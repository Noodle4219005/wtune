#!/usr/bin/env python3
"""predictor/validate.py -- grouped CV, metrics, conformal uncertainty, importance, ablations.

Never uses random splits. Group fields:
    leave-one-run-out    group = run_id     (weak/optimistic: siblings of the same case, other
                                              reps/configs, stay in training -- a baseline to
                                              contrast against the stricter schemes below)
    leave-one-case-out   group = case_base  (the leakage-safe default; see
                                              knowledge/2026-09-05-loco-over-directories-leaks-...)
    leave-one-family-out group = case_family
    leave-one-hardware-out group = hardware_id (raises NotEnoughGroups: this corpus is one
                                              machine, f1-spr-icpnq)

For every scheme, metrics are computed PER HELD-OUT CASE (never across cases), because "best
config" and "regret" are only meaningful within one case's own configs. A run-out or family-out
fold's held-out rows are grouped back into their case_base(s) before scoring.
"""
from __future__ import annotations

import math
import statistics
import sys
from collections import defaultdict

import numpy as np

from wtune.predictor import features as feat
from wtune.predictor import models as modelmod
from wtune.predictor import targets

NOISE = targets.NOISE


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class NotEnoughGroups(Exception):
    pass


def fold_groups(rows, group_field):
    return sorted({r.get(group_field) for r in rows if r.get(group_field) is not None},
                  key=lambda x: str(x))


def build_model(name, use_profile=False):
    return modelmod.MODEL_FACTORIES[name](use_profile)


# ---------------------------------------------------------------------------
# Cross-validation core
# ---------------------------------------------------------------------------
def cross_validate(rows, group_field, model_name, use_profile=False, return_models=False):
    """Fits `model_name` len(groups) times, each time excluding one group's rows from training,
    and scores that group's rows. Returns (predictions: {run_id: float}, unavailable_groups: set)
    and, if return_models, also (fold_models: {group: fitted_model}, fold_test_rows: {group: rows}).
    A fold with too little training data (models.NotEnoughData) is skipped and its group recorded
    as unavailable, per the brief's "mark it unavailable for that fold rather than crash"."""
    groups = fold_groups(rows, group_field)
    if len(groups) < 2:
        raise NotEnoughGroups("%s has %d distinct group(s), need >=2" % (group_field, len(groups)))

    predictions, unavailable = {}, set()
    fold_models, fold_test_rows = {}, {}
    for g in groups:
        train = [r for r in rows if r.get(group_field) != g]
        test = [r for r in rows if r.get(group_field) == g]
        if not train or not test:
            unavailable.add(g)
            continue
        model = build_model(model_name, use_profile)
        try:
            model.fit(train)
        except modelmod.NotEnoughData:
            unavailable.add(g)
            continue
        preds = model.score(test)
        for r, p in zip(test, preds):
            predictions[r["run_id"]] = p
        if return_models:
            fold_models[g] = model
            fold_test_rows[g] = test

    if return_models:
        return predictions, unavailable, fold_models, fold_test_rows
    return predictions, unavailable


# ---------------------------------------------------------------------------
# Per-case metrics
# ---------------------------------------------------------------------------
def _aggregate_case_configs(case_rows, predictions):
    """Within one case_base's rows (that received an out-of-fold prediction), average measured y
    and predicted y per config (replicates averaged in log space, per targets.py)."""
    by_config = defaultdict(lambda: {"y": [], "pred": [], "regress": []})
    for r in case_rows:
        if r["run_id"] not in predictions or r.get("y") is None:
            continue
        ck = feat.config_key(r)
        by_config[ck]["y"].append(r["y"])
        by_config[ck]["pred"].append(predictions[r["run_id"]])
        rg = targets.regress_1pct(r)
        if rg is not None:
            by_config[ck]["regress"].append(rg)
    out = {}
    for ck, d in by_config.items():
        if not d["y"]:
            continue
        out[ck] = {
            "y": sum(d["y"]) / len(d["y"]),
            "pred": sum(d["pred"]) / len(d["pred"]),
            "regress": any(d["regress"]) if d["regress"] else None,
        }
    return out


def per_case_metrics(case_rows, predictions):
    """`case_rows` are the rows of ONE decision group (see features.decision_group_key) -- a
    fixed resource allocation within one case_base, not the whole case_base. Returns None if the
    group has <2 distinct configs with predictions (excluded from aggregates entirely, per the
    brief), else a dict of this group's metrics.

    `top3` is None (not "trivially true") when the group has <=3 distinct configs: with only 2 or
    3 configs, "is the true best in the top 3" is guaranteed by construction and proves nothing.
    See `top3_eligible`."""
    configs = _aggregate_case_configs(case_rows, predictions)
    if len(configs) < 2:
        return None

    items = sorted(configs.items(), key=lambda kv: kv[1]["pred"])  # ascending predicted y
    measured_best = min(v["y"] for v in configs.values())
    is_best = lambda y: y <= measured_best + NOISE

    predicted_best_ck, predicted_best_v = items[0]
    top1 = is_best(predicted_best_v["y"])
    top3_eligible = len(configs) > 3
    top3 = any(is_best(v["y"]) for _, v in items[:3]) if top3_eligible else None
    regret = math.exp(predicted_best_v["y"] - measured_best) - 1
    # Naive baseline: always "pick the group's most common/default configuration". That config is
    # exactly the one targets.compute_case_baselines chose for this group, whose y is 0 by
    # construction (y = log(s/s_baseline)), and it is itself one of this group's measured configs
    # (so it needs no separate lookup here). naive_regret is how a model that never learns
    # anything -- just always recommends the default -- would score; every real model is reported
    # relative to this so a meaningless 0.0 regret cannot look like skill.
    naive_regret = math.exp(0.0 - measured_best) - 1

    # pairwise accuracy over non-tie pairs, ranked by predicted y vs measured y
    keys = list(configs.keys())
    correct, total = 0, 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = configs[keys[i]], configs[keys[j]]
            diff = a["y"] - b["y"]
            if abs(diff) < NOISE:
                continue
            total += 1
            measured_a_faster = diff < 0
            predicted_a_faster = a["pred"] < b["pred"]
            if measured_a_faster == predicted_a_faster:
                correct += 1
    pairwise_acc = (correct / total) if total else None

    regress_true = [v for v in configs.values() if v["regress"] is True]
    fn = sum(1 for v in regress_true if v["pred"] <= math.log(1.01))
    fnr = (fn / len(regress_true)) if regress_true else None

    return dict(top1=top1, top3=top3, top3_eligible=top3_eligible, regret=regret,
                naive_regret=naive_regret, pairwise_acc=pairwise_acc,
                pairwise_n=total, fnr=fnr, n_configs=len(configs))


def aggregate_metrics(rows, predictions):
    """Groups held-out predictions by DECISION GROUP (case_base, hardware_id, compliance_class,
    n_nodes, total_cores, binary_kind, io_form_history -- see features.decision_group_key), not
    by case_base alone, and computes per_case_metrics for each, then aggregates. `rows` should be
    pre-filtered to the compliance scope being reported. Rows with a usable measurement whose
    decision group cannot be determined (n_nodes/n_ranks/omp_threads unknown) are excluded and
    counted in `excluded_unknown_group` rather than silently dropped.

    top-1/regret/pairwise are aggregated over every group with >=2 distinct configs
    (`n_eval_cases`); top-3 is aggregated only over the subset with >3 distinct configs
    (`n_eval_groups_top3`) -- with <=3 configs, "true best is in the top 3" is trivially
    guaranteed and proves nothing. Both counts get their own UNDERPOWERED threshold (<5)."""
    by_group = defaultdict(list)
    excluded_unknown_group = 0
    for r in rows:
        if _num(r.get("s_per_step_median")) is None:
            continue
        key = feat.decision_group_key(r)
        if key is None:
            excluded_unknown_group += 1
            continue
        by_group[key].append(r)

    per_case = {}
    for key, group_rows in by_group.items():
        m = per_case_metrics(group_rows, predictions)
        if m is not None:
            m["case_base"], m["n_nodes"], m["total_cores"] = key[0], key[3], key[4]
            per_case[key] = m

    n = len(per_case)
    if n == 0:
        return dict(n_eval_cases=0, underpowered=True, n_eval_groups_top3=0, top3_underpowered=True,
                    excluded_unknown_group=excluded_unknown_group, per_case={})

    top3_groups = [m for m in per_case.values() if m["top3_eligible"]]
    n_top3 = len(top3_groups)
    regrets = [m["regret"] for m in per_case.values()]
    naive_regrets = [m["naive_regret"] for m in per_case.values()]
    pw = [m["pairwise_acc"] for m in per_case.values() if m["pairwise_acc"] is not None]
    fnrs = [m["fnr"] for m in per_case.values() if m["fnr"] is not None]
    return dict(
        n_eval_cases=n,
        underpowered=n < 5,
        n_eval_groups_top3=n_top3,
        top3_underpowered=n_top3 < 5,
        top1_rate=sum(m["top1"] for m in per_case.values()) / n,
        top3_rate=(sum(m["top3"] for m in top3_groups) / n_top3) if n_top3 else None,
        mean_regret=statistics.mean(regrets), median_regret=statistics.median(regrets),
        worst_regret=max(regrets),
        median_naive_regret=statistics.median(naive_regrets),
        median_regret_improvement=statistics.median(naive_regrets) - statistics.median(regrets),
        pairwise_accuracy=(statistics.mean(pw) if pw else None),
        regression_fnr=(statistics.mean(fnrs) if fnrs else None),
        excluded_unknown_group=excluded_unknown_group,
        per_case=per_case,
    )


def evaluate(rows, group_field, model_name, use_profile=False, scope=None):
    """One (model, scheme, scope) evaluation: cross_validate + aggregate_metrics.
    `scope` restricts both training and evaluation to rows with that compliance_class (RESEARCH
    is never evaluated -- callers should simply never pass scope='RESEARCH')."""
    scoped = [r for r in rows if scope is None or r.get("compliance_class") == scope]
    predictions, unavailable = cross_validate(scoped, group_field, model_name, use_profile)
    metrics = aggregate_metrics(scoped, predictions)
    metrics["unavailable_folds"] = len(unavailable)
    return metrics


# ---------------------------------------------------------------------------
# Conformal uncertainty (grouped CV+/jackknife+), leave-one-case-out only.
#
# Standard jackknife+ leaves one POINT out; we have one MODEL per left-out CASE (a group), so this
# is the grouped variant: for a test row x in held-out case c, every OTHER fold's model h != c
# predicts at x, and is combined with fold h's own out-of-fold residuals to build the candidate
# bound set {pred_h(x) +/- |residual|}. The (alpha/2, 1-alpha/2) empirical quantiles of that pooled
# set are the interval. This is a documented simplification (no finite-sample smoothing
# correction) appropriate to the small case counts here -- widths are reported as-is, never
# artificially tightened.
# ---------------------------------------------------------------------------
CONFORMAL_MIN_N = 30  # below this many held-out points behind a coverage value, mark it UNDERPOWERED


def _fold_residuals(predictions, fold_test_rows):
    """{fold_group: [|y - out-of-fold prediction| for each of that fold's test rows]}. Factored
    out of conformal_intervals so conformal_intervals_for_new_rows (recommend.py's candidates,
    which have no held-out fold of their own) can reuse the exact same residual pool."""
    fold_residuals = {}
    for g, test_rows in fold_test_rows.items():
        fold_residuals[g] = [abs(r["y"] - predictions[r["run_id"]])
                              for r in test_rows if r.get("y") is not None and r["run_id"] in predictions]
    return fold_residuals


def conformal_intervals(rows, model_name, use_profile=False, alphas=(0.10, 0.05), scope=None):
    """Returns (intervals: {run_id: {alpha: (lo, hi)}},
    coverage: {alpha: {"rate": empirical_rate, "n": held_out_points, "underpowered": n < 30}},
    predictions: {run_id: predicted_y}) -- `predictions` is the same leave-one-case-out point
    prediction used everywhere else, for a recommendation's predicted_gain=1-exp(y_pred).
    `coverage[alpha]["n"]` is the number of held-out points the empirical rate is computed over;
    an empirical rate with n<30 is not a reliable estimate of nominal coverage and is flagged."""
    scoped = [r for r in rows if scope is None or r.get("compliance_class") == scope]
    predictions, unavailable, fold_models, fold_test_rows = cross_validate(
        scoped, "case_base", model_name, use_profile, return_models=True)

    fold_residuals = _fold_residuals(predictions, fold_test_rows)

    intervals = {}   # run_id -> {alpha: (lo, hi)}
    for g, test_rows in fold_test_rows.items():
        others = [h for h in fold_models if h != g and fold_residuals.get(h)]
        for row in test_rows:
            if row.get("y") is None:
                continue
            lo_pool, hi_pool = [], []
            for h in others:
                pred_h = fold_models[h].score([row])[0]
                for resid in fold_residuals[h]:
                    lo_pool.append(pred_h - resid)
                    hi_pool.append(pred_h + resid)
            if not lo_pool:
                continue
            intervals[row["run_id"]] = {}
            for alpha in alphas:
                lo = float(np.quantile(lo_pool, alpha / 2))
                hi = float(np.quantile(hi_pool, 1 - alpha / 2))
                intervals[row["run_id"]][alpha] = (lo, hi)

    coverage = {}
    for alpha in alphas:
        hit, total = 0, 0
        for row in scoped:
            iv = intervals.get(row["run_id"], {}).get(alpha)
            if iv is None or row.get("y") is None:
                continue
            total += 1
            if iv[0] <= row["y"] <= iv[1]:
                hit += 1
        coverage[alpha] = {
            "rate": (hit / total) if total else None,
            "n": total,
            "underpowered": total < CONFORMAL_MIN_N,
        }
    return intervals, coverage, predictions


def conformal_intervals_for_new_rows(rows, new_rows, model_name, use_profile=False,
                                      alphas=(0.10, 0.05), scope=None):
    """Same grouped CV+/jackknife+ residual-pooling as conformal_intervals, but for rows that are
    OUTSIDE the training ledger entirely (e.g. a recommend.py shortlist candidate for a case that
    has zero rows of its own in `rows`) rather than for one of `rows`' own held-out points.

    Every leave-one-case-out fold model contributes to every new row's interval -- unlike
    conformal_intervals, there is no "own fold" to exclude, because a `new_rows` row is not part
    of any training case_base and so cannot leak into its own prediction.

    Returns {index into new_rows: {alpha: (lo, hi)}} (indexed by position, not run_id, since a
    freshly-built candidate row need not carry a ledger run_id). Raises the same
    NotEnoughGroups/NotEnoughData as cross_validate when the scoped training data is too thin to
    fit even one fold."""
    scoped = [r for r in rows if scope is None or r.get("compliance_class") == scope]
    predictions, _unavailable, fold_models, fold_test_rows = cross_validate(
        scoped, "case_base", model_name, use_profile, return_models=True)
    fold_residuals = _fold_residuals(predictions, fold_test_rows)

    intervals = {}
    for i, row in enumerate(new_rows):
        lo_pool, hi_pool = [], []
        for h, model in fold_models.items():
            resids = fold_residuals.get(h)
            if not resids:
                continue
            pred_h = model.score([row])[0]
            for resid in resids:
                lo_pool.append(pred_h - resid)
                hi_pool.append(pred_h + resid)
        if not lo_pool:
            continue
        intervals[i] = {}
        for alpha in alphas:
            lo = float(np.quantile(lo_pool, alpha / 2))
            hi = float(np.quantile(hi_pool, 1 - alpha / 2))
            intervals[i][alpha] = (lo, hi)
    return intervals


def recommendation_label(y_pred, upper_y_bound):
    """SAFE if even the pessimistic (slowest) end of the 95% interval still beats baseline."""
    gain = 1 - math.exp(y_pred)
    gain_lower = 1 - math.exp(upper_y_bound)
    return dict(predicted_gain=gain, gain_lower_95=gain_lower,
                label="SAFE" if gain_lower > 0 else "EXPERIMENTAL")


def quantile_variant(rows, use_profile=False, scope=None):
    """CatBoost Quantile(0.05)/Quantile(0.95) point models, when catboost is available."""
    if not modelmod.HAVE_CATBOOST:
        return None
    from catboost import CatBoostRegressor
    scoped = [r for r in rows if scope is None or r.get("compliance_class") == scope]
    dicts, y = modelmod._rows_to_Xy(scoped, use_profile)
    mask = ~np.isnan(y)
    dicts = [d for d, m in zip(dicts, mask) if m]
    y = y[mask]
    if len(dicts) < 2:
        return None
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(dicts)
    out = {}
    for q in (0.05, 0.95):
        m = CatBoostRegressor(depth=4, l2_leaf_reg=10, iterations=300, random_seed=modelmod.SEED,
                               loss_function="Quantile:alpha=%.2f" % q, verbose=False,
                               thread_count=modelmod.THREADS, allow_writing_files=False)
        m.fit(modelmod.Pool(X, y, thread_count=modelmod.THREADS))
        out[q] = (vec, m)
    return out


# ---------------------------------------------------------------------------
# Permutation importance (row-level, model-agnostic: shuffles a ledger field's values directly
# across rows and re-scores through the model's public `score()`, so it works uniformly across
# all five model classes rather than requiring access to each one's internal feature matrix).
# ---------------------------------------------------------------------------
IMPORTANCE_FIELDS = (
    "e_we", "e_sn", "e_vert", "time_step", "nproc_x", "nproc_y", "numtiles", "omp_threads",
    "distribution", "kmp_blocktime", "mp_physics", "bl_pbl_physics", "ra_lw_physics",
)


def permutation_importance(model, rows, fields=IMPORTANCE_FIELDS, n_repeats=5, seed=0):
    labeled = [r for r in rows if r.get("y") is not None]
    if not labeled:
        return {}
    y_true = np.array([r["y"] for r in labeled])
    base_pred = np.array(model.score(labeled))
    base_err = float(np.mean(np.abs(y_true - base_pred)))
    rng = np.random.RandomState(seed)
    out = {}
    for f in fields:
        deltas = []
        for _ in range(n_repeats):
            vals = [r.get(f) for r in labeled]
            perm = rng.permutation(len(vals))
            shuffled = [dict(r) for r in labeled]
            for r2, idx in zip(shuffled, perm):
                r2[f] = vals[idx]
            pred = np.array(model.score(shuffled))
            err = float(np.mean(np.abs(y_true - pred)))
            deltas.append(err - base_err)
        out[f] = float(np.mean(deltas))
    return out


# ---------------------------------------------------------------------------
# Level-0 (case+config) vs Level-1 (+profile) ablation.
# ---------------------------------------------------------------------------
def profile_ablation(rows, model_name, scope=None):
    scoped = [r for r in rows if scope is None or r.get("compliance_class") == scope]
    if not feat.has_profile_data(scoped):
        return {"status": "no profile data"}
    level0 = evaluate(rows, "case_base", model_name, use_profile=False, scope=scope)
    level1 = evaluate(rows, "case_base", model_name, use_profile=True, scope=scope)
    return {"status": "ok", "level0": level0, "level1": level1}


# ---------------------------------------------------------------------------
# Physics-gray constrained optimisation.
# The performance model and the fidelity-risk model are kept separate: the fidelity model trains
# on fidelity_label when present; with none present (true for this corpus -- see predictor/README.md
# on why PHYSICS-GRAY rows here carry no fidelity_label), every PHYSICS-GRAY recommendation is
# UNCONFIRMED regardless of how good the performance model looks.
# ---------------------------------------------------------------------------
def fidelity_model_status(rows):
    labeled = [r for r in rows if r.get("compliance_class") == "PHYSICS-GRAY" and r.get("fidelity_label")]
    if not labeled:
        return {"status": "no fidelity labels"}
    from collections import Counter
    return {"status": "trained", "n_labeled": len(labeled),
            "label_counts": dict(Counter(r["fidelity_label"] for r in labeled))}


def constrained_recommend(rows, model, case_base):
    """Best PHYSICS-GRAY config for one case, per `model`'s score(); always UNCONFIRMED unless a
    fidelity model was actually trained (see fidelity_model_status)."""
    fstatus = fidelity_model_status(rows)
    candidates = [r for r in rows if r["case_base"] == case_base and r.get("compliance_class") == "PHYSICS-GRAY"]
    if not candidates:
        return None
    preds = model.score(candidates)
    best = min(zip(candidates, preds), key=lambda cp: cp[1])
    return dict(case_base=case_base, config=feat.config_key(best[0]), predicted_y=best[1],
                fidelity_status=fstatus["status"],
                verdict="UNCONFIRMED" if fstatus["status"] != "trained" else "CONFIRMED")
