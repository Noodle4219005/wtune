#!/usr/bin/env python3
"""predictor/benchmark.py -- CLI: fit nothing here interactively, run the whole grouped-CV
benchmark and write metrics.csv, per_case.csv, conformal.csv, importance.csv, summary.md.

    python3 benchmark.py --ledger predictor/ledger/runs.csv --out <dir>

Deterministic: every model that has a random_state/seed uses models.SEED (0); numpy's global RNG
is also seeded for the permutation-importance shuffles and the conformal quantiles.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from wtune.predictor import models as modelmod
from wtune.predictor import targets
from wtune.predictor import validate

SCOPES = ("STRICT", "PHYSICS-GRAY")
SCHEMES = (
    ("leave-one-run-out", "run_id"),
    ("leave-one-case-out", "case_base"),
    ("leave-one-family-out", "case_family"),
    ("leave-one-hardware-out", "hardware_id"),
)


def _f(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_ledger(path):
    with open(path, newline="") as fh:
        rows = []
        for r in csv.DictReader(fh):
            row = dict(r)
            for key in ("e_we", "e_sn", "e_vert", "dx", "dy", "time_step", "radt", "cudt",
                        "history_interval", "n_nodes", "n_ranks", "omp_threads", "nproc_x",
                        "nproc_y", "numtiles", "kmp_blocktime", "s_per_step_median",
                        "s_per_step_mean", "n_steps_measured", "diffwrf_differing_fields",
                        "mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics",
                        "sf_surface_physics", "ra_lw_physics", "ra_sw_physics",
                        "profile_micro_share", "profile_radiation_share", "profile_dynamics_share",
                        "profile_mpi_share", "profile_io_share", "profile_rank_imbalance",
                        "profile_thread_imbalance"):
                row[key] = _f(row.get(key))
            for key in ("job_id", "rep_index", "case_raw", "case_base", "case_family",
                        "hardware_id", "binary_kind", "compliance_class", "distribution",
                        "fidelity_label", "notes", "run_id", "source_dataset"):
                if row.get(key) == "":
                    row[key] = None
            rows.append(row)
    return rows


def run_metrics_matrix(rows):
    metric_rows, per_case_rows = [], []
    best_by_scope = {}
    underpowered_scopes = set()  # scopes where every model's LOCO result was UNDERPOWERED
    ledger_notes = {}  # scope -> {"excluded_unknown_group": n, "io_form_splits": m} (data facts,
                        # not model-specific -- computed once per scope from the raw ledger rows)
    for scope in SCOPES:
        scope_rows = [r for r in rows if r.get("compliance_class") == scope]
        ledger_notes[scope] = dict(
            excluded_unknown_group=targets.count_unknown_decision_group(scope_rows),
            io_form_splits=targets.count_io_form_splits(scope_rows),
        )
        loco_regret = {}
        any_loco_ok = False
        avail_map = modelmod.available_models()
        for model_name in modelmod.MODEL_FACTORIES:
            avail, reason = avail_map[model_name]
            for scheme_name, group_field in SCHEMES:
                row_out = dict(scope=scope, model=model_name, scheme=scheme_name)
                if not avail:
                    row_out.update(status="UNAVAILABLE", reason=reason)
                    metric_rows.append(row_out)
                    continue
                try:
                    m = validate.evaluate(rows, group_field, model_name, use_profile=False,
                                           scope=scope)
                except validate.NotEnoughGroups as e:
                    row_out.update(status="NotEnoughGroups", reason=str(e))
                    metric_rows.append(row_out)
                    continue
                except modelmod.NotEnoughData as e:
                    row_out.update(status="NotEnoughData", reason=str(e))
                    metric_rows.append(row_out)
                    continue
                status = "UNDERPOWERED" if m.get("underpowered") else "ok"
                top3_status = "UNDERPOWERED" if m.get("top3_underpowered") else "ok"
                row_out.update(status=status, top3_status=top3_status,
                                n_eval_cases=m.get("n_eval_cases"),
                                n_eval_groups_top3=m.get("n_eval_groups_top3"),
                                top1_rate=m.get("top1_rate"), top3_rate=m.get("top3_rate"),
                                mean_regret=m.get("mean_regret"), median_regret=m.get("median_regret"),
                                worst_regret=m.get("worst_regret"),
                                median_naive_regret=m.get("median_naive_regret"),
                                median_regret_improvement=m.get("median_regret_improvement"),
                                pairwise_accuracy=m.get("pairwise_accuracy"),
                                regression_fnr=m.get("regression_fnr"),
                                excluded_unknown_group=m.get("excluded_unknown_group"),
                                unavailable_folds=m.get("unavailable_folds"))
                metric_rows.append(row_out)
                if scheme_name == "leave-one-case-out":
                    any_loco_ok = True  # evaluate() ran successfully (ok OR underpowered)
                    if status == "ok":
                        loco_regret[model_name] = m["median_regret"]
                for key, cm in m.get("per_case", {}).items():
                    per_case_rows.append(dict(scope=scope, model=model_name, scheme=scheme_name,
                                               **cm))
        if loco_regret:
            best_by_scope[scope] = min(loco_regret, key=loco_regret.get)
        elif any_loco_ok:
            underpowered_scopes.add(scope)
    return metric_rows, per_case_rows, best_by_scope, underpowered_scopes, ledger_notes


def run_conformal(rows, best_by_scope):
    conformal_rows = []
    for scope, model_name in best_by_scope.items():
        try:
            intervals, coverage, predictions = validate.conformal_intervals(
                rows, model_name, scope=scope)
        except (validate.NotEnoughGroups, modelmod.NotEnoughData):
            continue
        run_by_id = {r["run_id"]: r for r in rows}
        for run_id, alpha_map in intervals.items():
            r = run_by_id[run_id]
            y_pred = predictions.get(run_id, r.get("y"))
            for alpha, (lo, hi) in alpha_map.items():
                rec = validate.recommendation_label(y_pred, hi)
                cov = coverage.get(alpha) or {}
                conformal_rows.append(dict(
                    scope=scope, model=model_name, run_id=run_id, case_base=r["case_base"],
                    alpha=alpha, lower=lo, upper=hi, measured_y=r.get("y"),
                    covered=(lo <= r["y"] <= hi) if r.get("y") is not None else None,
                    coverage_at_alpha=cov.get("rate"), coverage_n=cov.get("n"),
                    coverage_underpowered=cov.get("underpowered"),
                    predicted_gain=rec["predicted_gain"], gain_lower_95=rec["gain_lower_95"],
                    label=rec["label"],
                ))
    return conformal_rows


def run_importance(rows, best_by_scope):
    importance_rows = []
    for scope, model_name in best_by_scope.items():
        scope_rows = [r for r in rows if r.get("compliance_class") == scope and r.get("y") is not None]
        if len(scope_rows) < 2:
            continue
        model = validate.build_model(model_name)
        try:
            model.fit(scope_rows)
        except modelmod.NotEnoughData:
            continue
        perm = validate.permutation_importance(model, scope_rows)
        native = model.feature_importance() if hasattr(model, "feature_importance") else {}
        for f, v in perm.items():
            importance_rows.append(dict(scope=scope, model=model_name, feature=f,
                                         permutation_importance=v,
                                         native_importance=None))
        for f, v in native.items():
            importance_rows.append(dict(scope=scope, model=model_name, feature=f,
                                         permutation_importance=None,
                                         native_importance=v))
    return importance_rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def write_summary(path, metric_rows, best_by_scope, underpowered_scopes, ledger_notes,
                   conformal_rows, importance_rows, ablations):
    lines = ["# WTune configuration predictor -- benchmark summary\n"]
    lines.append("Every metric below is computed PER DECISION GROUP (case_base, hardware_id, "
                  "compliance_class, n_nodes, total_cores, binary_kind, io_form_history) -- "
                  "configs are only ranked/regretted/paired against alternatives that hold the "
                  "resource allocation fixed. See predictor/features.py `decision_group_key`.\n")
    for scope in SCOPES:
        lines.append("\n## Scope: %s\n" % scope)
        notes = ledger_notes.get(scope, {})
        lines.append("- decision-group data notes: %d row(s) excluded from ranking (unknown "
                      "n_nodes/total_cores); io_form_history splits %d group(s) that would "
                      "otherwise merge different I/O backends."
                      % (notes.get("excluded_unknown_group", 0), notes.get("io_form_splits", 0)))
        best = best_by_scope.get(scope)
        if not best:
            if scope in underpowered_scopes:
                lines.append("- **no model selected (underpowered)**: every model's "
                              "leave-one-case-out result had <5 eligible decision groups -- see "
                              "metrics.csv `n_eval_cases`/`status` columns.\n")
            else:
                lines.append("No model produced a usable leave-one-case-out result in this scope "
                              "(NotEnoughGroups/NotEnoughData for all five models -- see metrics.csv).\n")
            underpowered = sorted({(mm["model"], mm["scheme"]) for mm in metric_rows
                                    if mm["scope"] == scope and mm.get("status") == "UNDERPOWERED"})
            if underpowered:
                lines.append("- UNDERPOWERED (model, scheme) pairs: " +
                              ", ".join("%s/%s" % up for up in underpowered))
            continue
        loco = [m for m in metric_rows if m["scope"] == scope and m["model"] == best
                and m["scheme"] == "leave-one-case-out"]
        m = loco[0] if loco else {}
        lines.append("- **best model (leave-one-case-out median regret): `%s`**" % best)
        lines.append("- n_eval_cases (decision groups, >=2 configs) = %s%s" % (m.get("n_eval_cases"),
                      " (UNDERPOWERED: <5 held-out groups)" if m.get("status") == "UNDERPOWERED" else ""))
        lines.append("- top-3 hit rate = %s (eligible groups with >3 configs: n=%s%s)"
                      % (m.get("top3_rate"), m.get("n_eval_groups_top3"),
                         " -- UNDERPOWERED: <5" if m.get("top3_status") == "UNDERPOWERED" else ""))
        lines.append("- median regret = %s, worst regret = %s"
                      % (m.get("median_regret"), m.get("worst_regret")))
        lines.append("- naive baseline (always the group's default config) median regret = %s; "
                      "regret improvement over naive = %s"
                      % (m.get("median_naive_regret"), m.get("median_regret_improvement")))
        cov = [c for c in conformal_rows if c["scope"] == scope]
        if cov:
            by_alpha = {}
            for c in cov:
                by_alpha[c["alpha"]] = (c["coverage_at_alpha"], c["coverage_n"], c["coverage_underpowered"])
            lines.append("- empirical conformal coverage: %s"
                          % ", ".join("%.0f%% nominal -> %.3f (n=%s%s)"
                                      % ((1 - a) * 100, v, n, " UNDERPOWERED: <30" if up else "")
                                      for a, (v, n, up) in sorted(by_alpha.items()) if v is not None))
        top_feats = sorted(
            [i for i in importance_rows if i["scope"] == scope and i["permutation_importance"] is not None],
            key=lambda i: -abs(i["permutation_importance"]))[:5]
        if top_feats:
            lines.append("- top features (permutation importance, |MAE delta|): "
                          + ", ".join("%s=%.4f" % (i["feature"], i["permutation_importance"])
                                      for i in top_feats))
        abl = ablations.get(scope, {})
        if abl.get("status") == "no profile data":
            lines.append("- profile (Level-1) ablation: no profile data")
        elif abl.get("status") == "ok":
            l0, l1 = abl["level0"], abl["level1"]
            lines.append("- profile (Level-1) ablation: level0 median_regret=%s -> "
                          "level1 median_regret=%s" % (l0.get("median_regret"), l1.get("median_regret")))
        underpowered = sorted({(mm["model"], mm["scheme"]) for mm in metric_rows
                                if mm["scope"] == scope and mm.get("status") == "UNDERPOWERED"})
        if underpowered:
            lines.append("- UNDERPOWERED (model, scheme) pairs [top-1/regret, n<5 groups]: " +
                          ", ".join("%s/%s" % up for up in underpowered))
        underpowered_top3 = sorted({(mm["model"], mm["scheme"]) for mm in metric_rows
                                     if mm["scope"] == scope and mm.get("status") == "ok"
                                     and mm.get("top3_status") == "UNDERPOWERED"})
        if underpowered_top3:
            lines.append("- UNDERPOWERED (model, scheme) pairs [top-3, n<5 eligible groups]: " +
                          ", ".join("%s/%s" % up for up in underpowered_top3))

    lines.append("\n## Exact n per metric\n")
    lines.append("See metrics.csv `n_eval_cases`/`n_eval_groups_top3` columns per (scope, model, "
                  "scheme) row; per_case.csv has one row per held-out DECISION GROUP actually "
                  "evaluated (n_nodes/total_cores columns distinguish groups sharing a case_base).\n")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    np.random.seed(modelmod.SEED)
    os.makedirs(args.out, exist_ok=True)

    rows = load_ledger(args.ledger)
    rows = [r for r in rows if r.get("compliance_class") != "RESEARCH"]  # never evaluated
    baselines = targets.compute_case_baselines(rows)
    targets.add_y(rows, baselines)

    metric_rows, per_case_rows, best_by_scope, underpowered_scopes, ledger_notes = \
        run_metrics_matrix(rows)
    conformal_rows = run_conformal(rows, best_by_scope)
    importance_rows = run_importance(rows, best_by_scope)
    ablations = {scope: validate.profile_ablation(rows, model_name, scope=scope)
                 for scope, model_name in best_by_scope.items()}

    write_csv(os.path.join(args.out, "metrics.csv"), metric_rows,
              ["scope", "model", "scheme", "status", "reason", "n_eval_cases", "n_eval_groups_top3",
               "top3_status", "top1_rate", "top3_rate", "mean_regret", "median_regret",
               "worst_regret", "median_naive_regret", "median_regret_improvement",
               "pairwise_accuracy", "regression_fnr", "excluded_unknown_group", "unavailable_folds"])
    write_csv(os.path.join(args.out, "per_case.csv"), per_case_rows,
              ["scope", "model", "scheme", "case_base", "n_nodes", "total_cores", "top1", "top3",
               "top3_eligible", "regret", "naive_regret", "pairwise_acc", "pairwise_n", "fnr",
               "n_configs"])
    write_csv(os.path.join(args.out, "conformal.csv"), conformal_rows,
              ["scope", "model", "run_id", "case_base", "alpha", "lower", "upper", "measured_y",
               "covered", "coverage_at_alpha", "coverage_n", "coverage_underpowered",
               "predicted_gain", "gain_lower_95", "label"])
    write_csv(os.path.join(args.out, "importance.csv"), importance_rows,
              ["scope", "model", "feature", "permutation_importance", "native_importance"])
    write_summary(os.path.join(args.out, "summary.md"), metric_rows, best_by_scope,
                  underpowered_scopes, ledger_notes, conformal_rows, importance_rows, ablations)

    print("wrote metrics.csv (%d rows), per_case.csv (%d rows), conformal.csv (%d rows), "
          "importance.csv (%d rows), summary.md to %s"
          % (len(metric_rows), len(per_case_rows), len(conformal_rows), len(importance_rows),
             args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
