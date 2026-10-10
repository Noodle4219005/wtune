#!/usr/bin/env python3
"""predictor/physics.py -- PHYSICS-GRAY performance + fidelity models (decision 0051).

Reads physics-campaign ledger rows (ledger.LEDGER_FIELDS, notes 'mode=physics template=Pxx
family= edits=k=v;.. decomp= fidelity_nd_max= probe=knob:verdict,..|none'). Two separate models:
performance (cost vs the case's P00) and fidelity (drift vs the twin's chaos floor,
fidelity_gate.py), both models.py's estimators over physics-change features. Exploratory until
the campaign lands (only test_physics.py's synthetic ledger so far).
CLI: python3 physics.py --ledger predictor/ledger/runs.csv --out <dir>"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark
import features as feat
import models as modelmod
import targets

# campaign/configs.py is the single source of truth for how a template edit op changes a namelist.
_CAMPAIGN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "campaign")
if _CAMPAIGN_DIR not in sys.path:
    sys.path.insert(0, _CAMPAIGN_DIR)
import configs as campaign_configs

if modelmod.HAVE_CATBOOST:
    from catboost import CatBoostRegressor

SEED = modelmod.SEED


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class NotEnoughGroups(Exception):
    pass


# ---------------------------------------------------------------------------
# Notes parsing (the physics-campaign contract; these tokens are not ledger columns).
# ---------------------------------------------------------------------------
def parse_notes(notes: str | None) -> dict[str, str | None]:
    """Space-separated key=value tokens (no spaces inside a value; partition on the FIRST '=' so
    a value that itself contains '=' -- e.g. edits=swint_opt=1 -- survives intact). Missing/empty
    notes -> {}."""
    out = {}
    if not notes:
        return out
    for tok in notes.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
        else:
            out[tok] = None
    return out


def _parse_pairs(s, pair_sep, kv_sep):
    out = {}
    if not s or s == "none":
        return out
    for tok in s.split(pair_sep):
        if not tok:
            continue
        if kv_sep in tok:
            k, v = tok.split(kv_sep, 1)
            out[k] = v
    return out


def parse_edits(s: str | None) -> dict[str, str]:
    """notes' edits=<k=v;k=v> value -> {k: v}."""
    return _parse_pairs(s, ";", "=")


def parse_probe(s: str | None) -> dict[str, str]:
    """notes' probe=<knob>:<verdict>,<knob>:<verdict>|none value -> {knob: verdict}."""
    return _parse_pairs(s, ",", ":")


def is_physics_row(row: dict) -> bool:
    return parse_notes(row.get("notes")).get("mode") == "physics"


def physics_rows(rows: list[dict]) -> list[dict]:
    """Every mode=physics row (P00 reference, twin, and Pxx candidates alike -- callers that need
    only candidates filter on `_template_of(r) not in (None, "P00")` themselves)."""
    return [r for r in rows if is_physics_row(r)]


def _template_of(row):
    return parse_notes(row.get("notes")).get("template")


def _status_of(row):
    return parse_notes(row.get("notes")).get("status")


def _log_s(row):
    """log of wall-clock cost per SIMULATED second (s_per_step_median / time_step), not per step:
    a template that changes time_step (P03 x1.2) keeps per-step cost flat but needs 1/1.2 as many
    steps, so per-step cost hid its real ~17% time-to-solution change (campaign 1077168-1077173).
    Falls back to per-step cost only when time_step is unknown."""
    s = _num(row.get("s_per_step_median"))
    if not s or s <= 0:
        return None
    dt = _num(row.get("time_step"))
    return math.log(s / dt) if dt and dt > 0 else math.log(s)


# ---------------------------------------------------------------------------
# Grouping: baseline = P00 at the SAME decomp (not features.decision_group_key, whose
# compliance_class would separate STRICT P00 from its PHYSICS-GRAY candidates).
# ---------------------------------------------------------------------------
def physics_group_key(row: dict) -> tuple | None:
    notes = parse_notes(row.get("notes"))
    decomp = notes.get("decomp")
    if row.get("case_base") is None or row.get("hardware_id") is None or decomp is None:
        return None
    n_nodes = feat.resolved_n_nodes(row)
    cores = feat.total_cores(row)
    if n_nodes is None or cores is None:
        return None
    return (row["case_base"], row["hardware_id"], n_nodes, cores, decomp)


def physics_groups(rows: list[dict]) -> dict[tuple, dict]:
    """{group_key: {"baseline": [row,...] (template P00), "candidates": [row,...] (template!=P00,
    at the SAME decomp as the baseline)}}. The twin row (P00 at TWIN_DECOMP) lands in its own
    group, with no candidates ever sharing its decomp -- harmless, just never used as a
    performance baseline (its job is the fidelity chaos floor, computed by fidelity_gate.py, not
    by this module). Non-physics rows are silently ignored (call physics_rows() first if unsure)."""
    groups = defaultdict(lambda: {"baseline": [], "candidates": []})
    for row in physics_rows(rows):
        key = physics_group_key(row)
        if key is None:
            continue
        template = _template_of(row)
        if template is None:
            continue  # data-quality guard: a physics row with no template= is not usable
        bucket = "baseline" if template == "P00" else "candidates"
        groups[key][bucket].append(row)
    return dict(groups)


# ---------------------------------------------------------------------------
# physics_suite resolution (module_check_a_mundo.F:3078-3283; wiki 2026-09-14-wrf-4-8-physics-
# namelist-compatibility-rules...): only a -1 sentinel is filled, as setup_physics_suite does.
# ---------------------------------------------------------------------------
PHYSICS_SUITE_TABLE = {
    "conus": dict(cu_physics=6, mp_physics=8, ra_lw_physics=4, ra_sw_physics=4,
                  bl_pbl_physics=2, sf_sfclay_physics=2, sf_surface_physics=2),
    "tropical": dict(cu_physics=16, mp_physics=6, ra_lw_physics=4, ra_sw_physics=4,
                     bl_pbl_physics=1, sf_sfclay_physics=1, sf_surface_physics=2),
}

# From campaign/cases.json: a ledger row has no physics_suite column, so attach_change_context
# resolves a P00 row's -1 sentinels through this.
CASE_PHYSICS_SUITE = {"conus25": "conus", "conus12": "conus", "maria1km": "tropical"}


def resolve_physics_suite(native_physics: dict, physics_suite: str | None = None) -> dict:
    """`native_physics` with every -1 sentinel replaced by `physics_suite`'s resolved value.
    physics_suite=None or unrecognised leaves -1 as-is -- there is nothing to resolve it against,
    and this never guesses."""
    resolved = dict(native_physics)
    table = PHYSICS_SUITE_TABLE.get(physics_suite, {})
    for k, v in resolved.items():
        if v == -1 and k in table:
            resolved[k] = table[k]
    return resolved


# ---------------------------------------------------------------------------
# Performance target: y = log(s_candidate / s_P00), replicates averaged in log space. A template
# with no usable replicate gets y=None and is reported INFEASIBLE, not dropped.
# ---------------------------------------------------------------------------
def performance_targets(rows: list[dict]) -> dict[tuple, dict]:
    """{group_key: {"baseline_log_s": float|None,
                     "templates": {template: {"y": float|None, "n_ok": int, "n_failed": int,
                                               "rows": [row,...]}}}}"""
    out = {}
    for key, g in physics_groups(rows).items():
        base_logs = [x for x in (_log_s(r) for r in g["baseline"]) if x is not None]
        baseline_log_s = sum(base_logs) / len(base_logs) if base_logs else None
        by_template = defaultdict(list)
        for r in g["candidates"]:
            by_template[_template_of(r)].append(r)
        templates = {}
        for tmpl, trows in by_template.items():
            logs = [_log_s(r) for r in trows]
            ok_logs = [x for x in logs if x is not None]
            n_failed = sum(1 for x in logs if x is None)
            y = None
            if ok_logs and baseline_log_s is not None:
                y = (sum(ok_logs) / len(ok_logs)) - baseline_log_s
            templates[tmpl] = dict(y=y, n_ok=len(ok_logs), n_failed=n_failed, rows=trows)
        out[key] = dict(baseline_log_s=baseline_log_s, templates=templates)
    return out


def add_perf_targets(rows: list[dict]) -> dict[tuple, dict]:
    """Attaches row["y"] (this module's performance target, same key models.py's estimators read)
    to every physics candidate row in place. Baseline/twin (template=="P00") rows and candidates
    whose group has no usable P00 baseline get y=None. Returns performance_targets()'s summary."""
    summary = performance_targets(rows)
    for g in summary.values():
        for tdata in g["templates"].values():
            for r in tdata["rows"]:
                r["y"] = tdata["y"]
    return summary


def infeasible_candidates(rows: list[dict]) -> list[tuple]:
    """[(group_key, template, n_failed)] for every candidate template where every replicate in
    its physics group FAILED (n_ok == 0) -- these are the mp_physics=6-style "does the probe's
    REJECT actually hold" candidates that might not even finish."""
    out = []
    for key, g in performance_targets(rows).items():
        for tmpl, tdata in g["templates"].items():
            if tdata["n_ok"] == 0:
                out.append((key, tmpl, tdata["n_failed"]))
    return out


# ---------------------------------------------------------------------------
# Fidelity targets: log10(fidelity_nd_max) and the WITHIN-FLOOR < SMALL-DRIFT < LARGE-DRIFT
# ordinal label. INVALID/unlabeled rows are excluded from training and counted.
# ---------------------------------------------------------------------------
FIDELITY_ORDER = {"WITHIN-FLOOR": 0, "SMALL-DRIFT": 1, "LARGE-DRIFT": 2}
FIDELITY_LABELS = ("WITHIN-FLOOR", "SMALL-DRIFT", "LARGE-DRIFT")  # ordinal order
DEFAULT_ALLOWED_LABELS = ("WITHIN-FLOOR", "SMALL-DRIFT")


def fidelity_nd(row: dict) -> float | None:
    notes = row.get("_phys_notes") or parse_notes(row.get("notes"))
    v = _num(notes.get("fidelity_nd_max"))
    return v if v and v > 0 else None


def fidelity_dataset(rows: list[dict]) -> dict[str, list[dict]]:
    """{"trainable": [rows with label in FIDELITY_ORDER], "invalid": [label==INVALID rows],
    "unlabeled": [mode=physics rows with no fidelity_label at all -- FAILED runs, or rep-2
    timing-only replicates that never had history enabled]}."""
    trainable, invalid, unlabeled = [], [], []
    for r in physics_rows(rows):
        label = (r.get("fidelity_label") or "").strip()
        if label in FIDELITY_ORDER:
            trainable.append(r)
        elif label == "INVALID":
            invalid.append(r)
        else:
            unlabeled.append(r)
    return dict(trainable=trainable, invalid=invalid, unlabeled=unlabeled)


def add_fidelity_targets(rows: list[dict]) -> dict[str, list[dict]]:
    """Attaches row["y_fidelity_label"] (0/1/2, ordinal) and row["y_fidelity_nd"]
    (log10(fidelity_nd_max)) in place for trainable rows; None for invalid/unlabeled. Returns
    fidelity_dataset()'s split."""
    ds = fidelity_dataset(rows)
    for r in ds["trainable"]:
        r["y_fidelity_label"] = FIDELITY_ORDER[r["fidelity_label"]]
        nd = fidelity_nd(r)
        r["y_fidelity_nd"] = math.log10(nd) if nd else None
    for r in ds["invalid"] + ds["unlabeled"]:
        r["y_fidelity_label"] = None
        r["y_fidelity_nd"] = None
    return ds


# ---------------------------------------------------------------------------
# Data sufficiency.
# ---------------------------------------------------------------------------
UNDERPOWERED_MIN_CASES = 5      # qualifying cases
MIN_CANDIDATES_PER_CASE = 3     # distinct templates with a target

# Fewer out-of-fold label predictions than this per template = insufficient evidence ->
# UNCONFIRMED (decision 0051: lafe P05 predicted SMALL-DRIFT in-sample, measured LARGE-DRIFT).
MIN_OOF_DANGEROUS_MISS = 5

# Lead-review gates (2026-09-15): A blocks a template measured LARGE-DRIFT in at least this
# fraction of training cases (karina/P05 holdout miss); B requires a real predicted speedup.
DEFAULT_MAX_TEMPLATE_LARGE_RATE = 0.5   # fraction of labeled training cases
DEFAULT_MIN_SPEEDUP_PCT = -1.0     # % time-to-solution vs P00, negative = faster


def data_sufficiency(rows: list[dict]) -> dict:
    """Cases (case_base) with >=3 distinct candidate templates that received a performance target
    (call after add_perf_targets). UNDERPOWERED when fewer than 5 such cases exist."""
    by_case = defaultdict(set)
    for r in physics_rows(rows):
        if _template_of(r) in (None, "P00") or r.get("y") is None:
            continue
        by_case[r.get("case_base")].add(_template_of(r))
    qualifying = sorted(c for c, tmpls in by_case.items() if len(tmpls) >= MIN_CANDIDATES_PER_CASE)
    return dict(n_qualifying_cases=len(qualifying), qualifying_cases=qualifying,
                underpowered=len(qualifying) < UNDERPOWERED_MIN_CASES)


# ---------------------------------------------------------------------------
# Physics-change features: the row's edited columns vs the P00 values attach_change_context
# stashes in row["_phys_from"] (physics_change_features is then pure in the row).
# ---------------------------------------------------------------------------
CHANGE_FIELDS = ("mp_physics", "bl_pbl_physics", "sf_sfclay_physics",
                  "ra_lw_physics", "ra_sw_physics", "radt", "time_step")


def attach_change_context(rows: list[dict]) -> list[dict]:
    """For every physics candidate row, attach `_phys_from` (dict of the P00 baseline's own
    CHANGE_FIELDS values in the same physics group, physics_suite-resolved for
    conus25/conus12/maria1km) and `_phys_notes` (parsed notes, cached). Idempotent. `rows` must be
    the FULL ledger (or at least include the P00 baseline rows), not just the candidates."""
    for g in physics_groups(rows).values():
        base_vals = {}
        if g["baseline"]:
            b0 = g["baseline"][0]
            base_vals = {f: b0.get(f) for f in CHANGE_FIELDS}
            suite = CASE_PHYSICS_SUITE.get(b0.get("case_base"))
            if suite:
                base_vals = resolve_physics_suite(base_vals, suite)
        for r in g["candidates"]:
            r["_phys_from"] = base_vals
            r["_phys_notes"] = parse_notes(r.get("notes"))
    return rows


def physics_change_features(row: dict) -> dict[str, float | None]:
    notes = row.get("_phys_notes") or parse_notes(row.get("notes"))
    frm = row.get("_phys_from") or {}
    template, family = notes.get("template"), notes.get("family")
    edits = parse_edits(notes.get("edits"))
    out = {}
    if template:
        out["template_%s" % template] = 1.0
    if family:
        out["family_%s" % family] = 1.0

    radt_from, radt_to = _num(frm.get("radt")), _num(row.get("radt"))
    if radt_from:
        out["radt_ratio"] = radt_to / radt_from if radt_to is not None else None
    ts_from, ts_to = _num(frm.get("time_step")), _num(row.get("time_step"))
    if ts_from:
        out["time_step_ratio"] = ts_to / ts_from if ts_to is not None else None
    out["swint_on"] = 1.0 if edits.get("swint_opt") == "1" else 0.0

    for f in ("mp_physics", "bl_pbl_physics", "sf_sfclay_physics", "ra_lw_physics", "ra_sw_physics"):
        fv, tv = _num(frm.get(f)), _num(row.get(f))
        if fv is not None:
            out["%s_from" % f] = fv
        if tv is not None:
            out["%s_to" % f] = tv
        if fv is not None and tv is not None:
            out["%s_changed" % f] = 1.0 if fv != tv else 0.0
    return out


def probe_features(row: dict) -> dict[str, float]:
    notes = row.get("_phys_notes") or parse_notes(row.get("notes"))
    verdicts = list(parse_probe(notes.get("probe")).values())
    return {
        "probe_any_reject": 1.0 if any(v == "REJECT" for v in verdicts) else 0.0,
        "probe_any_confirm": 1.0 if any(v == "CONFIRM" for v in verdicts) else 0.0,
        "probe_any_pass_wrf": 1.0 if any(v == "PASS->WRF" for v in verdicts) else 0.0,
        "probe_n_knobs": float(len(verdicts)),
    }


def row_features(row: dict) -> dict[str, Any]:
    """The one feature dict shared by every model in this module (performance, fidelity_nd,
    fidelity_label): case description (grid/native-physics-as-edited/hardware, via
    feat.case_features) + the physics change itself + the probe verdict on the edited knob(s)."""
    d = {}
    d.update(feat.case_features(row))
    d.update(physics_change_features(row))
    d.update(probe_features(row))
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Models: models.LinearModel / CatBoostPointModel's estimators over row_features() (decomp is
# fixed within a group, so config_features would be noise). `target` picks the label key.
# ---------------------------------------------------------------------------
class PhysicsLinearModel:
    name = "ridge"

    def __init__(self, target="y", feature_fn=row_features):
        self.target = target
        self.feature_fn = feature_fn

    def fit(self, train_rows: list[dict]) -> PhysicsLinearModel:
        dicts = [self.feature_fn(r) for r in train_rows]
        y = np.array([r.get(self.target) for r in train_rows], dtype=float)
        mask = ~np.isnan(y)
        dicts = [d for d, m in zip(dicts, mask) if m]
        y = y[mask]
        if len(dicts) < 2:
            raise modelmod.NotEnoughData(
                "PhysicsLinearModel(%s) needs >=2 labeled rows, got %d" % (self.target, len(dicts)))
        groups = np.array([r.get("case_base") for r, m in zip(train_rows, mask) if m])
        n_groups = len(set(groups))

        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(dicts)
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if n_groups >= 3:
            cv = list(GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups))
            est = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=cv)
        else:
            est = Ridge(alpha=1.0)
        est.fit(X, y)

        self._vec, self._imputer, self._scaler, self._est = vec, imputer, scaler, est
        return self

    def score(self, rows: list[dict]) -> list[float]:
        dicts = [self.feature_fn(r) for r in rows]
        X = self._vec.transform(dicts)
        X = self._imputer.transform(X)
        X = self._scaler.transform(X)
        return list(self._est.predict(X))


class PhysicsCatBoostModel:
    name = "catboost_point"

    def __init__(self, target="y", feature_fn=row_features):
        self.target = target
        self.feature_fn = feature_fn

    def fit(self, train_rows: list[dict]) -> PhysicsCatBoostModel:
        if not modelmod.HAVE_CATBOOST:
            raise modelmod.NotEnoughData("catboost not available")
        dicts = [self.feature_fn(r) for r in train_rows]
        y = np.array([r.get(self.target) for r in train_rows], dtype=float)
        mask = ~np.isnan(y)
        dicts = [d for d, m in zip(dicts, mask) if m]
        y = y[mask]
        if len(dicts) < 2:
            raise modelmod.NotEnoughData(
                "PhysicsCatBoostModel(%s) needs >=2 labeled rows, got %d" % (self.target, len(dicts)))
        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(dicts)
        model = CatBoostRegressor(depth=4, l2_leaf_reg=10, iterations=500, random_seed=SEED,
                                   loss_function="RMSE", verbose=False,
                                   thread_count=modelmod.THREADS, allow_writing_files=False)
        model.fit(modelmod.Pool(X, y, thread_count=modelmod.THREADS))
        self._vec, self._model = vec, model
        return self

    def score(self, rows: list[dict]) -> list[float]:
        dicts = [self.feature_fn(r) for r in rows]
        X = self._vec.transform(dicts)
        return list(self._model.predict(X, thread_count=modelmod.THREADS))


def _label_from_ordinal_score(score):
    idx = int(round(max(0.0, min(2.0, score))))
    return FIDELITY_LABELS[idx]


# ---------------------------------------------------------------------------
# Evaluation: leave-one-case-out (primary) and leave-one-template-out.
# ---------------------------------------------------------------------------
def cross_validate_physics(rows: list[dict], group_field: str, model_factory: Callable,
                           target: str, return_models: bool = False) -> tuple:
    """Leave-one-group-out CV over group_field ('case_base', or 'template' with P00 never
    held out) -> (predictions {run_id: y}, unavailable_groups); a fold too thin to fit is
    unavailable, never raised. return_models=True adds (fold_models, fold_test_rows) for the
    jackknife+ helpers. NotEnoughGroups with fewer than 2 groups."""
    usable = [r for r in rows if r.get(target) is not None]
    if group_field == "template":
        groups = sorted({_template_of(r) for r in usable} - {None, "P00"})
    else:
        groups = sorted({r.get(group_field) for r in usable if r.get(group_field) is not None})
    if len(groups) < 2:
        raise NotEnoughGroups("%s has %d distinct group(s), need >=2" % (group_field, len(groups)))

    predictions, unavailable = {}, set()
    fold_models, fold_test_rows = {}, {}
    for g in groups:
        if group_field == "template":
            train = [r for r in usable if _template_of(r) != g]
            test = [r for r in usable if _template_of(r) == g]
        else:
            train = [r for r in usable if r.get(group_field) != g]
            test = [r for r in usable if r.get(group_field) == g]
        if not train or not test:
            unavailable.add(g)
            continue
        model = model_factory()
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
# Defect 2 (job 1077893: P03 failed in 3 of 6 cases yet ranked #1): per-template failure record.
# ---------------------------------------------------------------------------
def template_infeasibility(rows: list[dict]) -> dict[str, dict]:
    """{template: {n_cases_tried, n_cases_failed, failed_cases (sorted), rate}} across
    cases: a case FAILED when every replicate in its physics group had no usable cost (n_ok == 0,
    as infeasible_candidates). Templates never attempted are absent."""
    tried, failed = defaultdict(set), defaultdict(set)
    for key, g in performance_targets(rows).items():
        case_base = key[0]
        for tmpl, tdata in g["templates"].items():
            if tdata["n_ok"] + tdata["n_failed"] == 0:
                continue
            tried[tmpl].add(case_base)
            if tdata["n_ok"] == 0:
                failed[tmpl].add(case_base)
    out = {}
    for tmpl, cases in tried.items():
        n = len(cases)
        k = len(failed.get(tmpl, ()))
        out[tmpl] = dict(n_cases_tried=n, n_cases_failed=k,
                          failed_cases=sorted(failed.get(tmpl, ())), rate=(k / n if n else None))
    return out


# ---------------------------------------------------------------------------
# Gate A: a model-free track record the regression cannot talk itself out of (karina/P05).
# ---------------------------------------------------------------------------
def template_large_drift_rate(rows: list[dict]) -> dict[str, dict]:
    """Model-free track record from MEASURED labels only: {template:
    {n_cases_labeled, n_cases_large, large_cases (sorted), rate}}, one vote per (case, template).
    Templates with no labeled case are absent."""
    labeled, large = defaultdict(set), defaultdict(set)
    for r in physics_rows(rows):
        tmpl = _template_of(r)
        if tmpl in (None, "P00"):
            continue
        label = (r.get("fidelity_label") or "").strip()
        if label not in FIDELITY_ORDER:
            continue
        case_base = r.get("case_base")
        if case_base is None:
            continue
        labeled[tmpl].add(case_base)
        if label == "LARGE-DRIFT":
            large[tmpl].add(case_base)
    out = {}
    for tmpl, cases in labeled.items():
        n = len(cases)
        k = len(large.get(tmpl, ()))
        out[tmpl] = dict(n_cases_labeled=n, n_cases_large=k,
                          large_cases=sorted(large.get(tmpl, ())), rate=(k / n if n else None))
    return out


# ---------------------------------------------------------------------------
# Defect 3 (lafe P05 in-sample SMALL-DRIFT, measured LARGE-DRIFT): out-of-fold dangerous misses.
# ---------------------------------------------------------------------------
def per_template_dangerous_miss(rows: list[dict], allowed_labels: tuple = DEFAULT_ALLOWED_LABELS
                                ) -> dict[str, dict]:
    """Per template, how often the leave-one-case-out OUT-OF-FOLD label prediction (ridge on
    y_fidelity_label) was allowed while the measured label was LARGE-DRIFT -> {template: {n,
    dangerous_miss_n, dangerous_miss_rate, insufficient (n < MIN_OOF_DANGEROUS_MISS)}}. A template
    with no out-of-fold prediction is absent (treat as insufficient, n=0)."""
    label_rows = [r for r in physics_rows(rows) if r.get("y_fidelity_label") is not None]
    try:
        predictions, _unavailable = cross_validate_physics(
            label_rows, "case_base", lambda: PhysicsLinearModel(target="y_fidelity_label"),
            "y_fidelity_label")
    except (NotEnoughGroups, modelmod.NotEnoughData):
        predictions = {}
    pred_label = {rid: _label_from_ordinal_score(p) for rid, p in predictions.items()}
    by_tmpl = defaultdict(lambda: dict(n=0, dangerous=0))
    for r in label_rows:
        tmpl = _template_of(r)
        if tmpl in (None, "P00"):
            continue
        pl = pred_label.get(r["run_id"])
        if pl is None:
            continue
        d = by_tmpl[tmpl]
        d["n"] += 1
        if pl in allowed_labels and r.get("fidelity_label") == "LARGE-DRIFT":
            d["dangerous"] += 1
    out = {}
    for tmpl, d in by_tmpl.items():
        n = d["n"]
        out[tmpl] = dict(n=n, dangerous_miss_n=d["dangerous"],
                          dangerous_miss_rate=(d["dangerous"] / n if n else None),
                          insufficient=n < MIN_OOF_DANGEROUS_MISS)
    return out


# ---------------------------------------------------------------------------
# Defect 4: a real time-to-solution interval -- validate.py's grouped jackknife+ over the
# leave-one-case-out folds, set up once per shortlist call.
# ---------------------------------------------------------------------------
def _perf_fold_residuals(predictions, fold_test_rows):
    fold_residuals = {}
    for g, test_rows in fold_test_rows.items():
        fold_residuals[g] = [abs(r["y"] - predictions[r["run_id"]])
                              for r in test_rows if r.get("y") is not None and r["run_id"] in predictions]
    return fold_residuals


def performance_conformal_setup(perf_rows: list[dict], model_cls: type = PhysicsLinearModel
                                ) -> tuple[dict | None, dict | None, str | None]:
    """(fold_models, fold_residuals, reason). `reason` is None iff the first two are usable;
    otherwise both are None and `reason` explains why (too few case_base groups, or every fold's
    residual pool came up empty) -- never raises."""
    try:
        predictions, _unavail, fold_models, fold_test_rows = cross_validate_physics(
            perf_rows, "case_base", lambda: model_cls(target="y"), "y", return_models=True)
    except (NotEnoughGroups, modelmod.NotEnoughData) as e:
        return None, None, "time-to-solution interval unavailable: %s" % e
    fold_residuals = _perf_fold_residuals(predictions, fold_test_rows)
    if not any(fold_residuals.values()):
        return None, None, ("time-to-solution interval unavailable: no fold produced a usable "
                             "residual pool (too few labeled performance rows per case)")
    return fold_models, fold_residuals, None


def performance_conformal_interval_pct(fold_models: dict, fold_residuals: dict, row: dict,
                                       alpha: float = 0.05) -> tuple[float, float] | None:
    """Jackknife+ interval for one NEW row's time-to-solution PERCENT CHANGE vs P00 (negative =
    faster), pooling every fold's residuals against that fold's own prediction at `row`. Returns
    (lo_pct, hi_pct) or None if no fold produced a residual (should not happen once
    performance_conformal_setup itself returned non-None fold_models/fold_residuals, but checked
    defensively per row anyway since a specific `row` could still fail every fold's .score())."""
    lo_pool, hi_pool = [], []
    for g, model in fold_models.items():
        resids = fold_residuals.get(g)
        if not resids:
            continue
        pred_g = model.score([row])[0]
        for resid in resids:
            lo_pool.append(pred_g - resid)
            hi_pool.append(pred_g + resid)
    if not lo_pool:
        return None
    lo = float(np.quantile(lo_pool, alpha / 2))
    hi = float(np.quantile(hi_pool, 1 - alpha / 2))
    return (math.exp(lo) - 1) * 100.0, (math.exp(hi) - 1) * 100.0


# ---------------------------------------------------------------------------
# Per-case performance metrics: regret of the predicted-fastest candidate among those with an
# allowed MEASURED label; naive = always P00 (naive_is_best: no allowed candidate beats P00).
# ---------------------------------------------------------------------------
def per_case_perf_metrics(case_rows: list[dict], predictions: dict,
                          allowed_labels: tuple = DEFAULT_ALLOWED_LABELS) -> dict | None:
    by_template = defaultdict(list)  # template -> [(measured_y, predicted_y)]
    for r in case_rows:
        if r.get("fidelity_label") not in allowed_labels:
            continue
        if r.get("y") is None or r["run_id"] not in predictions:
            continue
        by_template[_template_of(r)].append((r["y"], predictions[r["run_id"]]))
    if not by_template:
        return None

    template_meas = {t: sum(y for y, _ in v) / len(v) for t, v in by_template.items()}
    template_pred = {t: sum(p for _, p in v) / len(v) for t, v in by_template.items()}
    measured_best_y = min(template_meas.values())
    naive_is_best = measured_best_y >= 0.0  # no allowed candidate actually beats P00

    ranked = sorted(template_pred.items(), key=lambda kv: kv[1])
    predicted_best_t = ranked[0][0]
    regret = math.exp(template_meas[predicted_best_t] - measured_best_y) - 1
    naive_regret = math.exp(0.0 - measured_best_y) - 1

    top3_eligible = len(ranked) > 3
    top3 = None
    if top3_eligible:
        is_best = lambda y: y <= measured_best_y + targets.NOISE
        top3 = any(is_best(template_meas[t]) for t, _ in ranked[:3])

    keys = list(template_meas.keys())
    correct, total = 0, 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            diff = template_meas[keys[i]] - template_meas[keys[j]]
            if abs(diff) < targets.NOISE:
                continue
            total += 1
            if (diff < 0) == (template_pred[keys[i]] < template_pred[keys[j]]):
                correct += 1
    pairwise_acc = (correct / total) if total else None

    return dict(n_allowed_templates=len(template_meas), predicted_best=predicted_best_t,
                regret=regret, naive_regret=naive_regret, naive_is_best=naive_is_best,
                top3=top3, top3_eligible=top3_eligible, pairwise_acc=pairwise_acc, pairwise_n=total)


def aggregate_perf_metrics(candidates: list[dict], predictions: dict,
                           allowed_labels: tuple = DEFAULT_ALLOWED_LABELS) -> dict:
    by_case = defaultdict(list)
    for r in candidates:
        by_case[r.get("case_base")].append(r)
    per_case = {}
    for case, rows_c in by_case.items():
        m = per_case_perf_metrics(rows_c, predictions, allowed_labels)
        if m is not None:
            per_case[case] = m
    n = len(per_case)
    if n == 0:
        return dict(n_eval_cases=0, underpowered=True, n_eval_groups_top3=0, top3_rate=None,
                    mean_regret=None, median_regret=None, worst_regret=None,
                    median_naive_regret=None, naive_is_best_rate=None, pairwise_accuracy=None,
                    per_case={})
    top3_group = [m for m in per_case.values() if m["top3_eligible"]]
    regrets = [m["regret"] for m in per_case.values()]
    naive_regrets = [m["naive_regret"] for m in per_case.values()]
    pw = [m["pairwise_acc"] for m in per_case.values() if m["pairwise_acc"] is not None]
    return dict(
        n_eval_cases=n, underpowered=n < UNDERPOWERED_MIN_CASES,
        n_eval_groups_top3=len(top3_group),
        top3_rate=(sum(m["top3"] for m in top3_group) / len(top3_group)) if top3_group else None,
        mean_regret=statistics.mean(regrets), median_regret=statistics.median(regrets),
        worst_regret=max(regrets), median_naive_regret=statistics.median(naive_regrets),
        naive_is_best_rate=sum(1 for m in per_case.values() if m["naive_is_best"]) / n,
        pairwise_accuracy=(statistics.mean(pw) if pw else None),
        per_case=per_case,
    )


# ---------------------------------------------------------------------------
# Fidelity evaluation: label confusion (dangerous miss = predicted allowed, measured LARGE-DRIFT)
# and an average-rank Spearman of predicted vs measured log10(nd) (no scipy).
# ---------------------------------------------------------------------------
def spearman_correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    ra, rb = rank(a), rank(b)
    n = len(a)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return None
    return cov / math.sqrt(var_a * var_b)


def fidelity_confusion(rows: list[dict], predicted_label_by_run_id: dict[str, str]) -> dict:
    """Confusion matrix (measured_label, predicted_label) over the 3 trainable labels, plus
    dangerous-miss stats: predicted allowed (WITHIN-FLOOR/SMALL-DRIFT) but measured LARGE-DRIFT."""
    confusion = Counter()
    dangerous = 0
    predicted_allowed_n = 0
    n = 0
    for r in rows:
        measured = r.get("fidelity_label")
        pred = predicted_label_by_run_id.get(r["run_id"])
        if measured not in FIDELITY_ORDER or pred is None:
            continue
        n += 1
        confusion[(measured, pred)] += 1
        if pred in DEFAULT_ALLOWED_LABELS:
            predicted_allowed_n += 1
            if measured == "LARGE-DRIFT":
                dangerous += 1
    return dict(n=n, confusion=dict(confusion), dangerous_miss_n=dangerous,
                predicted_allowed_n=predicted_allowed_n,
                dangerous_miss_rate=(dangerous / predicted_allowed_n) if predicted_allowed_n else None)


# ---------------------------------------------------------------------------
# Probe confusion: probe verdict (PROBE-OK vs REJECT) against measured label and speedup -- the
# check on the heuristic mp_physics REJECT (decision 0051 update).
# ---------------------------------------------------------------------------
def probe_confusion(rows: list[dict]) -> dict:
    fidelity_table, speed_table = Counter(), Counter()
    n_fidelity, n_speed = 0, 0
    for r in physics_rows(rows):
        if _template_of(r) in (None, "P00"):
            continue
        notes = r.get("_phys_notes") or parse_notes(r.get("notes"))
        pm = parse_probe(notes.get("probe"))
        if not pm:
            continue
        verdicts = list(pm.values())
        probe_bucket = "REJECT" if any(v == "REJECT" for v in verdicts) else "PROBE-OK"
        label = r.get("fidelity_label")
        if label in FIDELITY_ORDER:
            outcome = "ALLOWED" if label in DEFAULT_ALLOWED_LABELS else "LARGE-DRIFT"
            fidelity_table[(probe_bucket, outcome)] += 1
            n_fidelity += 1
        if r.get("y") is not None:
            faster = r["y"] < -math.log(1.01)
            speed_table[(probe_bucket, "FASTER>1%" if faster else "NOT")] += 1
            n_speed += 1
    return dict(n_fidelity=n_fidelity, fidelity_confusion=dict(fidelity_table),
                n_speed=n_speed, speed_confusion=dict(speed_table))


# ---------------------------------------------------------------------------
# Candidate generation for a NEW case from campaign/physics_templates.json ({"templates": {"Pxx":
# {family, edits [{op, key, value|factor}], skip_cases, extra_edits, applies_to}}}). Reports what
# WOULD change; never edits namelist text.
# ---------------------------------------------------------------------------
def load_templates(templates_json_path: str) -> dict:
    with open(templates_json_path) as fh:
        doc = json.load(fh)
    return doc.get("templates", {})


def _cast_numeric(s):
    """A configs.py edit-summary string value back to the int/float/str physics_candidates_*
    already returned before this refactor (e.g. '6' -> 6, '.false.' stays a string)."""
    if s is None:
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        pass
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _apply_edit_via_configs(edited_physics, op, max_dom=1):
    """Apply one template edit op (set/mul/delete) to edited_physics[op['key']] in place,
    through campaign/configs.apply_physics_edit on a one-line synthetic namelist -- the same
    engine that edits real namelist text (rounding, clamps, the -1 suite 'mul' rule)."""
    key = op.get("key")
    section = op.get("section", "&physics")
    current = edited_physics.get(key)
    fake_text = ("%s\n/\n" % section if current is None
                 else "%s\n %s = %s,\n/\n" % (section, key, current))
    if op.get("op") == "delete":
        edited_physics.pop(key, None)
        return
    _new_text, _old, new_active = campaign_configs.apply_physics_edit(fake_text, op, max_dom=1)
    edited_physics[key] = _cast_numeric(new_active[0])


# ---------------------------------------------------------------------------
# Generic per-case pairing (decision 0051 update): physics_templates.json's per-case rules,
# generalised to an unseen case. Expects suite-resolved native physics.
# ---------------------------------------------------------------------------
_P05_GENERIC_SWAP = {1: 5, 5: 1, 2: 1}  # YSU(1)->MYNN(5); MYNN(5)->YSU(1); MYJ(2)->YSU(1)
_AEROSOL_AWARE_MP = (28, 29)   # Thompson aerosol-aware mp_physics ids


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def generic_extra_edits(template_id: str, native_physics_resolved: dict
                        ) -> tuple[list[dict], str | None]:
    """(extra edit ops, skip_reason) for a template on a case physics_templates.json has no
    per-case entry for. P05: swap PBL + surface layer together (YSU 1 -> MYNN 5, MYNN -> YSU,
    MYJ 2 -> YSU; others skip). P04: skip when native mp is already WSM6; with Thompson
    aerosol-aware mp (28/29) also aer_opt=0, use_aero_icbc=.false. P06 with mp 28/29: aer_opt=0.
    Others: ([], None). Reproduces the JSON's six hand-written cases (test_physics.py)."""
    if template_id == "P05":
        native_pbl = _as_int(native_physics_resolved.get("bl_pbl_physics"))
        target = _P05_GENERIC_SWAP.get(native_pbl)
        if target is None:
            return [], ("no generic PBL swap rule for native bl_pbl_physics=%r (only YSU(1)"
                         "->MYNN(5), MYNN(5)->YSU(1), MYJ(2)->YSU(1) are defined)" % (native_pbl,))
        return [
            dict(op="set", key="bl_pbl_physics", value=target, section="&physics"),
            dict(op="set", key="sf_sfclay_physics", value=target, section="&physics"),
        ], None

    if template_id in ("P04", "P06"):
        native_mp = _as_int(native_physics_resolved.get("mp_physics"))
        if template_id == "P04" and native_mp == 6:
            return [], "no-op: native mp_physics already WSM6"
        edits = []
        if native_mp in _AEROSOL_AWARE_MP:
            edits.append(dict(op="set", key="aer_opt", value=0, section="&physics"))
            if template_id == "P04":
                edits.append(dict(op="set", key="use_aero_icbc", value=".false.", section="&physics"))
        return edits, None

    return [], None


def _template_case_entry(spec, case_name):
    """(has_case_entry, extra_edits, skip_reason) from physics_templates.json's own per-case
    tables and 'applies_to' allowlist ONLY (no generic fallback) -- has_case_entry is True iff this exact case_name appears
    in the template's own 'skip_cases' or 'extra_edits', i.e. JSON always wins over the generic
    rule when both could apply."""
    skip_cases = spec.get("skip_cases") or {}
    extra_edits_map = spec.get("extra_edits") or {}
    applies_to = spec.get("applies_to")
    if applies_to is not None and case_name not in applies_to:
        # an allowlist is a per-case entry too, as in configs.physics_template_applicability
        # (D04/D06: the dict engine offered bigchill-only templates for karina)
        return True, [], "not in applies_to %s" % (applies_to,)
    has_case_entry = case_name in skip_cases or case_name in extra_edits_map
    return has_case_entry, extra_edits_map.get(case_name, []), skip_cases.get(case_name)


def physics_candidates_for_case(native_physics: dict, case_name: str, max_dom: int,
                                templates_json_path: str) -> list[dict]:
    """One entry per template for a case, from its suite-resolved native physics dict:
    {template, family, applicable, skip_reason, changes {key: {from, to}}, generic_rule_used}.
    JSON per-case entries win; an unseen case falls back to generic_extra_edits. max_dom is
    unused here (scalar d01 arithmetic); physics_candidates_from_namelist edits real text."""
    templates = load_templates(templates_json_path)
    resolved_native = resolve_physics_suite(native_physics, native_physics.get("physics_suite"))
    out = []
    for tmpl_id, spec in sorted(templates.items()):
        family = spec.get("family")
        has_case_entry, json_extra, json_skip = _template_case_entry(spec, case_name)
        if has_case_entry:
            skip_reason, extra_edits, generic_used = json_skip, json_extra, False
        else:
            extra_edits, skip_reason = generic_extra_edits(tmpl_id, resolved_native)
            # flag only a real generic decision (an edit or a skip), not every fall-through
            generic_used = bool(extra_edits) or skip_reason is not None
        entry = dict(template=tmpl_id, family=family, applicable=skip_reason is None,
                     skip_reason=skip_reason, changes={}, generic_rule_used=generic_used)
        if skip_reason is not None:
            out.append(entry)
            continue
        edited_physics = dict(resolved_native)
        edits = list(spec.get("edits") or []) + list(extra_edits)
        changes = {}
        for op in edits:
            key = op.get("key")
            frm = edited_physics.get(key)
            _apply_edit_via_configs(edited_physics, op, max_dom=1)
            changes[key] = {"from": frm, "to": edited_physics.get(key)}
        entry["changes"] = changes
        out.append(entry)
    return out


def physics_candidates_from_namelist(namelist_text: str, case_name: str, max_dom: int,
                                     templates_json_path: str,
                                     native_physics_resolved: dict | None = None) -> list[dict]:
    """Same result as physics_candidates_for_case, but applied to the case's REAL namelist text
    through campaign/configs.apply_physics_template (what recommend.py uses). An unseen case needs
    native_physics_resolved for the generic rule; a template configs.py rejects (ValueError) is
    reported not applicable with that message."""
    templates = campaign_configs.load_physics_templates(templates_json_path)
    native_physics_resolved = native_physics_resolved or {}
    out = []
    for tmpl_id in sorted(templates):
        spec = templates[tmpl_id]
        family = spec.get("family")
        has_case_entry, _json_extra, json_skip = _template_case_entry(spec, case_name)
        generic_edits = []
        if has_case_entry:
            applicable, skip_reason, generic_used = (json_skip is None), json_skip, False
        else:
            generic_edits, skip_reason = generic_extra_edits(tmpl_id, native_physics_resolved)
            applicable = skip_reason is None
            # Same "only flag a real decision" rule as physics_candidates_for_case above.
            generic_used = bool(generic_edits) or skip_reason is not None
        entry = dict(template=tmpl_id, family=family, applicable=applicable,
                     skip_reason=skip_reason, changes={}, generic_rule_used=generic_used)
        if not applicable:
            out.append(entry)
            continue
        # apply_physics_template already applies template['extra_edits'][case_name] itself when
        # has_case_entry -- only synthesize a fresh 'extra_edits' override for the generic path.
        effective_spec = spec if has_case_entry else dict(spec, extra_edits={case_name: generic_edits})
        try:
            _new_text, summary = campaign_configs.apply_physics_template(
                namelist_text, effective_spec, max_dom, case_id=case_name)
        except ValueError as e:
            entry["applicable"] = False
            entry["skip_reason"] = str(e)
            out.append(entry)
            continue
        changes = {}
        for e in summary:
            if e["op"] == "delete":
                changes[e["key"]] = {"from": None, "to": None}
                continue
            frm = _cast_numeric(e["old"][0]) if e["old"] else None
            to = _cast_numeric(e["new"][0]) if e["new"] else None
            changes[e["key"]] = {"from": frm, "to": to}
        entry["changes"] = changes
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Shortlist for a case with candidates but (possibly) no runs of its own yet -- what recommend.py
# --compliance PHYSICS-GRAY calls. Never auto-applies; only ranks and labels.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ShortlistGates:
    """Thresholds that force a candidate UNCONFIRMED (see predict_physics_shortlist)."""
    allowed_labels: tuple = DEFAULT_ALLOWED_LABELS
    max_infeasible_rate: float = 0.0
    max_dangerous_miss: float = 0.0
    max_template_large_rate: float = DEFAULT_MAX_TEMPLATE_LARGE_RATE
    min_speedup_pct: float = DEFAULT_MIN_SPEEDUP_PCT


@dataclasses.dataclass
class _ShortlistEvidence:
    """Models and per-template evidence fitted ONCE per shortlist call, shared by every candidate."""
    perf_model: Any
    perf_err: str | None
    fidelity_model: Any
    fidelity_err: str | None
    label_model: Any
    label_err: str | None
    infeasibility: dict
    dangerous_miss: dict
    fold_models: Any
    fold_residuals: Any
    tts_interval_note: str | None
    track_record: dict
    same_hw: bool


def _fit_shortlist_evidence(train_rows: list[dict], case_features: dict,
                            allowed_labels: tuple) -> _ShortlistEvidence:
    rows = physics_rows(train_rows)
    no_campaign_data = not rows
    perf_rows = [r for r in rows if _template_of(r) not in (None, "P00") and r.get("y") is not None]
    nd_rows = [r for r in rows if r.get("y_fidelity_nd") is not None]
    label_rows = [r for r in rows if r.get("y_fidelity_label") is not None]

    def try_fit(target, rows_in):
        if len(rows_in) < 2:
            if no_campaign_data:
                return None, "no PHYSICS-GRAY campaign data"
            return None, "not enough %s training rows (%d)" % (target, len(rows_in))
        try:
            return PhysicsLinearModel(target=target).fit(rows_in), None
        except modelmod.NotEnoughData as e:
            return None, str(e)

    perf_model, perf_err = try_fit("y", perf_rows)
    fidelity_model, fidelity_err = try_fit("y_fidelity_nd", nd_rows)
    label_model, label_err = try_fit("y_fidelity_label", label_rows)
    infeasibility = template_infeasibility(train_rows)
    dangerous_miss = per_template_dangerous_miss(train_rows, allowed_labels=allowed_labels)
    fold_models, fold_residuals, tts_interval_note = performance_conformal_setup(perf_rows)
    track_record = template_large_drift_rate(train_rows)
    hardware_id = case_features.get("hardware_id")
    return _ShortlistEvidence(
        perf_model=perf_model, perf_err=perf_err, fidelity_model=fidelity_model,
        fidelity_err=fidelity_err, label_model=label_model, label_err=label_err,
        infeasibility=infeasibility, dangerous_miss=dangerous_miss, fold_models=fold_models,
        fold_residuals=fold_residuals, tts_interval_note=tts_interval_note,
        track_record=track_record,
        same_hw=any(r.get("hardware_id") == hardware_id for r in perf_rows))


def _not_applicable_entry(cand: dict) -> dict:
    return dict(template=cand["template"], family=cand.get("family"), status="UNCONFIRMED",
                reasons=["not applicable to this case: %s" % cand.get("skip_reason")],
                predicted_time_to_solution_pct=None, time_to_solution_interval_pct=None,
                time_to_solution_interval_note=None, predicted_y=None,
                predicted_fidelity_label=None, predicted_fidelity_nd=None,
                probe_verdict=None, same_hardware_data=None,
                infeasibility=None, dangerous_miss=None, template_track_record=None)


def _candidate_row(case_features: dict, cand: dict, probe_verdicts: dict) -> tuple[dict, dict]:
    """The case's feature row with the candidate's edits applied -> (row, edited probe verdicts).
    A namelist-only key (swint_opt, aer_opt, ...) is no ledger column, so it reaches
    physics_change_features through a synthetic edits= notes string."""
    row = dict(case_features)
    row["_phys_from"] = {k: case_features.get(k) for k in CHANGE_FIELDS}
    for key, ch in cand["changes"].items():
        if key in row or key in CHANGE_FIELDS:
            row[key] = ch["to"]
    edited_verdicts = {k: probe_verdicts[k] for k in cand["changes"] if k in probe_verdicts}
    edits_str = ";".join("%s=%s" % (k, ch["to"]) for k, ch in cand["changes"].items())
    row["_phys_notes"] = dict(
        template=cand["template"], family=cand.get("family") or "", edits=edits_str,
        probe=",".join("%s:%s" % kv for kv in edited_verdicts.items()) or "none")
    return row, edited_verdicts


def _combined_probe_verdict(verdicts: list[str]) -> str | None:
    """REJECT > CONFIRM > PASS->WRF over the edited knobs' verdicts; None without any."""
    return ("REJECT" if any(v == "REJECT" for v in verdicts) else
            "CONFIRM" if any(v == "CONFIRM" for v in verdicts) else
            "PASS->WRF" if any(v == "PASS->WRF" for v in verdicts) else None)


def _infeasibility_gate(info: dict | None, gates: ShortlistGates) -> str | None:
    """Defect 2 (job 1077893): any training-case failure rate above the limit blocks."""
    if info is not None and info["rate"] is not None and info["rate"] > gates.max_infeasible_rate:
        return ("infeasible in %d/%d training case(s): %s"
                % (info["n_cases_failed"], info["n_cases_tried"], ", ".join(info["failed_cases"])))
    return None


def _dangerous_miss_gate(info: dict | None, gates: ShortlistGates) -> str | None:
    """Defect 3: too few out-of-fold label predictions always blocks; enough evidence blocks only
    above --max-dangerous-miss."""
    if info is None or info["insufficient"]:
        return ("insufficient out-of-fold fidelity evidence for this template (n=%d, need >=%d)"
                % (info["n"] if info else 0, MIN_OOF_DANGEROUS_MISS))
    if info["dangerous_miss_rate"] > gates.max_dangerous_miss:
        return ("out-of-fold dangerous-miss rate %.2f > --max-dangerous-miss=%.2f (%d/%d "
                "out-of-fold prediction(s) predicted allowed but measured LARGE-DRIFT)"
                % (info["dangerous_miss_rate"], gates.max_dangerous_miss,
                   info["dangerous_miss_n"], info["n"]))
    return None


def _track_record_gate(info: dict | None, gates: ShortlistGates) -> str | None:
    """Gate A: a template measured LARGE-DRIFT in too many training cases blocks, whatever the
    regression predicts (karina/P05 holdout miss)."""
    if info is not None and info["rate"] is not None and info["rate"] >= gates.max_template_large_rate:
        return ("LARGE-DRIFT in %d/%d training cases: %s"
                % (info["n_cases_large"], info["n_cases_labeled"], ", ".join(info["large_cases"])))
    return None


def _speed_gate(tts_pct: float | None, interval: tuple | None, gates: ShortlistGates) -> str | None:
    """Gate B: RECOMMENDABLE needs a real predicted speedup (point <= min_speedup_pct and, when an
    interval exists, its upper bound < 0); isc21/P02 at ~0 % is not a useful recommendation."""
    if tts_pct is None:
        return "no predicted speed benefit (no performance prediction)"
    if tts_pct > gates.min_speedup_pct:
        return ("no predicted speed benefit: predicted %+.2f%% (time-to-solution vs P00), need "
                "<= %+.2f%% (--min-speedup-pct)" % (tts_pct, gates.min_speedup_pct))
    if interval is not None and interval[1] >= 0:
        return ("no predicted speed benefit: interval upper bound %+.2f%% is not < 0 (point "
                "estimate %+.2f%%)" % (interval[1], tts_pct))
    return None


def _shortlist_entry(cand: dict, ev: _ShortlistEvidence, gates: ShortlistGates,
                     case_features: dict, probe_verdicts: dict) -> dict:
    row, edited_verdicts = _candidate_row(case_features, cand, probe_verdicts)
    reasons = []
    if cand.get("generic_rule_used"):
        reasons.append(
            "this template has no physics_templates.json per-case entry for this case -- the "
            "generic per-case pairing rule (physics.generic_extra_edits) supplied its extra "
            "edit(s)/skip decision instead")
    probe_verdict = _combined_probe_verdict(list(edited_verdicts.values()))
    if probe_verdict == "REJECT":
        reasons.append("probe REJECT on an edited knob")

    predicted_y = ev.perf_model.score([row])[0] if ev.perf_model else None
    if predicted_y is None:
        reasons.append(ev.perf_err)
    # y = log(cost per simulated second vs P00), so exp(y) - 1 is the time-to-solution change
    # (negative = faster; defect 4).
    predicted_tts_pct = ((math.exp(predicted_y) - 1) * 100.0) if predicted_y is not None else None
    tts_interval_pct = None
    if predicted_y is not None and ev.fold_models is not None:
        tts_interval_pct = performance_conformal_interval_pct(ev.fold_models, ev.fold_residuals, row)

    predicted_nd = None
    if ev.fidelity_model:
        predicted_nd = 10 ** ev.fidelity_model.score([row])[0]
    else:
        reasons.append(ev.fidelity_err)
    predicted_label = (_label_from_ordinal_score(ev.label_model.score([row])[0])
                       if ev.label_model else None)
    if ev.label_model is None:
        reasons.append(ev.label_err)
    predicted_allowed = predicted_label in gates.allowed_labels if predicted_label else False
    if predicted_label and not predicted_allowed:
        reasons.append("predicted fidelity label %s not in allowed set" % predicted_label)
    if not ev.same_hw:
        reasons.append("no same-hardware training data")

    infeas_info = ev.infeasibility.get(cand["template"])
    dang_info = ev.dangerous_miss.get(cand["template"])
    track_info = ev.track_record.get(cand["template"])
    blockers = [_infeasibility_gate(infeas_info, gates), _dangerous_miss_gate(dang_info, gates),
                _track_record_gate(track_info, gates),
                _speed_gate(predicted_tts_pct, tts_interval_pct, gates)]
    reasons += [b for b in blockers if b]
    status = "RECOMMENDABLE" if (
        predicted_allowed and probe_verdict != "REJECT" and ev.same_hw and predicted_y is not None
        and not any(blockers)) else "UNCONFIRMED"
    return dict(template=cand["template"], family=cand.get("family"), status=status,
                reasons=reasons, predicted_time_to_solution_pct=predicted_tts_pct,
                time_to_solution_interval_pct=tts_interval_pct,
                time_to_solution_interval_note=ev.tts_interval_note,
                predicted_y=predicted_y, predicted_fidelity_label=predicted_label,
                predicted_fidelity_nd=predicted_nd, probe_verdict=probe_verdict,
                same_hardware_data=ev.same_hw, infeasibility=infeas_info,
                dangerous_miss=dang_info, template_track_record=track_info)


def predict_physics_shortlist(train_rows: list[dict], case_features: dict, candidates: list[dict],
                              allowed_labels: tuple = DEFAULT_ALLOWED_LABELS,
                              probe_verdicts: dict | None = None,
                              max_infeasible_rate: float = 0.0, max_dangerous_miss: float = 0.0,
                              max_template_large_rate: float = DEFAULT_MAX_TEMPLATE_LARGE_RATE,
                              min_speedup_pct: float = DEFAULT_MIN_SPEEDUP_PCT) -> list[dict]:
    """Rank and label this case's physics candidates (never applies them). train_rows: the full
    ledger with perf/fidelity targets attached; case_features: the case's (suite-resolved) P00 row.
    A candidate is RECOMMENDABLE only with an allowed predicted label, no probe REJECT, same-
    hardware data and no gate blocking (ShortlistGates); every other one is UNCONFIRMED with its
    reasons. Sorted RECOMMENDABLE first, then by predicted_y (faster first)."""
    probe_verdicts = probe_verdicts or {}
    gates = ShortlistGates(allowed_labels=allowed_labels, max_infeasible_rate=max_infeasible_rate,
                           max_dangerous_miss=max_dangerous_miss,
                           max_template_large_rate=max_template_large_rate,
                           min_speedup_pct=min_speedup_pct)
    ev = _fit_shortlist_evidence(train_rows, case_features, allowed_labels)
    out = [_shortlist_entry(cand, ev, gates, case_features, probe_verdicts)
           if cand.get("applicable", True) else _not_applicable_entry(cand)
           for cand in candidates]
    out.sort(key=lambda d: (0 if d["status"] == "RECOMMENDABLE" else 1,
                            d["predicted_y"] if d["predicted_y"] is not None else float("inf")))
    return out


# ---------------------------------------------------------------------------
# CLI: the LOCO/LOTO evaluation -> metrics.csv + summary.md (exploratory, small n).
# ---------------------------------------------------------------------------
def physics_benchmark(ledger_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rows = benchmark.load_ledger(ledger_path)
    if not physics_rows(rows):
        with open(os.path.join(out_dir, "summary.md"), "w") as fh:
            fh.write("# WTune PHYSICS-GRAY predictor benchmark\n\nNo mode=physics rows found in "
                      "%s -- physics campaign has not run yet.\n" % ledger_path)
        print("wrote summary.md (no physics rows) to %s" % out_dir)
        return dict(status="no physics rows")

    attach_change_context(rows)
    add_perf_targets(rows)
    fidelity_ds = add_fidelity_targets(rows)
    infeasible = infeasible_candidates(rows)
    suff = data_sufficiency(rows)

    candidates = [r for r in physics_rows(rows) if _template_of(r) not in (None, "P00")]
    nd_rows = [r for r in physics_rows(rows) if r.get("y_fidelity_nd") is not None]
    label_rows = [r for r in physics_rows(rows) if r.get("y_fidelity_label") is not None]

    perf_metric_rows = []
    for scheme_name, group_field in (("leave-one-case-out", "case_base"),
                                      ("leave-one-template-out", "template")):
        for model_name, factory in (("ridge", lambda: PhysicsLinearModel(target="y")),
                                     ("catboost_point", lambda: PhysicsCatBoostModel(target="y"))):
            row_out = dict(target="performance", scheme=scheme_name, model=model_name)
            try:
                predictions, unavailable = cross_validate_physics(candidates, group_field, factory, "y")
            except (NotEnoughGroups, modelmod.NotEnoughData) as e:
                row_out.update(status=type(e).__name__, reason=str(e))
                perf_metric_rows.append(row_out)
                continue
            agg = aggregate_perf_metrics(candidates, predictions)
            row_out.update(status="UNDERPOWERED" if agg["underpowered"] else "ok",
                            unavailable_folds=len(unavailable),
                            n_eval_cases=agg["n_eval_cases"], n_eval_groups_top3=agg["n_eval_groups_top3"],
                            mean_regret=agg["mean_regret"], median_regret=agg["median_regret"],
                            worst_regret=agg["worst_regret"], median_naive_regret=agg["median_naive_regret"],
                            naive_is_best_rate=agg["naive_is_best_rate"],
                            pairwise_accuracy=agg["pairwise_accuracy"], top3_rate=agg["top3_rate"])
            perf_metric_rows.append(row_out)

    fidelity_metric_rows = []
    for model_name, factory in (("ridge", lambda: PhysicsLinearModel(target="y_fidelity_nd")),
                                 ("catboost_point", lambda: PhysicsCatBoostModel(target="y_fidelity_nd"))):
        row_out = dict(target="fidelity_nd", scheme="leave-one-case-out", model=model_name)
        try:
            predictions, unavailable = cross_validate_physics(nd_rows, "case_base", factory, "y_fidelity_nd")
        except (NotEnoughGroups, modelmod.NotEnoughData) as e:
            row_out.update(status=type(e).__name__, reason=str(e))
            fidelity_metric_rows.append(row_out)
            continue
        measured = [r["y_fidelity_nd"] for r in nd_rows if r["run_id"] in predictions]
        predicted = [predictions[r["run_id"]] for r in nd_rows if r["run_id"] in predictions]
        row_out.update(status="ok" if len(predictions) >= 5 else "UNDERPOWERED", n=len(predictions),
                        spearman_rho=spearman_correlation(measured, predicted),
                        unavailable_folds=len(unavailable))
        fidelity_metric_rows.append(row_out)

    for model_name, factory in (("ridge", lambda: PhysicsLinearModel(target="y_fidelity_label")),
                                 ("catboost_point", lambda: PhysicsCatBoostModel(target="y_fidelity_label"))):
        row_out = dict(target="fidelity_label", scheme="leave-one-case-out", model=model_name)
        try:
            predictions, unavailable = cross_validate_physics(label_rows, "case_base", factory,
                                                                "y_fidelity_label")
        except (NotEnoughGroups, modelmod.NotEnoughData) as e:
            row_out.update(status=type(e).__name__, reason=str(e))
            fidelity_metric_rows.append(row_out)
            continue
        pred_label = {rid: _label_from_ordinal_score(p) for rid, p in predictions.items()}
        conf = fidelity_confusion(label_rows, pred_label)
        acc = (sum(1 for r in label_rows if pred_label.get(r["run_id"]) == r["fidelity_label"])
               / conf["n"]) if conf["n"] else None
        row_out.update(status="ok" if conf["n"] >= 5 else "UNDERPOWERED", n=conf["n"],
                        label_accuracy=acc, dangerous_miss_n=conf["dangerous_miss_n"],
                        dangerous_miss_rate=conf["dangerous_miss_rate"],
                        unavailable_folds=len(unavailable))
        fidelity_metric_rows.append(row_out)

    pconf = probe_confusion(rows)
    all_metric_rows = perf_metric_rows + fidelity_metric_rows
    benchmark.write_csv(
        os.path.join(out_dir, "metrics.csv"), all_metric_rows,
        ["target", "scheme", "model", "status", "reason", "n", "n_eval_cases", "n_eval_groups_top3",
         "mean_regret", "median_regret", "worst_regret", "median_naive_regret", "naive_is_best_rate",
         "pairwise_accuracy", "top3_rate", "spearman_rho", "label_accuracy", "dangerous_miss_n",
         "dangerous_miss_rate", "unavailable_folds"])

    lines = ["# WTune PHYSICS-GRAY predictor benchmark (EXPLORATORY -- see decision 0051; n is "
             "small by construction until the physics campaign lands)\n"]
    lines.append("- data sufficiency: %d qualifying case(s) (>=3 candidates each) of %d needed -> %s"
                  % (suff["n_qualifying_cases"], UNDERPOWERED_MIN_CASES,
                     "UNDERPOWERED" if suff["underpowered"] else "ok"))
    lines.append("- infeasible candidates (every replicate FAILED): %d -> %s"
                  % (len(infeasible), infeasible))
    lines.append("- fidelity labels: %d trainable, %d INVALID, %d unlabeled/FAILED"
                  % (len(fidelity_ds["trainable"]), len(fidelity_ds["invalid"]), len(fidelity_ds["unlabeled"])))
    lines.append("\n## Performance (leave-one-case-out / leave-one-template-out)\n")
    for m in perf_metric_rows:
        lines.append("- %s/%s: status=%s median_regret=%s naive_median_regret=%s pairwise_acc=%s"
                      % (m["scheme"], m["model"], m["status"], m.get("median_regret"),
                         m.get("median_naive_regret"), m.get("pairwise_accuracy")))
    lines.append("\n## Fidelity\n")
    for m in fidelity_metric_rows:
        lines.append("- %s/%s: status=%s n=%s spearman_rho=%s label_accuracy=%s dangerous_miss_rate=%s"
                      % (m["target"], m["model"], m["status"], m.get("n"), m.get("spearman_rho"),
                         m.get("label_accuracy"), m.get("dangerous_miss_rate")))
    lines.append("\n## Probe verdict vs measured outcome\n")
    lines.append("- fidelity confusion (probe x allowed/LARGE-DRIFT), n=%d: %s"
                  % (pconf["n_fidelity"], pconf["fidelity_confusion"]))
    lines.append("- speed confusion (probe x >1%% faster), n=%d: %s"
                  % (pconf["n_speed"], pconf["speed_confusion"]))
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("wrote metrics.csv (%d rows), summary.md to %s" % (len(all_metric_rows), out_dir))
    return dict(status="ok", perf_metric_rows=perf_metric_rows, fidelity_metric_rows=fidelity_metric_rows,
                probe_confusion=pconf, data_sufficiency=suff, infeasible=infeasible)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    np.random.seed(SEED)
    physics_benchmark(args.ledger, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
