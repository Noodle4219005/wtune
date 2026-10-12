#!/usr/bin/env python3
"""predictor/targets.py -- case baselines, log-ratio targets, and pairwise labels.

Everything here works in log space, normalized within a case, per
knowledge/experiments/2026-09-07-config-cost-is-multiplicatively-separable-to-2-percent.md
(cost is multiplicatively separable across case/physics -> additive in log space) and per
knowledge/2026-09-05-loco-over-directories-leaks-...md (group by case_base, never by directory).

Stdlib only (math). Rows are the dicts produced by ledger.py (RESEARCH rows are expected to
already be excluded by the caller before any of this runs -- see validate.py).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from wtune.predictor.features import config_key, decision_group_key, decision_group_known

NOISE = math.log(1.005)  # run-to-run reproducibility floor (0.4-1.6% measured on 40-step karina
                          # runs); |log ratio| below this is not a reliable ordering, not a tie
                          # of "no difference" -- see the config-cost separability note.


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def scope_key(row):
    """The decision group: the resource allocation + comparison scope within which configs are
    directly comparable (baseline/ranking/pairwise/regret). See features.decision_group_key --
    (case_base, hardware_id, compliance_class, n_nodes, total_cores, binary_kind,
    io_form_history). A 4-node and an 8-node run of the same case_base are different scopes: the
    8-node run is faster because it has twice the hardware, not because its knobs are better.
    Returns None for a row whose resource allocation cannot be determined; every caller below
    skips such rows rather than merging them into a bogus catch-all group."""
    return decision_group_key(row)


def count_unknown_decision_group(rows):
    """Count of rows with a usable s_per_step_median whose decision group could not be
    determined (n_nodes/n_ranks/omp_threads/hardware_id missing) -- these are silently dropped
    from every baseline/ranking/pairwise/regret computation; this makes that exclusion visible."""
    return sum(1 for r in rows
               if _num(r.get("s_per_step_median")) not in (None,) and _num(r.get("s_per_step_median")) > 0
               and not decision_group_known(r))


def count_io_form_splits(rows):
    """How many (case_base, hardware_id, compliance_class, n_nodes, total_cores, binary_kind)
    groups contain more than one distinct io_form_history value -- i.e. how many groups the
    io_form_history component of the decision-group key actually splits apart. Only rows with a
    known decision group and a usable measurement count."""
    coarse = defaultdict(set)
    for r in rows:
        s = _num(r.get("s_per_step_median"))
        if s is None or s <= 0 or not decision_group_known(r):
            continue
        key = decision_group_key(r)  # (..., binary_kind, io_form_history)
        coarse[key[:-1]].add(key[-1])
    return sum(1 for vals in coarse.values() if len(vals) > 1)


def aggregate_by_config(rows):
    """Group rows by (decision group, config_key); average log(s_per_step_median) across
    replicates of the same config ("measured runtime of a config" per the brief). Returns
    {group: {config_key: {"mean_log_s": float, "n": int, "rows": [row, ...]}}}.
    Rows with no usable s_per_step_median, or an unresolvable decision group, are dropped."""
    out = defaultdict(dict)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        s = _num(row.get("s_per_step_median"))
        if s is None or s <= 0:
            continue
        sk = scope_key(row)
        if sk is None:
            continue  # unknown resource allocation: not comparable to anything
        grouped[sk][config_key(row)].append((math.log(s), row))
    for scope, configs in grouped.items():
        for ck, entries in configs.items():
            logs = [e[0] for e in entries]
            out[scope][ck] = {
                "mean_log_s": sum(logs) / len(logs),
                "n": len(logs),
                "rows": [e[1] for e in entries],
            }
    return out


def _reference_order(ck):
    """Outcome-independent tie-break for the baseline config: prefer the plain-MPI default
    (omp_threads 1, numtiles 1/unset), then lexical order. Never use measured speed here: the
    campaign runs every config the same number of times, so a speed tie-break made the baseline
    the measured-fastest config and the naive baseline's regret 0 by construction (2026-09-14)."""
    omp, numtiles = _num(ck[3]), _num(ck[2])
    return (omp not in (None, 1), numtiles not in (None, 1), tuple(str(v) for v in ck))


def compute_case_baselines(rows, overrides=None):
    """baseline = the most frequent configuration within (case_base, hardware_id, compliance
    scope), ties broken by _reference_order (outcome-independent). `overrides`
    is an optional {scope: config_key} map that forces a specific baseline (e.g. for a case where
    the modal config is not the intended "normal" one).
    Returns {scope: {"config_key": ck, "s_baseline": float, "n_configs": int}}."""
    overrides = overrides or {}
    agg = aggregate_by_config(rows)
    baselines = {}
    raw_counts = defaultdict(Counter)
    for row in rows:
        if _num(row.get("s_per_step_median")) is None:
            continue
        sk = scope_key(row)
        if sk is None:
            continue  # unknown resource allocation: excluded from baseline voting too
        raw_counts[sk][config_key(row)] += 1

    for scope, configs in agg.items():
        if scope in overrides and overrides[scope] in configs:
            ck = overrides[scope]
        else:
            counts = raw_counts[scope]
            best_n = max(counts.values())
            candidates = [c for c, n in counts.items() if n == best_n]
            ck = min(candidates, key=_reference_order)
        baselines[scope] = {
            "config_key": ck,
            "s_baseline": math.exp(configs[ck]["mean_log_s"]),
            "n_configs": len(configs),
        }
    return baselines


def add_y(rows, baselines):
    """Attach y = log(s_per_step_median / s_baseline) to each row in place. Rows whose scope has
    no baseline (e.g. RESEARCH, or a scope with zero usable rows) get y=None."""
    for row in rows:
        s = _num(row.get("s_per_step_median"))
        b = baselines.get(scope_key(row))
        if s is None or s <= 0 or b is None:
            row["y"] = None
        else:
            row["y"] = math.log(s / b["s_baseline"])
    return rows


def regress_1pct(row):
    """True if this row regresses (is slower than baseline) by more than 1%."""
    y = row.get("y")
    return None if y is None else y > math.log(1.01)


def pairwise_labels(rows, noise=NOISE):
    """All (A, B) config pairs within the same scope, at the aggregated-per-config level.
    Returns a list of dicts: scope, config_a, config_b, label (1 if A faster than B, 0 if slower),
    tie (bool), log_ratio = mean_log_s_a - mean_log_s_b."""
    agg = aggregate_by_config(rows)
    out = []
    for scope, configs in agg.items():
        keys = sorted(configs.keys(), key=repr)  # deterministic order
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                diff = configs[a]["mean_log_s"] - configs[b]["mean_log_s"]
                tie = abs(diff) < noise
                out.append({
                    "scope": scope, "config_a": a, "config_b": b,
                    "log_ratio": diff, "tie": tie,
                    "label": None if tie else (1 if diff < 0 else 0),
                })
    return out
