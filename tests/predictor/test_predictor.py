#!/usr/bin/env python3
"""predictor/tests/test_predictor.py -- pytest suite for phase A of the WTune config predictor.

Run with:
    pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml \\
        python -m pytest predictor/tests/test_predictor.py -v
"""
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wtune.predictor import benchmark
from wtune.predictor import features as feat
from wtune.predictor import ledger
from wtune.predictor import targets
from wtune.predictor import validate

HARDWARE_ID = ledger.HARDWARE_ID


# ---------------------------------------------------------------------------
# Synthetic ledger: 8 cases x 12 configs from a known multiplicative model + noise.
#   log(s_per_step) = base(case) + f(config) + noise
# f(config) is shared across all cases (varies only `numtiles`; everything else constant), so
# once y = log(s/s_baseline) removes each case's own base(case), the mapping config -> y is
# IDENTICAL across cases by construction -- exactly the separable structure the real corpus is
# claimed to have (2026-09-07 config-cost separability note), so a linear model in log space
# should achieve near-zero leave-one-case-out regret.
#
# f(config) MUST be an exact linear combination of the features ridge actually sees for a given
# `numtiles`: the raw "numtiles" feature (linear in nt) and "cells_per_tile" (features.py
# config_features, proportional to 1/nt -- everything else patch/decomp-derived is constant
# across this sweep since only numtiles varies). c1*nt + c2/nt is minimized at nt=sqrt(c2/c1);
# c2 = 13**2 * c1 puts that minimum exactly at nt=13. (2026-09-14: a previous version used
# 0.02*abs(nt-13), a V-shape with a kink at nt=13 that is NOT representable by any linear
# combination of the smooth numtiles/cells_per_tile features -- ridge could only fit a smooth
# curve through the kink, and the resulting bias alone produced ~1.9% median LOCO regret,
# failing the <=1% bar regardless of alpha. See predictor/tests -- ridge-regret investigation.)
# ---------------------------------------------------------------------------
CASES = ["caseA", "caseB", "caseC", "caseD", "caseE", "caseF", "caseG", "caseH"]
NUMTILES_VALUES = list(range(8, 20))  # 12 configs
BASE_LOG_S = {c: 0.10 * i for i, c in enumerate(CASES)}       # distinct case-level intercepts
F_CONFIG = {nt: 0.1 * nt + 16.9 / nt for nt in NUMTILES_VALUES}  # optimum at numtiles=13


def make_synthetic_ledger(noise_std=0.001, seed=0, with_variants=False):
    rng = np.random.RandomState(seed)
    rows = []
    for c in CASES:
        for nt in NUMTILES_VALUES:
            noise = rng.normal(0, noise_std)
            log_s = BASE_LOG_S[c] + F_CONFIG[nt] + noise
            case_raw = c
            if with_variants and nt == NUMTILES_VALUES[0]:
                case_raw = "%s__mp6bl1" % c  # a scheme-variant alias of the same case
            row = ledger.new_row(
                run_id="%s:nt%d" % (case_raw, nt), source_dataset="synthetic",
                job_id=None, rep_index=0,
                case_raw=case_raw, case_base=ledger.case_base_of(case_raw),
                case_family=ledger.case_family_of(ledger.case_base_of(case_raw)),
                hardware_id=HARDWARE_ID,
                binary_kind="pristine", compliance_class="STRICT",
                e_we=901.0, e_sn=901.0, e_vert=45.0, dx=3000.0, dy=3000.0, time_step=15.0,
                mp_physics=8, cu_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1,
                sf_surface_physics=2, ra_lw_physics=4, ra_sw_physics=4, radt=15.0, cudt=None,
                history_interval=None, io_form_history=None,
                n_nodes=4, n_ranks=448, omp_threads=1, nproc_x=8, nproc_y=56, numtiles=nt,
                distribution="cyclic", kmp_blocktime=None,
                s_per_step_median=math.exp(log_s), s_per_step_mean=math.exp(log_s),
                n_steps_measured=40, diffwrf_differing_fields=None, fidelity_label=None,
                profile_micro_share=None, profile_radiation_share=None,
                profile_dynamics_share=None, profile_mpi_share=None, profile_io_share=None,
                profile_rank_imbalance=None, profile_thread_imbalance=None,
                notes="synthetic",
            )
            rows.append(row)
    baselines = targets.compute_case_baselines(rows)
    targets.add_y(rows, baselines)
    return rows, baselines


# ---------------------------------------------------------------------------
# 1. ledger schema validation
# ---------------------------------------------------------------------------
def test_ledger_schema_validation(tmp_path):
    rows, _ = make_synthetic_ledger()
    out = tmp_path / "runs.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ledger.LEDGER_FIELDS})
    assert ledger.validate(str(out)) is True

    bad = tmp_path / "bad.csv"
    with open(bad, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        row = dict(rows[0])
        row["s_per_step_median"] = "-1.0"
        row["case_base"] = ""
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in ledger.LEDGER_FIELDS})
    assert ledger.validate(str(bad)) is False


def test_case_base_and_family_variants():
    assert ledger.case_base_of("karina_cat4") == "karina"
    assert ledger.case_base_of("karina_48h") == "karina"
    assert ledger.case_base_of("karina_14x64") == "karina"
    assert ledger.case_base_of("decay_karina_cat4") == "karina"
    assert ledger.case_base_of("probe_karina_cat4") == "karina"
    assert ledger.case_base_of("deep_conv_ot__mp10bl1") == "deep_conv_ot"
    assert ledger.case_base_of("arctic_sbl_16x28h12") == "arctic_sbl"
    assert ledger.case_family_of("karina") == "cyclone"


# ---------------------------------------------------------------------------
# 7. conformal coverage ~ nominal on synthetic data
# ---------------------------------------------------------------------------
def test_conformal_coverage_approximately_nominal():
    rows, _ = make_synthetic_ledger(noise_std=0.01, seed=1)
    intervals, coverage, predictions = validate.conformal_intervals(
        rows, "ridge", alphas=(0.10,), scope="STRICT")
    cov = coverage[0.10]
    assert cov["rate"] is not None
    # tiny n (8 groups): report honestly, only check it's in a broad plausible band, not exact 90%
    assert 0.3 <= cov["rate"] <= 1.0
    # the number of held-out points behind the rate must be reported and must match reality
    assert cov["n"] == sum(1 for r in rows if r.get("y") is not None
                            and r["run_id"] in intervals and 0.10 in intervals[r["run_id"]])
    # 96 synthetic points (8 cases x 12 configs) is well above the n=30 underpowered floor
    assert cov["n"] >= 30
    assert cov["underpowered"] is False


# ---------------------------------------------------------------------------
# 9. SAFE / EXPERIMENTAL rule
# ---------------------------------------------------------------------------
def test_safe_experimental_rule():
    safe = validate.recommendation_label(y_pred=-0.05, upper_y_bound=-0.01)
    assert safe["label"] == "SAFE"
    assert safe["gain_lower_95"] > 0
    experimental = validate.recommendation_label(y_pred=-0.05, upper_y_bound=0.02)
    assert experimental["label"] == "EXPERIMENTAL"
    assert experimental["gain_lower_95"] <= 0


# ---------------------------------------------------------------------------
# Helper shared by the decision-group tests below: an "8-node" variant of caseA's numtiles
# sweep, twice the ranks, uniformly faster purely because it has 2x the hardware -- not because
# any knob is "better". These must never be treated as alternative configs of the 4-node rows.
# ---------------------------------------------------------------------------
def _make_8node_variant(case_rows):
    variant_rows = []
    for r in case_rows:
        v = dict(r)
        v["run_id"] = r["run_id"] + ":8n"
        v["n_nodes"], v["n_ranks"] = 8, 896
        v["s_per_step_median"] = r["s_per_step_median"] / 2.0
        v["s_per_step_mean"] = v["s_per_step_median"]
        variant_rows.append(v)
    return variant_rows


# ---------------------------------------------------------------------------
# 12. decision groups separate different resource allocations (n_nodes/total_cores): a 4-node
# and an 8-node config of the same case_base must never produce a pairwise label or be counted
# in the same regret/top-k group.
# ---------------------------------------------------------------------------
def test_decision_group_separates_node_counts():
    rows, _ = make_synthetic_ledger(noise_std=0.0)
    case_rows = [r for r in rows if r["case_base"] == "caseA"]
    variant_rows = _make_8node_variant(case_rows)
    all_rows = case_rows + variant_rows
    baselines = targets.compute_case_baselines(all_rows)
    targets.add_y(all_rows, baselines)

    groups = {targets.scope_key(r) for r in all_rows}
    assert len(groups) == 2  # one decision group per node-count, same case_base

    # pairwise_labels draws pairs only from aggregate_by_config's per-group config dict, so no
    # pair can ever compare a 4-node config to an 8-node one
    pairs = targets.pairwise_labels(all_rows)
    assert len(pairs) == 2 * math.comb(len(NUMTILES_VALUES), 2)  # 12-choose-2 pairs PER group
    assert all(p["scope"] in groups for p in pairs)

    # feeding both groups' rows together (as validate.aggregate_metrics does from raw ledger
    # rows) must still score them as two separate 12-config groups, never one 24-config group
    predictions = {r["run_id"]: r["y"] for r in all_rows}  # perfect predictions
    by_group = defaultdict(list)
    for r in all_rows:
        by_group[feat.decision_group_key(r)].append(r)
    assert len(by_group) == 2
    for grp in by_group.values():
        m = validate.per_case_metrics(grp, predictions)
        assert m["n_configs"] == 12

    agg = validate.aggregate_metrics(all_rows, predictions)
    assert agg["n_eval_cases"] == 2  # 2 decision groups, not 1 case_base
    assert all(m["n_configs"] == 12 for m in agg["per_case"].values())


# ---------------------------------------------------------------------------
# 15. UNDERPOWERED rule and "no model selected" behaviour
# ---------------------------------------------------------------------------
def test_underpowered_no_model_selected():
    rows, _ = make_synthetic_ledger(noise_std=0.001, seed=3)
    few_rows = [r for r in rows if r["case_base"] in CASES[:3]]  # only 3 cases: always <5
    baselines = targets.compute_case_baselines(few_rows)
    targets.add_y(few_rows, baselines)
    metric_rows, per_case_rows, best_by_scope, underpowered_scopes, ledger_notes = \
        benchmark.run_metrics_matrix(few_rows)
    assert "STRICT" in underpowered_scopes
    assert "STRICT" not in best_by_scope  # no model selected from underpowered metrics
    loco_strict = [m for m in metric_rows
                   if m["scope"] == "STRICT" and m["scheme"] == "leave-one-case-out"
                   and m.get("n_eval_cases") is not None]
    assert loco_strict  # evaluate() actually ran (not NotEnoughGroups/NotEnoughData for all)
    for m in loco_strict:
        assert m["n_eval_cases"] < 5
        assert m["status"] == "UNDERPOWERED"


# ---------------------------------------------------------------------------
# campaign ledger adapter: rows already in LEDGER_FIELDS; failed runs are skipped, types
# are restored, and the per-pool hardware_id is kept (never overwritten with HARDWARE_ID)
# ---------------------------------------------------------------------------
def test_campaign_ledger_adapter_types_and_failed_runs(tmp_path):
    base = {k: "" for k in ledger.LEDGER_FIELDS}
    base.update(run_id="lafe_C02_r1_1", source_dataset="strict_campaign", job_id="1",
                rep_index="1", case_raw="lafe", case_base="lafe", case_family="boundary_layer_fine_scale",
                hardware_id="f1-spr-icpnp", binary_kind="pristine", compliance_class="STRICT",
                e_we="1000", e_sn="1000", e_vert="100", dx="2500", dy="2500", time_step="10",
                io_form_history="14", n_nodes="4", n_ranks="224", omp_threads="2",
                nproc_x="14", nproc_y="16", numtiles="1", distribution="block", kmp_blocktime="0",
                s_per_step_median="1.31462", s_per_step_mean="1.9", n_steps_measured="20")
    failed = dict(base, run_id="lafe_C07_r1_1", numtiles="14", s_per_step_median="",
                  s_per_step_mean="", n_steps_measured="0")
    path = tmp_path / "ledger_lafe_1.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS)
        w.writeheader()
        w.writerow(base)
        w.writerow(failed)

    rows = ledger.adapt_campaign_ledger(str(path))
    assert len(rows) == 1
    r = rows[0]
    assert r["hardware_id"] == "f1-spr-icpnp"
    assert r["n_ranks"] == 224 and r["omp_threads"] == 2 and r["numtiles"] == 1
    assert r["s_per_step_median"] == pytest.approx(1.31462)
    assert feat.decision_group_key(r) is not None


# ---------------------------------------------------------------------------
# kmp_blocktime: a campaign ledger CSV cell is a plain string ("0", "default", or ""); recommend.py
# builds in-memory candidate rows with kmp_blocktime as {None, int} (None means unset/"default").
# Before normalizing in the adapter, features.config_key((..., "0", ...)) != (..., 0, ...) for what
# is otherwise the identical configuration -- this asserts the two now compare equal.
# ---------------------------------------------------------------------------
def test_campaign_kmp_blocktime_normalizes_to_match_in_memory_config_key(tmp_path):
    base = {k: "" for k in ledger.LEDGER_FIELDS}
    base.update(run_id="conus25_C02_r1_1", source_dataset="strict_campaign", case_raw="conus25",
                case_base="conus25", hardware_id="f1-spr-icpnp", binary_kind="pristine",
                compliance_class="STRICT", n_nodes="4", n_ranks="224", omp_threads="2",
                nproc_x="14", nproc_y="16", numtiles="8", distribution="cyclic",
                kmp_blocktime="0", s_per_step_median="1.0")
    default_row = dict(base, run_id="conus25_C03_r1_1", numtiles="9", kmp_blocktime="default")
    path = tmp_path / "ledger_conus25_kmp.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS)
        w.writeheader()
        w.writerow(base)
        w.writerow(default_row)

    rows = ledger.adapt_campaign_ledger(str(path))
    campaign_row = next(r for r in rows if r["run_id"] == "conus25_C02_r1_1")
    campaign_default_row = next(r for r in rows if r["run_id"] == "conus25_C03_r1_1")
    assert campaign_row["kmp_blocktime"] == 0 and isinstance(campaign_row["kmp_blocktime"], int)
    assert campaign_default_row["kmp_blocktime"] is None

    in_memory_row = ledger.new_row(
        run_id="candidate", case_base="conus25", hardware_id="f1-spr-icpnp",
        n_nodes=4, n_ranks=224, omp_threads=2, nproc_x=14, nproc_y=16, numtiles=8,
        distribution="cyclic", kmp_blocktime=0)
    assert feat.config_key(campaign_row) == feat.config_key(in_memory_row)

    in_memory_default = ledger.new_row(
        run_id="candidate2", case_base="conus25", hardware_id="f1-spr-icpnp",
        n_nodes=4, n_ranks=224, omp_threads=2, nproc_x=14, nproc_y=16, numtiles=9,
        distribution="cyclic", kmp_blocktime=None)
    assert feat.config_key(campaign_default_row) == feat.config_key(in_memory_default)


# ---------------------------------------------------------------------------
# campaign rows: correctness-mode rows are not timing replicates, and a campaign's physics is not
# judged against another dataset's modal physics for the same case_base (2026-09-14 fixes)
# ---------------------------------------------------------------------------
def test_campaign_correctness_rows_skipped_and_classification_scoped(tmp_path):
    base = {k: "" for k in ledger.LEDGER_FIELDS}
    base.update(source_dataset="strict_campaign", rep_index="1", case_raw="karina",
                case_base="karina", hardware_id="f1-spr-icpnp", binary_kind="pristine",
                compliance_class="STRICT", mp_physics="8", io_form_history="14", n_nodes="4",
                n_ranks="448", omp_threads="1", nproc_x="16", nproc_y="28", distribution="cyclic",
                kmp_blocktime="0", s_per_step_median="0.28", n_steps_measured="30",
                notes="config=C01 mode=timing")
    timing = [dict(base, run_id="k_C01_r%d" % i, numtiles="1") for i in (1, 2)]
    timing += [dict(base, run_id="k_C02_r%d" % i, numtiles="8") for i in (1, 2)]
    corr = dict(base, run_id="k_C02_corr", numtiles="8", notes="config=C02 mode=correctness")
    path = tmp_path / "ledger_karina_9.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS)
        w.writeheader()
        for r in timing + [corr]:
            w.writerow(r)
    rows = ledger.adapt_campaign_ledger(str(path))
    assert len(rows) == 4 and not any(r["run_id"] == "k_C02_corr" for r in rows)

    # an older dataset for the same case_base, different hardware and different physics, with more
    # rows so its physics would be the modal config if datasets were pooled
    old = []
    for i in range(6):
        old.append(ledger.new_row(run_id="old%d" % i, source_dataset="old", case_raw="karina",
                                  case_base="karina", hardware_id="f1-spr-icpnq",
                                  binary_kind="pristine", mp_physics=6, n_nodes=4, n_ranks=448,
                                  omp_threads=1, nproc_x=16, nproc_y=28, numtiles=1,
                                  distribution="cyclic", kmp_blocktime="0",
                                  s_per_step_median=0.3))
    ledger.classify_rows(rows + old)
    assert all(r["compliance_class"] == "STRICT" for r in rows)


def _physics_ledger_row(run_id, template, notes_extra, **over):
    base = {k: "" for k in ledger.LEDGER_FIELDS}
    base.update(run_id=run_id, source_dataset="strict_campaign", rep_index="1",
                case_raw="conus12", case_base="conus12", hardware_id="f1-spr-icpnq",
                binary_kind="pristine", compliance_class="PHYSICS-GRAY", e_we="425", e_sn="300",
                e_vert="50", dx="12000", dy="12000", time_step="72", mp_physics="8",
                io_form_history="14", n_nodes="4", n_ranks="448", omp_threads="1", nproc_x="16",
                nproc_y="28", numtiles="8", distribution="cyclic", kmp_blocktime="0",
                s_per_step_median="1.25", n_steps_measured="30", fidelity_label="SMALL-DRIFT",
                notes="mode=physics template=%s decomp=C02 %s" % (template, notes_extra))
    base.update(over)
    return base


def test_fidelity_fields_do_not_widen_the_written_csv(tmp_path):
    """The whole point of using `notes` was to leave the 48-column header alone: build()'s
    DictWriter uses extrasaction='ignore', so the extra row keys must never reach the file."""
    src = tmp_path / "ledger_conus12_fid2.csv"
    with open(src, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS)
        w.writeheader()
        w.writerow(_physics_ledger_row("c12_A02_r1_1", "A02",
                                       "fidelity_nd_max=0.9 fidelity_floor_ratio_max=9.0"))
    out = tmp_path / "out" / "runs.csv"
    ledger.build(str(tmp_path / "no_such_wiki_dir"), str(out), campaign_ledgers=[str(src)])
    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ledger.LEDGER_FIELDS
        written = next(reader)
    assert "fidelity_nd_max" not in written
    assert "fidelity_nd_max=0.9" in written["notes"]
