#!/usr/bin/env python3
"""predictor/tests/test_decomp_search.py -- pytest suite for predictor/decomp_search.py
(WTune D3: active decomposition search).

Run with:
    pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml \\
        python -m pytest predictor/tests/test_decomp_search.py -v
"""
import csv
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wtune.predictor import benchmark
from wtune.predictor import decomp_search as dsearch
from wtune.predictor import ledger as ledgermod
from wtune.predictor import models as modelmod
from wtune.predictor import targets
import configs as campaign_configs   # decomp_search.py itself inserts campaign/ onto sys.path

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

HARDWARE_ID = "f1-spr-icpnp"

CASE_FIELDS = dict(
    e_we=901, e_sn=901, e_vert=45, dx=3000, dy=3000, time_step=15,
    mp_physics=8, cu_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1, sf_surface_physics=2,
    ra_lw_physics=4, ra_sw_physics=4, radt=15, cudt=0, history_interval=None, io_form_history=14,
)


def _mk_cfg(ranks, omp, nx, ny, numtiles, tile_shape="1d_y", dist="cyclic", blocktime="0"):
    return dict(ranks=ranks, omp=omp, nx=nx, ny=ny, numtiles=numtiles, tile_shape=tile_shape,
                numtiles_x=0, numtiles_y=0, tile_sz_x=0, tile_sz_y=0, tile_strategy=0,
                dist=dist, blocktime=blocktime)


# ---------------------------------------------------------------------------
# propose_core: stop rule
# ---------------------------------------------------------------------------
def _labeled_rows(cfgs, ys, case_name="prior"):
    return [dict(dsearch.candidate_row(case_name, CASE_FIELDS, HARDWARE_ID, 4, c,
                                        run_id="r:%d" % i), y=y)
            for i, (c, y) in enumerate(zip(cfgs, ys))]


def test_propose_core_stops_at_max_rounds():
    cfgs = [_mk_cfg(448, 1, 8, 56, nt) for nt in (1, 8)]
    labeled = _labeled_rows(cfgs, [0.0, -0.01])
    state = dict(configs={}, measured={}, rounds=[{"round": 1}, {"round": 2}])
    result = dsearch.propose_core(
        cfgs, labeled, "case", CASE_FIELDS, HARDWARE_ID, 4, state, batch_size=4,
        incumbent_cfg=None, incumbent_y=None, max_rounds=2)
    assert result["stopped"] is True
    assert "max_rounds" in result["reason"]
    assert result["round"] == 3


# ---------------------------------------------------------------------------
# state round-trip
# ---------------------------------------------------------------------------
def test_state_round_trip(tmp_path):
    path = str(tmp_path / "state.json")
    assert dsearch.load_state(path) == dict(configs={}, measured={}, rounds=[])
    state = dict(configs={"G0001": _mk_cfg(448, 1, 8, 56, 8)},
                 measured={"G0001": [{"run_id": "x", "rep_index": 0, "s_per_step_median": 0.29}]},
                 rounds=[{"round": 1, "proposed_ids": ["G0001"]}])
    dsearch.save_state(path, state)
    loaded = dsearch.load_state(path)
    assert loaded == state


def _write_campaign_style_ledger(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledgermod.LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ledgermod.LEDGER_FIELDS})


def test_ingest_records_measurements_and_skips_unknown_ids(tmp_path):
    state = dict(configs={"R01D0001": _mk_cfg(448, 1, 8, 56, 1)}, measured={}, rounds=[])
    rows = [
        ledgermod.new_row(
            run_id="karina_R01D0001_r1_1078123", source_dataset="strict_campaign",
            job_id="1078123", rep_index=1, case_raw="karina", case_base="karina",
            case_family="tropical_cyclone", hardware_id=HARDWARE_ID, binary_kind="pristine",
            compliance_class="STRICT", e_we=901, e_sn=901, e_vert=45, dx=3000, dy=3000,
            time_step=15, n_nodes=4, n_ranks=448, omp_threads=1, nproc_x=8, nproc_y=56,
            numtiles=1, distribution="cyclic", kmp_blocktime=0,
            s_per_step_median=0.29, s_per_step_mean=0.30, n_steps_measured=31,
            notes="config=R01D0001 mode=timing config_source=generated numtiles_x=0 "
                  "numtiles_y=0 tile_sz_x=0 tile_sz_y=0 tile_strategy=0"),
        # unknown config id -- must be skipped, not crash
        ledgermod.new_row(
            run_id="karina_UNKNOWNID_r1_1078123", source_dataset="strict_campaign",
            job_id="1078123", rep_index=1, case_raw="karina", case_base="karina",
            hardware_id=HARDWARE_ID, binary_kind="pristine", compliance_class="STRICT",
            e_we=901, e_sn=901, e_vert=45, n_nodes=4, n_ranks=224, omp_threads=2,
            nproc_x=4, nproc_y=56, numtiles=1, distribution="cyclic", kmp_blocktime=0,
            s_per_step_median=0.31, notes="config=UNKNOWNID mode=timing"),
    ]
    path = tmp_path / "new_ledger.csv"
    _write_campaign_style_ledger(rows, str(path))

    n_in, n_skip, _reasons = dsearch.ingest(state, str(path))
    assert n_in == 1
    assert n_skip == 1
    assert state["measured"]["R01D0001"][0]["s_per_step_median"] == 0.29
    assert state["measured"]["R01D0001"][0]["rep_index"] == 1


# ---------------------------------------------------------------------------
# Ensemble uncertainty (referenced by decomp_search.py's own module docstring).
# ---------------------------------------------------------------------------
class TestEnsembleUncertainty:

    def test_fit_ensemble_raises_not_enough_data(self):
        with pytest.raises(modelmod.NotEnoughData):
            dsearch.fit_ensemble([], n_models=4)


# ---------------------------------------------------------------------------
# Synthetic ground-truth pool: the search must find the known optimum in fewer runs than random.
# ---------------------------------------------------------------------------
def _synthetic_pool():
    """12 candidates over (ranks,omp) in {(448,1),(224,2),(112,4),(64,7)} x numtiles in
    {1,8,16} (tile_shape='1d_y' throughout -- the axis under test is ranks/omp/numtiles, not
    tiling). True y = 0.01*|omp-2| + 0.002*(numtiles-8)^2/64 -- a clean, single global minimum at
    (omp=2, numtiles=8), y=0, unambiguous enough for a depth-4 CatBoost ensemble to learn from a
    noiseless prior."""
    out = []
    for ranks, omp in ((448, 1), (224, 2), (112, 4), (64, 7)):
        for nt in (1, 8, 16):
            cfg = _mk_cfg(ranks, omp, ranks, 1, nt)
            y = 0.01 * abs(omp - 2) + 0.002 * ((nt - 8) ** 2) / 64.0
            out.append((cfg, y))
    return out


def test_search_beats_random_on_synthetic_ground_truth():
    candidates_with_y = _synthetic_pool()
    # Prior: the SAME 12 configs, already measured (as if on a sibling case) with the identical
    # true-y function -- gives the surrogate a clean signal to learn before any of THIS case's
    # own points are revealed (matches the brief: "trained on other cases + this case's own
    # already-measured configs").
    prior_rows = []
    for cfg, y in candidates_with_y:
        row = dsearch.candidate_row("priorcase", CASE_FIELDS, HARDWARE_ID, 4, cfg,
                                     run_id="prior:%s" % (dsearch.candidate_key(cfg),))
        row["y"] = y
        prior_rows.append(row)

    d3 = dsearch._run_adaptive_loop(
        candidates_with_y, prior_rows, "targetcase", CASE_FIELDS, HARDWARE_ID, 4,
        reference_predicate=None, kappa=1.5, max_per_rank_omp=2, ensemble_size=5, seed=0)
    rnd = dsearch._random_baseline(candidates_with_y, n_seeds=50, seed0=1)

    d3_regret = [r for _n, r in d3]
    assert len(d3_regret) == len(candidates_with_y)
    # the search must eventually land exactly on the true optimum (pool is finite and exhausted)
    assert d3_regret[-1] == pytest.approx(0.0, abs=1e-9)
    # with a clean, informative prior, an LCB-guided search should not need MORE runs on average
    # than blind random sampling to get within noise of optimal -- checked well before the pool
    # (size 12) is exhausted.
    assert d3_regret[2] <= rnd[2] + 1e-9


# ---------------------------------------------------------------------------
# End-to-end smoke test against the REAL karina STRICT campaign ledger (32 measured rows, 16
# templates x 2 reps -- /home/u5727520/wrf-bench/wscore/campaign/ledger_karina_1075297.csv). Not
# a WRF run, just real, already-measured data; confirms propose() runs end-to-end against a
# campaign/cases.json-registered case with no crash and a schema-valid, loadable proposal.
# ---------------------------------------------------------------------------
_REAL_KARINA_LEDGER = "/home/u5727520/wrf-bench/wscore/campaign/ledger_karina_1075297.csv"


@pytest.mark.skipif(not os.path.exists(_REAL_KARINA_LEDGER), reason="real campaign ledger not on this host")
def test_propose_end_to_end_with_real_karina_ledger(tmp_path):
    state = dsearch.load_state(str(tmp_path / "does-not-exist.json"))
    result, configs_out = dsearch.propose(
        "karina", [_REAL_KARINA_LEDGER], state, batch_size=4, ensemble_size=5, ensemble_seed=0)
    assert result["stopped"] is False
    assert 0 < len(configs_out) <= 4
    out_path = tmp_path / "proposal.json"
    with open(out_path, "w") as f:
        json.dump(dict(configs=configs_out), f)
    loaded = campaign_configs.load_generated_configs(str(out_path))
    assert [c["id"] for c in loaded] == [c["id"] for c in configs_out]


# ---------------------------------------------------------------------------
# Baseline/scale-consistency regression tests (2026-09-15 lafe/karina round-2 bug): a case's own
# already-measured rows (candidate_row()/case_row_fields(), built from state['measured']) and rows
# loaded from a real campaign ledger (benchmark.load_ledger) must land in the SAME
# features.decision_group_key scope, and the same physical measurement must never be counted twice
# when it appears via both state['measured'] and a --ledger file.
# ---------------------------------------------------------------------------
class TestBaselineScaleConsistency:

    def test_normalize_decision_group_fields_leaves_none_and_blank_alone(self):
        # None/'' are legitimate, distinct groups (features.decision_group_known) -- not a type
        # mismatch to paper over.
        row_none, row_blank = dict(io_form_history=None), dict(io_form_history="")
        dsearch._normalize_decision_group_fields([row_none, row_blank])
        assert row_none["io_form_history"] is None
        assert row_blank["io_form_history"] == ""


    def test_dedup_rows_fallback_key_survives_job_id_rep_index_type_mismatch(self):
        # No run_id -- falls back to (case_base, config_key, job_id, rep_index). job_id/rep_index
        # must be compared as strings regardless of Python type, the same trap io_form_history fell
        # into (int 14 vs str "14"): a real ledger row's job_id/rep_index stay raw CSV strings
        # (benchmark.load_ledger never casts them), while ingest()'s int(parsed["rep"]) makes
        # state['measured'][cid][i]['rep_index'] an int.
        cfg = _mk_cfg(448, 1, 8, 56, 1)
        a = dsearch.candidate_row("c", CASE_FIELDS, HARDWARE_ID, 4, cfg, run_id=None,
                                   job_id="99", rep_index=1)
        b = dsearch.candidate_row("c", CASE_FIELDS, HARDWARE_ID, 4, cfg, run_id=None,
                                   job_id=99, rep_index="1")
        assert dsearch._dedup_rows([a, b]) == [a]

    def test_propose_for_case_baselines_incumbent_against_merged_case_reference(self, tmp_path):
        """The lafe symptom, minimized: REF is this case's own already-measured C01-equivalent
        config, also present (2 earlier reps) in the real STRICT campaign ledger. OTHER_A/OTHER_B
        are two round candidates that collide on features.config_key once tile_shape is dropped
        (candidate_key's own docstring) and, pre-fix, formed an isolated round-only scope's modal
        "baseline" -- outvoting the true reference and making the truly-fastest incumbent's y come
        out far more negative than it should."""
        hardware_id = HARDWARE_ID
        case_json = dict(e_we=901, e_sn=901, e_vert=45, max_dom=1)
        ref_cfg = _mk_cfg(448, 1, 8, 56, 1)                      # the C01-equivalent reference
        other_a = _mk_cfg(224, 2, 8, 28, 16, dist="block")       # 1d_y
        other_b = dict(_mk_cfg(224, 2, 8, 28, 16, dist="block", tile_shape="2d"),
                       numtiles_x=4, numtiles_y=4, tile_strategy=3)  # SAME config_key as other_a

        state = dict(
            configs={"REF": ref_cfg, "OTHER_A": other_a, "OTHER_B": other_b},
            measured={
                "REF": [dict(run_id="testcase_REF_r1_1", job_id="1", rep_index=1,
                             s_per_step_median=1.0),
                        dict(run_id="testcase_REF_r2_1", job_id="1", rep_index=2,
                             s_per_step_median=1.0)],
                "OTHER_A": [dict(run_id="testcase_OTHER_A_r1_1", job_id="1", rep_index=1,
                                 s_per_step_median=9.0),
                            dict(run_id="testcase_OTHER_A_r2_1", job_id="1", rep_index=2,
                                 s_per_step_median=9.0)],
                "OTHER_B": [dict(run_id="testcase_OTHER_B_r1_1", job_id="1", rep_index=1,
                                 s_per_step_median=8.0),
                            dict(run_id="testcase_OTHER_B_r2_1", job_id="1", rep_index=2,
                                 s_per_step_median=8.0)],
            },
            hardware_id=hardware_id, nodes=4, total_cores=448, rounds=[{"round": 1}],
        )

        prior_row_kwargs = dict(CASE_FIELDS)
        prior_row_kwargs.update(
            source_dataset="strict_campaign", case_raw="testcase", case_base="testcase",
            case_family=None, hardware_id=hardware_id, binary_kind="pristine",
            compliance_class="STRICT", n_nodes=4, distribution="cyclic", kmp_blocktime=0,
            n_ranks=448, omp_threads=1, nproc_x=8, nproc_y=56, numtiles=1, io_form_history=14)
        prior_rows_raw = [
            ledgermod.new_row(run_id="testcase_C01_r1_9001", job_id="9001", rep_index=1,
                               s_per_step_median=2.0, **prior_row_kwargs),
            ledgermod.new_row(run_id="testcase_C01_r2_9001", job_id="9001", rep_index=2,
                               s_per_step_median=2.0, **prior_row_kwargs),
        ]
        ledger_path = tmp_path / "prior_ledger.csv"
        _write_campaign_style_ledger(prior_rows_raw, str(ledger_path))
        prior_rows = benchmark.load_ledger(str(ledger_path))
        # This fixture must genuinely reproduce the type mismatch (real ledger rows keep
        # io_form_history as the raw CSV string) -- pin that before trusting the rest of the test.
        assert prior_rows[0]["io_form_history"] == "14"

        result, _configs_out = dsearch.propose_for_case(
            "testcase", case_json, CASE_FIELDS, prior_rows, state, batch_size=4,
            total_cores=448, nodes=4, hardware_id=hardware_id, ensemble_size=5, ensemble_seed=0)

        assert result["incumbent"] == ref_cfg
        # geometric mean of {2.0, 2.0 (prior C01), 1.0, 1.0 (this case's own REF)} = sqrt(2);
        # y = ln(1.0) - ln(sqrt(2)) = -ln(2)/2. Deterministic (incumbent_y never touches the
        # surrogate) -- see bc -l verification in the task's own debug session.
        assert result["incumbent_y"] == pytest.approx(-math.log(2) / 2, abs=1e-6)

        # Without decomp_search's own normalization the result must now be identical too:
        # features.decision_group_key normalizes io_form_history and config_key keeps OTHER_B's
        # tile geometry apart from OTHER_A (2026-09-15), so neither half of the bug reproduces.
        case_measured_rows_raw = dsearch.build_case_measured_rows(
            state, "testcase", CASE_FIELDS, hardware_id, 4)
        unnormalized_combined = list(prior_rows) + case_measured_rows_raw
        unnormalized_baselines = targets.compute_case_baselines(unnormalized_combined)
        targets.add_y(unnormalized_combined, unnormalized_baselines)
        broken_cfg, broken_y = dsearch.incumbent_from_measured(case_measured_rows_raw)
        assert broken_cfg == ref_cfg
        assert broken_y == pytest.approx(result["incumbent_y"], abs=1e-6)

    def test_propose_for_case_row_counts_unaffected_by_duplicate_ledger_of_own_measured_round(
            self, tmp_path):
        """The karina symptom, minimized: the case's own already-measured round is ALSO passed via
        --ledger (2026-09-15: EXTRA_LEDGERS duplicated the ledger that had just been ingested into
        the state file). Every measurement must be counted once in the combined/labeled row set
        (and hence in the ensemble's training weight and the baseline vote) regardless of how many
        sources name it."""
        hardware_id = HARDWARE_ID
        ref_cfg = _mk_cfg(448, 1, 8, 56, 1)
        fast_cfg = _mk_cfg(224, 2, 8, 28, 12, dist="block")
        state = dict(
            configs={"REF": ref_cfg, "FAST": fast_cfg},
            measured={
                "REF": [dict(run_id="dup_REF_r1_555", job_id="555", rep_index=1,
                             s_per_step_median=2.0),
                        dict(run_id="dup_REF_r2_555", job_id="555", rep_index=2,
                             s_per_step_median=2.0)],
                "FAST": [dict(run_id="dup_FAST_r1_555", job_id="555", rep_index=1,
                              s_per_step_median=1.0),
                         dict(run_id="dup_FAST_r2_555", job_id="555", rep_index=2,
                              s_per_step_median=1.0)],
            },
            hardware_id=hardware_id, nodes=4, total_cores=448, rounds=[{"round": 1}],
        )
        case_measured_rows = dsearch.build_case_measured_rows(state, "dup", CASE_FIELDS,
                                                                hardware_id, 4)
        assert len(case_measured_rows) == 4  # REF x2 + FAST x2

        dup_kwargs = dict(CASE_FIELDS)
        dup_kwargs.update(
            source_dataset="strict_campaign", case_raw="dup", case_base="dup",
            hardware_id=hardware_id, binary_kind="pristine", compliance_class="STRICT",
            n_nodes=4, n_ranks=448, omp_threads=1, nproc_x=8, nproc_y=56, numtiles=1,
            distribution="cyclic", kmp_blocktime=0, io_form_history=14)
        duplicate_ledger_rows = [
            ledgermod.new_row(run_id="dup_REF_r1_555", job_id="555", rep_index=1,
                               s_per_step_median=2.0, **dup_kwargs),
            ledgermod.new_row(run_id="dup_REF_r2_555", job_id="555", rep_index=2,
                               s_per_step_median=2.0, **dup_kwargs),
        ]
        path = tmp_path / "duplicate_round1_ledger.csv"
        _write_campaign_style_ledger(duplicate_ledger_rows, str(path))
        prior_rows_dup = benchmark.load_ledger(str(path))

        combined_no_dup = dsearch._dedup_rows(case_measured_rows + [])
        combined_with_dup = dsearch._dedup_rows(case_measured_rows + list(prior_rows_dup))
        assert len(combined_no_dup) == len(combined_with_dup) == 4  # never inflates to 6

        case_json = dict(e_we=901, e_sn=901, e_vert=45, max_dom=1)
        result_no_dup, _ = dsearch.propose_for_case(
            "dup", case_json, CASE_FIELDS, [], dict(state), batch_size=4, total_cores=448,
            nodes=4, hardware_id=hardware_id, ensemble_size=5, ensemble_seed=0)
        result_with_dup, _ = dsearch.propose_for_case(
            "dup", case_json, CASE_FIELDS, prior_rows_dup, dict(state), batch_size=4,
            total_cores=448, nodes=4, hardware_id=hardware_id, ensemble_size=5, ensemble_seed=0)
        assert result_no_dup["incumbent"] == result_with_dup["incumbent"] == fast_cfg
        assert result_no_dup["incumbent_y"] == pytest.approx(result_with_dup["incumbent_y"], abs=1e-9)
        assert result_no_dup["n_training_rows"] == result_with_dup["n_training_rows"]


# ---------------------------------------------------------------------------
# Hermetic reproduction against the REAL 2026-09-15 lafe/karina D3 state + ledger snapshots
# (predictor/tests/fixtures/) -- pins the corrected behavior against the exact data that exposed
# the bug, not just the hand-built synthetic analog above.
# ---------------------------------------------------------------------------
_LAFE_STATE_R1 = os.path.join(_FIXTURES, "state_lafe_r1.json")
_LAFE_PRIOR_LEDGER = os.path.join(_FIXTURES, "ledger_lafe_1075295.csv")
_KARINA_STATE_R1R2 = os.path.join(_FIXTURES, "state_karina_r1r2.json")
_KARINA_PRIOR_LEDGER = os.path.join(_FIXTURES, "ledger_karina_1075297.csv")
_KARINA_ROUND1_LEDGER = os.path.join(_FIXTURES, "ledger_karina_1078093.csv")


@pytest.mark.skipif(
    not (os.path.exists(_LAFE_STATE_R1) and os.path.exists(_LAFE_PRIOR_LEDGER)),
    reason="fixture not present (ledger CSVs are gitignored real campaign data)")
def test_real_lafe_round2_incumbent_is_now_on_the_campaign_reference_scale():
    """Reproduces the exact 2026-09-15 lafe round-2 STOP: propose() computed incumbent_y=-0.37442
    for R01D0003 (measured 1.304/1.317 s/step) against an isolated round-1-only baseline of
    ~1.90 s/step, instead of the true C01-equivalent reference (~1.52-1.56 s/step, since R01D0001
    -- this case's own measured C01-equivalent -- merges with the prior campaign's own C01).
    baseline_config_key becomes (8, 56, 1, 1, 'cyclic', 0, 4, 448) with s_baseline=1.52418 (the
    true C01-equivalent scale), and incumbent_y=-0.15606 (Slurm-job-verified against this exact
    fixture data, job 1078178, debug_baseline_scale.py): incumbent_from_measured() picks the single
    best-measured REPLICATE's y (R01D0003 rep 1, s=1.30395), not a config's mean-of-replicates y,
    so this is ln(1.30395/1.52418), not ln(mean(1.30395,1.31708)/1.52418)."""
    state = dsearch.load_state(_LAFE_STATE_R1)
    result, _configs_out = dsearch.propose(
        "lafe", [_LAFE_PRIOR_LEDGER], state, batch_size=8, ensemble_size=5, ensemble_seed=0)
    assert result.get("incumbent_y") is not None
    assert result["incumbent_y"] == pytest.approx(-0.1560599127209296, abs=1e-4)
    ic = result["incumbent"]
    assert (ic["ranks"], ic["omp"], ic["nx"], ic["ny"]) == (224, 2, 8, 28)  # R01D0003


@pytest.mark.skipif(
    not (os.path.exists(_KARINA_STATE_R1R2) and os.path.exists(_KARINA_ROUND1_LEDGER)),
    reason="fixture not present")
def test_real_karina_incumbent_unchanged_whether_or_not_own_round_ledger_is_duplicated():
    """Reproduces the 2026-09-15 karina round-2 bug: the round-1 ledger was passed both via
    --ledger/EXTRA_LEDGERS and (already) via state['measured'] -- with the fix, including it a
    second time must not change the result."""
    result_only, _ = dsearch.propose(
        "karina", [_KARINA_PRIOR_LEDGER], dsearch.load_state(_KARINA_STATE_R1R2), batch_size=8,
        ensemble_size=5, ensemble_seed=0)
    result_dup, _ = dsearch.propose(
        "karina", [_KARINA_PRIOR_LEDGER, _KARINA_ROUND1_LEDGER],
        dsearch.load_state(_KARINA_STATE_R1R2), batch_size=8, ensemble_size=5, ensemble_seed=0)
    assert result_only.get("incumbent_y") is not None
    assert result_only["incumbent_y"] == pytest.approx(result_dup["incumbent_y"], abs=1e-9)
    assert result_only.get("n_training_rows") == result_dup.get("n_training_rows")


def test_propose_for_case_max_omp_excludes_threaded_candidates(tmp_path):
    case_json = dict(e_we=505, e_sn=505, e_vert=45, max_dom=1)
    rows = []
    for p in ("ledger_isc21_1075293.csv", "ledger_lafe_1075295.csv", "ledger_karina_1075297.csv"):
        path = os.path.join("/home/u5727520/wrf-bench/wscore/campaign", p)
        if not os.path.exists(path):
            pytest.skip("campaign ledgers not present")
        rows.extend(benchmark.load_ledger(path))
    result, configs_out = dsearch.propose_for_case(
        "maxomptest", case_json, dsearch.case_row_fields(case_json), rows,
        dict(configs={}, measured={}, rounds=[]), batch_size=8, total_cores=448, nodes=4,
        hardware_id=HARDWARE_ID, ensemble_size=3, ensemble_seed=0, max_omp=1)
    assert not result["stopped"]
    assert configs_out and all(c["omp"] == 1 for c in configs_out)


def test_tile_rows_uses_smallest_patch_and_tile_geometry():
    base = dict(ranks=448, omp=1, nx=14, ny=32, patch_x=40, patch_y=14, numtiles=8, tile_shape="1d_y",
                numtiles_x=0, numtiles_y=0, tile_sz_x=0, tile_sz_y=0)
    assert dsearch.tile_rows(base) == 1
    assert dsearch.tile_rows(dict(base, numtiles=2)) == 7
    assert dsearch.tile_rows(dict(base, tile_shape="2d", numtiles_x=4, numtiles_y=2)) == 7
    assert dsearch.tile_rows(dict(base, tile_shape="tile_sz", tile_sz_y=5)) == 5
