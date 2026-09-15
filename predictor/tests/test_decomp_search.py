#!/usr/bin/env python3
"""predictor/tests/test_decomp_search.py -- pytest suite for predictor/decomp_search.py
(WTune D3: active decomposition search).

Run with:
    pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml \\
        python -m pytest predictor/tests/test_decomp_search.py -v
"""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decomp_search as dsearch
import decomp_space as ds
import ledger as ledgermod
import models as modelmod
import configs as campaign_configs   # decomp_search.py itself inserts campaign/ onto sys.path

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
# candidate_key
# ---------------------------------------------------------------------------
def test_candidate_key_distinguishes_tile_sz_variants():
    """decomp_space._candidate_key (deliberately NOT reused here -- see decomp_search.
    candidate_key's own docstring) would collide two different tile_sz variants under
    tile_shape=='tile_sz'; decomp_search's own key must not."""
    a = dict(_mk_cfg(448, 1, 8, 56, 8, tile_shape="tile_sz"), tile_sz_x=28, tile_sz_y=4)
    b = dict(_mk_cfg(448, 1, 8, 56, 8, tile_shape="tile_sz"), tile_sz_x=14, tile_sz_y=8)
    assert dsearch.candidate_key(a) != dsearch.candidate_key(b)
    assert dsearch.candidate_key(a) == dsearch.candidate_key(dict(a))


# ---------------------------------------------------------------------------
# select_batch: diversity cap + forced inclusion
# ---------------------------------------------------------------------------
def _scored(cfgs, acqs):
    return [(acq, c, acq, 0.0) for acq, c in zip(acqs, cfgs)]


def test_select_batch_respects_diversity_cap():
    # 5 candidates all in the SAME (ranks, omp) group, ranked by acquisition -- cap=2 must keep
    # only the best 2, however large the requested batch.
    cfgs = [_mk_cfg(448, 1, 8, 56, nt) for nt in (1, 8, 12, 13, 14)]
    acqs = [0.0, -0.05, -0.02, -0.10, -0.01]  # numtiles=13 (idx3) best, then numtiles=8 (idx1)
    scored = _scored(cfgs, acqs)
    chosen = dsearch.select_batch(scored, batch_size=10, max_per_rank_omp=2)
    assert len(chosen) == 2
    assert {c["numtiles"] for c in chosen} == {13, 8}


def test_select_batch_diversity_cap_spans_multiple_groups():
    cfgs = ([_mk_cfg(448, 1, 8, 56, nt) for nt in (1, 8, 12)]
            + [_mk_cfg(224, 2, 4, 56, nt) for nt in (1, 12, 16)])
    acqs = [0.0, -0.01, -0.02, -0.05, -0.06, -0.03]
    scored = _scored(cfgs, acqs)
    chosen = dsearch.select_batch(scored, batch_size=10, max_per_rank_omp=1)
    groups = [(c["ranks"], c["omp"]) for c in chosen]
    assert len(chosen) == 2  # 1 per group, 2 groups
    assert sorted(groups) == [(224, 2), (448, 1)]
    # best (lowest) acquisition within each group: 448x1 trio acqs=[0.0,-0.01,-0.02] -> nt=12;
    # 224x2 trio acqs=[-0.05,-0.06,-0.03] -> nt=12.
    by_group = {(c["ranks"], c["omp"]): c for c in chosen}
    assert by_group[(448, 1)]["numtiles"] == 12
    assert by_group[(224, 2)]["numtiles"] == 12


def test_select_batch_forced_included_even_if_not_top_acquisition():
    cfgs = [_mk_cfg(448, 1, 8, 56, nt) for nt in (1, 8, 16)]
    acqs = [-0.5, -0.4, -0.01]  # numtiles=1 (idx0) is by far the best acquisition
    scored = _scored(cfgs, acqs)
    forced = [cfgs[2]]  # numtiles=16, the WORST acquisition -- must still appear
    chosen = dsearch.select_batch(scored, batch_size=2, max_per_rank_omp=5, forced=forced)
    assert len(chosen) == 2
    assert cfgs[2] in chosen
    assert cfgs[0] in chosen  # the remaining slot goes to the best acquisition


def test_select_batch_forced_still_counts_against_diversity_cap():
    cfgs = [_mk_cfg(448, 1, 8, 56, nt) for nt in (1, 8, 16)]
    acqs = [-0.5, -0.4, -0.01]
    scored = _scored(cfgs, acqs)
    forced = [cfgs[2]]
    chosen = dsearch.select_batch(scored, batch_size=5, max_per_rank_omp=1, forced=forced)
    assert len(chosen) == 1  # cap=1 for this single (ranks,omp) group; forced fills the only slot
    assert chosen == [cfgs[2]]


def test_select_batch_skips_strip_shapes_and_blocktime_duplicates():
    # round-1 proposals 2026-09-15 were dominated by 64x1 strips (15x slower in smoke 1078036)
    # and pairs differing only in KMP_BLOCKTIME
    strip = dict(_mk_cfg(64, 7, 64, 1, 13), patch_x=14, patch_y=901)
    good = dict(_mk_cfg(448, 1, 8, 56, 8), patch_x=113, patch_y=17)
    good_bt = dict(good, blocktime="default")
    other = dict(_mk_cfg(224, 2, 8, 28, 14), patch_x=113, patch_y=33)
    scored = [(-1.0, strip, 0.0, 1.0), (-0.2, good, -0.1, 0.1), (-0.19, good_bt, -0.1, 0.1),
              (-0.1, other, -0.05, 0.05)]
    chosen = dsearch.select_batch(scored, batch_size=4, max_per_rank_omp=2)
    assert strip not in chosen                      # aspect 64 > 20
    assert not (good in chosen and good_bt in chosen)  # same shape, blocktime only
    assert other in chosen


def test_select_batch_mixes_exploitation_and_exploration():
    # A has the best predicted mean but a poor LCB; B has the best LCB (high sigma) but a hopeless
    # mean (> log 1.5); C is a reasonable exploration pick. Exploitation must take A, exploration C.
    a = dict(_mk_cfg(448, 1, 8, 56, 8), patch_x=113, patch_y=17)
    b = dict(_mk_cfg(224, 2, 8, 28, 14), patch_x=113, patch_y=33)
    c = dict(_mk_cfg(112, 4, 8, 14, 16), patch_x=113, patch_y=65)
    scored = [(-0.05, a, -0.10, 0.03), (-2.0, b, 0.60, 1.7), (-0.30, c, 0.10, 0.27)]
    chosen = dsearch.select_batch(scored, batch_size=2, max_per_rank_omp=2)
    assert a in chosen and c in chosen and b not in chosen


# ---------------------------------------------------------------------------
# propose_core: stop rule
# ---------------------------------------------------------------------------
def _labeled_rows(cfgs, ys, case_name="prior"):
    return [dict(dsearch.candidate_row(case_name, CASE_FIELDS, HARDWARE_ID, 4, c,
                                        run_id="r:%d" % i), y=y)
            for i, (c, y) in enumerate(zip(cfgs, ys))]


def test_propose_core_stops_when_no_candidate_beats_incumbent_beyond_noise():
    cfgs = [_mk_cfg(448, 1, 8, 56, nt) for nt in (1, 8, 12, 13, 14, 16)]
    ys = [0.0, -0.01, -0.005, -0.008, -0.003, 0.001]
    labeled = _labeled_rows(cfgs, ys)
    state = dict(configs={}, measured={}, rounds=[{"round": 1}])  # pretend round 1 already ran
    # incumbent is already essentially the best of the lot (y=-0.01); every remaining candidate's
    # own labeled y is within NOISE of it, so a surrogate fit on exactly this data should not see
    # a confident further improvement.
    result = dsearch.propose_core(
        cfgs, labeled, "case", CASE_FIELDS, HARDWARE_ID, 4, state, batch_size=4,
        incumbent_cfg=cfgs[1], incumbent_y=-0.01, kappa=0.0, ensemble_size=5, ensemble_seed=0,
        max_rounds=6)
    assert result["stopped"] is True
    assert "noise floor" in result["reason"]


def test_propose_core_does_not_stop_when_incumbent_is_clearly_bad():
    cfgs = [_mk_cfg(448, 1, 8, 56, nt) for nt in (1, 8, 12, 13, 14, 16)]
    ys = [0.0, -0.01, -0.005, -0.20, -0.003, 0.001]  # numtiles=13 is a clear, large win
    labeled = _labeled_rows(cfgs, ys)
    state = dict(configs={}, measured={}, rounds=[{"round": 1}])
    result = dsearch.propose_core(
        cfgs, labeled, "case", CASE_FIELDS, HARDWARE_ID, 4, state, batch_size=4,
        incumbent_cfg=cfgs[0], incumbent_y=0.0, kappa=0.0, ensemble_size=5, ensemble_seed=0,
        max_rounds=6)
    assert result["stopped"] is False
    assert len(result["chosen"]) > 0


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


def test_propose_core_round1_forces_reference_when_no_incumbent_yet():
    cfgs = [_mk_cfg(448, 1, 8, 56, nt) for nt in (1, 8, 12)]
    labeled = _labeled_rows(cfgs, [0.0, -0.5, -0.01])  # numtiles=8 (idx1) has by far the best y
    state = dict(configs={}, measured={}, rounds=[])
    ref_predicate = dict(ranks=448, omp=1, nx=8, ny=56, numtiles=1, dist="cyclic", blocktime="0")
    result = dsearch.propose_core(
        cfgs, labeled, "case", CASE_FIELDS, HARDWARE_ID, 4, state, batch_size=1,
        incumbent_cfg=None, incumbent_y=None, kappa=0.0, ensemble_size=5, ensemble_seed=0,
        max_rounds=6, reference_predicate=ref_predicate)
    assert result["stopped"] is False
    assert result["round"] == 1
    # batch_size=1 and the reference is FORCED -> it must be the one chosen, even though
    # numtiles=8's own labeled y is much better (the point of forcing round 1's anchor).
    assert len(result["chosen"]) == 1
    assert result["chosen"][0]["numtiles"] == 1


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


# ---------------------------------------------------------------------------
# proposal JSON schema == decomp_space.to_campaign_configs, accepted by
# campaign/configs.py's own CONFIG_FILE loader (read-only use of real code).
# ---------------------------------------------------------------------------
def test_proposal_json_matches_to_campaign_configs_schema_and_loads(tmp_path):
    candidates, _axis = ds.enumerate_space(
        dict(e_we=901, e_sn=901, e_vert=45, max_dom=1), total_cores=448, nodes=4,
        hardware_id=HARDWARE_ID, omp_options=[1],
        tile_options=dict(numtiles_multiples=(1,), max_numtiles=16, include_2d=True,
                           tile_sz_variants=()))
    chosen = candidates[:5]
    configs_out = ds.to_campaign_configs(chosen, prefix="R01D")
    path = tmp_path / "proposal.json"
    with open(path, "w") as f:
        json.dump(dict(configs=configs_out), f)

    loaded = campaign_configs.load_generated_configs(str(path))
    assert [c["id"] for c in loaded] == [c["id"] for c in configs_out]
    for c in loaded:
        for key in ("id", "ranks", "omp", "nx", "ny", "numtiles", "dist", "blocktime",
                    "config_source"):
            assert key in c
        legal = ds.legal_shapes_exact([901], [901], c["ranks"])
        assert (c["nx"], c["ny"]) in legal


# ---------------------------------------------------------------------------
# parse_run_id / ingest
# ---------------------------------------------------------------------------
def test_parse_run_id():
    parsed = dsearch.parse_run_id("karina_R01D0003_r1_1078123")
    assert parsed == dict(case="karina", cid="R01D0003", rep="1", job="1078123")
    assert dsearch.parse_run_id("not-a-run-id") is None


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
    def test_ensemble_produces_nonzero_spread_and_a_usable_mean(self):
        cfgs_ys = _synthetic_pool()  # 12 distinct configs, a real (if noiseless) surface
        cfgs = [c for c, _y in cfgs_ys]
        labeled = _labeled_rows(cfgs, [y for _c, y in cfgs_ys])
        models_ = dsearch.fit_ensemble(labeled, n_models=8, seed=0)
        assert 1 <= len(models_) <= 8
        score_rows = [dsearch.candidate_row("q", CASE_FIELDS, HARDWARE_ID, 4, c, "q:%d" % i)
                      for i, c in enumerate(cfgs)]
        mu, sigma = dsearch.ensemble_predict(models_, score_rows)
        assert len(mu) == len(cfgs)
        assert (sigma >= 0).all()
        assert sigma.sum() > 0, "bootstrap ensemble produced zero spread across all candidates"

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
