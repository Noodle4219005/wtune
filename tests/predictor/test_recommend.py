#!/usr/bin/env python3
"""predictor/tests/test_recommend.py -- pytest suite for predictor/recommend.py.

Run with:
    pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml \\
        python -m pytest predictor/tests/test_recommend.py -v

Builds a small synthetic ledger (three case_base groups, all on the same grid as the "main"
synthetic namelist below, all hardware_id=HARDWARE_ID_TRAIN) plus several synthetic
namelist.input files. Every test uses --model ridge (unless the test is specifically about model
choice) so results are deterministic and do not depend on catboost/lightgbm being installed in
the test environment.
"""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decomp_space
import ledger
import recommend
import targets

# recommend.py itself inserts campaign/ onto sys.path on import (see recommend.py's module-level
# sys.path setup) -- safe to import configs only AFTER `import recommend` above.
import configs as campaign_configs

HARDWARE_ID_TRAIN = "hw-train-synthetic"

# Main synthetic case: same grid as karina (901x901x45), which the real campaign has already
# shown resolves legal shapes for every STRICT template (see campaign/configs.py TEMPLATES /
# knowledge notes) -- avoids a fixture that accidentally has zero legal candidates.
MAIN_NAMELIST = """\
&time_control
 run_days                            = 0,
 run_hours                           = 2,
 run_minutes                         = 0,
 history_interval                    = 60,
 io_form_history                     = 2,
 /

&domains
 max_dom                             = 1,
 e_we                                = 901,
 e_sn                                = 901,
 e_vert                              = 45,
 dx                                  = 3000,
 dy                                  = 3000,
 time_step                           = 15,
 /

&physics
 mp_physics                          = 8,
 cu_physics                          = 0,
 bl_pbl_physics                      = 1,
 sf_sfclay_physics                   = 1,
 sf_surface_physics                  = 2,
 ra_lw_physics                       = 4,
 ra_sw_physics                       = 4,
 radt                                = 15,
 cudt                                = 0,
 /
"""

# Same case as MAIN_NAMELIST but run_days/run_hours/run_minutes are ALL zero -- a real-world
# placeholder shape (lafe's actual template on disk is exactly this: 0/0/0, overwritten by the
# STRICT campaign's own runner before every launch). Exercises coordinator review item 1: the
# verify-plan must never emit "--run-minutes 0".
ZERO_RUN_NAMELIST = MAIN_NAMELIST.replace(
    "run_hours                           = 2,", "run_hours                           = 0,")

# conus12's real e_we/e_sn/e_vert (campaign/cases.json) -- much smaller than the 901x901x45
# training grid above, so this is genuinely out-of-range, not an arbitrary fabricated grid.
SMALL_NAMELIST = """\
&time_control
 run_days                            = 0,
 run_hours                           = 1,
 run_minutes                         = 0,
 history_interval_m                  = 30,
 io_form_history                     = 2,
 /

&domains
 max_dom                             = 1,
 e_we                                = 425,
 e_sn                                = 300,
 e_vert                              = 50,
 dx                                  = 12000,
 dy                                  = 12000,
 time_step                           = 72,
 /

&physics
 mp_physics                          = 6,
 cu_physics                          = 0,
 bl_pbl_physics                      = 1,
 sf_sfclay_physics                   = 1,
 sf_surface_physics                  = 2,
 ra_lw_physics                       = 4,
 ra_sw_physics                       = 4,
 radt                                = 15,
 cudt                                = 0,
 /
"""

# Two-domain namelist for coordinator review item 2 (--max-dom): d02 is deliberately tiny
# (200x200), which makes every ranks=448 STRICT template illegal once d02 counts toward legality
# (ceil(200/nx)>=10 and ceil(200/ny)>=10 cannot both hold when nx*ny=448), while d01 alone
# (901x901) is legal for essentially every template -- so --max-dom 1 vs the namelist's own
# max_dom=2 must produce different (and --max-dom 1 strictly larger) legal candidate sets.
MULTI_DOMAIN_NAMELIST = """\
&time_control
 run_days                            = 0,
 run_hours                           = 0,
 run_minutes                         = 0,
 history_interval                    = 60, 0,
 io_form_history                     = 2,
 /

&domains
 max_dom                             = 2,
 e_we                                = 901, 200,
 e_sn                                = 901, 200,
 e_vert                              = 45, 45,
 dx                                  = 3000, 1000,
 dy                                  = 3000, 1000,
 time_step                           = 15,
 /

&physics
 mp_physics                          = 8,
 cu_physics                          = 0,
 bl_pbl_physics                      = 1,
 sf_sfclay_physics                   = 1,
 sf_surface_physics                  = 2,
 ra_lw_physics                       = 4,
 ra_sw_physics                       = 4,
 radt                                = 15,
 cudt                                = 0,
 /
"""

TRAIN_CASES = ["trainA", "trainB", "trainC"]
NUMTILES_VALUES = [8, 10, 12, 13, 14, 16]
BASE_LOG_S = {c: 0.05 * i for i, c in enumerate(TRAIN_CASES)}
F_CONFIG = {nt: 0.05 * nt + 8.0 / nt for nt in NUMTILES_VALUES}  # convex, optimum near nt=13


def _case_rows(case_base, hardware_id, e_we, e_sn, e_vert, intercept, noise_std=0.0005, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for nt in NUMTILES_VALUES:
        log_s = intercept + F_CONFIG[nt] + rng.normal(0, noise_std)
        rows.append(ledger.new_row(
            run_id="%s:nt%d" % (case_base, nt), source_dataset="synthetic-recommend-test",
            job_id=None, rep_index=0,
            case_raw=case_base, case_base=case_base, case_family=ledger.case_family_of(case_base),
            hardware_id=hardware_id,
            binary_kind="pristine", compliance_class="STRICT",
            e_we=float(e_we), e_sn=float(e_sn), e_vert=float(e_vert), dx=3000.0, dy=3000.0,
            time_step=15.0, mp_physics=8, cu_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1,
            sf_surface_physics=2, ra_lw_physics=4, ra_sw_physics=4, radt=15.0, cudt=0.0,
            history_interval=60.0, io_form_history=2,
            n_nodes=4, n_ranks=448, omp_threads=1, nproc_x=8, nproc_y=56, numtiles=nt,
            distribution="cyclic", kmp_blocktime=0,
            s_per_step_median=math.exp(log_s), s_per_step_mean=math.exp(log_s),
            n_steps_measured=40, diffwrf_differing_fields=None, fidelity_label=None,
            profile_micro_share=None, profile_radiation_share=None,
            profile_dynamics_share=None, profile_mpi_share=None, profile_io_share=None,
            profile_rank_imbalance=None, profile_thread_imbalance=None,
            notes="synthetic",
        ))
    return rows


def make_training_rows(noise_std=0.0005, seed=0):
    rows = []
    for c in TRAIN_CASES:
        rows += _case_rows(c, HARDWARE_ID_TRAIN, 901, 901, 45, BASE_LOG_S[c],
                            noise_std=noise_std, seed=seed)
    return rows


def write_ledger_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ledger.LEDGER_FIELDS})


@pytest.fixture
def ledger_csv(tmp_path):
    path = tmp_path / "runs.csv"
    write_ledger_csv(make_training_rows(), str(path))
    return str(path)


@pytest.fixture
def main_namelist(tmp_path):
    path = tmp_path / "namelist.input"
    path.write_text(MAIN_NAMELIST)
    return str(path)


@pytest.fixture
def small_namelist(tmp_path):
    path = tmp_path / "namelist.input.small"
    path.write_text(SMALL_NAMELIST)
    return str(path)


def _legal_ids_for_main_grid():
    kept, skipped, _aliases = campaign_configs.resolve_case(
        dict(e_we=901, e_sn=901, e_vert=45, max_dom=1))
    return {c["id"] for c in kept}, {tid for tid, _reason in skipped}


# ---------------------------------------------------------------------------
# 1. Deterministic shortlist, legal-only, <=top_k, reference included
# ---------------------------------------------------------------------------
def test_deterministic_shortlist_legal_only(ledger_csv, main_namelist):
    legal_ids, skipped_ids = _legal_ids_for_main_grid()
    assert legal_ids, "fixture bug: no legal STRICT candidates for the 901x901x45 grid"

    result = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5)

    assert result["model_used"] == "ridge"
    all_ids = [e["config_id"] for e in result["all_ranked"]]
    assert all_ids, "no candidates produced"
    assert set(all_ids) <= legal_ids
    assert not (set(all_ids) & skipped_ids)
    assert len(result["shortlist"]) <= 5

    assert result["reference"] is not None
    assert result["reference"]["config_id"] == "C01"  # targets._reference_order: omp=1,numtiles=1

    # design hook (brief item 7): every candidate carries compliance_class + fidelity_status
    for e in result["all_ranked"]:
        assert e["compliance_class"] == "STRICT"
        assert e["fidelity_status"] == "N/A (STRICT)"
        # magnitude companion (coordinator review item 4): always present regardless of primary
        assert e["magnitude_model"] == result["magnitude_model_used"]
        assert isinstance(e["magnitude_value"], float)
        assert e["label"] in ("SAFE", "EXPERIMENTAL")

    # in-hardware, in-range grid: must NOT be forced EXPERIMENTAL by the zero-hardware-rows rule
    assert result["hardware_match"]["n_rows"] == len(TRAIN_CASES) * len(NUMTILES_VALUES)
    assert not any("zero training rows share hardware_id" in w for w in result["warnings"])
    assert result["grid_out_of_range"] is False
    # this synthetic case's grid is an EXACT match to trainA/B/C's -- honestly IN-SAMPLE
    assert any(w.startswith("IN-SAMPLE") for w in result["warnings"])

    # determinism: same inputs -> byte-identical shortlist
    result2 = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5)
    ids1 = [(e["config_id"], e["value"]) for e in result["shortlist"]]
    ids2 = [(e["config_id"], e["value"]) for e in result2["shortlist"]]
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# 2. Zero hardware match -> every label EXPERIMENTAL + warning
# ---------------------------------------------------------------------------
def test_no_hardware_match_forces_experimental(ledger_csv, main_namelist):
    result = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id="hw-never-trained-on", nodes=4, compliance="STRICT", model="ridge", top_k=5)

    assert result["hardware_match"]["n_rows"] == 0
    assert any("zero training rows share hardware_id" in w for w in result["warnings"])
    assert result["all_ranked"], "fixture bug: expected legal candidates"
    assert all(e["label"] == "EXPERIMENTAL" for e in result["all_ranked"])


# ---------------------------------------------------------------------------
# 3. Out-of-range grid warning (conus12's real, much-smaller grid vs the 901x901x45 training set)
# -- also a genuinely NEW case: no IN-SAMPLE warning should fire here.
# ---------------------------------------------------------------------------
def test_out_of_range_grid_warns(ledger_csv, small_namelist):
    result = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=small_namelist, case_name="smallcase",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5)

    assert result["grid_out_of_range"] is True
    assert any("out-of-range extrapolation" in w for w in result["warnings"])
    assert result["grid_neighbors"]
    assert {n["case_base"] for n in result["grid_neighbors"]} <= set(TRAIN_CASES)
    assert any("nearest training case" in w for w in result["warnings"])
    assert not any(w.startswith("IN-SAMPLE") for w in result["warnings"])


# ---------------------------------------------------------------------------
# 4. PHYSICS-GRAY is refused clearly, not silently treated as STRICT
# ---------------------------------------------------------------------------
def test_physics_gray_refused(ledger_csv, main_namelist):
    with pytest.raises(recommend.UnsupportedComplianceError, match="fidelity"):
        recommend.recommend(
            ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
            hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="PHYSICS-GRAY", model="ridge")


def test_unsupported_compliance_refused_via_cli(ledger_csv, main_namelist, capsys):
    # PHYSICS-GRAY is now supported (main() dispatches it to recommend_physics_gray -- see the
    # PHYSICS-GRAY test class below); a genuinely unknown compliance level must still be refused
    # cleanly via the CLI, through the same recommend()/UnsupportedComplianceError path as before.
    rc = recommend.main([
        "--ledger", ledger_csv, "--namelist", main_namelist, "--case-name", "synthtest",
        "--hardware-id", HARDWARE_ID_TRAIN, "--compliance", "BOGUS-LEVEL",
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "fidelity" in err.lower()


# ---------------------------------------------------------------------------
# 5. Namelist parsing never invents values it did not read
# ---------------------------------------------------------------------------
def test_namelist_parsing_never_guesses(main_namelist):
    case_fields = recommend.parse_case_from_namelist(main_namelist)
    assert case_fields["e_we"] == 901
    assert case_fields["e_sn"] == 901
    assert case_fields["e_vert"] == 45
    assert case_fields["mp_physics"] == 8
    assert case_fields["max_dom"] == 1


def test_report_formats_without_crashing(ledger_csv, main_namelist):
    result = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5)
    report = recommend.format_report(result)
    assert "WTune STRICT config shortlist" in report
    assert "What to run to verify" in report
    assert "magnitude (" in report


# ---------------------------------------------------------------------------
# Coordinator review item 1: verify-plan run length must never be a useless "--run-minutes 0",
# and must state how the length was derived.
# ---------------------------------------------------------------------------
def test_verify_run_minutes_uses_cases_json_when_case_is_known(main_namelist):
    case_fields = recommend.parse_case_from_namelist(main_namelist)  # time_step=15
    # isc21 (campaign/cases.json): steps=25, dt=15 -> 375s -> ceil(375/60)=7 min
    minutes, source = recommend.verify_run_minutes("isc21", case_fields)
    assert minutes == 7
    assert "cases.json" in source


def test_verify_run_minutes_falls_back_to_documented_default(main_namelist):
    case_fields = recommend.parse_case_from_namelist(main_namelist)  # time_step=15
    # not a real campaign case name -> 30 d01 steps x time_step=15s = 450s -> ceil(450/60)=8 min
    minutes, source = recommend.verify_run_minutes("not-a-real-case", case_fields)
    assert minutes == 8
    assert "default" in source


def test_verify_run_minutes_reports_reason_when_undeterminable():
    minutes, source = recommend.verify_run_minutes("not-a-real-case", dict(time_step=None))
    assert minutes is None
    assert "cannot derive" in source


def test_verify_plan_never_emits_run_minutes_zero(ledger_csv, tmp_path):
    path = tmp_path / "namelist.input.zerorun"
    path.write_text(ZERO_RUN_NAMELIST)
    result = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=str(path), case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5)
    assert result["verify_plan"]["run_minutes"] not in (0, None)
    for c in result["verify_plan"]["configs"]:
        assert "--run-minutes 0 " not in c["namelist_edit_command"]


# ---------------------------------------------------------------------------
# Coordinator review item 2: --max-dom overrides the namelist's own max_dom for legality AND the
# verify-plan, can only narrow (never exceed what the namelist actually has).
# ---------------------------------------------------------------------------
def test_resolve_max_dom_narrows_and_rejects_widening():
    assert recommend.resolve_max_dom(None, {"max_dom": 2}) == 2
    assert recommend.resolve_max_dom(1, {"max_dom": 2}) == 1
    with pytest.raises(ValueError, match="exceeds this namelist"):
        recommend.resolve_max_dom(3, {"max_dom": 2})


def test_max_dom_override_narrows_legality_and_verify_plan(ledger_csv, tmp_path):
    path = tmp_path / "namelist.input.multidom"
    path.write_text(MULTI_DOMAIN_NAMELIST)
    namelist_path = str(path)

    result_full = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=namelist_path, case_name="multidom",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5)
    assert result_full["max_dom"] == 2
    assert result_full["all_ranked"], (
        "fixture bug: expected at least one template still legal at max_dom=2 "
        "(skipped: %s)" % result_full["skipped"])
    ids_full = {e["config_id"] for e in result_full["all_ranked"]}

    result_narrow = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=namelist_path, case_name="multidom",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5,
        max_dom=1)
    assert result_narrow["max_dom"] == 1
    ids_narrow = {e["config_id"] for e in result_narrow["all_ranked"]}

    # d02=200x200 makes every ranks=448 template illegal once it counts; d01-only legality (the
    # namelist's own max_dom=2 vs an explicit --max-dom=1) must admit strictly more templates.
    assert len(ids_narrow) > len(ids_full)

    for c in result_narrow["verify_plan"]["configs"]:
        assert "--max-dom 1 " in c["namelist_edit_command"]
    for c in result_full["verify_plan"]["configs"]:
        assert "--max-dom 2 " in c["namelist_edit_command"]

    with pytest.raises(ValueError, match="exceeds this namelist"):
        recommend.recommend(
            ledger_path=ledger_csv, namelist_path=namelist_path, case_name="multidom",
            hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge",
            max_dom=5)


# ---------------------------------------------------------------------------
# Coordinator review item 3: in-sample detection (by case_base name AND by identical grid under a
# different name) plus --holdout-case.
# ---------------------------------------------------------------------------
def test_in_sample_check_by_name_and_grid_directly():
    scope_rows = make_training_rows()  # trainA/B/C, all sharing the exact 901x901x45 grid
    case_fields = dict(e_we=901.0, e_sn=901.0, e_vert=45.0)

    by_name, grid_matches = recommend._in_sample_check(scope_rows, "trainA", case_fields)
    assert by_name is True
    assert grid_matches == set(TRAIN_CASES)

    by_name2, grid_matches2 = recommend._in_sample_check(scope_rows, "brand_new_case", case_fields)
    assert by_name2 is False
    assert grid_matches2 == set(TRAIN_CASES)  # still flagged in-sample via grid, honestly

    # holding trainA's own rows out does not erase a grid-twin's flag (trainB/C still match)
    scope_without_a = [r for r in scope_rows if r["case_base"] != "trainA"]
    by_name3, grid_matches3 = recommend._in_sample_check(scope_without_a, "trainA", case_fields)
    assert by_name3 is False
    assert grid_matches3 == {"trainB", "trainC"}


def test_in_sample_warning_fires_end_to_end(ledger_csv, main_namelist):
    result = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="trainA",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5)
    in_sample = [w for w in result["warnings"] if w.startswith("IN-SAMPLE")]
    assert in_sample
    assert "trainA" in in_sample[0]


def test_holdout_case_drops_rows_from_training(ledger_csv, main_namelist):
    result = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="trainA",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5,
        holdout_case="trainA")
    assert any("dropped %d" % len(NUMTILES_VALUES) in w and "trainA" in w
               for w in result["warnings"])
    assert result["hardware_match"]["n_rows"] == (len(TRAIN_CASES) - 1) * len(NUMTILES_VALUES)
    assert result["hardware_match"]["n_cases"] == len(TRAIN_CASES) - 1


def test_holdout_unknown_case_warns_zero_dropped(ledger_csv, main_namelist):
    result = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5,
        holdout_case="no-such-case")
    assert any("ZERO training rows" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Coordinator review item 4: a magnitude companion (catboost_point, fallback ridge) always
# supplies the percent-change estimate + SAFE/EXPERIMENTAL label, even when the primary/requested
# ranking model is catboost_pairwise (which cannot produce either on its own).
# ---------------------------------------------------------------------------
def test_magnitude_companion_present_for_every_primary_model(ledger_csv, main_namelist):
    for primary in ("ridge", "catboost_point", "catboost_pairwise"):
        result = recommend.recommend(
            ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
            hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model=primary, top_k=5)
        assert result["magnitude_model_used"] in ("catboost_point", "ridge")
        for e in result["all_ranked"]:
            assert e["magnitude_model"] == result["magnitude_model_used"]
            assert isinstance(e["magnitude_value"], float)
            assert e["label"] in ("SAFE", "EXPERIMENTAL")


def test_pairwise_primary_no_longer_forces_every_label_experimental(ledger_csv, main_namelist):
    result = recommend.recommend(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="catboost_pairwise",
        top_k=5)
    if result["model_used"] != "catboost_pairwise":
        pytest.skip("catboost/lightgbm not available in this environment; fell back to %r"
                    % result["model_used"])
    assert result["model_used"] == "catboost_pairwise"
    assert all(e["score_kind"] == "ranking_score" for e in result["all_ranked"])
    # the fix: labels come from the magnitude companion, not from is_pairwise forcing EXPERIMENTAL
    assert any(e["label"] == "SAFE" for e in result["all_ranked"]), (
        "expected at least one SAFE label from the magnitude companion on this tightly "
        "separable synthetic ledger")


# ---------------------------------------------------------------------------
# Coordinator review item 5: "nearest training cases by grid size" must list same-hardware
# neighbours first, even when a cross-hardware case is numerically closer in grid size.
# ---------------------------------------------------------------------------
NEIGHBOR_NAMELIST = MAIN_NAMELIST.replace(
    "e_we                                = 901,", "e_we                                = 905,"
).replace(
    "e_sn                                = 901,", "e_sn                                = 905,")


@pytest.fixture
def mixed_hardware_ledger_csv(tmp_path):
    rows = (_case_rows("trainSame", HARDWARE_ID_TRAIN, 901, 901, 45, 0.0)
            + _case_rows("trainOther", HARDWARE_ID_TRAIN, 850, 850, 40, 0.05)
            + _case_rows("trainCross", "hw-cross-vendor", 905, 905, 45, 0.1))
    path = tmp_path / "runs_mixed_hw.csv"
    write_ledger_csv(rows, str(path))
    return str(path)


def test_grid_neighbors_list_same_hardware_first(mixed_hardware_ledger_csv, tmp_path):
    path = tmp_path / "namelist.input.neighbor"
    path.write_text(NEIGHBOR_NAMELIST)  # query grid = 905x905x45, an EXACT match to trainCross

    result = recommend.recommend(
        ledger_path=mixed_hardware_ledger_csv, namelist_path=str(path), case_name="neighbortest",
        hardware_id=HARDWARE_ID_TRAIN, nodes=4, compliance="STRICT", model="ridge", top_k=5)

    neighbors = result["grid_neighbors"]
    assert len(neighbors) == 3
    # trainCross has grid DISTANCE 0 (exact match) -- objectively closer than either
    # same-hardware case -- yet same-hardware neighbours must still sort first.
    assert neighbors[0]["case_base"] == "trainSame"
    assert neighbors[0]["hardware_id"] == HARDWARE_ID_TRAIN
    assert neighbors[0]["same_hardware"] is True
    assert neighbors[-1]["case_base"] == "trainCross"
    assert neighbors[-1]["same_hardware"] is False
    assert any(n["same_hardware"] is False for n in neighbors)
    assert any("CROSS-HW" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# --compliance PHYSICS-GRAY (decision 0051): recommend_physics_gray / format_physics_report /
# the CLI's PHYSICS-GRAY dispatch. main_namelist (901x901x45, dx=3000) is used as the "new case"
# throughout -- --probe-screen or a monkeypatched run_probe_screen supplies the SpeedyWeather
# probe text so no test ever shells out to Julia.
# ---------------------------------------------------------------------------
PROBE_SCREEN_TEXT = (
    "=== wtune SCREEN -- autotuner funnel ===\n"
    "STAGE 2 -- SpeedyWeather cheap physics vet:\n"
    "  swint_opt       [PHYSICS-GRAY ] CONFIRM    ok  [SIM-VALIDATED]\n"
    "  mp_physics      [PHYSICS-GRAY ] REJECT     convective regime needs double-moment  [HEURISTIC]\n"
)

PHYS_HW = HARDWARE_ID_TRAIN
PHYS_NATIVE = dict(mp_physics=8, cu_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1,
                    sf_surface_physics=2, ra_lw_physics=4, ra_sw_physics=4, radt=15.0,
                    cudt=0.0, time_step=15.0)


@pytest.fixture
def probe_screen_file(tmp_path):
    path = tmp_path / "screen_synthtest.txt"
    path.write_text(PROBE_SCREEN_TEXT)
    return str(path)


def _physics_notes(template, family, edits_str, decomp, sim_minutes=30, hist_minutes=30,
                    fidelity_nd_max=None, probe=None, fidelity_floor_ratio_max=None):
    parts = ["mode=physics", "template=%s" % template, "family=%s" % family,
             "edits=%s" % edits_str, "decomp=%s" % decomp,
             "sim_minutes=%d" % sim_minutes, "hist_minutes=%d" % hist_minutes]
    if fidelity_nd_max is not None:
        parts.append("fidelity_nd_max=%.4f" % fidelity_nd_max)
    # The chaos-floor ratio is the other number strict_campaign.sbatch writes into notes (see
    # configs.fidelity_notes_tokens); --objective accuracy needs it for the below-floor rule.
    # Defaults to absent so every existing fixture row's notes string is byte-identical.
    if fidelity_floor_ratio_max is not None:
        parts.append("fidelity_floor_ratio_max=%.4f" % fidelity_floor_ratio_max)
    parts.append("probe=%s" % (",".join("%s:%s" % kv for kv in (probe or {}).items()) or "none"))
    return " ".join(parts)


def _phys_row(run_id, template, family, edits_dict, physics_fields, decomp, rep, s,
              fidelity_label=None, nd=None, probe=None, hardware_id=PHYS_HW, case="phystrain",
              floor_ratio=None):
    return ledger.new_row(
        run_id=run_id, source_dataset="strict_campaign", job_id="1", rep_index=rep,
        case_raw=case, case_base=case, case_family=ledger.case_family_of(case),
        hardware_id=hardware_id, binary_kind="pristine",
        compliance_class=("STRICT" if template == "P00" else "PHYSICS-GRAY"),
        e_we=901.0, e_sn=901.0, e_vert=45.0, dx=3000.0, dy=3000.0,
        time_step=physics_fields["time_step"], mp_physics=physics_fields["mp_physics"],
        cu_physics=physics_fields["cu_physics"], bl_pbl_physics=physics_fields["bl_pbl_physics"],
        sf_sfclay_physics=physics_fields["sf_sfclay_physics"],
        sf_surface_physics=physics_fields["sf_surface_physics"],
        ra_lw_physics=physics_fields["ra_lw_physics"], ra_sw_physics=physics_fields["ra_sw_physics"],
        radt=physics_fields["radt"], cudt=physics_fields["cudt"],
        history_interval=(30.0 if rep == 1 else 0.0), io_form_history=2,
        n_nodes=4, n_ranks=448, omp_threads=1, nproc_x=8, nproc_y=56, numtiles=1,
        distribution="cyclic", kmp_blocktime=0,
        s_per_step_median=s, s_per_step_mean=s, n_steps_measured=30,
        diffwrf_differing_fields=None, fidelity_label=(fidelity_label if rep == 1 else None),
        profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
        profile_mpi_share=None, profile_io_share=None, profile_rank_imbalance=None,
        profile_thread_imbalance=None,
        notes=_physics_notes(template, family, ";".join("%s=%s" % kv for kv in edits_dict.items()),
                              decomp, fidelity_nd_max=(nd if rep == 1 else None), probe=probe,
                              fidelity_floor_ratio_max=(floor_ratio if rep == 1 else None)),
    )


PHYS_TRAIN_CASES = ["phystrain%d" % i for i in range(6)]  # >=5 cases: leave-one-case-out
# out-of-fold dangerous-miss evidence (physics.MIN_OOF_DANGEROUS_MISS=5) needs at least that many
# distinct case_base groups per template, or every candidate is forced UNCONFIRMED as
# "insufficient evidence" regardless of everything else (see defect 3's fix).


def make_physics_training_rows(cases=PHYS_TRAIN_CASES):
    """`len(cases)` synthetic cases (same grid as main_namelist) x (P00 + P01 + P02) x 2 reps each:
    P01 always measured 5% faster and WITHIN-FLOOR (probe-clean) -> should end up RECOMMENDABLE;
    P02 always measured faster still but LARGE-DRIFT (probe-clean too) -> must stay UNCONFIRMED on
    fidelity grounds alone, independent of the probe. Deliberately identical numbers across cases
    (this fixture is about status LOGIC, not about across-case variance) -- >=5 cases so
    per_template_dangerous_miss has enough leave-one-case-out out-of-fold evidence to call both
    templates' rates 0.0 (never insufficient), same defaults every caller of predict_physics_
    shortlist/recommend_physics_gray relies on."""
    rows = []
    for case in cases:
        for rep in (1, 2):
            rows.append(_phys_row("%s_P00_r%d" % (case, rep), "P00", "none", {}, PHYS_NATIVE,
                                   "C02", rep, 1.0, case=case))
        for rep in (1, 2):
            rows.append(_phys_row("%s_P01_r%d" % (case, rep), "P01", "radiation", {"swint_opt": "1"},
                                   PHYS_NATIVE, "C02", rep, 0.95, fidelity_label="WITHIN-FLOOR",
                                   nd=0.01, probe={"swint_opt": "CONFIRM"}, case=case))
        p02_fields = dict(PHYS_NATIVE, radt=30.0)
        for rep in (1, 2):
            rows.append(_phys_row("%s_P02_r%d" % (case, rep), "P02", "radiation",
                                   {"radt": "30", "swint_opt": "1"}, p02_fields, "C02", rep, 0.90,
                                   fidelity_label="LARGE-DRIFT", nd=0.5,
                                   probe={"radt": "CONFIRM", "swint_opt": "CONFIRM"}, case=case))
    return rows


FINE_DX_NAMELIST = MAIN_NAMELIST.replace(
    "dx                                  = 3000,", "dx                                  = 500,"
).replace("dy                                  = 3000,", "dy                                  = 500,")


@pytest.fixture
def fine_dx_namelist(tmp_path):
    path = tmp_path / "namelist.input.finedx"
    path.write_text(FINE_DX_NAMELIST)
    return str(path)


def test_physics_gray_no_training_data_all_unconfirmed(ledger_csv, main_namelist, probe_screen_file):
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file)
    assert result["compliance"] == "PHYSICS-GRAY"
    assert result["shortlist"], "fixture bug: expected at least one applicable template"
    assert all(c["status"] == "UNCONFIRMED" for c in result["shortlist"])
    assert any("no PHYSICS-GRAY campaign data" in r
               for c in result["shortlist"] for r in c["reasons"])
    assert not any(c["template"] == "P00" for c in result["shortlist"])

    p04 = next(c for c in result["shortlist"] if c["template"] == "P04")
    assert p04["probe_verdict"] == "REJECT"
    assert any("REJECT" in r for r in p04["reasons"])


def test_physics_gray_recommendable_with_synthetic_ledger(tmp_path, ledger_csv, main_namelist,
                                                            probe_screen_file):
    phys_ledger_path = tmp_path / "physics_runs.csv"
    write_ledger_csv(make_physics_training_rows(), str(phys_ledger_path))

    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file)

    by_t = {c["template"]: c for c in result["shortlist"]}
    assert by_t["P01"]["status"] == "RECOMMENDABLE"
    # time-to-solution vs P00: NEGATIVE = faster (defect 4 rename/sign-flip from the old
    # predicted_gain_pct, which used the opposite convention).
    assert (by_t["P01"]["predicted_time_to_solution_pct"] is not None
            and by_t["P01"]["predicted_time_to_solution_pct"] < 0)
    assert by_t["P01"]["infeasibility"]["rate"] == 0.0
    assert by_t["P01"]["dangerous_miss"]["insufficient"] is False
    assert by_t["P01"]["dangerous_miss"]["dangerous_miss_rate"] == 0.0

    # probe REJECT on the edited knob forces UNCONFIRMED regardless of prediction
    assert by_t["P04"]["status"] == "UNCONFIRMED"
    assert by_t["P04"]["probe_verdict"] == "REJECT"

    # LARGE-DRIFT-labeled P02 must stay UNCONFIRMED even though it is faster and probe-clean
    assert by_t["P02"]["status"] == "UNCONFIRMED"
    assert by_t["P02"]["probe_verdict"] != "REJECT"

    statuses = [c["status"] for c in result["shortlist"]]
    first_unconfirmed = statuses.index("UNCONFIRMED") if "UNCONFIRMED" in statuses else len(statuses)
    assert all(s == "RECOMMENDABLE" for s in statuses[:first_unconfirmed])


def test_physics_gray_cross_hardware_all_unconfirmed(tmp_path, ledger_csv, main_namelist,
                                                       probe_screen_file):
    phys_ledger_path = tmp_path / "physics_runs.csv"
    write_ledger_csv(make_physics_training_rows(), str(phys_ledger_path))

    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id="hw-never-trained-on", physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file)
    assert all(c["status"] == "UNCONFIRMED" for c in result["shortlist"])
    assert any("zero PHYSICS-GRAY training rows share hardware_id" in w for w in result["warnings"])


def test_physics_gray_probe_invalid_below_1km(ledger_csv, fine_dx_namelist, probe_screen_file):
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=fine_dx_namelist, case_name="finecase",
        hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file)
    assert result["probe_valid"] is False
    assert any("< 1000 m" in w for w in result["warnings"])
    # the screen text's REJECT verdict must be DISREGARDED entirely below 1 km
    assert all(c.get("probe_verdict") is None for c in result["shortlist"])


def test_physics_gray_run_probe_requires_lat_lon_source(ledger_csv, main_namelist):
    with pytest.raises(ValueError, match="SpeedyWeather probe"):
        recommend.recommend_physics_gray(
            ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
            hardware_id=HARDWARE_ID_TRAIN)


def test_physics_gray_run_probe_mocked(monkeypatch, ledger_csv, main_namelist):
    monkeypatch.setattr(
        recommend, "run_probe_screen",
        lambda namelist_path, latlon=None, wps=None, wrfinput=None, **kw: PROBE_SCREEN_TEXT)
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, latlon="10.0,120.0")
    assert result["probe_source"].startswith("auto --run-probe")
    assert result["probe_verdicts"].get("mp_physics") == "REJECT"


def test_physics_gray_json_output_schema_via_cli(ledger_csv, main_namelist, probe_screen_file, tmp_path):
    json_path = tmp_path / "out.json"
    rc = recommend.main([
        "--ledger", ledger_csv, "--namelist", main_namelist, "--case-name", "synthtest",
        "--hardware-id", HARDWARE_ID_TRAIN, "--compliance", "PHYSICS-GRAY",
        "--probe-screen", probe_screen_file, "--json", str(json_path),
    ])
    assert rc == 0
    with open(json_path) as fh:
        data = json.load(fh)
    assert data["compliance"] == "PHYSICS-GRAY"
    for key in ("shortlist", "candidates", "warnings", "verify_plan", "auto_apply_disclaimer"):
        assert key in data


def test_physics_gray_format_report_does_not_crash(ledger_csv, main_namelist, probe_screen_file):
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file)
    report = recommend.format_physics_report(result)
    assert "WTune PHYSICS-GRAY template shortlist" in report
    assert "Evidence warnings" in report
    assert "time-to-sol vs P00" in report
    assert "infeasible" in report and "OOF-miss" in report


# ---------------------------------------------------------------------------
# Defect 1 (real physics-campaign smoke, job 1077893): --holdout-case was accepted but never
# actually applied to --physics-ledger rows, so a "holdout" run for a case already in the given
# ledger silently trained on that case's own rows and never warned IN-SAMPLE.
# ---------------------------------------------------------------------------
def test_physics_gray_holdout_case_drops_physics_ledger_rows_and_in_sample_warning(
        tmp_path, ledger_csv, main_namelist, probe_screen_file):
    phys_ledger_path = tmp_path / "physics_runs.csv"
    write_ledger_csv(make_physics_training_rows(), str(phys_ledger_path))
    held_out = PHYS_TRAIN_CASES[0]

    # Recommending for a case already IN the training ledger, WITHOUT --holdout-case: in-sample,
    # and the training-evidence count must include all 6 synthetic cases.
    in_sample = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name=held_out,
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file)
    assert any("IN-SAMPLE" in w and held_out in w for w in in_sample["warnings"])
    assert any("%d case(s)" % len(PHYS_TRAIN_CASES) in w for w in in_sample["warnings"]
               if "training evidence" in w)

    # --holdout-case for that SAME case_base must actually drop its rows (defect 1's fix) --
    # the IN-SAMPLE warning disappears and the training-evidence count drops by exactly one case.
    held = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name=held_out,
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file, holdout_case=held_out)
    assert not any("IN-SAMPLE" in w for w in held["warnings"])
    assert any(("--holdout-case=%r" % held_out) in w and "dropped" in w for w in held["warnings"])
    assert any("%d case(s)" % (len(PHYS_TRAIN_CASES) - 1) in w for w in held["warnings"]
               if "training evidence" in w)


def test_physics_gray_holdout_case_unmatched_warns_zero_dropped(
        tmp_path, ledger_csv, main_namelist, probe_screen_file):
    phys_ledger_path = tmp_path / "physics_runs.csv"
    write_ledger_csv(make_physics_training_rows(), str(phys_ledger_path))
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file, holdout_case="no-such-case")
    assert any("matched ZERO PHYSICS-GRAY training rows" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Defect 2: per-template infeasibility evidence and gating (--max-infeasible-rate).
# ---------------------------------------------------------------------------
def test_physics_gray_infeasible_template_forced_unconfirmed(tmp_path, ledger_csv, main_namelist,
                                                                probe_screen_file):
    rows = make_physics_training_rows()
    # P03 crashes (both reps FAILED) in every one of the 6 training cases -- template_infeasibility
    # must catch this and force UNCONFIRMED regardless of anything else (the real physics-campaign
    # smoke job 1077893 ranked exactly this kind of candidate #1 RECOMMENDABLE).
    for case in PHYS_TRAIN_CASES:
        for rep in (1, 2):
            rows.append(_phys_row("%s_P03_r%d" % (case, rep), "P03", "dynamics",
                                   {"time_step": "18"}, dict(PHYS_NATIVE, time_step=18.0), "C02",
                                   rep, None, case=case))
    phys_ledger_path = tmp_path / "physics_runs.csv"
    write_ledger_csv(rows, str(phys_ledger_path))

    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file)
    by_t = {c["template"]: c for c in result["shortlist"]}
    assert by_t["P03"]["status"] == "UNCONFIRMED"
    assert by_t["P03"]["infeasibility"]["n_cases_failed"] == len(PHYS_TRAIN_CASES)
    assert by_t["P03"]["infeasibility"]["rate"] == 1.0
    assert any("infeasible in %d/%d" % (len(PHYS_TRAIN_CASES), len(PHYS_TRAIN_CASES)) in r
               for r in by_t["P03"]["reasons"])

    # --max-infeasible-rate=1.0 (accept up to 100% observed failure) lifts the infeasibility gate
    # alone.
    lenient = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file, max_infeasible_rate=1.0)
    by_t_lenient = {c["template"]: c for c in lenient["shortlist"]}
    assert not any("infeasible in" in r for r in by_t_lenient["P03"]["reasons"])


# ---------------------------------------------------------------------------
# Defect 3: leave-one-case-out out-of-fold dangerous-miss gating -- a single-case training ledger
# gives ZERO leave-one-case-out folds, which must read as "insufficient evidence", never as a
# clean pass.
# ---------------------------------------------------------------------------
def test_physics_gray_insufficient_oof_evidence_forces_unconfirmed(
        tmp_path, ledger_csv, main_namelist, probe_screen_file):
    rows = make_physics_training_rows(cases=["onlycase"])
    phys_ledger_path = tmp_path / "physics_runs.csv"
    write_ledger_csv(rows, str(phys_ledger_path))

    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file)
    by_t = {c["template"]: c for c in result["shortlist"]}
    assert by_t["P01"]["status"] == "UNCONFIRMED"
    assert by_t["P01"]["dangerous_miss"] is None
    assert any("insufficient out-of-fold fidelity evidence" in r for r in by_t["P01"]["reasons"])


# ---------------------------------------------------------------------------
# Defect 5: isc21's known bit-identical (floor_zero) TWIN_DECOMP=C06 must be overridden to C13;
# every other case keeps the generic default; the general floor_zero hint is always printed.
# ---------------------------------------------------------------------------
def test_default_twin_decomp_known_floor_zero_override():
    twin, note = recommend._default_twin_decomp("C02", "isc21")
    assert twin == "C13"
    assert note is not None and "isc21" in note and "TWIN_DECOMP=C13" in note

    twin2, note2 = recommend._default_twin_decomp("C02", "karina")
    assert twin2 == "C06"
    assert note2 is None

    # generic collision-avoidance still fires when DECOMP itself resolves to the module default twin
    twin3, note3 = recommend._default_twin_decomp("C06", "karina")
    assert twin3 == "C13"
    assert note3 is None


def test_physics_gray_verify_plan_uses_twin_override_for_isc21(ledger_csv, main_namelist,
                                                                  probe_screen_file):
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="isc21",
        hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file, decomp="C02")
    assert result["verify_plan"]["twin_decomp"] == "C13"
    assert any("isc21" in w and "TWIN_DECOMP=C13" in w for w in result["warnings"])
    assert any("floor_zero hint" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Lead-review gate A (2026-09-15): template track-record prior (--max-template-large-rate).
# ---------------------------------------------------------------------------
def test_physics_gray_track_record_gate_reason_and_flag(tmp_path, ledger_csv, main_namelist,
                                                          probe_screen_file):
    # make_physics_training_rows()'s P02 is LARGE-DRIFT in every one of the 6 synthetic cases --
    # a track record clearly over the default 0.5 threshold.
    rows = make_physics_training_rows()
    phys_ledger_path = tmp_path / "physics_runs.csv"
    write_ledger_csv(rows, str(phys_ledger_path))

    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file)
    p02 = next(c for c in result["shortlist"] if c["template"] == "P02")
    assert p02["status"] == "UNCONFIRMED"
    assert p02["template_track_record"]["rate"] == 1.0
    assert any("LARGE-DRIFT in %d/%d training cases" % (len(PHYS_TRAIN_CASES), len(PHYS_TRAIN_CASES))
               in r for r in p02["reasons"])

    lenient = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file, max_template_large_rate=1.01)
    p02_lenient = next(c for c in lenient["shortlist"] if c["template"] == "P02")
    assert not any("LARGE-DRIFT in" in r for r in p02_lenient["reasons"])

    report = recommend.format_physics_report(result)
    assert "train LARGE" in report
    assert "LARGE %d/%d" % (len(PHYS_TRAIN_CASES), len(PHYS_TRAIN_CASES)) in report


# ---------------------------------------------------------------------------
# Lead-review gate B (2026-09-15): speed-benefit requirement (--min-speedup-pct).
# ---------------------------------------------------------------------------
def test_physics_gray_speed_gate_no_benefit_forces_unconfirmed(tmp_path, ledger_csv, main_namelist,
                                                                 probe_screen_file):
    # P01 measured essentially UNCHANGED speed (identical s_per_step to P00) across all 6
    # training cases, always WITHIN-FLOOR -- the motivating real example: isc21/P02 measured
    # ~0% per step and must not be recommended as a speed change just for being fidelity-safe.
    rows = []
    for case in PHYS_TRAIN_CASES:
        for rep in (1, 2):
            rows.append(_phys_row("%s_P00_r%d" % (case, rep), "P00", "none", {}, PHYS_NATIVE,
                                   "C02", rep, 1.0, case=case))
        for rep in (1, 2):
            rows.append(_phys_row("%s_P01_r%d" % (case, rep), "P01", "radiation",
                                   {"swint_opt": "1"}, PHYS_NATIVE, "C02", rep, 1.0,
                                   fidelity_label="WITHIN-FLOOR", nd=0.01,
                                   probe={"swint_opt": "CONFIRM"}, case=case))
    phys_ledger_path = tmp_path / "physics_runs.csv"
    write_ledger_csv(rows, str(phys_ledger_path))

    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[str(phys_ledger_path)],
        probe_screen=probe_screen_file)
    p01 = next(c for c in result["shortlist"] if c["template"] == "P01")
    assert p01["status"] == "UNCONFIRMED"
    assert any("no predicted speed benefit" in r for r in p01["reasons"])

    report = recommend.format_physics_report(result)
    assert "time-to-sol vs P00" in report


# ---------------------------------------------------------------------------
# WTune D4: --space full / --propose-batch (recommend_full_space, run_propose_batch). Uses its
# OWN ledger fixture with hardware_id="f1-spr-icpnp" (a REAL entry in features.HARDWARE_TABLE) --
# decomp_space.enumerate_space reads that table unconditionally (NUMA-aligned omp options), unlike
# the STRICT-templates path above, which never touches it and so tolerates the shared fixtures'
# synthetic HARDWARE_ID_TRAIN.
# ---------------------------------------------------------------------------
FULL_SPACE_HARDWARE = "f1-spr-icpnp"
FULL_SPACE_OMP_OPTIONS = [1, 2]
FULL_SPACE_TILE_OPTIONS = dict(numtiles_multiples=(1,), max_numtiles=16, include_2d=True,
                                tile_sz_variants=())


@pytest.fixture
def full_space_ledger_csv(tmp_path):
    rows = []
    for c in TRAIN_CASES:
        rows += _case_rows(c, FULL_SPACE_HARDWARE, 901, 901, 45, BASE_LOG_S[c])
    path = tmp_path / "full_space_runs.csv"
    write_ledger_csv(rows, str(path))
    return str(path)


def test_space_full_returns_only_legal_candidates(full_space_ledger_csv, main_namelist):
    result = recommend.recommend_full_space(
        ledger_path=full_space_ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=FULL_SPACE_HARDWARE, nodes=4, top_k=25,
        omp_options=FULL_SPACE_OMP_OPTIONS, tile_options=FULL_SPACE_TILE_OPTIONS)

    assert result["total_candidates"] > 0
    assert 0 < len(result["shortlist"]) <= 25

    # cross-check every reported shape against decomp_space's own ground-truth legality gate
    for e in result["shortlist"]:
        legal = decomp_space.legal_shapes_exact([901], [901], e["n_ranks"])
        assert (e["nproc_x"], e["nproc_y"]) in legal, "reported candidate %s is ILLEGAL" % e

    # total_candidates matches an independent re-enumeration with the same parameters
    _candidates2, axis2 = decomp_space.enumerate_space(
        dict(e_we=901, e_sn=901, e_vert=45, max_dom=1), total_cores=448, nodes=4,
        hardware_id=FULL_SPACE_HARDWARE, omp_options=FULL_SPACE_OMP_OPTIONS,
        tile_options=FULL_SPACE_TILE_OPTIONS)
    assert result["total_candidates"] == axis2["total"]


def test_space_full_top_k_size(full_space_ledger_csv, main_namelist):
    result = recommend.recommend_full_space(
        ledger_path=full_space_ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=FULL_SPACE_HARDWARE, nodes=4, top_k=3,
        omp_options=FULL_SPACE_OMP_OPTIONS, tile_options=FULL_SPACE_TILE_OPTIONS)
    assert len(result["shortlist"]) == 3
    assert [e["rank"] for e in result["shortlist"]] == [1, 2, 3]
    assert result["total_candidates"] > 3


def test_space_full_extrapolation_fires_for_unseen_ranks_omp_and_tile_shape(
        full_space_ledger_csv, main_namelist):
    # Training rows (_case_rows) are ALL ranks=448/omp=1/nproc_x=8/nproc_y=56/tile_shape=1d_y --
    # every ranks=224 (omp=2) candidate, and every tile_shape="2d" candidate, must be flagged.
    # top_k covers the WHOLE constrained pool so the assertions do not depend on ranking order.
    result = recommend.recommend_full_space(
        ledger_path=full_space_ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=FULL_SPACE_HARDWARE, nodes=4, top_k=100000,
        omp_options=FULL_SPACE_OMP_OPTIONS, tile_options=FULL_SPACE_TILE_OPTIONS)

    assert any(e["omp_threads"] == 2 for e in result["shortlist"]), (
        "fixture bug: no omp=2 candidate in the enumerated pool")
    assert any(e["tile_shape"] == "2d" for e in result["shortlist"]), (
        "fixture bug: no 2d-tile candidate in the enumerated pool")

    for e in result["shortlist"]:
        if e["omp_threads"] == 2:
            assert e["extrapolated"] is True
            assert any("ranks x omp" in r for r in e["extrapolation_reasons"])
            assert e["label"] == "EXPERIMENTAL"
        if e["tile_shape"] == "2d":
            assert e["extrapolated"] is True
            assert any("tile shape" in r for r in e["extrapolation_reasons"])
            assert e["label"] == "EXPERIMENTAL"

    # a candidate at exactly a MEASURED point (448x1, 8x56, numtiles=8, 1d_y) must NOT be flagged
    measured = next(e for e in result["shortlist"]
                     if e["n_ranks"] == 448 and e["omp_threads"] == 1
                     and e["nproc_x"] == 8 and e["nproc_y"] == 56
                     and e["tile_shape"] == "1d_y" and e["numtiles"] == 8)
    assert measured["extrapolated"] is False
    assert measured["extrapolation_reasons"] == []


def test_space_templates_output_unchanged(ledger_csv, main_namelist, tmp_path, capsys):
    """--space defaults to 'templates' and must not change recommend()'s own output: run the CLI
    with and without an explicit --space templates and diff the JSON byte-for-byte."""
    json1 = tmp_path / "out1.json"
    json2 = tmp_path / "out2.json"
    rc1 = recommend.main([
        "--ledger", ledger_csv, "--namelist", main_namelist, "--case-name", "synthtest",
        "--hardware-id", HARDWARE_ID_TRAIN, "--model", "ridge", "--json", str(json1)])
    capsys.readouterr()
    rc2 = recommend.main([
        "--ledger", ledger_csv, "--namelist", main_namelist, "--case-name", "synthtest",
        "--hardware-id", HARDWARE_ID_TRAIN, "--model", "ridge", "--space", "templates",
        "--json", str(json2)])
    capsys.readouterr()
    assert rc1 == 0 and rc2 == 0
    with open(json1) as f:
        d1 = json.load(f)
    with open(json2) as f:
        d2 = json.load(f)
    assert d1 == d2


def test_propose_batch_writes_valid_config_file(full_space_ledger_csv, main_namelist, tmp_path):
    result, configs_out = recommend.run_propose_batch(
        full_space_ledger_csv, main_namelist, "synthtest", FULL_SPACE_HARDWARE, 4, None, 5)
    assert result["stopped"] is False
    assert 0 < len(configs_out) <= 5

    for cfg in configs_out:
        for key in ("id", "ranks", "omp", "nx", "ny", "numtiles", "dist", "blocktime",
                    "config_source"):
            assert key in cfg
        assert cfg["config_source"] == "generated"

    # accepted by campaign/configs.py's own CONFIG_FILE loader (read-only use)
    out_path = tmp_path / "propose_batch.json"
    with open(out_path, "w") as f:
        json.dump(dict(configs=configs_out), f)
    loaded = campaign_configs.load_generated_configs(str(out_path))
    assert [c["id"] for c in loaded] == [c["id"] for c in configs_out]
    for c in loaded:
        legal = decomp_space.legal_shapes_exact([901], [901], c["ranks"])
        assert (c["nx"], c["ny"]) in legal



# ---------------------------------------------------------------------------
# Accuracy-first mode: recommend.py --objective accuracy (2026-09-17).
#
# The contract has four parts, and each has its own test below:
#   1. --objective perf (and omitting the flag) is unchanged in every byte.
#   2. accuracy mode ranks by the numeric fidelity error, not by predicted speed, while still
#      reporting predicted speed for every candidate.
#   3. a candidate whose floor_ratio <= 1 is "indistinguishable from changing the decomposition"
#      and is NEVER presented as an accuracy improvement.
#   4. the STRICT decomposition stays a free axis -- it does not change the physics, so the
#      fastest legal one is still what gets recommended.
# ---------------------------------------------------------------------------

# MAIN_NAMELIST has no &dynamics group at all, so A02/A03/A04 and D01-D03 cannot resolve against
# it (apply_physics_edit raises "no &dynamics section found" and the candidate comes back not
# applicable). A separate fixture rather than an edit to MAIN_NAMELIST: adding &dynamics there
# would silently make the whole D-series applicable in every pre-existing test too.
ACCURACY_NAMELIST = MAIN_NAMELIST + "\n&dynamics\n diff_opt = 1,\n km_opt = 4,\n /\n"


@pytest.fixture
def accuracy_namelist(tmp_path):
    path = tmp_path / "namelist.input.accuracy"
    path.write_text(ACCURACY_NAMELIST)
    return str(path)


def make_accuracy_training_rows(cases=PHYS_TRAIN_CASES):
    """The perf fixture plus measured chaos-floor ratios, and two extra templates that separate
    the three interesting fidelity regimes:
      P01  nd 0.01, floor_ratio 4.2  -- small real drift, well above the floor
      P02  nd 0.50, floor_ratio 18.0 -- large real drift
      D02  nd 0.02, floor_ratio 0.80 -- BELOW the chaos floor: re-decomposing moves the solution
                                        more than dropping an RK stage does (the real measured
                                        conus12/RK2 number at 12 h was 1.593, i.e. only just
                                        above; 0.80 is the same shape pushed under the floor)
      A01  nd 0.30, floor_ratio 9.5  -- an accuracy-direction template with a real correction
    """
    rows = []
    for case in cases:
        for rep in (1, 2):
            rows.append(_phys_row("%s_P00_r%d" % (case, rep), "P00", "none", {}, PHYS_NATIVE,
                                   "C02", rep, 1.0, case=case))
        for rep in (1, 2):
            rows.append(_phys_row("%s_P01_r%d" % (case, rep), "P01", "radiation",
                                   {"swint_opt": "1"}, PHYS_NATIVE, "C02", rep, 0.95,
                                   fidelity_label="WITHIN-FLOOR", nd=0.01, floor_ratio=4.2,
                                   probe={"swint_opt": "CONFIRM"}, case=case))
        p02_fields = dict(PHYS_NATIVE, radt=30.0)
        for rep in (1, 2):
            rows.append(_phys_row("%s_P02_r%d" % (case, rep), "P02", "radiation",
                                   {"radt": "30", "swint_opt": "1"}, p02_fields, "C02", rep, 0.90,
                                   fidelity_label="LARGE-DRIFT", nd=0.5, floor_ratio=18.0,
                                   probe={"radt": "CONFIRM", "swint_opt": "CONFIRM"}, case=case))
        for rep in (1, 2):
            rows.append(_phys_row("%s_D02_r%d" % (case, rep), "D02", "dynamics", {"rk_ord": "2"},
                                   PHYS_NATIVE, "C02", rep, 0.80,
                                   fidelity_label="WITHIN-FLOOR", nd=0.02, floor_ratio=0.80,
                                   case=case))
        a01_fields = dict(PHYS_NATIVE, time_step=12.0)
        for rep in (1, 2):
            rows.append(_phys_row("%s_A01_r%d" % (case, rep), "A01", "dynamics",
                                   {"time_step": "12"}, a01_fields, "C02", rep, 1.25,
                                   fidelity_label="LARGE-DRIFT", nd=0.30, floor_ratio=9.5,
                                   case=case))
    return rows


@pytest.fixture
def accuracy_physics_ledger(tmp_path):
    path = tmp_path / "physics_runs_accuracy.csv"
    write_ledger_csv(make_accuracy_training_rows(), str(path))
    return str(path)


def _err_keys(entries):
    """The ordering build_accuracy_view actually promises: a known error sorts before an unknown
    one in BOTH directions, so "no evidence" can never outrank a measured number."""
    return [(e["fidelity_error"] is None, e["fidelity_error"]) for e in entries]


def _acc_kwargs(ledger_csv, namelist, probe_screen_file, physics_ledger):
    return dict(ledger_path=ledger_csv, namelist_path=namelist, case_name="synthtest",
                hardware_id=HARDWARE_ID_TRAIN, physics_ledgers=[physics_ledger],
                probe_screen=probe_screen_file)


def _accuracy_result(ledger_csv, namelist, probe_screen_file, physics_ledger):
    return recommend.recommend_physics_gray(
        objective="accuracy",
        **_acc_kwargs(ledger_csv, namelist, probe_screen_file, physics_ledger))


# --- 1. perf mode is untouched -------------------------------------------------------------
def test_objective_perf_is_the_default_and_adds_nothing(ledger_csv, accuracy_namelist,
                                                         probe_screen_file,
                                                         accuracy_physics_ledger):
    common = _acc_kwargs(ledger_csv, accuracy_namelist, probe_screen_file,
                         accuracy_physics_ledger)
    default = recommend.recommend_physics_gray(**common)
    explicit = recommend.recommend_physics_gray(objective="perf", **common)
    assert list(default) == list(explicit)
    assert "accuracy_view" not in default and "objective" not in default
    assert recommend.format_physics_report(default) == recommend.format_physics_report(explicit)
    # the accuracy blocks are genuinely absent, not merely empty
    assert "ACCURACY-FIRST view" not in recommend.format_physics_report(default)


def test_accuracy_mode_leaves_the_perf_shortlist_byte_identical(ledger_csv, accuracy_namelist,
                                                                 probe_screen_file,
                                                                 accuracy_physics_ledger):
    common = _acc_kwargs(ledger_csv, accuracy_namelist, probe_screen_file,
                         accuracy_physics_ledger)
    perf = recommend.recommend_physics_gray(**common)
    acc = recommend.recommend_physics_gray(objective="accuracy", **common)
    assert perf["shortlist"] == acc["shortlist"]
    assert perf["verify_plan"] == acc["verify_plan"]
    assert perf["warnings"] == acc["warnings"]
    assert acc["objective"] == "accuracy"


def test_unknown_objective_is_rejected(ledger_csv, main_namelist, probe_screen_file):
    with pytest.raises(ValueError, match="unknown objective"):
        recommend.recommend_physics_gray(
            ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
            hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file, objective="fastest")


def test_cli_objective_accuracy_refuses_strict_compliance(ledger_csv, main_namelist, capsys):
    rc = recommend.main(["--ledger", ledger_csv, "--namelist", main_namelist,
                         "--case-name", "synthtest", "--hardware-id", HARDWARE_ID_TRAIN,
                         "--model", "ridge", "--objective", "accuracy"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "requires --compliance PHYSICS-GRAY" in err
    assert "never change the physics" in err


# --- 2. ranking is by fidelity error, speed still reported ---------------------------------
def test_accuracy_mode_ranks_cost_levers_by_least_departure(ledger_csv, accuracy_namelist,
                                                             probe_screen_file,
                                                             accuracy_physics_ledger):
    result = _accuracy_result(ledger_csv, accuracy_namelist, probe_screen_file,
                              accuracy_physics_ledger)
    av = result["accuracy_view"]
    ranked = av["ranked_cost_levers"]
    assert ranked, "fixture bug: expected applicable cost-lever templates"
    assert _err_keys(ranked) == sorted(_err_keys(ranked)), "ascending fidelity error"
    assert all(e["direction"] == "cost" for e in ranked)
    assert [e["accuracy_rank"] for e in ranked] == list(range(1, len(ranked) + 1))
    # predicted speed is still reported -- accuracy mode changes the RANKING, not the evidence
    assert all("predicted_time_to_solution_pct" in e for e in av["entries"])
    assert any(e["predicted_time_to_solution_pct"] is not None for e in av["entries"])
    # the perf shortlist orders by predicted speed, so the two orderings must genuinely differ
    perf_order = [c["template"] for c in result["shortlist"]]
    assert [e["template"] for e in ranked] != [t for t in perf_order if t in
                                               {e["template"] for e in ranked}]


def test_accuracy_levers_are_ranked_by_largest_correction_not_smallest(
        ledger_csv, accuracy_namelist, probe_screen_file, accuracy_physics_ledger):
    """nd is a distance from the reference. For a cost lever that IS the error it introduced, so
    lower is better. For an A-series template the same number is the correction the extra work
    applies, and a small one means the reference was already converged -- ranking them ascending
    alongside the cost levers would reward exactly the templates that do nothing."""
    av = _accuracy_result(ledger_csv, accuracy_namelist, probe_screen_file,
                          accuracy_physics_ledger)["accuracy_view"]
    ranked = av["ranked_accuracy_levers"]
    assert ranked, "fixture bug: expected applicable A-series templates"
    assert all(e["direction"] == "accuracy" for e in ranked)
    assert all(e["template"].startswith("A") for e in ranked)
    measured = [e["fidelity_error"] for e in ranked if e["fidelity_error"] is not None]
    assert measured == sorted(measured, reverse=True)
    # an un-measured template must never outrank one with a real measured correction
    first_unknown = next((i for i, e in enumerate(ranked) if e["fidelity_error"] is None),
                         len(ranked))
    assert all(e["fidelity_error"] is not None for e in ranked[:first_unknown])


def test_template_direction_comes_from_the_json():
    assert recommend.template_direction("A01") == "accuracy"
    assert recommend.template_direction("A05") == "accuracy"
    for tid in ("P00", "P02", "P03", "D01", "D02", "D06"):
        assert recommend.template_direction(tid) == "cost", tid
    # a template id the JSON has never heard of is a cost lever, not a crash
    assert recommend.template_direction("Z99") == "cost"


def test_measured_fidelity_by_template_reads_both_numbers(accuracy_physics_ledger):
    import benchmark
    rows = benchmark.load_ledger(accuracy_physics_ledger)
    measured = recommend.measured_fidelity_by_template(rows)
    assert measured["D02"]["floor_ratio"]["median"] == pytest.approx(0.80)
    assert measured["D02"]["nd"]["median"] == pytest.approx(0.02)
    assert measured["P02"]["floor_ratio"]["median"] == pytest.approx(18.0)
    # rep 2 carries no fidelity at all, so one measurement per training case
    assert measured["A01"]["nd"]["n"] == len(PHYS_TRAIN_CASES)
    assert sorted(measured["A01"]["nd"]["cases"]) == sorted(PHYS_TRAIN_CASES)


def test_measured_fidelity_also_reads_a_parsed_ledger_column():
    """ledger.adapt_campaign_ledger parses the notes tokens into row fields; rows arriving that
    way must give the same answer as rows that still carry only the raw notes string."""
    row = ledger.new_row(run_id="r1", case_base="c", source_dataset="strict_campaign",
                          notes="mode=physics template=A02 probe=none")
    row["fidelity_nd_max"], row["fidelity_floor_ratio_max"] = 0.25, 3.5
    measured = recommend.measured_fidelity_by_template([row])
    assert measured["A02"]["nd"]["median"] == pytest.approx(0.25)
    assert measured["A02"]["floor_ratio"]["median"] == pytest.approx(3.5)


# --- 3. the chaos-floor hard rule ----------------------------------------------------------
def test_below_the_chaos_floor_is_never_an_accuracy_improvement(ledger_csv, accuracy_namelist,
                                                                 probe_screen_file,
                                                                 accuracy_physics_ledger):
    av = _accuracy_result(ledger_csv, accuracy_namelist, probe_screen_file,
                          accuracy_physics_ledger)["accuracy_view"]
    by_t = {e["template"]: e for e in av["entries"]}
    d02 = by_t["D02"]
    assert d02["floor_ratio"] == pytest.approx(0.80)
    assert d02["below_floor"] is True
    assert "indistinguishable from changing the decomposition" in d02["below_floor_note"]
    # never ranked as an improvement, in either direction...
    assert d02 not in av["ranked_cost_levers"]
    assert d02 not in av["ranked_accuracy_levers"]
    assert d02["accuracy_rank"] is None
    # ...and never allowed to anchor the Pareto front, however cheap it is
    assert d02["on_pareto"] is False
    assert d02["pareto_excluded_reason"] == "below the chaos floor"
    assert d02 in av["below_floor"]
    assert av["chaos_floor_ratio"] == 1.0
    # a candidate whose drift IS above the floor is unaffected
    assert by_t["P01"]["below_floor"] is False


@pytest.mark.parametrize("floor_ratio,expect_below", [
    (0.5, True), (0.999, True), (1.0, True),     # at the floor is still AT the floor
    (1.001, False), (1.593, False), (18.0, False),
])
def test_chaos_floor_boundary_is_inclusive(floor_ratio, expect_below):
    """floor_ratio <= 1 is the rule: a ratio of exactly 1 means the candidate moved the solution
    exactly as much as re-decomposing did, which is not a distinguishable change. 1.593 is the
    real measured conus12/RK2 number and is ABOVE the floor -- the rule must not swallow it."""
    cand = dict(template="X01", family="dynamics", status="UNCONFIRMED",
                predicted_time_to_solution_pct=-1.0, time_to_solution_interval_pct=None,
                predicted_fidelity_label=None, predicted_fidelity_nd=0.05)
    measured = {"X01": {"nd": {"median": 0.05, "n": 1, "cases": ["c"]},
                        "floor_ratio": {"median": floor_ratio, "n": 1, "cases": ["c"]}}}
    entry = recommend._accuracy_entry(cand, measured)
    assert entry["below_floor"] is expect_below
    assert (entry["below_floor_note"] is not None) is expect_below


def test_accuracy_entry_prefers_the_prediction_but_falls_back_to_measurement():
    measured = {"X01": {"nd": {"median": 0.42, "n": 3, "cases": ["a", "b"]},
                        "floor_ratio": {"median": 5.0, "n": 3, "cases": ["a", "b"]}}}
    base = dict(template="X01", family="dynamics", status="UNCONFIRMED",
                predicted_time_to_solution_pct=-1.0, time_to_solution_interval_pct=None,
                predicted_fidelity_label=None)
    predicted = recommend._accuracy_entry(dict(base, predicted_fidelity_nd=0.07), measured)
    assert predicted["fidelity_error"] == pytest.approx(0.07)
    assert "predicted" in predicted["fidelity_error_source"]
    fallback = recommend._accuracy_entry(dict(base, predicted_fidelity_nd=None), measured)
    assert fallback["fidelity_error"] == pytest.approx(0.42)
    assert "measured median of 3 row(s) over 2 training case(s)" in fallback["fidelity_error_source"]
    # the chaos floor has no regression at all -- it is always the measured number
    assert fallback["floor_ratio_source"].startswith("measured median")
    nothing = recommend._accuracy_entry(dict(base, predicted_fidelity_nd=None), {})
    assert nothing["fidelity_error"] is None and nothing["floor_ratio"] is None


def test_report_states_the_hard_rule_and_the_measured_example(ledger_csv, accuracy_namelist,
                                                               probe_screen_file,
                                                               accuracy_physics_ledger):
    result = _accuracy_result(ledger_csv, accuracy_namelist, probe_screen_file,
                              accuracy_physics_ledger)
    text = recommend.format_physics_report(result)
    assert "ACCURACY-FIRST view" in text
    assert "indistinguishable from changing the decomposition" in text
    assert "1.593" in text          # the measured conus12/RK2 example
    assert "NOT accuracy improvements" in text
    assert "Pareto front over (predicted cost, fidelity error)" in text
    assert "free axis" in text
    # the perf shortlist table is still printed above it, unchanged
    assert "-- Shortlist (" in text


# --- the Pareto front ----------------------------------------------------------------------
def _pareto_entry(template, cost, error, below_floor=False):
    return dict(template=template, direction="cost", below_floor=below_floor,
                predicted_time_to_solution_pct=cost, fidelity_error=error,
                on_pareto=False, pareto_excluded_reason=None)


def test_pareto_front_keeps_only_non_dominated_points():
    a = _pareto_entry("A", -10.0, 0.50)   # cheapest, worst error      -> perf-first end
    b = _pareto_entry("B", -5.0, 0.20)    # a genuine middle trade
    c = _pareto_entry("C", +5.0, 0.01)    # most expensive, best error -> acc-first end
    d = _pareto_entry("D", -4.0, 0.40)    # dominated by B on both axes
    front = recommend.pareto_front([a, b, c, d])
    assert [e["template"] for e in front] == ["A", "B", "C"]
    assert d["on_pareto"] is False and d["pareto_excluded_reason"] is None
    assert [e["on_pareto"] for e in (a, b, c)] == [True, True, True]
    # the two ends of the front are exactly perf-first and acc-first
    assert min(front, key=lambda e: e["predicted_time_to_solution_pct"])["template"] == "A"
    assert min(front, key=lambda e: e["fidelity_error"])["template"] == "C"


def test_pareto_front_records_why_a_point_was_left_off():
    below = _pareto_entry("BF", -20.0, 0.001, below_floor=True)
    no_cost = _pareto_entry("NC", None, 0.10)
    no_error = _pareto_entry("NE", -1.0, None)
    ok = _pareto_entry("OK", -2.0, 0.05)
    front = recommend.pareto_front([below, no_cost, no_error, ok])
    assert [e["template"] for e in front] == ["OK"]
    assert below["pareto_excluded_reason"] == "below the chaos floor"
    assert no_cost["pareto_excluded_reason"] == "no predicted time-to-solution"
    assert no_error["pareto_excluded_reason"] == "no fidelity error (predicted or measured)"


def test_pareto_front_keeps_identical_points_together():
    a, b = _pareto_entry("A", -3.0, 0.1), _pareto_entry("B", -3.0, 0.1)
    front = recommend.pareto_front([a, b])
    assert {e["template"] for e in front} == {"A", "B"}


def test_pareto_front_is_marked_on_the_real_view(ledger_csv, accuracy_namelist,
                                                  probe_screen_file, accuracy_physics_ledger):
    av = _accuracy_result(ledger_csv, accuracy_namelist, probe_screen_file,
                          accuracy_physics_ledger)["accuracy_view"]
    front = av["pareto_front"]
    assert front, "expected at least one operating point with both coordinates known"
    assert all(e["on_pareto"] for e in front)
    assert not any(e["below_floor"] for e in front)
    costs = [e["predicted_time_to_solution_pct"] for e in front]
    assert costs == sorted(costs)
    # a front point is never beaten on both axes at once
    for e in front:
        assert not any(o is not e
                       and o["predicted_time_to_solution_pct"] <= e["predicted_time_to_solution_pct"]
                       and o["fidelity_error"] <= e["fidelity_error"]
                       and (o["predicted_time_to_solution_pct"] < e["predicted_time_to_solution_pct"]
                            or o["fidelity_error"] < e["fidelity_error"])
                       for o in front)


# --- 4. the decomposition is a free axis ---------------------------------------------------
def test_decomposition_stays_the_fastest_legal_one(ledger_csv, accuracy_namelist,
                                                    probe_screen_file, accuracy_physics_ledger):
    common = _acc_kwargs(ledger_csv, accuracy_namelist, probe_screen_file,
                         accuracy_physics_ledger)
    perf = recommend.recommend_physics_gray(**common)
    acc = recommend.recommend_physics_gray(objective="accuracy", **common)
    assert perf["verify_plan"]["decomp"] == acc["verify_plan"]["decomp"]
    note = acc["accuracy_view"]["decomposition_note"]
    assert "free axis" in note.lower()
    assert "never change the physics" in note
    assert perf["verify_plan"]["decomp"] in note


def test_accuracy_verify_plan_is_two_stage_and_names_the_a_series(ledger_csv, accuracy_namelist,
                                                                   probe_screen_file,
                                                                   accuracy_physics_ledger):
    av = _accuracy_result(ledger_csv, accuracy_namelist, probe_screen_file,
                          accuracy_physics_ledger)["accuracy_view"]
    vp = av["verify_plan"]
    assert vp["accuracy_templates"], "expected applicable A-series templates"
    assert all(t.startswith("A") for t in vp["accuracy_templates"])
    assert "MODE=physics" in vp["stage1"]
    assert "+".join(vp["accuracy_templates"]) in vp["stage1"]
    # stage 2 is the whole point: measure against the CONVERGED solution, not against P00
    assert any("--reference <A-TEMPLATE" in c for c in vp["stage2"])
    assert "STAGE 2" in vp["note"] and "better-converged" in vp["note"]


def test_accuracy_mode_works_with_no_campaign_data_at_all(ledger_csv, accuracy_namelist,
                                                           probe_screen_file):
    """Today's production state: no PHYSICS-GRAY rows exist, so there is no fidelity error and no
    chaos-floor measurement for anything. Accuracy mode must still produce an honest, empty-front
    report rather than inventing a ranking."""
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=accuracy_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file, objective="accuracy")
    av = result["accuracy_view"]
    assert av["entries"], "every applicable template must still be listed"
    assert all(e["fidelity_error"] is None for e in av["entries"])
    assert all(e["floor_ratio"] is None for e in av["entries"])
    assert av["below_floor"] == []
    assert av["pareto_front"] == []
    assert all(e["pareto_excluded_reason"] is not None for e in av["entries"])
    text = recommend.format_physics_report(result)
    assert "ACCURACY-FIRST view" in text
    assert "(empty" in text


def test_cli_objective_accuracy_end_to_end(tmp_path, ledger_csv, accuracy_namelist,
                                            probe_screen_file, accuracy_physics_ledger, capsys):
    out_json = tmp_path / "acc.json"
    rc = recommend.main(["--ledger", ledger_csv, "--namelist", accuracy_namelist,
                         "--case-name", "synthtest", "--hardware-id", HARDWARE_ID_TRAIN,
                         "--compliance", "PHYSICS-GRAY", "--model", "ridge",
                         "--physics-ledger", accuracy_physics_ledger,
                         "--probe-screen", probe_screen_file,
                         "--objective", "accuracy", "--json", str(out_json)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ACCURACY-FIRST view" in out
    with open(out_json) as fh:
        payload = json.load(fh)
    assert payload["objective"] == "accuracy"
    assert payload["accuracy_view"]["chaos_floor_ratio"] == 1.0


def test_cli_without_objective_matches_objective_perf(ledger_csv, accuracy_namelist,
                                                       probe_screen_file,
                                                       accuracy_physics_ledger, capsys):
    argv = ["--ledger", ledger_csv, "--namelist", accuracy_namelist, "--case-name", "synthtest",
            "--hardware-id", HARDWARE_ID_TRAIN, "--compliance", "PHYSICS-GRAY", "--model", "ridge",
            "--physics-ledger", accuracy_physics_ledger, "--probe-screen", probe_screen_file]
    assert recommend.main(argv) == 0
    default_out = capsys.readouterr().out
    assert recommend.main(argv + ["--objective", "perf"]) == 0
    explicit_out = capsys.readouterr().out
    assert default_out == explicit_out
    assert "ACCURACY-FIRST view" not in default_out


# ---------------------------------------------------------------------------
# Cost-share-aware template filtering (2026-09-17): the ledger's profile_*_share columns bound how
# much a template can possibly save. SC26-P1's radiation is 1.9% of step cost, SC26-P2's is 47%
# (~/SC26-WRF docs/testcase-atlas), so the same radt template is worth measuring on one and not the
# other. A low share DEMOTES (still listed, with the number and the reason); it never hides.
# ---------------------------------------------------------------------------
def _sl(template, family, y=-0.05, status="RECOMMENDABLE"):
    return dict(template=template, family=family, status=status, reasons=[], predicted_y=y)


def test_bucket_share_demotes_a_low_share_family_and_says_why():
    shortlist = [_sl("P02", "radiation"), _sl("D03", "dynamics")]
    shares = {"profile_radiation_share": 0.019, "profile_dynamics_share": 0.60}
    out, notes = recommend.annotate_bucket_shares(shortlist, shares, min_bucket_share=0.05)
    by_id = {e["template"]: e for e in out}
    p02 = by_id["P02"]
    assert p02["bucket"] == "profile_radiation_share"
    assert p02["bucket_share"] == pytest.approx(0.019)
    assert p02["upper_bound_saving"] == pytest.approx(0.019)   # cannot save more than the bucket
    assert p02["bucket_demoted"] is True
    assert p02["status"] == "UNCONFIRMED"
    assert any("DEMOTED on cost share" in r and "1.9%" in r for r in p02["reasons"])
    assert by_id["D03"]["bucket_demoted"] is False
    assert by_id["D03"]["status"] == "RECOMMENDABLE"
    # demoted candidates are ranked last, not removed
    assert [e["template"] for e in out] == ["D03", "P02"]
    assert any("DEMOTED" in n and "P02" in n for n in notes)


def test_bucket_share_keeps_a_high_share_family_even_at_the_same_threshold():
    """The SC26-P2 half of the contrast: same template, same threshold, 47% bucket -> not demoted."""
    out, _notes = recommend.annotate_bucket_shares(
        [_sl("P02", "radiation")], {"profile_radiation_share": 0.47}, min_bucket_share=0.05)
    assert out[0]["bucket_demoted"] is False
    assert out[0]["status"] == "RECOMMENDABLE"
    assert out[0]["upper_bound_saving"] == pytest.approx(0.47)


def test_bucket_share_absent_profile_row_changes_nothing():
    shortlist = [_sl("P02", "radiation"), _sl("D03", "dynamics")]
    out, notes = recommend.annotate_bucket_shares(shortlist, {}, min_bucket_share=0.05)
    assert out is shortlist
    assert all("bucket" not in e and "bucket_demoted" not in e for e in out)
    assert all(e["status"] == "RECOMMENDABLE" for e in out)
    assert any("no profile row" in n for n in notes)


def test_bucket_share_never_demotes_a_diagnostics_template():
    """D04 (debug_level/history cadence) acts on no physics bucket, so no share can bound it --
    demoting it on a cost share would be a category error, not a conservative default."""
    shares = {"profile_radiation_share": 0.001, "profile_dynamics_share": 0.001,
              "profile_micro_share": 0.001}
    out, _notes = recommend.annotate_bucket_shares([_sl("D04", "diagnostics")], shares,
                                                    min_bucket_share=0.05)
    assert out[0]["bucket"] is None
    assert out[0]["bucket_demoted"] is False
    assert out[0]["status"] == "RECOMMENDABLE"


def test_bucket_share_with_no_measurement_for_that_bucket_is_not_a_demotion():
    out, _notes = recommend.annotate_bucket_shares(
        [_sl("P04", "microphysics")], {"profile_radiation_share": 0.47}, min_bucket_share=0.05)
    assert out[0]["bucket"] == "profile_micro_share"
    assert out[0]["bucket_share"] is None
    assert out[0]["bucket_demoted"] is False
    assert any("no measured profile_micro_share" in r for r in out[0]["reasons"])


def test_family_bucket_map_covers_every_template_family_in_use():
    families = {t.get("family") for t in campaign_configs.PHYSICS_TEMPLATES.values()}
    assert families <= set(recommend.FAMILY_PROFILE_BUCKET), families
    # pbl_surface and nesting share the dynamics bucket: the ledger has no profile_pbl_share column
    assert "profile_pbl_share" not in ledger.LEDGER_FIELDS
    assert recommend.FAMILY_PROFILE_BUCKET["pbl_surface"] == "profile_dynamics_share"
    assert recommend.FAMILY_PROFILE_BUCKET["nesting"] == "profile_dynamics_share"


def test_case_profile_shares_reads_this_cases_rows_only_and_takes_the_median():
    rows = [dict(case_base="a", profile_radiation_share=0.02),
            dict(case_base="a", profile_radiation_share=0.04),
            dict(case_base="b", profile_radiation_share=0.47),
            dict(case_base="a", profile_micro_share=None)]
    assert recommend.case_profile_shares(rows, "a") == {"profile_radiation_share": 0.03}
    assert recommend.case_profile_shares(rows, "b") == {"profile_radiation_share": 0.47}
    assert recommend.case_profile_shares(rows, "c") == {}
    # percent-shaped values normalise to fractions (the atlas quotes 1.9% / 47%)
    assert recommend.case_profile_shares(
        [dict(case_base="a", profile_radiation_share=47.0)], "a") == {
            "profile_radiation_share": pytest.approx(0.47)}


def test_physics_gray_end_to_end_demotes_a_low_radiation_share_case(ledger_csv, main_namelist,
                                                                     probe_screen_file):
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file,
        profile_shares={"profile_radiation_share": 0.019, "profile_dynamics_share": 0.55})
    by_id = {c["template"]: c for c in result["shortlist"]}
    assert by_id["P01"]["bucket_demoted"] is True
    assert by_id["P02"]["bucket_demoted"] is True
    assert by_id["D03"]["bucket_demoted"] is False
    # the verify plan must not spend 4-node slots on a bucket that cannot pay
    assert not any(t in ("P01", "P02", "P06", "A05") for t in result["verify_plan"]["pconfigs"])
    assert "1.9%" in recommend.format_physics_report(result)


def test_physics_gray_end_to_end_without_profile_shares_is_unchanged(ledger_csv, main_namelist,
                                                                      probe_screen_file):
    common = dict(ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
                  hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file)
    result = recommend.recommend_physics_gray(**common)
    assert result["profile_shares"] == {}
    assert all("bucket_demoted" not in c for c in result["shortlist"])
    assert any("no profile row" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Task-mandated frozen keys (2026-09-17). SC26-P1's problem statement fixes time_step = 10 and
# debug_level = 50; a recommendation that changes one is a rules violation, not a trade-off.
# Tested with a SYNTHETIC frozen_keys list -- no campaign/cases.json entry declares one.
# ---------------------------------------------------------------------------
def test_frozen_key_templates_are_excluded_with_the_reason_shown(ledger_csv, main_namelist,
                                                                  probe_screen_file):
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file,
        frozen_keys=["time_step"])
    cands = {c["template"]: c for c in result["candidates"]}
    # P03 (time_step x1.2) and A01 (time_step x0.8) both edit the frozen key
    for tid in ("P03", "A01"):
        assert cands[tid]["applicable"] is False, tid
        assert "time_step" in cands[tid]["skip_reason"], tid
        assert cands[tid]["frozen_key_conflict"] == ["time_step"], tid
    assert cands["P01"]["applicable"] is True
    by_id = {c["template"]: c for c in result["shortlist"]}
    assert by_id["P03"]["status"] == "UNCONFIRMED"
    assert any("time_step" in r for r in by_id["P03"]["reasons"])
    assert "P03" not in result["verify_plan"]["pconfigs"]
    report = recommend.format_physics_report(result)
    assert "frozen key" in report
    assert result["frozen_keys"] == ["time_step"]


def test_no_frozen_keys_is_the_default_and_excludes_nothing(ledger_csv, main_namelist,
                                                             probe_screen_file):
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=main_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file)
    assert result["frozen_keys"] == []
    cands = {c["template"]: c for c in result["candidates"]}
    assert cands["P03"]["applicable"] is True
    assert all("frozen_key_conflict" not in c for c in result["candidates"])


def test_exclude_frozen_key_templates_also_catches_a_generic_per_case_pairing():
    """A key that only appears in the RESOLVED changes (a generic extra-edit rule, not the
    template's own JSON edits) must still be caught -- otherwise the pairing smuggles it back in."""
    cands = [dict(template="P04", family="microphysics", applicable=True, skip_reason=None,
                  changes={"mp_physics": {"from": 8, "to": 6},
                           "aer_opt": {"from": 3, "to": 0}})]
    out, notes = recommend.exclude_frozen_key_templates(cands, ["aer_opt"], "synth",
                                                         templates={"P04": {"edits": []}})
    assert out[0]["applicable"] is False
    assert out[0]["frozen_key_conflict"] == ["aer_opt"]
    assert any("aer_opt" in n for n in notes)


# ---------------------------------------------------------------------------
# WRF extensions (chem/hydro/WRFDA/moving nest): grid/rank-only notes on every recommend path
# ---------------------------------------------------------------------------
_CHEM_NOTE = "EXTENSION: WRF-Chem detected"


@pytest.fixture
def chem_namelist(tmp_path):
    path = tmp_path / "chem" / "namelist.input"
    path.parent.mkdir()
    path.write_text(MAIN_NAMELIST + "\n &chem\n chem_opt = 301,\n chemdt = 3,\n photdt = 30,\n /\n")
    return str(path)


def test_recommend_default_path_carries_extension_note(ledger_csv, chem_namelist):
    result = recommend.recommend(ledger_path=ledger_csv, namelist_path=chem_namelist,
                                 case_name="synthtest", hardware_id=HARDWARE_ID_TRAIN)
    assert any(w.startswith(_CHEM_NOTE) for w in result["warnings"])


def test_recommend_physics_gray_carries_extension_note(ledger_csv, chem_namelist, probe_screen_file):
    result = recommend.recommend_physics_gray(
        ledger_path=ledger_csv, namelist_path=chem_namelist, case_name="synthtest",
        hardware_id=HARDWARE_ID_TRAIN, probe_screen=probe_screen_file)
    assert any(w.startswith(_CHEM_NOTE) for w in result["warnings"])


def test_recommend_full_space_carries_extension_note(full_space_ledger_csv, chem_namelist):
    result = recommend.recommend_full_space(
        ledger_path=full_space_ledger_csv, namelist_path=chem_namelist, case_name="synthtest",
        hardware_id=FULL_SPACE_HARDWARE, nodes=4, top_k=3,
        omp_options=FULL_SPACE_OMP_OPTIONS, tile_options=FULL_SPACE_TILE_OPTIONS)
    assert any(w.startswith(_CHEM_NOTE) for w in result["warnings"])


def test_extension_dmpar_only_never_proposes_openmp(ledger_csv, full_space_ledger_csv, chem_namelist,
                                                    monkeypatch):
    import wtune_extensions
    monkeypatch.setattr(wtune_extensions, "CHEM_MIN_J_PATCH", 1)   # isolate the omp cap from the j guard
    r = recommend.recommend(ledger_path=ledger_csv, namelist_path=chem_namelist,
                            case_name="synthtest", hardware_id=HARDWARE_ID_TRAIN, top_k=50)
    assert r["all_ranked"] and all(e["omp_threads"] == 1 for e in r["all_ranked"])
    f = recommend.recommend_full_space(
        ledger_path=full_space_ledger_csv, namelist_path=chem_namelist, case_name="synthtest",
        hardware_id=FULL_SPACE_HARDWARE, nodes=4, top_k=100000,
        omp_options=FULL_SPACE_OMP_OPTIONS, tile_options=FULL_SPACE_TILE_OPTIONS)
    assert f["shortlist"] and all(e["omp_threads"] == 1 for e in f["shortlist"])


def test_chem_j_patch_guard_in_recommend_paths(ledger_csv, full_space_ledger_csv, chem_namelist):
    import wtune_extensions as we
    r = recommend.recommend(ledger_path=ledger_csv, namelist_path=chem_namelist,
                            case_name="synthtest", hardware_id=HARDWARE_ID_TRAIN, top_k=50)
    e_sn = we.chem_e_sn_for_namelist(chem_namelist)
    assert e_sn
    f = recommend.recommend_full_space(
        ledger_path=full_space_ledger_csv, namelist_path=chem_namelist, case_name="synthtest",
        hardware_id=FULL_SPACE_HARDWARE, nodes=4, top_k=100000,
        omp_options=FULL_SPACE_OMP_OPTIONS, tile_options=FULL_SPACE_TILE_OPTIONS)
    assert any("CHEM_MIN_J_PATCH" in w for w in f["warnings"])
    assert all(we.chem_min_j_patch(e_sn, e["nproc_y"]) >= we.CHEM_MIN_J_PATCH for e in f["shortlist"])
