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
                    fidelity_nd_max=None, probe=None):
    parts = ["mode=physics", "template=%s" % template, "family=%s" % family,
             "edits=%s" % edits_str, "decomp=%s" % decomp,
             "sim_minutes=%d" % sim_minutes, "hist_minutes=%d" % hist_minutes]
    if fidelity_nd_max is not None:
        parts.append("fidelity_nd_max=%.4f" % fidelity_nd_max)
    parts.append("probe=%s" % (",".join("%s:%s" % kv for kv in (probe or {}).items()) or "none"))
    return " ".join(parts)


def _phys_row(run_id, template, family, edits_dict, physics_fields, decomp, rep, s,
              fidelity_label=None, nd=None, probe=None, hardware_id=PHYS_HW, case="phystrain"):
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
                              decomp, fidelity_nd_max=(nd if rep == 1 else None), probe=probe),
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
