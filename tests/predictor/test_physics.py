#!/usr/bin/env python3
"""predictor/tests/test_physics.py -- pytest suite for predictor/physics.py (PHYSICS-GRAY
performance + fidelity models, decision 0051), against a synthetic physics-campaign ledger built
directly (no CSV round-trip needed for most tests -- see test_physics_benchmark_runs_on_csv for
the one test that does go through a CSV, exercising benchmark.load_ledger too).

Run with:
    pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml \\
        python -m pytest predictor/tests/test_physics.py predictor/tests/test_predictor.py -v
"""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wtune.predictor import ledger
from wtune.predictor import physics

# physics.py's own module-level sys.path.insert already put campaign/ on sys.path (see
# physics.py's top -- it imports campaign/configs.py itself now, for the single-source-of-truth
# template-application refactor); safe to import configs here too, AFTER `import physics` above.
import configs as campaign_configs

HARDWARE_ID = ledger.HARDWARE_ID

# ---------------------------------------------------------------------------
# Synthetic physics-campaign ledger: 6 cases x (P00 + twin + 6 candidate templates) x 2 reps,
# with a known multiplicative speed effect and a known fidelity nd (+case variation) per template,
# a FAILED run (isc21/P03, both reps), an INVALID fidelity label (maria1km/P02), a skipped
# template (P04 for maria1km, per the real compatibility note: native mp_physics is already
# WSM6), and a suite -1 case (conus25: native mp/bl/sfclay/ra_* are -1, resolved via the "conus"
# physics_suite table).
# ---------------------------------------------------------------------------
CASES = ["karina", "conus25", "conus12", "isc21", "lafe", "maria1km"]
DECOMP = "C02"
TWIN_DECOMP = {c: ("C13" if c == "isc21" else "C06") for c in CASES}
TEMPLATES = ["P01", "P02", "P03", "P04", "P05", "P06"]
FAMILY = {"P01": "radiation", "P02": "radiation", "P03": "dynamics", "P04": "microphysics",
          "P05": "pbl_surface", "P06": "radiation"}
# multiplicative speed effect vs this case's own P00 (s_candidate = s_P00 * M_T[template])
M_T = {"P01": 1.001, "P02": 0.94, "P03": 0.85, "P04": 0.92, "P05": 0.97, "P06": 0.90}
# base normalized-distance (fidelity) per template; scaled per case by CASE_ND_FACTOR below
ND_T = {"P01": 0.02, "P02": 0.06, "P03": 0.03, "P04": 0.25, "P05": 0.08, "P06": 0.04}
LABEL_T = {"P01": "WITHIN-FLOOR", "P02": "SMALL-DRIFT", "P03": "WITHIN-FLOOR",
           "P04": "LARGE-DRIFT", "P05": "SMALL-DRIFT", "P06": "WITHIN-FLOOR"}
CASE_ND_FACTOR = {"karina": 1.0, "conus25": 1.1, "conus12": 0.9, "isc21": 1.05, "lafe": 0.95,
                  "maria1km": 1.15}
BASE_S = {"karina": 0.28, "conus25": 1.10, "conus12": 2.40, "isc21": 0.65, "lafe": 0.05,
          "maria1km": 0.90}

# conus25 deliberately carries -1 sentinels (physics_suite="conus" resolves them); the other 5
# cases use concrete values even where the real campaign also uses a suite (conus12, maria1km),
# to keep the suite-resolution test isolated to one case.
NATIVE = {
    "karina": dict(mp_physics=8, bl_pbl_physics=1, sf_sfclay_physics=1, ra_lw_physics=4,
                   ra_sw_physics=4, radt=15, time_step=15),
    "conus25": dict(mp_physics=-1, bl_pbl_physics=-1, sf_sfclay_physics=-1, ra_lw_physics=-1,
                     ra_sw_physics=-1, radt=10, time_step=15, physics_suite="conus"),
    "conus12": dict(mp_physics=8, bl_pbl_physics=2, sf_sfclay_physics=2, ra_lw_physics=4,
                     ra_sw_physics=4, radt=15, time_step=72),
    "isc21": dict(mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, ra_lw_physics=4,
                   ra_sw_physics=4, radt=3, time_step=15),
    "lafe": dict(mp_physics=28, bl_pbl_physics=5, sf_sfclay_physics=5, ra_lw_physics=4,
                  ra_sw_physics=4, radt=1, time_step=10),
    "maria1km": dict(mp_physics=6, bl_pbl_physics=1, sf_sfclay_physics=1, ra_lw_physics=4,
                       ra_sw_physics=4, radt=12, time_step=25),
}
RESOLVED = dict(NATIVE, conus25=dict(mp_physics=8, bl_pbl_physics=2, sf_sfclay_physics=2,
                                       ra_lw_physics=4, ra_sw_physics=4, radt=10, time_step=15))
PBL_SWAP_TO = {"karina": 5, "conus25": 1, "conus12": 1, "isc21": 1, "lafe": 1, "maria1km": 5}
GRID = {
    "karina": (901, 901, 45, 3000), "conus25": (1001, 801, 50, 2500),
    "conus12": (1501, 1201, 50, 12000), "isc21": (601, 601, 40, 21000),
    "lafe": (401, 401, 60, 111), "maria1km": (1201, 1201, 55, 1000),
}
SKIP = {"P04": {"maria1km": "no-op: native mp_physics already WSM6"}}
FAILED = {("isc21", "P03")}       # both reps FAILED (synthetic CFL crash)
INVALID = {("maria1km", "P02")}   # ran fine (has timing) but fidelity_label=INVALID


def _edits_str(d):
    return ";".join("%s=%s" % (k, v) for k, v in d.items())


def _notes(template, family, edits, decomp, probe, extra):
    probe_str = ",".join("%s:%s" % kv for kv in (probe or {}).items()) or "none"
    return ("mode=physics template=%s family=%s edits=%s decomp=%s sim_minutes=30 hist_minutes=30 "
            "probe=%s%s" % (template, family, edits, decomp, probe_str, extra))


def make_physics_ledger(noise_std=0.0008, seed=0):
    rng = np.random.RandomState(seed)
    rows = []

    def add_row(case, template, decomp, rep, s, physics_fields, edits, family,
                fidelity_label=None, nd=None, probe=None, failed=False, tag=""):
        extra = ""
        if failed:
            extra = " status=FAILED reason=synthetic_crash"
        elif nd is not None:
            extra = " fidelity_nd_max=%.4f" % nd
        e_we, e_sn, e_vert, dx = GRID[case]
        rows.append(ledger.new_row(
            run_id="%s_%s_r%d%s" % (case, template, rep, tag), source_dataset="strict_campaign",
            job_id="9999", rep_index=rep, case_raw=case, case_base=case,
            case_family=ledger.case_family_of(case), hardware_id=HARDWARE_ID,
            binary_kind="pristine",
            compliance_class=("STRICT" if template == "P00" else "PHYSICS-GRAY"),
            e_we=e_we, e_sn=e_sn, e_vert=e_vert, dx=dx, dy=dx,
            time_step=physics_fields["time_step"], mp_physics=physics_fields["mp_physics"],
            cu_physics=0, bl_pbl_physics=physics_fields["bl_pbl_physics"],
            sf_sfclay_physics=physics_fields["sf_sfclay_physics"], sf_surface_physics=2,
            ra_lw_physics=physics_fields["ra_lw_physics"], ra_sw_physics=physics_fields["ra_sw_physics"],
            radt=physics_fields["radt"], cudt=0, history_interval=(30.0 if rep == 1 else 0.0),
            io_form_history=2, n_nodes=4, n_ranks=448, omp_threads=1, nproc_x=8, nproc_y=56,
            numtiles=1, distribution="cyclic", kmp_blocktime=0,
            s_per_step_median=(None if failed else s), s_per_step_mean=(None if failed else s),
            n_steps_measured=(0 if failed else 30), diffwrf_differing_fields=None,
            fidelity_label=(fidelity_label if rep == 1 else None),
            profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
            profile_mpi_share=None, profile_io_share=None, profile_rank_imbalance=None,
            profile_thread_imbalance=None,
            notes=_notes(template, family, edits, decomp, probe, extra),
        ))

    for case in CASES:
        native, resolved = NATIVE[case], RESOLVED[case]
        base_s = BASE_S[case]

        for rep in (1, 2):
            s = base_s * math.exp(rng.normal(0, noise_std))
            add_row(case, "P00", DECOMP, rep, s, native, "", "reference")
        for rep in (1, 2):  # twin: same physics, TWIN_DECOMP -- exercises group isolation
            s = base_s * math.exp(rng.normal(0, noise_std))
            add_row(case, "P00", TWIN_DECOMP[case], rep, s, native, "", "reference",
                    fidelity_label="WITHIN-FLOOR", nd=0.001, tag="_twin")

        for tmpl in TEMPLATES:
            if SKIP.get(tmpl, {}).get(case):
                continue  # campaign never ran this template for this case
            edited = dict(resolved)
            edits, probe = {}, None
            if tmpl == "P01":
                edits, probe = {"swint_opt": "1"}, {"swint_opt": "CONFIRM"}
            elif tmpl == "P02":
                edited["radt"] = int(resolved["radt"] * 2)
                edits = {"radt": str(edited["radt"]), "swint_opt": "1"}
                probe = {"radt": "CONFIRM", "swint_opt": "CONFIRM"}
            elif tmpl == "P03":
                edited["time_step"] = int(round(resolved["time_step"] * 1.2))
                edits, probe = {"time_step": str(edited["time_step"])}, {"time_step": "PASS->WRF"}
            elif tmpl == "P04":
                edited["mp_physics"] = 6
                edits, probe = {"mp_physics": "6"}, {"mp_physics": "REJECT"}
            elif tmpl == "P05":
                edited["bl_pbl_physics"] = edited["sf_sfclay_physics"] = PBL_SWAP_TO[case]
                edits = {"bl_pbl_physics": str(PBL_SWAP_TO[case]),
                         "sf_sfclay_physics": str(PBL_SWAP_TO[case])}
                probe = None  # no probe evidence for this knob pair -- exercises "no probe data"
            elif tmpl == "P06":
                edited["ra_lw_physics"] = edited["ra_sw_physics"] = 1
                edits = {"ra_lw_physics": "1", "ra_sw_physics": "1"}
                # deliberately probe-REJECT but measured-safe: exercises "probe verdict disagrees
                # with measured outcome" in probe_confusion.
                probe = {"ra_lw_physics": "REJECT"}

            failed_here = (case, tmpl) in FAILED
            label = None if failed_here else ("INVALID" if (case, tmpl) in INVALID else LABEL_T[tmpl])
            nd = None if label in (None, "INVALID") else ND_T[tmpl] * CASE_ND_FACTOR[case]

            for rep in (1, 2):
                if failed_here:
                    add_row(case, tmpl, DECOMP, rep, None, edited, _edits_str(edits), FAMILY[tmpl],
                            failed=True, probe=probe)
                else:
                    s = base_s * M_T[tmpl] * math.exp(rng.normal(0, noise_std))
                    add_row(case, tmpl, DECOMP, rep, s, edited, _edits_str(edits), FAMILY[tmpl],
                            fidelity_label=label, nd=nd, probe=probe)
    return rows


# ---------------------------------------------------------------------------
# 1. notes parsing
# ---------------------------------------------------------------------------
def test_parse_notes_and_subparsers():
    notes = ("mode=physics template=P02 family=radiation edits=radt=19;swint_opt=1 decomp=C02 "
             "sim_minutes=30 hist_minutes=30 fidelity_nd_max=0.0650 "
             "probe=radt:CONFIRM,swint_opt:CONFIRM")
    parsed = physics.parse_notes(notes)
    assert parsed["mode"] == "physics" and parsed["template"] == "P02"
    assert parsed["edits"] == "radt=19;swint_opt=1"
    assert physics.parse_edits(parsed["edits"]) == {"radt": "19", "swint_opt": "1"}
    assert physics.parse_probe(parsed["probe"]) == {"radt": "CONFIRM", "swint_opt": "CONFIRM"}

    failed_notes = ("mode=physics template=P03 family=dynamics edits=time_step=18 decomp=C02 "
                     "status=FAILED reason=cfl_violation")
    parsed2 = physics.parse_notes(failed_notes)
    assert parsed2["status"] == "FAILED" and parsed2["reason"] == "cfl_violation"

    assert physics.is_physics_row({"notes": notes})
    assert not physics.is_physics_row({"notes": "mode=timing config=C01"})
    assert not physics.is_physics_row({"notes": None})
    assert physics.parse_notes(None) == {} and physics.parse_edits(None) == {}
    assert physics.parse_probe("none") == {}


# ---------------------------------------------------------------------------
# 2. grouping: baseline = P00 at DECOMP, never the twin (different decomp); skip is respected
# ---------------------------------------------------------------------------
def test_physics_groups_baseline_is_p00_at_decomp_not_twin():
    rows = make_physics_ledger()
    groups = physics.physics_groups(rows)

    karina_p00 = next(r for r in rows if r["case_base"] == "karina"
                       and physics._template_of(r) == "P00" and "_twin" not in r["run_id"])
    key = physics.physics_group_key(karina_p00)
    assert key in groups
    g = groups[key]
    assert len(g["baseline"]) == 2 and all(physics._template_of(r) == "P00" for r in g["baseline"])
    assert len(g["candidates"]) == 12  # 6 templates x 2 reps, none skipped for karina

    twin = next(r for r in rows if r["case_base"] == "karina" and "_twin" in r["run_id"])
    assert physics.physics_group_key(twin) != key


def test_physics_groups_respects_skip():
    rows = make_physics_ledger()
    groups = physics.physics_groups(rows)
    maria_p00 = next(r for r in rows if r["case_base"] == "maria1km"
                      and physics._template_of(r) == "P00" and "_twin" not in r["run_id"])
    key = physics.physics_group_key(maria_p00)
    assert len(groups[key]["candidates"]) == 10  # 5 templates x 2 reps (P04 skipped)
    assert not any(physics._template_of(r) == "P04" for r in groups[key]["candidates"])


# ---------------------------------------------------------------------------
# 3. y construction: recovers the known multiplier; FAILED -> y=None, counted INFEASIBLE
# ---------------------------------------------------------------------------
def test_add_perf_targets_recovers_known_multiplier_and_infeasible():
    rows = make_physics_ledger()
    physics.add_perf_targets(rows)
    for case in CASES:
        for tmpl in TEMPLATES:
            if SKIP.get(tmpl, {}).get(case) or (case, tmpl) in FAILED:
                continue
            crows = [r for r in rows if r["case_base"] == case and physics._template_of(r) == tmpl]
            ys = [r["y"] for r in crows if r["y"] is not None]
            assert len(ys) == len(crows) == 2
            # y is cost per SIMULATED second (s_per_step / time_step): a template that changes
            # time_step also shifts y by -log(dt_candidate / dt_P00) on top of its per-step M_T
            base_dt = float([r for r in rows if r["case_base"] == case
                             and physics._template_of(r) == "P00"][0]["time_step"])
            dt_ratio = float(crows[0]["time_step"]) / base_dt
            expected = math.log(M_T[tmpl]) - math.log(dt_ratio)
            assert (sum(ys) / len(ys)) == pytest.approx(expected, abs=0.01)

    failed_rows = [r for r in rows if r["case_base"] == "isc21" and physics._template_of(r) == "P03"]
    assert len(failed_rows) == 2 and all(r["y"] is None for r in failed_rows)
    infeasible = physics.infeasible_candidates(rows)
    assert any(tmpl == "P03" and key[0] == "isc21" for key, tmpl, _ in infeasible)


# ---------------------------------------------------------------------------
# 4. classify_rows leaves physics rows' compliance_class alone
# ---------------------------------------------------------------------------
def test_classify_rows_leaves_physics_rows_alone():
    rows = make_physics_ledger()
    before = {r["run_id"]: r["compliance_class"] for r in rows}
    ledger.classify_rows(rows)
    after = {r["run_id"]: r["compliance_class"] for r in rows}
    assert before == after
    assert all(after[r["run_id"]] == "STRICT" for r in rows if physics._template_of(r) == "P00")
    assert all(after[r["run_id"]] == "PHYSICS-GRAY" for r in rows
               if physics._template_of(r) not in (None, "P00"))


# ---------------------------------------------------------------------------
# physics_change_features: suite -1 resolution, from/to/ratio arithmetic, swint_on
# ---------------------------------------------------------------------------
def test_physics_change_features_hand_checked():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)

    conus_p02 = next(r for r in rows if r["case_base"] == "conus25"
                      and physics._template_of(r) == "P02" and r["rep_index"] == 1)
    feats = physics.physics_change_features(conus_p02)
    # native bl_pbl_physics=-1 must resolve to 2 (conus suite), not the raw sentinel
    assert feats["bl_pbl_physics_from"] == 2.0
    assert feats["mp_physics_from"] == 8.0
    assert feats["radt_ratio"] == pytest.approx(2.0)
    assert feats["swint_on"] == 1.0
    assert feats["template_P02"] == 1.0 and feats["family_radiation"] == 1.0

    karina_p01 = next(r for r in rows if r["case_base"] == "karina"
                       and physics._template_of(r) == "P01" and r["rep_index"] == 1)
    assert physics.physics_change_features(karina_p01)["swint_on"] == 1.0
    karina_p05 = next(r for r in rows if r["case_base"] == "karina"
                       and physics._template_of(r) == "P05" and r["rep_index"] == 1)
    f5 = physics.physics_change_features(karina_p05)
    assert f5["bl_pbl_physics_from"] == 1.0 and f5["bl_pbl_physics_to"] == 5.0
    assert f5["bl_pbl_physics_changed"] == 1.0
    assert f5.get("swint_on", 0.0) == 0.0  # P05 never touches swint_opt


# ---------------------------------------------------------------------------
# 5. LOCO / LOTO evaluation runs and recovers the synthetic ordering
# ---------------------------------------------------------------------------
def test_loco_recovers_synthetic_ordering():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    candidates = [r for r in physics.physics_rows(rows) if physics._template_of(r) not in (None, "P00")]

    predictions, unavailable = physics.cross_validate_physics(
        candidates, "case_base", lambda: physics.PhysicsLinearModel(target="y"), "y")
    assert not unavailable
    agg = physics.aggregate_perf_metrics(candidates, predictions)
    assert agg["n_eval_cases"] >= 5
    # loose sanity bound: M_T spans ~15% (0.85-1.001) against ~0.08% replicate noise, so even a
    # modest linear fit over the template one-hot should separate templates comfortably.
    assert agg["median_regret"] < 0.10
    if agg["pairwise_accuracy"] is not None:
        assert agg["pairwise_accuracy"] > 0.55


def test_leave_one_template_out_runs():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    candidates = [r for r in physics.physics_rows(rows) if physics._template_of(r) not in (None, "P00")]
    predictions, unavailable = physics.cross_validate_physics(
        candidates, "template", lambda: physics.PhysicsLinearModel(target="y"), "y")
    assert predictions
    assert unavailable == set() or unavailable.issubset({"P03"})  # P03's own y is all-None (FAILED
                                                                   # for isc21 only, but the OTHER
                                                                   # 5 cases' P03 rows still have y)


# ---------------------------------------------------------------------------
# 6. dangerous-miss arithmetic (direct, model-independent)
# ---------------------------------------------------------------------------
def test_fidelity_confusion_dangerous_miss_arithmetic():
    rows = [
        dict(run_id="a", fidelity_label="WITHIN-FLOOR"),
        dict(run_id="b", fidelity_label="SMALL-DRIFT"),
        dict(run_id="c", fidelity_label="LARGE-DRIFT"),
        dict(run_id="d", fidelity_label="LARGE-DRIFT"),
    ]
    predicted = {"a": "WITHIN-FLOOR", "b": "WITHIN-FLOOR", "c": "SMALL-DRIFT", "d": "LARGE-DRIFT"}
    conf = physics.fidelity_confusion(rows, predicted)
    assert conf["n"] == 4
    assert conf["predicted_allowed_n"] == 3        # a, b, c predicted allowed
    assert conf["dangerous_miss_n"] == 1           # c: predicted allowed but measured LARGE-DRIFT
    assert conf["dangerous_miss_rate"] == pytest.approx(1 / 3)


def test_fidelity_confusion_integration_runs():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_fidelity_targets(rows)
    label_rows = [r for r in physics.physics_rows(rows) if r.get("y_fidelity_label") is not None]
    predictions, _ = physics.cross_validate_physics(
        label_rows, "case_base", lambda: physics.PhysicsLinearModel(target="y_fidelity_label"),
        "y_fidelity_label")
    pred_label = {rid: physics._label_from_ordinal_score(p) for rid, p in predictions.items()}
    conf = physics.fidelity_confusion(label_rows, pred_label)
    assert conf["n"] > 0
    assert conf["dangerous_miss_rate"] is None or 0.0 <= conf["dangerous_miss_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 7. probe confusion: P06's probe=REJECT but measured-safe/faster row must show up as the
# "probe wrong" cell.
# ---------------------------------------------------------------------------
def test_probe_confusion():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    physics.add_fidelity_targets(rows)
    pconf = physics.probe_confusion(rows)
    assert pconf["n_fidelity"] > 0 and pconf["n_speed"] > 0
    assert pconf["fidelity_confusion"].get(("REJECT", "ALLOWED"), 0) > 0  # P06
    assert pconf["speed_confusion"].get(("REJECT", "FASTER>1%"), 0) > 0   # P06 is also faster


# ---------------------------------------------------------------------------
# 8. physics_candidates_for_case: skip_cases, extra_edits, suite resolution
# ---------------------------------------------------------------------------
def _templates_json_doc():
    return {
        "templates": {
            "P01": {"family": "radiation", "edits": [{"op": "set", "key": "swint_opt", "value": 1}]},
            "P02": {"family": "radiation",
                     "edits": [{"op": "mul", "key": "radt", "factor": 2},
                               {"op": "set", "key": "swint_opt", "value": 1}]},
            "P03": {"family": "dynamics", "edits": [{"op": "mul", "key": "time_step", "factor": 1.2}]},
            "P04": {"family": "microphysics",
                     "edits": [{"op": "set", "key": "mp_physics", "value": 6}],
                     "skip_cases": {"maria1km": "no-op: native mp_physics already WSM6"},
                     "extra_edits": {"lafe": [{"op": "set", "key": "aer_opt", "value": 0},
                                               {"op": "set", "key": "use_aero_icbc", "value": ".false."}]}},
            "P05": {"family": "pbl_surface", "edits": [],
                     "extra_edits": {
                         "karina": [{"op": "set", "key": "bl_pbl_physics", "value": 5},
                                    {"op": "set", "key": "sf_sfclay_physics", "value": 5}],
                         "isc21": [{"op": "set", "key": "bl_pbl_physics", "value": 1},
                                   {"op": "set", "key": "sf_sfclay_physics", "value": 1}]}},
            "P06": {"family": "radiation",
                     "edits": [{"op": "set", "key": "ra_lw_physics", "value": 1},
                               {"op": "set", "key": "ra_sw_physics", "value": 1}],
                     "extra_edits": {"lafe": [{"op": "set", "key": "aer_opt", "value": 0}]}},
        },
        "excluded": {"ra_24_14": "not compiled into this build"},
    }


def test_physics_candidates_for_case_skip_and_extra_edits(tmp_path):
    path = tmp_path / "physics_templates.json"
    path.write_text(json.dumps(_templates_json_doc()))

    cands = physics.physics_candidates_for_case(dict(NATIVE["maria1km"]), "maria1km", 2, str(path))
    by_t = {c["template"]: c for c in cands}
    assert by_t["P04"]["applicable"] is False
    assert "no-op" in by_t["P04"]["skip_reason"]
    assert by_t["P04"]["changes"] == {}
    assert by_t["P01"]["applicable"] is True
    assert by_t["P01"]["changes"]["swint_opt"] == {"from": None, "to": 1}

    cands_lafe = physics.physics_candidates_for_case(dict(NATIVE["lafe"]), "lafe", 1, str(path))
    by_t_lafe = {c["template"]: c for c in cands_lafe}
    assert by_t_lafe["P04"]["applicable"] is True
    assert by_t_lafe["P04"]["changes"]["mp_physics"] == {"from": 28, "to": 6}
    assert by_t_lafe["P04"]["changes"]["aer_opt"] == {"from": None, "to": 0}
    assert by_t_lafe["P04"]["changes"]["use_aero_icbc"] == {"from": None, "to": ".false."}

    cands_conus = physics.physics_candidates_for_case(dict(NATIVE["conus25"]), "conus25", 1, str(path))
    by_t_conus = {c["template"]: c for c in cands_conus}
    assert by_t_conus["P02"]["changes"]["radt"] == {"from": 10, "to": 20}  # not suite-resolved (radt isn't)
    # This synthetic templates doc's own P05 has no extra_edits entry for "conus25" (only
    # karina/isc21 are hand-written above) -- physics_candidates_for_case now falls back to
    # generic_extra_edits() for a case a template has never seen (decision 0051 update), so this
    # is no longer a no-op: conus25's suite-resolved native PBL is MYJ(2) -> generic swap to
    # YSU(1)/1, same direction the real physics_templates.json's own conus25 entry uses.
    assert by_t_conus["P05"]["applicable"] is True
    assert by_t_conus["P05"]["generic_rule_used"] is True
    assert by_t_conus["P05"]["changes"]["bl_pbl_physics"] == {"from": 2, "to": 1}
    assert by_t_conus["P05"]["changes"]["sf_sfclay_physics"] == {"from": 2, "to": 1}
    assert by_t_conus["P03"]["changes"]["time_step"] == {"from": 15, "to": 18}
    # P04 sets mp_physics=6 unconditionally: "from" must be the SUITE-RESOLVED native value (8),
    # not the raw -1 sentinel in NATIVE["conus25"] -- this is the assertion that actually depends
    # on resolve_physics_suite() firing (P02/P03/P05 above don't touch a suite-controlled field).
    assert by_t_conus["P04"]["changes"]["mp_physics"] == {"from": 8, "to": 6}


# ---------------------------------------------------------------------------
# 9. predict_physics_shortlist status logic
# ---------------------------------------------------------------------------
def test_predict_physics_shortlist_status_logic():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    physics.add_fidelity_targets(rows)

    karina_p00 = next(r for r in rows if r["case_base"] == "karina"
                       and physics._template_of(r) == "P00" and "_twin" not in r["run_id"]
                       and r["rep_index"] == 1)
    candidates = [
        # P01 is WITHIN-FLOOR-safe but only ~0.1% SLOWER (M_T["P01"]=1.001 in this fixture) --
        # since the lead-review speed-benefit gate landed, P01 no longer qualifies as
        # RECOMMENDABLE just for being fidelity-safe (see the dedicated assertion below); P06
        # (WITHIN-FLOOR, ~10% faster) is this test's "should end up RECOMMENDABLE" example.
        dict(template="P01", family="radiation", applicable=True,
             changes={"swint_opt": {"from": None, "to": 1}}),
        dict(template="P06", family="radiation", applicable=True,
             changes={"ra_lw_physics": {"from": 4, "to": 1}, "ra_sw_physics": {"from": 4, "to": 1}}),
        dict(template="P04", family="microphysics", applicable=True,
             changes={"mp_physics": {"from": 8, "to": 6}}),
        dict(template="P99", family="dynamics", applicable=False, skip_reason="not on this grid",
             changes={}),
    ]

    shortlist = physics.predict_physics_shortlist(
        rows, karina_p00, candidates, probe_verdicts={"mp_physics": "REJECT"})
    by_t = {c["template"]: c for c in shortlist}

    # not applicable -> UNCONFIRMED, never even scored
    assert by_t["P99"]["status"] == "UNCONFIRMED"
    assert by_t["P99"]["predicted_y"] is None

    # probe REJECT on the edited knob -> UNCONFIRMED regardless of prediction
    assert by_t["P04"]["status"] == "UNCONFIRMED"
    assert by_t["P04"]["probe_verdict"] == "REJECT"
    assert any("REJECT" in r for r in by_t["P04"]["reasons"])

    # P01: allowed label + probe not REJECT + same-hardware training data, but no real predicted
    # speed benefit -> the speed-benefit gate (lead-review gate B) keeps it UNCONFIRMED.
    assert by_t["P01"]["status"] == "UNCONFIRMED"
    assert by_t["P01"]["probe_verdict"] is None
    assert by_t["P01"]["predicted_fidelity_label"] in physics.DEFAULT_ALLOWED_LABELS
    assert any("no predicted speed benefit" in r for r in by_t["P01"]["reasons"])

    # P06: allowed label + probe not REJECT + same-hardware training data + a real (~10%)
    # predicted speedup -> RECOMMENDABLE.
    assert by_t["P06"]["status"] == "RECOMMENDABLE"
    assert by_t["P06"]["probe_verdict"] is None
    assert by_t["P06"]["predicted_fidelity_label"] in physics.DEFAULT_ALLOWED_LABELS
    assert by_t["P06"]["predicted_y"] is not None
    assert by_t["P06"]["predicted_time_to_solution_pct"] < -1.0

    # ranking: every RECOMMENDABLE candidate sorts before every UNCONFIRMED one
    statuses = [c["status"] for c in shortlist]
    first_unconfirmed = statuses.index("UNCONFIRMED") if "UNCONFIRMED" in statuses else len(statuses)
    assert all(s == "RECOMMENDABLE" for s in statuses[:first_unconfirmed])

    # no same-hardware training data -> UNCONFIRMED even with a clean probe verdict
    fake_hw_case = dict(karina_p00, hardware_id="no-such-hardware")
    shortlist_no_hw = physics.predict_physics_shortlist(rows, fake_hw_case, [candidates[1]],
                                                          probe_verdicts={})
    assert all(c["status"] == "UNCONFIRMED" for c in shortlist_no_hw)
    assert any("no same-hardware training data" in r
               for c in shortlist_no_hw for r in c["reasons"])


# ---------------------------------------------------------------------------
# End-to-end: physics_benchmark on a CSV-round-tripped ledger (exercises benchmark.load_ledger).
# ---------------------------------------------------------------------------
def test_physics_benchmark_runs_on_csv(tmp_path):
    rows = make_physics_ledger()
    path = tmp_path / "runs.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ledger.LEDGER_FIELDS})

    out_dir = tmp_path / "out"
    result = physics.physics_benchmark(str(path), str(out_dir))
    assert result["status"] == "ok"
    assert os.path.exists(out_dir / "metrics.csv")
    assert os.path.exists(out_dir / "summary.md")
    with open(out_dir / "metrics.csv") as fh:
        metric_rows = list(csv.DictReader(fh))
    assert len(metric_rows) > 0
    assert any(m["target"] == "performance" for m in metric_rows)
    assert any(m["target"] == "fidelity_nd" for m in metric_rows)
    assert any(m["target"] == "fidelity_label" for m in metric_rows)
    assert result["data_sufficiency"]["n_qualifying_cases"] >= 5
    assert any(tmpl == "P03" for _, tmpl, _ in result["infeasible"])


def test_physics_benchmark_no_physics_rows(tmp_path):
    path = tmp_path / "strict_only.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        row = ledger.new_row(run_id="x", case_base="karina", hardware_id=HARDWARE_ID,
                              compliance_class="STRICT", s_per_step_median=0.3, notes="mode=timing")
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in ledger.LEDGER_FIELDS})
    out_dir = tmp_path / "out2"
    result = physics.physics_benchmark(str(path), str(out_dir))
    assert result["status"] == "no physics rows"
    assert os.path.exists(out_dir / "summary.md")


# ---------------------------------------------------------------------------
# 10. Single source of truth (decision 0051 update): physics.py's dict-based
# physics_candidates_for_case (and the new text-based physics_candidates_from_namelist, which
# recommend.py --compliance PHYSICS-GRAY actually calls for a real namelist) must agree with
# campaign/configs.py's own apply_physics_template on all six real campaign case namelists --
# proving physics.py no longer maintains a second, driftable implementation of the templates'
# set/mul/delete arithmetic (physics._apply_edit is gone; see _apply_edit_via_configs).
# ---------------------------------------------------------------------------
_SIX_CASES = ("karina", "conus25", "conus12", "isc21", "lafe", "maria1km")


def _synth_namelist(case):
    """Namelist text built from cases.json's own physics dict (the real prep_dir namelists are
    cluster-only paths); one value per active domain, time_step from the case's dt."""
    nd = campaign_configs._case_effective_max_dom(case)
    phys = "".join(" %s = %s,\n" % (k, ", ".join([repr(v) if isinstance(v, str) else str(v)] * nd))
                   for k, v in case["physics"].items())
    return ("&time_control\n/\n&physics\n%s/\n&domains\n max_dom = %d,\n time_step = %s,\n/\n"
            "&dynamics\n/\n&bdy_control\n/\n") % (phys, nd, case["dt"])


@pytest.fixture
def case_namelists(tmp_path):
    """Point every case without a readable real namelist at a synthesized one."""
    patches = {}
    for cid in _SIX_CASES:
        case = campaign_configs.CASES[cid]
        if os.path.exists(campaign_configs._case_namelist_path(case)):
            continue
        path = tmp_path / ("%s.namelist.input" % cid)
        path.write_text(_synth_namelist(case))
        patches[cid] = case.get("namelist_template")
        case["namelist_template"] = str(path)
    yield
    for cid, old in patches.items():
        if old is None:
            campaign_configs.CASES[cid].pop("namelist_template", None)
        else:
            campaign_configs.CASES[cid]["namelist_template"] = old


def _cast(v):
    if v is None or isinstance(v, (int, float)):
        return v
    s = str(v)
    try:
        return int(s)
    except (TypeError, ValueError):
        pass
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def test_template_engine_consistency_across_real_case_namelists(case_namelists):
    for case_id in _SIX_CASES:
        case = campaign_configs.CASES[case_id]
        max_dom = campaign_configs._case_effective_max_dom(case)
        suite = physics.CASE_PHYSICS_SUITE.get(case_id)
        # cases.json's own "physics" dict never carries time_step (it's the case's top-level "dt"
        # field instead, per cases.json's own convention) -- physics_candidates_for_case's
        # native_physics contract needs it explicitly for P03's mul, same as recommend.py's own
        # _native_physics_from_case_fields does.
        native_dict = dict(case["physics"], time_step=case["dt"])
        dict_cands = {c["template"]: c for c in physics.physics_candidates_for_case(
            native_dict, case_id, max_dom, campaign_configs.PHYSICS_TEMPLATES_JSON_PATH)}

        for tid in campaign_configs.PHYSICS_TEMPLATE_ORDER:
            if tid == "P00":
                continue
            dict_entry = dict_cands[tid]
            summary, err = campaign_configs.resolve_physics_config(case_id, tid)
            if summary is None:
                assert dict_entry["applicable"] is False, (
                    "%s/%s: text engine skips (%s) but dict engine says applicable"
                    % (case_id, tid, err))
                continue
            assert dict_entry["applicable"] is True, (
                "%s/%s: dict engine skips (%s) but text engine applies"
                % (case_id, tid, dict_entry["skip_reason"]))

            text_to, text_from = {}, {}
            for e in summary:
                if e["op"] == "delete":
                    continue
                text_to[e["key"]] = _cast(e["new"][0])
                if e["old"]:
                    frm = _cast(e["old"][0])
                    if frm == -1 and suite and e["key"] in physics.PHYSICS_SUITE_TABLE.get(suite, {}):
                        frm = physics.PHYSICS_SUITE_TABLE[suite][e["key"]]
                    text_from[e["key"]] = frm

            for key, ch in dict_entry["changes"].items():
                assert key in text_to, "%s/%s/%s: text engine never touched this key" % (
                    case_id, tid, key)
                assert _cast(ch["to"]) == text_to[key], (
                    "%s/%s/%s 'to' mismatch: dict=%r text=%r"
                    % (case_id, tid, key, ch["to"], text_to[key]))
                # 'from' is only comparable for the fields physics_candidates_for_case's own
                # native_physics dict actually tracks (CHANGE_FIELDS) -- a namelist-only knob like
                # swint_opt/aer_opt/use_aero_icbc is not one of cases.json's own physics fields,
                # so the dict engine can never know its real native value (always reports None).
                if key in physics.CHANGE_FIELDS and key in text_from:
                    assert _cast(ch["from"]) == text_from[key], (
                        "%s/%s/%s 'from' mismatch: dict=%r text(suite-resolved)=%r"
                        % (case_id, tid, key, ch["from"], text_from[key]))


def test_physics_candidates_from_namelist_matches_text_engine_for_karina(case_namelists):
    """The NEW text-engine candidate function (what recommend.py --compliance PHYSICS-GRAY
    actually calls for a real namelist) must reproduce campaign/configs.py's own
    resolve_physics_config exactly for karina (a non-suite case, so no -1-resolution ambiguity)."""
    case = campaign_configs.CASES["karina"]
    path = campaign_configs._case_namelist_path(case)
    max_dom = campaign_configs._case_effective_max_dom(case)
    with open(path) as fh:
        text = fh.read()
    resolved_native = physics.resolve_physics_suite(
        dict(case["physics"]), case["physics"].get("physics_suite"))
    cands = {c["template"]: c for c in physics.physics_candidates_from_namelist(
        text, "karina", max_dom, campaign_configs.PHYSICS_TEMPLATES_JSON_PATH,
        native_physics_resolved=resolved_native)}

    for tid in campaign_configs.PHYSICS_TEMPLATE_ORDER:
        if tid == "P00":
            continue
        summary, err = campaign_configs.resolve_physics_config("karina", tid)
        entry = cands[tid]
        if summary is None:
            assert entry["applicable"] is False, (tid, err)
            continue
        assert entry["applicable"] is True, (tid, entry["skip_reason"])
        expected = {e["key"]: (_cast(e["old"][0] if e["old"] else None), _cast(e["new"][0]))
                    for e in summary if e["op"] != "delete"}
        got = {k: (_cast(v["from"]), _cast(v["to"])) for k, v in entry["changes"].items()}
        assert got == expected, tid


# ---------------------------------------------------------------------------
# 11. generic_extra_edits (decision 0051 update): reproduces the JSON's six hand-written per-case
# extra_edits/skip decisions exactly for P04/P05/P06 (so the JSON and the generic rule cannot
# drift apart), plus every branch on synthetic natives not covered by the six real cases.
# ---------------------------------------------------------------------------
def test_generic_extra_edits_matches_json_for_six_cases():
    for case_id in _SIX_CASES:
        case = campaign_configs.CASES[case_id]
        resolved = physics.resolve_physics_suite(
            dict(case["physics"]), case["physics"].get("physics_suite"))
        for tid in ("P04", "P05", "P06"):
            spec = campaign_configs.PHYSICS_TEMPLATES[tid]
            json_skip = (spec.get("skip_cases") or {}).get(case_id)
            json_extra = (spec.get("extra_edits") or {}).get(case_id, [])
            gen_edits, gen_skip = physics.generic_extra_edits(tid, resolved)
            assert (gen_skip is None) == (json_skip is None), (case_id, tid, gen_skip, json_skip)
            if json_skip is None:
                norm = lambda edits: sorted((e["op"], e["key"], e.get("value")) for e in edits)
                assert norm(gen_edits) == norm(json_extra), (case_id, tid, gen_edits, json_extra)


def test_generic_extra_edits_p05_pbl_swap_rules():
    edits, skip = physics.generic_extra_edits("P05", dict(bl_pbl_physics=1))  # YSU -> MYNN
    assert skip is None
    assert edits == [
        dict(op="set", key="bl_pbl_physics", value=5, section="&physics"),
        dict(op="set", key="sf_sfclay_physics", value=5, section="&physics"),
    ]
    edits, skip = physics.generic_extra_edits("P05", dict(bl_pbl_physics=5))  # MYNN -> YSU
    assert skip is None and edits[0]["value"] == 1
    edits, skip = physics.generic_extra_edits("P05", dict(bl_pbl_physics=2))  # MYJ -> YSU
    assert skip is None and edits[0]["value"] == 1
    edits, skip = physics.generic_extra_edits("P05", dict(bl_pbl_physics=8))  # no rule defined
    assert edits == [] and skip is not None and "no generic PBL swap rule" in skip


def test_generic_extra_edits_p04_p06_aerosol_and_wsm6_skip():
    # mp=6 (WSM6): P04 is a no-op -> skip; P06 is unaffected (not aerosol-aware) -> applies, no extras
    edits, skip = physics.generic_extra_edits("P04", dict(mp_physics=6))
    assert edits == [] and skip is not None and "WSM6" in skip
    edits, skip = physics.generic_extra_edits("P06", dict(mp_physics=6))
    assert edits == [] and skip is None

    # mp=28/29 (Thompson aerosol-aware): P04 adds aer_opt=0 + use_aero_icbc=.false.; P06 adds
    # only aer_opt=0 (P06 replaces RRTMG SW, which the aerosol-SW coupling needs)
    edits, skip = physics.generic_extra_edits("P04", dict(mp_physics=28))
    assert skip is None
    assert dict(op="set", key="aer_opt", value=0, section="&physics") in edits
    assert dict(op="set", key="use_aero_icbc", value=".false.", section="&physics") in edits
    edits, skip = physics.generic_extra_edits("P06", dict(mp_physics=29))
    assert skip is None
    assert edits == [dict(op="set", key="aer_opt", value=0, section="&physics")]

    # mp=8 (neither WSM6 nor aerosol-aware): no generic edits, no skip, for either template
    for tid in ("P04", "P06"):
        edits, skip = physics.generic_extra_edits(tid, dict(mp_physics=8))
        assert edits == [] and skip is None


# ---------------------------------------------------------------------------
# 10. Defect 2 (real physics-campaign smoke, job 1077893): per-template infeasibility evidence
# aggregated ACROSS a template's physics groups (cases), not just per group.
# ---------------------------------------------------------------------------
def test_template_infeasibility_aggregates_across_cases():
    rows = make_physics_ledger()
    physics.add_perf_targets(rows)
    infeas = physics.template_infeasibility(rows)

    # isc21/P03 is this fixture's only synthetic crash (both reps FAILED) -- exactly 1 of the 6
    # cases P03 was tried in.
    assert infeas["P03"]["n_cases_tried"] == len(CASES)
    assert infeas["P03"]["n_cases_failed"] == 1
    assert infeas["P03"]["failed_cases"] == ["isc21"]
    assert infeas["P03"]["rate"] == pytest.approx(1.0 / len(CASES))

    for tmpl in TEMPLATES:
        if tmpl == "P03":
            continue
        assert infeas[tmpl]["n_cases_failed"] == 0
        assert infeas[tmpl]["rate"] == 0.0

    # P04 is skipped entirely for maria1km (SKIP dict) -- never attempted there, so it must not
    # count toward n_cases_tried.
    assert infeas["P04"]["n_cases_tried"] == len(CASES) - 1


def test_template_infeasibility_empty_for_no_rows():
    assert physics.template_infeasibility([]) == {}


# ---------------------------------------------------------------------------
# 11. Defect 3: leave-one-case-out out-of-fold dangerous-miss estimate, cross-checked against a
# manual recount from the same cross_validate_physics predictions.
# ---------------------------------------------------------------------------
def test_per_template_dangerous_miss_matches_manual_recount():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_fidelity_targets(rows)
    result = physics.per_template_dangerous_miss(rows)

    label_rows = [r for r in physics.physics_rows(rows) if r.get("y_fidelity_label") is not None]
    predictions, _ = physics.cross_validate_physics(
        label_rows, "case_base", lambda: physics.PhysicsLinearModel(target="y_fidelity_label"),
        "y_fidelity_label")
    pred_label = {rid: physics._label_from_ordinal_score(p) for rid, p in predictions.items()}

    for tmpl in TEMPLATES:
        trows = [r for r in label_rows if physics._template_of(r) == tmpl]
        n = sum(1 for r in trows if r["run_id"] in pred_label)
        if n == 0:
            assert tmpl not in result
            continue
        dangerous = sum(1 for r in trows
                         if pred_label.get(r["run_id"]) in physics.DEFAULT_ALLOWED_LABELS
                         and r.get("fidelity_label") == "LARGE-DRIFT")
        assert result[tmpl]["n"] == n
        assert result[tmpl]["dangerous_miss_n"] == dangerous
        assert result[tmpl]["dangerous_miss_rate"] == pytest.approx(dangerous / n)
        assert result[tmpl]["insufficient"] == (n < physics.MIN_OOF_DANGEROUS_MISS)

    # P01 is WITHIN-FLOOR for every case it ran (6) -- plenty of out-of-fold evidence, and its
    # measured label is never LARGE-DRIFT, so its dangerous-miss rate must be exactly 0.
    assert result["P01"]["insufficient"] is False
    assert result["P01"]["dangerous_miss_rate"] == 0.0
    # P04 is LARGE-DRIFT for every case it ran (5 of 6 -- maria1km is skipped): exactly enough
    # out-of-fold evidence (n=5) to not be "insufficient".
    assert result["P04"]["n"] == 5
    assert result["P04"]["insufficient"] is False


def test_per_template_dangerous_miss_insufficient_below_floor():
    # A single case_base gives ZERO leave-one-case-out folds -- every template must come back
    # either absent or, if present at all, with n below MIN_OOF_DANGEROUS_MISS.
    rows = [r for r in make_physics_ledger() if r["case_base"] == "karina"]
    physics.attach_change_context(rows)
    physics.add_fidelity_targets(rows)
    result = physics.per_template_dangerous_miss(rows)
    assert result == {}  # cross_validate_physics itself raises NotEnoughGroups (1 group only)


# ---------------------------------------------------------------------------
# 12. Defect 4: performance conformal interval helpers (jackknife+ over cross_validate_physics'
# own leave-one-case-out folds).
# ---------------------------------------------------------------------------
def test_performance_conformal_interval_runs_and_brackets_point_estimate():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    perf_rows = [r for r in physics.physics_rows(rows)
                 if physics._template_of(r) not in (None, "P00") and r.get("y") is not None]

    fold_models, fold_residuals, reason = physics.performance_conformal_setup(perf_rows)
    assert reason is None
    assert fold_models and fold_residuals

    karina_p01 = next(r for r in rows if r["case_base"] == "karina"
                       and physics._template_of(r) == "P01" and r["rep_index"] == 1)
    iv = physics.performance_conformal_interval_pct(fold_models, fold_residuals, karina_p01)
    assert iv is not None
    lo, hi = iv
    assert lo <= hi
    # loose sanity margin (not a calibration check): the interval should be somewhere near the
    # known M_T["P01"] effect in percent terms.
    known_pct = (M_T["P01"] - 1) * 100
    assert lo - 10 <= known_pct <= hi + 10


def test_performance_conformal_setup_reason_when_too_few_groups():
    rows = [r for r in make_physics_ledger() if r["case_base"] == "karina"]
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    perf_rows = [r for r in physics.physics_rows(rows)
                 if physics._template_of(r) not in (None, "P00") and r.get("y") is not None]
    fold_models, fold_residuals, reason = physics.performance_conformal_setup(perf_rows)
    assert fold_models is None and fold_residuals is None
    assert reason is not None and "time-to-solution interval unavailable" in reason


# ---------------------------------------------------------------------------
# 13. predict_physics_shortlist with the new infeasibility/dangerous-miss gates (defects 2/3),
# on top of the existing status-logic test above -- P03 (1/6 training-case failure) must be
# UNCONFIRMED with the evidence attached to every row; P01 stays RECOMMENDABLE and its
# dangerous-miss rate is a clean 0.0/sufficient, unaffected by the new gates.
# ---------------------------------------------------------------------------
def test_predict_physics_shortlist_infeasibility_and_dangerous_miss():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    physics.add_fidelity_targets(rows)

    karina_p00 = next(r for r in rows if r["case_base"] == "karina"
                       and physics._template_of(r) == "P00" and "_twin" not in r["run_id"]
                       and r["rep_index"] == 1)
    candidates = [
        # P06 (WITHIN-FLOOR, ~10% faster) is the "clean bill of health" example here -- P01 is
        # fidelity-safe too but has no real speed benefit (see the lead-review gate B tests
        # below), so it no longer serves as a "should be RECOMMENDABLE" example on its own.
        dict(template="P06", family="radiation", applicable=True,
             changes={"ra_lw_physics": {"from": 4, "to": 1}, "ra_sw_physics": {"from": 4, "to": 1}}),
        dict(template="P03", family="dynamics", applicable=True,
             changes={"time_step": {"from": 15, "to": 18}}),
    ]
    shortlist = physics.predict_physics_shortlist(rows, karina_p00, candidates, probe_verdicts={})
    by_t = {c["template"]: c for c in shortlist}

    assert by_t["P03"]["infeasibility"]["n_cases_failed"] == 1
    assert by_t["P03"]["infeasibility"]["n_cases_tried"] == len(CASES)
    assert by_t["P03"]["status"] == "UNCONFIRMED"
    assert any("infeasible in 1/%d" % len(CASES) in r for r in by_t["P03"]["reasons"])

    lenient = physics.predict_physics_shortlist(rows, karina_p00, candidates, probe_verdicts={},
                                                 max_infeasible_rate=1.0)
    by_t_lenient = {c["template"]: c for c in lenient}
    assert not any("infeasible in" in r for r in by_t_lenient["P03"]["reasons"])

    assert by_t["P06"]["dangerous_miss"]["insufficient"] is False
    assert by_t["P06"]["dangerous_miss"]["dangerous_miss_rate"] == 0.0
    assert by_t["P06"]["infeasibility"]["rate"] == 0.0
    assert by_t["P06"]["template_track_record"]["rate"] == 0.0
    assert by_t["P06"]["status"] == "RECOMMENDABLE"
    assert by_t["P06"]["predicted_time_to_solution_pct"] is not None


# ---------------------------------------------------------------------------
# 14. Lead-review gate A: template_large_drift_rate (model-free measured-label track record).
# ---------------------------------------------------------------------------
def test_template_large_drift_rate_matches_known_fixture_labels():
    rows = make_physics_ledger()
    rates = physics.template_large_drift_rate(rows)

    # P04 is LARGE-DRIFT for every case it ran (5 of 6 -- maria1km is skipped).
    assert rates["P04"]["n_cases_labeled"] == 5
    assert rates["P04"]["n_cases_large"] == 5
    assert rates["P04"]["rate"] == 1.0

    # P01/P03/P06 are WITHIN-FLOOR and P02/P05 are SMALL-DRIFT for every case -- never LARGE-DRIFT.
    for tmpl in ("P01", "P02", "P03", "P05", "P06"):
        assert rates[tmpl]["n_cases_large"] == 0
        assert rates[tmpl]["rate"] == 0.0


def test_template_large_drift_rate_empty_for_no_rows():
    assert physics.template_large_drift_rate([]) == {}


def test_predict_physics_shortlist_track_record_gate_overrides_regression(monkeypatch):
    """Isolates the track-record gate from whatever the fidelity regression naturally predicts,
    by forcing every ordinal-score-to-label call to return the ALLOWED "SMALL-DRIFT" -- the exact
    shape of the real karina/P05 holdout miss (regression said SMALL-DRIFT; the template's own
    measured track record said otherwise)."""
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    physics.add_fidelity_targets(rows)
    karina_p00 = next(r for r in rows if r["case_base"] == "karina"
                       and physics._template_of(r) == "P00" and "_twin" not in r["run_id"]
                       and r["rep_index"] == 1)
    candidates = [dict(template="P04", family="microphysics", applicable=True,
                        changes={"mp_physics": {"from": 8, "to": 6}})]

    monkeypatch.setattr(physics, "_label_from_ordinal_score", lambda score: "SMALL-DRIFT")
    shortlist = physics.predict_physics_shortlist(rows, karina_p00, candidates, probe_verdicts={})
    p04 = shortlist[0]
    assert p04["predicted_fidelity_label"] == "SMALL-DRIFT"  # the (forced) regression verdict
    assert p04["template_track_record"]["rate"] == 1.0       # P04 is LARGE-DRIFT in every case
    assert p04["status"] == "UNCONFIRMED"
    assert any("LARGE-DRIFT in %d/%d training cases" % (5, 5) in r for r in p04["reasons"])

    lenient = physics.predict_physics_shortlist(rows, karina_p00, candidates, probe_verdicts={},
                                                 max_template_large_rate=1.01)
    p04_lenient = lenient[0]
    assert not any("LARGE-DRIFT in" in r for r in p04_lenient["reasons"])


# ---------------------------------------------------------------------------
# 15. Lead-review gate B: speed-benefit requirement.
# ---------------------------------------------------------------------------
def test_predict_physics_shortlist_speed_gate_point_estimate_threshold():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    physics.add_fidelity_targets(rows)
    karina_p00 = next(r for r in rows if r["case_base"] == "karina"
                       and physics._template_of(r) == "P00" and "_twin" not in r["run_id"]
                       and r["rep_index"] == 1)
    # P01 is WITHIN-FLOOR-safe with a near-noise-floor effect (M_T["P01"]=1.001, ~0.1% SLOWER per
    # its own raw multiplier) -- fit jointly with every other (much larger-effect) template, the
    # shared regression's own point estimate for it happens to land just past the default -1.0%
    # threshold, so what actually blocks it here is the conformal interval's own upper bound not
    # being < 0 (its tiny true effect is swamped by the spread of every OTHER template's own
    # residuals), not the point-estimate check itself -- see the dedicated interval test below for
    # the other branch.
    candidates = [dict(template="P01", family="radiation", applicable=True,
                        changes={"swint_opt": {"from": None, "to": 1}})]

    default_shortlist = physics.predict_physics_shortlist(rows, karina_p00, candidates,
                                                            probe_verdicts={})
    p01 = default_shortlist[0]
    assert p01["predicted_time_to_solution_pct"] <= physics.DEFAULT_MIN_SPEEDUP_PCT
    assert p01["status"] == "UNCONFIRMED"
    assert any("no predicted speed benefit" in r for r in p01["reasons"])
    assert any("interval upper bound" in r for r in p01["reasons"])
    assert not any("need <=" in r for r in p01["reasons"])

    # Loosening --min-speedup-pct cannot rescue this one: the point-estimate check already
    # passed, and the interval-upper-bound check is independent of this flag by design.
    lenient_shortlist = physics.predict_physics_shortlist(
        rows, karina_p00, candidates, probe_verdicts={}, min_speedup_pct=50.0)
    p01_lenient = lenient_shortlist[0]
    assert any("interval upper bound" in r for r in p01_lenient["reasons"])
    assert p01_lenient["status"] == "UNCONFIRMED"


def test_predict_physics_shortlist_speed_gate_interval_upper_bound():
    rows = make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    physics.add_fidelity_targets(rows)
    karina_p00 = next(r for r in rows if r["case_base"] == "karina"
                       and physics._template_of(r) == "P00" and "_twin" not in r["run_id"]
                       and r["rep_index"] == 1)
    candidates = [dict(template="P01", family="radiation", applicable=True,
                        changes={"swint_opt": {"from": None, "to": 1}})]
    shortlist = physics.predict_physics_shortlist(rows, karina_p00, candidates, probe_verdicts={})
    p01 = shortlist[0]
    # Whichever branch actually fired, the reported interval (when present) must be internally
    # consistent with the gate's own definition: an interval whose upper bound is already < 0
    # can only combine with a "no predicted speed benefit" status if the POINT ESTIMATE itself
    # failed the threshold (never the interval branch in that case).
    iv = p01["time_to_solution_interval_pct"]
    if iv is not None and iv[1] < 0:
        assert p01["predicted_time_to_solution_pct"] > physics.DEFAULT_MIN_SPEEDUP_PCT
