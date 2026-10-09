#!/usr/bin/env python3
"""predictor/tests/test_physics_holdout_eval.py -- pytest suite for
predictor/physics_holdout_eval.py (leave-one-case-out evaluation of the PHYSICS-GRAY
recommendation logic against real measured outcomes).

Run with:
    pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml \\
        python -m pytest predictor/tests/test_physics_holdout_eval.py -v

Builds a synthetic 6-case physics ledger with a KNOWN, deterministic ground truth (P01 always
WITHIN-FLOOR and 5% faster; P02 always LARGE-DRIFT and 8% faster -- a "looks fast but must stay
UNCONFIRMED" trap; P03 crashes (FAILED, both reps) in two of the six cases) so every assertion
below is about the harness's LOGIC, not about a real model's numeric accuracy.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ledger
import physics_holdout_eval

HW = "hw-holdout-eval-synthetic"
CASES = ["case0", "case1", "case2", "case3", "case4", "case5"]
FAIL_CASES = {"case0", "case1"}  # P03 crashes (both reps) only in these two cases
NATIVE = dict(mp_physics=8, cu_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1,
              sf_surface_physics=2, ra_lw_physics=4, ra_sw_physics=4, radt=15.0,
              cudt=0.0, time_step=15.0)
GRID = dict(e_we=901.0, e_sn=901.0, e_vert=45.0, dx=3000.0, dy=3000.0)


def _templates_json_doc():
    """P01/P02/P03 only, no skip_cases/extra_edits (none of our synthetic case names appear in
    physics_templates.json anyway, and P01/P02/P03 never need generic per-case pairing -- see
    physics.generic_extra_edits) -- so every template is applicable to every synthetic case."""
    return {
        "templates": {
            "P01": {"family": "radiation", "edits": [{"op": "set", "key": "swint_opt", "value": 1}]},
            "P02": {"family": "radiation",
                     "edits": [{"op": "mul", "key": "radt", "factor": 2},
                               {"op": "set", "key": "swint_opt", "value": 1}]},
            "P03": {"family": "dynamics", "edits": [{"op": "mul", "key": "time_step", "factor": 1.2}]},
        },
    }


def _notes(template, family, edits_str, decomp, fidelity_nd_max=None, probe=None, failed=False):
    parts = ["mode=physics", "template=%s" % template, "family=%s" % family,
             "edits=%s" % edits_str, "decomp=%s" % decomp, "sim_minutes=30", "hist_minutes=30"]
    if fidelity_nd_max is not None:
        parts.append("fidelity_nd_max=%.4f" % fidelity_nd_max)
    parts.append("probe=%s" % (",".join("%s:%s" % kv for kv in (probe or {}).items()) or "none"))
    if failed:
        parts.append("status=FAILED reason=synthetic_crash")
    return " ".join(parts)


def _row(case, template, family, edits_str, physics_fields, decomp, rep, s,
         fidelity_label=None, nd=None, probe=None, failed=False):
    return ledger.new_row(
        run_id="%s_%s_r%d" % (case, template, rep), source_dataset="physics_campaign",
        job_id="1", rep_index=rep, case_raw=case, case_base=case,
        case_family=ledger.case_family_of(case), hardware_id=HW, binary_kind="pristine",
        compliance_class=("STRICT" if template == "P00" else "PHYSICS-GRAY"),
        e_we=GRID["e_we"], e_sn=GRID["e_sn"], e_vert=GRID["e_vert"], dx=GRID["dx"], dy=GRID["dy"],
        time_step=physics_fields["time_step"], mp_physics=physics_fields["mp_physics"],
        cu_physics=physics_fields["cu_physics"], bl_pbl_physics=physics_fields["bl_pbl_physics"],
        sf_sfclay_physics=physics_fields["sf_sfclay_physics"],
        sf_surface_physics=physics_fields["sf_surface_physics"],
        ra_lw_physics=physics_fields["ra_lw_physics"], ra_sw_physics=physics_fields["ra_sw_physics"],
        radt=physics_fields["radt"], cudt=physics_fields["cudt"],
        history_interval=(30.0 if rep == 1 else 0.0), io_form_history=2,
        n_nodes=4, n_ranks=448, omp_threads=1, nproc_x=8, nproc_y=56, numtiles=1,
        distribution="cyclic", kmp_blocktime=0,
        s_per_step_median=(None if failed else s), s_per_step_mean=(None if failed else s),
        n_steps_measured=(0 if failed else 30), diffwrf_differing_fields=None,
        fidelity_label=(None if failed else (fidelity_label if rep == 1 else None)),
        profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
        profile_mpi_share=None, profile_io_share=None, profile_rank_imbalance=None,
        profile_thread_imbalance=None,
        notes=_notes(template, family, edits_str, decomp,
                     fidelity_nd_max=(nd if (rep == 1 and not failed) else None), probe=probe,
                     failed=failed),
    )


def make_ledger_rows():
    rows = []
    for case in CASES:
        for rep in (1, 2):
            rows.append(_row(case, "P00", "none", "", NATIVE, "C02", rep, 1.0))
        for rep in (1, 2):
            rows.append(_row(case, "P01", "radiation", "swint_opt=1", NATIVE, "C02", rep, 0.95,
                              fidelity_label="WITHIN-FLOOR", nd=0.01,
                              probe={"swint_opt": "CONFIRM"}))
        p02_fields = dict(NATIVE, radt=30.0)
        for rep in (1, 2):
            rows.append(_row(case, "P02", "radiation", "radt=30;swint_opt=1", p02_fields, "C02",
                              rep, 0.92, fidelity_label="LARGE-DRIFT", nd=0.5,
                              probe={"radt": "CONFIRM", "swint_opt": "CONFIRM"}))
        p03_fields = dict(NATIVE, time_step=18.0)
        failed_here = case in FAIL_CASES
        for rep in (1, 2):
            if failed_here:
                rows.append(_row(case, "P03", "dynamics", "time_step=18", p03_fields, "C02", rep,
                                  None, probe={"time_step": "PASS->WRF"}, failed=True))
            else:
                rows.append(_row(case, "P03", "dynamics", "time_step=18", p03_fields, "C02", rep,
                                  0.80, fidelity_label="WITHIN-FLOOR", nd=0.02,
                                  probe={"time_step": "PASS->WRF"}))
    return rows


def _write_ledger_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ledger.LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ledger.LEDGER_FIELDS})


def test_holdout_eval_end_to_end(tmp_path):
    ledger_path = tmp_path / "physics_runs.csv"
    _write_ledger_csv(make_ledger_rows(), str(ledger_path))
    templates_path = tmp_path / "physics_templates.json"
    templates_path.write_text(json.dumps(_templates_json_doc()))
    out_dir = tmp_path / "holdout_eval_out"

    result = physics_holdout_eval.run_holdout_eval(
        [str(ledger_path)], str(out_dir), templates_json_path=str(templates_path))

    assert result["n_cases"] == 6
    assert os.path.exists(out_dir / "holdout_eval_metrics.csv")
    assert os.path.exists(out_dir / "holdout_eval_summary.md")
    with open(out_dir / "holdout_eval_metrics.csv") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 6 * 3  # 6 cases x (P01, P02, P03)

    # P01 is the always-safe, always-a-bit-faster template with clean, separable training
    # evidence in every leave-one-case-out fold (6 cases, never fails, never LARGE-DRIFT) -- it
    # must end up the top-1 RECOMMENDABLE, truly acceptable & feasible, for every held-out case.
    assert result["n_with_top1"] == 6
    assert result["n_top1_ok"] == 6
    for r in result["case_results"]:
        assert r["top1"] is not None and r["top1"]["template"] == "P01"
        assert r["top1"]["measured_truly_acceptable_feasible"] is True

    # P02 is always faster (8%) but always measured LARGE-DRIFT: a real model trained on the
    # other 5 cases (identical feature vector for P02 in every case) must recover that pattern
    # out-of-sample and refuse it -- this is exactly the trap the real campaign smoke job 1077893
    # fell into (lafe/P05 RECOMMENDABLE despite measured LARGE-DRIFT).
    for r in result["case_results"]:
        by_t = {row["template"]: row for row in r["per_template"]}
        assert by_t["P02"]["measured_fidelity_label"] == "LARGE-DRIFT"
        assert by_t["P02"]["predicted_status"] == "UNCONFIRMED"
        assert by_t["P02"]["dangerous_recommendation"] is False

    # P03 crashes in 2 of 6 cases: every held-out evaluation keeps at least one other case's
    # crash (or has too few out-of-fold fidelity predictions) in its training evidence, so
    # defects 2/3's gates must keep it UNCONFIRMED everywhere -- never a dangerous recommendation.
    for r in result["case_results"]:
        by_t = {row["template"]: row for row in r["per_template"]}
        assert by_t["P03"]["predicted_status"] == "UNCONFIRMED"
        assert by_t["P03"]["dangerous_recommendation"] is False

    assert result["total_dangerous"] == 0

    # Measured time-to-solution sanity: P01 ~ -5%, P02 ~ -8% (both faster, negative = faster).
    for r in result["case_results"]:
        by_t = {row["template"]: row for row in r["per_template"]}
        assert -6.0 < by_t["P01"]["measured_time_to_solution_pct"] < -4.0
        assert -9.0 < by_t["P02"]["measured_time_to_solution_pct"] < -7.0

    summary = (out_dir / "holdout_eval_summary.md").read_text()
    assert "cases with a top-1 RECOMMENDABLE template: 6/6" in summary
    assert "dangerous recommendations" in summary


def test_evaluate_case_returns_none_when_no_measured_candidates(tmp_path):
    # A case with only a P00 reference row (no candidates measured at all) has nothing to
    # evaluate -- must return None, not crash.
    rows = [_row("lonelycase", "P00", "none", "", NATIVE, "C02", rep, 1.0) for rep in (1, 2)]
    templates_path = tmp_path / "physics_templates.json"
    templates_path.write_text(json.dumps(_templates_json_doc()))
    result = physics_holdout_eval.evaluate_case(rows, "lonelycase",
                                                 templates_json_path=str(templates_path))
    assert result is None
