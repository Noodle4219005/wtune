#!/usr/bin/env python3
"""Tests for wtune_calibrate.py -- offline only, no cluster access.

Run: uv run --with pytest --with numpy --with netCDF4 python -m pytest -q \\
     test_wtune_calibrate.py test_wtune_predict.py
"""
import json
import os
import subprocess
import sys

import pytest

import wtune_calibrate as wc
import wtune_predict as wp

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_CALIBRATION = os.path.join(HERE, "machine", "calibration", "scc.json")


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------------
# decomposition selection (predictor/decomp_space.py integration)
# --------------------------------------------------------------------------

def test_pick_near_square_prefers_square_over_elongated():
    # 36 = 6x6 (square) or 4x9/9x4 (elongated); square must win.
    shapes = [(4, 9), (9, 4), (6, 6), (2, 18), (18, 2)]
    assert wc.pick_near_square(shapes) == (6, 6)


def test_pick_near_square_deterministic_tiebreak():
    # Two equally-square shapes (|nx-ny| tied) -> smaller max(nx,ny) wins, then nx ascending.
    shapes = [(3, 4), (4, 3), (2, 6)]
    assert wc.pick_near_square(shapes) == (3, 4)


SEVERE_LIKE_NAMELIST = """\
&time_control
 run_days = 0,
 run_hours = 24,
 run_minutes = 0,
 run_seconds = 0,
 history_interval = 180, 180,
 restart = .false.,
 io_form_history = 2
 /
&domains
 time_step = 180,
 max_dom = 2,
 e_we = 100, 118,
 e_sn = 100, 103,
 e_vert = 45, 45,
 parent_grid_ratio = 1, 3,
 parent_time_step_ratio = 1, 3,
 /
&physics
 /
"""


def test_build_decomp_finds_legal_near_square_shape(tmp_path):
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), SEVERE_LIKE_NAMELIST))
    shape = wc.build_decomp(nl, 2, 72, 10)
    assert shape == (8, 9)


def test_build_decomp_returns_none_when_geometry_forbids_every_shape(tmp_path):
    # e_we/e_sn=100 on d01 caps legal (nx,ny) at nx,ny<=10 -> nx*ny<=100; 144 has none.
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), SEVERE_LIKE_NAMELIST))
    assert wc.build_decomp(nl, 2, 144, 10) is None
    assert wc.build_decomp(nl, 2, 100, 10) == (10, 10)


# --------------------------------------------------------------------------
# plan: writes sbatch scripts + MANIFEST.json, skips infeasible rank counts
# --------------------------------------------------------------------------

def _plan_args(case_dir, out_dir, ranks_list="1,36,72,144,288"):
    return wc.argparse.Namespace(
        site="scc", case_dir=str(case_dir), out=str(out_dir), ranks_list=ranks_list,
        cores_per_node=72, steps=20, min_patch=10, time_limit="00:15:00")


def test_plan_skips_infeasible_ranks_and_writes_manifest(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), SEVERE_LIKE_NAMELIST)
    out_dir = tmp_path / "plan_out"

    rc = wc.plan_cmd(_plan_args(case_dir, out_dir))
    assert rc == 0

    manifest = json.loads((out_dir / "MANIFEST.json").read_text())
    planned_ranks = sorted(r["ranks"] for r in manifest["runs"])
    skipped_ranks = sorted(s["ranks"] for s in manifest["skipped"])
    assert planned_ranks == [1, 36, 72]
    assert skipped_ranks == [144, 288]
    for s in manifest["skipped"]:
        assert "floor(e/nproc)" in s["reason"]

    # largest surviving rank count gets the one-frame history write, others stay off
    by_ranks = {r["ranks"]: r for r in manifest["runs"]}
    assert by_ranks[72]["history_mode"] == "end_d01"
    assert by_ranks[1]["history_mode"] == "off"
    assert by_ranks[36]["history_mode"] == "off"
    assert by_ranks[36]["nx"] == 6 and by_ranks[36]["ny"] == 6

    for ranks in (1, 36, 72):
        sbatch_path = out_dir / ("plan_%d.sbatch" % ranks)
        assert sbatch_path.exists()
        # every generated script must be syntactically valid bash
        r = subprocess.run(["bash", "-n", str(sbatch_path)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        text = sbatch_path.read_text()
        assert "#SBATCH --exclude=head" in text
        assert "--no-adios2" in text
        assert "--set restart=.false." in text

    mprof = out_dir / "machine_profile.sbatch"
    assert mprof.exists()
    r = subprocess.run(["bash", "-n", str(mprof)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_plan_raises_when_every_requested_rank_count_is_infeasible(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), SEVERE_LIKE_NAMELIST)
    out_dir = tmp_path / "plan_out"
    with pytest.raises(wc.CalibrateError):
        wc.plan_cmd(_plan_args(case_dir, out_dir, ranks_list="144,288"))


# --------------------------------------------------------------------------
# fit: recovers startup_s/_base/_per_rank from synthetic runs with known numbers
# --------------------------------------------------------------------------

def _rsl_main_lines(seconds_list):
    lines = []
    for i, s in enumerate(seconds_list):
        lines.append("Timing for main: time 2023-03-31_00:%02d:00 on domain    1:   %.6f elapsed seconds"
                     % (i, s))
        # a domain-2 line must be ignored by the d01-only parser
        lines.append("Timing for main: time 2023-03-31_00:%02d:00 on domain    2:   %.6f elapsed seconds"
                     % (i, s / 3.0))
    return "\n".join(lines) + "\n"


def _make_run(runs_dir, ranks, jobid, main_seconds, startup_s, rc=0, ok=1,
              write_seconds=None, write_bytes=None, extra_rsl=""):
    run_dir = runs_dir / ("run_%d_%s" % (ranks, jobid))
    run_dir.mkdir(parents=True)
    sum_main = sum(main_seconds)
    wall_s = sum_main + startup_s
    rsl_text = _rsl_main_lines(main_seconds)
    if write_seconds is not None:
        rsl_text += ("Timing for Writing wrfout_d01_2023-03-31_01:00:00 for domain        1:   "
                     "%.6f elapsed seconds\n" % write_seconds)
        if write_bytes is not None:
            (run_dir / "wrfout_d01_2023-03-31_01:00:00").write_bytes(b"\0" * write_bytes)
    rsl_text += extra_rsl
    _write(str(run_dir / "rsl.error.0000"), rsl_text)
    wall_text = "WALL_S %.6f\nRC %d\nOK %d\nRANKS %d\nNODES 1\nNX 1\nNY 1\n" % (
        wall_s, rc, ok, ranks)
    _write(str(run_dir / "wall.txt"), wall_text)
    return run_dir


def _fit_args(runs_dir, out_path, base=REAL_CALIBRATION, cores_per_node=72, site=None, host="unit-test-host"):
    return wc.argparse.Namespace(
        runs=str(runs_dir), out=str(out_path), base=base, cores_per_node=cores_per_node,
        site=site, host=host)


def test_fit_recovers_known_startup_base_per_rank_and_cap(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # base_true=2.0 per_rank_true=0.4: startup(1)=2.4 startup(36)=16.4; cap_true=19.4 at ranks=72
    _make_run(runs_dir, 1, "j1", [25.0, 25.0, 25.0, 25.0], startup_s=2.4)
    _make_run(runs_dir, 36, "j2", [25.0, 25.0, 25.0, 25.0], startup_s=16.4)
    _make_run(runs_dir, 72, "j3", [25.0, 25.0, 25.0, 25.0], startup_s=19.4,
              write_seconds=10.0, write_bytes=1_000_000)

    out_path = tmp_path / "scc-auto-test.json"
    rc = wc.fit_cmd(_fit_args(runs_dir, out_path))
    assert rc == 0

    out = json.loads(out_path.read_text())
    assert out["startup_s"] == pytest.approx(19.4)
    assert out["startup_s_base"] == pytest.approx(2.0)
    assert out["startup_s_per_rank"] == pytest.approx(0.4)
    # 1,000,000 bytes / 10s = 100,000 B/s = 0.1 MB/s (decimal)
    assert out["history_write_rate_MBps_measured"] == pytest.approx(0.1)
    assert out["validity"]["n_runs_used"] == 3
    assert out["validity"]["n_runs_excluded"] == 0


def test_fit_excludes_failed_run_with_reason(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, 1, "j1", [25.0, 25.0], startup_s=2.4)
    _make_run(runs_dir, 36, "j2", [25.0, 25.0], startup_s=16.4)
    _make_run(runs_dir, 6, "jfail", [25.0], startup_s=3.0, rc=1, ok=0)  # failed run

    out_path = tmp_path / "scc-auto-test.json"
    wc.fit_cmd(_fit_args(runs_dir, out_path))
    out = json.loads(out_path.read_text())

    assert out["validity"]["n_runs_used"] == 2
    assert out["validity"]["n_runs_excluded"] == 1
    excluded = out["validity"]["excluded"][0]
    assert excluded["ranks"] == 6
    assert "did not complete" in excluded["reason"]
    assert "RC=1" in excluded["reason"]


def test_fit_provenance_measured_vs_transferred(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, 1, "j1", [10.0], startup_s=2.4)
    _make_run(runs_dir, 36, "j2", [10.0], startup_s=16.4)
    _make_run(runs_dir, 72, "j3", [10.0], startup_s=19.4)

    out_path = tmp_path / "scc-auto-test.json"
    wc.fit_cmd(_fit_args(runs_dir, out_path))
    out = json.loads(out_path.read_text())

    prov = out["calibration_provenance"]
    assert "startup_s" in prov["measured_fields"]
    assert "startup_s_base" in prov["measured_fields"]
    assert "MEASURED" in out["startup_s_source"]
    assert "unit-test-host" in out["startup_s_source"]

    # No run supplied a separable read-timing line -> read rate stays TRANSFERRED, not claimed measured.
    assert "derived.read_rate_bytes_per_s" not in prov["measured_fields"]
    assert "read_rate_transferred_note" in out["derived"]
    assert "TRANSFERRED" in out["derived"]["read_rate_transferred_note"]
    assert REAL_CALIBRATION in out["derived"]["read_rate_transferred_note"] or \
        os.path.basename(REAL_CALIBRATION) in out["derived"]["read_rate_transferred_note"]
    # the untouched read rate value must be byte-identical to the base file's
    base = json.load(open(REAL_CALIBRATION))
    assert out["derived"]["read_rate_bytes_per_s"] == base["derived"]["read_rate_bytes_per_s"]

    # every top-level key copied verbatim from base is enumerated, not silently presented
    assert "radiation" in prov["transferred_fields"]
    assert "drift_band_pct" in prov["transferred_fields"]


def test_fit_measures_read_rate_when_a_separable_line_exists(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run_dir = _make_run(runs_dir, 1, "j1", [10.0], startup_s=2.4,
                         extra_rsl=("Timing for input for domain        1:   "
                                    "5.000000 elapsed seconds\n"))
    (run_dir / "wrfinput_d01").write_bytes(b"\0" * 500_000)
    _make_run(runs_dir, 36, "j2", [10.0], startup_s=16.4)

    out_path = tmp_path / "scc-auto-test.json"
    wc.fit_cmd(_fit_args(runs_dir, out_path))
    out = json.loads(out_path.read_text())

    assert out["derived"]["read_rate_bytes_per_s"] == pytest.approx(100_000.0)  # 500000/5
    assert "MEASURED" in out["derived"]["read_rate_formula"]
    assert "derived.read_rate_bytes_per_s" in out["calibration_provenance"]["measured_fields"]


def test_fit_raises_when_no_run_directories_found(tmp_path):
    runs_dir = tmp_path / "empty"
    runs_dir.mkdir()
    with pytest.raises(wc.CalibrateError):
        wc.fit_cmd(_fit_args(runs_dir, tmp_path / "out.json"))


# --------------------------------------------------------------------------
# integration: the fitted calibration file loads and works inside wtune_predict
# --------------------------------------------------------------------------

PREDICT_FIXTURE_NAMELIST = """\
&time_control
 run_minutes = 60,
 history_interval = 60,
 io_form_history = 2,
 /
&domains
 max_dom = 1,
 e_we = 10,
 e_sn = 10,
 e_vert = 10,
 time_step = 60,
 /
"""


def test_fitted_calibration_loads_in_wtune_predict_build_prediction(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, 1, "j1", [10.0, 10.0], startup_s=2.4)
    _make_run(runs_dir, 36, "j2", [10.0, 10.0], startup_s=16.4)
    _make_run(runs_dir, 72, "j3", [10.0, 10.0], startup_s=19.4,
              write_seconds=5.0, write_bytes=500_000)

    out_path = tmp_path / "scc-auto-test.json"
    wc.fit_cmd(_fit_args(runs_dir, out_path))

    case_dir = tmp_path / "predict_case"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), PREDICT_FIXTURE_NAMELIST)

    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=1,
        config={"ranks": 72, "omp": 1, "nx": 8, "ny": 9, "numtiles": 1},
        ledger_paths=None, calibration_path=str(out_path),
        io_form_override=None, io_target="nfs")

    assert pred["total_s"] > 0
    assert pred["init"]["startup_calibrated"] is True
    # ranks=72 == cores_per_node -> the cap applies, min(cap, base+per_rank*72) == cap == 19.4
    assert pred["init"]["startup_term_s"] == pytest.approx(19.4)
    text = wp.render_text(pred)
    assert "TOTAL predicted wall" in text
