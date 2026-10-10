#!/usr/bin/env python3
"""Tests for wtune/calibrate.py -- offline only, no cluster access.

Run: uv run --with pytest --with numpy --with netCDF4 python -m pytest -q \\
     test_wtune_calibrate.py test_wtune_predict.py
"""
import json
import os
import subprocess
import sys

import pytest

from wtune import calibrate as wc
from wtune import predict as wp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ -> repo root
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


def test_pick_near_square_accounts_for_domain_aspect_ratio():
    # sc26p2-like domain (e_we=170, e_sn=130, aspect~1.308); at 72 ranks the near-tie is
    # (8,9) vs (9,8) -- the wider dimension (e_we) should get MORE ranks (nx=9) for the more
    # square PATCH: (170/9,130/8)=(18.9,16.25) [aspect 1.16] vs (170/8,130/9)=(21.25,14.4)
    # [aspect 1.47]; the plain |nx-ny| metric (aspect=1.0 default) cannot tell these apart.
    shapes = [(6, 12), (12, 6), (8, 9), (9, 8)]
    assert wc.pick_near_square(shapes, aspect=170 / 130) == (9, 8)
    assert wc.pick_near_square(shapes) in ((8, 9), (9, 8))  # aspect=1.0 default is ambiguous here


SC26P2_LIKE_NAMELIST = """\
&time_control
 run_days = 0,
 run_hours = 6,
 run_minutes = 0,
 run_seconds = 0,
 restart = .false.,
 io_form_history = 2
 /
&domains
 time_step = 60,
 max_dom = 1,
 e_we = 170,
 e_sn = 130,
 e_vert = 45,
 /
&physics
 /
"""


def test_build_decomp_picks_sc26p2_style_shapes_for_every_planned_rank_count(tmp_path):
    # the exact (ranks -> nx,ny) the coordinator specified for the sc26p2 startup_s_per_node
    # sweep (72->1 node, 144->2, 204->3, 221->4 at cores_per_node=72); 204/221 have only one
    # legal shape each (no choice), 72/144 are where pick_near_square's aspect-awareness matters.
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), SC26P2_LIKE_NAMELIST))
    assert wc.build_decomp(nl, 1, 72, 10) == (9, 8)
    assert wc.build_decomp(nl, 1, 144, 10) == (12, 12)
    assert wc.build_decomp(nl, 1, 204, 10) == (17, 12)
    assert wc.build_decomp(nl, 1, 221, 10) == (17, 13)


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
        # regression (scc job 1090/1091, 2026-09-25): scc.sh's site_hygiene() reads
        # $ALLOC_NODES itself; calling it before ALLOC_NODES is assigned made `set -u` abort
        # the WHOLE script silently (its stderr was redirected away), with an empty .out and
        # no wall.txt -- bash -n cannot catch this (it is a runtime-only ordering bug), so it
        # is pinned here by textual position instead.
        assert text.index("ALLOC_NODES=") < text.index("site_hygiene reap")
        # regression (scc jobs 1090-1099, 2026-09-25): the run dir never linked the cached
        # Thompson lookup tables, so WRF computed them from scratch inside every measured
        # startup window (qr_acr_qg alone: 32.3s at 1 rank), contaminating the whole fit.
        # $W/run/namelist.input is skipped (a real file, would corrupt edit-namelist's write-
        # through) but every other file is linked with no name/extension filter, and the three
        # Thompson tables are linked from $OUT/thompson_tables when cached there.
        assert '[ "$b" = "namelist.input" ] && continue' in text
        for tbl in ("freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat"):
            assert tbl in text
        assert "$OUT/thompson_tables" in text
        assert "wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*" not in text  # the old blanket blacklist is gone

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
              write_seconds=None, write_bytes=None, extra_rsl="", nodes=1,
              input_bytes=None):
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
    wall_text = "WALL_S %.6f\nRC %d\nOK %d\nRANKS %d\nNODES %d\nNX 1\nNY 1\n" % (
        wall_s, rc, ok, ranks, nodes)
    _write(str(run_dir / "wall.txt"), wall_text)
    if input_bytes is not None:
        (run_dir / "wrfinput_d01").write_bytes(b"\0" * input_bytes)
    return run_dir


def _fit_args(runs_dir, out_path, base=REAL_CALIBRATION, cores_per_node=72, site=None,
              host="unit-test-host", no_read_rate=False, no_read_rate_reason=None):
    runs = runs_dir if isinstance(runs_dir, list) else [str(runs_dir)]
    return wc.argparse.Namespace(
        runs=[str(r) for r in runs], out=str(out_path), base=base, cores_per_node=cores_per_node,
        site=site, host=host, no_read_rate=no_read_rate, no_read_rate_reason=no_read_rate_reason)


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


def test_fit_excludes_thompson_table_contaminated_run(tmp_path):
    # regression (scc jobs 1090-1099, 2026-09-25): a run whose rsl shows the table being
    # computed from scratch at init is NOT comparable to one that linked the cached table --
    # it must be excluded even though it completed successfully (RC=0/OK=1).
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, 1, "j1", [10.0], startup_s=2.4)  # clean
    _make_run(runs_dir, 36, "j2", [10.0], startup_s=16.4,
              extra_rsl=("ThompMP: computing qr_acr_qg table: qr_acr_qg_V4.dat\n"
                         "Timing for table computation qr_acr_qg_V4.dat:   32.25516 elapsed seconds\n"))

    out_path = tmp_path / "scc-auto-test.json"
    wc.fit_cmd(_fit_args(runs_dir, out_path))
    out = json.loads(out_path.read_text())

    assert out["validity"]["n_runs_used"] == 1
    assert out["validity"]["n_runs_excluded"] == 1
    excluded = out["validity"]["excluded"][0]
    assert excluded["ranks"] == 36
    assert "contaminated" in excluded["reason"]
    assert "Thompson" in excluded["reason"]


def test_fit_no_read_rate_forces_transfer_even_with_a_separable_line(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run_dir = _make_run(runs_dir, 1, "j1", [10.0], startup_s=2.4,
                         extra_rsl=("Timing for input for domain        1:   "
                                    "5.000000 elapsed seconds\n"))
    (run_dir / "wrfinput_d01").write_bytes(b"\0" * 500_000)
    _make_run(runs_dir, 36, "j2", [10.0], startup_s=16.4)

    out_path = tmp_path / "scc-auto-test.json"
    wc.fit_cmd(_fit_args(runs_dir, out_path, no_read_rate=True,
                          no_read_rate_reason="NFS client cache warmed by back-to-back runs"))
    out = json.loads(out_path.read_text())

    base = json.load(open(REAL_CALIBRATION))
    assert out["derived"]["read_rate_bytes_per_s"] == base["derived"]["read_rate_bytes_per_s"]
    assert "derived.read_rate_bytes_per_s" not in out["calibration_provenance"]["measured_fields"]
    assert "NFS client cache" in out["derived"]["read_rate_transferred_note"]
    assert "--no-read-rate" in out["derived"]["read_rate_transferred_note"]


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


def test_fit_recovers_startup_s_per_node_from_multi_node_runs(tmp_path):
    # cap_true=19.4 (1 node), per_node_true=4.2 -> pure startup 19.4/23.6/27.8/32.0 at
    # nodes=1/2/3/4 (32.0 at 4 nodes matches the coordinator's own sc26p2 evidence number).
    # Each run's raw (wall-sum_main) startup = pure + read_s_pred, read_s_pred = 2,000,000
    # bytes / 1,000,000 B/s effective_read_rate = 2.0s constant, so raw = pure + 2.0.
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, 72, "j1", [100.0], startup_s=21.4, nodes=1, input_bytes=2_000_000)
    _make_run(runs_dir, 144, "j2", [100.0], startup_s=25.6, nodes=2, input_bytes=2_000_000)
    _make_run(runs_dir, 204, "j3", [100.0], startup_s=29.8, nodes=3, input_bytes=2_000_000)
    _make_run(runs_dir, 221, "j4", [100.0], startup_s=34.0, nodes=4, input_bytes=2_000_000)

    base = tmp_path / "base.json"
    _write(str(base), json.dumps({"derived": {"read_rate_bytes_per_s": 1_000_000.0}}))

    out_path = tmp_path / "scc-auto-test.json"
    rc = wc.fit_cmd(_fit_args(runs_dir, out_path, base=str(base)))
    assert rc == 0
    out = json.loads(out_path.read_text())

    assert out["startup_s_per_node"] == pytest.approx(4.2, abs=1e-6)
    assert "MEASURED" in out["startup_s_per_node_source"]
    assert "startup_s_per_node" in out["calibration_provenance"]["measured_fields"]
    # startup_s itself stays the RAW (not read-subtracted) 1-node value: 21.4, not 19.4.
    assert out["startup_s"] == pytest.approx(21.4)


def test_fit_rejects_negative_startup_s_per_node_slope(tmp_path):
    # regression (scc 2026-09-25): a contaminated 1-node point inflates cap_pure so far above
    # the real multi-node points that the least-squares slope against (nodes-1) comes out
    # NEGATIVE -- physically implausible (the model's whole premise is that the cap GROWS with
    # node count) and must be rejected, not written, symmetric to the per_rank guard above.
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, 72, "j1", [10.0], startup_s=105.757, nodes=1, input_bytes=2_000_000)
    _make_run(runs_dir, 144, "j2", [10.0], startup_s=47.0, nodes=2, input_bytes=2_000_000)
    _make_run(runs_dir, 204, "j3", [10.0], startup_s=41.3, nodes=3, input_bytes=2_000_000)

    base = tmp_path / "base.json"
    _write(str(base), json.dumps({"derived": {"read_rate_bytes_per_s": 1_000_000.0}}))

    out_path = tmp_path / "scc-auto-test.json"
    rc = wc.fit_cmd(_fit_args(runs_dir, out_path, base=str(base)))
    assert rc == 0
    out = json.loads(out_path.read_text())

    assert "startup_s_per_node" not in out
    assert "REJECTED" in out["startup_s_per_node_note"]
    assert "negative" in out["startup_s_per_node_note"].lower()
    assert "startup_s_per_node" not in out["calibration_provenance"]["measured_fields"]


def test_fit_startup_s_per_node_absent_without_multi_node_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, 1, "j1", [10.0], startup_s=2.4, nodes=1)
    _make_run(runs_dir, 36, "j2", [10.0], startup_s=16.4, nodes=1)

    out_path = tmp_path / "scc-auto-test.json"
    wc.fit_cmd(_fit_args(runs_dir, out_path))
    out = json.loads(out_path.read_text())

    assert "startup_s_per_node" not in out
    assert "not fitted" in out["startup_s_per_node_note"]
    assert "startup_s_per_node" not in out["calibration_provenance"]["measured_fields"]


def test_fit_rejects_negative_per_rank_slope(tmp_path):
    # regression (scc 2026-09-25): a contaminated 1-rank point (startup=105.757s, vs 36 ranks'
    # 38.239s) fits a NEGATIVE per_rank=-1.929 -- fed into predict_init's min(startup_s+...,
    # startup_s_base+per_rank*ranks), that makes the second term fall without bound as ranks
    # grows, so predicting a large-rank-count case with the resulting calibration returned a
    # deeply NEGATIVE total_s (conus12 at 288 ranks: -376.6s). The fit must refuse to write a
    # negative per_rank rather than silently corrupt every downstream prediction.
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_run(runs_dir, 1, "j1", [82.749], startup_s=105.757, nodes=1)
    _make_run(runs_dir, 36, "j2", [5.890], startup_s=38.239, nodes=1)
    _make_run(runs_dir, 72, "j3", [5.065], startup_s=48.182, nodes=1)

    out_path = tmp_path / "scc-auto-test.json"
    rc = wc.fit_cmd(_fit_args(runs_dir, out_path))
    assert rc == 0
    out = json.loads(out_path.read_text())

    assert "startup_s_base" not in out or out["startup_s_base"] == json.load(open(REAL_CALIBRATION)).get("startup_s_base")
    assert "startup_s_per_rank" not in out or out["startup_s_per_rank"] == json.load(open(REAL_CALIBRATION)).get("startup_s_per_rank")
    assert "startup_s_base" not in out["calibration_provenance"]["measured_fields"]
    assert "startup_s_per_rank" not in out["calibration_provenance"]["measured_fields"]
    assert "REJECTED" in out["startup_s_scaling_note"]
    assert "negative" in out["startup_s_scaling_note"].lower()

    # and predict_init must not be handed a negative slope at all: build_prediction on a
    # large rank count must not go negative.
    case_dir = tmp_path / "predict_case"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), PREDICT_FIXTURE_NAMELIST)
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=4,
        config={"ranks": 288, "omp": 1, "nx": 18, "ny": 16, "numtiles": 1},
        ledger_paths=None, calibration_path=str(out_path),
        io_form_override=None, io_target="nfs")
    assert pred["total_s"] > 0


def test_fit_pools_multiple_runs_directories(tmp_path):
    # --runs is repeatable: one small-rank sweep (severe2023-shaped) + one separate multi-node
    # sweep (sc26p2-shaped) both feed the same fit, without the caller merging directories.
    runs_a = tmp_path / "runs_severe"
    runs_a.mkdir()
    runs_b = tmp_path / "runs_sc26p2"
    runs_b.mkdir()
    _make_run(runs_a, 1, "j1", [10.0], startup_s=2.4, nodes=1)
    _make_run(runs_a, 36, "j2", [10.0], startup_s=16.4, nodes=1)
    _make_run(runs_b, 72, "j3", [10.0], startup_s=19.4, nodes=1)
    _make_run(runs_b, 144, "j4", [10.0], startup_s=23.6, nodes=2)

    out_path = tmp_path / "out.json"
    rc = wc.fit_cmd(_fit_args([runs_a, runs_b], out_path))
    assert rc == 0
    out = json.loads(out_path.read_text())

    assert out["validity"]["n_runs_used"] == 4
    assert out["startup_s"] == pytest.approx(19.4)       # from runs_b's lone nodes==1 point
    assert out["startup_s_base"] == pytest.approx(2.0)    # from runs_a's ranks<72 points
    assert out["startup_s_per_rank"] == pytest.approx(0.4)
    assert set(out["calibration_provenance"]["runs_dirs"]) == {str(runs_a.resolve()), str(runs_b.resolve())}


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
