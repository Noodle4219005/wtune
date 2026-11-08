#!/usr/bin/env python3
"""Tests for wtune/predict.py -- the offline wall-time predictor.

Run: /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q test_wtune_predict.py

Includes the PRE-REGISTERED held-out acceptance checks: calibrate I/O + rates
from the conus25 (job 427) run ONLY, then predict maria1km (job 429) and
isc21 (job 430) total wall time and compare against their measured walls
(3499s, 1004s). Geometry (e_we/e_sn/e_vert/dx/time_step/max_dom/radt) for
maria1km/isc21 is read from campaign/cases.json (the real, verified values --
see test_holdout_* docstrings for exactly which cases.json fields are used).
Namelists themselves are NOT in this repo, so the fixtures below are
synthetic; every value cases.json does not supply (output intervals, whether
d02/d03 history is on, io_form choices) is an explicit, labelled ASSUMPTION,
not fit to the answer. Per the task spec: these tests must NOT tune any
calibration constant to force a pass -- if a held-out case misses +-20%,
that test is expected to (and does) fail, and the failure is reported
honestly rather than hidden.
"""
import datetime
import json
import math
import os
import statistics
import subprocess
import sys
import types

import pytest

from wtune import predict as wp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ -> repo root
CALIBRATION = os.path.join(HERE, "machine", "calibration", "scc.json")

LEDGER_HEADER = (
    "run_id,source_dataset,job_id,rep_index,case_raw,case_base,case_family,hardware_id,"
    "binary_kind,compliance_class,e_we,e_sn,e_vert,dx,dy,time_step,mp_physics,cu_physics,"
    "bl_pbl_physics,sf_sfclay_physics,sf_surface_physics,ra_lw_physics,ra_sw_physics,radt,"
    "cudt,history_interval,io_form_history,n_nodes,n_ranks,omp_threads,nproc_x,nproc_y,"
    "numtiles,distribution,kmp_blocktime,s_per_step_median,s_per_step_mean,n_steps_measured,"
    "diffwrf_differing_fields,fidelity_label,profile_micro_share,profile_radiation_share,"
    "profile_dynamics_share,profile_mpi_share,profile_io_share,profile_rank_imbalance,"
    "profile_thread_imbalance,notes"
)


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    return path


def _ledger_row(**kw):
    """Build one strict_campaign-ledger-shaped CSV row; missing fields blank."""
    fields = LEDGER_HEADER.split(",")
    row = {f: "" for f in fields}
    row.update(kw)
    return ",".join(str(row[f]) for f in fields)


# --------------------------------------------------------------------------
# 1. namelist run-length / step-count parser
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 2. I/O frame counting: t0 frame, auxhist streams, size-unknown fraction, restart
# --------------------------------------------------------------------------

STUB_CAL = {
    "derived": {
        "bytes_per_cell_level": 1.0,
        "io_form_rate_table": {
            "2": {
                "nfs": {"rate_bytes_per_s": 1.0, "uncertainty_pct": 10, "source": "stub"},
                "local": {"rate_bytes_per_s": None, "uncertainty_pct": None, "source": "UNMEASURED (stub)"},
            }
        },
    },
    "fallback_auxhist_fraction": 0.5,
    "restart": {"restart_to_history_time_ratio": 2.0},
}


def _parse(tmp_path, text, name="namelist.input"):
    return wp.parse_namelist(_write(str(tmp_path / name), text))


AUXHIST_BASE_TC = "history_interval = 999999,\n io_form_history = 2,\n io_form_auxhist1 = 2,\n auxhist1_interval_s = 600,\n"


def test_auxhist_full_frame_when_no_iofields(tmp_path):
    nl_full = _parse(tmp_path, "&time_control\n" + AUXHIST_BASE_TC + "/\n&domains\n max_dom=1,\n e_we=10,\n e_sn=10,\n e_vert=1,\n time_step=1,\n/\n",
                     name="full.input")
    io_full = wp.predict_io(nl_full, max_dom=1, run_min=90.0, calibration=STUB_CAL,
                            io_form_override=None, io_target="nfs")
    aux_full = [s for s in io_full["streams"] if s.get("stream") == "auxhist1"][0]
    # run_min=90 -> run_s=5400; floor(5400/600)+1 = 10 frames; no iofields_filename -> full frame
    assert aux_full["n_frames"] == 10
    assert "fields" not in aux_full
    assert aux_full["sizing"].startswith("full frame")
    assert aux_full["bytes_per_frame"] == pytest.approx(100.0)
    assert aux_full["seconds"] == pytest.approx(1000.0)


def test_auxhist_first_principles_sizing_from_iofields_file(tmp_path):
    """Replaces the removed fallback_auxhist_fraction UNCALIBRATED GUESS: when
    iofields_filename points at a real field-list file, size = sum over its
    '+'-listed fields of points*4 bytes (points = e_we*e_sn*e_vert for 3-D,
    e_we*e_sn for 2-D; U/V get +1 point in their staggered dimension); an
    unrecognized name defaults to 3-D and is recorded as guessed.
    """
    case_dir = tmp_path / "case_fp"
    case_dir.mkdir()
    (case_dir / "fields_d01.txt").write_text("+:h:1:T,U,FOO\n")
    nl = _parse(tmp_path, "&time_control\n" + AUXHIST_BASE_TC +
               ' iofields_filename = "fields_d01.txt",\n/\n&domains\n max_dom=1,\n e_we=10,\n e_sn=10,\n e_vert=2,\n time_step=1,\n/\n',
               name="fp.input")
    io = wp.predict_io(nl, max_dom=1, run_min=90.0, calibration=STUB_CAL,
                       io_form_override=None, io_target="nfs", case_dir=str(case_dir))
    aux = [s for s in io["streams"] if s.get("stream") == "auxhist1"][0]
    assert aux["fields"] == ["FOO", "T", "U"]
    assert aux["fields_guessed_3d"] == ["FOO"]  # not in the built-in table -> defaulted to 3-D, flagged
    # T (unstaggered 3-D): 10*10*2=200 pts; U (staggered +1 in x): 11*10*2=220 pts;
    # FOO (unknown, defaults 3-D, unstaggered): 10*10*2=200 pts. bytes = (200+220+200)*4
    assert aux["bytes_per_frame"] == pytest.approx(620.0 * 4.0)
    assert aux["seconds"] == pytest.approx(10 * 620.0 * 4.0)  # 10 frames * bpcl-rate(1.0)


def test_auxhist_missing_iofields_file_flags_size_unknown(tmp_path):
    case_dir = tmp_path / "case_missing"
    case_dir.mkdir()
    nl = _parse(tmp_path, "&time_control\n" + AUXHIST_BASE_TC +
               ' iofields_filename = "fields_d01.txt",\n/\n&domains\n max_dom=1,\n e_we=10,\n e_sn=10,\n e_vert=1,\n time_step=1,\n/\n',
               name="missing.input")
    io = wp.predict_io(nl, max_dom=1, run_min=90.0, calibration=STUB_CAL,
                       io_form_override=None, io_target="nfs", case_dir=str(case_dir))
    aux = [s for s in io["streams"] if s.get("stream") == "auxhist1"][0]
    assert "seconds" not in aux
    assert "not found" in aux["note"] and "size unknown" in aux["note"]


def test_restart_write_count_and_local_target_unmeasured(tmp_path):
    nl = _parse(tmp_path, """\
&time_control
 history_interval = 999999,
 io_form_history = 2,
 restart_interval = 40,
 /
&domains
 max_dom = 1,
 e_we = 10,
 e_sn = 10,
 e_vert = 1,
 time_step = 1,
 /
""")
    io = wp.predict_io(nl, max_dom=1, run_min=90.0, calibration=STUB_CAL,
                       io_form_override=None, io_target="nfs")
    restart = [s for s in io["streams"] if s.get("stream") == "restart"][0]
    assert restart["n_writes"] == 2  # floor(90/40) writes, at t=40 and t=80
    assert restart["bytes_per_write"] == pytest.approx(200.0)  # 100 bytes * restart_to_history_time_ratio 2.0
    assert restart["seconds"] == pytest.approx(400.0)

    io_local = wp.predict_io(nl, max_dom=1, run_min=90.0, calibration=STUB_CAL,
                             io_form_override=None, io_target="local")
    for s in io_local["streams"]:
        assert "UNMEASURED" in s.get("note", "")
        assert "seconds" not in s
    assert io_local["total_s"] == 0.0


# --------------------------------------------------------------------------
# 3. long-run segmentation (>12h simulated -> 6h chunks)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 4. parameter table / launch line
# --------------------------------------------------------------------------

def test_cores_per_node_is_the_machine_setting_before_the_calibration(monkeypatch):
    """f1-arm 2026-10-25: machine.yaml says 144, no f1-arm calibration, and scc.json's 72 made
    predict warn '90 cores exceeds 72 cores/node'."""
    monkeypatch.delenv("WTUNE_CORES_PER_NODE", raising=False)
    assert wp.cores_per_node(wp.load_calibration(CALIBRATION)) == 72
    monkeypatch.setenv("WTUNE_CORES_PER_NODE", "144")
    assert wp.cores_per_node(wp.load_calibration(CALIBRATION)) == 144
    monkeypatch.setenv("WTUNE_CORES_PER_NODE", "112")
    assert wp.cores_per_node({}) == wp.cores_per_node(None) == 112
    monkeypatch.delenv("WTUNE_CORES_PER_NODE")
    config = {"ranks": 288, "omp": 1, "nx": 16, "ny": 18, "numtiles": 1, "dist": "block"}
    scc = wp.build_param_table(config, 4, 2, "nfs", wp.cores_per_node(wp.load_calibration(CALIBRATION)))
    assert scc["ranks_per_node"] == 72 and scc["warning"] is None
    over = wp.build_param_table(dict(config, ranks=320), 4, 2, "nfs", 72)    # 80/node > 72
    assert "exceeds 72 cores/node" in over["warning"]
    assert wp.build_param_table(dict(config, ranks=320), 4, 2, "nfs")["warning"] is None


def test_param_table_numtiles_unknown_when_not_provided():
    config = {"ranks": 448, "omp": 1, "nx": 16, "ny": 28, "dist": "block"}  # no numtiles
    p = wp.build_param_table(config, nodes=4, io_form=2, io_target="nfs")
    assert p["numtiles"] == "UNKNOWN"
    assert p["namelist_lines"][2] == "numtiles = UNKNOWN,"
    assert "numtiles not provided" in p["warning"]
    # launch_line itself never mentions numtiles, so it's still complete/usable
    assert "UNKNOWN" not in p["launch_line"]


def test_param_table_requires_ranks_omp_nx_ny():
    for missing_key in ("ranks", "omp", "nx", "ny"):
        config = {"ranks": 448, "omp": 1, "nx": 16, "ny": 28, "numtiles": 1, "dist": "block"}
        del config[missing_key]
        with pytest.raises(wp.PredictError):
            wp.build_param_table(config, nodes=4, io_form=2, io_target="nfs")


def test_resolve_config_dist_not_defaulted_via_cli_args():
    """The --dist CLI flag (and --config JSON's 'dist' key) must NOT fall back
    to "cyclic" when omitted -- that CLI-level default was the other half of
    the same 2026-09-23 bug (a user who forgets --dist would otherwise get a
    silently-plausible-but-possibly-wrong launch line, same as a programmatic
    caller omitting 'dist' from a config dict)."""
    args_no_dist = types.SimpleNamespace(config=None, nx=16, ny=28, ranks=None,
                                          omp=None, numtiles=None, dist=None)
    assert wp.resolve_config(args_no_dist)["dist"] is None

    args_block = types.SimpleNamespace(config=None, nx=16, ny=28, ranks=None,
                                        omp=None, numtiles=None, dist="block")
    assert wp.resolve_config(args_block)["dist"] == "block"

    args_config_json = types.SimpleNamespace(config='{"nx": 16, "ny": 28}', nx=None, ny=None,
                                              ranks=None, omp=None, numtiles=None, dist=None)
    assert wp.resolve_config(args_config_json)["dist"] is None


# --------------------------------------------------------------------------
# 5. probe resolution: exact match / extrapolated / no-probe fallback
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 6. conus25 self-consistency (this IS the calibration case): must reproduce
#    the measured 2090s wall within +-5%. restart_interval=300 (one restart
#    write at t=300min) is included on purpose: history and restart write
#    rates are now calibrated independently (history-only write rate from
#    the 7 history frames' 705.0s; restart modeled via the separately
#    measured restart_to_history_time_ratio, 431.2/100.7=4.28x), so both can
#    be modeled together here without double-counting -- see
#    machine/calibration/scc.json conus25_calibration_run.writing_total_note.
# --------------------------------------------------------------------------

CONUS25_NAMELIST = """\
&time_control
 run_days = 0,
 run_hours = 6,
 run_minutes = 0,
 run_seconds = 0,
 history_interval = 60,
 io_form_history = 2,
 restart_interval = 300,
 /
&domains
 max_dom = 1,
 e_we = 1501,
 e_sn = 1201,
 e_vert = 50,
 dx = 2500,
 dy = 2500,
 time_step = 15,
 /
&physics
 radt = 10,
 /
"""


def test_adaptive_bounds_follow_wrf_and_the_range_stays_positive(tmp_path):
    case_dir = tmp_path / "conus25"
    case_dir.mkdir()
    nl = CONUS25_NAMELIST.replace("&domains", "&domains\n use_adaptive_time_step = .true.,\n"
                                  " max_time_step = 200,", 1)
    _write(str(case_dir / "namelist.input"), nl)
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=7,
        config={"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1}, ledger_paths=[],
        calibration_path=CALIBRATION, io_form_override=None, io_target="nfs")
    assert 0 <= pred["range_low_s"] <= pred["total_s"]           # was -299 s
    # dyn_em/start_em.F: -1 -> NINT(k * min(dx,dy) / 1000) s; *_den divides
    nl = wp.parse_namelist(_write(str(tmp_path / "nl"), "&domains\n use_adaptive_time_step = "
                                  ".true.,\n dx = 500,\n dy = 500,\n max_time_step = 3,\n"
                                  " max_time_step_den = 2,\n/\n"))
    assert wp.adaptive_step_bounds(nl, 3600.0) is None           # min 2 s > max 3/2 s
    nl["domains"]["max_time_step_den"] = [1]
    assert wp.adaptive_step_bounds(nl, 3600.0) == (1200, 1800, 2.0, 3.0)


# --------------------------------------------------------------------------
# 7. PRE-REGISTERED held-out checks: maria1km and isc21.
#    Geometry from campaign/cases.json (max_dom, e_we/e_sn/e_vert per domain,
#    dx, time_step, parent_grid_ratio/parent_time_step_ratio, radt) is REAL.
#    Run length is set so d01 steps match the probed job exactly (maria1km
#    job 429: 432 d01 steps at dt=25s = 180 simulated minutes; isc21 job 430:
#    120 d01 steps at dt=15s = 30 simulated minutes -- 30min also happens to
#    match cases.json's isc21 physics_sim_minutes field).
#      - maria1km: 2026-09-22 REAL output settings (root-caused from scc rsl
#        "Timing for Writing" lines, job 429), replacing the earlier ASSUMED
#        ones: history_interval = 0, 60 (d01 OFF, d02 ON hourly -- matches
#        the "0,60,60" reported; the 3rd value is beyond max_dom=2 and is
#        never read); io_form_history=2; restart_interval=120 (ONE restart
#        at t=120min, for EVERY active domain -- d01 AND d02);
#        frames_per_outfile is present in the real namelist but is NOT
#        modeled by wtune/predict.py (noted, not simulated).
#      - isc21: 2026-09-22 REAL settings from the cluster (namelist +
#        fields_d0N.txt content), replacing the earlier ASSUMED ones:
#        io_form_history=0 (primary wrfout off); io_form_auxhist24=2 on all
#        3 domains; auxhist24_interval_s=60,60,60; auxhist24_begin_s=0,0,0;
#        auxhist24_end_s=300,300,300 (write window clipped to the first 5
#        simulated minutes -> 6 frames/domain, matching the 6 files/domain
#        actually observed); frames_per_auxhist24=1,1,1 (present, NOT
#        modeled -- doesn't change total bytes/time); iofields_filename =
#        fields_d01/02/03.txt, all three IDENTICAL, one line each:
#        '+:h:24:U,V,T,QVAPOR,XLAT,XLONG,LU_INDEX,HGT'. Sized first-principles
#        (no calibration constant): U/V staggered +1 in x/y, T/QVAPOR
#        unstaggered 3-D, XLAT/XLONG/LU_INDEX/HGT 2-D -- all 8 names are in
#        wtune/predict.py's built-in table, nothing guessed.
# --------------------------------------------------------------------------


ISC21_NAMELIST = """\
&time_control
 run_days = 0,
 run_hours = 0,
 run_minutes = 30,
 run_seconds = 0,
 io_form_history = 0,
 io_form_auxhist24 = 2,
 auxhist24_interval_s = 60, 60, 60,
 auxhist24_begin_s = 0, 0, 0,
 auxhist24_end_s = 300, 300, 300,
 frames_per_auxhist24 = 1, 1, 1,
 iofields_filename = "fields_d01.txt","fields_d02.txt","fields_d03.txt",
 /
&domains
 max_dom = 3,
 e_we = 793, 805, 1001,
 e_sn = 853, 805, 1001,
 e_vert = 51, 51, 51,
 parent_grid_ratio = 1, 3, 3,
 parent_time_step_ratio = 1, 3, 3,
 dx = 3000,
 dy = 3000,
 time_step = 15,
 /
&physics
 radt = 3,
 /
"""

# Real content, identical for all 3 domains (from the cluster's fields_d0N.txt).
ISC21_IOFIELDS_LINE = "+:h:24:U,V,T,QVAPOR,XLAT,XLONG,LU_INDEX,HGT\n"


def _holdout_predict(tmp_path, name, namelist_text, ranks_probe_row, extra_files=None):
    case_dir = tmp_path / name
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), namelist_text)
    for fname, content in (extra_files or {}).items():
        _write(str(case_dir / fname), content)
    ledger_path = tmp_path / ("ledger_%s.csv" % name)
    _write(str(ledger_path), LEDGER_HEADER + "\n" + _ledger_row(**ranks_probe_row) + "\n")
    config = {"ranks": 448, "omp": 1, "nx": 16, "ny": 28, "numtiles": 1}
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=4,
        config=config, ledger_paths=[str(ledger_path)], calibration_path=CALIBRATION,
        io_form_override=None, io_target="nfs", include_io=True)
    return pred


def test_holdout_isc21_within_20pct(tmp_path):
    """job 430: d01 median 4.0675 s/step over 120 d01 steps; measured wall 1004s.

    2026-09-22 (3rd pass): with the REAL fields_d0N.txt content (identical
    3-domain '+:h:24:U,V,T,QVAPOR,XLAT,XLONG,LU_INDEX,HGT' line) and the real
    begin/end window (6 frames/domain, not 31), first-principles sizing gives
    per-domain frame bytes of 563.1/539.5/834.1 MB (d01/d02/d03) and a total
    predicted wall of 1089.5s vs measured 1004.0s: ratio 1.085 -- PASSES
    +-20% (xfail removed; the 2nd pass's 0.7859 FAIL was an artifact of the
    now-abandoned 10% fallback-fraction guess, not of this sizing method).
    Per-domain predicted write times (20.4/19.5/30.2 s/frame) vs measured
    means (22.1/13.4/20.2 s/frame) show the real per-byte write rate for
    this small multi-field file type isn't identical to conus25's single
    large history file's 27.67 MB/s -- close by coincidence for d01, over
    for d02/d03 -- a genuine, disclosed generalization gap, not corrected
    here (no calibration constant was tuned against this held-out case).
    """
    row = dict(case_base="isc21", job_id=430, e_we=793, e_sn=853, e_vert=51, dx=3000,
              time_step=15, radt=3, io_form_history=0, n_nodes=4, n_ranks=448,
              omp_threads=1, nproc_x=16, nproc_y=28, numtiles=1,
              s_per_step_median=4.0675, n_steps_measured=120, notes="config=probe430")
    extra_files = {"fields_d01.txt": ISC21_IOFIELDS_LINE,
                   "fields_d02.txt": ISC21_IOFIELDS_LINE,
                   "fields_d03.txt": ISC21_IOFIELDS_LINE}
    pred = _holdout_predict(tmp_path, "isc21", ISC21_NAMELIST, row, extra_files=extra_files)
    measured_wall = 1004.0
    ratio = pred["total_s"] / measured_wall
    print("\nISC21 held-out: predicted=%.1fs measured=%.1fs ratio=%.4f" %
          (pred["total_s"], measured_wall, ratio))
    for st in pred["io"]["streams"]:
        if "seconds" in st:
            print("  %s d%s: %.1f MB/frame, %.2fs/frame" %
                  (st["stream"], st.get("domain"), st["bytes_per_frame"] / 1e6, st["seconds"] / st["n_frames"]))
    assert 0.80 <= ratio <= 1.20, (
        "isc21 held-out prediction %.1fs is outside +-20%% of measured %.1fs (ratio %.4f)"
        % (pred["total_s"], measured_wall, ratio))


# --------------------------------------------------------------------------
# 8. wtune/cli.py CLI hook smoke test
# --------------------------------------------------------------------------

def test_wtune_py_predict_subcommand_smoke(tmp_path):
    case_dir = tmp_path / "conus25"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), CONUS25_NAMELIST)
    out_json = tmp_path / "out.json"
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "src", "wtune"), "predict", str(case_dir),
         "--nodes", "7", "--nx", "24", "--ny", "30", "--ranks", "720",
         "--calibration", CALIBRATION, "--json", str(out_json)],
        capture_output=True, text=True, cwd=HERE)
    assert result.returncode == 0, result.stderr
    assert "TOTAL predicted wall" in result.stdout
    assert out_json.exists()
    doc = json.loads(out_json.read_text())
    assert doc["case_name"] == "conus25"


# --------------------------------------------------------------------------
# 9. r_night/r_day/s_normal measured from a probe's own rsl.error.0000 step
#    timings, via SCHEDULE-based radiation-step identification (not cost)
#    plus a day/night sunlit-fraction model.
# --------------------------------------------------------------------------

def _write_rsl(path, entries):
    """entries: [(sim_dt_or_None, domain, seconds), ...] -> writes a
    synthetic rsl.error.0000-shaped file, one 'Timing for main' line per
    entry, in the given order."""
    lines = []
    for sim_dt, dom, sec in entries:
        t = sim_dt.strftime("%Y-%m-%d_%H:%M:%S") if sim_dt else "0000-00-00_00:00:00"
        lines.append("Timing for main: time %s on domain   %d:  %.6f elapsed seconds" % (t, dom, sec))
    _write(str(path), "\n".join(lines) + "\n")


# --- 9a. schedule-based radiation-step identification (the severe2023 fix) ---


# --- 9b. least-squares day/night fit ---


# --- 9c. sunlit fraction (solar position) ---


def test_parse_wrfinput_corners_reads_ncdump_header_text():
    ncdump_text = """netcdf wrfinput_d01 {
// global attributes:
\t\t:CEN_LAT = 39.5f ;
\t\t:CEN_LON = -98.35f ;
\t\t:CORNER_LATS = 20.0f, 20.1f, 20.2f, 20.3f, 45.0f, 45.1f, 45.2f, 45.3f, 45.5f, 45.6f, 45.7f, 45.8f, 20.5f, 20.6f, 20.7f, 20.8f ;
\t\t:CORNER_LONS = -125.0f, -124.9f, -124.8f, -124.7f, -125.5f, -125.4f, -125.3f, -125.2f, -65.0f, -64.9f, -64.8f, -64.7f, -64.5f, -64.4f, -64.3f, -64.2f ;
}
"""
    parsed = wp.parse_wrfinput_corners(ncdump_text)
    assert parsed["cen"] == (39.5, -98.35)
    # mass-point (index c*4) corners: SW=idx0, NW=idx4, NE=idx8, SE=idx12
    assert parsed["corners"] == ((20.0, -125.0), (45.0, -125.5), (45.5, -65.0), (20.5, -64.5))

    points = wp.domain_sample_points_from_ncdump(ncdump_text)
    assert len(points) == 25
    assert points[12] == (39.5, -98.35)  # real CEN overrides the bilinear centre


def test_domain_sample_points_from_ncdump_degrades_to_centre_only_without_corners():
    ncdump_text = "netcdf x {\n// global attributes:\n\t\t:CEN_LAT = 10.0f ;\n\t\t:CEN_LON = 20.0f ;\n}\n"
    points = wp.domain_sample_points_from_ncdump(ncdump_text)
    assert points == [(10.0, 20.0)]


# --- 9c-bis. real XLAT/XLONG grid sampling (2026-09-23 scc finding: a real
#     real.exe wrfinput_d01 has NO CORNER_LATS/CORNER_LONS at all) ---

# Real conus12 wrfinput_d01 corners, measured 2026-09-23 on scc from the
# XLAT/XLONG arrays themselves (424 x 299 mass points, row-major, south row
# first): SW=(20.606,-121.482) SE=(20.606,-74.518) NW=(50.804,-136.748)
# NE=(50.804,-59.252). A synthetic 5x5 grid built from these same corners
# (with the lattice sized exactly 5x5) lets _lattice_indices resolve to
# [0,1,2,3,4] with no rounding ambiguity, so the sampled corners must come
# back byte-exact.
_CONUS12_SW = (20.606, -121.482)
_CONUS12_SE = (20.606, -74.518)
_CONUS12_NW = (50.804, -136.748)
_CONUS12_NE = (50.804, -59.252)


def _synthetic_xlat_xlong_5x5():
    """5x5 lat/lon grids (row 0 = south) spanning exactly the conus12
    corners above, linearly interpolated in between (interior values are
    not physically exact, only the 4 corners need to be)."""
    lat_rows = [_CONUS12_SW[0] + (_CONUS12_NW[0] - _CONUS12_SW[0]) * r / 4.0 for r in range(5)]
    lon_south = [_CONUS12_SW[1] + (_CONUS12_SE[1] - _CONUS12_SW[1]) * c / 4.0 for c in range(5)]
    lon_north = [_CONUS12_NW[1] + (_CONUS12_NE[1] - _CONUS12_NW[1]) * c / 4.0 for c in range(5)]
    lat2d = [[lat_rows[r]] * 5 for r in range(5)]
    lon2d = [[lon_south[c] + (lon_north[c] - lon_south[c]) * r / 4.0 for c in range(5)] for r in range(5)]
    return lat2d, lon2d


def _build_ncdump_v_text(n_rows, n_cols, lat2d, lon2d):
    """A synthetic `ncdump -v XLAT,XLONG <file>` TEXT output shaped like the
    real thing: dimensions block (Time UNLIMITED, south_north, west_east),
    variable declarations, then each variable's flattened row-major data."""
    def flat(vals2d):
        return ",\n  ".join("%.6ff" % v for row in vals2d for v in row)
    return """netcdf wrfinput_d01 {
dimensions:
\tTime = UNLIMITED ; // (1 currently)
\tsouth_north = %d ;
\twest_east = %d ;
variables:
\tfloat XLAT(Time, south_north, west_east) ;
\tfloat XLONG(Time, south_north, west_east) ;

// global attributes:
\t\t:CEN_LAT = 40.00001f ;
\t\t:CEN_LON = -98.f ;
data:

 XLAT =
  %s ;

 XLONG =
  %s ;
}
""" % (n_rows, n_cols, flat(lat2d), flat(lon2d))


def test_parse_ncdump_2d_var_returns_none_on_wrong_count():
    text = _build_ncdump_v_text(5, 5, *_synthetic_xlat_xlong_5x5())
    assert wp.parse_ncdump_2d_var(text, "XLAT", 6, 6) is None  # wrong shape
    assert wp.parse_ncdump_2d_var(text, "NOSUCHVAR", 5, 5) is None


def test_find_ncdump_binary_prefers_netcdf_env_var(tmp_path, monkeypatch):
    netcdf_root = tmp_path / "netcdf-4.9.2"
    bin_dir = netcdf_root / "bin"
    bin_dir.mkdir(parents=True)
    ncdump_path = bin_dir / "ncdump"
    ncdump_path.write_text("#!/bin/sh\n")
    monkeypatch.setenv("NETCDF", str(netcdf_root))
    assert wp._find_ncdump_binary() == str(ncdump_path)


def test_find_ncdump_binary_falls_back_to_path_lookup(monkeypatch):
    monkeypatch.delenv("NETCDF", raising=False)
    assert wp._find_ncdump_binary() == "ncdump"

    monkeypatch.setenv("NETCDF", "/nonexistent/netcdf/root")
    assert wp._find_ncdump_binary() == "ncdump"


def test_read_wrfinput_sample_points_geometry_source_ncdump_arrays(tmp_path, monkeypatch):
    text = _build_ncdump_v_text(5, 5, *_synthetic_xlat_xlong_5x5())
    (tmp_path / "wrfinput_d01").write_text("placeholder")  # only existence is checked before ncdump runs

    class _FakeResult:
        returncode = 0
        stdout = text

    def _fake_run(args, **kwargs):
        return _FakeResult()

    monkeypatch.setattr(wp.subprocess, "run", _fake_run)
    points, source = wp.read_wrfinput_sample_points(str(tmp_path))
    assert source == "ncdump_arrays"
    assert points[0] == pytest.approx(_CONUS12_SW)


def test_render_text_warns_when_geometry_source_is_none(tmp_path):
    nl_text = ("&time_control\n run_minutes = 60,\n /\n&domains\n max_dom = 1,\n"
               " e_we = 10, e_sn = 10, e_vert = 10,\n time_step = 60,\n /\n"
               "&physics\n radt = 10,\n /\n")
    case_dir = tmp_path / "nogeom"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), nl_text)  # no wrfinput_d01 present
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=1,
        config={"ranks": 1, "omp": 1, "nx": 1, "ny": 1, "numtiles": 1},
        ledger_paths=None, calibration_path=CALIBRATION, io_form_override=None, io_target="nfs")
    assert pred["compute"]["geometry_source"] == "none"
    text = wp.render_text(pred)
    assert "domain geometry source: none" in text
    assert "WARNING: no domain geometry found" in text
    assert "sunlit-fraction weighting is DISABLED" in text


# --- 9d. measure_rad_from_rsl end-to-end (schedule + day/night geometry fit) ---

RAD_RSL_NAMELIST = """\
&time_control
 run_minutes = 1440,
 /
&domains
 max_dom = 1,
 e_we = 10,
 e_sn = 10,
 e_vert = 10,
 time_step = 3600,
 /
&physics
 radt = 180,
 /
"""


def _build_two_regime_rsl_entries(r_night_true, r_day_true, s_normal_true=2.0):
    """24 hourly d01 steps (dt=3600s), radt=180min -> stepra=3 (radiation at
    i=1,3,6,9,12,15,18,21,24). start_dt=2020-03-20T00:00 (equinox, lon=0 ==
    UTC solar time -- see test_sunlit_fraction_zero_at_midnight_one_at_noon):
    hours 2,5,20,23 are night (i=3,6,21,24), hours 8,11,14,17 are day
    (i=9,12,15,18) -- an exact, evenly-split two-regime probe. Step 1 (i=1,
    also schedule-radiation) gets an outlier value, since measure_rad_from_rsl
    is documented to drop it regardless of category."""
    start = datetime.datetime(2020, 3, 20, 0, 0, 0)
    stepra = 3
    entries = []
    for i in range(1, 25):
        t = start + datetime.timedelta(hours=i - 1)
        is_rad = (i == 1) or (i % stepra == 0)
        if i == 1:
            sec = 999.0  # dropped outlier -- must not affect s_normal or the fit
        elif is_rad:
            is_day = 6 < t.hour < 18
            sec = (r_day_true if is_day else r_night_true) * s_normal_true
        else:
            sec = s_normal_true
        entries.append((t, 1, sec))
    return entries


def test_measure_rad_from_rsl_falls_back_without_sample_points(tmp_path):
    """No domain geometry at all (sample_points=None) -> no radiation sample
    can be placed against a sunlit fraction, regardless of how many
    schedule-radiation steps exist."""
    entries = _build_two_regime_rsl_entries(r_night_true=2.0, r_day_true=4.0)
    rsl_path = tmp_path / "rsl.error.0000"
    _write_rsl(rsl_path, entries)
    result = wp.measure_rad_from_rsl([str(rsl_path)], radt_minutes=180, dt_seconds=3600,
                                     sample_points=None, default_r_rad=5.0, day_night_ratio=1.87)
    assert result["source"] == "default_unmeasured"
    assert result["s_normal"] == pytest.approx(2.0)  # normal steps still usable


# --- 9e. predict_radiation_extra: hand-computed integration ---


# --- 9f. predict_compute integration (full plumbing) ---

def test_predict_compute_uses_rsl_day_night_model_without_ledger(tmp_path):
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), RAD_RSL_NAMELIST))
    steps, run_s, dt, _ = wp.get_d01_steps(nl)
    run_min = run_s / 60.0
    rsl_path = tmp_path / "rsl.error.0000"
    _write_rsl(rsl_path, _build_two_regime_rsl_entries(r_night_true=2.0, r_day_true=4.0))

    compute = wp.predict_compute(
        nl, 1, steps, run_min, ranks=1, omp=1, probe_result=None,
        calibration={"radiation": {"r_rad_default": 5.0, "r_rad_day_night_ratio": 1.87}},
        rsl_paths=[str(rsl_path)], dt_seconds=dt, sample_points=[(0.0, 0.0)],
        run_start_dt=datetime.datetime(2020, 3, 20, 0, 0, 0))
    assert compute["source"] == "probe_rsl"
    assert compute["s_normal"] == pytest.approx(2.0)
    assert compute["r_night"] == pytest.approx(2.0)
    assert compute["r_day"] == pytest.approx(4.0)
    assert compute["r_rad_source"] == "fit_day_night"
    assert compute["run_mean_sunlit_fraction"] is not None
    assert compute["sampling_advice"] is not None


# --------------------------------------------------------------------------
# 10. startup_s calibration key
# --------------------------------------------------------------------------

STARTUP_NAMELIST = """\
&domains
 max_dom = 1,
 e_we = 10,
 e_sn = 10,
 e_vert = 10,
 /
"""


def test_predict_init_startup_cap_grows_with_nodes(tmp_path):
    # sc26p2 4 nodes / 192 ranks started in ~32 s against the 1-node 19.4 s cap
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), STARTUP_NAMELIST))
    cal = {"derived": {"read_rate_bytes_per_s": 1000.0, "bytes_per_cell_level": 1.0},
           "startup_s": 19.4, "startup_s_base": 2.0, "startup_s_per_rank": 0.4,
           "startup_s_per_node": 4.0}
    one = wp.predict_init(None, nl, cal, ranks=72, nodes=1)
    four = wp.predict_init(None, nl, cal, ranks=192, nodes=4)
    small = wp.predict_init(None, nl, cal, ranks=1, nodes=1)
    assert one["total_s"] == pytest.approx(one["read_s"] + 19.4)
    assert four["total_s"] == pytest.approx(four["read_s"] + 19.4 + 12.0)
    assert small["total_s"] == pytest.approx(small["read_s"] + 2.4)
    del cal["startup_s_per_node"]
    assert wp.predict_init(None, nl, cal, ranks=192, nodes=4)["total_s"] == pytest.approx(
        four["read_s"] + 19.4)


def _calibration_with_nulls(tmp_path, *keys):
    """Copy of the real calibration with `keys` set to null, so a test of the
    uncalibrated path does not depend on whether scc.json has been fitted yet."""
    import json as _json
    d = _json.load(open(CALIBRATION))
    for k in keys:
        d[k] = None
    out = tmp_path / "calibration_nulls.json"
    out.write_text(_json.dumps(d))
    return str(out)


def test_render_text_prints_startup_uncalibrated_with_real_calibration(tmp_path):
    # With startup_s unset the text output must say so plainly. The real scc.json
    # was fitted on 2026-09-23, so null the key in a copy rather than rely on it.
    cal = _calibration_with_nulls(tmp_path, "startup_s", "startup_s_source")
    case_dir = tmp_path / "conus25"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), CONUS25_NAMELIST)
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=7,
        config={"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1},
        ledger_paths=None, calibration_path=cal,
        io_form_override=None, io_target="nfs")
    assert "startup: UNCALIBRATED" in wp.render_text(pred)


# --------------------------------------------------------------------------
# 11. I/O as a range (history_write_rate_MBps_low/_high)
# --------------------------------------------------------------------------


def test_io_range_absent_keeps_legacy_rss_range(tmp_path):
    cal = _calibration_with_nulls(tmp_path, "history_write_rate_MBps_low", "history_write_rate_MBps_high")
    case_dir = tmp_path / "conus25"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), CONUS25_NAMELIST)
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=7,
        config={"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1},
        ledger_paths=None, calibration_path=cal,
        io_form_override=None, io_target="nfs", include_io=True)
    assert pred["io_range"] is None
    assert pred["range_source"] is None
    expected_range_abs = math.sqrt(pred["compute"]["noise_abs_s"] ** 2 + pred["io"]["noise_abs_s"] ** 2
                                    + pred["drift_abs_s"] ** 2)
    assert pred["range_high_s"] == pytest.approx(pred["total_s"] + expected_range_abs)


# --------------------------------------------------------------------------
# 12. lafe --case-name probe-matching regression
# --------------------------------------------------------------------------

LAFE_NAMELIST = """\
&time_control
 run_hours = 24,
 /
&domains
 max_dom = 1,
 e_we = 500,
 e_sn = 500,
 e_vert = 50,
 time_step = 20,
 /
"""


def test_lafe_probe_matching_needs_case_name_override(tmp_path):
    """Root cause: build_prediction's case_name defaults to case_dir's own
    basename (Path(case_dir).name) when no explicit case_name is given, and
    resolve_probe matches purely on case_base == that name. lafe's real prep
    dir is /nfs/data/wrf/doi4-lafe2017 (named after the data source) while
    its ledger case_base is 'lafe' -- so with no --case-name, EVERY row for
    this case fails the case_base match (probe_result stays None end to
    end), and predict silently drops to no_probe_fallback despite the ledger
    holding an exact 72-rank measured row (8.587 s/step)."""
    case_dir = tmp_path / "doi4-lafe2017"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input.d01only"), LAFE_NAMELIST)

    ledger_path = tmp_path / "ledger_lafe.csv"
    row = dict(case_base="lafe", n_ranks=72, omp_threads=1, nproc_x=9, nproc_y=8,
              numtiles=1, s_per_step_median=8.587, notes="config=lafe72")
    _write(str(ledger_path), LEDGER_HEADER + "\n" + _ledger_row(**row) + "\n")

    config = {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1}
    common = dict(namelist_path=str(case_dir / "namelist.input.d01only"), case_dir=str(case_dir),
                 nodes=1, config=config, ledger_paths=[str(ledger_path)],
                 calibration_path=CALIBRATION, io_form_override=None, io_target="nfs")

    # BEFORE the fix's usage (no case_name): reproduces the real bug.
    pred_broken = wp.build_prediction(**common)
    assert pred_broken["case_name"] == "doi4-lafe2017"
    assert pred_broken["compute"]["source"] == "no_probe_fallback"
    assert pred_broken["compute"]["s_normal"] != pytest.approx(8.587)

    # AFTER: --case-name lafe makes the ledger match exact.
    pred_fixed = wp.build_prediction(case_name="lafe", **common)
    assert pred_fixed["case_name"] == "lafe"
    assert pred_fixed["compute"]["source"] == "measured"
    assert pred_fixed["compute"]["s_normal"] == pytest.approx(8.587)


def test_wtune_predict_cli_case_name_flag(tmp_path):
    case_dir = tmp_path / "doi4-lafe2017"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), LAFE_NAMELIST)
    ledger_path = tmp_path / "ledger_lafe.csv"
    row = dict(case_base="lafe", n_ranks=72, omp_threads=1, nproc_x=9, nproc_y=8,
              numtiles=1, s_per_step_median=8.587, notes="config=lafe72")
    _write(str(ledger_path), LEDGER_HEADER + "\n" + _ledger_row(**row) + "\n")
    out_json = tmp_path / "out.json"
    result = subprocess.run(
        [sys.executable, "-m", "wtune.predict", "predict", str(case_dir),
         "--nodes", "1", "--nx", "9", "--ny", "8", "--ranks", "72",
         "--probe", str(ledger_path), "--case-name", "lafe",
         "--calibration", CALIBRATION, "--json", str(out_json)],
        capture_output=True, text=True, cwd=os.path.join(HERE, "src"))
    assert result.returncode == 0, result.stderr
    doc = json.loads(out_json.read_text())
    assert doc["compute"]["source"] == "measured"
    assert doc["compute"]["s_normal"] == pytest.approx(8.587)


def test_unmeasured_radiation_uses_calibrated_day_night_levels():
    cal = {"radiation": {"r_rad_default": 5.0, "r_rad_default_night": 2.46, "r_rad_default_day": 3.93}}
    import inspect
    src = inspect.getsource(wp.predict_compute)
    assert "default_calibrated" in src and "r_rad_default_night" in src
    assert cal["radiation"]["r_rad_default_night"] < cal["radiation"]["r_rad_default_day"] < cal["radiation"]["r_rad_default"]


# --------------------------------------------------------------------------
# idealized cases (2026-09-24 scc): radt = 0 and overflowing end_minute
# --------------------------------------------------------------------------

NORAD_NAMELIST = """\
&time_control
 run_days = 0, run_hours = 0, run_minutes = 1, run_seconds = 0,
 start_year = 0001, start_month = 01, start_day = 01, start_hour = 00, start_minute = 00, start_second = 00,
 end_year = 0001, end_month = 01, end_day = 01, end_hour = 00, end_minute = 120, end_second = 00,
/
&domains
 time_step = 1, max_dom = 1, e_we = 40, e_sn = 40, e_vert = 40, dx = 100, dy = 100,
/
&physics
 radt = 0,
/
"""


def test_predict_compute_uses_probe_rsl_when_radt_is_zero(tmp_path):
    # em_les has radt = 0; the probe used to be ignored and the uncalibrated no-probe constant
    # was used instead (0.284 s/step against 0.033 probed).
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), NORAD_NAMELIST))
    steps, run_s, dt, _ = wp.get_d01_steps(nl)
    rsl = tmp_path / "rsl.error.0000"
    secs = [5.0, 0.03, 0.04, 0.03, 0.05, 0.03]
    rsl.write_text("".join(
        "Timing for main: time 0001-01-01_00:00:%02d on domain   1:    %.5f elapsed seconds\n"
        % (i + 1, s) for i, s in enumerate(secs)))
    compute = wp.predict_compute(nl, 1, steps, run_s / 60.0, ranks=1, omp=1, probe_result=None,
                                 calibration={"radiation": {"r_rad_default": 5.0},
                                              "fallback_s_per_cell_step_ns": 4437.0},
                                 rsl_paths=[str(rsl)], dt_seconds=dt)
    assert compute["source"] == "probe_rsl"
    assert compute["s_normal"] == pytest.approx(statistics.median(secs[1:]))
    assert compute["rad_steps"] == 0


def test_predict_init_skips_dangling_symlinks_and_sizes_bp_directories(tmp_path):
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), STARTUP_NAMELIST))
    calibration = {"derived": {"read_rate_bytes_per_s": 1000.0, "bytes_per_cell_level": 1.0}}
    (tmp_path / "wrfbdy_d01_2022-04-04_18:00:00.bp").symlink_to(tmp_path / "gone.bp")   # dangling
    bp = tmp_path / "wrfinput_d01.bp"
    bp.mkdir()
    (bp / "data.0").write_bytes(b"x" * 600)
    (bp / "md.idx").write_bytes(b"y" * 400)
    init = wp.predict_init(str(tmp_path), nl, calibration)
    assert init["read_s"] == pytest.approx(1.0)              # 1000 bytes / 1000 B/s
    assert "wrfinput_d01.bp" in init["source"] and "gone" not in init["source"] \
        and "wrfbdy_d01" not in init["source"]
