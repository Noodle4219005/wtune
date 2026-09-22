#!/usr/bin/env python3
"""Tests for wtune_predict.py -- the offline wall-time predictor.

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
import json
import math
import os
import subprocess
import sys
import types

import pytest

import wtune_predict as wp

HERE = os.path.dirname(os.path.abspath(__file__))
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

def test_run_length_and_steps_basic(tmp_path):
    nl_path = _write(str(tmp_path / "namelist.input"), """\
&time_control
 run_hours = 2,
 /
&domains
 time_step = 60,
 /
""")
    nl = wp.parse_namelist(nl_path)
    steps, run_s, dt, note = wp.get_d01_steps(nl)
    assert steps == 120
    assert run_s == 7200.0
    assert dt == 60.0
    assert note is None


def test_run_length_days_hours_minutes_seconds_combine(tmp_path):
    nl_path = _write(str(tmp_path / "namelist.input"), """\
&time_control
 run_days = 1,
 run_hours = 2,
 run_minutes = 3,
 run_seconds = 4,
 /
&domains
 time_step = 10,
 /
""")
    nl = wp.parse_namelist(nl_path)
    run_s = wp.get_run_seconds(nl)
    assert run_s == 1 * 86400 + 2 * 3600 + 3 * 60 + 4


def test_multiple_assignments_on_one_line(tmp_path):
    """sc26p2 bug (2026-09-22): a real namelist packs several 'key = value,'
    assignments onto ONE line, e.g.
    ' run_days = 0, run_hours = 48, run_minutes = 0, run_seconds = 0,'
    and similarly for time_step/time_step_fract_num/time_step_fract_den. The
    old reader took the whole rest of the line as run_days's value list, so
    run_hours/run_minutes/run_seconds were never set at all (silently 0) and
    the run length came out zero -> PredictError. The reader must split on
    every top-level 'identifier=' on the line, not just the first."""
    nl_path = _write(str(tmp_path / "namelist.input"), """\
&time_control
 run_days = 0, run_hours = 48, run_minutes = 0, run_seconds = 0,
 history_interval = 60,
 /
&domains
 max_dom = 1,
 e_we = 100,
 e_sn = 100,
 e_vert = 30,
 time_step = 15, time_step_fract_num = 1, time_step_fract_den = 2,
 /
""")
    nl = wp.parse_namelist(nl_path)
    assert nl["time_control"]["run_days"] == [0]
    assert nl["time_control"]["run_hours"] == [48]
    assert nl["time_control"]["run_minutes"] == [0]
    assert nl["time_control"]["run_seconds"] == [0]
    assert nl["domains"]["time_step"] == [15]
    assert nl["domains"]["time_step_fract_num"] == [1]
    assert nl["domains"]["time_step_fract_den"] == [2]

    run_s = wp.get_run_seconds(nl)
    assert run_s == 48 * 3600
    steps, run_s2, dt, note = wp.get_d01_steps(nl)
    assert dt == 15.5
    assert steps == round(48 * 3600 / 15.5)


def test_multiple_assignments_with_quoted_string_value(tmp_path):
    """A quoted string value must not confuse the per-line assignment
    splitter (no '=' or bare identifier inside the quotes is mistaken for
    the next key)."""
    nl_path = _write(str(tmp_path / "namelist.input"), """\
&time_control
 title = "run = 48h test", run_days = 2, run_hours = 0,
 /
""")
    nl = wp.parse_namelist(nl_path)
    assert nl["time_control"]["title"] == ["run = 48h test"]
    assert nl["time_control"]["run_days"] == [2]
    assert nl["time_control"]["run_hours"] == [0]


def test_timestep_fraction_and_non_exact_multiple_flagged(tmp_path):
    nl_path = _write(str(tmp_path / "namelist.input"), """\
&time_control
 run_seconds = 155,
 /
&domains
 time_step = 15,
 time_step_fract_num = 1,
 time_step_fract_den = 2,
 /
""")
    nl = wp.parse_namelist(nl_path)
    steps, run_s, dt, note = wp.get_d01_steps(nl)
    assert dt == 15.5
    assert steps == 10  # 155/15.5 == 10 exactly
    assert note is None

    nl_path2 = _write(str(tmp_path / "namelist2.input"), """\
&time_control
 run_seconds = 100,
 /
&domains
 time_step = 64,
 /
""")
    nl2 = wp.parse_namelist(nl_path2)
    steps2, _, _, note2 = wp.get_d01_steps(nl2)
    assert steps2 == round(100 / 64)
    assert note2 is not None and "not an exact multiple" in note2


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


def test_history_frame_count_includes_t0(tmp_path):
    nl = _parse(tmp_path, """\
&time_control
 history_interval = 30,
 io_form_history = 2,
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
    hist = [s for s in io["streams"] if s.get("stream") == "history"][0]
    # floor(90/30) intervals + 1 initial frame at t0 = 4 frames (t=0,30,60,90)
    assert hist["n_frames"] == 4
    assert hist["bytes_per_frame"] == pytest.approx(100.0)  # 10*10*1 cells * bpcl=1.0
    assert hist["seconds"] == pytest.approx(400.0)  # 4 frames * 100 bytes / rate=1.0
    assert io["total_s"] == pytest.approx(400.0)


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

def test_30h_run_renders_5_segments():
    compute = {"steps": 7200, "s_normal": 1.0, "rad_extra_s": 0.0}
    segments = wp.segment_compute(compute, run_hours=30.0, dt=15.0)
    assert segments is not None
    assert len(segments) == 5
    assert sum(seg["steps"] for seg in segments) == 7200
    assert all(seg["steps"] == 1440 for seg in segments)
    assert segments[0]["sim_time"] == "0.0-6.0h"
    assert segments[-1]["sim_time"] == "24.0-30.0h"


def test_short_run_not_segmented():
    compute = {"steps": 100, "s_normal": 1.0, "rad_extra_s": 0.0}
    assert wp.segment_compute(compute, run_hours=6.0, dt=15.0) is None


# --------------------------------------------------------------------------
# 4. parameter table / launch line
# --------------------------------------------------------------------------

def test_param_table_launch_line_and_namelist_lines():
    config = {"ranks": 448, "omp": 2, "nx": 16, "ny": 14, "numtiles": 12, "dist": "cyclic"}
    p = wp.build_param_table(config, nodes=4, io_form=2, io_target="nfs")
    assert p["ranks_per_node"] == 112
    assert p["warning"] is None
    assert p["launch_line"] == "mpirun -np 448 --map-by ppr:112:node:PE=2 --bind-to core --rank-by node"
    assert p["env"]["OMP_NUM_THREADS"] == "2"
    assert p["env"]["OMP_STACKSIZE"] == "1G"
    assert p["namelist_lines"] == ["nproc_x = 16,", "nproc_y = 14,", "numtiles = 12,"]


def test_param_table_launch_line_matches_scc_sites_site_launch():
    """campaign/sites/scc.sh site_launch(): --map-by ppr:<tpn>:node:PE=<omp> --bind-to
    core, PLUS --rank-by node for dist=cyclic only (block is OMPI5's own default
    rank-by, so site_launch adds nothing extra for it). build_param_table's
    launch_line must match exactly, taking dist from config (not hardcoded)."""
    config_block = {"ranks": 448, "omp": 1, "nx": 16, "ny": 28, "numtiles": 1, "dist": "block"}
    p_block = wp.build_param_table(config_block, nodes=4, io_form=2, io_target="nfs")
    assert p_block["dist"] == "block"
    assert p_block["launch_line"] == "mpirun -np 448 --map-by ppr:112:node:PE=1 --bind-to core"
    assert "--rank-by" not in p_block["launch_line"]

    config_cyclic = dict(config_block, dist="cyclic")
    p_cyclic = wp.build_param_table(config_cyclic, nodes=4, io_form=2, io_target="nfs")
    assert p_cyclic["dist"] == "cyclic"
    assert p_cyclic["launch_line"] == "mpirun -np 448 --map-by ppr:112:node:PE=1 --bind-to core --rank-by node"

    # config with no 'dist' key at all must NOT guess cyclic (2026-09-23 scc blind-test
    # bug: this used to silently default to "cyclic" and print --rank-by node for a
    # config whose real/measured dist was "block" -- a launch line for a DIFFERENT run).
    config_default = {"ranks": 448, "omp": 1, "nx": 16, "ny": 28, "numtiles": 1}
    p_default = wp.build_param_table(config_default, nodes=4, io_form=2, io_target="nfs")
    assert p_default["dist"] == "UNKNOWN"
    assert p_default["launch_line"] == p_block["launch_line"]  # no --rank-by when unknown
    assert p_default["launch_line"] != p_cyclic["launch_line"]
    assert "dist not provided" in p_default["warning"]


def test_param_table_warns_on_uneven_ranks_per_node():
    config = {"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1, "dist": "block"}
    p = wp.build_param_table(config, nodes=7, io_form=2, io_target="nfs")
    assert p["warning"] is not None
    assert "not evenly divisible" in p["warning"]


def test_param_table_numtiles_unknown_when_not_provided():
    config = {"ranks": 448, "omp": 1, "nx": 16, "ny": 28, "dist": "block"}  # no numtiles
    p = wp.build_param_table(config, nodes=4, io_form=2, io_target="nfs")
    assert p["numtiles"] == "UNKNOWN"
    assert p["namelist_lines"][2] == "numtiles = UNKNOWN,"
    assert "numtiles not provided" in p["warning"]
    # launch_line itself never mentions numtiles, so it's still complete/usable
    assert "UNKNOWN" not in p["launch_line"]


def test_param_table_io_form_unknown_when_none():
    config = {"ranks": 448, "omp": 1, "nx": 16, "ny": 28, "numtiles": 1, "dist": "block"}
    p = wp.build_param_table(config, nodes=4, io_form=None, io_target="nfs")
    assert p["io_form"] == "UNKNOWN"


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

def test_resolve_probe_exact_match():
    rows = [{"case_base": "conus25", "n_ranks": "720", "omp_threads": "1",
             "nproc_x": "24", "nproc_y": "30", "numtiles": "1",
             "s_per_step_median": "0.5555"}]
    config = {"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1}
    result = wp.resolve_probe(rows, "conus25", config)
    assert result["source"] == "measured"
    assert result["s_normal"] == pytest.approx(0.5555)


def test_resolve_probe_extrapolated_scales_by_ranks():
    rows = [{"case_base": "conus25", "n_ranks": "360", "omp_threads": "1",
             "nproc_x": "12", "nproc_y": "30", "numtiles": "1",
             "s_per_step_median": "1.0"}]
    config = {"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1}
    result = wp.resolve_probe(rows, "conus25", config)
    assert result["source"] == "extrapolated"
    # ideal scaling: double the ranks -> half the time
    assert result["s_normal"] == pytest.approx(0.5)


def test_resolve_probe_no_data_for_case_returns_none():
    rows = [{"case_base": "other_case", "n_ranks": "720", "s_per_step_median": "0.5"}]
    config = {"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1}
    assert wp.resolve_probe(rows, "conus25", config) is None


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


def test_conus25_self_consistency_within_5pct(tmp_path):
    case_dir = tmp_path / "conus25"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), CONUS25_NAMELIST)

    ledger_path = tmp_path / "ledger_conus25.csv"
    row = _ledger_row(case_base="conus25", job_id=427, e_we=1501, e_sn=1201, e_vert=50,
                      dx=2500, dy=2500, time_step=15, radt=10, history_interval=60,
                      io_form_history=2, n_nodes=7, n_ranks=720, omp_threads=1,
                      nproc_x=24, nproc_y=30, numtiles=1, distribution="cyclic",
                      s_per_step_median=0.5555, s_per_step_mean=1.3136,
                      n_steps_measured=1440, notes="config=cal427")
    _write(str(ledger_path), LEDGER_HEADER + "\n" + row + "\n")

    config = {"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1}
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=7,
        config=config, ledger_paths=[str(ledger_path)], calibration_path=CALIBRATION,
        io_form_override=None, io_target="nfs")

    assert pred["compute"]["source"] == "measured"
    measured_wall = 2090.0
    ratio = pred["total_s"] / measured_wall
    print("\nCONUS25 self-consistency: predicted=%.1fs measured=%.1fs ratio=%.4f" %
          (pred["total_s"], measured_wall, ratio))
    assert abs(ratio - 1.0) <= 0.05, (
        "conus25 self-consistency must reproduce 2090s within +-5%%, got %.1fs (ratio %.4f)"
        % (pred["total_s"], ratio))


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
#        modeled by wtune_predict.py (noted, not simulated).
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
#        wtune_predict.py's built-in table, nothing guessed.
# --------------------------------------------------------------------------

MARIA1KM_NAMELIST = """\
&time_control
 run_days = 0,
 run_hours = 3,
 run_minutes = 0,
 run_seconds = 0,
 history_interval = 0, 60, 60,
 io_form_history = 2,
 restart_interval = 120,
 frames_per_outfile = 0, 1,
 /
&domains
 max_dom = 2,
 e_we = 900, 1661,
 e_sn = 900, 1591,
 e_vert = 50, 50,
 dx = 5000,
 dy = 5000,
 time_step = 25,
 parent_grid_ratio = 1, 5,
 parent_time_step_ratio = 1, 5,
 /
&physics
 radt = 12, 12,
 /
"""

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
        io_form_override=None, io_target="nfs")
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


def test_holdout_maria1km_within_20pct(tmp_path):
    """job 429: d01 median 4.1615 s/step over 432 d01 steps (d01 wall already
    includes the 2160 nested d02 substeps at parent_time_step_ratio=5);
    measured wall 3499s.

    ROOT-CAUSED AND FIXED 2026-09-22 (was a pre-registered FAIL in the 1st
    calibration pass: predicted 2361 s vs 3499 s, ratio 0.675). The 1st pass's
    conus25 calibration blended one restart write's time into the history
    write rate (17.17 MB/s) and guessed the restart was 2x a history frame's
    BYTES; both were wrong, and the maria1km fixture had also guessed d02
    history OFF and no restart (ASSUMED, before the real settings were read
    from scc rsl output). With conus25 recalibrated from ONLY its own
    measurements -- history-only write rate 27.67 MB/s (7 frames / 705.0s)
    and a MEASURED restart_to_history_time_ratio=4.28 (431.2s restart /
    100.7s mean history-frame write) -- and this fixture now using maria1km's
    REAL output settings (d02-only hourly history, restart_interval=120 for
    BOTH domains), this model predicts 3486.0 s vs measured 3499.0 s, ratio
    0.9963: within +-20% (and even +-5%). No constant was tuned on maria1km
    itself -- every calibration number above comes from conus25 (job 427) alone.
    """
    row = dict(case_base="maria1km", job_id=429, e_we=900, e_sn=900, e_vert=50, dx=5000,
              time_step=25, radt=12, io_form_history=2, n_nodes=4, n_ranks=448,
              omp_threads=1, nproc_x=16, nproc_y=28, numtiles=1,
              s_per_step_median=4.1615, n_steps_measured=432, notes="config=probe429")
    pred = _holdout_predict(tmp_path, "maria1km", MARIA1KM_NAMELIST, row)
    measured_wall = 3499.0
    ratio = pred["total_s"] / measured_wall
    print("\nMARIA1KM held-out: predicted=%.1fs measured=%.1fs ratio=%.4f" %
          (pred["total_s"], measured_wall, ratio))
    assert 0.80 <= ratio <= 1.20, (
        "maria1km held-out prediction %.1fs is outside +-20%% of measured %.1fs (ratio %.4f) "
        "-- see docstring: this is a genuine model miss, not tuned away"
        % (pred["total_s"], measured_wall, ratio))


# --------------------------------------------------------------------------
# 8. wtune.py CLI hook smoke test
# --------------------------------------------------------------------------

def test_wtune_py_predict_subcommand_smoke(tmp_path):
    case_dir = tmp_path / "conus25"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), CONUS25_NAMELIST)
    out_json = tmp_path / "out.json"
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "wtune.py"), "predict", str(case_dir),
         "--nodes", "7", "--nx", "24", "--ny", "30", "--ranks", "720",
         "--calibration", CALIBRATION, "--json", str(out_json)],
        capture_output=True, text=True, cwd=HERE)
    assert result.returncode == 0, result.stderr
    assert "TOTAL predicted wall" in result.stdout
    assert out_json.exists()
    doc = json.loads(out_json.read_text())
    assert doc["case_name"] == "conus25"
    assert doc["steps"] == 1440
