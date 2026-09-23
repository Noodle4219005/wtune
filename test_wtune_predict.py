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
import datetime
import json
import math
import os
import statistics
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


# ---------------------------------------------------------------------------
# get_run_seconds: date-span fallback (job 587/588 postmortem shapes, transcribed verbatim from
# the real cluster namelists -- leading-zero months, per-domain comma lists, irregular spacing
# all preserved, since those are exactly what break a naive parser). See
# tools/test_temporal_validation.py's NML_LAFE_SHAPE/NML_ISC21_SHAPE for the same two cases
# exercised through read_run_length_hours, test_wtune.py for extract_case, and
# tools/test_metdata.py for the WPS GRIB-time count.
# ---------------------------------------------------------------------------
LAFE_TIME_CONTROL = """\
 &time_control
 run_days                            = 0,
 run_hours                           = 0,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 2017, 2017, 2017,
 start_month                         = 08,   08,   08,
 start_day                           = 23,   23,   23,
 start_hour                          = 06,   06,   06,
 start_minute                        = 00,   00,   00,
 start_second                        = 00,   00,   00,
 end_year                            = 2017, 2017, 2017,
 end_month                           = 08,   08,   08,
 end_day                             = 24,   24,   24,
 end_hour                            = 06,   06,   06,
 end_minute                          = 00,   00,   00,
 end_second                          = 00,   00,   00,
 interval_seconds                    = 21600,
 /
"""

ISC21_TIME_CONTROL = """\
&time_control
   run_days = 0,
   run_hours = 0,
   run_minutes = 30,
   run_seconds = 00,
   start_year =  2016, 2016, 2016
   start_month = 08,   08,   08,
   start_day =   29,   29,   29,
   start_hour =  09,   09,   09,
   end_year =    2016, 2016, 2016,
   end_month =   08,   08,   08,
   end_day =     30,   30,   30,
   end_hour =    00,   00,   00,
   interval_seconds = 10800,
 /
"""


def test_get_run_seconds_lafe_shape_uses_date_span(tmp_path):
    """Every run_* key is 0 -> falls back to start_date (2017-08-23 06:00) -> end_date
    (2017-08-24 06:00) = exactly 86400s (24h), not 0 and not a PredictError."""
    nl_path = _write(str(tmp_path / "namelist.input"), LAFE_TIME_CONTROL)
    nl = wp.parse_namelist(nl_path)
    assert wp.get_run_seconds(nl) == 86400.0


def test_get_run_seconds_isc21_shape_run_minutes_wins_over_date_span(tmp_path):
    """run_minutes=30 alone is nonzero -> 1800s exactly, even though start->end spans a
    DIFFERENT 15h (2016-08-29 09:00 -> 2016-08-30 00:00) that must NOT win."""
    nl_path = _write(str(tmp_path / "namelist.input"), ISC21_TIME_CONTROL)
    nl = wp.parse_namelist(nl_path)
    assert wp.get_run_seconds(nl) == 1800.0


def test_get_run_seconds_raises_predicterror_when_neither_source_available():
    nl = {"time_control": {"run_days": [0], "run_hours": [0], "run_minutes": [0],
                            "run_seconds": [0]}}
    with pytest.raises(wp.PredictError):
        wp.get_run_seconds(nl)


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

def test_classify_steps_detected_finds_radiation_steps_below_old_cost_threshold():
    """The OLD heuristic ('step time > 1.8x the pooled median') is exactly
    what missed severe2023's night radiation steps (measured cost ~1.05x a
    normal step). Build a synthetic wall-time series where radiation steps
    (schedule period stepra=6: i==1 or i%6==0) cost only 1.3x a normal step
    -- well under 1.8x -- and verify classify_steps_detected still finds
    them all, purely from the schedule."""
    stepra = 6
    n = 24
    normal_s, night_rad_s = 2.0, 2.6  # ratio 1.3, BELOW the old 1.8x threshold
    assert night_rad_s / normal_s < 1.8
    wall = [(night_rad_s if ((i == 1) or (i % stepra == 0)) else normal_s) for i in range(1, n + 1)]
    flags = wp.classify_steps_detected(wall, stepra)
    expected = [(i == 1) or (i % stepra == 0) for i in range(1, n + 1)]
    assert flags == expected
    assert sum(flags) == 5  # i = 1, 6, 12, 18, 24


def test_steps_per_radiation_call_rounds_fortran_nint():
    # conus12: radt=15min, dt=72s -> 15*60/72=12.5; Fortran NINT rounds
    # halves away from zero (13), NOT Python round()'s round-half-to-even (12).
    assert wp.steps_per_radiation_call(15, 72) == 13
    assert wp.steps_per_radiation_call(9, 10) == 54  # sc26p1 worked example


# --- 9b. least-squares day/night fit ---

def test_fit_day_night_r_rad_recovers_known_levels_from_two_regime_probe():
    r_night_true, r_day_true = 2.0, 4.0
    samples = [(frac, r_night_true + (r_day_true - r_night_true) * frac)
              for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    r_night, r_day, source = wp.fit_day_night_r_rad(samples, day_night_ratio=1.87)
    assert source == "fit_day_night"
    assert r_night == pytest.approx(r_night_true)
    assert r_day == pytest.approx(r_day_true)


def test_fit_day_night_r_rad_single_night_regime_uses_ratio():
    # sunlit fractions all near 0 (spread 0.1 < RADIATION_FIT_MIN_SUNLIT_SPREAD) --
    # a probe whose t0 fixed it to a single (night) regime, e.g. matthew's 00Z t0.
    samples = [(0.0, 1.9), (0.05, 1.95), (0.1, 2.0)]
    r_night, r_day, source = wp.fit_day_night_r_rad(samples, day_night_ratio=1.87)
    assert source == "ratio_from_night"
    mean_y = statistics.mean(s[1] for s in samples)
    assert r_night == pytest.approx(mean_y)
    assert r_day == pytest.approx(mean_y * 1.87)


def test_fit_day_night_r_rad_single_day_regime_uses_ratio():
    samples = [(0.9, 3.8), (0.95, 3.9), (1.0, 4.0)]
    r_night, r_day, source = wp.fit_day_night_r_rad(samples, day_night_ratio=1.87)
    assert source == "ratio_from_day"
    mean_y = statistics.mean(s[1] for s in samples)
    assert r_day == pytest.approx(mean_y)
    assert r_night == pytest.approx(mean_y / 1.87)


# --- 9c. sunlit fraction (solar position) ---

def test_sunlit_fraction_zero_at_midnight_one_at_noon_mid_latitude():
    """A small domain (a tight cluster of sample points) around 40N, 0E.
    lon=0 makes local solar time == UTC exactly (this test's own ASSUMPTION
    choice, matching solar_cos_zenith's own equation-of-time-free model), so
    UTC 12:00 is local solar noon and UTC 00:00 is local solar midnight.
    2020-03-20 is within a day of the equinox, so declination is ~0 and does
    not change the day/night sign at this latitude."""
    sample_points = [(40.0, 0.0), (40.1, 0.05), (39.9, -0.05), (40.05, -0.02)]
    noon = datetime.datetime(2020, 3, 20, 12, 0, 0)
    midnight = datetime.datetime(2020, 3, 20, 0, 0, 0)
    assert wp.sunlit_fraction(sample_points, noon) == pytest.approx(1.0)
    assert wp.sunlit_fraction(sample_points, midnight) == pytest.approx(0.0)


def test_sample_lattice_5x5_uses_real_center_not_corner_average():
    sw, nw, ne, se = (20.0, -125.0), (45.0, -125.0), (45.0, -65.0), (20.0, -65.0)
    real_cen = (33.0, -97.0)  # deliberately NOT the corners' arithmetic mean (32.5, -95.0)
    pts = wp.sample_lattice_5x5(sw, nw, ne, se, real_cen)
    assert len(pts) == 25
    assert pts[12] == real_cen  # index (v=0.5,u=0.5) is the 13th point (row 2, col 2) of a 5x5 lattice
    assert pts[0] == sw
    assert pts[4] == se
    assert pts[20] == nw
    assert pts[24] == ne


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


def test_parse_wrfinput_corners_missing_cen_lat_returns_none():
    assert wp.parse_wrfinput_corners("netcdf x {\n// global attributes:\n}\n") is None


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


def test_parse_ncdump_dims_reads_south_north_west_east():
    lat2d, lon2d = _synthetic_xlat_xlong_5x5()
    text = _build_ncdump_v_text(5, 5, lat2d, lon2d)
    dims = wp.parse_ncdump_dims(text)
    assert dims["south_north"] == 5
    assert dims["west_east"] == 5
    assert "Time" not in dims  # UNLIMITED, no bare integer on that line


def test_parse_ncdump_2d_var_and_grid_lattice_match_conus12_corners():
    lat2d, lon2d = _synthetic_xlat_xlong_5x5()
    text = _build_ncdump_v_text(5, 5, lat2d, lon2d)
    dims = wp.parse_ncdump_dims(text)
    parsed_lat = wp.parse_ncdump_2d_var(text, "XLAT", dims["south_north"], dims["west_east"])
    parsed_lon = wp.parse_ncdump_2d_var(text, "XLONG", dims["south_north"], dims["west_east"])
    assert parsed_lat[0][0] == pytest.approx(_CONUS12_SW[0])
    assert parsed_lon[0][0] == pytest.approx(_CONUS12_SW[1])

    points = wp.sample_lattice_5x5_from_grid(parsed_lat, parsed_lon)
    assert len(points) == 25
    assert points[0] == pytest.approx(_CONUS12_SW)     # (j=0,i=0): south-west
    assert points[4] == pytest.approx(_CONUS12_SE)     # (j=0,i=4): south-east
    assert points[20] == pytest.approx(_CONUS12_NW)    # (j=4,i=0): north-west
    assert points[24] == pytest.approx(_CONUS12_NE)    # (j=4,i=4): north-east


def test_parse_ncdump_2d_var_returns_none_on_wrong_count():
    text = _build_ncdump_v_text(5, 5, *_synthetic_xlat_xlong_5x5())
    assert wp.parse_ncdump_2d_var(text, "XLAT", 6, 6) is None  # wrong shape
    assert wp.parse_ncdump_2d_var(text, "NOSUCHVAR", 5, 5) is None


def test_sample_points_via_ncdump_arrays_end_to_end(tmp_path, monkeypatch):
    """Full _sample_points_via_ncdump_arrays pipeline, with subprocess.run
    monkeypatched to return the synthetic ncdump -v text instead of
    actually invoking a real ncdump binary (none is required on this dev
    machine -- see _find_ncdump_binary's own docstring for why scc's own
    ncdump isn't even on PATH there)."""
    text = _build_ncdump_v_text(5, 5, *_synthetic_xlat_xlong_5x5())

    class _FakeResult:
        returncode = 0
        stdout = text

    def _fake_run(args, **kwargs):
        assert args[0].endswith("ncdump")
        assert "-v" in args
        return _FakeResult()

    monkeypatch.setattr(wp.subprocess, "run", _fake_run)
    points = wp._sample_points_via_ncdump_arrays(tmp_path / "wrfinput_d01")
    assert points[0] == pytest.approx(_CONUS12_SW)
    assert points[24] == pytest.approx(_CONUS12_NE)


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


def test_read_wrfinput_sample_points_geometry_source_none_without_file(tmp_path):
    points, source = wp.read_wrfinput_sample_points(str(tmp_path))
    assert points is None
    assert source == "none"


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


def test_measure_rad_from_rsl_two_regime_end_to_end(tmp_path):
    entries = _build_two_regime_rsl_entries(r_night_true=2.0, r_day_true=4.0)
    rsl_path = tmp_path / "rsl.error.0000"
    _write_rsl(rsl_path, entries)
    result = wp.measure_rad_from_rsl(
        [str(rsl_path)], radt_minutes=180, dt_seconds=3600, sample_points=[(0.0, 0.0)],
        default_r_rad=5.0, day_night_ratio=1.87)

    assert result["source"] == "fit_day_night"
    assert result["s_normal"] == pytest.approx(2.0)
    assert result["r_night"] == pytest.approx(2.0)
    assert result["r_day"] == pytest.approx(4.0)
    assert result["n_radiation_steps"] == 8  # i=3,6,9,12,15,18,21,24 (step 1 dropped)
    assert result["mean_sunlit_fraction"] == pytest.approx(0.5)  # 4 night + 4 day


def test_measure_rad_from_rsl_single_night_regime_end_to_end(tmp_path):
    """Restrict the probe window to entirely-night hours (like matthew's 00Z
    t0): only i=1 (dropped) and i=3,6 fall in-window, both night."""
    start = datetime.datetime(2020, 3, 20, 0, 0, 0)
    entries = [(start + datetime.timedelta(hours=0), 1, 999.0),               # i=1, dropped
              (start + datetime.timedelta(hours=1), 1, 2.0),                  # i=2, normal
              (start + datetime.timedelta(hours=2), 1, 1.9 * 2.0),            # i=3, radiation, night (02:00)
              (start + datetime.timedelta(hours=3), 1, 2.0),                  # i=4
              (start + datetime.timedelta(hours=4), 1, 2.0),                  # i=5
              (start + datetime.timedelta(hours=5), 1, 2.1 * 2.0)]            # i=6, radiation, night (05:00)
    rsl_path = tmp_path / "rsl.error.0000"
    _write_rsl(rsl_path, entries)
    result = wp.measure_rad_from_rsl([str(rsl_path)], radt_minutes=180, dt_seconds=3600,
                                     sample_points=[(0.0, 0.0)], default_r_rad=5.0,
                                     day_night_ratio=1.87)
    assert result["source"] == "ratio_from_night"
    assert result["r_night"] == pytest.approx(2.0)  # mean(1.9, 2.1)
    assert result["r_day"] == pytest.approx(2.0 * 1.87)


def test_measure_rad_from_rsl_pools_multiple_files(tmp_path):
    entries = _build_two_regime_rsl_entries(r_night_true=2.0, r_day_true=4.0)
    half = len(entries) // 2
    p1, p2 = tmp_path / "rsl.error.0000", tmp_path / "rsl.error.0001"
    # step 1 must be present in EACH file (its own file-relative index 0 is
    # what gets dropped), so replicate it into both halves.
    _write_rsl(p1, [entries[0]] + entries[1:half])
    _write_rsl(p2, [entries[0]] + entries[half:])
    result = wp.measure_rad_from_rsl(
        [str(p1), str(p2)], radt_minutes=180, dt_seconds=3600, sample_points=[(0.0, 0.0)],
        default_r_rad=5.0, day_night_ratio=1.87)
    assert result["source"] == "fit_day_night"
    assert result["r_night"] == pytest.approx(2.0)
    assert result["r_day"] == pytest.approx(4.0)


def test_measure_rad_from_rsl_falls_back_when_lt2_radiation_samples(tmp_path):
    # radt=180min, dt=3600s -> stepra=3 (radiation at i=1,3,6,...); a 5-step
    # window only has i=1 (dropped) and i=3 as schedule-radiation -> exactly
    # 1 usable radiation sample, below the fit's own >=2 requirement.
    start = datetime.datetime(2020, 3, 20, 0, 0, 0)
    entries = [(start + datetime.timedelta(hours=i), 1, 2.0) for i in range(5)]
    rsl_path = tmp_path / "rsl.error.0000"
    _write_rsl(rsl_path, entries)
    result = wp.measure_rad_from_rsl([str(rsl_path)], radt_minutes=180, dt_seconds=3600,
                                     sample_points=[(0.0, 0.0)], default_r_rad=5.0,
                                     day_night_ratio=1.87)
    assert result["source"] == "default_unmeasured"
    assert result["r_night"] == 5.0
    assert result["r_day"] == 5.0
    assert result["n_radiation_steps"] == 1


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

def test_predict_radiation_extra_integrates_to_hand_computed_value():
    run_start_dt = datetime.datetime(2020, 3, 20, 0, 0, 0)
    sample_points = [(0.0, 0.0)]  # equator/prime-meridian: sunrise/sunset ~06:00/18:00 UTC at equinox
    rad_steps, rad_extra_s, mean_frac, sampling_advice = wp.predict_radiation_extra(
        steps=12, radt=180, dt_seconds=3600, s_normal=2.0, r_night=1.5, r_day=3.0,
        run_start_dt=run_start_dt, sample_points=sample_points, default_r_rad=5.0)
    # stepra=NINT(180*60/3600)=3 -> radiation steps i=1,3,6,9,12, offsets 0,2,5,8,11h.
    # hours 0,2,5 = night (frac=0); hours 8,11 = day (frac=1).
    assert rad_steps == 5
    expected = 3 * (1.5 - 1.0) * 2.0 + 2 * (3.0 - 1.0) * 2.0
    assert rad_extra_s == pytest.approx(expected)
    assert rad_extra_s == pytest.approx(11.0)
    assert mean_frac == pytest.approx(2.0 / 5.0)
    assert sampling_advice["night_sunlit_fraction"] == pytest.approx(0.0)
    assert sampling_advice["day_sunlit_fraction"] == pytest.approx(1.0)


def test_predict_radiation_extra_uniform_fallback_without_geometry():
    rad_steps, rad_extra_s, mean_frac, sampling_advice = wp.predict_radiation_extra(
        steps=12, radt=180, dt_seconds=3600, s_normal=2.0, r_night=1.5, r_day=3.0,
        run_start_dt=None, sample_points=None, default_r_rad=5.0)
    assert rad_steps == 5
    assert rad_extra_s == pytest.approx(5 * ((1.5 + 3.0) / 2.0 - 1.0) * 2.0)
    assert mean_frac is None
    assert sampling_advice is None


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


def test_predict_compute_ledger_s_normal_wins_rsl_still_sets_day_night(tmp_path):
    """Per spec: --probe (ledger) keeps supplying s_normal when it has a
    match; --probe-rsl only supplies s_normal when the ledger gave none.
    r_night/r_day, however, are always eligible to come from --probe-rsl."""
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), RAD_RSL_NAMELIST))
    steps, run_s, dt, _ = wp.get_d01_steps(nl)
    run_min = run_s / 60.0
    rsl_path = tmp_path / "rsl.error.0000"
    _write_rsl(rsl_path, _build_two_regime_rsl_entries(r_night_true=2.0, r_day_true=4.0))

    probe_result = {"s_normal": 9.0, "source": "measured", "spread_pct": None}
    compute = wp.predict_compute(
        nl, 1, steps, run_min, ranks=1, omp=1, probe_result=probe_result,
        calibration={"radiation": {"r_rad_default": 5.0, "r_rad_day_night_ratio": 1.87}},
        rsl_paths=[str(rsl_path)], dt_seconds=dt, sample_points=[(0.0, 0.0)],
        run_start_dt=datetime.datetime(2020, 3, 20, 0, 0, 0))
    assert compute["s_normal"] == pytest.approx(9.0)  # ledger value, unchanged
    assert compute["source"] == "measured"
    assert compute["r_night"] == pytest.approx(2.0)  # measured from rsl, not the default
    assert compute["r_day"] == pytest.approx(4.0)
    assert compute["r_rad_source"] == "fit_day_night"


def test_predict_compute_default_r_rad_when_no_probe_rsl_given(tmp_path):
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), RAD_RSL_NAMELIST))
    steps, run_s, dt, _ = wp.get_d01_steps(nl)
    run_min = run_s / 60.0
    compute = wp.predict_compute(nl, 1, steps, run_min, ranks=1, omp=1, probe_result=None,
                                 calibration={"radiation": {"r_rad_default": 5.0},
                                              "fallback_s_per_cell_step_ns": 1.0},
                                 rsl_paths=None, dt_seconds=dt)
    assert compute["r_night"] == 5.0
    assert compute["r_day"] == 5.0
    assert compute["r_rad_source"] == "default_unmeasured"


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


def test_predict_init_adds_startup_s_when_present(tmp_path):
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), STARTUP_NAMELIST))
    calibration = {"derived": {"read_rate_bytes_per_s": 1000.0, "bytes_per_cell_level": 1.0},
                   "startup_s": 20.0, "startup_s_source": "unit-test"}
    init = wp.predict_init(None, nl, calibration)
    assert init["startup_calibrated"] is True
    assert init["startup_s"] == 20.0
    assert init["startup_s_source"] == "unit-test"
    assert init["total_s"] == pytest.approx(init["read_s"] + 20.0)


def test_predict_init_startup_uncalibrated_when_absent(tmp_path):
    nl = wp.parse_namelist(_write(str(tmp_path / "namelist.input"), STARTUP_NAMELIST))
    calibration = {"derived": {"read_rate_bytes_per_s": 1000.0, "bytes_per_cell_level": 1.0}}
    init = wp.predict_init(None, nl, calibration)
    assert init["startup_calibrated"] is False
    assert init["startup_s"] is None
    assert init["total_s"] == pytest.approx(init["read_s"])


def test_render_text_prints_startup_uncalibrated_with_real_calibration(tmp_path):
    # machine/calibration/scc.json currently has startup_s: null (pending the
    # 2026-09-23 ground-truth fit) -- assert the text output says so plainly.
    case_dir = tmp_path / "conus25"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), CONUS25_NAMELIST)
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=7,
        config={"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1},
        ledger_paths=None, calibration_path=CALIBRATION,
        io_form_override=None, io_target="nfs")
    assert "startup: UNCALIBRATED" in wp.render_text(pred)


# --------------------------------------------------------------------------
# 11. I/O as a range (history_write_rate_MBps_low/_high)
# --------------------------------------------------------------------------

IORANGE_NAMELIST = """\
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


def test_io_range_brackets_point_estimate_and_widens_exactly(tmp_path):
    case_dir = tmp_path / "iorange_case"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), IORANGE_NAMELIST)

    # bytes_per_frame = bpcl(1000) * cells(10*10*10=1000) = 1,000,000 bytes;
    # n_frames = floor(60/60)+1 = 2 -> 2,000,000 bytes total.
    # point rate 1,000,000 B/s -> 2.0s; low(slow)=0.5MB/s=500,000B/s -> 4.0s;
    # high(fast)=2.0MB/s=2,000,000B/s -> 1.0s. All exact, no rounding.
    calibration = {
        "derived": {
            "bytes_per_cell_level": 1000.0,
            "io_form_rate_table": {
                "2": {"nfs": {"rate_bytes_per_s": 1000000.0, "uncertainty_pct": 10, "source": "point"}}},
            "read_rate_bytes_per_s": 1000000.0,
        },
        "fallback_s_per_cell_step_ns": 1.0,
        "drift_band_pct": 5.0,
        "history_write_rate_MBps_low": 0.5,
        "history_write_rate_MBps_high": 2.0,
    }
    cal_path = tmp_path / "calibration.json"
    _write(str(cal_path), json.dumps(calibration))

    config = {"ranks": 1, "omp": 1, "nx": 1, "ny": 1, "numtiles": 1}
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=1,
        config=config, ledger_paths=None, calibration_path=str(cal_path),
        io_form_override=None, io_target="nfs")

    io_range = pred["io_range"]
    assert io_range is not None
    assert io_range["fast_total_s"] == pytest.approx(1.0)
    assert io_range["slow_total_s"] == pytest.approx(4.0)
    assert pred["io"]["total_s"] == pytest.approx(2.0)  # point estimate unchanged

    compute_total = pred["compute"]["compute_total_s"]
    drift_pct = pred["drift_pct"]
    compute_drift = compute_total * (drift_pct / 100.0)
    expected_low = max(0.0, compute_total - compute_drift) + io_range["fast_total_s"] + pred["init"]["total_s"]
    expected_high = (compute_total + compute_drift) + io_range["slow_total_s"] + pred["init"]["total_s"]

    assert pred["range_low_s"] == pytest.approx(expected_low)
    assert pred["range_high_s"] == pytest.approx(expected_high)
    assert pred["range_low_s"] <= pred["total_s"] <= pred["range_high_s"]
    # widens by exactly the I/O fast/slow spread plus the compute-only drift band on both sides
    assert (pred["range_high_s"] - pred["range_low_s"]) == pytest.approx(
        (io_range["slow_total_s"] - io_range["fast_total_s"]) + 2 * compute_drift)

    text = wp.render_text(pred)
    assert "I/O rate range" in text
    assert "write-rate bracket" in text


def test_io_range_absent_keeps_legacy_rss_range(tmp_path):
    case_dir = tmp_path / "conus25"
    case_dir.mkdir()
    _write(str(case_dir / "namelist.input"), CONUS25_NAMELIST)
    pred = wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=7,
        config={"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1},
        ledger_paths=None, calibration_path=CALIBRATION,
        io_form_override=None, io_target="nfs")
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
        [sys.executable, os.path.join(HERE, "wtune_predict.py"), "predict", str(case_dir),
         "--nodes", "1", "--nx", "9", "--ny", "8", "--ranks", "72",
         "--probe", str(ledger_path), "--case-name", "lafe",
         "--calibration", CALIBRATION, "--json", str(out_json)],
        capture_output=True, text=True, cwd=HERE)
    assert result.returncode == 0, result.stderr
    doc = json.loads(out_json.read_text())
    assert doc["compute"]["source"] == "measured"
    assert doc["compute"]["s_normal"] == pytest.approx(8.587)
