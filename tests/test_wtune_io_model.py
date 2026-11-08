#!/usr/bin/env python3
"""wtune/io_model.py: run-length I/O terms of the whole-run prediction (release 0.8.2).

Calibration evidence (scc, 2026-10): climate-ea50 150x110x45 50 km, dt 240 s, 142,560 steps
(396 d), history 6 h (frames_per_outfile 4), interval_seconds 21600, sst_update=1 auxinput4 6 h,
N=5 120x3, 14 segments. Tune predicted 46:14 (compute_total 44:09 + init 125.6 s); actual 89:38
summed; rsl: history+restart writes 1598 x 0.74 s = 1181 s, lateral boundary 1584 x 0.31 s =
494 s, other in-step ~260 s (sst_update reads etc.), per-segment startup 14 x ~53 s = 745 s."""
import pytest

from wtune import io_model as iom
from wtune import predict as wp
from wtune import run as wr

CLIMATE_EA50 = """ &time_control
 run_days = 0, run_hours = 0, run_minutes = 0, run_seconds = 34214400,
 start_year = 2020, start_month = 01, start_day = 01, start_hour = 00,
 interval_seconds = 21600
 history_interval = 360,
 frames_per_outfile = 4,
 restart_interval = 43200,
 io_form_history = 2
 io_form_restart = 2
 auxinput4_interval = 360,
 io_form_auxinput4 = 2
 /
 &domains
 time_step = 240, max_dom = 1, e_we = 150, e_sn = 110, e_vert = 45,
 /
 &physics
 radt = 32, sst_update = 1,
 /
"""


THREE_DOM = """ &time_control
 run_hours = 12,
 start_year = 2020, 2020, 2020, start_month = 06, 06, 06, start_day = 01, 01, 01,
 interval_seconds = 21600
 history_interval = 60, 30, 15,
 history_begin = 0, 0, 90,
 restart_interval = 7200,
 io_form_history = 2
 io_form_restart = 14
 /
 &domains
 time_step = 60, max_dom = 3,
 e_we = 100, 100, 91, e_sn = 100, 91, 91, e_vert = 45, 45, 45,
 parent_id = 1, 1, 2, parent_grid_ratio = 1, 3, 3, parent_time_step_ratio = 1, 3, 3,
 /
 &physics
 radt = 30, 10, 10,
 /
"""


def _nl(tmp_path, text, name="namelist.input"):
    p = tmp_path / name
    p.write_text(text)
    return wp.parse_namelist(str(p))


# --- the copied formula equals wtune/run/'s ------------------------------------------------
def test_constants_are_the_wtune_run_copies():
    assert iom.CONUS25_RESTART_POINTS == wr.CONUS25_RESTART_POINTS
    assert iom.RESTART_IO_ANCHOR_S == wr.RESTART_IO_ANCHOR_S
    assert iom.HISTORY_S_PER_POINT_FRAME == wr.HISTORY_S_PER_POINT_FRAME


def test_three_domain_history_begin_and_adios2_restart(tmp_path):
    io = iom.run_length_io(_nl(tmp_path, THREE_DOM), io="netcdf")
    assert io["history_frames"] == [13, 25, 43]     # 0..12 h; d03 from 01:30 every 15 min
    assert io["io_form_restart"] == 14 and io["n_restart"] == 0     # 7200 min > 12 h
    assert io["n_boundary"] == 2 and io["aux_s"] == 0.0


# --- namelist edge cases ----------------------------------------------------------------------
def test_switches_turn_terms_off_and_restart_default(tmp_path):
    text = (CLIMATE_EA50.replace("restart_interval = 43200,", "")
            .replace("io_form_history = 2", "io_form_history = 0")
            .replace("sst_update = 1", "sst_update = 0")
            .replace(" &physics", " &bdy_control\n specified = .false.\n /\n &physics"))
    io = iom.run_length_io(_nl(tmp_path, text), io="netcdf")
    assert io["history_s"] == 0.0 and io["aux_s"] == 0.0 and io["boundary_s"] == 0.0
    assert io["restart_interval_min"] == 1440.0 and io["n_restart"] == 396   # WRF default 1 day


def test_unknown_restart_io_form_uses_the_netcdf_anchor():
    assert iom.restart_io_s([90e6], 11) == iom.restart_io_s([90e6], 2) == (324.0, 120.0)


@pytest.mark.parametrize("spelling,fmt", [(None, "adios2"), ("adios2", "adios2"), (14, "adios2"),
                                          ("ADIOS2", "adios2"), ("netcdf", "netcdf"),
                                          (2, "netcdf"), ("nc", "netcdf")])
def test_io_format_selection(spelling, fmt):
    assert iom.normalize_io_format(spelling) == fmt


def test_unknown_io_format_is_refused():
    with pytest.raises(ValueError):
        iom.normalize_io_format("hdf5")
