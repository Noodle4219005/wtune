#!/usr/bin/env python3
"""wtune_io_model.py: run-length I/O terms of the whole-run prediction (release 0.8.2).

Calibration evidence (scc, 2026-10): climate-ea50 150x110x45 50 km, dt 240 s, 142,560 steps
(396 d), history 6 h (frames_per_outfile 4), interval_seconds 21600, sst_update=1 auxinput4 6 h,
N=5 120x3, 14 segments. Tune predicted 46:14 (compute_total 44:09 + init 125.6 s); actual 89:38
summed; rsl: history+restart writes 1598 x 0.74 s = 1181 s, lateral boundary 1584 x 0.31 s =
494 s, other in-step ~260 s (sst_update reads etc.), per-segment startup 14 x ~53 s = 745 s."""
import pytest

import wtune_io_model as iom
import wtune_predict as wp
import wtune_run as wr

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

SC26_P2 = """ &time_control
 run_days = 0, run_hours = 48, run_minutes = 0, run_seconds = 0,
 start_year = 2005, start_month = 08, start_day = 27, start_hour = 00,
 interval_seconds = 21600
 history_interval = 60,
 restart_interval = 100000,
 io_form_history = 2
 io_form_restart = 2
 /
 &domains
 time_step = 72, max_dom = 1, e_we = 170, e_sn = 130, e_vert = 40,
 /
 &physics
 radt = 12,
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


# --- the copied formula equals wtune_run.py's ------------------------------------------------
def test_constants_are_the_wtune_run_copies():
    assert iom.CONUS25_RESTART_POINTS == wr.CONUS25_RESTART_POINTS
    assert iom.RESTART_IO_ANCHOR_S == wr.RESTART_IO_ANCHOR_S
    assert iom.HISTORY_S_PER_POINT_FRAME == wr.HISTORY_S_PER_POINT_FRAME


@pytest.mark.parametrize("text", [CLIMATE_EA50, SC26_P2, THREE_DOM])
@pytest.mark.parametrize("io_form", [2, 14])
def test_per_unit_costs_and_frame_counts_match_wtune_run(tmp_path, text, io_form):
    nl = _nl(tmp_path, text)
    geo = wr.case_geometry(nl)
    over = wr.fallback_overheads(geo, io_form)
    w, r = iom.restart_io_s(geo["points"], io_form)
    assert (w, r) == (over["restart_write_s"], over["restart_read_s"])
    assert iom.history_frame_s(geo["points"]) == over["history_frame_s"]
    io = iom.run_length_io(nl)
    assert io["points"] == geo["points"]
    for d in range(geo["max_dom"]):
        assert io["history_frames"][d] == wr.history_frames(geo, d, 0, geo["steps"])
    # whole-run history seconds == wtune_run.history_wall over the whole run
    assert io["history_s"] == pytest.approx(wr.history_wall(geo, over, 0, geo["steps"]))


def test_three_domain_history_begin_and_adios2_restart(tmp_path):
    io = iom.run_length_io(_nl(tmp_path, THREE_DOM))
    assert io["history_frames"] == [13, 25, 43]     # 0..12 h; d03 from 01:30 every 15 min
    assert io["io_form_restart"] == 14 and io["n_restart"] == 0     # 7200 min > 12 h
    assert io["n_boundary"] == 2 and io["aux_s"] == 0.0


# --- climate-ea50 calibration -----------------------------------------------------------------
def test_climate_ea50_terms_against_the_rsl_decomposition(tmp_path):
    io = iom.run_length_io(_nl(tmp_path, CLIMATE_EA50))
    assert io["run_s"] == 142560 * 240.0
    assert io["history_frames"] == [1585] and io["n_restart"] == 13      # 1598 writes in the rsl
    assert io["history_frames"][0] + io["n_restart"] == 1598
    assert io["n_boundary"] == 1584 and io["n_aux"] == [1584]
    # model: history 1585 x 0.669 = 1060 s + restart 13 x 2.67 = 35 s = 1095 s vs 1181 (-7%)
    assert io["history_s"] == pytest.approx(1060.3, abs=0.5)
    assert io["restart_s"] == pytest.approx(34.7, abs=0.1)
    assert io["history_s"] + io["restart_s"] == pytest.approx(1181.0, rel=0.30)
    assert (io["history_s"] + io["restart_s"]) / 1181.0 == pytest.approx(0.927, abs=0.01)
    # boundary 1584 x 0.31 = 491 s vs 494; aux 1584 x 0.164 = 260 s (the whole "other" bucket)
    assert io["boundary_s"] == pytest.approx(491.0, abs=0.5)
    assert io["boundary_s"] == pytest.approx(494.0, rel=0.30)
    assert io["aux_s"] == pytest.approx(260.0, abs=0.5)
    assert io["total_s"] == pytest.approx(1846.1, abs=1.0)


def test_climate_ea50_whole_run_prediction_and_breakdown(tmp_path):
    # init = round0's MEASURED startup 49.9 s (5 runs), not predict_init's 125.6 s (which charges
    # the whole 13-month wrfbdy as an init read -- the boundary term already counts it per update)
    io = iom.run_length_io(_nl(tmp_path, CLIMATE_EA50))
    compute, init = 44 * 60 + 9.0, 49.9
    total = compute + init + io["total_s"]
    assert total == pytest.approx(4545.0, abs=1.0)                   # 75:45 (was 46:14)
    # actual 89:38 = 5378 s over 14 segments; one unsegmented run pays startup once, so the
    # comparable actual drops 13 extra segment startups (~53 s each): 5378 - 689 = 4689 s (-3.1%)
    assert total == pytest.approx(5378.0 - 13 * 53.0, rel=0.05)
    # with wtune run's 14 segments each paying init_s once more: 4545 + 13 x 49.9 = 5194 (-3.4%)
    assert total + 13 * init == pytest.approx(5378.0, rel=0.05)
    assert iom.breakdown_line(compute, io, init) == (
        "compute 2649s + history 1060s + boundary 491s + aux 260s + restart 35s + init 50s "
        "= 4545s")
    assert "~50 s" in iom.segment_note(init)


# --- short case: SC26-P2 48 h 12 km -----------------------------------------------------------
def test_sc26p2_what_the_new_terms_add(tmp_path):
    # Tune predicted 2:36 (156 s) = t0-anchor 2400 steps x 0.0150 s/step (36.0 s) + the 120 s
    # DEFAULT startup in effect at the sweep; actual 2:21 (141 s). The I/O terms add ~42 s,
    # almost all history (49 frames x 0.80 s at 170x130x40 -- the same per-point rate the
    # climate-ea50 rsl measured, 0.72 s/frame at 150x110x45). With round0's measured startup
    # 54.1 s as init: 36.0 + 41.6 + 54.1 = 131.7 s (2:12), -6.6% vs 141 s (was +10.6%).
    io = iom.run_length_io(_nl(tmp_path, SC26_P2))
    assert io["history_frames"] == [49] and io["n_boundary"] == 8 and io["n_restart"] == 0
    assert io["history_s"] == pytest.approx(39.0, abs=0.1)
    assert io["boundary_s"] == pytest.approx(2.5, abs=0.1)
    assert io["aux_s"] == 0.0 and io["restart_s"] == 0.0
    assert io["total_s"] == pytest.approx(41.6, abs=0.1)
    compute, init = 2400 * 0.0150, 54.1
    total = compute + init + io["total_s"]
    assert total == pytest.approx(131.7, abs=0.2)
    assert abs(total - 156.0) <= 30.0 and total == pytest.approx(141.0, rel=0.10)
    assert iom.breakdown_line(compute, io, init) == (
        "compute 36s + history 39s + boundary 3s + aux 0s + restart 0s + init 54s = 132s")


# --- namelist edge cases ----------------------------------------------------------------------
def test_switches_turn_terms_off_and_restart_default(tmp_path):
    text = (CLIMATE_EA50.replace("restart_interval = 43200,", "")
            .replace("io_form_history = 2", "io_form_history = 0")
            .replace("sst_update = 1", "sst_update = 0")
            .replace(" &physics", " &bdy_control\n specified = .false.\n /\n &physics"))
    io = iom.run_length_io(_nl(tmp_path, text))
    assert io["history_s"] == 0.0 and io["aux_s"] == 0.0 and io["boundary_s"] == 0.0
    assert io["restart_interval_min"] == 1440.0 and io["n_restart"] == 396   # WRF default 1 day


def test_unknown_restart_io_form_uses_the_netcdf_anchor():
    assert iom.restart_io_s([90e6], 11) == iom.restart_io_s([90e6], 2) == (324.0, 120.0)
