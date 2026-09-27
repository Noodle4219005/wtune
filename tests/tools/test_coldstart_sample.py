#!/usr/bin/env python3
"""Offline tests for tools/coldstart_sample.py. No network access -- every network-touching
function (metdata.fetch_*, grib_state.trajectory) is either not exercised here or is exercised
only through the pure pieces (select_source's date-rule logic, which is itself offline).

Run with:
  python3 -m pytest -q tools/test_coldstart_sample.py
"""
import datetime
import json
import os
import re
import statistics
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coldstart_sample as cs   # noqa: E402
import wtune_predict as predict  # noqa: E402


_WRF_TIME_FMT = cs._WRF_TIME_FMT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NAMELIST_INPUT_TMPL = """\
 &time_control
 run_days                            = 0,
 run_hours                           = {run_hours},
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = {sy},
 start_month                         = {smo},
 start_day                           = {sd},
 start_hour                          = {sh},
 start_minute                        = 00,
 start_second                        = 00,
 end_year                            = {ey},
 end_month                           = {emo},
 end_day                             = {ed},
 end_hour                            = {eh},
 end_minute                          = 00,
 end_second                          = 00,
 interval_seconds                    = 10800
 history_interval_m                  = 60,
 frames_per_outfile                  = 1,
 restart                             = .false.,
 restart_interval                    = 660,
 io_form_history                     = 2,
 /

 &domains
 time_step                           = 72,
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = 1,
 e_we                                = 425,
 e_sn                                = 300,
 e_vert                              = 50,
 dx                                  = 12000,
 dy                                  = 12000,
 parent_id                           = 0,
 i_parent_start                      = 0,
 j_parent_start                      = 0,
 parent_grid_ratio                   = 1,
 num_metgrid_levels                  = 34,
 num_metgrid_soil_levels             = 4,
 nproc_x                             = 9,
 nproc_y                             = 8,
 /

 &physics
 radt                                = 15,
 ifire                               = {ifire},
 /
"""

NAMELIST_WPS_TMPL = """\
&share
 wrf_core = 'ARW',
 max_dom = 1,
 start_date = '2020-01-01_00:00:00',
 end_date   = '2020-01-01_03:00:00',
 interval_seconds = 10800,
 io_form_geogrid = 2,
/

&geogrid
 parent_id         = 1,
 parent_grid_ratio = 1,
 i_parent_start    = 1,
 j_parent_start    = 1,
 e_we              = 425,
 e_sn              = 300,
 geog_data_res     = 'default',
 dx                = 12000,
 dy                = 12000,
 map_proj          = 'lambert',
 truelat1          = 30,
 truelat2          = 60,
 stand_lon         = -98,
 ref_lat           = 40.00001,
 ref_lon           = -98,
 geog_data_path    = '/nfs/data/wrf/WPS_GEOG/'
/

&ungrib
 out_format = 'WPS',
 prefix = 'GFS',
/

&metgrid
 fg_name = 'GFS',
 io_form_metgrid = 2,
/
"""


def write_namelist(path, sy=2020, smo=1, sd=1, sh=0, ey=2020, emo=1, ed=1, eh=3, ifire=0,
                   run_hours=3, extra=""):
    with open(path, "w") as f:
        f.write(NAMELIST_INPUT_TMPL.format(sy=sy, smo=smo, sd=sd, sh=sh,
                                            ey=ey, emo=emo, ed=ed, eh=eh, ifire=ifire,
                                            run_hours=run_hours))
        if extra:
            # wtune_predict.parse_namelist merges repeated '&group' blocks into the same dict
            # (groups.setdefault), so a second small &physics/&chem/&fdda block just adds keys.
            f.write("\n" + extra)


def write_namelist_wps(path):
    with open(path, "w") as f:
        f.write(NAMELIST_WPS_TMPL)


# ---------------------------------------------------------------------------
# 1. namelist.wps from a wrfinput-attr dict (conus12 facts)
# ---------------------------------------------------------------------------

class TestNamelistWpsFromWrfinputAttrs(unittest.TestCase):
    def test_conus12_single_domain_fields(self):
        attrs = {
            "MAP_PROJ": 1.0, "TRUELAT1": 30.0, "TRUELAT2": 60.0, "STAND_LON": -98.0,
            "CEN_LAT": 40.00001, "CEN_LON": -98.0, "MOAD_CEN_LAT": 40.00001,
            "WEST-EAST_GRID_DIMENSION": 425.0, "SOUTH-NORTH_GRID_DIMENSION": 300.0,
            "DX": 12000.0, "DY": 12000.0,
        }
        nl = {"domains": {"e_we": [425], "e_sn": [300], "dx": [12000.0], "max_dom": [1]}}
        geo = cs.geo_from_wrfinput_attrs(attrs, nl)

        self.assertEqual(geo["map_proj"], "lambert")
        self.assertEqual(geo["truelat1"], 30.0)
        self.assertEqual(geo["truelat2"], 60.0)
        self.assertEqual(geo["stand_lon"], -98.0)
        self.assertEqual(geo["ref_lat"], 40.00001)
        self.assertEqual(geo["ref_lon"], -98.0)
        self.assertEqual(geo["e_we"], 425)
        self.assertEqual(geo["e_sn"], 300)
        self.assertEqual(geo["dx"], 12000.0)

        start_dt = datetime.datetime(2019, 11, 26, 15, 0, 0)
        end_dt = datetime.datetime(2019, 11, 26, 18, 0, 0)
        fields = cs.build_namelist_wps_fields(nl, geo, start_dt, end_dt, 10800)
        self.assertEqual(fields["e_we"], [425])
        self.assertEqual(fields["e_sn"], [300])
        self.assertEqual(fields["parent_id"], [1])
        self.assertEqual(fields["parent_grid_ratio"], [1])
        self.assertEqual(fields["i_parent_start"], [1])
        self.assertEqual(fields["j_parent_start"], [1])

        text = cs.render_namelist_wps(fields)
        self.assertIn("map_proj          = 'lambert',", text)
        self.assertIn("truelat1          = 30,", text)
        self.assertIn("truelat2          = 60,", text)
        self.assertIn("stand_lon         = -98,", text)
        self.assertIn("ref_lat           = 40.00001,", text)
        self.assertIn("ref_lon           = -98,", text)
        self.assertIn("e_we              = 425,", text)
        self.assertIn("e_sn              = 300,", text)

    def test_e_we_mismatch_is_an_error(self):
        attrs = {"MAP_PROJ": 1.0, "CEN_LAT": 40.0, "CEN_LON": -98.0,
                 "WEST-EAST_GRID_DIMENSION": 100.0, "SOUTH-NORTH_GRID_DIMENSION": 100.0,
                 "DX": 12000.0, "DY": 12000.0}
        nl = {"domains": {"e_we": [425], "e_sn": [300], "dx": [12000.0]}}
        with self.assertRaises(cs.ColdstartError):
            cs.geo_from_wrfinput_attrs(attrs, nl)


# ---------------------------------------------------------------------------
# 2. nested case (max_dom=2): per-domain lists copied from namelist.input
# ---------------------------------------------------------------------------

class TestDomainListFieldsNested(unittest.TestCase):
    def test_nest_copies_namelist_input_values(self):
        nl = {"domains": {
            "max_dom": [2],
            "e_we": [425, 100],
            "e_sn": [300, 80],
            "parent_id": [1, 1],
            "parent_grid_ratio": [1, 3],
            "i_parent_start": [1, 50],
            "j_parent_start": [1, 40],
        }}
        dom = cs.domain_list_fields(nl, 2)
        # domain 1 (d01): geogrid self-parent convention, NOT namelist.input's own (often 0) values
        self.assertEqual(dom["parent_id"][0], 1)
        self.assertEqual(dom["i_parent_start"][0], 1)
        self.assertEqual(dom["j_parent_start"][0], 1)
        # domain 2: copied verbatim from namelist.input
        self.assertEqual(dom["parent_id"][1], 1)
        self.assertEqual(dom["parent_grid_ratio"][1], 3)
        self.assertEqual(dom["i_parent_start"][1], 50)
        self.assertEqual(dom["j_parent_start"][1], 40)
        self.assertEqual(dom["e_we"], [425, 100])
        self.assertEqual(dom["e_sn"], [300, 80])


# ---------------------------------------------------------------------------
# 3. inspect blockers
# ---------------------------------------------------------------------------

class TestInspectBlockers(unittest.TestCase):
    def test_no_projection_source_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            write_namelist(os.path.join(d, "namelist.input"))
            rows, blockers, nl = cs.inspect_case(d)
            self.assertTrue(any("no projection source" in b for b in blockers))

    def test_pre_2015_date_is_NOT_blocked_when_a_fallback_chain_exists(self):
        # 2026-09-24 fix: a pre-2015 date used to be reported as "no GFS 0.25deg source" and
        # blocked outright -- wrong, since NCEI/FNL/CFSR cover it. See
        # TestForcingChainDateBands below for the exact chain per date.
        with tempfile.TemporaryDirectory() as d:
            write_namelist(os.path.join(d, "namelist.input"), sy=2010, ey=2010)
            write_namelist_wps(os.path.join(d, "namelist.wps"))
            rows, blockers, nl = cs.inspect_case(d)
            self.assertEqual(blockers, [])

    def test_pre_1979_date_with_no_grib_dir_is_still_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            write_namelist(os.path.join(d, "namelist.input"), sy=1975, ey=1975)
            write_namelist_wps(os.path.join(d, "namelist.wps"))
            rows, blockers, nl = cs.inspect_case(d)
            self.assertTrue(any("no forcing source covers this date" in b for b in blockers),
                            blockers)

    def test_ifire_positive_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            write_namelist(os.path.join(d, "namelist.input"), ifire=2)
            write_namelist_wps(os.path.join(d, "namelist.wps"))
            rows, blockers, nl = cs.inspect_case(d)
            self.assertTrue(any("ifire=2" in b for b in blockers), blockers)

    def test_completable_case_has_no_blockers(self):
        with tempfile.TemporaryDirectory() as d:
            write_namelist(os.path.join(d, "namelist.input"))
            write_namelist_wps(os.path.join(d, "namelist.wps"))
            rows, blockers, nl = cs.inspect_case(d)
            self.assertEqual(blockers, [])


# ---------------------------------------------------------------------------
# 4. window picking on a synthetic 72h run
# ---------------------------------------------------------------------------

class TestWindowPicking(unittest.TestCase):
    def setUp(self):
        self.start_dt = datetime.datetime(2020, 1, 1, 0, 0, 0)
        self.run_end_dt = self.start_dt + datetime.timedelta(hours=72)
        self.interval_seconds = 10800
        self.W, self.stepra, self.measure_steps = cs.compute_window_seconds(60, 72.0, 15.0)
        self.candidates = cs.candidate_times(self.start_dt, self.run_end_dt,
                                              self.interval_seconds, self.W)
        self.sample_points = cs.domain_lattice_from_geom(35.0, -98.0, 12000.0, 425,
                                                           dy_m=12000.0, e_sn=300)

    def test_t0_always_first(self):
        picks = cs.select_windows(self.candidates, 60 * 60.0, self.W, 4,
                                   sample_points=self.sample_points)
        self.assertEqual(picks[0], (self.candidates[0], "t0"))

    def test_day_and_night_picks_differ(self):
        picks = cs.select_windows(self.candidates, 60 * 60.0, self.W, 4,
                                   sample_points=self.sample_points)
        by_reason = {reason: t for t, reason in picks}
        self.assertIn("day", by_reason)
        self.assertIn("night", by_reason)
        self.assertNotEqual(by_reason["day"], by_reason["night"])

    def test_all_windows_satisfy_constraints_and_k_cap(self):
        K = 4
        picks = cs.select_windows(self.candidates, 60 * 60.0, self.W, K,
                                   sample_points=self.sample_points)
        self.assertLessEqual(len(picks), K)
        for t, _reason in picks:
            self.assertLessEqual(t + datetime.timedelta(seconds=self.W), self.run_end_dt)
            elapsed = (t - self.start_dt).total_seconds()
            self.assertEqual(elapsed % self.interval_seconds, 0)

    def test_k_cap_respected_with_larger_k(self):
        picks = cs.select_windows(self.candidates, 60 * 60.0, self.W, 8,
                                   sample_points=self.sample_points)
        self.assertLessEqual(len(picks), 8)
        # distinct times
        self.assertEqual(len(picks), len(set(t for t, _ in picks)))


# ---------------------------------------------------------------------------
# 5. W computation exact
# ---------------------------------------------------------------------------

class TestWindowSecondsExact(unittest.TestCase):
    def test_dt72_radt15(self):
        W, stepra, measure_steps = cs.compute_window_seconds(60, 72.0, 15.0)
        # steps_per_radiation_call(15, 72) = floor(15*60/72 + 0.5) = floor(12.5+0.5) = 13
        self.assertEqual(stepra, 13)
        self.assertEqual(measure_steps, max(30, 2 * 13))
        self.assertEqual(measure_steps, 30)
        self.assertEqual(W, 60 * 60.0 + 30 * 72.0)
        self.assertEqual(W, 5760.0)

    def test_measure_steps_driven_by_stepra_when_large(self):
        # radt=2min, dt=10s -> stepra = floor(2*60/10+0.5) = floor(12.5)=12 -> 2*stepra=24 < 30
        W, stepra, measure_steps = cs.compute_window_seconds(0, 10.0, 2.0)
        self.assertEqual(stepra, 12)
        self.assertEqual(measure_steps, 30)
        # radt=20min, dt=10s -> stepra = floor(20*60/10+0.5)=120 -> 2*stepra=240 > 30
        W2, stepra2, measure_steps2 = cs.compute_window_seconds(0, 10.0, 20.0)
        self.assertEqual(stepra2, 120)
        self.assertEqual(measure_steps2, 240)
        self.assertEqual(W2, 240 * 10.0)


# ---------------------------------------------------------------------------
# 6. wrf window namelist: key-count invariants; absent history keys not added
# ---------------------------------------------------------------------------

REAL_NAMELIST_TEXT = """\
 &time_control
 run_days                            = 0,
 run_hours                           = 0,
 run_minutes                         = 0,
 run_seconds                         = 2160,
 start_year                          = 2019,
 start_month                         = 11,
 start_day                           = 26,
 start_hour                          = 15,
 start_minute                        = 00,
 start_second                        = 00,
 end_year                            = 2019,
 end_month                           = 11,
 end_day                             = 26,
 end_hour                            = 15,
 end_minute                          = 36,
 end_second                          = 00,
 interval_seconds                    = 10800
 history_interval_m                  = 60,
 restart                             = .false.,
 restart_interval                    = 660,
 io_form_history                     = 2,
 /

 &domains
 time_step                           = 72,
 max_dom                             = 1,
 e_we                                = 425,
 e_sn                                = 300,
 e_vert                              = 50,
 dx                                  = 12000,
 dy                                  = 12000,
 num_metgrid_levels                  = 34,
 num_metgrid_soil_levels             = 4,
 nproc_x                             = 9,
 nproc_y                             = 8,
 /

 &physics
 physics_suite                       = 'conus'
 radt                                = 15,
 /
"""


class TestWrfWindowNamelist(unittest.TestCase):
    def setUp(self):
        self.T = datetime.datetime(2019, 11, 26, 15, 0, 0)
        self.txt = cs.build_wrf_window_namelist_text(
            REAL_NAMELIST_TEXT, self.T, win_steps=31, dt_seconds=72.0,
            nproc_x=6, nproc_y=12, numtiles=1)

    def test_key_counts_exactly_one(self):
        for key in ("start_year", "start_month", "start_day", "start_hour", "end_year",
                    "run_days", "run_hours", "run_minutes", "run_seconds", "restart",
                    "restart_interval", "history_interval_m", "nproc_x", "nproc_y"):
            self.assertEqual(cs.count_key(self.txt, key), 1, key)

    def test_absent_history_keys_not_added(self):
        for key in ("history_interval_s", "history_interval", "history_interval_h",
                    "auxhist1_interval"):
            self.assertEqual(cs.count_key(REAL_NAMELIST_TEXT, key), 0)
            self.assertEqual(cs.count_key(self.txt, key), 0)

    def test_values(self):
        self.assertEqual(cs.get_first(self.txt, "restart"), ".false.")
        self.assertEqual(cs.get_first(self.txt, "restart_interval"), "999999")
        self.assertEqual(cs.get_first(self.txt, "history_interval_m"), "0")
        self.assertEqual(int(cs.get_first(self.txt, "run_seconds")), 31 * 72)
        self.assertEqual(cs.get_first(self.txt, "nproc_x"), "6")
        self.assertEqual(cs.get_first(self.txt, "nproc_y"), "12")
        # untouched key survives verbatim
        self.assertIn("physics_suite", self.txt)

    def test_no_key_dropped(self):
        keys_before = set(m for m in
                          __import__("re").findall(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*=",
                                                    REAL_NAMELIST_TEXT))
        keys_after = set(m for m in
                         __import__("re").findall(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*=",
                                                   self.txt))
        self.assertTrue(keys_before.issubset(keys_after))


# ---------------------------------------------------------------------------
# 7. score: spin-up exclusion + exact trapezoid interpolation
# ---------------------------------------------------------------------------

def _rsl_line(sim_dt, elapsed):
    return "Timing for main: time %s on domain   1:   %.5f elapsed seconds" % (
        sim_dt.strftime(_WRF_TIME_FMT), elapsed)


class TestScoreSpinupExclusion(unittest.TestCase):
    N_SPINUP = 50    # deliberately a MAJORITY of the file, so a broken/missing spin-up filter
    N_CLEAN = 39     # would flip the median to 999.0 (see test_contamination_would_flip_median)

    def _write_rsl(self, d, T, dt, spinup_seconds):
        lines = []
        # N_SPINUP "spin-up" steps, all before T+spinup, deliberately huge/wrong elapsed cost
        for i in range(1, self.N_SPINUP + 1):
            lines.append(_rsl_line(T + datetime.timedelta(seconds=dt * i), 999.0))
        # N_CLEAN post-spin-up steps, constant elapsed=1.0
        for i in range(1, self.N_CLEAN + 1):
            t = T + datetime.timedelta(seconds=spinup_seconds + dt * i)
            lines.append(_rsl_line(t, 1.0))
        win_dir = os.path.join(d, "win_0", "wrf")
        os.makedirs(win_dir)
        with open(os.path.join(win_dir, "rsl.error.0000"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_pre_spinup_steps_excluded_from_s_k(self):
        with tempfile.TemporaryDirectory() as d:
            T = datetime.datetime(2020, 1, 1, 0, 0, 0)
            dt = 72.0
            # spinup_seconds chosen so it falls strictly between the last spin-up step's sim
            # time (dt*N_SPINUP) and the first post-spin-up step's sim time (spinup+dt)
            spinup_seconds = dt * self.N_SPINUP + 1.0
            self._write_rsl(d, T, dt, spinup_seconds)

            plan = {
                "dt_s": dt, "radt_min": 100000.0,
                "stepra": predict.steps_per_radiation_call(100000.0, dt),
                "spinup_min": spinup_seconds / 60.0,
                "W_seconds": spinup_seconds + 30 * dt,
                "run_start": T.strftime(_WRF_TIME_FMT),
                "windows": [{"t": T.strftime(_WRF_TIME_FMT), "reason": "t0"}],
                "_plan_dir": d,
            }
            windows_out, pooled, anchors = cs.score_windows_from_rsl(plan, None, {})
            self.assertEqual(windows_out[0]["n_normal"], self.N_CLEAN - 1)  # 1 flagged (i==1)
            self.assertAlmostEqual(windows_out[0]["s_k"], 1.0, places=9)

    def test_contamination_would_flip_median(self):
        # sanity check that the test above is actually discriminating: with N_SPINUP=50 a
        # MAJORITY of the file, a broken/missing spin-up filter (kept = every step, not just
        # post-spinup) would flip the median away from 1.0 -- confirming the assertion above
        # would fail if score_windows_from_rsl stopped filtering by simulated time.
        unfiltered_normal = [999.0] * (self.N_SPINUP - 1) + [1.0] * self.N_CLEAN
        self.assertNotAlmostEqual(statistics.median(unfiltered_normal), 1.0, places=3)


class TestInterpolateStepCostsExactTrapezoid(unittest.TestCase):
    def test_linear_ramp_between_two_windows(self):
        anchors = [(0.0, 1.0), (5760.0, 2.0)]
        step_times = [i * 72.0 for i in range(1, 81)]  # 80 steps, last exactly at 5760.0
        s_values = cs.interpolate_step_costs(step_times, anchors)

        expected = [1.0 + (t / 5760.0) * 1.0 for t in step_times]
        for got, want in zip(s_values, expected):
            self.assertAlmostEqual(got, want, places=9)

        self.assertAlmostEqual(s_values[-1], 2.0, places=9)
        total = sum(s_values)
        expected_total = sum(expected)
        self.assertAlmostEqual(total, expected_total, places=6)
        # closed form for this linear ramp: N*s0 + (s1-s0)/T * sum(step_times)
        n = len(step_times)
        closed_form = n * 1.0 + (1.0 / 5760.0) * sum(step_times)
        self.assertAlmostEqual(total, closed_form, places=6)

    def test_held_flat_outside_anchor_range(self):
        anchors = [(100.0, 1.0), (200.0, 2.0)]
        s_values = cs.interpolate_step_costs([0.0, 50.0, 300.0], anchors)
        self.assertEqual(s_values[0], 1.0)
        self.assertEqual(s_values[1], 1.0)
        self.assertEqual(s_values[2], 2.0)


# ---------------------------------------------------------------------------
# run.sbatch: hard user constraints (no LD_PRELOAD, no wrf.exe wrapper, --exclude=head mandatory)
# ---------------------------------------------------------------------------

class TestRunSbatchHardConstraints(unittest.TestCase):
    def test_generated_sbatch_has_no_forbidden_content(self):
        plan = {
            "windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"},
                        {"t": "2019-11-26_18:00:00", "reason": "late"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertNotIn("LD_PRELOAD", text)
        self.assertNotIn("--wrap", text)
        self.assertIn("--exclude=head", text)
        self.assertIn("--signal=B:USR1@90", text)
        self.assertIn("site_launch", text)
        self.assertIn("site_hygiene", text)

    def test_extra_excluded_nodes_from_env(self):
        plan = {"windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}], "nodes": 1,
                "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1}}
        with mock.patch.dict(os.environ, {"WTUNE_SBATCH_EXCLUDE": "work5, work7"}):
            text = cs.render_run_sbatch(plan, site="scc")
        self.assertIn("#SBATCH --exclude=head,work5,work7\n", text)

    def test_plan_dir_written_as_absolute_path(self):
        # Slurm runs a copy of the script from its spool dir: BASH_SOURCE cannot locate the plan.
        plan = {
            "windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
            "_plan_dir": "/scratch/cs plan/severe2023",
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertIn("\nPLAN='/scratch/cs plan/severe2023'\n", text)
        self.assertIn("#SBATCH --output=/scratch/cs plan/severe2023/run-%j.out", text)
        self.assertNotIn("BASH_SOURCE", text)


# ---------------------------------------------------------------------------
# T7 (2026-09-24, jobs on scc): --ntasks-per-node must be ceil(ranks/nodes), not floor -- floor
# under-requests Slurm slots and real.exe's own mpirun -np ranks --map-by ppr:ceil:node then
# fails with rc=213 'not enough slots'.
# ---------------------------------------------------------------------------

class TestNtasksPerNodeIsCeil(unittest.TestCase):
    def test_90_ranks_4_nodes_requests_23_per_node(self):
        plan = {
            "windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
            "nodes": 4, "config": {"ranks": 90, "omp": 1, "nx": 9, "ny": 10, "numtiles": 1},
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertIn("--ntasks-per-node=23", text)
        self.assertIn("ppr:23:node:PE=1 --bind-to core ./real.exe", text)


# ---------------------------------------------------------------------------
# K5 (2026-09-24): ML state completion hook -- default OFF, opt-in via plan['ml_model'].
# ---------------------------------------------------------------------------

class TestMlModelHook(unittest.TestCase):
    def _plan(self, ml_model=None):
        return {
            "windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
            "ml_model": ml_model,
        }

    def test_default_off_no_ml_code_at_all(self):
        text = cs.render_run_sbatch(self._plan(ml_model=None), site="scc")
        self.assertNotIn("ML_MODEL=", text)
        self.assertNotIn("TOOL_SCM=", text)
        self.assertNotIn("GRIB_PY=", text)
        self.assertNotIn("--targets-from-report", text)
        # LAST_ML_STATUS stays initialized (so the status line is well-formed either way), but
        # the `if [ -n "${ML_MODEL:-}" ` guard means the apply step itself never runs.
        self.assertIn("LAST_ML_STATUS=\"off\"", text)

    def test_ml_model_wires_grib_python_and_apply_step(self):
        text = cs.render_run_sbatch(self._plan(ml_model="/plan/dir/report.json"), site="scc")
        self.assertIn("ML_MODEL=/plan/dir/report.json", text)
        self.assertIn("TOOL_SCM=", text)
        self.assertIn("state_correction_model.py", text)
        self.assertIn("GRIB_PY=${GRIB_PYTHON:-$HOME/wtune/envs/grib/.pixi/envs/default/bin/python}",
                     text)
        self.assertIn("\"$GRIB_PY\" \"$TOOL_SCM\" apply", text)
        self.assertIn("--targets-from-report", text)
        # never /usr/bin/python3 for the ML step specifically
        self.assertNotIn("/usr/bin/python3", text)

    def test_windows_status_records_ml_fields(self):
        text = cs.render_run_sbatch(self._plan(ml_model="/plan/dir/report.json"), site="scc")
        self.assertIn("ml=$LAST_ML_STATUS ml_targets=$LAST_ML_TARGETS", text)

    def test_read_windows_status_tolerates_trailing_ml_fields(self):
        # the parser only anchors on the leading WINDOW/rc/ok fields (no end-of-line anchor) --
        # appended ml=/ml_targets= fields must not break it.
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "windows_status.txt"), "w") as f:
                f.write("WINDOW 0 rc=0 ok=1 real=decomp wrf=ok ml=applied ml_targets=dT__low\n")
            status = cs.read_windows_status(d)
        self.assertEqual(status, {0: True})

    def test_build_plan_stores_ml_model_absolute_path(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            write_namelist(os.path.join(case_dir, "namelist.input"))
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            report_path = os.path.join(d, "report.json")
            with open(report_path, "w") as f:
                f.write("{}")
            plan = cs.build_plan(case_dir, os.path.join(d, "plan"), spinup_min=1,
                                 ml_model=report_path)
            self.assertEqual(plan["ml_model"], os.path.abspath(report_path))

    def test_build_plan_default_ml_model_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            write_namelist(os.path.join(case_dir, "namelist.input"))
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            plan = cs.build_plan(case_dir, os.path.join(d, "plan"), spinup_min=1)
            self.assertIsNone(plan["ml_model"])


# ---------------------------------------------------------------------------
# plan: end-to-end smoke test (no network -- select_source's date rule is pure/offline)
# ---------------------------------------------------------------------------

class TestBuildPlanSmoke(unittest.TestCase):
    def test_plan_writes_plan_json_and_namelist_wps(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            out_dir = os.path.join(d, "plan")
            os.makedirs(case_dir)
            write_namelist(os.path.join(case_dir, "namelist.input"),
                            sy=2020, smo=1, sd=1, sh=0, ey=2020, emo=1, ed=2, eh=0)
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            plan = cs.build_plan(case_dir, out_dir, k=4, nodes=1, ranks=72, omp=1,
                                 nx=9, ny=8, spinup_min=1)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "plan.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "namelist.wps")))
            self.assertGreaterEqual(len(plan["windows"]), 1)
            self.assertLessEqual(len(plan["windows"]), 4)
            reasons = {w["reason"] for w in plan["windows"]}
            self.assertIn("t0", reasons)
            for t in plan["grib_times"]:
                datetime.datetime.strptime(t, _WRF_TIME_FMT)  # parses cleanly


# ---------------------------------------------------------------------------
# T12 (2026-09-24, job 965): vertical_grid must be read at PLAN time (login node, netCDF4/numpy
# available) and stored in plan.json -- fix-real-namelist (batch time, /usr/bin/python3, no
# numpy) only reads it back (see TestNoNumpyOnComputeNode's
# test_fix_real_namelist_applies_vertical_grid_from_plan_json_with_numpy_blocked for that half).
# ---------------------------------------------------------------------------

try:
    import netCDF4 as _netCDF4_probe   # noqa: F401
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


class TestGenerateVtableGfsPrmsl(unittest.TestCase):
    # A realistic slice of WPS 4.6's Vtable.GFS around the PMSL entry: the active line maps from
    # MSLET (GRIB1 param 130 / GRIB2 0-3-192), and a commented-out PRMSL alternative (GRIB1 param
    # 2 / GRIB2 0-3-1) sits right below it -- exactly the shape T13's cluster finding describes.
    FIXTURE = (
        "GRIB1| Level| From | To  | metgrid  | metgrid | metgrid                                 "
        "|GRIB2|GRIB2|GRIB2|GRIB2|\n"
        "Param| Type |Level1|Level2| Name    | Units   | Description                             "
        "|Discp|Catgy|Param|Level|\n"
        "-----+------+------+------+----------+---------+-----------------------------------------"
        "+-----+-----+-----+-----+\n"
        "  130 | 102  |   0  |      | PMSL     | Pa      | Sea-level Pressure                      "
        "|  0  |  3  | 192 | 101 |\n"
        "#  2 | 102  |   0  |      | PMSL     | Pa      | Sea-level Pressure                      "
        "|  0  |  3  |  1  | 101 |\n"
        "  011 | 100  |      |      | TT       | K       | Temperature                             "
        "|  0  |  0  |  0  | 100 |\n"
    )

    def test_exactly_one_active_pmsl_line_param_2_and_0_3_1(self):
        out = cs.generate_vtable_gfs_prmsl(self.FIXTURE)
        active_pmsl = [ln for ln in out.split("\n")
                       if "PMSL" in ln and ln.strip() and not ln.strip().startswith("#")]
        self.assertEqual(len(active_pmsl), 1, active_pmsl)
        cols = [c.strip() for c in active_pmsl[0].split("|")]
        self.assertEqual(cols[0], "2")     # GRIB1 param
        self.assertEqual(cols[7], "0")     # GRIB2 discipline
        self.assertEqual(cols[8], "3")     # GRIB2 category
        self.assertEqual(cols[9], "1")     # GRIB2 param
        self.assertNotIn("192", active_pmsl[0])

    def test_other_lines_untouched(self):
        out = cs.generate_vtable_gfs_prmsl(self.FIXTURE)
        self.assertIn("TT", out)
        self.assertIn("-----+------+------+------+", out)
        # the original commented PRMSL line is still there too, verbatim (harmless leftover)
        self.assertIn("#  2 | 102", out)

    def test_no_active_pmsl_line_raises(self):
        no_active = self.FIXTURE.replace("  130 | 102", "#  130 | 102")
        with self.assertRaises(cs.ColdstartError):
            cs.generate_vtable_gfs_prmsl(no_active)

    def test_two_active_pmsl_lines_raises(self):
        two_active = self.FIXTURE.replace("#  2 | 102", "   2 | 102")
        with self.assertRaises(cs.ColdstartError):
            cs.generate_vtable_gfs_prmsl(two_active)


@unittest.skipUnless(HAS_NETCDF4, "netCDF4 not installed in this dev environment")
class TestReadVerticalGridOrWarn(unittest.TestCase):
    def test_reads_znw_and_p_top_into_plan_json_schema(self):
        import netCDF4
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wrfinput_d01")
            ds = netCDF4.Dataset(path, "w")
            ds.createDimension("Time", None)
            ds.createDimension("bottom_top_stag", 4)
            znw = ds.createVariable("ZNW", "f4", ("Time", "bottom_top_stag"))
            znw[0] = [1.0, 0.6, 0.3, 0.0]
            p_top = ds.createVariable("P_TOP", "f4", ("Time",))
            p_top[0] = 5000.0
            ds.close()

            vg, warning = cs.read_vertical_grid_or_warn(path)
            self.assertIsNone(warning)
            self.assertEqual(vg, {"eta_levels": [1.0, 0.6, 0.3, 0.0], "p_top_pa": 5000.0})
            # exactly what build_plan/cmd_rebuild_plan store into plan.json -- round-trips clean
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"vertical_grid": vg}, f)
                plan_json_path = f.name
            with open(plan_json_path) as f:
                reloaded = json.load(f)
            self.assertEqual(reloaded["vertical_grid"], vg)
            os.remove(plan_json_path)

    def test_no_wrfinput_is_not_a_failure(self):
        vg, warning = cs.read_vertical_grid_or_warn(None)
        self.assertIsNone(vg)
        self.assertIsNone(warning)
        vg, warning = cs.read_vertical_grid_or_warn("/does/not/exist/wrfinput_d01")
        self.assertIsNone(vg)
        self.assertIsNone(warning)

    def test_unreadable_wrfinput_returns_loud_warning_not_silent_none(self):
        # T12: a wrfinput that EXISTS but can't yield ZNW/P_TOP must come back with a non-None
        # warning string (for the plan to record loudly), not just a quiet None.
        import netCDF4
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wrfinput_d01")
            ds = netCDF4.Dataset(path, "w")
            ds.createDimension("Time", None)
            ds.close()   # no ZNW, no P_TOP at all

            vg, warning = cs.read_vertical_grid_or_warn(path)
            self.assertIsNone(vg)
            self.assertIsNotNone(warning)
            self.assertIn("ZNW", warning)


# ---------------------------------------------------------------------------
# Item 1 [T]: per-window namelist.wps time gaps -- fix-wps-namelist must narrow &share
# start_date/end_date to exactly that window's own [t, t_end], not the whole run's min/max.
# ---------------------------------------------------------------------------

class TestFixWpsNamelistPerWindow(unittest.TestCase):
    def _base_fields(self):
        attrs = {"MAP_PROJ": 1.0, "TRUELAT1": 30.0, "TRUELAT2": 60.0, "STAND_LON": -98.0,
                 "CEN_LAT": 40.00001, "CEN_LON": -98.0,
                 "WEST-EAST_GRID_DIMENSION": 425.0, "SOUTH-NORTH_GRID_DIMENSION": 300.0,
                 "DX": 12000.0, "DY": 12000.0}
        nl = {"domains": {"e_we": [425], "e_sn": [300], "dx": [12000.0], "max_dom": [1]}}
        geo = cs.geo_from_wrfinput_attrs(attrs, nl)
        # deliberately span the WHOLE run (this is what build_plan's own PLAN/namelist.wps
        # holds) so the per-window narrowing is a real, observable change.
        return cs.build_namelist_wps_fields(
            nl, geo, datetime.datetime(2020, 1, 1, 0, 0, 0),
            datetime.datetime(2020, 1, 4, 0, 0, 0), 10800)

    def test_wps_fields_for_window_exact_span(self):
        base = self._base_fields()
        T = datetime.datetime(2020, 1, 1, 0, 0, 0)
        T_end = datetime.datetime(2020, 1, 1, 1, 36, 0)
        fields = cs.wps_fields_for_window(base, T, T_end)
        self.assertEqual(fields["start_date"], "2020-01-01_00:00:00")
        self.assertEqual(fields["end_date"], "2020-01-01_01:36:00")
        # every other field (projection, per-domain geometry) is untouched
        for key in ("map_proj", "ref_lat", "ref_lon", "e_we", "e_sn", "dx", "dy"):
            self.assertEqual(fields[key], base[key])

    def test_two_windows_0h_and_48h_each_cover_exactly_their_own_span(self):
        base = self._base_fields()
        windows = [
            {"t": "2020-01-01_00:00:00", "t_end": "2020-01-01_01:36:00", "reason": "t0"},
            {"t": "2020-01-03_00:00:00", "t_end": "2020-01-03_01:36:00", "reason": "late"},  # +48h
        ]
        with tempfile.TemporaryDirectory() as d:
            plan_dir = os.path.join(d, "plan")
            os.makedirs(os.path.join(plan_dir, "wps"))
            plan = {"windows": windows, "namelist_wps_fields": base}
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)

            written = []
            for win in (0, 1):
                fields = cs.cmd_fix_wps_namelist(plan_dir, win)
                written.append(fields)
                with open(os.path.join(plan_dir, "wps", "namelist.wps")) as f:
                    text = f.read()
                nlw = predict.parse_namelist(os.path.join(plan_dir, "wps", "namelist.wps"))
                self.assertEqual(predict.g(nlw, "share", "start_date"), windows[win]["t"])
                self.assertEqual(predict.g(nlw, "share", "end_date"), windows[win]["t_end"])
                self.assertIn(windows[win]["t"], text)
                self.assertIn(windows[win]["t_end"], text)

            # the two windows' spans are genuinely different (not e.g. both silently the
            # whole-run span, which is exactly the bug this fixes)
            self.assertNotEqual(written[0]["start_date"], written[1]["start_date"])
            self.assertNotEqual(written[0]["end_date"], written[1]["end_date"])
            self.assertEqual(written[0]["start_date"], "2020-01-01_00:00:00")
            self.assertEqual(written[1]["start_date"], "2020-01-03_00:00:00")


# ---------------------------------------------------------------------------
# Item 2 [T]: nests -- real dir links met_em for every domain, wrf dir links every
# wrfinput_d0N + wrfbdy_d01 + (conditionally) wrflowinp_d0N.
# ---------------------------------------------------------------------------

class TestNestsInRunSbatch(unittest.TestCase):
    def test_max_dom_2_rendered_sbatch_links_every_domain(self):
        plan = {
            "windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
            "max_dom": 2,
        }
        text = cs.render_run_sbatch(plan, site="scc")
        # real dir: link command covers every domain's met_em, not just d01 (met_em.d01.*.nc may
        # still appear elsewhere, e.g. metgrid's own post-step existence sentinel check)
        self.assertIn('ln -sf "$PLAN"/wps/met_em.d0*.*.nc .', text)
        self.assertNotIn('ln -sf "$PLAN"/wps/met_em.d01.*.nc .', text)  # old d01-only link gone
        # wrf dir: every wrfinput_d0N + wrfbdy_d01 (only d01 has a time-varying boundary)
        self.assertIn("wrfinput_d0*", text)
        self.assertIn("wrfbdy_d01", text)
        # wrflowinp_d0N linked only when present (nullglob-guarded, never a hard requirement)
        self.assertIn("wrflowinp_d0*", text)
        self.assertIn("WRFLOWINP", text)


# ---------------------------------------------------------------------------
# Item 3: TOOL path -- run.sbatch must reference the actual installed script location, not the
# nonexistent "$PLAN/../coldstart_sample.py".
# ---------------------------------------------------------------------------

class TestToolPathInSbatch(unittest.TestCase):
    def test_tool_is_absolute_path_to_this_script(self):
        plan = {
            "windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertNotIn('$PLAN/../coldstart_sample.py', text)
        self.assertIn(cs._SELF_PATH, text)
        self.assertTrue(os.path.isabs(cs._SELF_PATH))
        self.assertIn('"$TOOL"', text)


# ---------------------------------------------------------------------------
# Item 4 [T]: numpy is absent on scc's compute-node python3. Importing coldstart_sample, and
# every compute-node-facing subcommand, must not require it (metdata/grib_state are lazy and
# only needed by inspect/plan --state/fetch).
# ---------------------------------------------------------------------------

class TestNoNumpyOnComputeNode(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("numpy", "metdata", "grib_state")}
        for k in ("numpy", "metdata", "grib_state"):
            sys.modules.pop(k, None)
        sys.modules["numpy"] = None   # sys.modules[name] = None makes `import name` raise ImportError

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_import_succeeds_with_numpy_blocked(self):
        sys.modules.pop("coldstart_sample", None)
        import coldstart_sample as cs2   # noqa: F401  -- must not raise
        self.assertIsNotNone(cs2)

    def test_fix_wps_namelist_does_not_import_numpy(self):
        with tempfile.TemporaryDirectory() as d:
            plan_dir = os.path.join(d, "plan")
            os.makedirs(os.path.join(plan_dir, "wps"))
            fields = {"max_dom": 1, "start_date": "x", "end_date": "y", "map_proj": "lambert",
                     "ref_lat": 1.0, "ref_lon": 1.0, "dx": 1.0, "dy": 1.0,
                     "geog_data_res": ["default"], "geog_data_path": "/tmp",
                     "parent_id": [1], "parent_grid_ratio": [1], "i_parent_start": [1],
                     "j_parent_start": [1], "e_we": [10], "e_sn": [10], "interval_seconds": 10800}
            plan = {"windows": [{"t": "2020-01-01_00:00:00", "t_end": "2020-01-01_01:00:00",
                                 "reason": "t0"}],
                    "namelist_wps_fields": fields}
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)
            cs.cmd_fix_wps_namelist(plan_dir, 0)   # must not raise ImportError
        self.assertNotIn("metdata", sys.modules)
        self.assertNotIn("grib_state", sys.modules)

    def test_fix_real_and_wrf_namelist_do_not_import_numpy(self):
        with tempfile.TemporaryDirectory() as d:
            case_nl = os.path.join(d, "namelist.input")
            write_namelist(case_nl)
            plan_dir = os.path.join(d, "plan")
            real_dir = os.path.join(plan_dir, "win_0", "real")
            wrf_dir = os.path.join(plan_dir, "win_0", "wrf")
            os.makedirs(real_dir)
            os.makedirs(wrf_dir)
            open(os.path.join(real_dir, "met_em.d01.2019-11-26_15:00:00.nc"), "w").close()
            plan = {
                "namelist_input": case_nl,
                "windows": [{"t": "2019-11-26_15:00:00", "t_end": "2019-11-26_18:00:00",
                            "reason": "t0"}],
                "measure_steps": 30, "spinup_min": 1, "dt_s": 72.0,
            }
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)
            with mock.patch.object(cs, "find_ncdump", return_value="/bin/true"), \
                 mock.patch.object(cs, "_read_met_em_dims", return_value=(34, 4, 28)):
                cs.cmd_fix_real_namelist(plan_dir, 0)   # must not raise ImportError
                cs.cmd_fix_wrf_namelist(plan_dir, 0, 6, 12, 1)   # must not raise ImportError
        self.assertNotIn("metdata", sys.modules)
        self.assertNotIn("grib_state", sys.modules)

    def test_fix_real_namelist_applies_vertical_grid_from_plan_json_with_numpy_blocked(self):
        # T12 (2026-09-24, job 965): plan.json['vertical_grid'] (read at PLAN time, on the login
        # node) must be APPLIED here at batch time using only stdlib -- no numpy/netCDF4 import,
        # confirmed by this test's own numpy-blocked setUp.
        with tempfile.TemporaryDirectory() as d:
            case_nl = os.path.join(d, "namelist.input")
            write_namelist(case_nl, extra="""
 &domains
 dzbot                               = 30,
 dzstretch_s                         = 1.11,
 dzstretch_u                         = 1.10,
 auto_levels_opt                     = 2,
 p_top_requested                     = 5000,
 /
""")
            plan_dir = os.path.join(d, "plan")
            real_dir = os.path.join(plan_dir, "win_0", "real")
            os.makedirs(real_dir)
            open(os.path.join(real_dir, "met_em.d01.2019-11-26_15:00:00.nc"), "w").close()
            plan = {
                "namelist_input": case_nl,
                "windows": [{"t": "2019-11-26_15:00:00", "t_end": "2019-11-26_18:00:00",
                            "reason": "t0"}],
                "measure_steps": 30, "spinup_min": 1, "dt_s": 72.0, "max_dom": 1,
                "vertical_grid": {"eta_levels": [1.0, 0.6, 0.3, 0.0], "p_top_pa": 4000.0},
            }
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)
            with mock.patch.object(cs, "find_ncdump", return_value="/bin/true"), \
                 mock.patch.object(cs, "_read_met_em_dims", return_value=(34, 4, 28)):
                cs.cmd_fix_real_namelist(plan_dir, 0)   # must not raise ImportError
            with open(os.path.join(real_dir, "namelist.input")) as f:
                txt = f.read()
        self.assertEqual(cs.get_first(txt, "eta_levels"), "1.0")
        self.assertEqual(cs.get_first(txt, "e_vert"), "4")
        self.assertEqual(cs.get_first(txt, "p_top_requested"), "4000.0")
        for key in ("dzbot", "dzstretch_s", "dzstretch_u", "auto_levels_opt"):
            self.assertEqual(cs.count_key(txt, key), 0)
        self.assertNotIn("metdata", sys.modules)
        self.assertNotIn("grib_state", sys.modules)

    def test_stage_does_not_import_numpy(self):
        with tempfile.TemporaryDirectory() as d:
            plan_dir = os.path.join(d, "plan")
            os.makedirs(plan_dir)
            with open(os.path.join(plan_dir, "namelist.wps"), "w") as f:
                f.write(NAMELIST_WPS_TMPL)
            plan = {"windows": [{"t": "2020-01-01_00:00:00", "reason": "t0"}],
                    "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1}}
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)
            saved_env = {k: os.environ.pop(k, None) for k in ("WPS_DIR", "WRF_TREE")}
            try:
                cs.stage_plan(plan_dir, site="scc")   # must not raise ImportError
            finally:
                for k, v in saved_env.items():
                    if v is not None:
                        os.environ[k] = v
        self.assertNotIn("metdata", sys.modules)
        self.assertNotIn("grib_state", sys.modules)

    def test_score_does_not_import_numpy(self):
        with tempfile.TemporaryDirectory() as d:
            case_nl = os.path.join(d, "namelist.input")
            write_namelist(case_nl)
            plan_dir = os.path.join(d, "plan")
            wrf_dir = os.path.join(plan_dir, "win_0", "wrf")
            os.makedirs(wrf_dir)
            T = datetime.datetime(2019, 11, 26, 15, 0, 0)
            lines = [_rsl_line(T + datetime.timedelta(seconds=72 * i), 1.0) for i in range(1, 35)]
            with open(os.path.join(wrf_dir, "rsl.error.0000"), "w") as f:
                f.write("\n".join(lines) + "\n")
            geo = {"ref_lat": 40.0, "ref_lon": -98.0, "dx": 12000.0, "dy": 12000.0,
                   "e_we": 425, "e_sn": 300}
            plan = {
                "namelist_input": case_nl, "case_dir": "/does/not/exist",
                "dt_s": 72.0, "radt_min": 15.0,
                "stepra": predict.steps_per_radiation_call(15.0, 72.0),
                "spinup_min": 1.0, "W_seconds": 60.0 + 30 * 72.0,
                "run_start": T.strftime(_WRF_TIME_FMT),
                "windows": [{"t": T.strftime(_WRF_TIME_FMT), "reason": "t0"}],
                "geo": geo, "nodes": 1,
                "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
            }
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)
            cs.score_plan(plan_dir)   # must not raise ImportError (wtune_predict is stdlib-only)
        self.assertNotIn("metdata", sys.modules)
        self.assertNotIn("grib_state", sys.modules)


# ---------------------------------------------------------------------------
# Item 5: fetch's default --cache-dir is PLAN/grib (matches run.sbatch's own
# `"$PLAN"/grib/*.grib2` link_grib.csh argument), and the old GRIB_DIR fallback is gone.
# ---------------------------------------------------------------------------

class TestFetchCacheDirAndGribFallback(unittest.TestCase):
    def test_default_cache_dir_resolves_under_plan(self):
        with tempfile.TemporaryDirectory() as d:
            plan_dir = os.path.join(d, "plan")
            os.makedirs(plan_dir)
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump({"windows": []}, f)
            captured = {}
            with mock.patch.object(cs, "resolve_and_fetch_forcing",
                                   side_effect=lambda plan, plan_dir_arg, cache_dir, **kw:
                                   captured.update(cache_dir=cache_dir)):
                rc = cs.main(["fetch", "--plan", plan_dir])
            self.assertEqual(rc, 0)
            self.assertEqual(captured["cache_dir"], os.path.join(plan_dir, "grib"))

    def test_explicit_cache_dir_overrides_default(self):
        with tempfile.TemporaryDirectory() as d:
            plan_dir = os.path.join(d, "plan")
            os.makedirs(plan_dir)
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump({"windows": []}, f)
            captured = {}
            with mock.patch.object(cs, "resolve_and_fetch_forcing",
                                   side_effect=lambda plan, plan_dir_arg, cache_dir, **kw:
                                   captured.update(cache_dir=cache_dir)):
                cs.main(["fetch", "--plan", plan_dir, "--cache-dir", "/explicit/dir"])
            self.assertEqual(captured["cache_dir"], "/explicit/dir")

    def test_run_sbatch_has_no_grib_dir_fallback(self):
        plan = {
            "windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertNotIn("GRIB_DIR", text)
        self.assertIn('"$PLAN"/grib/*.grib2', text)
        # T9 (2026-09-24): the fallback glob (used only when grib_manifest.txt is missing) also
        # covers FNL's .grib and ncei_g3/g4's .grb -- the primary path reads the exact file list
        # `fetch` wrote instead of a glob.
        self.assertIn('"$PLAN"/grib/*.grib', text)
        self.assertIn('"$PLAN"/grib/*.grb', text)
        self.assertIn("grib_manifest.txt", text)
        self.assertIn("no forcing files", text)


# ---------------------------------------------------------------------------
# Item 6 [T]: score must not read the case's own wrfinput -- build_prediction's case_dir must be
# PLAN/win_0/real (the generated wrfinput), never plan["case_dir"].
# ---------------------------------------------------------------------------

class TestScoreDoesNotLeakCaseDir(unittest.TestCase):
    def test_build_prediction_receives_win0_real_not_case_dir(self):
        with tempfile.TemporaryDirectory() as d:
            case_nl = os.path.join(d, "namelist.input")
            write_namelist(case_nl)
            plan_dir = os.path.join(d, "plan")
            wrf_dir = os.path.join(plan_dir, "win_0", "wrf")
            os.makedirs(wrf_dir)
            T = datetime.datetime(2019, 11, 26, 15, 0, 0)
            lines = [_rsl_line(T + datetime.timedelta(seconds=72 * i), 1.0) for i in range(1, 35)]
            with open(os.path.join(wrf_dir, "rsl.error.0000"), "w") as f:
                f.write("\n".join(lines) + "\n")
            geo = {"ref_lat": 40.0, "ref_lon": -98.0, "dx": 12000.0, "dy": 12000.0,
                   "e_we": 425, "e_sn": 300}
            leakage_case_dir = "/definitely/not/used/" + d
            plan = {
                "namelist_input": case_nl, "case_dir": leakage_case_dir,
                "dt_s": 72.0, "radt_min": 15.0,
                "stepra": predict.steps_per_radiation_call(15.0, 72.0),
                "spinup_min": 1.0, "W_seconds": 60.0 + 30 * 72.0,
                "run_start": T.strftime(_WRF_TIME_FMT),
                "windows": [{"t": T.strftime(_WRF_TIME_FMT), "reason": "t0"}],
                "geo": geo, "nodes": 1,
                "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
            }
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)

            captured = {}
            real_build_prediction = predict.build_prediction

            def spy(namelist_path, case_dir, nodes, config, **kwargs):
                captured["case_dir"] = case_dir
                return real_build_prediction(namelist_path, case_dir, nodes, config, **kwargs)

            with mock.patch.object(predict, "build_prediction", side_effect=spy):
                cs.score_plan(plan_dir)

            self.assertEqual(captured["case_dir"], os.path.join(plan_dir, "win_0", "real"))
            self.assertNotEqual(captured["case_dir"], leakage_case_dir)


# ---------------------------------------------------------------------------
# Item 7: a failed window is recorded ok=0 and skipped by score, not treated as a hard error;
# score errors only when every window is unusable.
# ---------------------------------------------------------------------------

class TestWindowsStatusSkipping(unittest.TestCase):
    def _plan_with_two_windows(self, d):
        T0 = datetime.datetime(2020, 1, 1, 0, 0, 0)
        T1 = datetime.datetime(2020, 1, 2, 0, 0, 0)
        for k, T in enumerate((T0, T1)):
            wrf_dir = os.path.join(d, "win_%d" % k, "wrf")
            os.makedirs(wrf_dir)
            lines = [_rsl_line(T + datetime.timedelta(seconds=72 * i), float(k + 1))
                    for i in range(1, 35)]
            with open(os.path.join(wrf_dir, "rsl.error.0000"), "w") as f:
                f.write("\n".join(lines) + "\n")
        return {
            "dt_s": 72.0, "radt_min": 100000.0,
            "stepra": predict.steps_per_radiation_call(100000.0, 72.0),
            "spinup_min": 1.0, "W_seconds": 60.0 + 30 * 72.0,
            "run_start": T0.strftime(_WRF_TIME_FMT),
            "windows": [{"t": T0.strftime(_WRF_TIME_FMT), "reason": "t0"},
                       {"t": T1.strftime(_WRF_TIME_FMT), "reason": "late"}],
            "_plan_dir": d,
        }

    def test_ok0_window_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            plan = self._plan_with_two_windows(d)
            with open(os.path.join(d, "windows_status.txt"), "w") as f:
                f.write("WINDOW 0 rc=1 ok=0\nWINDOW 1 rc=0 ok=1\n")
            windows_out, pooled, anchors = cs.score_windows_from_rsl(plan, None, {})
            self.assertEqual(len(windows_out), 1)
            self.assertEqual(windows_out[0]["t"], plan["windows"][1]["t"])

    def test_all_ok0_raises(self):
        with tempfile.TemporaryDirectory() as d:
            plan = self._plan_with_two_windows(d)
            with open(os.path.join(d, "windows_status.txt"), "w") as f:
                f.write("WINDOW 0 rc=1 ok=0\nWINDOW 1 rc=1 ok=0\n")
            with self.assertRaises(cs.ColdstartError):
                cs.score_windows_from_rsl(plan, None, {})

    def test_interrupted_suffix_still_parses_as_ok0(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "windows_status.txt"), "w") as f:
                f.write("WINDOW 0 rc=1 ok=0 INTERRUPTED sig=USR1\n")
            status = cs.read_windows_status(d)
            self.assertEqual(status, {0: False})

    def test_no_status_file_means_all_usable(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cs.read_windows_status(d), {})


# ---------------------------------------------------------------------------
# Item 8: USR1/TERM/INT trap kills the in-flight WRF, reaps, records INTERRUPTED, and exits 3;
# the EXIT trap alone must not double-run (or skip) the reap.
# ---------------------------------------------------------------------------

class TestUsr1TrapExitsThree(unittest.TestCase):
    def test_sbatch_signal_handling_shape(self):
        plan = {
            "windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertIn("trap 'on_signal USR1' USR1", text)
        self.assertIn("trap 'on_signal TERM' TERM", text)
        self.assertIn("trap 'on_signal INT' INT", text)
        self.assertIn("trap on_exit EXIT", text)
        self.assertIn("exit 3", text)
        self.assertIn("INTERRUPTED", text)
        self.assertIn("SIGNAL_TRAP_DONE", text)
        # the EXIT trap must be guarded against re-running after a signal already handled it
        m = re.search(r"on_exit\(\) \{(.*?)\n\}", text, re.S)
        self.assertIsNotNone(m)
        self.assertIn("SIGNAL_TRAP_DONE", m.group(1))


# ---------------------------------------------------------------------------
# Item 9: --measure-steps was parsed and silently ignored -- removed entirely.
# ---------------------------------------------------------------------------

class TestMeasureStepsFlagRemoved(unittest.TestCase):
    def test_plan_subcommand_rejects_measure_steps(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            write_namelist(os.path.join(case_dir, "namelist.input"))
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            with self.assertRaises(SystemExit):
                cs.main(["plan", "--case-dir", case_dir, "--out", os.path.join(d, "plan"),
                        "--measure-steps", "auto"])


# ---------------------------------------------------------------------------
# Item 10 [T]: inspect must report inputs that neither GFS (Vtable.GFS) nor the site's
# WPS_GEOG can supply -- blockers for genuinely unsupported physics options, warnings ("filled
# differently: ...") for options that are completable but not quite what the case asked for.
# The geog check reads an actual directory listing (never a hardcoded dataset-name list).
# ---------------------------------------------------------------------------

SCC_GEOG_LISTING = (
    "albedo_modis", "greenfrac_fpar_modis", "lai_modis_10m", "lai_modis_30s",
    "maxsnowalb_modis", "modis_landuse_20class_30s_with_lakes",
    "orogwd_10m", "orogwd_1deg", "orogwd_20m", "orogwd_2deg", "orogwd_30m",
    "soiltemp_1deg", "soiltype_bot_30s", "soiltype_top_30s", "topo_gmted2010_30s",
    "varsso", "varsso_10m", "varsso_2m", "varsso_5m",
)


def _make_geog_dir(tmpdir, names=SCC_GEOG_LISTING):
    geog = os.path.join(tmpdir, "WPS_GEOG")
    os.makedirs(geog)
    for n in names:
        os.makedirs(os.path.join(geog, n))
    return geog


def _capability_notes(rows):
    return [note for (item, _ok, _how, note) in rows if item == "capability"]


class TestGeogDirectoryListing(unittest.TestCase):
    def test_listing_is_lowercased_and_real(self):
        with tempfile.TemporaryDirectory() as d:
            geog = _make_geog_dir(d)
            names = cs.geog_directory_listing(geog)
            self.assertIn("modis_landuse_20class_30s_with_lakes", names)
            self.assertIn("albedo_modis", names)
            self.assertNotIn("lake_depth", names)

    def test_missing_directory_is_none_not_empty(self):
        self.assertIsNone(cs.geog_directory_listing("/no/such/path/at/all"))

    def test_has_dataset_true_false_and_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            geog = _make_geog_dir(d)
            names = cs.geog_directory_listing(geog)
            self.assertTrue(cs.geog_has_dataset(names, "modis", "landuse"))
            self.assertFalse(cs.geog_has_dataset(names, "lake_depth"))
            self.assertIsNone(cs.geog_has_dataset(None, "lake_depth"))

    def test_detect_geog_landuse_modis_with_lakes(self):
        with tempfile.TemporaryDirectory() as d:
            names = cs.geog_directory_listing(_make_geog_dir(d))
            name, cat = cs.detect_geog_landuse(names)
            self.assertEqual(name, "modis_landuse_20class_30s_with_lakes")
            self.assertEqual(cat, 21)

    def test_detect_geog_landuse_usgs(self):
        with tempfile.TemporaryDirectory() as d:
            names = cs.geog_directory_listing(_make_geog_dir(d, names=("usgs_landuse_30s",)))
            name, cat = cs.detect_geog_landuse(names)
            self.assertEqual(name, "usgs_landuse_30s")
            self.assertEqual(cat, 24)

    def test_detect_geog_landuse_none_present(self):
        with tempfile.TemporaryDirectory() as d:
            names = cs.geog_directory_listing(_make_geog_dir(d, names=("albedo_modis",)))
            self.assertEqual(cs.detect_geog_landuse(names), (None, None))


class TestInspectCapabilityBlockers(unittest.TestCase):
    def _inspect(self, d, extra="", geog_names=SCC_GEOG_LISTING):
        write_namelist(os.path.join(d, "namelist.input"), extra=extra)
        write_namelist_wps(os.path.join(d, "namelist.wps"))
        geog = _make_geog_dir(d, names=geog_names) if geog_names is not None else \
            os.path.join(d, "no_such_geog")
        return cs.inspect_case(d, geog_path=geog)

    def test_sf_lake_physics_without_lake_depth_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&physics\n sf_lake_physics = 1,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            self.assertTrue(any("sf_lake_physics=1" in b and "lake_depth" in b for b in blockers),
                            blockers)

    def test_sf_lake_physics_with_lake_depth_present_is_not_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&physics\n sf_lake_physics = 1,\n /\n"
            names = SCC_GEOG_LISTING + ("lake_depth_30s",)
            rows, blockers, nl = self._inspect(d, extra=extra, geog_names=names)
            self.assertFalse(any("lake_depth" in b for b in blockers), blockers)

    def test_sf_urban_physics_bep_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&physics\n sf_urban_physics = 2,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            self.assertTrue(any("sf_urban_physics=2" in b for b in blockers), blockers)

    def test_sf_urban_physics_bem_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&physics\n sf_urban_physics = 3,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            self.assertTrue(any("sf_urban_physics=3" in b for b in blockers), blockers)

    def test_sf_urban_physics_slab_scheme_not_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&physics\n sf_urban_physics = 1,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            self.assertFalse(any("sf_urban_physics" in b for b in blockers), blockers)

    def test_chem_opt_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&chem\n chem_opt = 401,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            self.assertTrue(any("chem_opt=401" in b for b in blockers), blockers)

    def test_obs_nudge_opt_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&fdda\n obs_nudge_opt = 1,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            self.assertTrue(any("obs_nudge_opt=1" in b for b in blockers), blockers)


class TestInspectCapabilityWarnings(unittest.TestCase):
    def _inspect(self, d, extra="", geog_names=SCC_GEOG_LISTING):
        write_namelist(os.path.join(d, "namelist.input"), extra=extra)
        write_namelist_wps(os.path.join(d, "namelist.wps"))
        geog = _make_geog_dir(d, names=geog_names)
        return cs.inspect_case(d, geog_path=geog)

    def test_hydrometeor_warning_always_present_on_a_completable_case(self):
        with tempfile.TemporaryDirectory() as d:
            rows, blockers, nl = self._inspect(d)
            self.assertEqual(blockers, [])
            notes = _capability_notes(rows)
            self.assertTrue(any("hydrometeors and number concentrations start at zero" in n
                                for n in notes), notes)

    def test_mp_physics_28_aerosol_warning(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&physics\n mp_physics = 28,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            notes = _capability_notes(rows)
            self.assertTrue(any("mp_physics=28" in n and "aerosol" in n for n in notes), notes)

    def test_mp_physics_other_no_aerosol_warning(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&physics\n mp_physics = 6,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            notes = _capability_notes(rows)
            self.assertFalse(any("mp_physics=" in n for n in notes), notes)

    def test_num_land_cat_mismatch_warning(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&domains\n num_land_cat = 24,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            notes = _capability_notes(rows)
            self.assertTrue(any("num_land_cat=24" in n and "21 categories" in n for n in notes),
                            notes)

    def test_num_land_cat_matching_no_warning(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&domains\n num_land_cat = 21,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            notes = _capability_notes(rows)
            self.assertFalse(any("num_land_cat=" in n for n in notes), notes)

    def test_ruc_surface_with_wrong_soil_layers_warning(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&physics\n sf_surface_physics = 3,\n num_soil_layers = 6,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            notes = _capability_notes(rows)
            self.assertTrue(any("sf_surface_physics=3" in n and "num_soil_layers=6" in n
                                for n in notes), notes)

    def test_ruc_surface_with_four_soil_layers_no_warning(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&physics\n sf_surface_physics = 3,\n num_soil_layers = 4,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            notes = _capability_notes(rows)
            self.assertFalse(any("sf_surface_physics=3" in n for n in notes), notes)

    def test_grid_fdda_warning(self):
        with tempfile.TemporaryDirectory() as d:
            extra = "&fdda\n grid_fdda = 1,\n /\n"
            rows, blockers, nl = self._inspect(d, extra=extra)
            notes = _capability_notes(rows)
            self.assertTrue(any("grid_fdda=1" in n for n in notes), notes)


# ---------------------------------------------------------------------------
# Item 10, fix-real-namelist: num_land_cat overwrite from met_em must be announced (old -> new),
# not silent -- the overwrite itself is still correct and stays.
# ---------------------------------------------------------------------------

class TestFixRealNamelistAnnouncesLandCatChange(unittest.TestCase):
    def test_prints_old_to_new_when_it_changes(self):
        with tempfile.TemporaryDirectory() as d:
            case_nl = os.path.join(d, "namelist.input")
            write_namelist(case_nl, extra="&domains\n num_land_cat = 24,\n /\n")
            plan_dir = os.path.join(d, "plan")
            real_dir = os.path.join(plan_dir, "win_0", "real")
            os.makedirs(real_dir)
            open(os.path.join(real_dir, "met_em.d01.2019-11-26_15:00:00.nc"), "w").close()
            plan = {
                "namelist_input": case_nl,
                "windows": [{"t": "2019-11-26_15:00:00", "t_end": "2019-11-26_18:00:00",
                            "reason": "t0"}],
            }
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)
            with mock.patch.object(cs, "find_ncdump", return_value="/bin/true"), \
                 mock.patch.object(cs, "_read_met_em_dims", return_value=(34, 4, 21)):
                buf = []
                with mock.patch("builtins.print", side_effect=lambda *a: buf.append(" ".join(str(x) for x in a))):
                    cs.cmd_fix_real_namelist(plan_dir, 0)
            self.assertTrue(any("num_land_cat: 24 -> 21" in line for line in buf), buf)
            with open(os.path.join(real_dir, "namelist.input")) as f:
                out_text = f.read()
            self.assertEqual(cs.get_first(out_text, "num_land_cat"), "21")

    def test_no_announcement_when_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            case_nl = os.path.join(d, "namelist.input")
            write_namelist(case_nl, extra="&domains\n num_land_cat = 21,\n /\n")
            plan_dir = os.path.join(d, "plan")
            real_dir = os.path.join(plan_dir, "win_0", "real")
            os.makedirs(real_dir)
            open(os.path.join(real_dir, "met_em.d01.2019-11-26_15:00:00.nc"), "w").close()
            plan = {
                "namelist_input": case_nl,
                "windows": [{"t": "2019-11-26_15:00:00", "t_end": "2019-11-26_18:00:00",
                            "reason": "t0"}],
            }
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)
            with mock.patch.object(cs, "find_ncdump", return_value="/bin/true"), \
                 mock.patch.object(cs, "_read_met_em_dims", return_value=(34, 4, 21)):
                buf = []
                with mock.patch("builtins.print", side_effect=lambda *a: buf.append(" ".join(str(x) for x in a))):
                    cs.cmd_fix_real_namelist(plan_dir, 0)
            self.assertFalse(any("num_land_cat:" in line for line in buf), buf)


# ---------------------------------------------------------------------------
# Round 3 (2026-09-24): pre-2015 forcing fallback chain. Chain date-bands / URL patterns / FNL
# cutover / CFSR tar spans are tested directly in tools/test_metdata.py; here: cadence
# realignment, ungrib expected-file lists, per-window source choice + probe fallback, inspect's
# new chain-based (not blocked-by-default) reporting, and the run.sbatch/stage_plan wiring
# (per-window prefix/Vtable, geogrid/metgrid TBL-dir symlinks and output verification).
# ---------------------------------------------------------------------------

class TestAlignToInterval(unittest.TestCase):
    def test_03z_aligns_to_06z_on_6h_grid_shift_10800(self):
        t = datetime.datetime(2001, 6, 1, 3, 0, 0)
        aligned = cs.align_to_interval(t, 6)
        self.assertEqual(aligned, datetime.datetime(2001, 6, 1, 6, 0, 0))
        self.assertEqual((aligned - t).total_seconds(), 10800)

    def test_already_aligned_time_has_no_shift(self):
        t = datetime.datetime(2020, 1, 1, 0, 0, 0)
        self.assertEqual(cs.align_to_interval(t, 3), t)


class TestBuildPlanCadenceForFnl(unittest.TestCase):
    def test_case_start_03z_fnl_t0_window_06z_shift_10800(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            out_dir = os.path.join(d, "plan")
            os.makedirs(case_dir)
            # 2001-06-01 resolves to [d083002 (FNL)] only -- see TestForcingChainDateBands in
            # tools/test_metdata.py. run_hours=48 so the window-interval (6h) grid has enough
            # candidates past spin-up+measure.
            write_namelist(os.path.join(case_dir, "namelist.input"), sy=2001, smo=6, sd=1, sh=3,
                            ey=2001, emo=6, ed=3, eh=3, run_hours=48)
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            plan = cs.build_plan(case_dir, out_dir, k=4, nodes=1, ranks=72, omp=1, nx=9, ny=8,
                                 spinup_min=1)
            self.assertEqual(plan["forcing_chain_primary"]["dataset"], "d083002")
            self.assertEqual(plan["window_interval_seconds"], 21600)
            t0_window = next(w for w in plan["windows"] if w["reason"] == "t0")
            self.assertEqual(t0_window["t"], "2001-06-01_06:00:00")
            self.assertEqual(t0_window["shift_s"], 10800.0)
            # window interval 21600 (6h) is what's written into namelist.wps too
            self.assertEqual(plan["namelist_wps_fields"]["interval_seconds"], 21600)
            self.assertEqual(plan["namelist_wps_fields"]["prefix"], "FNL")


class TestIntervalSecondsOverrideInRealNamelist(unittest.TestCase):
    def test_interval_seconds_written_when_overridden(self):
        T = datetime.datetime(2019, 11, 26, 15, 0, 0)
        T_end = datetime.datetime(2019, 11, 26, 18, 0, 0)
        txt = cs.build_real_namelist_text(REAL_NAMELIST_TEXT, T, T_end, 34, 4, 21,
                                          interval_seconds=21600)
        self.assertEqual(cs.get_first(txt, "interval_seconds"), "21600")

    def test_interval_seconds_unchanged_when_not_overridden(self):
        T = datetime.datetime(2019, 11, 26, 15, 0, 0)
        T_end = datetime.datetime(2019, 11, 26, 18, 0, 0)
        txt = cs.build_real_namelist_text(REAL_NAMELIST_TEXT, T, T_end, 34, 4, 21)
        self.assertEqual(cs.get_first(txt, "interval_seconds"), "10800")


class TestExpectedWpsIntermediateFiles(unittest.TestCase):
    def test_6h_window_over_12h_is_three_files_with_prefix(self):
        T = datetime.datetime(2010, 1, 1, 0, 0, 0)
        T_end = datetime.datetime(2010, 1, 1, 12, 0, 0)
        files = cs.expected_wps_intermediate_files("FNL", T, T_end, 6 * 3600)
        self.assertEqual(files, ["FNL:2010-01-01_00", "FNL:2010-01-01_06", "FNL:2010-01-01_12"])


class TestChooseForcingForWindowProbeFallback(unittest.TestCase):
    def test_g4_missing_one_file_falls_back_to_fnl(self):
        T = datetime.datetime(2010, 1, 1, 0, 0, 0)
        T_end = datetime.datetime(2010, 1, 1, 6, 0, 0)
        chain = cs.resolve_forcing_chain_for(T, now=datetime.datetime(2026, 9, 24))
        self.assertEqual([s.dataset for s in chain], ["ncei_g4", "d083002"])

        skipped_one = []

        def fake_probe(entries):
            results = []
            ok = True
            for fn, urls in entries:
                if fn.startswith("ncei_g4") and not skipped_one:
                    skipped_one.append(fn)
                    results.append({"filename": fn, "chosen_url": None})
                    ok = False
                else:
                    results.append({"filename": fn, "chosen_url": urls[0]})
            return ok, results

        chosen, tried = cs.choose_forcing_for_window(T, T_end, chain, probe_fn=fake_probe)
        self.assertEqual(chosen.dataset, "d083002")
        self.assertEqual(tried[0]["dataset"], "ncei_g4")
        self.assertFalse(tried[0]["ok"])
        self.assertEqual(tried[0]["missing"], skipped_one)
        self.assertEqual(tried[1]["dataset"], "d083002")
        self.assertTrue(tried[1]["ok"])


class TestFetchCadenceMismatch(unittest.TestCase):
    def test_fetch_fails_clearly_when_chosen_source_cadence_differs_from_plan(self):
        with tempfile.TemporaryDirectory() as d:
            plan = {
                "windows": [{"t": "2010-01-01_00:00:00", "t_end": "2010-01-01_06:00:00",
                            "reason": "t0"}],
                "window_interval_seconds": 3 * 3600,   # plan assumed ncei_g4's 3h cadence
                "forcing_chain_primary": {"dataset": "ncei_g4"},
                "forcing_pin": None,
            }

            def fake_probe(entries):
                if any(fn.startswith("ncei_g4") for fn, _ in entries):
                    return False, [{"filename": fn, "chosen_url": None} for fn, _ in entries]
                return True, [{"filename": fn, "chosen_url": u[0]} for fn, u in entries]

            with self.assertRaises(cs.ColdstartError) as ctx:
                cs.resolve_and_fetch_forcing(plan, d, os.path.join(d, "grib"),
                                             probe_fn=fake_probe)
            self.assertIn("--forcing", str(ctx.exception))
            self.assertIn("d083002", str(ctx.exception))


class TestFetchAlwaysWritesRecordOnFailure(unittest.TestCase):
    # T4b (2026-09-24): a failed `fetch` must still write PLAN/forcing.json (status: 'failed',
    # every window's candidates_tried) and name the first missing URL + its final status in the
    # error, instead of leaving no trace of what was actually probed.
    def test_failure_writes_forcing_json_and_names_first_missing_url(self):
        with tempfile.TemporaryDirectory() as d:
            plan = {
                "windows": [{"t": "2010-01-01_00:00:00", "t_end": "2010-01-01_06:00:00",
                            "reason": "t0"}],
                "window_interval_seconds": None,
                "forcing_chain_primary": {"dataset": "ncei_g4"},
                "forcing_pin": None,
            }

            def fake_probe(entries):
                results = []
                for fn, urls in entries:
                    results.append({
                        "filename": fn, "chosen_url": None,
                        "checked": [{"url": urls[0], "status": None,
                                    "attempts": [{"status": 503}, {"status": 503},
                                                {"status": 404}]}]})
                return False, results

            with self.assertRaises(cs.ColdstartError) as ctx:
                cs.resolve_and_fetch_forcing(plan, d, os.path.join(d, "grib"),
                                             probe_fn=fake_probe)
            msg = str(ctx.exception)
            self.assertIn("first missing:", msg)
            self.assertIn("404", msg)

            with open(os.path.join(d, "forcing.json")) as f:
                forcing = json.load(f)
            self.assertEqual(forcing["status"], "failed")
            self.assertEqual(len(forcing["windows"]), 1)
            self.assertIsNone(forcing["windows"][0]["chosen"])
            self.assertTrue(forcing["windows"][0]["candidates_tried"])
            self.assertTrue(all("first_missing" in t
                                for t in forcing["windows"][0]["candidates_tried"]))


class TestForcingPin(unittest.TestCase):
    def test_forcing_pin_returns_single_source_chain(self):
        chain = cs.resolve_forcing_chain_for(datetime.datetime(2010, 1, 1), forcing_pin="d083002")
        self.assertEqual([s.dataset for s in chain], ["d083002"])

    def test_unknown_forcing_pin_raises(self):
        with self.assertRaises(cs.ColdstartError):
            cs.resolve_forcing_chain_for(datetime.datetime(2010, 1, 1), forcing_pin="not_a_dataset")


class TestInspectPre2015NoLongerBlockedCliExitCodes(unittest.TestCase):
    def test_2005_case_exits_0(self):
        with tempfile.TemporaryDirectory() as d:
            write_namelist(os.path.join(d, "namelist.input"), sy=2005, ey=2005)
            write_namelist_wps(os.path.join(d, "namelist.wps"))
            rc = cs.main(["inspect", "--case-dir", d])
            self.assertEqual(rc, 0)

    def test_1975_case_exits_3(self):
        with tempfile.TemporaryDirectory() as d:
            write_namelist(os.path.join(d, "namelist.input"), sy=1975, ey=1975)
            write_namelist_wps(os.path.join(d, "namelist.wps"))
            rc = cs.main(["inspect", "--case-dir", d])
            self.assertEqual(rc, 3)

    def test_2005_case_has_coarser_resolution_warning(self):
        with tempfile.TemporaryDirectory() as d:
            write_namelist(os.path.join(d, "namelist.input"), sy=2005, ey=2005)
            write_namelist_wps(os.path.join(d, "namelist.wps"))
            rows, blockers, nl = cs.inspect_case(d)
            self.assertEqual(blockers, [])
            notes = _capability_notes(rows)
            self.assertTrue(any("forcing resolution" in n and "coarser than 0.25" in n
                                for n in notes), notes)

    def test_1990_case_has_reanalysis_fallback_warning(self):
        with tempfile.TemporaryDirectory() as d:
            write_namelist(os.path.join(d, "namelist.input"), sy=1990, ey=1990)
            write_namelist_wps(os.path.join(d, "namelist.wps"))
            rows, blockers, nl = cs.inspect_case(d)
            self.assertEqual(blockers, [])
            notes = _capability_notes(rows)
            self.assertTrue(any("reanalysis_fallback" in n for n in notes), notes)


# ---------------------------------------------------------------------------
# Item H (2026-09-24, jobs 941/942): geogrid.exe/metgrid.exe MPI_Abort(errorcode=0) on a missing
# GEOGRID.TBL/METGRID.TBL, so mpirun's own rc is 0 -- run.sbatch must independently verify
# geo_em/met_em outputs + the log's 'Successful completion', and stage_plan must symlink the
# geogrid/ and metgrid/ TBL directories (never writing through an existing real directory).
# ---------------------------------------------------------------------------

class TestGeogridMetgridTblDirs(unittest.TestCase):
    def test_rendered_sbatch_has_geo_em_checks_for_max_dom_2(self):
        plan = {
            "windows": [{"t": "2019-11-26_15:00:00", "t_end": "2019-11-26_18:00:00",
                        "reason": "t0"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
            "max_dom": 2,
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertIn("geo_em.d01.nc", text)
        self.assertIn("geo_em.d02.nc", text)
        self.assertIn("Successful completion", text)
        self.assertIn("geogrid.log*", text)
        self.assertIn("metgrid.log*", text)

    def test_symlink_dir_refuses_to_overwrite_real_directory(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src")
            os.makedirs(src)
            dst = os.path.join(d, "dst")
            os.makedirs(dst)   # a REAL directory, not a symlink -- must not be replaced
            with self.assertRaises(cs.ColdstartError):
                cs._symlink_dir(src, dst)

    def test_symlink_dir_replaces_an_existing_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            src1 = os.path.join(d, "src1")
            src2 = os.path.join(d, "src2")
            os.makedirs(src1)
            os.makedirs(src2)
            dst = os.path.join(d, "dst")
            cs._symlink_dir(src1, dst)
            cs._symlink_dir(src2, dst)   # must not raise -- replaces the old symlink
            self.assertEqual(os.path.realpath(dst), os.path.realpath(src2))

    def test_stage_creates_geogrid_and_metgrid_dir_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            wps_dir = os.path.join(d, "WPS")
            os.makedirs(os.path.join(wps_dir, "geogrid"))
            os.makedirs(os.path.join(wps_dir, "metgrid"))
            os.makedirs(os.path.join(wps_dir, "ungrib", "Variable_Tables"))
            open(os.path.join(wps_dir, "ungrib", "Variable_Tables", "Vtable.GFS"), "w").close()

            case_nl = os.path.join(d, "namelist.input")
            write_namelist(case_nl)
            plan_dir = os.path.join(d, "plan")
            os.makedirs(plan_dir)
            with open(os.path.join(plan_dir, "namelist.wps"), "w") as f:
                f.write(NAMELIST_WPS_TMPL)
            plan = {"windows": [{"t": "2020-01-01_00:00:00", "t_end": "2020-01-01_03:00:00",
                                "reason": "t0"}],
                    "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
                    "max_dom": 1}
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)

            saved = {k: os.environ.get(k) for k in ("WPS_DIR", "WRF_TREE")}
            os.environ["WPS_DIR"] = wps_dir
            os.environ.pop("WRF_TREE", None)
            try:
                cs.stage_plan(plan_dir, site="scc")
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            geogrid_link = os.path.join(plan_dir, "wps", "geogrid")
            metgrid_link = os.path.join(plan_dir, "wps", "metgrid")
            self.assertTrue(os.path.islink(geogrid_link))
            self.assertTrue(os.path.islink(metgrid_link))
            self.assertEqual(os.path.realpath(geogrid_link),
                             os.path.realpath(os.path.join(wps_dir, "geogrid")))
            self.assertEqual(os.path.realpath(metgrid_link),
                             os.path.realpath(os.path.join(wps_dir, "metgrid")))

    def test_stage_generates_vtable_gfs_prmsl_from_real_vtable_gfs(self):
        # T13 (2026-09-24): stage_plan must GENERATE PLAN/wps/Vtable.GFS_PRMSL (a real file, not
        # a symlink -- it doesn't exist in $WPS_DIR at all) from the real Vtable.GFS content.
        with tempfile.TemporaryDirectory() as d:
            wps_dir = os.path.join(d, "WPS")
            os.makedirs(os.path.join(wps_dir, "ungrib", "Variable_Tables"))
            with open(os.path.join(wps_dir, "ungrib", "Variable_Tables", "Vtable.GFS"), "w") as f:
                f.write(TestGenerateVtableGfsPrmsl.FIXTURE)

            case_nl = os.path.join(d, "namelist.input")
            write_namelist(case_nl)
            plan_dir = os.path.join(d, "plan")
            os.makedirs(plan_dir)
            with open(os.path.join(plan_dir, "namelist.wps"), "w") as f:
                f.write(NAMELIST_WPS_TMPL)
            plan = {"windows": [{"t": "2020-01-01_00:00:00", "t_end": "2020-01-01_03:00:00",
                                "reason": "t0"}],
                    "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
                    "max_dom": 1}
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)

            saved = {k: os.environ.get(k) for k in ("WPS_DIR", "WRF_TREE")}
            os.environ["WPS_DIR"] = wps_dir
            os.environ.pop("WRF_TREE", None)
            try:
                cs.stage_plan(plan_dir, site="scc")
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            prmsl_path = os.path.join(plan_dir, "wps", "Vtable.GFS_PRMSL")
            self.assertFalse(os.path.islink(prmsl_path))   # a real file, not a symlink
            with open(prmsl_path) as f:
                text = f.read()
            active_pmsl = [ln for ln in text.split("\n")
                          if "PMSL" in ln and ln.strip() and not ln.strip().startswith("#")]
            self.assertEqual(len(active_pmsl), 1)
            self.assertIn("|  3  |  1  |", active_pmsl[0])

    def test_stage_restaging_never_writes_through_a_stale_symlink(self):
        # A prior stage's Vtable.GFS_PRMSL could be a leftover symlink (e.g. from an old code
        # path) -- restaging must remove it first, never open(path, 'w') straight through it
        # into whatever it points at.
        with tempfile.TemporaryDirectory() as d:
            wps_dir = os.path.join(d, "WPS")
            os.makedirs(os.path.join(wps_dir, "ungrib", "Variable_Tables"))
            with open(os.path.join(wps_dir, "ungrib", "Variable_Tables", "Vtable.GFS"), "w") as f:
                f.write(TestGenerateVtableGfsPrmsl.FIXTURE)

            case_nl = os.path.join(d, "namelist.input")
            write_namelist(case_nl)
            plan_dir = os.path.join(d, "plan")
            os.makedirs(os.path.join(plan_dir, "wps"))
            with open(os.path.join(plan_dir, "namelist.wps"), "w") as f:
                f.write(NAMELIST_WPS_TMPL)
            plan = {"windows": [{"t": "2020-01-01_00:00:00", "t_end": "2020-01-01_03:00:00",
                                "reason": "t0"}],
                    "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
                    "max_dom": 1}
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan, f)

            # a decoy file elsewhere that a stale symlink might point at
            decoy = os.path.join(d, "decoy_unrelated_file.txt")
            with open(decoy, "w") as f:
                f.write("MUST NOT BE TOUCHED")
            os.symlink(decoy, os.path.join(plan_dir, "wps", "Vtable.GFS_PRMSL"))

            saved = {k: os.environ.get(k) for k in ("WPS_DIR", "WRF_TREE")}
            os.environ["WPS_DIR"] = wps_dir
            os.environ.pop("WRF_TREE", None)
            try:
                cs.stage_plan(plan_dir, site="scc")
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            with open(decoy) as f:
                self.assertEqual(f.read(), "MUST NOT BE TOUCHED")
            prmsl_path = os.path.join(plan_dir, "wps", "Vtable.GFS_PRMSL")
            self.assertFalse(os.path.islink(prmsl_path))


# ---------------------------------------------------------------------------
# Round 4 (2026-09-24): real.exe hangs at SOME decompositions on scc (sc26p1 3x3/5x5,
# severe2023 5x5; mechanism unknown, explicitly NOT an odd/square rule) -- real.exe now launches
# with the predicted config's own decomposition under `timeout 600`, retries serial (1x1) under
# `timeout 1800` on rc=124/missing outputs, and records which path succeeded. OMP_NUM_THREADS
# must match the ground-truth runner. fetch's byte total must not double-count a shared path.
# ---------------------------------------------------------------------------

class TestRealExeLaunchAndRetry(unittest.TestCase):
    def setUp(self):
        plan = {
            "windows": [{"t": "2019-11-26_15:00:00", "t_end": "2019-11-26_18:00:00",
                        "reason": "t0"}],
            "nodes": 4, "config": {"ranks": 100, "omp": 2, "nx": 10, "ny": 10, "numtiles": 1},
            "max_dom": 1,
        }
        self.text = cs.render_run_sbatch(plan, site="scc")

    def test_real_exe_uses_config_decomposition_ranks_per_node(self):
        self.assertIn("timeout 600 mpirun -np 100 --map-by ppr:25:node:PE=1 --bind-to core "
                      "./real.exe", self.text)

    def test_serial_fallback_branch_present(self):
        self.assertIn("--serial", self.text)
        self.assertIn("timeout 1800 mpirun -np 1 --map-by ppr:1:node:PE=1 --bind-to core "
                      "./real.exe", self.text)
        self.assertIn("rc -eq 124", self.text)
        self.assertIn("real_path=serial", self.text)
        self.assertIn("real_path=decomp", self.text)

    def test_windows_status_records_real_path(self):
        self.assertIn("real=$LAST_REAL_PATH", self.text)

    def test_omp_num_threads_1_before_real_and_config_omp_before_wrf(self):
        self.assertIn("export OMP_NUM_THREADS=1", self.text)
        self.assertIn("export OMP_NUM_THREADS=2", self.text)
        self.assertIn("unset OMP_PLACES OMP_PROC_BIND", self.text)
        # OMP_NUM_THREADS=1 must come BEFORE the real.exe launch, and =2 BEFORE the actual
        # site_launch CALL (not just the module docstring's mention of the word "site_launch").
        i_omp1 = self.text.index("export OMP_NUM_THREADS=1")
        i_real = self.text.index("timeout 600 mpirun")
        i_omp2 = self.text.index("export OMP_NUM_THREADS=2")
        i_launch = self.text.index("site_launch 100 25 2 block")
        self.assertLess(i_omp1, i_real)
        self.assertLess(i_omp2, i_launch)
        self.assertLess(i_real, i_omp2)   # real.exe still ran under OMP=1, not 2


class TestFixRealNamelistNprocFromConfig(unittest.TestCase):
    def _stage(self, d, serial=False):
        case_nl = os.path.join(d, "namelist.input")
        write_namelist(case_nl)
        plan_dir = os.path.join(d, "plan")
        real_dir = os.path.join(plan_dir, "win_0", "real")
        os.makedirs(real_dir)
        open(os.path.join(real_dir, "met_em.d01.2019-11-26_15:00:00.nc"), "w").close()
        plan = {
            "namelist_input": case_nl,
            "windows": [{"t": "2019-11-26_15:00:00", "t_end": "2019-11-26_18:00:00",
                        "reason": "t0"}],
            "config": {"ranks": 100, "omp": 2, "nx": 10, "ny": 10, "numtiles": 1},
        }
        with open(os.path.join(plan_dir, "plan.json"), "w") as f:
            json.dump(plan, f)
        with mock.patch.object(cs, "find_ncdump", return_value="/bin/true"), \
             mock.patch.object(cs, "_read_met_em_dims", return_value=(34, 4, 21)):
            cs.cmd_fix_real_namelist(plan_dir, 0, serial=serial)
        with open(os.path.join(real_dir, "namelist.input")) as f:
            return f.read()

    def test_decomposed_writes_config_nx_ny(self):
        with tempfile.TemporaryDirectory() as d:
            txt = self._stage(d, serial=False)
            self.assertEqual(cs.get_first(txt, "nproc_x"), "10")
            self.assertEqual(cs.get_first(txt, "nproc_y"), "10")

    def test_serial_writes_1_1_regardless_of_config(self):
        with tempfile.TemporaryDirectory() as d:
            txt = self._stage(d, serial=True)
            self.assertEqual(cs.get_first(txt, "nproc_x"), "1")
            self.assertEqual(cs.get_first(txt, "nproc_y"), "1")


class TestFetchByteDedup(unittest.TestCase):
    def test_shared_path_counted_once_across_windows(self):
        # Two windows whose SINGLE valid time is identical -> metdata.gfs_rda_wps_entry (real,
        # unmocked) resolves both to the SAME (cycle, fhour) -> the SAME cache filename. This is
        # exactly the 2026-09-24 finding: an RDA whole-file source answering more than one
        # "need" with the same physical path must be counted once, not once per need.
        with tempfile.TemporaryDirectory() as d:
            cache_dir = os.path.join(d, "grib")
            downloaded = []

            def fake_probe(entries):
                return True, [{"filename": fn, "chosen_url": u[0]} for fn, u in entries]

            def fake_download(url, dest):
                downloaded.append(dest)
                if not os.path.exists(dest):
                    with open(dest, "wb") as f:
                        f.write(b"x" * 1500)   # stand-in for a ~1.5 GB RDA whole file

            plan = {
                "windows": [
                    {"t": "2010-01-01_00:00:00", "t_end": "2010-01-01_00:00:00", "reason": "t0"},
                    {"t": "2010-01-01_00:00:00", "t_end": "2010-01-01_00:00:00", "reason": "late"},
                ],
                "window_interval_seconds": 3 * 3600,
                "forcing_chain_primary": {"dataset": "d084001"},
                "forcing_pin": "d084001",
            }
            total_bytes, forcing = cs.resolve_and_fetch_forcing(
                plan, d, cache_dir, probe_fn=fake_probe, download_fn=fake_download)

            self.assertEqual(len(set(downloaded)), 1)   # both windows hit the same dest path
            self.assertEqual(total_bytes, 1500)          # not 3000 (2x the shared file)
            self.assertEqual(sorted(os.listdir(cache_dir)),
                             sorted(set(os.path.basename(p) for p in downloaded)))
            # T9 (2026-09-24): forcing.json['files'] and grib_manifest.txt must both record the
            # exact basenames -- run.sbatch's link_grib.csh step reads the manifest instead of
            # assuming a *.grib2 extension.
            self.assertEqual(forcing["files"], sorted(set(os.path.basename(p) for p in downloaded)))
            with open(os.path.join(d, "grib_manifest.txt")) as f:
                manifest_lines = [ln.strip() for ln in f if ln.strip()]
            self.assertEqual(manifest_lines, forcing["files"])


# ---------------------------------------------------------------------------
# R6-D (2026-09-24): date-shifted cold-start sampling. Windows are placed/run at the ANALOG
# timeline; night/day picks are still ranked by the STATED timeline's own sun position; score's
# s_k anchors map back to the stated timeline; the compute range widens by analog quality.
# ---------------------------------------------------------------------------

class TestSelectWindowsSunlitShift(unittest.TestCase):
    def setUp(self):
        self.start_dt = datetime.datetime(2020, 1, 1, 0, 0, 0)
        self.run_end_dt = self.start_dt + datetime.timedelta(hours=72)
        self.interval_seconds = 10800
        self.W, self.stepra, self.measure_steps = cs.compute_window_seconds(60, 72.0, 15.0)
        self.candidates = cs.candidate_times(self.start_dt, self.run_end_dt,
                                              self.interval_seconds, self.W)
        self.sample_points = cs.domain_lattice_from_geom(35.0, -98.0, 12000.0, 425,
                                                          dy_m=12000.0, e_sn=300)

    def test_zero_shift_reproduces_unshifted_picks_exactly(self):
        picks_a = cs.select_windows(self.candidates, 3600.0, self.W, 4,
                                    sample_points=self.sample_points, sunlit_shift_s=0.0)
        picks_b = cs.select_windows(self.candidates, 3600.0, self.W, 4,
                                    sample_points=self.sample_points)
        self.assertEqual(picks_a, picks_b)

    def test_large_shift_can_change_day_night_picks(self):
        # shifting by 12h flips which candidate's STATED-timeline sun position looks like day
        # vs night, without changing which candidates exist (still the same analog-timeline set).
        picks_shifted = cs.select_windows(self.candidates, 3600.0, self.W, 4,
                                          sample_points=self.sample_points,
                                          sunlit_shift_s=12 * 3600.0)
        picks_unshifted = cs.select_windows(self.candidates, 3600.0, self.W, 4,
                                            sample_points=self.sample_points, sunlit_shift_s=0.0)
        by_reason_shifted = {r: t for t, r in picks_shifted}
        by_reason_unshifted = {r: t for t, r in picks_unshifted}
        for t, _r in picks_shifted:
            self.assertIn(t, self.candidates)
        self.assertNotEqual(by_reason_shifted.get("day"), by_reason_unshifted.get("day"))


class TestBuildPlanDateShift(unittest.TestCase):
    def test_date_shift_to_places_windows_on_analog_timeline(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            out_dir = os.path.join(d, "plan")
            os.makedirs(case_dir)
            write_namelist(os.path.join(case_dir, "namelist.input"),
                           sy=2020, smo=1, sd=1, sh=0, run_hours=6)
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            plan = cs.build_plan(case_dir, out_dir, k=1, nodes=1, ranks=72, omp=1, nx=9, ny=8,
                                 spinup_min=1, date_shift_to="2021-06-15_00:00:00")
            self.assertEqual(plan["stated_start"], "2020-01-01_00:00:00")
            self.assertEqual(plan["analog_start"], "2021-06-15_00:00:00")
            expected_shift = (datetime.datetime(2021, 6, 15) - datetime.datetime(2020, 1, 1)).total_seconds()
            self.assertEqual(plan["shift_s"], expected_shift)
            self.assertEqual(plan["analog_quality"], "manual")
            t0_window = next(w for w in plan["windows"] if w["reason"] == "t0")
            self.assertTrue(t0_window["t"].startswith("2021-06-15"))
            # run_start/run_end stay on the STATED timeline (what the real future run will see)
            self.assertEqual(plan["run_start"], "2020-01-01_00:00:00")

    def test_date_shift_to_and_analog_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            write_namelist(os.path.join(case_dir, "namelist.input"))
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            with self.assertRaises(cs.ColdstartError):
                cs.build_plan(case_dir, os.path.join(d, "plan"), spinup_min=1,
                              date_shift_to="2021-06-15_00:00:00",
                              analog_path=os.path.join(d, "analog.json"))

    def test_analog_path_reads_chosen_date_and_quality(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            write_namelist(os.path.join(case_dir, "namelist.input"),
                           sy=2020, smo=1, sd=1, sh=0, run_hours=6)
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            analog_path = os.path.join(d, "analog.json")
            with open(analog_path, "w") as f:
                json.dump({"chosen": {"date": "2022-03-10_00:00:00"}, "quality": "fair"}, f)
            plan = cs.build_plan(case_dir, os.path.join(d, "plan"), k=1, spinup_min=1,
                                 analog_path=analog_path)
            self.assertEqual(plan["analog_start"], "2022-03-10_00:00:00")
            self.assertEqual(plan["analog_quality"], "fair")


class TestScoreAnchorShift(unittest.TestCase):
    def test_anchor_maps_analog_midpoint_back_to_stated_timeline(self):
        with tempfile.TemporaryDirectory() as d:
            T_analog = datetime.datetime(2021, 6, 15, 0, 0, 0)   # window's own (analog) start
            dt = 72.0
            lines = [_rsl_line(T_analog + datetime.timedelta(seconds=dt * i), 1.0)
                    for i in range(1, 40)]
            wrf_dir = os.path.join(d, "win_0", "wrf")
            os.makedirs(wrf_dir)
            with open(os.path.join(wrf_dir, "rsl.error.0000"), "w") as f:
                f.write("\n".join(lines) + "\n")

            shift_s = (T_analog - datetime.datetime(2020, 1, 1, 0, 0, 0)).total_seconds()
            plan = {
                "dt_s": dt, "radt_min": 100000.0,
                "stepra": predict.steps_per_radiation_call(100000.0, dt),
                "spinup_min": 1.0, "W_seconds": 60.0 + 30 * dt,
                "run_start": "2020-01-01_00:00:00",   # STATED timeline
                "shift_s": shift_s,
                "windows": [{"t": T_analog.strftime(_WRF_TIME_FMT), "reason": "t0"}],
                "_plan_dir": d,
            }
            windows_out, pooled, anchors = cs.score_windows_from_rsl(plan, None, {})
            # mid_dt (analog) = T_analog + spinup + (W-spinup)/2; mapped back by -shift_s, then
            # made relative to run_start_dt (stated) = 2020-01-01 -- since T_analog - shift_s ==
            # stated start (2020-01-01) by construction, the anchor is exactly the
            # spinup+(W-spinup)/2 offset, identical to the unshifted case.
            spinup_s = plan["spinup_min"] * 60.0
            expected_anchor_s = spinup_s + (plan["W_seconds"] - spinup_s) / 2.0
            self.assertAlmostEqual(anchors[0][0], expected_anchor_s, places=6)


class TestScoreAnalogQualityRangeWidening(unittest.TestCase):
    def _plan_dir_with_one_window(self, d):
        case_nl = os.path.join(d, "namelist.input")
        write_namelist(case_nl)
        plan_dir = os.path.join(d, "plan")
        wrf_dir = os.path.join(plan_dir, "win_0", "wrf")
        os.makedirs(wrf_dir)
        T = datetime.datetime(2019, 11, 26, 15, 0, 0)
        lines = [_rsl_line(T + datetime.timedelta(seconds=72 * i), 1.0) for i in range(1, 35)]
        with open(os.path.join(wrf_dir, "rsl.error.0000"), "w") as f:
            f.write("\n".join(lines) + "\n")
        geo = {"ref_lat": 40.0, "ref_lon": -98.0, "dx": 12000.0, "dy": 12000.0,
              "e_we": 425, "e_sn": 300}
        plan_base = {
            "namelist_input": case_nl, "case_dir": "/does/not/matter",
            "dt_s": 72.0, "radt_min": 15.0,
            "stepra": predict.steps_per_radiation_call(15.0, 72.0),
            "spinup_min": 1.0, "W_seconds": 60.0 + 30 * 72.0,
            "run_start": T.strftime(_WRF_TIME_FMT), "shift_s": 0.0,
            "windows": [{"t": T.strftime(_WRF_TIME_FMT), "reason": "t0"}],
            "geo": geo, "nodes": 1,
            "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
        }
        return plan_dir, plan_base

    def _score_with_quality(self, quality):
        with tempfile.TemporaryDirectory() as d:
            plan_dir, plan_base = self._plan_dir_with_one_window(d)
            plan_base["analog_quality"] = quality
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan_base, f)
            return cs.score_plan(plan_dir)

    def test_good_quality_widens_to_5pct(self):
        result = self._score_with_quality("good")
        self.assertGreaterEqual(result["drift_frac"], 0.05 - 1e-9)
        self.assertIn("analog_quality=good", result["range_note"])

    def test_fair_quality_widens_to_10pct(self):
        result = self._score_with_quality("fair")
        self.assertGreaterEqual(result["drift_frac"], 0.10 - 1e-9)

    def test_poor_quality_widens_to_15pct(self):
        result = self._score_with_quality("poor")
        self.assertGreaterEqual(result["drift_frac"], 0.15 - 1e-9)

    def test_no_analog_quality_leaves_range_note_none(self):
        with tempfile.TemporaryDirectory() as d:
            plan_dir, plan_base = self._plan_dir_with_one_window(d)
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump(plan_base, f)
            result = cs.score_plan(plan_dir)
            self.assertIsNone(result["range_note"])


# ---------------------------------------------------------------------------
# Round 7 (2026-09-24): T1 `plan --times` (pinned windows, --k ignored), T2 `plan --real-only`
# (real.exe-only windows, score refuses).
# ---------------------------------------------------------------------------

class TestPlanTimesPinned(unittest.TestCase):
    def _case_dir(self, d, run_hours=48):
        case_dir = os.path.join(d, "case")
        os.makedirs(case_dir)
        write_namelist(os.path.join(case_dir, "namelist.input"),
                       sy=2020, smo=1, sd=1, sh=0, run_hours=run_hours)
        write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
        return case_dir

    def test_times_exact_in_plan_json(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._case_dir(d)
            times = [datetime.datetime(2020, 1, 1, 3, 0, 0), datetime.datetime(2020, 1, 1, 9, 0, 0)]
            plan = cs.build_plan(case_dir, os.path.join(d, "plan"), spinup_min=1, times=times)
            self.assertEqual([w["t"] for w in plan["windows"]],
                             ["2020-01-01_03:00:00", "2020-01-01_09:00:00"])
            self.assertEqual([w["reason"] for w in plan["windows"]], ["pinned", "pinned"])

    def test_k_is_ignored_with_times_more_times_than_k(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._case_dir(d)
            times = [datetime.datetime(2020, 1, 1, 3, 0, 0), datetime.datetime(2020, 1, 1, 9, 0, 0),
                     datetime.datetime(2020, 1, 1, 15, 0, 0)]
            plan = cs.build_plan(case_dir, os.path.join(d, "plan"), spinup_min=1, k=1, times=times)
            self.assertEqual(len(plan["windows"]), 3)   # not capped at k=1

    def test_off_grid_time_raises_naming_time_and_grid(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._case_dir(d)
            times = [datetime.datetime(2020, 1, 1, 1, 0, 0)]   # not on the 3h forcing grid
            with self.assertRaises(cs.ColdstartError) as ctx:
                cs.build_plan(case_dir, os.path.join(d, "plan"), spinup_min=1, times=times)
            msg = str(ctx.exception)
            self.assertIn("2020-01-01_01:00:00", msg)
            self.assertIn("3h cadence", msg)

    def test_time_that_does_not_fit_the_run_raises(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = self._case_dir(d, run_hours=6)
            # run is only 6h (ends 2020-01-01_06:00:00). spinup_min=200 -> W = 200*60 + 30*72 =
            # 14160s (~3.93h); a window pinned at 03:00 (3h of run remaining) cannot fit it.
            times = [datetime.datetime(2020, 1, 1, 3, 0, 0)]
            with self.assertRaises(cs.ColdstartError) as ctx:
                cs.build_plan(case_dir, os.path.join(d, "plan"), spinup_min=200, times=times)
            self.assertIn("does not fit in the run", str(ctx.exception))


class TestPlanRealOnly(unittest.TestCase):
    def test_real_only_recorded_in_plan(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            write_namelist(os.path.join(case_dir, "namelist.input"), sy=2020, smo=1, sd=1, sh=0)
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            plan = cs.build_plan(case_dir, os.path.join(d, "plan"), spinup_min=1, real_only=True)
            self.assertTrue(plan["real_only"])

    def test_default_is_not_real_only(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            write_namelist(os.path.join(case_dir, "namelist.input"), sy=2020, smo=1, sd=1, sh=0)
            write_namelist_wps(os.path.join(case_dir, "namelist.wps"))
            plan = cs.build_plan(case_dir, os.path.join(d, "plan"), spinup_min=1)
            self.assertFalse(plan["real_only"])

    def test_rendered_sbatch_has_no_wrf_exe_launch_or_site_launch_call(self):
        plan = {
            "windows": [{"t": "2020-01-01_03:00:00", "t_end": "2020-01-01_06:00:00",
                        "reason": "pinned"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
            "real_only": True,
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertNotIn("fix-wrf-namelist", text)
        self.assertNotIn("window $k: wrf.exe", text)
        self.assertFalse(any(line.strip().startswith("site_launch ")
                             for line in text.splitlines()))
        self.assertNotIn("SITE_LAUNCH_CMD", text)
        self.assertIn("wrf=$LAST_WRF_STATUS", text)

    def test_real_step_links_nocolons_met_em_names(self):
        plan = {
            "windows": [{"t": "2020-01-01_03:00:00", "t_end": "2020-01-01_06:00:00",
                        "reason": "pinned"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertIn('ln -sf "$f" "${f//:/_}"', text)
        self.assertLess(text.index("wps/met_em.d0*.*.nc ."), text.index('"${f//:/_}"'))

    def test_per_window_min_scales_wrf_timeout_and_time_limit(self):
        # job 982 (maria1km): the fixed 15 min wrf.exe timeout killed a 1 km window at 899 s
        plan = {
            "windows": [{"t": "2020-01-01_03:00:00", "t_end": "2020-01-01_06:00:00",
                        "reason": "pinned"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
        }
        self.assertIn("timeout 900 ", cs.render_run_sbatch(plan, site="scc"))
        text = cs.render_run_sbatch(plan, site="scc", per_window_estimate_min=60)
        self.assertIn("timeout 3600 ", text)
        self.assertNotIn("timeout 900 ", text)
        self.assertIn("#SBATCH --time=02:40:00", text)

    def test_rendered_sbatch_still_has_real_exe_timeout_and_serial_fallback(self):
        plan = {
            "windows": [{"t": "2020-01-01_03:00:00", "t_end": "2020-01-01_06:00:00",
                        "reason": "pinned"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
            "real_only": True,
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertIn("timeout 600 mpirun -np 72 --map-by ppr:72:node:PE=1 --bind-to core "
                     "./real.exe", text)
        self.assertIn("timeout 1800 mpirun -np 1 --map-by ppr:1:node:PE=1 --bind-to core "
                     "./real.exe", text)
        self.assertIn("--serial", text)

    def test_normal_plan_still_has_wrf_exe_and_ok_status(self):
        plan = {
            "windows": [{"t": "2020-01-01_03:00:00", "t_end": "2020-01-01_06:00:00",
                        "reason": "pinned"}],
            "nodes": 1, "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1},
            "real_only": False,
        }
        text = cs.render_run_sbatch(plan, site="scc")
        self.assertIn("fix-wrf-namelist", text)
        self.assertTrue(any(line.strip().startswith("site_launch ")
                            for line in text.splitlines()))
        self.assertIn("LAST_WRF_STATUS=\"ok\"", text)

    def test_score_refuses_real_only_plan(self):
        with tempfile.TemporaryDirectory() as d:
            plan_dir = os.path.join(d, "plan")
            os.makedirs(plan_dir)
            with open(os.path.join(plan_dir, "plan.json"), "w") as f:
                json.dump({"real_only": True, "windows": []}, f)
            with self.assertRaises(cs.ColdstartError) as ctx:
                cs.score_plan(plan_dir)
            self.assertIn("--real-only", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TestDownloadRetriesAndFailsOnMissing(unittest.TestCase):
    # yagi2024 rebuild (scc job 1002): a swallowed transient failure left fnl_2024090312 out
    T = datetime.datetime(2024, 9, 3, 12)
    T_end = datetime.datetime(2024, 9, 3, 18)

    def test_transient_failure_is_retried(self):
        calls = []

        def flaky(url, dest):
            calls.append(url)
            if len(calls) <= 2:          # first attempt: both grib2 and grib1 urls fail
                raise OSError("503")
            open(dest, "w").close()

        with tempfile.TemporaryDirectory() as d:
            import metdata as md
            paths = cs._download_forcing_for_window(md.SRC_FNL, self.T, self.T_end, d,
                                                    download_fn=flaky, sleep_fn=lambda s: None)
            names = sorted(os.path.basename(p) for p in paths)
        self.assertEqual(names, ["fnl_2024090312.grib", "fnl_2024090318.grib"])

    def test_persistent_failure_raises_naming_the_file(self):
        def only_18z(url, dest):
            if "_12_00" in url:
                raise OSError("503")
            open(dest, "w").close()

        with tempfile.TemporaryDirectory() as d:
            import metdata as md
            with self.assertRaisesRegex(cs.ColdstartError, "fnl_2024090312.grib"):
                cs._download_forcing_for_window(md.SRC_FNL, self.T, self.T_end, d,
                                                download_fn=only_18z, sleep_fn=lambda s: None)


class TestWindowEndTimeWrittenWithoutCaseMinuteKeys(unittest.TestCase):
    # maria1km (scc job 1006): namelist has no end_minute/end_second, so d02 stopped at 10:00:00
    NL = """&time_control
 run_hours = 3,
 start_year = 2017, 2017,
 start_month = 09, 09,
 start_day = 20, 20,
 start_hour = 09, 09,
 end_year = 2017, 2017,
 end_month = 09, 09,
 end_day = 20, 20,
 end_hour = 12, 12,
/
&domains
 max_dom = 2,
 time_step = 25,
/
"""

    def _vals(self, text, key):
        return cs.get_first(text, key)

    def test_wrf_window_end_has_minute_and_second_for_every_domain(self):
        T = datetime.datetime(2017, 9, 20, 9, 0, 0)
        txt = cs.build_wrf_window_namelist_text(self.NL, T, 202, 25, 16, 18, 1)
        self.assertRegex(txt, r"end_hour\s*=\s*10, 10,")
        self.assertRegex(txt, r"end_minute\s*=\s*24, 24,")
        self.assertRegex(txt, r"end_second\s*=\s*10, 10,")
        self.assertRegex(txt, r"start_minute\s*=\s*00, 00,")

    def test_real_namelist_end_has_minute_for_every_domain(self):
        T = datetime.datetime(2017, 9, 20, 9)
        txt = cs.build_real_namelist_text(self.NL, T, T + datetime.timedelta(minutes=90), 34, 4, 21)
        self.assertRegex(txt, r"end_minute\s*=\s*30, 30,")


class TestAerosolClimatologyConstants(unittest.TestCase):
    # lafe (scc job 1009): mp_physics=28 + use_aero_icbc needs the QNWFA/QNIFA climatology in
    # met_em, which no GFS-family analysis carries
    def _nl(self, physics, domains=""):
        with tempfile.NamedTemporaryFile("w", suffix=".input", delete=False) as f:
            f.write("&domains\n max_dom = 1,\n%s/\n&physics\n%s/\n" % (domains, physics))
        self.addCleanup(os.remove, f.name)
        return cs.predict.parse_namelist(f.name)

    def test_no_aerosol_input_adds_nothing(self):
        self.assertEqual(cs.wps_constants_for(self._nl(" mp_physics = 8,\n"), "/nonexistent"), [])

    def test_use_aero_icbc_resolves_sigma_file_and_renders(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "QNWFA_QNIFA_SIGMA_MONTHLY.dat"), "w").close()
            nl = self._nl(" mp_physics = 28,\n use_aero_icbc = .true.,\n", " wif_input_opt = 1,\n")
            got = cs.wps_constants_for(nl, d)
        self.assertEqual(got, [os.path.join(os.path.abspath(d), "QNWFA_QNIFA_SIGMA_MONTHLY.dat")])
        fields = {"max_dom": 1, "start_date": "x", "end_date": "y", "interval_seconds": 10800,
                  "parent_id": [1], "parent_grid_ratio": [1], "i_parent_start": [1],
                  "j_parent_start": [1], "e_we": [10], "e_sn": [10], "geog_data_res": ["default"],
                  "dx": 1000, "dy": 1000, "map_proj": "lambert", "ref_lat": 0, "ref_lon": 0,
                  "geog_data_path": "/g", "constants_name": got}
        self.assertIn(" constants_name = '%s'," % got[0], cs.render_namelist_wps(fields))

    def test_missing_file_raises_at_plan_time(self):
        nl = self._nl(" mp_physics = 28,\n use_aero_icbc = .true.,\n")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(cs.ColdstartError, "QNWFA_QNIFA_SIGMA_MONTHLY.dat"):
                cs.wps_constants_for(nl, d)

    def test_wif_input_opt_2_needs_black_carbon_file(self):
        nl = self._nl(" mp_physics = 28,\n use_aero_icbc = .true.,\n", " wif_input_opt = 2,\n")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(cs.ColdstartError, "QNWFA_QNIFA_QNBCA_SIGMA_MONTHLY.dat"):
                cs.wps_constants_for(nl, d)


class TestSetListValueWithoutTrailingComma(unittest.TestCase):
    # yagi2024 rebuild (scc job 1021): "start_year = 2024, 2024,2024" got a duplicate line
    def test_replaces_in_place(self):
        t = "&time_control\nstart_year                          = 2024, 2024,2024\n/\n"
        out = cs.set_list(t, "start_year", 2024, 2)
        self.assertEqual(cs.count_key(out, "start_year"), 1)
        self.assertIn("start_year                          = 2024, 2024,", out)

    def test_packed_line_neighbor_kept(self):
        t = "&time_control\n start_hour = 12, 12,12 end_hour = 06, 06,\n/\n"
        out = cs.set_list(t, "start_hour", "00", 2)
        self.assertEqual(cs.count_key(out, "start_hour"), 1)
        self.assertIn("end_hour = 06, 06,", out)
        self.assertIn("start_hour = 00, 00,", out)

    def test_single_value_without_comma(self):
        out = cs.set_list("&domains\n max_dom = 2\n/\n", "max_dom", 1, 1)
        self.assertIn("max_dom = 1,", out)
        self.assertEqual(cs.count_key(out, "max_dom"), 1)


def test_wrf_window_namelist_inserts_decomposition_when_case_omits_it():
    # conus25's namelist has no nproc_x/nproc_y/numtiles; C3 job 1237's rebuild window then ran
    # WRF's default decomposition instead of the config being predicted (+21% per step).
    import datetime as _dt
    text = ("&time_control\n run_days = 0,\n run_hours = 6,\n run_minutes = 0,\n"
            " run_seconds = 0,\n start_year = 2019,\n start_month = 11,\n start_day = 26,\n"
            " start_hour = 18,\n end_year = 2019,\n end_month = 11,\n end_day = 27,\n"
            " end_hour = 00,\n history_interval = 60,\n/\n"
            "&domains\n time_step = 15,\n max_dom = 1,\n e_we = 100,\n/\n")
    out = cs.build_wrf_window_namelist_text(text, _dt.datetime(2019, 11, 26, 22), 50, 15.0,
                                            12, 24, 8)
    dom = out.split("&domains", 1)[1]
    assert cs.count_key(dom, "nproc_x") == 1 and "nproc_x = 12," in dom.replace("  ", " ")
    assert cs.count_key(dom, "nproc_y") == 1 and cs.count_key(dom, "numtiles") == 1
    assert cs.count_key(out.split("&domains", 1)[0], "numtiles") == 0
    auto = cs.build_wrf_window_namelist_text(text, _dt.datetime(2019, 11, 26, 22), 50, 15.0,
                                             -1, -1, 1)
    assert cs.count_key(auto, "nproc_x") == 0 and cs.count_key(auto, "numtiles") == 0


def test_era5_pin_fetch_writes_intermediate_manifest(tmp_path):
    netCDF4 = __import__("pytest").importorskip("netCDF4")
    import numpy as _np
    case = tmp_path / "case"; case.mkdir()
    with netCDF4.Dataset(case / "wrfinput_d01", "w") as d:
        d.createDimension("Time", 1); d.createDimension("south_north", 2); d.createDimension("west_east", 2)
        for name, val in (("XLAT", [[20.0, 20.0], [50.0, 50.0]]), ("XLONG", [[-120.0, -70.0], [-120.0, -70.0]])):
            v = d.createVariable(name, "f4", ("Time", "south_north", "west_east")); v[0] = _np.array(val)
    plan = {"case_dir": str(case), "forcing_pin": cs.ERA5_DATASET,
            "windows": [{"t": "2019-11-26_21:00:00", "t_end": "2019-11-26_23:00:00"}]}
    seen = {}

    def runner(cmd):
        seen["cmd"] = cmd
        out = cmd[cmd.index("--out") + 1]
        for t in cmd[cmd.index("--times") + 1].split(","):
            open(os.path.join(out, "ERA5:" + t), "wb").write(b"x")

    total, forcing = cs._fetch_era5_intermediate(plan, str(tmp_path), runner=runner)
    assert seen["cmd"][seen["cmd"].index("--bbox") + 1] == "20.000,50.000,-120.000,-70.000"
    assert forcing["files"] == ["ERA5:2019-11-26_21", "ERA5:2019-11-26_22", "ERA5:2019-11-26_23"]
    assert forcing["windows"][0]["chosen"]["vtable"] == "none" and total == 3
    pwf = cs._per_window_forcing_from_forcing_json(str(tmp_path), plan)
    assert pwf == [{"prefix": "ERA5", "vtable": "none", "forcing_interval_h": 1}]
    assert cs.resolve_forcing_chain_for(None, forcing_pin=cs.ERA5_DATASET)[0].prefix == "ERA5"


class TestShortMeasureWindow(unittest.TestCase):
    """--measure-steps short (decision 0054 speed attempt): a window of stepra+10 measured steps
    is too short for classify_steps_detected, so radiation steps are located by known phase."""

    def test_explicit_measure_steps_sets_window(self):
        W, stepra, measure_steps = cs.compute_window_seconds(10, 36.0, 10.0, measure_steps=27)
        self.assertEqual(stepra, 17)
        self.assertEqual(measure_steps, 27)
        self.assertEqual(W, 600.0 + 27 * 36.0)

    def test_measure_steps_below_one_radiation_period_refused(self):
        with self.assertRaises(cs.ColdstartError):
            cs.compute_window_seconds(10, 36.0, 10.0, measure_steps=17)

    def test_short_window_flags_radiation_steps_by_known_phase(self):
        with tempfile.TemporaryDirectory() as d:
            T = datetime.datetime(2021, 7, 20, 0, 0, 0)
            dt, stepra, spin_steps, measure = 36.0, 17, 17, 27
            lines = []
            for n in range(1, spin_steps + measure + 1):
                sec = 5.0 if n % stepra == 1 else 1.0
                lines.append(_rsl_line(T + datetime.timedelta(seconds=dt * n), sec))
            os.makedirs(os.path.join(d, "win_0", "wrf"))
            with open(os.path.join(d, "win_0", "wrf", "rsl.error.0000"), "w") as f:
                f.write("\n".join(lines) + "\n")
            spinup_seconds = dt * spin_steps + 1.0
            plan = {"dt_s": dt, "radt_min": 10.0, "stepra": stepra, "measure_steps": measure,
                    "spinup_min": spinup_seconds / 60.0,
                    "W_seconds": spinup_seconds + measure * dt,
                    "run_start": T.strftime(_WRF_TIME_FMT),
                    "windows": [{"t": T.strftime(_WRF_TIME_FMT), "reason": "rebuild"}],
                    "_plan_dir": d}
            out, _pooled, _anchors = cs.score_windows_from_rsl(plan, None, {})
            # post-spin-up steps n=18..44: radiation at n=18 and n=35
            self.assertEqual(out[0]["n_rad"], 2)
            self.assertEqual(out[0]["n_normal"], measure - 2)
            self.assertAlmostEqual(out[0]["s_k"], 1.0)
            self.assertAlmostEqual(out[0]["r_local"], 5.0)
