#!/usr/bin/env python3
"""Offline tests for tools/case_rebuild.py. No real network access (fetch orchestration is
best-effort/not exercised here -- see case_rebuild.py's own module docstring). netCDF4 is NOT
installed in this dev environment (only on the cluster's grib pixi env) -- the netCDF4-reading
layer (read_wrfinput_fields_netcdf4) is therefore not independently exercised here either, same
skip-if-missing convention tools/test_grib_state.py already uses; every metric/scoring function
it feeds is tested directly against plain numpy arrays instead.

Run with:
  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q tools/test_case_rebuild.py
"""
import datetime
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wtune.tools import case_rebuild as cr   # noqa: E402
from wtune.tools import coldstart_sample as cs  # noqa: E402
from wtune import predict as predict  # noqa: E402

_WRF_TIME_FMT = cr._WRF_TIME_FMT

try:
    import netCDF4  # noqa: F401
    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False


# ---------------------------------------------------------------------------
# R6-A: complete_namelist
# ---------------------------------------------------------------------------

BASE_NL_TEXT = """\
 &time_control
 run_days                            = 0,
 run_hours                           = 3,
 run_minutes                         = 0,
 run_seconds                         = 0,
 interval_seconds                    = 10800
 /

 &domains
 max_dom                             = 1,
 e_we                                = 100,
 e_sn                                = 100,
 dx                                  = 12000,
 /

 &physics
 /
"""

BASE_FACTS = dict(
    max_dom=1, num_metgrid_levels=32, num_metgrid_soil_levels=4, num_land_cat=21,
    interval_seconds=21600,
    start_dt=datetime.datetime(2020, 1, 1, 0, 0, 0),
    end_dt=datetime.datetime(2020, 1, 1, 6, 0, 0),
    dx_km=12.0, parent_grid_ratio=[1],
)


class TestCompleteNamelistFillOnlyIfMissing(unittest.TestCase):
    def setUp(self):
        self.txt, self.changes = cr.complete_namelist(BASE_NL_TEXT, BASE_FACTS)


    def test_mp_physics_7_gets_do_radar_ref(self):
        txt, changes = cr.complete_namelist(
            BASE_NL_TEXT.replace("&physics\n", "&physics\n mp_physics = 7,\n"), BASE_FACTS)
        self.assertEqual(cs.get_first(txt, "do_radar_ref"), "1")
        self.assertIn("do_radar_ref", [c["key"] for c in changes])
        txt, _ = cr.complete_namelist(
            BASE_NL_TEXT.replace("&physics\n", "&physics\n mp_physics = 17,\n"), BASE_FACTS)
        self.assertEqual(cs.count_key(txt, "do_radar_ref"), 0)


    def test_physics_suite_with_minus1_placeholders_gets_a_note(self):
        # T14 (2026-09-24, cluster check against conus12): physics_suite='CONUS' with
        # mp_physics=-1 (a placeholder meaning 'use the suite default') is a real, valid
        # namelist -- record a note so a future comparison tool doesn't flag -1 vs an explicit
        # scheme value as a mismatch. Nothing is MODIFIED (old==new).
        text = BASE_NL_TEXT.replace(
            "&physics\n",
            "&physics\n physics_suite = 'CONUS',\n mp_physics = -1,\n ra_lw_physics = -1,\n")
        txt, changes = cr.complete_namelist(text, BASE_FACTS)
        note = next(c for c in changes if c["key"] == "physics_suite_placeholders")
        self.assertEqual(note["old"], "'CONUS'")
        self.assertEqual(note["new"], "'CONUS'")
        self.assertIn("mp_physics", note["reason"])
        self.assertIn("ra_lw_physics", note["reason"])
        self.assertIn("EQUIVALENT", note["reason"])
        # the namelist text itself is untouched by this note
        self.assertEqual(cs.get_first(txt, "mp_physics"), "-1")


# ---------------------------------------------------------------------------
# T14 (2026-09-24, cluster check against real namelists): complete_namelist's fill_if_missing/
# sync must decide EACH key's namelist dimension from wtune_catalog_registry.json's own
# 'per_domain' field, not a hardcoded ndom -- frames_per_outfile is per_domain=True there but
# was previously always written as a single value.
# ---------------------------------------------------------------------------

class TestCompleteNamelistDimensionFromRegistry(unittest.TestCase):
    def test_frames_per_outfile_is_broadcast_per_domain(self):
        registry = cr.load_registry()
        self.assertTrue(cr.registry_per_domain("frames_per_outfile", registry))
        facts = dict(BASE_FACTS, max_dom=2, parent_grid_ratio=[1, 3])
        txt, changes = cr.complete_namelist(BASE_NL_TEXT, facts, registry=registry)
        m = re.search(r"(?<![A-Za-z0-9_])frames_per_outfile\s*=\s*([^\n/]+)", txt)
        self.assertEqual(m.group(1).strip().rstrip(","), "1, 1")


# ---------------------------------------------------------------------------
# T16 (2026-09-24, job 986 follow-up): complete_namelist must guarantee the same 8 groups K1
# guarantees (T15), creating any missing group EMPTY except &bdy_control (K1's same defaults,
# fill-only-if-missing) and &namelist_quilt (its 2 keys, only if absent).
# ---------------------------------------------------------------------------

class TestCompleteNamelistGuaranteesAllEightGroups(unittest.TestCase):
    def _parse(self, txt):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "namelist.input")
            with open(path, "w") as f:
                f.write(txt)
            return predict.parse_namelist(path)


    def test_all_groups_already_present_makes_no_group_changes(self):
        text = BASE_NL_TEXT.replace(
            "&physics\n",
            "&physics\n/\n\n&fdda\n/\n\n&dynamics\n/\n\n&bdy_control\n spec_bdy_width = 5,\n"
            " spec_zone = 1,\n relax_zone = 4,\n specified = .true.,\n nested = .false.,\n/\n\n"
            "&grib2\n/\n\n&namelist_quilt\n nio_tasks_per_group = 0,\n nio_groups = 1,\n")
        txt, changes = cr.complete_namelist(text, BASE_FACTS)
        self.assertFalse(any(c["key"].startswith("&") for c in changes))


# ---------------------------------------------------------------------------
# R6-A: registry validation
# ---------------------------------------------------------------------------

class TestRegistryValidation(unittest.TestCase):
    def test_known_keys_produce_no_warnings(self):
        registry = cr.load_registry()
        nl = {"time_control": {"run_days": [0], "interval_seconds": [10800]},
             "domains": {"max_dom": [1]}}
        warnings = cr.validate_namelist_against_registry(nl, registry)
        self.assertEqual(warnings, [])

    def test_unknown_key_is_a_warning_not_deleted(self):
        registry = cr.load_registry()
        nl = {"time_control": {"totally_made_up_key_xyz": [1]}}
        warnings = cr.validate_namelist_against_registry(nl, registry)
        self.assertEqual(len(warnings), 1)
        self.assertIn("totally_made_up_key_xyz", warnings[0])
        self.assertIn("time_control", warnings[0])


    def test_group_the_catalog_does_not_cover_gets_one_warning_not_per_key(self):
        # T11 (2026-09-24): wtune_catalog_registry.json has no 'namelist_quilt' group at all,
        # even though nio_groups/nio_tasks_per_group are real WRF keys -- must not report them
        # as 'unknown namelist key', only that the GROUP itself isn't in the catalog.
        registry = cr.load_registry()
        self.assertNotIn("namelist_quilt", set(g.lower() for g in registry.get("groups", {})))
        nl = {"namelist_quilt": {"nio_groups": [1], "nio_tasks_per_group": [4]}}
        warnings = cr.validate_namelist_against_registry(nl, registry)
        self.assertEqual(len(warnings), 1)
        self.assertIn("namelist_quilt", warnings[0])
        self.assertIn("not in Registry catalog", warnings[0])
        self.assertNotIn("nio_groups", warnings[0])
        self.assertFalse(any("unknown namelist key" in w for w in warnings))


# ---------------------------------------------------------------------------
# T11 (2026-09-24, job 962 -- highest priority after T3, blocks V1): complete_namelist's
# fill-if-missing/sync must place a newly-inserted key into ITS OWN Registry group, not always
# &time_control -- real.exe FATAL'd 'Maybe here?: p_top_requested =' because p_top_requested (a
# &domains key) landed in &time_control.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# K1 (2026-09-24): namelist-from-wrfinput -- pure attr->namelist mapping (no netCDF4 needed).
# ---------------------------------------------------------------------------

class TestNamelistFieldsFromWrfinputFactsPure(unittest.TestCase):
    def _facts(self):
        attrs = {
            "WEST-EAST_GRID_DIMENSION": 425, "SOUTH-NORTH_GRID_DIMENSION": 300,
            "BOTTOM-TOP_GRID_DIMENSION": 45, "DX": 12000.0, "DY": 12000.0,
            "GRID_ID": 1, "PARENT_ID": 0, "I_PARENT_START": 1, "J_PARENT_START": 1,
            "PARENT_GRID_RATIO": 1, "DT": 72.0,
            "MP_PHYSICS": 8, "RA_LW_PHYSICS": 4, "RA_SW_PHYSICS": 4,
            "SF_SFCLAY_PHYSICS": 1, "SF_SURFACE_PHYSICS": 2, "BL_PBL_PHYSICS": 1,
            "CU_PHYSICS": 6, "DIFF_OPT": 1, "KM_OPT": 4, "DAMP_OPT": 3,
            "HYPSOMETRIC_OPT": 2, "USE_THETA_M": 1, "GWD_OPT": 1, "SF_URBAN_PHYSICS": 0,
            "NUM_LAND_CAT": 21, "SIMULATION_START_DATE": "2019-11-26_12:00:00",
        }
        d01 = {"attrs": attrs, "znw": [1.0, 0.5, 0.0], "p_top": 5000.0,
              "dzs": [0.1, 0.3, 0.6, 1.0], "times": [datetime.datetime(2019, 11, 26, 12, 0, 0)]}
        return [d01]


    def test_non_integral_dt_writes_fract_fields(self):
        facts = self._facts()
        facts[0]["attrs"]["DT"] = 45.5
        groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(
            facts, wrfbdy_times=None, hours=6.0)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["domains"]["time_step"], [45])
        self.assertEqual(gdict["domains"]["time_step_fract_num"], [50])
        self.assertEqual(gdict["domains"]["time_step_fract_den"], [100])

    def test_times_mismatch_with_simulation_start_date_notes_it(self):
        facts = self._facts()
        facts[0]["times"] = [datetime.datetime(2019, 11, 26, 13, 0, 0)]   # != SIM_START 12:00
        groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(
            facts, wrfbdy_times=None, hours=6.0)
        self.assertTrue(any("SIMULATION_START_DATE" in n and "Times[0]" in n for n in notes))

    def test_missing_simulation_start_date_raises(self):
        facts = self._facts()
        del facts[0]["attrs"]["SIMULATION_START_DATE"]
        with self.assertRaises(cr.RebuildError):
            cr.namelist_fields_from_wrfinput_facts(facts, wrfbdy_times=None, hours=6.0)

    def test_missing_grid_attr_raises(self):
        facts = self._facts()
        del facts[0]["attrs"]["BOTTOM-TOP_GRID_DIMENSION"]
        with self.assertRaises(cr.RebuildError):
            cr.namelist_fields_from_wrfinput_facts(facts, wrfbdy_times=None, hours=6.0)


    def _two_domain_facts(self):
        d01 = self._facts()[0]
        d02_attrs = dict(d01["attrs"])
        d02_attrs.update({"WEST-EAST_GRID_DIMENSION": 151, "SOUTH-NORTH_GRID_DIMENSION": 121,
                          "DX": 4000.0, "DY": 4000.0, "GRID_ID": 2, "PARENT_ID": 1,
                          "I_PARENT_START": 50, "J_PARENT_START": 40, "PARENT_GRID_RATIO": 3})
        return [d01, {"attrs": d02_attrs}]

    def _render_and_parse(self, wrfbdy_times=None, wrfbdy_bdy_width=None):
        registry = cr.load_registry()
        groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(
            self._two_domain_facts(), wrfbdy_times=wrfbdy_times, hours=6.0, registry=registry,
            wrfbdy_bdy_width=wrfbdy_bdy_width)
        text = cr.render_namelist_input(groups)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "namelist.input")
            with open(path, "w") as f:
                f.write(text)
            parsed = predict.parse_namelist(path)
        return groups, parsed, provenance, notes, registry


    def test_spec_bdy_width_from_wrfbdy_bdy_width_when_present(self):
        # T15(c): the wrfbdy file's own bdy_width dimension, when known, wins over the
        # Registry default -- provenance 'wrfbdy'.
        groups, parsed, provenance, notes, registry = self._render_and_parse(wrfbdy_bdy_width=4)
        bdy = parsed["bdy_control"]
        self.assertEqual(bdy["spec_bdy_width"], [4])
        self.assertEqual(provenance["spec_bdy_width"], "wrfbdy")


@unittest.skipUnless(HAS_NETCDF4, "netCDF4 not installed in this dev environment")
class TestCmdNamelistFromWrfinput(unittest.TestCase):
    def _write_wrfinput(self, path, e_we=425, e_sn=300, e_vert=45, grid_id=1, parent_id=0,
                        i_parent_start=1, j_parent_start=1, parent_grid_ratio=1, dx=12000.0,
                        start="2019-11-26_12:00:00", with_vertical=True):
        import netCDF4
        ds = netCDF4.Dataset(path, "w")
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("bottom_top_stag", e_vert + 1)
        ds.createDimension("soil_layers_stag", 4)
        ds.setncattr("WEST-EAST_GRID_DIMENSION", e_we)
        ds.setncattr("SOUTH-NORTH_GRID_DIMENSION", e_sn)
        ds.setncattr("BOTTOM-TOP_GRID_DIMENSION", e_vert)
        ds.setncattr("DX", dx); ds.setncattr("DY", dx)
        ds.setncattr("GRID_ID", grid_id); ds.setncattr("PARENT_ID", parent_id)
        ds.setncattr("I_PARENT_START", i_parent_start)
        ds.setncattr("J_PARENT_START", j_parent_start)
        ds.setncattr("PARENT_GRID_RATIO", parent_grid_ratio)
        ds.setncattr("DT", 72.0)
        for k, v in (("MP_PHYSICS", 8), ("RA_LW_PHYSICS", 4), ("RA_SW_PHYSICS", 4),
                    ("SF_SFCLAY_PHYSICS", 1), ("SF_SURFACE_PHYSICS", 2),
                    ("BL_PBL_PHYSICS", 1), ("CU_PHYSICS", 6), ("DIFF_OPT", 1),
                    ("KM_OPT", 4), ("DAMP_OPT", 3), ("HYPSOMETRIC_OPT", 2),
                    ("USE_THETA_M", 1), ("GWD_OPT", 1), ("SF_URBAN_PHYSICS", 0),
                    ("NUM_LAND_CAT", 21)):
            ds.setncattr(k, v)
        ds.setncattr("SIMULATION_START_DATE", start)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.array(list(start), "S1")
        if with_vertical:
            znw = ds.createVariable("ZNW", "f4", ("Time", "bottom_top_stag"))
            znw[0] = np.linspace(1.0, 0.0, e_vert + 1)
            p_top = ds.createVariable("P_TOP", "f4", ("Time",))
            p_top[0] = 5000.0
            dzs = ds.createVariable("DZS", "f4", ("Time", "soil_layers_stag"))
            dzs[0] = [0.1, 0.3, 0.6, 1.0]
        ds.close()

    def _write_wrfbdy(self, path, start, hours_span, interval_h=3):
        import netCDF4
        ds = netCDF4.Dataset(path, "w")
        n = hours_span // interval_h + 1
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        base = datetime.datetime.strptime(start, cr._WRF_TIME_FMT)
        strs = [(base + datetime.timedelta(hours=interval_h * i)).strftime(cr._WRF_TIME_FMT)
               for i in range(n)]
        times[:] = np.array([list(x) for x in strs], "S1")
        ds.close()

    def test_every_key_and_provenance_single_domain_with_wrfbdy(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_wrfinput(os.path.join(d, "wrfinput_d01"))
            self._write_wrfbdy(os.path.join(d, "wrfbdy_d01"), "2019-11-26_12:00:00", 6,
                               interval_h=3)
            out_path = os.path.join(d, "namelist.input")
            text, provenance, notes = cr.cmd_namelist_from_wrfinput(d, out_path)

            self.assertTrue(os.path.exists(out_path))
            prov_path = os.path.join(d, "namelist_provenance.json")
            self.assertTrue(os.path.exists(prov_path))
            with open(prov_path) as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["provenance"], provenance)

            self.assertEqual(cs.get_first(text, "e_we"), "425")
            self.assertEqual(cs.get_first(text, "e_vert"), "45")
            self.assertEqual(cs.get_first(text, "p_top_requested"), "5000.0")
            self.assertEqual(cs.count_key(text, "eta_levels"), 1)
            self.assertEqual(cs.get_first(text, "num_soil_layers"), "4")
            self.assertEqual(cs.get_first(text, "interval_seconds"), "10800")
            self.assertEqual(cs.get_first(text, "run_hours"), "9")   # 6h span + one 3h interval
            self.assertEqual(provenance["interval_seconds"], "wrfbdy")
            self.assertEqual(provenance["run_hours"], "wrfbdy")
            self.assertEqual(provenance["eta_levels"], "wrfinput_attr")
            self.assertEqual(provenance["radt"], "rule_default")

    def test_hours_override_and_no_wrfbdy(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_wrfinput(os.path.join(d, "wrfinput_d01"))
            out_path = os.path.join(d, "namelist.input")
            text, provenance, notes = cr.cmd_namelist_from_wrfinput(d, out_path, hours=12.0)
            self.assertEqual(cs.get_first(text, "run_hours"), "12")
            self.assertEqual(provenance["run_hours"], "given")
            self.assertEqual(provenance["interval_seconds"], "rule_default")


# ---------------------------------------------------------------------------
# K2 (2026-09-24): new-domain -- for a user who only knows where and when.
# ---------------------------------------------------------------------------

class TestNewDomainFields(unittest.TestCase):
    START = datetime.datetime(2020, 6, 1, 0, 0, 0)


    def test_lat_75_is_polar(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            75.0, 10.0, (1200.0, 1200.0), 12.0, self.START, 6.0)
        self.assertEqual(wps["map_proj"], "polar")
        self.assertEqual(wps["truelat1"], 75.0)
        self.assertIsNone(wps["truelat2"])


    def test_tropical_physics_suite_and_convection_permitting(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            10.0, 100.0, (400.0, 400.0), 3.0, self.START, 6.0)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["physics"]["physics_suite"], ["'TROPICAL'"])
        self.assertEqual(gdict["physics"]["cu_physics"], [0])   # dx_km=3 < 4


class TestCmdNewDomain(unittest.TestCase):
    def test_refuses_partial_four_inputs(self):
        with self.assertRaises(cr.RebuildError):
            cr.cmd_new_domain(center="40,-98", size_km=None, dx_km="12", start="2020-06-01_00:00:00",
                              hours=6.0, out_dir="/tmp/whatever")

    def test_refuses_none_of_the_four(self):
        with self.assertRaises(cr.RebuildError):
            cr.cmd_new_domain(hours=6.0, out_dir="/tmp/whatever")


# ---------------------------------------------------------------------------
# K3 (2026-09-24, zhengzhou -- wrfinput_d0N but no wrfbdy_d01): finalize --keep-wrfinput keeps
# real.exe's own wrfbdy/wrflowinp (boundaries) but the case's ORIGINAL wrfinput (interior).
# compare() itself (grid_identity_check + metrics) is mocked here -- it needs real netCDF4
# files, and its own internals are covered elsewhere; this tests the ORCHESTRATION K3 adds:
# which file gets kept, that compare() is called with the right two paths, that a grid mismatch
# (compare() raising) propagates as a refusal, and that the report records the choice.
# ---------------------------------------------------------------------------

class TestCmdRebuildFinalizeKeepWrfinput(unittest.TestCase):
    def _setup(self, d, with_original_wrfinput=True):
        out_dir = os.path.join(d, "out")
        case_dir = os.path.join(d, "case")
        real_dir = os.path.join(out_dir, "real")
        wps_dir = os.path.join(out_dir, "wps")
        os.makedirs(real_dir)
        os.makedirs(wps_dir)
        os.makedirs(case_dir)
        plan = {"case_dir": case_dir, "max_dom": 1,
               "namelist_input": os.path.join(case_dir, "namelist.input"),
               "span_start": "2020-01-01_00:00:00", "span_end": "2020-01-01_06:00:00",
               "stated_start": "2020-01-01_00:00:00", "analog_start": None, "shift_s": 0.0,
               "forcing_chain_primary": {"dataset": "d084001", "prefix": "GFS",
                                        "vtable": "Vtable.GFS", "forcing_interval_h": 3},
               "forcing_pin": None}
        with open(os.path.join(out_dir, "plan.json"), "w") as f:
            json.dump(plan, f)
        with open(os.path.join(real_dir, "wrfinput_d01"), "w") as f:
            f.write("REAL-WRFINPUT")
        with open(os.path.join(real_dir, "wrfbdy_d01"), "w") as f:
            f.write("REAL-WRFBDY")
        with open(os.path.join(real_dir, "namelist.input"), "w") as f:
            f.write("&time_control\n/\n")
        if with_original_wrfinput:
            with open(os.path.join(case_dir, "wrfinput_d01"), "w") as f:
                f.write("ORIGINAL-WRFINPUT")
        return out_dir, case_dir, real_dir

    def test_default_copies_reals_own_wrfinput(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir, case_dir, real_dir = self._setup(d)
            with mock.patch.object(cs, "inspect_case", return_value=([], [], {})):
                cr.cmd_rebuild_finalize(out_dir, keep_wrfinput=False)
            with open(os.path.join(out_dir, "case", "wrfinput_d01")) as f:
                self.assertEqual(f.read(), "REAL-WRFINPUT")

    def test_keep_wrfinput_keeps_original_and_calls_compare(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir, case_dir, real_dir = self._setup(d)
            fake_report = {"verdict": "consistent"}
            with mock.patch.object(cs, "inspect_case", return_value=([], [], {})), \
                 mock.patch.object(cr, "compare", return_value=fake_report) as mock_compare:
                report = cr.cmd_rebuild_finalize(out_dir, keep_wrfinput=True)
            mock_compare.assert_called_once_with(
                os.path.join(case_dir, "wrfinput_d01"), os.path.join(real_dir, "wrfinput_d01"))
            with open(os.path.join(out_dir, "case", "wrfinput_d01")) as f:
                self.assertEqual(f.read(), "ORIGINAL-WRFINPUT")
            # wrfbdy still comes from real.exe -- the case had none of its own
            with open(os.path.join(out_dir, "case", "wrfbdy_d01")) as f:
                self.assertEqual(f.read(), "REAL-WRFBDY")
            self.assertEqual(report["keep_wrfinput"]["d01_vs_analysis_t0_compare"], fake_report)
            self.assertIn("wrfinput_d01", report["keep_wrfinput"]["domains"])
            self.assertIn("boundaries", report["keep_wrfinput"]["note"])
            self.assertIn("interior", report["keep_wrfinput"]["note"])

    def test_keep_wrfinput_without_original_raises(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir, case_dir, real_dir = self._setup(d, with_original_wrfinput=False)
            with mock.patch.object(cs, "inspect_case", return_value=([], [], {})):
                with self.assertRaises(cr.RebuildError):
                    cr.cmd_rebuild_finalize(out_dir, keep_wrfinput=True)

    def test_keep_wrfinput_propagates_grid_mismatch_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir, case_dir, real_dir = self._setup(d)
            with mock.patch.object(cs, "inspect_case", return_value=([], [], {})), \
                 mock.patch.object(cr, "compare",
                                   side_effect=cr.RebuildError("grid identity check FAILED: x")):
                with self.assertRaises(cr.RebuildError):
                    cr.cmd_rebuild_finalize(out_dir, keep_wrfinput=True)


# ---------------------------------------------------------------------------
# K4 (2026-09-24): `complete` front door -- inventory + path selection. Pure filesystem checks,
# no netCDF4 needed for the DECISION (files are touched empty). One test per inventory (a)-(f).
# ---------------------------------------------------------------------------

class TestInventoryAndChoosePathSixCases(unittest.TestCase):
    def _touch(self, d, *names):
        for name in names:
            open(os.path.join(d, name), "w").close()


    def test_b_wrfinput_and_wrfbdy_no_namelist_is_k1_only(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "wrfinput_d01", "wrfbdy_d01")
            inv = cr.inventory_case_dir(d)
            decision = cr.choose_completion_path(inv, have_center_size_dx_start=False)
            self.assertEqual(decision["path"], "k1_and_or_k3")
            self.assertTrue(decision["needs_k1"])
            self.assertFalse(decision["needs_k3"])

    def test_c_wrfinput_and_namelist_no_wrfbdy_is_k3_only(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "wrfinput_d01", "namelist.input")
            inv = cr.inventory_case_dir(d)
            decision = cr.choose_completion_path(inv, have_center_size_dx_start=False)
            self.assertEqual(decision["path"], "k1_and_or_k3")
            self.assertFalse(decision["needs_k1"])
            self.assertTrue(decision["needs_k3"])

    def test_d_namelist_and_projection_only_is_existing_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "namelist.input", "namelist.wps")
            inv = cr.inventory_case_dir(d)
            decision = cr.choose_completion_path(inv, have_center_size_dx_start=False)
            self.assertEqual(decision["path"], "existing_rebuild")


    def test_wrfinput_with_namelist_and_wrfbdy_but_no_namelist_wps_is_still_as_is(self):
        # namelist.input carries its own projection (via a wrfinput_d01, which is also present)
        # -- (a) does not require namelist.wps specifically, only namelist.input+wrfinput+wrfbdy.
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "namelist.input", "wrfinput_d01", "wrfbdy_d01")
            inv = cr.inventory_case_dir(d)
            decision = cr.choose_completion_path(inv, have_center_size_dx_start=False)
            self.assertEqual(decision["path"], "as_is")


class TestCmdComplete(unittest.TestCase):
    def test_refuses_partial_four_domain_args(self):
        with tempfile.TemporaryDirectory() as d:
            in_dir = os.path.join(d, "in")
            os.makedirs(in_dir)
            with self.assertRaises(cr.RebuildError):
                cr.cmd_complete(in_dir, os.path.join(d, "out"), center="40,-98", size_km=None,
                                dx_km=12.0, start="2020-06-01_00:00:00")

    def test_refuses_when_nothing_locates_the_domain(self):
        with tempfile.TemporaryDirectory() as d:
            in_dir = os.path.join(d, "in")
            os.makedirs(in_dir)
            with self.assertRaises(cr.RebuildError):
                cr.cmd_complete(in_dir, os.path.join(d, "out"))

    def test_k2_then_rebuild_path_calls_new_domain_and_rebuild_plan(self):
        with tempfile.TemporaryDirectory() as d:
            in_dir = os.path.join(d, "in")
            out_dir = os.path.join(d, "out")
            os.makedirs(in_dir)
            fake_plan = {"span_start": "2020-06-01_00:00:00"}
            with mock.patch.object(cr, "cmd_rebuild_plan", return_value=fake_plan) as mock_plan:
                report = cr.cmd_complete(in_dir, out_dir, center="40,-98", size_km="1200,1200",
                                         dx_km=12.0, start="2020-06-01_00:00:00", hours=6.0)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "namelist.input")))
            mock_plan.assert_called_once()
            call_kwargs = mock_plan.call_args
            self.assertEqual(call_kwargs.args[0], out_dir)   # case_dir_for_plan = out_dir (K2's)
            self.assertEqual(report["decision"]["path"], "k2_then_rebuild")
            self.assertTrue(os.path.exists(os.path.join(out_dir, "completion_report.json")))


# ---------------------------------------------------------------------------
# R6-A: rebuild_span
# ---------------------------------------------------------------------------

class TestRebuildSpan(unittest.TestCase):
    def test_unshifted_span_equals_stated(self):
        nl = {"time_control": {"start_year": [2020], "start_month": [1], "start_day": [1],
                               "start_hour": [0], "run_hours": [6]}}
        stated_start, stated_end, analog_start, span_start, span_end, run_s = cr.rebuild_span(nl)
        self.assertIsNone(analog_start)
        self.assertEqual(span_start, stated_start)
        self.assertEqual(span_end, stated_end)
        self.assertEqual(run_s, 21600)

    def test_date_shift_to_moves_the_span_only(self):
        nl = {"time_control": {"start_year": [2020], "start_month": [1], "start_day": [1],
                               "start_hour": [0], "run_hours": [6]}}
        stated_start, stated_end, analog_start, span_start, span_end, run_s = cr.rebuild_span(
            nl, date_shift_to="2021-06-15_00:00:00")
        self.assertEqual(stated_start, datetime.datetime(2020, 1, 1))
        self.assertEqual(analog_start, datetime.datetime(2021, 6, 15))
        self.assertEqual(span_start, datetime.datetime(2021, 6, 15))
        self.assertEqual(span_end, datetime.datetime(2021, 6, 15, 6, 0, 0))
        self.assertEqual(run_s, 21600)   # run length itself is unchanged by the shift


# ---------------------------------------------------------------------------
# T4b (2026-09-24): rebuild fetch must always write forcing.json, even on failure, naming the
# first missing URL + its final status -- mirrors coldstart_sample.resolve_and_fetch_forcing.
# ---------------------------------------------------------------------------

class TestRebuildFetchAlwaysWritesRecordOnFailure(unittest.TestCase):
    def test_failure_writes_forcing_json_and_names_first_missing_url(self):
        with tempfile.TemporaryDirectory() as d:
            plan = {"span_start": "2010-01-01_00:00:00", "span_end": "2010-01-01_06:00:00",
                   "forcing_pin": None}
            with open(os.path.join(d, "plan.json"), "w") as f:
                json.dump(plan, f)

            def fake_probe(entries):
                results = []
                for fn, urls in entries:
                    results.append({
                        "filename": fn, "chosen_url": None,
                        "checked": [{"url": urls[0], "status": None,
                                    "attempts": [{"status": 503}, {"status": 404}]}]})
                return False, results

            with self.assertRaises(cr.RebuildError) as ctx:
                cr.cmd_rebuild_fetch(d, probe_fn=fake_probe)
            msg = str(ctx.exception)
            self.assertIn("first missing:", msg)
            self.assertIn("404", msg)

            with open(os.path.join(d, "forcing.json")) as f:
                forcing = json.load(f)
            self.assertEqual(forcing["status"], "failed")
            self.assertIsNone(forcing["chosen"])
            self.assertTrue(forcing["candidates_tried"])


# ---------------------------------------------------------------------------
# R6-A: render_rebuild_sbatch -- text-content invariants (no cluster access)
# ---------------------------------------------------------------------------

class TestRenderRebuildSbatch(unittest.TestCase):
    def _plan(self, max_dom=1, verify_steps=0):
        return {
            "nodes": 2, "config": {"ranks": 100, "omp": 2, "nx": 10, "ny": 10, "numtiles": 1},
            "max_dom": max_dom, "verify_steps": verify_steps,
            "span_start": "2020-01-01_00:00:00", "span_end": "2020-01-01_06:00:00",
            "interval_seconds": 10800,
        }

    def _chosen(self):
        return {"prefix": "GFS", "vtable": "Vtable.GFS", "forcing_interval_h": 3}


    def test_wrf_smoke_run_present_when_verify_steps_given(self):
        text = cr.render_rebuild_sbatch(self._plan(verify_steps=20), self._chosen())
        self.assertIn("wrf.exe smoke run: 20 steps", text)
        self.assertIn("export OMP_NUM_THREADS=2", text)
        self.assertIn("unset OMP_PLACES OMP_PROC_BIND", text)


    def test_ml_model_wires_grib_python_and_apply_step(self):
        plan = self._plan()
        plan["ml_model"] = "/out/dir/report.json"
        text = cr.render_rebuild_sbatch(plan, self._chosen())
        self.assertIn("ML_MODEL=/out/dir/report.json", text)
        self.assertIn("state_correction_model.py", text)
        self.assertIn("GRIB_PY=${GRIB_PYTHON:-%s/envs/grib/.pixi/envs/default/bin/python}" % cs._TREE,
                     text)
        self.assertIn("\"$GRIB_PY\" \"$TOOL_SCM\" apply", text)
        self.assertIn("--targets-from-report", text)
        self.assertIn("REBUILD_DIR/ml_status.txt", text)


# ---------------------------------------------------------------------------
# R6-B: pure metric functions (numpy arrays, no netCDF4 needed)
# ---------------------------------------------------------------------------

class TestCompareMetrics(unittest.TestCase):

    def test_max_abs_diff_exact(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 5.0, 3.0])
        self.assertEqual(cr.max_abs_diff(a, b), 3.0)

    def test_agreement_pct_exact(self):
        a = np.array([1, 1, 0, 0])
        b = np.array([1, 0, 0, 0])
        self.assertEqual(cr.agreement_pct(a, b), 75.0)

    def test_bias_exact(self):
        a = np.array([5.0, 5.0])
        b = np.array([3.0, 4.0])
        self.assertAlmostEqual(cr.bias(a, b), 1.5, places=9)


    def test_compute_pw_exact(self):
        qvapor = np.ones((2, 1, 1)) * 0.01
        p = np.array([[[90000.0]], [[50000.0]]])
        pb = np.zeros((2, 1, 1))
        pw = cr.compute_pw(qvapor, p, pb)
        # dp = p_full[0]-p_full[1] = 40000 (only nz-1=1 layer contributes); pw = 0.01*40000/9.81
        self.assertAlmostEqual(float(pw[0, 0]), 0.01 * 40000.0 / cr.G_ACCEL, places=9)

    def test_theta_to_temp_exact(self):
        theta_pert = np.array([[[0.0]]])   # theta_full = 300 K
        p = np.array([[[100000.0]]])       # p_full = P0 -> (p/p0)^(Rd/cp) = 1
        pb = np.zeros((1, 1, 1))
        t = cr.theta_to_temp(theta_pert, p, pb)
        self.assertAlmostEqual(float(t[0, 0, 0]), 300.0, places=9)


    def test_interp_pressure_level_field_matches_per_column_loop(self):
        rng = np.random.default_rng(7)
        nz, ny, nx = 12, 9, 11
        p = np.sort(rng.uniform(5000.0, 105000.0, (nz, ny, nx)), axis=0)[::-1].copy()
        p[3, 0, 0] = p[4, 0, 0]                      # equal neighbours -> field[idx]
        p[0, 1, 1] = 80000.0                         # target above the surface pressure
        f = rng.normal(280.0, 10.0, (nz, ny, nx))
        for target in (85000.0, 50000.0, 100000.0, 2000.0):
            got = cr.interp_pressure_level_field(f, p, target)
            want = np.array([[cr.interp_pressure_level_column(f[:, j, i], p[:, j, i], target)
                              for i in range(nx)] for j in range(ny)])
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-9, equal_nan=True)
            self.assertTrue(np.array_equal(np.isnan(got), np.isnan(want)))


class TestGridIdentityCheck(unittest.TestCase):
    def _grid(self, dx=12000.0, cen_lat=40.0, cen_lon=-98.0, we=100, sn=100):
        return {
            "attrs": {"MAP_PROJ": 1, "DX": dx, "DY": dx, "TRUELAT1": 30.0, "TRUELAT2": 60.0,
                     "STAND_LON": -98.0, "CEN_LAT": cen_lat, "CEN_LON": cen_lon,
                     "WEST-EAST_GRID_DIMENSION": we, "SOUTH-NORTH_GRID_DIMENSION": sn,
                     "BOTTOM-TOP_GRID_DIMENSION": 45},
            "dims": {"west_east": we - 1, "south_north": sn - 1, "bottom_top": 44},
            "corners": {"SW": (35.0, -103.0), "SE": (35.0, -93.0), "NW": (45.0, -103.0),
                       "NE": (45.0, -93.0), "center": (cen_lat, cen_lon)},
        }


    def test_center_beyond_tolerance_fails(self):
        a = self._grid(cen_lat=40.0)
        b = self._grid(cen_lat=40.5)
        ok, reasons = cr.grid_identity_check(a, b)
        self.assertFalse(ok)


    def _mercator_grid(self, truelat2):
        g = self._grid()
        g["attrs"]["MAP_PROJ"] = 3
        g["attrs"]["TRUELAT2"] = truelat2
        return g

    def test_mercator_grids_differing_only_in_unused_truelat2_pass(self):
        # T10 (2026-09-24, matthew): MAP_PROJ=3 (Mercator) does not use TRUELAT2 -- a leftover
        # 60 in the original wrfinput vs WRF's own "unset" sentinel 1e20 in the rebuilt one must
        # not refuse an otherwise-identical grid.
        a = self._mercator_grid(60.0)
        b = self._mercator_grid(1e20)
        ok, reasons = cr.grid_identity_check(a, b)
        self.assertTrue(ok, reasons)
        self.assertFalse(any("TRUELAT2" in r for r in reasons))


class TestCompareVerdict(unittest.TestCase):

    def test_different_static_data(self):
        verdict, static_ok, state_ok = cr.compare_verdict(100.0, 50.0, 0.95, 0.9, 0.8)
        self.assertEqual(verdict, "different-static-data")
        self.assertFalse(static_ok)
        self.assertTrue(state_ok)


# ---------------------------------------------------------------------------
# R6-C: candidate_dates (DOY window + wrap), block_average, scoring/ranking
# ---------------------------------------------------------------------------


class TestClimatologicalAnalog(unittest.TestCase):
    """T14 (2026-09-26): climatological_analog() replaces max(candidate_dates(...)) for the
    no-target case -- that was NOT "same DOY, most recent year" (it was the +-doy_window
    window's LAST day), e.g. stated 2026-09-26 with doy_window=45 -> 2025-11-10, not
    2025-09-26."""


    def test_leap_doy_366_falls_back_to_dec31_in_non_leap_y1(self):
        stated = datetime.datetime(2028, 12, 31, 12, 0, 0)   # DOY 366 (2028 is a leap year)
        got = cr.climatological_analog(stated, (2021, 2025), hour="same")   # 2025 is not leap
        self.assertEqual(got, datetime.datetime(2025, 12, 31, 12, 0, 0))


    def test_no_target_cmd_analog_writes_corrected_date(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            with open(os.path.join(case_dir, "namelist.input"), "w") as f:
                f.write(" &time_control\n start_year=2026,\n start_month=9,\n start_day=26,\n"
                        " start_hour=0,\n run_hours=6,\n /\n &domains\n max_dom=1,\n /\n")
            out_dir = os.path.join(d, "out")
            result = cr.cmd_analog(case_dir, out_dir, years="2021-2025", doy_window=45)
            self.assertEqual(result["quality"], "no-target")
            self.assertEqual(result["chosen"]["date"][:10], "2025-09-26")
            with open(os.path.join(out_dir, "analog.json")) as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["chosen"]["date"][:10], "2025-09-26")


# ---------------------------------------------------------------------------
# T3 (2026-09-24): analog target must come from the case's OWN wrfinput_d01, never from
# fetching GRIB at the stated start date; wind rotation grid-relative -> earth-relative.
# ---------------------------------------------------------------------------

class TestRotateGridToEarth(unittest.TestCase):
    def test_90_degree_rotation(self):
        # cosalpha=0, sinalpha=1 (a 90 degree map rotation) -> u_e = -v, v_e = u
        u = np.array([3.0])
        v = np.array([4.0])
        cosalpha = np.array([0.0])
        sinalpha = np.array([1.0])
        u_e, v_e = cr.rotate_grid_to_earth(u, v, cosalpha, sinalpha)
        self.assertAlmostEqual(float(u_e[0]), -4.0, places=9)
        self.assertAlmostEqual(float(v_e[0]), 3.0, places=9)


class TestDestagger(unittest.TestCase):
    def test_destagger_u_averages_last_axis(self):
        u = np.array([[[1.0, 3.0, 5.0]]])   # [bottom_top=1, south_north=1, west_east_stag=3]
        out = cr.destagger_u(u)
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_allclose(out[0, 0], [2.0, 4.0])

    def test_destagger_v_averages_axis_1(self):
        v = np.array([[[1.0], [3.0], [5.0]]])   # [bottom_top=1, south_north_stag=3, west_east=1]
        out = cr.destagger_v(v)
        self.assertEqual(out.shape, (1, 2, 1))
        np.testing.assert_allclose(out[0, :, 0], [2.0, 4.0])


@unittest.skipUnless(HAS_NETCDF4, "netCDF4 not installed in this dev environment")
class TestTargetFieldsFromWrfinput(unittest.TestCase):
    def test_synthetic_wrfinput(self):
        import netCDF4
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wrfinput_d01")
            ds = netCDF4.Dataset(path, "w")
            ds.createDimension("Time", None)
            ds.createDimension("bottom_top", 2)
            ds.createDimension("bottom_top_stag", 3)
            ds.createDimension("south_north", 2)
            ds.createDimension("south_north_stag", 3)
            ds.createDimension("west_east", 2)
            ds.createDimension("west_east_stag", 3)

            p = ds.createVariable("P", "f4", ("Time", "bottom_top", "south_north", "west_east"))
            pb = ds.createVariable("PB", "f4", ("Time", "bottom_top", "south_north", "west_east"))
            p[0] = 0.0
            pb[0, 0] = 90000.0
            pb[0, 1] = 50000.0

            t2 = ds.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
            t2[0] = 290.0
            ds.close()

            fields = cr.target_fields_from_wrfinput(path)
            self.assertIn("tmp2m", fields)
            np.testing.assert_allclose(fields["tmp2m"], 290.0)
            # no PH/PHB/T/U/V/QVAPOR given -> those keys legitimately absent, not crashed
            self.assertNotIn("hgt500", fields)


# ---------------------------------------------------------------------------
# T8 (2026-09-24, matthew -- highest priority after T3/T11, blocks V1): complete_namelist's
# vertical_grid override -- eta_levels/p_top_requested come straight from the wrfinput's own
# ZNW/P_TOP and dzbot/dzstretch_s/dzstretch_u/auto_levels_opt (which conflict with an explicit
# eta_levels list) are removed outright.
# ---------------------------------------------------------------------------

class TestCompleteNamelistVerticalGridOverride(unittest.TestCase):
    def _nl_with_dzstretch(self):
        return """\
&time_control
run_days                            = 0,
run_hours                           = 3,
run_minutes                         = 0,
run_seconds                         = 0,
interval_seconds                    = 10800
/

&domains
max_dom                             = 1,
e_we                                = 100,
e_sn                                = 100,
dx                                  = 12000,
e_vert                              = 45,
p_top_requested                     = 5000,
dzbot                               = 30,
dzstretch_s                         = 1.11,
dzstretch_u                         = 1.10,
auto_levels_opt                     = 2,
/

&physics
/
"""

    def test_eta_levels_written_exactly_dz_keys_removed_change_log(self):
        # T12: vertical_grid's dict key is 'p_top_pa' (plan.json's own schema), not
        # 'p_top_requested' -- that name is reserved for the NAMELIST key it's written into.
        vertical_grid = {"eta_levels": [1.0, 0.6, 0.3, 0.0], "p_top_pa": 4000.0}
        txt, changes = cr.complete_namelist(self._nl_with_dzstretch(), BASE_FACTS,
                                            vertical_grid=vertical_grid)
        # eta_levels written exactly
        self.assertEqual(cs.get_first(txt, "eta_levels"), "1.0")
        eta_line = re.search(r"eta_levels\s*=\s*([^\n/]+)", txt, re.I).group(1)
        self.assertEqual([float(x) for x in eta_line.rstrip(", \n").split(",")],
                         [1.0, 0.6, 0.3, 0.0])
        # e_vert synced to len(eta_levels), p_top_requested overridden (not the namelist's 5000)
        self.assertEqual(cs.get_first(txt, "e_vert"), "4")
        self.assertEqual(cs.get_first(txt, "p_top_requested"), "4000.0")
        # conflicting keys removed
        for key in ("dzbot", "dzstretch_s", "dzstretch_u", "auto_levels_opt"):
            self.assertEqual(cs.count_key(txt, key), 0)
        # change log entries present for every one of the above
        changed_keys = {c["key"] for c in changes}
        for key in ("eta_levels", "e_vert", "p_top_requested", "dzbot", "dzstretch_s",
                   "dzstretch_u", "auto_levels_opt"):
            self.assertIn(key, changed_keys)


class TestCmdAnalogTargetNeverFromStatedDate(unittest.TestCase):
    def test_target_built_from_wrfinput_not_fetched_at_stated_start(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            with open(os.path.join(case_dir, "namelist.input"), "w") as f:
                f.write(" &time_control\n start_year=2020,\n start_month=1,\n start_day=10,\n"
                        " start_hour=0,\n run_hours=6,\n /\n &domains\n max_dom=1,\n /\n")
            wi_path = os.path.join(case_dir, "wrfinput_d01")
            open(wi_path, "w").close()   # just needs to exist -- reading it is mocked below

            fake_target = {"hgt500": np.ones((20, 20)) * 5000.0}
            fetch_calls = []

            def fake_fetch(dt, lat2d, lon2d, cache_dir, grib_dir=None, source="auto"):
                fetch_calls.append(dt)
                return {"hgt500": np.ones((20, 20)) * 5000.0}

            with mock.patch.object(cr, "target_fields_from_wrfinput", return_value=fake_target), \
                 mock.patch.object(cr, "_fetch_and_regrid_candidate", side_effect=fake_fetch), \
                 mock.patch.object(cr.cs, "_grib_state") as fake_gs_mod, \
                 mock.patch.object(cr.cs, "_metdata") as fake_md_mod:
                fake_gs_mod.return_value.target_grid.return_value = (
                    np.zeros((20, 20)), np.zeros((20, 20)), "bbox")
                fake_md_mod.return_value.select_source.return_value = ("aws", "recent")
                out = cr.cmd_analog(case_dir, os.path.join(d, "out"), years="2021-2021",
                                    doy_window=1, top=2)

            stated_start = datetime.datetime(2020, 1, 10, 0, 0, 0)
            self.assertNotIn(stated_start, fetch_calls)
            self.assertTrue(all(c.year in (2021,) for c in fetch_calls))
            self.assertNotEqual(out["quality"], "no-target")


class TestScoreCandidateAndRanking(unittest.TestCase):
    def setUp(self):
        self.target = {"f": np.array([[10.0, 20.0], [30.0, 40.0]])}

    def test_field_score_exact_values(self):
        T = self.target["f"]
        cases = {
            "perfect": (T.copy(), 0.0),
            "scaled": (2 * T, 0.5),
            "anti": (np.array([[40.0, 30.0], [20.0, 10.0]]), 1.5),
        }
        for name, (field, expected_score) in cases.items():
            corr, nrmse = cr.anomaly_pattern_corr_nrmse(field, T)
            score = cr.field_score(corr, nrmse)
            self.assertAlmostEqual(score, expected_score, places=5, msg=name)


class TestQualityFlag(unittest.TestCase):
    def test_good(self):
        self.assertEqual(cr.quality_flag([0.9, 0.85, 0.95]), "good")

    def test_fair(self):
        self.assertEqual(cr.quality_flag([0.9, 0.65, 0.95]), "fair")

    def test_poor(self):
        self.assertEqual(cr.quality_flag([0.9, 0.5, 0.95]), "poor")

    def test_no_data_is_poor(self):
        self.assertEqual(cr.quality_flag([]), "poor")


# ---------------------------------------------------------------------------
# Numpy-free module import (case_rebuild.py's rebuild fix-span-* subcommands run inside a batch
# job on scc's /usr/bin/python3, which has no numpy).
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()


class TestWind850Verdict(unittest.TestCase):


    def test_nan_wind_fails_and_absent_wind_is_ignored(self):
        self.assertEqual(cr.compare_verdict(10.0, 95.0, 0.95, 0.9, 0.8,
                                            corr_u850=float("nan"), corr_v850=0.99)[0],
                         "state-mismatch")
        self.assertEqual(cr.compare_verdict(10.0, 95.0, 0.95, 0.9, 0.8)[0], "consistent")


def test_plan_is_fitted_to_a_machine_with_fewer_cores(monkeypatch):
    # haitang2017 (2026-10-26): prep wps planned 144 ranks (1x144) for a 144-core node; after
    # cores_per_node 72, prep real asserted ppn=144 > 72
    import wtune.tools.case_rebuild as cr
    assert cr.near_square(144) == (12, 12) and cr.near_square(72) == (8, 9) and cr.near_square(7) == (1, 7)
    monkeypatch.setenv("WTUNE_CORES_PER_NODE", "72")
    plan = dict(nodes=1, config=dict(ranks=144, omp=1, nx=1, ny=144, numtiles=1))
    assert cr.fit_plan_to_machine(plan)["config"] == dict(ranks=72, omp=1, nx=8, ny=9, numtiles=1)
    small = dict(nodes=2, config=dict(ranks=100, omp=1, nx=10, ny=10, numtiles=1))
    assert cr.fit_plan_to_machine(small)["config"]["ranks"] == 100        # fits: unchanged
