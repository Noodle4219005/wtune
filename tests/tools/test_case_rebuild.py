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
import math
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


class TestCompleteNamelistAlwaysSync(unittest.TestCase):
    def setUp(self):
        self.txt, self.changes = cr.complete_namelist(BASE_NL_TEXT, BASE_FACTS)

    def test_met_em_facts_synced(self):
        self.assertEqual(cs.get_first(self.txt, "num_metgrid_levels"), "32")
        self.assertEqual(cs.get_first(self.txt, "num_metgrid_soil_levels"), "4")
        self.assertEqual(cs.get_first(self.txt, "num_land_cat"), "21")

    def test_interval_seconds_synced_to_forcing(self):
        self.assertEqual(cs.get_first(self.txt, "interval_seconds"), "21600")
        change = next(c for c in self.changes if c["key"] == "interval_seconds")
        self.assertEqual(change["old"], "10800")
        self.assertEqual(change["new"], "21600")

    def test_dates_synced_to_span(self):
        self.assertEqual(cs.get_first(self.txt, "start_year"), "2020")
        self.assertEqual(cs.get_first(self.txt, "end_hour"), "6")
        self.assertEqual(cs.get_first(self.txt, "run_seconds"), "21600")

    def test_run_hours_overwritten_when_span_disagrees(self):
        # original run_hours=3 (10800s) but the span is 6h (21600s) -- must be overwritten, not
        # left alone (this is "always sync", not "fill only if missing").
        self.assertEqual(cs.get_first(self.txt, "run_hours"), "0")
        self.assertEqual(cs.get_first(self.txt, "run_seconds"), "21600")


class TestCompleteNamelistFillOnlyIfMissing(unittest.TestCase):
    def setUp(self):
        self.txt, self.changes = cr.complete_namelist(BASE_NL_TEXT, BASE_FACTS)

    def test_e_vert_and_ptop_filled(self):
        self.assertEqual(cs.get_first(self.txt, "e_vert"), "45")
        self.assertEqual(cs.get_first(self.txt, "p_top_requested"), "5000")

    def test_time_step_is_6x_dx_km(self):
        # dx_km=12.0 -> largest int <= 72.0 that is >=1 -> 72 (matches conus12's own real value)
        self.assertEqual(cs.get_first(self.txt, "time_step"), "72")

    def test_history_interval_default_when_absent(self):
        self.assertEqual(cs.get_first(self.txt, "history_interval"), "60")

    def test_restart_defaults(self):
        self.assertEqual(cs.get_first(self.txt, "restart"), ".false.")
        # run_minutes = 21600/60 = 360
        self.assertEqual(cs.get_first(self.txt, "restart_interval"), "360")

    def test_io_form_defaults(self):
        for key in ("io_form_history", "io_form_restart", "io_form_input", "io_form_boundary"):
            self.assertEqual(cs.get_first(self.txt, key), "2")

    def test_physics_suite_added_when_no_scheme_key_at_all(self):
        self.assertEqual(cs.get_first(self.txt, "physics_suite"), "'CONUS'")
        change = next(c for c in self.changes if c["key"] == "physics_suite")
        self.assertIn("no &physics scheme key", change["reason"])

    def test_never_overrides_a_present_user_value(self):
        text_with_e_vert = BASE_NL_TEXT.replace("&domains\n", "&domains\n e_vert = 33,\n")
        txt, changes = cr.complete_namelist(text_with_e_vert, BASE_FACTS)
        self.assertEqual(cs.get_first(txt, "e_vert"), "33")
        self.assertFalse(any(c["key"] == "e_vert" for c in changes))

    def test_physics_suite_not_added_when_a_scheme_key_present(self):
        text_with_mp = BASE_NL_TEXT.replace("&physics\n", "&physics\n mp_physics = 6,\n")
        txt, changes = cr.complete_namelist(text_with_mp, BASE_FACTS)
        self.assertEqual(cs.count_key(txt, "physics_suite"), 0)

    def test_history_interval_not_added_when_history_interval_m_present(self):
        text_with_him = BASE_NL_TEXT.replace("&time_control\n",
                                             "&time_control\n history_interval_m = 30,\n")
        txt, changes = cr.complete_namelist(text_with_him, BASE_FACTS)
        self.assertEqual(cs.count_key(txt, "history_interval"), 0)
        self.assertEqual(cs.get_first(txt, "history_interval_m"), "30")

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

    def test_no_placeholder_note_when_physics_suite_absent(self):
        txt, changes = cr.complete_namelist(BASE_NL_TEXT, BASE_FACTS)
        self.assertFalse(any(c["key"] == "physics_suite_placeholders" for c in changes))

    def test_no_placeholder_note_when_no_keys_are_minus1(self):
        text = BASE_NL_TEXT.replace(
            "&physics\n", "&physics\n physics_suite = 'CONUS',\n mp_physics = 8,\n")
        txt, changes = cr.complete_namelist(text, BASE_FACTS)
        self.assertFalse(any(c["key"] == "physics_suite_placeholders" for c in changes))


class TestCompleteNamelistNestedParentTimeStepRatio(unittest.TestCase):
    def test_parent_time_step_ratio_filled_per_nest(self):
        facts = dict(BASE_FACTS, max_dom=2, parent_grid_ratio=[1, 3])
        txt, changes = cr.complete_namelist(BASE_NL_TEXT, facts)
        self.assertEqual(cs.get_first(txt, "parent_time_step_ratio"), "1")
        # full list check (set_list_values wrote "1, 3,")
        m = __import__("re").search(r"parent_time_step_ratio\s*=\s*([^\n/]+)", txt)
        self.assertEqual(m.group(1).strip().rstrip(","), "1, 3")


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

    def test_restart_is_single_valued_regardless_of_max_dom(self):
        registry = cr.load_registry()
        self.assertFalse(cr.registry_per_domain("restart", registry))
        facts = dict(BASE_FACTS, max_dom=2, parent_grid_ratio=[1, 3])
        txt, changes = cr.complete_namelist(BASE_NL_TEXT, facts, registry=registry)
        m = re.search(r"(?<![A-Za-z0-9_])restart\s*=\s*([^\n/]+)", txt)
        self.assertEqual(m.group(1).strip().rstrip(","), ".false.")

    def test_single_domain_unaffected(self):
        # max_dom=1: per-domain vs single-valued is not observable (both render one value) --
        # confirms the fix does not disturb the existing single-domain behavior.
        txt, changes = cr.complete_namelist(BASE_NL_TEXT, BASE_FACTS)
        self.assertEqual(cs.get_first(txt, "frames_per_outfile"), "1")


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

    def test_k2_output_becomes_complete_through_complete_namelist(self):
        # K2 (new-domain) only ever writes &time_control/&domains/&physics -- complete_namelist
        # must fill in the rest automatically.
        groups, wps, provenance, notes = cr.new_domain_fields(
            40.0, -98.0, (1200.0, 1200.0), 12.0, datetime.datetime(2020, 6, 1), 6.0)
        k2_text = cr.render_namelist_input(groups)
        parsed_before = self._parse(k2_text)
        for missing in ("fdda", "dynamics", "bdy_control", "grib2", "namelist_quilt"):
            self.assertNotIn(missing, parsed_before)

        facts = dict(BASE_FACTS, max_dom=1)
        txt, changes = cr.complete_namelist(k2_text, facts)
        parsed_after = self._parse(txt)
        for group in ("time_control", "domains", "physics", "fdda", "dynamics", "bdy_control",
                     "grib2", "namelist_quilt"):
            self.assertIn(group, parsed_after, "missing &%s after complete_namelist" % group)
        self.assertEqual(parsed_after["fdda"], {})
        self.assertEqual(parsed_after["grib2"], {})
        bdy = parsed_after["bdy_control"]
        self.assertEqual(bdy["spec_bdy_width"], [5])
        self.assertEqual(bdy["spec_zone"], [1])
        self.assertEqual(bdy["relax_zone"], [4])
        self.assertEqual(bdy["specified"], [True])
        self.assertEqual(bdy["nested"], [False])
        quilt = parsed_after["namelist_quilt"]
        self.assertEqual(quilt["nio_tasks_per_group"], [0])
        self.assertEqual(quilt["nio_groups"], [1])

    def test_bdy_control_partial_present_keeps_user_value_fills_the_rest(self):
        text = BASE_NL_TEXT.replace(
            "&physics\n", "&physics\n/\n\n&bdy_control\n specified = .true.,\n")
        txt, changes = cr.complete_namelist(text, BASE_FACTS)
        parsed = self._parse(txt)
        bdy = parsed["bdy_control"]
        self.assertEqual(bdy["specified"], [True])   # user's own value, untouched
        self.assertEqual(bdy["spec_bdy_width"], [5])
        self.assertEqual(bdy["spec_zone"], [1])
        self.assertEqual(bdy["relax_zone"], [4])
        self.assertEqual(bdy["nested"], [False])
        self.assertFalse(any(c["key"] == "specified" for c in changes))
        self.assertTrue(any(c["key"] == "spec_bdy_width" for c in changes))

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

    def test_registry_has_physics_group_with_known_scheme_keys(self):
        registry = cr.load_registry()
        valid = cr.registry_valid_keys(registry)
        for key in ("mp_physics", "ra_lw_physics", "cu_physics"):
            self.assertIn(key, valid)

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

class TestCompleteNamelistGroupPlacement(unittest.TestCase):
    def test_missing_p_top_requested_and_e_vert_land_in_domains(self):
        nl_text = """\
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
        self.assertEqual(cs.count_key(nl_text, "p_top_requested"), 0)
        self.assertEqual(cs.count_key(nl_text, "e_vert"), 0)
        txt, changes = cr.complete_namelist(nl_text, BASE_FACTS)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "namelist.input")
            with open(path, "w") as f:
                f.write(txt)
            parsed = predict.parse_namelist(path)
        self.assertIn("p_top_requested", parsed.get("domains", {}))
        self.assertNotIn("p_top_requested", parsed.get("time_control", {}))
        self.assertIn("e_vert", parsed.get("domains", {}))
        self.assertNotIn("e_vert", parsed.get("time_control", {}))
        # still lands correctly in &time_control (unaffected by the group fix)
        self.assertIn("restart", parsed.get("time_control", {}))

    def test_no_change_logged_for_numerically_equal_values(self):
        # '01' -> '1' (or '10800' -> '10800.0') is not a change -- only the SPELLING differs.
        nl_text = BASE_NL_TEXT.replace("interval_seconds                    = 10800",
                                       "interval_seconds                    = 010800")
        facts = dict(BASE_FACTS, interval_seconds=10800)   # same value, leading-zero spelling
        txt, changes = cr.complete_namelist(nl_text, facts)
        self.assertFalse(any(c["key"] == "interval_seconds" for c in changes))


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

    def test_full_mapping_and_provenance(self):
        wrfbdy_times = [datetime.datetime(2019, 11, 26, 12, 0, 0) + datetime.timedelta(hours=3 * i)
                       for i in range(3)]   # 0h/3h/6h -> interval=3h
        groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(
            self._facts(), wrfbdy_times=wrfbdy_times, hours=None)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["domains"]["e_we"], [425])
        self.assertEqual(gdict["domains"]["e_sn"], [300])
        self.assertEqual(gdict["domains"]["e_vert"], [45])
        self.assertEqual(gdict["domains"]["dx"], [12000.0])
        self.assertEqual(gdict["domains"]["parent_grid_ratio"], [1])
        self.assertEqual(gdict["domains"]["parent_time_step_ratio"], [1])
        self.assertEqual(gdict["domains"]["time_step"], [72])
        self.assertEqual(gdict["domains"]["eta_levels"], [1.0, 0.5, 0.0])
        self.assertEqual(gdict["domains"]["p_top_requested"], [5000.0])
        self.assertEqual(gdict["domains"]["max_dom"], [1])
        self.assertEqual(gdict["physics"]["num_soil_layers"], [4])
        self.assertEqual(gdict["physics"]["num_land_cat"], [21])
        self.assertEqual(gdict["physics"]["mp_physics"], [8])
        # T15(a): gwd_opt/damp_opt/diff_opt/km_opt/use_theta_m are &dynamics per the Registry
        # catalog, not &physics -- job 986's wrf.exe FATAL traced back to this misplacement.
        self.assertEqual(gdict["dynamics"]["gwd_opt"], [1])
        self.assertEqual(gdict["dynamics"]["damp_opt"], [3])
        self.assertEqual(gdict["physics"]["radt"], [12])   # round(dx_km=12.0)
        self.assertEqual(gdict["time_control"]["start_year"], [2019])
        self.assertEqual(gdict["time_control"]["start_month"], [11])
        self.assertEqual(gdict["time_control"]["start_day"], [26])
        self.assertEqual(gdict["time_control"]["start_hour"], [12])
        self.assertEqual(gdict["time_control"]["interval_seconds"], [10800])   # 3h, from wrfbdy
        # run length = last wrfbdy Times (start+6h) + one interval (3h) = 9h
        self.assertEqual(gdict["time_control"]["run_days"], [0])
        self.assertEqual(gdict["time_control"]["run_hours"], [9])
        self.assertEqual(gdict["time_control"]["run_minutes"], [0])
        self.assertEqual(gdict["time_control"]["run_seconds"], [0])
        self.assertEqual(gdict["time_control"]["history_interval"], [60])
        self.assertEqual(gdict["time_control"]["restart"], [".false."])
        self.assertEqual(gdict["time_control"]["restart_interval"], [9 * 60])
        for k in ("io_form_history", "io_form_restart", "io_form_input", "io_form_boundary"):
            self.assertEqual(gdict["time_control"][k], [2])

        self.assertEqual(provenance["e_we"], "wrfinput_attr")
        self.assertEqual(provenance["eta_levels"], "wrfinput_attr")
        self.assertEqual(provenance["p_top_requested"], "wrfinput_attr")
        self.assertEqual(provenance["num_soil_layers"], "wrfinput_attr")
        self.assertEqual(provenance["interval_seconds"], "wrfbdy")
        self.assertEqual(provenance["run_hours"], "wrfbdy")
        self.assertEqual(provenance["radt"], "rule_default")
        self.assertEqual(provenance["history_interval"], "rule_default")
        self.assertEqual(provenance["restart"], "rule_default")

    def test_hours_override_takes_precedence_and_is_tagged_given(self):
        groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(
            self._facts(), wrfbdy_times=None, hours=6.0)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["time_control"]["run_hours"], [6])
        self.assertEqual(provenance["run_hours"], "given")
        self.assertEqual(provenance["interval_seconds"], "rule_default")   # no wrfbdy at all
        self.assertTrue(any("rule_default 3h" in n for n in notes))

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

    def test_two_domains_parent_fields_per_domain(self):
        d01 = self._facts()[0]
        d02_attrs = dict(d01["attrs"])
        d02_attrs.update({"WEST-EAST_GRID_DIMENSION": 151, "SOUTH-NORTH_GRID_DIMENSION": 121,
                          "DX": 4000.0, "DY": 4000.0, "GRID_ID": 2, "PARENT_ID": 1,
                          "I_PARENT_START": 50, "J_PARENT_START": 40, "PARENT_GRID_RATIO": 3})
        d02 = {"attrs": d02_attrs}
        groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(
            [d01, d02], wrfbdy_times=None, hours=6.0)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["domains"]["e_we"], [425, 151])
        self.assertEqual(gdict["domains"]["parent_id"], [0, 1])
        self.assertEqual(gdict["domains"]["parent_grid_ratio"], [1, 3])
        self.assertEqual(gdict["domains"]["parent_time_step_ratio"], [1, 3])
        self.assertEqual(gdict["domains"]["max_dom"], [2])

    def test_physics_key_dimension_comes_from_the_registry_catalog(self):
        # T14 (2026-09-24, cluster check against conus12/sc26p1/zhengzhou real namelists):
        # damp_opt was rendered 'damp_opt = 3, 3, 3,' (broadcast per domain) but WRF's Registry
        # declares it dimension 1 -- a strict namelist reader can reject the extra values. The
        # expected dimension for each tested key is read from the SAME catalog K1 itself
        # consults (registry_per_domain), not hardcoded here, so this test tracks the catalog
        # rather than encoding an assumption about which keys are which.
        d01 = self._facts()[0]
        d02_attrs = dict(d01["attrs"])
        d02_attrs.update({"WEST-EAST_GRID_DIMENSION": 151, "SOUTH-NORTH_GRID_DIMENSION": 121,
                          "DX": 4000.0, "DY": 4000.0, "GRID_ID": 2, "PARENT_ID": 1,
                          "I_PARENT_START": 50, "J_PARENT_START": 40, "PARENT_GRID_RATIO": 3})
        d02 = {"attrs": d02_attrs}
        registry = cr.load_registry()
        groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(
            [d01, d02], wrfbdy_times=None, hours=6.0, registry=registry)
        gdict = {g: dict(items) for g, items in groups}
        all_items = {}
        for _g, items in groups:
            all_items.update(dict(items))
        physics_keys = [key for key, _attr in cr._PHYSICS_ATTRS]
        checked_single = checked_per_domain = 0
        for key in physics_keys:
            if key not in all_items:
                continue   # not every physics option is present in every fixture (none skipped
                          # here, but stay honest about the general case)
            expected_len = 2 if cr.registry_per_domain(key, registry) else 1
            self.assertEqual(len(all_items[key]), expected_len,
                             "%s: expected %d value(s) per registry_per_domain=%s, got %r"
                             % (key, expected_len,
                                cr.registry_per_domain(key, registry), all_items[key]))
            if expected_len == 1:
                checked_single += 1
            else:
                checked_per_domain += 1
        # sanity: the fixture actually exercises BOTH dimensions (damp_opt/hypsometric_opt/
        # use_theta_m single; mp_physics/gwd_opt/... per-domain), not just one.
        self.assertGreater(checked_single, 0)
        self.assertGreater(checked_per_domain, 0)
        # exact spot-checks named in the cluster finding -- damp_opt/gwd_opt are &dynamics per
        # the catalog (T15(a)), not &physics.
        self.assertEqual(gdict["dynamics"]["damp_opt"], [3])
        self.assertEqual(gdict["physics"]["mp_physics"], [8, 8])
        self.assertEqual(gdict["dynamics"]["gwd_opt"], [1, 1])

    def test_frames_per_outfile_is_per_domain_per_registry(self):
        # T14: frames_per_outfile is per_domain=True in the catalog -- was hardcoded to a
        # single value before.
        d01 = self._facts()[0]
        d02 = {"attrs": dict(d01["attrs"], **{"WEST-EAST_GRID_DIMENSION": 151,
                                              "SOUTH-NORTH_GRID_DIMENSION": 121, "GRID_ID": 2,
                                              "PARENT_ID": 1})}
        registry = cr.load_registry()
        groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(
            [d01, d02], wrfbdy_times=None, hours=6.0, registry=registry)
        gdict = {g: dict(items) for g, items in groups}
        self.assertTrue(cr.registry_per_domain("frames_per_outfile", registry))
        self.assertEqual(gdict["time_control"]["frames_per_outfile"], [1, 1])

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

    def test_every_checked_key_is_in_its_catalog_group(self):
        # T15(a) (2026-09-24, job 986): each key's group is looked up from the SAME catalog K1
        # itself consults (registry_group_for_key) -- not hardcoded here either.
        groups, parsed, provenance, notes, registry = self._render_and_parse()
        for key in ("diff_opt", "km_opt", "damp_opt", "use_theta_m", "hypsometric_opt",
                   "gwd_opt", "time_step", "eta_levels", "p_top_requested"):
            expected_group = cr.registry_group_for_key(key, registry)
            self.assertIsNotNone(expected_group, "%s has no catalog group at all" % key)
            self.assertIn(key, parsed.get(expected_group, {}),
                          "%s: expected in &%s (per the catalog), parsed groups were %s"
                          % (key, expected_group, sorted(parsed.keys())))

    def test_all_eight_groups_present_even_when_empty(self):
        # T15(b) (2026-09-24, job 986): wrf.exe FATAL'd 'ERROR reading namelist namelist_quilt'
        # because K1's rendered namelist had NO &namelist_quilt group at all (WRF's reader
        # errors on a missing GROUP, not just a missing key) -- every group WRF reads must be
        # present, even empty ones this fixture doesn't populate (&fdda, &grib2).
        groups, parsed, provenance, notes, registry = self._render_and_parse()
        for group in ("time_control", "domains", "physics", "fdda", "dynamics", "bdy_control",
                     "grib2", "namelist_quilt"):
            self.assertIn(group, parsed, "missing &%s entirely" % group)
        self.assertEqual(parsed.get("fdda", "MISSING"), {})
        self.assertEqual(parsed.get("grib2", "MISSING"), {})
        self.assertTrue(any("does not record which groups" in n for n in notes))

    def test_bdy_control_exact_values_rule_default(self):
        # T15(c): no wrfbdy fixture given here -> spec_bdy_width rule_default (5); specified is
        # .true. for d01 only, nested is .true. for every OTHER domain.
        groups, parsed, provenance, notes, registry = self._render_and_parse()
        bdy = parsed["bdy_control"]
        self.assertEqual(bdy["spec_bdy_width"], [5])
        self.assertEqual(bdy["spec_zone"], [1])
        self.assertEqual(bdy["relax_zone"], [4])
        self.assertEqual(bdy["specified"], [True, False])
        self.assertEqual(bdy["nested"], [False, True])
        self.assertEqual(provenance["spec_bdy_width"], "rule_default")
        self.assertEqual(provenance["specified"], "rule_default")

    def test_spec_bdy_width_from_wrfbdy_bdy_width_when_present(self):
        # T15(c): the wrfbdy file's own bdy_width dimension, when known, wins over the
        # Registry default -- provenance 'wrfbdy'.
        groups, parsed, provenance, notes, registry = self._render_and_parse(wrfbdy_bdy_width=4)
        bdy = parsed["bdy_control"]
        self.assertEqual(bdy["spec_bdy_width"], [4])
        self.assertEqual(provenance["spec_bdy_width"], "wrfbdy")

    def test_namelist_quilt_rule_defaults(self):
        # T15(d)
        groups, parsed, provenance, notes, registry = self._render_and_parse()
        quilt = parsed["namelist_quilt"]
        self.assertEqual(quilt["nio_tasks_per_group"], [0])
        self.assertEqual(quilt["nio_groups"], [1])
        self.assertEqual(provenance["nio_tasks_per_group"], "rule_default")
        self.assertEqual(provenance["nio_groups"], "rule_default")


class TestRenderNamelistInput(unittest.TestCase):
    def test_renders_group_key_value_lines(self):
        groups = [("time_control", [("run_hours", [6])]),
                 ("domains", [("e_we", [425, 151]), ("eta_levels", [1.0, 0.5, 0.0])])]
        text = cr.render_namelist_input(groups)
        self.assertIn("&time_control", text)
        self.assertIn(" run_hours = 6,", text)
        self.assertIn("&domains", text)
        self.assertIn(" e_we = 425, 151,", text)
        self.assertIn(" eta_levels = 1.0, 0.5, 0.0,", text)
        self.assertEqual(text.count("/"), 2)


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

    def test_lat_10_is_mercator(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            10.0, 100.0, (1200.0, 1200.0), 12.0, self.START, 6.0)
        self.assertEqual(wps["map_proj"], "mercator")
        self.assertEqual(wps["truelat1"], 10.0)
        self.assertIsNone(wps["truelat2"])
        self.assertIsNone(wps["stand_lon"])

    def test_lat_40_is_lambert(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            40.0, -98.0, (1200.0, 1200.0), 12.0, self.START, 6.0)
        self.assertEqual(wps["map_proj"], "lambert")
        self.assertEqual(wps["truelat1"], 40.0)
        self.assertEqual(wps["truelat2"], 40.0)
        self.assertEqual(wps["stand_lon"], -98.0)

    def test_lat_75_is_polar(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            75.0, 10.0, (1200.0, 1200.0), 12.0, self.START, 6.0)
        self.assertEqual(wps["map_proj"], "polar")
        self.assertEqual(wps["truelat1"], 75.0)
        self.assertIsNone(wps["truelat2"])

    def test_e_we_for_w1200_dx12_is_101(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            40.0, -98.0, (1200.0, 1200.0), 12.0, self.START, 6.0)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["domains"]["e_we"], [101])
        self.assertEqual(wps["e_we"], [101])

    def test_tropical_physics_suite_and_convection_permitting(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            10.0, 100.0, (400.0, 400.0), 3.0, self.START, 6.0)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["physics"]["physics_suite"], ["'TROPICAL'"])
        self.assertEqual(gdict["physics"]["cu_physics"], [0])   # dx_km=3 < 4

    def test_conus_physics_suite_no_cu_override_above_4km(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            40.0, -98.0, (1200.0, 1200.0), 12.0, self.START, 6.0)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["physics"]["physics_suite"], ["'CONUS'"])
        self.assertNotIn("cu_physics", gdict["physics"])

    def test_radt_equals_round_dx_km(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            40.0, -98.0, (1200.0, 1200.0), 12.0, self.START, 6.0)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["physics"]["radt"], [12])

    def test_time_step_is_floor_6x_dx(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            40.0, -98.0, (1200.0, 1200.0), 12.5, self.START, 6.0)
        gdict = {g: dict(items) for g, items in groups}
        self.assertEqual(gdict["domains"]["time_step"], [75])   # floor(6*12.5)=75

    def test_given_vs_rule_default_provenance(self):
        groups, wps, provenance, notes = cr.new_domain_fields(
            40.0, -98.0, (1200.0, 1200.0), 12.0, self.START, 6.0)
        self.assertEqual(provenance["ref_lat"], "given")
        self.assertEqual(provenance["ref_lon"], "given")
        self.assertEqual(provenance["dx"], "given")
        self.assertEqual(provenance["start_year"], "given")
        self.assertEqual(provenance["run_hours"], "given")
        self.assertEqual(provenance["e_we"], "rule_default")
        self.assertEqual(provenance["map_proj"], "rule_default")
        self.assertEqual(provenance["truelat1"], "rule_default")
        self.assertEqual(provenance["radt"], "rule_default")


class TestCmdNewDomain(unittest.TestCase):
    def test_refuses_partial_four_inputs(self):
        with self.assertRaises(cr.RebuildError):
            cr.cmd_new_domain(center="40,-98", size_km=None, dx_km="12", start="2020-06-01_00:00:00",
                              hours=6.0, out_dir="/tmp/whatever")

    def test_refuses_none_of_the_four(self):
        with self.assertRaises(cr.RebuildError):
            cr.cmd_new_domain(hours=6.0, out_dir="/tmp/whatever")

    def test_writes_namelist_input_and_wps_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = os.path.join(d, "domain")
            cr.cmd_new_domain(center="40,-98", size_km="1200,1200", dx_km="12",
                              start="2020-06-01_00:00:00", hours=6.0, out_dir=out_dir)
            nl_path = os.path.join(out_dir, "namelist.input")
            nlw_path = os.path.join(out_dir, "namelist.wps")
            prov_path = os.path.join(out_dir, "namelist_provenance.json")
            self.assertTrue(os.path.exists(nl_path))
            self.assertTrue(os.path.exists(nlw_path))
            self.assertTrue(os.path.exists(prov_path))
            with open(nl_path) as f:
                nl_text = f.read()
            self.assertEqual(cs.get_first(nl_text, "e_we"), "101")
            with open(nlw_path) as f:
                nlw_text = f.read()
            self.assertIn("map_proj          = 'lambert',", nlw_text)


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

    def test_a_everything_present_is_as_is(self):
        with tempfile.TemporaryDirectory() as d:
            self._touch(d, "namelist.input", "namelist.wps", "wrfinput_d01", "wrfbdy_d01")
            inv = cr.inventory_case_dir(d)
            decision = cr.choose_completion_path(inv, have_center_size_dx_start=False)
            self.assertEqual(decision["path"], "as_is")
            self.assertFalse(decision["needs_k1"])
            self.assertFalse(decision["needs_k3"])

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

    def test_e_nothing_but_center_size_dx_start_is_k2_then_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            inv = cr.inventory_case_dir(d)
            decision = cr.choose_completion_path(inv, have_center_size_dx_start=True)
            self.assertEqual(decision["path"], "k2_then_rebuild")

    def test_f_nothing_at_all_refuses_naming_what_is_needed(self):
        with tempfile.TemporaryDirectory() as d:
            inv = cr.inventory_case_dir(d)
            decision = cr.choose_completion_path(inv, have_center_size_dx_start=False)
            self.assertEqual(decision["path"], "refuse")
            self.assertIn("namelist.input", decision["reason"])
            self.assertIn("wrfinput_d0N", decision["reason"])
            self.assertIn("--center/--size-km/--dx-km/--start", decision["reason"])

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

    def test_real_step_links_nocolons_met_em_names(self):
        # V6 lafe-real/-fict (scc jobs 1059/1060): nocolons=.true. -> real.exe wants '_' names
        text = cr.render_rebuild_sbatch(self._plan(), self._chosen())
        self.assertIn('ln -sf "$f" "${f//:/_}"', text)
        self.assertLess(text.index('/wps/met_em.d0*.*.nc .'), text.index('"${f//:/_}"'))

    def test_no_forbidden_content(self):
        text = cr.render_rebuild_sbatch(self._plan(), self._chosen())
        self.assertNotIn("LD_PRELOAD", text)
        self.assertNotIn("--wrap", text)
        self.assertIn("--exclude=head", text)
        self.assertIn("--signal=B:USR1@90", text)

    def test_real_exe_timeout_600_and_serial_1800_fallback(self):
        text = cr.render_rebuild_sbatch(self._plan(), self._chosen())
        self.assertIn("timeout 600 mpirun -np 100 --map-by ppr:50:node:PE=1 --bind-to core "
                     "./real.exe", text)
        self.assertIn("timeout 1800 mpirun -np 1 --map-by ppr:1:node:PE=1 --bind-to core "
                     "./real.exe", text)
        self.assertIn("--serial", text)

    def test_alloc_nodes_defined_before_first_run_hygiene_call_and_has_usr1_trap(self):
        # T5 (2026-09-24, jobs 954/955): site_hygiene's `mpirun -np "$ALLOC_NODES"` failed
        # 'ALLOC_NODES: unbound variable' because the rebuild renderer's own hand-copy of the
        # preamble never defined it. Must now come from the shared
        # coldstart_sample.render_signal_trap_and_hygiene_preamble, same as run.sbatch.
        text = cr.render_rebuild_sbatch(self._plan(), self._chosen())
        lines = text.splitlines()
        alloc_idx = next(i for i, ln in enumerate(lines) if ln.startswith("ALLOC_NODES="))
        first_hygiene_idx = next(i for i, ln in enumerate(lines)
                                 if "run_hygiene" in ln and i != alloc_idx)
        self.assertLess(alloc_idx, first_hygiene_idx)
        self.assertIn("trap 'on_signal USR1' USR1", text)
        self.assertIn("trap 'on_signal TERM' TERM", text)
        self.assertIn("trap 'on_signal INT' INT", text)
        self.assertIn("trap on_exit EXIT", text)

    def test_uses_rebuild_dir_not_bare_out_assigned_after_site_source(self):
        # T6 (2026-09-24, jobs 957/958): campaign/sites/scc.sh assigns its own global
        # OUT=$HOME/wtune-runs/campaign, clobbering a bare OUT= set before sourcing it. Renamed
        # to REBUILD_DIR (a name no campaign/sites/*.sh assigns) and moved after the source line.
        text = cr.render_rebuild_sbatch(self._plan(), self._chosen())
        lines = text.splitlines()
        self.assertFalse(any(ln.startswith("OUT=") for ln in lines))
        source_idx = next(i for i, ln in enumerate(lines) if ln.startswith('source "$CAMPAIGN'))
        rebuild_dir_idx = next(i for i, ln in enumerate(lines) if ln.startswith("REBUILD_DIR="))
        self.assertGreater(rebuild_dir_idx, source_idx)
        self.assertIn('cd "$REBUILD_DIR/wps"', text)
        self.assertIn('cd "$REBUILD_DIR/real"', text)
        self.assertNotIn('"$OUT/', text)
        self.assertNotIn('"$OUT"', text)

    def test_grib_manifest_read_with_widened_fallback_glob(self):
        # T9 (2026-09-24): link exactly what `rebuild fetch` wrote (grib_manifest.txt) instead
        # of assuming a *.grib2 extension (FNL is .grib, ncei_g3/g4 are .grb).
        text = cr.render_rebuild_sbatch(self._plan(), self._chosen())
        self.assertIn("grib_manifest.txt", text)
        self.assertIn('"$REBUILD_DIR"/grib/*.grib2', text)
        self.assertIn('"$REBUILD_DIR"/grib/*.grib ', text)
        self.assertIn('"$REBUILD_DIR"/grib/*.grb', text)

    def test_ntasks_per_node_is_ceil_not_floor(self):
        # T7 (2026-09-24): 90 ranks / 4 nodes must request ceil=23/node, not floor=22 (floor
        # under-requests Slurm slots -> real.exe's own mpirun then fails rc=213 'not enough
        # slots').
        plan = self._plan()
        plan["nodes"] = 4
        plan["config"]["ranks"] = 90
        text = cr.render_rebuild_sbatch(plan, self._chosen())
        self.assertIn("--ntasks-per-node=23", text)
        self.assertIn("ppr:23:node:PE=1 --bind-to core ./real.exe", text)

    def test_geo_em_checks_for_max_dom_2(self):
        text = cr.render_rebuild_sbatch(self._plan(max_dom=2), self._chosen())
        self.assertIn("geo_em.d01.nc", text)
        self.assertIn("geo_em.d02.nc", text)
        self.assertIn("Successful completion", text)

    def test_no_wrf_smoke_run_when_verify_steps_zero(self):
        text = cr.render_rebuild_sbatch(self._plan(verify_steps=0), self._chosen())
        self.assertNotIn("wrf.exe smoke run", text)

    def test_wrf_smoke_run_present_when_verify_steps_given(self):
        text = cr.render_rebuild_sbatch(self._plan(verify_steps=20), self._chosen())
        self.assertIn("wrf.exe smoke run: 20 steps", text)
        self.assertIn("export OMP_NUM_THREADS=2", text)
        self.assertIn("unset OMP_PLACES OMP_PROC_BIND", text)

    def test_ml_model_default_off_no_ml_code_at_all(self):
        plan = self._plan()
        plan["ml_model"] = None
        text = cr.render_rebuild_sbatch(plan, self._chosen())
        self.assertNotIn("ML_MODEL=", text)
        self.assertNotIn("TOOL_SCM=", text)
        self.assertNotIn("--targets-from-report", text)
        self.assertNotIn("ml_status.txt", text)

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
    def test_rmse_exact(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 3.0], [3.0, 5.0]])
        self.assertAlmostEqual(cr.rmse(a, b), (0.5) ** 0.5, places=9)

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

    def test_pattern_correlation_matches_numpy_corrcoef(self):
        a = np.array([1.0, 2.0, 4.0, 3.0])
        b = np.array([2.0, 1.0, 5.0, 4.0])
        expected = float(np.corrcoef(a, b)[0, 1])
        self.assertAlmostEqual(cr.pattern_correlation(a, b), expected, places=9)

    def test_pattern_correlation_perfect_and_anti(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(cr.pattern_correlation(a, a), 1.0, places=9)
        self.assertAlmostEqual(cr.pattern_correlation(a, -a), -1.0, places=9)

    def test_pattern_correlation_constant_field_is_nan(self):
        a = np.array([1.0, 2.0, 3.0])
        c = np.array([5.0, 5.0, 5.0])
        self.assertTrue(np.isnan(cr.pattern_correlation(a, c)))

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

    def test_interp_pressure_level_column_exact_bracket(self):
        field_k = np.array([300.0, 280.0, 260.0])
        p_k = np.array([100000.0, 85000.0, 70000.0])
        self.assertEqual(cr.interp_pressure_level_column(field_k, p_k, 85000.0), 280.0)

    def test_interp_pressure_level_column_below_ground_is_nan(self):
        field_k = np.array([300.0, 280.0, 260.0])
        p_k = np.array([100000.0, 85000.0, 70000.0])
        self.assertTrue(math.isnan(cr.interp_pressure_level_column(field_k, p_k, 110000.0)))

    def test_interp_pressure_level_column_midpoint_linear_in_lnp(self):
        import math as _m
        field_k = np.array([0.0, 100.0])
        p_k = np.array([100000.0, 50000.0])
        target = _m.sqrt(100000.0 * 50000.0)   # log-midpoint -> exactly halfway in ln(p)
        val = cr.interp_pressure_level_column(field_k, p_k, target)
        self.assertAlmostEqual(val, 50.0, places=6)


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

    def test_identical_grids_pass(self):
        a = self._grid()
        b = self._grid()
        ok, reasons = cr.grid_identity_check(a, b)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_different_dx_fails(self):
        a = self._grid(dx=12000.0)
        b = self._grid(dx=4000.0)
        ok, reasons = cr.grid_identity_check(a, b)
        self.assertFalse(ok)
        self.assertTrue(any("DX" in r for r in reasons))

    def test_center_beyond_tolerance_fails(self):
        a = self._grid(cen_lat=40.0)
        b = self._grid(cen_lat=40.5)
        ok, reasons = cr.grid_identity_check(a, b)
        self.assertFalse(ok)

    def test_within_tolerance_passes(self):
        a = self._grid(cen_lat=40.0)
        b = self._grid(cen_lat=40.005)
        ok, reasons = cr.grid_identity_check(a, b)
        self.assertTrue(ok)

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

    def test_lambert_grids_differing_in_truelat2_still_fail(self):
        # MAP_PROJ=1 (Lambert) DOES use TRUELAT2 -- this must still be caught.
        a = self._grid()
        b = self._grid()
        b["attrs"]["TRUELAT2"] = 1e20
        ok, reasons = cr.grid_identity_check(a, b)
        self.assertFalse(ok)
        self.assertTrue(any("TRUELAT2" in r for r in reasons))


class TestCompareVerdict(unittest.TestCase):
    def test_consistent(self):
        verdict, static_ok, state_ok = cr.compare_verdict(10.0, 95.0, 0.95, 0.9, 0.8)
        self.assertEqual(verdict, "consistent")
        self.assertTrue(static_ok and state_ok)

    def test_different_static_data(self):
        verdict, static_ok, state_ok = cr.compare_verdict(100.0, 50.0, 0.95, 0.9, 0.8)
        self.assertEqual(verdict, "different-static-data")
        self.assertFalse(static_ok)
        self.assertTrue(state_ok)

    def test_state_mismatch_dominates_even_with_good_static(self):
        verdict, static_ok, state_ok = cr.compare_verdict(10.0, 95.0, 0.5, 0.9, 0.8)
        self.assertEqual(verdict, "state-mismatch")
        self.assertTrue(static_ok)
        self.assertFalse(state_ok)


# ---------------------------------------------------------------------------
# R6-C: candidate_dates (DOY window + wrap), block_average, scoring/ranking
# ---------------------------------------------------------------------------

class TestCandidateDatesDoyWrap(unittest.TestCase):
    def test_jan10_window45_wraps_nov26_to_feb24(self):
        stated = datetime.datetime(2020, 1, 10, 0, 0, 0)
        cands = cr.candidate_dates(stated, (2021, 2021), 45)
        self.assertIn(datetime.datetime(2020, 11, 26, 0, 0, 0), cands)
        self.assertIn(datetime.datetime(2021, 2, 24, 0, 0, 0), cands)
        self.assertEqual(min(cands), datetime.datetime(2020, 11, 26, 0, 0, 0))
        self.assertEqual(max(cands), datetime.datetime(2021, 2, 24, 0, 0, 0))

    def test_2021_2025_exact_length(self):
        stated = datetime.datetime(2020, 1, 10, 0, 0, 0)
        cands = cr.candidate_dates(stated, (2021, 2025), 45)
        self.assertEqual(len(cands), 5 * (2 * 45 + 1))   # 5 years x 91-day window each

    def test_same_hour_as_stated(self):
        stated = datetime.datetime(2020, 1, 10, 6, 0, 0)
        cands = cr.candidate_dates(stated, (2021, 2021), 5)
        self.assertTrue(all(c.hour == 6 for c in cands))


class TestClimatologicalAnalog(unittest.TestCase):
    """T14 (2026-09-26): climatological_analog() replaces max(candidate_dates(...)) for the
    no-target case -- that was NOT "same DOY, most recent year" (it was the +-doy_window
    window's LAST day), e.g. stated 2026-09-26 with doy_window=45 -> 2025-11-10, not
    2025-09-26."""

    def test_same_doy_same_hour_most_recent_year(self):
        stated = datetime.datetime(2026, 9, 26, 0, 0, 0)
        got = cr.climatological_analog(stated, (2021, 2025), hour="same")
        self.assertEqual(got, datetime.datetime(2025, 9, 26, 0, 0, 0))

    def test_leap_doy_366_falls_back_to_dec31_in_non_leap_y1(self):
        stated = datetime.datetime(2028, 12, 31, 12, 0, 0)   # DOY 366 (2028 is a leap year)
        got = cr.climatological_analog(stated, (2021, 2025), hour="same")   # 2025 is not leap
        self.assertEqual(got, datetime.datetime(2025, 12, 31, 12, 0, 0))

    def test_jan5_maps_to_jan5_of_y1(self):
        stated = datetime.datetime(2031, 1, 5, 18, 0, 0)
        got = cr.climatological_analog(stated, (2021, 2025), hour="same")
        self.assertEqual(got, datetime.datetime(2025, 1, 5, 18, 0, 0))

    def test_leap_doy_366_in_leap_y1_maps_directly(self):
        stated = datetime.datetime(2028, 12, 31, 0, 0, 0)   # DOY 366 (2028 is a leap year)
        got = cr.climatological_analog(stated, (2021, 2024), hour="same")   # 2024 IS leap too
        self.assertEqual(got, datetime.datetime(2024, 12, 31, 0, 0, 0))

    def test_explicit_hour_override(self):
        stated = datetime.datetime(2026, 9, 26, 3, 0, 0)
        got = cr.climatological_analog(stated, (2021, 2025), hour=18)
        self.assertEqual(got, datetime.datetime(2025, 9, 26, 18, 0, 0))

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


class TestBlockAverage(unittest.TestCase):
    def test_exact_averages_evenly_divisible(self):
        field = np.arange(16.0).reshape(4, 4)
        out = cr.block_average(field, block=2, min_blocks=1)
        self.assertEqual(out.shape, (2, 2))
        # top-left 2x2 block = [[0,1],[4,5]] -> mean = 2.5
        self.assertAlmostEqual(out[0, 0], 2.5, places=9)

    def test_min_blocks_floor_shrinks_block_size(self):
        field = np.arange(100.0).reshape(10, 10)
        out = cr.block_average(field, block=8, min_blocks=10)
        # 10//8=1 < 10 -> shrinks to block=10//10=1 -> no averaging, same shape
        self.assertEqual(out.shape, (10, 10))
        np.testing.assert_allclose(out, field)

    def test_nan_aware_block_ignores_nan_and_averages_the_rest(self):
        field = np.full((4, 4), np.nan)
        field[0, 0] = 10.0
        field[1, 1] = 20.0   # same 2x2 block as [0,0]
        out = cr.block_average(field, block=2, min_blocks=1)
        self.assertEqual(out.shape, (2, 2))
        self.assertAlmostEqual(out[0, 0], 15.0, places=9)   # nanmean(10, 20) = 15, not poisoned

    def test_all_nan_block_stays_nan(self):
        field = np.full((4, 4), np.nan)
        field[0, 0] = 10.0   # only the top-left block has data
        out = cr.block_average(field, block=2, min_blocks=1)
        self.assertTrue(math.isnan(out[1, 1]))   # bottom-right block is entirely NaN


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

    def test_zero_rotation_is_identity(self):
        u = np.array([5.0])
        v = np.array([-2.0])
        cosalpha = np.array([1.0])
        sinalpha = np.array([0.0])
        u_e, v_e = cr.rotate_grid_to_earth(u, v, cosalpha, sinalpha)
        self.assertAlmostEqual(float(u_e[0]), 5.0, places=9)
        self.assertAlmostEqual(float(v_e[0]), -2.0, places=9)

    def test_broadcast_2d_alpha_against_3d_wind(self):
        u = np.ones((2, 1, 1)) * 3.0
        v = np.ones((2, 1, 1)) * 4.0
        cosalpha = np.zeros((1, 1))
        sinalpha = np.ones((1, 1))
        u_e, v_e = cr.rotate_grid_to_earth(u, v, cosalpha, sinalpha)
        self.assertEqual(u_e.shape, (2, 1, 1))
        np.testing.assert_allclose(u_e, -4.0)
        np.testing.assert_allclose(v_e, 3.0)


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


@unittest.skipUnless(HAS_NETCDF4, "netCDF4 not installed in this dev environment")
class TestVerticalGridFromWrfinput(unittest.TestCase):
    def test_znw_and_p_top_read_exactly(self):
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

            vg = cr.vertical_grid_from_wrfinput(path)
            self.assertEqual(vg["eta_levels"], [1.0, 0.6, 0.3, 0.0])
            self.assertEqual(vg["p_top_requested"], 5000.0)

    def test_missing_znw_raises(self):
        import netCDF4
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wrfinput_d01")
            ds = netCDF4.Dataset(path, "w")
            ds.createDimension("Time", None)
            p_top = ds.createVariable("P_TOP", "f4", ("Time",))
            p_top[0] = 5000.0
            ds.close()
            # T12 (2026-09-24): vertical_grid_from_wrfinput moved to coldstart_sample.py (see
            # cr.vertical_grid_from_wrfinput's alias) -- it now raises cs.ColdstartError, not
            # cr.RebuildError.
            with self.assertRaises(cs.ColdstartError):
                cr.vertical_grid_from_wrfinput(path)


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

    def test_eta_levels_lands_in_domains_group(self):
        vertical_grid = {"eta_levels": [1.0, 0.5, 0.0], "p_top_pa": 5000.0}
        txt, changes = cr.complete_namelist(self._nl_with_dzstretch(), BASE_FACTS,
                                            vertical_grid=vertical_grid)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "namelist.input")
            with open(path, "w") as f:
                f.write(txt)
            parsed = predict.parse_namelist(path)
        self.assertIn("eta_levels", parsed.get("domains", {}))

    def test_no_vertical_grid_falls_back_to_fill_if_missing_and_keeps_dzstretch(self):
        txt, changes = cr.complete_namelist(self._nl_with_dzstretch(), BASE_FACTS,
                                            vertical_grid=None)
        # unchanged: e_vert/p_top_requested already present -> fill_if_missing is a no-op
        self.assertEqual(cs.get_first(txt, "e_vert"), "45")
        self.assertEqual(cs.get_first(txt, "p_top_requested"), "5000")
        self.assertEqual(cs.count_key(txt, "dzstretch_s"), 1)
        self.assertFalse(any(c["key"] == "eta_levels" for c in changes))


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

    def test_constant_field_scores_inf(self):
        const = {"f": np.full((2, 2), 25.0)}
        result = cr.score_candidate(const, self.target)
        self.assertEqual(result["total"], float("inf"))

    def test_ranking_orders_best_to_worst(self):
        T = self.target["f"]
        candidates = [
            (datetime.datetime(2021, 1, 1), {"f": np.array([[40.0, 30.0], [20.0, 10.0]])}),  # anti
            (datetime.datetime(2022, 1, 1), {"f": T.copy()}),                                # perfect
            (datetime.datetime(2023, 1, 1), {"f": 2 * T}),                                   # scaled
            (datetime.datetime(2024, 1, 1), {"f": np.array([[11.0, 19.0], [31.0, 39.0]])}),  # noisy
            (datetime.datetime(2025, 1, 1), {"f": np.full((2, 2), 25.0)}),                   # const
        ]
        ranked, kept = cr.cmd_analog_offline_rank(self.target, candidates, top_n_trajectory=3)
        ranked_dates = [d for d, _s in ranked]
        self.assertEqual(ranked_dates, [
            datetime.datetime(2022, 1, 1),   # perfect: score 0
            datetime.datetime(2024, 1, 1),   # noisy: small score
            datetime.datetime(2023, 1, 1),   # scaled: 0.5
            datetime.datetime(2021, 1, 1),   # anti: 1.5
            datetime.datetime(2025, 1, 1),   # const: inf
        ])
        self.assertEqual(len(kept), 3)
        self.assertEqual([d for d, _s in kept], ranked_dates[:3])

    def test_two_stage_select_keeps_exactly_top_n(self):
        ranked = [(datetime.datetime(2020, 1, i), {"total": float(i)}) for i in range(1, 21)]
        kept = cr.two_stage_select(ranked, top_n=10)
        self.assertEqual(len(kept), 10)
        self.assertEqual(kept, ranked[:10])


class TestQualityFlag(unittest.TestCase):
    def test_good(self):
        self.assertEqual(cr.quality_flag([0.9, 0.85, 0.95]), "good")

    def test_fair(self):
        self.assertEqual(cr.quality_flag([0.9, 0.65, 0.95]), "fair")

    def test_poor(self):
        self.assertEqual(cr.quality_flag([0.9, 0.5, 0.95]), "poor")

    def test_no_data_is_poor(self):
        self.assertEqual(cr.quality_flag([]), "poor")

    def test_nan_is_poor(self):
        self.assertEqual(cr.quality_flag([0.9, float("nan")]), "poor")


class TestBuildAnalogOutput(unittest.TestCase):
    def test_no_target_climatological_analog_via_cmd_analog(self):
        with tempfile.TemporaryDirectory() as d:
            case_dir = os.path.join(d, "case")
            os.makedirs(case_dir)
            with open(os.path.join(case_dir, "namelist.input"), "w") as f:
                f.write(" &time_control\n start_year=2020,\n start_month=1,\n start_day=10,\n"
                        " start_hour=0,\n run_hours=6,\n /\n &domains\n max_dom=1,\n /\n")
            out_dir = os.path.join(d, "out")
            result = cr.cmd_analog(case_dir, out_dir, years="2021-2022", doy_window=5)
            self.assertEqual(result["quality"], "no-target")
            self.assertIsNotNone(result["chosen"])
            with open(os.path.join(out_dir, "analog.json")) as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["quality"], "no-target")


# ---------------------------------------------------------------------------
# Numpy-free module import (case_rebuild.py's rebuild fix-span-* subcommands run inside a batch
# job on scc's /usr/bin/python3, which has no numpy).
# ---------------------------------------------------------------------------

class TestNoNumpyAtModuleImport(unittest.TestCase):
    def test_import_succeeds_with_numpy_blocked(self):
        saved = {k: sys.modules.get(k) for k in ("numpy", "wtune.tools.case_rebuild")}
        for k in ("numpy", "wtune.tools.case_rebuild"):
            sys.modules.pop(k, None)
            vars(sys.modules["wtune.tools"]).pop(k.rpartition(".")[2], None)
        sys.modules["numpy"] = None
        try:
            from wtune.tools import case_rebuild as cr2   # noqa: F401
            self.assertIsNotNone(cr2)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


if __name__ == "__main__":
    unittest.main()


class TestWind850Verdict(unittest.TestCase):
    # V3 (conus12 relabelled 2019 -> 2023): Z500 0.816 / T850 0.842 / PW 0.687 was a thin miss;
    # U850 0.130 / V850 0.150 is what separates it from real dates (0.955-0.994)
    def test_wind_alone_flags_state_mismatch(self):
        verdict, _, state_ok = cr.compare_verdict(10.0, 95.0, 0.95, 0.9, 0.8,
                                                  corr_u850=0.13, corr_v850=0.15)
        self.assertEqual(verdict, "state-mismatch")
        self.assertFalse(state_ok)

    def test_real_date_winds_stay_consistent(self):
        verdict, _, _ = cr.compare_verdict(10.0, 95.0, 0.996, 0.994, 0.986,
                                           corr_u850=0.955, corr_v850=0.97)
        self.assertEqual(verdict, "consistent")

    def test_nan_wind_fails_and_absent_wind_is_ignored(self):
        self.assertEqual(cr.compare_verdict(10.0, 95.0, 0.95, 0.9, 0.8,
                                            corr_u850=float("nan"), corr_v850=0.99)[0],
                         "state-mismatch")
        self.assertEqual(cr.compare_verdict(10.0, 95.0, 0.95, 0.9, 0.8)[0], "consistent")
