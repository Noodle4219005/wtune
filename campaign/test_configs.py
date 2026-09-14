#!/usr/bin/env python3
"""Unit tests for campaign/configs.py. Stdlib unittest, Python 3.6 compatible."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import configs  # noqa: E402


class TestTemplates(unittest.TestCase):
    def test_ranks_times_omp_is_448(self):
        for t in configs.TEMPLATES:
            self.assertEqual(t["ranks"] * t["omp"], configs.TOTAL_CORES, t["id"])

    def test_numtiles_matches_template_table(self):
        expected = {
            "C01": 1, "C02": 8, "C03": 8, "C04": 8,
            "C05": 13, "C06": 12, "C07": 14, "C08": 2, "C09": 12, "C10": 12,
            "C11": 12, "C12": 12,
            "C13": 16, "C14": 12, "C15": 16, "C16": 13,
        }
        got = {t["id"]: t["numtiles"] for t in configs.TEMPLATES}
        self.assertEqual(got, expected)


class TestLegalShapes(unittest.TestCase):
    def test_patch_dims_ceil(self):
        self.assertEqual(configs.patch_dims(901, 16), 57)   # 901/16 = 56.31 -> 57
        self.assertEqual(configs.patch_dims(900, 30), 30)   # exact division

    def test_legal_shapes_filters_below_min_patch(self):
        # 425 x 300 at 448 ranks: (56, 8) gives patch_x=ceil(425/56)=8 < 10 -> illegal
        shapes = configs.legal_shapes(425, 300, 448)
        pairs = [(nx, ny) for nx, ny, px, py in shapes]
        self.assertNotIn((56, 8), pairs)
        self.assertNotIn((8, 56), pairs)
        # (16, 28) gives patch_x=ceil(425/16)=27>=10, patch_y=ceil(300/28)=11>=10 -> legal
        self.assertIn((16, 28), pairs)

    def test_legality_uses_wrf_integer_division(self):
        # share/module_check_a_mundo.F:3014-3029: e_we/nproc_x < 10 (integer division) aborts
        self.assertTrue(configs.wrf_patch_legal(100, 10))
        self.assertFalse(configs.wrf_patch_legal(99, 10))    # ceil would give 10, WRF gives 9
        # conus12 e_sn=300, nproc_y=32: ceil 10 but floor 9 -> WRF aborts, must be illegal
        pairs = [(nx, ny) for nx, ny, px, py in configs.legal_shapes(425, 300, 448)]
        self.assertNotIn((14, 32), pairs)

    def test_conus12_448_ranks_has_legal_shapes(self):
        # documents that 425x300 at 448 ranks is NOT fully illegal (some templates
        # still resolve); this exercises the case the task flagged as "may lack
        # alt shapes" without assuming it lacks ALL shapes.
        shapes, method = configs.ranked_shapes(425, 300, 50, 448, 1)
        self.assertGreater(len(shapes), 0)
        self.assertIn(method, ("wtune_decomp", "fallback"))

    def test_multidomain_shape_legal_on_d01_illegal_on_smaller_nest_is_rejected(self):
        # d01=2000x2000 (large, patch never binds), d02=150x900 (e_we tiny) -- a
        # shape with a big nx (many x-slices) is legal on d01 but illegal on d02
        # because ceil(150/nx) < 10 once nx > 15.
        e_we = [2000, 150]
        e_sn = [2000, 900]
        shapes_d01_only = configs.legal_shapes(2000, 2000, 448)
        pairs_d01_only = [(nx, ny) for nx, ny, px, py in shapes_d01_only]
        # (28, 16): patch_x=ceil(2000/28)=72>=10 legal on d01 alone
        self.assertIn((28, 16), pairs_d01_only)
        shapes_multi = configs.legal_shapes(e_we, e_sn, 448)
        pairs_multi = [(nx, ny) for nx, ny, px, py in shapes_multi]
        # same (28, 16): patch_x on d02 = ceil(150/28) = 6 < 10 -> must be rejected
        self.assertNotIn((28, 16), pairs_multi)
        # sanity: multi-domain legality is a strict subset of d01-only legality
        self.assertTrue(set(pairs_multi) <= set(pairs_d01_only))
        self.assertGreater(len(pairs_multi), 0)

    def test_resolve_config_applies_multidomain_filter_via_case(self):
        case = dict(e_we=2000, e_sn=2000, e_vert=50, max_dom=2,
                    e_we_list=[2000, 150], e_sn_list=[2000, 900])
        template = dict(id="TX", ranks=448, omp=1, shape="best", numtiles=8,
                         dist="cyclic", blocktime="0")
        cfg, err = configs.resolve_config(case, template)
        self.assertIsNone(err)
        # the resolved shape must itself be legal on the small d02, not just d01
        self.assertGreaterEqual(configs.patch_dims(150, cfg["nx"]), 10)
        self.assertGreaterEqual(configs.patch_dims(900, cfg["ny"]), 10)


class TestDedupe(unittest.TestCase):
    def test_dedupe_collapses_identical_configs(self):
        resolved = [
            dict(id="A", ranks=448, omp=1, nx=16, ny=28, numtiles=8, dist="cyclic", blocktime="0"),
            dict(id="B", ranks=448, omp=1, nx=16, ny=28, numtiles=8, dist="cyclic", blocktime="0"),
            dict(id="C", ranks=224, omp=2, nx=14, ny=16, numtiles=12, dist="cyclic", blocktime="0"),
        ]
        kept, aliases = configs.dedupe(resolved)
        self.assertEqual([c["id"] for c in kept], ["A", "C"])
        self.assertEqual(aliases, {"A": ["B"]})

    def test_dedupe_no_duplicates_is_noop(self):
        resolved = [
            dict(id="A", ranks=448, omp=1, nx=16, ny=28, numtiles=1, dist="cyclic", blocktime="0"),
            dict(id="B", ranks=448, omp=1, nx=16, ny=28, numtiles=8, dist="cyclic", blocktime="0"),
        ]
        kept, aliases = configs.dedupe(resolved)
        self.assertEqual(len(kept), 2)
        self.assertEqual(aliases, {})


class TestBuildOrder(unittest.TestCase):
    def test_reps_two_is_palindrome(self):
        order = configs.build_order(["A", "B", "C"], 2)
        self.assertEqual(order, ["A", "B", "C", "C", "B", "A"])

    def test_reps_three_alternates(self):
        order = configs.build_order(["A", "B"], 3)
        self.assertEqual(order, ["A", "B", "B", "A", "A", "B"])

    def test_reps_zero_is_empty(self):
        self.assertEqual(configs.build_order(["A", "B"], 0), [])

    def test_reps_one_is_forward_only(self):
        self.assertEqual(configs.build_order(["A", "B"], 1), ["A", "B"])


class TestNamelistEditing(unittest.TestCase):
    """2026-09-14 all-ADIOS2 switch: edit_namelist's `history_interval` positional
    parameter is gone (run_minutes plus history_mode now fully determine it, since
    every campaign call already kept them equal) and adios2=True is the default, so
    most calls below now need history_mode="end_d01" (to exercise the old
    always-write-run-length behaviour these tests were written to check) and a
    `ranks` kwarg (adios2_numaggregators). io_form_history's asserted value moves
    from the old unconditional 2 to ADIOS2's 14. See TestAdios2 and TestHistoryMode
    for the new behaviour's own dedicated coverage, and
    test_io_form_input_and_boundary_overrides_apply_with_no_adios2 below for why
    the old io_form-override test now needs adios2=False to keep meaning what it
    originally meant."""

    SYNTHETIC_WITH_EXISTING = (
        "&time_control\n"
        " run_days=0, run_hours=1, run_minutes=0, run_seconds=0,\n"
        " history_interval=999, frames_per_outfile=1,\n"
        " io_form_history=14,\n"
        "/\n"
        "&domains\n"
        " nproc_x = 16, nproc_y = 28,\n"
        " numtiles = 8,\n"
        " time_step=15, e_we=901, e_sn=901, e_vert=45,\n"
        "/\n"
    )

    SYNTHETIC_WITHOUT_EXISTING = (
        "&time_control\n"
        " run_days=0, run_hours=1, run_minutes=0, run_seconds=0,\n"
        " history_interval=999, frames_per_outfile=1,\n"
        " io_form_history=14,\n"
        "/\n"
        "&domains\n"
        " time_step=15, e_we=901, e_sn=901, e_vert=45,\n"
        "/\n"
    )

    SYNTHETIC_HISTORY_M = (
        "&time_control\n"
        " run_days = 0,\n"
        " run_hours = 6,\n"
        " run_minutes = 0,\n"
        " history_interval_m = 60\n"
        " io_form_history = 2\n"
        "/\n"
        "&domains\n"
        " nproc_x = 8, nproc_y = 56,\n"
        " time_step = 15,\n"
        "/\n"
    )

    def test_edit_with_existing_decomp_lines(self):
        out = configs.edit_namelist(self.SYNTHETIC_WITH_EXISTING, 14, 16, 12, 8,
                                     history_mode="end_d01", ranks=224)
        self.assertNotIn("nproc_x = 16", out)
        self.assertNotIn("numtiles = 8", out)
        self.assertIn("nproc_x=14, nproc_y=16, numtiles=12,", out)
        self.assertEqual(out.count("nproc_x"), 1)
        self.assertEqual(out.count("numtiles"), 1)
        self.assertIn("run_minutes=8", out)
        self.assertIn("run_hours=0", out)
        self.assertIn("run_days=0", out)
        self.assertIn("history_interval=8,", out)
        self.assertIn("history_begin = 8,", out)
        # io_form_history is ADIOS2's 14 now (this template already started at 14,
        # unlike the old test which started it at 14 and asserted the old
        # unconditional override brought it down to 2 -- see TestAdios2 for
        # dedicated before/after io_form coverage on a template that starts != 14)
        self.assertIn("io_form_history=14,", out)
        # ADIOS2 settings that don't need a pre-existing line always get inserted
        self.assertIn('input_inname="wrfinput_d<domain>.bp", bdy_inname="wrfbdy_d<domain>.bp",', out)
        self.assertIn("&namelist_adios2\n adios2_numaggregators=224,\n/\n", out)
        # time_step untouched (not a campaign-edited knob)
        self.assertIn("time_step=15", out)

    def test_edit_without_existing_decomp_lines(self):
        out = configs.edit_namelist(self.SYNTHETIC_WITHOUT_EXISTING, 14, 16, 12, 8,
                                     history_mode="end_d01", ranks=224)
        self.assertIn("nproc_x=14, nproc_y=16, numtiles=12,", out)
        idx_domains = out.find("&domains")
        idx_decomp = out.find("nproc_x=14")
        self.assertGreater(idx_decomp, idx_domains)

    def test_edit_uses_history_interval_m_when_present(self):
        out = configs.edit_namelist(self.SYNTHETIC_HISTORY_M, 4, 28, 6, 5,
                                     history_mode="end_d01", ranks=112)
        self.assertIn("history_interval_m", out)
        self.assertIn("history_interval_m = 5", out)
        self.assertNotIn("history_interval = 5", out)
        self.assertIn("history_begin_m = 5,", out)
        self.assertIn("nproc_x=4, nproc_y=28, numtiles=6,", out)
        self.assertNotIn("nproc_x = 8", out)

    def test_edit_preserves_physics_and_restart_lines(self):
        text = self.SYNTHETIC_WITH_EXISTING + "&physics\n mp_physics=8,\n/\n"
        out = configs.edit_namelist(text, 14, 16, 12, 8, history_mode="end_d01", ranks=224)
        self.assertIn("mp_physics=8,", out)

    # -- WTune D1 decomposition-space work: numtiles_x/y/tile_sz_x/y/tile_strategy --------------
    def test_tile_geometry_args_default_none_is_byte_identical(self):
        """Every new edit_namelist param defaults to None: a call with none of them given must
        produce EXACTLY the same text as before these params existed."""
        with_extra = configs.edit_namelist(self.SYNTHETIC_WITHOUT_EXISTING, 14, 16, 12, 8,
                                            history_mode="end_d01", ranks=224)
        without_extra_kwargs = configs.edit_namelist(
            self.SYNTHETIC_WITHOUT_EXISTING, 14, 16, 12, 8, history_mode="end_d01", ranks=224,
            numtiles_x=None, numtiles_y=None, tile_sz_x=None, tile_sz_y=None, tile_strategy=None)
        self.assertEqual(with_extra, without_extra_kwargs)
        self.assertNotIn("numtiles_x", with_extra)
        self.assertNotIn("tile_strategy", with_extra)

    def test_tile_geometry_args_written_once_on_domains_line_when_given(self):
        out = configs.edit_namelist(self.SYNTHETIC_WITHOUT_EXISTING, 14, 16, 12, 8,
                                     history_mode="end_d01", ranks=224,
                                     numtiles_x=2, numtiles_y=6, tile_sz_x=0, tile_sz_y=0,
                                     tile_strategy=3)
        self.assertIn("nproc_x=14, nproc_y=16, numtiles=12, numtiles_x=2, numtiles_y=6, "
                      "tile_sz_x=0, tile_sz_y=0, tile_strategy=3,", out)
        for key in ("numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy"):
            self.assertEqual(out.count(key), 1, key)

    def test_tile_geometry_args_partial_only_writes_given_keys(self):
        out = configs.edit_namelist(self.SYNTHETIC_WITHOUT_EXISTING, 14, 16, 12, 8,
                                     history_mode="end_d01", ranks=224, numtiles_x=4)
        self.assertIn("numtiles_x=4", out)
        self.assertNotIn("numtiles_y", out)
        self.assertNotIn("tile_sz_x", out)
        self.assertNotIn("tile_strategy", out)

    def test_tile_geometry_args_replace_stale_existing_line(self):
        text = self.SYNTHETIC_WITHOUT_EXISTING.replace(
            "&domains\n", "&domains\n numtiles_x = 99,\n")
        out = configs.edit_namelist(text, 14, 16, 12, 8, history_mode="end_d01", ranks=224,
                                     numtiles_x=2, numtiles_y=6)
        self.assertNotIn("numtiles_x = 99", out)
        self.assertEqual(out.count("numtiles_x"), 1)
        self.assertIn("numtiles_x=2, numtiles_y=6,", out)

    SYNTHETIC_3DOMAIN = (
        "&time_control\n"
        " run_days = 0, run_hours = 1, run_minutes = 0,\n"
        " history_interval_s = 00, 00, 00,\n"
        " io_form_history = 0,\n"
        " restart = .false.,\n"
        " io_form_input = 11,\n"
        " io_form_boundary = 11,\n"
        "/\n"
        "&domains\n"
        " max_dom = 3,\n"
        " time_step=15, e_we=793, 805, 1001, e_sn=853, 805, 1001,\n"
        "/\n"
    )

    def test_numtiles_is_a_single_value_even_with_nests(self):
        # numtiles is namelist-global; a per-domain list makes WRF abort with
        # "ERROR reading namelist domains" (smoke jobs 1074506-1074508)
        out = configs.edit_namelist(self.SYNTHETIC_3DOMAIN, 16, 28, 8, 30, max_dom=3, ranks=448)
        self.assertIn("nproc_x=16, nproc_y=28, numtiles=8,\n", out)
        self.assertNotIn("numtiles=8, 8", out)
        self.assertEqual(out.count("numtiles"), 1)

    def test_history_interval_s_variant_converts_minutes_to_seconds(self):
        # nest_history_on defaults to False: d01 gets the run length, nested
        # domains (d02/d03) get 0 -- see TestNestHistory for the dedicated coverage
        out = configs.edit_namelist(self.SYNTHETIC_3DOMAIN, 16, 28, 8, 30, max_dom=3,
                                     history_mode="end_d01", ranks=448)
        self.assertIn("history_interval_s = 1800, 0, 0,", out)
        self.assertNotIn("history_interval_s = 00", out)
        self.assertIn("history_begin_s = 1800, 0, 0,", out)
        # io_form_history is always forced to 14 (ADIOS2) regardless of the native
        # value (0 here); _set_scalar preserves this template's own "key = value"
        # spacing style (unlike SYNTHETIC_WITH_EXISTING's "key=value" style above)
        self.assertIn("io_form_history = 14,", out)
        self.assertIn("io_form_input = 14,", out)
        self.assertIn("io_form_boundary = 14,", out)

    def test_set_max_dom_override_on_3domain_template_keeps_perdomain_lists_valid(self):
        out = configs.edit_namelist(
            self.SYNTHETIC_3DOMAIN, 16, 28, 8, 30, max_dom=2,
            history_mode="end_d01", ranks=448, overrides={"max_dom": 2},
        )
        self.assertIn("max_dom = 2", out)
        self.assertNotIn("max_dom = 3", out)
        # numtiles stays a single value; per-domain lists (history_interval_s) follow max_dom
        self.assertIn("nproc_x=16, nproc_y=28, numtiles=8,\n", out)
        # nest_history_on defaults to False: d01 gets the run length, nest gets 0
        self.assertIn("history_interval_s = 1800, 0,", out)

    def test_restart_false_override(self):
        text = self.SYNTHETIC_3DOMAIN.replace("restart = .false.,", "restart = .true.,")
        out = configs.edit_namelist(text, 16, 28, 8, 30, max_dom=2, ranks=448,
                                     overrides={"restart": ".false."})
        self.assertIn("restart = .false.", out)
        self.assertNotIn("restart = .true.", out)

    def test_io_form_input_and_boundary_overrides_apply_with_no_adios2(self):
        # 2026-09-14 all-ADIOS2: this override path only still works with
        # adios2=False. With the new default (adios2=True) the same override can
        # no longer win -- see TestAdios2.test_overrides_cannot_revert_io_form.
        out = configs.edit_namelist(
            self.SYNTHETIC_3DOMAIN, 16, 28, 8, 30, max_dom=2, adios2=False,
            overrides={"max_dom": 2, "io_form_input": 2, "io_form_boundary": 2},
        )
        self.assertIn("io_form_input = 2", out)
        self.assertIn("io_form_boundary = 2", out)
        self.assertNotIn("io_form_input = 11", out)
        self.assertNotIn("io_form_boundary = 11", out)

    def test_edit_namelist_cli_parses_repeated_set_flags(self):
        overrides = {}
        for kv in ["max_dom=2", "restart=.false.", "io_form_input=2"]:
            key, sep, value = kv.partition("=")
            self.assertTrue(sep)
            overrides[key] = value
        self.assertEqual(overrides, {"max_dom": "2", "restart": ".false.", "io_form_input": "2"})


class TestDeleteKeys(unittest.TestCase):
    SYNTHETIC_DELETE_KEYS = (
        "&time_control\n"
        " run_days=0, run_hours=1, run_minutes=0,\n"
        " history_interval=999,\n"
        " io_form_history=14,\n"
        "/\n"
        "&domains\n"
        " time_step=15, e_we=901, e_sn=901, e_vert=45,\n"
        "/\n"
        "&physics\n"
        " bl_mynn_tkebudget = 0, 0, 0,\n"
        " icloud_bl = 1,\n"
        "/\n"
        "&noah_mp\n"
        " opt_run=3, ! former comment\n"
        " dveg=9 ! former 3\n"
        " opt_crs=1,\n"
        "/\n"
    )

    def test_delete_key_lines_is_case_insensitive_and_leaves_others_untouched(self):
        out = configs._delete_key_lines(self.SYNTHETIC_DELETE_KEYS, "BL_MYNN_TKEBUDGET", ignore_case=True)
        out = configs._delete_key_lines(out, "Opt_Run", ignore_case=True)
        self.assertNotIn("tkebudget", out.lower())
        self.assertNotIn("opt_run", out.lower())
        # untouched lines, including the one with an inline "!" comment, stay byte-identical
        self.assertIn(" icloud_bl = 1,\n", out)
        self.assertIn(" dveg=9 ! former 3\n", out)
        self.assertIn(" opt_crs=1,\n", out)

    def test_delete_key_lines_default_is_case_sensitive(self):
        # without ignore_case, a differently-cased key is left alone
        out = configs._delete_key_lines(self.SYNTHETIC_DELETE_KEYS, "BL_MYNN_TKEBUDGET")
        self.assertIn("bl_mynn_tkebudget = 0, 0, 0,\n", out)

    def test_edit_namelist_delete_keys_param_removes_both_lafe_keys(self):
        out = configs.edit_namelist(self.SYNTHETIC_DELETE_KEYS, 14, 16, 12, 8, ranks=224,
                                     delete_keys=["bl_mynn_tkebudget", "opt_run"])
        self.assertNotIn("tkebudget", out.lower())
        self.assertNotIn("opt_run", out.lower())
        self.assertIn(" icloud_bl = 1,\n", out)
        self.assertIn(" dveg=9 ! former 3\n", out)
        self.assertIn(" opt_crs=1,\n", out)

    def test_edit_namelist_without_delete_keys_is_unaffected(self):
        out = configs.edit_namelist(self.SYNTHETIC_DELETE_KEYS, 14, 16, 12, 8, ranks=224)
        self.assertIn("bl_mynn_tkebudget = 0, 0, 0,\n", out)
        self.assertIn("opt_run=3, ! former comment\n", out)


class TestNestHistory(unittest.TestCase):
    """nest_history_on's interaction with history_mode="end_d01" (the mode that
    actually writes a d01 frame). See TestHistoryMode for history_mode itself
    (off/end_d01, all three history-interval variants, max_dom 1/2/3)."""

    SYNTHETIC_NEST_MINUTES = (
        "&time_control\n"
        " run_days = 0, run_hours = 1, run_minutes = 0,\n"
        " history_interval = 999, 999,\n"
        " io_form_history = 2,\n"
        "/\n"
        "&domains\n"
        " max_dom = 2,\n"
        " time_step=15, e_we=1000, 1001, e_sn=1000, 1001,\n"
        "/\n"
    )

    SYNTHETIC_NEST_MINUTES_M = (
        "&time_control\n"
        " run_days = 0, run_hours = 1, run_minutes = 0,\n"
        " history_interval_m = 999, 999, 999,\n"
        " io_form_history = 2,\n"
        "/\n"
        "&domains\n"
        " max_dom = 3,\n"
        " time_step=15, e_we=793, 805, 1001, e_sn=853, 805, 1001,\n"
        "/\n"
    )

    SYNTHETIC_3DOMAIN = TestNamelistEditing.SYNTHETIC_3DOMAIN

    def test_nest_history_off_by_default_minutes_variant_max_dom_2(self):
        out = configs.edit_namelist(self.SYNTHETIC_NEST_MINUTES, 4, 28, 6, 30, max_dom=2,
                                     history_mode="end_d01", ranks=112)
        self.assertIn("history_interval = 30, 0,", out)

    def test_nest_history_off_by_default_minutes_m_variant_max_dom_3(self):
        out = configs.edit_namelist(self.SYNTHETIC_NEST_MINUTES_M, 16, 28, 8, 30, max_dom=3,
                                     history_mode="end_d01", ranks=448)
        self.assertIn("history_interval_m = 30, 0, 0,", out)

    def test_nest_history_off_by_default_seconds_variant_max_dom_3(self):
        out = configs.edit_namelist(self.SYNTHETIC_3DOMAIN, 16, 28, 8, 30, max_dom=3,
                                     history_mode="end_d01", ranks=448)
        self.assertIn("history_interval_s = 1800, 0, 0,", out)

    def test_nest_history_on_keeps_all_domains_minutes_variant(self):
        out = configs.edit_namelist(self.SYNTHETIC_NEST_MINUTES, 4, 28, 6, 30, max_dom=2,
                                     history_mode="end_d01", ranks=112, nest_history_on=True)
        self.assertIn("history_interval = 30, 30,", out)
        self.assertIn("history_begin = 30, 30,", out)

    def test_nest_history_on_keeps_all_domains_seconds_variant(self):
        out = configs.edit_namelist(self.SYNTHETIC_3DOMAIN, 16, 28, 8, 30, max_dom=3,
                                     history_mode="end_d01", ranks=448, nest_history_on=True)
        self.assertIn("history_interval_s = 1800, 1800, 1800,", out)
        self.assertIn("history_begin_s = 1800, 1800, 1800,", out)

    def test_single_domain_history_unchanged_regardless_of_nest_flag(self):
        base = TestNamelistEditing.SYNTHETIC_WITH_EXISTING
        out_default = configs.edit_namelist(base, 14, 16, 12, 8, history_mode="end_d01", ranks=224)
        out_nest_on = configs.edit_namelist(base, 14, 16, 12, 8, history_mode="end_d01", ranks=224,
                                             nest_history_on=True)
        self.assertEqual(out_default, out_nest_on)
        self.assertIn("history_interval=8,", out_default)


class TestHistoryMode(unittest.TestCase):
    """history_mode itself: 'off' (timing runs, no domain writes history) and
    'end_d01' (correctness runs, d01 writes only its final frame via
    history_begin==history_interval==run_minutes -- registry.io_boilerplate:57-62,
    share/set_timekeeping.f90:2745-2757), across all three history-interval
    variants and max_dom 1/2/3."""

    SINGLE = TestNamelistEditing.SYNTHETIC_WITH_EXISTING
    NEST_MINUTES = TestNestHistory.SYNTHETIC_NEST_MINUTES
    NEST_MINUTES_M = TestNestHistory.SYNTHETIC_NEST_MINUTES_M
    NEST_SECONDS = TestNamelistEditing.SYNTHETIC_3DOMAIN

    def test_off_zeroes_single_domain_max_dom_1(self):
        out = configs.edit_namelist(self.SINGLE, 14, 16, 12, 8, history_mode="off", ranks=224)
        self.assertIn("history_interval=0,", out)

    def test_off_zeroes_every_domain_minutes_variant_max_dom_2(self):
        out = configs.edit_namelist(self.NEST_MINUTES, 4, 28, 6, 30, max_dom=2,
                                     history_mode="off", ranks=112)
        self.assertIn("history_interval = 0, 0,", out)

    def test_off_zeroes_every_domain_minutes_m_variant_max_dom_3(self):
        out = configs.edit_namelist(self.NEST_MINUTES_M, 16, 28, 8, 30, max_dom=3,
                                     history_mode="off", ranks=448)
        self.assertIn("history_interval_m = 0, 0, 0,", out)

    def test_off_zeroes_every_domain_seconds_variant_max_dom_3(self):
        out = configs.edit_namelist(self.NEST_SECONDS, 16, 28, 8, 30, max_dom=3,
                                     history_mode="off", ranks=448)
        self.assertIn("history_interval_s = 0, 0, 0,", out)

    def test_end_d01_sets_history_begin_single_domain_max_dom_1(self):
        out = configs.edit_namelist(self.SINGLE, 14, 16, 12, 8, history_mode="end_d01", ranks=224)
        self.assertIn("history_interval=8,", out)
        self.assertIn("history_begin = 8,", out)

    def test_end_d01_sets_history_begin_minutes_variant_max_dom_2(self):
        out = configs.edit_namelist(self.NEST_MINUTES, 4, 28, 6, 30, max_dom=2,
                                     history_mode="end_d01", ranks=112)
        self.assertIn("history_interval = 30, 0,", out)
        self.assertIn("history_begin = 30, 0,", out)

    def test_end_d01_sets_history_begin_minutes_m_variant_max_dom_3(self):
        out = configs.edit_namelist(self.NEST_MINUTES_M, 16, 28, 8, 30, max_dom=3,
                                     history_mode="end_d01", ranks=448)
        self.assertIn("history_interval_m = 30, 0, 0,", out)
        self.assertIn("history_begin_m = 30, 0, 0,", out)

    def test_end_d01_sets_history_begin_seconds_variant_max_dom_3(self):
        out = configs.edit_namelist(self.NEST_SECONDS, 16, 28, 8, 30, max_dom=3,
                                     history_mode="end_d01", ranks=448)
        self.assertIn("history_interval_s = 1800, 0, 0,", out)
        self.assertIn("history_begin_s = 1800, 0, 0,", out)

    def test_invalid_history_mode_raises(self):
        with self.assertRaises(ValueError):
            configs.edit_namelist(self.SINGLE, 14, 16, 12, 8, history_mode="bogus", ranks=224)


class TestAdios2(unittest.TestCase):
    """All-ADIOS2 switch (2026-09-14 user decision): io_form_input/boundary/
    history/restart=14, restart_interval effectively disabled, input_inname/
    bdy_inname and a &namelist_adios2 group with adios2_numaggregators=ranks.
    Recipe proven in regress_bp.sbatch/job 1072441."""

    SYNTHETIC_NO_ADIOS2 = (
        "&time_control\n"
        " run_days=0, run_hours=1, run_minutes=0,\n"
        " history_interval=999,\n"
        " io_form_history=2,\n"
        " io_form_input=2,\n"
        " io_form_boundary=2,\n"
        " io_form_restart=2,\n"
        " restart_interval=60,\n"
        "/\n"
        "&domains\n"
        " time_step=15, e_we=901, e_sn=901, e_vert=45,\n"
        "/\n"
    )

    SYNTHETIC_STALE_ADIOS2 = (
        "&time_control\n"
        " run_days=0, run_hours=1, run_minutes=0,\n"
        " history_interval=999,\n"
        " input_inname=\"wrfinput_d<domain>\", bdy_inname=\"wrfbdy_d<domain>\",\n"
        " io_form_history=2,\n"
        " io_form_input=11,\n"
        " io_form_boundary=11,\n"
        " io_form_restart=2,\n"
        " restart_interval=60,\n"
        "/\n"
        "&domains\n"
        " time_step=15, e_we=901, e_sn=901, e_vert=45,\n"
        "/\n"
        "&namelist_adios2\n"
        " adios2_numaggregators=99,\n"
        "/\n"
    )

    def test_inserts_all_settings_on_namelist_without_existing_adios2_entries(self):
        out = configs.edit_namelist(self.SYNTHETIC_NO_ADIOS2, 14, 16, 12, 8, ranks=224)
        self.assertIn("io_form_input=14,", out)
        self.assertIn("io_form_boundary=14,", out)
        self.assertIn("io_form_history=14,", out)
        self.assertIn("io_form_restart=14,", out)
        self.assertIn("restart_interval=1000000,", out)
        self.assertIn('input_inname="wrfinput_d<domain>.bp", bdy_inname="wrfbdy_d<domain>.bp",', out)
        self.assertIn("&namelist_adios2\n adios2_numaggregators=224,\n/\n", out)
        self.assertEqual(out.count("input_inname"), 1)
        self.assertEqual(out.count("&namelist_adios2"), 1)

    def test_replaces_existing_entries_no_duplicates(self):
        out = configs.edit_namelist(self.SYNTHETIC_STALE_ADIOS2, 14, 16, 12, 8, ranks=224)
        self.assertEqual(out.count("input_inname"), 1)
        self.assertEqual(out.count("bdy_inname"), 1)
        self.assertEqual(out.count("&namelist_adios2"), 1)
        self.assertEqual(out.count("adios2_numaggregators"), 1)
        self.assertIn("adios2_numaggregators=224,", out)
        self.assertNotIn("adios2_numaggregators=99", out)
        self.assertNotIn('input_inname="wrfinput_d<domain>"', out)  # stale (no .bp) line gone
        self.assertIn('input_inname="wrfinput_d<domain>.bp"', out)
        self.assertIn("io_form_input=14,", out)
        self.assertIn("io_form_boundary=14,", out)
        self.assertIn("io_form_restart=14,", out)
        self.assertNotIn("io_form_input=11", out)
        self.assertNotIn("io_form_restart=2,", out)

    def test_overrides_cannot_revert_io_form(self):
        # 2026-09-14: cases.json's lafe used to override io_form_input/boundary to
        # 2 (pnetcdf->plain netCDF); with adios2=True (default) that override
        # must lose -- _apply_adios2 runs after overrides in edit_namelist.
        out = configs.edit_namelist(
            self.SYNTHETIC_NO_ADIOS2, 14, 16, 12, 8, ranks=224,
            overrides={"io_form_input": 2, "io_form_boundary": 2},
        )
        self.assertIn("io_form_input=14,", out)
        self.assertIn("io_form_boundary=14,", out)
        self.assertNotIn("io_form_input=2,", out)
        self.assertNotIn("io_form_boundary=2,", out)

    def test_inserts_missing_io_form_keys_like_isc21s_real_template(self):
        # isc21's real namelist.input has io_form_history but NO io_form_restart/
        # io_form_input/io_form_boundary lines at all -- _set_scalar alone would
        # silently no-op on the missing three, leaving them at WRF's default
        # while isc21 is fed .bp (ADIOS2) inputs. Must be inserted, not skipped.
        text = (
            "&time_control\n"
            " run_days=0, run_hours=1, run_minutes=0,\n"
            " history_interval_s = 00, 00, 00,\n"
            " io_form_history=0,\n"
            "/\n"
            "&domains\n"
            " max_dom=3,\n"
            " time_step=15, e_we=793, 805, 1001, e_sn=853, 805, 1001,\n"
            "/\n"
        )
        out = configs.edit_namelist(text, 16, 28, 8, 7, max_dom=3, ranks=448)
        self.assertIn("io_form_history=14,", out)
        self.assertIn("io_form_restart=14,", out)
        self.assertIn("io_form_input=14,", out)
        self.assertIn("io_form_boundary=14,", out)
        self.assertEqual(out.count("io_form_input"), 1)
        self.assertEqual(out.count("io_form_boundary"), 1)
        self.assertEqual(out.count("io_form_restart"), 1)

    def test_ranks_required_when_adios2_enabled(self):
        with self.assertRaises(ValueError):
            configs.edit_namelist(self.SYNTHETIC_NO_ADIOS2, 14, 16, 12, 8)

    def test_no_adios2_preserves_legacy_behaviour(self):
        out = configs.edit_namelist(self.SYNTHETIC_NO_ADIOS2, 14, 16, 12, 8, adios2=False)
        self.assertIn("io_form_history=2,", out)
        self.assertIn("io_form_input=2,", out)      # untouched: no ADIOS2 rewriting
        self.assertIn("io_form_boundary=2,", out)
        self.assertIn("io_form_restart=2,", out)
        self.assertIn("restart_interval=60,", out)  # untouched: no restart-interval override
        self.assertNotIn("input_inname", out)
        self.assertNotIn("&namelist_adios2", out)

    def test_numaggregators_equals_ranks_argument(self):
        for ranks in (112, 224, 448):
            out = configs.edit_namelist(self.SYNTHETIC_NO_ADIOS2, 14, 16, 12, 8, ranks=ranks)
            self.assertIn("adios2_numaggregators=%d," % ranks, out)


class TestStepStats(unittest.TestCase):
    def _series_with_radiation_spikes(self, n_cycles, cycle_len, normal=6.0, spike=44.0, init=100.0):
        rest = []
        for i in range(n_cycles * cycle_len):
            rest.append(spike if (i + 1) % cycle_len == 0 else normal)
        return [init] + rest, rest

    def test_median_excludes_spikes_mean_all_includes_them(self):
        values, rest = self._series_with_radiation_spikes(n_cycles=6, cycle_len=4)
        stats = configs.step_stats(values)
        n_spikes = sum(1 for v in rest if v == 44.0)
        self.assertEqual(stats["n_all"], len(rest))
        self.assertEqual(stats["n_measured"], len(rest) - n_spikes)
        self.assertEqual(stats["n_excluded"], 1 + n_spikes)
        self.assertAlmostEqual(stats["median"], 6.0)
        self.assertAlmostEqual(stats["mean_all"], sum(rest) / len(rest))
        self.assertGreater(stats["mean_all"], stats["median"])

    def test_empty_input(self):
        stats = configs.step_stats([])
        self.assertIsNone(stats["median"])
        self.assertIsNone(stats["mean_all"])
        self.assertEqual(stats["n_measured"], 0)
        self.assertEqual(stats["n_excluded"], 0)
        self.assertEqual(stats["n_all"], 0)

    def test_single_step_only_no_measurable_steps(self):
        stats = configs.step_stats([12.3])
        self.assertEqual(stats["n_all"], 0)
        self.assertEqual(stats["n_excluded"], 1)
        self.assertIsNone(stats["median"])
        self.assertIsNone(stats["mean_all"])


class TestStepStatsCLI(unittest.TestCase):
    def test_cli_output_format(self):
        import contextlib
        import io
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with open(path, "w") as f:
                for v in [100.0, 6.0, 6.0, 44.0, 6.0]:
                    f.write("%.4f\n" % v)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = configs._cmd_step_stats([path])
            self.assertEqual(rc, 0)
            out = buf.getvalue().strip()
            self.assertRegex(out, r"^med=[\d.]+ mean=[\d.]+ n=\d+ excl=\d+ nall=\d+$")
        finally:
            os.remove(path)

    def test_cli_output_format_empty_file(self):
        import contextlib
        import io
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = configs._cmd_step_stats([path])
            self.assertEqual(rc, 0)
            out = buf.getvalue().strip()
            self.assertEqual(out, "med= mean= n=0 excl=0 nall=0")
        finally:
            os.remove(path)


class TestCasesJsonStrictCampaignEdits(unittest.TestCase):
    def test_lafe_has_namelist_delete_keys(self):
        lafe = configs.CASES["lafe"]
        self.assertIn("namelist_delete_keys", lafe)
        # iofields_filename / ignore_iofields_warning were added 2026-09-14 after the
        # modify_io_masks1 SIGSEGV (smoke job 1074691)
        self.assertEqual(set(lafe["namelist_delete_keys"]),
                         {"bl_mynn_tkebudget", "opt_run", "iofields_filename", "ignore_iofields_warning"})

    def test_isc21_disables_validation_aux_stream(self):
        isc21 = configs.CASES["isc21"]
        self.assertEqual(isc21.get("namelist_overrides", {}).get("io_form_auxhist24"), 0)

    def test_all_cases_have_bp_dir(self):
        for cid, case in configs.CASES.items():
            self.assertIn("bp_dir", case, cid)
            self.assertTrue(case["bp_dir"], cid)

    def test_input_files_never_contain_netcdf_wrfinput_or_wrfbdy(self):
        # wrfinput_d0N/wrfbdy_d01 are always BP now (symlinked from bp_dir by the
        # runner); input_files is only non-netCDF auxiliary files.
        for cid, case in configs.CASES.items():
            for f in case["input_files"]:
                self.assertFalse(f.startswith("wrfinput"), "%s: %s" % (cid, f))
                self.assertFalse(f.startswith("wrfbdy"), "%s: %s" % (cid, f))

    def test_lafe_namelist_overrides_no_longer_sets_io_form(self):
        # 2026-09-14 all-ADIOS2: the old io_form_input/boundary=2 (pnetcdf->netCDF)
        # override is gone -- edit_namelist's ADIOS2 pass would just override it
        # right back to 14 anyway, so keeping it in cases.json was pure noise.
        lafe = configs.CASES["lafe"]
        overrides = lafe.get("namelist_overrides", {})
        self.assertNotIn("io_form_input", overrides)
        self.assertNotIn("io_form_boundary", overrides)


class TestComplianceKnobs(unittest.TestCase):
    def test_all_varied_knobs_are_strict(self):
        sys.path.insert(0, configs.REPO_ROOT)
        try:
            import wtune_compliance
        except Exception as e:  # pragma: no cover - only if repo layout changes
            self.skipTest("wtune_compliance not importable: %s" % e)
        for name in configs._COMPLIANCE_KNOBS:
            self.assertEqual(wtune_compliance.classify(name), wtune_compliance.STRICT, name)


class TestCasesJson(unittest.TestCase):
    def test_all_cases_have_required_fields(self):
        required = ("prep_dir", "bp_dir", "restart", "input_files", "history_key", "dt",
                    "e_we", "e_sn", "e_vert", "dx", "dy", "case_family", "steps", "physics")
        for cid, case in configs.CASES.items():
            for field in required:
                self.assertIn(field, case, "%s missing %s" % (cid, field))

    def test_case_order_matches_cases_json_keys(self):
        self.assertEqual(sorted(configs.CASE_ORDER), sorted(configs.CASES.keys()))


class TestAdios2MultiKeyLine(unittest.TestCase):
    """karina's template writes several io_form keys on ONE line; ADIOS2 editing must rewrite them in place,
    not insert duplicates while leaving the old '=2' values (smoke job 1074653: 'error opening wrfinput_d01.bp')."""

    KARINA_LIKE = (
        "&time_control\n"
        " run_hours=0, run_minutes=8, history_interval=8,\n"
        " io_form_history=2, io_form_restart=2, io_form_input=2, io_form_boundary=2,\n"
        "/\n"
        "&domains\n"
        " time_step=15, e_we=901, e_sn=901, e_vert=45,\n"
        "/\n"
    )

    def test_multi_key_line_rewritten_without_duplicates(self):
        out = configs.edit_namelist(self.KARINA_LIKE, 7, 32, 12, 8, ranks=224)
        for key in ("io_form_history", "io_form_restart", "io_form_input", "io_form_boundary"):
            self.assertEqual(out.count(key + "="), 1, key)
            self.assertIn(key + "=14", out, key)
        self.assertNotIn("=2,", out.split("&domains")[0].replace("nio_groups=1,", ""))


class TestQuiltingOff(unittest.TestCase):
    """ADIOS2 editing must always disable WRF quilting (smoke job 1074562 hung with the LAFE template's
    nio_tasks_per_group=32, nio_groups=4)."""

    LAFE_LIKE = (
        "&time_control\n"
        " run_minutes = 0,\n"
        " history_interval = 5, 5, 5,\n"
        " io_form_history = 11,\n"
        " io_form_input = 11,\n"
        " io_form_boundary = 11,\n"
        " io_form_restart = 11,\n"
        "/\n"
        "&domains\n"
        " max_dom = 2,\n"
        " time_step = 10,\n"
        "/\n"
        "&namelist_quilt\n"
        " nio_tasks_per_group = 32,\n"
        " nio_groups = 4,\n"
        "! nio_tasks_per_group = 0,\n"
        "! nio_groups = 1,\n"
        "/\n"
    )

    def test_quilt_group_replaced_with_quilting_off(self):
        out = configs.edit_namelist(self.LAFE_LIKE, 4, 56, 12, 5, ranks=224, max_dom=2)
        self.assertEqual(out.count("&namelist_quilt"), 1)
        self.assertIn("nio_tasks_per_group=0,", out)
        self.assertIn("nio_groups=1,", out)
        self.assertNotIn("nio_tasks_per_group = 32", out)
        self.assertNotIn("nio_groups = 4", out)

    def test_quilt_group_added_when_absent(self):
        text = self.LAFE_LIKE.split("&namelist_quilt")[0]
        out = configs.edit_namelist(text, 4, 56, 12, 5, ranks=224, max_dom=2)
        self.assertEqual(out.count("&namelist_quilt"), 1)
        self.assertIn("nio_tasks_per_group=0,", out)


class TestHistoryModeInterval(unittest.TestCase):
    """MODE=physics fidelity replicates (decision 0051): --history-mode interval
    --history-minutes N writes a d01 frame every N minutes starting at t=0
    (history_begin*=0), nested domains still off by default -- same
    nest_history_on-gated rule as 'off'/'end_d01'."""

    SINGLE = TestNamelistEditing.SYNTHETIC_WITH_EXISTING
    NEST_MINUTES = TestNestHistory.SYNTHETIC_NEST_MINUTES
    NEST_MINUTES_M = TestNestHistory.SYNTHETIC_NEST_MINUTES_M
    NEST_SECONDS = TestNamelistEditing.SYNTHETIC_3DOMAIN

    def test_interval_single_domain(self):
        out = configs.edit_namelist(self.SINGLE, 14, 16, 12, 60, history_mode="interval",
                                     history_minutes=10, ranks=224)
        self.assertIn("history_interval=10,", out)
        self.assertIn("history_begin = 0,", out)
        self.assertIn("run_minutes=60", out)

    def test_interval_minutes_variant_max_dom_2(self):
        out = configs.edit_namelist(self.NEST_MINUTES, 4, 28, 6, 60, max_dom=2,
                                     history_mode="interval", history_minutes=10, ranks=112)
        self.assertIn("history_interval = 10, 0,", out)
        self.assertIn("history_begin = 0, 0,", out)

    def test_interval_minutes_m_variant_max_dom_3(self):
        out = configs.edit_namelist(self.NEST_MINUTES_M, 16, 28, 8, 60, max_dom=3,
                                     history_mode="interval", history_minutes=10, ranks=448)
        self.assertIn("history_interval_m = 10, 0, 0,", out)
        self.assertIn("history_begin_m = 0, 0, 0,", out)

    def test_interval_seconds_variant_max_dom_3(self):
        out = configs.edit_namelist(self.NEST_SECONDS, 16, 28, 8, 30, max_dom=3,
                                     history_mode="interval", history_minutes=5, ranks=448)
        self.assertIn("history_interval_s = 300, 0, 0,", out)
        self.assertIn("history_begin_s = 0, 0, 0,", out)

    def test_interval_requires_history_minutes(self):
        with self.assertRaises(ValueError):
            configs.edit_namelist(self.SINGLE, 14, 16, 12, 60, history_mode="interval", ranks=224)

    def test_off_and_end_d01_still_work_alongside_interval(self):
        # 'interval' is additive -- the two pre-existing modes are unaffected.
        out_off = configs.edit_namelist(self.SINGLE, 14, 16, 12, 8, history_mode="off", ranks=224)
        self.assertIn("history_interval=0,", out_off)
        out_end = configs.edit_namelist(self.SINGLE, 14, 16, 12, 8, history_mode="end_d01", ranks=224)
        self.assertIn("history_interval=8,", out_end)


class TestFindNamelistKey(unittest.TestCase):
    """_find_namelist_key's own-value-list-only extraction: it must stop before a
    different key's assignment jammed onto the same line, and must not fabricate
    values beyond what the line actually has."""

    def test_single_value_own_line(self):
        text = "&physics\n mp_physics=8, ra_lw_physics=4,\n/\n"
        found = configs._find_namelist_key(text, "mp_physics")
        self.assertIsNotNone(found)
        self.assertEqual(found["values"], ["8"])
        # ra_lw_physics on the SAME line must not leak into mp_physics's own values
        self.assertNotIn("4", found["values"])

    def test_multi_value_own_line(self):
        text = "&physics\n radt = 3, 3, 3,\n/\n"
        found = configs._find_namelist_key(text, "radt")
        self.assertEqual(found["values"], ["3", "3", "3"])

    def test_stops_before_next_key_jammed_on_same_line(self):
        text = "&domains\n time_step=15, e_we=901, e_sn=901, e_vert=45,\n/\n"
        found = configs._find_namelist_key(text, "time_step")
        self.assertEqual(found["values"], ["15"])
        # replacing [value_start:value_end] must leave the rest of the line intact
        rebuilt = text[:found["value_start"]] + "18," + text[found["value_end"]:]
        self.assertIn("time_step=18, e_we=901, e_sn=901, e_vert=45,", rebuilt)

    def test_missing_key_returns_none(self):
        text = "&physics\n mp_physics=8,\n/\n"
        self.assertIsNone(configs._find_namelist_key(text, "swint_opt"))

    def test_commented_line_never_matches(self):
        text = "&physics\n! swint_opt = 0,\n mp_physics=8,\n/\n"
        self.assertIsNone(configs._find_namelist_key(text, "swint_opt"))


class TestApplyPhysicsEdit(unittest.TestCase):
    """apply_physics_edit: set/mul/delete on single- and multi-domain namelists,
    trailing (beyond max_dom) entries left untouched, and the -1 physics_suite
    'mul' hard error."""

    SINGLE_DOMAIN = "&physics\n mp_physics=8, radt=15,\n/\n&domains\n time_step=15, e_we=901, e_sn=901, e_vert=45,\n/\n"
    THREE_DOMAIN = "&physics\n mp_physics = 8, 8, 8,\n radt = 3, 3, 3,\n swint_opt = 1,\n/\n&domains\n max_dom=3,\n time_step = 15,\n/\n"
    SUITE = "&physics\n physics_suite = 'conus'\n mp_physics = -1,\n radt = 10,\n/\n&domains\n time_step=15,\n/\n"
    # lafe-like: 3 native values on-disk, but the case's own max_dom override trims to 1
    TRIMMED = "&physics\n mp_physics = 28, 28, 28,\n radt = 1, 1, 1,\n/\n&domains\n max_dom = 3,\n time_step = 10,\n/\n"
    # maria1km-like: 3 native values, max_dom=2, and d03's OWN value differs from d01/d02
    NONUNIFORM_TRAILING = "&physics\n radt = 12, 12, 30,\n/\n&domains\n max_dom = 2,\n time_step = 25,\n/\n"

    def test_set_on_existing_single_value(self):
        out, old, new = configs.apply_physics_edit(
            self.SINGLE_DOMAIN, {"op": "set", "key": "mp_physics", "value": 6}, 1)
        self.assertEqual(old, ["8"])
        self.assertEqual(new, ["6"])
        # SINGLE_DOMAIN's own "key=value" spacing (no spaces) is preserved --
        # only the value list is rewritten, never the "key = " prefix.
        self.assertIn("mp_physics=6,", out)

    def test_set_inserts_single_value_when_absent(self):
        out, old, new = configs.apply_physics_edit(
            self.SINGLE_DOMAIN, {"op": "set", "key": "swint_opt", "value": 1}, 1)
        self.assertEqual(old, [])
        self.assertEqual(new, ["1"])
        self.assertIn("swint_opt = 1,", out)
        # even with max_dom>1 elsewhere in the file, an inserted key gets exactly one value
        out2, _old2, new2 = configs.apply_physics_edit(
            self.THREE_DOMAIN, {"op": "set", "key": "diff_opt", "value": 1}, 3)
        self.assertEqual(new2, ["1"])
        self.assertEqual(out2.count("diff_opt"), 1)

    def test_mul_multiplies_every_active_domain_independently(self):
        out, old, new = configs.apply_physics_edit(
            self.THREE_DOMAIN, {"op": "mul", "key": "radt", "factor": 2, "round": "int"}, 3)
        self.assertEqual(old, ["3", "3", "3"])
        self.assertEqual(new, ["6", "6", "6"])
        self.assertIn("radt = 6, 6, 6,", out)

    def test_mul_leaves_trailing_beyond_max_dom_untouched(self):
        # lafe-like: max_dom override trims 3 -> 1; entries beyond index 0 must
        # be preserved byte-for-byte, per the task's own trim-is-harmless note.
        out, old, new = configs.apply_physics_edit(
            self.TRIMMED, {"op": "mul", "key": "radt", "factor": 2, "round": "int"}, 1)
        self.assertEqual(old, ["1"])
        self.assertEqual(new, ["2"])
        self.assertIn("radt = 2, 1, 1,", out)

    def test_mul_leaves_nonuniform_trailing_domain_untouched(self):
        # maria1km-like: d03's OWN native value (30) differs from d01/d02 (12) and
        # must stay 30, not be derived from d01's new value.
        out, old, new = configs.apply_physics_edit(
            self.NONUNIFORM_TRAILING, {"op": "mul", "key": "radt", "factor": 2, "round": "int"}, 2)
        self.assertEqual(old, ["12", "12"])
        self.assertEqual(new, ["24", "24"])
        self.assertIn("radt = 24, 24, 30,", out)

    def test_mul_time_step_rounds_to_int(self):
        # conus12-like: dt=72 -> 72*1.2=86.4 -> 86
        text = "&domains\n time_step=72, e_we=425, e_sn=300, e_vert=50,\n/\n"
        out, old, new = configs.apply_physics_edit(
            text, {"op": "mul", "key": "time_step", "factor": 1.2, "round": "int"}, 1)
        self.assertEqual(old, ["72"])
        self.assertEqual(new, ["86"])
        self.assertIn("time_step=86, e_we=425, e_sn=300, e_vert=50,", out)

    def test_mul_time_step_exact_multiple(self):
        # karina-like: dt=15 -> 15*1.2=18 exactly
        text = "&domains\n time_step=15, e_we=901, e_sn=901, e_vert=45,\n/\n"
        out, old, new = configs.apply_physics_edit(
            text, {"op": "mul", "key": "time_step", "factor": 1.2, "round": "int"}, 1)
        self.assertEqual(new, ["18"])
        self.assertIn("time_step=18, e_we=901, e_sn=901, e_vert=45,", out)

    def test_mul_on_suite_default_minus_one_is_an_error(self):
        with self.assertRaises(ValueError):
            configs.apply_physics_edit(self.SUITE, {"op": "mul", "key": "mp_physics", "factor": 2}, 1)

    def test_set_on_suite_default_minus_one_overrides_only_that_key(self):
        out, old, new = configs.apply_physics_edit(
            self.SUITE, {"op": "set", "key": "mp_physics", "value": 6}, 1)
        self.assertEqual(old, ["-1"])
        self.assertEqual(new, ["6"])
        self.assertIn("mp_physics = 6,", out)
        # radt (not a suite-default key here) is untouched by this edit
        self.assertIn("radt = 10,", out)

    def test_delete_removes_key(self):
        out, old, new = configs.apply_physics_edit(
            self.SINGLE_DOMAIN, {"op": "delete", "key": "radt"}, 1)
        self.assertIsNone(old)
        self.assertIsNone(new)
        self.assertNotIn("radt", out)

    def test_delete_missing_key_is_an_error(self):
        with self.assertRaises(ValueError):
            configs.apply_physics_edit(self.SINGLE_DOMAIN, {"op": "delete", "key": "swint_opt"}, 1)

    def test_mul_missing_key_is_an_error(self):
        with self.assertRaises(ValueError):
            configs.apply_physics_edit(self.SINGLE_DOMAIN, {"op": "mul", "key": "swint_opt", "factor": 2}, 1)

    def test_mul_respects_min_clamp(self):
        text = "&physics\n radt = 1,\n/\n"
        out, old, new = configs.apply_physics_edit(
            text, {"op": "mul", "key": "radt", "factor": 0.1, "round": "int", "min": 1}, 1)
        self.assertEqual(new, ["1"])  # 1*0.1=0.1 -> round 0 -> clamped up to min 1


class TestApplyPhysicsTemplate(unittest.TestCase):
    def test_p00_reference_is_a_noop(self):
        template = {"family": "none", "edits": []}
        out, summary = configs.apply_physics_template(TestApplyPhysicsEdit.SINGLE_DOMAIN, template, 1)
        self.assertEqual(out, TestApplyPhysicsEdit.SINGLE_DOMAIN)
        self.assertEqual(summary, [])

    def test_multi_edit_template_applies_in_order(self):
        template = {
            "family": "radiation",
            "edits": [
                {"op": "mul", "key": "radt", "factor": 2, "round": "int"},
                {"op": "set", "key": "swint_opt", "value": 1},
            ],
        }
        out, summary = configs.apply_physics_template(TestApplyPhysicsEdit.SINGLE_DOMAIN, template, 1)
        # radt: SINGLE_DOMAIN's own no-space style is preserved (only the value changes)
        self.assertIn("radt=30,", out)
        # swint_opt: inserted from absent, so it gets the insert path's own " = " style
        self.assertIn("swint_opt = 1,", out)
        self.assertEqual([e["key"] for e in summary], ["radt", "swint_opt"])
        self.assertEqual(summary[0]["new"], ["30"])

    def test_raises_leaves_no_way_to_get_a_partial_result(self):
        template = {"family": "microphysics", "edits": [{"op": "mul", "key": "mp_physics", "factor": 2}]}
        with self.assertRaises(ValueError):
            configs.apply_physics_template(TestApplyPhysicsEdit.SUITE, template, 1)


class TestFormatPhysicsEdits(unittest.TestCase):
    def test_empty_summary_is_empty_string(self):
        self.assertEqual(configs.format_physics_edits([]), "")

    def test_set_and_mul_join_with_semicolons_no_spaces(self):
        summary = [
            dict(key="radt", op="mul", old=["15"], new=["30"]),
            dict(key="swint_opt", op="set", old=[], new=["1"]),
        ]
        s = configs.format_physics_edits(summary)
        self.assertEqual(s, "radt=30;swint_opt=1")
        self.assertNotIn(" ", s)

    def test_multidomain_new_values_slash_joined(self):
        summary = [dict(key="mp_physics", op="set", old=["-1", "-1"], new=["6", "6"])]
        self.assertEqual(configs.format_physics_edits(summary), "mp_physics=6/6")

    def test_delete_shows_deleted_marker(self):
        summary = [dict(key="opt_run", op="delete", old=None, new=None)]
        self.assertEqual(configs.format_physics_edits(summary), "opt_run=DELETED")


class TestProbeVerdicts(unittest.TestCase):
    SAMPLE_SCREEN_TEXT = (
        "STAGE 2 -- SpeedyWeather cheap physics vet:\n"
        "  io_form_history [STRICT       ] PASS->WRF   physics-independent  [WRF-VALIDATED]\n"
        "  radt            [PHYSICS-GRAY ] REJECT      cloud-cap ~8.6 min at forecast wind 18 m/s  [REJECTED]\n"
        "  swint_opt       [PHYSICS-GRAY ] CONFIRM     terminator run  [PROBE-ADVISORY]\n"
        "  mp_physics      [PHYSICS-GRAY ] REJECT      convective regime  [REJECTED]\n"
        "  time_step       [PHYSICS-GRAY ] PASS->WRF   acoustic CFL is dynamics  [MODEL-ONLY]\n"
    )

    def test_parse_probe_verdicts(self):
        v = configs.parse_probe_verdicts(self.SAMPLE_SCREEN_TEXT)
        self.assertEqual(v["radt"], "REJECT")
        self.assertEqual(v["swint_opt"], "CONFIRM")
        self.assertEqual(v["mp_physics"], "REJECT")
        self.assertEqual(v["time_step"], "PASS->WRF")
        self.assertEqual(v["io_form_history"], "PASS->WRF")

    def test_probe_verdicts_for_case_reads_screen_file(self):
        import tempfile
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "screen_testcase.txt"), "w") as f:
            f.write(self.SAMPLE_SCREEN_TEXT)
        s = configs.probe_verdicts_for_case("testcase", ["radt", "swint_opt"], screen_dir=d)
        self.assertEqual(s, "probe=radt:REJECT,swint_opt:CONFIRM")

    def test_probe_verdicts_unknown_key_is_notfound(self):
        import tempfile
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "screen_testcase.txt"), "w") as f:
            f.write(self.SAMPLE_SCREEN_TEXT)
        s = configs.probe_verdicts_for_case("testcase", ["diff_opt"], screen_dir=d)
        self.assertEqual(s, "probe=diff_opt:NOTFOUND")

    def test_probe_verdicts_missing_file_is_none(self):
        import tempfile
        d = tempfile.mkdtemp()
        s = configs.probe_verdicts_for_case("no_such_case", ["radt"], screen_dir=d)
        self.assertEqual(s, "probe=none")

    def test_probe_verdicts_no_keys_is_none(self):
        self.assertEqual(configs.probe_verdicts_for_case("karina", [], screen_dir="/nonexistent"), "probe=none")


class TestPhysicsTagTemplate(unittest.TestCase):
    def test_ref_and_twin_tags(self):
        self.assertEqual(configs.physics_tag_template("P00_ref"), ("P00", "reference"))
        self.assertEqual(configs.physics_tag_template("P00_twin"), ("P00", "twin"))

    def test_candidate_tag(self):
        self.assertEqual(configs.physics_tag_template("P02"), ("P02", "candidate"))


class TestPhysicsTemplatesJson(unittest.TestCase):
    """physics_templates.json itself: loading, and the final P00-P06 set (decision 0051,
    revised 2026-09-15 from the WRF 4.8 source compatibility survey)."""

    ALL = ["P00", "P01", "P02", "P03", "P04", "P05", "P06"]

    def test_loads_and_has_p00_through_p06(self):
        self.assertEqual(set(configs.PHYSICS_TEMPLATES), set(self.ALL))
        self.assertEqual(configs.PHYSICS_TEMPLATE_ORDER, self.ALL)

    def test_p00_has_no_edits(self):
        self.assertEqual(configs.PHYSICS_TEMPLATES["P00"]["edits"], [])

    def test_p01_sets_swint_opt(self):
        edits = configs.PHYSICS_TEMPLATES["P01"]["edits"]
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["op"], "set")
        self.assertEqual(edits[0]["key"], "swint_opt")
        self.assertEqual(edits[0]["value"], 1)

    def test_p02_multiplies_radt_and_sets_swint_opt(self):
        edits = configs.PHYSICS_TEMPLATES["P02"]["edits"]
        keys = [e["key"] for e in edits]
        self.assertEqual(keys, ["radt", "swint_opt"])
        self.assertEqual(edits[0]["op"], "mul")
        self.assertEqual(edits[0]["factor"], 2)

    def test_p03_multiplies_time_step(self):
        edits = configs.PHYSICS_TEMPLATES["P03"]["edits"]
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["key"], "time_step")
        self.assertEqual(edits[0]["factor"], 1.2)
        self.assertEqual(edits[0].get("round"), "int")

    def test_p04_sets_mp_physics_skips_maria1km_and_pairs_lafe_aerosol(self):
        template = configs.PHYSICS_TEMPLATES["P04"]
        self.assertEqual(template["edits"], [{"op": "set", "key": "mp_physics", "value": 6, "section": "&physics"}])
        self.assertIn("maria1km", template.get("skip_cases", {}))
        self.assertNotIn("lafe", template.get("skip_cases", {}))
        lafe_keys = [e["key"] for e in template["extra_edits"]["lafe"]]
        self.assertEqual(lafe_keys, ["aer_opt", "use_aero_icbc"])

    def test_p05_pbl_swap_per_case_pairs(self):
        extra = configs.PHYSICS_TEMPLATES["P05"]["extra_edits"]
        expect = {"karina": 5, "maria1km": 5, "isc21": 1, "lafe": 1, "conus25": 1, "conus12": 1}
        for case_id, bl in expect.items():
            vals = {e["key"]: e["value"] for e in extra[case_id]}
            self.assertEqual(vals, {"bl_pbl_physics": bl, "sf_sfclay_physics": bl}, case_id)

    def test_p06_radiation_swap_and_lafe_aer_opt(self):
        template = configs.PHYSICS_TEMPLATES["P06"]
        self.assertEqual([(e["key"], e["value"]) for e in template["edits"]],
                         [("ra_lw_physics", 1), ("ra_sw_physics", 1)])
        self.assertEqual(template["extra_edits"]["lafe"][0]["key"], "aer_opt")

    def test_extra_edits_applied_only_for_named_case(self):
        text = "&physics\n mp_physics = 28,\n aer_opt = 3,\n use_aero_icbc = .true.,\n/\n"
        template = configs.PHYSICS_TEMPLATES["P04"]
        lafe_text, lafe_summary = configs.apply_physics_template(text, template, 1, case_id="lafe")
        self.assertIn("aer_opt = 0,", lafe_text)
        self.assertIn("use_aero_icbc = .false.,", lafe_text)
        self.assertEqual([e["key"] for e in lafe_summary], ["mp_physics", "aer_opt", "use_aero_icbc"])
        other_text, other_summary = configs.apply_physics_template(text, template, 1, case_id="karina")
        self.assertIn("aer_opt = 3,", other_text)
        self.assertEqual([e["key"] for e in other_summary], ["mp_physics"])

    def test_every_template_has_a_family(self):
        for tid, t in configs.PHYSICS_TEMPLATES.items():
            self.assertIn("family", t, tid)
            self.assertIn("description", t, tid)


class TestResolvePhysicsConfig(unittest.TestCase):
    """resolve_physics_config against synthetic case/template dicts and a real
    temp namelist file on disk (it must read the case's own namelist, exactly
    like edit-namelist itself would)."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def _write_namelist(self, name, text):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_resolves_set_edit_against_real_file(self):
        path = self._write_namelist("nl1.input", TestApplyPhysicsEdit.SINGLE_DOMAIN)
        case = dict(namelist_template=path, max_dom=1)
        template = {"family": "microphysics", "edits": [{"op": "set", "key": "mp_physics", "value": 6}]}
        summary, reason = configs.resolve_physics_config(
            "c1", "PX", cases={"c1": case}, templates={"PX": template})
        self.assertIsNone(reason)
        self.assertEqual(summary[0]["new"], ["6"])

    def test_applies_to_restricts_cases(self):
        path = self._write_namelist("nl2.input", TestApplyPhysicsEdit.SINGLE_DOMAIN)
        case = dict(namelist_template=path, max_dom=1)
        template = {"family": "microphysics", "edits": [], "applies_to": ["other_case"]}
        summary, reason = configs.resolve_physics_config(
            "c1", "PX", cases={"c1": case}, templates={"PX": template})
        self.assertIsNone(summary)
        self.assertIn("applies_to", reason)

    def test_skip_cases_excludes_with_reason(self):
        path = self._write_namelist("nl3.input", TestApplyPhysicsEdit.SINGLE_DOMAIN)
        case = dict(namelist_template=path, max_dom=1)
        template = {"family": "microphysics", "edits": [], "skip_cases": {"c1": "TODO: needs a survey"}}
        summary, reason = configs.resolve_physics_config(
            "c1", "PX", cases={"c1": case}, templates={"PX": template})
        self.assertIsNone(summary)
        self.assertEqual(reason, "TODO: needs a survey")

    def test_mul_on_suite_minus_one_reports_as_skip_reason(self):
        path = self._write_namelist("nl4.input", TestApplyPhysicsEdit.SUITE)
        case = dict(namelist_template=path, max_dom=1)
        template = {"family": "microphysics", "edits": [{"op": "mul", "key": "mp_physics", "factor": 2}]}
        summary, reason = configs.resolve_physics_config(
            "c1", "PX", cases={"c1": case}, templates={"PX": template})
        self.assertIsNone(summary)
        self.assertIn("-1", reason)

    def test_max_dom_override_wins_over_case_max_dom(self):
        path = self._write_namelist("nl5.input", TestApplyPhysicsEdit.TRIMMED)
        case = dict(namelist_template=path, max_dom=3, namelist_overrides={"max_dom": 1})
        template = {"family": "dynamics", "edits": [{"op": "mul", "key": "radt", "factor": 2, "round": "int"}]}
        summary, reason = configs.resolve_physics_config(
            "c1", "PX", cases={"c1": case}, templates={"PX": template})
        self.assertIsNone(reason)
        self.assertEqual(summary[0]["old"], ["1"])
        self.assertEqual(summary[0]["new"], ["2"])

    def test_unknown_case_or_template(self):
        summary, reason = configs.resolve_physics_config("nope", "P00", cases={}, templates={})
        self.assertIsNone(summary)
        self.assertIn("unknown case", reason)


class TestPhysicsLedgerColumns(unittest.TestCase):
    def test_unedited_columns_show_case_native_values(self):
        case = dict(dt=15, physics=dict(mp_physics=8, radt=15, cu_physics=0))
        cols = configs.physics_ledger_columns(case, [])
        self.assertEqual(cols["time_step"], 15)
        self.assertEqual(cols["mp_physics"], 8)
        self.assertEqual(cols["radt"], 15)

    def test_edited_column_shows_new_value_others_stay_native(self):
        case = dict(dt=15, physics=dict(mp_physics=8, radt=15, cu_physics=0))
        summary = [dict(key="radt", op="mul", old=["15"], new=["30"])]
        cols = configs.physics_ledger_columns(case, summary)
        self.assertEqual(cols["radt"], "30")
        self.assertEqual(cols["mp_physics"], 8)  # untouched, still native
        self.assertEqual(cols["time_step"], 15)

    def test_time_step_edit_overrides_time_step_column(self):
        case = dict(dt=72, physics=dict(mp_physics=-1))
        summary = [dict(key="time_step", op="mul", old=["72"], new=["86"])]
        cols = configs.physics_ledger_columns(case, summary)
        self.assertEqual(cols["time_step"], "86")


class TestPhysicsListAndOrder(unittest.TestCase):
    """cmd_physics_list / cmd_physics_order against the REAL cases.json and
    physics_templates.json -- an end-to-end sanity check that every campaign
    case's real namelist.input is readable and every P00-P04 template
    resolves (applicable or a documented skip) without raising."""

    def test_physics_list_every_real_case(self):
        import contextlib
        import io
        for cid in configs.CASE_ORDER:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = configs.cmd_physics_list([cid])
            self.assertEqual(rc, 0, cid)
            out = buf.getvalue()
            self.assertIn("P00", out)
            self.assertIn("P04", out)

    def test_physics_list_json_mode(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_physics_list(["karina", "--json"])
        self.assertEqual(rc, 0)
        applicable = json.loads(buf.getvalue())
        self.assertIsInstance(applicable, list)
        self.assertNotIn("P00", applicable)

    def test_physics_list_unknown_case(self):
        rc = configs.cmd_physics_list(["no_such_case"])
        self.assertEqual(rc, 2)

    def test_physics_order_default_pconfigs_includes_ref_and_twin_first(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_physics_order(["karina", "--reps", "2"])
        self.assertEqual(rc, 0)
        lines = buf.getvalue().strip().split("\n")
        self.assertEqual(lines[0], "P00_ref")
        self.assertEqual(lines[1], "P00_twin")
        # palindrome: last two entries are the same pair, reversed
        self.assertEqual(lines[-1], "P00_ref")
        self.assertEqual(lines[-2], "P00_twin")

    def test_physics_order_explicit_pconfigs(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_physics_order(["karina", "--pconfigs", "P01+P03", "--reps", "1"])
        self.assertEqual(rc, 0)
        lines = buf.getvalue().strip().split("\n")
        self.assertEqual(lines, ["P00_ref", "P00_twin", "P01", "P03"])


class TestPhysicsDescribe(unittest.TestCase):
    def test_describe_applicable_template(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_physics_describe(["karina", "P01"])
        self.assertEqual(rc, 0)
        fields = buf.getvalue().strip("\n").split("\t")
        self.assertEqual(len(fields), 13)
        self.assertEqual(fields[0], "radiation")
        self.assertIn("swint_opt=1", fields[1])

    def test_describe_skip_returns_two_fields(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_physics_describe(["maria1km", "P04"])
        self.assertEqual(rc, 0)
        fields = buf.getvalue().strip("\n").split("\t")
        self.assertEqual(fields[0], "SKIP")

    def test_describe_unknown_case(self):
        rc = configs.cmd_physics_describe(["no_such_case", "P00"])
        self.assertEqual(rc, 2)

    def test_describe_unknown_template(self):
        rc = configs.cmd_physics_describe(["karina", "P99"])
        self.assertEqual(rc, 2)


class TestConfigFileGeneratedConfigs(unittest.TestCase):
    """CONFIG_FILE (WTune D1 decomposition-space work): campaign/configs.py's own side of
    consuming a predictor/decomp_space.py `to_campaign_configs`-shaped JSON file. The JSON here
    is built by hand (not via `import decomp_space`) -- decomp_space.py uses `from __future__
    import annotations` (Python >= 3.7 only, same reason wtune_decomp.py cannot be imported under
    this file's own python3.6-compatibility requirement -- see configs.py's module docstring), so
    this file, which must stay importable under the campaign's python3.6, cannot import it either.
    predictor/tests/test_decomp_space.py owns decomp_space.py's own correctness."""

    SAMPLE_CONFIGS = [
        dict(id="G0001", ranks=448, omp=1, nx=8, ny=56, numtiles=8, dist="cyclic", blocktime="0",
             patch_x=113, patch_y=17, numtiles_x=2, numtiles_y=4, tile_sz_x=0, tile_sz_y=0,
             tile_strategy=3, tile_shape="2d", config_source="generated"),
        dict(id="G0002", ranks=224, omp=2, nx=4, ny=56, numtiles=13, dist="block", blocktime="default",
             patch_x=199, patch_y=17, numtiles_x=0, numtiles_y=0, tile_sz_x=0, tile_sz_y=0,
             tile_strategy=0, tile_shape="1d_y", config_source="generated"),
    ]

    def _write_config_file(self, configs_list=None):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "sample.json")
        with open(path, "w") as f:
            json.dump(configs_list if configs_list is not None else self.SAMPLE_CONFIGS, f)
        return path

    def test_load_generated_configs_bare_list(self):
        path = self._write_config_file()
        loaded = configs.load_generated_configs(path)
        self.assertEqual(loaded, self.SAMPLE_CONFIGS)

    def test_load_generated_configs_configs_key_wrapper(self):
        path = self._write_config_file()
        with open(path, "w") as f:
            json.dump({"configs": self.SAMPLE_CONFIGS, "coverage": {}}, f)
        loaded = configs.load_generated_configs(path)
        self.assertEqual(loaded, self.SAMPLE_CONFIGS)

    def test_cmd_case_generated_tsv_header_and_rows(self):
        import contextlib
        import io
        path = self._write_config_file()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_case_generated("karina", path)
        self.assertEqual(rc, 0)
        lines = buf.getvalue().strip("\n").split("\n")
        header = lines[0].split("\t")
        self.assertEqual(header, [
            "config_id", "ranks", "omp", "nproc_x", "nproc_y", "numtiles", "dist", "blocktime",
            "patch_x", "patch_y", "numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y",
            "tile_strategy", "config_source",
        ])
        row0 = lines[1].split("\t")
        self.assertEqual(row0[0], "G0001")
        self.assertEqual(row0[1:10], ["448", "1", "8", "56", "8", "cyclic", "0", "113", "17"])
        self.assertEqual(row0[10:], ["2", "4", "0", "0", "3", "generated"])
        row1 = lines[2].split("\t")
        self.assertEqual(row1[0], "G0002")
        self.assertEqual(row1[-1], "generated")

    def test_cmd_case_generated_unknown_case(self):
        path = self._write_config_file()
        rc = configs.cmd_case_generated("no_such_case", path)
        self.assertEqual(rc, 2)

    def test_main_dispatches_to_generated_when_config_file_env_set(self):
        """configs.py CASE_ID, run through main(), reads CONFIG_FILE from the environment (as the
        real CLI invocation in campaign/strict_campaign.sbatch does) instead of resolving
        TEMPLATES."""
        import contextlib
        import io
        path = self._write_config_file()
        old = os.environ.get("CONFIG_FILE")
        os.environ["CONFIG_FILE"] = path
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = configs.main(["karina"])
            self.assertEqual(rc, 0)
            self.assertIn("G0001", buf.getvalue())
            self.assertIn("G0002", buf.getvalue())
        finally:
            if old is None:
                os.environ.pop("CONFIG_FILE", None)
            else:
                os.environ["CONFIG_FILE"] = old

    def test_main_without_config_file_env_is_unaffected(self):
        """The same dispatch point, with CONFIG_FILE absent, must still resolve the ordinary
        16 STRICT templates (byte-identical default behaviour)."""
        import contextlib
        import io
        old = os.environ.pop("CONFIG_FILE", None)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = configs.main(["karina"])
            self.assertEqual(rc, 0)
            self.assertIn("C01", buf.getvalue())
            self.assertNotIn("G0001", buf.getvalue())
        finally:
            if old is not None:
                os.environ["CONFIG_FILE"] = old

    def test_cmd_order_generated_preserves_file_order_and_palindrome(self):
        import contextlib
        import io
        path = self._write_config_file()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_order("karina", 2, config_file=path)
        self.assertEqual(rc, 0)
        lines = buf.getvalue().strip("\n").split("\n")
        self.assertEqual(lines, ["G0001", "G0002", "G0002", "G0001"])

    def test_cmd_order_without_config_file_uses_templates(self):
        rc_out = []
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_order("karina", 1, config_file=None)
        self.assertEqual(rc, 0)
        lines = buf.getvalue().strip("\n").split("\n")
        self.assertTrue(all(l.startswith("C") for l in lines), lines)


if __name__ == "__main__":
    unittest.main()
