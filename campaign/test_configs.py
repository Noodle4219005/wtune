#!/usr/bin/env python3
"""Unit tests for campaign/configs.py. Stdlib unittest, Python 3.6 compatible."""
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
        self.assertEqual(set(lafe["namelist_delete_keys"]), {"bl_mynn_tkebudget", "opt_run"})

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


if __name__ == "__main__":
    unittest.main()
