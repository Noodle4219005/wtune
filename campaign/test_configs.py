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


class TestMaxOmp(unittest.TestCase):
    """case["max_omp"] (e.g. c34/marshall in campaign/cases.json: WRF-Fire crashes rc=137/139
    at the first d01 step with any omp>1) must keep resolve_config/resolve_case from ever
    proposing an omp>max_omp template, while an absent key changes nothing."""

    def test_resolve_config_rejects_omp_above_case_cap(self):
        case = dict(configs.CASES["karina"], max_omp=1)
        template = dict(id="C06", ranks=224, omp=2, shape="best", numtiles=12,
                         dist="cyclic", blocktime="0")
        cfg, err = configs.resolve_config(case, template)
        self.assertIsNone(cfg)
        self.assertEqual(err, "omp(2) exceeds case max_omp=1")

    def test_resolve_config_keeps_omp_at_or_below_case_cap(self):
        case = dict(configs.CASES["karina"], max_omp=1)
        template = dict(id="C01", ranks=448, omp=1, shape="best", numtiles=1,
                         dist="cyclic", blocktime="0")
        cfg, err = configs.resolve_config(case, template)
        self.assertIsNone(err)
        self.assertEqual(cfg["omp"], 1)

    def test_resolve_config_absent_max_omp_is_unchanged_behaviour(self):
        # karina itself carries no max_omp key -- omp=2 templates must resolve exactly as
        # before this feature existed.
        case = configs.CASES["karina"]
        self.assertNotIn("max_omp", case)
        template = dict(id="C06", ranks=224, omp=2, shape="best", numtiles=12,
                         dist="cyclic", blocktime="0")
        cfg, err = configs.resolve_config(case, template)
        self.assertIsNone(err)
        self.assertEqual(cfg["omp"], 2)

    def test_resolve_case_with_max_omp_1_keeps_only_omp_1_templates(self):
        # karina resolves all 16 TEMPLATES with no cap (asserted by the no-cap branch above);
        # capping it to max_omp=1 must keep exactly the four omp=1 templates (C01-C04) and
        # skip every omp=2/omp=4 template with the new reason string.
        case = dict(configs.CASES["karina"], max_omp=1)
        kept, skipped, aliases = configs.resolve_case(case)
        self.assertEqual(sorted(c["id"] for c in kept), ["C01", "C02", "C03", "C04"])
        skipped_by_id = dict(skipped)
        for tid in ("C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12"):
            self.assertEqual(skipped_by_id[tid], "omp(2) exceeds case max_omp=1")
        for tid in ("C13", "C14", "C15", "C16"):
            self.assertEqual(skipped_by_id[tid], "omp(4) exceeds case max_omp=1")

    def test_c34_and_marshall_carry_max_omp_1_in_cases_json(self):
        for case_id in ("c34", "marshall"):
            self.assertEqual(configs.CASES[case_id].get("max_omp"), 1, case_id)


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

    def test_run_seconds_is_zeroed(self):
        # decision 0055: a rebuilt-segment template (coldstart real/wrf-window namelist) carries
        # run_seconds=<window>; WRF adds it to run_minutes, so it must be zeroed.
        text = self.SYNTHETIC_WITH_EXISTING.replace("run_seconds=0", "run_seconds=4830")
        out = configs.edit_namelist(text, 14, 16, 12, 8, history_mode="off", ranks=224)
        self.assertIn("run_seconds=0,", out)
        self.assertIn("run_minutes=8", out)
        self.assertNotIn("4830", out)

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
        self.assertIn("&namelist_adios2\n"
                      " adios2_numaggregators=224,\n"
                      " adios2_compression_enable=.true.,\n"
                      ' adios2_blosc_compressor="zstd",\n'
                      "/\n", out)
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

    def test_off_with_nwp_diagnostics_uses_never_elapsing_d01_interval(self):
        # WRF aborts on nwp_diagnostics!=0 with history_interval(1)==0 (smoke job 1078290)
        text = self.NEST_MINUTES.replace("&time_control\n", "&time_control\n nwp_diagnostics = 1,\n", 1)
        self.assertNotEqual(text, self.NEST_MINUTES)
        out = configs.edit_namelist(text, 4, 28, 6, 30, max_dom=2, history_mode="off", ranks=112)
        n = configs.HISTORY_NEVER_MINUTES
        self.assertIn("history_interval = %d, %d," % (n, n), out)
        self.assertIn("history_begin = %d, %d," % (n, n), out)
        self.assertIn("nwp_diagnostics = 1,", out)

    def test_off_with_hailcast_on_nest_only_uses_never_elapsing_interval_every_domain(self):
        # hailcast_opt aborts when ANY domain's history_interval is 0 (debug job 1078305)
        for line in (" hailcast_opt = 1, 1,\n", " hailcast_opt = 0, 1,\n", " afwa_diag_opt = 1,\n"):
            text = self.NEST_MINUTES.replace("&time_control\n", "&time_control\n" + line, 1)
            out = configs.edit_namelist(text, 4, 28, 6, 30, max_dom=2, history_mode="off", ranks=112)
            n = configs.HISTORY_NEVER_MINUTES
            self.assertIn("history_interval = %d, %d," % (n, n), out)
            self.assertIn("history_begin = %d, %d," % (n, n), out)

    def test_off_with_nwp_diagnostics_zero_or_commented_is_unchanged(self):
        for line in (" nwp_diagnostics = 0,\n", "! nwp_diagnostics = 1,\n", " hailcast_opt = 0, 0,\n"):
            text = self.NEST_MINUTES.replace("&time_control\n", "&time_control\n" + line, 1)
            out = configs.edit_namelist(text, 4, 28, 6, 30, max_dom=2, history_mode="off", ranks=112)
            self.assertIn("history_interval = 0, 0,", out)
            self.assertNotIn("history_begin", out)

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
        self.assertIn("&namelist_adios2\n"
                      " adios2_numaggregators=224,\n"
                      " adios2_compression_enable=.true.,\n"
                      ' adios2_blosc_compressor="zstd",\n'
                      "/\n", out)
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


class TestAdios2Compression(unittest.TestCase):
    """The compression keys must be WRITTEN, not inherited from the Registry.

    WRF's ADIOS2 backend treats every compression failure as a non-fatal
    wrf_debug(WARN), so a run that silently fell back to uncompressed BP looks
    exactly like a run that compressed. Recording the settings in the run's own
    namelist is the first half of making that impossible to miss; the second
    half is tools/bp_compression_check.py, which measures the output.
    """

    def _out(self, ranks=224):
        return configs.edit_namelist(TestAdios2.SYNTHETIC_NO_ADIOS2, 14, 16, 12, 8,
                                     ranks=ranks)

    def test_compression_keys_are_written_explicitly(self):
        out = self._out()
        self.assertIn(" adios2_compression_enable=.true.,\n", out)
        self.assertIn(' adios2_blosc_compressor="zstd",\n', out)

    def test_codec_comes_from_the_module_constants(self):
        # The measured choice lives in one place; the emitter must not hardcode it.
        self.assertEqual(configs.ADIOS2_BLOSC_COMPRESSOR, "zstd")
        self.assertEqual(configs.ADIOS2_COMPRESSION_ENABLE, ".true.")
        out = self._out()
        self.assertIn(' adios2_blosc_compressor="%s",' % configs.ADIOS2_BLOSC_COMPRESSOR, out)
        self.assertIn(" adios2_compression_enable=%s," % configs.ADIOS2_COMPRESSION_ENABLE, out)

    def test_each_key_appears_exactly_once(self):
        out = self._out()
        self.assertEqual(out.count("adios2_compression_enable"), 1)
        self.assertEqual(out.count("adios2_blosc_compressor"), 1)
        self.assertEqual(out.count("adios2_numaggregators"), 1)

    def test_numaggregators_behaviour_is_unchanged(self):
        # The compression keys are additive: numaggregators still equals ranks,
        # still leads the block, and is still the first line after the group header.
        for ranks in (112, 224, 448):
            out = configs.edit_namelist(TestAdios2.SYNTHETIC_NO_ADIOS2, 14, 16, 12, 8,
                                        ranks=ranks)
            self.assertIn("&namelist_adios2\n adios2_numaggregators=%d,\n" % ranks, out)

    def test_stale_block_is_replaced_not_appended(self):
        out = configs.edit_namelist(TestAdios2.SYNTHETIC_STALE_ADIOS2, 14, 16, 12, 8,
                                    ranks=224)
        self.assertEqual(out.count("&namelist_adios2"), 1)
        self.assertEqual(out.count("adios2_compression_enable"), 1)
        self.assertEqual(out.count("adios2_blosc_compressor"), 1)

    def test_a_template_that_already_sets_a_different_codec_is_overwritten(self):
        stale = TestAdios2.SYNTHETIC_NO_ADIOS2.rstrip("\n") + (
            "\n&namelist_adios2\n"
            " adios2_numaggregators=7,\n"
            ' adios2_blosc_compressor="lz4",\n'
            " adios2_compression_enable=.false.,\n"
            "/\n")
        out = configs.edit_namelist(stale, 14, 16, 12, 8, ranks=224)
        self.assertNotIn('adios2_blosc_compressor="lz4"', out)
        self.assertNotIn("adios2_compression_enable=.false.", out)
        self.assertNotIn("adios2_numaggregators=7,", out)
        self.assertIn('adios2_blosc_compressor="zstd",', out)

    def test_no_adios2_still_writes_no_compression_keys(self):
        out = configs.edit_namelist(TestAdios2.SYNTHETIC_NO_ADIOS2, 14, 16, 12, 8,
                                    adios2=False)
        self.assertNotIn("&namelist_adios2", out)
        self.assertNotIn("adios2_blosc_compressor", out)
        self.assertNotIn("adios2_compression_enable", out)

    def test_block_is_valid_fortran_namelist_shape(self):
        out = self._out()
        start = out.index("&namelist_adios2")
        block = out[start:out.index("\n/\n", start) + 3]
        lines = block.strip().splitlines()
        self.assertEqual(lines[0], "&namelist_adios2")
        self.assertEqual(lines[-1], "/")
        for line in lines[1:-1]:
            self.assertTrue(line.startswith(" "), line)
            self.assertTrue(line.endswith(","), line)
            self.assertIn("=", line)



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
        sys.path.insert(0, os.path.join(configs.REPO_ROOT, "src"))
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
    # cost levers added 2026-09-16; D07 (time_step_sound=2) added 2026-09-17 as the first
    # template proposed by a forward prediction (predictor/stability.py) rather than after a run
    DYNAMICS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]
    ACCURACY = ["A01", "A02", "A03", "A04", "A05"]         # accuracy levers added 2026-09-17

    def test_loads_and_has_p00_through_p06(self):
        self.assertEqual(set(configs.PHYSICS_TEMPLATES),
                         set(self.ALL + self.DYNAMICS + self.ACCURACY))
        self.assertEqual(configs.PHYSICS_TEMPLATE_ORDER,
                         sorted(self.ALL + self.DYNAMICS + self.ACCURACY))

    def test_dynamics_templates_are_in_the_dynamics_family_and_edit_that_section(self):
        for tid in ("D01", "D02", "D03", "D07"):
            tpl = configs.PHYSICS_TEMPLATES[tid]
            self.assertEqual(tpl["family"], "dynamics")
            for e in tpl["edits"]:
                self.assertEqual(e["section"], "&dynamics")

    def test_d04_d06_families_and_sections(self):
        self.assertEqual(configs.PHYSICS_TEMPLATES["D04"]["family"], "diagnostics")
        self.assertEqual(configs.PHYSICS_TEMPLATES["D05"]["family"], "pbl_surface")
        self.assertEqual(configs.PHYSICS_TEMPLATES["D06"]["family"], "nesting")
        sections = {e["section"] for tid in ("D04", "D05", "D06")
                    for e in configs.PHYSICS_TEMPLATES[tid]["edits"]}
        self.assertEqual(sections, {"&time_control", "&physics", "&bdy_control"})

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
    def test_describe_p00_has_no_empty_fields_so_bash_read_cannot_shift_columns(self):
        # regression: P00 has no edits; an empty TSV field made bash `IFS=$'\t' read` shift
        # every later column (time_step got mp_physics) in campaign ledgers 1077168-1077173
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_physics_describe(["karina", "P00"])
        self.assertEqual(rc, 0)
        fields = buf.getvalue().strip("\n").split("\t")
        self.assertEqual(len(fields), 13)
        self.assertTrue(all(f != "" for f in fields), fields)
        self.assertEqual(fields[1], "-")            # empty edits written as "-"
        self.assertEqual(fields[3], str(configs.CASES["karina"]["dt"]))  # time_step stays in place

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


class TestResolveDecompPair(unittest.TestCase):
    """resolve_decomp_pair (2026-09-17): the single rule set behind MODE=physics's
    DECOMP/TWIN_DECOMP, for both STRICT template ids and a CONFIG_FILE-generated set. The case
    that forced it is sc26p1, which resolves ZERO of C01-C16 (d02 has 100 cells, WRF needs
    floor(e/nproc) >= 10 on every active domain, so both domains cap at 10x10 = 100 ranks while
    every template is anchored at 448/224/112)."""

    GENERATED = [
        dict(id="G0001", ranks=100, omp=1, nx=10, ny=10, numtiles=4, dist="block", blocktime="0",
             patch_x=15, patch_y=15, numtiles_x=2, numtiles_y=2, tile_sz_x=0, tile_sz_y=0,
             tile_strategy=3, tile_shape="2d", config_source="generated"),
        dict(id="G0002", ranks=50, omp=2, nx=5, ny=10, numtiles=8, dist="cyclic", blocktime="0",
             patch_x=30, patch_y=15, numtiles_x=0, numtiles_y=0, tile_sz_x=0, tile_sz_y=0,
             tile_strategy=0, tile_shape="1d_y", config_source="generated"),
        # byte-for-byte the same decomposition as G0001 under a different id: the twin-difference
        # rule must catch this, not just an id equality check.
        dict(id="G0003", ranks=100, omp=1, nx=10, ny=10, numtiles=4, dist="block", blocktime="0",
             patch_x=15, patch_y=15, numtiles_x=2, numtiles_y=2, tile_sz_x=0, tile_sz_y=0,
             tile_strategy=3, tile_shape="2d", config_source="generated"),
        # same as G0001 except for the tile geometry -- different enough to be a legal twin
        dict(id="G0004", ranks=100, omp=1, nx=10, ny=10, numtiles=4, dist="block", blocktime="0",
             patch_x=15, patch_y=15, numtiles_x=4, numtiles_y=1, tile_sz_x=0, tile_sz_y=0,
             tile_strategy=3, tile_shape="2d", config_source="generated"),
    ]

    def _config_file(self, configs_list=None):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "pair.json")
        with open(path, "w") as f:
            json.dump({"case_id": "sc26p1",
                       "configs": self.GENERATED if configs_list is None else configs_list}, f)
        return path

    # ---- STRICT template ids: unchanged behaviour -----------------------------------

    def test_strict_template_pair_resolves_without_a_config_file(self):
        ref, twin, reason = configs.resolve_decomp_pair("karina", "C02", "C06")
        self.assertIsNone(reason)
        self.assertEqual(ref["id"], "C02")
        self.assertEqual(twin["id"], "C06")
        self.assertEqual(ref["ranks"], 448)
        self.assertEqual(twin["ranks"], 224)

    def test_unknown_strict_id_lists_what_is_available(self):
        ref, twin, reason = configs.resolve_decomp_pair("karina", "C99", "C06")
        self.assertIsNone(ref)
        self.assertIsNone(twin)
        self.assertIn("C99", reason)
        self.assertIn("available ids", reason)
        self.assertIn("C02", reason)

    def test_case_resolving_no_strict_template_says_so_and_names_the_way_out(self):
        """sc26p1: the whole reason this function exists. The message must not read as
        'bad id' -- it must say the case resolves none of them, and point at decomp_space."""
        ref, _twin, reason = configs.resolve_decomp_pair("sc26p1", "C02", "C06")
        self.assertIsNone(ref)
        self.assertIn("ZERO", reason)
        self.assertIn("decomp_space.py", reason)
        self.assertIn("CONFIG_FILE", reason)

    def test_unknown_case(self):
        ref, _twin, reason = configs.resolve_decomp_pair("no_such_case", "C02", "C06")
        self.assertIsNone(ref)
        self.assertIn("unknown case", reason)

    # ---- generated config set -------------------------------------------------------

    def test_generated_pair_resolves_for_a_case_with_no_strict_template(self):
        path = self._config_file()
        ref, twin, reason = configs.resolve_decomp_pair("sc26p1", "G0001", "G0002",
                                                        config_file=path)
        self.assertIsNone(reason)
        self.assertEqual((ref["ranks"], ref["nx"], ref["ny"]), (100, 10, 10))
        self.assertEqual((twin["ranks"], twin["nx"], twin["ny"]), (50, 5, 10))
        self.assertEqual(ref["numtiles_x"], 2)

    def test_id_absent_from_the_config_file(self):
        path = self._config_file()
        ref, _twin, reason = configs.resolve_decomp_pair("sc26p1", "G0001", "G9999",
                                                         config_file=path)
        self.assertIsNone(ref)
        self.assertIn("TWIN_DECOMP", reason)
        self.assertIn("G9999", reason)
        self.assertIn(path, reason)

    def test_empty_config_file_is_refused_with_the_expected_shape_named(self):
        path = self._config_file([])
        ref, _twin, reason = configs.resolve_decomp_pair("sc26p1", "G0001", "G0002",
                                                         config_file=path)
        self.assertIsNone(ref)
        self.assertIn("no configs", reason)

    # ---- the twin must be a DIFFERENT decomposition ---------------------------------

    def test_identical_id_is_refused(self):
        path = self._config_file()
        ref, _twin, reason = configs.resolve_decomp_pair("sc26p1", "G0001", "G0001",
                                                         config_file=path)
        self.assertIsNone(ref)
        self.assertIn("SAME decomposition", reason)
        self.assertIn("chaos floor", reason)

    def test_different_id_but_identical_decomposition_is_also_refused(self):
        path = self._config_file()
        ref, _twin, reason = configs.resolve_decomp_pair("sc26p1", "G0001", "G0003",
                                                         config_file=path)
        self.assertIsNone(ref)
        self.assertIn("SAME decomposition", reason)
        self.assertIn("LARGE-DRIFT", reason)

    def test_differing_only_in_tile_geometry_is_accepted(self):
        path = self._config_file()
        ref, twin, reason = configs.resolve_decomp_pair("sc26p1", "G0001", "G0004",
                                                        config_file=path)
        self.assertIsNone(reason)
        self.assertEqual(twin["id"], "G0004")

    def test_identical_strict_template_pair_is_refused_too(self):
        ref, _twin, reason = configs.resolve_decomp_pair("karina", "C02", "C02")
        self.assertIsNone(ref)
        self.assertIn("SAME decomposition", reason)

    # ---- id hygiene: run_id and notes both have to survive the id --------------------

    def test_underscore_in_a_config_id_is_refused(self):
        path = self._config_file([dict(self.GENERATED[0], id="G_0001"), self.GENERATED[1]])
        ref, _twin, reason = configs.resolve_decomp_pair("sc26p1", "G_0001", "G0002",
                                                         config_file=path)
        self.assertIsNone(ref)
        self.assertIn("underscore", reason)
        self.assertIn("run_id", reason)

    def test_whitespace_in_a_config_id_is_refused(self):
        path = self._config_file([dict(self.GENERATED[0], id="G 0001"), self.GENERATED[1]])
        ref, _twin, reason = configs.resolve_decomp_pair("sc26p1", "G 0001", "G0002",
                                                         config_file=path)
        self.assertIsNone(ref)
        self.assertIn("whitespace", reason)

    def test_empty_decomp_id(self):
        ref, _twin, reason = configs.resolve_decomp_pair("karina", "", "C06")
        self.assertIsNone(ref)
        self.assertIn("DECOMP is empty", reason)


class TestDecompPairCLI(unittest.TestCase):
    """configs.py decomp-pair -- the fail-fast gate strict_campaign.sbatch's MODE=physics runs
    before the first WRF srun."""

    def _capture(self, argv, env_config_file=None):
        import contextlib
        import io
        out, err = io.StringIO(), io.StringIO()
        old = os.environ.get("CONFIG_FILE")
        if env_config_file is not None:
            os.environ["CONFIG_FILE"] = env_config_file
        else:
            os.environ.pop("CONFIG_FILE", None)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = configs.main(argv)
        finally:
            if old is None:
                os.environ.pop("CONFIG_FILE", None)
            else:
                os.environ["CONFIG_FILE"] = old
        return rc, out.getvalue(), err.getvalue()

    def test_strict_pair_prints_two_rows_and_rc_zero(self):
        rc, out, err = self._capture(["decomp-pair", "karina", "C02", "C06"])
        self.assertEqual(rc, 0, err)
        lines = out.strip("\n").split("\n")
        self.assertEqual(lines[0].split("\t"), [
            "role", "config_id", "ranks", "omp", "nproc_x", "nproc_y", "numtiles", "dist",
            "blocktime", "numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy",
            "config_source",
        ])
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[1].split("\t")[:3], ["reference", "C02", "448"])
        self.assertEqual(lines[2].split("\t")[:3], ["twin", "C06", "224"])
        # a STRICT template has no tile geometry; it must read as the 0 the sbatch passes
        self.assertEqual(lines[1].split("\t")[9:], ["0", "0", "0", "0", "0", "strict_template"])

    def test_failure_is_rc2_with_one_tagged_stderr_line_and_no_stdout(self):
        rc, out, err = self._capture(["decomp-pair", "sc26p1", "C02", "C06"])
        self.assertEqual(rc, 2)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("DECOMP-PAIR ERROR: "), err)
        self.assertEqual(len(err.strip("\n").split("\n")), 1)

    def test_config_file_env_var_is_honoured(self):
        path = TestResolveDecompPair()._config_file()
        rc, out, err = self._capture(["decomp-pair", "sc26p1", "G0001", "G0002"],
                                     env_config_file=path)
        self.assertEqual(rc, 0, err)
        rows = [l.split("\t") for l in out.strip("\n").split("\n")[1:]]
        self.assertEqual([r[1] for r in rows], ["G0001", "G0002"])
        self.assertEqual(rows[0][9:], ["2", "2", "0", "0", "3", "generated"])

    def test_config_file_flag_overrides_the_env_var(self):
        path = TestResolveDecompPair()._config_file()
        rc, out, err = self._capture(
            ["decomp-pair", "sc26p1", "G0001", "G0002", "--config-file", path])
        self.assertEqual(rc, 0, err)
        self.assertIn("G0001", out)


MMLSQUOTA_SAMPLE = """\
                         Block Limits                                    |     File Limits
Filesystem type             KB      quota      limit   in_doubt    grace |    files   quota    limit in_doubt    grace  Remarks
home1      USR       978611440 1048576000 1153433600    5725280     none |  3750507       0        0      102     none
project    GOV115003  USR           12672          0          0          0     none |        8       0        0        0     none
work1      USR       100004448  104857600  209715200    1997808     none |  1859892       0        0      120     none
work2      USR            none
scratch    USR           12672          0          0          0     none |        8       0        0        0     none
"""


class TestQuotaFreeGib(unittest.TestCase):
    def test_home1_free_gib_matches_known_sample(self):
        # 1048576000 KB quota - 978611440 KB used = 69964560 KB free = 66.72... GiB
        free = configs.quota_free_gib(MMLSQUOTA_SAMPLE, filesystem="home1")
        self.assertIsNotNone(free)
        self.assertAlmostEqual(free, (1048576000 - 978611440) / (1024.0 * 1024.0), places=3)
        self.assertAlmostEqual(free, 66.72, places=1)

    def test_ignores_header_banner_and_noquota_rows(self):
        # the header/banner lines and 'work2 USR none' (no numeric columns) must not
        # be mistaken for a real row -- only 'home1  USR  <int> <int> <int>' matches
        free = configs.quota_free_gib(MMLSQUOTA_SAMPLE, filesystem="work2")
        self.assertIsNone(free)

    def test_work1_near_soft_quota_small_headroom(self):
        # 104857600 KB quota - 100004448 KB used = 4853152 KB ~= 4.63 GiB free -- well
        # under a 60 GiB MIN_FREE_GIB guard threshold, the near-full-quota case the guard
        # exists to catch (a used > quota row would work the same way and go negative,
        # this sample is the observed near-full case from the mmlsquota wiki notes)
        free = configs.quota_free_gib(MMLSQUOTA_SAMPLE, filesystem="work1")
        self.assertAlmostEqual(free, (104857600 - 100004448) / (1024.0 * 1024.0), places=3)
        self.assertLess(free, 60)

    def test_quota_zero_is_no_limit_not_free_gib(self):
        # a quota column of 0 means "not tracked at the user level" (see the fileset-vs-
        # user mmlsquota trap note -- 'project''s real row has this shape too, but with an
        # extra fileset-name column between the filesystem and USR that the parser never
        # matches at all; 'scratch' isolates the quota=0 case on its own) -- must return
        # None, not a huge false-positive free_gib from (0 - used)
        free = configs.quota_free_gib(MMLSQUOTA_SAMPLE, filesystem="scratch")
        self.assertIsNone(free)

    def test_project_row_extra_fileset_column_does_not_match(self):
        # 'project    GOV115003  USR ...' has an extra fileset-name column between the
        # filesystem and USR -- the parser must not mistake GOV115003 for the used-KB
        # column or otherwise misparse this row; it should simply not match
        free = configs.quota_free_gib(MMLSQUOTA_SAMPLE, filesystem="project")
        self.assertIsNone(free)

    def test_missing_filesystem_returns_none(self):
        self.assertIsNone(configs.quota_free_gib(MMLSQUOTA_SAMPLE, filesystem="work3"))

    def test_empty_or_garbled_text_returns_none(self):
        self.assertIsNone(configs.quota_free_gib("", filesystem="home1"))
        self.assertIsNone(configs.quota_free_gib("mmlsquota: command not found\n", filesystem="home1"))


class TestQuotaFreeGibCLI(unittest.TestCase):
    def _run_cli(self, stdin_text, argv=None):
        import contextlib
        import io
        buf_out = io.StringIO()
        old_stdin = sys.stdin
        sys.stdin = io.StringIO(stdin_text)
        try:
            with contextlib.redirect_stdout(buf_out):
                rc = configs._cmd_quota_free_gib(argv or [])
        finally:
            sys.stdin = old_stdin
        return rc, buf_out.getvalue().strip()

    def test_cli_prints_free_gib_for_home1(self):
        rc, out = self._run_cli(MMLSQUOTA_SAMPLE)
        self.assertEqual(rc, 0)
        self.assertAlmostEqual(float(out), 66.72, places=1)

    def test_cli_prints_na_and_nonzero_rc_when_unparseable(self):
        rc, out = self._run_cli("mmlsquota: command not found\n")
        self.assertEqual(rc, 1)
        self.assertEqual(out, "NA")

    def test_cli_respects_filesystem_flag(self):
        rc, out = self._run_cli(MMLSQUOTA_SAMPLE, argv=["--filesystem", "work1"])
        self.assertEqual(rc, 0)
        self.assertLess(float(out), 60)


class TestCasesJsonEnv(unittest.TestCase):
    """CASES_JSON env var swaps the case file configs.py loads at import; unset keeps cases.json."""

    def _import_in_subprocess(self, env_value):
        import subprocess
        env = dict(os.environ)
        env.pop("CASES_JSON", None)
        if env_value is not None:
            env["CASES_JSON"] = env_value
        code = "import json, configs; print(json.dumps([configs.CASES_JSON_PATH, sorted(configs.CASES)]))"
        p = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.abspath(__file__)),
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, check=True)
        return json.loads(p.stdout.strip().splitlines()[-1])

    def test_default_is_cases_json(self):
        path, ids = self._import_in_subprocess(None)
        self.assertEqual(path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.json"))
        self.assertIn("karina", ids)

    def test_env_points_at_fragment(self):
        import tempfile
        frag = {"cases": {"fragcase": dict(configs.CASES["karina"])}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(frag, f)
        try:
            path, ids = self._import_in_subprocess(f.name)
            self.assertEqual(path, f.name)
            self.assertEqual(ids, ["fragcase"])
        finally:
            os.unlink(f.name)


class TestTotalCoresEnv(unittest.TestCase):
    """WTUNE_TOTAL_CORES env var (decision 0050 step 8, scc site) swaps TOTAL_CORES at import;
    unset keeps the f1 default of 448 -- same subprocess-import pattern as TestCasesJsonEnv,
    since TOTAL_CORES is bound once at module import time."""

    def _import_in_subprocess(self, env_value):
        import subprocess
        env = dict(os.environ)
        env.pop("WTUNE_TOTAL_CORES", None)
        if env_value is not None:
            env["WTUNE_TOTAL_CORES"] = env_value
        code = "import configs; print(configs.TOTAL_CORES)"
        p = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.abspath(__file__)),
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, check=True)
        return int(p.stdout.strip().splitlines()[-1])

    def test_default_is_448(self):
        self.assertEqual(self._import_in_subprocess(None), 448)

    def test_env_overrides_total_cores(self):
        self.assertEqual(self._import_in_subprocess("288"), 288)


class TestCaseOverlay(unittest.TestCase):
    """campaign/cases.<site>.json overlays cases.json per-case when SITE != f1 (decision 0050
    step 8); SITE unset/"f1" must leave CASES byte-identical to before SITE existed."""

    def test_f1_default_is_unmodified(self):
        cases = configs.load_cases(site="f1")
        self.assertEqual(cases["conus25"]["prep_dir"],
                         "/home/u5727520/wrf-bench/atlas/v4.4_bench_conus2.5km")

    def test_scc_overlay_replaces_prep_dir_and_bp_dir_only(self):
        cases = configs.load_cases(site="scc")
        c = cases["conus25"]
        self.assertEqual(c["prep_dir"], "/nfs/data/wrf/v4.4_bench_conus2.5km")
        # scc has no ADIOS2 .bp inputs: empty bp_dir => strict_campaign converts prep_dir into
        # its BP cache (stage-bp-inputs, 2026-10-03); netCDF only via the legacy escape
        self.assertEqual(c["bp_dir"], "")
        # geometry untouched by the overlay
        self.assertEqual(c["e_we"], configs.load_cases(site="f1")["conus25"]["e_we"])

    def test_netcdf_input_format_keeps_inputs_at_io_form_2(self):
        nl = "&time_control\n io_form_history = 2,\n/\n&domains\n/\n"
        old = os.environ.get("WTUNE_INPUT_FORMAT")
        try:
            os.environ["WTUNE_INPUT_FORMAT"] = "netcdf"
            out = configs._apply_adios2(nl, 72, 1)
        finally:
            if old is None:
                os.environ.pop("WTUNE_INPUT_FORMAT", None)
            else:
                os.environ["WTUNE_INPUT_FORMAT"] = old
        self.assertRegex(out, r"io_form_input\s*=\s*2")
        self.assertRegex(out, r"io_form_boundary\s*=\s*2")
        self.assertRegex(out, r"io_form_history\s*=\s*14")
        self.assertNotIn("wrfinput_d<domain>.bp", out)

    def test_scc_overlay_expands_home_and_tilde(self):
        cases = configs.load_cases(site="scc")
        self.assertNotIn("$HOME", cases["isc21"]["prep_dir"])
        self.assertTrue(os.path.isabs(cases["isc21"]["prep_dir"]))

    def test_unknown_site_with_no_overlay_file_is_unmodified(self):
        cases = configs.load_cases(site="nosuchsite")
        self.assertEqual(cases["conus25"]["prep_dir"],
                         "/home/u5727520/wrf-bench/atlas/v4.4_bench_conus2.5km")

    def test_module_level_cases_respects_site_env_at_import(self):
        import subprocess
        env = dict(os.environ)
        env["SITE"] = "scc"
        code = "import configs; print(configs.CASES['conus25']['prep_dir'])"
        p = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.abspath(__file__)),
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, check=True)
        self.assertEqual(p.stdout.strip().splitlines()[-1], "/nfs/data/wrf/v4.4_bench_conus2.5km")


class TestSbatchNodeCount(unittest.TestCase):
    """strict_campaign.sbatch derives ranks-per-node and the ledger n_nodes column from the
    allocation (SLURM_JOB_NUM_NODES, default 4) instead of a hard-coded 4, so a -N1 cf112 run of a
    small case launches correctly while the 4-node default stays identical."""

    def _text(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "strict_campaign.sbatch")) as f:
            return f.read()

    def test_no_hardcoded_four_nodes(self):
        s = self._text()
        self.assertNotIn("ranks / 4", s)
        self.assertNotIn('N_NODES="4"', s)
        self.assertEqual(s.count("ntasks_per_node=$(( ranks / ALLOC_NODES ))"), 2)
        self.assertEqual(s.count('N_NODES="$ALLOC_NODES"'), 3)

    def test_default_is_four_and_follows_allocation(self):
        import subprocess
        line = [l for l in self._text().splitlines() if l.startswith("ALLOC_NODES=")]
        self.assertEqual(line, ["ALLOC_NODES=${SLURM_JOB_NUM_NODES:-4}"])
        for env_nodes, want in ((None, "4"), ("4", "4"), ("1", "1")):
            env = dict(os.environ)
            env.pop("SLURM_JOB_NUM_NODES", None)
            if env_nodes is not None:
                env["SLURM_JOB_NUM_NODES"] = env_nodes
            out = subprocess.run(["bash", "-c", line[0] + '; ranks=56; echo "$ALLOC_NODES $(( ranks / ALLOC_NODES ))"'],
                                 env=env, stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.split()
            self.assertEqual(out, [want, str(56 // int(want))])


class TestConfigExtras(unittest.TestCase):
    """physics_namelists / window_frame per-config fields (decision 0059): the
    `configs.py config-extras CASE` TSV strict_campaign.sbatch reads, plus the runner wiring
    markers `wtune tune --pipeline full` greps for (physics_runner_supported)."""

    CFGS = [
        {"id": "ANCHOR", "ranks": 72, "window_frame": True,
         "physics_namelists": {"pt1": "/x/a1.input", "pt2": "/x/a2.input"}},
        {"id": "TWIN", "ranks": 72},
        {"id": "P1", "physics_namelists": {"pt1": "/x/p1.input"}, "window_frame": False},
    ]

    def test_rows_select_this_case_and_default_empty(self):
        rows = configs.config_extras_rows(self.CFGS, "pt1")
        self.assertEqual(rows, [("ANCHOR", "/x/a1.input", "1"), ("TWIN", "", "0"),
                                ("P1", "/x/p1.input", "0")])
        rows2 = configs.config_extras_rows(self.CFGS, "pt2")
        self.assertEqual([r[1] for r in rows2], ["/x/a2.input", "", ""])

    def test_window_frame_levels(self):
        lv = configs.window_frame_level
        self.assertEqual([lv(v) for v in (None, False, 0, "0", "")], [0] * 5)
        self.assertEqual([lv(v) for v in (True, 1, "end", "1")], [1] * 4)
        self.assertEqual([lv(v) for v in (2, "2", "mid_end")], [2] * 3)

    def test_cli_prints_tsv_and_needs_config_file(self):
        import contextlib
        import io
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "c.json")
        with open(path, "w") as f:
            json.dump({"configs": self.CFGS}, f)
        old = os.environ.get("CONFIG_FILE")
        os.environ["CONFIG_FILE"] = path
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = configs.main(["config-extras", "pt1"])
            self.assertEqual(rc, 0)
            self.assertEqual(buf.getvalue().splitlines(),
                             ["ANCHOR\t/x/a1.input\t1", "TWIN\t\t0", "P1\t/x/p1.input\t0"])
            del os.environ["CONFIG_FILE"]
            self.assertEqual(configs.main(["config-extras", "pt1"]), 2)
        finally:
            if old is None:
                os.environ.pop("CONFIG_FILE", None)
            else:
                os.environ["CONFIG_FILE"] = old

    def test_sbatch_wires_the_fields(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "strict_campaign.sbatch")) as f:
            text = f.read()
        self.assertIn("physics_namelists", text)       # wtune_tune.PHYSICS_RUNNER_MARKER
        self.assertIn("config-extras", text)
        self.assertIn('--history-mode "$RUN_HISTORY_ARG"', text)
        self.assertNotIn('--history-mode "$HISTORY_MODE_ARG" --ranks "$ranks"', text)
        self.assertIn("window_frame=", text)
        self.assertIn("window_frame_mid=", text)
        self.assertIn('rm -rf "$d"/wrfrst_d0*_*', text)

    def test_mid_end_mode_writes_mid_and_end_frames(self):
        text = ("&time_control\n run_minutes = 0,\n history_interval = 180,\n"
                " history_begin = 7,\n/\n&domains\n max_dom = 1,\n/\n")
        out = configs.edit_namelist(text, 8, 9, 1, 31, max_dom=1, history_mode="mid_end_d01",
                                    ranks=72)
        self.assertRegex(out, r"history_begin\s*=\s*15")      # frames at 15 and 15+16 = 31
        self.assertRegex(out, r"history_interval\s*=\s*16")
        with self.assertRaises(ValueError):
            configs.edit_namelist(text, 8, 9, 1, 1, max_dom=1, history_mode="mid_end_d01",
                                  ranks=72)

    def test_end_d01_writes_one_frame_at_window_end_for_a_restart_window(self):
        text = ("&time_control\n run_minutes = 0,\n history_interval = 180,\n"
                " history_begin = 0,\n start_year = 2016,\n start_month = 10,\n"
                " start_day = 6,\n start_hour = 0,\n/\n&domains\n max_dom = 1,\n/\n")
        out = configs.edit_namelist(text, 8, 9, 1, 30, max_dom=1, history_mode="end_d01",
                                    ranks=72)
        self.assertRegex(out, r"history_interval\s*=\s*30")
        self.assertRegex(out, r"history_begin\s*=\s*30")
        self.assertRegex(out, r"nproc_x\s*=\s*8")


class TestPointFused(unittest.TestCase):
    """POINT_FUSED / ANCHOR_END / WINDOW_EXACT (decisions 0055/0059): resolve_point_plan,
    post_spinup_steps, edit_namelist run_seconds, and the sbatch wiring (bash -n + static guards)."""

    LINE = " Timing for main: time 2019-11-26_%s on domain   %d:    %.5f elapsed seconds\n"

    def _rsl(self, n, dt=10, rad=None, nested=True):
        out = []
        for k in range(1, n + 1):
            t = k * dt
            stamp = "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)
            v = 5.0 if rad and k in rad else 0.05
            out.append(self.LINE % (stamp, 1, v))
            if nested:
                out.append(self.LINE % (stamp, 2, 0.01))
        return "".join(out)

    def test_default_order_unchanged(self):
        o = ["ANCHOR", "C1", "C2", "ANCHOREND"]
        r = configs.resolve_point_plan(o)
        self.assertEqual(r["order"], o)
        self.assertFalse(r["fused"])
        self.assertTrue(r["anchor_end"])

    def test_anchor_end_off_omits_run(self):
        r = configs.resolve_point_plan(["ANCHOR", "C1", "C2", "ANCHOREND"], anchor_end=False)
        self.assertEqual(r["order"], ["ANCHOR", "C1", "C2"])
        self.assertFalse(r["anchor_end"])
        r = configs.resolve_point_plan(["ANCHOR", "C1", "ANCHOREND"], anchor_end=True)
        self.assertIn("ANCHOREND", r["order"])

    def test_fused_single_config_drops_anchors_and_defaults_drift_gate_off(self):
        r = configs.resolve_point_plan(["ANCHOR", "W", "ANCHOREND"], point_fused=True)
        self.assertTrue(r["fused"])
        self.assertEqual(r["order"], ["W"])
        self.assertEqual(r["cid"], "W")
        self.assertFalse(r["anchor_end"])

    def test_fused_explicit_anchor_end_is_ignored_with_note(self):
        r = configs.resolve_point_plan(["ANCHOR", "W", "ANCHOREND"], True, anchor_end=True)
        self.assertTrue(r["fused"])
        self.assertFalse(r["anchor_end"])
        self.assertIn("ANCHOR_END=1", r["note"])

    def test_fused_refused_for_multi_config_or_reps_or_window_frame(self):
        for order in (["ANCHOR", "A", "B", "ANCHOREND"], ["W", "W"]):
            r = configs.resolve_point_plan(order, point_fused=True)
            self.assertFalse(r["fused"], order)
            self.assertEqual(r["order"], order)
            self.assertIn("POINT_FUSED ignored", r["note"])
        r = configs.resolve_point_plan(["ANCHOR", "W"], point_fused=True, window_frame=True)
        self.assertFalse(r["fused"])

    def test_fused_with_only_anchor_uses_it(self):
        r = configs.resolve_point_plan(["ANCHOR"], point_fused=True)
        self.assertTrue(r["fused"])
        self.assertEqual(r["order"], ["ANCHOR"])

    def test_fused_selects_post_spinup_steps_only(self):
        # dt=10, 60 s spin-up => steps 7.. count; 20 steps total -> 14 selected, d01 only
        text = self._rsl(20)
        vals = configs.post_spinup_steps(text, 60, 10)
        self.assertEqual(len(vals), 14)
        self.assertTrue(all(v == 0.05 for v in vals))   # nested 0.01 lines never leak in

    def test_fused_excludes_spinup_radiation_and_step_stats_excludes_window_radiation(self):
        text = self._rsl(20, rad={1, 7, 13})   # step 1 (spin-up), 7 (first window), 13 (in window)
        vals = configs.post_spinup_steps(text, 60, 10)
        self.assertEqual(vals[0], 5.0)           # step 7 is the first post-spin-up step
        st = configs.step_stats(vals)
        self.assertEqual(st["n_all"], len(vals) - 1)
        self.assertAlmostEqual(st["median"], 0.05)
        self.assertEqual(st["n_excluded"], 2)    # dropped first + radiation step 13

    def test_fused_spinup_covering_everything_gives_no_steps(self):
        self.assertEqual(configs.post_spinup_steps(self._rsl(6), 60, 10), [])

    def test_fused_steps_cli(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as f:
            f.write(self._rsl(10))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = configs.main(["fused-steps", f.name, "--spin-seconds", "60", "--dt", "10"])
        os.unlink(f.name)
        self.assertEqual(rc, 0)
        self.assertEqual(len(buf.getvalue().split()), 4)

    def test_point_plan_cli_emits_shell_assignments(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            configs.main(["point-plan", "--fused", "1", "--anchor-end", "", "ANCHOR", "W", "ANCHOREND"])
        out = buf.getvalue()
        self.assertIn("POINT_ORDER=('W')", out)
        self.assertIn("POINT_FUSED_ON=1", out)
        self.assertIn("POINT_ANCHOR_END=0", out)

    NL = ("&time_control\n run_days = 0,\n run_hours = 0,\n run_minutes = 5,\n"
          " run_seconds = 0,\n/\n&domains\n max_dom = 1,\n/\n")

    def test_run_seconds_default_unchanged_and_exact_tail(self):
        base = configs.edit_namelist(self.NL, 4, 4, 1, 11, max_dom=1, adios2=False)
        same = configs.edit_namelist(self.NL, 4, 4, 1, 11, max_dom=1, adios2=False, run_seconds=None)
        self.assertEqual(base, same)
        self.assertRegex(base, r"run_seconds\s*=\s*0")
        t = configs.edit_namelist(self.NL, 4, 4, 1, 11, max_dom=1, adios2=False, run_seconds=40)
        self.assertRegex(t, r"run_seconds\s*=\s*40")
        self.assertRegex(t, r"run_minutes\s*=\s*11")

    def test_sbatch_syntax_and_defaults_wiring(self):
        import subprocess
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strict_campaign.sbatch")
        p = subprocess.run(["bash", "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(p.returncode, 0, p.stderr)
        text = open(path).read()
        # fused path skips the spin-up block; default (FUSED_ON=0) keeps it
        self.assertIn('[ "$FUSED_ON" != "1" ]', text)
        self.assertIn("FUSED_ON=0\n", text)
        self.assertIn('POINT_FUSED:-0', text)
        self.assertIn('--anchor-end "${ANCHOR_END:-}"', text)


if __name__ == "__main__":
    unittest.main()


class TestDynamicsTemplates(unittest.TestCase):
    """D01-D03 (&dynamics cost levers, 2026-09-16). time_step_sound/moist_adv_opt are absent from
    every campaign namelist, so they are inserted -- and must be inserted per active domain."""

    NL = ("&domains\n time_step=15, max_dom=2,\n/\n"
          "&physics\n mp_physics = 8, 8,\n/\n"
          "&dynamics\n diff_opt=1, km_opt=4,\n/\n")

    def _tpl(self, tid):
        return configs.PHYSICS_TEMPLATES[tid]

    def test_d01_inserts_time_step_sound_for_every_active_domain(self):
        out, summary = configs.apply_physics_template(self.NL, self._tpl("D01"), max_dom=2)
        self.assertIn("time_step_sound = 3, 3,", out)
        self.assertIn("&dynamics", out.split("time_step_sound")[0][-200:])
        self.assertEqual(summary[0]["new"], ["3", "3"])

    def test_d01_single_domain_inserts_one_value(self):
        out, _ = configs.apply_physics_template(self.NL, self._tpl("D01"), max_dom=1)
        self.assertIn("time_step_sound = 3,", out)
        self.assertNotIn("time_step_sound = 3, 3,", out)

    def test_d02_rk_ord_is_scalar_not_per_domain(self):
        out, _ = configs.apply_physics_template(self.NL, self._tpl("D02"), max_dom=2)
        self.assertIn("rk_ord = 2,", out)
        self.assertNotIn("rk_ord = 2, 2,", out)

    def test_d03_sets_both_advection_limiters(self):
        out, summary = configs.apply_physics_template(self.NL, self._tpl("D03"), max_dom=2)
        self.assertIn("moist_adv_opt = 0, 0,", out)
        self.assertIn("scalar_adv_opt = 0, 0,", out)
        self.assertEqual([e["key"] for e in summary], ["moist_adv_opt", "scalar_adv_opt"])

    def test_dynamics_templates_do_not_touch_physics_section(self):
        for tid in ("D01", "D02", "D03"):
            out, _ = configs.apply_physics_template(self.NL, self._tpl(tid), max_dom=2)
            self.assertIn(" mp_physics = 8, 8,", out)
            self.assertIn(" diff_opt=1, km_opt=4,", out)


class TestDiagnosticsAndCadenceTemplates(unittest.TestCase):
    """D04-D06 (2026-09-16): diagnostics off, PBL call interval, one-way nesting."""

    NL = ("&time_control\n nwp_diagnostics = 1,\n/\n"
          "&domains\n time_step=15, max_dom=2,\n/\n"
          "&physics\n mp_physics = 8, 8,\n hailcast_opt = 1, 1,\n do_radar_ref = 1,\n/\n"
          "&bdy_control\n spec_bdy_width = 5,\n feedback = 1,\n/\n")

    def test_d04_zeroes_all_three_diagnostics(self):
        out, summary = configs.apply_physics_template(
            self.NL, configs.PHYSICS_TEMPLATES["D04"], max_dom=2)
        self.assertIn("nwp_diagnostics = 0,", out)
        self.assertIn("hailcast_opt = 0, 0,", out)
        self.assertIn("do_radar_ref = 0,", out)
        self.assertEqual([e["key"] for e in summary],
                         ["nwp_diagnostics", "hailcast_opt", "do_radar_ref"])

    def test_d04_only_applies_to_the_two_cases_that_enable_them(self):
        tpl = configs.PHYSICS_TEMPLATES["D04"]
        self.assertEqual(set(tpl["applies_to"]), {"bigchill", "elevconv"})
        ok, _why = configs.physics_template_applicability("bigchill", tpl)
        self.assertTrue(ok)
        ok, why = configs.physics_template_applicability("karina", tpl)
        self.assertFalse(ok)
        self.assertIn("applies_to", why)

    def test_d05_inserts_bldt_for_every_active_domain(self):
        out, _ = configs.apply_physics_template(
            self.NL, configs.PHYSICS_TEMPLATES["D05"], max_dom=2)
        self.assertIn("bldt = 10, 10,", out)
        self.assertIn("&physics", out.split("bldt")[0][-300:])

    def test_d06_sets_feedback_zero_in_bdy_control(self):
        out, summary = configs.apply_physics_template(
            self.NL, configs.PHYSICS_TEMPLATES["D06"], max_dom=2)
        self.assertIn("feedback = 0,", out)
        self.assertEqual(summary[0]["old"], ["1"])
        self.assertIn(" spec_bdy_width = 5,", out)

    def test_d06_is_nested_cases_only(self):
        tpl = configs.PHYSICS_TEMPLATES["D06"]
        self.assertNotIn("karina", tpl["applies_to"])
        self.assertIn("zhengzhou", tpl["applies_to"])


class TestSoundStepCostTemplate(unittest.TestCase):
    """D07 (2026-09-17): time_step_sound=2, the halved acoustic sub-step count that
    predictor/stability.py predicted FORWARD for sc26p1 (d01 acoustic CFL 0.094, d02 0.283 at
    WRF's auto count of 4; 2 puts them at 0.189/0.567, both inside CFL_SAFE=0.60)."""

    NL = ("&domains\n time_step=10, max_dom=2,\n/\n"
          "&physics\n mp_physics = 4, 4,\n/\n"
          "&dynamics\n diff_opt=1, km_opt=4,\n/\n")

    def test_inserts_two_for_every_active_domain(self):
        out, summary = configs.apply_physics_template(
            self.NL, configs.PHYSICS_TEMPLATES["D07"], max_dom=2)
        self.assertIn("time_step_sound = 2, 2,", out)
        self.assertIn("&dynamics", out.split("time_step_sound")[0][-200:])
        self.assertEqual(summary[0]["new"], ["2", "2"])

    def test_value_is_even_and_below_d01s_odd_three(self):
        # WRF's auto path returns even counts only, and the one odd value measured (D01's 3)
        # hung conus12 before step 1 in both replicates (job 1080698) -- so the cost direction
        # is taken 4 -> 2, never 4 -> 3.
        edits = configs.PHYSICS_TEMPLATES["D07"]["edits"]
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["key"], "time_step_sound")
        self.assertEqual(edits[0]["value"], 2)
        self.assertEqual(edits[0]["value"] % 2, 0)
        self.assertTrue(edits[0]["per_domain"])

    def test_is_a_cost_lever_not_an_accuracy_one(self):
        # --objective accuracy must never mistake a COARSER acoustic integration for a
        # converged reference; A02 (which raises the count) is the accuracy mirror.
        self.assertEqual(configs.PHYSICS_TEMPLATES["D07"]["direction"], "cost")
        self.assertEqual(configs.PHYSICS_TEMPLATES["A02"]["direction"], "accuracy")

    def test_applies_only_to_the_case_with_the_measured_margin(self):
        tpl = configs.PHYSICS_TEMPLATES["D07"]
        self.assertEqual(tpl["applies_to"], ["sc26p1"])
        ok, _why = configs.physics_template_applicability("sc26p1", tpl)
        self.assertTrue(ok)
        # conus12 would land at acoustic CFL 340*(72/2)/12000 = 1.02, past the classical limit
        ok, why = configs.physics_template_applicability("conus12", tpl)
        self.assertFalse(ok)
        self.assertIn("applies_to", why)

    def test_description_cites_the_source_and_the_measured_numbers(self):
        d = configs.PHYSICS_TEMPLATES["D07"]["description"]
        self.assertIn("dyn_em/solve_em.F:451-468", d)
        self.assertIn("0.094", d)
        self.assertIn("0.283", d)
        self.assertIn("1080698", d)

    def test_the_value_two_is_the_halved_auto_count_for_this_case(self):
        """The 2 is not a magic number. Recomputed here in plain arithmetic rather than by
        importing predictor/stability.py, which this file cannot do (it uses `from __future__
        import annotations`, Python >= 3.7, and test_configs.py must stay importable under the
        campaign's python3.6 -- same reason it cannot import decomp_space.py). The stability
        module's own side of this is pinned in predictor/tests/test_stability.py."""
        case = configs.CASES["sc26p1"]
        dt = float(case["dt"])                       # 10 s, and parent_time_step_ratio_list is
        self.assertEqual(case["parent_time_step_ratio_list"], [1, 1])   # [1,1], so d02 shares it
        for dx, expect_cfl in zip(case["dx_list"], (0.094, 0.283)):
            auto = max(2 * (int(300.0 * dt / dx - 0.01) + 1), 4)        # solve_em.F:462
            self.assertEqual(auto, 4)
            self.assertAlmostEqual(340.0 * (dt / auto) / dx, expect_cfl, places=3)
            # the template's value doubles that CFL and must stay inside the advisory band
            self.assertLess(340.0 * (dt / 2) / dx, 0.60)
        self.assertEqual(configs.PHYSICS_TEMPLATES["D07"]["edits"][0]["value"], 4 // 2)


class TestAccuracyTemplates(unittest.TestCase):
    """A01-A05 (accuracy levers, 2026-09-17): the mirrors of the P/D cost levers. Every A-template
    that edits a max_domains-shaped key must be per_domain, because time_step_sound /
    moist_adv_opt / scalar_adv_opt / h_*_adv_order are absent from every campaign namelist and are
    therefore INSERTED -- a non-per_domain insert would write d01 only and leave every nest at the
    Registry default. time_step is the exception: it is declared scalar in Registry.EM_COMMON:2262
    ('namelist,domains 1'), so A01 must NOT be per_domain."""

    NL = ("&domains\n time_step=15, max_dom=2,\n/\n"
          "&physics\n mp_physics = 8, 8,\n radt = 15, 15,\n/\n"
          "&dynamics\n diff_opt=1, km_opt=4,\n/\n")

    def _tpl(self, tid):
        return configs.PHYSICS_TEMPLATES[tid]

    def test_every_accuracy_template_is_marked_accuracy_direction(self):
        for tid in ("A01", "A02", "A03", "A04", "A05"):
            self.assertEqual(self._tpl(tid).get("direction"), "accuracy", tid)

    def test_no_cost_template_claims_the_accuracy_direction(self):
        for tid, tpl in configs.PHYSICS_TEMPLATES.items():
            if not tid.startswith("A"):
                self.assertNotEqual(tpl.get("direction"), "accuracy", tid)

    def test_max_domains_shaped_accuracy_edits_are_per_domain(self):
        for tid in ("A02", "A03", "A04", "A05"):
            for e in self._tpl(tid)["edits"]:
                self.assertTrue(e.get("per_domain"), "%s/%s" % (tid, e["key"]))

    def test_a01_shrinks_time_step_and_is_not_per_domain(self):
        edits = self._tpl("A01")["edits"]
        self.assertEqual(len(edits), 1)
        self.assertEqual((edits[0]["key"], edits[0]["op"], edits[0]["factor"]),
                         ("time_step", "mul", 0.8))
        self.assertEqual(edits[0].get("round"), "int")
        self.assertEqual(edits[0]["section"], "&domains")
        self.assertFalse(edits[0].get("per_domain"))
        out, summary = configs.apply_physics_template(self.NL, self._tpl("A01"), max_dom=2)
        self.assertIn("time_step=12,", out)
        self.assertEqual(summary[0]["new"], ["12"])

    def test_a01_is_the_mirror_of_p03(self):
        # P03 multiplies the same key upward; together they bracket the case's native dt.
        self.assertEqual(configs.PHYSICS_TEMPLATES["P03"]["edits"][0]["key"], "time_step")
        self.assertGreater(configs.PHYSICS_TEMPLATES["P03"]["edits"][0]["factor"], 1.0)
        self.assertLess(self._tpl("A01")["edits"][0]["factor"], 1.0)

    def test_a01_min_clamp_keeps_a_one_second_step_legal(self):
        nl = self.NL.replace("time_step=15,", "time_step=1,")
        out, _ = configs.apply_physics_template(nl, self._tpl("A01"), max_dom=2)
        self.assertIn("time_step=1,", out)
        self.assertNotIn("time_step=0,", out)

    def test_a02_inserts_an_even_time_step_sound_for_every_active_domain(self):
        edit = self._tpl("A02")["edits"][0]
        self.assertEqual(edit["value"], 6)
        # dyn_em/solve_em.F:615 halves this with INTEGER division on RK substep 2, and WRF's own
        # auto formula (solve_em.F:451-468) returns even numbers only -- odd values truncate.
        self.assertEqual(edit["value"] % 2, 0)
        self.assertGreater(edit["value"], configs.PHYSICS_TEMPLATES["D01"]["edits"][0]["value"])
        out, summary = configs.apply_physics_template(self.NL, self._tpl("A02"), max_dom=2)
        self.assertIn("time_step_sound = 6, 6,", out)
        self.assertIn("&dynamics", out.split("time_step_sound")[0][-200:])
        self.assertEqual(summary[0]["new"], ["6", "6"])

    def test_a03_sets_monotonic_transport_on_every_active_domain(self):
        out, summary = configs.apply_physics_template(self.NL, self._tpl("A03"), max_dom=2)
        self.assertIn("moist_adv_opt = 2, 2,", out)
        self.assertIn("scalar_adv_opt = 2, 2,", out)
        self.assertEqual([e["key"] for e in summary], ["moist_adv_opt", "scalar_adv_opt"])
        # D03 turns the same limiter off; A03 turns it up. Registry default is 1.
        self.assertEqual([e["value"] for e in configs.PHYSICS_TEMPLATES["D03"]["edits"]], [0, 0])

    def test_a03_skips_isc21_which_is_already_monotonic(self):
        ok, why = configs.physics_template_applicability("isc21", self._tpl("A03"))
        self.assertFalse(ok)
        self.assertIn("moist_adv_opt=2", why)

    def test_a04_sets_sixth_order_horizontal_advection_on_every_active_domain(self):
        out, summary = configs.apply_physics_template(self.NL, self._tpl("A04"), max_dom=2)
        self.assertIn("h_mom_adv_order = 6, 6,", out)
        self.assertIn("h_sca_adv_order = 6, 6,", out)
        self.assertEqual([e["key"] for e in summary], ["h_mom_adv_order", "h_sca_adv_order"])

    def test_a04_order_is_even_and_at_the_halo_ceiling(self):
        # module_advect_em.F implements 6/5/4/3/2 (and 0); dyn_em/solve_em.F:414 aborts above 6
        # because the communication halos only exist to order 6.
        for e in self._tpl("A04")["edits"]:
            self.assertEqual(e["value"], 6, e["key"])

    def test_a04_skips_isc21_because_monotonic_transport_is_order_5_only(self):
        ok, why = configs.physics_template_applicability("isc21", self._tpl("A04"))
        self.assertFalse(ok)
        self.assertIn("advect_scalar_mono", why)

    def test_a03_and_a04_incompatibility_is_recorded_in_excluded(self):
        import json as _json
        with open(configs.PHYSICS_TEMPLATES_JSON_PATH) as f:
            excluded = _json.load(f)["excluded"]
        key = next(k for k in excluded if k.startswith("A03 + A04"))
        self.assertIn("9643", excluded[key])
        self.assertIn("10016", excluded[key])

    def test_a05_halves_radt_without_rounding_or_a_min_clamp(self):
        edit = self._tpl("A05")["edits"][0]
        self.assertEqual((edit["key"], edit["op"], edit["factor"]), ("radt", "mul", 0.5))
        # A min clamp of 1 would silently make lafe/marshall (native radt=1) no-ops, and radt is
        # REAL in Registry.EM_COMMON:2473, so no integer rounding is needed either.
        self.assertIsNone(edit.get("round"))
        self.assertIsNone(edit.get("min"))
        out, summary = configs.apply_physics_template(self.NL, self._tpl("A05"), max_dom=2)
        self.assertIn("radt = 7.5, 7.5,", out)
        self.assertEqual(summary[0]["new"], ["7.5", "7.5"])

    def test_a05_halving_a_one_minute_radt_is_not_clamped_away(self):
        nl = self.NL.replace("radt = 15, 15,", "radt = 1, 1,")
        out, _ = configs.apply_physics_template(nl, self._tpl("A05"), max_dom=2)
        self.assertIn("radt = 0.5, 0.5,", out)

    def test_a05_does_not_pair_swint_opt_the_way_p02_does(self):
        self.assertEqual([e["key"] for e in self._tpl("A05")["edits"]], ["radt"])
        self.assertIn("swint_opt", [e["key"] for e in configs.PHYSICS_TEMPLATES["P02"]["edits"]])

    def test_accuracy_templates_do_not_touch_unrelated_sections(self):
        for tid in ("A01", "A02", "A03", "A04"):
            out, _ = configs.apply_physics_template(self.NL, self._tpl(tid), max_dom=2)
            self.assertIn(" mp_physics = 8, 8,", out)
            self.assertIn(" diff_opt=1, km_opt=4,", out)

    def test_every_accuracy_template_has_a_family_and_a_real_description(self):
        for tid in ("A01", "A02", "A03", "A04", "A05"):
            tpl = self._tpl(tid)
            self.assertIn(tpl["family"], ("dynamics", "radiation"), tid)
            # Same bar as P00-P06/D01-D06: the description must cite the source it was read from.
            self.assertGreater(len(tpl["description"]), 400, tid)
            self.assertIn(".F:", tpl["description"], tid)


class TestFidelityNotesTokens(unittest.TestCase):
    """configs.fidelity_notes_tokens(): the shell-side extraction strict_campaign.sbatch
    MODE=physics reads into FID_LABEL/FID_ND/FID_FLOOR/FID_DECIDING, which then become the
    `fidelity_nd_max=` / `fidelity_floor_ratio_max=` tokens in the ledger `notes` column. It used
    to be an inline heredoc (twice) and so could never be tested without running the sbatch."""

    PAYLOAD = {
        "label": "SMALL-DRIFT",
        "deciding": "nd(QICE)=0.032 > 0.03 and floor_ratio(QICE)=1.593 > 2",
        "per_field_summary_at_last_time": {
            "T": {"nd": 0.012, "floor_ratio": 1.2, "floor_zero": False},
            "QICE": {"nd": 0.032, "floor_ratio": 1.593, "floor_zero": False},
            "U": {"nd": 0.004, "floor_ratio": 0.8, "floor_zero": False},
        },
    }

    def test_reports_the_maxima_over_fields_not_the_thresholds(self):
        label, nd, floor, deciding = configs.fidelity_notes_tokens(self.PAYLOAD)
        self.assertEqual(label, "SMALL-DRIFT")
        self.assertEqual(nd, "0.032")
        self.assertEqual(floor, "1.593")
        self.assertNotIn(" ", deciding)
        self.assertIn("nd(QICE)=0.032", deciding)

    def test_every_field_is_a_single_shell_word(self):
        # `read -r A B C D` must see exactly four words, or FID_DECIDING swallows the rest.
        line = " ".join(configs.fidelity_notes_tokens(self.PAYLOAD))
        self.assertEqual(len(line.split()), 4)

    def test_tokens_become_one_notes_token_each(self):
        _label, nd, floor, _deciding = configs.fidelity_notes_tokens(self.PAYLOAD)
        notes = "mode=physics template=A01 fidelity_nd_max=%s fidelity_floor_ratio_max=%s" % (nd, floor)
        parsed = dict(t.split("=", 1) for t in notes.split())
        self.assertEqual(float(parsed["fidelity_nd_max"]), 0.032)
        self.assertEqual(float(parsed["fidelity_floor_ratio_max"]), 1.593)

    def test_missing_pieces_fall_back_to_na_never_to_a_crash(self):
        self.assertEqual(configs.fidelity_notes_tokens({}), ("NA", "NA", "NA", "NA"))
        self.assertEqual(
            configs.fidelity_notes_tokens({"label": "INVALID", "deciding": None,
                                            "per_field_summary_at_last_time": None}),
            ("INVALID", "NA", "NA", "NA"))

    def test_floor_zero_fields_contribute_no_floor_ratio(self):
        # fidelity_gate.py leaves floor_ratio null when the twin was bit-identical to the
        # reference; that field must not be counted, and must not crash the max().
        payload = {"label": "WITHIN-FLOOR", "deciding": "all key fields within floor",
                   "per_field_summary_at_last_time": {
                       "T": {"nd": 0.001, "floor_ratio": None, "floor_zero": True},
                       "U": {"nd": 0.002, "floor_ratio": 0.5, "floor_zero": False}}}
        _l, nd, floor, _d = configs.fidelity_notes_tokens(payload)
        self.assertEqual((nd, floor), ("0.002", "0.5"))

    def test_booleans_are_not_mistaken_for_numbers(self):
        # isinstance(True, int) is True in Python; a bool must never become a fidelity number.
        payload = {"per_field_summary_at_last_time": {"T": {"nd": True, "floor_ratio": False}}}
        _l, nd, floor, _d = configs.fidelity_notes_tokens(payload)
        self.assertEqual((nd, floor), ("NA", "NA"))

    def test_cli_prints_one_line_and_refuses_a_missing_file(self):
        import io as _io
        import contextlib
        import json as _json
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            _json.dump(self.PAYLOAD, fh)
            path = fh.name
        try:
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = configs.main(["fidelity-tokens", path])
            self.assertEqual(rc, 0)
            self.assertEqual(buf.getvalue().strip().split(), list(configs.fidelity_notes_tokens(self.PAYLOAD)))
        finally:
            os.unlink(path)
        buf, err = _io.StringIO(), _io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = configs.main(["fidelity-tokens", path + ".gone"])
        # Nothing printed -> the sbatch's `read` leaves FID_* empty -> the same notes shape as a
        # MODE=physics run that produced no fidelity JSON at all.
        self.assertEqual(rc, 2)
        self.assertEqual(buf.getvalue(), "")


class TestSbatchFidelityExtraction(unittest.TestCase):
    """strict_campaign.sbatch MODE=physics must call the tested helper instead of carrying two
    byte-identical inline python heredocs, and must still emit both numeric notes tokens."""

    def _text(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "strict_campaign.sbatch")) as f:
            return f.read()

    def test_calls_configs_fidelity_tokens_in_both_arms(self):
        s = self._text()
        self.assertEqual(s.count('configs.py" fidelity-tokens'), 2)
        self.assertNotIn("per_field_summary_at_last_time", s)

    def test_both_numeric_tokens_still_reach_the_notes_column(self):
        s = self._text()
        for tok in ("fidelity_nd_max=$FID_ND", "fidelity_floor_ratio_max=$FID_FLOOR"):
            self.assertIn(tok, s)
        # The SKIPPED-row notes template keeps the same tokens with empty values, which is the
        # "no fidelity JSON" shape predictor/ledger.py must read back as None.
        self.assertIn("fidelity_nd_max= fidelity_floor_ratio_max=", s)


class TestHailcastNestHistoryInterval(unittest.TestCase):
    """hailcast_opt aborts at init on a domain whose history_interval is 0 ("HAILCAST Error : No
    'history_interval' defined in namelist"), which killed the bigchill/elevconv reference runs
    (jobs 1081089/1081090) because nest history is off by default. The nest must get a
    never-elapsing interval instead of 0 -- still writes nothing."""

    NEST = ("&time_control\n history_interval = 60, 60,\n/\n"
            "&domains\n time_step=15, max_dom=2,\n/\n"
            "&physics\n mp_physics = 8, 8,\n hailcast_opt = 1, 1,\n/\n")
    NEST_NO_DIAG = NEST.replace(" hailcast_opt = 1, 1,\n", "")

    def test_interval_mode_gives_the_nest_a_never_elapsing_interval_when_hailcast_is_on(self):
        out = configs.edit_namelist(self.NEST, 4, 28, 6, 30, max_dom=2, history_mode="interval",
                                     history_minutes=10, ranks=112)
        n = configs.HISTORY_NEVER_MINUTES
        self.assertIn("history_interval = 10, %d," % n, out)
        self.assertNotIn("history_interval = 10, 0,", out)

    def test_nest_still_gets_zero_without_those_diagnostics(self):
        out = configs.edit_namelist(self.NEST_NO_DIAG, 4, 28, 6, 30, max_dom=2,
                                     history_mode="interval", history_minutes=10, ranks=112)
        self.assertIn("history_interval = 10, 0,", out)

    def test_end_d01_mode_too(self):
        out = configs.edit_namelist(self.NEST, 4, 28, 6, 30, max_dom=2, history_mode="end_d01",
                                     ranks=112)
        self.assertNotIn("history_interval = 30, 0,", out)


class TestValuesByDomain(unittest.TestCase):
    """apply_physics_edit's 'set' op with a per-domain value list (2026-09-17). The scalar 'value'
    form is unchanged and still the default; 'values_by_domain' exists because a nest does not
    always want its parent's value (SC26-P1's d01 carries 3x its d02's acoustic margin at the same
    dt -- predictor/stability.py)."""

    PRESENT = ("&dynamics\n time_step_sound = 4, 4,\n/\n")
    ABSENT = ("&domains\n max_dom = 2,\n/\n&dynamics\n diff_opt = 1, 1,\n/\n")

    def test_scalar_value_is_still_the_default_and_unchanged(self):
        out, old, new = configs.apply_physics_edit(
            self.PRESENT, {"op": "set", "key": "time_step_sound", "value": 6,
                           "section": "&dynamics"}, 2)
        self.assertEqual((old, new), (["4", "4"], ["6", "6"]))
        self.assertIn("time_step_sound = 6, 6,", out)

    def test_list_form_writes_one_value_per_domain(self):
        out, old, new = configs.apply_physics_edit(
            self.PRESENT, {"op": "set", "key": "time_step_sound",
                           "values_by_domain": [2, 4], "section": "&dynamics"}, 2)
        self.assertEqual((old, new), (["4", "4"], ["2", "4"]))
        self.assertIn("time_step_sound = 2, 4,", out)

    def test_mapping_form_accepts_json_string_keys_and_is_one_based(self):
        out, _old, new = configs.apply_physics_edit(
            self.PRESENT, {"op": "set", "key": "time_step_sound",
                           "values_by_domain": {"1": 2, "2": 4}, "section": "&dynamics"}, 2)
        self.assertEqual(new, ["2", "4"])
        self.assertIn("time_step_sound = 2, 4,", out)
        out2, _o2, new2 = configs.apply_physics_edit(
            self.PRESENT, {"op": "set", "key": "time_step_sound",
                           "values_by_domain": {1: 2, 2: 4}, "section": "&dynamics"}, 2)
        self.assertEqual(new2, ["2", "4"])
        self.assertEqual(out, out2)

    def test_length_mismatch_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            configs.apply_physics_edit(
                self.PRESENT, {"op": "set", "key": "time_step_sound",
                               "values_by_domain": [2, 4, 6], "section": "&dynamics"}, 2)
        self.assertIn("3 value(s)", str(cm.exception))
        self.assertIn("time_step_sound", str(cm.exception))

    def test_missing_or_out_of_range_domain_index_is_refused(self):
        with self.assertRaises(ValueError) as missing:
            configs.apply_physics_edit(
                self.PRESENT, {"op": "set", "key": "time_step_sound",
                               "values_by_domain": {1: 2}, "section": "&dynamics"}, 2)
        self.assertIn("missing domain(s) [2]", str(missing.exception))
        with self.assertRaises(ValueError) as outside:
            configs.apply_physics_edit(
                self.PRESENT, {"op": "set", "key": "time_step_sound",
                               "values_by_domain": {1: 2, 2: 4, 3: 6}, "section": "&dynamics"}, 2)
        self.assertIn("[3]", str(outside.exception))

    def test_absent_key_is_inserted_per_domain_without_an_explicit_per_domain_flag(self):
        # values_by_domain is per-domain by construction, so it implies per_domain on insert.
        out, old, new = configs.apply_physics_edit(
            self.ABSENT, {"op": "set", "key": "time_step_sound",
                          "values_by_domain": [2, 4], "section": "&dynamics"}, 2)
        self.assertEqual((old, new), ([], ["2", "4"]))
        self.assertIn("time_step_sound = 2, 4,", out)

    def test_a02_scalar_path_still_works_for_a_single_domain_case(self):
        out, _summary = configs.apply_physics_template(
            self.ABSENT, configs.PHYSICS_TEMPLATES["A02"], max_dom=1)
        self.assertIn("time_step_sound = 6,", out)
        self.assertNotIn("time_step_sound = 6, 6,", out)


class TestFrozenKeys(unittest.TestCase):
    """cases.json 'frozen_keys' (2026-09-17): keys the TASK mandates (SC26-P1 fixes time_step=10
    and debug_level=50). A template that edits one is refused outright -- that is a rules
    violation, not a tuning trade-off. Tested against a SYNTHETIC case; no cases.json entry
    declares frozen keys today."""

    NL = ("&domains\n time_step = 10, max_dom = 1,\n/\n"
          "&physics\n radt = 9,\n/\n")
    TPL = {"family": "dynamics",
           "edits": [{"op": "mul", "key": "time_step", "factor": 1.2, "round": "int",
                      "section": "&domains"}]}
    RADT = {"family": "radiation",
            "edits": [{"op": "mul", "key": "radt", "factor": 2, "round": "int",
                       "section": "&physics"}]}

    def test_case_frozen_keys_lowercases_and_defaults_to_empty(self):
        self.assertEqual(configs.case_frozen_keys({"frozen_keys": ["Time_Step", " debug_level "]}),
                         ["time_step", "debug_level"])
        self.assertEqual(configs.case_frozen_keys({}), [])
        self.assertEqual(configs.case_frozen_keys(None), [])

    def test_template_frozen_key_conflicts_sees_edits_and_per_case_extra_edits(self):
        self.assertEqual(configs.template_frozen_key_conflicts(self.TPL, ["time_step"]),
                         ["time_step"])
        self.assertEqual(configs.template_frozen_key_conflicts(self.TPL, ["radt"]), [])
        paired = dict(self.RADT, extra_edits={"synth": [{"op": "set", "key": "debug_level",
                                                          "value": 0, "section": "&time_control"}]})
        self.assertEqual(configs.template_frozen_key_conflicts(paired, ["debug_level"],
                                                                case_id="synth"), ["debug_level"])
        # ...and only for the named case.
        self.assertEqual(configs.template_frozen_key_conflicts(paired, ["debug_level"],
                                                                case_id="other"), [])

    def test_apply_physics_template_refuses_a_frozen_key_and_names_key_and_case(self):
        with self.assertRaises(ValueError) as cm:
            configs.apply_physics_template(self.NL, self.TPL, 1, case_id="sc26p1",
                                            frozen_keys=["time_step"])
        msg = str(cm.exception)
        self.assertIn("time_step", msg)
        self.assertIn("sc26p1", msg)
        self.assertIn("frozen", msg)

    def test_a_frozen_key_refuses_before_any_edit_is_applied(self):
        two = {"family": "dynamics",
               "edits": [{"op": "mul", "key": "radt", "factor": 2, "round": "int",
                          "section": "&physics"},
                         {"op": "mul", "key": "time_step", "factor": 1.2, "round": "int",
                          "section": "&domains"}]}
        with self.assertRaises(ValueError):
            configs.apply_physics_template(self.NL, two, 1, case_id="sc26p1",
                                            frozen_keys=["time_step"])
        # nothing partially applied: the unfrozen edit in the same template did not run either
        ok, _summary = configs.apply_physics_template(self.NL, two, 1, case_id="sc26p1")
        self.assertIn("radt = 18,", ok)

    def test_a_template_that_touches_nothing_frozen_still_applies(self):
        out, summary = configs.apply_physics_template(self.NL, self.RADT, 1, case_id="sc26p1",
                                                       frozen_keys=["time_step", "debug_level"])
        self.assertIn("radt = 18,", out)
        self.assertEqual([e["key"] for e in summary], ["radt"])

    def test_applicability_reports_the_frozen_key(self):
        ok, reason = configs.physics_template_applicability("sc26p1", self.TPL,
                                                             frozen_keys=["time_step"])
        self.assertFalse(ok)
        self.assertIn("time_step", reason)
        self.assertIn("sc26p1", reason)
        self.assertEqual(configs.physics_template_applicability("sc26p1", self.TPL),
                         (True, None))

    def test_resolve_physics_config_refuses_a_frozen_key_for_a_synthetic_case(self):
        import tempfile
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "namelist.input")
        with open(path, "w") as fh:
            fh.write(self.NL)
        cases = {"sc26p1": {"namelist_template": path, "max_dom": 1,
                            "frozen_keys": ["time_step", "debug_level"]}}
        templates = {"X": self.TPL, "R": self.RADT}
        summary, reason = configs.resolve_physics_config("sc26p1", "X", cases=cases,
                                                          templates=templates)
        self.assertIsNone(summary)
        self.assertIn("time_step", reason)
        ok_summary, ok_reason = configs.resolve_physics_config("sc26p1", "R", cases=cases,
                                                               templates=templates)
        self.assertIsNone(ok_reason)
        self.assertEqual([e["key"] for e in ok_summary], ["radt"])


class TestRestartFrom(unittest.TestCase):
    """edit_namelist's restart_from/restart_io_form/write_restart_interval (WTune probe
    multi-point redesign, 2026-09-26): a RANKING-PROBE run starting from an existing WRF restart
    file instead of from t0 -- see edit_namelist's own docstring and wtune_probe_plan.py."""

    SYNTHETIC_3DOM_RESTART = (
        "&time_control\n"
        " run_days = 0, run_hours = 6, run_minutes = 0, run_seconds = 0,\n"
        " start_year = 2019, 2019, 2019,\n"
        " start_month = 11, 11, 11,\n"
        " start_day = 25, 25, 25,\n"
        " start_hour = 12, 12, 12,\n"
        " start_minute = 00, 00, 00,\n"
        " restart = .false.,\n"
        " restart_interval = 1440,\n"
        " io_form_restart = 2,\n"
        "/\n"
        "&domains\n"
        " max_dom = 3,\n"
        " time_step=15, e_we=793, 805, 1001, e_sn=853, 805, 1001,\n"
        "/\n"
    )

    def test_restart_from_sets_start_for_every_domain_and_restart_true(self):
        import datetime
        t = datetime.datetime(2019, 11, 26, 23, 0, 0)
        out = configs.edit_namelist(self.SYNTHETIC_3DOM_RESTART, 16, 28, 8, 50, max_dom=3,
                                     adios2=False, restart_from=t)
        self.assertIn("start_year = 2019, 2019, 2019,", out)   # unchanged: same year
        self.assertIn("start_month = 11, 11, 11,", out)
        self.assertIn("start_day = 26, 26, 26,", out)
        self.assertIn("start_hour = 23, 23, 23,", out)
        self.assertIn("start_minute = 0, 0, 0,", out)
        self.assertIn("start_second = 0, 0, 0,", out)
        self.assertIn("restart = .true.,", out)
        self.assertNotIn("restart = .false.,", out)
        # run length = steps*dt handled by the caller via run_minutes (50 here); this function
        # only overrides start_*/restart, so run_minutes=50 still comes through untouched
        # (spacing preserved from the template's own "key = value" style -- see _set_scalar).
        self.assertIn("run_minutes = 50", out)

    def test_restart_from_overrides_restart_timers_stored_in_the_restart_file(self):
        import re
        # G0 (jobs 1371/1372): without it WRF reuses the spin-up's restart interval from the
        # restart file and writes a restart inside the candidate's measured window.
        import datetime
        t = datetime.datetime(2019, 11, 26, 23, 0, 0)
        out = configs.edit_namelist(self.SYNTHETIC_3DOM_RESTART, 16, 28, 8, 50, max_dom=3,
                                     adios2=False, restart_from=t)
        self.assertEqual(len(re.findall(r"(?m)^\s*override_restart_timers\s*=\s*\.true\.", out)), 1)
        again = configs.edit_namelist(out, 16, 28, 8, 50, max_dom=3, adios2=False, restart_from=t)
        self.assertEqual(len(re.findall(r"(?m)^\s*override_restart_timers\s*=", again)), 1)
        plain = configs.edit_namelist(self.SYNTHETIC_3DOM_RESTART, 16, 28, 8, 50, max_dom=3,
                                       adios2=False)
        self.assertNotIn("override_restart_timers", plain)

    def test_restart_from_none_is_byte_identical_to_before(self):
        out_a = configs.edit_namelist(self.SYNTHETIC_3DOM_RESTART, 16, 28, 8, 50, max_dom=3,
                                       adios2=False)
        out_b = configs.edit_namelist(self.SYNTHETIC_3DOM_RESTART, 16, 28, 8, 50, max_dom=3,
                                       adios2=False, restart_from=None, restart_io_form=None,
                                       write_restart_interval=None)
        self.assertEqual(out_a, out_b)
        self.assertIn("restart = .false.,", out_a)   # untouched: no restart_from given

    def test_restart_io_form_overrides_after_adios2(self):
        import datetime
        t = datetime.datetime(2019, 11, 26, 23, 0, 0)
        out = configs.edit_namelist(self.SYNTHETIC_3DOM_RESTART, 16, 28, 8, 50, max_dom=3,
                                     adios2=True, ranks=448, restart_from=t, restart_io_form=2)
        # ADIOS2 forces io_form_restart=14 first; restart_io_form=2 must win (netCDF, so the
        # restart file is decomposition-independent -- see edit_namelist's docstring). Spacing
        # preserved from the template's own "key = value" style (the line already existed).
        self.assertIn("io_form_restart = 2,", out)
        self.assertNotIn("io_form_restart = 14", out)
        # history/input/boundary stay on ADIOS2 -- only restart's OWN format is overridden
        self.assertIn("io_form_history=14,", out)

    def test_write_restart_interval_sets_interval_without_forcing_restart_true(self):
        out = configs.edit_namelist(self.SYNTHETIC_3DOM_RESTART, 16, 28, 8, 360, max_dom=3,
                                     adios2=False, write_restart_interval=60,
                                     restart_io_form=2)
        self.assertIn("restart_interval = 60", out)
        self.assertNotIn("restart_interval = 1440", out)
        self.assertIn("restart = .false.,", out)   # spin-up writer still starts at t0
        self.assertIn("io_form_restart = 2", out)

    def test_write_restart_interval_survives_adios2_default(self):
        # adios2=True forces restart_interval=1000000 (see _apply_adios2's docstring) -- without
        # write_restart_interval winning afterward, the spin-up run could never actually write a
        # restart at t0+spinup.
        out = configs.edit_namelist(self.SYNTHETIC_3DOM_RESTART, 16, 28, 8, 60, max_dom=3,
                                     adios2=True, ranks=448, write_restart_interval=60)
        self.assertIn("restart_interval = 60,", out)
        self.assertNotIn("restart_interval = 1000000", out)


class TestRestartTime(unittest.TestCase):
    """case_start_datetime/cmd_restart_time -- the `configs.py restart-time CASE MINUTES` CLI
    campaign/strict_campaign.sbatch's RANKING_START=spinup block calls once per job to compute
    the --restart-from value every candidate's second edit-namelist pass uses (2026-09-26,
    end-to-end wiring)."""

    NL_TEXT = (
        "&time_control\n"
        " start_year = 2019, 2019, 2019,\n"
        " start_month = 11, 11, 11,\n"
        " start_day = 25, 25, 25,\n"
        " start_hour = 12, 12, 12,\n"
        "/\n"
    )

    def test_case_start_datetime_defaults_minute_second_to_zero(self):
        import datetime
        self.assertEqual(configs.case_start_datetime(self.NL_TEXT),
                          datetime.datetime(2019, 11, 25, 12, 0, 0))

    def test_case_start_datetime_reads_explicit_minute_second(self):
        import datetime
        text = self.NL_TEXT.replace(
            " start_hour = 12, 12, 12,\n",
            " start_hour = 12, 12, 12,\n start_minute = 30, 30, 30,\n"
            " start_second = 15, 15, 15,\n")
        self.assertEqual(configs.case_start_datetime(text),
                          datetime.datetime(2019, 11, 25, 12, 30, 15))

    def test_cmd_restart_time_prints_t0_plus_spinup(self):
        import tempfile
        tmpdir = tempfile.mkdtemp()
        nl_path = os.path.join(tmpdir, "namelist.input")
        with open(nl_path, "w") as f:
            f.write(self.NL_TEXT)
        configs.CASES["_test_restart_time_case"] = {"prep_dir": tmpdir}
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = configs.main(["restart-time", "_test_restart_time_case", "90"])
            self.assertEqual(rc, 0)
            self.assertEqual(buf.getvalue().strip(), "2019-11-25_13:30:00")
        finally:
            del configs.CASES["_test_restart_time_case"]

    def test_cmd_restart_time_unknown_case_is_an_error(self):
        rc = configs.main(["restart-time", "_no_such_case_at_all", "60"])
        self.assertEqual(rc, 2)

    def test_cmd_restart_time_wrong_arg_count_is_an_error(self):
        rc = configs.main(["restart-time", "onlyone"])
        self.assertEqual(rc, 2)



class TestNamelistFileStaging(unittest.TestCase):
    """Files the namelist names are staged into run dirs (climate-ea50, scc probe jobs 2541-2545:
    iofields_d01.txt never staged -> WRF 4.8 hung at input_input until the 1260 s timeout)."""

    NML = """&time_control
 run_hours = 6,
 iofields_filename = "iofields_d01.txt", 'io/fields_d02.txt',  ignore_iofields_warning = .true.,
 auxinput4_inname = "wrflowinp_d<domain>", auxinput4_interval = 360, 360,
 auxinput1_inname = "met_em.d<domain>.<date>",
 auxinput11_inname = 'obs_d<domain>.txt',
 auxinput5_inname = 'wrfchemi_d<domain>_<date>',
 ! iofields_filename = 'commented.txt',
/
&domains
 max_dom = 2,
/
&physics
 sst_update = 1,
/
&fdda
 grid_fdda = 1, 0,
 obs_nudge_opt = 0, 1,
/
"""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "prep")
        self.run_dir = os.path.join(self.tmp, "run")
        os.makedirs(os.path.join(self.src, "io"))
        os.makedirs(self.run_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def _touch(self, *names, **kw):
        base = kw.get("base", self.src)
        for n in names:
            with open(os.path.join(base, n), "w") as f:
                f.write("x")

    def test_refs_resolve_domain_placeholder_and_requiredness(self):
        refs = configs.namelist_file_refs(self.NML)
        req = sorted(n for k, n, r in refs if r)
        self.assertEqual(req, ["io/fields_d02.txt", "iofields_d01.txt", "wrffdda_d01",
                               "wrflowinp_d01", "wrflowinp_d02"])
        opt = sorted(n for k, n, r in refs if not r)
        self.assertEqual(opt, ["OBS_DOMAIN2*", "obs_d01.txt", "obs_d02.txt", "tslist",
                               "wrfchemi_d01_<date>", "wrfchemi_d02_<date>"])
        self.assertFalse(any(k == "auxinput1_inname" for k, _n, _r in refs))   # real.exe only

    def test_refs_none_specified_and_sst_off(self):
        refs = configs.namelist_file_refs(
            "&time_control\n iofields_filename = 'NONE_SPECIFIED',\n/\n&domains\n max_dom=1,\n/\n")
        self.assertEqual(refs, [("tslist", "tslist", False)])

    def test_stage_links_present_files_and_reports_missing(self):
        self._touch("iofields_d01.txt", "io/fields_d02.txt", "wrflowinp_d01", "wrflowinp_d02",
                    "OBS_DOMAIN201", "OBS_DOMAIN202", "tslist",
                    "wrfchemi_d01_2020-01-01_00:00:00")
        self._touch("wrflowinp_d02", base=self.run_dir)           # already staged: kept as is
        res = configs.stage_namelist_files(self.NML, self.run_dir, [self.src])
        for name in ("iofields_d01.txt", "io/fields_d02.txt", "wrflowinp_d01", "OBS_DOMAIN201",
                     "OBS_DOMAIN202", "tslist", "wrfchemi_d01_2020-01-01_00:00:00"):
            p = os.path.join(self.run_dir, name)
            self.assertTrue(os.path.islink(p), name)
            self.assertEqual(os.readlink(p), os.path.join(self.src, name))
        self.assertFalse(os.path.islink(os.path.join(self.run_dir, "wrflowinp_d02")))
        self.assertIn(("auxinput4_inname", "wrflowinp_d02"), res["present"])
        self.assertEqual(res["missing"], [("gfdda_inname", "wrffdda_d01")])
        lines = configs.format_namelist_stage(res, [self.src])
        self.assertTrue(any(ln.startswith("WARNING") and "wrffdda_d01" in ln for ln in lines))

    def test_stage_second_search_dir_and_iofields_warning(self):
        other = os.path.join(self.tmp, "orig")
        os.makedirs(other)
        self._touch("iofields_d01.txt", base=other)
        nml = '&time_control\n iofields_filename = "iofields_d01.txt", "gone_d02.txt",\n/\n' \
              "&domains\n max_dom = 2,\n/\n"
        res = configs.stage_namelist_files(nml, self.run_dir, [self.src, other, "", "/nonexistent"])
        self.assertEqual(os.readlink(os.path.join(self.run_dir, "iofields_d01.txt")),
                         os.path.join(other, "iofields_d01.txt"))
        self.assertEqual(res["missing"], [("iofields_filename", "gone_d02.txt")])
        warn = configs.format_namelist_stage(res, [self.src, other])[-1]
        self.assertIn("hangs at input_input", warn)
        self.assertFalse(os.path.lexists(os.path.join(self.run_dir, "gone_d02.txt")))

    def test_cli_links_and_exits_zero_with_missing(self):
        import subprocess
        self._touch("iofields_d01.txt")
        nml = os.path.join(self.run_dir, "namelist.input")
        with open(nml, "w") as f:
            f.write('&time_control\n iofields_filename = "iofields_d01.txt", "x_d02.txt",\n/\n'
                    "&domains\n max_dom = 2,\n/\n")
        p = subprocess.run([sys.executable, os.path.join(configs.CAMPAIGN_DIR, "configs.py"),
                            "link-namelist-files", nml, self.run_dir, self.src],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("linked iofields_d01.txt", p.stdout)
        self.assertIn("WARNING", p.stderr)
        self.assertTrue(os.path.islink(os.path.join(self.run_dir, "iofields_d01.txt")))


class TestStrictCampaignStagesNamelistFiles(unittest.TestCase):
    SBATCH = os.path.join(configs.CAMPAIGN_DIR, "strict_campaign.sbatch")

    def test_every_wrf_exe_link_is_preceded_by_link_namelist_files(self):
        with open(self.SBATCH) as f:
            lines = f.read().splitlines()
        idx = [i for i, ln in enumerate(lines) if 'ln -sf "$W/main/wrf.exe" ./wrf.exe' in ln]
        self.assertEqual(len(idx), 3)
        for i in idx:
            self.assertEqual(lines[i - 1].strip(), "link_namelist_files", lines[i])

    def test_case_loader_emits_namelist_ref_dirs_incl_rebuilt_origin(self):
        import re as _re
        import subprocess
        import tempfile
        with open(self.SBATCH) as f:
            text = f.read()
        m = _re.search(r"eval \"\$\(python3 - \"\$CASE\" <<'PY'\n(.*?)\nPY\n", text, _re.S)
        self.assertIsNotNone(m)
        tmp = tempfile.mkdtemp()
        base = dict(prep_dir="/p/orig", bp_dir="", dt=60, steps=10, case_family="x", e_we=10,
                    e_sn=10, e_vert=10, dx=1, dy=1, input_files=[], physics={})
        rebuilt = dict(base, prep_dir="/p/win_0/real", namelist_template="/p/plan/nml.input",
                       rebuilt_from="orig")
        cases = os.path.join(tmp, "cases.json")
        with open(cases, "w") as f:
            json.dump({"cases": {"orig": base, "orig__rb0": rebuilt}}, f)
        env = dict(os.environ, CASES_JSON=cases, CAMPAIGN=configs.CAMPAIGN_DIR, SITE="f1")
        out = subprocess.run([sys.executable, "-", "orig__rb0"], input=m.group(1), env=env,
                             stdout=subprocess.PIPE, universal_newlines=True).stdout
        line = [ln for ln in out.splitlines() if ln.startswith("NAMELIST_REF_DIRS=")]
        self.assertEqual(line, ["NAMELIST_REF_DIRS='/p/win_0/real /p/plan /p/orig'"])



class TestAdios2InputStaging(unittest.TestCase):
    """2026-10-03 user decision: ADIOS2 for every WRF input stream. adios2_input_streams /
    _apply_adios2's aux switch / stage-bp-inputs (conversion through a FAKE src/wtune_bp found via
    WTUNE_SRC, lock-serialized) / strict_campaign.sbatch's staging block (extracted, run with a stub
    interpreter) / nc2bp.sbatch."""

    HERE = os.path.dirname(os.path.abspath(__file__))
    NL = (" &time_control\n io_form_history = 2,\n io_form_input = 2,\n io_form_boundary = 2,\n"
          " auxinput4_inname = \"wrflowinp_d<domain>\",\n io_form_auxinput4 = 2,\n /\n"
          " &domains\n max_dom = 2,\n /\n &physics\n sst_update = 1,\n /\n"
          " &fdda\n grid_fdda = 1, 0,\n gfdda_inname = \"wrffdda_d<domain>\",\n"
          " io_form_gfdda = 2,\n /\n")
    FAKE_BP = (
        "import importlib.util, os, time\n"
        "_spec = importlib.util.spec_from_file_location('wtune_bp_real', os.environ['FAKE_BP_REAL'])\n"
        "_real = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_real)\n"
        "# the real per-record helpers (stdlib only); conversion is faked\n"
        "PER_RECORD_NAME_RE = _real.PER_RECORD_NAME_RE\n"
        "is_per_record_dir, record_files = _real.is_per_record_dir, _real.record_files\n"
        "bp_input_links = _real.bp_input_links\n"
        "per_record_namelist_edits = _real.per_record_namelist_edits\n"
        "ensure_dir, open_lock = _real.ensure_dir, _real.open_lock\n"
        "def _mk_bp(p):\n"
        "    os.makedirs(p)\n"
        "    for f in ('md.idx', 'data.0'):\n"
        "        open(os.path.join(p, f), 'w').close()\n"
        "def require_adios2_python():\n"
        "    if os.environ.get('FAKE_BP_FAIL'):\n"
        "        raise RuntimeError('no Python that can import adios2')\n"
        "    return '/fake/python'\n"
        "def convert_inputs(case_dir, cache_dir, names=None, engine='BP5'):\n"
        "    os.makedirs(cache_dir, exist_ok=True)\n"
        "    out = {}\n"
        "    for n in names:\n"
        "        dst = os.path.join(cache_dir, n)\n"
        "        if not os.path.isdir(dst):          # the manifest's 'current' check\n"
        "            time.sleep(float(os.environ.get('FAKE_BP_SLEEP', '0')))\n"
        "            with open(os.environ['FAKE_BP_LOG'], 'a') as f:\n"
        "                f.write('convert %s %s\\n' % (n, engine))\n"
        "            if PER_RECORD_NAME_RE.match(n):\n"
        "                for t in os.environ.get('FAKE_BP_RECORDS', '2020-01-01_00:00:00 '\n"
        "                                        '2020-01-01_06:00:00').split():\n"
        "                    _mk_bp(os.path.join(dst, '%s_%s.bp' % (n, t)))\n"
        "            else:\n"
        "                _mk_bp(dst)\n"
        "        out[n] = dst\n"
        "    return out\n")

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="wtune-bp-test-")
        self.src = os.path.join(self.tmp, "src")
        os.makedirs(self.src)
        with open(os.path.join(self.src, "wtune_bp.py"), "w") as f:
            f.write(self.FAKE_BP)
        self.prep = os.path.join(self.tmp, "prep")
        os.makedirs(self.prep)
        for n in ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02"):
            with open(os.path.join(self.prep, n), "w") as f:
                f.write("CDF")
        self.nl = os.path.join(self.prep, "namelist.input")
        with open(self.nl, "w") as f:
            f.write(self.NL)
        self.log = os.path.join(self.tmp, "convert.log")
        self.env = dict(os.environ, WTUNE_SRC=self.src, FAKE_BP_LOG=self.log,
                        WTUNE_BP_CACHE=os.path.join(self.tmp, "cache"),
                        FAKE_BP_REAL=os.path.join(os.path.dirname(self.HERE), "src",
                                                  "wtune_bp.py"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stage(self, *args, env=None):
        import subprocess
        return subprocess.Popen([sys.executable, os.path.join(self.HERE, "configs.py"),
                                 "stage-bp-inputs", "sc26p1"] + list(args),
                                env=env or self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)

    @staticmethod
    def _assignments(stdout):
        import shlex
        return dict(shlex.split(l)[0].split("=", 1) if shlex.split(l) else (l.rstrip("="), "")
                    for l in stdout.splitlines() if "=" in l)

    def _converted(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as f:
            return f.read().split("\n")[:-1]

    def test_streams_and_bp_innames(self):
        sts = {st["stream"]: st for st in configs.adios2_input_streams(self.NL)}
        self.assertEqual(list(sts), ["input", "boundary", "auxinput4", "gfdda"])
        self.assertEqual(sts["input"]["names"], ["wrfinput_d01", "wrfinput_d02"])
        self.assertEqual(sts["input"]["required"], ["wrfinput_d01"])
        self.assertEqual(sts["auxinput4"]["inname"], "wrflowinp_d<domain>.bp")
        self.assertEqual(sts["gfdda"]["names"], ["wrffdda_d01"])
        self.assertEqual(configs.adios2_stream_edits(sts["gfdda"]),
                         [("io_form_gfdda", 14, "fdda"),
                          ("gfdda_inname", '"wrffdda_d<domain>.bp"', "fdda")])
        # an inname already carrying .bp is not doubled
        nl = self.NL.replace('"wrflowinp_d<domain>"', '"wrflowinp_d<domain>.bp"')
        aux = [s for s in configs.adios2_input_streams(nl) if s["stream"] == "auxinput4"][0]
        self.assertEqual((aux["names"][0], aux["inname"]),
                         ("wrflowinp_d01", "wrflowinp_d<domain>.bp"))

    def _apply(self, aux_env):
        old = {k: os.environ.get(k) for k in ("WTUNE_INPUT_FORMAT", "WTUNE_AUX_BP_STREAMS")}
        try:
            os.environ.pop("WTUNE_INPUT_FORMAT", None)
            if aux_env is None:
                os.environ.pop("WTUNE_AUX_BP_STREAMS", None)
            else:
                os.environ["WTUNE_AUX_BP_STREAMS"] = aux_env
            return configs._apply_adios2(self.NL, 72, 2)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_apply_adios2_switches_only_the_aux_streams_available_as_bp(self):
        out = self._apply("auxinput4")
        for key in ("io_form_input", "io_form_boundary", "io_form_history", "io_form_restart",
                    "io_form_auxinput4"):
            self.assertRegex(out, r"(?m)^\s*%s\s*=\s*14\s*,\s*$" % key)
        self.assertIn('auxinput4_inname = "wrflowinp_d<domain>.bp"', out)
        self.assertRegex(out, r"(?m)^\s*io_form_gfdda\s*=\s*2,")          # not in the list
        self.assertIn('gfdda_inname = "wrffdda_d<domain>"', out)
        both = self._apply("auxinput4,gfdda")
        fdda = both[both.index("&fdda"):]
        self.assertRegex(fdda.split("/")[0], r"io_form_gfdda\s*=\s*14")
        self.assertIn('gfdda_inname = "wrffdda_d<domain>.bp"', both)
        none = self._apply(None)
        self.assertRegex(none, r"(?m)^\s*io_form_auxinput4\s*=\s*2,")
        self.assertIn('input_inname="wrfinput_d<domain>.bp"', none)

    def test_bp_cache_dir_is_keyed_by_case_and_prep_dir(self):
        old = os.environ.get("WTUNE_BP_CACHE")
        try:
            os.environ["WTUNE_BP_CACHE"] = "/c"
            a = configs.bp_cache_dir("x", self.prep)
            self.assertTrue(a.startswith("/c/x/") and len(os.path.basename(a)) == 12)
            self.assertNotEqual(a, configs.bp_cache_dir("x", self.tmp))
            self.assertEqual(a, configs.bp_cache_dir("x", self.prep + "/"))
        finally:
            if old is None:
                os.environ.pop("WTUNE_BP_CACHE", None)
            else:
                os.environ["WTUNE_BP_CACHE"] = old

    def test_stage_converts_links_and_reports_aux(self):
        p = self._stage(self.nl, self.prep)
        out, err = p.communicate()
        self.assertEqual(p.returncode, 0, err)
        a = self._assignments(out)
        cache = configs.bp_cache_dir("sc26p1", self.prep, os.path.join(self.tmp, "cache"))
        self.assertEqual(a["BP_DIR"], cache)
        self.assertEqual(a["WTUNE_AUX_BP_STREAMS"], "auxinput4")      # no wrffdda_d01 at all
        self.assertEqual(a["BP_CONVERTED"].split(), ["wrfinput_d01", "wrfinput_d02", "wrfbdy_d01",
                                                     "wrflowinp_d01", "wrflowinp_d02"])
        for n in ("wrfinput_d01", "wrfinput_d02"):
            self.assertEqual(os.readlink(os.path.join(cache, n + ".bp")), os.path.join(cache, n))
        # multi-record inputs: one BP per record in cache/<name>/, reported, no flat links
        self.assertEqual(a["WTUNE_PER_RECORD_BP"], "wrfbdy_d01 wrflowinp_d01 wrflowinp_d02")
        for n in ("wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02"):
            self.assertFalse(os.path.lexists(os.path.join(cache, n + ".bp")))
            self.assertEqual(sorted(os.listdir(os.path.join(cache, n))),
                             ["%s_2020-01-01_00:00:00.bp" % n, "%s_2020-01-01_06:00:00.bp" % n])
            self.assertFalse(os.path.lexists(os.path.join(cache, "%s_2020-01-01_00:00:00.bp" % n)))
        self.assertIn("BP input wrfbdy_d01 -> %s (per record, 2 files)"
                      % os.path.join(cache, "wrfbdy_d01"), err)
        self.assertIn("gfdda input wrffdda_d01 found nowhere", err)
        # second run: everything current, nothing reconverted, links kept
        p = self._stage(self.nl, self.prep)
        p.communicate()
        self.assertEqual(p.returncode, 0)
        self.assertEqual(len(self._converted()), 5)

    def test_existing_bp_in_prep_is_used_not_converted(self):
        os.makedirs(os.path.join(self.prep, "wrfinput_d02.bp"))
        # a per-record container already in prep_dir: linked as cache/<name>, not converted
        for t in ("2020-01-01_00:00:00", "2020-01-01_06:00:00"):
            d = os.path.join(self.prep, "wrfbdy_d01.bp", "wrfbdy_d01_%s.bp" % t)
            os.makedirs(d)
            for f in ("md.idx", "data.0"):
                open(os.path.join(d, f), "w").close()
        p = self._stage(self.nl, self.prep)
        out, err = p.communicate()
        self.assertEqual(p.returncode, 0, err)
        a = self._assignments(out)
        cache = a["BP_DIR"]
        self.assertEqual(os.readlink(os.path.join(cache, "wrfinput_d02.bp")),
                         os.path.join(self.prep, "wrfinput_d02.bp"))
        self.assertEqual(os.readlink(os.path.join(cache, "wrfbdy_d01")),
                         os.path.join(self.prep, "wrfbdy_d01.bp"))
        self.assertIn("wrfbdy_d01", a["WTUNE_PER_RECORD_BP"].split())
        self.assertNotIn("convert wrfbdy_d01 BP5", self._converted())
        self.assertNotIn("convert wrfinput_d02 BP5", self._converted())

    def test_existing_single_multirecord_bp_is_reconverted_per_record(self):
        # a single BP wrfbdy (multi-record: WRF reads record 1 everywhere) next to its netCDF:
        # the netCDF is converted per record instead; without the netCDF it is used + WARNING
        os.makedirs(os.path.join(self.prep, "wrfbdy_d01.bp"))
        p = self._stage(self.nl, self.prep)
        out, err = p.communicate()
        self.assertEqual(p.returncode, 0, err)
        self.assertIn("convert wrfbdy_d01 BP5", self._converted())
        self.assertIn("wrfbdy_d01", self._assignments(out)["WTUNE_PER_RECORD_BP"].split())
        os.remove(os.path.join(self.prep, "wrfbdy_d01"))
        p = self._stage(self.nl, self.prep)
        out, err = p.communicate()
        self.assertEqual(p.returncode, 0, err)
        self.assertIn("is a single multi-record BP", err)
        cache = self._assignments(out)["BP_DIR"]
        self.assertEqual(os.readlink(os.path.join(cache, "wrfbdy_d01.bp")),
                         os.path.join(self.prep, "wrfbdy_d01.bp"))

    def test_concurrent_jobs_convert_once(self):
        env = dict(self.env, FAKE_BP_SLEEP="0.3")
        procs = [self._stage(self.nl, self.prep, env=env) for _ in range(3)]
        outs = [p.communicate() for p in procs]
        self.assertEqual([p.returncode for p in procs], [0, 0, 0], [o[1] for o in outs])
        self.assertEqual(sorted(self._converted()),
                         sorted("convert %s BP5" % n for n in (
                             "wrfinput_d01", "wrfinput_d02", "wrfbdy_d01", "wrflowinp_d01",
                             "wrflowinp_d02")))
        self.assertEqual(len(set(self._assignments(o[0])["BP_DIR"] for o in outs)), 1)

    def test_converter_failure_and_missing_wrfinput_are_fatal(self):
        p = self._stage(self.nl, self.prep, env=dict(self.env, FAKE_BP_FAIL="1"))
        _, err = p.communicate()
        self.assertEqual(p.returncode, 1)
        self.assertIn("FATAL: ADIOS2 input staging failed for sc26p1", err)
        os.remove(os.path.join(self.prep, "wrfinput_d01"))
        p = self._stage(self.nl, self.prep)
        _, err = p.communicate()
        self.assertEqual(p.returncode, 1)
        self.assertIn("FATAL: wrfinput/wrfbdy not available", err)

    def test_check_mode_on_an_f1_bp_dir_writes_nothing(self):
        bp = os.path.join(self.tmp, "bp")
        for n in ("wrfinput_d01.bp", "wrfbdy_d01.bp", "wrflowinp_d01.bp"):
            os.makedirs(os.path.join(bp, n))
        before = sorted(os.listdir(bp))
        p = self._stage(self.nl, bp, "--cache-dir", bp, "--check")
        out, err = p.communicate()
        self.assertEqual(p.returncode, 0, err)
        a = self._assignments(out)
        self.assertEqual((a["BP_DIR"], a["WTUNE_AUX_BP_STREAMS"], a["BP_CONVERTED"]),
                         (bp, "", ""))                    # wrflowinp_d02.bp missing -> aux4 off
        self.assertEqual(sorted(os.listdir(bp)), before)
        self.assertEqual(self._converted(), [])

    # ---- strict_campaign.sbatch staging block, extracted ---------------------------------
    def _sbatch(self, name="strict_campaign.sbatch"):
        with open(os.path.join(self.HERE, name)) as f:
            return f.read()

    def _block(self):
        s = self._sbatch()
        i = s.index('WTUNE_AUX_BP_STREAMS=""\n')
        j = s.index("export WTUNE_AUX_BP_STREAMS\n", i) + len("export WTUNE_AUX_BP_STREAMS\n")
        return s[i:j]

    def _run_block(self, bp_dir, pybin_body, extra_env=None):
        import subprocess
        stub = os.path.join(self.tmp, "pybin")
        with open(stub, "w") as f:
            f.write("#!/usr/bin/env bash\necho \"$*\" >> %s/pybin.args\n%s\n"
                    % (self.tmp, pybin_body))
        os.chmod(stub, 0o755)
        script = ("set -uo pipefail\nPYBIN=%s\nCAMPAIGN=/camp\nCASE=sc26p1\nPREP_DIR=/prep\n"
                  "NAMELIST_TEMPLATE=/prep/namelist.input\nBP_DIR='%s'\n"
                  "NAMELIST_REF_DIRS='/prep'\n" % (stub, bp_dir) + self._block() +
                  'echo "OUT BP_DIR=$BP_DIR FMT=${WTUNE_INPUT_FORMAT:-} AUX=$WTUNE_AUX_BP_STREAMS '
                  'REF=$NAMELIST_REF_DIRS"\n')
        env = dict(os.environ)
        env.pop("WTUNE_INPUT_FORMAT", None)
        env.update(extra_env or {})
        p = subprocess.run(["bash", "-c", script], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
        args = ""
        if os.path.exists(os.path.join(self.tmp, "pybin.args")):
            with open(os.path.join(self.tmp, "pybin.args")) as f:
                args = f.read()
        return p, args

    def test_sbatch_empty_bp_dir_stages_prep_into_cache_and_runs_adios2(self):
        p, args = self._run_block("", "echo \"BP_DIR='/cache/sc26p1/abc'\"; "
                                      "echo \"WTUNE_AUX_BP_STREAMS='auxinput4'\"")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(args.split(), ["/camp/configs.py", "stage-bp-inputs", "sc26p1",
                                        "/prep/namelist.input", "/prep"])
        self.assertIn("OUT BP_DIR=/cache/sc26p1/abc FMT=adios2 AUX=auxinput4 "
                      "REF=/cache/sc26p1/abc /prep", p.stdout)

    def test_sbatch_f1_bp_dir_is_only_checked(self):
        p, args = self._run_block("/f1/bp", "echo \"BP_DIR='/f1/bp'\"; "
                                            "echo \"WTUNE_AUX_BP_STREAMS=''\"")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(args.split(), ["/camp/configs.py", "stage-bp-inputs", "sc26p1",
                                        "/prep/namelist.input", "/f1/bp", "--cache-dir", "/f1/bp",
                                        "--check"])
        self.assertIn("OUT BP_DIR=/f1/bp FMT=adios2 AUX= REF=/f1/bp /prep", p.stdout)

    def test_sbatch_staging_failure_aborts_the_job(self):
        p, _ = self._run_block("", "echo 'FATAL: x' >&2; exit 1")
        self.assertEqual(p.returncode, 1)
        self.assertIn("FATAL: ADIOS2 input staging failed for case sc26p1", p.stderr)
        self.assertNotIn("OUT ", p.stdout)

    def test_sbatch_netcdf_is_only_an_explicit_legacy_escape(self):
        p, args = self._run_block("/f1/bp", "exit 1", {"WTUNE_INPUT_FORMAT": "netcdf"})
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(args, "")                              # no staging at all
        self.assertIn("WARNING: WTUNE_INPUT_FORMAT=netcdf -- LEGACY", p.stderr)
        self.assertIn("OUT BP_DIR= FMT=netcdf AUX= REF=/prep", p.stdout)
        s = self._sbatch()
        self.assertNotIn('if [ -z "$BP_DIR" ]; then export WTUNE_INPUT_FORMAT=netcdf', s)
        # staged before the first run dir links its inputs
        self.assertLess(s.index("export WTUNE_AUX_BP_STREAMS\n"), s.index("  link_wrf_inputs\n"))

    def test_sbatch_scripts_parse_and_nc2bp_has_no_f1_paths(self):
        import subprocess
        for name in ("strict_campaign.sbatch", "nc2bp.sbatch"):
            subprocess.run(["bash", "-n", os.path.join(self.HERE, name)], check=True)
        s = self._sbatch("nc2bp.sbatch")
        self.assertNotIn("/home/u5727520", s)
        self.assertIn('configs.py" stage-bp-inputs "$CASE" "$NAMELIST_TEMPLATE" "$PREP_DIR"', s)


class TestPerRecordInputs(unittest.TestCase):
    """WRF 4.8's ADIOS2 reader never selects a BP step, so wrfbdy/wrflowinp/wrffdda are staged one
    BP per record (<BP_DIR>/<name>/<name>_<date>.bp, WTUNE_PER_RECORD_BP) and read with
    multi_bdy_files / dated innames. WRF opens the file of the run's START time, so a probe
    restarting off a record time reads that stream from netCDF instead (_apply_adios2)."""

    HERE = os.path.dirname(os.path.abspath(__file__))
    NL = (" &time_control\n start_year = 2020, 2020,\n start_month = 01, 01,\n"
          " start_day = 01, 01,\n start_hour = 00, 00,\n start_minute = 00, 00,\n"
          " start_second = 00, 00,\n interval_seconds = 21600,\n run_minutes = 60,\n"
          " io_form_history = 2,\n io_form_input = 2,\n io_form_boundary = 2,\n"
          " io_form_restart = 2,\n restart_interval = 60,\n"
          " auxinput4_inname = \"wrflowinp_d<domain>\",\n io_form_auxinput4 = 2,\n"
          " auxinput4_interval = 360, 360,\n /\n"
          " &domains\n max_dom = 2,\n time_step = 60,\n /\n"
          " &physics\n sst_update = 1,\n /\n")

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="wtune-pr-test-")
        self.bp = os.path.join(self.tmp, "bp")
        for n in ("wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02"):
            for t in ("2020-01-01_00:00:00", "2020-01-01_06:00:00"):
                os.makedirs(os.path.join(self.bp, n, "%s_%s.bp" % (n, t)))
        self._old = {k: os.environ.get(k) for k in (
            "WTUNE_INPUT_FORMAT", "WTUNE_AUX_BP_STREAMS", "WTUNE_PER_RECORD_BP", "WTUNE_BP_DIR")}
        os.environ.pop("WTUNE_INPUT_FORMAT", None)
        os.environ["WTUNE_AUX_BP_STREAMS"] = "auxinput4"
        os.environ["WTUNE_PER_RECORD_BP"] = "wrfbdy_d01 wrflowinp_d01 wrflowinp_d02"
        os.environ["WTUNE_BP_DIR"] = self.bp

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @staticmethod
    def _vals(text, key):
        return configs._namelist_string_values(text, key)

    def _edit(self, text, restart_from=None):
        return configs.edit_namelist(text, 8, 9, 1, 60, max_dom=2, ranks=72,
                                     restart_from=restart_from)

    def test_set_namelist_value_lists_packed_lines_and_absent_groups(self):
        t = (" &time_control\n frames_per_auxinput4 = 999999, 999999, io_form_history = 2,\n"
             " /\n &domains\n max_dom = 2,\n /\n")
        out = configs._set_namelist_value(t, "frames_per_auxinput4", "1, 1", "time_control")
        self.assertIn(" frames_per_auxinput4 = 1, 1, io_form_history = 2,\n", out)
        out = configs._set_namelist_value(out, "multi_bdy_files", ".true.", "bdy_control")
        self.assertTrue(out.endswith("&bdy_control\n multi_bdy_files = .true.,\n/\n"))
        out = configs._set_namelist_value(out, "multi_bdy_files", ".false.", "bdy_control")
        self.assertEqual(out.count("multi_bdy_files"), 1)
        self.assertIn("multi_bdy_files = .false.,", out)
        t2 = " &bdy_control\n spec_bdy_width = 5,\n /\n"
        self.assertEqual(configs._set_namelist_value(t2, "multi_bdy_files", ".true.", "bdy_control"),
                         " &bdy_control\n multi_bdy_files = .true.,\n spec_bdy_width = 5,\n /\n")
        q = configs._set_namelist_value(' &fdda\n gfdda_inname = "a, b",\n /\n', "gfdda_inname",
                                        '"wrffdda_d<domain>_<date>.bp"', "fdda")
        self.assertIn('gfdda_inname = "wrffdda_d<domain>_<date>.bp",\n', q)

    def test_cold_start_on_a_record_reads_per_record_bp(self):
        import re
        out = self._edit(self.NL)
        self.assertEqual(self._vals(out, "io_form_boundary"), ["14"])
        self.assertEqual(self._vals(out, "bdy_inname"), ["wrfbdy_d<domain>_<date>.bp"])
        self.assertEqual(self._vals(out, "multi_bdy_files"), [".true."])
        self.assertIn("&bdy_control", out)
        self.assertEqual(self._vals(out, "io_form_auxinput4"), ["14"])
        self.assertEqual(self._vals(out, "auxinput4_inname"), ["wrflowinp_d<domain>_<date>.bp"])
        self.assertEqual(self._vals(out, "frames_per_auxinput4"), ["1", "1"])
        self.assertEqual(self._vals(out, "input_inname"), ["wrfinput_d<domain>.bp"])
        for k in ("bdy_inname", "auxinput4_inname", "multi_bdy_files", "frames_per_auxinput4"):
            self.assertEqual(sum(1 for ln in out.splitlines()
                                 if re.search(r"(?<![A-Za-z0-9_])%s\s*=" % k, ln)), 1, k)

    def test_restart_on_a_record_time_keeps_per_record(self):
        import datetime
        first = self._edit(self.NL)
        out = self._edit(first, restart_from=datetime.datetime(2020, 1, 1, 6))
        self.assertEqual(self._vals(out, "bdy_inname"), ["wrfbdy_d<domain>_<date>.bp"])
        self.assertEqual(self._vals(out, "io_form_auxinput4"), ["14"])

    def test_restart_off_record_falls_back_to_netcdf_and_undoes_the_first_pass(self):
        import contextlib
        import datetime
        import io
        first = self._edit(self.NL)                      # the sbatch's pass 1 (cold start t0)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = self._edit(first, restart_from=datetime.datetime(2020, 1, 1, 1))  # t0 + 60 min
        self.assertEqual(self._vals(out, "io_form_boundary"), ["2"])
        self.assertIsNone(self._vals(out, "bdy_inname"))          # WRF default wrfbdy_d<domain>
        self.assertEqual(self._vals(out, "multi_bdy_files"), [".false."])
        self.assertEqual(self._vals(out, "io_form_auxinput4"), ["2"])
        self.assertEqual(self._vals(out, "auxinput4_inname"), ["wrflowinp_d<domain>"])
        self.assertEqual(self._vals(out, "frames_per_auxinput4"), ["999999", "999999"])
        self.assertEqual(self._vals(out, "input_inname"), ["wrfinput_d<domain>.bp"])
        self.assertEqual(self._vals(out, "io_form_input"), ["14"])
        self.assertEqual(self._vals(out, "start_hour"), ["1", "1"])
        self.assertIn("boundary input has no record at the run start "
                      "(wrfbdy_d01_2020-01-01_01:00:00.bp)", err.getvalue())
        # idempotent for the I/O keys (_apply_restart_from itself reorders the start_* lines)
        with contextlib.redirect_stderr(io.StringIO()):
            again = self._edit(out, restart_from=datetime.datetime(2020, 1, 1, 1))
        for k in ("io_form_boundary", "bdy_inname", "multi_bdy_files", "io_form_auxinput4",
                  "auxinput4_inname", "frames_per_auxinput4", "input_inname", "io_form_input"):
            self.assertEqual(self._vals(again, k), self._vals(out, k), k)

    def test_one_domain_missing_its_record_falls_back_the_whole_stream(self):
        import contextlib
        import io
        import shutil
        shutil.rmtree(os.path.join(self.bp, "wrflowinp_d02",
                                   "wrflowinp_d02_2020-01-01_00:00:00.bp"))
        with contextlib.redirect_stderr(io.StringIO()):
            out = self._edit(self.NL)
        self.assertEqual(self._vals(out, "bdy_inname"), ["wrfbdy_d<domain>_<date>.bp"])
        self.assertEqual(self._vals(out, "io_form_auxinput4"), ["2"])
        self.assertEqual(self._vals(out, "auxinput4_inname"), ["wrflowinp_d<domain>"])
        self.assertIsNone(self._vals(out, "frames_per_auxinput4"))   # never set, not added

    def test_aux_stream_not_available_is_not_switched(self):
        os.environ["WTUNE_AUX_BP_STREAMS"] = ""
        out = self._edit(self.NL)
        self.assertEqual(self._vals(out, "io_form_auxinput4"), ["2"])
        self.assertEqual(self._vals(out, "auxinput4_inname"), ["wrflowinp_d<domain>"])
        self.assertEqual(self._vals(out, "bdy_inname"), ["wrfbdy_d<domain>_<date>.bp"])

    def test_no_per_record_inputs_keeps_the_single_bp_innames(self):
        os.environ["WTUNE_PER_RECORD_BP"] = ""
        out = self._edit(self.NL)
        self.assertIn(' input_inname="wrfinput_d<domain>.bp", bdy_inname="wrfbdy_d<domain>.bp",',
                      out)
        self.assertNotIn("multi_bdy_files", out)

    def test_stage_namelist_files_skips_a_per_record_dir_for_a_netcdf_name(self):
        prep = os.path.join(self.tmp, "prep")
        run = os.path.join(self.tmp, "run")
        os.makedirs(prep)
        os.makedirs(run)
        for n in ("wrflowinp_d01", "wrflowinp_d02"):
            open(os.path.join(prep, n), "w").close()
        res = configs.stage_namelist_files(self.NL, run, [self.bp, prep])
        self.assertEqual(os.readlink(os.path.join(run, "wrflowinp_d01")),
                         os.path.join(prep, "wrflowinp_d01"))
        self.assertEqual(res["missing"], [])
        # a per-record namelist stages every record file of every domain
        nl = self.NL.replace('"wrflowinp_d<domain>"', '"wrflowinp_d<domain>_<date>.bp"')
        run2 = os.path.join(self.tmp, "run2")
        os.makedirs(run2)
        configs.stage_namelist_files(nl, run2, [os.path.join(self.bp, "wrflowinp_d01"),
                                                os.path.join(self.bp, "wrflowinp_d02")])
        self.assertEqual(sorted(os.listdir(run2)), [
            "wrflowinp_d01_2020-01-01_00:00:00.bp", "wrflowinp_d01_2020-01-01_06:00:00.bp",
            "wrflowinp_d02_2020-01-01_00:00:00.bp", "wrflowinp_d02_2020-01-01_06:00:00.bp"])

    def test_sbatch_link_wrf_inputs_links_every_record_and_the_netcdf(self):
        import subprocess
        with open(os.path.join(self.HERE, "strict_campaign.sbatch")) as f:
            s = f.read()
        i = s.index("link_per_record() {")
        j = s.index("\ncheck_disk_guard () {", i)
        prep = os.path.join(self.tmp, "prep")
        run = os.path.join(self.tmp, "run")
        os.makedirs(prep)
        os.makedirs(run)
        open(os.path.join(prep, "wrfbdy_d01"), "w").close()
        script = ("set -uo pipefail\nMAX_DOM=2\nPREP_DIR=%s\nBP_DIR=%s\n"
                  "WTUNE_PER_RECORD_BP='wrfbdy_d01 wrflowinp_d01 wrflowinp_d02'\n"
                  "%s\ncd %s && link_wrf_inputs\n" % (prep, self.bp, s[i:j], run))
        p = subprocess.run(["bash", "-c", script], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        names = sorted(os.listdir(run))
        self.assertIn("wrfbdy_d01", names)                       # netCDF, for an off-record start
        self.assertNotIn("wrfbdy_d01.bp", names)
        for n in ("wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02"):
            rec = "%s_2020-01-01_06:00:00.bp" % n
            self.assertEqual(os.readlink(os.path.join(run, rec)), os.path.join(self.bp, n, rec))
        self.assertIn("wrfinput_d02.bp", names)

class TestFreshProbeInputs(unittest.TestCase):
    """Head-side fresh per-record copies for probe jobs (WTUNE_BP_FRESH_DIR/copy_<i>/<name>/):
    every wrf.exe run opens its own inodes -- NFS open stall, wiki 2026-10-03 adios2 gate."""

    HERE = os.path.dirname(os.path.abspath(__file__))
    NL = (" &time_control\n start_year = 2020,\n start_month = 01,\n start_day = 01,\n"
          " start_hour = 00,\n interval_seconds = 21600,\n /\n &domains\n max_dom = 1,\n /\n")

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="wtune-fresh-test-")
        self.prep = os.path.join(self.tmp, "prep")
        os.makedirs(self.prep)
        for n in ("wrfinput_d01", "wrfbdy_d01"):
            with open(os.path.join(self.prep, n), "wb") as f:
                f.write(b"CDF\x01" + b"\0" * 60)
        with open(os.path.join(self.prep, "namelist.input"), "w") as f:
            f.write(self.NL)
        self.cases = os.path.join(self.tmp, "cases.json")
        with open(self.cases, "w") as f:
            json.dump({"cases": {"pt": {"prep_dir": self.prep, "bp_dir": ""}}}, f)
        self.cache = configs.bp_cache_dir("pt", self.prep, os.path.join(self.tmp, "cache"))
        self._old = os.environ.get("WTUNE_BP_CACHE")
        os.environ["WTUNE_BP_CACHE"] = os.path.join(self.tmp, "cache")
        self.copies = []

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        if self._old is None:
            os.environ.pop("WTUNE_BP_CACHE", None)
        else:
            os.environ["WTUNE_BP_CACHE"] = self._old

    def _bp(self, current=True, reflink=True):
        import re as _re
        import types
        test = self

        def fresh_input_copy(sources, dest, stats=None):
            test.copies.append((dict(sources), dest))
            stats.update(reflink=reflink, bytes=0 if reflink else 10)
            return {n: os.path.join(dest, n) for n in sources}
        return types.SimpleNamespace(
            PER_RECORD_NAME_RE=_re.compile(r"^(wrfbdy|wrflowinp|wrffdda)_d\d\d$"),
            is_per_record_dir=lambda p: False, fresh_input_copy=fresh_input_copy,
            cached_inputs=lambda d, c, names: ({n: os.path.join(c, n) for n in names}
                                               if current else None))

    def test_copies_per_run_from_the_current_cache(self):
        dest = os.path.join(self.tmp, "job", "bp_fresh")
        got = configs.fresh_probe_inputs("pt", dest, 3, cases_json=self.cases, bp=self._bp())
        self.assertEqual(got, dest)
        self.assertEqual([d for _s, d in self.copies],
                         [os.path.join(dest, "copy_%d" % i) for i in range(3)])
        self.assertEqual(self.copies[0][0], {"wrfbdy_d01": os.path.join(self.cache, "wrfbdy_d01")})

    def test_without_reflink_one_copy_only(self):
        dest = os.path.join(self.tmp, "job", "bp_fresh")
        configs.fresh_probe_inputs("pt", dest, 5, cases_json=self.cases,
                                   bp=self._bp(reflink=False))
        self.assertEqual(len(self.copies), 1)

    def test_cache_not_current_gives_none_with_a_warning(self):
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got = configs.fresh_probe_inputs("pt", os.path.join(self.tmp, "d"), 2,
                                             cases_json=self.cases, bp=self._bp(current=False))
        self.assertIsNone(got)
        self.assertEqual(self.copies, [])
        self.assertIn("not current for wrfbdy_d01 (run campaign/nc2bp.sbatch first)",
                      err.getvalue())

    def test_bp_cache_status(self):
        self.assertEqual(configs.bp_cache_status("pt", self.cases, bp=self._bp())[0], "current")
        st, detail = configs.bp_cache_status("pt", self.cases, bp=self._bp(current=False))
        self.assertEqual(st, "stale")
        self.assertIn("wrfbdy_d01", detail)
        self.assertEqual(configs.bp_cache_status("nope", self.cases, bp=self._bp())[0], "none")

    def _link_script(self, fresh_dir, runs):
        with open(os.path.join(self.HERE, "strict_campaign.sbatch")) as f:
            s = f.read()
        body = s[s.index("link_per_record() {"):s.index("\ncheck_disk_guard () {")]
        # link_wrf_inputs must run in this shell (it advances FRESH_RUN_IDX): no subshell
        cmds = "".join("mkdir run%d; cd run%d; link_wrf_inputs; cd ..\n" % (i, i)
                       for i in range(runs))
        return ("set -uo pipefail\nMAX_DOM=1\nPREP_DIR=%s\nBP_DIR=%s\n"
                "WTUNE_PER_RECORD_BP='wrfbdy_d01'\nWTUNE_BP_FRESH_DIR='%s'\n%s\ncd %s\n%s"
                % (self.prep, self.cache, fresh_dir, body, self.tmp, cmds))

    def test_sbatch_links_copy_i_for_run_i_then_reuses_the_last(self):
        import subprocess
        rec = "wrfbdy_d01_2020-01-01_00:00:00.bp"
        fresh = os.path.join(self.tmp, "bp_fresh")
        for i in range(2):
            os.makedirs(os.path.join(fresh, "copy_%d" % i, "wrfbdy_d01", rec))
        os.makedirs(os.path.join(self.cache, "wrfbdy_d01", rec))
        p = subprocess.run(["bash", "-c", self._link_script(fresh, 3)], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        links = [os.readlink(os.path.join(self.tmp, "run%d" % i, rec)) for i in range(3)]
        self.assertEqual(links, [os.path.join(fresh, "copy_0", "wrfbdy_d01", rec),
                                 os.path.join(fresh, "copy_1", "wrfbdy_d01", rec),
                                 os.path.join(fresh, "copy_1", "wrfbdy_d01", rec)])
        self.assertIn("fresh BP copies exhausted at run 2", p.stderr)
        # no fresh dir: the cache, as before
        p = subprocess.run(["bash", "-c", self._link_script("", 1).replace(
            "cd %s\n" % self.tmp, "cd %s\nrm -rf run0\n" % self.tmp)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(os.readlink(os.path.join(self.tmp, "run0", rec)),
                         os.path.join(self.cache, "wrfbdy_d01", rec))

    def test_sbatch_warns_without_fresh_dir_and_cleans_up_its_copies(self):
        with open(os.path.join(self.HERE, "strict_campaign.sbatch")) as f:
            s = f.read()
        self.assertIn("no WTUNE_BP_FRESH_DIR: runs share the cache's per-record inodes", s)
        self.assertIn('rm -rf "$WTUNE_BP_FRESH_DIR"', s)

if __name__ == "__main__":
    unittest.main()
