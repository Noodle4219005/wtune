#!/usr/bin/env python3
"""Unit tests for campaign/configs.py. Stdlib unittest, Python 3.6 compatible."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# another test module may have imported configs with SITE=scc (scc_sweep sets it): start clean
os.environ.pop("SITE", None)
sys.modules.pop("configs", None)
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


    def test_conus12_448_ranks_has_legal_shapes(self):
        # documents that 425x300 at 448 ranks is NOT fully illegal (some templates
        # still resolve); this exercises the case the task flagged as "may lack
        # alt shapes" without assuming it lacks ALL shapes.
        shapes, method = configs.ranked_shapes(425, 300, 50, 448, 1)
        self.assertGreater(len(shapes), 0)
        self.assertIn(method, ("wtune.decomp", "fallback"))


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


    def test_c34_and_marshall_carry_max_omp_1_in_cases_json(self):
        for case_id in ("c34", "marshall"):
            self.assertEqual(configs.CASES[case_id].get("max_omp"), 1, case_id)


class TestBuildOrder(unittest.TestCase):


    def test_reps_zero_is_empty(self):
        self.assertEqual(configs.build_order(["A", "B"], 0), [])


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


    def test_tile_geometry_args_partial_only_writes_given_keys(self):
        out = configs.edit_namelist(self.SYNTHETIC_WITHOUT_EXISTING, 14, 16, 12, 8,
                                     history_mode="end_d01", ranks=224, numtiles_x=4)
        self.assertIn("numtiles_x=4", out)
        self.assertNotIn("numtiles_y", out)
        self.assertNotIn("tile_sz_x", out)
        self.assertNotIn("tile_strategy", out)


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


    def test_edit_namelist_delete_keys_param_removes_both_lafe_keys(self):
        out = configs.edit_namelist(self.SYNTHETIC_DELETE_KEYS, 14, 16, 12, 8, ranks=224,
                                     delete_keys=["bl_mynn_tkebudget", "opt_run"])
        self.assertNotIn("tkebudget", out.lower())
        self.assertNotIn("opt_run", out.lower())
        self.assertIn(" icloud_bl = 1,\n", out)
        self.assertIn(" dveg=9 ! former 3\n", out)
        self.assertIn(" opt_crs=1,\n", out)


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

    AUX24 = (
        "&time_control\n"
        " run_days = 0, run_hours = 0, run_minutes = 30,\n"
        " history_interval_s = 0, 0, 0,\n"
        " io_form_auxhist24 = 2\n"
        " auxhist24_interval_s = 60, 60, 60,\n"
        " auxhist24_begin_s = 0, 0, 0,\n"
        " auxhist3_interval = 0, 0, 0,\n"
        "/\n"
        "&domains\n"
        " max_dom = 3,\n"
        " time_step=15, e_we=793, 805, 1001, e_sn=853, 805, 1001,\n"
        "/\n"
    )

    def test_aux_stream_never_begins_in_every_mode(self):
        for mode in ("off", "end_d01", "mid_end_d01"):
            out = configs.edit_namelist(self.AUX24, 16, 28, 8, 30, max_dom=3,
                                        history_mode=mode, ranks=448)
            self.assertIn("auxhist24_begin_s = 60000000, 60000000, 60000000,", out)
            self.assertNotIn("auxhist24_begin_s = 0", out)
            self.assertIn("auxhist24_interval_s = 60, 60, 60,", out)


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


    def test_ranks_required_when_adios2_enabled(self):
        with self.assertRaises(ValueError):
            configs.edit_namelist(self.SYNTHETIC_NO_ADIOS2, 14, 16, 12, 8)


class TestStepStats(unittest.TestCase):
    def _series_with_radiation_spikes(self, n_cycles, cycle_len, normal=6.0, spike=44.0, init=100.0):
        rest = []
        for i in range(n_cycles * cycle_len):
            rest.append(spike if (i + 1) % cycle_len == 0 else normal)
        return [init] + rest, rest


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
            import wtune.compliance
        except Exception as e:  # pragma: no cover - only if repo layout changes
            self.skipTest("wtune.compliance not importable: %s" % e)
        for name in configs._COMPLIANCE_KNOBS:
            self.assertEqual(wtune.compliance.classify(name), wtune.compliance.STRICT, name)


class TestCasesJson(unittest.TestCase):
    def test_all_cases_have_required_fields(self):
        required = ("prep_dir", "bp_dir", "restart", "input_files", "history_key", "dt",
                    "e_we", "e_sn", "e_vert", "dx", "dy", "case_family", "steps", "physics")
        for cid, case in configs.CASES.items():
            for field in required:
                self.assertIn(field, case, "%s missing %s" % (cid, field))

    def test_case_order_matches_cases_json_keys(self):
        self.assertEqual(sorted(configs.CASE_ORDER), sorted(configs.CASES.keys()))


class TestHistoryModeInterval(unittest.TestCase):
    """MODE=physics fidelity replicates (decision 0051): --history-mode interval
    --history-minutes N writes a d01 frame every N minutes starting at t=0
    (history_begin*=0), nested domains still off by default -- same
    nest_history_on-gated rule as 'off'/'end_d01'."""

    SINGLE = TestNamelistEditing.SYNTHETIC_WITH_EXISTING
    NEST_MINUTES = TestNestHistory.SYNTHETIC_NEST_MINUTES
    NEST_MINUTES_M = TestNestHistory.SYNTHETIC_NEST_MINUTES_M
    NEST_SECONDS = TestNamelistEditing.SYNTHETIC_3DOMAIN


    def test_interval_minutes_m_variant_max_dom_3(self):
        out = configs.edit_namelist(self.NEST_MINUTES_M, 16, 28, 8, 60, max_dom=3,
                                     history_mode="interval", history_minutes=10, ranks=448)
        self.assertIn("history_interval_m = 10, 0, 0,", out)
        self.assertIn("history_begin_m = 0, 0, 0,", out)


    def test_interval_requires_history_minutes(self):
        with self.assertRaises(ValueError):
            configs.edit_namelist(self.SINGLE, 14, 16, 12, 60, history_mode="interval", ranks=224)


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


class TestFormatPhysicsEdits(unittest.TestCase):

    def test_set_and_mul_join_with_semicolons_no_spaces(self):
        summary = [
            dict(key="radt", op="mul", old=["15"], new=["30"]),
            dict(key="swint_opt", op="set", old=[], new=["1"]),
        ]
        s = configs.format_physics_edits(summary)
        self.assertEqual(s, "radt=30;swint_opt=1")
        self.assertNotIn(" ", s)


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


class TestConfigFileGeneratedConfigs(unittest.TestCase):
    """CONFIG_FILE (WTune D1 decomposition-space work): campaign/configs.py's own side of
    consuming a predictor/decomp_space.py `to_campaign_configs`-shaped JSON file. The JSON here
    is built by hand (not via `import decomp_space`) -- decomp_space.py uses `from __future__
    import annotations` (Python >= 3.7 only, same reason wtune/decomp.py cannot be imported under
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


    def test_empty_config_file_is_refused_with_the_expected_shape_named(self):
        path = self._config_file([])
        ref, _twin, reason = configs.resolve_decomp_pair("sc26p1", "G0001", "G0002",
                                                         config_file=path)
        self.assertIsNone(ref)
        self.assertIn("no configs", reason)

    # ---- the twin must be a DIFFERENT decomposition ---------------------------------


    def test_identical_strict_template_pair_is_refused_too(self):
        ref, _twin, reason = configs.resolve_decomp_pair("karina", "C02", "C02")
        self.assertIsNone(ref)
        self.assertIn("SAME decomposition", reason)

    # ---- id hygiene: run_id and notes both have to survive the id --------------------


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


    def test_quota_zero_is_no_limit_not_free_gib(self):
        # a quota column of 0 means "not tracked at the user level" (see the fileset-vs-
        # user mmlsquota trap note -- 'project''s real row has this shape too, but with an
        # extra fileset-name column between the filesystem and USR that the parser never
        # matches at all; 'scratch' isolates the quota=0 case on its own) -- must return
        # None, not a huge false-positive free_gib from (0 - used)
        free = configs.quota_free_gib(MMLSQUOTA_SAMPLE, filesystem="scratch")
        self.assertIsNone(free)


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
        env.pop("SITE", None)        # the scc overlay would add its own cases
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
        self.assertEqual(s.count("ntasks_per_node=$(( ranks / ALLOC_NODES ))"), 1)
        self.assertEqual(s.count('N_NODES="$ALLOC_NODES"'), 1)

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
        self.assertIn("physics_namelists", text)       # wtune.tune.PHYSICS_RUNNER_MARKER
        self.assertIn("config-extras", text)
        self.assertIn('--history-mode "$RUN_HISTORY_ARG"', text)
        self.assertNotIn('--history-mode "$HISTORY_MODE_ARG" --ranks "$ranks"', text)
        self.assertIn("window_frame=", text)
        self.assertIn("window_frame_mid=", text)
        self.assertIn('rm -rf "$d"/wrfrst_d0*_*', text)

    def test_restart_pass_gets_run_sec_args(self):
        text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "strict_campaign.sbatch")).read()
        passes = [b for b in text.split("edit-namelist")[1:]
                  if '--restart-from "$RESTART_TIME"' in b.split("\n\n")[0]]  # the spin-up pass
        self.assertEqual(len(passes), 1)
        self.assertIn('--run-minutes "$RUN_MIN_EFF" "${RUN_SEC_ARGS[@]}"', passes[0])

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


    def test_anchor_end_off_omits_run(self):
        r = configs.resolve_point_plan(["ANCHOR", "C1", "C2", "ANCHOREND"], anchor_end=False)
        self.assertEqual(r["order"], ["ANCHOR", "C1", "C2"])
        self.assertFalse(r["anchor_end"])
        r = configs.resolve_point_plan(["ANCHOR", "C1", "ANCHOREND"], anchor_end=True)
        self.assertIn("ANCHOREND", r["order"])


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


class TestDiagnosticsAndCadenceTemplates(unittest.TestCase):
    """D04-D06 (2026-09-16): diagnostics off, PBL call interval, one-way nesting."""

    NL = ("&time_control\n nwp_diagnostics = 1,\n/\n"
          "&domains\n time_step=15, max_dom=2,\n/\n"
          "&physics\n mp_physics = 8, 8,\n hailcast_opt = 1, 1,\n do_radar_ref = 1,\n/\n"
          "&bdy_control\n spec_bdy_width = 5,\n feedback = 1,\n/\n")


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


    def test_a01_is_the_mirror_of_p03(self):
        # P03 multiplies the same key upward; together they bracket the case's native dt.
        self.assertEqual(configs.PHYSICS_TEMPLATES["P03"]["edits"][0]["key"], "time_step")
        self.assertGreater(configs.PHYSICS_TEMPLATES["P03"]["edits"][0]["factor"], 1.0)
        self.assertLess(self._tpl("A01")["edits"][0]["factor"], 1.0)


    def test_a04_order_is_even_and_at_the_halo_ceiling(self):
        # module_advect_em.F implements 6/5/4/3/2 (and 0); dyn_em/solve_em.F:414 aborts above 6
        # because the communication halos only exist to order 6.
        for e in self._tpl("A04")["edits"]:
            self.assertEqual(e["value"], 6, e["key"])


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


    def test_a05_does_not_pair_swint_opt_the_way_p02_does(self):
        self.assertEqual([e["key"] for e in self._tpl("A05")["edits"]], ["radt"])
        self.assertIn("swint_opt", [e["key"] for e in configs.PHYSICS_TEMPLATES["P02"]["edits"]])


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


    def test_floor_zero_fields_contribute_no_floor_ratio(self):
        # fidelity_gate.py leaves floor_ratio null when the twin was bit-identical to the
        # reference; that field must not be counted, and must not crash the max().
        payload = {"label": "WITHIN-FLOOR", "deciding": "all key fields within floor",
                   "per_field_summary_at_last_time": {
                       "T": {"nd": 0.001, "floor_ratio": None, "floor_zero": True},
                       "U": {"nd": 0.002, "floor_ratio": 0.5, "floor_zero": False}}}
        _l, nd, floor, _d = configs.fidelity_notes_tokens(payload)
        self.assertEqual((nd, floor), ("0.002", "0.5"))


class TestValuesByDomain(unittest.TestCase):
    """apply_physics_edit's 'set' op with a per-domain value list (2026-09-17). The scalar 'value'
    form is unchanged and still the default; 'values_by_domain' exists because a nest does not
    always want its parent's value (SC26-P1's d01 carries 3x its d02's acoustic margin at the same
    dt -- predictor/stability.py)."""

    PRESENT = ("&dynamics\n time_step_sound = 4, 4,\n/\n")
    ABSENT = ("&domains\n max_dom = 2,\n/\n&dynamics\n diff_opt = 1, 1,\n/\n")


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
    file instead of from t0 -- see edit_namelist's own docstring and wtune/probe_plan.py."""

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


    def test_write_restart_interval_sets_interval_without_forcing_restart_true(self):
        out = configs.edit_namelist(self.SYNTHETIC_3DOM_RESTART, 16, 28, 8, 360, max_dom=3,
                                     adios2=False, write_restart_interval=60,
                                     restart_io_form=2)
        self.assertIn("restart_interval = 60", out)
        self.assertNotIn("restart_interval = 1440", out)
        self.assertIn("restart = .false.,", out)   # spin-up writer still starts at t0
        self.assertIn("io_form_restart = 2", out)


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
        self.assertEqual(len(idx), 2)
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
    _apply_adios2's aux switch / stage-bp-inputs (conversion through a FAKE src/wtune/bp.py found via
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
        os.makedirs(os.path.join(self.src, "wtune"))
        with open(os.path.join(self.src, "wtune", "bp.py"), "w") as f:
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
                                                  "wtune", "bp.py"))

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

    def test_mp_physics_7_gets_do_radar_ref(self):
        nl = " &physics\n mp_physics = 7, 7,\n do_radar_ref = 0,\n /\n"
        out = configs._force_gsfcgce_radar_ref(nl)
        self.assertIn(" do_radar_ref = 1,\n", out)
        self.assertEqual(configs._force_gsfcgce_radar_ref(out), out)
        other = " &physics\n mp_physics = 17, 8,\n /\n"
        self.assertEqual(configs._force_gsfcgce_radar_ref(other), other)
        self.assertIn("do_radar_ref = 1,", self._edit(self.NL.replace("&physics", "&physics\n mp_physics = 7, 7,")))


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


class TestSbatchRefusesRetiredModes(unittest.TestCase):
    """strict_campaign.sbatch runs MODE=timing with a CONFIG_FILE only; the script stops before
    its python resolution, so an empty site file is enough to run it here."""
    HERE = os.path.dirname(os.path.abspath(__file__))

    def _run(self, **env):
        import subprocess
        import tempfile
        stub = tempfile.mkdtemp()
        os.makedirs(os.path.join(stub, "sites"))
        open(os.path.join(stub, "sites", "f1.sh"), "w").close()
        full = {k: v for k, v in os.environ.items() if k not in ("MODE", "CONFIG_FILE", "SITE")}
        full.update(env, CAMPAIGN=stub)
        return subprocess.run(["bash", os.path.join(self.HERE, "strict_campaign.sbatch")], env=full,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)

    def test_old_modes_are_refused_and_point_at_the_archive(self):
        for mode in ("physics", "correctness"):
            p = self._run(MODE=mode, CONFIG_FILE="/x.json")
            self.assertEqual(p.returncode, 1, mode)
            self.assertIn("MODE must be 'timing', got '%s'" % mode, p.stderr)
            self.assertIn("archive/unused-2026-10/campaign/strict_campaign_full.sbatch", p.stderr)

    def test_missing_config_file_is_refused(self):
        for env in ({}, {"CONFIG_FILE": ""}):
            p = self._run(MODE="timing", **env)
            self.assertEqual(p.returncode, 1)
            self.assertIn("CONFIG_FILE env var is required", p.stderr)
            self.assertEqual(p.stdout, "")


if __name__ == "__main__":
    unittest.main()
