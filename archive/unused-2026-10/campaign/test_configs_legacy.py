"""Tests of the removed configs.py subcommands and of the MODE=physics script text, cut from
campaign/test_configs.py as at 30572ce (one file, not runnable as is)."""

# ---- from campaign/test_configs.py
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


# ---- from campaign/test_configs.py
    def test_physics_order_explicit_pconfigs(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = configs.cmd_physics_order(["karina", "--pconfigs", "P01+P03", "--reps", "1"])
        self.assertEqual(rc, 0)
        lines = buf.getvalue().strip().split("\n")
        self.assertEqual(lines, ["P00_ref", "P00_twin", "P01", "P03"])


# ---- from campaign/test_configs.py
class TestPhysicsDescribe(unittest.TestCase):
    def setUp(self):
        # describe reads the case's real namelist (an scc-only path); point karina at a temp copy
        import tempfile
        from unittest import mock
        d = tempfile.mkdtemp()
        path = os.path.join(d, "namelist.input")
        with open(path, "w") as f:
            f.write(TestApplyPhysicsEdit.SINGLE_DOMAIN)
        patch = mock.patch.dict(configs.CASES["karina"],
                                {"namelist_template": path, "max_dom": 1})
        patch.start()
        self.addCleanup(patch.stop)

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


# ---- from campaign/test_configs.py
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


# ---- from campaign/test_configs.py
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


# ---- from campaign/test_configs.py
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
