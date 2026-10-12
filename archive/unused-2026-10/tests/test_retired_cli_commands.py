# Tests of the retired `wtune profile` / `wtune knobs` commands, cut verbatim from
# tests/test_wtune_case.py and tests/test_wtune_compliance.py (CliSmokeTests) at 18a9b9f.
# Not runnable as is (fixtures and imports stayed in the original files).


# from tests/test_wtune_case.py
def test_cli_via_wtune_subcommand(conus_namelist, tmp_path):
    """The actual documented entry point: python3 wtune.py profile <case_dir> ..."""
    out_json = os.path.join(str(tmp_path), 'profile2.json')
    case_dir = os.path.dirname(conus_namelist)
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, 'wtune.py'), 'profile', case_dir,
         '--cores-per-node', '72', '--json', out_json],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=HERE, timeout=60)
    assert result.returncode == 0, result.stderr.decode()
    with open(out_json) as fh:
        payload = json.load(fh)
    for key in wc.PROFILE_KEYS:
        assert key in payload


# from tests/test_wtune_compliance.py
    def test_knobs_strict_hides_physics_gray_labels(self):
        namelist = (
            "&time_control\n"
            " run_hours = 6,\n"
            " history_interval = 60,\n"
            " io_form_history = 2,\n"
            " io_form_restart = 2,\n"
            "/\n"
            "&domains\n"
            " e_we = 100,\n"
            " e_sn = 100,\n"
            " e_vert = 50,\n"
            " dx = 2500.0,\n"
            " dy = 2500.0,\n"
            " time_step = 15,\n"
            " max_dom = 1,\n"
            " nproc_x = 8,\n"
            " nproc_y = 56,\n"
            "/\n"
            "&physics\n"
            " mp_physics = 8,\n"
            " cu_physics = 1,\n"
            " radt = 5,\n"
            " sf_sfclay_physics = 1,\n"
            " sf_surface_physics = 2,\n"
            " bl_pbl_physics = 1,\n"
            " num_soil_layers = 4,\n"
            "/\n"
            "&dynamics\n"
            " non_hydrostatic = .true.,\n"
            "/\n"
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'namelist.input')
            with open(path, 'w') as f:
                f.write(namelist)
            out_all = subprocess.check_output(
                [sys.executable, os.path.join(ROOT, 'wtune.py'), 'knobs', '--namelist', path, '--all'],
                cwd=ROOT).decode()
            # sanity: this namelist DOES trigger a RESULTS-tagged rule (cu_physics at dx=2.5km).
            self.assertIn('cu_physics', out_all)
            out_strict = subprocess.check_output(
                [sys.executable, os.path.join(ROOT, 'wtune.py'), 'knobs', '--namelist', path,
                 '--all', '--compliance', 'strict'],
                cwd=ROOT).decode()
        self.assertNotIn('[PHYSICS-GRAY]', out_strict)
        self.assertNotIn('[RESEARCH]', out_strict)
        self.assertIn('hidden as unclassified/disallowed at --compliance strict', out_strict)
