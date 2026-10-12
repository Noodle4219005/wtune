# The wtune_knobs.py <-> compliance-catalog parity checks, cut verbatim from
# tests/test_wtune_compliance.py (AdvisorCoverageTests) at 33a1ef7. Not runnable as is.


# Params the knob advisor (wtune_knobs.py) emits today -- all PHYSICS-GRAY.
KNOBS_PARAMS = [
    'num_soil_layers', 'sf_sfclay_physics', 'km_opt', 'diff_opt', 'o3input', 'cu_physics',
    'radt', 'diff_6th_opt', 'non_hydrostatic', 'mp_physics', 'w_damping', 'epssm', 'gwd_opt',
    'sst_update', 'tmn_update', 'bucket_mm', 'prec_acc_dt', 'fractional_seaice',
    'slope_rad/topo_shading', 'sf_urban_physics',
]


def _extract_knobs_params():
    text = open(os.path.join(SRC, 'wtune_knobs.py')).read()
    return set(re.findall(r"_finding\('([^']+)'", text))


    def test_knobs_params_all_classified(self):
        for name in KNOBS_PARAMS:
            self.assertNotEqual(classify(name), UNCLASSIFIED, name)


    def test_knobs_params_match_source(self):
        found = _extract_knobs_params()
        self.assertEqual(found, set(KNOBS_PARAMS),
                          'wtune_knobs.py _finding() names vs hard-coded KNOBS_PARAMS differ: '
                          'extra-in-source={} extra-in-list={}'.format(
                              found - set(KNOBS_PARAMS), set(KNOBS_PARAMS) - found))
