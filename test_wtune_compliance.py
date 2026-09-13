import copy
import os
import re
import subprocess
import sys
import tempfile
import unittest

from wtune_compliance import (
    CATALOG, GATE_FOR_NUMERICS, LEVELS, PHYSICS_GRAY, REQUIRED_FIELDS, RESEARCH,
    STRICT, UNCLASSIFIED, admits, classify, validate,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

# Params the knob advisor (wtune_knobs.py) emits today -- all PHYSICS-GRAY.
KNOBS_PARAMS = [
    'num_soil_layers', 'sf_sfclay_physics', 'km_opt', 'diff_opt', 'o3input', 'cu_physics',
    'radt', 'diff_6th_opt', 'non_hydrostatic', 'mp_physics', 'w_damping', 'epssm', 'gwd_opt',
    'sst_update', 'tmn_update', 'bucket_mm', 'prec_acc_dt', 'fractional_seaice',
    'slope_rad/topo_shading', 'sf_urban_physics',
]

# Params the physics-candidate screener (wtune_physics.py _effective_opinion) emits today.
PHYSICS_CANDIDATE_PARAMS = [
    'io_form_history', 'io_form_restart', 'io_form_input', 'numtiles', 'nproc_x', 'nproc_y',
    'swint_opt', 'radt', 'time_step', 'mp_physics', 'h_sca_adv_order', 'h_mom_adv_order',
    'use_adaptive_time_step', 'non_hydrostatic', 'diff_opt', 'cudt', 'epssm', 'cu_physics',
    'e_vert',
]


def _entry(name):
    for e in CATALOG:
        if e['name'] == name:
            return copy.deepcopy(e)
    raise KeyError(name)


def _extract_knobs_params():
    text = open(os.path.join(ROOT, 'wtune_knobs.py')).read()
    return set(re.findall(r"_finding\('([^']+)'", text))


def _extract_physics_params():
    text = open(os.path.join(ROOT, 'wtune_physics.py')).read()
    m = re.search(r"def _effective_opinion\(.*?\n(?=def )", text, re.S)
    self_assert = m is not None
    block = m.group(0) if m else ''
    names = set(re.findall(r"name == '([A-Za-z0-9_]+)'", block))
    for group in re.findall(r"name in \(([^)]*)\)", block):
        for part in group.split(','):
            part = part.strip().strip("'").strip('"')
            if part:
                names.add(part)
    return names, self_assert


class CatalogValidationTests(unittest.TestCase):
    def test_catalog_is_valid_and_complete(self):
        errs = validate()
        self.assertEqual(errs, [])
        for e in CATALOG:
            for field in REQUIRED_FIELDS:
                self.assertIn(field, e, 'entry {!r} missing {}'.format(e.get('name'), field))

    def test_rejects_execution_interposition_on_strict(self):
        e = _entry('numtiles')
        self.assertEqual(e['classification'], STRICT)
        e['execution_interposition'] = True
        errs = validate([e])
        self.assertTrue(any('execution_interposition requires RESEARCH' in m for m in errs), errs)

    def test_rejects_source_change_on_physics_gray(self):
        e = _entry('mp_physics')
        self.assertEqual(e['classification'], PHYSICS_GRAY)
        e['source_change'] = True
        errs = validate([e])
        self.assertTrue(any('source_change requires RESEARCH' in m for m in errs), errs)

    def test_rejects_strict_with_solution_numerics(self):
        e = _entry('numtiles')
        self.assertEqual(e['classification'], STRICT)
        e['numerics_change'] = 'solution'
        e['requires_correctness_gate'] = GATE_FOR_NUMERICS['solution']
        errs = validate([e])
        self.assertTrue(any('STRICT cannot change the solution' in m for m in errs), errs)

    def test_rejects_physics_gray_without_solution_numerics(self):
        e = _entry('mp_physics')
        self.assertEqual(e['classification'], PHYSICS_GRAY)
        e['numerics_change'] = 'none'
        e['requires_correctness_gate'] = GATE_FOR_NUMERICS['none']
        errs = validate([e])
        self.assertTrue(any('PHYSICS-GRAY must declare numerics_change=solution' in m for m in errs), errs)

    def test_rejects_gate_mismatched_with_numerics_change(self):
        e = _entry('mp_physics')
        self.assertEqual(e['numerics_change'], 'solution')
        e['requires_correctness_gate'] = 'diffwrf'
        errs = validate([e])
        self.assertTrue(any('does not match numerics_change' in m for m in errs), errs)

    def test_rejects_missing_field(self):
        e = _entry('numtiles')
        del e['kind']
        errs = validate([e])
        self.assertTrue(any('missing fields' in m and 'kind' in m for m in errs), errs)

    def test_rejects_duplicate_name(self):
        e1 = _entry('numtiles')
        e2 = _entry('numtiles')
        errs = validate([e1, e2])
        self.assertTrue(any('duplicate name' in m for m in errs), errs)


class ClassifyTests(unittest.TestCase):
    def test_known_names(self):
        self.assertEqual(classify('numtiles'), STRICT)
        self.assertEqual(classify('mp_physics'), PHYSICS_GRAY)
        self.assertEqual(classify('plov_micro_staging'), RESEARCH)

    def test_wildcard_and_combined_names(self):
        self.assertEqual(classify('I_MPI_ADJUST_ALLREDUCE'), STRICT)
        self.assertEqual(classify('slope_rad/topo_shading'), PHYSICS_GRAY)

    def test_unknown_name_is_unclassified(self):
        self.assertEqual(classify('definitely_not_a_knob'), UNCLASSIFIED)


class AdmitsTests(unittest.TestCase):
    def test_strict_admits_only_strict(self):
        self.assertTrue(admits('strict', STRICT))
        self.assertFalse(admits('strict', PHYSICS_GRAY))
        self.assertFalse(admits('strict', RESEARCH))

    def test_physics_gray_admits_strict_and_physics_gray(self):
        self.assertTrue(admits('physics-gray', STRICT))
        self.assertTrue(admits('physics-gray', PHYSICS_GRAY))
        self.assertFalse(admits('physics-gray', RESEARCH))

    def test_research_admits_all_three(self):
        for cls in (STRICT, PHYSICS_GRAY, RESEARCH):
            self.assertTrue(admits('research', cls))

    def test_unclassified_never_admitted(self):
        for level in LEVELS:
            self.assertFalse(admits(level, UNCLASSIFIED))

    def test_unknown_level_raises(self):
        with self.assertRaises(ValueError):
            admits('bogus-level', STRICT)


class AdvisorCoverageTests(unittest.TestCase):
    def test_knobs_params_all_classified(self):
        for name in KNOBS_PARAMS:
            self.assertNotEqual(classify(name), UNCLASSIFIED, name)

    def test_physics_candidate_params_all_classified(self):
        for name in PHYSICS_CANDIDATE_PARAMS:
            self.assertNotEqual(classify(name), UNCLASSIFIED, name)

    def test_knobs_params_match_source(self):
        found = _extract_knobs_params()
        self.assertEqual(found, set(KNOBS_PARAMS),
                          'wtune_knobs.py _finding() names vs hard-coded KNOBS_PARAMS differ: '
                          'extra-in-source={} extra-in-list={}'.format(
                              found - set(KNOBS_PARAMS), set(KNOBS_PARAMS) - found))

    def test_physics_candidate_params_match_source(self):
        found, matched = _extract_physics_params()
        self.assertTrue(matched, 'could not locate _effective_opinion() body in wtune_physics.py')
        self.assertEqual(found, set(PHYSICS_CANDIDATE_PARAMS),
                          'wtune_physics.py _effective_opinion() names vs hard-coded '
                          'PHYSICS_CANDIDATE_PARAMS differ: extra-in-source={} extra-in-list={}'.format(
                              found - set(PHYSICS_CANDIDATE_PARAMS), set(PHYSICS_CANDIDATE_PARAMS) - found))


class CliSmokeTests(unittest.TestCase):
    def test_compliance_validate_exits_zero(self):
        rc = subprocess.call([sys.executable, os.path.join(ROOT, 'wtune.py'), 'compliance', 'validate'],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertEqual(rc, 0)

    def test_compliance_classify_numtiles_prints_strict(self):
        out = subprocess.check_output(
            [sys.executable, os.path.join(ROOT, 'wtune.py'), 'compliance', 'classify', 'numtiles'],
            cwd=ROOT).decode()
        self.assertIn('numtiles STRICT', out)

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


if __name__ == '__main__':
    unittest.main()
