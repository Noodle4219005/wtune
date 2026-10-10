import copy
import os
import re
import subprocess
import sys
import unittest

from wtune.compliance import (CATALOG, GATE_FOR_NUMERICS, LEVELS, PHYSICS_GRAY, REQUIRED_FIELDS, RESEARCH, STRICT, UNCLASSIFIED, ComplianceError, admits, assert_strict_only, classify, validate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ -> repo root
SRC = os.path.join(ROOT, "src")

# Params the physics-candidate screener (wtune/physics.py _effective_opinion) emits today.
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


def _extract_physics_params():
    text = open(os.path.join(SRC, 'wtune', 'physics.py')).read()
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


class LegalValuesTests(unittest.TestCase):
    def test_every_entry_has_legal_values(self):
        for e in CATALOG:
            self.assertIn('legal_values', e, e['name'])
            lv = e['legal_values']
            for field in ('type', 'values', 'min', 'max', 'constraints', 'source', 'verified'):
                self.assertIn(field, lv, '{} legal_values missing {}'.format(e['name'], field))

    def test_every_strict_and_physics_gray_knob_has_legal_values(self):
        for e in CATALOG:
            if e['classification'] in (STRICT, PHYSICS_GRAY):
                lv = e['legal_values']
                self.assertTrue(lv['type'], e['name'])
                self.assertTrue(lv['source'], '{} legal_values.source is empty'.format(e['name']))

    def test_mp_physics_enum_contains_6_and_8(self):
        lv = _entry('mp_physics')['legal_values']
        self.assertEqual(lv['type'], 'enum')
        self.assertIn(6, lv['values'])   # WSM6
        self.assertIn(8, lv['values'])   # Thompson

    def test_cu_physics_excludes_an_arw_fatal_value(self):
        # module_check_a_mundo.F:671-681: "Option cu_physics = 4 should not be used for ARW;
        # cu_physics = 95 is suggested" -- cu_physics=4 (scalesasscheme) is fatal for ARW.
        lv = _entry('cu_physics')['legal_values']
        self.assertEqual(lv['type'], 'enum')
        self.assertIn(4, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 16, 93, 94, 95, 96, 99])  # full Registry set has 4
        self.assertNotIn(4, lv['values'])  # but it is excluded as ARW-fatal
        self.assertIn(95, lv['values'])    # the suggested replacement stays legal

    def test_numtiles_is_int_with_min_1(self):
        lv = _entry('numtiles')['legal_values']
        self.assertEqual(lv['type'], 'int')
        self.assertEqual(lv['min'], 1)


class ValidateRejectsBadLegalValuesTests(unittest.TestCase):
    def test_rejects_empty_enum_values(self):
        e = _entry('mp_physics')
        e['legal_values'] = dict(e['legal_values'], type='enum', values=[])
        errs = validate([e])
        self.assertTrue(any('enum requires non-empty values' in m for m in errs), errs)

    def test_rejects_min_greater_than_max(self):
        e = _entry('numtiles')
        e['legal_values'] = dict(e['legal_values'], min=10, max=1)
        errs = validate([e])
        self.assertTrue(any('min > max' in m for m in errs), errs)

    def test_rejects_gain_status_with_evidence_lacking_slug(self):
        e = _entry('numtiles')
        self.assertEqual(e['status'], 'gain')
        e['evidence'] = 'no bracket reference here'
        errs = validate([e])
        self.assertTrue(any('requires evidence to contain at least one' in m for m in errs), errs)

    def test_rejects_unmeasured_status_with_evidence_not_starting_unmeasured(self):
        e = _entry('tile_sz_x')
        self.assertEqual(e['status'], 'unmeasured')
        e['evidence'] = 'not the right prefix'
        errs = validate([e])
        self.assertTrue(any("requires evidence to start with 'unmeasured'" in m for m in errs), errs)


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


class AssertStrictOnlyTests(unittest.TestCase):
    """decision 0056: the tuner's final recommended configuration may only auto-apply
    STRICT-classified namelist changes."""

    def test_strict_only_change_passes(self):
        orig = {'nproc_x': [-1], 'nproc_y': [-1], 'numtiles': [1], 'time_step': [60]}
        final = dict(orig, nproc_x=[8], nproc_y=[9], numtiles=[4])
        assert_strict_only(orig, final)      # must not raise

    def test_changing_mp_physics_raises(self):
        orig = {'nproc_x': [-1], 'mp_physics': [1]}
        final = dict(orig, mp_physics=[6])
        with self.assertRaises(ComplianceError) as cm:
            assert_strict_only(orig, final)
        self.assertIn('mp_physics', str(cm.exception))

    def test_changing_time_step_raises(self):
        orig = {'nproc_x': [-1], 'time_step': [60]}
        final = dict(orig, time_step=[45])
        with self.assertRaises(ComplianceError) as cm:
            assert_strict_only(orig, final)
        self.assertIn('time_step', str(cm.exception))

    def test_no_changes_never_raises_even_for_a_non_strict_key(self):
        orig = {'mp_physics': [6]}
        assert_strict_only(orig, dict(orig))   # identical -> nothing changed, no raise

    def test_probe_window_keys_are_exempt(self):
        orig = {'run_hours': [12], 'start_hour': [0], 'end_hour': [12]}
        final = {'run_hours': [6], 'start_hour': [0], 'end_hour': [6]}
        assert_strict_only(orig, final)        # must not raise despite the changed window

    def test_unclassified_changed_key_also_raises(self):
        orig = {'not_a_real_knob': [1]}
        final = {'not_a_real_knob': [2]}
        with self.assertRaises(ComplianceError):
            assert_strict_only(orig, final)


class AdvisorCoverageTests(unittest.TestCase):
    def test_physics_candidate_params_all_classified(self):
        for name in PHYSICS_CANDIDATE_PARAMS:
            self.assertNotEqual(classify(name), UNCLASSIFIED, name)

    def test_physics_candidate_params_match_source(self):
        found, matched = _extract_physics_params()
        self.assertTrue(matched, 'could not locate _effective_opinion() body in wtune_physics.py')
        self.assertEqual(found, set(PHYSICS_CANDIDATE_PARAMS),
                          'wtune_physics.py _effective_opinion() names vs hard-coded '
                          'PHYSICS_CANDIDATE_PARAMS differ: extra-in-source={} extra-in-list={}'.format(
                              found - set(PHYSICS_CANDIDATE_PARAMS), set(PHYSICS_CANDIDATE_PARAMS) - found))


class CliSmokeTests(unittest.TestCase):
    def test_compliance_validate_exits_zero(self):
        rc = subprocess.call([sys.executable, os.path.join(ROOT, 'src', 'wtune'), 'compliance', 'validate'],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertEqual(rc, 0)

    def test_compliance_classify_numtiles_prints_strict(self):
        out = subprocess.check_output(
            [sys.executable, os.path.join(ROOT, 'src', 'wtune'), 'compliance', 'classify', 'numtiles'],
            cwd=ROOT).decode()
        self.assertIn('numtiles STRICT', out)


if __name__ == '__main__':
    unittest.main()
