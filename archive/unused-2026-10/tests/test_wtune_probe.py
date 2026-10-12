import os
import tempfile
import unittest

import wtune_probe as probe


class ProbeParsingTests(unittest.TestCase):
    def test_parse_rsl_separates_solver_io_and_mesh_evidence(self):
        text = ("Ntasks in X 8 in Y 56\n"
                "Timing for main step 0.400 elapsed\n"
                "Timing for main step 0.600 elapsed\n"
                "Timing for Writing history 0.250 elapsed\n"
                "SUCCESS COMPLETE\n")
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as handle:
            handle.write(text)
            path = handle.name
        try:
            result = probe.parse_rsl(path)
        finally:
            os.unlink(path)

        self.assertTrue(result['ok'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['solver_median_s'], 0.5)
        self.assertEqual(result['io_total_s'], 0.25)
        self.assertEqual(result['io_writes'], 1)
        self.assertEqual(result['mesh'], (8, 56))

    def test_parse_rsl_rejects_truncated_run_as_need_evidence(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as handle:
            handle.write('Timing for main step 0.400 elapsed\n')
            path = handle.name
        try:
            result = probe.parse_rsl(path)
        finally:
            os.unlink(path)

        self.assertFalse(result['ok'])
        self.assertEqual(result['status'], 'NEED-EVIDENCE')
        self.assertIn('SUCCESS COMPLETE', result['reason'])


class ProbeComparisonTests(unittest.TestCase):
    def test_compare_preserves_declared_objective_and_missing_evidence_gate(self):
        result = probe.compare([], [1.0], objective='io_total_s')

        self.assertEqual(result['objective'], 'io_total_s')
        self.assertEqual(result['verdict'], 'NEED-EVIDENCE')
        self.assertIn('missing measurements', result['reason'])


if __name__ == '__main__':
    unittest.main()
