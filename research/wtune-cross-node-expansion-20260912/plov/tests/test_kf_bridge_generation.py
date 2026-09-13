import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class KFBridgeGenerationTests(unittest.TestCase):
    def invoke(self, mutation=None, count=79):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            names = ['a%d' % i for i in range(count)]
            source = 'subroutine KF_eta_CPS(' + ','.join(names) + ')\n'
            source += 'real(4) :: ' + ','.join(names) + '\nend subroutine KF_eta_CPS\n'
            schema = dict(narg=79, ordered_args=names,
                          source_sha256=hashlib.sha256(source.encode()).hexdigest())
            if mutation:
                mutation(schema)
            (td / 'source.f90').write_text(source)
            (td / 'schema.json').write_text(json.dumps(schema))
            output = td / 'bridge.c'
            output.write_text('preserved')
            result = subprocess.run([sys.executable, str(ROOT / 'gen_kf_bridge.py'),
                '--source', str(td / 'source.f90'), '--schema', str(td / 'schema.json'),
                '--output', str(output)], capture_output=True, text=True, timeout=30)
            return result, output.read_text()

    def test_matching_audited_source_and_order_generate_bridge(self):
        result, output = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('__wrap_module_cu_kfeta_mp_kf_eta_cps_', output)
        self.assertIn('wt_kf_bridge_dispatch', output)

    def test_source_drift_is_refused_without_overwriting(self):
        result, output = self.invoke(lambda s: s.update(source_sha256='0' * 64))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('source hash', result.stderr)
        self.assertEqual(output, 'preserved')

    def test_order_drift_is_refused_even_with_matching_hash(self):
        result, output = self.invoke(lambda s: s['ordered_args'].reverse())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('argument order', result.stderr)
        self.assertEqual(output, 'preserved')

    def test_wrong_core_arity_is_refused(self):
        result, output = self.invoke(lambda s: s.update(narg=78))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('79', result.stderr)
        self.assertEqual(output, 'preserved')

    def test_source_arity_cannot_disagree_with_claimed_core_arity(self):
        result, output = self.invoke(count=78)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('79', result.stderr)
        self.assertEqual(output, 'preserved')


if __name__ == '__main__':
    unittest.main()
