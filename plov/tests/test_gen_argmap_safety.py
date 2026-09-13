import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / 'gen_argmap.py'

def fixture(name, profile):
    ns = {}; exec(GEN.read_text(), ns)
    geom = ns['MICRO_GEOM'] if profile == 'micro' else ns['GEOM']
    scal = ns['MICRO_SCALARS'] if profile == 'micro' else ns['RAD_SCALARS']
    w3 = ns['MICRO_W3'] + ns['MICRO_OPT3'] + ns['MICRO_EXTRA_W3'] if profile == 'micro' else ns['RAD_W3'] + ns['RAD_EXTRA']
    w2 = ns['MICRO_W2'] + ns['MICRO_EXTRA_W2'] if profile == 'micro' else ns['RAD_W2']
    args = sorted(set(geom + scal + w3 + w2))
    lines = ['      SUBROUTINE %s(%s)' % (name, ', '.join(args))]
    lines += ['      REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: %s' % x for x in sorted(set(w3))]
    lines += ['      REAL, DIMENSION(ims:ime,jms:jme) :: %s' % x for x in sorted(set(w2))]
    lines += ['      INTEGER :: %s' % x for x in sorted(set(geom + scal) - set(w3) - set(w2))]
    return '\n'.join(lines) + '\n'

def run(src, sub, out):
    return subprocess.run([sys.executable, str(GEN), str(src), sub, str(out)], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

class GenArgmapSafetyTests(unittest.TestCase):
    def test_unsupported_cumulus_diagnostic_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); src = d / 'cu.f90'; src.write_text('SUBROUTINE cumulus_driver(a)\nEND\n')
            out = d / 'existing.h'; out.write_bytes(b'keep me\n'); digest = hashlib.sha256(out.read_bytes()).digest()
            result = run(src, 'cumulus_driver', out)
            self.assertNotEqual(result.returncode, 0); self.assertIn('unsupported subroutine profile', result.stderr)
            self.assertEqual(out.read_bytes(), b'keep me\n'); self.assertEqual(hashlib.sha256(out.read_bytes()).digest(), digest)

    def test_invalid_and_missing_inputs_preserve_output(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); out = d / 'existing.h'; out.write_bytes(b'sentinel')
            cases = [(d / 'bad.F', 'SUBROUTINE microphysics_driver(a)\n'), (d / 'bad.f90', '#define X 1\nSUBROUTINE microphysics_driver(a)\n'), (d / 'missing.f90', None)]
            for src, content in cases:
                if content is not None: src.write_text(content)
                result = run(src, 'microphysics_driver', out)
                self.assertNotEqual(result.returncode, 0); self.assertEqual(out.read_bytes(), b'sentinel')

    def test_valid_fixture_with_F_suffix_is_rejected_without_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); src = d / 'valid.F'; src.write_text(fixture('microphysics_driver', 'micro'))
            out = d / 'existing.h'; out.write_bytes(b'sentinel')
            result = run(src, 'microphysics_driver', out)
            self.assertNotEqual(result.returncode, 0); self.assertIn('compiled .f90', result.stderr); self.assertEqual(out.read_bytes(), b'sentinel')

    def test_valid_fixture_with_cpp_directive_is_rejected_without_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); src = d / 'valid.f90'; src.write_text('#define X 1\n' + fixture('microphysics_driver', 'micro'))
            out = d / 'existing.h'; out.write_bytes(b'sentinel')
            result = run(src, 'microphysics_driver', out)
            self.assertNotEqual(result.returncode, 0); self.assertIn('preprocessor directive', result.stderr); self.assertEqual(out.read_bytes(), b'sentinel')

    def test_line_marker_is_allowed_and_first_sorted_argument_is_literal_zero(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); src = d / 'marked.f90'; src.write_text('# 1 "compiled.f90"\n' + fixture('microphysics_driver', 'micro'))
            out = d / 'marked.h'; result = run(src, 'microphysics_driver', out)
            self.assertEqual(result.returncode, 0, result.stderr); self.assertIn('#define WT_A_CHANNEL_SWITCH 0', out.read_text())

    def test_supported_profiles_generate_expected_literals(self):
        for sub, profile, macro, needle in [('microphysics_driver', 'micro', 'WT', '#define WT_A_TH'), ('radiation_driver', 'rad', 'WTR', '#define WTR_A_DT')]:
            with tempfile.TemporaryDirectory() as td:
                d = Path(td); src = d / (profile + '.f90'); src.write_text(fixture(sub, profile)); out = d / (profile + '.h')
                result = run(src, sub, out)
                self.assertEqual(result.returncode, 0, result.stderr); text = out.read_text()
                self.assertIn('#define %s_NARG' % macro, text); self.assertIn(needle, text); self.assertIn('%s_argname' % macro.lower(), text)

if __name__ == '__main__':
    unittest.main()
