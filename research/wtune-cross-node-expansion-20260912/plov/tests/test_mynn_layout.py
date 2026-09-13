"""Local source-schema/layout checks only; no native Fortran or WRF execution."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

PLOV = Path(__file__).resolve().parents[1]


class MynnLayoutTests(unittest.TestCase):
    def run_command(self, argv, **kwargs):
        return subprocess.run(argv, text=True, capture_output=True, timeout=60, **kwargs)

    def test_generated_header_matches_and_schema_change_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'schema.h'
            generator = PLOV / 'gen_mynn_schema.py'
            result = self.run_command([sys.executable, str(generator), '--output', str(output)])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes(), (PLOV / 'wt_mynn_schema.h').read_bytes())
            schema = json.loads((PLOV / 'mynn_schema.json').read_text())
            rows = re.findall(r'\{"(\w+)", WT_MYNN_(\w+), (\d), (\d), WT_MYNN_(INOUT|IN|OUT),', output.read_text())
            self.assertEqual(rows, [(a['name'], a['type'].upper(), str(a['rank']),
                                    str(int(a['optional'])), a['intent'].upper())
                                   for a in schema['arguments']])
            indices = re.findall(r'WT_MYNN_(\w+) = (\d+),', output.read_text())
            self.assertEqual(indices, [(a['name'].upper(), str(a['index'])) for a in schema['arguments']])
            namespace = {'__name__': 'generator_under_test', '__file__': str(generator)}
            exec(compile(generator.read_text(), str(generator), 'exec'), namespace)
            with self.assertRaisesRegex(ValueError, 'unknown branch condition'):
                namespace['condition']('arbitrary_expression()', 'qv')
            for mutation in ('order', 'condition', 'width'):
                bad = json.loads(json.dumps(schema))
                if mutation == 'order':
                    bad['arguments'][0], bad['arguments'][1] = bad['arguments'][1], bad['arguments'][0]
                    for i, arg in enumerate(bad['arguments']):
                        arg['index'] = i
                elif mutation == 'condition':
                    bad['arguments'][0]['access_condition'] = 'arbitrary_expression()'
                else:
                    bad['arguments'][0]['byte_width'] = 4
                path = Path(tmp) / 'bad.json'
                path.write_text(json.dumps(bad))
                before = output.read_bytes()
                result = self.run_command([sys.executable, str(generator), '--schema', str(path), '--output', str(output)])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('schema SHA-256 mismatch', result.stderr)
                self.assertEqual(output.read_bytes(), before)

    def compile_and_run(self, sanitizers):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / 'mynn layout test'
            command = shlex.split(os.environ.get('CC', 'cc')) + [
                '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-pedantic',
                '-I', str(PLOV), str(PLOV / 'wt_mynn_layout.c'),
                str(PLOV / 'tests/test_mynn_layout.c'), '-o', str(executable)]
            if sanitizers:
                command += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            result = self.run_command(command)
            self.assertEqual(result.returncode, 0, result.stderr)
            env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                       UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
            result = self.run_command([str(executable)], env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('MYNN layout checks PASS', result.stdout)

    def test_layout(self):
        self.compile_and_run(False)

    def test_layout_sanitized(self):
        self.compile_and_run(True)


if __name__ == '__main__':
    unittest.main()
