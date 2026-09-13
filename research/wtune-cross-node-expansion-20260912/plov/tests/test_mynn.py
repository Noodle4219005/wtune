"""Synthetic owner-local C transaction fixtures; never invoke native MYNN/WRF."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

PLOV = Path(__file__).resolve().parents[1]


class MynnTransactionTests(unittest.TestCase):
    def compile_and_run(self, sanitize):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / 'mynn transaction'
            argv = shlex.split(os.environ.get('CC', 'cc')) + [
                '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-pedantic',
                '-pthread', '-fno-builtin-memcpy', '-I', str(PLOV), str(PLOV / 'wt_mynn_layout.c'),
                str(PLOV / 'wt_mynn.c'), str(PLOV / 'tests/test_mynn.c'),
                '-Wl,--wrap=malloc', '-Wl,--wrap=free', '-Wl,--wrap=memcpy', '-o', str(executable)]
            if sanitize:
                argv += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            result = subprocess.run(argv, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                       UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
            result = subprocess.run([str(executable)], text=True, capture_output=True,
                                    timeout=60, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('MYNN transaction checks PASS', result.stdout)

    def test_transaction(self):
        self.compile_and_run(False)

    def test_transaction_sanitized(self):
        self.compile_and_run(True)


if __name__ == '__main__':
    unittest.main()
