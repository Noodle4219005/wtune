import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
class KfDispatchTests(unittest.TestCase):
    def run_fixture(self, extra_flags):
        compiler = shlex.split(os.environ.get("CC", "cc"))
        with tempfile.TemporaryDirectory(prefix="wt-kf-dispatch-") as td:
            executable = str(Path(td) / "fixture")
            command = compiler + shlex.split(os.environ.get("CFLAGS", "")) + extra_flags + ["-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", "-Iplov", "plov/wt_cumulus.c", "plov/wt_kf_dispatch.c", "plov/tests/test_kf_dispatch.c", "-o", executable]
            built = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([executable], text=True, capture_output=True, timeout=60)
            self.assertEqual(ran.returncode, 0, ran.stderr)

    def test_local_dispatch_fixture(self):
        self.run_fixture([])

    def test_missing_runtime_probe_refuses_staging(self):
        self.run_fixture(["-DWT_KF_NO_OMP_HOOK"])
if __name__ == "__main__": unittest.main()
