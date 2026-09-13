import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class KfLiveTests(unittest.TestCase):
    def test_live_scope_runtime(self):
        cc = shlex.split(os.environ.get("CC", "cc"))
        with tempfile.TemporaryDirectory(prefix="wt-kf-live-") as tmp:
            executable = str(Path(tmp) / "fixture")
            command = cc + shlex.split(os.environ.get("CFLAGS", "")) + [
                "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
                "-Iplov", "plov/wt_live_spans.c", "plov/wt_kf_capacity.c",
                "plov/wt_kf_dispatch.c", "plov/wt_cumulus.c", "plov/wt_kf_live.c",
                "plov/tests/test_kf_live.c", "-o", executable]
            built = subprocess.run(command, cwd=ROOT, capture_output=True,
                                   text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            ran = subprocess.run([executable], capture_output=True, text=True, timeout=60)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn("KF live scope checks passed", ran.stdout)


if __name__ == "__main__":
    unittest.main()
