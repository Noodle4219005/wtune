import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SfclayCapacityTests(unittest.TestCase):
    def test_capacity_provider_fixture(self):
        cc = shlex.split(os.environ["CC"])
        with tempfile.TemporaryDirectory(prefix="wt-sfclay-capacity-") as tmp:
            exe = Path(tmp) / "fixture"
            built = subprocess.run(cc + ["-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-Iplov", "plov/tests/test_sfclay_capacity.c",
                "plov/wt_sfclay_capacity.c", "plov/wt_live_spans.c", "-o", str(exe)], cwd=ROOT,
                text=True, capture_output=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            ran = subprocess.run([str(exe)], text=True, capture_output=True, timeout=60)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn("SFCLAY capacity checks passed", ran.stdout)


if __name__ == "__main__":
    unittest.main()
