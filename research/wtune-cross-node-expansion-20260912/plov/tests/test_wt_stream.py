import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StreamTests(unittest.TestCase):
    def test_process_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / "stream")
            command = shlex.split(os.environ.get("CC", "cc")) + [
                "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT),
                str(ROOT / "wt_stream.c"), str(ROOT / "tests/test_wt_stream.c"),
                "-o", exe,
            ]
            subprocess.run(command, check=True, timeout=60)
            result = subprocess.run([exe], capture_output=True, check=True, timeout=60)
            self.assertIn(b"PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
