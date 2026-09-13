import os
import shlex
import subprocess
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WireCodecTest(unittest.TestCase):
    def test_c_wire_codec_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "test_wt_wire")
            subprocess.check_call(
                shlex.split(os.environ.get("CC", "cc")) + shlex.split(os.environ.get("CFLAGS", "")) + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-I", ROOT,
                 os.path.join(ROOT, "wt_wire.c"),
                 os.path.join(ROOT, "tests", "test_wt_wire.c"), "-o", exe],
                timeout=60,
            )
            self.assertIn(b"PASS", subprocess.check_output([exe], timeout=60))


if __name__ == "__main__":
    unittest.main()
