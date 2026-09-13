import os
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class KFCPSTest(unittest.TestCase):
    def test_schema_header_is_current(self):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "gen_kf_cps_schema.py"),
                               "--check", os.path.join(ROOT, "wt_kf_cps_schema.h")], timeout=60)

    def test_c_transaction_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "fixture")
            subprocess.check_call(shlex.split(os.environ.get("CC", "cc")) +
                                  ["-std=c11", "-Wall", "-Wextra", "-Werror",
                                   "-I", ROOT, os.path.join(ROOT, "wt_cumulus.c"),
                                   os.path.join(ROOT, "tests", "test_kf_cps_stage.c"), "-o", exe], timeout=60)
            self.assertIn(b"PASS", subprocess.check_output([exe], timeout=60))

if __name__ == "__main__":
    unittest.main()
