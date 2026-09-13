import os
import shlex
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
NORMAL = "staged caller-owned noinit opaque unconfigured omp negative-omp fractional tiles capacity wrong-type budget capture-fail query-status query-fail prequery stale-span direct-surface nonowner missing-omp missing-query missing-capture chain-proof init-proof invalid-flag configure-active lease lease-proof0 reused-owner writer-preference active-writer-unconfigured active-writer-disabled scheme-mismatch".split()
FATAL = {
    **dict.fromkeys("zero-outer zero-surface wrong-nargs recursive-init nested-outer nested-surface bad-token omp-reentry query-reentry capture-reentry".split(), 0),
    **dict.fromkeys("guard readonly owner postquery postquery-fail postspan query-postspan reader-init dispatch-reentry token-in-native".split(), 1),
}
class SfclayLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="wt-sf-live-")
        cls.exe = Path(cls.tmp.name) / "live"
        sources = ["tests/test_sfclay_live.c", "wt_sfclay_live.c", "wt_sfclay_capacity.c", "wt_sfclay.c", "wt_live_spans.c"]
        if (ROOT / "plov/wt_sfclay_init_lease.c").exists():
            sources.append("wt_sfclay_init_lease.c")
        cmd = shlex.split(os.environ["CC"]) + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", "-g", "-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-Iplov"]
        cmd += ["plov/" + s for s in sources] + ["-Wl,--wrap=pthread_cond_wait", "-o", str(cls.exe)]
        r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=60)
        if r.returncode:
            raise AssertionError(r.stdout + r.stderr)
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    def test_normal(self):
        for case in NORMAL:
            with self.subTest(case=case):
                r = subprocess.run([str(self.exe), case], text=True, capture_output=True, timeout=5)
                self.assertEqual(r.returncode, 0, case + "\n" + r.stdout + r.stderr)
                self.assertIn("PASS", r.stdout)
    def test_fatal_without_replay(self):
        for case, calls in FATAL.items():
            with self.subTest(case=case):
                r = subprocess.run([str(self.exe), case], text=True, capture_output=True, timeout=5)
                self.assertEqual(r.returncode, -signal.SIGABRT, case + "\n" + r.stdout + r.stderr)
                self.assertEqual(r.stdout.count("native\n"), calls)
                self.assertNotIn("CHECK", r.stderr)
                self.assertNotIn("Sanitizer", r.stderr)
if __name__ == "__main__":
    unittest.main()
