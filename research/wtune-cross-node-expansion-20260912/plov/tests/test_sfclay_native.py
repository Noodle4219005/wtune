import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/20260913-full-scope/sf-native-rewrite"
FLAGS = [
    "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
    "-Wformat=2", "-Wstrict-prototypes", "-Wmissing-prototypes",
    "-Wshadow", "-Wconversion", "-Wsign-conversion",
    "-g", "-O1", "-fsanitize=address,undefined",
    "-fno-sanitize-recover=undefined",
    "-fno-omit-frame-pointer", "-fopenmp", "-pthread", "-Iplov",
]


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(prefix="native-", dir=ARTIFACTS)
        cls.build = Path(cls.tmp.name)
        cls.env = os.environ.copy()
        for name in ("TMPDIR", "TMP", "TEMP"):
            cls.env[name] = str(ARTIFACTS)
        cls.env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"
        cls.env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        cls.cc = shlex.split(os.environ["CC"])
        cls.adapter = cls._compile("adapter", [
            "plov/tests/sfclay_native_bootstrap_fixture.c",
            "plov/wt_sfclay_native.c", "plov/wt_kf_native.c",
            "plov/wt_live_spans.c",
        ])
        cls.live = cls._compile("live", [
            "plov/tests/sfclay_native_live_fixture.c",
            "plov/wt_sfclay_native.c", "plov/wt_sfclay_native_query.c",
            "plov/wt_sfclay_live.c", "plov/wt_sfclay_init_lease.c",
            "plov/wt_sfclay_capacity.c", "plov/wt_sfclay.c",
            "plov/wt_live_spans.c",
        ])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def _compile(cls, name, sources):
        exe = cls.build / name
        command = cls.cc + FLAGS + sources + ["-o", str(exe)]
        result = subprocess.run(
            command, cwd=ROOT, env=cls.env, text=True, capture_output=True,
            timeout=60,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return exe

    def run_case(self, executable, case):
        result = subprocess.run(
            [str(executable), case], cwd=ROOT, env=self.env,
            text=True, capture_output=True, timeout=8,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "PASS " + case + "\n")
        self.assertNotIn("Sanitizer", result.stderr)
        self.assertNotIn("runtime error:", result.stderr)

    def test_bounded_adapter_contract(self):
        for case in ("proof-gates", "rollback", "immutable", "selector-all",
                     "omp-callback", "kf-independent"):
            with self.subTest(case=case):
                self.run_case(self.adapter, case)

    def test_actual_live_path(self):
        for case in ("proof0", "omp", "selected-miss", "selected-staged"):
            with self.subTest(case=case):
                self.run_case(self.live, case)


if __name__ == "__main__":
    unittest.main()
