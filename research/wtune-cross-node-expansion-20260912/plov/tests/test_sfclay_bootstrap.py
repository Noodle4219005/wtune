import csv
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/20260913-full-scope/sf-native-rewrite"
COMMON_FLAGS = [
    "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
    "-Wformat=2", "-Wstrict-prototypes", "-Wmissing-prototypes",
    "-Wshadow", "-Wconversion", "-Wsign-conversion",
    "-g", "-O1", "-fsanitize=address,undefined",
    "-fno-sanitize-recover=undefined",
    "-fno-omit-frame-pointer", "-Iplov",
]


class BootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(prefix="bootstrap-", dir=ARTIFACTS)
        cls.build = Path(cls.tmp.name)
        cls.env = os.environ.copy()
        for name in ("TMPDIR", "TMP", "TEMP"):
            cls.env[name] = str(ARTIFACTS)
        cls.env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"
        cls.env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        cls.cc = shlex.split(os.environ["CC"])
        cls.executables = {
            "proof0": cls._compile("proof0", []),
            "source": cls._compile("source", ["-DWT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED=1"]),
            "caller": cls._compile("caller", [
                "-DWT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED=1",
                "-DWT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED=1",
            ]),
            "proof1": cls._compile("proof1", [
                "-DWT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED=1",
                "-DWT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED=1",
                "-DWT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED=1",
                '-DWT_SFCLAY_BUILD_PROFILE_ID="fixture,profile"',
            ]),
            "unsafe_profile": cls._compile("unsafe_profile", [
                "-DWT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED=1",
                "-DWT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED=1",
                "-DWT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED=1",
                '-DWT_SFCLAY_BUILD_PROFILE_ID="bad,\\\"quote\\\"\\n# forged"',
            ]),
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def _compile(cls, name, defines, expect_success=True):
        exe = cls.build / name
        command = cls.cc + COMMON_FLAGS + defines + [
            "plov/tests/sfclay_bootstrap_fixture.c",
            "plov/wt_sfclay_bootstrap.c", "-o", str(exe),
        ]
        result = subprocess.run(
            command, cwd=ROOT, env=cls.env, text=True, capture_output=True,
            timeout=60,
        )
        if expect_success and result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return exe if expect_success else result

    def run_case(self, build, case, extra_env=None):
        env = self.env.copy()
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(
            [str(self.executables[build]), case], cwd=ROOT, env=env,
            text=True, capture_output=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Sanitizer", result.stderr)
        self.assertNotIn("runtime error:", result.stderr)
        return result.stdout

    def test_proof_macros_reject_noncanonical_and_inconsistent_builds(self):
        bad = [
            ["-DWT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED=2"],
            ["-DWT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED=0",
             "-DWT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED=1"],
            ["-DWT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED=0",
             "-DWT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED=1"],
        ]
        for index, defines in enumerate(bad):
            with self.subTest(defines=defines):
                result = self._compile(f"rejected-{index}", defines, False)
                self.assertNotEqual(result.returncode, 0)

    def test_runtime_disabled_and_each_missing_proof(self):
        self.run_case("proof1", "before-start")
        self.run_case("proof1", "disabled")
        self.run_case("proof1", "negative-request")
        self.run_case("proof1", "invalid-selector")
        self.run_case("proof0", "expect-source-proof", {
            "WT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED": "1",
            "WT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED": "1",
            "WT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED": "1",
        })
        self.run_case("source", "expect-caller-proof")
        self.run_case("caller", "expect-initializer-proof")

    def test_strict_positive_decimal_budgets_and_exact_values(self):
        invalid = ["", "0", " 1", "+1", "-1", "1 ", "1x",
                   "184467440737095516160000"]
        for variable in ("WT_SFCLAY_MAX_SPANS", "WT_SFCLAY_MAX_NODES",
                         "WT_SFCLAY_BYTE_BUDGET"):
            for value in invalid:
                with self.subTest(variable=variable, value=value):
                    self.run_case("proof1", "invalid-budget", {variable: value})
        self.run_case("proof1", "valid-budget", {
            "WT_SFCLAY_MAX_SPANS": "17",
            "WT_SFCLAY_MAX_NODES": "19",
            "WT_SFCLAY_BYTE_BUDGET": "1048579",
        })

    def test_duplicate_and_failed_start_are_terminal(self):
        self.run_case("proof1", "duplicate")
        self.run_case("proof1", "native-rejected")

    def test_effective_state_requires_actual_staged_counter(self):
        self.run_case("proof1", "zero-requests")
        self.run_case("proof1", "post-fallback")
        self.run_case("proof1", "actual-staged")

    def test_dump_is_comment_prefixed_csv_and_null_safe(self):
        output = self.run_case("proof1", "dump")
        rows = []
        for line in output.splitlines():
            self.assertTrue(line.startswith("# "), line)
            rows.append(next(csv.reader([line[2:]])))
        values = {row[1]: row[2] for row in rows}
        self.assertEqual(values["profile"], "fixture,profile")
        self.assertEqual(values["requested"], "7")
        self.assertEqual(values["staged"], "3")
        self.assertEqual(values["fallback"], "5")
        self.assertEqual(values["fatal"], "11")
        for name in ("disabled", "source-abi", "no-scope", "omp",
                     "native-query", "tiles", "argument-abi", "capacity",
                     "prepare", "init"):
            self.assertIn("reason." + name, values)
        unsafe = self.run_case("unsafe_profile", "dump-invalid-profile")
        self.assertNotIn("\n# forged", unsafe)
        self.assertTrue(all(line.startswith("# ") for line in unsafe.splitlines()))


if __name__ == "__main__":
    unittest.main()
