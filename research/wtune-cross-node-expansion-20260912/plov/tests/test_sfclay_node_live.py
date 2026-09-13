#!/usr/bin/env python3
"""Strict sanitizer test for the SFCLAY production node-live chain."""

import os
from pathlib import Path
import shlex
import signal
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/20260913-node-local-physics/sfclay/live"
DEFAULT_CC = Path("/home/u5727520/wrf-stack/wtune-fortran-dev-20260912/"
                  ".pixi/envs/default/bin/x86_64-conda-linux-gnu-cc")

SOURCES = [
    "plov/tests/test_sfclay_node_live.c", "plov/wt_sfclay_node_live.c",
    "plov/wt_sfclay_native.c", "plov/wt_sfclay_native_query.c",
    "plov/wt_sfclay_live.c", "plov/wt_sfclay_init_lease.c",
    "plov/wt_sfclay_capacity.c", "plov/wt_live_spans.c",
    "plov/wt_sfclay_batch.c", "plov/wt_sfclay_batch_codec.c",
    "plov/wt_sfclay_node.c", "plov/wt_node_local.c",
    "plov/wt_kf_node_shm.c", "plov/wt_sfclay_bootstrap.c",
    "plov/wt_sfclay_dispatch.c", "plov/wt_sfclay_scope.c",
    "plov/wt_sfclay_scope_bridge.c", "plov/wt_sfclay_bridge.c",
    "plov/wt_sfclay.c",
]


def compile_fixture() -> Path:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    scratch = ARTIFACT / "scratch"
    scratch.mkdir(exist_ok=True)
    cc = shlex.split(os.environ.get("CC", "")) or [str(DEFAULT_CC)]
    executable = ARTIFACT / "test_sfclay_node_live"
    command = [*cc, *shlex.split(os.environ.get("CFLAGS", "")),
               "-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
               "-pedantic", "-fopenmp", "-pthread",
               "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
               "-fno-omit-frame-pointer", "-DWT_SFCLAY_NODE_LOCAL",
               "-DWT_SFCLAY_BUILD_SOURCE_ABI_VERIFIED=1",
               "-DWT_SFCLAY_BUILD_CALLER_CHAIN_VERIFIED=1",
               "-DWT_SFCLAY_BUILD_INITIALIZER_RELOCATION_VERIFIED=1",
               "-DWT_SFCLAY_BUILD_CONCURRENT_VERIFIED=1",
               '-DWT_SFCLAY_BUILD_PROFILE_ID="sfclay-node-live-fixture"',
               "-DWT_SFCLAY_PSIMTB_F77=test_psimtb",
               "-DWT_SFCLAY_PSIHTB_F77=test_psihtb",
               "-DWT_SFCLAY_INIT_WRAP=module_sf_sfclay_mp_sfclayinit_",
               "-DWT_SFCLAY_INIT_REAL=fixture_sfclay_init_real",
               "-DWT_SFCLAY_BRIDGE_TEST_HELPERS",
               "-DWT_SFCLAY_SCOPE_TEST_HELPERS", "-Iplov", *SOURCES,
               "-Wl,--wrap=module_first_rk_step_part1_mp_first_rk_step_part1_",
               "-Wl,--wrap=module_sf_sfclay_mp_sfclay_", "-o", str(executable)]
    environment = os.environ.copy()
    environment.update(TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    result = subprocess.run(command, cwd=ROOT, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=60)
    (ARTIFACT / "compile.txt").write_text(
        " ".join(shlex.quote(x) for x in command) + "\n" +
        result.stdout + result.stderr)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return executable


class SfclayNodeLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.executable = compile_fixture()
        cls.environment = os.environ.copy()
        cls.environment.update(
            ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:detect_leaks=1",
            UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1",
            OMP_NUM_THREADS="2", WT_SFCLAY_NODE_G0_TRACE="1",
            TMPDIR=str(ARTIFACT / "scratch"), TMP=str(ARTIFACT / "scratch"),
            TEMP=str(ARTIFACT / "scratch"),
        )

    def run_case(self, case: str, succeeds: bool = True,
                 marker: str = "production node-live fork/OMP2 chain passed") -> str:
        result = subprocess.run([str(self.executable), case], cwd=ROOT,
                                env=self.environment, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True,
                                timeout=60)
        (ARTIFACT / f"{case}.txt").write_text(
            " ".join(shlex.quote(x) for x in [str(self.executable), case]) + "\n" +
            result.stdout + result.stderr + f"returncode={result.returncode}\n")
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(marker, result.stdout)
        else:
            self.assertEqual(result.returncode, -signal.SIGABRT,
                             result.stdout + result.stderr)
        self.assertNotIn("ERROR: AddressSanitizer", result.stderr)
        self.assertNotIn("runtime error:", result.stderr)
        return result.stdout + result.stderr

    def test_cross_process_omp2_and_compatibility(self) -> None:
        output = self.run_case("normal")
        self.assertIn("event=helper", output)
        shadow = self.run_case("shadow")
        self.assertIn("shadow_checked=", shadow)
        self.assertIn("event=helper", shadow)
        owner = self.run_case("owner-only")
        self.assertIn("mode=forced-g0-owner-only", owner)
        for case in ("init-mismatch", "epoch-mismatch", "outside-scope",
                     ):
            with self.subTest(case=case):
                self.run_case(case)
        for case in ("bad-shadow", "bad-owner-only", "bad-combined"):
            with self.subTest(case=case):
                self.run_case(case,marker="malformed mode rejected")

    def test_post_native_failures_abort_without_replay(self) -> None:
        for case in ("reinit", "postnative-guard"):
            with self.subTest(case=case):
                self.run_case(case, succeeds=False)


if __name__ == "__main__":
    unittest.main()
