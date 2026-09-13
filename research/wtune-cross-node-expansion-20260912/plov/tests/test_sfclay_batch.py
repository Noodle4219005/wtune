#!/usr/bin/env python3
"""Strict sanitizer test for independent SFCLAY row batches."""

from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT.parent / "artifacts/20260913-node-local-physics/sfclay/batch"


def run_fixture(artifact: pathlib.Path) -> None:
    artifact = artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    scratch = artifact / "scratch"
    scratch.mkdir(exist_ok=True)
    binary = artifact / "test_sfclay_batch"
    compiler = shlex.split(os.environ.get("CC", ""))
    if not compiler:
        compiler = [shutil.which("x86_64-conda-linux-gnu-cc") or "cc"]
    command = [
        *compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
        "-pthread", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
        "-fno-sanitize-recover=all", f"-I{ROOT}",
        str(ROOT / "wt_sfclay_batch.c"),
        str(ROOT / "wt_sfclay_batch_codec.c"),
        str(ROOT / "tests/test_sfclay_batch.c"), "-o", str(binary),
    ]
    environment = os.environ.copy()
    environment.update(
        ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:detect_leaks=1",
        UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1",
        TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
    )
    log_path = artifact / "test-sfclay-batch.txt"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n"); log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=60, env=environment)
        run = subprocess.run([str(binary)], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=60,
                             env=environment, check=False)
        output = run.stdout.decode("utf-8", errors="replace")
        log.write(shlex.join([str(binary)]) + "\n" + output); log.flush()
    if run.returncode or "SFCLAY row batch tests passed" not in output:
        raise AssertionError(output)


class SfclayBatchTest(unittest.TestCase):
    def test_sfclay_batch(self) -> None:
        run_fixture(DEFAULT_ARTIFACT)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_fixture(pathlib.Path(sys.argv[1]))
    else:
        unittest.main()
