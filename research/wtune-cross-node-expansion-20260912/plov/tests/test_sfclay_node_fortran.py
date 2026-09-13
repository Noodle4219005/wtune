#!/usr/bin/env python3
"""Frozen GNU SFCLAY through an actual forked node-local helper."""

from __future__ import annotations

import hashlib
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = pathlib.Path(os.environ.get(
    "WTUNE_SFCLAY_SOURCE",
    "/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF/phys/module_sf_sfclay.F",
))
SOURCE_SHA256 = "dfd7aae5ef1f0bd979246480af5276a1ee956bc216ec8dc6e82eb963ded761dc"
SYMBOL = "__module_sf_sfclay_MOD_sfclay"
DEFAULT_ARTIFACT = ROOT / "artifacts/20260913-node-local-physics/sfclay/fortran"


def run_logged(command: list[str], log, environment: dict[str, str]) -> None:
    log.write(shlex.join(command) + "\n"); log.flush()
    subprocess.run(command, cwd=ROOT, check=True, stdout=log,
                   stderr=subprocess.STDOUT, timeout=60, env=environment)


def run_fixture(artifact: pathlib.Path) -> None:
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise AssertionError("frozen SFCLAY source hash changed")
    artifact = artifact.resolve(); artifact.mkdir(parents=True, exist_ok=True)
    scratch = artifact / "scratch"; scratch.mkdir(exist_ok=True)
    cc = shlex.split(os.environ.get("CC", "")) or [
        shutil.which("x86_64-conda-linux-gnu-cc") or "cc"]
    fc = shlex.split(os.environ.get("FC", "")) or [
        shutil.which("x86_64-conda-linux-gnu-gfortran") or "gfortran"]
    environment = os.environ.copy()
    environment.update(
        ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:detect_leaks=1",
        UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1",
        TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
    )
    bridge = artifact / "bridge.c"
    fobj = artifact / "sfclay.o"
    exe = artifact / "test_sfclay_node_fortran"
    csources = [
        "plov/tests/test_sfclay_node_fortran.c", "plov/wt_sfclay.c",
        "plov/wt_node_local.c", "plov/wt_sfclay_batch.c",
        "plov/wt_sfclay_batch_codec.c", "plov/wt_sfclay_node.c",
    ]
    objects: list[pathlib.Path] = []
    log_path = artifact / "test-sfclay-node-fortran.txt"
    with log_path.open("w", encoding="utf-8") as log:
        run_logged([sys.executable, "plov/gen_sfclay_bridge.py", "--source",
                    str(SOURCE), "--schema", "plov/sfclay_schema.json",
                    "--symbol", SYMBOL, "--output", str(bridge)], log, environment)
        run_logged([*fc, "-cpp", "-DEM_CORE=1", "-ffree-form", "-fcheck=all",
                    "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                    "-J", str(artifact), "-c", str(SOURCE), "-o", str(fobj)],
                   log, environment)
        for source in csources:
            output = artifact / (pathlib.Path(source).stem + ".o")
            objects.append(output)
            run_logged([*cc, "-std=c11", "-O1", "-g", "-Wall", "-Wextra",
                        "-Werror", "-pedantic", "-pthread",
                        "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                        "-fno-sanitize-recover=all", "-Iplov", "-c", source,
                        "-o", str(output)], log, environment)
        bridge_obj = artifact / "bridge.o"
        run_logged([*cc, "-std=c11", "-O1", "-g", "-Wall", "-Wextra",
                    "-Werror", "-pedantic", "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all", "-DWT_SFCLAY_BRIDGE_TEST_HELPERS",
                    "-Iplov", "-c", str(bridge), "-o", str(bridge_obj)],
                   log, environment)
        run_logged([*fc, "-pthread", "-fsanitize=address,undefined",
                    "-fno-sanitize-recover=all", str(fobj),
                    *(str(x) for x in objects), str(bridge_obj),
                    "-Wl,--wrap=" + SYMBOL, "-o", str(exe)], log, environment)
        command = [str(exe), str(artifact)]
        log.write(shlex.join(command) + "\n"); log.flush()
        run = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=60,
                             env=environment, check=False)
        output = run.stdout.decode("utf-8", errors="replace")
        log.write(output); log.flush()
    if run.returncode or "passed 73 cases" not in output:
        raise AssertionError(output)


class SfclayNodeFortranTest(unittest.TestCase):
    def test_frozen_sfclay_helper(self) -> None:
        run_fixture(DEFAULT_ARTIFACT)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_fixture(pathlib.Path(sys.argv[1]))
    else:
        unittest.main()
