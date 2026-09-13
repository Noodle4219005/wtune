"""Compile local C/Fortran fixtures; this does not validate the native WRF ABI."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

PLOV = Path(__file__).resolve().parents[1]


class LiveSpansTests(unittest.TestCase):
    def test_c_registry(self):
        cc = shlex.split(os.environ.get("CC", "cc"))
        with tempfile.TemporaryDirectory(prefix="wt-live-spans-c-") as tmp:
            executable = str(Path(tmp) / "test")
            args = cc + ["-std=c11", "-Wall", "-Wextra", "-Werror",
                         "-DWT_TEST_C_ONLY", "-I", str(PLOV),
                         str(PLOV / "wt_live_spans.c"),
                         str(PLOV / "tests/test_live_spans.c"), "-o", executable]
            result = subprocess.run(args, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([executable], text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("live span checks passed", result.stdout)

    def test_registry_and_fortran_grid(self):
        cc = shlex.split(os.environ.get("CC", "cc"))
        fc = shlex.split(os.environ.get("FC", "/usr/bin/gfortran"))
        self.assertIsNotNone(shutil.which(cc[0]), "C compiler required")
        self.assertIsNotNone(shutil.which(fc[0]), "Fortran compiler required")
        with tempfile.TemporaryDirectory(prefix="wt-live-spans-") as tmp:
            def run(args):
                result = subprocess.run(args, cwd=tmp, text=True,
                                        capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0,
                                 " ".join(map(str, args)) + "\n" +
                                 result.stdout + result.stderr)
                return result.stdout
            cflags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(PLOV)]
            run(cc + cflags + ["-c", str(PLOV / "wt_live_spans.c"), "-o", "spans.o"])
            run(cc + cflags + ["-c", str(PLOV / "tests/test_live_spans.c"), "-o", "test.o"])
            fflags = ["-std=f2018", "-Wall", "-Wextra", "-Werror", "-fcheck=all"]
            run(fc + fflags + ["-c", str(PLOV / "tests/grid_spans_fixture.f90"), "-o", "fixture.o"])
            run(fc + fflags + ["-c", str(PLOV / "wt_grid_spans.f90"), "-o", "grid.o"])
            run(cc + ["test.o", "spans.o", "fixture.o", "grid.o",
                      "-lgfortran", "-lquadmath", "-o", "test"])
            self.assertIn("live span checks passed", run([str(Path(tmp) / "test")]))


if __name__ == "__main__":
    unittest.main()
