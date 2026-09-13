"""Compile actual query glue against small fake modules; no native WRF ABI claim."""
import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Result(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in
                ("grid_id", "actual_scheme", "actual_fractional", "expected_scheme", "status")]


class SfclayNativeQueryTests(unittest.TestCase):
    def command(self, command, cwd):
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, " ".join(command) + "\n" + result.stdout + result.stderr)

    def build(self, scheme=91, wide=None, alternate=False):
        temp = tempfile.TemporaryDirectory(prefix="wt-sfclay-query-")
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        fc = shlex.split(os.environ.get("FC", "gfortran"))
        cc = shlex.split(os.environ.get("CC", "cc"))
        flags = ["-cpp", "-std=f2018", "-O2", "-fPIC", "-Wall", "-Wextra", "-Werror", "-fcheck=all"]
        if wide:
            flags += ["-fdefault-" + wide + "-8"]
        fixture = ROOT / "plov/tests/sfclay_native_query_fixture.f90"
        self.command(fc + flags + [f"-DMOCK_SCHEME={scheme}", "-c", str(fixture), "-o", "fixture.o"], directory)
        source = ROOT / "plov/wt_sfclay_native_query.f90"
        extra = ["-Dwt_sfclay_native_query=fixture_query_override"] if alternate else []
        self.command(fc + flags + extra + ["-c", str(source), "-o", "query-f.o"], directory)
        cflags = shlex.split(os.environ.get("CFLAGS", "")) + ["-std=c11", "-O2", "-fPIC", "-Wall", "-Wextra", "-Werror"]
        if alternate:
            cflags += ["-DWT_SFCLAY_NATIVE_QUERY_F77=fixture_query_override_"]
        self.command(cc + cflags + ["-c", str(ROOT / "plov/wt_sfclay_native_query.c"), "-o", "query-c.o"], directory)
        shared = directory / "query.so"
        self.command(cc + ["-shared", "query-c.o", "query-f.o", "fixture.o", "-lgfortran", "-lquadmath", "-o", str(shared)], directory)
        library = ctypes.CDLL(str(shared))
        library.fixture_configure.argtypes = [ctypes.c_int] * 5 + [ctypes.POINTER(ctypes.c_void_p)] * 2
        library.fixture_configure.restype = None
        library.fixture_getter_calls.restype = ctypes.c_int
        library.wt_sfclay_native_query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(Result)]
        library.wt_sfclay_native_query.restype = ctypes.c_int
        return library

    def configure(self, library, grid_id=1, scheme=91, fractional=0, model_scheme=None, model_fractional=None):
        grid, cfg = ctypes.c_void_p(), ctypes.c_void_p()
        library.fixture_configure(grid_id, scheme, fractional,
                                  scheme if model_scheme is None else model_scheme,
                                  fractional if model_fractional is None else model_fractional,
                                  ctypes.byref(grid), ctypes.byref(cfg))
        return grid, cfg

    def query(self, library, grid, cfg):
        result = Result(*([99] * 5))
        accepted = library.wt_sfclay_native_query(grid, cfg, ctypes.byref(result))
        return accepted, result

    def test_actual_configuration_and_compiled_constant(self):
        library = self.build()
        for grid_id in (1, 5):
            grid, cfg = self.configure(library, grid_id)
            accepted, result = self.query(library, grid, cfg)
            self.assertEqual(accepted, 1)
            self.assertEqual(tuple(getattr(result, name) for name, _ in Result._fields_), (grid_id, 91, 0, 91, 0))
            self.assertEqual(library.fixture_getter_calls(), 2)
        for inputs, status in (({"scheme": 92}, 5), ({"fractional": 1}, 6),
                               ({"model_scheme": 92}, 4), ({"model_fractional": 1}, 4)):
            with self.subTest(inputs=inputs):
                grid, cfg = self.configure(library, **inputs)
                accepted, result = self.query(library, grid, cfg)
                self.assertEqual((accepted, result.status), (0, status))
                self.assertEqual(library.fixture_getter_calls(), 2)

    def test_invalid_grid_never_calls_getters_and_null_inputs_refuse(self):
        library = self.build()
        for grid_id in (0, -1, 6, 2147483647):
            grid, cfg = self.configure(library, grid_id)
            accepted, result = self.query(library, grid, cfg)
            self.assertEqual((accepted, result.status), (0, 3))
            self.assertEqual(library.fixture_getter_calls(), 0)
            self.assertEqual((result.actual_scheme, result.actual_fractional), (0, 0))
        grid, cfg = self.configure(library)
        for gp, cp in ((None, cfg), (grid, None), (None, None)):
            accepted, result = self.query(library, gp, cp)
            self.assertEqual((accepted, result.status), (0, 1))
            self.assertEqual((result.grid_id, result.actual_scheme, result.actual_fractional, result.expected_scheme), (0, 0, 0, 0))
        self.assertEqual(library.wt_sfclay_native_query(grid, cfg, None), 0)
        self.assertEqual(library.fixture_getter_calls(), 0)

    def test_wide_default_kinds_refuse_before_invalid_pointer_dereference(self):
        for wide in ("real", "integer"):
            with self.subTest(wide=wide):
                library = self.build(wide=wide)
                self.configure(library)
                accepted, result = self.query(library, ctypes.c_void_p(1), ctypes.c_void_p(1))
                self.assertEqual((accepted, result.status), (0, 2))
                self.assertEqual((result.grid_id, result.actual_scheme, result.actual_fractional, result.expected_scheme), (0, 0, 0, 0))
                # Direct plain-Fortran call also initializes all outputs before its kind guard.
                outputs = [ctypes.c_int(99) for _ in range(5)]
                native = library.wt_sfclay_native_query_
                native.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_int)] * 5
                native.restype = None
                native(ctypes.c_void_p(1), ctypes.c_void_p(1), *(ctypes.byref(value) for value in outputs))
                self.assertEqual([value.value for value in outputs], [0, 0, 0, 0, 2])
                self.assertEqual(library.fixture_getter_calls(), 0)

    def test_arbitrary_compiled_scheme_and_fortran_symbol_override(self):
        library = self.build(scheme=137, alternate=True)
        grid, cfg = self.configure(library, scheme=137)
        accepted, result = self.query(library, grid, cfg)
        self.assertEqual((accepted, result.expected_scheme, result.status), (1, 137, 0))
        grid, cfg = self.configure(library, scheme=91)
        accepted, result = self.query(library, grid, cfg)
        self.assertEqual((accepted, result.actual_scheme, result.expected_scheme, result.status), (0, 91, 137, 5))


if __name__ == "__main__":
    unittest.main()
