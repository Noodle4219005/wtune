"""Compile the RK selector query against ordinary mock Fortran modules."""
import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Result(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in (
        "status", "rk_ord", "zadvect_implicit", "h_sca_adv_order",
        "v_sca_adv_order", "cu_physics", "diff_opt", "bl_pbl_physics",
        "time_step", "specified", "nested", "periodic_x", "polar",
        "open_xs", "open_xe", "open_ys", "open_ye")]


class RkNativeQueryTests(unittest.TestCase):
    def command(self, command, cwd):
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0,
                         " ".join(command) + "\n" + completed.stdout + completed.stderr)

    def build(self, wide=None, alternate=False):
        temp = tempfile.TemporaryDirectory(prefix="wt-rk-query-")
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        fc = shlex.split(os.environ.get("FC", "gfortran"))
        cc = shlex.split(os.environ.get("CC", "cc"))
        fflags = ["-cpp", "-std=f2018", "-O2", "-fPIC", "-Wall", "-Wextra", "-Werror", "-fcheck=all"]
        if wide:
            fflags += ["-fdefault-" + wide + "-8"]
        self.command(fc + fflags + ["-c", str(ROOT / "plov/tests/rk_native_query_fixture.f90"), "-o", "fixture.o"], directory)
        extra = ["-Dwt_rk_native_query=fixture_rk_query_override"] if alternate else []
        self.command(fc + fflags + extra + ["-c", str(ROOT / "plov/wt_rk_native_query.f90"), "-o", "query-f.o"], directory)
        cflags = shlex.split(os.environ.get("CFLAGS", "")) + ["-std=c11", "-O2", "-fPIC", "-Wall", "-Wextra", "-Werror"]
        if alternate:
            cflags += ["-DWT_RK_NATIVE_QUERY_F77=fixture_rk_query_override_"]
        self.command(cc + cflags + ["-c", str(ROOT / "plov/wt_rk_native_query.c"), "-o", "query-c.o"], directory)
        shared = directory / "query.so"
        self.command(cc + ["-shared", "query-c.o", "query-f.o", "fixture.o", "-lgfortran", "-lquadmath", "-o", str(shared)], directory)
        library = ctypes.CDLL(str(shared))
        library.fixture_configure.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]
        library.fixture_set_global_time_step.argtypes = [ctypes.c_int]
        library.fixture_getter_calls.restype = ctypes.c_int
        library.wt_rk_native_query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(Result)]
        library.wt_rk_native_query.restype = ctypes.c_int
        return library

    @staticmethod
    def values():
        return (ctypes.c_int * 16)(7, -3, 2, 5, 137, 9, 41, 12,
                                   -1, 2, -3, 4, -5, 6, -7, 8)

    def configure(self, library):
        cfg = ctypes.c_void_p()
        library.fixture_configure(self.values(), ctypes.byref(cfg))
        return cfg

    def query(self, library, cfg, rk_step=2):
        result = Result(*([99] * len(Result._fields_)))
        return library.wt_rk_native_query(cfg, rk_step, ctypes.byref(result)), result

    def test_selector_values_are_exact_and_logicals_normalized(self):
        library = self.build()
        cfg = self.configure(library)
        library.fixture_set_global_time_step(73)
        accepted, result = self.query(library, cfg, 11)
        self.assertEqual(accepted, 1)
        self.assertEqual(tuple(getattr(result, name) for name, _ in Result._fields_),
                         (0, 7, 1, 2, 5, 137, 9, 41, 73, 1, 1, 1, 1, 1, 1, 1, 1))
        self.assertEqual(library.fixture_getter_calls(), 1)

    def test_global_getter_change_is_visible_each_query(self):
        library = self.build()
        cfg = self.configure(library)
        library.fixture_set_global_time_step(19)
        self.assertEqual(self.query(library, cfg)[1].time_step, 19)
        library.fixture_set_global_time_step(88)
        self.assertEqual(self.query(library, cfg)[1].time_step, 88)
        self.assertEqual(library.fixture_getter_calls(), 2)

    def test_invalid_step_and_nulls_refuse_without_getters(self):
        library = self.build()
        cfg = self.configure(library)
        for step in (0, -1, -2147483648):
            accepted, result = self.query(library, cfg, step)
            self.assertEqual((accepted, result.status), (0, 3))
            self.assertEqual(tuple(getattr(result, name) for name, _ in Result._fields_[1:]), (0,) * 16)
        accepted, result = self.query(library, None)
        self.assertEqual((accepted, result.status), (0, 1))
        self.assertEqual(tuple(getattr(result, name) for name, _ in Result._fields_[1:]), (0,) * 16)
        self.assertEqual(library.wt_rk_native_query(cfg, 1, None), 0)
        self.assertEqual(library.fixture_getter_calls(), 0)

    def test_wide_kinds_refuse_before_cfg_access_and_symbol_override_works(self):
        for wide in ("integer", "real"):
            with self.subTest(wide=wide):
                library = self.build(wide=wide)
                accepted, result = self.query(library, ctypes.c_void_p(1))
                self.assertEqual((accepted, result.status), (0, 2))
                self.assertEqual(tuple(getattr(result, name) for name, _ in Result._fields_[1:]), (0,) * 16)
                self.assertEqual(library.fixture_getter_calls(), 0)
        library = self.build(alternate=True)
        cfg = self.configure(library)
        accepted, result = self.query(library, cfg)
        self.assertEqual((accepted, result.status, result.rk_ord), (1, 0, 7))


if __name__ == "__main__":
    unittest.main()
