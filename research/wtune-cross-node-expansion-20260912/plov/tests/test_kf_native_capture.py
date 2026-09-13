"""Real C/Fortran/OpenMP glue with a fake domain; never native WRF ABI evidence."""
import ctypes
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PLOV = ROOT / "plov"
HARNESS = r'''
#include "wt_kf_native.h"
#include "wt_kf_live.h"
#include "wt_kf_cps_schema.h"
#include <assert.h>
#include <omp.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
extern void *native_fixture_grid(int);
extern void native_fixture_set_id(int);
static atomic_int calls, bad_team;
static void native(void **a) { (void)a; ++calls; }
static void check_capture(void *grid, int id, int expected) {
    wt_live_spans *spans = wt_live_spans_create(4);
    int tiles = 99;
    assert(spans && wt_live_spans_begin(spans));
    native_fixture_set_id(id);
    assert(wt_kf_capture_grid(grid, spans, 1, &tiles) == expected);
    assert(tiles == (expected ? 1 : 0));
    assert(wt_live_spans_finish(spans) == (expected ? 0 : 2));
    wt_live_spans_destroy(spans);
}
int main(int argc, char **argv) {
    struct wt_kf_live_counters counts;
    assert(argc == 2);
    int mode = atoi(argv[1]);
    if (mode == 0) {
        wt_kf_live_dispatch((void **)1, native, WT_KF_NARG);
        wt_kf_live_get_counters(&counts);
        assert(calls == 1 && counts.reason[WT_KF_LIVE_DISABLED] == 1);
        return 0;
    }
    if (mode >= 3) {
        void *grid = native_fixture_grid(1);
        if (mode == 7) {
            struct wt_kf_live_config external = {0};
            assert(wt_kf_live_configure(&external));
            assert(!wt_kf_native_configure_for_grid(1,1,0,100,100,100000,1));
            check_capture(grid, 2, 1); /* failed install leaves standalone all-grid semantics */
            return 0;
        }
        assert(!wt_kf_native_configure_for_grid(1,1,0,100,100,100000,-1));
        if (mode == 8) assert(!wt_kf_native_configure_for_grid(1,1,0,0,100,100000,2));
        int selector = mode == 4 ? 0 : mode == 5 ? 2 : 1;
        assert(wt_kf_native_configure_for_grid(1,1,0,100,100,100000,selector));
        check_capture(grid, 1, selector != 2);
        check_capture(grid, 2, selector != 1);
        if (selector) { check_capture(grid, 0, 0); check_capture(grid, -1, 0); }
        assert(!wt_kf_native_configure_for_grid(1,1,0,100,100,100000,selector == 1 ? 2 : 1));
        assert(!wt_kf_native_configure(1,1,0,100,100,100000));
        assert(!wt_kf_native_configure_for_grid(1,1,0,100,100,100000,-1));
        check_capture(grid, 1, selector != 2);
        check_capture(grid, 2, selector != 1);
        if (selector == 1) {
            native_fixture_set_id(2);
            wt_cumulus_grid_begin(grid);
            wt_kf_live_dispatch((void **)1, native, WT_KF_NARG);
            wt_kf_live_get_counters(&counts);
            assert(calls == 1 && counts.reason[WT_KF_LIVE_CAPACITY] == 1);
            wt_cumulus_grid_end();
        }
        return 0;
    }
    assert(wt_kf_native_configure(1, mode != 1, 0, 100, 100, 100000));
    assert(!wt_kf_native_configure(1, 1, 0, 100, 100, 100000));
    wt_cumulus_grid_begin(mode == 1 ? (void *)1 : native_fixture_grid(1));
    if (mode == 1) {
        wt_kf_live_dispatch((void **)1, native, WT_KF_NARG);
        wt_kf_live_get_counters(&counts);
        assert(calls == 1 && counts.reason[WT_KF_LIVE_SOURCE_ABI] == 1);
    } else {
        #pragma omp parallel num_threads(2)
        {
            if (omp_get_num_threads() != 2 || !omp_in_parallel()) ++bad_team;
            wt_kf_live_dispatch((void **)1, native, WT_KF_NARG);
        }
        if (bad_team) {
            fputs("Actual OpenMP runtime did not provide two active threads\n", stderr);
            return 77;
        }
        wt_kf_live_get_counters(&counts);
        assert(calls == 2 && counts.requested == 2 && counts.reason[WT_KF_LIVE_OMP] == 2);
    }
    wt_cumulus_grid_end();
    return 0;
}
'''


class NativeCaptureTests(unittest.TestCase):
    def run_command(self, command, directory, **kwargs):
        result = subprocess.run(command, cwd=directory, capture_output=True,
                                text=True, timeout=60, **kwargs)
        self.assertEqual(result.returncode, 0,
                         " ".join(map(str, command)) + "\n" + result.stdout + result.stderr)
        return result

    def build(self, directory, kind_flags=(), override=False):
        cc = shlex.split(os.environ.get("CC", "cc"))
        fc = shlex.split(os.environ.get("FC", "gfortran"))
        fixture = (PLOV / "tests/grid_spans_fixture.f90").read_text()
        match = re.search(r"(?ims)^module module_domain_type\s*$.*?^end module\s*$", fixture)
        self.assertIsNotNone(match, "Reusable fake module_domain_type declaration changed")
        module = match.group(0)
        self.assertEqual(module.count("  type domain\n"), 1,
                         "Reusable fake domain declaration changed")
        module = module.replace("  type domain\n", "  type domain\n    integer :: num_tiles, id\n", 1)
        (directory / "module_domain_type.f90").write_text(module + "\n")
        fflags = ["-std=f2018", "-Wall", "-Wextra", "-Werror", "-fcheck=all", "-fPIC"] + list(kind_flags)
        sources = [directory / "module_domain_type.f90", PLOV / "wt_grid_spans.f90",
                   PLOV / "wt_grid_capture.f90", PLOV / "tests/native_capture_fixture.f90"]
        if override:
            original = sources[2].read_text()
            renamed = original.replace("subroutine wt_grid_capture(", "subroutine custom_grid_capture(", 1)
            renamed = renamed.replace("subroutine wt_grid_identity(", "subroutine custom_grid_identity(", 1)
            self.assertNotEqual(renamed, original, "Capture symbol declaration changed")
            sources[2] = directory / "custom_grid_capture.f90"
            sources[2].write_text(renamed)
        objects = []
        for number, source in enumerate(sources):
            obj = f"fortran{number}.o"
            self.run_command(fc + fflags + ["-c", str(source), "-o", obj], directory)
            objects.append(obj)
        cflags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-fopenmp", "-I", str(PLOV)]
        for name in ["wt_live_spans", "wt_kf_capacity", "wt_cumulus", "wt_kf_dispatch", "wt_kf_live", "wt_kf_native"]:
            extra = ["-DWT_GRID_CAPTURE_F77=custom_grid_capture_", "-DWT_GRID_ID_F77=custom_grid_identity_"] if override and name == "wt_kf_native" else []
            self.run_command(cc + cflags + extra + ["-c", str(PLOV / f"{name}.c"), "-o", f"{name}.o"], directory)
            objects.append(f"{name}.o")
        library = directory / "capture.so"
        self.run_command(cc + ["-shared", "-fopenmp"] + objects + ["-lgfortran", "-lquadmath", "-o", str(library)], directory)
        return library, cc, cflags, objects

    def inspect_capture(self, library, accepted, override=False):
        dll = ctypes.CDLL(str(library))
        ptr, size, integer = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        dll.native_fixture_grid.argtypes = [integer]
        dll.native_fixture_grid.restype = ptr
        dll.native_fixture_interior.restype = ptr
        dll.native_fixture_set_id.argtypes = [integer]
        identity = getattr(dll, "custom_grid_identity_" if override else "wt_grid_identity_")
        identity.argtypes = [ptr, ctypes.POINTER(integer), ctypes.POINTER(integer)]
        identity.restype = None
        grid = dll.native_fixture_grid(1)
        for value in [1, 2, 0, -1]:
            dll.native_fixture_set_id(value)
            identifier, status = integer(99), integer(99)
            identity(grid, ctypes.byref(identifier), ctypes.byref(status))
            valid = accepted and value > 0
            self.assertEqual(status.value, 0 if valid else 2)
            self.assertEqual(identifier.value, value if valid else 0)
        dll.wt_live_spans_create.argtypes = [size]
        dll.wt_live_spans_create.restype = ptr
        dll.wt_live_spans_begin.argtypes = [ptr]
        dll.wt_live_spans_begin.restype = ctypes.c_uint64
        dll.wt_live_spans_finish.argtypes = [ptr]
        dll.wt_live_spans_end.argtypes = [ptr]
        dll.wt_live_spans_destroy.argtypes = [ptr]
        dll.wt_kf_capture_grid.argtypes = [ptr, ptr, size, ctypes.POINTER(integer)]
        dll.wt_live_spans_lookup.argtypes = [ptr, ctypes.c_uint64, ptr, integer, size, size, ctypes.POINTER(size)]
        context = dll.wt_live_spans_create(8)
        self.assertTrue(context)
        try:
            for tile_count in [1, 2, 0, -1]:
                generation = dll.wt_live_spans_begin(context)
                tiles = integer(99)
                result = dll.wt_kf_capture_grid(dll.native_fixture_grid(tile_count), context, 1, ctypes.byref(tiles))
                valid = accepted and tile_count > 0
                self.assertEqual(result, int(valid))
                self.assertEqual(tiles.value, tile_count if valid else 0)
                self.assertEqual(dll.wt_live_spans_finish(context), 0 if valid else 2)
                remaining = size(999)
                result = dll.wt_live_spans_lookup(context, generation, dll.native_fixture_interior(), 1, 4, 32, ctypes.byref(remaining))
                self.assertEqual(result, 0 if valid else 1)
                self.assertEqual(remaining.value, 32 if valid else 0)
                dll.wt_live_spans_end(context)
            dll.wt_live_spans_begin(context)
            tiles = integer(99)
            self.assertEqual(dll.wt_kf_capture_grid(dll.native_fixture_grid(1), context, 0, ctypes.byref(tiles)), 0)
            self.assertEqual(tiles.value, 0)
        finally:
            dll.wt_live_spans_destroy(context)

    def test_c_fortran_capture_and_actual_openmp(self):
        with tempfile.TemporaryDirectory(prefix="wt-native-capture-") as tmp:
            directory = Path(tmp)
            library, cc, cflags, objects = self.build(directory)
            self.inspect_capture(library, True)
            (directory / "harness.c").write_text(HARNESS)
            self.run_command(cc + cflags + ["harness.c"] + objects +
                             ["-lgfortran", "-lquadmath", "-o", "harness"], directory)
            environment = dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="2")
            for mode in range(9):
                self.run_command([str(directory / "harness"), str(mode)], directory, env=environment)

    def test_wide_default_real_refuses_capture(self):
        with tempfile.TemporaryDirectory(prefix="wt-native-real8-") as tmp:
            library, *_ = self.build(Path(tmp), ["-fdefault-real-8"])
            self.inspect_capture(library, False)

    def test_wide_default_integer_and_logical_refuse_capture(self):
        with tempfile.TemporaryDirectory(prefix="wt-native-int8-") as tmp:
            library, *_ = self.build(Path(tmp), ["-fdefault-integer-8"])
            self.inspect_capture(library, False)

    def test_plain_fortran_symbol_override(self):
        with tempfile.TemporaryDirectory(prefix="wt-native-symbol-") as tmp:
            library, *_ = self.build(Path(tmp), override=True)
            self.inspect_capture(library, True, override=True)


if __name__ == "__main__":
    unittest.main()
