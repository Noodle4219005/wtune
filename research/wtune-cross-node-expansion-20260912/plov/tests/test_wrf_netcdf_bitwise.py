#!/usr/bin/env python3
"""Direct-libnetcdf tests for the WRF bitwise comparator."""

import ctypes
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/wrf_netcdf_bitwise.py"
ARTIFACT = ROOT / "artifacts/20260913-node-local-physics/netcdf-bitwise"
LIBRARY = Path("/home/u5727520/wrf-stack/intel/lib/libnetcdf.so")
NC_CLOBBER = 0
NC_64BIT_OFFSET = 0x0200
NC_NETCDF4 = 0x1000
NC_UNLIMITED = 0
NC_GLOBAL = -1
NC_CHAR, NC_INT, NC_FLOAT, NC_DOUBLE, NC_STRING = 2, 4, 5, 6, 12


class Writer:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        ip = ctypes.POINTER(ctypes.c_int)
        sp = ctypes.POINTER(ctypes.c_size_t)
        self.lib.nc_strerror.argtypes = [ctypes.c_int]
        self.lib.nc_strerror.restype = ctypes.c_char_p
        self.lib.nc_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ip]
        self.lib.nc_close.argtypes = [ctypes.c_int]
        self.lib.nc_def_dim.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ip]
        self.lib.nc_def_var.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_int, ip, ip]
        self.lib.nc_def_grp.argtypes = [ctypes.c_int, ctypes.c_char_p, ip]
        self.lib.nc_put_att_text.argtypes = [ctypes.c_int, ctypes.c_int,
                                             ctypes.c_char_p, ctypes.c_size_t,
                                             ctypes.c_char_p]
        self.lib.nc_put_att.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                        ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        self.lib.nc_enddef.argtypes = [ctypes.c_int]
        self.lib.nc_put_vara.argtypes = [ctypes.c_int, ctypes.c_int, sp, sp,
                                         ctypes.c_void_p]

    def check(self, status, operation):
        if status:
            detail = self.lib.nc_strerror(status).decode("utf-8", "replace")
            raise RuntimeError("%s: %s" % (operation, detail))

    def text_attribute(self, ncid, varid, name, value):
        raw = value.encode("ascii")
        self.check(self.lib.nc_put_att_text(ncid, varid, name.encode("ascii"),
                                            len(raw), raw), "put text attribute")

    def create(self, path, mode, float_bits=None, times=None, field_type=NC_FLOAT,
               y_length=2, omit_flag=False, history="one", child_group=False,
               string_variable=False, variable_attr_bits=0x3f000000,
               empty_time=False, title="WRF synthetic state"):
        ncid = ctypes.c_int()
        self.check(self.lib.nc_create(os.fsencode(str(path)), NC_CLOBBER | mode,
                                      ctypes.byref(ncid)), "create")
        try:
            dims = {}
            for name, length in (("Time", NC_UNLIMITED), ("DateStrLen", 19),
                                 ("south_north", y_length), ("west_east", 2)):
                dimid = ctypes.c_int()
                self.check(self.lib.nc_def_dim(ncid.value, name.encode("ascii"), length,
                                               ctypes.byref(dimid)), "define dimension")
                dims[name] = dimid.value
            variables = {}
            variables["Times"] = self.define_var(ncid.value, "Times", NC_CHAR,
                                                   [dims["Time"], dims["DateStrLen"]])
            variables["T"] = self.define_var(ncid.value, "T", field_type,
                                               [dims["Time"], dims["south_north"],
                                                dims["west_east"]])
            if not omit_flag:
                variables["FLAG"] = self.define_var(ncid.value, "FLAG", NC_INT,
                                                      [dims["south_north"], dims["west_east"]])
            if string_variable:
                variables["Label"] = self.define_var(ncid.value, "Label", NC_STRING,
                                                       [dims["Time"]])
            self.text_attribute(ncid.value, NC_GLOBAL, "TITLE", title)
            self.text_attribute(ncid.value, NC_GLOBAL, "history", history)
            self.text_attribute(ncid.value, variables["T"], "units", "K")
            attr = (ctypes.c_ubyte * 4).from_buffer_copy(
                struct.pack("=I", variable_attr_bits))
            self.check(self.lib.nc_put_att(ncid.value, variables["T"], b"scale",
                                           NC_FLOAT, 1, attr), "put numeric attribute")
            if child_group:
                group = ctypes.c_int()
                self.check(self.lib.nc_def_grp(ncid.value, b"child", ctypes.byref(group)),
                           "define child group")
            self.check(self.lib.nc_enddef(ncid.value), "end define mode")
            bits = float_bits or [0x3f800000, 0x00000000, 0x7fc00001, 0x40000000]
            if not empty_time:
                times = (times or "2026-01-01_00:00:00").encode("ascii")
                self.write(ncid.value, variables["Times"], (0, 0), (1, 19), times)
                if field_type == NC_FLOAT:
                    payload = b"".join(struct.pack("=I", value) for value in bits)
                else:
                    payload = b"".join(struct.pack("=d", float(index + 1))
                                        for index in range(2 * y_length))
                self.write(ncid.value, variables["T"], (0, 0, 0),
                           (1, y_length, 2), payload)
            if not omit_flag:
                flags = struct.pack("=%di" % (2 * y_length),
                                    *range(10, 10 + 2 * y_length))
                self.write(ncid.value, variables["FLAG"], (0, 0),
                           (y_length, 2), flags)
        finally:
            self.check(self.lib.nc_close(ncid.value), "close")

    def define_var(self, ncid, name, xtype, dims):
        varid = ctypes.c_int()
        dimids = (ctypes.c_int * max(1, len(dims)))(*dims)
        self.check(self.lib.nc_def_var(ncid, name.encode("ascii"), xtype, len(dims),
                                       dimids, ctypes.byref(varid)), "define variable")
        return varid.value

    def write(self, ncid, varid, start, count, payload):
        starts = (ctypes.c_size_t * len(start))(*start)
        counts = (ctypes.c_size_t * len(count))(*count)
        buffer = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        self.check(self.lib.nc_put_vara(ncid, varid, starts, counts, buffer),
                   "write variable")


class WrfNetcdfBitwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACT.mkdir(parents=True, exist_ok=True)
        cls.writer = Writer(LIBRARY)

    def make(self, name, mode, **kwargs):
        path = ARTIFACT / name
        self.writer.create(path, mode, **kwargs)
        return path

    def compare(self, name, left, right):
        report = ARTIFACT / (name + ".json")
        if report.exists():
            report.unlink()
        command = [sys.executable, str(TOOL), "--library", str(LIBRARY),
                   "--report", str(report), str(left), str(right)]
        result = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True,
                                timeout=60)
        return result, json.loads(report.read_text())

    def test_identical_cross_format_ignores_only_named_global_metadata(self):
        left = self.make("equal-cdf2.nc", NC_64BIT_OFFSET, history="classic history")
        right = self.make("equal-netcdf4.nc", NC_NETCDF4, history="hdf5 history")
        result, report = self.compare("equal", left, right)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(report["result"], "identical")
        self.assertEqual(report["statistics"]["compared_variables"], 3)
        self.assertGreater(report["statistics"]["compared_bytes"], 0)
        self.assertEqual(report["ignored_global_attributes"]["names"],
                         ["_NCProperties", "history"])
        self.assertIn("history", report["ignored_global_attributes"]["present_left"])
        self.assertNotEqual(report["left"]["sha256"], report["right"]["sha256"])

    def test_every_bit_sensitive_case_is_different(self):
        base = self.make("base.nc", NC_64BIT_OFFSET)
        cases = {
            "ulp": dict(float_bits=[0x3f800001, 0, 0x7fc00001, 0x40000000]),
            "signed-zero": dict(float_bits=[0x3f800000, 0x80000000,
                                             0x7fc00001, 0x40000000]),
            "nan-payload": dict(float_bits=[0x3f800000, 0, 0x7fc00002, 0x40000000]),
            "times": dict(times="2026-01-01_00:00:01"),
            "type": dict(field_type=NC_DOUBLE),
            "dimensions": dict(y_length=3,
                               float_bits=[0x3f800000, 0, 0x7fc00001,
                                           0x40000000, 0x40400000, 0x40800000]),
            "missing-variable": dict(omit_flag=True),
            "variable-attribute": dict(variable_attr_bits=0x3f000001),
            "global-attribute": dict(title="changed WRF state"),
        }
        for name, options in cases.items():
            with self.subTest(name=name):
                changed = self.make(name + ".nc", NC_NETCDF4, **options)
                result, report = self.compare(name, base, changed)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(report["result"], "different")
                self.assertIsNotNone(report["first_difference"])

    def test_unreadable_unsupported_and_grouped_inputs_are_errors(self):
        base = self.make("error-base.nc", NC_64BIT_OFFSET)
        truncated = ARTIFACT / "truncated.nc"
        truncated.write_bytes(base.read_bytes()[:16])
        string_file = self.make("string.nc", NC_NETCDF4, string_variable=True)
        grouped = self.make("grouped.nc", NC_NETCDF4, child_group=True)
        empty = self.make("empty.nc", NC_NETCDF4, empty_time=True)
        for name, candidate in (("truncated", truncated), ("string", string_file),
                                ("grouped", grouped), ("empty", empty)):
            with self.subTest(name=name):
                result, report = self.compare("error-" + name, candidate, candidate)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(report["result"], "error")
                self.assertIn("error", report)

    def test_slab_planner_never_exceeds_eight_mib(self):
        spec = importlib.util.spec_from_file_location("wrf_netcdf_bitwise", str(TOOL))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        planned = list(module.slabs((3, 4097, 1025), 8))
        self.assertGreater(len(planned), 1)
        self.assertTrue(all(size <= 8 * 1024 * 1024 for _, _, size in planned))
        self.assertEqual(sum(size for _, _, size in planned), 3 * 4097 * 1025 * 8)


if __name__ == "__main__":
    unittest.main()
