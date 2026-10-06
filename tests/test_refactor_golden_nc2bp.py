"""Golden call log of src/tools/wrf_nc2bp.py against a FAKE adios2 module (real adios2 is not
installed in envs/wtune, so test_wrf_nc2bp.py skips): every declare/define/open/put/step call,
the files written and the stderr warning, for a single-step, a multi-record and a split-per-record
conversion plus the error paths. Pinned before the round-3 tidy-up split convert().

Regenerate (only for an intended behaviour change):
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_nc2bp.py
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import types

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src", "tools", "wrf_nc2bp.py")
GOLDEN = os.path.join(HERE, "fixtures", "golden", "nc2bp", "calls.json")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"
SRC_OVERRIDE = os.environ.get("WTUNE_GOLDEN_SRC")     # regen from an older revision's file
DATES = ["2026-01-01_00:00:00", "2026-01-01_06:00:00", "2026-01-01_12:00:00"]


def _fmt(v):
    if isinstance(v, np.ndarray):
        return "arr(%s,%s,%s)" % (v.dtype, list(v.shape),
                                  hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:12])
    if isinstance(v, (np.generic,)):
        return "%s(%r)" % (type(v).__name__, v.item())
    if isinstance(v, (list, tuple)):
        return [_fmt(x) for x in v]
    return repr(v)


def _fake_adios2(log):
    mod = types.ModuleType("adios2")

    class Var:
        def __init__(self, name):
            self.name = name

    class Engine:
        def __init__(self, path):
            self.path = path

        def begin_step(self):
            log.append("begin_step")

        def put(self, var, value, mode):
            log.append(["put", var.name, _fmt(value), _fmt(mode)])

        def end_step(self):
            log.append("end_step")

        def close(self):
            log.append("close")

    class IO:
        def set_engine(self, name):
            log.append(["set_engine", name])

        def define_attribute(self, name, value, variable_name=None):
            log.append(["attr", name, _fmt(value), variable_name])

        def define_variable(self, name, *a, **k):
            log.append(["var", name, _fmt(list(a)), _fmt(sorted(k.items()))])
            return Var(name)

        def open(self, output, mode):
            log.append(["open", os.path.basename(output), _fmt(mode)])
            os.makedirs(output, exist_ok=True)
            return Engine(output)

    class Adios:
        def declare_io(self, name):
            log.append(["declare_io", name])
            return IO()

    mod.Adios = Adios
    mod.string_to_mode = lambda m: ("MODE_" + m, None)
    mod.bindings = types.SimpleNamespace(Mode=types.SimpleNamespace(Sync="SYNC"))
    return mod


def _write_input(path, nrec, dates):
    from netCDF4 import Dataset
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("west_east", 3)
        ds.createDimension("south_north", 2)
        ds.setncattr("TITLE", "OUTPUT FROM REAL_EM")
        ds.setncattr("DX", np.float32(3000.0))
        ds.setncattr("ITEMS", np.array([1, 2], dtype=np.int32))
        t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        for k in range(nrec):
            t[k, :] = np.array(list(dates[k]), dtype="S1")
        u = ds.createVariable("U", "f4", ("Time", "south_north", "west_east"))
        u[:] = np.arange(nrec * 6, dtype=np.float32).reshape(nrec, 2, 3)
        u.setncattr("MemoryOrder", "XY ")
        u.setncattr("description", "x wind")
        u.setncattr("FieldType", np.int32(104))
        i = ds.createVariable("MAPFAC", "i4", ("south_north", "west_east"), fill_value=-9)
        i[:] = np.arange(6, dtype=np.int32).reshape(2, 3)
        d = ds.createVariable("D", "f8", ("Time",))
        d[:] = np.arange(nrec, dtype=np.float64)
        md = ds.createVariable("md___thisbdytime", "S1", ("Time", "DateStrLen"))
        for k in range(nrec):
            md[k, :] = np.array(list(dates[k]), dtype="S1")


def _load(log):
    name = "wrf_nc2bp_golden_under_test"
    sys.modules["adios2"] = _fake_adios2(log)
    spec = importlib.util.spec_from_file_location(name, SRC_OVERRIDE or SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _collect(tmp):
    out = {}
    cases = {
        "single": (1, {}), "multi": (3, {}), "bp5": (1, dict(engine_name="BP5")),
        "split": (3, dict(split_prefix="wrfbdy_d01")), "bad-engine": (1, dict(engine_name="BP3")),
        "dup": (2, dict(split_prefix="wrfbdy_d01")),
    }
    for cid, (nrec, kw) in cases.items():
        log = []
        saved = sys.modules.get("adios2")
        try:
            mod = _load(log)
            d = tmp / cid
            d.mkdir()
            dates = DATES if cid != "dup" else [DATES[0], DATES[0]]
            _write_input(str(d / "in.nc"), nrec, dates)
            err = io.StringIO()
            res = None
            with contextlib.redirect_stderr(err):
                try:
                    res = [os.path.relpath(p, str(d)) for p in
                           mod.convert(str(d / "in.nc"), str(d / "out.bp"), **kw)]
                except Exception as e:  # noqa: BLE001 -- pinned by type + message
                    res = "%s: %s" % (type(e).__name__, str(e).replace(str(d), "<D>"))
            out[cid] = {"result": res, "stderr": err.getvalue().replace(str(d), "<D>"),
                        "log": log, "tree": sorted(os.path.relpath(os.path.join(r, f), str(d))
                                                   for r, ds, fs in os.walk(str(d / "out.bp"))
                                                   for f in ds + fs)
                        if (d / "out.bp").exists() else None}
        finally:
            if saved is None:
                sys.modules.pop("adios2", None)
            else:
                sys.modules["adios2"] = saved
    log = []
    saved = sys.modules.get("adios2")
    mod = _load(log)
    out["helpers"] = {"record_file_name": mod.record_file_name("wrfbdy_d01", DATES[0]),
                      "field_types": [int(mod._field_type_for_dtype(np.dtype(t)))
                                      for t in ("float32", "float64", "int32")]}
    try:
        mod.record_file_name("x", "short")
    except ValueError as e:
        out["helpers"]["bad_date"] = str(e)
    if saved is None:
        sys.modules.pop("adios2", None)
    else:
        sys.modules["adios2"] = saved
    return json.loads(json.dumps(out))


def test_nc2bp_call_log_matches_golden(tmp_path):
    got = _collect(tmp_path)
    if REGEN:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(got, fh, indent=1, sort_keys=True)
            fh.write("\n")
    with open(GOLDEN) as fh:
        want = json.load(fh)
    assert got == want
