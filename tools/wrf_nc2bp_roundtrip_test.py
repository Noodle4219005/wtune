#!/usr/bin/env python3
"""Small synthetic WRF netCDF -> BP -> netCDF bit-exact regression test."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


HERE = Path(__file__).resolve().parent
CONVERTER = HERE / "wrf_nc2bp.py"
VENDORED = Path("/home/u5727520/SC26-WRF/scripts/vendor/wrf-adios2-to-netcdf4/convert.py")


def _load_converter():
    spec = importlib.util.spec_from_file_location("wrf_bp_to_nc", VENDORED)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import vendored converter {VENDORED}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_source(path: Path) -> None:
    dates = ["2026-01-01_00:00:00", "2026-01-01_03:00:00"]
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 3)
        ds.createDimension("west_east_stag", 4)
        ds.createDimension("bottom_top", 2)
        ds.createDimension("bdy_width", 1)

        ds.setncattr("TITLE", "synthetic WRF input")
        ds.setncattr("MAP_PROJ", np.int32(3))
        ds.setncattr("DX", np.float32(3000.25))
        ds.setncattr("INTS", np.array([1, -2, 7], dtype=np.int32))
        ds.setncattr("REALS", np.array([1.5, -2.25], dtype=np.float32))

        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:, :] = np.asarray([list(x) for x in dates], dtype="S1")

        xlat = ds.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
        xlat[:] = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 10.0
        xlat.setncattr("FieldType", np.int32(104))
        xlat.setncattr("MemoryOrder", "XY ")
        xlat.setncattr("description", "latitude")
        xlat.setncattr("units", "degree_north")
        xlat.setncattr("stagger", "")
        xlat.setncattr("coordinates", "XLONG XLAT")

        u = ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))
        u[:] = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 4)
        u.setncattr("FieldType", np.int32(104))
        u.setncattr("MemoryOrder", "XYZ")
        u.setncattr("description", "x wind")
        u.setncattr("units", "m s-1")
        u.setncattr("stagger", "X")

        bdy = ds.createVariable("U_BXS", "f4", ("Time", "bdy_width", "bottom_top", "south_north"))
        bdy[:] = np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2) - 1.0
        bdy.setncattr("FieldType", np.int32(104))
        bdy.setncattr("MemoryOrder", "XSZ")
        bdy.setncattr("description", "bdy x wind")
        bdy.setncattr("units", "m s-1")
        bdy.setncattr("stagger", "X")

        integer = ds.createVariable("I_FIELD", "i4", ("Time", "south_north", "west_east"))
        integer[:] = np.arange(12, dtype=np.int32).reshape(2, 2, 3) - 4
        integer.setncattr("FieldType", np.int32(106))
        integer.setncattr("MemoryOrder", "XY ")
        integer.setncattr("description", "integer field")

        scalar = ds.createVariable("SCALAR", "f4", ("Time",))
        scalar[:] = np.array([2.5, -7.0], dtype=np.float32)
        scalar.setncattr("FieldType", np.int32(104))
        scalar.setncattr("MemoryOrder", "0")
        scalar.setncattr("description", "scalar field")

        for name, values in {
            "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_": dates,
            "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_": [
                "2026-01-01_03:00:00",
                "2026-01-01_06:00:00",
            ],
        }.items():
            var = ds.createVariable(name, "S1", ("Time", "DateStrLen"))
            var[:, :] = np.asarray([list(x) for x in values], dtype="S1")


def _attrs(ds: Dataset, variable: str | None) -> dict[str, object]:
    obj = ds if variable is None else ds.variables[variable]
    return {name: obj.getncattr(name) for name in obj.ncattrs() if name != "_FillValue"}


def _assert_attr_equal(left: object, right: object, label: str) -> None:
    if isinstance(left, str) or isinstance(right, str):
        assert left == right, f"{label}: {left!r} != {right!r}"
        return
    assert np.array_equal(np.asarray(left), np.asarray(right)), f"{label}: {left!r} != {right!r}"
    assert np.asarray(left).dtype == np.asarray(right).dtype, f"{label}: dtype changed"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="wrf_nc2bp_test_") as temp:
        root = Path(temp)
        source = root / "synthetic.nc"
        bp = root / "synthetic.bp"
        roundtrip = root / "roundtrip.nc"
        _make_source(source)

        subprocess.run(
            [sys.executable, str(CONVERTER), str(source), str(bp)],
            check=True,
        )

        reverse = _load_converter()
        reverse.convert(str(bp), str(roundtrip))

        with Dataset(source, "r") as left, Dataset(roundtrip, "r") as right:
            assert set(left.dimensions) == set(right.dimensions)
            for name in left.dimensions:
                assert len(left.dimensions[name]) == len(right.dimensions[name]), name
            assert set(left.variables) == set(right.variables)
            for name in left.variables:
                a = np.asarray(left.variables[name][:])
                b = np.asarray(right.variables[name][:])
                assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} != {b.dtype}"
                assert np.array_equal(a, b), f"{name}: data changed"
                assert left.variables[name].dimensions == right.variables[name].dimensions, name
                for attr in _attrs(left, name):
                    _assert_attr_equal(
                        left.variables[name].getncattr(attr),
                        right.variables[name].getncattr(attr),
                        f"{name}:{attr}",
                    )
            for attr in _attrs(left, None):
                _assert_attr_equal(left.getncattr(attr), right.getncattr(attr), f"global:{attr}")

        print("PASS: synthetic netCDF -> BP5 -> netCDF vendored round trip; all variables, raw dtypes, dimensions, and attributes match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
