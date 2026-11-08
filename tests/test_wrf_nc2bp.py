"""Synthetic WRF netCDF -> BP -> netCDF bit-exact regression (restored from
8ba1629:tools/wrf_nc2bp_roundtrip_test.py). Runs under the wtune env: the converters go through
wtune.bp, which runs them as a subprocess of the envs/adios2 interpreter (or
$WTUNE_ADIOS2_PYTHON); skipped when no adios2-capable interpreter is found."""

import os
import sys

import numpy as np
import pytest

import wtune.bp

netCDF4 = pytest.importorskip("netCDF4")
Dataset = netCDF4.Dataset

if wtune.bp.adios2_python() is None:
    pytest.skip("no adios2-capable interpreter (pixi install in envs/adios2, or set "
                "WTUNE_ADIOS2_PYTHON)", allow_module_level=True)


def _make_source(path):
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


def test_api_roundtrip_check_and_cached_inputs(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    _make_source(str(case / "wrfinput_d01"))
    _make_source(str(case / "wrfbdy_d01"))
    cache = tmp_path / "cache"
    m = wtune.bp.convert_inputs(str(case), str(cache))
    assert sorted(m) == ["wrfbdy_d01", "wrfinput_d01"]
    assert wtune.bp.is_bp(m["wrfinput_d01"])                 # single-record: one BP, unchanged
    assert wtune.bp.is_per_record_dir(m["wrfbdy_d01"])       # multi-record: one BP per record
    for name, bp in m.items():
        r = wtune.bp.roundtrip_check(str(case / name), bp, str(tmp_path / "rt"))
        assert r["ok"], r["mismatches"]
    assert not os.listdir(tmp_path / "rt")
    assert "adios2" not in sys.modules     # never imported into the wtune process


def test_cli_nc2bp_check(tmp_path, capsys):
    source = tmp_path / "wrfinput_d01"
    _make_source(str(source))
    out = tmp_path / "out"
    assert wtune.bp.main(["nc2bp", str(source), "--out", str(out), "--check"]) == 0
    assert wtune.bp.is_bp(str(out / "wrfinput_d01"))
    assert "CHECK PASS" in capsys.readouterr().out
    assert wtune.bp.main(["bp2nc", str(out / "wrfinput_d01"), "--out", str(out)]) == 0
    assert (out / "wrfinput_d01.nc").is_file()


def test_split_roundtrip_detects_wrong_record(tmp_path):
    """The per-record round trip pairs file i with source record i: swapping two records'
    contents must fail."""
    import shutil
    source = tmp_path / "wrfbdy_d01"
    out = tmp_path / "split"
    _make_source(str(source))
    wtune.bp.nc2bp(str(source), str(out), split_prefix="wrfbdy_d01")
    a, b = [out / r for r in wtune.bp.record_files(str(out))]
    tmp = out / "swap"
    shutil.move(str(a), str(tmp)); shutil.move(str(b), str(a)); shutil.move(str(tmp), str(b))
    r = wtune.bp.roundtrip_check(str(source), str(out), str(tmp_path / "rt"))
    assert not r["ok"]
