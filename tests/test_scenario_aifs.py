"""Tests for scenario.generators.aifs -- pure netCDF-to-WPS-intermediate conversion against a
synthetic file (no network, no real AIFS runner). Covers: field/level/unit mapping, bbox crop
(with margin), geopotential -> height conversion, lon 0..360 handling, and missing-variable ->
fields_missing."""

import datetime

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

import scenario.wps_intermediate as wi
from scenario.base import ScenarioRequest, get_generator
from scenario.generators.aifs import G, netcdf_to_records


def _make_synthetic_nc(path, omit=()):
    ds = netCDF4.Dataset(path, "w")
    try:
        ds.createDimension("time", 1)
        ds.createDimension("level", 2)
        ds.createDimension("lat", 4)
        ds.createDimension("lon", 5)

        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "hours since 2031-07-15 00:00:00"
        time_var[:] = [0.0]

        level_var = ds.createVariable("level", "f4", ("level",))
        level_var[:] = [500.0, 850.0]  # hPa

        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lat_var[:] = [40.0, 30.0, 20.0, 10.0]  # descending

        lon_var = ds.createVariable("lon", "f4", ("lon",))
        lon_var[:] = [350.0, 355.0, 0.0, 5.0, 10.0]  # 0..360 convention (== -10..10)

        t_var = ds.createVariable("t", "f4", ("time", "level", "lat", "lon"))
        t_var[:] = np.full((1, 2, 4, 5), 250.0, dtype=np.float32)

        u_var = ds.createVariable("u", "f4", ("time", "level", "lat", "lon"))
        u_var[:] = np.full((1, 2, 4, 5), 5.0, dtype=np.float32)

        v_var = ds.createVariable("v", "f4", ("time", "level", "lat", "lon"))
        v_var[:] = np.full((1, 2, 4, 5), -3.0, dtype=np.float32)

        q_var = ds.createVariable("q", "f4", ("time", "level", "lat", "lon"))
        q_var[:] = np.full((1, 2, 4, 5), 0.001, dtype=np.float32)

        z_var = ds.createVariable("z", "f4", ("time", "level", "lat", "lon"))
        z_var[:] = np.full((1, 2, 4, 5), 500.0 * G, dtype=np.float32)  # -> GHT = 500 m

        var_2t = ds.createVariable("2t", "f4", ("time", "lat", "lon"))
        var_2t[:] = np.full((1, 4, 5), 288.0, dtype=np.float32)

        var_10u = ds.createVariable("10u", "f4", ("time", "lat", "lon"))
        var_10u[:] = np.full((1, 4, 5), 2.0, dtype=np.float32)

        var_10v = ds.createVariable("10v", "f4", ("time", "lat", "lon"))
        var_10v[:] = np.full((1, 4, 5), -1.0, dtype=np.float32)

        if "sp" not in omit:
            var_sp = ds.createVariable("sp", "f4", ("time", "lat", "lon"))
            var_sp[:] = np.full((1, 4, 5), 101300.0, dtype=np.float32)

        var_skt = ds.createVariable("skt", "f4", ("time", "lat", "lon"))
        var_skt[:] = np.full((1, 4, 5), 290.0, dtype=np.float32)

        var_lsm = ds.createVariable("lsm", "f4", ("time", "lat", "lon"))
        var_lsm[:] = np.full((1, 4, 5), 1.0, dtype=np.float32)

        # deliberately omit "msl" and "z_sfc" -> PMSL and SOILHGT must land in fields_missing
    finally:
        ds.close()


def test_netcdf_to_records_fields_levels_units_and_missing(tmp_path):
    nc_path = tmp_path / "synthetic.nc"
    _make_synthetic_nc(nc_path)

    bbox = (15.0, 35.0, -8.0, 8.0)
    per_time = netcdf_to_records(nc_path, bbox)
    assert len(per_time) == 1
    t, records, missing = per_time[0]
    assert t == datetime.datetime(2031, 7, 15, 0)

    assert "PMSL" in missing
    assert "SOILHGT" in missing

    by_field_level = {(r.field, round(r.xlvl)): r for r in records}
    assert ("TT", 50000) in by_field_level
    assert ("TT", 85000) in by_field_level
    assert by_field_level[("TT", 50000)].units == "K"
    assert by_field_level[("UU", 50000)].units == "m s-1"
    assert by_field_level[("SPECHUMD", 50000)].units == "kg kg-1"

    ght = by_field_level[("GHT", 50000)]
    assert np.allclose(ght.slab, 500.0)  # z / 9.80665 == 500 m

    tt_surface = by_field_level[("TT", int(wi.SURFACE_PA))]
    assert np.allclose(tt_surface.slab, 288.0)


def test_netcdf_to_records_bbox_crop_and_lat_ascending(tmp_path):
    nc_path = tmp_path / "synthetic.nc"
    _make_synthetic_nc(nc_path)

    # tight bbox around lat 20..30, lon -5..5, no margin room to spare -> only lat {20,30},
    # lon {-5,0,5} survive a margin_deg=5 crop (source margin is fixed at 5 deg in the module).
    bbox = (25.0, 25.0, 0.0, 0.0)
    per_time = netcdf_to_records(nc_path, bbox)
    _, records, _ = per_time[0]
    tt = next(r for r in records if r.field == "TT" and round(r.xlvl) == 50000)
    assert tt.ny == 2  # lat 20, 30 (25 +- 5 margin)
    assert tt.nx == 3  # lon -5, 0, 5
    assert tt.startlat == 20.0  # ascending, SW corner
    assert tt.deltalat == 10.0


def test_aifs_generator_end_to_end_with_nc_paths(tmp_path):
    nc_path = tmp_path / "synthetic.nc"
    _make_synthetic_nc(nc_path)

    gen = get_generator("aifs")
    valid_time = datetime.datetime(2031, 7, 15, 0)
    req = ScenarioRequest(
        bbox=(15.0, 35.0, -8.0, 8.0), start=valid_time, end=valid_time, interval_h=6,
        out_dir=tmp_path / "out",
    )
    forcing = gen.generate(req, nc_paths={valid_time: nc_path})
    assert len(forcing.files) == 1
    # PSFC/PMSL are atmospheric state (fill.PRESSURE_GROUP): having PSFC (this synthetic file's
    # "sp") is enough that generate() does not raise, but PMSL must still come out with NO fill
    # rule -- fill may never supply either from the analog (real.exe derives one from the other).
    # It stays in fields_missing (genuinely absent from the final output), just unfilled.
    assert "PMSL" in forcing.fields_missing
    assert "PMSL" not in forcing.fill_rules
    assert "SOILHGT" in forcing.fields_missing
    assert forcing.fill_rules["SOILHGT"] == "WPS_GEOG"

    recs = wi.read_fields(forcing.files[0])
    assert any(r.field == "TT" for r in recs)


def test_aifs_generator_raises_when_both_psfc_and_pmsl_are_absent(tmp_path):
    """Neither PSFC nor PMSL at all (fill.PRESSURE_GROUP entirely absent) is a real capability
    gap -- fill may not supply either from the analog, so this must raise, not silently fill."""
    from scenario.base import ScenarioError

    nc_path = tmp_path / "synthetic_no_pressure.nc"
    _make_synthetic_nc(nc_path, omit=("sp",))

    gen = get_generator("aifs")
    valid_time = datetime.datetime(2031, 7, 15, 0)
    req = ScenarioRequest(
        bbox=(15.0, 35.0, -8.0, 8.0), start=valid_time, end=valid_time, interval_h=6,
        out_dir=tmp_path / "out",
    )
    with pytest.raises(ScenarioError, match="PSFC/PMSL"):
        gen.generate(req, nc_paths={valid_time: nc_path})
