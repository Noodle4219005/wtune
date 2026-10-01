"""StitchNodes --range scaled to the output interval (SC26-P2: hourly wrfout with the 6-hourly
8 deg range let a 1007 hPa low 653 km from Katrina take the t0 fix)."""
import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

import analysis.tc_track as tc_track


def _time_nc(path, values, units):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", None)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = units
        t[:] = np.asarray(values, dtype=float)
    return netCDF4.Dataset(path)


@pytest.mark.parametrize("interval_h, expected", [
    (6.0, 8.0),      # the convention the default was set for: unchanged
    (3.0, 4.0),
    (1.0, 2.0),      # 8/6 = 1.33 deg -> floor 2.0
    (12.0, 8.0),     # never above stitch_range_deg
    (None, 8.0),     # unknown interval: unchanged
])
def test_stitch_range_for_interval(interval_h, expected):
    assert tc_track.stitch_range_for_interval(tc_track.TCTrackParams(), interval_h) == expected


def test_output_interval_hours_from_units(tmp_path):
    with _time_nc(tmp_path / "h.nc", [0, 1, 2, 3], "hours since 2005-08-27 00:00:00") as ds:
        assert tc_track._output_interval_hours(ds) == pytest.approx(1.0)
    with _time_nc(tmp_path / "m.nc", [0, 180, 360], "minutes since 2005-08-27 00:00:00") as ds:
        assert tc_track._output_interval_hours(ds) == pytest.approx(3.0)
    with _time_nc(tmp_path / "one.nc", [0], "hours since 2005-08-27 00:00:00") as ds:
        assert tc_track._output_interval_hours(ds) is None
