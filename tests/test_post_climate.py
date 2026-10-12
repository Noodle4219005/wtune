"""Synthetic-data tests for post/recipes/climate.py and the ERA5/GPCP monthly fetch helpers.
No real long run exists yet; these check month bookkeeping, precipitation differencing, seasonal
means, masking/interpolation and (when `cdo` is installed) the CDO stage on tiny WRF-like files."""
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytest

from wtune.post.recipes import climate
from wtune.post.fetch import era5_monthly, gpcp_monthly


def test_analysis_months_excludes_spinup_and_incomplete_tail():
    t0, t1 = datetime(2022, 12, 1), datetime(2024, 1, 1)
    months = climate.analysis_months(t0, t1, 30.0)
    assert months[0] == (2023, 1) and months[-1] == (2023, 12) and len(months) == 12
    # spin-up parameter: 0 days keeps December 2022, 45 days drops January 2023 too
    assert climate.analysis_months(t0, t1, 0.0)[0] == (2022, 12)
    assert climate.analysis_months(t0, t1, 45.0)[0] == (2023, 2)
    # a month whose closing 00 UTC frame is missing is not complete
    assert (2023, 12) not in climate.analysis_months(t0, datetime(2023, 12, 31, 18), 30.0)
    assert climate.analysis_months(t0, datetime(2022, 12, 20), 30.0) == []


def test_complete_seasons_djf_uses_jan_feb_dec_of_analysis_year():
    months = [(2023, m) for m in range(1, 13)]
    s = climate.complete_seasons(months)
    assert s["DJF"] == [(2023, 1), (2023, 2), (2023, 12)] and len(s) == 4
    assert "DJF" not in climate.complete_seasons([(2023, m) for m in range(1, 12)])   # no Dec
    assert climate.complete_seasons(months + [(2024, 1)]).get("DJF") is None           # ambiguous


def test_monthly_totals_difference_and_reset():
    acc = [np.full((2, 2), 10.0), np.full((2, 2), 25.0), np.full((2, 2), 5.0)]      # last: reset
    tot = climate.monthly_totals(acc)
    assert tot[0][0, 0] == 10.0 and tot[1][0, 0] == 15.0 and np.isnan(tot[2]).all()


def test_season_mean_is_day_weighted():
    monthly = {(2023, 1): np.array([[1.0]]), (2023, 2): np.array([[2.0]]), (2023, 12): np.array([[3.0]])}
    got = climate.season_mean(monthly, [(2023, 1), (2023, 2), (2023, 12)])[0, 0]
    assert got == pytest.approx((31 * 1 + 28 * 2 + 31 * 3) / 90.0)


def test_box_mean_mask_nan_and_weights():
    lat, lon = np.meshgrid(np.linspace(30, 40, 11), np.linspace(100, 110, 11), indexing="ij")
    f = np.where(lat > 35, 10.0, 0.0)
    land = lon < 105
    m, n = climate.box_mean(f, lat, lon, (30, 40, 100, 110), land)
    assert n == int((land).sum()) and 0.0 < m < 10.0
    f2 = f.copy()
    f2[:, :3] = np.nan
    assert climate.box_mean(f2, lat, lon, (30, 40, 100, 110), land)[1] == n - 3 * 11
    assert np.isnan(climate.box_mean(f, lat, lon, (0, 1, 0, 1), land)[0])


def test_interp_to_grid_handles_lon_convention_and_nan():
    ref_lat = np.array([40.0, 30.0, 20.0])               # north-first like ERA5
    ref_lon = np.array([100.0, 110.0, 120.0])
    fld = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, np.nan, 3.0]])
    lat2d = np.array([[30.0, 35.0, 25.0]])
    lon2d = np.array([[105.0, 110.0, 110.0]])
    out = climate.interp_to_grid(ref_lat, ref_lon, fld, lat2d, lon2d)
    assert out[0, 0] == pytest.approx(1.5) and out[0, 1] == pytest.approx(2.0)
    assert np.isnan(out[0, 2])                            # NaN neighbour propagates
    # reference on 0..360, model on -180..180
    out2 = climate.interp_to_grid(ref_lat, ref_lon + 100.0, fld, lat2d, lon2d + 100.0 - 360.0)
    assert np.isfinite(out2[0, 0])


def test_blank_boundary():
    out = climate.blank_boundary(np.ones((6, 7)), 2)
    assert np.isnan(out[:2]).all() and np.isnan(out[:, -2:]).all() and np.isfinite(out[2:-2, 2:-2]).all()


def test_annual_cycle_plot_smoke(tmp_path):
    months = [(2023, m) for m in range(1, 13)]
    x = np.arange(12.0)
    mod = {n: 280 + x for n, _, _ in climate.BOXES}
    ref = {n: 281 + x for n, _, _ in climate.BOXES}
    series = {"T": {"model": dict(mod, **{"W North Pacific": None}), "ref": ref, "ref_label": "ERA5", "why": "SST constant"},
              "P": {"model": mod, "ref": None, "ref_label": "GPCP", "why": "no data"}}
    p = climate.plot_annual_cycle(months, series, str(tmp_path / "a.png"), "t", ["note"])
    assert (tmp_path / "a.png").stat().st_size > 1000


def test_seasonal_bias_plot_smoke_and_symmetric_levels(tmp_path):
    ccrs = pytest.importorskip("cartopy.crs")
    lat, lon = np.meshgrid(np.linspace(20, 40, 20), np.linspace(100, 130, 30), indexing="ij")
    d = {"DJF": np.sin(lon / 10.0) * 2, "JJA": np.cos(lat / 5.0)}
    p = climate.plot_seasonal_bias(d, lat, lon, ccrs.PlateCarree(), (lon < 110).astype(float),
                                   str(tmp_path / "b.png"), "t", "T2", "ERA5", "K")
    assert (tmp_path / "b.png").stat().st_size > 1000


# ---- fetch helpers ----------------------------------------------------------------------

def test_month_starts_spinup_and_completeness():
    ms = era5_monthly.month_starts(datetime(2022, 12, 1), datetime(2024, 1, 1), 30.0, 6)
    assert ms[0] == datetime(2023, 1, 1) and ms[-1] == datetime(2023, 12, 1) and len(ms) == 12
    assert era5_monthly.month_starts(datetime(2022, 12, 1), datetime(2024, 1, 1), 0.0)[0] == datetime(2022, 12, 1)


def test_sample_times_counts():
    assert len(era5_monthly.sample_times(datetime(2023, 2, 1), 6)) == 28 * 4
    assert len(era5_monthly.sample_times(datetime(2023, 12, 1), 1)) == 31 * 24


def test_list_month_files_newest_processing_date_wins():
    html = ('<a href="gpcp_v02r03_monthly_d202301_c20230411.nc">x</a> '
            '<a href="gpcp_v02r03_monthly_d202301_c20230512.nc">x</a> '
            '<a href="gpcp_v02r03_monthly_d202302_c20230505.nc">x</a>')
    got = gpcp_monthly.list_month_files(html)
    assert got == {"202301": "gpcp_v02r03_monthly_d202301_c20230512.nc",
                   "202302": "gpcp_v02r03_monthly_d202302_c20230505.nc"}


def test_era5_aggregate_mean_ignores_nan():
    xr = pytest.importorskip("xarray")
    times = np.array(["2023-01-01T00", "2023-01-01T06"], dtype="datetime64[ns]")
    data = np.array([[[1.0, np.nan], [3.0, 4.0]], [[3.0, np.nan], [5.0, 8.0]]])
    ds = xr.Dataset({"2m_temperature": (("time", "latitude", "longitude"), data)},
                    coords={"time": times, "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]})
    out = era5_monthly._aggregate(ds, "t2m", [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 6)],
                                  np.array([0, 1]), np.array([0, 1]))
    assert out[0, 0] == 2.0 and np.isnan(out[0, 1]) and out[1, 1] == 6.0


# ---- CDO stage on tiny WRF-like frames ----------------------------------------------------

@pytest.mark.skipif(shutil.which("cdo") is None, reason="cdo not installed")
def test_model_monthly_cdo_pipeline(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    ny, nx = 3, 4
    t = datetime(2023, 1, 1)
    frames, acc, k = [], 0.0, 0
    while t <= datetime(2023, 4, 1):
        path = tmp_path / ("wrfout_d01_%s" % t.strftime("%Y-%m-%d_%H:%M:%S"))
        with netCDF4.Dataset(str(path), "w") as nc:
            nc.createDimension("Time", None)
            nc.createDimension("DateStrLen", 19)
            nc.createDimension("south_north", ny)
            nc.createDimension("west_east", nx)
            tv = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
            tv[0, :] = np.array(list(t.strftime("%Y-%m-%d_%H:%M:%S")), dtype="S1")
            xt = nc.createVariable("XTIME", "f4", ("Time",))     # real wrfout carries this
            xt.units = "minutes since 2023-01-01 00:00:00"
            xt[0] = (t - datetime(2023, 1, 1)).total_seconds() / 60.0
            for name in ("T2", "SST", "RAINC", "RAINNC"):
                nc.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            nc.variables["T2"][0] = 270.0 + t.month
            nc.variables["SST"][0] = 300.0
            nc.variables["RAINC"][0] = acc
            nc.variables["RAINNC"][0] = 0.0
        frames.append((t.strftime("%Y-%m-%d_%H:%M:%S"), str(path)))
        t += timedelta(hours=6)
        acc += 1.0                          # 1 mm per 6 h step
    out = climate.model_monthly(frames, str(tmp_path / "work"), ny, nx, 6 * 3600)
    assert out["T2"][(2023, 2)][0, 0] == pytest.approx(272.0)
    order = sorted(out["PR_acc"])
    totals = dict(zip(order, climate.monthly_totals([out["PR_acc"][kk] for kk in order])))
    assert totals[(2023, 1)][0, 0] == pytest.approx(31 * 4)      # steps in January
    assert totals[(2023, 2)][0, 0] == pytest.approx(28 * 4)
    assert totals[(2023, 3)][0, 0] == pytest.approx(31 * 4)
