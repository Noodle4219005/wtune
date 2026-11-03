"""Regression tests for the 8bdcd3c source-review findings in post/recipes/*.py: earth-relative
winds, hemisphere sign, per-figure error handling, frame time index, units, out-of-domain
points, Hovmoller centre fallback, climate boxes outside the domain, bucketed rain. wrf-python
and netCDF4 are replaced by small fakes; the arrays are synthetic."""
import sys
import types
from datetime import datetime

import numpy as np
import pytest

from wtune.post.recipes import (climate, common, convection, moisture, pbl, precip, synoptic, tc_extra, terrain)
from wtune.post.recipes import tc_structure as ts


class _DS:
    """A stand-in for an open netCDF4.Dataset."""

    def __init__(self, variables=None, **attrs):
        self.variables = variables or {}
        self.__dict__.update(attrs)

    def getncattr(self, name):
        return self.__dict__[name]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---- 1 + 5: 850 hPa barbs are earth-relative; the below-ground mask uses the frame's index

def test_thetae850_uses_uvmet_and_the_frames_psfc(tmp_path):
    ny, nx, names, saved = 3, 4, [], {}
    three = np.ones((2, ny, nx))

    def getvar(ds, name, timeidx=None, units=None):
        names.append(name)
        return {"theta_e": three * 330.0,
                "uvmet": np.stack([three * 7.0, three * -3.0]),
                "ua": three * 99.0, "va": three * 99.0}[name]
    wrf = types.SimpleNamespace(getvar=getvar,
                                interplevel=lambda f, p, lev: np.asarray(f)[0])
    psfc = np.stack([np.full((ny, nx), 100000.0), np.full((ny, nx), 80000.0)])
    c = synoptic._Chart(wrf=wrf, ds=_DS({"PSFC": psfc}), ti=1, np=np,
                        save_map=lambda p, la, lo, title, **kw: saved.update(kw),
                        out_dir=str(tmp_path), run_label="r", tlabel="t", time_str="t",
                        pressure=three, lats=np.zeros((ny, nx)), lons=np.zeros((ny, nx)))
    synoptic._thetae850_chart(c, lambda *a, **k: {"id": a[0]})
    assert "uvmet" in names and "ua" not in names and "va" not in names
    assert saved["hatch_mask"].all()                 # PSFC[1] = 800 hPa: all below ground
    assert np.ma.getdata(saved["u"]).max() == 7.0 and np.ma.getdata(saved["v"]).min() == -3.0


# ---- 2: tc_extra rotates U10/V10 to earth axes and drops rings cut by the domain edge

def test_earth_relative_removes_the_grid_rotation_leak():
    lat0, lon0, alpha = 20.0, 130.0, np.radians(30.0)
    lat = np.array([20.0, 20.5, 20.0, 19.5])
    lon = np.array([130.5, 130.0, 129.5, 130.0])          # E, N, W, S of the centre
    ue, vn = np.array([0.0, -10.0, 0.0, 10.0]), np.array([10.0, 0.0, -10.0, 0.0])  # CCW vortex
    ug = ue * np.cos(alpha) + vn * np.sin(alpha)           # what WRF stores (grid axes)
    vg = vn * np.cos(alpha) - ue * np.sin(alpha)
    u, v = tc_extra.earth_relative(ug, vg, np.sin(alpha), np.cos(alpha))
    np.testing.assert_allclose(tc_extra.tangential_wind(lat0, lon0, lat, lon, u, v), 10.0)
    leaked = tc_extra.tangential_wind(lat0, lon0, lat, lon, ug, vg)
    np.testing.assert_allclose(leaked, 10.0 * np.cos(alpha))     # the unrotated answer is low
    u0, v0 = tc_extra.earth_relative(ug, vg, 0.0, 1.0)           # unrotated grid: unchanged
    assert (u0 == ug).all() and (v0 == vg).all()


def test_ring_means_blank_rings_cut_by_the_domain_edge():
    dx, bins = 10.0, [0, 50, 100, 150]
    x = np.arange(41) * dx
    full_x, full_y = np.meshgrid(x - 200.0, x - 200.0)       # centre in the middle
    corner_x, corner_y = np.meshgrid(x, x)                   # centre on the domain corner
    vt = np.ones((41, 41))
    assert np.isfinite(tc_extra.ring_means(vt, np.hypot(full_x, full_y), bins, dx)).all()
    cut = tc_extra.ring_means(vt, np.hypot(corner_x, corner_y), bins, dx)
    assert np.isnan(cut[1:]).all()                           # a quarter of each ring survives


def test_radius_time_plot_is_cyclonic_positive_in_the_sh(monkeypatch, tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    n, lat0, lon0, alpha = 61, -20.0, 150.0, np.radians(25.0)
    lat, lon = np.meshgrid(lat0 + (np.arange(n) - 30) * 0.1, lon0 + (np.arange(n) - 30) * 0.1,
                           indexing="ij")
    dy, dx = lat - lat0, (lon - lon0) * np.cos(np.radians(lat0))
    r = np.maximum(np.hypot(dx, dy), 1e-9)
    ue, vn = 20.0 * dy / r, -20.0 * dx / r                   # clockwise = cyclonic in the SH
    path = str(tmp_path / "wrfout_d01")
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("Time", 1)
        nc.createDimension("y", n)
        nc.createDimension("x", n)
        nc.DX = 11000.0
        for name, a in (("XLAT", lat), ("XLONG", lon),
                        ("U10", ue * np.cos(alpha) + vn * np.sin(alpha)),
                        ("V10", vn * np.cos(alpha) - ue * np.sin(alpha)),
                        ("SINALPHA", np.full((n, n), np.sin(alpha))),
                        ("COSALPHA", np.full((n, n), np.cos(alpha)))):
            nc.createVariable(name, "f8", ("Time", "y", "x"))[0] = a
    monkeypatch.setattr(tc_extra.atcf, "parse_file", lambda p: [
        {"tau_h": h, "time": datetime(2020, 1, 1), "lat": lat0, "lon": lon0} for h in (0, 6)])
    monkeypatch.setattr(tc_extra, "list_frame_refs", lambda run_dir, domain=1: [
        ("2020-01-01_00:00:00", path, 0), ("2020-01-01_06:00:00", path, 0)])
    grids = []
    import matplotlib.axes
    real = matplotlib.axes.Axes.pcolormesh
    monkeypatch.setattr(matplotlib.axes.Axes, "pcolormesh",
                        lambda self, x, y, g, **kw: grids.append(g) or real(self, x, y, g, **kw))
    assert tc_extra.radius_time_plot("/run", "adeck", str(tmp_path / "rt.png"))
    g = grids[0]
    np.testing.assert_allclose(g[1:8], 20.0, atol=0.5)      # +20 m/s, not -20 or 20 cos(alpha)
    assert np.isnan(g[-1]).all()                            # 375-400 km: beyond the domain


def test_radius_time_plot_has_a_column_only_where_a_frame_is(monkeypatch, tmp_path):
    # hourly a-deck on 3-hourly output: two empty columns out of three (haitang2017); a history
    # time a few time steps off the hour is the same frame
    monkeypatch.setattr(tc_extra.atcf, "parse_file", lambda p: [
        {"tau_h": h, "time": datetime(2017, 7, 31), "lat": 25.0, "lon": 120.0}
        for h in range(0, 10)])
    refs = [("2017-07-31_00:00:00", "/f0", 0), ("2017-07-31_03:01:30", "/f3", 0),
            ("2017-07-31_06:00:00", "/f6", 0), ("2017-07-31_09:30:00", "/f9", 0)]
    monkeypatch.setattr(tc_extra, "list_frame_refs", lambda run_dir, domain=1: refs)
    opened = []

    class DS:
        def __init__(self, p):
            opened.append(p)
            self.variables = {k: np.zeros((1, 5, 5)) for k in
                              ("XLAT", "XLONG", "U10", "V10", "SINALPHA")}
            self.variables["COSALPHA"] = np.ones((1, 5, 5))

        def getncattr(self, name):
            return 3000.0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr("netCDF4.Dataset", DS)
    monkeypatch.setattr(tc_extra, "ring_means", lambda vt, rr, bins, dx: np.ones(len(bins) - 1))
    xs = []
    import matplotlib.axes
    real = matplotlib.axes.Axes.pcolormesh
    monkeypatch.setattr(matplotlib.axes.Axes, "pcolormesh",
                        lambda self, x, y, g, **kw: xs.append(list(x)) or real(self, x, y, g, **kw))
    assert tc_extra.radius_time_plot("/run", "adeck", str(tmp_path / "rt.png"))
    assert xs[0] == [0.0, 3.0, 6.0] and opened == ["/f0", "/f3", "/f6"]   # 09:30 is too far
    assert tc_extra.nearest_frame(refs, datetime(2017, 7, 31, 9)) is None


# ---- 3: cyclonic is positive in both hemispheres

def test_rmw_and_vorticity_follow_the_hemisphere():
    rc = np.arange(5.0, 100.0, 10.0)
    vt_nh = 40.0 * np.exp(-((rc - 35.0) / 20.0) ** 2) - 2.0      # peak at 35 km, -2 far out
    assert ts.cyclonic_sign(15.0) == 1.0 and ts.cyclonic_sign(-15.0) == -1.0
    nh = ts.rmw_profile(vt_nh * ts.cyclonic_sign(15.0), rc)
    sh = ts.rmw_profile(-vt_nh * ts.cyclonic_sign(-15.0), rc)    # the SH mirror image
    assert nh[0] == sh[0] == 35.0
    assert ts.rmw_profile(-vt_nh, rc)[0] != 35.0                 # what the unsigned Vt gave

    avo = np.array([[20.0, 30.0], [-20.0, -30.0]])
    lats = np.array([[10.0, 10.0], [-10.0, -10.0]])
    same, label = synoptic.cyclonic_avo(np, avo, np.abs(lats))
    assert same is avo and label == "absolute vorticity (10$^{-5}$ s$^{-1}$)"   # NH: unchanged
    flipped, label = synoptic.cyclonic_avo(np, avo, lats)
    assert (flipped == np.abs(avo)).all() and "cyclonic +" in label


def test_rz_wind_figure_says_so_in_the_sh_only(monkeypatch, tmp_path):
    import matplotlib.figure
    labels = []
    real = matplotlib.figure.Figure.colorbar
    monkeypatch.setattr(matplotlib.figure.Figure, "colorbar",
                        lambda self, cf, **kw: labels.append(kw["label"]) or real(self, cf, **kw))
    r, z = np.arange(5.0, 300.0, 10.0), np.arange(0.0, 18.5, 0.5)
    vt = 30.0 * np.exp(-((r[None] - 40.0) / 30.0) ** 2) * np.ones((len(z), 1))
    rz = dict(dx=10.0, r=r, z_km=z, vt=vt, vr=vt * 0.0, w=vt * 0.0, ter_km=r * 0.0,
              rmw=ts.rmw_profile(vt, r), source="track")
    ts.rz_wind_figure(rz, str(tmp_path / "nh.png"), "r", 1, "t")
    ts.rz_wind_figure(dict(rz, sh=True), str(tmp_path / "sh.png"), "r", 1, "t")
    assert "cyclonic" not in labels[0] and "cyclonic" in labels[1]


# ---- 4: one failing figure is one missing entry; the figures already drawn survive

def _ids(out):
    return [(e["id"], bool(e.get("missing"))) for e in out]


def test_convection_figures_survive_a_missing_field(monkeypatch, tmp_path):
    def map_field(run_dir, domain, field, *a, **k):
        if field != "mdbz":
            raise KeyError("QVAPOR")
        return "p.png"
    monkeypatch.setattr(convection, "list_frames", lambda r, domain=1: [("t", "p")])
    monkeypatch.setattr(convection, "_post_spinup_frames", lambda r, d, s: [("t", "p", 0)])
    monkeypatch.setattr(convection, "_map_field", map_field)
    monkeypatch.setattr(convection, "_uh_time_max", lambda *a: (_ for _ in ()).throw(
        ValueError("'uhel' is not a valid variable name")))
    monkeypatch.setattr(convection, "_max_reflectivity_point",
                        lambda *a: (_ for _ in ()).throw(IndexError("index 9 out of bounds")))
    monkeypatch.setattr(convection, "_radar_figures", lambda *a: [])
    info = {"domains": [{"domain": 1, "dx_km": 3.0, "cu_physics": 0}]}
    out = convection.figures(["/run"], ["r"], str(tmp_path), info)
    assert _ids(out) == [("conv_max_reflectivity", False), ("conv_mcape_mcin", True),
                         ("conv_uh25_max", True), ("conv_skewt", True)]
    assert "lacks QVAPOR" in out[1]["reason"] and "lacks uhel" in out[2]["reason"]
    assert out[3]["reason"].startswith("IndexError")


def test_synoptic_charts_survive_a_missing_field(monkeypatch, tmp_path):
    monkeypatch.setattr(synoptic, "_post_spinup_time",
                        lambda *a: (("2020-01-01_00:00:00", "/p", 0), []))
    monkeypatch.setattr("netCDF4.Dataset", lambda p: _DS())
    monkeypatch.setitem(sys.modules, "wrf", types.SimpleNamespace())
    monkeypatch.setattr("wtune.post.conventions.ds_time_label", lambda ds, t: t)
    monkeypatch.setattr(synoptic, "_read_shared_fields", lambda c: None)
    for name in ("_z500_vort_chart", "_slp_thickness_chart", "_thetae850_chart",
                 "_t2_wind10_chart"):
        monkeypatch.setattr(synoptic, name, lambda c, *a, _n=name: {"id": _n})
    monkeypatch.setattr(synoptic, "_jet250_chart",
                        lambda c, fig: (_ for _ in ()).throw(KeyError("U")))
    monkeypatch.setattr(synoptic, "_model_bbox", lambda np_, ds: (0, 1, 0, 1))
    monkeypatch.setattr(synoptic, "_era5_diff_entry", lambda *a: {"id": "era5"})
    monkeypatch.setattr(synoptic, "_era5_850_entry", lambda *a: {"id": "era5_850"})
    out = synoptic.figures(["/run"], ["r"], str(tmp_path), {})
    assert _ids(out) == [("_z500_vort_chart", False), ("_slp_thickness_chart", False),
                         ("_thetae850_chart", False), ("syn_250_jet", True),
                         ("_t2_wind10_chart", False), ("era5", False), ("era5_850", False)]
    assert out[3]["reason"] == "history stream lacks U (iofields-reduced output?)"


def test_synoptic_reads_the_850_hpa_fields_while_the_wrfout_is_open(monkeypatch, tmp_path):
    # haitang2017: the ERA5 entries ran after `with Dataset` closed the file; reading the 850 hPa
    # wind there raised "NetCDF: Not a valid ID" and the whole synoptic set became one entry
    state = {"open": False}

    class DS(_DS):
        def __enter__(self):
            state["open"] = True
            return self

        def __exit__(self, *a):
            state["open"] = False
            return False
    monkeypatch.setattr(synoptic, "_post_spinup_time",
                        lambda *a: (("2020-01-01_00:00:00", "/p", 0), []))
    monkeypatch.setattr("netCDF4.Dataset", lambda p: DS())
    monkeypatch.setitem(sys.modules, "wrf", types.SimpleNamespace())
    monkeypatch.setattr("wtune.post.conventions.ds_time_label", lambda ds, t: t)
    monkeypatch.setattr(synoptic, "_read_shared_fields", lambda c: None)
    for name in ("_z500_vort_chart", "_slp_thickness_chart", "_thetae850_chart",
                 "_t2_wind10_chart", "_jet250_chart"):
        monkeypatch.setattr(synoptic, name, lambda c, *a, _n=name: {"id": _n})
    monkeypatch.setattr(synoptic, "_model_bbox", lambda np_, ds: (0, 1, 0, 1))
    monkeypatch.setattr(synoptic, "_era5_diff_entry", lambda *a: {"id": "era5"})
    seen = []
    monkeypatch.setattr(synoptic, "_model_850", lambda c: seen.append(state["open"]) or {"U850": 1})
    monkeypatch.setattr(synoptic, "_era5_850_entry",
                        lambda c, r, p, b, f, mi, m, box: {"id": "850", "m": m})
    out = synoptic.figures(["/run"], ["r"], str(tmp_path), {})
    assert seen == [True] and out[-1] == {"id": "850", "m": {"U850": 1}}
    monkeypatch.setattr(synoptic, "_model_850", lambda c: (_ for _ in ()).throw(KeyError("QVAPOR")))
    out = synoptic.figures(["/run"], ["r"], str(tmp_path), {})
    assert [o["id"] for o in out][:5] == ["_z500_vort_chart", "_slp_thickness_chart",
                                         "_thetae850_chart", "_jet250_chart", "_t2_wind10_chart"]
    assert "QVAPOR" in out[-1]["m"]


def test_pbl_time_height_survives_a_failing_cross_section(monkeypatch, tmp_path):
    frames = [("2020-01-01_00:00:00", "/a", 0), ("2020-01-01_01:00:00", "/b", 0)]
    monkeypatch.setattr(pbl, "list_frame_refs", lambda r, domain=1: frames)
    monkeypatch.setattr(pbl, "_obs_dir", lambda r, p=None: None)
    monkeypatch.setattr(pbl, "pick_column", lambda r, f, s: (1, 1, "col", None))
    monkeypatch.setattr("netCDF4.Dataset", lambda p: _DS({"XLONG": np.zeros((1, 3, 3))}))
    t = datetime(2020, 1, 1)
    monkeypatch.setattr(pbl, "column_series",
                        lambda f, j, i: [(t, {"pblh_model": 1.0}), (t, {"pblh_model": 2.0})])
    monkeypatch.setattr(pbl, "plot_time_height", lambda *a: "th.png")
    monkeypatch.setattr(pbl, "plot_cross_section",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("PBLH")))
    info = {"domains": [{"domain": 1, "dx_km": 1.0, "bl_pbl_physics": 1}]}
    out = pbl.figures(["/run"], ["r"], str(tmp_path), info)
    assert _ids(out) == [("pbl_time_height", False), ("pbl_cross_terrain", True),
                         ("pbl_skewt_obs", True)]
    assert "lacks PBLH" in out[1]["reason"]


def test_terrain_and_moisture_figures_survive_a_missing_field(monkeypatch, tmp_path):
    def boom(exc):
        return lambda *a, **k: (_ for _ in ()).throw(exc)
    info = {"domains": [{"domain": 1}]}
    monkeypatch.setattr(terrain, "_terrain_wind_map", lambda *a, **k: "w.png")
    monkeypatch.setattr(terrain, "_cross_section_figure", boom(KeyError("W")))
    monkeypatch.setattr(terrain, "_rain_over_terrain", boom(KeyError("RAINC")))
    monkeypatch.setattr(terrain, "_valley_time_height", lambda *a, **k: ("v.png", None))
    monkeypatch.setattr(terrain, "list_frame_refs", boom(IndexError("x")))
    monkeypatch.setattr(terrain, "_point_stat", boom(ValueError("bad obs")))
    out = terrain.figures(["/run"], ["r"], str(tmp_path), info, project_dir=str(tmp_path))
    assert _ids(out) == [("terrain_wind_map", False), ("terrain_crosssection", True),
                         ("terrain_rain", True), ("terrain_valley_timeheight", False),
                         ("terrain_leeward_station", True), ("terrain_point_stat", True)]
    assert "lacks RAINC" in out[2]["reason"]

    monkeypatch.setattr(moisture, "_pw_figure", lambda *a, **k: "pw.png")
    monkeypatch.setattr(moisture, "_pw_change_figure", boom(KeyError("QVAPOR")))
    monkeypatch.setattr(moisture, "_ivt_figure", lambda *a, **k: "ivt.png")
    monkeypatch.setattr(moisture, "_cross_section_figure", boom(IndexError("x")))
    monkeypatch.setattr(moisture, "_point_time_height", boom(KeyError("HGT")))
    out = moisture.figures(["/run"], ["r"], str(tmp_path), info, tc_centre=(20.0, 130.0))
    assert _ids(out) == [("moisture_pw", False), ("moisture_pw_chg6h", True),
                         ("moisture_ivt", False), ("moisture_crosssection", True),
                         ("moisture_timeheight", True)]



def test_terrain_point_stat_not_repeated_on_the_precip_domain(monkeypatch, tmp_path):
    for name in ("_terrain_wind_map", "_rain_over_terrain"):
        monkeypatch.setattr(terrain, name, lambda *a, **k: None)
    monkeypatch.setattr(terrain, "_cross_section_figure", lambda *a, **k: None)
    monkeypatch.setattr(terrain, "_valley_time_height", lambda *a, **k: (None, "x"))
    monkeypatch.setattr(terrain, "list_frame_refs", lambda *a, **k: [])
    calls = []
    monkeypatch.setattr(terrain, "_point_stat",
                        lambda *a: calls.append(a[2]) or {"id": "terrain_point_stat"})
    one = {"types": ["precip", "terrain"], "domains": [{"domain": 1}]}
    out = terrain.figures(["/run"], ["r"], str(tmp_path), one, project_dir=str(tmp_path))
    assert calls == [] and "precip_stn_series" in out[-1]["text"]
    nested = dict(one, domains=[{"domain": 1}, {"domain": 2}], terrain_domains=[1])
    terrain.figures(["/run"], ["r"], str(tmp_path), nested, project_dir=str(tmp_path))
    terrain.figures(["/run"], ["r"], str(tmp_path), dict(one, types=["terrain"]),
                    project_dir=str(tmp_path))
    assert calls == [1, 1]

# ---- 6: valley time-height barbs are in kt and say so

def test_valley_time_height_barbs_are_in_kt(monkeypatch, tmp_path):
    import matplotlib.axes
    ny = nx = 15
    frames = [("2020-01-01_0%d:00:00" % h, "/f%d" % h, 0) for h in (0, 3)]
    monkeypatch.setattr(terrain, "list_frame_refs", lambda run_dir, domain=1: frames)
    monkeypatch.setattr("netCDF4.Dataset", lambda p: _DS())
    z = np.linspace(100.0, 3000.0, 12)[:, None, None] * np.ones((12, ny, nx))

    def getvar(ds, name, timeidx=0, units=None):
        return {"ter": np.zeros((ny, nx)), "theta": 290.0 + z / 500.0, "z": z,
                "uvmet": np.stack([z * 0.0 + 10.0, z * 0.0])}[name]
    monkeypatch.setitem(sys.modules, "wrf", types.SimpleNamespace(
        getvar=getvar, latlon_coords=lambda v, as_np=True: (np.zeros((ny, nx)),) * 2))
    monkeypatch.setattr(terrain, "_find_valley_point", lambda hgt, landmask, bounds, **k: (7, 7))
    barbs, titles = [], []
    monkeypatch.setattr(matplotlib.axes.Axes, "barbs",
                        lambda self, x, y, u, v, **kw: barbs.append((u, v)))
    real = matplotlib.axes.Axes.set_title
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title",
                        lambda self, s, **kw: titles.append(s) or real(self, s, **kw))
    path, reason = terrain._valley_time_height("/run", "r", 1, str(tmp_path), bdy_width=2)
    assert path and reason is None and barbs
    np.testing.assert_allclose(barbs[0][0], 10.0 * 1.943844)     # 10 m/s drawn as 19.4 kt
    assert "wind barbs in kt" in titles[-1]


# ---- 7: a time-height point outside the domain is a missing entry, not a wrapped index

def test_point_time_height_outside_the_domain(monkeypatch, tmp_path):
    frames = [("2020-01-01_00:00:00", "/a", 0), ("2020-01-01_01:00:00", "/b", 0)]
    monkeypatch.setattr(moisture, "list_frame_refs", lambda run_dir, domain=1: frames)
    monkeypatch.setattr("netCDF4.Dataset", lambda p: _DS({"XLAT": np.zeros((1, 5, 6))}))
    monkeypatch.setitem(sys.modules, "wrf", types.SimpleNamespace(
        ll_to_xy=lambda ds, lat, lon: np.array([-3, 2])))
    with pytest.raises(moisture.PointOutsideDomain, match="outside d02"):
        moisture._point_time_height("/run", "r", 2, str(tmp_path), (10.0, 120.0))

    for name in ("_pw_figure", "_pw_change_figure", "_ivt_figure", "_cross_section_figure"):
        monkeypatch.setattr(moisture, name, lambda *a, **k: "p.png")
    out = moisture.figures(["/run"], ["r"], str(tmp_path), {"domains": [{"domain": 2}]},
                           tc_centre=(10.0, 120.0))
    assert out[-1]["id"] == "moisture_timeheight" and out[-1]["missing"]
    assert "is outside d02" in out[-1]["reason"]


# ---- 8: with a track, a Hovmoller frame it does not cover has no centre (NaN column)

def test_hovmoller_frame_without_a_track_point_is_blank(monkeypatch):
    slp = np.full((9, 9), 1010.0)
    slp[3:6, 3:6] = 990.0
    asked = []

    def getvar(ds, name, timeidx=0, meta=False, units=None):
        asked.append(name)
        return slp if name == "slp" else np.ones((3, 9, 9)) * 1000.0
    monkeypatch.setattr(ts.cl, "_wrf", lambda: types.SimpleNamespace(
        getvar=getvar, ll_to_xy=lambda ds, lat, lon, timeidx=0, meta=False: np.array([-5, 2])))
    ds = _DS({"XLAT": np.zeros((1, 9, 9)), "HGT": np.zeros((1, 9, 9))}, DX=10000.0)
    track = {"2020-01-01_06:00:00": (20.0, 130.0, 1)}
    t0 = "2020-01-01_00:00:00"
    assert ts.frame_centre(ds, t0, track, 0, slp_fallback=False) is None
    # a track point outside the domain is no centre either
    assert ts.frame_centre(ds, "2020-01-01_06:00:00", track, 0, slp_fallback=False) is None
    assert ts.frame_centre(ds, t0, track, 0)[:2] == (4, 4)       # other figures: unchanged
    assert ts.frame_centre(ds, t0, {}, 0)[:2] == (4, 4)

    monkeypatch.setattr("netCDF4.Dataset", lambda p: ds)
    del asked[:]
    assert ts.eye_profile("/p", t0, track, 0, np.array([0.0, 500.0])) == (None, None)
    assert "slp" not in asked
    _times, arr = ts.hovmoller_assemble([(t0, None), ("2020-01-01_06:00:00", [1.0, 2.0])], 2)
    assert np.isnan(arr[:, 0]).all() and (arr[:, 1] == [1.0, 2.0]).all()


# ---- 9: a climate box with no cell in the domain says so

def test_climate_box_outside_domain(monkeypatch, tmp_path):
    lat, lon = np.meshgrid(np.linspace(30.0, 45.0, 16), np.linspace(-110.0, -90.0, 21),
                           indexing="ij")                             # central US
    inside = np.ones(lat.shape, bool)
    assert climate.boxes_outside(lat, lon, inside) == [b[0] for b in climate.BOXES]
    assert climate.boxes_outside(lat, lon + 225.0, inside) == [
        "South China", "Taiwan"]                                      # 30-45N, 115-135E
    assert climate.boxes_outside(lat, lon + 225.0, ~inside) == [b[0] for b in climate.BOXES]

    import matplotlib.axes
    texts = []
    real = matplotlib.axes.Axes.text
    monkeypatch.setattr(matplotlib.axes.Axes, "text",
                        lambda self, x, y, s, **kw: texts.append(s) or real(self, x, y, s, **kw))
    months = [(2023, m) for m in range(1, 13)]
    nan = np.full(12, np.nan)
    model = {b[0]: nan for b in climate.BOXES}
    series = {k: {"model": model, "ref": None, "ref_label": "", "why": "SST constant",
                  "outside": ["Taiwan"]} for k in ("T", "P")}
    climate.plot_annual_cycle(months, series, str(tmp_path / "c.png"), "r", [])
    assert texts.count("box outside domain") == 2 and texts.count("SST constant") == 6


# ---- 10: accumulated rain includes the bucket counters

def _rain_ds(bucket=None, with_counters=True):
    v = {"RAINC": np.array([[[1.0, 2.0]], [[3.0, 4.0]]], "f4"),
         "RAINNC": np.array([[[10.0, 20.0]], [[30.0, 40.0]]], "f4")}
    if with_counters:
        v["I_RAINC"] = np.array([[[0, 1]], [[2, 0]]], "i4")
        v["I_RAINNC"] = np.array([[[0, 0]], [[1, 3]]], "i4")
    return _DS(v) if bucket is None else _DS(v, BUCKET_MM=bucket)


def test_climate_cdo_expr_adds_the_bucket_part():
    assert common._bucket_mm(_rain_ds(-1.0)) == 0.0                        # bucket off
    assert common._bucket_mm(_rain_ds(100.0, with_counters=False)) == 0.0
    assert common._bucket_mm(_rain_ds(100.0)) == 100.0 and common._bucket_mm(_rain_ds()) == 0.0
    assert climate.rain_expr(0.0) == "PR=RAINC+RAINNC"
    assert climate.rain_expr(100.0) == "PR=RAINC+RAINNC+(I_RAINC+I_RAINNC)*100.0"


def test_precip_model_accum_with_bucket(monkeypatch):
    v = dict(_rain_ds().variables, XLAT=np.zeros((2, 1, 2)), XLONG=np.zeros((2, 1, 2)))
    monkeypatch.setattr("netCDF4.Dataset", lambda p: _DS(v, BUCKET_MM=100.0))
    _lat, _lon, mm = precip._model_accum(("/a", 0), ("/a", 1))
    # RAINC+RAINNC alone goes 11 -> 33 and 22 -> 44 (+22 both); the counters add 300 and 200
    assert (mm == [[322.0, 222.0]]).all()


def test_ensemble_rain_acc_with_bucket(monkeypatch):
    from wtune.post.recipes import ensemble
    v = dict(_rain_ds().variables)
    v = {k: np.repeat(np.repeat(a, 4, axis=1), 2, axis=2) for k, a in v.items()}   # (2, 4, 4)
    monkeypatch.setattr("netCDF4.Dataset", lambda p: _DS(v, BUCKET_MM=100.0))
    monkeypatch.setattr(ensemble, "frame_ref", lambda r, d, t: ("/a", 1))
    monkeypatch.setattr("wtune.post.caseinfo.spec_bdy_width", lambda r: 1)
    monkeypatch.setitem(sys.modules, "wrf", types.SimpleNamespace(
        getvar=lambda ds, name, timeidx=0: ds.variables[name][timeidx],
        latlon_coords=lambda var, as_np=True: (None, None), get_cartopy=lambda var=None: None))
    arr = ensemble._member_field("/run", 1, "t", field="rain_acc")[0]
    assert (arr[1:3, 1:3] == [[333.0, 344.0], [333.0, 344.0]]).all()


# ---- follow-up: a failed diffwrf comparison is one missing envelope entry

def test_repro_envelope_compare_error_is_one_missing_entry(monkeypatch, tmp_path):
    from wtune.post.compare import CompareError
    monkeypatch.setattr(common, "_domain_physics", lambda r: (None, [1]))
    monkeypatch.setattr(common, "plot_domain_map", lambda *a: str(tmp_path / "d.png"))
    monkeypatch.setattr(common, "list_frames", lambda r, domain=1: [("t0", "/p")])
    monkeypatch.setattr(common, "plot_repro_envelope",
                        lambda *a: (_ for _ in ()).throw(CompareError("diffwrf rc=2")))
    out = common.figures(["/a", "/b"], ["a", "b"], str(tmp_path), {}, decomposition_only=True)
    assert _ids(out) == [("common_domain_map", False), ("common_repro_envelope", True)]
    assert out[1]["reason"] == "diffwrf rc=2"


def test_era5_850_box_stats_are_the_case_area_means(tmp_path):
    # a d01-wide RMSE says little about the Taiwan Strait; the box means of the pairs do
    netCDF4 = pytest.importorskip("netCDF4")
    p = str(tmp_path / "pairs.nc")
    lat, lon = np.arange(20.0, 26.0), np.arange(118.0, 124.0)
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        f = np.full((6, 6), 10.0)
        f[:2] = 20.0                                       # 20-21N: outside the box below
        ds.createVariable("FCST_WSPD850_ALL", "f4", ("lat", "lon"))[:] = f
        ds.createVariable("OBS_WSPD850_ALL", "f4", ("lat", "lon"))[:] = np.full((6, 6), 8.0)
    got = synoptic.box_stats(p, (22.0, 25.0, 119.0, 122.0))
    assert got == {"model": 10.0, "era5": 8.0, "bias": 2.0, "n": 16}
    assert synoptic.box_stats(p, (40.0, 41.0, 119.0, 122.0)) is None
