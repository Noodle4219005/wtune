"""post/recipes/tc_structure.py: storm-relative rotation (Rankine vortex), azimuthal-mean bins,
environment-annulus logic incl. the small-domain fallback, smoothed-SLP centre, track reading,
Hovmoller frame selection/assembly, and every figure rendered once (matplotlib Agg) from a
synthetic wrfout on which wrf-python runs for real (Lambert grid from an independent pyproj LCC,
as tests/test_post_cf_pressure.py). The 3-D volume (another module) is monkeypatched.
"""
import csv
import os

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("wrf")
pyproj = pytest.importorskip("pyproj")
pytest.importorskip("scipy")
pytest.importorskip("cartopy")

from wtune.post.recipes import tc_structure as ts

NY = NX = 50
DX = 12000.0
P_TOP = 5000.0
LCC = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=26 +lon_0=-89 +a=6370000 +b=6370000 +units=m"
SIGMA_STAG = np.concatenate([[1.0, 0.995, 0.985, 0.97], np.linspace(0.94, 0.0, 17)])
NZ = len(SIGMA_STAG) - 1
TIMES = ["2005-08-28_00:00:00", "2005-08-28_03:00:00", "2005-08-28_06:00:00"]
SIGMA = 5.670374419e-8


def _lonlat():
    x = (np.arange(NX) - (NX - 1) / 2.0) * DX
    y = (np.arange(NY) - (NY - 1) / 2.0) * DX
    xx, yy = np.meshgrid(x, y)
    tr = pyproj.Transformer.from_crs(pyproj.CRS(LCC), "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xx, yy)
    return lat, lon


def _rankine(r_km, vmax=45.0, rmax=48.0):
    r = np.asarray(r_km, "f8")
    return np.where(r < rmax, vmax * r / rmax, vmax * rmax / np.maximum(r, 1e-9))


def write_wrfout(path, time_str, centre=(25, 25), vortex=True, hydrometeors=True,
                 westerly=0.0, cold=False, hill=True):
    """One-frame synthetic wrfout. vortex: Rankine TC (grid-relative winds, low SLP, warm
    core, eyewall updraft and condensate) centred on grid cell `centre` (j, i). Otherwise a
    uniform grid-relative westerly `westerly` with a shallow snow band along row centre[0]."""
    jc, ic = centre
    lat, lon = _lonlat()
    jj, ii = np.mgrid[0:NY, 0:NX]
    r2 = np.hypot((ii - ic) * DX, (jj - jc) * DX) / 1000.0
    hgt = np.zeros((NY, NX))
    if hill:
        hgt[-8:, :6] = 600.0                                   # NW corner hill
    dp = 4000.0 * np.exp(-(r2 / 70.0) ** 2) if vortex else 0.0 * r2
    psfc = (101500.0 - dp) * np.exp(-hgt / 8000.0)
    pstag = P_TOP + (psfc[None] - P_TOP) * SIGMA_STAG[:, None, None]
    pmass = 0.5 * (pstag[1:] + pstag[:-1])
    zstag = hgt[None] + 7500.0 * np.log(psfc[None] / pstag.clip(min=1.0))
    zstag[-1] = zstag[-2] + 4000.0
    zm = 0.5 * (zstag[1:] + zstag[:-1])
    zkm = zm / 1000.0
    theta = (255.0 if cold else 298.0) + 3.5 * zkm
    if vortex:
        theta = theta + 8.0 * np.exp(-(r2 / 60.0) ** 2)[None] * np.exp(-((zkm - 9.0) / 4.0) ** 2)
    qv = (0.003 if cold else 0.016) * np.exp(-zkm / 2.5)
    ring = np.exp(-((r2 - 48.0) / 20.0) ** 2)[None]
    if vortex:
        qc = 6e-4 * ring * (zkm < 6)
        qr = 1e-3 * ring * (zkm < 5)
        qi = 1e-4 * ring * (zkm > 8) * (zkm < 14)
        qs = 4e-4 * (ring + 0.3 * np.exp(-((r2 - 160.0) / 25.0) ** 2)[None]) * (zkm > 4) * (zkm < 12)
        qg = 2e-4 * ring * (zkm > 5) * (zkm < 9)
    else:
        band = np.exp(-((jj - jc) / 2.0) ** 2)[None] * (ii > 8)[None] * (ii < NX - 8)[None]
        qc = 2e-4 * band * (zkm < 1.8)
        qr = 0.0 * qc
        qi = 3e-5 * band * (zkm < 2.2)
        qs = 5e-4 * band * (zkm < 2.0)
        qg = 0.0 * qc

    def winds(xi, yj, zz):
        X = (xi - ic) * DX / 1000.0
        Y = (yj - jc) * DX / 1000.0
        r = np.hypot(X, Y)
        if not vortex:
            return np.full(np.broadcast(zz, X).shape, westerly), np.zeros(np.broadcast(zz, X).shape)
        vt = _rankine(r)[None] * np.clip(1.0 - zz / 16.0, 0.0, None)
        vr = -6.0 * np.exp(-zz / 1.0) * (r / (r + 50.0))[None] + 4.0 * np.exp(-((zz - 13) / 2.0) ** 2) * (r / (r + 50.0))[None]
        with np.errstate(invalid="ignore", divide="ignore"):
            c, s = np.where(r > 0, X / r, 0.0), np.where(r > 0, Y / r, 0.0)
        return vr * c - vt * s, vr * s + vt * c

    yy_u, xx_u = np.mgrid[0:NY, 0:NX + 1]
    U, _ = winds(xx_u - 0.5, yy_u, np.concatenate([zkm, zkm[:, :, -1:]], axis=2))
    yy_v, xx_v = np.mgrid[0:NY + 1, 0:NX]
    _, V = winds(xx_v, yy_v - 0.5, np.concatenate([zkm, zkm[:, -1:, :]], axis=1))
    zskm = zstag / 1000.0
    W = (1.5 * np.exp(-((r2 - 48.0) / 15.0) ** 2)[None] * np.sin(np.pi * np.clip(zskm, 0, 16) / 16.0)
         - 0.3 * np.exp(-(r2 / 20.0) ** 2)[None] * (zskm < 16)) if vortex else \
        0.4 * np.exp(-((jj - jc) / 2.0) ** 2)[None] * np.sin(np.pi * np.clip(zskm, 0, 2.5) / 2.5)
    u10, v10 = winds(ii.astype(float), jj.astype(float), np.zeros((1, NY, NX)))
    cloudy = (qc + qi + qs).sum(axis=0) > 1e-4
    tb = np.where(cloudy, 205.0 if vortex else 250.0, 285.0)
    with netCDF4.Dataset(path, "w") as d:
        for n, s in [("Time", 1), ("DateStrLen", 19), ("south_north", NY), ("west_east", NX),
                     ("south_north_stag", NY + 1), ("west_east_stag", NX + 1),
                     ("bottom_top", NZ), ("bottom_top_stag", NZ + 1)]:
            d.createDimension(n, s)
        d.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = \
            np.array([list(time_str)], dtype="S1")

        def var(name, dims, arr):
            d.createVariable(name, "f4", ("Time",) + dims)[:] = np.asarray(arr, "f4")[None]
        m2 = ("south_north", "west_east")
        m3 = ("bottom_top",) + m2
        s3 = ("bottom_top_stag",) + m2
        var("XLAT", m2, lat)
        var("XLONG", m2, lon)
        var("HGT", m2, hgt)
        var("LANDMASK", m2, (hgt > 0).astype(float))
        var("PSFC", m2, psfc)
        var("U10", m2, 0.8 * u10[0])
        var("V10", m2, 0.8 * v10[0])
        var("PBLH", m2, np.full((NY, NX), 1200.0))
        var("OLR", m2, SIGMA * tb ** 4)
        var("P", m3, np.zeros_like(pmass))
        var("PB", m3, pmass)
        var("T", m3, theta - 300.0)
        var("QVAPOR", m3, qv)
        if hydrometeors:
            for n, a in (("QCLOUD", qc), ("QRAIN", qr), ("QICE", qi), ("QSNOW", qs),
                         ("QGRAUP", qg)):
                var(n, m3, np.broadcast_to(a, pmass.shape))
        var("PH", s3, np.zeros_like(zstag))
        var("PHB", s3, zstag * 9.81)
        var("W", s3, np.broadcast_to(W, zstag.shape))
        var("U", ("bottom_top", "south_north", "west_east_stag"), U)
        var("V", ("bottom_top", "south_north_stag", "west_east"), V)
        var("MU", m2, np.zeros((NY, NX)))
        var("MUB", m2, psfc - P_TOP)
        var("DNW", ("bottom_top",), np.diff(SIGMA_STAG))
        var("C1H", ("bottom_top",), np.ones(NZ))
        var("C2H", ("bottom_top",), np.zeros(NZ))
        d.createVariable("P_TOP", "f4", ("Time",))[:] = [P_TOP]
        for k, v in dict(MAP_PROJ=1, TRUELAT1=30.0, TRUELAT2=60.0, STAND_LON=-89.0,
                         MOAD_CEN_LAT=26.0, CEN_LAT=26.0, CEN_LON=-89.0, POLE_LAT=90.0,
                         POLE_LON=0.0, DX=DX, DY=DX, P_TOP=P_TOP,
                         SIMULATION_START_DATE="2005-08-28_00:00:00",
                         TITLE=" OUTPUT FROM WRF V4.5.1 MODEL").items():
            d.setncattr(k, v)
    return path


def make_run(tmp_path, n=3, moving=True, **kw):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    for k, t in enumerate(TIMES[:n]):
        write_wrfout(str(run / ("wrfout_d01_" + t)), t,
                     centre=(25, 24 + (k if moving else 0)), **kw)
    return str(run)


CASE = dict(types=["common", "tc"], domains=[dict(domain=1, dx_km=12.0)])


# ---------------------------------------------------------------- numerics

def test_rankine_vortex_rotates_to_pure_tangential():
    X, Y = ts.grid_offsets_km((41, 41), 20, 20, 5.0)
    r = np.hypot(X, Y)
    vt_true = _rankine(r)
    with np.errstate(invalid="ignore", divide="ignore"):
        c, s = np.where(r > 0, X / r, 0), np.where(r > 0, Y / r, 0)
    u, v = -vt_true * s, vt_true * c                       # cyclonic (counter-clockwise)
    vr, vt = ts.radial_tangential(u, v, X, Y)
    assert np.allclose(vr, 0.0, atol=1e-9)
    assert np.allclose(vt, vt_true, atol=1e-9)
    assert (vt[r > 0] > 0).all()
    # leading (level) dimension broadcasts
    vr3, vt3 = ts.radial_tangential(np.stack([u, 2 * u]), np.stack([v, 2 * v]), X, Y)
    assert vt3.shape == (2, 41, 41) and np.allclose(vt3[1], 2 * vt_true)


def test_pure_inflow_is_negative_radial():
    X, Y = ts.grid_offsets_km((21, 21), 10, 10, 1.0)
    r = np.hypot(X, Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        u, v = np.where(r > 0, -3 * X / r, 0), np.where(r > 0, -3 * Y / r, 0)
    vr, vt = ts.radial_tangential(u, v, X, Y)
    assert np.allclose(vr[r > 0], -3.0) and np.allclose(vt, 0.0, atol=1e-12)


def test_azimuthal_mean_bins_and_coverage():
    dx = 4.0
    X, Y = ts.grid_offsets_km((101, 101), 50, 50, dx)
    r = np.hypot(X, Y)
    edges = ts.radius_edges(dx, 300.0)
    assert edges[0] == 0 and edges[1] == dx and edges[-1] == 300.0 and len(edges) == 76
    m = ts.azimuthal_mean(r, r, edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    ok = np.isfinite(m)
    assert np.allclose(m[ok][1:], centres[ok][1:], atol=0.6 * dx)
    const = ts.azimuthal_mean(np.full((3, 101, 101), 7.0), r, edges)
    assert const.shape == (3, 75) and np.allclose(const[np.isfinite(const)], 7.0)
    # the domain holds a full ring only to 200 km: coverage masking blanks the cut rings
    cov = ts.azimuthal_mean(r, r, edges, dx_km=dx)
    assert np.isfinite(cov[centres < 190]).all()
    assert np.isnan(cov[centres > 270]).all()
    # NaN cells are ignored, not propagated
    f = np.where(X > 0, np.nan, 1.0)
    assert np.allclose(ts.azimuthal_mean(f, r, edges)[1:40], 1.0)


def test_environment_annulus_full_and_small_domain_fallback():
    assert ts.environment_annulus(1200.0) == (500.0, 1000.0, False)
    r_in, r_out, reduced = ts.environment_annulus(300.0)
    assert reduced and r_out == 300.0 and r_in == 150.0
    with pytest.raises(ValueError):
        ts.environment_annulus(0.0)
    assert ts.full_annulus_radius_km((100, 200), 50, 60, 12.0, bdy_width=5) == (99 - 5 - 50) * 12.0
    assert ts.full_annulus_radius_km((100, 200), 3, 60, 12.0, bdy_width=5) == 0.0
    assert "largest full annulus" in ts.annulus_label((150.0, 300.0, True))
    assert "500-1000 km" in ts.annulus_label((500.0, 1000.0, False))


def test_temperature_anomaly_eye_minus_annulus():
    dx = 10.0
    X, Y = ts.grid_offsets_km((161, 161), 80, 80, dx)
    r = np.hypot(X, Y)
    t = 280.0 + 6.0 * np.exp(-(r / 40.0) ** 2)
    t3 = np.stack([t, t - 10.0])
    ann = ts.environment_annulus(ts.full_annulus_radius_km(r.shape, 80, 80, dx))
    assert ann[2] and ann[1] == 800.0
    env = ts.annulus_mean(t3, r, ann)
    assert np.allclose(env, [280.0, 270.0], atol=1e-6)
    prof = ts.eye_anomaly(t3, r, dx, ann)
    assert np.allclose(prof, 6.0 * np.mean(np.exp(-(np.array([0, dx, dx, dx, dx]) / 40.0) ** 2)),
                       atol=1e-6)


def test_rmw_profile_and_symmetric_levels():
    rc = np.array([5.0, 15.0, 25.0, 35.0])
    vt = np.array([[10, 30, 20, 5], [np.nan] * 4, [1, 2, 3, 4]], dtype=float)
    out = ts.rmw_profile(vt, rc)
    assert out[0] == 15.0 and np.isnan(out[1]) and out[2] == 35.0
    lv = ts.symmetric_levels(np.array([-2.3, 4.6]))
    assert lv[0] == -5 and lv[-1] == 5 and np.allclose(np.diff(lv), 1.0)


def test_centre_from_smoothed_slp_excludes_boundary():
    jj, ii = np.mgrid[0:40, 0:50]
    slp = 1010.0 - 30.0 * np.exp(-(((jj - 22) ** 2 + (ii - 31) ** 2) / 8.0))
    slp[0, 0] = 900.0                                       # boundary spike
    slp[20, 10] = 990.0                                     # single-cell spike: smoothed away
    assert ts.centre_from_slp(slp, bdy_width=5) == (22, 31)


def test_track_csv_and_hovmoller_selection(tmp_path):
    p = tmp_path / "track_domains_run0.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["time", "domain", "dx_km", "lat", "lon"])
        w.writeheader()
        w.writerow(dict(time=TIMES[0], domain="d02", dx_km=4, lat=25.5, lon=-88.0))
    assert ts.read_track_csv(str(p)) == {TIMES[0]: (25.5, -88.0, 2)}
    assert ts.read_track_csv(None) == {}
    times = ["2005-08-28_%02d:00:00" % h for h in range(24)]
    assert ts.hovmoller_frames(times) == times
    sel = ts.hovmoller_frames(times, cap=5)
    assert len(sel) == 5 and sel[0] == times[0] and sel[-1] == times[-1]
    assert ts.hovmoller_frames(times, every="6h") == times[::6]
    dts, arr = ts.hovmoller_assemble([(times[0], np.arange(4.0)), (times[1], None)], 4)
    assert arr.shape == (4, 2) and np.allclose(arr[:, 0], np.arange(4)) and np.isnan(arr[:, 1]).all()
    assert dts[1].hour == 1


# ---------------------------------------------------------------- with wrf-python

def test_frame_centre_from_slp_and_from_track(tmp_path):
    run = make_run(tmp_path, n=1)
    path = os.path.join(run, "wrfout_d01_" + TIMES[0])
    with netCDF4.Dataset(path) as ds:
        j, i, src = ts.frame_centre(ds, TIMES[0], {}, 5)
        assert (j, i) == (25, 24) and src == "minimum SLP (smoothed)"
        lat, lon = float(ds.variables["XLAT"][0, 30, 20]), float(ds.variables["XLONG"][0, 30, 20])
        j2, i2, src2 = ts.frame_centre(ds, TIMES[0], {TIMES[0]: (lat, lon, 1)}, 5)
        assert (j2, i2) == (30, 20) and src2 == "minimum-SLP track"


def test_rz_analysis_recovers_vortex_and_warm_core(tmp_path):
    run = make_run(tmp_path, n=1)
    path = os.path.join(run, "wrfout_d01_" + TIMES[0])
    rz = ts.rz_analysis(path, (25, 24, "test"), 5)
    k1 = int(np.argmin(np.abs(rz["z_km"] - 1.0)))
    vt1 = rz["vt"][k1]
    assert np.nanmax(vt1) > 30.0
    # RMW at ~1 km near the Rankine 48 km (one 12 km bin)
    assert abs(rz["rmw"][k1] - 48.0) <= 12.0
    assert np.nanmax(np.abs(rz["vr"][k1][:4])) < np.nanmax(vt1)
    assert rz["annulus"][2] is True                         # 600 km domain: fallback annulus
    k9 = int(np.argmin(np.abs(rz["z_km"] - 9.0)))
    tp9 = rz["tprime"][k9]
    assert tp9[0] > 2.0 and tp9[0] > np.nanmax(tp9[-5:]) + 2.0     # warm core aloft
    assert np.nanmax(rz["q_gkg"]) > 0.1


def _fake_volume(calls, raise_exc=None):
    def render(wrfout_path, out_png, work_dir, *, center_latlon=None, box_km=None,
               cutaway=False, title="", timeidx=0):
        calls.append(dict(path=wrfout_path, center=center_latlon, box=box_km, cutaway=cutaway))
        if raise_exc:
            raise raise_exc
        open(out_png, "wb").close()
        return out_png
    return render


def test_figures_render_every_tc_structure_figure(tmp_path, monkeypatch):
    from wtune.post.recipes import paraview3d
    calls = []
    monkeypatch.setattr(paraview3d, "render_condensate_volume", _fake_volume(calls),
                        raising=False)
    run = make_run(tmp_path)
    out = ts.figures(run, "run0", str(tmp_path / "out"), CASE)
    by_id = {f["id"]: f for f in out}
    want = ["tc_struct_reflectivity_2km", "tc_struct_ir_proxy", "tc_struct_rz_wind",
            "tc_struct_rz_warmcore", "tc_struct_rz_condensate", "tc_struct_xsec",
            "tc_struct_warmcore_hovmoller", "tc_struct_condensate_3d"]
    for fid in want:
        assert fid in by_id, fid
        assert not by_id[fid]["missing"], (fid, by_id[fid].get("reason"))
        assert os.path.exists(by_id[fid]["path"])
    assert "under-resolved" in by_id["tc_struct_reflectivity_2km"]["note"]
    assert "largest full annulus" in by_id["tc_struct_rz_warmcore"]["note"]
    assert "proxy, not a satellite-channel simulation" in by_id["tc_struct_ir_proxy"]["note"]
    assert calls and calls[0]["cutaway"] and calls[0]["box"] == 600.0
    assert calls[0]["center"] is not None and calls[0]["path"].endswith(TIMES[-1])


def test_ir_not_drawn_twice_when_cloud_set_runs_and_3d_missing(tmp_path, monkeypatch):
    from wtune.post.recipes import paraview3d
    monkeypatch.setattr(paraview3d, "render_condensate_volume",
                        _fake_volume([], ImportError("not yet")), raising=False)
    run = make_run(tmp_path, n=1)
    case = dict(CASE, types=["common", "tc", "cloud"])
    monkeypatch.setattr(ts, "rz_analysis", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    monkeypatch.setattr(ts, "reflectivity_2km_figure", lambda *a, **k: ("p", 1.0))
    monkeypatch.setattr(ts, "xsec_figure", lambda *a, **k: "p")
    out = ts.figures(run, "run0", str(tmp_path / "out"), case)
    ids = [f["id"] for f in out]
    assert "tc_struct_ir_proxy" not in ids
    m3d = [f for f in out if f["id"] == "tc_struct_condensate_3d"][0]
    assert m3d["missing"] and "not yet" in m3d["reason"]
    hov = [f for f in out if f["id"] == "tc_struct_warmcore_hovmoller"][0]
    assert hov["missing"] and "2+ frames" in hov["reason"]
    assert all(f["missing"] for f in out if f["id"].startswith("tc_struct_rz_"))


# ---------------------------------------------------------------- registration / wiring

def test_catalog_and_dispatch_registration():
    from wtune.post import caseinfo, recipe
    cat = recipe.load_catalog()
    for fid in ("tc_struct_reflectivity_2km", "tc_struct_ir_proxy", "tc_struct_rz_wind",
                "tc_struct_rz_warmcore", "tc_struct_rz_condensate", "tc_struct_xsec",
                "tc_struct_warmcore_hovmoller", "tc_struct_condensate_3d"):
        assert set(cat[fid]) == {"zh", "en"}
    assert "proxy, not a satellite-channel simulation" not in cat["tc_struct_ir_proxy"]["en"]["question"]
    assert "cloud" in caseinfo.ALL_TYPES_ORDER and "cloud" in recipe.TYPE_DISPATCH


def test_recipe_passes_track_csv_to_structure_set(tmp_path, monkeypatch):
    from wtune.post import recipe
    seen = {}

    def fake(run_dir, label, out_dir, case_info, track_csv=None, every=None, times=None,
             paraview=True):
        seen.update(track_csv=track_csv, out_dir=out_dir, every=every, paraview=paraview)
        return []
    monkeypatch.setattr(ts, "figures", fake)
    recipe._tc_structure_figures(["r"], ["run0"], str(tmp_path), CASE, every="6h")
    assert seen["track_csv"] is None and seen["every"] == "6h"
    os.makedirs(tmp_path / "tc")
    (tmp_path / "tc" / "track_domains_run0.csv").write_text("time,domain,lat,lon\n")
    recipe._tc_structure_figures(["r"], ["run0"], str(tmp_path), CASE)
    assert seen["track_csv"].endswith("track_domains_run0.csv")
    assert seen["out_dir"].endswith("tc_structure")
    recipe._tc_structure_figures(["r"], ["run0"], str(tmp_path), CASE, use_track=False)
    assert seen["track_csv"] is None


def test_no_code_paths_in_captions():
    import re
    from wtune.post import recipe
    cat = recipe.load_catalog()
    for fid, entry in cat.items():
        if fid.startswith(("tc_struct_", "cloud_")):
            for lang in ("zh", "en"):
                for text in entry[lang].values():
                    assert not re.search(r"post/|\.py\b", text), (fid, lang, text)


# ---------------------------------------------------------------- legends list only what is drawn

def _capture_legends_and_titles(monkeypatch):
    import matplotlib.axes
    import matplotlib.figure
    legends, titles = [], []
    real_legend = matplotlib.figure.Figure.legend
    real_title = matplotlib.axes.Axes.set_title

    def legend(self, *a, **kw):
        legends.append([h.get_label() for h in kw.get("handles", [])])
        return real_legend(self, *a, **kw)

    def set_title(self, label, *a, **kw):
        titles.append(" ".join(label.split()))
        return real_title(self, label, *a, **kw)
    monkeypatch.setattr(matplotlib.figure.Figure, "legend", legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", set_title)
    return legends, titles


def test_reflectivity_legend_lists_only_reached_wind_radii(tmp_path, monkeypatch):
    legends, titles = _capture_legends_and_titles(monkeypatch)
    calm = str(tmp_path / "calm")
    write_wrfout(calm, TIMES[0], vortex=False)                  # no wind at all
    ts.reflectivity_2km_figure(calm, TIMES[0], (25, 25, "test"), str(tmp_path / "c.png"),
                               "run0", 1)
    assert not any(l.endswith(" kt") for l in legends[-1])
    assert "no 10 m wind >= 34 kt" in titles[-1]
    storm = str(tmp_path / "storm")
    write_wrfout(storm, TIMES[0])
    ts.reflectivity_2km_figure(storm, TIMES[0], (25, 25, "test"), str(tmp_path / "s.png"),
                               "run0", 1)
    assert "10 m wind 34 kt" in legends[-1]
    assert "no 10 m wind" not in titles[-1]


def test_xsec_legend_merges_both_panels_and_lists_only_drawn(tmp_path, monkeypatch):
    legends, _ = _capture_legends_and_titles(monkeypatch)
    path = str(tmp_path / "storm")
    write_wrfout(path, TIMES[0])                                # w -0.3..1.5 m/s
    ts.xsec_figure(path, TIMES[0], (25, 25, "test"), str(tmp_path / "x.png"), "run0", 1)
    labels = legends[-1]
    assert len(labels) == len(set(labels))
    assert [l for l in labels if l.startswith("w down")] == ["w down -0.25 m s$^{-1}$"]
    assert "0 °C" in labels and "-20 °C" in labels


def test_rz_legend_omits_absent_terrain_and_negative_vt(tmp_path, monkeypatch):
    legends, _ = _capture_legends_and_titles(monkeypatch)
    path = str(tmp_path / "flat")
    write_wrfout(path, TIMES[0], hill=False)
    rz = ts.rz_analysis(path, (25, 25, "test"), 5)
    assert np.nanmin(rz["vt"]) > -5.0
    ts.rz_wind_figure(rz, str(tmp_path / "w.png"), "run0", 1, "t")
    labels = legends[-1]
    assert "RMW(z)" in labels
    assert not any("below ground" in l or "< 0" in l for l in labels)
