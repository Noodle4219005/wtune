"""post/recipes/cloud.py: section-line geometry along / perpendicular to a given wind (incl.
clipping at the domain edge), mean boundary-layer wind, max-TWP column, height interpolation
with the near-ground fill, the enhanced-IR palette, the "cloud" case type, and every figure
rendered once (matplotlib Agg) from a synthetic lake-effect-like wrfout on which wrf-python runs
for real (builder shared with tests/test_post_tc_structure.py). The 3-D volume (another module)
is monkeypatched.
"""
import math
import os

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("wrf")
pytest.importorskip("pyproj")
pytest.importorskip("scipy")
pytest.importorskip("cartopy")

from wtune.post.recipes import cloud as cl  # noqa: E402
from test_post_tc_structure import NX, NY, TIMES, write_wrfout  # noqa: E402

CASE = dict(types=["common", "precip", "cloud"], domains=[dict(domain=1, dx_km=12.0)])


def _lake_run(tmp_path, n=3, **kw):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    for t in TIMES[:n]:
        write_wrfout(str(run / ("wrfout_d01_" + t)), t, centre=(25, 25), vortex=False,
                     westerly=10.0, cold=True, **kw)
    return str(run)


# ---------------------------------------------------------------- geometry / numerics

def test_section_directions_along_and_perpendicular():
    along, perp, calm = cl.section_directions(3.0, 4.0)
    assert not calm
    assert along == pytest.approx((0.6, 0.8))
    assert perp == pytest.approx((-0.8, 0.6))            # wind rotated +90 deg (left)
    assert along[0] * perp[0] + along[1] * perp[1] == pytest.approx(0.0)
    assert cl.section_directions(0.1, 0.2) == ((1.0, 0.0), (0.0, 1.0), True)


def test_line_endpoints_follow_the_wind_and_clip_each_side():
    s, e, (back, fwd) = cl.line_endpoints(50, 40, 1.0, 0.0, 12.5, 200, 100)
    assert s == pytest.approx((37.5, 40)) and e == pytest.approx((62.5, 40))
    ux, uy = 0.6, 0.8
    s, e, (back, fwd) = cl.line_endpoints(50, 40, ux, uy, 20.0, 200, 100)
    d = (e[0] - s[0], e[1] - s[1])
    assert d[0] * uy - d[1] * ux == pytest.approx(0.0, abs=1e-9)     # parallel to the wind
    assert back == fwd == 20.0
    # near the east edge: the downwind side is shortened, the upwind side keeps its length
    s, e, (back, fwd) = cl.line_endpoints(195, 40, 1.0, 0.0, 12.5, 200, 100)
    assert e == pytest.approx((199, 40)) and fwd == pytest.approx(4.0) and back == 12.5
    # diagonal hitting the north edge first
    s, e, (back, fwd) = cl.line_endpoints(50, 95, ux, uy, 20.0, 200, 100)
    assert e[1] == pytest.approx(99.0) and 0 <= e[0] <= 199 and fwd == pytest.approx(4.0 / 0.8)
    x = cl.signed_distance_km(np.array([s, e]), 50, 95, ux, uy, 12.0)
    assert x[0] == pytest.approx(-back * 12.0) and x[1] == pytest.approx(fwd * 12.0)


def test_mean_bl_wind_and_max_column():
    z = np.array([25.0, 200.0, 600.0, 1100.0, 2000.0])
    u = np.array([2.0, 4.0, 6.0, 8.0, 30.0])
    v = np.zeros(5)
    assert cl.mean_bl_wind(u, v, z, 1000.0) == (4.0, 0.0)
    assert cl.mean_bl_wind(u, v, z, 5.0) == (2.0, 0.0)              # at least the lowest level
    f = np.zeros((30, 40))
    f[1, 1] = 99.0                                                  # boundary zone: ignored
    f[12, 20] = 5.0
    assert cl.max_column(f, 5) == (12, 20)


def test_interp_heights_fills_only_between_ground_and_lowest_level():
    z = np.broadcast_to(np.array([50.0, 300.0, 800.0])[:, None, None], (3, 2, 2)).copy()
    z[:, 1, 1] += 1000.0                                            # a column on a 1000 m hill
    hgt = np.array([[0.0, 0.0], [0.0, 1000.0]])
    f = np.broadcast_to(np.array([1.0, 2.0, 3.0])[:, None, None], (3, 2, 2))
    # (wrf.interplevel returns NaN for a level exactly equal to a model level, so the test
    # levels avoid 50/300/800 m)
    out = cl.interp_heights(f, z, [0.0, 400.0, 500.0], hgt)
    assert out[0, 0, 0] == 1.0                                      # filled to the ground
    assert np.isnan(out[0, 1, 1]) and np.isnan(out[1, 1, 1])        # below ground: NaN
    assert out[1, 0, 0] == pytest.approx(2.2) and out[2, 0, 0] == pytest.approx(2.4)


def test_ir_palette_grey_above_240_coloured_below():
    cmap, norm, levels = cl.ir_colormap()
    warm, mild = cmap(norm(300.0)), cmap(norm(245.0))
    assert warm[0] == warm[1] == warm[2] and mild[0] == mild[1] == mild[2]
    assert warm[0] < mild[0]                                        # warm = dark
    cold = cmap(norm(200.0))
    assert not (cold[0] == cold[1] == cold[2])                      # enhanced colour
    assert "proxy, not a satellite-channel simulation" in cl.IR_TITLE


def test_cloud_case_type_detected_from_hydrometeors(tmp_path):
    from wtune.post import caseinfo
    run = _lake_run(tmp_path, n=1)
    ok, ev = caseinfo.detect_cloud(run)
    assert ok and "QSNOW" in ev
    dry = tmp_path / "dry"
    dry.mkdir()
    write_wrfout(str(dry / ("wrfout_d01_" + TIMES[0])), TIMES[0], vortex=False,
                 hydrometeors=False)
    assert caseinfo.detect_cloud(str(dry))[0] is False
    assert caseinfo.detect_cloud(str(tmp_path / "none"))[0] is False


# ---------------------------------------------------------------- figures

def _fake_volume(calls, exc=None):
    def render(wrfout_path, out_png, work_dir, *, center_latlon=None, box_km=None,
               cutaway=False, title="", timeidx=0):
        calls.append(dict(center=center_latlon, box=box_km, cutaway=cutaway))
        if exc:
            raise exc
        open(out_png, "wb").close()
        return out_png
    return render


def test_figures_render_every_cloud_figure(tmp_path, monkeypatch):
    from wtune.post.recipes import paraview3d
    calls = []
    monkeypatch.setattr(paraview3d, "render_condensate_volume", _fake_volume(calls),
                        raising=False)
    run = _lake_run(tmp_path)
    out = cl.figures([run], ["run0"], str(tmp_path / "out"), CASE, every="3h")
    ids = [f["id"] for f in out]
    for f in out:
        assert not f["missing"], (f["id"], f.get("reason"))
        assert os.path.exists(f["path"])
    # B1 at every --every frame, the rest at the last frame only
    assert ids.count("cloud_water_path") == 3
    for fid in ("cloud_top_height", "cloud_xsec", "cloud_ir_proxy", "cloud_condensate_3d"):
        assert ids.count(fid) == 1
        assert [f for f in out if f["id"] == fid][0]["valid_time"] == TIMES[-1]
    assert calls == [dict(center=None, box=None, cutaway=False)]


def test_missing_entries_without_hydrometeors_or_3d(tmp_path, monkeypatch):
    from wtune.post.recipes import paraview3d
    monkeypatch.setattr(paraview3d, "render_condensate_volume",
                        _fake_volume([], NotImplementedError("later")), raising=False)
    dry = tmp_path / "dry"
    dry.mkdir()
    write_wrfout(str(dry / ("wrfout_d01_" + TIMES[0])), TIMES[0], vortex=False,
                 hydrometeors=False)
    out = cl.figures([str(dry)], ["run0"], str(tmp_path / "o1"), CASE)
    assert len(out) == 1 and out[0]["missing"] and "QCLOUD" in out[0]["reason"]
    run = _lake_run(tmp_path, n=1)
    monkeypatch.setattr(cl, "cloud_xsec_figure", lambda *a, **k: "x")
    monkeypatch.setattr(cl, "cloud_top_figure", lambda *a, **k: "x")
    out = cl.figures([run], ["run0"], str(tmp_path / "o2"), CASE)
    m = [f for f in out if f["id"] == "cloud_condensate_3d"][0]
    assert m["missing"] and "later" in m["reason"]
    assert [f["id"] for f in out if f["id"] == "cloud_water_path"] == ["cloud_water_path"]


def test_xsec_line_runs_along_the_bl_wind(tmp_path, monkeypatch):
    """The along-wind section of a westerly case is the grid row through the max-TWP column."""
    run = _lake_run(tmp_path, n=1)
    seen = []
    real = cl.section

    def spy(field, z, start, end, levels):
        seen.append((tuple(start), tuple(end)))
        return real(field, z, start, end, levels)
    monkeypatch.setattr(cl, "section", spy)
    p = str(tmp_path / "x.png")
    cl.cloud_xsec_figure(os.path.join(run, "wrfout_d01_" + TIMES[0]), TIMES[0], p, "run0", 1, 5)
    (s0, e0), (s1, e1) = seen[0], seen[3]
    assert s0[1] == pytest.approx(25) and e0[1] == pytest.approx(25) and e0[0] > s0[0]
    assert s1[0] == pytest.approx(e1[0])                          # perpendicular: along grid y
    assert math.hypot(e0[0] - s0[0], e0[1] - s0[1]) <= 2 * 150.0 / 12.0 + 1e-6
    assert 0 <= min(s0[0], e0[0]) and max(s0[0], e0[0]) <= NX - 1
    assert 0 <= min(s1[1], e1[1]) and max(s1[1], e1[1]) <= NY - 1


def test_recipe_dispatches_cloud_with_cadence(tmp_path, monkeypatch):
    from wtune.post import recipe
    seen = {}

    def fake(run_dirs, labels, out_dir, case_info, every=None, times=None, paraview=True):
        seen.update(out_dir=out_dir, every=every, paraview=paraview)
        return []
    monkeypatch.setattr(cl, "figures", fake)
    recipe.TYPE_DISPATCH["cloud"](["r"], ["run0"], str(tmp_path), CASE, paraview=False,
                                  every="6h")
    assert seen == dict(out_dir=os.path.join(str(tmp_path), "cloud"), every="6h", paraview=False)
    cat = recipe.load_catalog()
    for fid in ("cloud_water_path", "cloud_top_height", "cloud_xsec", "cloud_ir_proxy",
                "cloud_condensate_3d"):
        assert set(cat[fid]) == {"zh", "en"}


def test_water_path_panels_omit_empty_species_but_keep_twp():
    z = np.zeros((4, 4))
    wp = dict(LWP=z + 50.0, RWP=z, IWP=z + 0.5, SWP=z + 300.0)
    wp["TWP"] = sum(wp.values())
    panels, omitted = cl.water_path_panels(wp)
    assert panels == ["LWP", "SWP", "TWP"] and omitted == ["RWP", "IWP"]
    assert np.allclose(wp["TWP"], 350.5)                             # TWP keeps every group
    p2, o2 = cl.water_path_panels(dict(LWP=z, TWP=z))
    assert p2 == ["TWP"] and o2 == ["LWP"]


def test_multi_panel_gridline_labels_outer_only(tmp_path):
    run = _lake_run(tmp_path, n=1)
    import cartopy.crs as ccrs  # noqa: F401
    wrf = pytest.importorskip("wrf")
    with netCDF4.Dataset(os.path.join(run, "wrfout_d01_" + TIMES[0])) as ds:
        lats = np.asarray(ds.variables["XLAT"][0])
        lons = np.asarray(ds.variables["XLONG"][0])
        crs = wrf.get_cartopy(wrfin=ds)
        fig, axes = cl.plan_figure(crs, 2, 2)
        flags = []
        for k, ax in enumerate(axes[:3]):
            info = cl.finish_plan(ax, ds, lats, lons, crs, caption=False,
                                  outer_labels=(k % 2 == 0, k + 2 >= 3))
            gl = info["gridliner"]
            assert gl.x_inline is False and gl.y_inline is False
            assert not gl.top_labels and not gl.right_labels
            assert gl.left_labels in ("y", False) and gl.bottom_labels in ("x", False)
            flags.append((bool(gl.left_labels), bool(gl.bottom_labels)))
        cl._plt().close(fig)
    # 3 panels on 2x2: left column = 0, 2; bottom row = 2 and panel 1 (nothing below it)
    assert flags == [(True, False), (False, True), (True, True)]


def test_water_path_suptitle_names_omitted_panel(tmp_path, monkeypatch):
    run = _lake_run(tmp_path, n=1)                                 # QRAIN is all zero here
    titles = []
    import matplotlib.figure
    real = matplotlib.figure.Figure.suptitle
    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle",
                        lambda self, t, **kw: (titles.append(t), real(self, t, **kw))[1])
    cl.water_path_figure(os.path.join(run, "wrfout_d01_" + TIMES[0]), TIMES[0],
                         str(tmp_path / "wp.png"), "run0", 1)
    assert "RWP < 1 g m$^{-2}$ everywhere (omitted" in titles[0]


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
        titles.append(label)
        return real_title(self, label, *a, **kw)
    monkeypatch.setattr(matplotlib.figure.Figure, "legend", legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", set_title)
    return legends, titles


def _shift_mdbz(monkeypatch, offset):
    """cl._wrf() whose composite reflectivity is shifted by `offset` dBZ (the lake fixture
    peaks at ~28 dBZ)."""
    import types
    wrf_mod = cl._wrf()

    def getvar(ds, name, *a, **kw):
        out = wrf_mod.getvar(ds, name, *a, **kw)
        return np.asarray(out, "f8") + offset if name == "mdbz" else out
    fake = types.SimpleNamespace(getvar=getvar, get_cartopy=wrf_mod.get_cartopy)
    monkeypatch.setattr(cl, "_wrf", lambda: fake)


@pytest.mark.parametrize("offset, want", [(0.0, [15.0, 25.0]), (-10.0, [15.0]), (-20.0, [])])
def test_cloud_top_legend_lists_only_reached_echo_thresholds(tmp_path, monkeypatch, offset, want):
    run = _lake_run(tmp_path, n=1)
    _shift_mdbz(monkeypatch, offset)
    legends, titles = _capture_legends_and_titles(monkeypatch)
    cl.cloud_top_figure(os.path.join(run, "wrfout_d01_" + TIMES[0]), TIMES[0],
                        str(tmp_path / "ct.png"), "run0", 1)
    labels = legends[0] if legends else []
    assert labels == ["composite reflectivity %g dBZ" % v for v in want]
    assert ("no composite reflectivity >= 15 dBZ" in " ".join(titles[-1].split())) == (not want)


def test_contour_handles_only_for_levels_reached():
    labels = lambda hs: [h.get_label() for h in hs]               # noqa: E731
    assert labels(cl.w_contour_handles([0.0, 0.3])) == ["w up 0.25 m s$^{-1}$"]
    assert labels(cl.w_contour_handles([-0.6, 1.2])) == ["w up 0.25/0.5/1 m s$^{-1}$",
                                                          "w down -0.25/-0.5 m s$^{-1}$"]
    assert cl.w_contour_handles([0.0, 0.1]) == [] and cl.w_contour_handles([np.nan]) == []
    iso = [(0.0, "solid", 1.8, "0 °C"), (-15.0, "dashed", 1.0, "-15 °C")]
    assert labels(cl.isotherm_handles([-10.0, 5.0], iso)) == ["0 °C"]
    assert labels(cl.isotherm_handles([-20.0, 5.0], iso)) == ["0 °C", "-15 °C"]
    assert cl.isotherm_handles([5.0, 9.0], iso) == []
    assert cl.echo_thresholds_reached([10.0, 18.0]) == [15.0]
    assert cl.echo_thresholds_reached([np.nan]) == []


def test_cloud_xsec_legend_lists_only_drawn_contours(tmp_path, monkeypatch):
    """Lake fixture: w is 0..0.38 m/s (only the +0.25 contour exists, no downdraft)."""
    run = _lake_run(tmp_path, n=1)
    legends, _ = _capture_legends_and_titles(monkeypatch)
    cl.cloud_xsec_figure(os.path.join(run, "wrfout_d01_" + TIMES[0]), TIMES[0],
                         str(tmp_path / "x.png"), "run0", 1, 5)
    labels = legends[0]
    assert "w up 0.25 m s$^{-1}$" in labels
    assert not any(l.startswith("w down") for l in labels)
    assert "PBLH" in labels
