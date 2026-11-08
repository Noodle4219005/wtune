"""post/recipes/cloud.py: section-line geometry along / perpendicular to a given wind (incl.
clipping at the domain edge), mean boundary-layer wind, max-TWP column, height interpolation
with the near-ground fill, the enhanced-IR palette, the "cloud" case type, and every figure
rendered once (matplotlib Agg) from a synthetic lake-effect-like wrfout on which wrf-python runs
for real (builder shared with tests/test_post_tc_structure.py). The 3-D volume (another module)
is monkeypatched.
"""
import os

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("wrf")
pytest.importorskip("pyproj")
pytest.importorskip("scipy")
pytest.importorskip("cartopy")

from wtune.post.recipes import cloud as cl  # noqa: E402
from test_post_tc_structure import TIMES, write_wrfout  # noqa: E402

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


def test_water_path_panels_omit_empty_species_but_keep_twp():
    z = np.zeros((4, 4))
    wp = dict(LWP=z + 50.0, RWP=z, IWP=z + 0.5, SWP=z + 300.0)
    wp["TWP"] = sum(wp.values())
    panels, omitted = cl.water_path_panels(wp)
    assert panels == ["LWP", "SWP", "TWP"] and omitted == ["RWP", "IWP"]
    assert np.allclose(wp["TWP"], 350.5)                             # TWP keeps every group
    p2, o2 = cl.water_path_panels(dict(LWP=z, TWP=z))
    assert p2 == ["TWP"] and o2 == ["LWP"]


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
    # the thresholds come from a lake-effect study; the title, drawn for every case type,
    # does not cite it
    assert "15/25 dBZ" in " ".join(titles[-1].split()) and "lake" not in titles[-1]


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
