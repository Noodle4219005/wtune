"""post/basemap.py: shared base map for every plan-view figure (user request 2026-10-01)."""
import os
import re
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from post import basemap  # noqa: E402


def _grid(ny=40, nx=50):
    lat = np.linspace(40.0, 44.0, ny)[:, None] * np.ones((1, nx))
    lon = np.linspace(-90.0, -84.0, nx)[None, :] * np.ones((ny, 1))
    hgt = 300.0 + 600.0 * np.exp(-((lat - 42) ** 2 + (lon + 87) ** 2))
    return lat, lon, hgt


def test_terrain_contours_and_caption_on_plain_axes():
    lat, lon, hgt = _grid()
    fig, ax = plt.subplots()
    ax.contourf(lon, lat, hgt)
    n0 = len(ax.collections)
    out = basemap.add_base_map(ax, None, lats=lat, lons=lon, hgt=hgt, nest_rings=[])
    assert out["terrain_interval_m"] and out["terrain_interval_m"] > 0
    assert len(ax.collections) > n0                     # contour lines were added
    assert "terrain HGT every" in out["caption"]
    plt.close(fig)


def test_terrain_skipped_when_shapes_differ():
    lat, lon, hgt = _grid()
    fig, ax = plt.subplots()
    out = basemap.add_base_map(ax, None, lats=lat, lons=lon, hgt=hgt[:-1], nest_rings=[])
    assert out["terrain_interval_m"] is None
    plt.close(fig)


def test_nest_ring_drawn_dashed():
    lat, lon, hgt = _grid()
    fig, ax = plt.subplots()
    ring = (2, [-88, -86, -86, -88, -88], [41, 41, 43, 43, 41])
    out = basemap.add_base_map(ax, None, lats=lat, lons=lon, terrain=False, nest_rings=[ring])
    assert out["nest"] == [2]
    assert any(l.get_linestyle() == "--" for l in ax.get_lines())
    plt.close(fig)


def test_cartopy_axes_get_gridline_labels():
    ccrs = pytest.importorskip("cartopy.crs")
    lat, lon, hgt = _grid()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(central_longitude=-87))
    ax.set_extent([-90, -84, 40, 44], crs=ccrs.PlateCarree())
    out = basemap.add_base_map(ax, ccrs, lats=lat, lons=lon, hgt=hgt, nest_rings=[])
    gl = out["gridliner"]
    assert gl is not None and gl.left_labels and gl.bottom_labels
    plt.close(fig)


def test_lambert_gridline_labels_outer_only(tmp_path):
    """0.8.1: on a Lambert-conformal WRF map cartopy's default x_inline put a row of longitude
    labels across the map interior; labels must sit outside the frame only (lon bottom, lat
    left, unrotated), and the base-map caption must not overprint the bottom labels."""
    ccrs = pytest.importorskip("cartopy.crs")
    lat = np.linspace(40.0, 44.0, 60)[:, None] * np.ones((1, 70))
    lon = np.linspace(-89.0, -83.0, 70)[None, :] * np.ones((60, 1))
    fig = plt.figure(figsize=(7, 6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
        central_longitude=-86, central_latitude=42, standard_parallels=(38, 46)))
    ax.set_extent([-89, -83, 40, 44], crs=ccrs.PlateCarree())
    ax.contourf(lon, lat, lat + lon, transform=ccrs.PlateCarree())
    gl = basemap.add_base_map(ax, ccrs, lats=lat, lons=lon, nest_rings=[])["gridliner"]
    assert gl.x_inline is False and gl.y_inline is False and gl.rotate_labels is False
    assert not gl.top_labels and not gl.right_labels
    assert gl.bottom_labels == "x" and gl.left_labels == "y"     # lon bottom only, lat left only
    fig.savefig(str(tmp_path / "lambert.png"), dpi=80)
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    shown = [t for t in gl.label_artists if t.get_visible()]
    bottom = [t.get_text() for t in gl.bottom_label_artists if t.get_visible()]
    left = [t.get_text() for t in gl.left_label_artists if t.get_visible()]
    assert bottom and all(t.endswith("W") for t in bottom)
    assert left and all(t.endswith("N") for t in left)
    assert len(shown) == len(bottom) + len(left)                  # nothing top/right/inline
    for t in shown:
        e = t.get_window_extent(renderer)
        cx, cy = 0.5 * (e.x0 + e.x1), 0.5 * (e.y0 + e.y1)
        assert not (frame.x0 < cx < frame.x1 and frame.y0 < cy < frame.y1), t.get_text()
        assert t.get_rotation() == 0
    cap = fig.texts[-1].get_window_extent(renderer)
    assert fig.texts[-1].get_text().startswith("Base map")
    assert not any(cap.overlaps(t.get_window_extent(renderer)) for t in shown)
    plt.close(fig)


def test_masked_barbs_interior_and_sentinels():
    lat, lon, _ = _grid(20, 20)
    u = np.full(lat.shape, 5.0)
    v = np.full(lat.shape, 5.0)
    u[10, 10] = 9.96921e36                               # wrf.interplevel below-ground fill
    fig, ax = plt.subplots()
    mask = basemap.interior_mask(lat.shape, bdy_width=5)
    b = basemap.masked_barbs(ax, lon, lat, u, v, 1, mask=mask)
    n = len(b.get_offsets())
    assert n == 10 * 10 - 1                              # 20x20 minus the 5-cell rim, minus sentinel
    xs = np.asarray(b.get_offsets())[:, 0]
    assert xs.min() > lon[0, 4] and xs.max() < lon[0, 15]
    plt.close(fig)


def test_masked_barbs_none_when_all_masked():
    lat, lon, _ = _grid(8, 8)
    fig, ax = plt.subplots()
    out = basemap.masked_barbs(ax, lon, lat, np.ones(lat.shape), np.ones(lat.shape), 1,
                               mask=np.zeros(lat.shape, bool))
    assert out is None
    plt.close(fig)


# every plan-view builder must go through the shared base map, never draw its own coastline
PLAN_VIEW_FILES = ["post/plotting.py", "post/obs_figs.py", "post/tc.py",
                   "post/recipes/mapfig.py", "post/recipes/common.py", "post/recipes/terrain.py",
                   "post/recipes/moisture.py", "post/recipes/synoptic.py",
                   "post/recipes/ensemble.py", "post/recipes/climate.py"]


@pytest.mark.parametrize("rel", PLAN_VIEW_FILES)
def test_no_private_coastline_drawing(rel):
    text = open(os.path.join(SRC, rel)).read()
    assert not re.search(r"cfeature\.(COASTLINE|BORDERS)|\.coastlines\(", text), rel
    assert "add_base_map" in text or "_coastline(" in text, rel
