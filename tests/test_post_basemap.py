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

from wtune.post import basemap  # noqa: E402


def _grid(ny=40, nx=50):
    lat = np.linspace(40.0, 44.0, ny)[:, None] * np.ones((1, nx))
    lon = np.linspace(-90.0, -84.0, nx)[None, :] * np.ones((ny, 1))
    hgt = 300.0 + 600.0 * np.exp(-((lat - 42) ** 2 + (lon + 87) ** 2))
    return lat, lon, hgt


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


def test_masked_barbs_none_when_all_masked():
    lat, lon, _ = _grid(8, 8)
    fig, ax = plt.subplots()
    out = basemap.masked_barbs(ax, lon, lat, np.ones(lat.shape), np.ones(lat.shape), 1,
                               mask=np.zeros(lat.shape, bool))
    assert out is None
    plt.close(fig)


# every plan-view builder must go through the shared base map, never draw its own coastline
PLAN_VIEW_FILES = ["post/plotting.py", "post/obs_figs.py", "post/tc.py",
                   "post/recipes/mapfig.py", "post/domain_map.py", "post/recipes/terrain.py",
                   "post/recipes/moisture.py", "post/recipes/synoptic.py",
                   "post/recipes/ensemble.py", "post/recipes/climate.py"]


@pytest.mark.parametrize("rel", PLAN_VIEW_FILES)
def test_no_private_coastline_drawing(rel):
    text = open(os.path.join(SRC, "wtune", rel)).read()
    assert not re.search(r"cfeature\.(COASTLINE|BORDERS)|\.coastlines\(", text), rel
    assert "add_base_map" in text or "_coastline(" in text, rel


def test_barbs_thinned_interleaved_and_never_underground():
    # haitang2017 terrain_wind_d03 (2026-10-26): two barb sets on the same points, ~25 across,
    # and 700 hPa barbs over the Central Mountain Range
    import numpy as np
    from types import SimpleNamespace
    from wtune.post import basemap
    assert basemap.barb_stride((301, 221)) == 14 and basemap.barb_stride((40, 40)) == 4
    psfc = np.full((1, 30, 30), 1000e2)
    psfc[0, 10:20, 10:20] = 650e2
    ds = SimpleNamespace(variables={"PSFC": psfc})
    ok = basemap.above_ground(ds, 700.0)
    assert ok[0, 0] and not ok[15, 15]
    drawn = []
    ax = SimpleNamespace(barbs=lambda lo, la, u, v, **kw: drawn.append(list(zip(la, lo))))
    lat, lon = np.meshgrid(np.arange(30.0), np.arange(30.0), indexing="ij")
    u = np.ones((30, 30))
    basemap.masked_barbs(ax, lon, lat, u, u, 6, mask=ok)
    basemap.masked_barbs(ax, lon, lat, u, u, 6, mask=ok, offset=3)
    assert not set(drawn[0]) & set(drawn[1])
    assert not [p for p in drawn[0] + drawn[1] if 10 <= p[0] < 20 and 10 <= p[1] < 20]
