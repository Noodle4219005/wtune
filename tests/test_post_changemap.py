"""Tests for post/changemap.py: the cdo-sub command construction and output naming (decision
0058 section 2 update: "差值一律用 cdo sub ... 命名在圖名後加間隔"), without a real cdo binary or
wrfout -- every external call (tools.find_on_path/run, post.extract.extract, netCDF4.Dataset,
post.recipes.mapfig.save_map) is faked so this stays a fast unit test of the plumbing, not an
integration test of cdo itself (that is covered on scc).
"""
import sys
import types

import numpy as np
import pytest

from wtune.post import changemap, tools


class _FakeVar:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, key):
        return self._arr[key]


class _FakeDataset:
    def __init__(self, path):
        self.path = path
        self.variables = {
            "XLAT": _FakeVar(np.zeros((1, 2, 2))),
            "XLONG": _FakeVar(np.zeros((1, 2, 2))),
            "slp": _FakeVar(np.array([[[1.0, 2.0], [3.0, 4.0]]])),
        }

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake_deps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "find_on_path", lambda name, install_hint=None: "/usr/bin/cdo")
    monkeypatch.setattr(tools, "run", lambda cmd, cwd=None, check=True: (calls.append(cmd), (0, "", ""))[1])

    extract_calls = []

    def fake_extract(run_dir, variables, out_path, domain=1, times=None, box=None, **kw):
        extract_calls.append((variables, times, box))
        open(out_path, "w").close()

    monkeypatch.setattr("wtune.post.extract.extract", fake_extract)
    monkeypatch.setitem(sys.modules, "netCDF4", types.SimpleNamespace(Dataset=_FakeDataset))

    import wtune.post.recipes.mapfig as mapfig
    monkeypatch.setattr(mapfig, "save_map", lambda *a, **kw: a[0])

    return dict(cdo_calls=calls, extract_calls=extract_calls)


def test_change_map_reports_missing_when_cdo_not_found(tmp_path, monkeypatch):
    def raise_not_found(name, install_hint=None):
        raise tools.ToolNotFoundError("cdo not found")
    monkeypatch.setattr(tools, "find_on_path", raise_not_found)

    record = changemap.change_map(
        "/fake/run", "run0", "slp", 1, str(tmp_path),
        "2005-08-27_00:00:00", "2005-08-27_06:00:00", "2005-08-27_00:00:00")
    assert record["missing"] is True
    assert record["dataset"] == "cdo"
    assert "cdo not found" in record["reason"]


def test_figures_for_change_maps_needs_at_least_two_times():
    assert changemap.figures_for_change_maps("/fake/run", "run0", 1, "/tmp/out", []) == []
    assert changemap.figures_for_change_maps("/fake/run", "run0", 1, "/tmp/out",
                                             ["2005-08-27_00:00:00"]) == []


def test_figures_for_change_maps_includes_rain_acc_period_accumulation(tmp_path, fake_deps):
    records = changemap.figures_for_change_maps(
        "/fake/run", "run0", 1, str(tmp_path),
        ["2005-08-27_00:00:00", "2005-08-27_06:00:00"], fields=("slp",))
    ids = [r["id"] for r in records]
    assert "chg_slp" in ids
    assert "chg_rain_acc" in ids


def test_a_tall_map_gets_a_tall_figure_not_blank_sides(tmp_path):
    # haitang2017 d03 (6 deg wide, 9 deg tall) on a fixed 8x6.4 left a third of the width blank
    pytest.importorskip("cartopy")
    import cartopy.crs as ccrs
    from PIL import Image
    from wtune.post.recipes import mapfig
    assert mapfig.map_height_in((0, 6, 0, 9)) == pytest.approx(6.2 * 1.5 + 1.6)
    assert mapfig.map_height_in((0, 80, 0, 45)) == pytest.approx(6.2 * 45 / 80 + 1.6)
    assert mapfig.map_height_in((0, 1, 0, 9)) == 11.0 and mapfig.map_height_in((0, 1, 0, 0)) == 6.4
    sizes = {}
    for name, (la0, la1, lo0, lo1) in {"tall": (19.4, 28.0, 116.3, 123.0),
                                       "wide": (-5.0, 51.0, 78.0, 160.0)}.items():
        lat, lon = np.meshgrid(np.linspace(la0, la1, 30), np.linspace(lo0, lo1, 25), indexing="ij")
        crs = ccrs.LambertConformal(central_longitude=(lo0 + lo1) / 2,
                                    central_latitude=(la0 + la1) / 2, standard_parallels=(20, 30))
        out = str(tmp_path / ("%s.png" % name))
        mapfig.save_map(out, lat, lon, "t", fill_field=lat, crs=crs)
        sizes[name] = Image.open(out).size
    assert sizes["tall"][0] == sizes["wide"][0]
    assert sizes["tall"][1] > sizes["tall"][0] > sizes["wide"][1]
