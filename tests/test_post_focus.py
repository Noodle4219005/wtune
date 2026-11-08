"""post/focus.py: the point and time the recipes' point figures and one-time maps use."""
import os

import netCDF4
import numpy as np

from wtune.post import focus

NY = NX = 60
TIMES = ["2017-07-31_00:00:00", "2017-07-31_06:00:00", "2017-07-31_07:00:00",
         "2017-07-31_08:00:00", "2017-07-31_09:00:00"]


def _core(j, i, amp):
    y, x = np.mgrid[0:NY, 0:NX]
    return amp * np.exp(-((x - i) ** 2 + (y - j) ** 2) / 20.0)


def _write(run, rain_by_time):
    lat = (20.0 + 0.03 * np.arange(NY))[:, None] * np.ones((1, NX))
    lon = np.ones((NY, 1)) * (118.0 + 0.03 * np.arange(NX))[None, :]
    for t, rain in zip(TIMES, rain_by_time):
        with netCDF4.Dataset(os.path.join(run, "wrfout_d03_%s" % t), "w") as d:
            d.DX = 3000.0
            for name, n in (("Time", 1), ("DateStrLen", 19), ("south_north", NY),
                            ("west_east", NX)):
                d.createDimension(name, n)
            d.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(t)], "S1")
            dims = ("Time", "south_north", "west_east")
            d.createVariable("XLAT", "f4", dims)[:] = lat[None]
            d.createVariable("XLONG", "f4", dims)[:] = lon[None]
            d.createVariable("RAINC", "f4", dims)[:] = np.zeros((1, NY, NX))
            d.createVariable("RAINNC", "f4", dims)[:] = rain[None]


def test_post_focus_box_restricts_the_search(tmp_path):
    acc = [np.zeros((NY, NX))] + [k * (_core(30, 40, 50.0) + _core(20, 20, 10.0))
                                  for k in range(1, 5)]
    _write(str(tmp_path), acc)
    lat = 20.0 + 0.03 * 20
    lon = 118.0 + 0.03 * 20
    f = focus.rain_focus(str(tmp_path), 3, spinup_hours=0,
                         box=[lat - 0.2, lat + 0.2, lon - 0.2, lon + 0.2])
    assert (abs(f["j"] - 20), abs(f["i"] - 20)) <= (1, 1) and "in post.focus" in f["rule"]


def test_frame_at_takes_the_nearest_frame():
    frames = [("2017-07-31_00:00:00", "a"), ("2017-07-31_03:00:00", "b"),
              ("2017-07-31_06:00:00", "c")]
    assert focus.frame_at(frames, "2017-07-31_05:00:00")[1] == "c"
    assert focus.frame_at(frames, "2017-07-31_01:30:00")[1] == "a"       # tie: earlier
    assert focus.frame_at(frames, None)[1] == "c"
    assert focus.frame_at([], "2017-07-31_05:00:00") is None


def test_one_time_maps_get_the_rain_peak(tmp_path, monkeypatch):
    # haitang2017: synoptic/IVT/PW/CAPE/cloud/terrain maps were drawn at F072, after the event
    from wtune.post import recipe
    from wtune.post.recipes import synoptic, terrain
    real = _core(30, 40, 1.0)
    _write(str(tmp_path), [0 * real, 10 * real, 30 * real, 130 * real, 140 * real])
    case = dict(types=["synoptic", "terrain"], domains=[dict(domain=3, dx_km=3.0)])
    peak = recipe._map_time(str(tmp_path), case, 6, None)
    assert peak == "2017-07-31_08:00:00"
    _write(str(tmp_path), [0 * real] * len(TIMES))
    assert recipe._map_time(str(tmp_path), case, 6, None) is None
    seen = {}
    for mod in (synoptic, terrain):
        def fake(*a, map_time=None, _m=mod.__name__, **k):
            seen[_m] = map_time
            return []
        monkeypatch.setattr(mod, "figures", fake)
    r = recipe._RecipeRun([str(tmp_path)], ["run0"], str(tmp_path / "out"), case,
                          map_time=peak)
    recipe._run_simple(recipe.TYPE_DISPATCH["synoptic"], r)
    recipe._run_terrain(recipe.TYPE_DISPATCH["terrain"], r)
    assert seen == {synoptic.__name__: peak, terrain.__name__: peak}


def test_valley_and_cloud_section_points_prefer_post_focus(monkeypatch, tmp_path):
    """haitang2017 run72: the valley time-height (26.21N 119.02E) and the d03 cloud section
    (26.61N 116.34E) were picked domain-wide, in Fujian, outside post.focus over Taiwan."""
    from wtune.post import recipe
    from wtune.post.focus import box_mask
    from wtune.post.recipes import cloud, terrain
    lat, lon = np.meshgrid(np.linspace(20, 27, 15), np.linspace(116, 123, 15), indexing="ij")
    box = [21.8, 25.4, 119.9, 122.1]
    inside = box_mask(lat, lon, box)
    hgt = np.full((15, 15), 500.0)
    hgt[12, 4] = 0.0                       # deepest valley, outside the box
    hgt[7, 9] = 100.0                      # shallower valley, inside
    bounds = (2, 13, 2, 13)
    assert terrain._find_valley_point(hgt, None, bounds) == (12, 4)
    assert terrain._find_valley_point(hgt, None, bounds, inside=inside) == (7, 9)
    hgt[7, 9] = 500.0                      # nothing qualifies inside: domain-wide pick
    assert terrain._find_valley_point(hgt, None, bounds, inside=inside) == (12, 4)
    twp = np.zeros((15, 15))
    twp[12, 4], twp[7, 9] = 5.0, 1.0
    assert cloud.max_column(twp, 2) == (12, 4)
    assert cloud.max_column(twp, 2, inside=inside) == (7, 9)
    assert box_mask(lat, lon, None) is None

    seen = {}
    monkeypatch.setattr(terrain, "figures", lambda *a, focus_box=None, **k:
                        seen.setdefault("terrain", focus_box) and [])
    monkeypatch.setattr(cloud, "figures", lambda *a, focus_box=None, **k:
                        seen.setdefault("cloud", focus_box) and [])
    case = dict(types=["terrain", "cloud"], domains=[dict(domain=3, dx_km=3.0)])
    r = recipe._RecipeRun([str(tmp_path)], ["run0"], str(tmp_path / "out"), case,
                          focus_box=box)
    recipe._run_terrain(recipe.TYPE_DISPATCH["terrain"], r)
    recipe._run_cloud(recipe.TYPE_DISPATCH["cloud"], r)
    assert seen == {"terrain": box, "cloud": box}


def test_precip_gets_the_rain_peak_for_its_obs_window(monkeypatch, tmp_path):
    from wtune.post import recipe
    from wtune.post.recipes import precip
    seen = {}

    def fake(run_dirs, labels, out_dir, case_info, spinup_hours=6, project_dir=None,
             map_time=None):
        seen["map_time"] = map_time
        return []
    monkeypatch.setattr(precip, "figures", fake)
    case = dict(types=["precip"], domains=[dict(domain=3, dx_km=3.0)])
    r = recipe._RecipeRun([str(tmp_path)], ["run0"], str(tmp_path / "out"), case,
                          map_time="2017-08-01_23:00:00")
    recipe._run_simple(recipe.TYPE_DISPATCH["precip"], r)
    assert seen == {"map_time": "2017-08-01_23:00:00"}
