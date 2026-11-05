"""Defects found validating SCC26 Sundowner Run 1 (3 domains, Santa Barbara): post.times for the
one-time maps, wind overlay scope, accepted time formats and atomic `project set`, relative
--out, gust sampling note, T2/sea/tick appearance, recipe-driven station comparison."""
import os

import pytest
import yaml

from test_post_agent_control import _project, _run, _set, _post, TIMES   # noqa: F401
from wtune import project as wp


# -- C: time formats and atomic project set ------------------------------------------------------

@pytest.mark.parametrize("text", ["2022-04-05_06:00", "2022-04-05_06:00:00", "2022-04-05T06",
                                  "2022-04-05 06:00", "2022-04-05T06:00:00", "2022-04-05_06"])
def test_parse_iso_time_accepts_wrf_and_space_spellings(text):
    from datetime import datetime
    assert wp.parse_iso_time(text) == datetime(2022, 4, 5, 6, 0)


def test_project_set_stores_non_canonical_times_canonically(tmp_path):
    project = _project(tmp_path)
    assert _set(project, "--post-times", "2022-04-05_06:00:00,2022-04-05 09:30,2022-04-05T12") == 0
    assert _post(project)["times"] == ["2022-04-05T06:00", "2022-04-05T09:30", "2022-04-05T12"]


def test_bad_flag_rejects_all_names_flag_value_and_examples_and_writes_nothing(tmp_path, capsys):
    project = _project(tmp_path)
    before = (project / "project.yaml").read_text()
    assert _set(project, "--post-stations", "KSBA", "--post-times", "yesterday",
                "--post-station-method", "cubic", "--post-overlay", "wind10") == 1
    err = capsys.readouterr().err
    assert "--post-times" in err and "'yesterday'" in err and "2022-04-05_06:00:00" in err
    assert "--post-station-method" in err and "'cubic'" in err      # every bad flag, not the first
    assert (project / "project.yaml").read_text() == before        # valid flags were not written


# -- A: post.times / negligible rain decide the one-time-map time -------------------------------

def _focus_run(tmp_path, scale):
    import numpy as np
    import test_post_focus as tf
    real = tf._core(30, 40, 1.0)
    tf._write(str(tmp_path), [0 * real, 10 * real * scale, 30 * real * scale,
                              130 * real * scale, 140 * real * scale])
    return dict(types=["terrain"], domains=[dict(domain=3, dx_km=3.0)])


def test_map_time_uses_first_post_time_snapped_to_a_frame(tmp_path):
    from wtune.post import recipe
    case = _focus_run(tmp_path, 1.0)
    notes = []
    t = recipe._map_time(str(tmp_path), case, 6, None,
                         post_times=["2017-07-31T06:40", "2017-07-31T09:00"], notes=notes)
    assert t == "2017-07-31_07:00:00"           # not the rain peak 08:00
    assert notes == []


def test_map_time_post_time_outside_the_run_falls_back_with_a_note(tmp_path):
    from wtune.post import recipe
    case = _focus_run(tmp_path, 1.0)
    notes = []
    t = recipe._map_time(str(tmp_path), case, 6, None, post_times=["2017-08-05T00"], notes=notes)
    assert t == "2017-07-31_08:00:00"
    assert len(notes) == 1 and "outside" in notes[0]["text"]


def test_map_time_negligible_peak_rain_means_last_frame_and_a_note(tmp_path, capsys):
    from wtune.post import focus, recipe
    case = _focus_run(tmp_path, 1e-5)           # peak rate far below 0.1 mm/h
    notes = []
    assert recipe._map_time(str(tmp_path), case, 6, None, notes=notes) is None
    assert len(notes) == 1 and "below %g mm/h" % focus.MIN_PEAK_RATE_MM_H in notes[0]["text"]
    assert "last frame" in notes[0]["text"] and "--post-times" in notes[0]["text"]
    assert "last frame" in capsys.readouterr().out


def test_map_time_real_rain_still_uses_the_peak(tmp_path):
    from wtune.post import recipe
    case = _focus_run(tmp_path, 1.0)
    notes = []
    assert recipe._map_time(str(tmp_path), case, 6, None, notes=notes) == "2017-07-31_08:00:00"
    assert notes == []


def test_run_recipe_wires_post_times_and_the_note(tmp_path, monkeypatch):
    from wtune.post import recipe
    from wtune.post.recipes import common
    project = _project(tmp_path, "  types: auto\n  times: [2019-01-30T03]\n")
    run = _run(tmp_path)
    seen = {}
    monkeypatch.setattr(common, "figures", lambda *a, **k: [])
    for name in ("_data_products", "_time_series_figures", "_verification_advice"):
        monkeypatch.setattr(recipe, name, lambda *a, **k: [])
    monkeypatch.setattr(recipe, "_build_report", lambda *a, **k: str(tmp_path / "r.html"))
    monkeypatch.setattr(recipe.caseinfo, "detect_types", lambda *a, **k: dict(
        types=["common"], domains=[dict(domain=3)], evidence={}))
    real = recipe._map_time
    monkeypatch.setattr(recipe, "_map_time",
                        lambda *a, **k: seen.setdefault("t", real(*a, **k)))
    recipe._run_recipe([run], str(tmp_path / "out"), project_dir=str(project))
    assert seen["t"] == "2019-01-30_03:00:00"


def test_convection_maps_follow_map_time_and_none_means_last_frame(monkeypatch, tmp_path):
    from wtune.post.recipes import convection
    times = []

    def map_field(run_dir, domain, field, t, *a, **k):
        times.append(t)
        return "p.png"
    frames = [("2019-01-30_00:00:00", "p"), ("2019-01-30_03:00:00", "p"),
              ("2019-01-30_06:00:00", "p")]
    monkeypatch.setattr(convection, "list_frames", lambda r, domain=1: frames)
    monkeypatch.setattr(convection, "_post_spinup_frames", lambda r, d, s: [
        (t, p, 0) for t, p in frames[1:]])
    monkeypatch.setattr(convection, "_map_field", map_field)
    monkeypatch.setattr(convection, "_uh_time_max", lambda *a: (_ for _ in ()).throw(
        ValueError("x")))
    monkeypatch.setattr(convection, "_max_reflectivity_point", lambda *a: (_ for _ in ()).throw(
        IndexError("x")))
    monkeypatch.setattr(convection, "_radar_figures", lambda *a: [])
    info = {"domains": [{"domain": 1, "dx_km": 3.0, "cu_physics": 0}]}
    convection.figures(["/run"], ["r"], str(tmp_path), info, map_time="2019-01-30_02:00:00")
    assert set(times) == {"2019-01-30_03:00:00"}
    del times[:]
    convection.figures(["/run"], ["r"], str(tmp_path), info, map_time=None)
    assert set(times) == {"2019-01-30_06:00:00"}


# -- D: figures.json paths are relative to the post directory, whatever --out was ---------------

def test_relative_out_dir_gives_figures_json_paths_relative_to_it(tmp_path, monkeypatch):
    import json
    from wtune.post import check, recipe
    from wtune.post.recipes import common
    run = _run(tmp_path, ndom=1)
    monkeypatch.chdir(tmp_path)

    def figures(run_dirs, labels, out_dir, case_info, **kw):
        path = os.path.join(out_dir, "common_domain_map.png")
        with open(path, "wb") as fh:
            fh.write(b"png")
        return [recipe.fig("common_domain_map", path, labels[0], domain="d01")]
    monkeypatch.setattr(common, "figures", figures)
    for name in ("_data_products", "_verification_advice"):
        monkeypatch.setattr(recipe, name, lambda *a, **k: [])
    monkeypatch.setattr(recipe, "_build_report", lambda *a, **k: "r.html")
    monkeypatch.setattr(recipe.caseinfo, "detect_types", lambda *a, **k: dict(
        types=["common"], domains=[dict(domain=1)], evidence={}))
    res = recipe.run_recipe([run], os.path.join("post", "xx-recipe"))
    doc = json.load(open(os.path.join("post", "xx-recipe", "figures.json")))
    paths = [f["path"] for f in doc["figures"] if f.get("path")]
    assert paths == ["common_domain_map.png"]
    assert os.path.isabs(res["out_dir"])
    for p in paths:
        assert os.path.isfile(os.path.join("post", "xx-recipe", p))
    # review.json uses the same base: check finds the figure
    post = tmp_path / "post" / "xx-recipe"
    (post / "review.json").write_text(json.dumps(dict(
        schema="wtune-post-review/1", figures=[dict(path="common_domain_map.png",
                                                    verdict="ok", issues=[])])))
    errors, warnings = [], []
    check.check_review(str(post), errors, warnings)
    assert errors == []


def test_write_index_relativizes_cwd_relative_paths(tmp_path, monkeypatch):
    import json
    from wtune.post import recipe
    monkeypatch.chdir(tmp_path)
    os.makedirs("post/a")
    recipe.write_index("post/a", [], [], {}, [recipe.fig("x", "post/a/sub/x.png", "r")], [])
    assert json.load(open("post/a/figures.json"))["figures"][0]["path"] == "sub/x.png"


# -- B: wind10 overlay also on the time-series t2 / wspd10 maps ---------------------------------

def _fake_timeseries(tmp_path, monkeypatch):
    from wtune.post import plotting, timeseries
    run = _run(tmp_path, ndom=1)
    calls = []

    def plot_field(run_dir, field, out_dir, **kw):
        calls.append((field, kw.get("overlay_wind")))
        p = os.path.join(out_dir, "%s_x.png" % field)
        open(p, "wb").write(b"png")
        return [p]
    monkeypatch.setattr(plotting, "plot_field", plot_field)
    monkeypatch.setattr(timeseries, "fixed_scale", lambda *a, **k: (None, None, "neither"))
    return run, calls, timeseries


def test_timeseries_wind_overlay_on_t2_and_wspd10_only(tmp_path, monkeypatch):
    run, calls, timeseries = _fake_timeseries(tmp_path, monkeypatch)
    timeseries.figures_for_time_series(run, "r", 1, str(tmp_path / "o"), times=[TIMES[1]],
                                       overlay=["wind10"])
    assert dict(calls) == dict(slp=False, t2=True, wspd10=True, rain_acc=False, pw=False)


def test_timeseries_has_no_barbs_without_the_overlay(tmp_path, monkeypatch):
    run, calls, timeseries = _fake_timeseries(tmp_path, monkeypatch)
    timeseries.figures_for_time_series(run, "r", 1, str(tmp_path / "o"), times=[TIMES[1]])
    assert not any(v for _f, v in calls)


def test_recipe_passes_map_overlay_to_the_time_series(tmp_path, monkeypatch):
    from wtune.post import recipe
    from wtune.post.recipes import common
    project = _project(tmp_path, "  types: auto\n  map_overlay: [wind10]\n")
    run = _run(tmp_path)
    seen = {}
    monkeypatch.setattr(common, "figures", lambda *a, **k: [])
    for name in ("_data_products", "_verification_advice"):
        monkeypatch.setattr(recipe, name, lambda *a, **k: [])
    monkeypatch.setattr(recipe, "_time_series_figures",
                        lambda *a, **k: seen.update(k) or [])
    monkeypatch.setattr(recipe, "_build_report", lambda *a, **k: str(tmp_path / "r.html"))
    monkeypatch.setattr(recipe.caseinfo, "detect_types", lambda *a, **k: dict(
        types=["common"], domains=[dict(domain=3)], evidence={}))
    recipe._run_recipe([run], str(tmp_path / "out"), project_dir=str(project), every="3h")
    assert seen["overlay"] == ["wind10"]


# -- E: gust sampling bias ----------------------------------------------------------------------

def _gust_compare(tmp_path, out=None):
    import test_post_station_compare as ts
    run = ts._run(tmp_path)
    rows = [("A", 33.0, 101.0, 100, 0, "gust", 12.0), ("A", 33.0, 101.0, 100, 60, "gust", 11.0),
            ("A", 33.0, 101.0, 100, 0, "t2", 20.0)]
    return ts.sc.compare(str(run), str(ts._stations(tmp_path, rows)), ["A"], ["gust", "t2"],
                         out_dir=out)


def test_gust_stats_carry_a_sampling_note_and_n(tmp_path):
    res = _gust_compare(tmp_path)
    by = {s["variable"]: s for s in res["stats"]}
    assert by["gust"]["note"] == "obs only when gusts occur; model gust source = WSPD10MAX"
    assert by["t2"]["note"] == ""
    line = [n for n in res["notes"] if n.startswith("gust sampling bias")]
    assert len(line) == 1 and "n=2" in line[0]


def test_station_stats_csv_has_a_note_column_and_the_gust_panel_says_so(tmp_path, monkeypatch):
    import csv
    import matplotlib.axes
    titles = []
    real = matplotlib.axes.Axes.set_title
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title",
                        lambda self, t, *a, **k: titles.append(t) or real(self, t, *a, **k))
    _gust_compare(tmp_path, out=str(tmp_path / "o"))
    rows = {r["variable"]: r for r in csv.DictReader(open(tmp_path / "o" / "station_stats.csv"))}
    assert rows["gust"]["note"].startswith("obs only when gusts occur")
    assert rows["t2"]["note"] == ""
    assert any("gust (gust reports only)" in t for t in titles)
    assert not any("t2 (gust reports only)" in t for t in titles)


# -- G: recipe runs the station comparison; `post stations` reads the project --------------------

def _station_project(tmp_path, post, with_dataset=True):
    import test_post_station_compare as ts
    project = tmp_path / "p"
    (project / "analysis").mkdir(parents=True)
    (project / "project.yaml").write_text("name: p\npost:\n" + post)
    run = ts._run(tmp_path)
    if with_dataset:
        d = ts._stations(tmp_path, [("A", 33.0, 101.0, 100, 0, "t2", 20.0),
                                    ("A", 33.0, 101.0, 100, 60, "t2", 21.0),
                                    ("B", 33.0, 101.0, 100, 0, "t2", 19.0)], name="ds")
        obs = project / "input" / "obs"
        obs.mkdir(parents=True)
        os.rename(str(d), str(obs / "asos"))
    return project, run


def _recipe_stub(monkeypatch, tmp_path):
    from wtune.post import recipe
    from wtune.post.recipes import common
    monkeypatch.setattr(common, "figures", lambda *a, **k: [])
    for name in ("_data_products", "_time_series_figures", "_verification_advice"):
        monkeypatch.setattr(recipe, name, lambda *a, **k: [])
    monkeypatch.setattr(recipe, "_build_report", lambda *a, **k: str(tmp_path / "r.html"))
    monkeypatch.setattr(recipe, "_map_time", lambda *a, **k: None)
    monkeypatch.setattr(recipe.caseinfo, "detect_types", lambda *a, **k: dict(
        types=["common"], domains=[dict(domain=1)], evidence={}))
    return recipe


def test_recipe_runs_the_station_comparison_when_post_stations_is_set(tmp_path, monkeypatch):
    import json
    project, run = _station_project(
        tmp_path, "  types: auto\n  stations: [A]\n  station_vars: [t2]\n"
                  "  station_method: bilinear\n  calm_kt: 2.0\n")
    recipe = _recipe_stub(monkeypatch, tmp_path)
    from wtune.post import station_compare
    seen = {}
    real = station_compare.compare
    monkeypatch.setattr(station_compare, "compare",
                        lambda *a, **k: seen.update(a=a, k=k) or real(*a, **k))
    out = tmp_path / "out"
    res = recipe._run_recipe([str(run)], str(out), project_dir=str(project))
    assert seen["a"][2] == ["A"] and seen["a"][3] == ["t2"]
    assert seen["k"]["method"] == "bilinear" and seen["k"]["calm_kt"] == 2.0
    assert seen["k"]["out_dir"] == str(out)
    ids = [f["id"] for f in res["figures"]]
    assert ids.count("station_series") >= 1
    doc = json.load(open(out / "figures.json"))
    paths = [f["path"] for f in doc["figures"] if f["id"] == "station_series"]
    assert paths and all((out / p).is_file() for p in paths)       # registered, relative to out
    assert (out / "station_stats.csv").is_file()


def test_recipe_without_a_stations_dataset_notes_instead_of_failing(tmp_path, monkeypatch):
    project, run = _station_project(tmp_path, "  types: auto\n  stations: [A]\n",
                                    with_dataset=False)
    recipe = _recipe_stub(monkeypatch, tmp_path)
    res = recipe._run_recipe([str(run)], str(tmp_path / "out"), project_dir=str(project))
    assert [n for n in res["notes"] if n["source"] == "stations" and "obs-check" in n["text"]]
    assert not (tmp_path / "out" / "station_stats.csv").exists()


def test_recipe_without_post_stations_does_not_touch_stations(tmp_path, monkeypatch):
    project, run = _station_project(tmp_path, "  types: auto\n")
    recipe = _recipe_stub(monkeypatch, tmp_path)
    from wtune.post import station_compare
    monkeypatch.setattr(station_compare, "compare",
                        lambda *a, **k: pytest.fail("compare must not run"))
    res = recipe._run_recipe([str(run)], str(tmp_path / "out"), project_dir=str(project))
    assert not [n for n in res["notes"] if n["source"] == "stations"]


def test_post_stations_cli_reads_the_project_for_flags_not_given(tmp_path, monkeypatch):
    from wtune.post import cli, station_compare
    project, run = _station_project(
        tmp_path, "  types: auto\n  stations: [A]\n  station_vars: [t2, gust]\n"
                  "  station_method: bilinear\n  calm_kt: 2.5\n")
    seen = {}

    def fake(run_dir, obs, st, vs, **kw):
        seen.update(st=st, vs=vs, **kw)
        return dict(series=[], stats=[], notes=[], figures=[], files=[])
    monkeypatch.setattr(station_compare, "compare", fake)
    base = ["stations", "--run", str(run), "--obs", str(project / "input/obs/asos"),
            "--out", str(tmp_path / "o")]
    assert cli.main(base + ["--project", str(project)]) == 0
    assert (seen["st"], seen["vs"], seen["method"], seen["calm_kt"]) == (
        ["A"], ["t2", "gust"], "bilinear", 2.5)
    # a flag given on the command line wins; without --project the old defaults hold
    assert cli.main(base + ["--project", str(project), "--method", "nearest",
                            "--stations", "B"]) == 0
    assert seen["method"] == "nearest" and seen["st"] == ["B"] and seen["calm_kt"] == 2.5
    assert cli.main(base) == 0
    assert (seen["st"], seen["vs"], seen["method"], seen["calm_kt"]) == (
        ["all"], ["all"], "nearest", 3.0)


# -- F: T2 colour scale, sea colour, tick density, zoom extent ----------------------------------

def test_t2_scale_is_sequential_unless_the_data_crosses_freezing():
    import numpy as np
    from wtune.post.recipes.synoptic import t2_scale
    warm = np.linspace(2.0, 25.0, 500)
    lev, cmap, ext = t2_scale(warm)
    assert cmap == "viridis" and ext == "neither"
    assert lev[0] <= 2.0 and lev[-1] >= 25.0 and lev[0] > -5.0       # not -24 .. +24
    lev, cmap, ext = t2_scale(np.linspace(-12.0, 9.0, 500))
    assert cmap == "RdBu_r" and ext == "both" and lev[0] == -lev[-1] and 0.0 in lev
    assert t2_scale(np.linspace(-30.0, -10.0, 100))[1] == "viridis"   # all-frozen: no freezing line


def test_time_series_t2_scale_follows_the_data_range(tmp_path):
    import netCDF4
    import numpy as np
    from wtune.post import timeseries
    run = _run(tmp_path, ndom=1, times=[TIMES[0]])
    for name in os.listdir(run):
        with netCDF4.Dataset(os.path.join(run, name), "a") as d:
            lat = np.asarray(d["XLAT"][0])
            for nm, val in (("PSFC", 100000.0), ("Q2", 0.005)):
                d.createVariable(nm, "f4", ("Time", "south_north", "west_east"))[0] = val
            d.createVariable("T2", "f4", ("Time", "south_north", "west_east"))[0] = (
                273.15 + 2.0 + 23.0 * (lat - lat.min()) / (lat.max() - lat.min()))
    lev, cmap, ext = timeseries.fixed_scale(run, "t2", 1, [TIMES[0]])
    assert cmap == "viridis" and lev[0] > -5 and lev[-1] < 40
    assert "sequential" in timeseries.SCALE_RULE["t2"]


def test_sea_is_masked_by_landmask_else_by_height():
    import numpy as np
    from wtune.post import domain_map
    hgt = np.array([[0.0, 0.4, 3.0], [0.0, 10.0, 0.0]])
    assert np.isnan(domain_map.sea_masked(hgt)).tolist() == [[True, True, False], [True, False, True]]
    lm = np.array([[0, 1, 1], [1, 1, 0]])               # LANDMASK wins over HGT <= 0.5
    assert np.isnan(domain_map.sea_masked(hgt, lm)).tolist() == [[True, False, False],
                                                                 [False, False, True]]


def _capture_draw(monkeypatch):
    """The figure/axes draw() saved (draw closes it): [(fig, extent, contourf kwargs list)]."""
    import matplotlib.figure
    from cartopy.mpl.geoaxes import GeoAxes
    seen = dict(contourf=[], figs=[])
    real_cf = GeoAxes.contourf
    real_save = matplotlib.figure.Figure.savefig
    monkeypatch.setattr(GeoAxes, "contourf",
                        lambda self, *a, **k: seen["contourf"].append(k) or real_cf(self, *a, **k))

    def save(self, *a, **k):
        seen["figs"].append(self)
        return real_save(self, *a, **k)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", save)
    return seen


def test_domain_map_paints_the_sea_light_blue_and_zoom_uses_the_nest_extent(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import netCDF4
    import numpy as np
    from wtune.post import domain_map
    from wtune.post.recipes import common
    run = _run(tmp_path)
    for name in os.listdir(run):
        with netCDF4.Dataset(os.path.join(run, name), "a") as d:
            lat = np.asarray(d["XLAT"][0])
            d.createVariable("LANDMASK", "f4", ("Time", "south_north", "west_east"))[0] = (
                lat > np.median(lat)).astype("f4")
    n0 = len(plt.get_fignums())
    seen = _capture_draw(monkeypatch)
    out = tmp_path / "o"
    out.mkdir()
    (path, *_rest), = [(z[1],) for z in common.plot_domain_zooms(run, 3, str(out), "r", ["d03"])]
    assert os.path.isfile(path)
    assert any(k.get("colors") == [domain_map.SEA_COLOUR] for k in seen["contourf"])
    assert len(plt.get_fignums()) == n0                 # no orphan figure left open
    # the zoom axes show d03's box, not d01's: d03 spans ~1.2 deg of latitude
    fig = seen["figs"][-1]
    import cartopy.crs as ccrs
    ext = fig.axes[0].get_extent(crs=ccrs.PlateCarree())
    assert ext[3] - ext[2] < 2.0 and ext[1] - ext[0] < 2.5


def test_small_domain_gets_three_or_more_ticks_labelled_with_hemispheres():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from wtune.post import basemap
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    gl = basemap.outer_gridlines(ax)
    basemap.gridline_step(gl, 1.2, 1.5)                 # the 3 km Santa Barbara nest
    lon_ticks = gl.xlocator.tick_values(-120.3, -118.8)
    lat_ticks = gl.ylocator.tick_values(34.05, 35.26)
    assert len([t for t in lon_ticks if -120.3 <= t <= -118.8]) >= 3
    assert len([t for t in lat_ticks if 34.05 <= t <= 35.26]) >= 3
    assert gl.xformatter(-119.5).endswith("W") and "\N{DEGREE SIGN}" in gl.xformatter(-119.5)
    assert gl.yformatter(34.5).endswith("N")
    plt.close(fig)
    # a large map keeps the old fixed step
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    gl = basemap.outer_gridlines(ax)
    basemap.gridline_step(gl, 20.0, 25.0)
    assert list(gl.xlocator.locs)[:2] == [-180, -175]
    plt.close(fig)
