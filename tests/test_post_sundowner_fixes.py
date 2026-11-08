"""Defects found validating SCC26 Sundowner Run 1 (3 domains, Santa Barbara): post.times for the
one-time maps, wind overlay scope, accepted time formats and atomic `project set`, relative
--out, gust sampling note, T2/sea/tick appearance, recipe-driven station comparison."""
import os


from test_post_agent_control import _project, _run, _set, _post, TIMES   # noqa: F401


# -- C: time formats and atomic project set ------------------------------------------------------


def test_project_set_stores_non_canonical_times_canonically(tmp_path):
    project = _project(tmp_path)
    assert _set(project, "--post-times", "2022-04-05_06:00:00,2022-04-05 09:30,2022-04-05T12") == 0
    assert _post(project)["times"] == ["2022-04-05T06:00", "2022-04-05T09:30", "2022-04-05T12"]


# -- A: post.times / negligible rain decide the one-time-map time -------------------------------

def _focus_run(tmp_path, scale):
    import test_post_focus as tf
    real = tf._core(30, 40, 1.0)
    tf._write(str(tmp_path), [0 * real, 10 * real * scale, 30 * real * scale,
                              130 * real * scale, 140 * real * scale])
    return dict(types=["terrain"], domains=[dict(domain=3, dx_km=3.0)])


def test_map_time_post_time_outside_the_run_falls_back_with_a_note(tmp_path):
    from wtune.post import recipe
    case = _focus_run(tmp_path, 1.0)
    notes = []
    t = recipe._map_time(str(tmp_path), case, 6, None, post_times=["2017-08-05T00"], notes=notes)
    assert t == "2017-07-31_08:00:00"
    assert len(notes) == 1 and "outside" in notes[0]["text"]


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
