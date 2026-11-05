"""Agent-controlled post-processing: the `wtune project set --post-*` flags (times, map domains,
figure switches, stations, wind overlay), their project.yaml keys, and how the recipe, the
domain map and `wtune post check` use them."""
import json
import os

import netCDF4
import numpy as np
import pytest
import yaml

from wtune import project as wp
from wtune.post import check, domain_map, recipe
from wtune.post import cli as post_cli
from wtune.post.recipes import common

TIMES = ["2019-01-30_00:00:00", "2019-01-30_03:00:00", "2019-01-30_06:00:00",
         "2019-01-30_09:00:00"]
# (south, north, west, east, nx, ny) -- the third box is the Santa Barbara style nest
BOXES = {1: (30.0, 40.0, -125.0, -113.0, 40, 36), 2: (33.0, 36.5, -121.5, -117.5, 40, 36),
         3: (34.05, 35.26, -120.31, -118.83, 40, 36)}


def _project(tmp_path, post="  types: auto\n"):
    project = tmp_path / "p"
    (project / "analysis").mkdir(parents=True)
    (project / "project.yaml").write_text("name: p\npost:\n" + post)
    return project


def _wrfout(path, dom, t):
    s, n, w, e, nx, ny = BOXES[dom]
    lat = np.linspace(s, n, ny)[:, None] * np.ones((1, nx))
    lon = np.ones((ny, 1)) * np.linspace(w, e, nx)[None, :]
    hgt = 800.0 * np.exp(-(((lat - 34.6) / 0.6) ** 2 + ((lon + 119.8) / 0.6) ** 2)) + 5.0
    with netCDF4.Dataset(path, "w") as d:
        d.DX = 27000.0 / 3 ** (dom - 1)
        d.MAP_PROJ, d.TRUELAT1, d.TRUELAT2 = 1, 30.0, 60.0
        d.STAND_LON, d.MOAD_CEN_LAT, d.CEN_LON, d.POLE_LAT, d.POLE_LON = -119.0, 35.0, -119.0, 90.0, 0.0
        for name, size in (("Time", None), ("DateStrLen", 19), ("south_north", ny),
                           ("west_east", nx)):
            d.createDimension(name, size)
        d.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(t)], "S1")
        dims = ("Time", "south_north", "west_east")
        for name, arr in (("XLAT", lat), ("XLONG", lon), ("HGT", hgt),
                          ("U10", np.full((ny, nx), 5.0)), ("V10", np.full((ny, nx), -2.0)),
                          ("COSALPHA", np.ones((ny, nx))), ("SINALPHA", np.zeros((ny, nx)))):
            d.createVariable(name, "f4", dims)[:] = arr[None]


def _run(tmp_path, ndom=3, times=TIMES):
    run = tmp_path / "run"
    run.mkdir()
    for dom in range(1, ndom + 1):
        for t in times:
            _wrfout(str(run / ("wrfout_d%02d_%s" % (dom, t))), dom, t)
    return str(run)


def _set(project, *args):
    return wp.main(["set", "--project", str(project)] + list(args))


def _post(project):
    return yaml.safe_load((project / "project.yaml").read_text())["post"]


def test_every_post_flag_is_written_and_read_back(tmp_path, capsys):
    project = _project(tmp_path)
    assert _set(project, "--post-times", "2019-01-30T06,2019-01-30T09:30",
                "--post-map-domains", "d01,d03", "--post-figures-off", "timeseries_slp,anim_t2",
                "--post-figures-on", "common_domain_map_zoom",
                "--post-stations", "KSBA,KSMX1", "--post-station-vars", "t2,wspd10,wdir10,gust",
                "--post-station-method", "bilinear", "--post-calm-kt", "3",
                "--post-overlay", "wind10") == 0
    post = _post(project)
    assert post["types"] == "auto"                      # untouched keys survive
    assert post["times"] == ["2019-01-30T06", "2019-01-30T09:30"]
    assert post["map_domains"] == ["d01", "d03"]
    assert post["figures_off"] == ["timeseries_slp", "anim_t2"]
    assert post["figures_on"] == ["common_domain_map_zoom"]
    assert post["stations"] == ["KSBA", "KSMX1"]
    assert post["station_vars"] == ["t2", "wspd10", "wdir10", "gust"]
    assert post["station_method"] == "bilinear" and post["calm_kt"] == 3.0
    assert post["map_overlay"] == ["wind10"]
    back = wp.project_post_defaults(wp.load_project(str(project)))
    assert back["times"] == post["times"] and back["calm_kt"] == 3.0
    assert back["map_overlay"] == ["wind10"] and back["station_method"] == "bilinear"
    assert "post drawing options" in capsys.readouterr().out
    info = wp.info_project(str(project))
    assert "map_domains=['d01', 'd03']" in info and "station_method=bilinear" in info


def test_none_clears_and_all_collapses(tmp_path):
    project = _project(tmp_path)
    assert _set(project, "--post-map-domains", "d02,all", "--post-times", "2019-01-30T06") == 0
    assert _post(project)["map_domains"] == ["all"]
    assert _set(project, "--post-map-domains", "none", "--post-times", "none") == 0
    back = wp.project_post_defaults(wp.load_project(str(project)))
    assert back["map_domains"] == [] and back["times"] == []


@pytest.mark.parametrize("args, message", [
    (["--post-times", "2019/01/30 06"], "YYYY-MM-DDTHH"),
    (["--post-times", "2019-01-30T25"], "YYYY-MM-DDTHH"),
    (["--post-stations", "KSBA,K-1"], "letters and digits"),
    (["--post-station-vars", "t2,rain"], "unknown variable rain"),
    (["--post-station-method", "cubic"], "nearest|bilinear"),
    (["--post-calm-kt", "0"], "calm threshold"),
    (["--post-calm-kt", "abc"], "calm threshold"),
    (["--post-overlay", "slp"], "unknown overlay slp"),
    (["--post-map-domains", "3"], "dNN"),
    (["--post-figures-off", "no_such_figure"], "unknown figure id no_such_figure"),
    (["--post-figures-on", "timeseries_slp,bogus"], "unknown figure id bogus"),
])
def test_bad_values_are_rejected_with_a_specific_message(tmp_path, capsys, args, message):
    project = _project(tmp_path)
    before = (project / "project.yaml").read_text()
    assert _set(project, *args) == 1
    assert message in capsys.readouterr().err
    assert (project / "project.yaml").read_text() == before


def test_unknown_figure_error_lists_catalog_ids(tmp_path, capsys):
    project = _project(tmp_path)
    assert _set(project, "--post-figures-off", "nope") == 1
    err = capsys.readouterr().err
    assert "common_domain_map," in err and "first 40 of" in err


# -- post check ------------------------------------------------------------------------------

def _review_with(tmp_path, project, args):
    post = project / "post" / "1"
    (post / "figures").mkdir(parents=True, exist_ok=True)
    (post / "figures" / "m.png").write_bytes(b"png")
    (post / "review.json").write_text(json.dumps(dict(
        schema="wtune-post-review/1", figures=[dict(
            path="figures/m.png", verdict="fix", issues=["d03 not shaded"],
            action=dict(kind="project_set", args=args))])))
    errors, warnings = [], []
    check.check_review(str(post), errors, warnings, project_dir=str(project))
    return errors, warnings


def test_check_accepts_the_new_flags_and_rejects_unknown_ones(tmp_path):
    project = _project(tmp_path)
    good = ["--post-times", "2019-01-30T06", "--post-map-domains", "d01,d03",
            "--post-figures-off", "timeseries_slp", "--post-overlay", "wind10",
            "--post-stations", "KSBA", "--post-station-vars", "t2,gust"]
    errors, _w = _review_with(tmp_path, project, good)
    assert errors == []
    errors, _w = _review_with(tmp_path, project, ["--post-figures-off", "not_a_figure"])
    assert any("rejected" in e and "not_a_figure" in e for e in errors)
    errors, _w = _review_with(tmp_path, project, ["--post-times", "yesterday"])
    assert any("rejected" in e and "yesterday" in e for e in errors)
    errors, _w = _review_with(tmp_path, project, ["--post-bogus", "x"])
    assert any("project_set args must use" in e for e in errors)
    errors, _w = _review_with(tmp_path, project, ["--machine", "x"])
    assert any("project_set args must use" in e for e in errors)


# -- recipe: times priority ------------------------------------------------------------------

def _capture_recipe(monkeypatch):
    seen = {}

    def fake(run_dirs, out, **kw):
        seen.update(kw)
        os.makedirs(out, exist_ok=True)
        return dict(out_dir=out, figures=[], notes=[], report_path=out + "/report.html",
                    case_info=dict(types=["common"]))
    monkeypatch.setattr(recipe, "run_recipe", fake)
    return seen


def test_times_priority_cli_then_project_times_then_every(tmp_path, monkeypatch):
    project = _project(tmp_path, "  types: auto\n  every: 6h\n  times: [2019-01-30T06]\n")
    run = _run(tmp_path, ndom=1)
    seen = _capture_recipe(monkeypatch)

    def go(*extra):
        seen.clear()
        assert post_cli.main(["recipe", run, "--project", str(project)] + list(extra)) == 0
        return seen.get("every"), seen.get("times")

    # post.times beats post.every
    assert go() == (None, ["2019-01-30_06:00:00"])
    # the command line beats post.times (and post.every)
    assert go("--times", "2019013009") == (None, ["2019013009"])
    # a 30-min offset snaps to the nearest frame later, but the project time is kept exact here
    assert _set(project, "--post-times", "2019-01-30T03:00") == 0
    assert go()[1] == ["2019-01-30_03:00:00"]
    # without post.times the project's every applies again
    assert _set(project, "--post-times", "none") == 0
    assert go() == ("6h", None)
    # --every on the command line still wins over the project default every
    assert go("--every", "3h") == ("3h", None)


def test_project_time_outside_the_run_names_the_available_range(tmp_path, monkeypatch, capsys):
    project = _project(tmp_path, "  types: auto\n  times: [2019-02-05T00]\n")
    run = _run(tmp_path, ndom=1)
    _capture_recipe(monkeypatch)
    assert post_cli.main(["recipe", run, "--project", str(project)]) == 1
    err = capsys.readouterr().err
    assert "2019-02-05T00" in err and "2019-01-30T00:00 .. 2019-01-30T09:00" in err
    assert "4 frames" in err


# -- domain map ------------------------------------------------------------------------------

def _png(path):
    return os.path.isfile(path) and os.path.getsize(path) > 5000


def test_default_domain_map_shades_only_d01_and_draws_no_zoom(tmp_path, monkeypatch):
    run = _run(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    path = common.plot_domain_map(run, 3, str(out), "r")
    assert _png(path) and not (out / "common_domain_map_d03.png").exists()
    calls = []
    real = domain_map.draw
    monkeypatch.setattr(domain_map, "draw", lambda *a, **k: calls.append(k) or real(*a, **k))
    common.plot_domain_map(run, 3, str(out), "r")
    assert calls[0]["extra_terrain"] == []              # d01 terrain only


def test_map_domains_shades_nests_and_draws_a_zoom_per_nest(tmp_path, monkeypatch):
    run = _run(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    calls = []
    real = domain_map.draw
    monkeypatch.setattr(domain_map, "draw", lambda *a, **k: calls.append(k) or real(*a, **k))
    path = common.plot_domain_map(run, 3, str(out), "r", map_domains=["d01", "d03"])
    assert _png(path) and len(calls[0]["extra_terrain"]) == 1      # d03 shaded over d01
    zooms = common.plot_domain_zooms(run, 3, str(out), "r", ["d01", "d03"])
    assert [(z[0], z[3]) for z in zooms] == [(3, None)] and _png(zooms[0][1])
    assert zooms[0][1].endswith("common_domain_map_d03.png")
    assert calls[-1]["zoom"]["dom"] == 3 and calls[-1]["wind"] is None
    zooms = common.plot_domain_zooms(run, 3, str(out), "r", ["all"])
    assert [z[0] for z in zooms] == [2, 3]


def test_wind_overlay_reads_u10_v10_at_the_requested_time(tmp_path):
    run = _run(tmp_path)
    t, lat, lon, u, v = domain_map.wind10_at(run, 3, "2019-01-30_06:00:00")
    assert t == "2019-01-30_06:00:00" and u.shape == lat.shape
    assert float(u[0, 0]) == pytest.approx(5.0) and float(v[0, 0]) == pytest.approx(-2.0)
    assert domain_map.wind10_at(run, 3, None)[0] == TIMES[-1]      # no time: the last frame
    out = tmp_path / "out"
    out.mkdir()
    zooms = common.plot_domain_zooms(run, 3, str(out), "r", ["d03"], ["wind10"],
                                     "2019-01-30_06:00:00")
    assert zooms[0][2] == "2019-01-30_06:00:00" and _png(zooms[0][1])


def test_overlay_without_u10_is_a_note_not_a_crash(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for dom in (1, 3):
        p = str(run / ("wrfout_d%02d_%s" % (dom, TIMES[0])))
        _wrfout(p, dom, TIMES[0])
        with netCDF4.Dataset(p, "a") as d:
            d.renameVariable("U10", "U10X")
    out = tmp_path / "out"
    out.mkdir()
    zooms = common.plot_domain_zooms(str(run), 3, str(out), "r", ["d03"], ["wind10"])
    assert "no U10/V10" in zooms[0][3] and _png(zooms[0][1])


def test_the_d03_label_sits_outside_its_box():
    doms = [dict(dom=1, x0=0, x1=1000, y0=0, y1=800), dict(dom=3, x0=400, x1=500, y0=300, y1=360),
            dict(dom=2, x0=400, x1=700, y0=300, y1=500)]
    a = domain_map._label_anchors(doms, 1000, 800)
    assert a[1][1] > 360                                # above d03's top edge, not inside it
    assert a[2][1] - a[1][1] > 0.02 * 800               # the neighbouring label is lifted off it


def test_recipe_runs_the_common_figures_with_project_map_domains(tmp_path, monkeypatch):
    project = _project(tmp_path, "  types: auto\n  map_domains: [d03]\n  map_overlay: [wind10]\n"
                                 "  times: [2019-01-30T06]\n")
    run = _run(tmp_path)
    seen = {}

    def fake_figures(run_dirs, labels, out_dir, case_info, **kw):
        seen.update(kw)
        return []
    monkeypatch.setattr(common, "figures", fake_figures)
    for name in ("_data_products", "_time_series_figures", "_verification_advice"):
        monkeypatch.setattr(recipe, name, lambda *a, **k: [])
    monkeypatch.setattr(recipe, "_build_report", lambda *a, **k: str(tmp_path / "r.html"))
    monkeypatch.setattr(recipe, "_type_tasks", lambda *a, **k: [])
    monkeypatch.setattr(recipe, "_map_time", lambda *a, **k: None)
    monkeypatch.setattr(recipe.caseinfo, "detect_types", lambda *a, **k: dict(
        types=["common"], domains=[dict(domain=3)], evidence={}))
    recipe._run_recipe([run], str(tmp_path / "out"), project_dir=str(project))
    assert seen["map_domains"] == ["d03"] and seen["overlay"] == ["wind10"]
    assert seen["overlay_time"] == "2019-01-30_06:00:00"


# -- figures_off / figures_on ----------------------------------------------------------------

def test_figures_off_drops_the_figure_from_the_index_and_the_disk(tmp_path):
    out = tmp_path / "out"
    (out / "timeseries").mkdir(parents=True)
    for name in ("a.png", "b.png"):
        (out / "timeseries" / name).write_bytes(b"png")
    figs = [recipe.fig("timeseries_slp", str(out / "timeseries" / "a.png"), "r"),
            recipe.fig("timeseries_t2", str(out / "timeseries" / "b.png"), "r"),
            recipe.missing("anim_slp", "ffmpeg", "x")]
    kept, notes = recipe.apply_figure_switches(str(out), figs, ["timeseries_slp", "anim_slp"],
                                               ["common_domain_map_zoom"])
    assert [f["id"] for f in kept] == ["timeseries_t2"]
    assert not (out / "timeseries" / "a.png").exists() and (out / "timeseries" / "b.png").exists()
    text = " ".join(n["text"] for n in notes)
    assert "nothing was forced" in text and "common_domain_map_zoom" in text
    recipe.write_index(str(out), ["r"], ["r"], {}, kept, [])
    doc = json.load(open(out / "figures.json"))
    assert [f["id"] for f in doc["figures"]] == ["timeseries_t2"]


def test_a_figure_outside_the_output_directory_is_never_deleted(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    other = tmp_path / "keep.png"
    other.write_bytes(b"png")
    kept, _n = recipe.apply_figure_switches(str(out), [recipe.fig("timeseries_slp", str(other),
                                                                   "r")], ["timeseries_slp"], [])
    assert kept == [] and other.exists()
