"""Regression tests for the post/ findings of the full-source review (8bdcd3c): fail-closed
`post compare`, report sections that drop nothing, truncated downloads, per-type / per-field
failure isolation in `post recipe`, bucketed rain, earth-relative tc_rmw winds, a-deck init
time, sub-hourly point_stat tags, --srun-minutes."""
import gzip
import io
import os
import sys
import types
import zlib
from datetime import datetime

import numpy as np
import pytest

import wtune.post.cli as cli
import wtune.post.compare as compare
import wtune.post.extract as extract
import wtune.post.recipe as recipe
import wtune.post.report as report
from wtune.post import atcf, changemap, met_pointstat, met_tc, sources, tc, timeseries, tools
from wtune.post.fetch import igra

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "post")
DIFFWRF_IDENTICAL = open(os.path.join(FIXTURES, "diffwrf_identical.txt")).read()
T0, T6 = "2005-08-27_00:00:00", "2005-08-27_06:00:00"


# ------------------------------------------------------------------ 1. compare fails closed

def _runs(tmp_path, a_times, b_times):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    for d, times in ((a, a_times), (b, b_times)):
        for t in times:
            (d / ("wrfout_d01_" + t)).write_text("")
    return str(a), str(b)


def _diffwrf(monkeypatch, rc=0, out=DIFFWRF_IDENTICAL, err=""):
    monkeypatch.setattr(tools, "find_diffwrf", lambda: "/fake/diffwrf")
    monkeypatch.setattr(tools, "run", lambda cmd, cwd=None, check=True: (rc, out, err))


@pytest.mark.parametrize("rc,out", [
    (1, ""),                                                    # diffwrf crashed
    (0, " could not open /b/wrfout\n Just plot   T\n"),         # unreadable B: no diff table
    (0, DIFFWRF_IDENTICAL + " Big difference: dim 1 lengths differ for T differ in b\n"),
    (0, "Diffing a b\n They differ big time.  Dates do not match\n"),
])
def test_compare_diffwrf_failure_is_an_error_not_identical(tmp_path, monkeypatch, rc, out):
    _diffwrf(monkeypatch, rc=rc, out=out, err="boom")
    a, b = _runs(tmp_path, [T0], [T0])
    with pytest.raises(compare.CompareError):
        compare.compare_runs(a, b)
    assert cli.main(["compare", a, b]) == 2


def test_parse_diffwrf_keeps_a_field_that_became_nan():
    row = ("        T2     12345    2   0.2950000000E+03                NaN  ***"
           "          NaN     Infinity\n")
    rows = compare.parse_diffwrf(DIFFWRF_IDENTICAL + row)
    assert [r["field"] for r in rows] == ["T2"] and rows[0]["ndifs"] == 12345
    assert np.isnan(rows[0]["rms2"]) and np.isinf(rows[0]["pntwise_max"])


def test_compare_frames_on_one_side_only_are_not_identical(tmp_path, monkeypatch, capsys):
    _diffwrf(monkeypatch)
    a, b = _runs(tmp_path, [T0, T6], [T0])
    result = compare.compare_runs(a, b)
    assert result["identical"] is False and result["n_frames_compared"] == 1
    assert result["only_in_a"] == [[1, T6]] and result["only_in_b"] == []
    assert cli.main(["compare", a, b]) == 1
    assert "only in run A" in capsys.readouterr().out


def test_compare_zero_common_frames_exits_nonzero(tmp_path, monkeypatch):
    _diffwrf(monkeypatch)
    a, b = _runs(tmp_path, [T0], [T6])
    assert cli.main(["compare", a, b]) == 2


def test_compare_identical_still_exits_zero(tmp_path, monkeypatch):
    _diffwrf(monkeypatch)
    a, b = _runs(tmp_path, [T0, T6], [T0, T6])
    assert cli.main(["compare", a, b]) == 0


# ------------------------------------------------------------------ 2. report drops nothing

def _report_text(tmp_path, types_, figures):
    case_info = dict(types=types_, evidence={}, run_length_h=1.0, d01_extent_km=(1.0, 1.0))
    path = report.build_recipe_report(str(tmp_path), case_info, figures, ["r"], ["r"], 6, {})
    return open(path).read()


def test_report_shows_whole_type_missing_entries_in_their_section(tmp_path):
    figs = [recipe.missing("%s_figures" % t, "wrfout", "why-%s" % t)
            for t in ("convection", "synoptic", "ensemble")]
    figs.append(recipe.missing("timeseries", "wrfout_d01_*", "no frames matched"))
    text = _report_text(tmp_path, ["common", "convection", "synoptic", "ensemble"], figs)
    for t in ("convection", "synoptic", "ensemble"):
        assert text.index("<h2>%s</h2>" % t) < text.index("why-%s" % t)
    assert text.index("<h2>Time series</h2>") < text.index("no frames matched")
    assert "<h2>other</h2>" not in text


def test_report_lists_unclaimed_entries_in_a_final_section(tmp_path):
    figs = [recipe.missing("common_x", "d", "claimed"),
            recipe.missing("pbl_figures", "d", "type-not-detected"),
            recipe.missing("mystery", "d", "nobody-claims-this")]
    text = _report_text(tmp_path, ["common"], figs)
    other = text.index("<h2>other</h2>")
    assert text.index("claimed") < other < text.index("type-not-detected")
    assert other < text.index("nobody-claims-this")
    assert text.count("nobody-claims-this") == 1 and text.count(">missing: common_x<") == 1


# ------------------------------------------------------------------ 3. truncated downloads

class _Resp(io.BytesIO):
    status = 200

    def __init__(self, body, length):
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_download_rejects_a_body_shorter_than_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    monkeypatch.setattr(sources.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(b"x" * 10, 100))
    dest = str(tmp_path / "f.bin")
    with pytest.raises(OSError, match="10 of 100"):
        sources.download("http://x/f.bin", dest, retries=1)
    assert not os.path.exists(dest)


@pytest.mark.parametrize("length", [10, None])
def test_download_accepts_a_complete_body(tmp_path, monkeypatch, length):
    monkeypatch.setattr(sources.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(b"x" * 10, length))
    dest = str(tmp_path / "f.bin")
    sha, downloaded = sources.download("http://x/f.bin", dest, retries=1)
    assert downloaded and sha == sources.sha256_file(dest) and os.path.getsize(dest) == 10


def test_fetch_bdeck_truncated_gzip_is_an_atcf_error(tmp_path, monkeypatch):
    blob = gzip.compress(b"AL, 12, 2005082700, , BEST, 0\n" * 200)[:-20]
    monkeypatch.setattr(atcf.urllib.request, "urlopen",
                        lambda url, timeout=None: _Resp(blob, None))
    with pytest.raises(atcf.ATCFError):
        atcf.fetch_bdeck(2005, "al", "12", str(tmp_path / "b.dat"))
    assert not (tmp_path / "b.dat").exists()


def _zip_member(payload, cut=0):
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    data = comp.compress(payload) + comp.flush()
    hdr = bytearray(30)
    hdr[:4], hdr[8:10] = b"PK\x03\x04", (8).to_bytes(2, "little")
    return bytes(hdr) + (data[:-cut] if cut else data)


def test_igra_stream_cut_before_member_end_raises(monkeypatch):
    payload = b"".join(b"line %d\n" % i for i in range(5000))
    monkeypatch.setattr(igra.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(_zip_member(payload), None))
    assert len(list(igra.stream_zip_lines("http://x/s.zip"))) == 5000
    monkeypatch.setattr(igra.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(_zip_member(payload, cut=50), None))
    with pytest.raises(RuntimeError, match="before the zip member was complete"):
        list(igra.stream_zip_lines("http://x/s.zip"))


# ------------------------------------------------------------------ 4. one type failing

def _recipe_run(tmp_path):
    return recipe._RecipeRun(["r"], ["r"], str(tmp_path), dict(types=[], domains=[]))


def test_type_failure_becomes_a_missing_entry_and_other_types_survive(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise tc.TCError("no track found")
    monkeypatch.setitem(recipe.TYPE_DISPATCH, "precip", boom)
    monkeypatch.setitem(recipe.TYPE_DISPATCH, "pbl", lambda *a, **k: [{"id": "pbl_ok"}])
    out = recipe._type_figures(["common", "precip", "pbl"], _recipe_run(tmp_path))
    assert [f["id"] for f in out] == ["precip_figures", "pbl_ok"]
    assert out[0]["missing"] and out[0]["reason"] == "TCError: no track found"


def test_type_failure_guard_lets_keyboard_interrupt_through(tmp_path, monkeypatch):
    def interrupt(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setitem(recipe.TYPE_DISPATCH, "precip", interrupt)
    with pytest.raises(KeyboardInterrupt):
        recipe._type_figures(["precip"], _recipe_run(tmp_path))


# ------------------------------------------------------------------ 5. one field failing

def test_time_series_field_without_its_variable_only_drops_that_field(tmp_path, monkeypatch):
    import wtune.post.plotting as plotting

    def scale(run_dir, field, domain, times):
        if field == "pw":
            raise extract.MissingFields(["QVAPOR"], "pw")
        return [0.0, 1.0], "viridis", "neither"

    def plot_field(run_dir, field, out_dir, **kw):
        path = os.path.join(out_dir, "%s_raw.png" % field)
        open(path, "w").close()
        return [path]

    monkeypatch.setattr(timeseries, "list_frames", lambda run_dir, domain=1: [(T0, "p"), (T6, "p")])
    monkeypatch.setattr(timeseries, "fixed_scale", scale)
    monkeypatch.setattr(plotting, "plot_field", plot_field)
    out = timeseries.figures_for_time_series("r", "r", 1, str(tmp_path))
    gone = [f for f in out if f["missing"]]
    assert [f["id"] for f in gone] == ["timeseries_pw"] and "QVAPOR" in gone[0]["reason"]
    drawn = [f["id"] for f in out if not f["missing"]]
    assert sorted(set(drawn)) == ["timeseries_rain_acc", "timeseries_slp", "timeseries_t2",
                                  "timeseries_wspd10"] and len(drawn) == 8


def test_change_map_field_without_its_variable_only_drops_that_field(tmp_path, monkeypatch):
    def diff(run_dir, field, domain, out_dir, a, b):
        if field == "pw":
            raise ValueError("'QVAPOR' is not a valid variable name")
        return np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2))

    monkeypatch.setattr(changemap, "_compute_diff", diff)
    monkeypatch.setattr(changemap, "change_map", lambda run_dir, label, field, *a, **k:
                        dict(id="chg_" + field, missing=False))
    out = changemap.figures_for_change_maps("r", "r", 1, str(tmp_path), [T0, T6],
                                            fields=("slp", "pw"))
    assert [(f["id"], f["missing"]) for f in out] == [
        ("chg_slp", False), ("chg_pw", True), ("chg_rain_acc", False)]
    assert "QVAPOR" in out[1]["reason"]


# ------------------------------------------------------------------ 6. bucketed rain

class _DS:
    def __init__(self, variables, **attrs):
        self.variables = variables
        self.__dict__.update(attrs)


def _rain_vars(with_tips=True):
    v = dict(RAINC=np.array([[[1.0, 2.0]], [[3.0, 4.0]]], "f4"),
             RAINNC=np.array([[[10.0, 20.0]], [[30.0, 40.0]]], "f4"))
    if with_tips:
        v.update(I_RAINC=np.array([[[0, 0]], [[1, 0]]], "i4"),
                 I_RAINNC=np.array([[[0, 0]], [[2, 5]]], "i4"))
    return v


def test_accumulated_rain_adds_bucket_tips():
    ds = _DS(_rain_vars(), BUCKET_MM=np.float32(100.0))
    np.testing.assert_allclose(extract.accumulated_rain(ds, 1), [[333.0, 544.0]])
    np.testing.assert_allclose(extract.accumulated_rain(ds, 0), [[11.0, 22.0]])


@pytest.mark.parametrize("ds", [
    _DS(_rain_vars(), BUCKET_MM=np.float32(-1.0)),      # bucket_mm off (WRF default)
    _DS(_rain_vars()),                                  # no attribute at all
    _DS(_rain_vars(with_tips=False), BUCKET_MM=100.0),  # iofields removed I_RAIN*
])
def test_accumulated_rain_without_buckets_is_rainc_plus_rainnc(ds):
    got = extract.accumulated_rain(ds, 1)
    np.testing.assert_array_equal(got, np.array([[33.0, 44.0]], "f4"))
    assert got.dtype == np.float32


def test_rain_acc_surface_field_includes_bucket_tips():
    import xarray as xr
    ds = _DS(_rain_vars(), BUCKET_MM=100.0)
    fake_wrf = types.SimpleNamespace(getvar=lambda ds, name, timeidx=0: xr.DataArray(
        ds.variables[name][timeidx], dims=("y", "x"), attrs={"projection": "p"}))
    total = extract._surface_field(ds, fake_wrf, "rain_acc", 1)
    np.testing.assert_allclose(total.values, [[333.0, 544.0]])
    assert total.attrs["projection"] == "p"


# ------------------------------------------------------------------ 7. tc_rmw winds

def test_tc_rmw_fields_use_earth_rotated_winds(tmp_path, monkeypatch):
    import netCDF4
    asked = []
    shape = (2, 3)

    def getvar(src, name, timeidx=0, units=None):
        asked.append(name)
        if name == "uvmet":
            return np.stack([np.full((4,) + shape, 7.0), np.full((4,) + shape, -3.0)])
        if name == "wspd_wdir10":
            return np.zeros((2,) + shape)
        return np.full((4,) + shape if name in ("ua", "va", "pressure") else shape, 99.0)

    fake = types.SimpleNamespace(
        getvar=getvar, interplevel=lambda f, p, levels: np.asarray(f)[:len(levels)],
        extract_times=lambda src, ti: np.datetime64("2005-08-27T00:00:00"))
    monkeypatch.setitem(sys.modules, "wrf", fake)
    src, out = str(tmp_path / "wrfout"), str(tmp_path / "f.nc")
    netCDF4.Dataset(src, "w").close()
    met_tc._wrf_fields_on_pressure_levels(src, out, levels=(850, 500))
    assert "uvmet" in asked and "ua" not in asked and "va" not in asked
    with netCDF4.Dataset(out) as ds:
        assert float(ds["u"][:].max()) == 7.0 and float(ds["v"][:].min()) == -3.0


# ------------------------------------------------------------------ 8. a-deck init time

def test_adeck_leads_count_from_the_run_start_not_first_detection(tmp_path):
    chosen = [dict(time=datetime(2005, 8, 27, 12), lat=24.6, lon=-83.3, pmin_hpa=959, vmax_ms=40),
              dict(time=datetime(2005, 8, 27, 18), lat=24.4, lon=-84.0, pmin_hpa=950, vmax_ms=45)]
    path = tc.build_adeck(chosen, [], str(tmp_path / "a.dat"), "AL", "12", tech="ctrl",
                          init_time=datetime(2005, 8, 27, 0))
    rows = atcf.parse_file(path)
    assert {r["time"] for r in rows} == {datetime(2005, 8, 27, 0)}
    assert sorted({r["tau_h"] for r in rows}) == [12, 18]


# ------------------------------------------------------------------ 9. point_stat tags

def test_point_stat_output_dirs_are_distinct_for_sub_hourly_frames(tmp_path, monkeypatch):
    import wtune.post.caseinfo as caseinfo
    dirs = []

    def to_latlon(lat, lon, fields, valid, out_nc, method=None):
        open(out_nc, "w").close()
        return out_nc

    monkeypatch.setattr(met_pointstat, "read_stations", lambda p: [dict(id="s")])
    monkeypatch.setattr(met_pointstat, "build_obs_text", lambda p, a, b: ("txt", ["s"]))
    monkeypatch.setattr(met_pointstat, "run_ascii2nc", lambda text, work: "obs.nc")
    monkeypatch.setattr(met_pointstat, "model_fields", lambda path, timeidx=0: (0, 0, {}, 0))
    monkeypatch.setattr(met_pointstat, "run_point_stat",
                        lambda fnc, obs, poly, out_dir, tag: dirs.append((out_dir, tag)))
    monkeypatch.setattr(met_pointstat, "summarize", lambda *a: "done")
    monkeypatch.setattr(met_pointstat.met_gridstat, "write_interior_polygon", lambda *a: "poly")
    monkeypatch.setattr(met_pointstat.met_gridstat, "curvilinear_to_latlon", to_latlon)
    monkeypatch.setattr(caseinfo, "spec_bdy_width", lambda run_dir: 5)
    frames = [("2005-08-27_00:00:00", "f", 0), ("2005-08-27_00:30:00", "f", 1)]
    assert met_pointstat.verify_run("r", 1, frames, "proj", str(tmp_path), 0) == "done"
    assert len(dirs) == 2 and len({d for d, _t in dirs}) == 2 and len({t for _d, t in dirs}) == 2


# ------------------------------------------------------------------ 10. --srun-minutes

@pytest.mark.parametrize("extra,minutes", [
    ([], "120"), (["--srun-minutes", "480"], "480"), (["--srun-minutes=30"], "30")])
def test_srun_minutes_sets_the_job_step_time_limit(monkeypatch, extra, minutes):
    from wtune.tools import scc_sweep
    seen = []
    monkeypatch.setattr(scc_sweep, "excluded_nodes", lambda: ["head"])

    def execvp(prog, cmd):
        seen.append(cmd)
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execvp", execvp)
    with pytest.raises(SystemExit):
        cli.main(["recipe", "runs/r1", "--srun"] + extra)
    cmd = seen[0]
    assert cmd[cmd.index("-t") + 1] == minutes
    assert cmd[cmd.index("post") + 1:] == ["recipe", "runs/r1"]


def test_srun_minutes_rejects_a_non_number():
    with pytest.raises(SystemExit, match="--srun-minutes"):
        cli.main(["recipe", "r", "--srun", "--srun-minutes", "soon"])


# ---------------------------------------------------------------------------
# Stage IV: one file failing on the network must not abort the fetch or pass unnoticed
# ---------------------------------------------------------------------------
def test_stage4_partial_download_keeps_the_rest_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    import argparse
    from wtune.post import cli, sources
    from wtune.post.fetch import stage4

    def fake_download(url, dest, known_sha=None, **_kw):
        if url == stage4.SITES_URL or "06h" in url:
            raise OSError("handshake timed out")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            fh.write("x")
        return "sha", True
    monkeypatch.setattr(sources, "download", fake_download)
    args = argparse.Namespace(project=str(tmp_path), run_dir=None, start="2019012800",
                              end="2019012812", bbox="35,47,-93,-77", accum="1,6", jobs=1)
    with pytest.raises(sources.FetchIncomplete, match="could not be downloaded"):
        stage4.run(args)
    prov = sources.read_provenance(str(tmp_path), "stage4")
    assert prov["files"] and all("01h" in f for f in prov["files"])     # the 1 h files are kept
    assert any("06h" in f for f in prov["failed"]) and stage4.SITES_FILE in prov["failed"]

    args.fetch_name, args.list = "stage4", False
    assert cli.cmd_fetch(args) == 1                                     # not a traceback, not 0
    assert "could not be downloaded" in capsys.readouterr().err


def test_post_user_errors_print_one_line_not_a_traceback(tmp_path, capsys):
    from wtune.post import cli
    assert cli.main(["plot", "--field", "t2", "--out", str(tmp_path / "o"), str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("error: ")
    from wtune.post import timeseries
    with pytest.raises(ValueError, match="use YYYYMMDDHH"):
        timeseries._nearest_time(["2019-01-30_00:00:00"], "last")
