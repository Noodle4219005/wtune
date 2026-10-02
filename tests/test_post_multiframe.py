"""Multi-frame wrfout (namelist frames_per_outfile > 1) and iofields-reduced history streams.

The frame index is built from each file's own Times variable, not from the file name: a file
holding 4 history times is 4 frames, each read at its own timeidx. The segmented fixture mirrors
the climate-ea50 restart layout: a restarted segment does not write its start time, so seg_1's
first file begins 6 h after the restart, and seg_0 ran one history step past the restart, so the
boundary valid time sits in BOTH segments (the later segment must win).

    seg_0/wrfout_d01_2023-01-01_00:00:00  00 06 12 18
    seg_0/wrfout_d01_2023-01-02_00:00:00  24 30 36 42      (hours since 2023-01-01_00)
    seg_0/wrfout_d01_2023-01-03_00:00:00  48 54            54 also in seg_1 -> seg_1 wins
    seg_1/wrfout_d01_2023-01-03_06:00:00  54 60 66 72      last valid time 72 h, timeidx 3
"""
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from post import caseinfo, extract, recipe  # noqa: E402
from post import vapor_bundle as vb  # noqa: E402

FMT = "%Y-%m-%d_%H:%M:%S"
T0 = datetime(2023, 1, 1)


def T(hours, t0=T0):
    return (t0 + timedelta(hours=hours)).strftime(FMT)


def _times_file(path, times, ny=4, nx=5):
    """Minimal wrfout: Times (+ XLAT/XLONG whose value encodes the time index)."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with netCDF4.Dataset(str(path), "w") as d:
        d.createDimension("Time", None)
        d.createDimension("DateStrLen", 19)
        d.createDimension("south_north", ny)
        d.createDimension("west_east", nx)
        d.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = \
            np.array([list(t) for t in times], dtype="S1")
        lat = d.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
        lon = d.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
        for k in range(len(times)):
            lat[k] = 20.0 + k
            lon[k] = 120.0
        d.setncattr("DX", 12000.0)
    return str(path)


def _segmented(tmp_path):
    out = tmp_path / "o2"
    f = {}
    f["a"] = _times_file(out / "seg_0" / ("wrfout_d01_%s" % T(0)), [T(h) for h in (0, 6, 12, 18)])
    f["b"] = _times_file(out / "seg_0" / ("wrfout_d01_%s" % T(24)), [T(h) for h in (24, 30, 36, 42)])
    f["c"] = _times_file(out / "seg_0" / ("wrfout_d01_%s" % T(48)), [T(48), T(54)])
    f["d"] = _times_file(out / "seg_1" / ("wrfout_d01_%s" % T(54)), [T(h) for h in (54, 60, 66, 72)])
    return str(out), f


# -- frame index ------------------------------------------------------------------------------

def test_frame_index_from_times_later_segment_wins(tmp_path):
    out, f = _segmented(tmp_path)
    refs = extract.list_frame_refs(out, domain=1)
    assert [t for t, _p, _i in refs] == [T(h) for h in range(0, 73, 6)]       # 13 valid times
    by = {t: (p, i) for t, p, i in refs}
    assert by[T(30)] == (f["b"], 1)
    assert by[T(48)] == (f["c"], 0)
    assert by[T(54)] == (f["d"], 0)              # duplicated boundary time: seg_1 wins
    assert by[T(72)] == (f["d"], 3)              # the last time sits at timeidx 3
    assert extract.frame_ref(out, 1, T(66)) == (f["d"], 2)
    assert extract.frame_path(out, 1, T(66)) == f["d"]
    assert extract.frame_ref(out, 1, T(78)) == (os.path.join(out, "wrfout_d01_%s" % T(78)), 0)
    # backward compatible (time, path) pairs, Times-based
    assert extract.list_frames(out, domain=1)[-1] == (T(72), f["d"])
    assert extract.frame_index(out)[(1, T(42))] == f["b"]
    # one entry per FILE (diffwrf compares whole files)
    assert sorted(extract.file_index(out).values()) == sorted(f.values())
    assert extract.list_frame_refs(out, domain=2) == []


def test_times_cache_follows_a_rewritten_file(tmp_path):
    out, f = _segmented(tmp_path)
    assert extract.list_frames(out)[-1][0] == T(72)
    _times_file(f["d"], [T(54), T(60)])           # the segment was re-run shorter
    st = os.stat(f["d"])
    os.utime(f["d"], ns=(st.st_atime_ns, st.st_mtime_ns + 2 * 10 ** 9))
    assert extract.list_frames(out)[-1][0] == T(60)


def test_unreadable_files_fall_back_to_the_file_name_time(tmp_path):
    run = tmp_path / "flat"
    run.mkdir()
    (run / ("wrfout_d01_%s" % T(0))).write_text("")            # empty
    (run / ("wrfout_d01_%s" % T(6))).write_text("not netcdf")  # garbage
    assert extract.list_frame_refs(str(run)) == [
        (T(0), str(run / ("wrfout_d01_%s" % T(0))), 0),
        (T(6), str(run / ("wrfout_d01_%s" % T(6))), 0)]


def test_run_length_window_and_last_time_use_every_valid_time(tmp_path):
    out, f = _segmented(tmp_path)
    assert caseinfo.run_length_hours(out) == pytest.approx(72.0)   # file names alone: 54 h
    from post import cf_pressure
    assert cf_pressure.select_frames(out, 1) == [(T(72), f["d"], 3)]
    assert [s[0] for s in cf_pressure.select_frames(out, 1, every="24h")] == \
        [T(0), T(24), T(48), T(72)]


def test_pw_6h_change_finds_both_frames_inside_one_file(tmp_path, monkeypatch):
    from post.recipes import moisture
    out, f = _segmented(tmp_path)
    captured = {}

    def fake_change_map(run_dir, run_label, field, domain, out_dir, t_prev, t_curr, t0, **kw):
        captured.update(t_prev=t_prev, t_curr=t_curr, t0=t0)
        return {"id": kw.get("fig_id"), "missing": False}
    monkeypatch.setattr("post.changemap.change_map", fake_change_map)
    assert moisture._pw_change_figure(out, "r", 1, str(tmp_path)) is not None
    assert (captured["t_prev"], captured["t_curr"], captured["t0"]) == (T(66), T(72), T(0))
    assert extract.frame_ref(out, 1, T(66)) == (f["d"], 2)


# -- climate-ea50 layout: 4 times per file, 6-hourly, 14 restart segments ---------------------

def _climate_run(tmp_path):
    out = tmp_path / "climate"
    start, end = datetime(2022, 12, 1), datetime(2024, 1, 1)
    step = timedelta(hours=6)
    n_seg = 14
    total_days = (end - start).days
    bounds = [start + timedelta(days=round(k * total_days / n_seg)) for k in range(n_seg)] + [end]
    for k in range(n_seg):
        t = bounds[k] if k == 0 else bounds[k] + step   # a restart does not write its start time
        times = []
        while t <= bounds[k + 1]:
            times.append(t.strftime(FMT))
            t += step
        for i in range(0, len(times), 4):
            chunk = times[i:i + 4]
            _times_file(out / ("seg_%d" % k) / ("wrfout_d01_%s" % chunk[0]), chunk, ny=2, nx=2)
    return str(out)


def test_climate_layout_run_length_and_twelve_months(tmp_path):
    from post.recipes import climate
    run = _climate_run(tmp_path)
    files = extract.file_index(run)
    assert max(t for _d, t in files) == "2023-12-31_06:00:00"     # the real case's last file
    refs = extract.list_frame_refs(run, domain=1)
    assert refs[-1][0] == "2024-01-01_00:00:00" and refs[-1][2] == 3
    assert caseinfo.run_length_hours(run) == pytest.approx(396 * 24.0)
    times, step_s, months = climate.run_months(refs)
    assert step_s == 6 * 3600
    assert months == [(2023, m) for m in range(1, 13)]
    assert set(climate.complete_seasons(months)) == {"DJF", "MAM", "JJA", "SON"}
    # what the file-name index produced: 395.25 d, 11 months
    names = sorted(datetime.strptime(t, FMT) for _d, t in files)
    assert (names[-1] - names[0]).total_seconds() / 86400.0 == pytest.approx(395.25)
    assert len(climate.analysis_months(names[0], names[-1])) == 11


def test_climate_cdo_inputs_cut_only_overridden_times(tmp_path, monkeypatch):
    from post.recipes import climate
    out, f = _segmented(tmp_path)
    calls = []
    monkeypatch.setattr(climate.tools, "run", lambda cmd, **kw: calls.append(cmd))
    work = str(tmp_path / "work")
    inputs, temps = climate._cdo_inputs(extract.list_frame_refs(out), work, "cdo")
    sel = os.path.join(work, "sel_00002.nc")
    assert inputs == [f["a"], f["b"], sel, f["d"]]        # seg_0's 54 h lost to seg_1
    assert temps == [sel]
    assert calls == [["cdo", "-s", "-O", "seltimestep,1", f["c"], sel]]


# -- data products at the right timeidx -------------------------------------------------------

def _stack(srcs, dst, skip=(), skip_attrs=()):
    """One multi-frame wrfout from single-frame files (variables without Time copied once)."""
    with netCDF4.Dataset(srcs[0]) as s0, netCDF4.Dataset(dst, "w") as d:
        d.setncatts({k: s0.getncattr(k) for k in s0.ncattrs() if k not in skip_attrs})
        for name, dim in s0.dimensions.items():
            d.createDimension(name, None if name == "Time" else len(dim))
        for name, v in s0.variables.items():
            if name not in skip:
                d.createVariable(name, v.dtype, v.dimensions)
        for k, src in enumerate(srcs):
            with netCDF4.Dataset(src) as s:
                for name, v in s.variables.items():
                    if name in skip:
                        continue
                    if v.dimensions and v.dimensions[0] == "Time":
                        d.variables[name][k] = v[0]
                    elif k == 0:
                        d.variables[name][...] = v[...]
    return dst


def _cf_multiframe(tmp_path, hours=(0, 6, 12)):
    """A flat run dir with ONE cf_pressure-fixture wrfout holding len(hours) times whose theta
    differs per time (T = 5 K * k), plus the single-frame sources it was stacked from."""
    pytest.importorskip("wrf")
    pytest.importorskip("pyproj")
    pytest.importorskip("scipy")
    from test_post_cf_pressure import _write_wrfout
    t0 = datetime(2005, 8, 28)
    srcs = []
    for k, h in enumerate(hours):
        p = str(tmp_path / "src" / ("single_%d.nc" % k))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        _write_wrfout(p, time_str=T(h, t0))
        with netCDF4.Dataset(p, "a") as d:
            d.variables["T"][:] = 5.0 * k
        srcs.append(p)
    run = tmp_path / "run"
    run.mkdir()
    multi = _stack(srcs, str(run / ("wrfout_d01_%s" % T(hours[0], t0))))
    return str(run), multi, srcs, [T(h, t0) for h in hours]


def test_cf3d_converts_each_valid_time_at_its_timeidx(tmp_path):
    from post import cf_pressure as cp
    run, multi, srcs, times = _cf_multiframe(tmp_path)
    res = cp.export(run, str(tmp_path / "out"), [1], 5, every="6h")
    assert [r["valid_time"] for r in res] == times and not any(r.get("error") for r in res)
    temps = []
    for r, src in zip(res, srcs):
        ref = cp.convert_frame(src, str(tmp_path / ("ref_%s.nc" % r["valid_time"][11:13])), 1, 5)
        with netCDF4.Dataset(r["path"]) as a, netCDF4.Dataset(ref["path"]) as b:
            ta = np.ma.filled(a["air_temperature"][:], np.nan)
            np.testing.assert_allclose(ta, np.ma.filled(b["air_temperature"][:], np.nan))
            assert a.getncattr("wtune_wrfout") == os.path.abspath(multi)
            temps.append(float(np.nanmean(ta)))
    assert temps[0] < temps[1] < temps[2]            # three different times, not timeidx 0 thrice
    last = cp.export(run, str(tmp_path / "out_last"), [1], 5)
    assert [r["valid_time"] for r in last] == times[-1:]


def test_extract_and_precip_accum_read_the_frame_timeidx(tmp_path):
    pytest.importorskip("wrf")
    from test_post_extract import _write_wrfout, LAT0, LON0
    from post.recipes import precip
    run = tmp_path / "run"
    run.mkdir()
    times = [T(h) for h in (0, 6, 12, 18)]
    path = str(run / ("wrfout_d01_%s" % times[0]))
    _write_wrfout(path, times)
    with netCDF4.Dataset(path, "a") as d:
        for k in range(4):
            d.variables["RAINC"][k] = 2.0 * k
    out = tmp_path / "pts.csv"
    extract.extract(str(run), ["rain_acc", "rain_rate"], str(out),
                    points=[(LAT0 + 1.0, LON0 + 1.0, "P")])
    rows = [ln.split(",") for ln in out.read_text().splitlines()
            if ln and ln[0].isdigit()]
    assert [r[0] for r in rows] == times
    assert [float(r[1]) for r in rows] == [0.0, 2.0, 4.0, 6.0]
    assert rows[0][2] == ""
    assert [float(r[2]) for r in rows[1:]] == pytest.approx([2.0 / 6.0] * 3)
    _lat, _lon, mm = precip._model_accum(extract.frame_ref(str(run), 1, times[1]),
                                         extract.frame_ref(str(run), 1, times[3]))
    np.testing.assert_allclose(mm, 4.0)


# -- VAPOR bundle: one time per file ----------------------------------------------------------

def test_vapor_bundle_refuses_multiframe_without_allow_copy(tmp_path):
    out, f = _segmented(tmp_path)
    root = tmp_path / "bundle"
    with pytest.raises(vb.NotHardLinkable) as e:
        vb.build_bundle(out, str(root), 1)
    msg = str(e.value)
    assert "multi-frame" in msg and "--allow-copy" in msg and "frames_per_outfile" in msg
    assert not root.exists()                       # refused before anything was written
    from post.recipes import vapor_bundle as recipe_vb
    case_info = {"domains": [{"domain": 1}]}
    rec = recipe_vb.figures(out, "r", str(root), case_info)
    assert len(rec) == 1 and rec[0]["missing"] and "multi-frame" in rec[0]["reason"]


def test_vapor_bundle_extracts_single_time_files_with_allow_copy(tmp_path):
    out, f = _segmented(tmp_path)
    r = vb.build_bundle(out, str(tmp_path / "bundle"), 1, every="6h", times=[T(66), T(72)],
                        allow_copy=True)
    assert [fr["valid_time"] for fr in r["frames"]] == [T(66), T(72)]
    assert [fr["timeidx"] for fr in r["frames"]] == [2, 3]
    assert {fr["mode"] for fr in r["frames"]} == {"extract"}
    for fr, k in zip(r["frames"], (2, 3)):
        p = os.path.join(r["dir"], fr["name"])
        assert extract.file_times(p) == [fr["valid_time"]]
        assert os.stat(p).st_ino != os.stat(f["d"]).st_ino
        with netCDF4.Dataset(p) as d:
            assert d.variables["XLAT"].shape[0] == 1
            assert float(d.variables["XLAT"][0, 0, 0]) == 20.0 + k
            assert d.getncattr("DX") == 12000.0
    manifest = json.load(open(r["manifest"]))
    assert manifest["copied_bytes"] == manifest["total_bytes"] > 0


# -- iofields-reduced history stream ------------------------------------------------------------

def test_reduced_history_stream_is_a_missing_entry_naming_the_fields(tmp_path):
    pytest.importorskip("wrf")
    pytest.importorskip("pyproj")
    pytest.importorskip("scipy")
    from test_post_cf_pressure import _write_wrfout
    from post import cf_pressure as cp
    from post.recipes import cf3d
    full = str(tmp_path / "full.nc")
    _write_wrfout(full, time_str="2005-08-28_06:00:00")
    run = tmp_path / "run"
    run.mkdir()
    reduced = _stack([full], str(run / "wrfout_d01_2005-08-28_06:00:00"),
                     skip=("QCLOUD", "P_TOP"), skip_attrs=("P_TOP",))
    want = "history stream lacks P_TOP, QCLOUD (iofields-reduced output?)"
    with pytest.raises(extract.MissingFields) as e:
        cp.convert_frame(reduced, str(tmp_path / "x.nc"), 1, 5)
    assert str(e.value) == want
    res = cp.export(str(run), str(tmp_path / "out"), [1], 5)
    assert res[0]["error"] == want
    rec = cf3d.figures(str(run), "r", str(tmp_path / "out2"), {"domains": [{"domain": 1}]}, 5)
    assert rec[0]["missing"] and rec[0]["reason"] == want and "Error" not in rec[0]["reason"]


def test_recipe_guard_turns_missing_fields_into_a_missing_entry():
    def raiser(exc):
        def fn(*a, **k):
            raise exc
        return fn
    for exc, name in ((AttributeError("NetCDF: Attribute not found"), None),
                      (KeyError("QCLOUD"), "QCLOUD"),
                      (ValueError("'QICE' is not a valid variable name"), "QICE"),
                      (extract.MissingFields(["P_TOP", "W"]), "P_TOP, W")):
        rec = recipe.guarded("climate", raiser(exc))
        assert len(rec) == 1 and rec[0]["missing"]
        assert rec[0]["reason"].startswith("history stream lacks")
        assert "AttributeError" not in rec[0]["reason"]
        if name:
            assert name in rec[0]["reason"]
    with pytest.raises(RuntimeError):
        recipe.guarded("climate", raiser(RuntimeError("a real bug")))
    assert recipe.guarded("climate", lambda: ["ok"]) == ["ok"]
