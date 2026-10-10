#!/usr/bin/env python3
"""pytest suite for wtune/run/ (`wtune run`: segmented WRF executor with restart hand-offs).
No job is ever submitted: sbatch/sacct/snapshot are fakes, every directory is a tmp dir."""
from conftest import swap_wtune_module
import datetime
import json
import os
import re
import subprocess

import sys
import types

import pytest

# Importing wtune_run -> scc_sweep sets SITE=scc and loads campaign `configs` with it; undo both so
# campaign/test_configs.py (collected later in the same process) gets its own f1 import.
_SITE_BEFORE = os.environ.get("SITE")
_CONFIGS_BEFORE = sys.modules.get("configs")

from wtune import predict as wp
from wtune import run as wr
from wtune import watch as wtune_watch

if _SITE_BEFORE is None:
    os.environ.pop("SITE", None)
if _CONFIGS_BEFORE is None:
    sys.modules.pop("configs", None)


def _load_real_bp():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wtune_bp_real", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__))), "src", "wtune", "bp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REAL_BP = _load_real_bp()        # its per-record helpers (stdlib only, no adios2) are used as-is


def _mk_bp(path):
    """A directory wtune_bp.is_bp recognizes (BP4: md.idx + data.0)."""
    os.makedirs(path, exist_ok=True)
    for f in ("md.idx", "data.0"):
        open(os.path.join(path, f), "w").close()


class FakeBP:
    """Stand-in for src/wtune/bp.py (the converter module): convert_inputs makes cache_dir/<name>
    BP directories (its real output naming; wrfbdy/wrflowinp/wrffdda as per-record containers of
    <name>_<date>.bp, dates from `records`) and records the call; no adios2 anywhere. The
    per-record helpers are the real module's."""
    XML = '<?xml version="1.0"?>\n<!-- wtune -->\n<adios-config>\n</adios-config>\n'
    DEFAULT_RECORDS = ["2020-06-01_00:00:00", "2020-06-01_06:00:00", "2020-06-01_12:00:00"]

    def __init__(self):
        self.calls = []
        self.fail = None
        self.records = {}
        self.cp_calls = []
        self.no_reflink = False

    def module(self):
        m = types.ModuleType("wtune_bp")
        m.ADIOS2_IO_FORM = 14
        m.require_adios2_python = self.require_adios2_python
        m.convert_inputs = self.convert_inputs
        m.adios2_xml = lambda engine_by_stream=None: self.XML
        m.is_bp = os.path.isdir
        for name in ("PER_RECORD_NAME_RE", "is_per_record_dir", "record_files",
                     "bp_input_links", "per_record_namelist_edits"):
            setattr(m, name, getattr(REAL_BP, name))
        m.fresh_input_copy = lambda src, dest, stats=None: REAL_BP.fresh_input_copy(
            src, dest, runner=self.cp, stats=stats)
        return m

    def cp(self, cmd, **kw):
        """Fake `cp`: reflink succeeds unless `no_reflink`; copies with shutil."""
        import shutil
        self.cp_calls.append(list(cmd))
        if "--reflink=always" in cmd and self.no_reflink:
            return subprocess.CompletedProcess(cmd, 1, b"", b"reflink not supported")
        shutil.copytree(cmd[-2], cmd[-1], symlinks=True)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    def require_adios2_python(self):
        if self.fail:
            raise RuntimeError(self.fail)
        return "/fake/adios2/python"

    def convert_inputs(self, case_dir, cache_dir, names=None, engine="BP4"):
        self.calls.append((case_dir, cache_dir, list(names), engine))
        out = {}
        for n in names:
            p = os.path.join(cache_dir, n)
            if REAL_BP.PER_RECORD_NAME_RE.match(n):
                for t in self.records.get(n, self.DEFAULT_RECORDS):
                    _mk_bp(os.path.join(p, "%s_%s.bp" % (n, t)))
            else:
                _mk_bp(p)
            out[n] = p
        return out


@pytest.fixture(autouse=True)
def fake_bp(monkeypatch):
    fb = FakeBP()
    swap_wtune_module(monkeypatch, "bp", fb.module())
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    return fb

NAMELIST_3DOM = """ &time_control
 run_days                            = 0,
 run_hours                           = 12,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 2020, 2020, 2020,
 start_month                         = 06,   06,   06,
 start_day                           = 01,   01,   01,
 start_hour                          = 00,   00,   00,
 end_year                            = 2020, 2020, 2020,
 end_month                           = 06,   06,   06,
 end_day                             = 01,   01,   01,
 end_hour                            = 12,   12,   12,
 interval_seconds                    = 21600
 input_from_file                     = .true.,.true.,.true.,
 history_interval                    = 60,  60,   60,
 frames_per_outfile                  = 1, 1, 1,
 restart                             = .false.,
 restart_interval                    = 7200,
 io_form_history                     = 2
 io_form_restart                     = 2
 io_form_input                       = 2
 io_form_boundary                    = 2
 /

 &domains
 time_step                           = 60,
 max_dom                             = 3,
 e_we                                = 100, 100, 91,
 e_sn                                = 100, 91,  91,
 e_vert                              = 45,  45,  45,
 parent_id                           = 1,   1,   2,
 parent_grid_ratio                   = 1,   3,   3,
 parent_time_step_ratio              = 1,   3,   3,
 /

 &physics
 radt                                = 30, 10, 10,
 /
"""


def _dt(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d_%H:%M:%S")


def _nl_from_text(tmp_path, text, name="namelist.input"):
    p = tmp_path / name
    p.write_text(text)
    return wp.parse_namelist(str(p))


def _geo(tmp_path, text=NAMELIST_3DOM):
    return wr.case_geometry(_nl_from_text(tmp_path, text, "geo.nml"))


def _cfg(nx=8, ny=9, numtiles=1, omp=1, nodes=1, cid="cli"):
    return wr.make_config(nx, ny, numtiles, omp, nodes, cid=cid)


# ---------------------------------------------------------------------------
# geometry / planning
# ---------------------------------------------------------------------------
def test_geometry_of_3dom_case(tmp_path):
    g = _geo(tmp_path)
    assert (g["steps"], g["dt"], g["max_dom"], g["quantum"]) == (720, 60.0, 3, 1)
    assert g["points"] == [100 * 100 * 45, 100 * 91 * 45, 91 * 91 * 45]
    assert g["end"] == _dt("2020-06-01_12:00:00")


def test_plan_fixed_minutes_last_shorter(tmp_path):
    g = _geo(tmp_path)
    n = wr.segment_steps_for_minutes(g, 300)
    assert wr.plan_segments_fixed(720, n) == [(0, 300), (300, 600), (600, 720)]


def test_segment_minutes_must_be_whole_steps(tmp_path):
    g = _geo(tmp_path, NAMELIST_3DOM.replace("time_step                           = 60,",
                                             "time_step                           = 36,"))
    assert g["quantum"] == 5                       # lcm(36, 60) = 180 s = 5 steps
    assert wr.segment_steps_for_minutes(g, 3) == 5
    with pytest.raises(wr.RunError):
        wr.segment_steps_for_minutes(g, 1)         # 60 s is not a multiple of 36 s


def test_stop_at_whole_step_and_validation(tmp_path):
    g = _geo(tmp_path)
    assert wr.resolve_end_step(g, None) == 720
    assert wr.resolve_end_step(g, "2020-06-01_06:00:00") == 360
    assert wr.resolve_end_step(g, "2020-06-02_00:00:00") == 720       # later than the end: capped
    with pytest.raises(wr.RunError):
        wr.resolve_end_step(g, "2020-06-01_06:00:30")                  # not on a 60 s step
    with pytest.raises(wr.RunError):
        wr.resolve_end_step(g, "2020-06-01_00:00:00")


def _synthetic_model(g, s=1.0):
    pts = [(g["start"], s), (g["end"], s)]
    comp = wr.compute_fn_from_points(g, pts, 1.0, 1.0, None)          # r=1: no radiation extra
    over = wr.overheads(init_s=100.0, restart_read_s=50.0, restart_write_s=50.0,
                        history_frame_s=[10.0, 10.0, 10.0], source="test")
    return comp, over


def test_compute_over_interval_is_difference_of_whole_run_totals(tmp_path):
    g = _geo(tmp_path)
    pts = [(g["start"], 1.0), (g["end"], 3.0)]
    comp = wr.compute_fn_from_points(g, pts, 2.0, 4.0, None)
    import coldstart_sample as cs
    anchors = [(0.0, 1.0), ((g["end"] - g["start"]).total_seconds(), 3.0)]
    whole = cs.predicted_compute_total(720, 60.0, g["stepra"], g["start"], anchors, None, 2.0, 4.0)[0]
    assert comp(0, 300) + comp(300, 720) == pytest.approx(whole)
    assert comp(0, 300) < comp(300, 600)            # rising s_normal


def _assert_l_grid(segs, L, n_end):
    assert segs[0]["start_step"] == 0 and segs[-1]["end_step"] == n_end
    for x, y in zip(segs, segs[1:]):
        assert x["end_step"] == y["start_step"]
    for s in segs:
        assert s["start_step"] % L == 0
        assert s["end_step"] % L == 0 or s["end_step"] == n_end
    assert [s["index"] for s in segs] == list(range(len(segs)))


def test_plan_fixed_minutes_is_the_restart_quantum(tmp_path):
    g = _geo(tmp_path)
    comp, over = _synthetic_model(g)
    L, segs = wr.plan_run(g, 700, [{"config": _cfg(), "compute": comp}], over,
                          segment_minutes=300)
    assert L == 300 and wr.restart_interval_minutes(g, L) == 300
    _assert_l_grid(segs, L, 700)
    assert [(s["start_step"], s["end_step"]) for s in segs] == [(0, 300), (300, 600), (600, 700)]
    assert [s["writes_restart"] for s in segs] == [True, True, False]
    L1, one = wr.plan_run(g, 720, [{"config": _cfg(), "compute": comp}], over)
    assert L1 is None and len(one) == 1
    assert wr.restart_interval_minutes(g, None) == wr.RESTART_OFF_MIN


def test_plan_wall_limit_with_synthetic_prediction(tmp_path):
    g = _geo(tmp_path)
    comp, over = _synthetic_model(g)
    cand = [{"config": _cfg(), "compute": comp}]
    limit = 400.0                                  # budget 320 s
    L, segs = wr.plan_run(g, 720, cand, over, wall_limit_s=limit)
    _assert_l_grid(segs, L, 720)
    for s in segs:
        assert s["predicted_wall_s"] <= 0.8 * limit + 1e-9
        assert s["time_limit"] <= "00:06:40"       # capped at --wall-limit
    # one more quantum would push the most expensive L-grid segment over the budget
    walls = [wr.segment_wall(g, comp, over, a, b, L + 1)
             for a, b in wr.plan_segments_fixed(720, L + 1)]
    assert max(walls) > 0.8 * limit
    # uniform: every segment but the last is exactly L long
    assert all(s["end_step"] - s["start_step"] == L for s in segs[:-1])


def test_plan_wall_limit_boundaries_whole_minutes_and_steps(tmp_path):
    txt = NAMELIST_3DOM.replace("time_step                           = 60,",
                                "time_step                           = 36,")
    g = _geo(tmp_path, txt)
    cand = [{"config": _cfg(), "compute": wr.constant_compute_fn(0.5)}]
    over = wr.overheads(init_s=30.0, restart_read_s=10.0, restart_write_s=10.0,
                        history_frame_s=[0, 0, 0])
    L, segs = wr.plan_run(g, g["steps"], cand, over, wall_limit_s=300.0)
    assert len(segs) > 2 and L % 5 == 0             # quantum lcm(36, 60) = 180 s = 5 steps
    assert (L * 36) % 60 == 0 and wr.restart_interval_minutes(g, L) == L * 36 // 60
    _assert_l_grid(segs, L, g["steps"])
    for s in segs[:-1]:
        assert _dt(s["end"]).second == 0


def test_fallback_io_scales_from_conus25_and_dorian_anchors(tmp_path):
    conus = NAMELIST_3DOM.replace("max_dom                             = 3,",
                                  "max_dom                             = 1,").replace(
        "e_we                                = 100, 100, 91,", "e_we = 1500,").replace(
        "e_sn                                = 100, 91,  91,", "e_sn = 1200,").replace(
        "e_vert                              = 45,  45,  45,", "e_vert = 50,")
    g = _geo(tmp_path, conus)
    assert wr.fallback_restart_io_s(g, 2) == pytest.approx((324.0, 120.0))
    assert wr.fallback_restart_io_s(g, 14) == pytest.approx((1.05, 2.3))
    over = wr.fallback_overheads(g, 14)
    assert (over["restart_write_s"], over["restart_read_s"]) == pytest.approx((1.05, 2.3))
    half = dict(g, points=[g["points"][0] / 2])
    assert wr.fallback_restart_io_s(half, 2) == pytest.approx((162.0, 60.0))
    dorian = dict(g, points=[664 * 700 * 40, 451 * 451 * 40], max_dom=2)
    d01, d02 = wr.fallback_history_frame_s(dorian, "netcdf")
    assert 7 * d01 + 25 * d02 == pytest.approx(300.5)          # the anchor reproduces itself
    assert d01 == pytest.approx(16.75, abs=0.05)               # f1 atlas measured ~15.7 s


def test_fallback_history_follows_the_run_io_format(tmp_path):
    g = _geo(tmp_path)
    nc = wr.fallback_overheads(g, 14, "netcdf")["history_frame_s"]
    ad = wr.fallback_overheads(g, 14, "adios2")["history_frame_s"]
    assert nc == wr.fallback_history_frame_s(g, "netcdf")
    assert ad == wr.fallback_history_frame_s(g)                # adios2 is the default, as --io
    assert all(x < n / 10 for x, n in zip(ad, nc))             # ADIOS2 anchor ~21x cheaper
    assert ad[0] == pytest.approx(g["points"][0] * wr.iom.ADIOS2_HISTORY_S_PER_POINT_FRAME)


def test_prediction_io_block_wins_over_fallback(tmp_path):
    g = _geo(tmp_path)
    fb = wr.fallback_overheads(g, 2)
    pred = {"result": {"points": [{"at": g["start"].isoformat(), "s_normal": 1.0}],
                       "r_night": 1.0, "r_day": 1.0, "init_s": 50.0}}
    _c, over = wr.model_from_prediction(pred, g, None, fb)
    assert over["restart_write_s"] == fb["restart_write_s"]
    assert over["history_frame_s"] == fb["history_frame_s"]
    pred["io"] = {"restart": {"per_write_s": 7.0}, "history": [{"median_s": 1.5}]}
    _c, over = wr.model_from_prediction(pred, g, None, fb)
    assert over["restart_write_s"] == over["restart_read_s"] == 7.0
    assert over["history_frame_s"][0] == 1.5 and over["history_frame_s"][1:] == \
        fb["history_frame_s"][1:]


def test_plan_wall_limit_too_small_raises(tmp_path):
    g = _geo(tmp_path)
    comp, over = _synthetic_model(g)
    with pytest.raises(wr.RunError):
        wr.plan_run(g, 720, [{"config": _cfg(), "compute": comp}], over, wall_limit_s=100.0)


def test_tune_report_picks_lowest_predicted_compute_per_segment(tmp_path):
    g = _geo(tmp_path)
    t0, t1 = g["start"].isoformat(), g["end"].isoformat()
    report = {"nodes": 1, "ranking": [
        {"id": "A", "nproc_x": 8, "nproc_y": 9, "numtiles": 1, "omp": 1, "dist": "block",
         "points": [{"at": t0, "s_normal": 1.0, "r": None, "n_rad": 0},
                    {"at": t1, "s_normal": 3.0, "r": None, "n_rad": 0}]},
        {"id": "B", "nproc_x": 6, "nproc_y": 12, "numtiles": 2, "omp": 1, "dist": "block",
         "points": [{"at": t0, "s_normal": 2.0, "r": None, "n_rad": 0}]},
        {"id": "C", "nproc_x": 4, "nproc_y": 4, "numtiles": 1, "omp": 1, "points": []}]}
    cands = wr.tune_candidates(report, g, None, {}, 1)
    assert [c["config"]["id"] for c in cands] == ["A", "B"]
    _L, segs = wr.plan_run(g, 720, cands, wr.overheads(), segment_minutes=360)
    assert [s["config"]["id"] for s in segs] == ["A", "B"]
    with pytest.raises(wr.RunError):
        wr.tune_candidates(report, g, None, {}, 2)          # measured at 1 node


def test_rescue_time_step():
    assert wr.rescue_time_step(60, 21600, (3600, 1800)) == 45
    assert wr.rescue_time_step(100, 6000, (3600,)) == 75
    assert wr.rescue_time_step(60, 3600, (3600, 7)) == 45     # 7 s cannot be kept: relaxed
    assert wr.rescue_time_step(1, 60, ()) is None
    assert wr.rescue_time_step(60, 3607, ()) is None          # 3607 = 3607 x 1: no divisor in [30, 45]


# ---------------------------------------------------------------------------
# segment namelist
# ---------------------------------------------------------------------------
def _render(k, start="2020-06-01_06:00:00", end="2020-06-01_12:00:00", L_min=180,
            text=NAMELIST_3DOM, **kw):
    return wr.render_segment_namelist(text, _dt(start), _dt(end), k,
                                      _cfg(nx=12, ny=6, numtiles=4), L_min, **kw)


def test_namelist_every_domain_start_end_and_run_length(tmp_path):
    txt = _render(1, "2020-06-01_06:00:00", "2020-06-01_09:00:00")
    nl = _nl_from_text(tmp_path, txt)
    tc = nl["time_control"]
    for key, want in (("start_year", 2020), ("start_month", 6), ("start_day", 1),
                      ("start_hour", 6), ("start_minute", 0), ("start_second", 0),
                      ("end_year", 2020), ("end_month", 6), ("end_day", 1), ("end_hour", 9),
                      ("end_minute", 0), ("end_second", 0)):
        assert tc[key] == [want] * 3, key
    assert (tc["run_days"], tc["run_hours"], tc["run_minutes"], tc["run_seconds"]) == \
        ([0], [0], [0], [10800])
    info = wtune_watch.run_info(nl)
    assert info["start"] == _dt("2020-06-01_06:00:00") and info["end"] == _dt("2020-06-01_09:00:00")
    assert all(d["end"] == info["end"] and d["start"] == info["start"] for d in info["domains"])
    assert nl["domains"]["nproc_x"] == [12] and nl["domains"]["nproc_y"] == [6]
    assert nl["domains"]["numtiles"] == [4]
    assert tc["history_interval"] == [60, 60, 60]            # history untouched
    assert "write_hist_at_0h_rst" not in txt
    lines = [ln.strip() for ln in txt.splitlines()]
    assert "io_form_restart                     = 14" in lines      # layout kept, own line
    assert "io_form_input                       = 2" in lines


def test_namelist_restart_keys_same_quantum_no_override(tmp_path):
    segs = [(0, "2020-06-01_00:00:00", "2020-06-01_03:00:00"),
            (1, "2020-06-01_03:00:00", "2020-06-01_06:00:00"),
            (2, "2020-06-01_06:00:00", "2020-06-01_08:00:00")]        # last one shorter
    for k, a, b in segs:
        txt = _render(k, a, b, L_min=180, restart_io_form=2 if k == 1 else 14)
        assert "override_restart_timers" not in txt
        tc = _nl_from_text(tmp_path, txt, "k%d" % k)["time_control"]
        assert tc["restart_interval"] == [180]                         # L in EVERY segment
        assert tc["restart"] == [k > 0]
        assert tc["io_form_restart"] == [2 if k == 1 else 14]


def test_namelist_case_override_restart_timers_forced_false(tmp_path):
    base = NAMELIST_3DOM.replace(" restart                             = .false.,",
                                 " restart                             = .false.,\n"
                                 " override_restart_timers = .true.,")
    for k in (0, 1):
        tc = _nl_from_text(tmp_path, _render(k, text=base), "o%d" % k)["time_control"]
        assert tc["override_restart_timers"] == [False]


def test_namelist_restart_interval_variants_and_time_step_override(tmp_path):
    base = NAMELIST_3DOM.replace(" restart_interval                    = 7200,",
                                 " restart_interval                    = 7200,\n"
                                 " restart_interval_h = 5,\n restart_interval_m = 30,")
    txt = wr.render_segment_namelist(base, _dt("2020-06-01_00:00:00"),
                                     _dt("2020-06-01_06:00:00"), 0, _cfg(), 360,
                                     time_step=45)
    tc = _nl_from_text(tmp_path, txt)["time_control"]
    assert tc["restart_interval_h"] == [0] and tc["restart_interval_m"] == [360]
    assert tc["restart_interval"] == [360]
    assert _nl_from_text(tmp_path, txt, "b")["domains"]["time_step"] == [45]


def test_namelist_diff_summary_lists_changes():
    diff = wr.namelist_diff_summary(NAMELIST_3DOM, _render(1))
    assert any(ln.startswith("+") and "restart " in ln and ".true." in ln for ln in diff)
    assert any(ln.startswith("-") and "run_hours" in ln for ln in diff)


# ---------------------------------------------------------------------------
# run dir
# ---------------------------------------------------------------------------
def _make_case(tmp_path, text=NAMELIST_3DOM):
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(text)
    for name in ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01", "rsl.error.0000", "wrf.out",
                 "namelist.output", "wrfout_d01_2020-06-01_00:00:00",
                 "wrfrst_d01_2020-06-01_06:00:00", "adios2.xml", "myiofields.txt"):
        (case / name).write_text("x")
    (case / "wrfinput_d03.bp").mkdir()
    (case / "junkdir").mkdir()
    return case


def _tree_state(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = os.path.join(dp, n)
            st = os.lstat(p)
            out[p] = (st.st_mtime_ns, st.st_size)
    return out


def test_setup_segment_dir_links_and_never_writes_case(tmp_path):
    case = _make_case(tmp_path)
    wrf = tmp_path / "WRF"
    (wrf / "run").mkdir(parents=True)
    for name in ("LANDUSE.TBL", "wrf.exe", "namelist.input", "myiofields.txt"):
        (wrf / "run" / name).write_text("w")
    prev = tmp_path / "out" / "seg_0"
    prev.mkdir(parents=True)
    rst = [prev / ("wrfrst_d%02d_2020-06-01_06:00:00" % d) for d in (1, 2, 3)]
    for p in rst:
        p.mkdir()                                   # BP4 restarts are directories
    before = _tree_state(case)
    seg = tmp_path / "out" / "seg_1"
    xml = wr.render_adios2_restart_xml(["2020-06-01_06:00:00", "2020-06-01_09:00:00"], 3)
    found = wr.restart_files(str(prev), "2020-06-01_06:00:00", 3)
    assert all(len(v) == 1 for v in found.values())
    wr.setup_segment_dir(str(case), str(seg), "NML", [p for v in found.values() for p in v], xml,
                         str(wrf))
    assert _tree_state(case) == before
    names = set(os.listdir(seg))
    assert {"wrfinput_d01", "wrfinput_d02", "wrfbdy_d01", "wrfinput_d03.bp", "myiofields.txt",
            "LANDUSE.TBL", "wrf.exe", "namelist.input", "adios2.xml"} <= names
    assert not names & {"rsl.error.0000", "wrf.out", "namelist.output", "junkdir",
                        "wrfout_d01_2020-06-01_00:00:00"}
    assert os.readlink(seg / "myiofields.txt") == str(case / "myiofields.txt")   # case wins
    for p in rst:
        assert os.readlink(seg / p.name) == str(p)
    assert not os.path.islink(seg / "namelist.input")
    assert (seg / "namelist.input").read_text() == "NML"
    assert not os.path.islink(seg / "adios2.xml")
    assert (seg / "adios2.xml").read_text() == (
        '<?xml version="1.0"?>\n<adios-config>\n'
        '  <io name="wrfrst_d01_2020-06-01_06:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d02_2020-06-01_06:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d03_2020-06-01_06:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d01_2020-06-01_09:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d02_2020-06-01_09:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '  <io name="wrfrst_d03_2020-06-01_09:00:00">\n    <engine type="BP4"/>\n  </io>\n'
        '</adios-config>\n')


def test_adios2_xml_keeps_case_entries():
    base = '<?xml version="1.0"?>\n<adios-config>\n  <io name="wrfout">\n  </io>\n</adios-config>\n'
    out = wr.render_adios2_restart_xml(["2020-06-01_06:00:00"], 1, base)
    assert out == ('<?xml version="1.0"?>\n<adios-config>\n  <io name="wrfout">\n  </io>\n'
                   '  <io name="wrfrst_d01_2020-06-01_06:00:00">\n    <engine type="BP4"/>\n'
                   '  </io>\n</adios-config>\n')


def test_setup_segment_dir_io_form2_links_case_adios2_xml_and_refuses_inside_case(tmp_path):
    case = _make_case(tmp_path)
    seg = tmp_path / "o" / "seg_0"
    wr.setup_segment_dir(str(case), str(seg), "NML")
    assert os.path.islink(seg / "adios2.xml")
    with pytest.raises(wr.RunError):
        wr.setup_segment_dir(str(case), str(case / "seg_0"), "NML")


# ---------------------------------------------------------------------------
# sbatch
# ---------------------------------------------------------------------------
def test_segment_sbatch_text(tmp_path, monkeypatch):
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "work5,work7")
    cfg = wr.make_config(6, 12, 2, 2, 2)
    text = wr.render_segment_sbatch(str(tmp_path / "seg_0"), 0, cfg, "01:30:00", "wtrun-x-s0",
                                    campaign_dir="/repo/campaign")
    assert ("#SBATCH -p compute -N 2 --exclusive --ntasks-per-node=72 --time=01:30:00 "
            "--exclude=head,work5,work7") in text
    assert "mpirun -np 72 --map-by ppr:36:node:PE=2 --bind-to core ./wrf.exe &" in text
    assert "export OMP_NUM_THREADS=2" in text
    assert "export OMP_PLACES=cores OMP_PROC_BIND=close" in text
    assert "ulimit -s unlimited" in text
    assert "run_hygiene reap+census" in text and "site_hygiene" in text
    assert 'source "$HOME/wtune-scc/env.sh"' in text
    assert "CAMPAIGN=${CAMPAIGN:-/repo/campaign}" in text
    last = text.strip().splitlines()[-1]
    assert last.startswith('echo "SEGMENT rc=$rc wall=$wall ok=$ok"')
    assert "SUCCESS COMPLETE WRF" in text
    assert subprocess.run(["bash", "-n"], input=text, text=True).returncode == 0


def test_segment_sbatch_omp1_unsets_places_and_cyclic(tmp_path, monkeypatch):
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    cfg = wr.make_config(10, 13, 1, 1, 2, dist="cyclic")
    text = wr.render_segment_sbatch(str(tmp_path), 1, cfg, "00:20:00", "j")
    assert "--exclude=head\n" in text
    assert "unset OMP_PLACES OMP_PROC_BIND" in text and "export OMP_NUM_THREADS=1" in text
    assert "mpirun -np 130 --map-by ppr:65:node:PE=1 --bind-to core --rank-by node ./wrf.exe" in text


def test_time_limit_rounding_and_cap():
    assert wr.time_limit_s(1100.0) == 2400           # 1.5*1100+600 = 2250 -> 2400
    assert wr.time_limit_s(1000.0) == 2100           # already on the 5-min grid
    assert wr.time_limit_s(1100.0, 1800) == 1800


# ---------------------------------------------------------------------------
# controller (fakes)
# ---------------------------------------------------------------------------
class Fakes:
    """sbatch creates the restart files the segment would write; sacct says COMPLETED; the
    snapshot outcome per submission comes from `outcomes` (default done)."""

    def __init__(self, outcomes=None):
        self.submitted = []
        self.outcomes = list(outcomes or [])
        self.snap_by_dir = {}

    def sbatch(self, cmd):
        path = cmd.split()[-1].strip("'")
        seg = os.path.dirname(path)
        nl = wp.parse_namelist(os.path.join(seg, "namelist.input"))
        info = wtune_watch.run_info(nl)
        outcome = self.outcomes.pop(0) if self.outcomes else "done"
        self.submitted.append((seg, outcome))
        if outcome == "done":
            for d in range(1, len(info["domains"]) + 1):
                rst = os.path.join(seg, "wrfrst_d%02d_%s" % (d, info["end"].strftime(
                    "%Y-%m-%d_%H:%M:%S")))
                os.makedirs(rst, exist_ok=True)
                open(os.path.join(rst, "md.idx"), "w").close()
        self.snap_by_dir[seg] = outcome
        return "Submitted batch job %d\n" % (1000 + len(self.submitted))

    def sacct(self, cmd):
        jid = cmd[cmd.index("-j") + 1]
        return "%s|COMPLETED|00:10:00\n" % jid

    def snapshot(self, rundir):
        outcome = self.snap_by_dir[rundir]
        if outcome == "done":
            return {"status": "done", "warnings": [], "sum_d01_elapsed_s": 300.0,
                    "n_d01_steps": 360}
        if outcome == "cfl":
            return {"status": "failed", "warnings": [
                {"code": "FAILED", "text": "NaN in a d01 Timing line"},
                {"code": "CFL", "text": "3 cfl line(s) in rsl.error.0000"}]}
        return {"status": "failed", "warnings": [{"code": "FAILED", "text": "forrtl: severe"}]}


def _run(tmp_path, fakes, extra=()):
    case = tmp_path / "case" if (tmp_path / "case").exists() else _make_case(tmp_path)
    out = tmp_path / "out"
    lines = []
    rc = wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--segment-minutes", "360",
                  "--out", str(out), "--poll-interval", "0"] + list(extra),
                 sbatch_runner=fakes.sbatch, sacct_runner=fakes.sacct,
                 snapshot_fn=fakes.snapshot, sleep=lambda s: None, log=lines.append)
    report = None
    if (out / "run_report.json").exists():
        report = json.loads((out / "run_report.json").read_text())
    return rc, out, report, lines


def test_controller_two_segment_happy_path(tmp_path):
    case = _make_case(tmp_path)
    before = _tree_state(case)
    f = Fakes()
    rc, out, rep, _ = _run(tmp_path, f)
    assert rc == 0 and rep["status"] == "completed"
    assert [os.path.basename(s) for s, _ in f.submitted] == ["seg_0", "seg_1"]
    assert [s["attempts"][0]["job_id"] for s in rep["segments"]] == ["1001", "1002"]
    assert rep["physics_gray"] is False
    assert [d["decision"] for d in rep["decisions"]] == ["next", "observed_overhead", "done"]
    assert rep["restart_interval_min"] == 360
    obs = rep["decisions"][1]
    assert obs["overhead_s"] == 300.0 and obs["compute_replaced"] is True   # 600 s - 300 s d01
    # later --time from observed: 360 steps x 300/360 s + 300 s overhead + segment-0 read
    s1 = rep["segments"][1]
    assert s1["predicted_wall_s"] == pytest.approx(300.0 + 300.0 + rep["overheads"]["restart_read_s"])
    assert s1["time_limit"] == wr._hms(wr.time_limit_s(s1["predicted_wall_s"]))
    seg1 = out / "seg_1"
    for d in (1, 2, 3):
        name = "wrfrst_d%02d_2020-06-01_06:00:00" % d
        assert os.readlink(seg1 / name) == str(out / "seg_0" / name)
    tc1 = wp.parse_namelist(str(seg1 / "namelist.input"))["time_control"]
    assert tc1["restart"] == [True] and "override_restart_timers" not in tc1
    assert tc1["restart_interval"] == [360]
    xml = (out / "seg_1" / "adios2.xml").read_text()
    assert "wrfrst_d01_2020-06-01_06:00:00" in xml and "wrfrst_d01_2020-06-01_12:00:00" in xml
    tc0 = wp.parse_namelist(str(out / "seg_0" / "namelist.input"))["time_control"]
    assert tc0["restart_interval"] == [360] and "override_restart_timers" not in tc0
    assert (out / "seg_0" / "segment.sbatch").exists()
    assert _tree_state(case) == before


def test_controller_failure_without_rescue_stops(tmp_path):
    f = Fakes(outcomes=["cfl"])
    rc, out, rep, _ = _run(tmp_path, f)
    assert rc == 1 and rep["status"] == "failed"
    assert len(f.submitted) == 1
    assert "segment 0" in rep["error"] and "CFL" in rep["error"]
    assert not (out / "seg_1").exists()


def test_controller_cfl_advice_without_rescue_writes_namelist_and_stops(tmp_path):
    # decision 0056: without --cfl-rescue, a CFL/NaN failure stops (same as before) but also
    # prints the recommended time_step and writes it as a namelist next to the segment --
    # never resubmits anything itself.
    f = Fakes(outcomes=["cfl"])
    rc, out, rep, lines = _run(tmp_path, f)
    assert rc == 1 and rep["status"] == "failed"
    assert len(f.submitted) == 1                       # nothing resubmitted
    assert not (out / "seg_1").exists()
    advice_path = out / "seg_0" / "segment.advice.namelist.input"
    assert advice_path.exists()
    advice_nl = wp.parse_namelist(str(advice_path))["domains"]["time_step"]
    assert advice_nl == [45]                            # same 0.75x dt rescue_time_step gives
    assert (out / "seg_0" / "namelist.input").exists()  # the real (failed) namelist, dt=60,
    real_nl = wp.parse_namelist(str(out / "seg_0" / "namelist.input"))["domains"]["time_step"]
    assert real_nl == [60]                              # is untouched by the advice
    assert any("PHYSICS-GRAY ADVICE" in ln for ln in lines)
    assert any("segment.advice.namelist.input" in ln for ln in lines)
    decision = next(d for d in rep["decisions"] if d["decision"] == "cfl_advice")
    assert decision["recommended_time_step"] == 45
    assert decision["advice_namelist"].endswith("segment.advice.namelist.input")


def test_controller_cfl_rescue_once_then_original_dt(tmp_path):
    f = Fakes(outcomes=["cfl", "done", "done"])
    rc, out, rep, lines = _run(tmp_path, f, ["--cfl-rescue"])
    assert rc == 0 and rep["physics_gray"] is True
    assert [os.path.basename(s) for s, _ in f.submitted] == ["seg_0", "seg_0_rescue", "seg_1"]
    dom = lambda d: wp.parse_namelist(str(out / d / "namelist.input"))["domains"]["time_step"]
    assert dom("seg_0") == [60] and dom("seg_0_rescue") == [45] and dom("seg_1") == [60]
    assert rep["segments"][0]["attempts"][1]["time_step"] == 45
    assert os.readlink(out / "seg_1" / "wrfrst_d01_2020-06-01_06:00:00") == \
        str(out / "seg_0_rescue" / "wrfrst_d01_2020-06-01_06:00:00")
    assert any("PHYSICS-GRAY" in ln for ln in lines)


def test_controller_cfl_rescue_at_most_once(tmp_path):
    f = Fakes(outcomes=["cfl", "cfl"])
    rc, out, rep, _ = _run(tmp_path, f, ["--cfl-rescue"])
    assert rc == 1 and len(f.submitted) == 2
    assert "already used" in rep["error"]


def test_dry_run_submits_nothing(tmp_path):
    def boom(*a, **k):
        raise AssertionError("submitted")
    case = _make_case(tmp_path)
    out = tmp_path / "out"
    lines = []
    rc = wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--wall-limit", "02:00:00",
                  "--segment-minutes", "480", "--out", str(out), "--dry-run"], sbatch_runner=boom, sacct_runner=boom,
                 snapshot_fn=boom, log=lines.append)
    assert rc == 0 and not out.exists()
    # per-record wrfbdy (interval_seconds 6 h): 480 min is cut down to the 360-min record grid
    assert "2 segment(s) cut on input record times (multiples of 06:00:00: interval_seconds " \
           "06:00:00)" in lines[0]
    assert any("--segment-minutes 480 rounded down to 360" in ln for ln in lines)
    assert any(ln.startswith("namelist diff, segment 0") for ln in lines)
    assert any(ln.startswith("namelist diff, segment 1") for ln in lines)
    assert any(re.search(r"seg 0 .*--time=", ln) for ln in lines)


def test_dry_run_with_prediction_file(tmp_path):
    case = _make_case(tmp_path)
    pred = {"hardware": {"nx": 8, "ny": 9, "omp": 1}, "nodes": 1,
            "result": {"points": [{"at": "2020-06-01T03:00:00", "s_normal": 0.5},
                                  {"at": "2020-06-01T09:00:00", "s_normal": 0.7}],
                       "r_night": 2.0, "r_day": 4.0, "init_s": 60.0},
            "io": {"history": [{"domain": 1, "median_s": 5.0}, {"domain": 2, "median_s": 3.0},
                               {"domain": 3, "median_s": 2.0}],
                   "restart": {"per_write_s": 40.0}}}
    p = tmp_path / "pred_rebuild.json"
    p.write_text(json.dumps(pred))
    lines = []
    rc = wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--wall-limit", "00:10:00",
                  "--prediction", str(p), "--out", str(tmp_path / "o"), "--dry-run"],
                 log=lines.append)
    assert rc == 0
    assert "overheads: prediction" in lines[0]
    segs = [ln for ln in lines if ln.startswith("  seg ")]
    assert len(segs) > 1


def test_out_inside_case_refused(tmp_path):
    case = _make_case(tmp_path)
    assert wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9",
                    "--out", str(case / "out"), "--dry-run"], log=lambda s: None) == 2


# ---------------------------------------------------------------------------
# 0.8.2 (climate-ea50): the case's restart cadence is kept; namelist-named files are staged
# ---------------------------------------------------------------------------
_RST_LINE = " restart_interval                    = 7200,"


def _with_restart(minutes, text=NAMELIST_3DOM):
    return text.replace(_RST_LINE, " restart_interval                    = %d," % minutes)


def test_restart_setting_one_segment_keeps_case_interval_below_run_length(tmp_path):
    g = _geo(tmp_path)                                      # 12 h of 60 s steps
    keep = wr.restart_setting(g, None, 720, 180 * 60.0)
    assert keep["set_min"] is None and keep["cadence_s"] == 10800.0
    assert "kept" in keep["label"] and "03:00:00" in keep["label"]
    assert wr.restart_write_steps(g, 0, 720, 10800.0) == [180, 360, 540, 720]
    off = wr.restart_setting(g, None, 720, 7200 * 60.0)    # >= run length: nothing to keep
    assert off["set_min"] == wr.RESTART_OFF_MIN and off["cadence_s"] is None
    assert "off (one segment" in off["label"]
    assert wr.restart_setting(g, None, 720, 0.0)["set_min"] == wr.RESTART_OFF_MIN


def test_restart_setting_segments_keep_a_dividing_case_interval_else_set_L(tmp_path):
    g = _geo(tmp_path)
    keep = wr.restart_setting(g, 360, 720, 120 * 60.0)     # 2 h divides the 6 h quantum
    assert keep["set_min"] is None and keep["cadence_s"] == 7200.0
    odd = wr.restart_setting(g, 360, 720, 250 * 60.0)
    assert odd["set_min"] == 360 and odd["cadence_s"] == 21600.0
    assert "does not divide" in odd["label"]
    assert wr.restart_setting(g, 360, 720, 0.0)["set_min"] == 360


def test_restart_write_steps_off_grid_cadence_rounds_up(tmp_path):
    g = _geo(tmp_path)
    assert wr.restart_write_steps(g, 0, 10, 90.0) == [2, 3, 5, 6, 8, 9]
    assert wr.restart_write_steps(g, 3, 9, 90.0) == [5, 6, 8, 9]
    assert wr.restart_write_steps(g, 0, 10, None) == []


def test_kept_restart_writes_are_in_the_predicted_wall(tmp_path):
    g = _geo(tmp_path)
    over = wr.overheads(restart_write_s=50.0)
    comp = wr.constant_compute_fn(1.0)
    _L, one_off = wr.plan_run(g, 720, [{"config": _cfg(), "compute": comp}], over)
    _L, one_kept = wr.plan_run(g, 720, [{"config": _cfg(), "compute": comp}], over,
                               case_restart_s=180 * 60.0)
    assert one_kept[0]["predicted_wall_s"] - one_off[0]["predicted_wall_s"] == pytest.approx(200.0)
    L, segs = wr.plan_run(g, 720, [{"config": _cfg(), "compute": comp}], over,
                          segment_minutes=360, case_restart_s=120 * 60.0)
    base = wr.segment_wall(g, comp, over, 0, 360, None)
    assert segs[0]["predicted_wall_s"] == pytest.approx(base + 3 * 50.0)      # 2, 4, 6 h


def test_namelist_none_leaves_case_restart_keys(tmp_path):
    base = NAMELIST_3DOM.replace(_RST_LINE, _RST_LINE + "\n restart_interval_h = 5,")
    txt = wr.render_segment_namelist(base, _dt("2020-06-01_00:00:00"), _dt("2020-06-01_06:00:00"),
                                     1, _cfg(), None)
    tc = _nl_from_text(tmp_path, txt)["time_control"]
    assert tc["restart_interval"] == [7200] and tc["restart_interval_h"] == [5]
    assert tc["restart"] == [True]


def test_single_segment_run_keeps_case_restart_interval(tmp_path):
    """climate-ea50: one segment used to get restart_interval = 1000000 -> no checkpoint."""
    case = _make_case(tmp_path, _with_restart(180))
    out = tmp_path / "out"
    f = Fakes()
    lines = []
    rc = wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--out", str(out),
                  "--poll-interval", "0"], sbatch_runner=f.sbatch, sacct_runner=f.sacct,
                 snapshot_fn=f.snapshot, sleep=lambda s: None, log=lines.append)
    assert rc == 0 and len(f.submitted) == 1
    assert "1 segment(s), restart every 03:00:00 (case's restart_interval kept; one segment)" \
        in lines[0]
    tc = wp.parse_namelist(str(out / "seg_0" / "namelist.input"))["time_control"]
    assert tc["restart_interval"] == [180]
    rep = json.loads((out / "run_report.json").read_text())
    assert rep["restart_interval_source"] == "case" and rep["restart_cadence_s"] == 10800.0
    xml = (out / "seg_0" / "adios2.xml").read_text()
    for t in ("03", "06", "09", "12"):
        assert "wrfrst_d01_2020-06-01_%s:00:00" % t in xml


def test_single_segment_run_without_case_restart_stays_off(tmp_path):
    _make_case(tmp_path)                                     # restart_interval 7200 > 12 h
    lines = []
    rc = wr.main([str(tmp_path / "case"), "--nodes", "1", "--nx", "8", "--ny", "9",
                  "--out", str(tmp_path / "o"), "--dry-run", "--resume", "0"], log=lines.append)
    assert rc == 0 and "restart off (one segment; case restart_interval" in lines[0]


def test_setup_segment_dir_stages_namelist_named_files(tmp_path):
    case = _make_case(tmp_path)
    (case / "io").mkdir()
    (case / "io" / "fields_d02.txt").write_text("x")
    (case / "namelist.iofields").write_text("x")             # skipped by the namelist.* rule
    (case / "wrflowinp_d01").write_text("x")
    nml = _with_restart(180).replace(
        " io_form_history                     = 2",
        " iofields_filename = 'namelist.iofields', 'io/fields_d02.txt', 'gone_d03.txt',\n"
        " io_form_history                     = 2")
    before = _tree_state(case)
    seg = tmp_path / "out" / "seg_0"
    res = wr.setup_segment_dir(str(case), str(seg), nml)
    assert os.readlink(seg / "io" / "fields_d02.txt") == str(case / "io" / "fields_d02.txt")
    assert os.readlink(seg / "namelist.iofields") == str(case / "namelist.iofields")
    assert res["missing"] == [("iofields_filename", "gone_d03.txt")]
    assert _tree_state(case) == before


def test_missing_iofields_is_a_plan_warning(tmp_path):
    nml = NAMELIST_3DOM.replace(" io_form_history                     = 2",
                                ' iofields_filename = "iofields_d01.txt",\n'
                                " io_form_history                     = 2")
    _make_case(tmp_path, nml)
    lines = []
    rc = wr.main([str(tmp_path / "case"), "--nodes", "1", "--nx", "8", "--ny", "9",
                  "--out", str(tmp_path / "o"), "--dry-run"], log=lines.append)
    assert rc == 0
    warn = [ln for ln in lines if "WARNING" in ln]
    assert len(warn) == 1 and "iofields_d01.txt" in warn[0]
    assert "hangs at input_input" in warn[0]


# ---------------------------------------------------------------------------
# ADIOS2 I/O (default since 2026-10-03; --io netcdf = legacy escape)
# ---------------------------------------------------------------------------
NAMELIST_AUX = NAMELIST_3DOM.replace(
    " &physics\n", " &physics\n sst_update = 1,\n").replace(
    " io_form_boundary                    = 2\n",
    " io_form_boundary                    = 2\n auxinput4_inname = \"wrflowinp_d<domain>\",\n"
    " io_form_auxinput4 = 2,\n auxinput4_interval = 360, 360, 360,\n") + (
    " &fdda\n grid_fdda = 1, 0, 0,\n gfdda_inname = \"wrffdda_d<domain>\",\n"
    " io_form_gfdda = 2,\n /\n")


def _streams(text):
    return [st for st in wr.sweep.campaign_configs.adios2_input_streams(text)]


def test_adios2_streams_of_case_with_sst_update_and_fdda():
    sts = {st["stream"]: st for st in _streams(NAMELIST_AUX)}
    assert list(sts) == ["input", "boundary", "auxinput4", "gfdda"]
    assert sts["input"]["names"] == ["wrfinput_d01", "wrfinput_d02", "wrfinput_d03"]
    assert sts["input"]["required"] == ["wrfinput_d01"]
    assert sts["boundary"]["names"] == ["wrfbdy_d01"]
    assert sts["auxinput4"]["inname"] == "wrflowinp_d<domain>.bp"
    assert sts["gfdda"]["names"] == ["wrffdda_d01"] and sts["gfdda"]["group"] == "fdda"
    assert [st["stream"] for st in _streams(NAMELIST_3DOM)] == ["input", "boundary"]


def test_namelist_adios2_every_io_form_14_and_bp_innames():
    cfg = _cfg()
    txt = wr.render_segment_namelist(NAMELIST_AUX, _dt("2020-06-01_00:00:00"),
                                     _dt("2020-06-01_06:00:00"), 0, cfg, 360,
                                     io_streams=_streams(NAMELIST_AUX))
    for key in ("io_form_history", "io_form_restart", "io_form_input", "io_form_boundary",
                "io_form_auxinput4", "io_form_gfdda"):
        assert re.search(r"(?m)^\s*%s\s*=\s*14\s*,?\s*$" % key, txt), key
        assert not re.search(r"(?m)^\s*%s\s*=\s*2\b" % key, txt), key
    for key, name in (("input_inname", "wrfinput_d<domain>.bp"),
                      ("bdy_inname", "wrfbdy_d<domain>.bp"),
                      ("auxinput4_inname", "wrflowinp_d<domain>.bp"),
                      ("gfdda_inname", "wrffdda_d<domain>.bp")):
        assert len(re.findall(r'(?m)^\s*%s\s*=\s*"%s"' % (key, re.escape(name)), txt)) == 1, key
    fdda = txt[txt.index("&fdda"):]
    assert "io_form_gfdda" in fdda.split("/")[0] and "gfdda_inname" in fdda.split("/")[0]
    assert re.search(r"(?ms)^&namelist_adios2\s*\n\s*adios2_numaggregators = %d," % cfg["ranks"],
                     txt)


def test_namelist_netcdf_legacy_keeps_case_io_forms():
    txt = wr.render_segment_namelist(NAMELIST_AUX, _dt("2020-06-01_00:00:00"),
                                     _dt("2020-06-01_06:00:00"), 0, _cfg(), 360)
    for key in ("io_form_history", "io_form_input", "io_form_boundary", "io_form_auxinput4",
                "io_form_gfdda"):
        assert re.search(r"(?m)^\s*%s\s*=\s*2\b" % key, txt), key
    assert "input_inname" not in txt and "adios2_numaggregators" not in txt


def test_run_adios2_default_converts_once_links_bp_inputs_and_writes_xml(tmp_path, fake_bp):
    case = _make_case(tmp_path)                 # wrfinput_d01/d02 + wrfbdy_d01 netCDF, d03 .bp
    before = _tree_state(case)
    f = Fakes()
    rc, out, rep, lines = _run(tmp_path, f)
    assert rc == 0 and rep["status"] == "completed"
    cache = out / "_bp_inputs"
    assert fake_bp.calls == [(str(case), str(cache), ["wrfinput_d01", "wrfinput_d02",
                                                      "wrfbdy_d01"], "BP4")]
    io_line = [ln for ln in lines if ln.startswith("I/O:")]
    assert io_line == ["I/O: ADIOS2 (history/restart/input/boundary), inputs from %s "
                       "(3 netCDF -> BP4, 1 existing BP); one BP per record: wrfbdy_d01" % cache]
    for seg in ("seg_0", "seg_1"):
        d = out / seg
        assert os.readlink(d / "wrfinput_d01.bp") == str(cache / "wrfinput_d01")
        assert os.readlink(d / "wrfinput_d02.bp") == str(cache / "wrfinput_d02")
        assert os.readlink(d / "wrfinput_d03.bp") == str(case / "wrfinput_d03.bp")
        # per-record wrfbdy: one link per record file, never the container itself
        assert not os.path.lexists(d / "wrfbdy_d01.bp")
        for t in FakeBP.DEFAULT_RECORDS:
            rec = "wrfbdy_d01_%s.bp" % t
            assert os.readlink(d / rec) == str(d / "_bp_inputs" / "wrfbdy_d01" / rec)
        nl = (d / "namelist.input").read_text()
        for key in ("io_form_history", "io_form_restart", "io_form_input", "io_form_boundary"):
            assert re.search(r"(?m)^\s*%s\s*=\s*14\b" % key, nl), (seg, key)
        assert 'input_inname = "wrfinput_d<domain>.bp"' in nl
        assert re.findall(r"(?m)^\s*bdy_inname\s*=\s*(.*)$", nl) == ['"wrfbdy_d<domain>_<date>.bp",']
        bdy = wp.parse_namelist(str(d / "namelist.input"))["bdy_control"]
        assert bdy["multi_bdy_files"] == [True]
        xml = (d / "adios2.xml").read_text()
        assert xml.count("<adios-config>") == 1 and "wrfrst_d01_2020-06-01_06:00:00" in xml
    # the case's own adios2.xml ("x") is not XML: segments get wtune_bp's + the restart entries
    assert rep["io"]["mode"] == "adios2" and rep["io"]["bp_cache"] == str(cache)
    assert _tree_state(case) == before


def test_run_adios2_aux_streams_converted_and_switched(tmp_path, fake_bp):
    case = _make_case(tmp_path, NAMELIST_AUX)
    for n in ("wrflowinp_d01", "wrflowinp_d02", "wrflowinp_d03", "wrffdda_d01"):
        (case / n).write_text("x")
    rc, out, rep, lines = _run(tmp_path, Fakes())
    assert rc == 0
    names = fake_bp.calls[0][2]
    assert names[-4:] == ["wrflowinp_d01", "wrflowinp_d02", "wrflowinp_d03", "wrffdda_d01"]
    assert any(ln.startswith("I/O: ADIOS2 (history/restart/input/boundary/aux4/gfdda)")
               for ln in lines)
    nl = (out / "seg_0" / "namelist.input").read_text()
    assert re.search(r"(?m)^\s*io_form_auxinput4\s*=\s*14\b", nl)
    assert re.search(r"(?m)^\s*io_form_gfdda\s*=\s*14\b", nl)
    for seg in ("seg_0", "seg_1"):
        for n in ("wrflowinp_d02", "wrffdda_d01", "wrfbdy_d01"):
            rec = "%s_2020-06-01_06:00:00.bp" % n
            assert os.readlink(out / seg / rec) == str(out / seg / "_bp_inputs" / n / rec)
        nl = (out / seg / "namelist.input").read_text()
        assert 'auxinput4_inname = "wrflowinp_d<domain>_<date>.bp",' in nl
        assert 'gfdda_inname = "wrffdda_d<domain>_<date>.bp",' in nl
        tc = wp.parse_namelist(str(out / seg / "namelist.input"))["time_control"]
        assert tc["frames_per_auxinput4"] == [1, 1, 1] and tc["frames_per_auxinput10"] == [1, 1, 1]


def test_run_adios2_missing_aux_file_keeps_that_stream_netcdf(tmp_path, fake_bp):
    _make_case(tmp_path, NAMELIST_AUX)                       # no wrflowinp/wrffdda at all
    lines = []
    rc = wr.main([str(tmp_path / "case"), "--nodes", "1", "--nx", "8", "--ny", "9",
                  "--out", str(tmp_path / "o"), "--dry-run"], log=lines.append)
    assert rc == 0 and fake_bp.calls == []                  # dry run converts nothing
    assert any("aux input stream auxinput4 keeps the case's own (netCDF) io_form" in ln
               for ln in lines)
    assert any(ln.startswith("I/O: ADIOS2 (history/restart/input/boundary),") for ln in lines)
    assert any(re.search(r"\+\s*io_form_input\s*=\s*14", ln) for ln in lines)
    assert not any(re.search(r"\+.*io_form_auxinput4", ln) for ln in lines)


def test_run_io_netcdf_is_the_legacy_escape(tmp_path, fake_bp):
    rc, out, rep, lines = _run(tmp_path, Fakes(), extra=["--io", "netcdf"])
    assert rc == 0 and fake_bp.calls == []
    assert any(ln.startswith("WARNING: --io netcdf is the LEGACY") for ln in lines)
    assert any(ln.startswith("I/O: LEGACY netCDF") for ln in lines)
    nl = (out / "seg_0" / "namelist.input").read_text()
    assert re.search(r"(?m)^\s*io_form_input\s*=\s*2\b", nl) and "input_inname" not in nl
    assert not os.path.lexists(out / "seg_0" / "wrfinput_d01.bp")
    assert rep["io"] == {"mode": "netcdf"}


def test_run_converter_failure_stops_before_any_submission(tmp_path, fake_bp, capsys):
    fake_bp.fail = "no Python that can `import adios2`"
    f = Fakes()
    rc, out, rep, _ = _run(tmp_path, f)
    assert rc == 2 and f.submitted == [] and rep is None
    assert "ADIOS2 input staging failed" in capsys.readouterr().err


def test_bp_cache_under_project_input_bp(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "project.yaml").write_text("name: p\n")
    monkeypatch.setenv("WTUNE_PROJECT", str(proj))
    assert wr.bp_cache_dir(str(tmp_path / "out")) == str(proj / "input" / "bp")
    monkeypatch.delenv("WTUNE_PROJECT")
    monkeypatch.chdir(tmp_path)
    assert wr.bp_cache_dir(str(tmp_path / "out")) == str(tmp_path / "out" / "_bp_inputs")


def test_merge_adios2_xml_dedupes_io_names():
    base = ('<?xml version="1.0"?>\n<adios-config>\n  <io name="wrfrst_d01_X">\n'
            '    <engine type="BP5"/>\n  </io>\n</adios-config>\n')
    extra = ('<?xml version="1.0"?>\n<adios-config>\n  <io name="wrfrst_d01_X">\n'
             '    <engine type="BP4"/>\n  </io>\n  <io name="a">\n  </io>\n</adios-config>\n')
    out = wr.merge_adios2_xml(base, extra)
    assert out.count('name="wrfrst_d01_X"') == 1 and 'name="a"' in out
    xml = wr.render_adios2_restart_xml(["X"], 1, out)
    assert xml.count('name="wrfrst_d01_X"') == 1


# ---------------------------------------------------------------------------
# per-record ADIOS2 inputs (WRF 4.8's ADIOS2 reader returns record 1 of a multi-record BP): one
# BP per record, named by WRF's clock time -> every segment cut on an input record time
# ---------------------------------------------------------------------------
def test_record_alignment_lcm_of_boundary_and_sst_intervals(tmp_path):
    g = _geo(tmp_path)
    nl = _nl_from_text(tmp_path, NAMELIST_AUX.replace("auxinput4_interval = 360, 360, 360",
                                                      "auxinput4_interval = 240, 240, 240"))
    a = wr.input_record_alignment(nl, g, ["wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02"])
    assert (a["steps"], a["seconds"]) == (720, 43200.0)          # lcm(6 h, 4 h) = 12 h
    assert a["label"] == "interval_seconds 06:00:00 + auxinput4_interval 04:00:00"
    b = wr.input_record_alignment(nl, g, ["wrfbdy_d01"])
    assert (b["steps"], b["seconds"]) == (360, 21600.0)
    assert wr.input_record_alignment(nl, g, ["wrfinput_d01"]) is None
    assert wr.input_record_alignment(nl, g, []) is None


def test_plan_segment_minutes_rounded_down_to_record_grid(tmp_path):
    g = _geo(tmp_path)
    comp, over = _synthetic_model(g)
    cand = [{"config": _cfg(), "compute": comp}]
    align = {"steps": 360, "seconds": 21600.0, "label": "interval_seconds 06:00:00"}
    notes = []
    L, segs = wr.plan_run(g, 720, cand, over, segment_minutes=500, align=align, notes=notes)
    assert L == 360 and [(s["start_step"], s["end_step"]) for s in segs] == [(0, 360), (360, 720)]
    assert notes == ["--segment-minutes 500 rounded down to 360: segment cuts must fall on input "
                     "record times (interval_seconds 06:00:00, every 06:00:00 from the run start)"]
    notes = []
    L, _ = wr.plan_run(g, 720, cand, over, segment_minutes=360, align=align, notes=notes)
    assert L == 360 and notes == []                              # already on the grid: no note
    L, segs = wr.plan_run(g, 720, cand, over, segment_minutes=720, align=align)
    assert L is None and len(segs) == 1                          # one segment: no cut at all
    with pytest.raises(wr.RunError, match=r"--segment-minutes 180 is shorter than one input "
                                          r"record interval.*every 06:00:00.*--segment-minutes 360"):
        wr.plan_run(g, 720, cand, over, segment_minutes=180, align=align)


def test_plan_wall_limit_searches_only_record_multiples(tmp_path):
    g = _geo(tmp_path)
    comp, over = _synthetic_model(g)
    cand = [{"config": _cfg(), "compute": comp}]
    align = {"steps": 240, "seconds": 14400.0, "label": "interval_seconds 04:00:00"}
    free, _ = wr.plan_run(g, 720, cand, over, wall_limit_s=1000.0)
    L, segs = wr.plan_run(g, 720, cand, over, wall_limit_s=1000.0, align=align)
    assert free % 240 and L == free // 240 * 240           # the largest record multiple that fits
    assert all(s["start_step"] % 240 == 0 for s in segs)
    with pytest.raises(wr.RunError, match="segment cuts must fall on input record times"):
        wr.plan_run(g, 720, cand, over, wall_limit_s=10.0, align=align)


def test_run_segment_minutes_shorter_than_boundary_interval_is_a_clear_error(tmp_path, capsys):
    case = _make_case(tmp_path)
    rc = wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--segment-minutes", "180",
                  "--out", str(tmp_path / "o"), "--dry-run"], log=lambda s: None)
    err = capsys.readouterr().err
    assert rc == 2 and "--segment-minutes 180 is shorter than one input record interval" in err
    # the legacy netCDF path has no record grid
    assert wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--segment-minutes",
                    "180", "--io", "netcdf", "--out", str(tmp_path / "o"), "--dry-run"],
                   log=lambda s: None) == 0


def test_run_sst_update_cuts_on_lcm_and_links_every_record(tmp_path, fake_bp):
    text = NAMELIST_AUX.replace("auxinput4_interval = 360, 360, 360",
                                "auxinput4_interval = 240, 240, 240")
    text = text.replace(" end_day                             = 01,   01,   01,",
                        " end_day                             = 02,   02,   02,")
    text = text.replace(" run_hours                           = 12,",
                        " run_hours                           = 36,")
    case = _make_case(tmp_path, text)
    for n in ("wrflowinp_d01", "wrflowinp_d02", "wrflowinp_d03", "wrffdda_d01"):
        (case / n).write_text("x")
    hours = ["2020-06-%02d_%02d:00:00" % (1 + h // 24, h % 24) for h in range(0, 37, 2)]
    for n in ("wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02", "wrflowinp_d03", "wrffdda_d01"):
        fake_bp.records[n] = hours
    f = Fakes()
    lines = []
    rc = wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--segment-minutes", "1000",
                  "--out", str(tmp_path / "out"), "--poll-interval", "0"],
                 sbatch_runner=f.sbatch, sacct_runner=f.sacct, snapshot_fn=f.snapshot,
                 sleep=lambda s: None, log=lines.append)
    assert rc == 0, lines
    rep = json.loads((tmp_path / "out" / "run_report.json").read_text())
    # lcm(6 h boundary, 4 h SST) = 12 h; 1000 min -> 720 min, never longer than asked
    assert [s["start"] for s in rep["segments"]] == ["2020-06-01_00:00:00", "2020-06-01_12:00:00",
                                                     "2020-06-02_00:00:00"]
    assert rep["io"]["per_record"] == ["wrfbdy_d01", "wrffdda_d01", "wrflowinp_d01",
                                       "wrflowinp_d02", "wrflowinp_d03"]
    assert rep["io"]["segment_unit_s"] == 43200.0
    assert any("--segment-minutes 1000 rounded down to 720" in ln for ln in lines)
    seg1 = tmp_path / "out" / "seg_1"
    for n in ("wrfbdy_d01", "wrflowinp_d03"):
        for t in hours:
            assert os.path.islink(seg1 / ("%s_%s.bp" % (n, t)))
    assert wp.parse_namelist(str(seg1 / "namelist.input"))["time_control"]["restart"] == [True]


def test_run_missing_record_at_a_segment_start_stops_before_submission(tmp_path, fake_bp, capsys):
    case = _make_case(tmp_path)
    fake_bp.records["wrfbdy_d01"] = ["2020-06-01_00:00:00", "2020-06-01_12:00:00"]
    f = Fakes()
    rc, out, rep, _ = _run(tmp_path, f)
    assert rc == 2 and f.submitted == []
    err = capsys.readouterr().err
    assert "missing at segment starts: seg 1 wrfbdy_d01_2020-06-01_06:00:00.bp" in err


def test_existing_per_record_dir_in_case_is_linked_not_converted(tmp_path, fake_bp):
    case = _make_case(tmp_path)
    (case / "wrfbdy_d01").unlink()
    for t in FakeBP.DEFAULT_RECORDS:
        _mk_bp(str(case / "wrfbdy_d01.bp" / ("wrfbdy_d01_%s.bp" % t)))
    rc, out, rep, lines = _run(tmp_path, Fakes())
    assert rc == 0
    assert "wrfbdy_d01" not in fake_bp.calls[0][2]
    rec = "wrfbdy_d01_2020-06-01_06:00:00.bp"
    assert os.readlink(out / "seg_1" / rec) == str(out / "seg_1" / "_bp_inputs" / "wrfbdy_d01"
                                                    / rec)
    assert ["cp", "-a", "--reflink=always", str(case / "wrfbdy_d01.bp"),
            str(out / "seg_1" / "_bp_inputs" / "wrfbdy_d01")] in fake_bp.cp_calls
    assert rep["io"]["per_record"] == ["wrfbdy_d01"]


def test_existing_single_multirecord_bp_is_reconverted_from_netcdf(tmp_path, fake_bp):
    case = _make_case(tmp_path)
    _mk_bp(str(case / "wrfbdy_d01.bp"))           # a single (multi-record) BP next to the netCDF
    rc, out, rep, lines = _run(tmp_path, Fakes())
    assert rc == 0 and "wrfbdy_d01" in fake_bp.calls[0][2]
    rec = "wrfbdy_d01_2020-06-01_06:00:00.bp"
    assert os.readlink(out / "seg_1" / rec) == str(out / "seg_1" / "_bp_inputs" / "wrfbdy_d01"
                                                    / rec)
    assert fake_bp.cp_calls[0][3] == str(out / "_bp_inputs" / "wrfbdy_d01")
    (case / "wrfbdy_d01").unlink()                 # no netCDF: used, with a warning
    lines = []
    assert wr.main([str(case), "--nodes", "1", "--nx", "8", "--ny", "9", "--out",
                    str(tmp_path / "o2"), "--dry-run"], log=lines.append) == 0
    assert any("single multi-record BP" in ln for ln in lines)


# ---------------------------------------------------------------------------
# fresh per-attempt input copy (NFS open stall, wiki 2026-10-03 adios2 gate)
# ---------------------------------------------------------------------------
def test_fresh_copy_per_segment_attempt_removed_after_success_kept_on_failure(tmp_path, fake_bp):
    f = Fakes(outcomes=["cfl", "done", "done"])
    rc, out, rep, lines = _run(tmp_path, f, ["--cfl-rescue"])
    assert rc == 0
    dests = [c[4] for c in fake_bp.cp_calls]
    assert dests == [str(out / d / "_bp_inputs" / "wrfbdy_d01")
                     for d in ("seg_0", "seg_0_rescue", "seg_1")]       # one copy per attempt
    assert all(c[:3] == ["cp", "-a", "--reflink=always"] for c in fake_bp.cp_calls)
    assert (out / "seg_0" / "_bp_inputs" / "wrfbdy_d01").is_dir()         # failed: kept
    assert not (out / "seg_0_rescue" / "_bp_inputs").exists()             # done: removed
    assert not (out / "seg_1" / "_bp_inputs").exists()
    rec = "wrfbdy_d01_2020-06-01_06:00:00.bp"
    assert os.readlink(out / "seg_1" / rec) == str(out / "seg_1" / "_bp_inputs" / "wrfbdy_d01"
                                                    / rec)
    assert sum("fresh per-record inputs" in ln and "(reflink)" in ln for ln in lines) == 3
    # wrfinput (read once at init) still links the shared cache
    assert os.readlink(out / "seg_1" / "wrfinput_d01.bp") == str(out / "_bp_inputs" /
                                                                 "wrfinput_d01")


def test_fresh_copy_falls_back_to_cp_a_without_reflink(tmp_path, fake_bp, capsys):
    fake_bp.no_reflink = True
    rc, out, rep, lines = _run(tmp_path, Fakes())
    assert rc == 0
    assert [c[:3] for c in fake_bp.cp_calls[:2]] == [["cp", "-a", "--reflink=always"],
                                                      ["cp", "-a", fake_bp.cp_calls[1][2]]]
    assert "WARNING: no reflink copy of" in capsys.readouterr().err
    assert any("fresh per-record inputs" in ln and "cp -a" in ln for ln in lines)


# ---------------------------------------------------------------------------
# pre-created output dirs (NFS CREATE stalls on rank 0, scc jobs 2962-2977)
# ---------------------------------------------------------------------------
def _nl(tmp_path, **kv):
    text = NAMELIST_3DOM
    for key, val in kv.items():
        text = re.sub(r"(?m)^ %s\s*=.*$" % key, " %s = %s" % (key, val), text)
    p = tmp_path / "nl.input"
    p.write_text(text)
    return wp.parse_namelist(str(p))


T0 = datetime.datetime(2020, 6, 1, 0)


def test_output_dir_names_two_domains_different_intervals(tmp_path):
    nl = _nl(tmp_path, history_interval="180, 60, 60", restart_interval="360")
    got = wr.output_dir_names(nl, T0, T0 + datetime.timedelta(hours=6), 2)
    assert [n for n in got if n.startswith("wrfout_d01")] == [
        "wrfout_d01_2020-06-01_%02d:00:00" % h for h in (0, 3, 6)]
    assert len([n for n in got if n.startswith("wrfout_d02")]) == 7
    assert sorted(n for n in got if n.startswith("wrfrst")) == [
        "wrfrst_d01_2020-06-01_06:00:00", "wrfrst_d02_2020-06-01_06:00:00"]


def test_output_dir_names_frames_per_outfile_and_restart_start_excluded(tmp_path):
    nl = _nl(tmp_path, frames_per_outfile="3, 3, 3", restart_interval="120")
    t1 = T0 + datetime.timedelta(hours=6)
    got = wr.output_dir_names(nl, t1, t1 + datetime.timedelta(hours=6), 1)
    assert [n for n in got if n.startswith("wrfout")] == [
        "wrfout_d01_2020-06-01_06:00:00", "wrfout_d01_2020-06-01_09:00:00",
        "wrfout_d01_2020-06-01_12:00:00"][:3]
    rst = [n for n in got if n.startswith("wrfrst")]
    assert "wrfrst_d01_2020-06-01_06:00:00" not in rst and len(rst) == 3


def test_output_dir_names_custom_history_outname_skips_history(tmp_path):
    nl = _nl(tmp_path, history_interval="60")
    nl["time_control"]["history_outname"] = ["out_d<domain>_<date>"]
    got = wr.output_dir_names(nl, T0, T0 + datetime.timedelta(hours=2), 1)
    assert not any(n.startswith("wrfout") for n in got)


def test_remove_empty_output_dirs_keeps_non_empty(tmp_path):
    names = ["wrfout_d01_a", "wrfrst_d01_b"]
    assert wr.precreate_output_dirs(str(tmp_path), names) == 2
    (tmp_path / names[1] / "md.idx").write_text("x")
    wr.remove_empty_output_dirs(str(tmp_path), names)
    assert not (tmp_path / names[0]).exists() and (tmp_path / names[1]).is_dir()


class _Spy(Fakes):
    """Records the empty dirs present at submit and what restart_files sees after the job."""
    def sbatch(self, cmd):
        seg = os.path.dirname(cmd.split()[-1].strip("'"))
        self.seen = getattr(self, "seen", {})
        self.seen[os.path.basename(seg)] = sorted(
            n for n in os.listdir(seg) if n.startswith(("wrfout", "wrfrst")))
        return super().sbatch(cmd)


def test_controller_precreates_and_cleans_before_handoff(tmp_path):
    f = _Spy()
    rc, out, rep, lines = _run(tmp_path, f)
    assert rc == 0
    assert sum(ln.startswith("  pre-created ") and ln.endswith(" output dirs") for ln in lines) == 2
    assert "wrfrst_d01_2020-06-01_06:00:00" in f.seen["seg_0"]
    assert "wrfout_d03_2020-06-01_00:00:00" in f.seen["seg_0"]
    seg0 = os.listdir(out / "seg_0")
    # leftovers removed; the hand-off restart (non-empty) survives and seg_1 linked it
    assert not [n for n in seg0 if n.startswith("wrfout")]
    assert os.path.isfile(out / "seg_0" / "wrfrst_d01_2020-06-01_06:00:00" / "md.idx")
    assert "wrfrst_d01_2020-06-01_06:00:00" in f.seen["seg_1"]      # seg_1's own start dir


def test_controller_io_netcdf_creates_nothing(tmp_path):
    f = _Spy()
    rc, out, rep, lines = _run(tmp_path, f, extra=["--io", "netcdf"])
    assert rc == 0 and not any("pre-created" in ln for ln in lines)
    assert f.seen["seg_0"] == []


def test_watch_and_post_ignore_empty_wrfout_dir(tmp_path):
    from post import extract
    run = tmp_path / "run"
    (run / "seg_0").mkdir(parents=True)
    (run / "seg_0" / "wrfout_d01_2020-06-01_03:00:00").mkdir()
    assert wtune_watch.latest_wrfout(str(run / "seg_0")) == (None, None)
    assert extract.list_frames(str(run), 1) == []
    assert extract.segment_dirs(str(run / "seg_0")) == []
    (run / "seg_0" / "wrfout_d01_2020-06-01_03:00:00" / "md.idx").write_text("x")
    assert wtune_watch.latest_wrfout(str(run / "seg_0"))[0] == datetime.datetime(2020, 6, 1, 3)


def test_existing_run_dir_is_refused_before_planning(tmp_path, capsys):
    f = Fakes()
    (tmp_path / "out" / "seg_0").mkdir(parents=True)
    rc, _out, _rep, lines = _run(tmp_path, f)
    assert rc != 0 and f.submitted == [] and lines == []      # refused before the plan is printed
    assert "already holds a run" in capsys.readouterr().err
