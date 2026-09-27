#!/usr/bin/env python3
"""pytest suite for wtune_run.py (`wtune run`: segmented WRF executor with restart hand-offs).
No job is ever submitted: sbatch/sacct/snapshot are fakes, every directory is a tmp dir."""
import datetime
import json
import os
import re
import subprocess

import pytest

import wtune_predict as wp
import wtune_run as wr
import wtune_watch

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
    assert (g["steps"], g["dt"], g["max_dom"], g["quantum"], g["hist_align"]) == (720, 60.0, 3, 1, 60)
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


def test_plan_wall_limit_with_synthetic_prediction(tmp_path):
    g = _geo(tmp_path)
    comp, over = _synthetic_model(g)
    cand = [{"config": _cfg(), "compute": comp}]
    limit = 400.0                                  # budget 320 s
    segs = wr.plan_run(g, 720, cand, over, wall_limit_s=limit)
    assert segs[0]["start_step"] == 0 and segs[-1]["end_step"] == 720
    for x, y in zip(segs, segs[1:]):
        assert x["end_step"] == y["start_step"]
    for s in segs:
        assert s["predicted_wall_s"] <= 0.8 * limit + 1e-9
        assert s["time_limit"] <= "00:06:40"       # capped at --wall-limit
    for s in segs[:-1]:
        assert s["end_step"] % g["hist_align"] == 0            # on the history grid
        a, b = s["start_step"], s["end_step"]
        # the next history-aligned end would not have fit
        assert wr.segment_wall(g, comp, over, a, b + g["hist_align"], 720) > 0.8 * limit
    assert [s["index"] for s in segs] == list(range(len(segs)))


def test_plan_wall_limit_boundaries_whole_minutes_and_steps(tmp_path):
    txt = NAMELIST_3DOM.replace("time_step                           = 60,",
                                "time_step                           = 36,").replace(
        "history_interval                    = 60,  60,   60,",
        "history_interval                    = 0,  0,   0,")
    g = _geo(tmp_path, txt)
    assert g["hist_align"] is None
    cand = [{"config": _cfg(), "compute": wr.constant_compute_fn(0.5)}]
    over = wr.overheads(init_s=30.0, restart_read_s=10.0, restart_write_s=10.0,
                        history_frame_s=[0, 0, 0])
    segs = wr.plan_run(g, g["steps"], cand, over, wall_limit_s=300.0)
    assert len(segs) > 2
    for s in segs[:-1]:
        secs = s["end_step"] * 36
        assert secs % 60 == 0 and s["end_step"] % 5 == 0
        assert _dt(s["end"]).second == 0


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
    segs = wr.plan_run(g, 720, cands, wr.overheads(), segment_minutes=360)
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
def _render(k, is_last, start="2020-06-01_06:00:00", end="2020-06-01_12:00:00", **kw):
    return wr.render_segment_namelist(NAMELIST_3DOM, _dt(start), _dt(end), k, is_last,
                                      _cfg(nx=12, ny=6, numtiles=4), **kw)


def test_namelist_every_domain_start_end_and_run_length(tmp_path):
    txt = _render(1, False, "2020-06-01_06:00:00", "2020-06-01_09:00:00")
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


def test_namelist_restart_keys_k0_vs_k1(tmp_path):
    nl0 = _nl_from_text(tmp_path, _render(0, False, "2020-06-01_00:00:00", "2020-06-01_06:00:00"),
                        "k0")
    tc0 = nl0["time_control"]
    assert tc0["restart"] == [False] and "override_restart_timers" not in tc0
    assert tc0["restart_interval"] == [360] and tc0["io_form_restart"] == [14]
    nl1 = _nl_from_text(tmp_path, _render(1, False, "2020-06-01_06:00:00",
                                          "2020-06-01_09:00:00", restart_io_form=2), "k1")
    tc1 = nl1["time_control"]
    assert tc1["restart"] == [True] and tc1["override_restart_timers"] == [True]
    assert tc1["restart_interval"] == [180] and tc1["io_form_restart"] == [2]
    last = _nl_from_text(tmp_path, _render(2, True), "k2")["time_control"]
    assert last["restart_interval"] == [wr.RESTART_OFF_MIN] and last["restart"] == [True]


def test_namelist_restart_interval_variants_and_time_step_override(tmp_path):
    base = NAMELIST_3DOM.replace(" restart_interval                    = 7200,",
                                 " restart_interval                    = 7200,\n"
                                 " restart_interval_h = 5,\n restart_interval_m = 30,")
    txt = wr.render_segment_namelist(base, _dt("2020-06-01_00:00:00"),
                                     _dt("2020-06-01_06:00:00"), 0, False, _cfg(),
                                     time_step=45)
    tc = _nl_from_text(tmp_path, txt)["time_control"]
    assert tc["restart_interval_h"] == [0] and tc["restart_interval_m"] == [360]
    assert tc["restart_interval"] == [360]
    assert _nl_from_text(tmp_path, txt, "b")["domains"]["time_step"] == [45]


def test_namelist_diff_summary_lists_changes():
    diff = wr.namelist_diff_summary(NAMELIST_3DOM, _render(1, False))
    assert any(ln.startswith("+") and "override_restart_timers" in ln for ln in diff)
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
                os.mkdir(os.path.join(seg, "wrfrst_d%02d_%s" % (d, info["end"].strftime(
                    "%Y-%m-%d_%H:%M:%S"))))
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
    assert [d["decision"] for d in rep["decisions"]] == ["next", "replan", "done"]
    seg1 = out / "seg_1"
    for d in (1, 2, 3):
        name = "wrfrst_d%02d_2020-06-01_06:00:00" % d
        assert os.readlink(seg1 / name) == str(out / "seg_0" / name)
    tc1 = wp.parse_namelist(str(seg1 / "namelist.input"))["time_control"]
    assert tc1["restart"] == [True] and tc1["override_restart_timers"] == [True]
    assert tc1["restart_interval"] == [wr.RESTART_OFF_MIN]
    assert "wrfrst_d01_2020-06-01_06:00:00" in (out / "seg_1" / "adios2.xml").read_text()
    assert (out / "seg_0" / "segment.sbatch").exists()
    assert _tree_state(case) == before


def test_controller_failure_without_rescue_stops(tmp_path):
    f = Fakes(outcomes=["cfl"])
    rc, out, rep, _ = _run(tmp_path, f)
    assert rc == 1 and rep["status"] == "failed"
    assert len(f.submitted) == 1
    assert "segment 0" in rep["error"] and "CFL" in rep["error"]
    assert not (out / "seg_1").exists()


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
                  "--segment-minutes", "240", "--out", str(out), "--dry-run"], sbatch_runner=boom, sacct_runner=boom,
                 snapshot_fn=boom, log=lines.append)
    assert rc == 0 and not out.exists()
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
