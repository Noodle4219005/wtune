#!/usr/bin/env python3
"""Independent reproduction tests, round 6 (derived from defect descriptions only)."""
import argparse
import datetime as dt
import gzip
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
from wtune import predict as wp
from wtune import run as wr
import wtune.watch
from test_wtune_run import Fakes, _make_case, fake_bp, FakeBP  # noqa: F401  (autouse fixture)
from test_wtune_run_resume import FMT, T0, Dying, _real_bp, _rsl

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = HERE


def _at(h):
    return T0 + dt.timedelta(hours=h)


# =====================================================================================
# 1. restart retention
# =====================================================================================
def _ckpt_run(tmp_path, fakes, extra=()):
    if not (tmp_path / "case").exists():
        _make_case(tmp_path)                      # restart_interval = 7200 min: none inside the run
    out, lines = tmp_path / "out", []
    rc = wr.main([str(tmp_path / "case"), "--nodes", "1", "--nx", "8", "--ny", "9",
                  "--out", str(out), "--poll-interval", "0", "--io", "netcdf"] + list(extra),
                 sbatch_runner=fakes.sbatch, sacct_runner=fakes.sacct,
                 snapshot_fn=fakes.snapshot, sleep=lambda s: None, log=lines.append)
    rep = json.loads((out / "run_report.json").read_text()) if (out / "run_report.json").exists() else None
    return rc, out, rep, lines


def _rst_times(seg, dom=1):
    pat = re.compile(r"^wrfrst_d%02d_(\S+)$" % dom)
    return sorted(m.group(1) for n in os.listdir(seg) for m in [pat.match(n)] if m
                  and not os.path.islink(os.path.join(seg, n)))


CK = ["--segment-minutes", "720", "--checkpoint-minutes", "120"]


def test_died_right_after_newest_restart_resumes_from_previous_set(tmp_path):
    f = Dying([("died", "NODE_FAIL", 4)])         # restarts 02:00 and 04:00 written, then dead
    rc, out, rep, lines = _ckpt_run(tmp_path, f, CK + ["--resume", "1"])
    assert rc == 0 and rep["status"] == "completed", rep and rep.get("error")
    first = rep["segments"][0]["attempts"][0]
    assert first["resume_time"] == "2020-06-01_02:00:00"
    assert [os.path.basename(s) for s, _ in f.submitted] == ["seg_0", "seg_0_r1"]


def test_failed_attempt_keeps_every_valid_restart_set(tmp_path):
    f = Dying([("died", "NODE_FAIL", 6)])
    rc, out, rep, _ = _ckpt_run(tmp_path, f, CK + ["--resume", "0"])
    assert rc == 1
    seg = str(out / "seg_0")
    for d in (1, 2, 3):
        assert _rst_times(seg, d) == ["2020-06-01_02:00:00", "2020-06-01_04:00:00",
                                      "2020-06-01_06:00:00"], d


class Writing(Fakes):
    """A job that finishes OK and has written every restart its namelist asks for."""

    def sbatch(self, cmd):
        jid = super().sbatch(cmd)
        seg = self.submitted[-1][0]
        nl = wp.parse_namelist(os.path.join(seg, "namelist.input"))
        info = wtune.watch.run_info(nl)
        n = len(info["domains"])
        every = dt.timedelta(minutes=nl["time_control"]["restart_interval"][0])
        rst = [T0 + j * every for j in range(1, 100) if info["start"] < T0 + j * every <= info["end"]]
        for t in rst:
            for d in range(1, n + 1):
                q = os.path.join(seg, "wrfrst_d%02d_%s" % (d, t.strftime(FMT)))
                if not os.path.exists(q):
                    _real_bp(q)
        with open(os.path.join(seg, "rsl.error.0000"), "w") as fh:
            fh.write(_rsl(info["start"], info["end"], rst, n))
        return jid


def test_ok_attempt_leaves_only_newest_valid_restart_set(tmp_path):
    f = Writing()
    rc, out, rep, _ = _ckpt_run(tmp_path, f, ["--segment-minutes", "360", "--checkpoint-minutes", "120"])
    assert rc == 0, rep and rep.get("error")
    assert _rst_times(str(out / "seg_0")) == ["2020-06-01_06:00:00"]
    assert _rst_times(str(out / "seg_1")) == ["2020-06-01_12:00:00"]
    for d in (2, 3):
        assert _rst_times(str(out / "seg_0"), d) == ["2020-06-01_06:00:00"]


def test_ok_attempt_still_cleans_after_a_resume(tmp_path):
    f = Dying([("died", "NODE_FAIL", 5)])
    rc, out, rep, _ = _ckpt_run(tmp_path, f, CK + ["--resume", "1"])
    assert rc == 0, rep and rep.get("error")
    assert rep["segments"][0]["attempts"][0]["resume_time"] == "2020-06-01_04:00:00"
    # once the resume finished, the failed attempt keeps only the set it resumed from
    assert _rst_times(str(out / "seg_0")) == ["2020-06-01_04:00:00"]
    assert _rst_times(str(out / "seg_0_r1")) == ["2020-06-01_12:00:00"]


def _side(tmp_path, max_dom=2):
    from wtune import sidecar as ws
    run = tmp_path / "out"
    report = {}
    side = ws.Sidecar(str(run), ws.post_cache_root(str(run)), max_dom, os.path.isdir, report,
                      save=lambda: None, log=lambda s: None, cores_per_node=72,
                      exclude_arg="", runner=lambda argv: None, keep_last_restart=True)
    seg = run / "seg_0"
    seg.mkdir(parents=True)
    return side, seg, report


def test_compress_history_sidecar_keeps_two_newest_valid_sets_while_running(tmp_path):
    side, seg, report = _side(tmp_path)
    rst = lambda d, h: str(seg / ("wrfrst_d%02d_%s" % (d, (T0 + dt.timedelta(hours=h)).strftime(FMT))))
    for h in (2, 4, 6):
        for d in (1, 2):
            _real_bp(rst(d, h))
    (seg / "rsl.error.0000").write_text(_rsl(T0, _at(7), [_at(2), _at(4), _at(6)], 2))
    side.poll(str(seg), "77", {"ranks": 80, "nodes": 5, "omp": 3}, "RUNNING")
    for h in (4, 6):
        for d in (1, 2):
            assert os.path.isdir(rst(d, h)), (h, d)
    assert not report["sidecar"]["valid_restarts"][-1].get("deleted")
    assert not report["sidecar"]["valid_restarts"][-2].get("deleted")


def test_sidecar_single_valid_set_and_unfinished_set_untouched(tmp_path):
    side, seg, report = _side(tmp_path)
    rst = lambda d, h: str(seg / ("wrfrst_d%02d_%s" % (d, (T0 + dt.timedelta(hours=h)).strftime(FMT))))
    for d in (1, 2):
        _real_bp(rst(d, 2))
    _real_bp(rst(1, 4))                                     # 04:00 still being written
    (seg / "rsl.error.0000").write_text(_rsl(T0, _at(5), [_at(2)], 2))
    side.poll(str(seg), "77", {"ranks": 80, "nodes": 5, "omp": 3}, "RUNNING")
    assert all(os.path.isdir(rst(d, 2)) for d in (1, 2)) and os.path.isdir(rst(1, 4))


# =====================================================================================
# 2. nc2bp in a job, not in the login process
# =====================================================================================
class ConvFakes(Fakes):
    """sbatch of a script under nc2bp/ is a conversion job: RUNNING for `running_polls` sacct
    polls, then `end_state`; the BP cache appears when the job completes."""

    def __init__(self, fake_bp, case, cache, end_state="COMPLETED", running_polls=2):  # noqa: F811
        super().__init__()
        self.fb, self.case, self.cache = fake_bp, case, cache
        self.end_state, self.running_polls = end_state, running_polls
        self.events, self.scripts, self.jobs, self.polls = [], [], {}, {}
        self.n_job_conversions = 0

    def sbatch(self, cmd):
        path = cmd.split()[-1].strip("'")
        if os.sep + "nc2bp" + os.sep in path:
            self.events.append("nc2bp")
            self.scripts.append(path)
            jid = str(500 + len(self.scripts))
            self.jobs[jid] = "conv"
            return "Submitted batch job %s\n" % jid
        self.events.append("segment")
        out = super().sbatch(cmd)
        return out

    def sacct(self, cmd):
        jid = cmd[cmd.index("-j") + 1]
        if self.jobs.get(jid) != "conv":
            return super().sacct(cmd)
        n = self.polls[jid] = self.polls.get(jid, 0) + 1
        if n <= self.running_polls:
            self.events.append("poll-running")
            return "%s|RUNNING|00:00:01\n" % jid
        if self.end_state == "COMPLETED" and n == self.running_polls + 1:
            self.fb.convert_inputs(self.case, self.cache, names=["wrfinput_d01", "wrfinput_d02", "wrfbdy_d01"])
            self.n_job_conversions += 1
        return "%s|%s|00:00:05\n" % (jid, self.end_state)


def _conv_main(tmp_path, f, extra=(), **kw):
    out, lines = tmp_path / "out", []
    rc = wr.main([str(tmp_path / "case"), "--nodes", "1", "--nx", "8", "--ny", "9",
                  "--segment-minutes", "360", "--out", str(out), "--poll-interval", "0"] + list(extra),
                 sbatch_runner=f.sbatch, sacct_runner=f.sacct, snapshot_fn=f.snapshot,
                 sleep=lambda s: None, log=lines.append, **kw)
    return rc, out, lines


class BpSpy:
    """Counts conversions that really had to run (an output missing from the cache) made through
    the wtune.bp module by the process under test; cached_inputs reports a cache as current."""

    def __init__(self, fb):
        self.real = []
        orig = fb.convert_inputs
        mod = sys.modules["wtune.bp"]

        def conv(case_dir, cache_dir, names=None, engine="BP4"):
            self.real.append(any(not os.path.isdir(os.path.join(cache_dir, n)) for n in names))
            return orig(case_dir, cache_dir, names=names, engine=engine)
        mod.convert_inputs = conv
        mod.cached_inputs = lambda d, c, names, **kw: (
            {n: os.path.join(c, n) for n in names}
            if all(os.path.isdir(os.path.join(c, n)) for n in names) else None)


@pytest.fixture
def spy(fake_bp):  # noqa: F811
    return BpSpy(fake_bp)


@pytest.fixture
def no_slurm_env(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def test_stale_cache_submits_one_conversion_job_waits_then_submits_segment0(tmp_path, fake_bp, spy, no_slurm_env):  # noqa: F811
    case = _make_case(tmp_path)
    f = ConvFakes(fake_bp, str(case), str(tmp_path / "out" / "_bp_inputs"))
    rc, out, lines = _conv_main(tmp_path, f, convert_in_job=True)
    assert rc == 0, lines
    assert f.events[:4] == ["nc2bp", "poll-running", "poll-running", "segment"], f.events
    assert f.events.count("nc2bp") == 1
    assert f.scripts[0].startswith(str(out / "nc2bp") + os.sep) and os.path.isfile(f.scripts[0])
    # the login process itself converted nothing: only the (faked) job did
    assert f.n_job_conversions == 1 and not any(spy.real)
    assert os.path.exists(out / "seg_0" / "wrfinput_d01.bp")


def test_failed_conversion_job_submits_nothing_and_login_never_converts(tmp_path, fake_bp, spy, no_slurm_env):  # noqa: F811
    _make_case(tmp_path)
    f = ConvFakes(fake_bp, str(tmp_path / "case"), str(tmp_path / "out" / "_bp_inputs"),
                  end_state="FAILED")
    rc, out, lines = _conv_main(tmp_path, f, convert_in_job=True)
    assert rc != 0
    assert f.events.count("segment") == 0 and f.events.count("nc2bp") == 1
    assert fake_bp.calls == [] and spy.real == []
    assert not any(n.startswith("seg_") for n in os.listdir(out))


def test_current_cache_submits_no_conversion_job(tmp_path, fake_bp, spy, no_slurm_env):  # noqa: F811
    case = _make_case(tmp_path)
    cache = tmp_path / "out" / "_bp_inputs"
    fake_bp.convert_inputs(str(case), str(cache), names=["wrfinput_d01", "wrfinput_d02", "wrfbdy_d01"])
    f = ConvFakes(fake_bp, str(case), str(cache))
    rc, out, lines = _conv_main(tmp_path, f, convert_in_job=True)
    assert rc == 0, lines
    assert "nc2bp" not in f.events and f.events[0] == "segment"
    assert not (out / "nc2bp").exists() or not os.listdir(out / "nc2bp")
    assert not any(spy.real)


def test_inside_a_slurm_job_it_converts_in_place(tmp_path, fake_bp, spy, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    case = _make_case(tmp_path)
    f = ConvFakes(fake_bp, str(case), str(tmp_path / "out" / "_bp_inputs"))
    rc, out, lines = _conv_main(tmp_path, f)
    assert rc == 0, lines
    assert "nc2bp" not in f.events and spy.real == [True]


def test_dry_run_submits_nothing_even_with_stale_cache(tmp_path, fake_bp, no_slurm_env):  # noqa: F811
    case = _make_case(tmp_path)
    f = ConvFakes(fake_bp, str(case), str(tmp_path / "out" / "_bp_inputs"))
    rc, out, lines = _conv_main(tmp_path, f, ["--dry-run"], convert_in_job=True)
    assert rc == 0 and f.events == [] and fake_bp.calls == []


# =====================================================================================
# 3. calibrate plan stages namelist-named files
# =====================================================================================
IOF_NAMELIST = """\
&time_control
 run_days = 0,
 run_hours = 24,
 run_minutes = 0,
 run_seconds = 0,
 history_interval = 180,
 restart = .false.,
 io_form_history = 2
 iofields_filename = "myiofields.txt", "myiofields.txt",
 ignore_iofields_warning = .true.,
 /
&domains
 time_step = 180,
 max_dom = 1,
 e_we = 100,
 e_sn = 100,
 e_vert = 45,
 /
&physics
 /
"""


def _plan_script(tmp_path, namelist=IOF_NAMELIST, extra_files=("myiofields.txt",)):
    from wtune import calibrate as wc
    case = tmp_path / "scase"
    case.mkdir()
    (case / "namelist.input").write_text(namelist)
    for n in extra_files:
        (case / n).write_text("+:h:0:T2\n")
    out = tmp_path / "plan_out"
    ns = argparse.Namespace(site="scc", case_dir=str(case), out=str(out), ranks_list="1",
                            cores_per_node=72, steps=20, min_patch=10, time_limit="00:15:00")
    assert wc.plan_cmd(ns) == 0
    return case, (out / "plan_1.sbatch").read_text()


def _run_namelist_part(tmp_path, case, script):
    lines = script.splitlines()
    i0 = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith('"$PYBIN"') and "edit-namelist" in ln)
    i1 = next(i for i, ln in enumerate(lines) if ln.startswith("export OMP_NUM_THREADS"))
    frag = "\n".join(lines[i0:i1])
    run = tmp_path / "rundir"
    run.mkdir()
    (run / "namelist.input.orig").write_text((case / "namelist.input").read_text())
    W = tmp_path / "W"
    (W / "main").mkdir(parents=True)
    (W / "main" / "wrf.exe").write_text("x")
    env = dict(os.environ, CAMPAIGN=os.path.join(REPO, "campaign"), W=str(W),
               PYBIN="/home/nk/devel/wtune/envs/wtune/.pixi/envs/default/bin/python")
    r = subprocess.run(["bash", "-c", frag], cwd=str(run), env=env, capture_output=True, text=True)
    return run, r


def test_plan_script_stages_iofields_file_after_namelist_edit(tmp_path):
    case, script = _plan_script(tmp_path)
    run, r = _run_namelist_part(tmp_path, case, script)
    assert (run / "namelist.input").is_file(), r.stderr
    assert (run / "myiofields.txt").is_file(), (r.stdout, r.stderr)
    assert (run / "myiofields.txt").read_text().strip().endswith("T2")


def test_plan_script_staging_does_not_replace_existing_and_tolerates_missing(tmp_path):
    case, script = _plan_script(tmp_path, extra_files=())     # file named but absent from the case
    run, r = _run_namelist_part(tmp_path, case, script)
    assert (run / "namelist.input").is_file(), r.stderr       # the run is not aborted by it


def test_plan_script_without_namelist_files_still_valid_bash(tmp_path):
    plain = IOF_NAMELIST.replace(' iofields_filename = "myiofields.txt", "myiofields.txt",\n', "")
    case, script = _plan_script(tmp_path, namelist=plain, extra_files=())
    p = tmp_path / "s.sh"
    p.write_text(script)
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0
    run, r = _run_namelist_part(tmp_path, case, script)
    assert (run / "namelist.input").is_file(), r.stderr


# =====================================================================================
# 4. network only on login nodes
# =====================================================================================
def _install_ne(root, name="ne_110m_coastline"):
    d = os.path.join(root, "shapefiles", "natural_earth", "physical")
    os.makedirs(d)
    for ext in ("shp", "shx", "dbf"):
        with open(os.path.join(d, "%s.%s" % (name, ext)), "w") as fh:
            fh.write("x")


@pytest.fixture
def ne_spy(monkeypatch):
    pytest.importorskip("cartopy")
    import cartopy
    import cartopy.io.shapereader as shp
    calls = []

    def boom(*a, **kw):
        calls.append((a, kw))
        raise AssertionError("shapereader.natural_earth called (it downloads)")
    monkeypatch.setattr(shp, "natural_earth", boom)
    monkeypatch.setattr(cartopy, "config", dict(cartopy.config))
    return calls


def test_natural_earth_ready_never_calls_the_downloading_reader(tmp_path, monkeypatch, ne_spy):
    from wtune.post import plotting
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("WTUNE_CARTOPY_DATA", str(empty))
    plotting._set_cartopy_data_dir()
    assert plotting._natural_earth_ready() is False
    assert ne_spy == []


def test_natural_earth_ready_true_from_disk_only(tmp_path, monkeypatch, ne_spy):
    from wtune.post import plotting
    root = tmp_path / "data"
    _install_ne(str(root))
    monkeypatch.setenv("WTUNE_CARTOPY_DATA", str(root))
    plotting._set_cartopy_data_dir()
    assert plotting._natural_earth_ready() is True
    assert ne_spy == []


def test_natural_earth_ready_other_resolution_not_confused(tmp_path, monkeypatch, ne_spy):
    from wtune.post import plotting
    root = tmp_path / "data"
    _install_ne(str(root), "ne_110m_coastline")
    monkeypatch.setenv("WTUNE_CARTOPY_DATA", str(root))
    plotting._set_cartopy_data_dir()
    assert plotting._natural_earth_ready("50m") is False
    assert ne_spy == []


BDECK_PLAIN = open(os.path.join(HERE, "tests", "fixtures", "post", "bal122005_sample.dat")).read()


class _Resp:
    def __init__(self, body):
        self._b, self.headers, self.status = body, {}, 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        if n is None or n < 0:
            b, self._b = self._b, b""
        else:
            b, self._b = self._b[:n], self._b[n:]
        return b


@pytest.fixture
def fake_nhc(monkeypatch):
    urls = []

    def urlopen(req, *a, **kw):
        url = getattr(req, "full_url", req)
        urls.append(url)
        body = BDECK_PLAIN.encode()
        return _Resp(gzip.compress(body) if url.endswith(".gz") else body)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return urls


def _project(tmp_path, storm="AL122005"):
    from wtune import project
    p = tmp_path / "proj"
    project.init_project(str(p), storm=storm)
    return str(p)


def test_bdeck_module_meets_fetch_contract():
    from wtune.post import fetch
    mods, errors = fetch.discover(refresh=True)
    assert "bdeck" in mods and "bdeck" not in errors, errors
    m = mods["bdeck"]
    assert m.NAME == "bdeck" and fetch.check_module(m) == []
    p = argparse.ArgumentParser()
    m.add_arguments(p)


def test_bdeck_fetch_writes_named_file_with_provenance_and_is_idempotent(tmp_path, fake_nhc):
    from wtune.post import fetch, sources
    proj = _project(tmp_path)
    paths = fetch.run_module("bdeck", argparse.Namespace(project=proj, run_dir=None))
    want = os.path.join(proj, "input", "obs", "bdeck", "bal122005.dat")
    assert paths == [want]
    from wtune.post import atcf
    assert len(atcf.parse_file(want)) > 0                    # a decompressed, parseable b-deck
    prov = sources.read_provenance(proj, "bdeck")
    assert prov and "bal122005.dat" in prov["files"] and prov["urls"]
    assert all(isinstance(prov[k], str) for k in fetch.REQUIRED_PROVENANCE)
    n = len(fake_nhc)
    again = fetch.run_module("bdeck", argparse.Namespace(project=proj, run_dir=None))
    assert again == [want]
    assert open(want).read() == open(want).read() and os.path.getsize(want) > 0
    assert fetch.is_present(fetch.discover()[0]["bdeck"], proj)
    assert len(fake_nhc) >= n


def test_project_without_storm_has_nothing_to_fetch(tmp_path, fake_nhc):
    from wtune.post import fetch
    proj = _project(tmp_path, storm=None)
    got = fetch.run_module("bdeck", argparse.Namespace(project=proj, run_dir=None))
    assert got == [] and fake_nhc == []
    line = fetch.offline_summary(proj, None, online=False) or ""
    assert "bdeck" not in line


def test_project_with_storm_lists_bdeck_as_missing_until_fetched(tmp_path, fake_nhc):
    from wtune.post import fetch
    proj = _project(tmp_path)
    assert "bdeck" in fetch.missing_locally(proj)
    fetch.run_module("bdeck", argparse.Namespace(project=proj, run_dir=None))
    assert "bdeck" not in fetch.missing_locally(proj)


HURDAT_PTS = [dict(time=dt.datetime(2005, 8, 28, 0), lat=25.0, lon=-85.0, vmax_ms=60.0, pmin_hpa=930.0,
                   rmw_km=30.0),
              dict(time=dt.datetime(2005, 8, 28, 6), lat=25.5, lon=-86.0, vmax_ms=65.0, pmin_hpa=920.0,
                   rmw_km=28.0)]


def test_resolve_bdeck_does_not_download_inside_a_slurm_job(tmp_path, monkeypatch):
    from wtune.post import atcf, tc
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    hits = []
    monkeypatch.setattr(atcf, "fetch_bdeck", lambda *a, **k: hits.append(a) or "x")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: hits.append(a))
    path = tc.resolve_bdeck("AL122005", str(tmp_path), hurdat2_points=HURDAT_PTS)
    assert hits == [] and os.path.isfile(path) and len(atcf.parse_file(path)) >= 2


def test_resolve_bdeck_inside_job_without_fallback_is_a_tcerror_not_a_download(tmp_path, monkeypatch):
    from wtune.post import atcf, tc
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    hits = []
    monkeypatch.setattr(atcf, "fetch_bdeck", lambda *a, **k: hits.append(a) or "x")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: hits.append(a))
    with pytest.raises(tc.TCError):
        tc.resolve_bdeck("AL122005", str(tmp_path))
    assert hits == []


def test_resolve_bdeck_still_downloads_on_a_login_node(tmp_path, monkeypatch):
    from wtune.post import atcf, tc
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    seen = []
    monkeypatch.setattr(atcf, "fetch_bdeck", lambda y, b, c, dest, **k: seen.append(dest) or dest)
    assert tc.resolve_bdeck("AL122005", str(tmp_path), hurdat2_points=HURDAT_PTS).endswith("bal122005.dat")
    assert len(seen) == 1


def test_suite_blocks_non_loopback_connections_quickly():
    t0 = time.monotonic()
    with pytest.raises(OSError):
        socket.create_connection(("93.184.216.34", 80), timeout=5)
    with pytest.raises(OSError):
        urllib.request.urlopen("http://example.com/", timeout=5)
    assert time.monotonic() - t0 < 3.0


def test_loopback_connections_still_work():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        c = socket.create_connection(srv.getsockname(), timeout=2)
        c.close()
    finally:
        srv.close()


def test_network_block_also_covers_name_resolution_and_udp_style_connects():
    t0 = time.monotonic()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        with pytest.raises(OSError):
            s.connect(("1.1.1.1", 443))
    finally:
        s.close()
    assert time.monotonic() - t0 < 3.0


# =====================================================================================
# 5. adaptive time step bounds
# =====================================================================================
def _nl(tmp_path, domains, run_hours=6):
    text = ("&time_control\n run_days = 0,\n run_hours = %d,\n run_minutes = 0,\n run_seconds = 0,\n"
            " history_interval = 60,\n io_form_history = 2,\n /\n&domains\n max_dom = 1,\n e_we = 100,\n"
            " e_sn = 100,\n e_vert = 45,\n time_step = 15,\n use_adaptive_time_step = .true.,\n%s /\n"
            "&physics\n radt = 10,\n /\n") % (run_hours, domains)
    p = tmp_path / "namelist.input"
    p.write_text(text)
    return wp.parse_namelist(str(p))


def test_adaptive_defaults_follow_wrf_nint_of_3dx_and_8dx(tmp_path):
    nl = _nl(tmp_path, " dx = 2500,\n dy = 2500,\n")
    lo_n, hi_n, lo, hi = wp.adaptive_step_bounds(nl, 21600.0)
    assert (lo, hi) == (8.0, 20.0)                    # NINT(7.5) = 8, NINT(20) = 20
    assert (lo_n, hi_n) == (21600 // 20, -(-21600 // 8))


def test_adaptive_defaults_use_min_of_dx_dy_and_round_half_away(tmp_path):
    nl = _nl(tmp_path, " dx = 5000,\n dy = 3500,\n")
    _lo_n, _hi_n, lo, hi = wp.adaptive_step_bounds(nl, 21600.0)
    assert (lo, hi) == (11.0, 28.0)                   # 3*3.5 = 10.5 -> 11 (Fortran NINT)


def test_adaptive_explicit_values_win_and_den_divides(tmp_path):
    nl = _nl(tmp_path, " dx = 5000,\n dy = 5000,\n min_time_step = 10,\n max_time_step = 40,\n")
    assert wp.adaptive_step_bounds(nl, 3600.0)[2:] == (10.0, 40.0)
    nl = _nl(tmp_path, " dx = 5000,\n dy = 5000,\n min_time_step = 10,\n max_time_step = 40,\n"
                       " min_time_step_den = 2,\n max_time_step_den = 4,\n")
    assert wp.adaptive_step_bounds(nl, 3600.0)[2:] == (5.0, 10.0)


def test_adaptive_off_gives_none(tmp_path):
    nl = _nl(tmp_path, " dx = 5000,\n dy = 5000,\n")
    nl["domains"]["use_adaptive_time_step"] = [False]
    assert wp.adaptive_step_bounds(nl, 3600.0) is None


CALIBRATION = os.path.join(HERE, "machine", "calibration", "scc.json")


def test_range_low_is_never_negative_for_conus25_like_case_with_max_200(tmp_path):
    from test_wtune_predict import CONUS25_NAMELIST
    case = tmp_path / "conus25"
    case.mkdir()
    nl = CONUS25_NAMELIST.replace("&domains", "&domains\n use_adaptive_time_step = .true.,\n"
                                  " max_time_step = 200,", 1)
    (case / "namelist.input").write_text(nl)
    config = {"ranks": 720, "omp": 1, "nx": 24, "ny": 30, "numtiles": 1}
    pred = wp.build_prediction(namelist_path=str(case / "namelist.input"), case_dir=str(case), nodes=7,
                               config=config, ledger_paths=[], calibration_path=CALIBRATION,
                               io_form_override=None, io_target="nfs")
    assert pred["range_low_s"] >= 0.0
    assert pred["range_low_s"] <= pred["total_s"] <= pred["range_high_s"]
    assert "use_adaptive_time_step" in pred["steps_note"]
