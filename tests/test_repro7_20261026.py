#!/usr/bin/env python3
"""Independent reproductions of a4c02b2 (second review): written from the behaviour
descriptions only. (a)/(b) conversion job under wtune stop / sbatch refusal, (c) restarts of a
failed attempt after its resume finished, (d) adaptive dt defaults, (e) post fetch bdeck,
(f) the test suite's network guard. Fakes only: no job, no network, no adios2."""
import argparse
import asyncio
import datetime as dt
import http.server
import json
import os
import signal
import socket
import threading
import urllib.request

import pytest

import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
from test_wtune_run import Fakes, _make_case, _run, fake_bp  # noqa: F401  (autouse fixture)
from test_wtune_run_resume import Dying
from wtune import predict as wp
from wtune import run as wr
import wtune.stopping

FMT = "%Y-%m-%d_%H:%M:%S"


@pytest.fixture(autouse=True)
def _reset_stop():
    yield
    wtune.stopping.STOP["reason"] = None


# ---------------------------------------------------------------------------
# (a)/(b) the conversion job of a login-node `wtune run`
# ---------------------------------------------------------------------------
class ConvFakes(Fakes):
    """Fakes plus a conversion job: `conv` decides what the nc2bp job does --
    "stop" (wtune stop arrives while it runs), "reject" (sbatch refuses it),
    "fail" (it ends FAILED), "done" (it converts the cache and completes)."""

    def __init__(self, conv, out, case, fb):
        super().__init__()
        self.conv, self.out, self.case, self.fb = conv, out, case, fb
        self.conv_job = None
        self.all_sbatch = []
        self.cancelled = []
        self.state_seen = []          # (job id, run.state.json jobs) at every sacct call
        self.handler_missing = False

    def sbatch(self, cmd):
        self.all_sbatch.append(cmd)
        if "nc2bp.sbatch" in cmd:
            if self.conv == "reject":
                raise RuntimeError("sbatch failed (rc=1): sbatch: error: Batch job submission "
                                   "failed: Invalid account or account/partition combination")
            self.conv_job = "2001"
            return "Submitted batch job 2001\n"
        return super().sbatch(cmd)

    def _state_jobs(self):
        p = self.out / "run.state.json"
        if not p.exists():
            return None
        return [str(j) for j in json.loads(p.read_text()).get("jobs", [])]

    def sacct(self, cmd):
        jid = cmd[cmd.index("-j") + 1]
        self.state_seen.append((jid, self._state_jobs()))
        if jid == self.conv_job:
            if self.conv == "stop":
                h = signal.getsignal(signal.SIGTERM)
                if not callable(h):
                    self.handler_missing = True
                    raise KeyboardInterrupt("no wtune stop handler during the conversion wait")
                if not wtune.stopping.STOP["reason"]:
                    h(signal.SIGTERM, None)          # what `wtune stop` (kill -TERM pid) does
                return "%s|RUNNING|00:01:00\n" % jid
            if self.conv == "fail":
                return "%s|FAILED|00:01:00\n" % jid
            if self.conv == "done":
                cache = os.path.join(str(self.out), "_bp_inputs")
                if not os.path.exists(os.path.join(cache, "wrfinput_d01")):
                    self.fb.convert_inputs(str(self.case), cache,
                                           ["wrfinput_d01", "wrfinput_d02", "wrfbdy_d01"])
                return "%s|COMPLETED|00:01:00\n" % jid
        return super().sacct(cmd)


def _main_conv(tmp_path, fakes):
    lines = []
    rc = wr.main([str(tmp_path / "case"), "--nodes", "1", "--nx", "8", "--ny", "9",
                  "--segment-minutes", "360", "--out", str(tmp_path / "out"),
                  "--poll-interval", "0"],
                 sbatch_runner=fakes.sbatch, sacct_runner=fakes.sacct,
                 snapshot_fn=fakes.snapshot, sleep=lambda s: None, log=lines.append,
                 scancel_runner=fakes.cancelled.append, convert_in_job=True)
    return rc, lines


def test_a_wtune_stop_during_conversion_wait_cancels_it_and_exits_130(tmp_path, fake_bp):  # noqa: F811
    case = _make_case(tmp_path)
    f = ConvFakes("stop", tmp_path / "out", case, fake_bp)
    try:
        rc, lines = _main_conv(tmp_path, f)
    except KeyboardInterrupt:
        pytest.fail("wtune stop during the conversion wait: no handler, the process would die "
                    "and leave job %s running" % f.conv_job)
    assert not f.handler_missing
    assert rc == 130
    assert f.cancelled == ["2001"]
    assert len(f.all_sbatch) == 1 and "nc2bp.sbatch" in f.all_sbatch[0]   # nothing else
    assert fake_bp.calls == []                                           # no login-node convert
    # run.state.json listed the conversion job while it was waited for
    assert f.state_seen and all(st == ["2001"] for j, st in f.state_seen if j == "2001")
    assert not (tmp_path / "out" / "seg_0").exists()


def test_b_sbatch_rejecting_the_conversion_job_is_a_clean_error(tmp_path, fake_bp, capsys):  # noqa: F811
    case = _make_case(tmp_path)
    f = ConvFakes("reject", tmp_path / "out", case, fake_bp)
    try:
        rc, lines = _main_conv(tmp_path, f)
    except RuntimeError as e:
        pytest.fail("traceback instead of an error line: %s" % e)
    assert rc == 2
    assert "error:" in capsys.readouterr().err
    assert fake_bp.calls == []                       # no conversion on the login node
    assert len(f.all_sbatch) == 1                    # no segment submitted
    assert not (tmp_path / "out" / "seg_0").exists()
    assert not (tmp_path / "out" / "run.state.json").exists()


def test_b2_a_failed_conversion_job_is_an_error_and_nothing_is_converted_here(tmp_path, fake_bp):  # noqa: F811
    case = _make_case(tmp_path)
    f = ConvFakes("fail", tmp_path / "out", case, fake_bp)
    rc, lines = _main_conv(tmp_path, f)
    assert rc == 2 and fake_bp.calls == [] and len(f.all_sbatch) == 1
    assert not (tmp_path / "out" / "run.state.json").exists()


def test_a2_after_the_conversion_job_the_segments_list_their_jobs(tmp_path, fake_bp):  # noqa: F811
    case = _make_case(tmp_path)
    f = ConvFakes("done", tmp_path / "out", case, fake_bp)
    rc, lines = _main_conv(tmp_path, f)
    assert rc == 0
    # the job converted (fake, inside sacct); the login node's later convert_inputs pass only
    # checks the now-current cache (the fake converter does not model that skip)
    assert fake_bp.calls and fake_bp.calls[0][1].endswith("_bp_inputs")
    seg_states = [(j, st) for j, st in f.state_seen if j != "2001"]
    assert seg_states and all(st is not None and j in st for j, st in seg_states)
    assert not (tmp_path / "out" / "run.state.json").exists()     # removed at the normal end


# ---------------------------------------------------------------------------
# (c) restarts of the failed attempt once its resume finished
# ---------------------------------------------------------------------------
def _sets(d):
    """{time: [domains]} of the restart sets WRF itself wrote into d (links excluded)."""
    out = {}
    for p in d.glob("wrfrst_d*_*"):
        if p.is_symlink():
            continue
        out.setdefault(p.name[len("wrfrst_dNN_"):], []).append(int(p.name[8:10]))
    return {t: sorted(v) for t, v in out.items()}


def test_c_failed_attempt_keeps_only_the_set_its_resume_started_from(tmp_path):
    _make_case(tmp_path)
    f = Dying([("died", "NODE_FAIL", 8)])
    rc, out, rep, _ = _run(tmp_path, f, ["--segment-minutes", "720", "--checkpoint-minutes",
                                         "120", "--io", "netcdf"])
    assert rc == 0 and rep["status"] == "completed"
    first = rep["segments"][0]["attempts"][0]
    assert first["resume_time"] == "2020-06-01_06:00:00"
    assert _sets(out / "seg_0") == {"2020-06-01_06:00:00": [1, 2, 3]}
    for d in (1, 2, 3):                              # the finished attempt's links still resolve
        assert os.path.exists(out / "seg_0_r1" / ("wrfrst_d%02d_2020-06-01_06:00:00" % d))
    assert _sets(out / "seg_0_r1") == {"2020-06-01_12:00:00": [1, 2, 3]}


def test_c2_resume_chain_r1_r2(tmp_path):
    _make_case(tmp_path)
    f = Dying([("died", "NODE_FAIL", 8), ("died", "NODE_FAIL", 11)])
    rc, out, rep, _ = _run(tmp_path, f, ["--segment-minutes", "720", "--checkpoint-minutes",
                                         "120", "--io", "netcdf", "--resume", "2"])
    assert rc == 0 and rep["resumes"] == 2
    a0, a1, a2 = rep["segments"][0]["attempts"]
    assert (a0["resume_time"], a1["resume_time"]) == ("2020-06-01_06:00:00",
                                                       "2020-06-01_10:00:00")
    for d in (1, 2, 3):
        assert os.path.exists(out / "seg_0_r2" / ("wrfrst_d%02d_2020-06-01_10:00:00" % d))
    assert _sets(out / "seg_0_r1") == {"2020-06-01_10:00:00": [1, 2, 3]}


def test_c3_resume_chain_trims_the_first_failed_attempt_too(tmp_path):
    # beyond the stated contract: in a chain seg_0 is also a failed attempt a resume started
    # from (06:00); whatever else it holds is never read again
    _make_case(tmp_path)
    f = Dying([("died", "NODE_FAIL", 8), ("died", "NODE_FAIL", 11)])
    rc, out, rep, _ = _run(tmp_path, f, ["--segment-minutes", "720", "--checkpoint-minutes",
                                         "120", "--io", "netcdf", "--resume", "2"])
    assert rc == 0
    assert _sets(out / "seg_0") == {"2020-06-01_06:00:00": [1, 2, 3]}


# ---------------------------------------------------------------------------
# (d) use_adaptive_time_step defaults
# ---------------------------------------------------------------------------
def _adaptive(tmp_path, dx, dy=None, extra=""):
    p = tmp_path / "adaptive.nml"
    p.write_text(" &domains\n use_adaptive_time_step = .true.,\n dx = %s,\n dy = %s,\n%s /\n"
                 % (dx, dy if dy is not None else dx, extra))
    return wp.adaptive_step_bounds(wp.parse_namelist(str(p)), 3600.0)


def test_d_defaults_round_half_away_from_zero_like_fortran_nint(tmp_path):
    # WRF start_em.F: min = NINT(3*min(dx,dy)/1000), max = NINT(8*min(dx,dy)/1000)
    assert _adaptive(tmp_path, 1500.0) == (300, 720, 5.0, 12.0)          # NINT(4.5) = 5, not 4
    lo, hi = _adaptive(tmp_path, 1562.5)[2:]
    assert (lo, hi) == (5.0, 13.0)                                       # NINT(12.5) = 13
    assert _adaptive(tmp_path, 3000.0, 1500.0)[2:] == (5.0, 12.0)        # min(dx, dy)


def test_d_den_divides_the_default_and_an_explicit_value(tmp_path):
    lo, hi = _adaptive(tmp_path, 1500.0, extra=" min_time_step_den = 2,\n max_time_step_den = 4,\n")[2:]
    assert (lo, hi) == (2.5, 3.0)
    lo, hi = _adaptive(tmp_path, 1500.0, extra=" min_time_step = 9,\n min_time_step_den = 2,\n"
                                               " max_time_step = 30,\n")[2:]
    assert (lo, hi) == (4.5, 30.0)


# ---------------------------------------------------------------------------
# (e) wtune post fetch bdeck
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_nhc(monkeypatch):
    from wtune.post import atcf
    calls = []

    def fetch(year, basin, cy, dest, timeout=30):
        calls.append((year, basin, cy, dest))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            fh.write("AL, 14, 2016092812,   , BEST,   0, 133N,  609W,  45, 1004\n")
        return dest
    monkeypatch.setattr(atcf, "fetch_bdeck", fetch)
    return calls


def _bdeck_argv(project, storm=None):
    from wtune import submit
    window = (dt.datetime(2016, 9, 28, 0), dt.datetime(2016, 10, 1, 0),
              (10.0, 30.0, -80.0, -50.0), dt.datetime(2016, 10, 1, 0))
    (name, cmd), = submit.fetch_steps(project, ["bdeck"], window)
    argv = cmd[2:]                                   # drop WTUNE_BIN "post"
    assert argv[:2] == ["fetch", "bdeck"]
    return argv + (["--storm", storm] if storm else [])


def test_e_bdeck_takes_wtune_submits_window_options(tmp_path, fake_nhc):
    from wtune.post import cli
    (tmp_path / "project.yaml").write_text("storm:\n  id: AL142016\n")
    try:
        rc = cli.main(_bdeck_argv(str(tmp_path)))
    except SystemExit as e:
        pytest.fail("argparse refused wtune submit's fetch step: exit %s" % e.code)
    assert rc == 0 and len(fake_nhc) == 1
    assert (tmp_path / "input" / "obs" / "bdeck" / "bal142016.dat").is_file()


def test_e_no_download_when_the_sha_matches_provenance(tmp_path, fake_nhc):
    from wtune.post import fetch, sources
    from wtune.post.fetch import bdeck
    (tmp_path / "project.yaml").write_text("storm:\n  id: AL142016\n")
    ns = argparse.Namespace(project=str(tmp_path), storm=None, run_dir=None, start=None,
                            end=None, bbox=None)
    got = fetch.run_module("bdeck", ns)
    assert len(fake_nhc) == 1 and sources.read_provenance(str(tmp_path), "bdeck")["files"]
    assert fetch.run_module("bdeck", ns) == got and len(fake_nhc) == 1    # sha matches: skipped
    with open(got[0], "a") as fh:                                         # corrupted locally
        fh.write("garbage\n")
    assert fetch.run_module("bdeck", ns) == got and len(fake_nhc) == 2    # re-downloaded
    assert bdeck.present(str(tmp_path))


def test_e_atcf_storm_without_project_yaml(tmp_path, fake_nhc):
    from wtune.post import cli
    proj = tmp_path / "bare"
    proj.mkdir()
    rc = cli.main(_bdeck_argv(str(proj), storm="al142016"))
    assert rc == 0 and len(fake_nhc) == 1
    assert (proj / "input" / "obs" / "bdeck" / "bal142016.dat").is_file()


# ---------------------------------------------------------------------------
# (f) the suite's network guard (conftest._no_network)
# ---------------------------------------------------------------------------
def _server(host, family=socket.AF_INET):
    srv = socket.socket(family, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, 0))
    srv.listen(4)
    return srv


def _ipv6():
    try:
        s = _server("::1", socket.AF_INET6)
    except OSError:
        return False
    s.close()
    return True


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2"])
def test_f_loopback_v4_whole_127_8(host):
    srv = _server(host)
    port = srv.getsockname()[1]
    with socket.create_connection((host, port), timeout=2):
        pass
    c = socket.socket()
    try:
        c.settimeout(2)
        assert c.connect_ex((host, port)) == 0
    finally:
        c.close()
        srv.close()


def test_f_loopback_v6_and_ipv4_mapped_and_localhost():
    srv = _server("127.0.0.1")
    port = srv.getsockname()[1]
    try:
        if _ipv6():
            c = socket.socket(socket.AF_INET6)
            try:
                c.settimeout(2)
                c.connect(("::ffff:127.0.0.1", port, 0, 0))     # IPv4-mapped
            finally:
                c.close()
            s6 = _server("::1", socket.AF_INET6)
            try:
                with socket.create_connection(("::1", s6.getsockname()[1]), timeout=2):
                    pass
            finally:
                s6.close()
        # 'localhost' may resolve to ::1 first; create_connection then falls back to 127.0.0.1
        with socket.create_connection(("localhost", port), timeout=2):
            pass
    finally:
        srv.close()


def test_f_http_server_on_all_interfaces_and_asyncio(monkeypatch):
    for v in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy",
              "ALL_PROXY"):
        monkeypatch.delenv(v, raising=False)

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass
    httpd = http.server.HTTPServer(("", 0), H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        port = httpd.server_address[1]
        for host in ("127.0.0.1", "localhost", "0.0.0.0"):
            assert urllib.request.urlopen("http://%s:%d/" % (host, port), timeout=5).read() == b"ok"
    finally:
        httpd.shutdown()
        httpd.server_close()

    async def echo():
        async def handle(r, w):
            w.write(await r.read(2))
            await w.drain()
            w.close()
        server = await asyncio.start_server(handle, "localhost", 0)
        port = server.sockets[0].getsockname()[1]
        r, w = await asyncio.open_connection("localhost", port)
        w.write(b"hi")
        await w.drain()
        got = await r.read(2)
        w.close()
        server.close()
        await server.wait_closed()
        return got
    assert asyncio.run(echo()) == b"hi"


def test_f_connect_ex_to_a_remote_address_is_refused_at_once():
    c = socket.socket()
    c.settimeout(1.0)                 # bounds the unguarded case; TEST-NET-1 never answers
    try:
        with pytest.raises(OSError):
            c.connect_ex(("192.0.2.1", 80))
    finally:
        c.close()


def test_f_host_name_lookup_never_reaches_the_resolver(monkeypatch):
    import _socket
    reached = []

    def resolver(host, *a, **kw):
        reached.append(host)
        raise socket.gaierror(socket.EAI_NONAME, "stub")
    monkeypatch.setattr(_socket, "getaddrinfo", resolver)
    for call in (lambda: socket.getaddrinfo("example.com", 80),
                 lambda: socket.create_connection(("example.com", 80), timeout=1)):
        with pytest.raises(OSError):
            call()
    assert reached == []


def test_f_gethostbyname_is_guarded_too():
    # socket.gethostbyname / gethostbyname_ex / gethostbyaddr resolve without getaddrinfo
    import _socket
    assert socket.gethostbyname is not _socket.gethostbyname
    assert socket.gethostbyname_ex is not _socket.gethostbyname_ex
