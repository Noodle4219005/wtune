#!/usr/bin/env python3
"""Probe jobs get fresh per-record BP input copies made on the submitting host before sbatch
(WTUNE_BP_FRESH_DIR) -- NFS open stall, wiki 2026-10-03 adios2 gate."""

import pytest

from wtune.tune import jobs
from wtune.tune._deps import sweep
import test_wtune_tune as T


def test_fresh_input_export_only_for_real_submission(monkeypatch, tmp_path, capsys):
    calls = []

    def fake(case_id, dest, runs, cases_json=None):
        calls.append((case_id, dest, runs, cases_json))
        return dest
    monkeypatch.setattr(sweep.campaign_configs, "fresh_probe_inputs", fake)
    assert jobs.fresh_input_export(T._ctx(), "matthew", str(tmp_path), 5) == []   # fake runner
    assert calls == []
    ctx = T._ctx(sbatch_runner=sweep.default_sbatch_runner)
    out = jobs.fresh_input_export(ctx, "pt_case", str(tmp_path), 5, "/c/cases.json")
    assert out == [("WTUNE_BP_FRESH_DIR", str(tmp_path / "bp_fresh"))]
    assert calls == [("pt_case", str(tmp_path / "bp_fresh"), 5, "/c/cases.json")]

    def boom(*a, **k):
        raise OSError("no such namelist")
    monkeypatch.setattr(sweep.campaign_configs, "fresh_probe_inputs", boom)
    assert jobs.fresh_input_export(ctx, "pt_case", str(tmp_path), 5) == []
    assert "WARNING: fresh BP input copy for pt_case failed (OSError" in capsys.readouterr().err


# ---- the BP cache is made current by ONE nc2bp job before the first probes (scc 2884/2885) ----
class _Cache:
    """Fake bp_cache_status: stale until the nc2bp job ran (or `fails`)."""

    def __init__(self, stale=True, fails=False):
        self.stale, self.fails, self.events = stale, fails, []

    def status(self, case_id, cases_json=None, site="scc", bp=None):
        return ("stale", "/c: wrfbdy_d01") if self.stale else ("current", "/c")

    def runner(self, inner):
        def run(cmd):
            if "nc2bp.sbatch" in cmd:
                self.events.append("nc2bp")
                self.stale = self.fails
                return "Submitted batch job 777\n"
            self.events.append("probe")
            return inner(cmd)
        return run


@pytest.fixture(autouse=True)
def _fresh_bp_memo():
    jobs._BP_READY.clear()        # conversion is memoized per process


def _wire(monkeypatch, cache):
    monkeypatch.setattr(jobs, "_submits_for_real", lambda ctx: True)
    monkeypatch.setattr(sweep.campaign_configs, "bp_cache_status", cache.status)

    def fresh(case_id, dest, runs, cases_json=None):
        cache.events.append("copy")
        return dest
    monkeypatch.setattr(sweep.campaign_configs, "fresh_probe_inputs", fresh)


def _cfgs(n=1):
    return [dict(id="C%d" % i, ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block",
                 blocktime=0) for i in range(n)]


def test_first_wave_waits_for_the_conversion_job_then_copies(monkeypatch, tmp_path, capsys):
    cache = _Cache()
    _wire(monkeypatch, cache)
    inner, calls, _ = T.make_fake_sbatch()
    ctx = T._ctx(sbatch_runner=cache.runner(inner))
    jobs.ensure_bp_inputs(ctx, [("matthew", None)], str(tmp_path / "bp_convert"))
    jobs.submit_configs_job(ctx, _cfgs(), 1, "w1", s_est=0.5)
    assert cache.events == ["nc2bp", "copy", "probe"]
    assert "bp inputs: converted matthew in 0 s (job 777)" in capsys.readouterr().out
    assert "777" in ctx.owned_jobs


def test_nc2bp_command_excludes_head_and_names_the_cases(monkeypatch, tmp_path):
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "gn07")
    cmd = jobs.nc2bp_command(T._ctx(), ["pt_0", "pt_1"], "/o/rebuilt_cases.json", "/o/conv")
    assert "--exclude=head,gn07 " in cmd and cmd.endswith("/nc2bp.sbatch")
    assert ",CHECK=0," in cmd and ",CASES=pt_0:pt_1,CASES_JSON=/o/rebuilt_cases.json " in cmd


def test_conversion_failure_stops_before_any_probe(monkeypatch, tmp_path):
    cache = _Cache(fails=True)
    _wire(monkeypatch, cache)
    real_fakes = T.make_scaling_fakes

    def fakes(*a, **k):
        sbatch, reader, job_nodes = real_fakes(*a, **k)
        return cache.runner(sbatch), reader, job_nodes
    monkeypatch.setattr(T, "make_scaling_fakes", fakes)
    with pytest.raises(SystemExit, match=r"ERROR: BP input conversion job 777 ended COMPLETED; "
                                         r"cache not current for matthew -- no probes submitted"):
        T._run_cli(monkeypatch, tmp_path, ["--minutes", "30", "--nodes", "1",
                                           "--no-node-preflight"])
    assert cache.events == ["nc2bp"]                                 # no probe submitted
