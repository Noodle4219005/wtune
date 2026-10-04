#!/usr/bin/env python3
"""Probe jobs get fresh per-record BP input copies made on the submitting host before sbatch
(WTUNE_BP_FRESH_DIR) -- NFS open stall, wiki 2026-10-03 adios2 gate."""
import os
import re

import pytest

from wtune_tune import jobs, prediction
from wtune_tune._deps import sweep
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


def test_submit_configs_job_exports_the_fresh_dir(monkeypatch):
    seen = []

    def fake_export(ctx, case_id, job_out_dir, runs, cases_json=None):
        seen.append((case_id, os.path.basename(job_out_dir), runs, cases_json))
        return [("WTUNE_BP_FRESH_DIR", os.path.join(job_out_dir, "bp_fresh"))]
    monkeypatch.setattr(jobs, "fresh_input_export", fake_export)
    runner, calls, _ = T.make_fake_sbatch()
    ctx = T._ctx(sbatch_runner=runner)
    cfgs = [dict(id="C%d" % i, ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block",
                 blocktime=0) for i in range(3)]
    jobs.submit_configs_job(ctx, cfgs, 2, "fresh_job", s_est=0.5)
    assert seen == [("matthew", "fresh_job", 3 * 2 + 2, None)]
    m = re.search(r"--export=(\S+)", calls[-1])
    assert ",WTUNE_BP_FRESH_DIR=%s,CONFIG_FILE=" % os.path.join(ctx.out_dir, "fresh_job",
                                                                 "bp_fresh") in m.group(1)


def test_submit_configs_job_without_fresh_dir_is_unchanged(monkeypatch):
    monkeypatch.setattr(jobs, "fresh_input_export", lambda *a, **k: [])
    runner, calls, _ = T.make_fake_sbatch()
    ctx = T._ctx(sbatch_runner=runner)
    jobs.submit_configs_job(ctx, [dict(id="C1", ranks=72, omp=1, nx=8, ny=9, numtiles=1,
                                       dist="block", blocktime=0)], 1, "plain", s_est=0.5)
    assert "WTUNE_BP_FRESH_DIR" not in calls[-1]


def test_trajectory_command_carries_extra_export():
    cmd = prediction.build_ranking_sbatch_command(
        "conus25", 4, 72, "01:00:00", 900, 2, 50, "/x/c.json", "/x", "60:120",
        extra_export=[("WTUNE_BP_FRESH_DIR", "/x/bp_fresh")])
    assert re.search(r"SPINUP_MINUTES=60:120(,CAMPAIGN=\S+)?,WTUNE_BP_FRESH_DIR=/x/bp_fresh ", cmd)
    plain = prediction.build_ranking_sbatch_command(
        "conus25", 4, 72, "01:00:00", 900, 2, 50, "/x/c.json", "/x", "60:120")
    assert "WTUNE_BP_FRESH_DIR" not in plain


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


def test_cache_current_submits_no_conversion_job(monkeypatch, tmp_path, capsys):
    cache = _Cache(stale=False)
    _wire(monkeypatch, cache)
    inner, calls, _ = T.make_fake_sbatch()
    ctx = T._ctx(sbatch_runner=cache.runner(inner))
    jobs.ensure_bp_inputs(ctx, [("matthew", None)], str(tmp_path / "bp_convert"))
    assert cache.events == [] and capsys.readouterr().out == "bp inputs: cache current\n"


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


def test_failed_conversion_job_state_is_an_error(monkeypatch, tmp_path):
    cache = _Cache()
    _wire(monkeypatch, cache)
    inner, _calls, _ = T.make_fake_sbatch()
    ctx = T._ctx(sbatch_runner=cache.runner(inner),
                 sacct_runner=T.make_fake_sacct({"777": ("FAILED", "00:00:04")}))
    with pytest.raises(RuntimeError, match="job 777 ended FAILED"):
        jobs.ensure_bp_inputs(ctx, [("matthew", None)], str(tmp_path / "c"))


def test_rebuilt_points_convert_in_one_job_with_their_case_file(monkeypatch, tmp_path):
    cache = _Cache()
    _wire(monkeypatch, cache)
    inner, _calls, _ = T.make_fake_sbatch()
    seen = []

    def runner(cmd):
        seen.append(cmd)
        return cache.runner(inner)(cmd)
    ctx = T._ctx(sbatch_runner=runner)
    jobs.ensure_bp_inputs(ctx, [("m__rb0", "/o/rebuilt_cases.json"),
                                ("m__rb1", "/o/rebuilt_cases.json")], str(tmp_path / "c"))
    assert len(seen) == 1 and ",CASES=m__rb0:m__rb1,CASES_JSON=/o/rebuilt_cases.json " in seen[0]
