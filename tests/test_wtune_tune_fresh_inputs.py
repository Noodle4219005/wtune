#!/usr/bin/env python3
"""Probe jobs get fresh per-record BP input copies made on the submitting host before sbatch
(WTUNE_BP_FRESH_DIR) -- NFS open stall, wiki 2026-10-03 adios2 gate."""
import os
import re

import pytest

import wtune_tune
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
