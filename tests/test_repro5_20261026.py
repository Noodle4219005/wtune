"""Independent reproductions of five 2026-10-26 defect reports (f1ed209 -> 160e113).
Fakes only at the Slurm / WRF boundary; nothing is submitted."""
import datetime as dt
import os
import re

import pytest

import test_wtune_tune as T
import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
import test_wtune_run_resume as TR
from test_wtune_run import Fakes, _make_case, _run, _with_restart, fake_bp  # noqa: F401
import wtune.tune
import wtune.tune.pipeline
import wtune.tune.prediction
import wtune.tune.report
import wtune.watch
from wtune import predict as wp

FMT = "%Y-%m-%d_%H:%M:%S"


# ---------------------------------------------------------------- defect 1

@pytest.fixture(autouse=True)
def _ascending_sweep(monkeypatch):
    # recorded with the ascending node sweep; the default is now promising-first
    monkeypatch.setenv("WTUNE_SWEEP_ORDER", "ascending")


# ---------------------------------------------------------------- defect 2
def _sweep_with_dead_n1(tmp_path, last_wave_exhausts=False):
    sbatch, reader, job_nodes = T.make_scaling_fakes({1: 1.0, 2: 0.5, 4: 0.3, 8: 0.2})
    states = {}

    def sb(cmd):
        out = sbatch(cmd)
        j = out.split()[-1]
        if job_nodes[j] == 1:
            states[j] = ("TIMEOUT", "01:00:00")
        return out

    def rd(case, ids):
        return reader(case, [j for j in ids if job_nodes[j] != 1])
    clock = T.FakeClock()

    def rd_exhaust(case, ids):
        rows = rd(case, ids)
        clock.t = 10 ** 6
        return rows
    ctx = T._ctx(hotspot_fill=False, sbatch_runner=sb, sacct_runner=T.make_fake_sacct(states),
                 ledger_reader=rd_exhaust if last_wave_exhausts else rd, clock=clock,
                 case_json=dict(T.MATTHEW_LIKE, prep_dir=str(tmp_path / "m")), budget_s=36000.0,
                 out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    return ctx, job_nodes


def test_d2_search_says_speedup_is_relative_to_smallest_measured_n(tmp_path, capsys):
    ctx, _jn = _sweep_with_dead_n1(tmp_path, last_wave_exhausts=True)
    report = wtune.tune.run_node_search(
        ctx, T._profile([{"nodes": n, "legal": True} for n in (1, 2, 4, 8)]),
        objective="time", deadline_s=None, min_efficiency=0.7, k_finalists=2, total_steps=100)
    ns = report["node_selection"]
    assert ns["table"][0]["nodes"] == 2 and ns["table"][0]["speedup"] == pytest.approx(1.0)
    out = capsys.readouterr().out
    assert re.search(r"relative to (the )?(N=)?\s*2\b", out, re.I), out
    text = "\n".join(wtune.tune.report._node_table_lines(ns))
    assert re.search(r"relative to (the )?(N=)?\s*2\b", text, re.I), text


# ---------------------------------------------------------------- defect 3
class Multi(Fakes):
    """Each job writes the restart sets valid(info) (rsl shows their write finished) and
    unfinished(info) (files only), then ends `done`."""

    def __init__(self, valid, unfinished=lambda info: []):
        super().__init__()
        self.valid, self.unfinished = valid, unfinished

    def sbatch(self, cmd):
        seg = os.path.dirname(cmd.split()[-1].strip("'"))
        info = wtune.watch.run_info(wp.parse_namelist(os.path.join(seg, "namelist.input")))
        valid, extra = self.valid(info), self.unfinished(info)
        with open(os.path.join(seg, "rsl.error.0000"), "w") as f:
            f.write(TR._rsl(info["start"], info["end"], set(valid), len(info["domains"])))
        for t in valid + extra:
            for d in range(1, len(info["domains"]) + 1):
                p = os.path.join(seg, "wrfrst_d%02d_%s" % (d, t.strftime(FMT)))
                os.makedirs(p, exist_ok=True)
                open(os.path.join(p, "md.idx"), "w").close()
        self.submitted.append((seg, "done"))
        self.snap_by_dir[seg] = "done"
        return "Submitted batch job %d\n" % (1000 + len(self.submitted))


def _h(info, *hours):
    return [info["start"].replace(hour=0) + dt.timedelta(hours=h) for h in hours]


def _sets(d):
    return sorted({n.split("_", 2)[2] for n in os.listdir(d) if n.startswith("wrfrst_d01_")})


def _t(h):
    return (TR.T0 + dt.timedelta(hours=h)).strftime(FMT)


# ---------------------------------------------------------------- defect 4
def _nl(adaptive, extra_domains="", dx=3000, dt_s=18, minutes=180):
    return ("&time_control\n run_minutes = %d,\n /\n&domains\n max_dom = 1,\n"
            " e_we = 100, e_sn = 100, e_vert = 40,\n dx = %d,\n dy = %d,\n time_step = %d,\n"
            "%s /\n&physics\n radt = 3,\n /\n"
            % (minutes, dx, dx, dt_s,
               (" use_adaptive_time_step = .true.,\n" if adaptive else "") + extra_domains))


def _tune_output(monkeypatch, tmp_path, capsys, text):
    (tmp_path / "namelist.input").write_text(text)
    T._run_cli(monkeypatch, tmp_path, ["-O0"])
    cap = capsys.readouterr()
    return cap.out + cap.err


def test_d4_tune_warns_about_adaptive_time_step(monkeypatch, tmp_path, capsys):
    out = _tune_output(monkeypatch, tmp_path, capsys, _nl(True, " max_time_step = 60,\n"))
    assert "adaptive" in out.lower()
