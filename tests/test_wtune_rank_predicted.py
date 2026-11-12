"""Decision 0055 point jobs: rebuilt-point case entries, ANCHOR/ANCHOREND wrapping, validity
gates, predicted-total ranking, P* choice, CLI/YAML plumbing, default campaign dir, the running
budget, and the bash side (syntax + foreign-load census + strict-restart skip row). Fake Slurm
throughout. The --rank-by predicted mode's own tests are in archive/unused-2026-10/tests/."""
import csv
import datetime as dt
import json
import os
import re
import subprocess
import types
from pathlib import Path

import pytest

from wtune.tools import scc_sweep
from wtune import tune as wt
from test_wtune_tune import _ctx, make_fake_sbatch

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "campaign" / "strict_campaign.sbatch"
HYGIENE = REPO / "campaign" / "node_hygiene.sh"
PM = wt.probe_plan.PointMeasurement
T0 = dt.datetime(2021, 7, 19, 18)
ANCHOR = dict(id="ANCHOR", ranks=288, omp=1, nx=16, ny=18, numtiles=1, dist="block",
              blocktime=0, config_source="anchor")


def _cfg(cid, nx=12, ny=24):
    return dict(ANCHOR, id=cid, nx=nx, ny=ny, config_source="generated")


# ---------------------------------------------------------------------------
# A: run_seconds zeroing
# ---------------------------------------------------------------------------
def test_edit_namelist_zeroes_run_seconds():
    text = ("&time_control\n run_days=0, run_hours=0, run_minutes=0, run_seconds=4830,\n"
            " history_interval=999,\n io_form_history=14,\n/\n"
            "&domains\n time_step=36, e_we=100, e_sn=100, e_vert=45,\n/\n")
    out = scc_sweep.campaign_configs.edit_namelist(text, 4, 4, 1, 21, ranks=16)
    assert "run_seconds=0," in out and "run_minutes=21," in out and "4830" not in out


# ---------------------------------------------------------------------------
# D1: rebuilt case entry
# ---------------------------------------------------------------------------
REAL_NL = """&time_control
 run_days = 0,
 run_hours = 0,
 run_minutes = 0,
 run_seconds = 7200,
 start_year = 2021, 2021,
 start_month = 07, 07,
 start_day = 19, 19,
 start_hour = 18, 18,
 start_minute = 00, 00,
 start_second = 00, 00,
 end_year = 2021, 2021,
 end_month = 07, 07,
 end_day = 19, 19,
 end_hour = 20, 20,
 end_minute = 00, 00,
 end_second = 00, 00,
 interval_seconds = 3600,
 history_interval = 60, 60,
 restart = .false.,
 restart_interval = 1440,
/
&domains
 time_step = 36,
 max_dom = 2,
 e_we = 100, 61,
 e_sn = 100, 61,
 nproc_x = 16,
 nproc_y = 18,
/
"""


def _fake_plan_dir(root, max_dom=2):
    pdir = root / "pt_0"
    real = pdir / "win_0" / "real"
    real.mkdir(parents=True)
    (real / "namelist.input").write_text(REAL_NL)
    for f in ["wrfbdy_d01"] + ["wrfinput_d%02d" % d for d in range(1, max_dom + 1)]:
        (real / f).write_text("x")
    plan = dict(windows=[dict(t="2021-07-19_18:00:00", t_end="2021-07-19_20:00:00")],
                dt_s=36.0, measure_steps=35, spinup_min=20, max_dom=max_dom)
    return pdir, plan


def test_rebuilt_case_entry(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "freezeH2O.dat").write_text("tbl")
    before = sorted(os.listdir(case_dir))
    pdir, plan = _fake_plan_dir(tmp_path / "out")
    entry0 = dict(prep_dir=str(case_dir), bp_dir="", dt=36, steps=35, max_dom=2,
                  input_files=["freezeH2O.dat"], physics={}, case_family="x")
    point = types.SimpleNamespace(at=T0 + dt.timedelta(minutes=20), reason="auto")
    cid, entry = wt.write_rebuilt_case_entry("my_case", entry0, 3, point, plan, str(pdir), 20,
                                             17, ANCHOR)
    assert cid == "mycasep3" and "_" not in cid
    assert entry["prep_dir"] == str(pdir / "win_0" / "real")
    assert entry["steps"] == max(40, 2 * 17) + 2 == 42     # 40-step floor (haitang2017, 2026-10-27)
    assert entry["bp_dir"] == "" and entry0["prep_dir"] == str(case_dir)   # original untouched
    tpl = Path(entry["namelist_template"]).read_text()
    assert re.search(r"run_seconds\s*=\s*0,", tpl)
    assert re.search(r"start_hour\s*=\s*18, 18,", tpl) and re.search(r"start_minute\s*=\s*00", tpl)
    assert os.path.islink(pdir / "win_0" / "real" / "freezeH2O.dat")
    assert sorted(os.listdir(case_dir)) == before


def test_rebuilt_case_entry_missing_real_output_raises(tmp_path):
    pdir, plan = _fake_plan_dir(tmp_path)
    os.remove(pdir / "win_0" / "real" / "wrfinput_d02")
    with pytest.raises(RuntimeError, match="wrfinput_d02"):
        wt.write_rebuilt_case_entry("c", dict(prep_dir="/x"), 0,
                                    types.SimpleNamespace(at=T0, reason="a"), plan, str(pdir),
                                    20, 17, ANCHOR)


def test_prepare_rebuilt_points_refuses_out_inside_case_dir(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    with pytest.raises(RuntimeError, match="inside the case dir"):
        wt.prepare_rebuilt_points(ctx, [], ANCHOR, str(tmp_path), None, {}, str(tmp_path / "x"),
                                  20, 17, 36.0)


def test_prepare_rebuilt_points_writes_one_cases_json(monkeypatch, tmp_path):
    monkeypatch.setenv("WPS_DIR", "/x")
    monkeypatch.setenv("WRF_TREE", "/x")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    seen = {}

    def plan_segments(cdir, pts, cfg, pdir, **k):
        seen.setdefault("real_only", []).append(k.get("real_only"))
        seen.setdefault("cfg", []).append(dict(cfg))
        seen.setdefault("measure_steps", []).append(k.get("measure_steps"))
        _p, plan = _fake_plan_dir(Path(pdir).parent / ("tmp_" + os.path.basename(pdir)))
        # move the fake real dir to pdir
        os.rename(str(_p), pdir)
        return plan

    monkeypatch.setattr(wt.probe_plan, "plan_rebuild_segments", plan_segments)
    monkeypatch.setattr(wt.coldstart, "resolve_and_fetch_forcing", lambda *a, **k: (0, {}))
    monkeypatch.setattr(wt.budget, "rebuild_segment_timeout_min", lambda *a, **k: 30)
    monkeypatch.setattr(wt.coldstart, "stage_plan", lambda pdir, **k: pdir + "/run.sbatch")
    # real_only prep needs one node (real.exe output is decomposition-independent)
    monkeypatch.setattr(wt.sweep, "build_anchor", lambda cj, cores, nodes, hw: dict(
        ANCHOR, ranks=72, nx=8, ny=9) if nodes == 1 else None)
    sb, calls, _ = make_fake_sbatch()
    ctx = _ctx(out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"), sbatch_runner=sb,
               sbatch_script=str(SBATCH))
    pts = [types.SimpleNamespace(at=T0 + dt.timedelta(hours=h, minutes=20), reason="auto")
           for h in (0, 7)]
    out = wt.prepare_rebuilt_points(ctx, pts, ANCHOR, str(case_dir), None,
                                    dict(prep_dir=str(case_dir), input_files=[]),
                                    str(tmp_path / "rebuilt"), 20, 17, 36.0)
    assert seen["real_only"] == [True, True]
    assert [(c["nodes"], c["ranks"], c["nx"], c["ny"]) for c in seen["cfg"]] == [(1, 72, 8, 9)] * 2
    assert all(m >= wt.rebuilt_steps(17) for m in seen["measure_steps"])
    assert [o["case_id"] for o in out] == ["matthewp0", "matthewp1"]
    cases = json.load(open(tmp_path / "rebuilt" / "rebuilt_cases.json"))["cases"]
    assert set(cases) == {"matthewp0", "matthewp1"}
    assert all("CAMPAIGN=%s" % SBATCH.parent in c for c in calls)
    assert os.listdir(case_dir) == []


# ---------------------------------------------------------------------------
# D3: gates via measure_candidates_at_points
# ---------------------------------------------------------------------------
def _rp(tmp_path, index=0):
    return dict(index=index, point=types.SimpleNamespace(at=T0 + dt.timedelta(minutes=21),
                                                         reason="auto"),
                case_id="matthewp%d" % index, T=T0, cases_json=str(tmp_path / "rc.json"),
                steps=36, spinup_min=21, stepra=17, dt=36.0, plan_dir="", job_id="1")


def _write_run(ledger_dir, case_id, cid, job, s, first_time="2021-07-19_18:21:36"):
    d = Path(ledger_dir) / ("%s_%s_r1_%s" % (case_id, cid, job))
    d.mkdir(parents=True, exist_ok=True)
    (d / "steps.txt").write_text("\n".join(str(s * (3.0 if i % 17 == 1 else 1.0))
                                           for i in range(1, 37)) + "\n")
    (d / "rsl.error.0000").write_text("Timing for main: time %s on domain   1:  1.0 elapsed "
                                      "seconds\n" % first_time)


def _run_measure(tmp_path, s_end=1.0, notes_by_cid=None, first_time=None):
    sb, calls, paths = make_fake_sbatch(start=500)
    ledger = tmp_path / "ledger"
    rows = []
    notes_by_cid = notes_by_cid or {}

    def reader(case, job_ids):
        return rows

    ctx = _ctx(out_dir=str(tmp_path / "out"), ledger_dir=str(ledger), sbatch_runner=sb,
               ledger_reader=reader)
    job = "501"
    rp = _rp(tmp_path)
    for cid, s in (("ANCHOR", 1.0), ("G1", 0.8), ("ANCHOREND", s_end)):
        _write_run(ledger, rp["case_id"], cid, job, s,
                   **({"first_time": first_time} if first_time else {}))
        notes = notes_by_cid.get(cid, "config=%s node_dirty=none restart_target=20 "
                                      "restart_used=1 foreign_load=0" % cid)
        rows.append(dict(run_id="%s_%s_r1_%s" % (rp["case_id"], cid, job),
                         s_per_step_median="" if "SKIPPED" in notes else str(s), notes=notes))
    return wt.measure_candidates_at_points(ctx, [_cfg("G1")], [rp], ANCHOR), calls


def test_gate_anchor_drift_raises(tmp_path):
    with pytest.raises(RuntimeError, match="anchor drift"):
        _run_measure(tmp_path, s_end=1.015)


def test_gate_restart_used_0_raises(tmp_path):
    with pytest.raises(RuntimeError, match="restart_used=0"):
        _run_measure(tmp_path, notes_by_cid={
            "G1": "status=SKIPPED reason=spinup_failed config=G1 restart_target=20 "
                  "restart_used=0 node_dirty=none foreign_load=0"})


def test_gate_foreign_load_raises(tmp_path):
    with pytest.raises(RuntimeError, match="foreign_load=1"):
        _run_measure(tmp_path, notes_by_cid={
            "ANCHOR": "config=ANCHOR node_dirty=none restart_target=20 restart_used=1 "
                      "foreign_load=1 foreign=work5:77:root:node:91.0"})


def test_gate_t0_start_detected_from_rsl(tmp_path):
    with pytest.raises(RuntimeError, match="did not resume"):
        _run_measure(tmp_path, first_time="2021-07-19_18:00:36")


def test_predicted_spinup_min():
    assert wt.predicted_spinup_min(36) == 21
    assert [wt.predicted_spinup_min(d) for d in (15, 10, 25)] == [20, 20, 20]
    assert wt.predicted_spinup_min(72) == 24            # 20..23 min are not multiples of 72 s
    with pytest.raises(SystemExit, match="time_step=1001s"):
        wt.predicted_spinup_min(7 * 11 * 13)       # 1001 s: no m in [20, 60] divides


# ---------------------------------------------------------------------------
# D4 / P*
# ---------------------------------------------------------------------------
def test_rank_by_predicted_total_orders_by_total():
    def pms(scale):
        return [PM(at=T0 + dt.timedelta(hours=h), reason="a", source="rebuild",
                   s_normal=scale * (1.0 + 0.1 * h), n_rad=2, r=3.0) for h in (1, 10)]
    m = {"ANCHOR": pms(1.0), "G1": pms(0.8), "G2": pms(1.3), "ANCHOREND": pms(0.1)}
    ranked = wt.rank_by_predicted_total(m, T0, T0 + dt.timedelta(hours=24), 36.0, 17, {}, None,
                                        None, None, {"ANCHOR": 288, "G1": 288, "G2": 288}, 4)
    assert [e["cid"] for e in ranked] == ["G1", "ANCHOR", "G2"]
    assert ranked[0]["pred_low_s"] < ranked[0]["pred_total_s"] < ranked[0]["pred_high_s"]


def test_build_sbatch_command_extra_export_refuses_comma():
    with pytest.raises(ValueError, match="sbatch --export"):
        scc_sweep.build_sbatch_command("c", 1, 72, "01:00:00", 600, 1, 30, "/x", "/o",
                                       extra_export=[("SPINUP_MINUTES", "20,40")])


# ---------------------------------------------------------------------------
# bash side
# ---------------------------------------------------------------------------
def test_bash_syntax():
    for f in (SBATCH, HYGIENE):
        subprocess.run(["bash", "-n", str(f)], check=True)


def test_census_flags_foreign_process(tmp_path):
    uid = os.getuid()
    fake = tmp_path / "ps"
    fake.write_text("#!/bin/sh\ncat <<'EOF'\n"
                    "    2     0     0 root      0.0 99999 kthreadd\n"
                    "   55     2     0 root     99.0 99999 kworker/0:1\n"
                    " 7001     1     0 root     91.0  3600 node\n"
                    " 7002     1     0 root      0.3 99999 sshd\n"
                    " 7004     1     0 root    100.0     0 check-new-release\n"
                    " 7005     1     0 root     50.0    29 young\n"
                    " 7003     1 %d me        99.0 99999 wrf.exe\n"
                    "EOF\n" % uid)
    fake.chmod(0o755)
    env = dict(os.environ, PATH="%s:%s" % (tmp_path, os.environ["PATH"]))
    out = subprocess.run(["bash", str(HYGIENE), "census"], capture_output=True, text=True,
                         env=env, check=True).stdout
    lines = [l for l in out.splitlines() if l.startswith("FOREIGN_LOAD ")]
    assert len(lines) == 1, out                        # young/new processes do not fire
    assert "pid=7001 user=root comm=node pcpu=91.0" in lines[0]


def test_strict_restart_skip_row(tmp_path):
    text = SBATCH.read_text()
    fn = re.search(r"(write_restart_skip_row \(\) \{.*?\n\}\n)", text, re.S).group(1)
    ledger = tmp_path / "ledger.csv"
    script = ('LEDGER=%s J=77 CASE=conus25p0 HARDWARE_ID=scc NODE_DIRTY_NOTE=node_dirty=none '
              'FOREIGN_LOAD_NOTE=foreign_load=0\n' % ledger + fn +
              'write_restart_skip_row conus25p0_G1_r1_77 G1 1 spinup_failed 20\n')
    subprocess.run(["bash", "-c", script], check=True, capture_output=True)
    row = list(csv.reader(open(ledger)))[0]
    assert row[0] == "conus25p0_G1_r1_77" and row[35] == ""
    assert "status=SKIPPED reason=spinup_failed config=G1 restart_target=20 restart_used=0" \
        in row[-1]
    assert len(row) == 48


def test_strict_restart_defaults_do_not_change_t0_notes():
    text = SBATCH.read_text()
    assert 'SPINUP_TIMEOUT_S=${SPINUP_TIMEOUT_S:-$TIMEOUT_S}' in text
    assert 'RANKING_STRICT_RESTART=${RANKING_STRICT_RESTART:-0}' in text
    # restart/foreign notes only when RANKING_ACTIVE (spinup+timing) or a foreign load fired
    assert re.search(r'if \[ "\$RANKING_ACTIVE" = "1" \]; then\n\s+NOTES="\$NOTES restart_target=',
                     text)


# ---------------------------------------------------------------------------
# running-time budget, hard wall cap (conus25-pred1 postmortem)
# ---------------------------------------------------------------------------
from test_wtune_tune import FakeClock, make_fake_scancel   # noqa: E402


def timeline_sacct(clock, timelines):
    """sacct fake driven by the virtual clock: timelines {job: [(t_until, State), ...]} -- the
    first entry with clock.t < t_until wins, COMPLETED after the last; a scancel'd job (in
    `cancelled`) reports CANCELLED."""
    cancelled = set()

    def runner(cmd):
        ids = cmd[cmd.index("-j") + 1].split(",")
        out = []
        for j in ids:
            state = "COMPLETED"
            if j in cancelled:
                state = "CANCELLED"
            else:
                for t_until, st in timelines.get(j, []):
                    if clock.t < t_until:
                        state = st
                        break
            out.append("%s|%s|00:00:10" % (j, state))
        return "\n".join(out)

    return runner, cancelled


def _running_ctx(tmp_path, clock, sacct, budget_s, max_wall_s=None, **kw):
    scancel, scalls = make_fake_scancel()
    ctx = _ctx(out_dir=str(tmp_path), clock=clock, sacct_runner=sacct, scancel_runner=scancel,
               budget_s=budget_s, poll_interval=10.0, **kw)
    ctx.budget_mode = "running"
    ctx.max_wall_s = max_wall_s
    return ctx, scalls


def test_running_budget_does_not_charge_queue_time(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {"1": [(600, "PENDING"), (900, "RUNNING")]})
    ctx, scalls = _running_ctx(tmp_path, clock, sacct, budget_s=400, max_wall_s=3000)
    states = wt.wait_for_jobs(ctx, ["1"], "round0")
    assert states["1"][0] == "COMPLETED" and scalls == []
    assert ctx.queue_only_s == pytest.approx(600)
    assert ctx.elapsed() == pytest.approx(300) and ctx.wall_elapsed() == pytest.approx(900)


def test_rad_factor_conus25_like():
    # 82 steps, stepra 40 -> 3 radiation steps: ~1.107, not the old blanket 1.5
    assert wt.rad_factor(82, 40) == pytest.approx(1 + 3 * 2.93 / 82)
    assert wt.rad_factor(82, None) == 1.0


# ---------------------------------------------------------------------------
# point jobs are exempt from the budget (conus25-pred3)
# ---------------------------------------------------------------------------
def _conc_ctx(tmp_path, clock, sacct, budget_s=5000.0, max_wall_s=10 ** 6):
    ctx, scalls = _running_ctx(tmp_path, clock, sacct, budget_s=budget_s, max_wall_s=max_wall_s,
                               nodes=4, max_nodes=12)
    ctx.startup_s = 120.0
    return ctx, scalls


def test_stage2_wall_cap_cancel_still_raises(tmp_path):
    clock = FakeClock()
    sacct, cancelled = timeline_sacct(clock, {"501": [(10 ** 9, "RUNNING")]})
    ctx, scalls = _conc_ctx(tmp_path, clock, sacct, budget_s=600, max_wall_s=1000)
    sb, _calls, _p = make_fake_sbatch(start=500)
    ctx.sbatch_runner = sb

    def scancel(ids):
        scalls.append(list(ids))
        cancelled.update(ids)

    ctx.scancel_runner = scancel
    with pytest.raises(RuntimeError, match="not COMPLETED"):
        wt.measure_candidates_at_points(ctx, [_cfg("G1")], [_rp(tmp_path)], ANCHOR)
    assert scalls == [["501"]]
