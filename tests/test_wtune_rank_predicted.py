"""Decision 0055 (--rank-by predicted): rebuilt-point case entries, ANCHOR/ANCHOREND wrapping,
validity gates, predicted-total ranking, P* choice, CLI/YAML plumbing, default campaign dir, and
the bash side (syntax + foreign-load census + strict-restart skip row). Fake Slurm throughout."""
import csv
import datetime as dt
import json
import os
import re
import subprocess
import types
from pathlib import Path

import pytest

import scc_sweep
import wtune_config
import wtune_tune as wt
from test_wtune_tune import _ctx, make_fake_sacct, make_fake_sbatch

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "campaign" / "strict_campaign.sbatch"
HYGIENE = REPO / "campaign" / "node_hygiene.sh"
PM = wt.wtune_probe_plan.PointMeasurement
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
    assert entry["steps"] == max(30, 2 * 17) + 2 == 36
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


def test_measure_steps_cover_candidate_minutes():
    steps, step_dt = wt.rebuilt_steps(17), 36.0
    ms = wt.rebuilt_measure_steps(steps, step_dt)
    run_min = -(-steps * 36 // 60)
    assert ms >= steps and ms * step_dt >= run_min * 60


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

    monkeypatch.setattr(wt.wtune_probe_plan, "plan_rebuild_segments", plan_segments)
    monkeypatch.setattr(wt.coldstart, "resolve_and_fetch_forcing", lambda *a, **k: (0, {}))
    monkeypatch.setattr(wt, "rebuild_segment_timeout_min", lambda *a, **k: 30)
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
# D2: chunks -- ANCHOR + ANCHOREND in every chunk (predicted), unchanged (t0)
# ---------------------------------------------------------------------------
def _chunk_configs(paths):
    return [[c["id"] for c in json.load(open(p.split(",")[0]))["configs"]]
            for p in paths.values()]


def test_t0_mode_chunks_and_command_unchanged(tmp_path):
    sb, calls, paths = make_fake_sbatch()
    ctx = _ctx(out_dir=str(tmp_path), sbatch_runner=sb)
    configs = [dict(ANCHOR)] + [_cfg("G%d" % i) for i in range(1, 5)]
    wt._submit_round0_wave(ctx, configs, 2, 0.01, "round0")
    ids = _chunk_configs(paths)
    assert ids == [["ANCHOR", "G2", "G4"], ["G1", "G3"]]
    assert all("RANKING_START" not in c and "ANCHOREND" not in json.dumps(ids) for c in calls)
    assert all("CASE=matthew," in c for c in calls)


def test_predicted_mode_every_chunk_has_anchor_and_anchor_end(tmp_path):
    sb, calls, paths = make_fake_sbatch()
    ctx = _ctx(out_dir=str(tmp_path), sbatch_runner=sb)
    ctx.ranking = wt.ranking_context("matthewp1", "/r/rebuilt_cases.json", 20, 36, ANCHOR)
    ctx.ranking["dt"] = 36.0
    configs = [dict(ANCHOR)] + [_cfg("G%d" % i) for i in range(1, 5)]
    wt._submit_round0_wave(ctx, configs, 2, 0.01, "round0")
    ids = _chunk_configs(paths)
    # predicted mode deals CANDIDATES only; every slot then gets its own ANCHOR/ANCHOREND
    assert ids == [["ANCHOR", "G1", "G3", "ANCHOREND"], ["ANCHOR", "G2", "G4", "ANCHOREND"]]
    for c in calls:
        assert "CASE=matthewp1," in c and "STEPS=36," in c
        for kv in ("CASES_JSON=/r/rebuilt_cases.json", "RANKING_START=spinup",
                   "SPINUP_MINUTES=20", "RANKING_STRICT_RESTART=1", "RANKING_RESTART_IO_FORM=14"):
            assert kv in c.split("--export=")[1].split()[0].split(",")
        assert re.search(r"SPINUP_TIMEOUT_S=\d+", c)
    end = json.load(open(list(paths.values())[0].split(",")[0]))["configs"][-1]
    assert end["config_source"] == "anchor_end" and end["nx"] == ANCHOR["nx"]


def test_wrap_with_anchors_is_idempotent():
    once = wt.wrap_with_anchors([_cfg("G1")], ANCHOR)
    assert [c["id"] for c in wt.wrap_with_anchors(once, ANCHOR)] == ["ANCHOR", "G1", "ANCHOREND"]


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


def test_gates_clean_pass(tmp_path):
    (meas, gates, jobs), calls = _run_measure(tmp_path)
    assert set(meas) == {"ANCHOR", "G1"} and gates[0]["ok"] and jobs == ["501"]
    assert meas["G1"][0].s_normal == pytest.approx(0.8)
    assert meas["G1"][0].at == T0 + dt.timedelta(minutes=21, seconds=36 * 36 / 2)
    c = calls[0]
    assert "REPS=1," in c and "RANKING_STRICT_RESTART=1" in c and "CASE=matthewp0," in c
    assert "TIMEOUT_S=1800" not in c


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


def test_gate_first_step_is_exact(tmp_path):
    # spin-up 21 min at dt 36 -> resumed first step g0 = 35+1 = 36 (18:21:36); 18:22:12 is 37
    with pytest.raises(RuntimeError, match="expected 36"):
        _run_measure(tmp_path, first_time="2021-07-19_18:22:12")


def test_predicted_spinup_min():
    assert wt.predicted_spinup_min(36) == 21
    assert [wt.predicted_spinup_min(d) for d in (15, 10, 25)] == [20, 20, 20]
    assert wt.predicted_spinup_min(72) == 24            # 20..23 min are not multiples of 72 s
    with pytest.raises(SystemExit, match="time_step=1001s"):
        wt.predicted_spinup_min(7 * 11 * 13)       # 1001 s: no m in [20, 60] divides


# ---------------------------------------------------------------------------
# stage 1: anchor normalization and exclusion
# ---------------------------------------------------------------------------
def _stage1_rows(job, anchor, anchor_end, cands, bad=None):
    bad = bad or {}
    rows = []
    for cid, s in [("ANCHOR", anchor)] + cands + [("ANCHOREND", anchor_end)]:
        notes = bad.get(cid, "config=%s node_dirty=none restart_target=21 restart_used=1 "
                             "foreign_load=0" % cid)
        rows.append(dict(run_id="matthewp1_%s_r1_%s" % (cid, job), s_per_step_median=str(s),
                         notes=notes))
    return rows


def test_stage1_normalizes_by_own_job_anchor(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.ranking = wt.ranking_context("matthewp1", "/r.json", 21, 36, ANCHOR)
    # job 11 on slow nodes (anchor 2.0), job 12 on fast nodes (anchor 1.0): raw medians say G1
    # (1.1) beats G2 (1.6), normalized says G2 (0.8x) beats G1 (1.1x).
    rows = (_stage1_rows("11", 2.0, 2.0, [("G2", 1.6)])
            + _stage1_rows("12", 1.0, 1.0, [("G1", 1.1)]))
    wt._record_stage1_gates(ctx, ["11", "12"], rows, "round0")
    res = wt.aggregate_by_cid(rows, {"G1": _cfg("G1"), "G2": _cfg("G2"), "ANCHOR": ANCHOR})
    ranked, excluded = wt.stage1_normalized_ranking(res, ctx.stage1_jobs)
    assert [e["cid"] for e in ranked] == ["G2", "G1"] and not excluded
    assert ranked[0]["norm_key"] == pytest.approx(0.8) and ranked[0]["median"] == 1.6


def test_stage1_anchor_end_missing_uses_anchor_alone(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.ranking = wt.ranking_context("matthewp1", "/r.json", 21, 36, ANCHOR)
    rows = [r for r in _stage1_rows("11", 2.0, 9.9, [("G1", 1.0)]) if "ANCHOREND" not in r["run_id"]]
    wt._record_stage1_gates(ctx, ["11"], rows, "round0")
    assert ctx.stage1_jobs["11"]["s_anchor_job"] == 2.0


def test_stage1_mean_of_anchor_and_anchor_end_and_drift_only_warns(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.ranking = wt.ranking_context("matthewp1", "/r.json", 21, 36, ANCHOR)
    rows = _stage1_rows("11", 1.0, 1.1, [("G1", 1.05)])        # 10% drift
    wt._record_stage1_gates(ctx, ["11"], rows, "round0")
    info = ctx.stage1_jobs["11"]
    assert info["s_anchor_job"] == pytest.approx(1.05) and not info["excluded"]
    res = wt.aggregate_by_cid(rows, {"G1": _cfg("G1")})
    ranked, _ex = wt.stage1_normalized_ranking(res, ctx.stage1_jobs)
    assert ranked[0]["norm_key"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_notes,why", [
    ("config=G1 node_dirty=none restart_target=21 restart_used=1 foreign_load=1 "
     "foreign=work5:77:root:node:91.0", "foreign_load=1"),
    ("status=SKIPPED reason=physics_mismatch config=G1 restart_target=21 restart_used=0 "
     "node_dirty=none foreign_load=0", "restart_used=0"),
])
def test_stage1_job_excluded(tmp_path, bad_notes, why):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.ranking = wt.ranking_context("matthewp1", "/r.json", 21, 36, ANCHOR)
    rows = (_stage1_rows("11", 1.0, 1.0, [("G1", 0.5), ("G3", 0.6)], bad={"G1": bad_notes})
            + _stage1_rows("12", 1.0, 1.0, [("G2", 0.9)]))
    wt._record_stage1_gates(ctx, ["11", "12"], rows, "round0")
    assert ctx.stage1_jobs["11"]["excluded"] and why in ctx.stage1_jobs["11"]["reason"]
    res = wt.aggregate_by_cid(rows, {c: _cfg(c) for c in ("G1", "G2", "G3")})
    ranked, excluded = wt.stage1_normalized_ranking(res, ctx.stage1_jobs)
    assert [e["cid"] for e in ranked] == ["G2"]
    assert {x["cid"] for x in excluded} == {"G1", "G3"}
    assert any(g.get("excluded") for g in ctx.ranking_gates)


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


def test_choose_primary_point(monkeypatch):
    pts = [types.SimpleNamespace(at=T0 + dt.timedelta(hours=h)) for h in (0, 6, 12, 18)]
    fr = {0: 0.2, 6: 0.9, 12: 0.9, 18: 0.1}
    monkeypatch.setattr(wt.wtune_predict, "sunlit_fraction",
                        lambda sp, at: fr[int((at - T0).total_seconds() // 3600)])
    idx, p, frac = wt.choose_primary_point(pts, [(0.0, 0.0)])
    assert idx == 1 and frac == 0.9                     # tie 6h/12h -> earliest
    assert wt.choose_primary_point(pts, None)[0] == 0   # no geometry -> earliest


# ---------------------------------------------------------------------------
# D6: CLI / YAML, default campaign dir
# ---------------------------------------------------------------------------
def test_cli_rank_by_flags():
    a = wt.build_arg_parser().parse_args(["conus25", "--rank-by", "predicted",
                                          "--restart-io-form", "2"])
    assert a.rank_by == "predicted" and a.restart_io_form == 2
    a = wt.build_arg_parser().parse_args(["conus25"])
    assert a.rank_by is None and a.restart_io_form is None
    with pytest.raises(SystemExit):
        wt.build_arg_parser().parse_args(["conus25", "--rank-by", "bogus"])
    with pytest.raises(SystemExit):
        wt.build_arg_parser().parse_args(["conus25", "--restart-io-form", "11"])


def test_yaml_rank_by_keys():
    cfg, errs = wtune_config.resolve_config({})
    assert cfg["probe"]["rank_by"] == "t0" and cfg["probe"]["restart_io_form"] == 14 and not errs
    _cfg2, errs = wtune_config.resolve_config({"probe": {"rank_by": "bogus",
                                                         "restart_io_form": 11}})
    assert any("rank_by" in e for e in errs) and any("restart_io_form" in e for e in errs)


def test_default_sbatch_script_is_this_trees_campaign():
    assert Path(scc_sweep.DEFAULT_SBATCH_SCRIPT) == SBATCH
    assert wt.build_arg_parser().parse_args(["x"]).sbatch_script == str(SBATCH)
    assert scc_sweep.campaign_dir_for(scc_sweep.DEFAULT_SBATCH_SCRIPT) == str(SBATCH.parent)
    assert scc_sweep.campaign_dir_for("/nonexistent/strict_campaign.sbatch") is None
    cmd = wt.build_ranking_sbatch_command("conus25", 4, 72, "01:00:00", 900, 1, 50, "/x/c.json",
                                          "/x", "60", sbatch_script=str(SBATCH))
    assert ",CAMPAIGN=%s" % SBATCH.parent in cmd


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


def test_campaign_sbatch_commands_honour_extra_exclude(monkeypatch):
    import scc_sweep as sweep
    import wtune_tune as wt
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    assert sweep.exclude_arg() == "--exclude=head"
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "work5, work7")
    cmd = sweep.build_sbatch_command("conus25", 4, 72, "01:00:00", 600, 1, 30, "/x/c.json", "/x")
    assert "--exclude=head,work5,work7 " in cmd
    cmd = wt.build_ranking_sbatch_command("conus25", 4, 72, "01:00:00", 600, 1, 30, "/x/c.json",
                                          "/x", "20")
    assert "--exclude=head,work5,work7 " in cmd


# ---------------------------------------------------------------------------
# running-time budget, hard wall cap, stage-1 cutoff, stage-2 reserve (conus25-pred1 postmortem)
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


def test_wall_mode_unchanged_charges_queue_time(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {"1": [(600, "PENDING"), (900, "RUNNING")]})
    ctx, scalls = _running_ctx(tmp_path, clock, sacct, budget_s=400)
    ctx.budget_mode = "wall"
    wt.wait_for_jobs(ctx, ["1"], "round0")
    assert scalls == [["1"]] and ctx.elapsed() == pytest.approx(400)


def test_hard_wall_cap_cancels_pending(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {"1": [(10 ** 9, "PENDING")]})
    ctx, scalls = _running_ctx(tmp_path, clock, sacct, budget_s=400, max_wall_s=100)
    wt.wait_for_jobs(ctx, ["1"], "stage2")
    assert scalls == [["1"]] and clock.t == pytest.approx(100) and ctx.elapsed() == 0


def test_running_budget_exhausted_while_running_cancels(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {"1": [(100, "PENDING"), (10 ** 9, "RUNNING")]})
    ctx, scalls = _running_ctx(tmp_path, clock, sacct, budget_s=200, max_wall_s=10 ** 6)
    wt.wait_for_jobs(ctx, ["1"], "stage2")
    assert scalls == [["1"]] and clock.t == pytest.approx(300)


def test_stage1_cutoff_cancels_unstarted_slot_as_not_measured(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {"1": [(500, "RUNNING")], "2": [(10 ** 9, "PENDING")]})
    ctx, scalls = _running_ctx(tmp_path, clock, sacct, budget_s=1000, max_wall_s=10 ** 6)
    ctx.job_cids = {"1": ["ANCHOR", "G1", "ANCHOREND"], "2": ["ANCHOR", "G2", "G3", "ANCHOREND"]}
    states = wt.wait_for_jobs(ctx, ["1", "2"], "round0",
                              pending_cutoff=lambda: ctx.elapsed() >= 200)
    assert scalls == [["2"]]                       # only the never-started slot
    assert ctx.not_measured == [dict(job_id="2", cids=["G2", "G3"], stage="round0",
                                     reason="job still PENDING at the stage-1 cutoff")]
    assert states["1"][0] == "COMPLETED" and clock.t == pytest.approx(500)
    assert "2" in ctx.cancelled_jobs


def test_estimate_stage2_waves_and_model():
    # 3 points, 2 slots -> 2 waves; no stepra -> rad factor 1: spin = 120 + 35; run = 120+30+36
    est = wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 1.0, 2, 120.0)
    assert est == pytest.approx(2 * ((120 + 35) + 5 * (150 + 36)))
    assert wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 1.0, 3, 120.0) == pytest.approx(est / 2)
    # stepra 17: 36-step run has 3 radiation steps at r_rad 3.93
    f = 1 + 3 * (wt.RANKING_R_RAD - 1) / 36.0
    assert wt.rad_factor(36, 17) == pytest.approx(f)
    est = wt.estimate_stage2_s(1, 1, 36, 21, 36.0, 1.0, 1, 120.0, 0.0, stepra=17)
    assert est == pytest.approx((120 + 35 * wt.rad_factor(35, 17)) + (120 + 36 * f))


def test_rad_factor_conus25_like():
    # 82 steps, stepra 40 -> 3 radiation steps: ~1.107, not the old blanket 1.5
    assert wt.rad_factor(82, 40) == pytest.approx(1 + 3 * 2.93 / 82)
    assert wt.rad_factor(82, None) == 1.0


def _reserved_ctx(tmp_path, budget_s, **kw):
    ctx = _ctx(out_dir=str(tmp_path), budget_s=budget_s, **kw)
    ctx.ranking = wt.ranking_context("matthewp0", "/r.json", 21, 36, ANCHOR)
    ctx.ranking["dt"] = 36.0
    ctx.s_prior = 1.0
    ctx.startup_s = 120.0
    ctx.reserve = dict(n_points=3, n_configs=5, steps=36, spinup_min=21, dt=36.0, slots=3)
    # as if a first stage-1 wave had run 3 jobs at once at 1.0 s/step: concurrency 3, full top-3
    ctx.stage1_max_running = 3
    ctx.stage1_jobs = {"0": dict(s_anchor_job=1.0, excluded=False)}
    return ctx


def test_reserve_blocks_stage1_jobs_that_would_eat_stage2(tmp_path):
    reserve = wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 1.0, 3, 120.0)      # 1642.5 s
    sb, calls, _p = make_fake_sbatch()
    ctx = _reserved_ctx(tmp_path, budget_s=reserve + 500, sbatch_runner=sb)
    assert wt.stage2_reserve_s(ctx) == pytest.approx(reserve)
    ok, why = wt._stage1_may_start(ctx, 400)
    assert ok
    ok, why = wt._stage1_may_start(ctx, 600)
    assert not ok and "stage-2 reserve" in why
    wt._submit_round0_wave(ctx, [dict(ANCHOR)] + [_cfg("G%d" % i) for i in range(1, 6)], 1,
                           1.0, "round0")
    assert calls == []                              # 7 runs + spin-up do not fit before stage 2
    # measured anchor normalizers replace the prior
    ctx.stage1_jobs = {"9": dict(s_anchor_job=0.5, excluded=False)}
    assert wt.stage2_reserve_s(ctx) == pytest.approx(
        wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 0.5, 3, 120.0))


def test_reserve_is_zero_and_checks_inert_in_t0_mode(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.reserve = dict(n_points=3, n_configs=5, steps=36, spinup_min=21, dt=36.0, slots=3)
    ctx.s_prior = 1.0
    assert wt.stage2_reserve_s(ctx) == 0.0
    assert wt._stage1_may_start(ctx, 10 ** 9) == (True, None)
    assert wt._stage1_cutoff(ctx) is None


def test_stage1_cutoff_fires_when_reserve_reached(tmp_path):
    clock = FakeClock()
    ctx = _reserved_ctx(tmp_path, budget_s=2000, clock=clock)
    cut = wt._stage1_cutoff(ctx)
    assert not cut()
    clock.t = 2000 - wt.stage2_reserve_s(ctx)
    assert cut()


def _rebuilt(n=3):
    return [dict(index=i, point=types.SimpleNamespace(at=T0), case_id="matthewp%d" % i, T=T0,
                 cases_json="/r.json", steps=36, spinup_min=21, stepra=17, dt=36.0)
            for i in range(n)]


def test_fit_stage2_shrinks_then_skips(tmp_path):
    clock = FakeClock()
    ctx = _ctx(out_dir=str(tmp_path), clock=clock, budget_s=10 ** 6, max_nodes=12, nodes=4)
    ctx.startup_s = 120.0
    top = [_cfg("G1"), _cfg("G2"), _cfg("G3")]
    s_by = {"ANCHOR": 1.0, "G1": 1.0, "G2": 1.0, "G3": 1.0}
    ctx.stage1_max_running = 3                       # 3 point jobs at once -> 1 wave
    fit, est, why = wt.fit_stage2(ctx, top, ANCHOR, _rebuilt(), s_by)
    assert [c["id"] for c in fit] == ["G1", "G2", "G3"] and why is None
    ctx.stage1_max_running = 1                       # serial cluster -> 3 waves
    assert wt.fit_stage2(ctx, top, ANCHOR, _rebuilt(), s_by)[1] == pytest.approx(3 * est)
    ctx.stage1_max_running = 3
    per_run = wt.ranking_cost_model(120.0, 30.0, 36, 35, 17, 1.0)["per_run_s"]
    ctx.budget_s = est - per_run / 2                 # one run short -> drop G3
    fit, est2, why = wt.fit_stage2(ctx, top, ANCHOR, _rebuilt(), s_by)
    assert [c["id"] for c in fit] == ["G1", "G2"] and est2 == pytest.approx(est - per_run)
    ctx.budget_s = 10.0
    fit, _e, why = wt.fit_stage2(ctx, top, ANCHOR, _rebuilt(), s_by)
    assert fit == [] and "best candidate" in why
    ctx.budget_s = 10 ** 6
    fit, _e, why = wt.fit_stage2(ctx, [_cfg("G1")], ANCHOR, _rebuilt(), s_by)   # <2 candidates
    assert [c["id"] for c in fit] == ["G1"] and why is None


def test_run_predicted_branch_sets_running_budget_and_max_points(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(wt, "resolve_case", lambda case, cpn: (
        dict(e_we=91, e_sn=100, e_vert=45, prep_dir=str(tmp_path)), str(tmp_path / "nl"),
        {"core_count_priors": {"efficiency_table": []}}))

    def fake_run_predicted(ctx, nodes, nodes_note, s_est):
        seen.update(mode=ctx.budget_mode, wall=ctx.max_wall_s,
                    max_points=ctx.probe_cfg["max_points"])
        return {}

    monkeypatch.setattr(wt, "run_predicted", fake_run_predicted)
    monkeypatch.setattr(wt, "render_predicted_report_text", lambda r: "")
    base = ["matthew", "--minutes", "60", "--nodes", "1", "--steps", "5",
            "--out", str(tmp_path / "o"), "--ledger-dir", str(tmp_path / "l"),
            "--rank-by", "predicted"]
    wt.run(wt.build_arg_parser().parse_args(base), sacct_runner=make_fake_sacct({}),
           clock=FakeClock())
    assert seen == dict(mode="running", wall=3 * 60 * 60.0, max_points=3)
    wt.run(wt.build_arg_parser().parse_args(base + ["--max-points", "5",
                                                    "--max-wall-minutes", "90"]),
           sacct_runner=make_fake_sacct({}), clock=FakeClock())
    assert seen == dict(mode="running", wall=90 * 60.0, max_points=5)


def test_probe_key_explicit(tmp_path):
    pytest.importorskip("yaml")
    y = tmp_path / "c.yaml"
    y.write_text("probe:\n  max_points: 5\n")
    assert wt._probe_key_explicit(str(y), "max_points")
    y.write_text("probe:\n  steps: 30\n")
    assert not wt._probe_key_explicit(str(y), "max_points")
    assert not wt._probe_key_explicit(None, "max_points")


# ---------------------------------------------------------------------------
# stage-1 shrink-to-fit, minimum, feasibility pre-check, 60-min conus25-like scenario
# (conus25 e2e attempt 2: every round0 slot SKIPPED -> "stage 1 measured nothing")
# ---------------------------------------------------------------------------
def _shrink_ctx(tmp_path, budget_s, sb):
    ctx = _reserved_ctx(tmp_path, budget_s=budget_s, sbatch_runner=sb)
    ctx.reserve = dict(n_points=3, n_configs=5, steps=36, spinup_min=21, dt=36.0, slots=3)
    return ctx


def test_stage1_slot_is_shrunk_not_skipped(tmp_path):
    # no stepra -> rad factor 1: per_run = 120+30+36 = 186, spin-up = 120+35 = 155,
    # reserve = 155 + 5*186 = 1085 (3 points, 3 concurrent)
    sb, calls, paths = make_fake_sbatch()
    budget = 1085 + (155 + 5 * 186) + 10             # fits ANCHOR + 3 + ANCHOREND, not more
    ctx = _shrink_ctx(tmp_path, budget, sb)
    assert wt.stage2_reserve_s(ctx) == pytest.approx(1085)
    cands = [_cfg("G%d" % i) for i in range(1, 7)]
    wt._submit_round0_wave(ctx, [dict(ANCHOR)] + cands, 1, 1.0, "round0")
    assert _chunk_configs(paths) == [["ANCHOR", "G1", "G2", "G3", "ANCHOREND"]]
    assert [t["cid"] for t in ctx.stage1_trimmed] == ["G4", "G5", "G6"]


def test_stage1_minimum_is_anchor_one_candidate_anchor_end(tmp_path):
    sb, calls, paths = make_fake_sbatch()
    ctx = _shrink_ctx(tmp_path, 1085 + (155 + 3 * 186), sb)          # exactly the minimum
    wt._submit_round0_wave(ctx, [_cfg("G1"), _cfg("G2")], 3, 1.0, "round0")
    # 2 candidates over min(3 slots, 2) = 2 slots; each slot is exactly the minimum
    assert _chunk_configs(paths) == [["ANCHOR", "G1", "ANCHOREND"], ["ANCHOR", "G2", "ANCHOREND"]]
    sb2, calls2, _p = make_fake_sbatch()
    ctx = _shrink_ctx(tmp_path / "b", 1085 + (155 + 3 * 186) - 1, sb2)   # 1 s short
    wt._submit_round0_wave(ctx, [_cfg("G1"), _cfg("G2")], 3, 1.0, "round0")
    assert calls2 == []                              # only now is a slot skipped


def _predicted_env(monkeypatch, n_points=3, dt=15.0, stepra=40, s_led=1.3):
    pts = [types.SimpleNamespace(at=T0 + dt_.timedelta(hours=6 * i, minutes=20), reason="auto",
                                 source="rebuild") for i in range(n_points)]
    monkeypatch.setattr(wt.wtune_predict, "parse_namelist", lambda p: None)
    monkeypatch.setattr(wt.wtune_predict, "get_d01_steps", lambda nl: (1440, 21600, dt, None))
    monkeypatch.setattr(wt, "plan_predicted_points", lambda ctx, spin: (
        None, T0, T0 + dt_.timedelta(hours=24), dt, stepra, pts, []))
    monkeypatch.setattr(wt.wtune_predict, "read_wrfinput_sample_points", lambda d: ([], None))
    monkeypatch.setattr(wt.sweep, "estimate_s_per_step", lambda *a, **k: (s_led, "ledger", 5))
    return pts


dt_ = dt


def test_feasibility_precheck_refuses_before_submitting(monkeypatch, tmp_path):
    _predicted_env(monkeypatch)
    called = []
    monkeypatch.setattr(wt, "prepare_rebuilt_points", lambda *a, **k: called.append(1))
    sb, calls, _p = make_fake_sbatch()
    ctx = _ctx(out_dir=str(tmp_path), budget_s=20 * 60.0, nodes=4, max_nodes=12,
               sbatch_runner=sb, case_json=dict(e_we=1501, e_sn=1201, e_vert=50,
                                                prep_dir=str(tmp_path)))
    ctx.budget_mode = "running"
    ctx.stage1_mode = "pstar"                        # pstar: --minutes must hold both stages
    with pytest.raises(SystemExit, match=r"predicted mode needs at least \d+ min of running time "
                                         r"for matthew at 4 node\(s\) \(have 20\)"):
        wt.run_predicted(ctx, 4, "user", 1.3)
    assert called == [] and calls == []


@pytest.mark.parametrize("stage2_extra_s", [0.0, 3000.0])
def test_conus25_like_runs_stage1_and_stage2(monkeypatch, tmp_path, stage2_extra_s):
    """Observed (attempt 1, job 1388): ~200 s per candidate run (restart read + ~87 s startup +
    82 steps x ~1.3 s); rebuild prep 7 min; 4 nodes, max-nodes 12. 80 min of running budget:
    with nothing observed yet, stage-2 concurrency is 1 (conservative), which puts the minimum
    at ~77 min for 3 points (60 min is now refused up front -- attempt 3 really needed ~81).
    stage2_extra_s > 0: stage 2 runs 50 min past its estimate -> completes, overrun reported."""
    _predicted_env(monkeypatch)
    clock = FakeClock()
    ledger_dir = tmp_path / "ledger"
    submitted = {}                                   # job -> (t_submit, duration, config ids)
    counter = {"n": 2000}

    def sbatch(cmd):
        counter["n"] += 1
        jid = str(counter["n"])
        path = re.search(r"CONFIG_FILE=(\S+)", cmd).group(1)
        ids = [c["id"] for c in json.load(open(path))["configs"]]
        submitted[jid] = (clock.t, (87 + 80 * 1.3) + 200.0 * len(ids), ids)
        return "Submitted batch job %s\n" % jid

    def sacct(cmd):
        out = []
        for j in cmd[cmd.index("-j") + 1].split(","):
            t_sub, dur, _ids = submitted.get(j, (0, 0, []))
            if clock.t < t_sub + 30:
                st = "PENDING"
            elif clock.t < t_sub + 30 + dur:
                st = "RUNNING"
            else:
                st = "COMPLETED"
            out.append("%s|%s|%s" % (j, st, wt.sweep.campaign_experiment._format_hms(dur / 60.0)
                                     if st == "COMPLETED" else "00:00:00"))
        return "\n".join(out)

    def reader(case, job_ids):
        rows = []
        for j in job_ids:
            _t, _d, ids = submitted[j]
            for i, cid in enumerate(ids):
                s = 1.3 if cid in ("ANCHOR", "ANCHOREND") else 1.3 * (0.9 + 0.01 * (int(j) % 7) + 0.003 * i)
                rows.append(dict(run_id="%s_%s_r1_%s" % (case, cid, j), s_per_step_median=str(s),
                                 notes="config=%s node_dirty=none restart_target=20 "
                                       "restart_used=1 foreign_load=0" % cid))
        return rows

    def prepare(ctx, points, anchor, *a, **k):
        clock.t += 420.0                             # 7 min of controller + prep running time
        return [dict(index=i, point=p, case_id="matthewp%d" % i, T=p.at - dt_.timedelta(minutes=20),
                     cases_json=str(tmp_path / "rc.json"), prep_dir="", template="", steps=82,
                     spinup_min=20, stepra=40, dt=15.0, plan_dir="", job_id="1")
                for i, p in enumerate(points)]

    stage2 = {}

    def measure(ctx, configs, rebuilt, anchor, s_by_cid=None, **k):
        stage2["cids"] = [c["id"] for c in configs]
        stage2["left_s"] = ctx.budget_s - ctx.elapsed()
        clock.t += stage2_extra_s
        meas = {}
        for c in [anchor] + list(configs):
            meas[c["id"]] = [PM(at=rp["point"].at, reason="auto", source="rebuild",
                                s_normal=s_by_cid.get(c["id"], 1.3), n_rad=2, r=3.5)
                             for rp in rebuilt]
        return meas, [], ["9001"]

    monkeypatch.setattr(wt, "prepare_rebuilt_points", prepare)
    monkeypatch.setattr(wt, "measure_candidates_at_points", measure)
    scancel, scalls = make_fake_scancel()
    ctx = _ctx(out_dir=str(tmp_path / "out"), ledger_dir=str(ledger_dir), budget_s=4800.0,
               nodes=4, max_nodes=12, clock=clock, poll_interval=20.0, sbatch_runner=sbatch,
               sacct_runner=sacct, scancel_runner=scancel, ledger_reader=reader,
               proposer=lambda state, bs, kappa=None: (dict(stopped=True, reason="stub"), []),
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path)))
    ctx.budget_mode = "running"
    ctx.max_wall_s = 3 * 3600.0
    ctx.stage1_mode = "pstar"                        # this scenario pins the P*-restart stage 1
    report = wt.run_predicted(ctx, 4, "user", 1.3)
    assert report["stage1_ranking"], "stage 1 measured nothing"
    assert len(report["stage1_ranking"]) >= 3
    assert stage2["cids"] and len(stage2["cids"]) >= 1
    assert report["ranking"] and report["stage2_skipped_reason"] is None
    assert scalls == []                              # nothing hit the budget or the wall cap
    assert "concurrent" in report["stage2_reserve_note"]
    assert report["stage1_max_running"] == 3 and report["stage2_concurrency"] == 3
    if stage2_extra_s:
        assert report["stage2_overrun_s"] > 0 and "STAGE 2 OVERRUN" in \
            wt.render_predicted_report_text(report)
    else:
        assert ctx.elapsed() <= ctx.budget_s and not report["stage2_overrun_s"]


def test_conus25_60min_is_refused_with_conservative_concurrency(monkeypatch, tmp_path):
    _predicted_env(monkeypatch)
    monkeypatch.setattr(wt, "prepare_rebuilt_points", lambda *a, **k: pytest.fail("submitted"))
    ctx = _ctx(out_dir=str(tmp_path), budget_s=3600.0, nodes=4, max_nodes=12,
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path)))
    ctx.budget_mode = "running"
    ctx.stage1_mode = "pstar"
    with pytest.raises(SystemExit, match=r"needs at least 7\d min .*1 concurrent"):
        wt.run_predicted(ctx, 4, "user", 1.3)


# ---------------------------------------------------------------------------
# attempt 3 (conus25-pred3): observed stage-1 concurrency; stage 2 exempt from the budget
# ---------------------------------------------------------------------------
def _conc_ctx(tmp_path, clock, sacct, budget_s=5000.0, max_wall_s=10 ** 6):
    ctx, scalls = _running_ctx(tmp_path, clock, sacct, budget_s=budget_s, max_wall_s=max_wall_s,
                               nodes=4, max_nodes=12)
    ctx.ranking = wt.ranking_context("matthewp0", "/r.json", 21, 36, ANCHOR)
    ctx.ranking["dt"] = 36.0
    ctx.s_prior = 1.0
    ctx.startup_s = 120.0
    ctx.reserve = dict(n_points=3, n_configs=5, steps=36, spinup_min=21, dt=36.0, slots=3)
    return ctx, scalls


def test_serial_stage1_jobs_give_concurrency_1_reserve(tmp_path):
    clock = FakeClock()
    # both submitted at t=0; the busy cluster runs them one at a time (1417/1418)
    sacct, _c = timeline_sacct(clock, {"1417": [(300, "RUNNING")],
                                       "1418": [(300, "PENDING"), (600, "RUNNING")]})
    ctx, _s = _conc_ctx(tmp_path, clock, sacct)
    assert wt.stage2_concurrency(ctx) == 1           # nothing observed yet: conservative
    wt.wait_for_jobs(ctx, ["1417", "1418"], "round0")
    assert ctx.stage1_max_running == 1 and wt.stage2_concurrency(ctx) == 1
    ctx.stage1_jobs = {"1417": dict(s_anchor_job=1.0, excluded=False)}
    assert wt.stage2_reserve_s(ctx) == pytest.approx(
        wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 1.0, 1, 120.0, 30.0))   # 3 waves
    assert "1 concurrent: observed max 1" in wt.describe_stage2_reserve(ctx)


def test_parallel_stage1_jobs_raise_concurrency_capped(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {j: [(300, "RUNNING")] for j in ("1", "2", "3", "4")})
    ctx, _s = _conc_ctx(tmp_path, clock, sacct)
    wt.wait_for_jobs(ctx, ["1", "2", "3", "4"], "round0")
    assert ctx.stage1_max_running == 4 and wt.stage2_concurrency(ctx) == 3   # floor(12/4)


def test_concurrency_not_observed_outside_stage1(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {j: [(300, "RUNNING")] for j in ("1", "2", "3")})
    ctx, _s = _conc_ctx(tmp_path, clock, sacct)
    ctx.ranking = None                               # rebuild prep / stage 2
    wt.wait_for_jobs(ctx, ["1", "2", "3"], "rebuild-prep")
    assert ctx.stage1_max_running == 0


def test_stage2_exempt_from_running_budget_completes(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {"1436": [(840, "RUNNING")],
                                       "1437": [(840, "PENDING"), (1680, "RUNNING")]})
    ctx, scalls = _conc_ctx(tmp_path, clock, sacct, budget_s=600)
    ctx.ranking = None
    states = wt.wait_for_jobs(ctx, ["1436", "1437"], "stage2", exempt_budget=True)
    assert scalls == [] and states["1437"][0] == "COMPLETED"
    assert ctx.elapsed() == pytest.approx(1680) and ctx.elapsed() - ctx.budget_s > 0


def test_stage2_exempt_still_hits_wall_cap(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {"1436": [(10 ** 9, "RUNNING")]})
    ctx, scalls = _conc_ctx(tmp_path, clock, sacct, budget_s=600, max_wall_s=1000)
    ctx.ranking = None
    wt.wait_for_jobs(ctx, ["1436"], "stage2", exempt_budget=True)
    assert scalls == [["1436"]] and clock.t == pytest.approx(1000)


def test_stage2_wall_cap_cancel_still_raises(tmp_path):
    clock = FakeClock()
    sacct, cancelled = timeline_sacct(clock, {"501": [(10 ** 9, "RUNNING")]})
    ctx, scalls = _conc_ctx(tmp_path, clock, sacct, budget_s=600, max_wall_s=1000)
    ctx.ranking = None
    sb, _calls, _p = make_fake_sbatch(start=500)
    ctx.sbatch_runner = sb

    def scancel(ids):
        scalls.append(list(ids))
        cancelled.update(ids)

    ctx.scancel_runner = scancel
    with pytest.raises(RuntimeError, match="not COMPLETED"):
        wt.measure_candidates_at_points(ctx, [_cfg("G1")], [_rp(tmp_path)], ANCHOR)
    assert scalls == [["501"]]


# ---------------------------------------------------------------------------
# stage1=t0 (default since conus25-pred4 G3): broad t0 search, rebuild prep beside it
# ---------------------------------------------------------------------------
def test_stage1_option_cli_yaml_default():
    assert wt.build_arg_parser().parse_args(["x"]).stage1 is None
    assert wt.build_arg_parser().parse_args(["x", "--stage1", "pstar"]).stage1 == "pstar"
    with pytest.raises(SystemExit):
        wt.build_arg_parser().parse_args(["x", "--stage1", "bogus"])
    cfg, errs = wtune_config.resolve_config({})
    assert cfg["probe"]["stage1"] == "t0" and not errs
    _c, errs = wtune_config.resolve_config({"probe": {"stage1": "bogus"}})
    assert any("probe.stage1" in e for e in errs)
    assert _ctx().stage1_mode == "t0"


def test_stage1_t0_feasibility_checks_stage1_only(monkeypatch, tmp_path, capsys):
    _predicted_env(monkeypatch)
    monkeypatch.setattr(wt, "submit_rebuilt_points", lambda *a, **k: pytest.fail("submitted"))
    ctx = _ctx(out_dir=str(tmp_path / "o"), budget_s=5 * 60.0, nodes=4, max_nodes=12, steps=20,
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path / "c")))
    ctx.budget_mode = "running"
    # one minimal t0 job = 3 x (120 + 20 x 1.3) = 438 s = 7:18 > 5 min -> refused
    with pytest.raises(SystemExit, match=r"needs at least 8 min .*stage-1 minimum 7:18 = one t0 "
                                         r"job"):
        wt.run_predicted(ctx, 4, "user", 1.3)
    # 10 min holds it: accepted even though stage 2 alone is estimated at ~65 min
    ctx.budget_s = 10 * 60.0

    class Submitted(Exception):
        pass

    monkeypatch.setattr(wt, "submit_rebuilt_points",
                        lambda *a, **k: (_ for _ in ()).throw(Submitted()))
    with pytest.raises(Submitted):
        wt.run_predicted(ctx, 4, "user", 1.3)
    out = capsys.readouterr().out
    assert re.search(r"stage 2 will need about \d+ min MORE, on top of --minutes \(3 point\(s\)"
                     r" x \(spin-up \+ 5 runs\), concurrency 1", out)


def test_stage1_t0_end_to_end(monkeypatch, tmp_path):
    _predicted_env(monkeypatch)
    clock = FakeClock()
    events = []
    submitted = {}
    counter = {"n": 3000}
    cmds = []

    def sbatch(cmd):
        counter["n"] += 1
        jid = str(counter["n"])
        cmds.append(cmd)
        path = re.search(r"CONFIG_FILE=(\S+)", cmd).group(1)
        ids = [c["id"] for c in json.load(open(path))["configs"]]
        submitted[jid] = (clock.t, 150.0 * len(ids), ids)
        events.append(("stage1-submit", jid))
        return "Submitted batch job %s\n" % jid

    def sacct(cmd):
        out = []
        for j in cmd[cmd.index("-j") + 1].split(","):
            t_sub, dur, _ids = submitted.get(j, (0, 0, []))
            st = ("PENDING" if clock.t < t_sub + 30 else
                  "RUNNING" if clock.t < t_sub + 30 + dur else "COMPLETED")
            out.append("%s|%s|%s" % (j, st, wt.sweep.campaign_experiment._format_hms(dur / 60.0)))
        return "\n".join(out)

    def num(cid):
        return int(re.sub(r"\D", "", cid) or 0)

    def reader(case, job_ids):
        assert case == "matthew"                     # t0 stage 1: the case's own ledgers
        rows = []
        for j in job_ids:
            _t, _d, ids = submitted[j]
            scale = 1.5 if int(j) % 2 else 1.0       # alternate slow/fast allocations
            for cid in ids:
                base = 1.3 if cid in ("ANCHOR", "ANCHOREND") else 1.3 * (0.8 + 0.01 * num(cid))
                rows.append(dict(run_id="%s_%s_r1_%s" % (case, cid, j),
                                 s_per_step_median=str(scale * base),
                                 notes="config=%s node_dirty=none" % cid))
        return rows

    def submit_prep(ctx, points, anchor, *a, **k):
        events.append(("prep-submit",))
        ctx.owned_jobs.extend(["p1", "p2", "p3"])
        return dict(job_ids=["p1", "p2", "p3"], points=points)

    def finish_prep(ctx, handle):
        events.append(("prep-wait",))
        return [dict(index=i, point=p, case_id="matthewp%d" % i, T=p.at - dt_.timedelta(minutes=20),
                     cases_json="/r.json", steps=82, spinup_min=20, stepra=40, dt=15.0,
                     job_id="p%d" % (i + 1))
                for i, p in enumerate(handle["points"])]

    stage2 = {}

    def measure(ctx, configs, rebuilt, anchor, s_by_cid=None, **k):
        events.append(("stage2",))
        stage2["cids"] = [c["id"] for c in configs]
        stage2["ranking_ctx"] = ctx.ranking
        meas = {c["id"]: [PM(at=rp["point"].at, reason="auto", source="rebuild",
                             s_normal=s_by_cid.get(c["id"], 1.3), n_rad=2, r=3.5)
                          for rp in rebuilt] for c in [anchor] + list(configs)}
        return meas, [], ["9001"]

    monkeypatch.setattr(wt, "submit_rebuilt_points", submit_prep)
    monkeypatch.setattr(wt, "finish_rebuilt_points", finish_prep)
    monkeypatch.setattr(wt, "prepare_rebuilt_points", lambda *a, **k: pytest.fail("pstar path"))
    monkeypatch.setattr(wt, "measure_candidates_at_points", measure)
    scancel, scalls = make_fake_scancel()
    ctx = _ctx(out_dir=str(tmp_path / "out"), ledger_dir=str(tmp_path / "l"), budget_s=5400.0,
               nodes=4, max_nodes=12, steps=20, clock=clock, poll_interval=20.0,
               sbatch_runner=sbatch, sacct_runner=sacct, scancel_runner=scancel,
               ledger_reader=reader,
               proposer=lambda state, bs, kappa=None: (dict(stopped=True, reason="stub"), []),
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path / "c")))
    ctx.budget_mode = "running"
    ctx.max_wall_s = 3 * 5400.0
    report = wt.run_predicted(ctx, 4, "user", 1.3)

    # order: prep submitted first, stage 1 ran beside it, prep awaited only before stage 2
    kinds = [e[0] for e in events]
    assert kinds[0] == "prep-submit"
    assert kinds.index("prep-wait") > max(i for i, k in enumerate(kinds) if k == "stage1-submit")
    assert kinds[-1] == "stage2" and kinds[-2] == "prep-wait"
    # stage 1 = plain t0 jobs: no ranking context, t0 STEPS, anchors in every job
    for c in cmds:
        exp = c.split("--export=")[1].split()[0]
        assert "RANKING_START" not in exp and "CASES_JSON" not in exp
        assert ",CASE=matthew," in exp and ",STEPS=20," in exp
    for jid, (_t, _d, ids) in submitted.items():
        assert ids[0] == "ANCHOR" and ids[-1] == "ANCHOREND" and len(ids) >= 3
    assert ctx.stage1_max_running >= 1 and report["stage1_mode"] == "t0"
    assert not report["stage1_excluded"]             # no restart_used required at t0
    # shortlist = the 3 smallest ANCHOR-NORMALIZED keys, not the raw medians
    ranked = report["stage1_ranking"]
    assert [e["norm_key"] for e in ranked] == sorted(e["norm_key"] for e in ranked)
    by_num = sorted((e["id"] for e in ranked), key=num)
    assert stage2["cids"] == by_num[:3]
    raw_top = [e["id"] for e in sorted(ranked, key=lambda e: e["median_s_per_step"])[:3]]
    assert raw_top != stage2["cids"]                 # the slow allocations would have skewed it
    assert stage2["ranking_ctx"] is None and report["ranking"] and scalls == []


# ---------------------------------------------------------------------------
# 2026-09-28 (conus25-pred5): --minutes is the stage-1 search budget only in stage1=t0
# ---------------------------------------------------------------------------
def _t0_fakes(clock, run_s=150.0, queue_s=30.0):
    submitted, cmds, counter = {}, [], {"n": 4000}

    def sbatch(cmd):
        counter["n"] += 1
        jid = str(counter["n"])
        cmds.append(cmd)
        path = re.search(r"CONFIG_FILE=(\S+)", cmd).group(1)
        ids = [c["id"] for c in json.load(open(path))["configs"]]
        submitted[jid] = (clock.t, run_s * len(ids), ids)
        return "Submitted batch job %s\n" % jid

    def sacct(cmd):
        out = []
        for j in cmd[cmd.index("-j") + 1].split(","):
            t_sub, dur, _ids = submitted.get(j, (0, 0, []))
            st = ("PENDING" if clock.t < t_sub + queue_s else
                  "RUNNING" if clock.t < t_sub + queue_s + dur else "COMPLETED")
            out.append("%s|%s|%s" % (j, st, wt.sweep.campaign_experiment._format_hms(dur / 60.0)))
        return "\n".join(out)

    def reader(case, job_ids):
        rows = []
        for j in job_ids:
            _t, _d, ids = submitted.get(j, (0, 0, []))
            for cid in ids:
                n = int(re.sub(r"\D", "", cid) or 0)
                s_ = 1.3 if cid in ("ANCHOR", "ANCHOREND") else 1.3 * (0.8 + 0.01 * n)
                rows.append(dict(run_id="%s_%s_r1_%s" % (case, cid, j), s_per_step_median=str(s_),
                                 notes="config=%s node_dirty=none" % cid))
        return rows

    return sbatch, sacct, reader, submitted, cmds


def _stub_proposer(state, bs, kappa=None):
    return dict(stopped=True, reason="stub"), []


def _t0_ctx(tmp_path, clock, fakes, budget_s, **kw):
    sbatch, sacct, reader, _sub, _cmds = fakes
    scancel, scalls = make_fake_scancel()
    ctx = _ctx(out_dir=str(tmp_path / "out"), ledger_dir=str(tmp_path / "l"), budget_s=budget_s,
               nodes=4, max_nodes=12, steps=20, clock=clock, poll_interval=20.0,
               sbatch_runner=sbatch, sacct_runner=sacct, scancel_runner=scancel,
               ledger_reader=reader, proposer=_stub_proposer,
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path / "c")),
               **kw)
    return ctx, scalls


def _t0_mode_candidate_count(monkeypatch, tmp_path, budget_s):
    """What the plain t0 tuner's search (round0 + adaptive rounds; confirm adds no candidate)
    measures on the same fake timeline and budget."""
    monkeypatch.setattr(wt.sweep, "estimate_s_per_step", lambda *a, **k: (1.3, "ledger", 5))
    clock = FakeClock()
    fakes = _t0_fakes(clock)
    ctx, _s = _t0_ctx(tmp_path / "t0", clock, fakes, budget_s)
    job_ids0, cbc, rows0 = wt.run_round0(ctx)
    res = wt.aggregate_by_cid(rows0, cbc)
    wt.run_adaptive_rounds(ctx, cbc, res, 1.3)
    return len([c for c in res if c != "ANCHOR"])


def _run_pred_t0(monkeypatch, tmp_path, budget_s, stage2_s=1200.0, stage2_cap_s=None):
    _predicted_env(monkeypatch)
    clock = FakeClock()
    fakes = _t0_fakes(clock)
    events = []

    def submit_prep(ctx, points, anchor, *a, **k):
        clock.t += 200.0                             # ERA5 fetch + stage: not stage-1 work
        events.append("prep-submit")
        return dict(job_ids=["p1", "p2", "p3"], points=points)

    def finish_prep(ctx, handle):
        events.append("prep-wait")
        return [dict(index=i, point=p, case_id="matthewp%d" % i, T=p.at - dt_.timedelta(minutes=20),
                     cases_json="/r.json", steps=82, spinup_min=20, stepra=40, dt=15.0,
                     job_id="p%d" % (i + 1)) for i, p in enumerate(handle["points"])]

    stage2 = {}

    def measure(ctx, configs, rebuilt, anchor, s_by_cid=None, **k):
        events.append("stage2")
        stage2["cids"] = [c["id"] for c in configs]
        clock.t += stage2_s                          # stage 2's own running time
        meas = {c["id"]: [PM(at=rp["point"].at, reason="auto", source="rebuild",
                             s_normal=s_by_cid.get(c["id"], 1.3), n_rad=2, r=3.5)
                          for rp in rebuilt] for c in [anchor] + list(configs)}
        return meas, [], ["9001"]

    monkeypatch.setattr(wt, "submit_rebuilt_points", submit_prep)
    monkeypatch.setattr(wt, "finish_rebuilt_points", finish_prep)
    monkeypatch.setattr(wt, "measure_candidates_at_points", measure)
    ctx, scalls = _t0_ctx(tmp_path / "pred", clock, fakes, budget_s)
    ctx.budget_mode = "running"
    ctx.max_wall_s = 3 * budget_s
    ctx.stage2_budget_s = stage2_cap_s
    report = wt.run_predicted(ctx, 4, "user", 1.3)
    return report, ctx, stage2, events, scalls


def test_stage1_t0_30min_measures_at_least_the_t0_search(monkeypatch, tmp_path):
    n_t0 = _t0_mode_candidate_count(monkeypatch, tmp_path, 1800.0)
    report, ctx, stage2, events, scalls = _run_pred_t0(monkeypatch, tmp_path, 1800.0)
    n_pred = len(report["stage1_ranking"]) + len(report["stage1_excluded"])
    assert n_pred >= n_t0 >= 1, (n_pred, n_t0)
    # stage 1 used (about) --minutes, stage 2 ran AFTER it and is reported separately
    assert report["stage1_running_min"] <= 30.0 + 1.0
    assert report["stage2_running_min"] >= 20.0
    assert report["total_running_min"] == pytest.approx(
        report["stage1_running_min"] + report["stage2_running_min"] + report["prep_submit_min"])
    assert report["prep_submit_min"] == pytest.approx(200.0 / 60.0)
    assert report["stage2_estimate_min"] > 0 and stage2["cids"] and len(stage2["cids"]) == 3
    assert report["stage2_overrun_s"] is None and scalls == []
    assert events[0] == "prep-submit" and events[-2:] == ["prep-wait", "stage2"]
    assert "stage 2 separate" in report["budget_semantics"]
    assert "running time: stage 1" in wt.render_predicted_report_text(report)
    # the wall cap was derived from --minutes + the stage-2 estimate, not 3x --minutes alone
    assert ctx.max_wall_s > 3 * 1800.0


def test_stage2_minutes_cap_trims_shortlist(monkeypatch, tmp_path):
    full, _c, stage2, _e, _s = _run_pred_t0(monkeypatch, tmp_path / "a", 1800.0)
    assert len(stage2["cids"]) == 3 and full["stage2_minutes_cap"] is None
    cap = full["stage2_estimate_s"] - 1.0            # one second short of the full top-3
    report, _c, stage2, _e, _s = _run_pred_t0(monkeypatch, tmp_path / "b", 1800.0,
                                               stage2_cap_s=cap)
    assert len(stage2["cids"]) == 2 and report["stage2_estimate_s"] <= cap
    assert report["stage2_minutes_cap"] == pytest.approx(cap / 60.0)


def test_fit_stage2_no_cap_never_trims(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path), budget_s=10.0, max_nodes=12, nodes=4)
    ctx.startup_s = 120.0
    top = [_cfg("G1"), _cfg("G2"), _cfg("G3")]
    fit, est, why = wt.fit_stage2(ctx, top, ANCHOR, _rebuilt(), {"ANCHOR": 1.0}, left_s=None)
    assert [c["id"] for c in fit] == ["G1", "G2", "G3"] and why is None and est > 10.0
    fit, est1, why = wt.fit_stage2(ctx, top, ANCHOR, _rebuilt(), {"ANCHOR": 1.0},
                                   left_s=est - 1.0)
    assert [c["id"] for c in fit] == ["G1", "G2"]


def test_stage1_t0_holds_no_stage2_reserve_pstar_does(tmp_path):
    ctx = _reserved_ctx(tmp_path, budget_s=3600.0)   # pstar-style: ranking context set
    assert wt.stage1_reserve_s(ctx) > 0
    ctx2 = _ctx(out_dir=str(tmp_path), budget_s=3600.0)
    ctx2.stage1 = dict(mode="t0", anchor=ANCHOR)
    ctx2.reserve = dict(ctx.reserve)
    ctx2.s_prior = 1.0
    ctx2.reserve_in_stage1 = False
    assert wt.stage1_reserve_s(ctx2) == 0.0
    assert wt._stage1_may_start(ctx2, 3500.0)[0] and not wt._stage1_may_start(ctx2, 3700.0)[0]


def test_stage2_minutes_cli_and_yaml():
    assert wt.build_arg_parser().parse_args(["x", "--stage2-minutes", "25"]).stage2_minutes == 25
    cfg, errs = wtune_config.resolve_config({})
    assert cfg["probe"]["stage2_minutes"] is None and not errs
    _c, errs = wtune_config.resolve_config({"probe": {"stage2_minutes": -1}})
    assert any("stage2_minutes" in e for e in errs)
