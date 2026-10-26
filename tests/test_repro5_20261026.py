"""Independent reproductions of five 2026-10-26 defect reports (f1ed209 -> 160e113).
Fakes only at the Slurm / WRF boundary; nothing is submitted."""
import csv
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
from wtune.tools import scc_sweep as sweep

FMT = "%Y-%m-%d_%H:%M:%S"


# ---------------------------------------------------------------- defect 1
def _fake_predict(monkeypatch, compute_s, init_s):
    def build(**kw):
        return {"total_s": compute_s + init_s,
                "compute": {"compute_total_s": compute_s, "r_rad_source": "x"},
                "init": {"total_s": init_s}}
    monkeypatch.setattr(wp, "build_prediction", build)


def _o0(tmp_path):
    ctx = T._ctx(max_nodes=4, out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    return ctx, wtune.tune.pipeline.run_level0(ctx, T.ALL_NODES_PROFILE, 1000)


@pytest.mark.parametrize("compute_s,init_s", [(80.0, 20.0), (0.4, 99.0), (500.0, 60.0)])
def test_d1_o0_breakdown_uses_predicts_compute_and_init(monkeypatch, tmp_path, compute_s, init_s):
    _fake_predict(monkeypatch, compute_s, init_s)
    ctx, rep = _o0(tmp_path)
    assert ctx.startup_s == 120.0
    p = rep["pipeline"]["prediction"]
    assert p["compute_s"] == pytest.approx(compute_s)
    assert p["init_s"] == pytest.approx(init_s)
    assert p["compute_s"] >= 0 and "compute -" not in p["breakdown"]


def test_d1_cold_start_s_per_step_is_predicts_compute_over_steps(monkeypatch, tmp_path):
    _fake_predict(monkeypatch, 80.0, 20.0)
    ctx = T._ctx(out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    cfg = dict(ranks=72, omp=1, nx=8, ny=9, numtiles=1)
    total, s, _src = wtune.tune.prediction.cold_start_prediction(ctx, 1, cfg, 1000)
    assert total == pytest.approx(100.0)
    assert s == pytest.approx(0.08)


def test_d1_cold_start_s_per_step_when_init_exceeds_startup(monkeypatch, tmp_path):
    _fake_predict(monkeypatch, 200.0, 300.0)
    ctx = T._ctx(out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    cfg = dict(ranks=72, omp=1, nx=8, ny=9, numtiles=1)
    total, s, _src = wtune.tune.prediction.cold_start_prediction(ctx, 1, cfg, 100)
    assert total == pytest.approx(500.0)
    assert s == pytest.approx(2.0)


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


def test_d2_unmeasured_probe_job_is_named_with_its_slurm_state(tmp_path, capsys):
    ctx, job_nodes = _sweep_with_dead_n1(tmp_path)
    sw = wtune.tune.run_node_sweep(ctx, [1, 2, 4])
    assert 1 not in sw["measurements"] and {2, 4} <= set(sw["measurements"])
    j1 = next(j for j, n in job_nodes.items() if n == 1)
    out = capsys.readouterr().out
    assert any(j1 in ln and "TIMEOUT" in ln for ln in out.splitlines()), out


def test_d2_every_measured_n_gets_no_unmeasured_line(tmp_path, capsys):
    sbatch, reader, job_nodes = T.make_scaling_fakes({1: 1.0, 2: 0.5})
    ctx = T._ctx(hotspot_fill=False, sbatch_runner=sbatch, ledger_reader=reader,
                 budget_s=36000.0, out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    wtune.tune.run_node_sweep(ctx, [1, 2])
    assert "TIMEOUT" not in capsys.readouterr().out


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


def _ns(first_n):
    rows = [dict(nodes=n, median=1.0 / n, n=2, spread=0.0, speedup=float(n) / first_n,
                 efficiency=1.0, wall_s=None, node_hours=None, maxrss_mb=None)
            for n in (first_n, first_n * 2)]
    return dict(table=rows, choices={}, objective="time", finalists=[], chosen=first_n)


def test_d2_report_table_names_the_reference_n_when_not_1():
    text = "\n".join(wtune.tune.report._node_table_lines(_ns(2)))
    assert re.search(r"relative to (the )?(N=)?\s*2\b", text, re.I), text


def test_d2_report_table_has_no_reference_note_when_n1_measured():
    text = "\n".join(wtune.tune.report._node_table_lines(_ns(1)))
    assert "relative to" not in text.lower()


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


def _plain_run(tmp_path, fake_bp_, fakes, case_text=None, extra=()):
    from test_wtune_run import NAMELIST_3DOM
    _make_case(tmp_path, case_text or NAMELIST_3DOM)
    return _run(tmp_path, fakes, list(extra))


def test_d3_own_restart_setting_keeps_only_newest_valid_set(tmp_path, fake_bp):  # noqa: F811
    f = Multi(lambda i: _h(i, *((2, 4, 6) if i["start"].hour == 0 else (8, 10, 12))))
    rc, out, rep, _ = _plain_run(tmp_path, fake_bp, f)
    assert rc == 0 and rep["status"] == "completed"
    assert _sets(str(out / "seg_0")) == [_t(6)]
    assert _sets(str(out / "seg_1")) == sorted([_t(6), _t(12)])      # 06:00 = link from seg_0
    link = out / "seg_1" / ("wrfrst_d01_" + _t(6))
    assert os.path.islink(link) and os.path.exists(link)
    assert os.path.exists(out / "seg_0" / ("wrfrst_d01_" + _t(6)))


def test_d3_not_yet_valid_newer_set_is_not_deleted(tmp_path, fake_bp):  # noqa: F811
    f = Multi(lambda i: _h(i, *((2, 4, 6) if i["start"].hour == 0 else (8, 12))),
              unfinished=lambda i: _h(i, *((7,) if i["start"].hour == 0 else (10,))))
    rc, out, rep, _ = _plain_run(tmp_path, fake_bp, f)
    assert rc == 0
    assert _sets(str(out / "seg_0")) == [_t(6), _t(7)]
    assert _sets(str(out / "seg_1")) == sorted([_t(6), _t(10), _t(12)])


def test_d3_case_own_restart_setting_deletes_nothing(tmp_path, fake_bp):  # noqa: F811
    from test_wtune_run import NAMELIST_3DOM
    f = Multi(lambda i: _h(i, *((2, 4, 6) if i["start"].hour == 0 else (8, 10, 12))))
    _make_case(tmp_path, _with_restart(120, NAMELIST_3DOM))     # 2 h divides the segment
    rc, out, rep, _ = _run(tmp_path, f)
    assert rc == 0
    assert _sets(str(out / "seg_0")) == [_t(2), _t(4), _t(6)]
    assert sorted(set(_sets(str(out / "seg_1"))) - {_t(6)}) == [_t(8), _t(10), _t(12)]


# ---------------------------------------------------------------- defect 4
def _nl(adaptive, extra_domains="", dx=3000, dt_s=18, minutes=180):
    return ("&time_control\n run_minutes = %d,\n /\n&domains\n max_dom = 1,\n"
            " e_we = 100, e_sn = 100, e_vert = 40,\n dx = %d,\n dy = %d,\n time_step = %d,\n"
            "%s /\n&physics\n radt = 3,\n /\n"
            % (minutes, dx, dx, dt_s,
               (" use_adaptive_time_step = .true.,\n" if adaptive else "") + extra_domains))


def _pred(tmp_path, name, text):
    d = tmp_path / name
    d.mkdir()
    (d / "namelist.input").write_text(text)
    return wp.build_prediction(
        namelist_path=str(d / "namelist.input"), case_dir=str(d), nodes=1,
        config={"ranks": 4, "omp": 1, "nx": 2, "ny": 2, "numtiles": 1}, ledger_paths=None,
        calibration_path=os.path.join(T.os.path.dirname(os.path.dirname(os.path.abspath(
            __file__))), "machine", "calibration", "scc.json"),
        io_form_override=None, io_target="nfs")


def test_d4_explicit_max_min_time_step_widen_the_range(tmp_path):
    ad = _pred(tmp_path, "ad", _nl(True, " max_time_step = 60,\n min_time_step = 9,\n"))
    fewest = _pred(tmp_path, "mx", _nl(False, dt_s=60))["total_s"]
    most = _pred(tmp_path, "mn", _nl(False, dt_s=9))["total_s"]
    assert ad["range_low_s"] <= fewest * 1.02
    assert ad["range_high_s"] >= most * 0.98
    assert "adaptive" in (ad["steps_note"] or "").lower()


def test_d4_default_max_min_are_8dx_and_3dx_seconds(tmp_path):
    ad = _pred(tmp_path, "ad", _nl(True, " max_time_step = -1,\n"))     # min absent
    fewest = _pred(tmp_path, "mx", _nl(False, dt_s=24))["total_s"]      # 8 * 3 km
    most = _pred(tmp_path, "mn", _nl(False, dt_s=9))["total_s"]         # 3 * 3 km
    assert ad["range_low_s"] <= fewest * 1.02
    assert ad["range_high_s"] >= most * 0.98
    assert "adaptive" in (ad["steps_note"] or "").lower()


def test_d4_without_adaptive_nothing_changes(tmp_path):
    plain = _pred(tmp_path, "p", _nl(False, " max_time_step = 60,\n min_time_step = 9,\n"))
    assert "adaptive" not in (plain["steps_note"] or "").lower()
    ref = _pred(tmp_path, "r", _nl(False))
    assert plain["range_low_s"] == pytest.approx(ref["range_low_s"])
    assert plain["range_high_s"] == pytest.approx(ref["range_high_s"])


def _tune_output(monkeypatch, tmp_path, capsys, text):
    (tmp_path / "namelist.input").write_text(text)
    T._run_cli(monkeypatch, tmp_path, ["-O0"])
    cap = capsys.readouterr()
    return cap.out + cap.err


def test_d4_tune_warns_about_adaptive_time_step(monkeypatch, tmp_path, capsys):
    out = _tune_output(monkeypatch, tmp_path, capsys, _nl(True, " max_time_step = 60,\n"))
    assert "adaptive" in out.lower()


def test_d4_tune_is_silent_without_adaptive(monkeypatch, tmp_path, capsys):
    out = _tune_output(monkeypatch, tmp_path, capsys, _nl(False))
    assert "adaptive time step" not in out.lower()


# ---------------------------------------------------------------- defect 5
def _led(path, rows):
    cols = ["case_base", "hardware_id", "s_per_step_median", "n_nodes", "notes", "e_we", "e_sn",
            "e_vert", "n_ranks", "omp_threads", "nproc_x", "nproc_y", "numtiles"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _lr(s, n, build=None):
    return {"case_base": "c", "hardware_id": "hw1", "s_per_step_median": s, "n_nodes": n,
            "notes": "config=C1 mode=timing" + (" build=%s" % build if build else ""),
            "e_we": 100, "e_sn": 100, "e_vert": 40, "n_ranks": 4 * n, "omp_threads": 1,
            "nproc_x": 2, "nproc_y": 2, "numtiles": 1}


def _est(tmp_path, rows, nodes, build_id=None):
    _led(tmp_path / "ledger_c_x.csv", rows)
    return sweep.estimate_s_per_step("c", nodes, str(tmp_path), hardware_id="hw1",
                                     fallback=10.0, build_id=build_id)


CURVE = [_lr(0.016, 1), _lr(0.012, 2), _lr(0.010, 4), _lr(0.0095, 8)]


@pytest.mark.parametrize("nodes,want", [(1, 0.016), (2, 0.012), (8, 0.0095), (16, 0.0095 / 2),
                                        (3, 0.010 * 4 / 3)])
def test_d5_uses_only_the_nearest_measured_n(tmp_path, nodes, want):
    s, src, _n = _est(tmp_path, CURVE, nodes)
    assert s == pytest.approx(want) and src == "ledger"


def test_d5_tie_in_log_distance_picks_one_neighbour_not_a_blend(tmp_path):
    s, _src, _n = _est(tmp_path, [_lr(0.016, 1), _lr(0.010, 4)], 2)
    assert s == pytest.approx(0.016 / 2) or s == pytest.approx(0.010 * 4 / 2)


def test_d5_median_over_rows_at_the_nearest_n(tmp_path):
    rows = [_lr(0.010, 4), _lr(0.014, 4), _lr(0.030, 4), _lr(0.016, 1)]
    s, _src, _n = _est(tmp_path, rows, 4)
    assert s == pytest.approx(0.014)


def test_d5_build_filter_same_build_first_never_another(tmp_path):
    rows = [_lr(0.016, 1, "A"), _lr(0.001, 2, "B"), _lr(0.012, 2, None)]
    s, src, _n = _est(tmp_path, rows, 2, build_id="A")
    assert (src, s) == ("ledger", pytest.approx(0.016 / 2))
    s, src, _n = _est(tmp_path, [_lr(0.001, 2, "B"), _lr(0.012, 4, None)], 2, build_id="A")
    assert src == "ledger_unknown_build" and s == pytest.approx(0.012 * 4 / 2)
    s, src, _n = _est(tmp_path, [_lr(0.001, 2, "B")], 2, build_id="A")
    assert src == "fallback_10s" and s == 10.0
