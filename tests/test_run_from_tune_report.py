"""wtune run --tune-report with an -O (decision 0059) report: the layout comes from the report's
winner (CLI flags override), the winner's measured median s/step is the rate when the report has
no ranking points, and the ppr rank mapping is checked against what tune measured."""
import json
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import wtune_run as wr  # noqa: E402
from test_wtune_run import Fakes, _make_case  # noqa: E402

WINNER = {"cid": "ANCHOR", "cores_per_node": 72, "median": 0.5, "nodes": 1, "numtiles": 2,
          "nx": 8, "ny": 9, "omp": 1, "ranks": 72, "reps": 3, "source": "t0 sweep ANCHOR"}


def _report(tmp_path, **kw):
    rep = {"nodes": 1, "ranking": [], "winner": dict(WINNER),
           "node_selection": {"chosen": 1, "chosen_note": "x", "layouts": [dict(WINNER)]}}
    rep.update(kw)
    path = tmp_path / "tune_report.json"
    path.write_text(json.dumps(rep))
    return str(path)


def _main(tmp_path, args, **kw):
    case = tmp_path / "case" if (tmp_path / "case").exists() else _make_case(tmp_path)
    lines = []
    rc = wr.main([str(case), "--out", str(tmp_path / "out"), "--poll-interval", "0"] + args,
                 log=lines.append, **kw)
    return rc, lines


def test_layout_sources_in_order():
    assert wr.tune_report_layout({"winner": WINNER})["nx"] == 8
    ns_only = {"node_selection": {"winner": dict(WINNER, nx=6, ny=12)}}
    assert (wr.tune_report_layout(ns_only)["nx"], wr.tune_report_layout(ns_only)["ny"]) == (6, 12)
    rows = [dict(WINNER, nodes=2, median=0.3), dict(WINNER, nodes=1, median=0.6, nx=4, ny=18),
            dict(WINNER, nodes=1, median=0.4, nx=6, ny=12)]
    lay = wr.tune_report_layout({"node_selection": {"chosen": 1, "layouts": rows}})
    assert (lay["nodes"], lay["nx"], lay["median"]) == (1, 6, 0.4)
    lay = wr.tune_report_layout({"node_selection": {"chosen": 1, "final": 2, "layouts": rows}})
    assert lay["nodes"] == 2
    assert wr.tune_report_layout({"ranking": []}) is None


def test_dry_run_takes_layout_and_rate_from_winner(tmp_path):
    rc, lines = _main(tmp_path, ["--tune-report", _report(tmp_path), "--segment-minutes", "360",
                                 "--dry-run"])
    assert rc == 0
    assert "layout from tune report: N=1 8x9 tiles 2 omp 1 (cores/node 72, 0.5000 s/step)" \
        in lines
    seg = [ln for ln in lines if ln.startswith("  seg 0")][0]
    assert "8x9 t2 omp1" in seg
    assert any("median 0.5000 s/step as the rate" in ln for ln in lines)
    assert not any("WARNING" in ln for ln in lines)


def test_cli_flags_override_report(tmp_path):
    rc, lines = _main(tmp_path, ["--tune-report", _report(tmp_path), "--nx", "6", "--ny", "12",
                                 "--dry-run"])
    assert rc == 0
    head = [ln for ln in lines if ln.startswith("layout from tune report")][0]
    assert "CLI overrides --nx --ny -> N=1 6x12 tiles 2 omp 1" in head
    assert "6x12 t2 omp1" in [ln for ln in lines if ln.startswith("  seg 0")][0]
    assert any("measured for N=1 8x9 tiles 2 omp 1, not this config" in ln for ln in lines)


def test_no_layout_anywhere_is_a_clear_error(tmp_path, capsys):
    rc, _ = _main(tmp_path, ["--dry-run"])
    assert rc == 2
    assert "--nodes, --nx, --ny required" in capsys.readouterr().err
    rc, _ = _main(tmp_path, ["--tune-report", _report(tmp_path, winner=None, node_selection={}),
                             "--dry-run"])
    assert rc == 2
    assert "--nodes, --nx, --ny required" in capsys.readouterr().err


def test_no_ranking_points_and_no_median_raises(tmp_path, capsys):
    rep = _report(tmp_path, winner=dict(WINNER, median=None), node_selection={})
    rc, _ = _main(tmp_path, ["--tune-report", rep, "--dry-run"])
    assert rc == 2
    assert "no winner with a measured median" in capsys.readouterr().err


def test_median_rate_run_replans_from_observed(tmp_path):
    f = Fakes()
    rc, _ = _main(tmp_path, ["--tune-report", _report(tmp_path), "--segment-minutes", "360"],
                  sbatch_runner=f.sbatch, sacct_runner=f.sacct, snapshot_fn=f.snapshot,
                  sleep=lambda s: None)
    assert rc == 0
    rep = json.loads((tmp_path / "out" / "run_report.json").read_text())
    s0 = rep["segments"][0]
    assert (s0["config"]["nx"], s0["config"]["ny"], s0["config"]["numtiles"]) == (8, 9, 2)
    assert s0["predicted_compute_s"] == 0.5 * 360
    obs = [d for d in rep["decisions"] if d["decision"] == "observed_overhead"][0]
    assert obs["compute_replaced"] is True


def test_rank_mapping_notes():
    # P1 winner: 80 ranks x 3 omp on 5 nodes = ppr 16/node x 3 = 48 cores/node, as measured
    assert wr.rank_mapping_notes(wr.make_config(8, 10, 1, 3, 5), 48) == []
    notes = wr.rank_mapping_notes(wr.make_config(8, 10, 1, 3, 6), None)
    assert len(notes) == 1 and "do not divide over 6 nodes" in notes[0]
    notes = wr.rank_mapping_notes(wr.make_config(8, 10, 1, 3, 5), 72)
    assert len(notes) == 1 and "the tune report measured 72" in notes[0]


# ---------------------------------------------------------------------------
# wtune tune: top-level "winner" = the FINAL pick (wtune_tune.final_winner)
# ---------------------------------------------------------------------------
import wtune_tune  # noqa: E402
import test_wtune_tune as tt  # noqa: E402


def _row(nodes, cores, nx, ny, omp=1, numtiles=1, median=0.5, cid="G1"):
    return dict(source="finalist search", nodes=nodes, cores_per_node=cores, ranks=nx * ny,
                omp=omp, nx=nx, ny=ny, numtiles=numtiles, median=median, reps=3, cid=cid)


def _best(nx, ny, omp=1, numtiles=1, cid="G1", median=None):
    return dict(id=cid, ranks=nx * ny, omp=omp, nproc_x=nx, nproc_y=ny, numtiles=numtiles,
                median_s_per_step=median)


def test_final_winner_is_the_matching_layouts_row():
    rows = [_row(4, 72, 16, 18, median=0.30, cid="G7"), _row(5, 72, 18, 20, median=0.28)]
    rep = {"nodes": 4, "cores_used_per_node": 72, "best": _best(16, 18, cid="G7"),
           "node_selection": {"winner": rows[1], "layouts": rows}}
    assert wtune_tune.final_winner(rep) == rows[0]     # the final pick, not the fastest row


def test_final_winner_built_from_ranking_row_when_no_layouts_row():
    rep = {"nodes": 6, "best": _best(12, 12, omp=3, numtiles=3, cid="ANCHOR", median=0.0153),
           "node_selection": {"layouts": [], "core_choice": {"6": {"cores": 72}}}}
    w = wtune_tune.final_winner(rep)
    assert (w["nodes"], w["cores_per_node"], w["nx"], w["ny"], w["omp"], w["numtiles"],
            w["ranks"], w["median"], w["cid"]) == (6, 72, 12, 12, 3, 3, 144, 0.0153, "ANCHOR")
    assert wtune_tune.final_winner({"nodes": 2, "best": _best(8, 9), "pipeline_only": True}) \
        is None


def _check_winner_is_final(report, out_dir):
    w, best = report["winner"], report["best"]
    assert w is not None and w["nodes"] == report["nodes"]
    assert w["cores_per_node"] == report["cores_used_per_node"]
    assert (w["ranks"], w["omp"], w["nx"], w["ny"], w["numtiles"]) == (
        best["ranks"], best["omp"], best["nproc_x"], best["nproc_y"], best["numtiles"])
    with open(os.path.join(out_dir, "tune_report.json")) as f:
        on_disk = json.load(f)
    assert on_disk["winner"] == json.loads(json.dumps(w))
    lay = wr.tune_report_layout(on_disk)
    assert (lay["nodes"], lay["nx"], lay["ny"]) == (w["nodes"], w["nx"], w["ny"])
    return w


def test_pipeline_without_phase_2_writes_the_final_winner(monkeypatch, tmp_path):
    report, env = tt._pipe_run(monkeypatch, tmp_path, planner=lambda planned, T, call: ([], 0.15))
    assert "rebuild_submit" not in env.events
    w = _check_winner_is_final(report, str(tmp_path / "out"))
    assert w["median"] is not None


def test_pipeline_phase_2_overturn_updates_the_winner(monkeypatch, tmp_path):
    class Overturn(tt._PipeEnv):
        """Phase 2 ranks the first (N, c) group -- phase 1's pick -- last."""

        def __init__(self, mp, *a, **k):
            super().__init__(mp, *a, **k)
            fake, calls = wtune_tune.rank_by_predicted_total, {"n": 0}

            def rank(*args):
                out = fake(*args)
                if calls["n"] == 0:
                    for e in out:
                        e["pred_total_s"] += 1e6
                calls["n"] += 1
                return out
            mp.setattr(wtune_tune, "rank_by_predicted_total", rank)

    monkeypatch.setattr(tt, "_PipeEnv", Overturn)
    report, _env = tt._pipe_run(monkeypatch, tmp_path)
    assert report["nodes_note"] == "phase 2 predicted-whole-run winner"
    phase1_n = report["node_selection"]["final"]
    w = _check_winner_is_final(report, str(tmp_path / "out"))
    assert w["nodes"] != phase1_n
