"""wtune run --tune-report with an -O (decision 0059) report: the layout comes from the report's
winner (CLI flags override), the winner's measured median s/step is the rate when the report has
no ranking points, and the ppr rank mapping is checked against what tune measured."""
import json
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from wtune import run as wr  # noqa: E402
from test_wtune_run import Fakes, _make_case, fake_bp  # noqa: E402,F401  (autouse: a fake src/wtune/bp.py for the default ADIOS2 input conversion)

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


def test_dry_run_after_o0_takes_its_unmeasured_layout(tmp_path):
    rep = _report(tmp_path, winner=None, node_selection=None, pipeline_only=True,
                  cores_used_per_node=72,
                  best={"id": "ANCHOR", "ranks": 72, "omp": 1, "numtiles": 1, "dist": "block",
                        "nproc_x": 8, "nproc_y": 9})
    rc, lines = _main(tmp_path, ["--tune-report", rep, "--dry-run"])
    assert rc == 0
    assert "layout from tune report: N=1 8x9 tiles 1 omp 1 (cores/node 72, ? s/step)" in lines
    assert any("-O0 tune report: its layout is used" in ln for ln in lines)
    assert any("s/step fallback (cell-step model" in ln for ln in lines)


def test_no_ranking_points_and_no_median_raises(tmp_path, capsys):
    rep = _report(tmp_path, winner=dict(WINNER, median=None), node_selection={})
    rc, _ = _main(tmp_path, ["--tune-report", rep, "--dry-run"])
    assert rc == 2
    assert "no winner with a measured median" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# wtune tune: top-level "winner" = the FINAL pick (wtune.tune.final_winner)
# ---------------------------------------------------------------------------
import wtune.tune  # noqa: E402
import test_wtune_tune as tt  # noqa: E402


def _row(nodes, cores, nx, ny, omp=1, numtiles=1, median=0.5, cid="G1"):
    return dict(source="finalist search", nodes=nodes, cores_per_node=cores, ranks=nx * ny,
                omp=omp, nx=nx, ny=ny, numtiles=numtiles, median=median, reps=3, cid=cid)


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


def _healthy_preflight(monkeypatch):
    """A fake all-healthy node pre-flight. Without it run() calls the REAL sinfo/srun: on scc,
    inside an allocation, only the test's own node can read tmp_path, so the probe leaves 1
    healthy node -> max_nodes 1 -> one finalist and no phase-2 overturn. node_preflight also
    writes WTUNE_PREFLIGHT_EXCLUDE into os.environ: let monkeypatch restore it."""
    monkeypatch.setenv("WTUNE_PREFLIGHT_EXCLUDE", "")
    return tt.make_health_runner({"work%d" % i: "idle" for i in range(1, 14)})[0]


def test_pipeline_phase_2_overturn_updates_the_winner(monkeypatch, tmp_path):
    class Overturn(tt._PipeEnv):
        """Phase 2 ranks the first (N, c) group -- phase 1's pick -- last."""

        def __init__(self, mp, *a, **k):
            super().__init__(mp, *a, **k)
            fake, calls = wtune.tune.points.rank_by_predicted_total, {"n": 0}

            def rank(*args):
                out = fake(*args)
                if calls["n"] == 0:
                    for e in out:
                        e["pred_total_s"] += 1e6
                calls["n"] += 1
                return out
            mp.setattr(wtune.tune.points, "rank_by_predicted_total", rank)

    monkeypatch.setattr(tt, "_PipeEnv", Overturn)
    report, _env = tt._pipe_run(monkeypatch, tmp_path,
                                preflight_runner=_healthy_preflight(monkeypatch))
    assert report["nodes_note"] == "phase 2 predicted-whole-run winner"
    phase1_n = report["node_selection"]["final"]
    w = _check_winner_is_final(report, str(tmp_path / "out"))
    assert w["nodes"] != phase1_n
