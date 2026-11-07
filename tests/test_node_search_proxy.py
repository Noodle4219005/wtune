import pytest

import wtune.tune
from wtune.tune import node_search as NS
from wtune.tune import proxy_stage as PS


class _Clock:
    t = 0.0

    def now(self):
        return self.t


def _cfg(cid, nx, ny, omp=1, nt=1):
    return dict(id=cid, ranks=nx * ny, omp=omp, nx=nx, ny=ny, numtiles=nt, dist="block", blocktime="0")


def _ctx(tmp_path, proxy=True, inputs=True):
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(" &domains\n max_dom = 1,\n /\n")
    if inputs:
        (case / "wrfinput_d01").write_text("x")
        (case / "wrfbdy_d01").write_text("x")
    return wtune.tune.Ctx(
        case="karina", case_json={}, namelist_path=str(case / "namelist.input"), nodes=1,
        cores_per_node=144, hardware_id="x", steps=40, seed=1, kappa=0.5, budget_s=3600.0, max_nodes=8,
        out_dir=str(tmp_path / "out"), ledger_dir=str(tmp_path / "l"), env_source="", sbatch_script="",
        sacct_cmd="sacct", poll_interval=0.0, sbatch_runner=lambda c: pytest.fail("no sbatch"),
        sacct_runner=lambda c: "", scancel_runner=lambda j: None, ledger_reader=lambda c, j: [],
        proposer=None, clock=_Clock(), proxy=proxy, wrf_tree="/trees/dmsm" if proxy else "")


ANCHORS = {n: _cfg("ANCHOR", 12 * n, 12) for n in (2, 4, 8)}
RANKING = {2: [(ANCHORS[2], 0.60, "one-node")],
           4: [(_cfg("D0001", 24, 24), 0.30, "band"), (ANCHORS[4], 0.32, "band")],
           8: [(ANCHORS[8], 0.25, "band"), (_cfg("D0001", 32, 36), 0.27, "one-node")]}
REAL = {4: 0.29, 8: 0.31}           # real medians: the proxy's best N (8) loses to N=4


def _patch(monkeypatch, ranking, calls):
    monkeypatch.setattr(NS, "_sweep_anchor", lambda ctx, n: (ANCHORS[n], None) if n in ANCHORS
                        else (None, "no legal decomposition"))

    def rank(ctx, legal):
        calls.append(("rank", legal))
        ctx.owned_jobs += ["901", "902"]
        return ranking
    monkeypatch.setattr(PS, "rank_across_nodes", rank)

    def tune(fctx, adaptive=True, confirm=True):
        calls.append(("tune", fctx.nodes, fctx.proxy_ranked, fctx.anchor_override, adaptive, confirm,
                      fctx.core_budget, fctx.full_space))
        m = REAL[fctx.nodes]
        return {"ANCHOR": dict(cid="ANCHOR", config=fctx.anchor_override, s_values=[m], median=m,
                               lo=m, hi=m, n=1)}, None, "ANCHOR"
    monkeypatch.setattr(NS._rounds, "run_tuning_at", tune)
    monkeypatch.setattr(NS._report, "build_report", lambda ctx, n, note, res, conf, cid, **k: dict(
        best=dict(id="ANCHOR", median_s_per_step=res["ANCHOR"]["median"]), full_run_prediction=None))
    monkeypatch.setattr(NS._prediction, "build_full_run_prediction", lambda *a, **k: dict(stub=True))

    def sweep(*a, **k):
        raise _RealSweep()
    monkeypatch.setattr(NS, "run_node_sweep", sweep)


class _RealSweep(Exception):
    pass


def _search(ctx, level):
    collect = {}
    report = NS.run_node_search(ctx, {}, "time", None, 0.7, 2, 1000, plan=dict(level=level),
                                collect=collect)
    return report, collect


def test_joint_proxy_tunes_the_two_best_n_and_chooses_by_real_medians(tmp_path, monkeypatch):
    calls = []
    _patch(monkeypatch, RANKING, calls)
    ctx = _ctx(tmp_path)
    report, collect = _search(ctx, 2)
    assert calls[0] == ("rank", [2, 4, 8])
    tuned = [c for c in calls if c[0] == "tune"]
    assert [c[1] for c in tuned] == [8, 4]                                  # lowest proxy scores first
    for _k, n, ranked, anchor, adaptive, confirm, cores, full in tuned:
        assert ranked is RANKING[n] and anchor is ANCHORS[n]
        assert (adaptive, confirm, cores, full) == (False, False, 144, False)
    assert report["nodes"] == 4 and collect["final_n"] == 4 and sorted(collect["outcomes"]) == [4, 8]
    assert collect["cores_info"] == {} and collect["sweep"]["results"] == {}
    ns = report["node_selection"]
    assert ns["note"] == "joint proxy search" and ns["finalists"] == [8, 4] and ns["sweep_jobs"] == ["901", "902"]
    assert [(r["nodes"], r["median"], r["fidelity"]) for r in ns["table"]] == [
        (2, 0.60, "one-node"), (4, 0.30, "band"), (8, 0.25, "band")]
    assert ns["table"][1]["layout"] == "24x24_o1_nt1"
    assert {s["nodes"] for s in ns["skipped"]} == {1, 3, 5, 6, 7}
    assert report["cores_used_per_node"] == 144 and report["full_run_prediction"] == dict(stub=True)
    assert ns["winner"]["nodes"] == 4
    assert wtune.tune.render_node_selection_text(ns)


def test_joint_proxy_falls_back_to_the_real_sweep_without_a_ranking(tmp_path, monkeypatch):
    calls = []
    _patch(monkeypatch, {}, calls)
    ctx = _ctx(tmp_path)
    assert NS._run_node_search_proxy(ctx, "time", None, 0.7, 1000, dict(level=2), lambda: 2, {}) is None
    with pytest.raises(_RealSweep):
        _search(ctx, 2)
    assert not [c for c in calls if c[0] == "tune"]


@pytest.mark.parametrize("level,proxy,inputs", [(1, True, True), (3, True, True), (2, False, True),
                                                (2, True, False)])
def test_real_sweep_at_o1_o3_proxy_off_or_without_inputs(tmp_path, monkeypatch, level, proxy, inputs):
    calls = []
    _patch(monkeypatch, RANKING, calls)
    with pytest.raises(_RealSweep):
        _search(_ctx(tmp_path, proxy=proxy, inputs=inputs), level)
    assert calls == []


def test_no_plan_means_the_real_sweep(tmp_path, monkeypatch):
    calls = []
    _patch(monkeypatch, RANKING, calls)
    with pytest.raises(_RealSweep):
        NS.run_node_search(_ctx(tmp_path), {}, "time", None, 0.7, 2, 1000)
    assert calls == []
