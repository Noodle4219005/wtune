import dataclasses
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
RANKING = {2: [(ANCHORS[2], 0.262, "one-node")],
           4: [(_cfg("D0001", 24, 24), 0.26, "band"), (ANCHORS[4], 0.32, "band")],
           8: [(ANCHORS[8], 0.25, "band"), (_cfg("D0001", 32, 36), 0.20, "one-node")]}


def _patch(monkeypatch, ranking, calls, consistent=None, extra_cores=None):
    extra_cores = extra_cores if extra_cores is not None else {}
    monkeypatch.setattr(NS, "_sweep_anchor", lambda ctx, n: (ANCHORS[n], None) if n in ANCHORS
                        else (None, "no legal decomposition"))

    def rank(ctx, legal, anchor_only_below=0, leaders=3, cores=None, cores_by_n=None, **kw):
        if kw.get("out_name") == "proxy-cores":    # the finalists' cores/node sweep
            calls.append(("cores", legal, cores_by_n))
            return extra_cores.get("ranking", {}), {}
        calls.append(("rank", legal))
        calls.append(("below", anchor_only_below))
        calls.append(("leaders", leaders))
        calls.append(("cores_by_n", cores_by_n))
        ctx.owned_jobs += ["901", "902"]
        return ({n: list(rows) for n, rows in ranking.items()},     # the search merges into it
                dict(consistent) if consistent is not None else {n: True for n in ranking})
    monkeypatch.setattr(PS, "rank_across_nodes", rank)
    monkeypatch.setattr(NS._rounds, "run_tuning_at", lambda *a, **k: pytest.fail("no real probe"))

    def report(ctx, n, note, res, conf, cid, **k):
        calls.append(("report", n, sorted(res), cid))
        b = min(res, key=lambda c: res[c]["median"])
        return dict(best=dict(id=b, median_s_per_step=res[b]["median"]), full_run_prediction=None)
    monkeypatch.setattr(NS._report, "build_report", report)
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


def test_joint_proxy_picks_by_the_proxy_scores_with_no_real_probe(tmp_path, monkeypatch):
    calls = []
    _patch(monkeypatch, RANKING, calls)
    ctx = _ctx(tmp_path)
    report, collect = _search(ctx, 2)
    assert calls[0] == ("rank", [2, 4, 8]) and calls[1] == ("below", 4)   # objective time: N < 8/2
    assert calls[2] == ("leaders", None)              # O2: PROXY_JOINT_LEADERS
    # only the leader's fidelity ranks: N=8's one-node 0.20 reads low and must not win
    assert [c for c in calls if c[0] == "report"] == [("report", 8, ["ANCHOR"], "ANCHOR"),
                                                       ("report", 4, ["ANCHOR", "D0001"], "ANCHOR")]
    assert report["nodes"] == 8 and collect["final_n"] == 8 and sorted(collect["outcomes"]) == [4, 8]
    assert collect["outcomes"][4].best_median == 0.26 and collect["outcomes"][8].cores == 144
    ns = report["node_selection"]
    assert ns["note"] == "joint proxy search" and ns["finalists"] == [8, 4] and ns["sweep_jobs"] == ["901", "902"]
    assert ns["position_consistent"] == {"2": True, "4": True, "8": True}
    assert [(r["nodes"], r["median"], r["fidelity"]) for r in ns["table"]] == [
        (2, 0.262, "one-node"), (4, 0.26, "band"), (8, 0.25, "band")]
    assert ns["table"][1]["layout"] == "24x24_o1_nt1"
    assert {s["nodes"] for s in ns["skipped"]} == {1, 3, 5, 6, 7}
    assert report["cores_used_per_node"] == 144 and report["full_run_prediction"] == dict(stub=True)
    assert ns["winner"]["nodes"] == 8 and ns["winner"]["median"] == 0.25
    assert ctx.owned_jobs == ["901", "902"]                             # nothing but the proxy jobs
    text = wtune.tune.render_node_selection_text(ns)
    assert "node selection (joint proxy" in text and "measured, ANCHOR" not in text
    assert "proxy-timed layouts" in text and "proxy band" in text
    assert "band     96x12_o1_nt1" in text and "one-node 24x24" not in text   # fidelity + layout per N


def test_inconsistent_crop_positions_at_the_best_n_add_one_finalist(tmp_path, monkeypatch):
    calls = []
    ranking = {n: ([(ANCHORS[2], 0.262, "full")] if n == 2 else RANKING[n]) for n in RANKING}
    _patch(monkeypatch, ranking, calls, consistent={2: True, 4: True, 8: False})
    report, collect = _search(_ctx(tmp_path), 2)
    ns = report["node_selection"]
    assert ns["finalists"] == [8, 4, 2] and ns["position_consistent"]["8"] is False
    assert report["nodes"] == 8 and sorted(collect["outcomes"]) == [2, 4, 8]
    calls.clear()
    _patch(monkeypatch, RANKING, calls, consistent={2: True, 4: False, 8: True})
    (tmp_path / "b").mkdir()
    report, _c = _search(_ctx(tmp_path / "b"), 2)
    assert report["node_selection"]["finalists"] == [8, 4]          # only the best N's flag counts


# cost: N=4 at 0.265 s/step costs 1.06 node-s/step, N=8 at 0.25 2.0 -- N=4 leads, as the real sweep's
# wtune.nodes.finalists picks it (the proxy path took the two fastest whatever the objective)
@pytest.mark.parametrize("objective,lead,finals", [("time", "band", [8]), ("time", "one-node", [4]),
                                                   ("cost", "band", [4, 8])])
def test_a_finalist_beyond_the_margin_behind_an_accurate_leader_is_dropped(tmp_path, monkeypatch,
                                                                           objective, lead, finals):
    calls = []
    ranking = {2: [(ANCHORS[2], 0.40, "one-node")], 4: [(ANCHORS[4], 0.25 * 1.06, "band")],
               8: [(ANCHORS[8], 0.25, lead)]}
    _patch(monkeypatch, ranking, calls)
    report = NS.run_node_search(_ctx(tmp_path), {}, objective, None, 0.7, 2, 1000, plan=dict(level=2),
                                collect={})
    assert report["node_selection"]["finalists"] == finals


def test_joint_proxy_falls_back_to_the_real_sweep_without_a_ranking(tmp_path, monkeypatch):
    calls = []
    _patch(monkeypatch, {}, calls)
    ctx = _ctx(tmp_path)
    assert NS._run_node_search_proxy(ctx, "time", None, 0.7, 1000, {}) is None
    with pytest.raises(_RealSweep):
        _search(ctx, 2)
    assert not [c for c in calls if c[0] == "report"]


def test_the_finalists_are_timed_at_the_other_core_counts_and_a_faster_one_wins(tmp_path, monkeypatch):
    # Sundowner on f1-arm: 4 nodes x 54 of a Grace's 72 cores 1.040 s/step, x 72 1.123; the
    # full-node proxy alone could not see it
    calls = []
    fewer = {8: [(_cfg("S0001", 36, 24), 0.22, "band")], 4: [(_cfg("S0002", 24, 18), 0.30, "band")]}
    _patch(monkeypatch, RANKING, calls, extra_cores=dict(ranking=fewer))
    ctx = _ctx(tmp_path)
    report, collect = _search(ctx, 2)
    sweep = [c for c in calls if c[0] == "cores"]
    options = [142, 128, 108, 96, 72]                              # 144's options but 144
    assert sweep == [("cores", [4, 8], {8: options, 4: options})]
    assert collect["outcomes"][8].cores == 108 and collect["outcomes"][8].best_median == 0.22
    assert collect["outcomes"][4].cores == 144                     # 0.30 at 108 is slower than 0.26
    ns = report["node_selection"]
    assert ns["core_choice"]["8"]["cores"] == 108 and ns["core_options"] == [144, 142, 128, 108, 96, 72]
    assert [(r["cores"], r["median"], r["chosen"]) for r in ns["core_tables"]["8"]] == [
        (144, 0.25, False), (108, 0.22, True)]
    assert report["cores_used_per_node"] == 108
    assert "joint proxy: each count's fastest layout, band" in wtune.tune.render_node_selection_text(ns)


def test_an_n_with_no_full_core_layout_is_timed_at_fewer_cores_not_dropped(tmp_path, monkeypatch):
    calls = []
    _patch(monkeypatch, RANKING, calls)
    # N=6 has no legal layout at 144 cores per node but has one at 108
    monkeypatch.setattr(NS, "_sweep_anchor", lambda ctx, n: (ANCHORS.get(n), None) if n in ANCHORS
                        else ((_cfg("ANCHOR", 27, 24), None) if n == 6 and ctx.core_budget == 108
                              else (None, "no legal decomposition")))
    ctx = _ctx(tmp_path)
    _search(ctx, 2)
    assert ("rank", [2, 4, 6, 8]) in calls and ("cores_by_n", {6: [108]}) in calls


def test_an_n_that_does_not_fit_in_memory_at_full_cores_is_timed_at_fewer_or_skipped(tmp_path, monkeypatch):
    calls = []
    _patch(monkeypatch, RANKING, calls)
    monkeypatch.setattr(NS, "_sweep_anchor", lambda ctx, n: (_cfg("ANCHOR", 12 * n, (ctx.core_budget or 144) // 12), None)
                        if n in ANCHORS else (None, "no legal decomposition"))

    class Mem:                                # N=2 fits only at <= 96 ranks/node, N=4 never
        def why_not(self, n, rpn):
            return "memory: too big" if n == 4 or (n == 2 and rpn > 96) else None
    monkeypatch.setattr(NS._memory, "model_for", lambda cj, gib: Mem())
    ctx = _ctx(tmp_path)
    report, _c = _search(ctx, 2)
    assert ("rank", [2, 8]) in calls and ("cores_by_n", {2: [96]}) in calls
    assert {"nodes": 4, "reason": "memory: too big"} in report["node_selection"]["skipped"]


def test_cyclic_only_search_runs_the_real_sweep_and_unused_flags_are_named(tmp_path, monkeypatch, capsys):
    calls = []
    _patch(monkeypatch, RANKING, calls)
    ctx = dataclasses.replace(_ctx(tmp_path), search_cfg={"rank_distribution": ["cyclic"]})
    assert NS._run_node_search_proxy(ctx, "time", None, 0.7, 1000, {}) is None
    assert "has no block layout" in capsys.readouterr().out and not calls
    (tmp_path / "b").mkdir()
    ctx = dataclasses.replace(_ctx(tmp_path / "b"), node_seeds="all", hotspot_fill=False)
    NS.run_node_search(ctx, {}, "time", None, 0.7, 2, 1000, plan=dict(level=2), collect={},
                       sweep_budget_frac=0.5)
    out = capsys.readouterr().out
    assert "does not use --node-seeds all, --hotspot-fill off, --sweep-budget-frac 0.5" in out


def test_the_pipeline_callbacks_see_the_proxy_scores(tmp_path, monkeypatch):
    calls, seen = [], {}
    _patch(monkeypatch, RANKING, calls)
    NS.run_node_search(_ctx(tmp_path), {}, "time", None, 0.7, 2, 1000, plan=dict(level=2), collect={},
                       on_wave=lambda meas: seen.update(wave=meas),
                       on_sweep=lambda rows, sw: seen.update(rows=sorted((r["nodes"], r["median"]) for r in rows)))
    assert seen["wave"] == {8: [0.25], 4: [0.26]} and seen["rows"] == [(4, 0.26), (8, 0.25)]
