"""pytest suite for wtune_nodes.py (pure node-count selection from measured s/step)."""
import pytest

import wtune_nodes as wn

NONMONO = {1: [1.0, 1.02], 2: [0.52, 0.53], 4: [0.30, 0.31], 8: [0.33, 0.34]}


def test_candidates_sorted_unique_capped():
    assert wn.candidate_node_counts([8, 1, 4, 4, 13, 16, 0], 13) == [1, 4, 8, 13]


def test_sweep_waves_max13():
    assert wn.sweep_waves([1, 2, 4, 8], 13) == [[1, 2, 4], [8]]
    assert wn.sweep_waves([8, 13], 13) == [[8], [13]]
    assert wn.sweep_waves([16], 13) == [[16]]          # N > max still gets its own wave


def test_summarize_fields_and_skips():
    rows = wn.summarize({**NONMONO, 16: [], 32: None})
    assert [r["nodes"] for r in rows] == [1, 2, 4, 8]
    r1, r2 = rows[0], rows[1]
    assert r1["median"] == pytest.approx(1.01) and r1["n"] == 2
    assert r1["spread"] == pytest.approx(0.02 / 1.01)
    assert r1["speedup"] == 1.0 and r1["efficiency"] == 1.0
    assert r2["speedup"] == pytest.approx(1.01 / 0.525)
    assert r2["efficiency"] == pytest.approx(1.01 / 0.525 / 2)
    assert wn.summarize({4: [0.3]})[0]["spread"] == 0.0
    assert wn.summarize({}) == []


def test_time_nonmonotone_picks_4_and_stop_early():
    rows = wn.summarize(NONMONO)
    assert wn.choose(rows, "time")[0] == 4
    assert wn.stop_early(rows) is True
    assert wn.stop_early(wn.summarize({n: v for n, v in NONMONO.items() if n <= 4})) is False
    assert wn.stop_early(wn.summarize({1: [1.0]})) is False


def test_time_tie_prefers_smaller_n():
    rows = wn.summarize({4: [0.300], 8: [0.295]})
    n, note = wn.choose(rows, "time")
    assert n == 4 and "objective=time" in note


def test_cost_unconstrained_and_deadline():
    rows = wn.summarize(NONMONO)
    # 100 steps: N=1 wall 101s nh .028; N=2 52.5s nh .029; N=4 30.5s nh .034 -> N=1
    assert wn.choose(rows, "cost", total_steps=100)[0] == 1
    # deadline 60s excludes N=1 (101s); N=2 (52.5s) is the cheapest feasible
    n, note = wn.choose(rows, "cost", total_steps=100, deadline_s=60.0)
    assert n == 2 and "node_hours" in note
    # overhead_s counts into wall
    assert wn.choose(rows, "cost", total_steps=100, deadline_s=60.0, overhead_s=10.0)[0] == 4


def test_cost_none_feasible_picks_fastest_and_says_so():
    rows = wn.summarize(NONMONO)
    n, note = wn.choose(rows, "cost", total_steps=100, deadline_s=1.0)
    assert n == 4 and "NO N meets deadline" in note


def test_knee():
    rows = wn.summarize({1: [1.0], 2: [0.55], 4: [0.35], 8: [0.30]})
    # eff: 1, .909, .714, .417 -> largest >= 0.7 is 4
    assert wn.choose(rows, "knee")[0] == 4
    assert wn.choose(rows, "knee", min_efficiency=0.9)[0] == 2
    # nothing but N_min qualifies -> N_min trivially has eff 1; use a table starting above 0.7
    assert wn.choose(rows, "knee", min_efficiency=1.5)[0] == 1


def test_table_columns():
    rows = wn.summarize({1: [1.0], 2: [0.5]})
    t = wn.table(rows, total_steps=3600, overhead_s=0.0)
    assert t[1]["wall_s"] == pytest.approx(1800.0)
    assert t[1]["node_hours"] == pytest.approx(1.0)
    t0 = wn.table(rows)
    assert t0[0]["wall_s"] is None and t0[0]["node_hours"] is None


def test_finalists():
    rows = wn.summarize(NONMONO)
    # time -> 4 (0.305); 8 (0.335) is 9.8% slower -> within 10%; 2 (0.525) is not
    assert wn.finalists(rows, "time", k=2) == [4, 8]
    assert wn.finalists(rows, "time", k=1) == [4]
    assert wn.finalists(rows, "time", k=3) == [4, 8]
    assert wn.finalists(rows, "time", k=2, within=0.01) == [4]


def test_value_errors():
    rows = wn.summarize(NONMONO)
    with pytest.raises(ValueError):
        wn.choose(rows, "bogus")
    with pytest.raises(ValueError):
        wn.choose([], "time")
    with pytest.raises(ValueError):
        wn.choose(rows, "cost")                       # total_steps missing


def test_seed_nodes_and_refine_targets():
    legal = list(range(1, 14))
    assert wn.seed_nodes(legal, 13) == [1, 2, 4, 8, 13]
    assert wn.seed_nodes([3, 5, 6], 13) == [3]                    # no seed legal -> smallest legal
    known = {1, 2, 4, 8, 13}
    assert wn.refine_targets(4, known, legal) == [3, 5]           # nearest legal each side
    assert wn.refine_targets(4, known | {3, 5}, legal, attempted={3, 5}) == []
    assert wn.refine_targets(13, known, legal) == [12]            # open upper side: nothing above
    assert wn.refine_targets(2, known, [1, 2, 4], attempted=()) == []   # 3 illegal
    assert wn.bracket(4, known) == (2, 8) and wn.bracket(13, known) == (8, None)


def test_default_core_options_and_choose_cores():
    assert wn.default_core_options(72) == [72, 70, 64, 48, 36]
    assert wn.default_core_options(112) == [112, 110, 100, 75, 56]
    rows = [dict(cores=72, median=0.300), dict(cores=64, median=0.270), dict(cores=48, median=0.40)]
    assert wn.choose_cores(rows)[0] == 64                         # 64 faster beyond noise
    tie = [dict(cores=72, median=0.300), dict(cores=64, median=0.296), dict(cores=48, median=0.40)]
    assert wn.choose_cores(tie)[0] == 72                          # within noise -> fuller c
    with pytest.raises(ValueError):
        wn.choose_cores([])
