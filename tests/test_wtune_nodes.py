"""pytest suite for wtune/nodes.py (pure node-count selection from measured s/step)."""
import pytest

from wtune import nodes as wn

NONMONO = {1: [1.0, 1.02], 2: [0.52, 0.53], 4: [0.30, 0.31], 8: [0.33, 0.34]}


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


def test_value_errors():
    rows = wn.summarize(NONMONO)
    with pytest.raises(ValueError):
        wn.choose(rows, "bogus")
    with pytest.raises(ValueError):
        wn.choose([], "time")
    with pytest.raises(ValueError):
        wn.choose(rows, "cost")                       # total_steps missing


def test_default_core_options_and_choose_cores():
    assert wn.default_core_options(72) == [72, 70, 64, 54, 48, 36]
    assert wn.default_core_options(112) == [112, 110, 100, 84, 75, 56]
    rows = [dict(cores=72, median=0.300), dict(cores=64, median=0.270), dict(cores=48, median=0.40)]
    assert wn.choose_cores(rows)[0] == 64                         # 64 faster beyond noise
    tie = [dict(cores=72, median=0.300), dict(cores=64, median=0.296), dict(cores=48, median=0.40)]
    assert wn.choose_cores(tie)[0] == 72                          # within noise -> fuller c
    with pytest.raises(ValueError):
        wn.choose_cores([])
