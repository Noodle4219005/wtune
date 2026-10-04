"""Phase-2 short-run guard: a plan costing >= the real run is refused when pipe.short_run is set."""
import types

from wtune_tune import pipeline as P
from wtune_tune.pipeline_state import ShortRun
from wtune_tune.common import UNSET


def _st(monkeypatch, estimates, short):
    plans = [("full", [], []), ("winner", [], [])]
    monkeypatch.setattr(P, "phase2_plans", lambda *a, **k: iter(plans))
    monkeypatch.setattr(P, "_phase2_groups", lambda lays, fw: [(1, 1, {}, [], {}, [])])
    est = iter(estimates)
    monkeypatch.setattr(P, "fit_stage2", lambda *a, **k: (None, next(est)))
    monkeypatch.setattr(P, "_phase2_gctx", lambda st, n, c: None)
    monkeypatch.setattr(P, "choose_point_mode", lambda *a: None)
    pipe = types.SimpleNamespace(short_run=short)
    return types.SimpleNamespace(pipe=pipe, rebuilt=None, sample_points=None, nl=None)


def test_short_run_rejects_plan_not_cheaper_than_real_run(monkeypatch):
    st = _st(monkeypatch, [500.0, 300.0], ShortRun(message="m", real_run_s=400.0))
    tried, chosen = P._phase2_choose_plan(st, [], False, 1000.0, False)
    assert [t["estimate_s"] for t in tried] == [500.0, 300.0]
    assert chosen[0] == "winner"


def test_short_run_none_fits_when_all_cost_more_than_run(monkeypatch):
    st = _st(monkeypatch, [500.0, 450.0], ShortRun(message="m", real_run_s=400.0))
    assert P._phase2_choose_plan(st, [], False, 1000.0, False)[1] is None


def test_without_short_run_behaviour_unchanged(monkeypatch):
    st = _st(monkeypatch, [500.0, 300.0], UNSET)
    tried, chosen = P._phase2_choose_plan(st, [], False, 1000.0, False)
    assert len(tried) == 1 and chosen[0] == "full"
