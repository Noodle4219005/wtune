# RETIRED 2026-10-10: the recommend.py tests cut from tests/predictor/test_verify_obs.py.
# Not runnable as is.
import pytest


# --------------------------------------------------------------------------------------
# how recommend.py's accuracy objective consumes this
# --------------------------------------------------------------------------------------

def _candidate(template, tts=-5.0, nd=0.02):
    return {"template": template, "family": "radiation", "status": "ok",
            "predicted_time_to_solution_pct": tts, "predicted_fidelity_nd": nd,
            "predicted_fidelity_label": "SMALL-DRIFT", "time_to_solution_interval_pct": None}


def test_recommend_falls_back_to_model_vs_model_without_observations():
    """No obs_scores -> byte-for-byte the pre-existing behaviour: nd, split by direction."""
    recommend = pytest.importorskip("recommend")
    view = recommend.build_accuracy_view([_candidate("P01"), _candidate("P02", nd=0.05)], [])
    assert view["judged_against"] == recommend.MODEL_ERROR
    assert view["ranked_observed"] == []
    assert [e["template"] for e in view["ranked_cost_levers"]] == ["P01", "P02"]
    assert all(e["fidelity_error_kind"] == recommend.MODEL_ERROR for e in view["entries"])


def test_recommend_prefers_the_observation_score_over_the_regression():
    recommend = pytest.importorskip("recommend")
    obs = {"P02": (46.3, "measured vs observations (track, ...)")}
    view = recommend.build_accuracy_view(
        [_candidate("P01"), _candidate("P02", nd=0.05)], [], obs_scores=obs,
        observations={"kind": "track", "path": "/x/hurdat2.txt"})
    assert view["judged_against"] == recommend.OBSERVED_ERROR
    assert [e["template"] for e in view["ranked_observed"]] == ["P02"]
    entry = view["ranked_observed"][0]
    assert entry["fidelity_error"] == pytest.approx(46.3)       # the obs score, not nd=0.05
    assert entry["fidelity_error_kind"] == recommend.OBSERVED_ERROR
    assert "observations" in entry["fidelity_error_source"]
    # the un-scored candidate still goes through the old path, in its own block
    assert [e["template"] for e in view["ranked_cost_levers"]] == ["P01"]


def test_the_chaos_floor_never_suppresses_an_observation_score():
    """floor_ratio is a model-vs-model concept; a real skill gain must not be filtered by it."""
    recommend = pytest.importorskip("recommend")
    rows = []          # no measured floor for P01; inject one directly instead
    view = recommend.build_accuracy_view(
        [_candidate("P01")], rows, obs_scores={"P01": (2.24, "observations")},
        observations={"kind": "station", "path": "/x/obs.csv"})
    entry = view["entries"][0]
    assert entry["below_floor"] is False
    assert entry["pareto_excluded_reason"] != "below the chaos floor"


def test_observation_scores_for_case_is_empty_without_a_run_or_a_block():
    recommend = pytest.importorskip("recommend")
    assert recommend.observation_scores_for_case("sc26p2", None) == {}
    assert recommend.observation_scores_for_case("sc26p2", {}) == {}
    fake_cases = {"nocase": {"dt": 72}}
    assert recommend.case_observations("nocase", fake_cases) is None
    assert recommend.case_observations("absent", fake_cases) is None

