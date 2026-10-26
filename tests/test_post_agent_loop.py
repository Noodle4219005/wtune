"""Decision 0078: the agent-in-the-loop half of `wtune post` -- ctx.kit (post/analysis_kit.py),
`wtune post brief` (post/brief.py) and `wtune post check` (post/check.py)."""
import json
import os

import numpy as np
import pytest

from wtune.post import analysis_kit as kit
from wtune.post import check


def test_contingency_scores_match_the_textbook_table():
    # hits 50, misses 20, false alarms 30, correct negatives 900 (Wilks ch. 9 style table)
    obs = np.array([1] * 50 + [1] * 20 + [0] * 30 + [0] * 900, dtype=float)
    fc = np.array([1] * 50 + [0] * 20 + [1] * 30 + [0] * 900, dtype=float)
    s = kit.contingency(obs, fc, 0.5)
    assert (s["hits"], s["misses"], s["false_alarms"], s["correct_negatives"]) == (50, 20, 30, 900)
    assert s["fbias"] == pytest.approx(80 / 70) and s["csi"] == pytest.approx(0.5)
    a_r = 80 * 70 / 1000
    assert s["ets"] == pytest.approx((50 - a_r) / (100 - a_r))


def test_normalized_thresholds_score_a_weak_but_well_placed_field():
    rng = np.random.default_rng(0)
    obs = rng.gamma(1.0, 20.0, size=(40, 40))
    weak = 0.4 * obs                                  # same pattern, 60 % too weak
    fixed = kit.scores_by_threshold(obs, weak, [50.0])[0]
    norm = kit.scores_by_threshold(obs, weak, [90], normalized=True)[0]
    assert norm["ets"] == pytest.approx(1.0) and fixed["ets"] < norm["ets"]
    assert norm["fcst_threshold"] == pytest.approx(0.4 * norm["threshold"])


def test_box_mean_nearest_hovmoller_and_track_error():
    lat, lon = np.meshgrid(np.linspace(21, 26, 51), np.linspace(119, 123, 41), indexing="ij")
    field = lat * 10.0
    assert kit.box_mean(lat, lon, field, [22.0, 23.0, 120.0, 121.0]) == pytest.approx(225.0)
    j, i, d = kit.nearest_index(lat, lon, 23.02, 121.01)
    assert (j, i) == (20, 20) and d < 3.0
    assert kit.match_points(lat, lon, field, [(23.0, 121.0), (40.0, 121.0)], max_km=20)[1][0] is None
    dist, hov = kit.hovmoller([field, field + 1], lat, lon, 22.0, 120.0, 25.0, 120.0, n=4)
    assert hov.shape == (2, 4) and hov[1, 0] - hov[0, 0] == pytest.approx(1.0)
    assert dist[-1] == pytest.approx(333.6, abs=1.0)
    err = kit.track_error([("t0", 24.0, 120.0), ("t1", 25.0, 120.0)], [("t1", 24.0, 120.0)])
    assert err[0][0] == "t1" and err[0][1] == pytest.approx(111.2, abs=0.5)


def test_obs_model_panels_draw_on_the_cwa_scale(tmp_path):
    pytest.importorskip("cartopy")
    lat, lon = np.meshgrid(np.linspace(21, 26, 30), np.linspace(119, 123, 25), indexing="ij")
    rain = 300 * np.exp(-((lat - 23.5) ** 2 + (lon - 120.8) ** 2))
    fig, axes = kit.obs_model_panels((lat, lon, rain), (lat, lon, 0.8 * rain), [1, 300],
                                     "24 h rain")
    fig.savefig(str(tmp_path / "p.png"))
    assert len(axes) == 2 and os.path.getsize(tmp_path / "p.png") > 10000


def _post_dir(tmp_path):
    post = tmp_path / "post" / "1"
    (post / "precip").mkdir(parents=True)
    (post / "precip" / "rain.png").write_bytes(b"png")
    (post / "precip" / "other.png").write_bytes(b"png")
    with open(post / "figures.json", "w") as fh:
        json.dump(dict(schema="wtune-post-figures/1", runs=[], case_info={}, notes=[], figures=[
            dict(id="precip_accum", path="precip/rain.png", missing=False),
            dict(id="terrain_rain", path="precip/other.png", missing=False),
            dict(id="syn_maps", missing=True, dataset="d01", reason="x")]), fh)
    with open(post / "scores.csv", "w") as fh:
        fh.write("threshold,ets,fbias\n50,0.214,0.62\n")
    return post


GOOD = """# Haitang 2017
## 1 Setup
Three domains, 3 km inner nest ([domains](precip/rain.png)).
## 2 Large scale
The monsoon trough ([rain](precip/rain.png)).
## 3 Target variables
Rain over the southwest plains ([map](precip/rain.png)).
## 4 Scores
ETS 0.21 at 50 mm with a bias of 0.62 ([scores](scores.csv)).
## 5 Mechanism
Southwesterly flow lifted by the terrain ([rain](precip/rain.png)).
## 6 Conclusion and limits
Too dry by a third ([scores](scores.csv)).
"""


def test_check_passes_a_cited_assessment_and_a_complete_review(tmp_path):
    post = _post_dir(tmp_path)
    (post / "assessment.md").write_text(GOOD)
    (post / "review.json").write_text(json.dumps(dict(schema="wtune-post-review/1", figures=[
        dict(path="precip/rain.png", verdict="ok", issues=[]),
        dict(path="precip/other.png", verdict="fix", issues=["colour bar saturates"])])))
    assert check.check(str(post)) == ([], [])


def test_check_rejects_uncited_paragraphs_invented_numbers_and_bad_reviews(tmp_path):
    post = _post_dir(tmp_path)
    bad = GOOD.replace("ETS 0.21", "ETS 0.35").replace(
        "Too dry by a third ([scores](scores.csv)).", "Too dry by a third.").replace(
        "## 5 Mechanism", "## 5 Notes").replace("(precip/rain.png)).\n## 2", "(precip/gone.png)).\n## 2")
    (post / "assessment.md").write_text(bad)
    (post / "review.json").write_text(json.dumps(dict(schema="wtune-post-review/1", figures=[
        dict(path="precip/rain.png", verdict="fix", issues=[])])))
    errors, warnings = check.check(str(post))
    text = "\n".join(errors)
    assert "0.35 is not in scores.csv" in text and "0.62" not in text
    assert "cites no figure" in text and "precip/gone.png does not exist" in text
    assert "no section for mechanism" in text and "verdict fix without issues" in text
    assert warnings and "not reviewed" in warnings[0]


def test_brief_lists_figures_with_their_questions(tmp_path):
    from wtune.post import brief
    post = _post_dir(tmp_path)
    doc = brief.build_brief(str(post))
    assert [f["id"] for f in doc["figures"]["drawn"]] == ["precip_accum", "terrain_rain"]
    assert doc["figures"]["drawn"][0]["question"]
    assert doc["figures"]["missing"][0]["id"] == "syn_maps"
    assert [s["step"] for s in doc["agent_steps"]] == ["review", "analysis", "assessment", "check"]
    with pytest.raises(FileNotFoundError, match="figures.json"):
        brief.build_brief(str(tmp_path))
