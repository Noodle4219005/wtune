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
        dict(path="precip/other.png", verdict="fix", issues=["colour bar saturates"],
             action=dict(kind="wtune_defect", text="rain scale tops at 203.2 mm"))])))
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


def _preview_dir(tmp_path, project):
    post = project / "post" / "preview"
    (post / "figures").mkdir(parents=True)
    (post / "figures" / "common_domain_map.png").write_bytes(b"png")
    with open(post / "figures.json", "w") as fh:
        json.dump(dict(schema="wtune-post-figures/1", kind="preview", runs=[], case_info={},
                       notes=[], figures=[dict(id="common_domain_map", missing=False,
                                               path="figures/common_domain_map.png")]), fh)
    return post


def _project(tmp_path):
    project = tmp_path / "p"
    (project / "analysis").mkdir(parents=True)
    (project / "project.yaml").write_text("name: p\npost:\n  types: auto\n")
    return project


def test_a_preview_review_needs_a_plan_and_valid_fix_actions(tmp_path):
    # user 2026-10-27: the agent gives drawing instructions and fixes, the preview's too
    project = _project(tmp_path)
    post = _preview_dir(tmp_path, project)
    review = dict(schema="wtune-post-review/1", figures=[dict(
        path="figures/common_domain_map.png", verdict="fix", issues=["d03 misses SW Taiwan"],
        action=dict(kind="project_set", args=["--post-focus", "21.8,25.4,119.9,122.1"]))],
        plan=[dict(figure="12 h rain, gauges vs model", why="paper Fig. 4",
                   how="CWA scale, 4 panels",
                   action=dict(kind="analysis", script="analysis/rain12h.py"))])
    (post / "review.json").write_text(json.dumps(review))
    errors, warnings = check.check(str(post), project_dir=str(project))
    assert errors == [] and any("rain12h.py does not exist yet" in w for w in warnings)
    review["plan"] = []
    review["figures"][0]["action"] = dict(kind="project_set", args=["--max-nodes", "8"])
    (post / "review.json").write_text(json.dumps(review))
    text = "\n".join(check.check(str(post), project_dir=str(project))[0])
    assert "needs `plan`" in text and "project_set args must use" in text
    review["figures"][0]["action"] = dict(kind="project_set",
                                          args=["--post-focus", "25.4,21.8,119.9,122.1"])
    review["figures"].append(dict(path="figures/common_domain_map.png", verdict="fix",
                                  issues=["x"]))
    (post / "review.json").write_text(json.dumps(review))
    text = "\n".join(check.check(str(post), project_dir=str(project))[0])
    assert "rejected" in text and "verdict fix without an action" in text


def test_project_set_drawing_options(tmp_path):
    from wtune import project as wp
    project = _project(tmp_path)
    assert wp.main(["set", "--project", str(project), "--post-focus", "21.8,25.4,119.9,122.1",
                    "--cross-section", "moisture=22,119.5,23.5,121.5"]) == 0
    doc = wp.load_project(str(project))
    assert doc["post"]["focus"] == [21.8, 25.4, 119.9, 122.1] and doc["post"]["types"] == "auto"
    assert doc["cross_sections"]["moisture"] == dict(lat0=22.0, lon0=119.5, lat1=23.5, lon1=121.5)
    assert wp.main(["set", "--project", str(project), "--post-focus", "none"]) == 0
    assert wp.load_project(str(project))["post"]["focus"] is None
    assert wp.main(["set", "--project", str(project), "--cross-section", "x=1,2,3,4"]) == 1


def test_preview_writes_its_figure_index(tmp_path):
    pytest.importorskip("cartopy")
    import shutil
    from wtune.post import brief, preview
    here = os.path.join(os.path.dirname(__file__), "fixtures", "namelists", "haitang2017")
    project = tmp_path / "p"
    (project / "case").mkdir(parents=True)
    for n in ("namelist.wps", "namelist.input"):
        shutil.copy(os.path.join(here, n), project / "case" / n)
    out = project / "post" / "preview"
    preview.run_preview(str(project), {}, str(out))
    index = json.load(open(out / "figures.json"))
    assert index["kind"] == "preview"
    assert [f["path"] for f in index["figures"]] == ["common_domain_map.png"]
    doc = brief.build_brief(str(out))
    assert doc["kind"] == "preview" and [s["step"] for s in doc["agent_steps"]][:2] == [
        "review", "plan"] and doc["figures"]["after_the_run"]


def test_projects_point_agents_at_the_instructions(tmp_path):
    from wtune import project as wp
    path = wp.init_project(str(tmp_path / "new"))
    text = open(os.path.join(path, "AGENTS.md")).read()
    for name in ("AGENT.md", "wtune-operate", "wtune-event", "wtune-post", "bin/wtune"):
        assert name in text
    old = _project(tmp_path)
    assert wp.main(["set", "--project", str(old), "--agents-md"]) == 0
    assert os.path.isfile(old / "AGENTS.md")


def test_a_bdeck_only_storm_is_a_project_setting(tmp_path):
    # haitang2017 (WP122017, JMA best track only): preview drew no track and detected no tc
    from wtune import project as wp
    from wtune.post import atcf, cli as post_cli
    from datetime import datetime
    project = _project(tmp_path)
    bdeck = tmp_path / "bwp122017.dat"
    bdeck.write_text("".join(atcf.format_line("WP", "12", datetime(2017, 7, 31, h), "BEST", 0,
                                              20.0 + h / 6, 117.0 + h / 12, 45, 985) + "\n"
                             for h in (0, 6, 12, 18)))
    assert wp.main(["set", "--project", str(project), "--storm", "WP122017",
                    "--bdeck", str(bdeck)]) == 0
    doc = wp.load_project(str(project))
    assert wp.project_storm(doc) == ("WP122017", None) and wp.project_bdeck(doc) == str(bdeck)
    assert post_cli._project_bdeck(None, str(project), None, "WP122017") == str(bdeck)
    assert post_cli._project_bdeck("x.dat", str(project), None, "WP122017") == "x.dat"
    pytest.importorskip("cartopy")
    from wtune.post import preview
    f = preview._best_track(str(tmp_path / "out"), "WP122017", None,
                            (datetime(2017, 7, 31), datetime(2017, 8, 1)), bdeck=str(bdeck))
    assert os.path.getsize(f["path"]) > 10000 and f["source"].startswith("b-deck")


def test_project_set_accepts_the_agent_drawing_flags_in_a_review(tmp_path):
    project = _project(tmp_path)
    post = _preview_dir(tmp_path, project)
    review = dict(schema="wtune-post-review/1", figures=[dict(
        path="figures/common_domain_map.png", verdict="fix", issues=["d03 has no terrain"],
        action=dict(kind="project_set", args=["--post-map-domains", "d01,d03",
                                              "--post-figures-off", "timeseries_slp"]))],
        plan=[dict(figure="d03 terrain", how="zoomed map", action=dict(
            kind="project_set", args=["--post-times=2019-01-30T06"]))])
    (post / "review.json").write_text(json.dumps(review))
    assert check.check(str(post), project_dir=str(project))[0] == []


def test_check_warns_when_a_station_in_the_stats_is_never_named(tmp_path):
    from wtune.post import check as chk
    (tmp_path / "station_stats.csv").write_text("station,domain,variable,n\nSBA,d01,t2,3\nIZA,d01,t2,3\n")
    (tmp_path / chk.ASSESSMENT_FILE).write_text("SBA is warm.\n")
    w = []
    chk.check_coverage(str(tmp_path), w)
    assert len(w) == 1 and "IZA" in w[0] and "SBA" not in w[0].split("never named:")[1]
    (tmp_path / chk.ASSESSMENT_FILE).write_text("SBA and IZA.\n")
    w = []
    chk.check_coverage(str(tmp_path), w)
    assert w == []
