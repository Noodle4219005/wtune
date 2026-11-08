#!/usr/bin/env python3
"""pytest suite for wtune/config.py (the `wtune tune` YAML search-space config loader).

Every test exercises the dict-level API (`resolve_config`, `filter_candidates`,
`candidate_allowed`, `is_fire_case`, ...) directly, so none of them require PyYAML. Only the
handful of tests that actually read a YAML *file* (`load_config`, the `init`/`show`/`validate`
CLI) import `yaml` and `pytest.importorskip("yaml")` first, so this whole module still runs (and
mostly still passes) in an environment where PyYAML is not installed -- it just skips those.

Run with:
  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q test_wtune_config.py
"""
import json

import pytest

from wtune import config as wcfg

MATTHEW_LIKE = dict(e_we=91, e_sn=100, e_vert=45)


def _valid_raw(**overrides):
    """A fully-specified, schema-valid raw dict (mirrors the task's own example config)."""
    raw = dict(
        site="scc", case="matthew", compliance="strict",
        budget=dict(minutes=30, max_nodes=13, nodes="auto", nodes_policy="knee"),
        search=dict(omp=[1, 2, 3, 4, 6, 8, 9, 12], numtiles=[1, 2, 4, 8, 12, 16, 24],
                    rank_distribution=["block", "cyclic"], aspect_max=8,
                    io=dict(enabled=False, io_form_history=[2, 11, 14], target=["nfs", "local"])),
        physics_gray=dict(enabled=False, knobs=dict(radt=dict(multiply=[1.0, 1.5, 2.0]),
                                                      time_step=dict(multiply=[1.0, 1.1, 1.2]))),
        probe=dict(steps=20, confirm_reps=2),
        prediction=dict(io_target="nfs", drift_band_pct=6, measure_r_rad=True),
        exclude=[dict(omp_gt=1, when="fire")],
    )
    raw.update(overrides)
    return raw


def test_out_of_range_numtiles_is_reported_with_the_catalog_bound():
    raw = _valid_raw()
    raw["search"]["numtiles"] = [65]   # numtiles' catalog max=64 (predictor/decomp_space.py:89)
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("search.numtiles" in e and "above" in e for e in errs)


def test_budget_nodes_search_and_search_knobs_validate():
    cfg, errs = wcfg.resolve_config(_valid_raw(budget=dict(
        minutes=10, max_nodes=13, nodes="search", node_objective="cost", deadline_minutes=90,
        min_efficiency=0.8, node_finalists=3, sweep_budget_frac=0.4, proxy_max_nodes=8)))
    assert errs == []
    assert cfg["budget"]["nodes"] == "search" and cfg["budget"]["node_objective"] == "cost"
    assert cfg["budget"]["proxy_max_nodes"] == 8
    for bad in (dict(node_objective="fastest"), dict(deadline_minutes=-1),
                dict(node_finalists=0), dict(sweep_budget_frac=1.5), dict(min_efficiency="x"),
                dict(proxy_max_nodes=0), dict(proxy_max_nodes=2.5)):
        _cfg, errs = wcfg.resolve_config(_valid_raw(budget=dict(minutes=10, nodes="search", **bad)))
        assert any("budget." in e for e in errs), bad


# ---------------------------------------------------------------------------
# shared candidate filter -- against a REAL decomp_space.enumerate_space grid
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def matthew_candidates():
    from wtune.predictor import decomp_space   # predictor/decomp_space.py; wtune.config's own sys.path bootstrap
    candidates, _axis = decomp_space.enumerate_space(MATTHEW_LIKE, total_cores=72, nodes=1,
                                                        hardware_id="scc-gh200")
    return candidates


def test_filter_candidates_removes_disallowed_aspect(matthew_candidates):
    assert any(max(c["nx"] / c["ny"], c["ny"] / c["nx"]) > 2 for c in matthew_candidates)
    search_cfg = dict(aspect_max=2)
    out = wcfg.filter_candidates(matthew_candidates, search_cfg, [], False)
    assert out
    assert all(max(c["nx"] / c["ny"], c["ny"] / c["nx"]) <= 2 for c in out)


def test_filter_candidates_combines_every_restriction_at_once(matthew_candidates):
    search_cfg = dict(omp=[1, 2], numtiles=[1, 2], rank_distribution=["block"], aspect_max=3)
    out = wcfg.filter_candidates(matthew_candidates, search_cfg, [], False)
    assert out
    for c in out:
        assert c["omp"] in (1, 2)
        assert c["numtiles"] in (1, 2)
        assert c["dist"] == "block"
        assert max(c["nx"] / c["ny"], c["ny"] / c["nx"]) <= 3
    assert len(out) < len(matthew_candidates)


# ---------------------------------------------------------------------------
# fire-case detection
# ---------------------------------------------------------------------------
def test_is_fire_case_true_from_cases_json_max_omp():
    assert wcfg.is_fire_case(case_json=dict(max_omp=1)) is True
    assert wcfg.is_fire_case(case_json=dict(max_omp=2)) is False
    assert wcfg.is_fire_case(case_json=dict()) is False


def test_is_fire_case_true_from_namelist_fire_group(tmp_path):
    nl = tmp_path / "namelist.input"
    nl.write_text("&fire\n ifire = 0, 2,\n/\n")
    assert wcfg.is_fire_case(case_json=None, namelist_path=str(nl)) is True


def test_resolve_node_counts_single_int():
    counts = wcfg.resolve_node_counts(dict(nodes=7, max_nodes=13))
    assert counts == [7]


# ---------------------------------------------------------------------------
# YAML-dependent tests -- skip cleanly when PyYAML is not installed in this env
# ---------------------------------------------------------------------------
def test_load_config_missing_pyyaml_gives_a_clear_actionable_error(monkeypatch, tmp_path):
    """Regardless of whether PyYAML happens to be installed in THIS test env, the lazy-import
    error path itself must be exercised: force `_load_yaml_module` to behave as if PyYAML were
    absent and check the message tells the user exactly what to run."""
    def fake_loader():
        raise SystemExit(wcfg._YAML_MISSING_MSG)
    monkeypatch.setattr(wcfg, "_load_yaml_module", fake_loader)
    p = tmp_path / "x.yaml"
    p.write_text("case: matthew\n")
    with pytest.raises(SystemExit) as exc_info:
        wcfg.load_config(str(p))
    assert "pixi add pyyaml" in str(exc_info.value)


def test_cmd_init_writes_a_file_that_validates(tmp_path):
    pytest.importorskip("yaml")
    out = tmp_path / "wtune.yaml"
    rc = wcfg.main(["init", "matthew", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    cfg = wcfg.load_config(str(out))
    assert cfg["case"] == "matthew"


def test_cmd_validate_exits_2_on_a_bad_file(tmp_path, capsys):
    pytest.importorskip("yaml")
    import yaml
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(dict(search=dict(omp=[0]))))
    rc = wcfg.main(["validate", str(p)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


def test_cmd_validate_exits_0_on_a_good_file(tmp_path):
    pytest.importorskip("yaml")
    import yaml
    p = tmp_path / "good.yaml"
    p.write_text(yaml.safe_dump(dict(case="matthew")))
    rc = wcfg.main(["validate", str(p)])
    assert rc == 0


def test_cmd_show_prints_resolved_config_and_candidate_counts(tmp_path, capsys):
    pytest.importorskip("yaml")
    import yaml
    p = tmp_path / "wtune.yaml"
    p.write_text(yaml.safe_dump(dict(case="matthew", budget=dict(nodes=[1, 2]),
                                       search=dict(omp=[1]))))
    rc = wcfg.main(["show", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out.split("\ncandidate counts")[0])
    assert parsed["case"] == "matthew"
    assert "nodes=1" in out and "nodes=2" in out
    assert "legal_total=" in out and "legal_after_search_filter=" in out


def test_cmd_show_reports_when_case_cannot_be_resolved(tmp_path, capsys):
    pytest.importorskip("yaml")
    import yaml
    p = tmp_path / "wtune.yaml"
    p.write_text(yaml.safe_dump(dict()))   # no case, no case_dir
    rc = wcfg.main(["show", str(p)])
    assert rc == 0   # still prints the resolved config; just can't preview counts
    out = capsys.readouterr().out
    assert "cannot preview candidate counts" in out
