#!/usr/bin/env python3
"""pytest suite for wtune_config.py (the `wtune tune` YAML search-space config loader).

Every test exercises the dict-level API (`resolve_config`, `filter_candidates`,
`candidate_allowed`, `is_fire_case`, ...) directly, so none of them require PyYAML. Only the
handful of tests that actually read a YAML *file* (`load_config`, the `init`/`show`/`validate`
CLI) import `yaml` and `pytest.importorskip("yaml")` first, so this whole module still runs (and
mostly still passes) in an environment where PyYAML is not installed -- it just skips those.

Run with:
  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q test_wtune_config.py
"""
import copy
import json
import os

import pytest

import wtune_config as wcfg

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


# ---------------------------------------------------------------------------
# load + defaults merge
# ---------------------------------------------------------------------------
def test_empty_config_resolves_to_pure_defaults():
    cfg, errs = wcfg.resolve_config({})
    assert errs == []
    assert cfg == wcfg.DEFAULTS
    assert cfg is not wcfg.DEFAULTS   # never hand back the live DEFAULTS object


def test_full_valid_example_resolves_cleanly():
    cfg, errs = wcfg.resolve_config(_valid_raw())
    assert errs == [], errs
    assert cfg["case"] == "matthew"
    assert cfg["search"]["omp"] == [1, 2, 3, 4, 6, 8, 9, 12]
    assert cfg["exclude"] == [dict(omp_gt=1, when="fire")]


def test_partial_config_merges_over_defaults_not_replaces_them():
    cfg, errs = wcfg.resolve_config(dict(case="matthew", budget=dict(minutes=45)))
    assert errs == []
    assert cfg["budget"]["minutes"] == 45
    assert cfg["budget"]["max_nodes"] == wcfg.DEFAULTS["budget"]["max_nodes"]   # untouched default
    assert cfg["budget"]["nodes"] == "auto"
    assert cfg["search"] == wcfg.DEFAULTS["search"]


def test_resolve_config_rejects_non_mapping_top_level():
    cfg, errs = wcfg.resolve_config([1, 2, 3])
    assert cfg is None
    assert errs and "mapping" in errs[0]


# ---------------------------------------------------------------------------
# unknown keys -- every level
# ---------------------------------------------------------------------------
def test_unknown_top_level_key_is_an_error():
    cfg, errs = wcfg.resolve_config(dict(_valid_raw(), bogus=1))
    assert cfg is None
    assert any("unknown top-level key" in e and "bogus" in e for e in errs)


def test_unknown_nested_keys_are_errors_and_all_are_reported():
    raw = _valid_raw()
    raw["budget"] = dict(raw["budget"], typo_minutes=5)
    raw["search"] = dict(raw["search"], typo_omp=[1])
    raw["search"]["io"] = dict(raw["search"]["io"], typo_target=["nfs"])
    raw["probe"] = dict(raw["probe"], typo_steps=1)
    raw["prediction"] = dict(raw["prediction"], typo_io=1)
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    joined = " | ".join(errs)
    assert "budget.typo_minutes" in joined
    assert "search.typo_omp" in joined
    assert "search.io.typo_target" in joined
    assert "probe.typo_steps" in joined
    assert "prediction.typo_io" in joined


def test_unknown_exclude_rule_key_is_an_error():
    raw = _valid_raw(exclude=[dict(omp_gt=1, when="fire", bogus=1)])
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("exclude[0].bogus" in e for e in errs)


def test_unknown_physics_gray_knob_key_is_an_error():
    raw = _valid_raw(compliance="physics-gray")
    raw["physics_gray"]["knobs"]["radt"]["bogus"] = 1
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("physics_gray.knobs.radt.bogus" in e for e in errs)


# ---------------------------------------------------------------------------
# every validation-error TYPE required by the task
# ---------------------------------------------------------------------------
def test_out_of_range_omp_is_reported_with_the_catalog_bound():
    raw = _valid_raw()
    raw["search"]["omp"] = [0, 4]   # 0 is below per_rank_omp_threads' catalog min=1
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("search.omp" in e and "0" in e and "below" in e for e in errs)


def test_out_of_range_numtiles_is_reported_with_the_catalog_bound():
    raw = _valid_raw()
    raw["search"]["numtiles"] = [65]   # numtiles' catalog max=64 (predictor/decomp_space.py:89)
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("search.numtiles" in e and "above" in e for e in errs)


def test_illegal_rank_distribution_value_is_reported():
    raw = _valid_raw()
    raw["search"]["rank_distribution"] = ["plane"]   # not in srun_distribution's catalog enum
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("search.rank_distribution" in e and "plane" in e for e in errs)


def test_illegal_io_form_history_value_is_reported():
    raw = _valid_raw()
    raw["search"]["io"]["io_form_history"] = [4]   # not in io_form_history's catalog enum [2,11,14]
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("search.io.io_form_history" in e and "4" in e for e in errs)


def test_non_positive_aspect_max_is_reported():
    raw = _valid_raw()
    raw["search"]["aspect_max"] = 0
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("search.aspect_max" in e for e in errs)


def test_physics_gray_knob_under_strict_compliance_is_reported():
    """physics_gray knobs require compliance: physics-gray -- once physics_gray is actually
    requested (enabled: true), not when it is merely declared-but-disabled (the shipped template
    itself keeps compliance: strict with disabled physics_gray knobs as documentation)."""
    raw = _valid_raw(compliance="strict")
    raw["physics_gray"]["enabled"] = True
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("physics_gray.knobs.radt: requires compliance: physics-gray" in e for e in errs)
    assert any("physics_gray.knobs.time_step: requires compliance: physics-gray" in e for e in errs)
    # and it ALSO reports every violation, not just the first: the unimplemented-feature gate too
    assert any(e == "declared but not yet supported by wtune tune: physics_gray" for e in errs)


def test_physics_gray_knob_declared_but_disabled_under_strict_is_fine():
    raw = _valid_raw(compliance="strict")   # physics_gray.enabled stays False
    cfg, errs = wcfg.resolve_config(raw)
    assert errs == [], errs


def test_physics_gray_knob_not_in_catalog_is_reported():
    raw = _valid_raw(compliance="physics-gray")
    raw["physics_gray"]["knobs"] = dict(not_a_real_knob=dict(multiply=[1.0]))
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("not_a_real_knob" in e and "not found in the compliance catalog" in e for e in errs)


def test_physics_gray_knob_wrong_classification_is_reported():
    """`nproc_x` is a real catalog entry but classified STRICT, not PHYSICS-GRAY -- must be
    rejected as a physics_gray knob even though it exists."""
    raw = _valid_raw(compliance="physics-gray")
    raw["physics_gray"]["knobs"] = dict(nproc_x=dict(multiply=[1.0]))
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any("nproc_x" in e and "not PHYSICS-GRAY" in e for e in errs)


def test_strict_section_knob_classification_is_sanity_checked():
    """`_check_catalog_range` also asserts the catalog entry it looked up is actually STRICT --
    regression guard for the "STRICT section knobs must be STRICT" requirement even though every
    real catalog entry used today already satisfies it."""
    entry = wcfg.wc.lookup("per_rank_omp_threads")
    assert entry["classification"] == wcfg.wc.STRICT


# ---------------------------------------------------------------------------
# honesty gate: declared-but-not-yet-supported search sections
# ---------------------------------------------------------------------------
def test_enabling_io_search_is_a_hard_error():
    raw = _valid_raw()
    raw["search"]["io"]["enabled"] = True
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any(e == "declared but not yet supported by wtune tune: search.io" for e in errs)


def test_enabling_physics_gray_search_is_a_hard_error():
    raw = _valid_raw(compliance="physics-gray")
    raw["physics_gray"]["enabled"] = True
    cfg, errs = wcfg.resolve_config(raw)
    assert cfg is None
    assert any(e == "declared but not yet supported by wtune tune: physics_gray" for e in errs)


def test_io_search_disabled_by_default_and_range_still_checked():
    cfg, errs = wcfg.resolve_config({})
    assert errs == []
    assert cfg["search"]["io"]["enabled"] is False


# ---------------------------------------------------------------------------
# budget.nodes: auto | int | [ints]
# ---------------------------------------------------------------------------
def test_budget_nodes_accepts_auto_int_and_list():
    for nodes_val in ("auto", 4, [1, 2, 4]):
        cfg, errs = wcfg.resolve_config(_valid_raw(budget=dict(minutes=10, max_nodes=13,
                                                                  nodes=nodes_val, nodes_policy="knee")))
        assert errs == [], (nodes_val, errs)


def test_budget_nodes_exceeding_max_nodes_is_reported():
    cfg, errs = wcfg.resolve_config(_valid_raw(budget=dict(minutes=10, max_nodes=4, nodes=99,
                                                              nodes_policy="knee")))
    assert cfg is None
    assert any("budget.nodes" in e and "exceeds" in e for e in errs)


# ---------------------------------------------------------------------------
# shared candidate filter -- against a REAL decomp_space.enumerate_space grid
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def matthew_candidates():
    import decomp_space   # predictor/decomp_space.py; wtune_config's own sys.path bootstrap
    candidates, _axis = decomp_space.enumerate_space(MATTHEW_LIKE, total_cores=72, nodes=1,
                                                        hardware_id="scc-gh200")
    return candidates


def test_filter_candidates_removes_disallowed_omp(matthew_candidates):
    assert any(c["omp"] not in (1, 2) for c in matthew_candidates)   # the grid has more than 1,2
    search_cfg = dict(omp=[1, 2])
    out = wcfg.filter_candidates(matthew_candidates, search_cfg, [], False)
    assert out
    assert all(c["omp"] in (1, 2) for c in out)


def test_filter_candidates_removes_disallowed_numtiles(matthew_candidates):
    assert any(c["numtiles"] not in (1,) for c in matthew_candidates)
    search_cfg = dict(numtiles=[1])
    out = wcfg.filter_candidates(matthew_candidates, search_cfg, [], False)
    assert out
    assert all(c["numtiles"] == 1 for c in out)


def test_filter_candidates_removes_disallowed_dist(matthew_candidates):
    search_cfg = dict(rank_distribution=["block"])
    out = wcfg.filter_candidates(matthew_candidates, search_cfg, [], False)
    assert out
    assert all(c["dist"] == "block" for c in out)


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


def test_exempt_search_still_lets_everything_through_that_is_not_excluded(matthew_candidates):
    search_cfg = dict(omp=[1])   # would normally drop every omp != 1 candidate
    out = wcfg.filter_candidates(matthew_candidates, search_cfg, [], False, exempt_search=True)
    assert len(out) == len(matthew_candidates)


# ---------------------------------------------------------------------------
# exclude rules, including the fire rule
# ---------------------------------------------------------------------------
def test_exclude_fire_rule_drops_omp_gt_1_only_when_fire():
    rule = [dict(omp_gt=1, when="fire")]
    omp1 = dict(omp=1, numtiles=1, nx=8, ny=9, dist="block")
    omp2 = dict(omp=2, numtiles=1, nx=8, ny=9, dist="block")
    assert wcfg.candidate_allowed(omp1, {}, rule, is_fire=True) is True     # omp=1 never matches omp_gt:1
    assert wcfg.candidate_allowed(omp2, {}, rule, is_fire=True) is False    # excluded: fire + omp>1
    assert wcfg.candidate_allowed(omp2, {}, rule, is_fire=False) is True    # not a fire case: rule inert


def test_exclude_rule_without_when_applies_unconditionally():
    rule = [dict(numtiles_gt=32)]
    big = dict(omp=1, numtiles=48, nx=8, ny=9, dist="block")
    small = dict(omp=1, numtiles=8, nx=8, ny=9, dist="block")
    assert wcfg.candidate_allowed(big, {}, rule, is_fire=False) is False
    assert wcfg.candidate_allowed(small, {}, rule, is_fire=False) is True


def test_anchor_exempt_from_search_but_not_from_exclude():
    rule = [dict(omp_gt=1, when="fire")]
    anchor = dict(omp=2, numtiles=1, nx=8, ny=9, dist="block", id="ANCHOR")
    # exempt_search=True normally bypasses every `search` restriction...
    assert wcfg.candidate_allowed(anchor, dict(omp=[1]), [], is_fire=False, exempt_search=True) is True
    # ...but never bypasses `exclude`.
    assert wcfg.candidate_allowed(anchor, dict(omp=[1]), rule, is_fire=True, exempt_search=True) is False


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


def test_is_fire_case_false_when_ifire_all_zero(tmp_path):
    nl = tmp_path / "namelist.input"
    nl.write_text("&fire\n ifire = 0, 0,\n/\n")
    assert wcfg.is_fire_case(case_json=None, namelist_path=str(nl)) is False


def test_is_fire_case_false_when_namelist_missing():
    assert wcfg.is_fire_case(case_json=None, namelist_path="/no/such/namelist.input") is False


# ---------------------------------------------------------------------------
# `config show` candidate counts (predictor/decomp_space.enumerate_space + the filter)
# ---------------------------------------------------------------------------
def test_count_legal_candidates_shrinks_under_a_restrictive_search_cfg():
    cfg, errs = wcfg.resolve_config(dict(case="matthew", search=dict(omp=[1])))
    assert errs == []
    total, filtered = wcfg.count_legal_candidates(MATTHEW_LIKE, nodes=1, cfg=cfg, is_fire=False)
    assert total > 0
    assert 0 < filtered < total


def test_count_legal_candidates_unrestricted_keeps_everything():
    cfg, errs = wcfg.resolve_config(dict(case="matthew"))
    assert errs == []
    total, filtered = wcfg.count_legal_candidates(MATTHEW_LIKE, nodes=1, cfg=cfg, is_fire=False)
    assert filtered == total


def test_resolve_node_counts_auto_caps_at_max_nodes():
    counts = wcfg.resolve_node_counts(dict(nodes="auto", max_nodes=4))
    assert counts == [1, 2, 4]


def test_resolve_node_counts_explicit_list():
    counts = wcfg.resolve_node_counts(dict(nodes=[1, 4, 8], max_nodes=13))
    assert counts == [1, 4, 8]


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


def test_load_config_reads_and_validates_a_real_file(tmp_path):
    pytest.importorskip("yaml")
    p = tmp_path / "wtune.yaml"
    import yaml
    p.write_text(yaml.safe_dump(_valid_raw()))
    cfg = wcfg.load_config(str(p))
    assert cfg["case"] == "matthew"


def test_load_config_raises_config_error_with_every_violation(tmp_path):
    pytest.importorskip("yaml")
    import yaml
    raw = _valid_raw()
    raw["search"]["omp"] = [0]
    raw["bogus_key"] = 1
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(wcfg.ConfigError) as exc_info:
        wcfg.load_config(str(p))
    errs = exc_info.value.errors
    assert any("bogus_key" in e for e in errs)
    assert any("search.omp" in e for e in errs)


def test_template_renders_valid_yaml_that_resolves_cleanly():
    pytest.importorskip("yaml")
    import yaml
    text = wcfg.render_template("conus25")
    raw = yaml.safe_load(text)
    assert raw["case"] == "conus25"
    cfg, errs = wcfg.resolve_config(raw)
    assert errs == [], errs


def test_examples_wtune_yaml_matches_the_rendered_template_and_validates():
    pytest.importorskip("yaml")
    import yaml
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ -> repo root
    path = os.path.join(here, "examples", "wtune.yaml")
    with open(path) as f:
        text = f.read()
    assert text == wcfg.render_template("conus25")
    raw = yaml.safe_load(text)
    cfg, errs = wcfg.resolve_config(raw)
    assert errs == [], errs


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
