"""Golden snapshot of wtune_config.resolve_config's validation messages (order included), pinned
before the round-2 refactor made _validate_semantics table-driven.

Regenerate (only when a behaviour change is intended) with:
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_config.py
"""
import json
import os

from wtune import config as wcfg

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "fixtures", "golden", "config", "resolve_config.json")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"

BAD_NUMBERS = [0, -1, True, "x", None, 1.5, 2]

CASES = {
    "defaults": {},
    "not_a_mapping": [1, 2],
    "top_level": {"case": "a", "case_dir": "/x", "compliance": "loose", "bogus": 1},
    "budget_types": {"budget": {"minutes": True, "max_nodes": 0, "nodes_policy": "x",
                                "node_objective": "speed", "deadline_minutes": -1,
                                "min_efficiency": "x", "sweep_budget_frac": 1.5,
                                "node_finalists": 0, "nodes": [0, 20, "a", True]}},
    "budget_bad_max_nodes_list": {"budget": {"max_nodes": True, "nodes": [1, 50]}},
    "budget_nodes_int": {"budget": {"max_nodes": 4, "nodes": 8}},
    "budget_nodes_zero": {"budget": {"nodes": 0}},
    "budget_nodes_str": {"budget": {"nodes": "many"}},
    "budget_nodes_bool": {"budget": {"nodes": False}},
    "budget_nodes_empty": {"budget": {"nodes": []}},
    "budget_ok": {"budget": {"minutes": 30, "max_nodes": 8, "nodes": [1, 2, 8],
                             "node_objective": "cost", "sweep_budget_frac": 1,
                             "node_finalists": 3, "deadline_minutes": 2.5}},
    "budget_not_mapping": {"budget": 5},
    "search_bad": {"search": {"omp": [0, 999, "a"], "numtiles": "x",
                              "rank_distribution": [], "aspect_max": -2,
                              "io": {"enabled": "yes", "io_form_history": [99, "a"],
                                     "target": ["nfs", "tape"], "zzz": 1}, "bad": 2}},
    "search_more_bad": {"search": {"omp": [], "rank_distribution": ["cyclic", "nope"],
                                   "aspect_max": True,
                                   "io": {"enabled": True, "io_form_history": [],
                                          "target": "nfs"}}},
    "search_ok": {"search": {"omp": [1, 2], "numtiles": [1, 4], "aspect_max": 3.0}},
    "search_not_mapping": {"search": "x"},
    "search_io_not_mapping": {"search": {"io": 3}},
    "physics_gray_bad": {"physics_gray": {"enabled": "y", "knobs": {
        "radt": {"multiply": [0.5, -1, True]}, "nonexistent_knob": {},
        "per_rank_omp_threads": {"multiply": []}, "x": 3, "y": {"zz": 1, "multiply": "a"}}}},
    "physics_gray_enabled_strict": {"physics_gray": {"enabled": True, "knobs": {
        "radt": {"multiply": [2]}}}},
    "physics_gray_enabled_gray": {"compliance": "physics-gray", "physics_gray": {
        "enabled": True, "knobs": {"radt": {"multiply": [2]}}}},
    "physics_gray_not_mapping": {"physics_gray": [1]},
    "physics_gray_knobs_not_mapping": {"physics_gray": {"knobs": [1], "extra": 1}},
    "probe_bad": {"probe": {"steps": 0, "confirm_reps": True, "spinup_min": -1,
                            "ranking_start": "late", "max_points": 0, "budget_pct": 101,
                            "rank_by": "x", "stage2_minutes": 0, "stage1": "y",
                            "restart_io_form": True, "points_by": "z",
                            "indicator_sources": ["era5", "imerg", "era5"],
                            "sources": ["coldstart", "rebuild", "bogus", "trajectory",
                                        "trajectory"],
                            "points": ["+30m", "50%", "2020-01-01_00:00:00", "+1d", 5,
                                       {"reason": "x"}, {"at": "+2h", "foo": 1, "source": "z"},
                                       {"at": "bad", "source": "coldstart"}],
                            "nope": 1}},
    "probe_more_bad": {"probe": {"steps": "many", "confirm_reps": 0, "spinup_min": "x",
                                 "max_points": True, "budget_pct": 0, "stage2_minutes": True,
                                 "restart_io_form": 5, "indicator_sources": [],
                                 "sources": "trajectory", "points": []}},
    "probe_types": {"probe": {"steps": 2.5, "budget_pct": "x", "stage2_minutes": "y",
                              "indicator_sources": "era5", "sources": [], "points": "x"}},
    "probe_ok": {"probe": {"steps": 30, "stage2_minutes": 15, "rank_by": "predicted",
                           "restart_io_form": 2, "stage1": "pstar",
                           "points_by": "cost-indicators", "points": ["+1h", {"at": "10%"}]}},
    "prediction_bad": {"prediction": {"io_target": "s3", "drift_band_pct": 101,
                                      "measure_r_rad": "no", "x": 1}},
    "prediction_bad_type": {"prediction": {"drift_band_pct": True}},
    "prediction_ok": {"prediction": {"io_target": "local", "drift_band_pct": 0,
                                     "measure_r_rad": True}},
    "exclude_bad": {"exclude": [{"when": "rain", "omp_gt": "a", "bogus": 1}, 5,
                                {"nx_lt": True, "ranks_eq": 4}]},
    "exclude_not_list": {"exclude": {"when": "fire"}},
    "exclude_ok": {"exclude": [{"when": "fire", "omp_gt": 1}]},
}


def test_resolve_config_golden():
    out = {}
    for name, raw in CASES.items():
        cfg, errors = wcfg.resolve_config(raw)
        out[name] = {"errors": errors, "cfg": cfg}
    text = json.dumps(out, indent=1) + "\n"
    if REGEN:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            fh.write(text)
    with open(GOLDEN) as fh:
        expected = fh.read()
    assert text == expected, "resolve_config output differs from golden %s" % GOLDEN
