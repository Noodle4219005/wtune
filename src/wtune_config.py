#!/usr/bin/env python3
"""wtune_config.py -- YAML search-space configuration for `wtune tune`.

Precedence: CLI flag > this file > DEFAULTS (= wtune_tune.py's former argparse defaults). Every
violation is reported (ConfigError); a declared-but-unsearched knob (search.io, physics_gray)
enabled is an error, never a silent no-op. Schema: examples/wtune.yaml. PyYAML is imported lazily.
CLI: wtune_config.py init <case> [--out F] | show <file> [--case N|--case-dir D] [--nodes N,..]
| validate <file> (exit 2 on error)."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREDICTOR_DIR = os.path.join(_HERE, "predictor")
_CAMPAIGN_DIR = os.path.join(os.path.dirname(_HERE), "campaign")  # campaign/ stays at repo root
for _p in (_HERE, _PREDICTOR_DIR, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wtune_compliance as wc

# ---------------------------------------------------------------------------
# built-in defaults -- duplicated from wtune_tune.py (it imports this module; the reverse would
# be circular).
# ---------------------------------------------------------------------------
DEFAULT_SITE = "scc"
DEFAULT_MAX_NODES = 13                 # scc compute pool: work[1-13]
DEFAULT_HARDWARE_ID = "scc-gh200"
DEFAULT_CORES_PER_NODE = 72            # scc GH200
AUTO_NODE_COUNTS = (1, 2, 4, 8, 13)     # scc_sweep DEFAULT_NODES; `config show` preview only
NODES_POLICY_CHOICES = ("min", "cap", "knee")
IO_TARGETS = ("nfs", "local")           # wtune_predict's own calibration targets (machine/calibration/*.json)
EXCLUDE_WHEN_VALUES = ("fire",)

DEFAULTS = {
    "site": DEFAULT_SITE,
    "case": None,
    "case_dir": None,
    "compliance": "strict",
    "budget": {
        "minutes": None,               # no built-in default: required via --minutes or this key
        "max_nodes": DEFAULT_MAX_NODES,
        "nodes": "auto",
        "nodes_policy": "knee",
        # --nodes search knobs (wtune_tune.py: None = the CLI default applies)
        "node_objective": None,        # time | cost | knee
        "deadline_minutes": None,
        "min_efficiency": None,
        "node_finalists": None,
        "sweep_budget_frac": None,
    },
    "search": {
        "omp": None,                   # None = unrestricted (decomp_space's own omp_options)
        "numtiles": None,
        "rank_distribution": None,
        "aspect_max": None,
        "io": {"enabled": False, "io_form_history": [2, 11, 14], "target": ["nfs", "local"]},
    },
    "physics_gray": {"enabled": False, "knobs": {}},
    # steps "auto" = max(30, stepra+10): >= 1 radiation step per probe. spinup_min/ranking_start
    # rank every candidate; points..sources place the winner's full-run prediction points
    # (wtune_probe_plan.plan_points). ranking_start defaults to t0: spinup start is not wired
    # into strict_campaign.sbatch yet (selecting it prints a NOTE and runs from t0).
    "probe": {"steps": "auto", "confirm_reps": 2, "spinup_min": 60, "ranking_start": "t0",
              "points": "auto", "max_points": 5, "budget_pct": 10.0,
              "sources": ["trajectory", "restart", "rebuild"],
              # rank_by=predicted: rank by predicted whole-run time from shared spun-up ERA5
              # points (decision 0055); restart_io_form is that restart's io_form (14 or 2).
              "rank_by": "t0", "restart_io_form": 14, "stage1": "t0",
              "stage2_minutes": None,
              # points_by cost-indicators: place points by a weather cost proxy (decision 0060);
              # indicator_sources in priority order.
              "points_by": "time",
              "indicator_sources": ["case_forcing", "era5", "gfs", "stage4", "mrms"]},
    "prediction": {"io_target": "nfs", "drift_band_pct": None, "measure_r_rad": False},
    "exclude": [],
}

TOP_KEYS = {"site", "case", "case_dir", "compliance", "budget", "search", "physics_gray",
            "probe", "prediction", "exclude"}
BUDGET_KEYS = {"minutes", "max_nodes", "nodes", "nodes_policy", "node_objective",
               "deadline_minutes", "min_efficiency", "node_finalists", "sweep_budget_frac"}
NODE_OBJECTIVE_CHOICES = ("time", "cost", "knee")
SEARCH_KEYS = {"omp", "numtiles", "rank_distribution", "aspect_max", "io"}
SEARCH_IO_KEYS = {"enabled", "io_form_history", "target"}
PHYSICS_GRAY_KEYS = {"enabled", "knobs"}
PHYSICS_GRAY_KNOB_KEYS = {"multiply"}
PROBE_KEYS = {"steps", "confirm_reps", "spinup_min", "ranking_start", "points", "max_points",
              "budget_pct", "sources", "rank_by", "restart_io_form", "stage1",
              "stage2_minutes", "points_by", "indicator_sources"}
POINTS_BY_CHOICES = ("time", "cost-indicators")
INDICATOR_SOURCE_CHOICES = ("case_forcing", "era5", "gfs", "stage4", "mrms")
RANKING_START_CHOICES = ("t0", "spinup")
RANK_BY_CHOICES = ("t0", "predicted")
RESTART_IO_FORM_CHOICES = (14, 2)
STAGE1_CHOICES = ("t0", "pstar")
# "coldstart" is the pre-2026-09-26 name, still accepted (wtune_probe_plan.canonical_source
# normalizes it to "rebuild" everywhere downstream) -- never rejected as an unknown source.
PROBE_SOURCE_CHOICES = ("trajectory", "restart", "rebuild")
PROBE_SOURCE_ALIASES = {"coldstart": "rebuild"}
_PROBE_POINT_OFFSET_RE = re.compile(r"^\+\d+(?:\.\d+)?[mh]$")
_PROBE_POINT_PCT_RE = re.compile(r"^\d+(?:\.\d+)?%$")
_PROBE_POINT_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}$")
PREDICTION_KEYS = {"io_target", "drift_band_pct", "measure_r_rad"}
EXCLUDE_RULE_KEYS = {"when"} | {"%s_%s" % (field, op)
                                 for field in ("omp", "numtiles", "nx", "ny", "ranks")
                                 for op in ("gt", "lt", "ge", "le", "eq", "ne")}

_YAML_MISSING_MSG = (
    "wtune_config: PyYAML is required to load a wtune tune YAML config file.\n"
    "Install it with: pixi add pyyaml   (or: pip install pyyaml)\n"
)


class ConfigError(Exception):
    """Raised with every validation violation found (never just the first)."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors) if self.errors else "invalid config")


# ---------------------------------------------------------------------------
# YAML loading (lazy import)
# ---------------------------------------------------------------------------
def _load_yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(_YAML_MISSING_MSG) from exc
    return yaml


def load_config(path: str) -> dict:
    """Read + validate a YAML config file. Returns the resolved, typed config dict, or raises
    ConfigError (all violations) / SystemExit (PyYAML missing)."""
    yaml = _load_yaml_module()
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except OSError as exc:
        raise ConfigError(["cannot read config file %s: %s" % (path, exc)])
    except Exception as exc:  # yaml.YAMLError and friends
        raise ConfigError(["cannot parse YAML in %s: %s" % (path, exc)])
    if raw is None:
        raw = {}
    cfg, errors = resolve_config(raw)
    if errors:
        raise ConfigError(errors)
    return cfg


# ---------------------------------------------------------------------------
# merge (defaults <- raw dict) + validate on plain dicts (no YAML dependency)
# ---------------------------------------------------------------------------
def _merge_section(name, raw_section, defaults_section, allowed_keys, errors):
    if raw_section is None:
        return copy.deepcopy(defaults_section)
    if not isinstance(raw_section, dict):
        errors.append("%s: must be a mapping, got %s" % (name, type(raw_section).__name__))
        return copy.deepcopy(defaults_section)
    out = copy.deepcopy(defaults_section)
    for k, v in raw_section.items():
        if k not in allowed_keys:
            errors.append("%s.%s: unknown key" % (name, k))
            continue
        out[k] = v
    return out


def _merge_search(raw, errors):
    out = copy.deepcopy(DEFAULTS["search"])
    if raw is None:
        return out
    if not isinstance(raw, dict):
        errors.append("search: must be a mapping, got %s" % type(raw).__name__)
        return out
    for k, v in raw.items():
        if k not in SEARCH_KEYS:
            errors.append("search.%s: unknown key" % k)
            continue
        if k == "io":
            out["io"] = _merge_section("search.io", v, DEFAULTS["search"]["io"], SEARCH_IO_KEYS, errors)
        else:
            out[k] = v
    return out


def _merge_physics_gray(raw, errors):
    out = copy.deepcopy(DEFAULTS["physics_gray"])
    if raw is None:
        return out
    if not isinstance(raw, dict):
        errors.append("physics_gray: must be a mapping, got %s" % type(raw).__name__)
        return out
    for k, v in raw.items():
        if k not in PHYSICS_GRAY_KEYS:
            errors.append("physics_gray.%s: unknown key" % k)
            continue
        if k != "knobs":
            out[k] = v
            continue
        if not isinstance(v, dict):
            errors.append("physics_gray.knobs: must be a mapping, got %s" % type(v).__name__)
            continue
        knobs_out = {}
        for knob_name, knob_cfg in v.items():
            if not isinstance(knob_cfg, dict):
                errors.append("physics_gray.knobs.%s: must be a mapping" % knob_name)
                continue
            for kk in knob_cfg:
                if kk not in PHYSICS_GRAY_KNOB_KEYS:
                    errors.append("physics_gray.knobs.%s.%s: unknown key" % (knob_name, kk))
            knobs_out[knob_name] = dict(knob_cfg)
        out["knobs"] = knobs_out
    return out


def _merge_exclude(raw, errors):
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append("exclude: must be a list, got %s" % type(raw).__name__)
        return []
    out = []
    for i, rule in enumerate(raw):
        if not isinstance(rule, dict):
            errors.append("exclude[%d]: must be a mapping" % i)
            continue
        for k in rule:
            if k not in EXCLUDE_RULE_KEYS:
                errors.append("exclude[%d].%s: unknown key" % (i, k))
        when = rule.get("when")
        if when is not None and when not in EXCLUDE_WHEN_VALUES:
            errors.append("exclude[%d].when: %r not in %s" % (i, when, list(EXCLUDE_WHEN_VALUES)))
        out.append(dict(rule))
    return out


def _check_int_list(name, values, errors, min_len=1):
    if not isinstance(values, list) or len(values) < min_len:
        errors.append("%s: must be a non-empty list" % name)
        return []
    ok = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            errors.append("%s: %r is not an integer" % (name, v))
            continue
        ok.append(v)
    return ok


def _check_catalog_range(name, values, catalog_name, errors, expect_class=wc.STRICT):
    """Every value in `values` must be inside catalog_name's legal_values range/enum, and the
    catalog entry itself must carry `expect_class`."""
    entry = wc.lookup(catalog_name)
    if entry is None:
        errors.append("%s: internal error -- compliance catalog has no entry %r" % (name, catalog_name))
        return
    if entry["classification"] != expect_class:
        errors.append("%s: catalog entry %r is classified %s, not %s"
                       % (name, catalog_name, entry["classification"], expect_class))
    lv = entry["legal_values"]
    if lv["type"] == "enum":
        legal = set(lv["values"])
        for v in values:
            if v not in legal:
                errors.append("%s: %r is not in the catalog's legal set %s (%s)"
                               % (name, v, sorted(legal, key=str), catalog_name))
    else:
        lo, hi = lv.get("min"), lv.get("max")
        for v in values:
            if lo is not None and v < lo:
                errors.append("%s: %r is below the catalog minimum %r (%s)" % (name, v, lo, catalog_name))
            if hi is not None and v > hi:
                errors.append("%s: %r is above the catalog maximum %r (%s)" % (name, v, hi, catalog_name))


def _is_number(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float))


def _is_int(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, int)


def _choice(choices: tuple, optional: bool = False) -> tuple[Callable[[Any], bool], str]:
    """(is_valid, message) for a value that must be one of `choices` (or None if optional)."""
    return (lambda v: (optional and v is None) or v in choices), "%%r not in %s" % (list(choices),)


def _positive_number(v: Any) -> bool:
    return _is_number(v) and v > 0


def _optional(check: Callable[[Any], bool]) -> Callable[[Any], bool]:
    return lambda v: v is None or check(v)


_POS_NUMBER = "must be a positive number, got %r"
_POS_INT = "must be a positive integer, got %r"
_BOOL = "must be a bool, got %r"

# (key, is_valid, message % value) per section, checked in this order.
_BUDGET_RULES = [
    ("minutes", _optional(_positive_number), _POS_NUMBER),
    ("max_nodes", lambda v: _is_int(v) and v > 0, _POS_INT),
    ("nodes_policy", *_choice(NODES_POLICY_CHOICES)),
    ("node_objective", *_choice(NODE_OBJECTIVE_CHOICES, optional=True)),
    ("deadline_minutes", _optional(_positive_number), _POS_NUMBER),
    ("min_efficiency", _optional(_positive_number), _POS_NUMBER),
    ("sweep_budget_frac", _optional(_positive_number), _POS_NUMBER),
    ("sweep_budget_frac", lambda v: not (_is_number(v) and v > 1), "must be <= 1, got %r"),
    ("node_finalists", _optional(lambda v: _is_int(v) and v >= 1), _POS_INT),
]
_SEARCH_RULES = [
    ("aspect_max", _optional(_positive_number), _POS_NUMBER),
]
_PROBE_RULES = [
    ("steps", lambda v: v == "auto" or (_is_int(v) and v > 0),
     "must be a positive integer or 'auto', got %r"),
    ("confirm_reps", lambda v: _is_int(v) and v > 0, _POS_INT),
    ("spinup_min", lambda v: _is_number(v) and v >= 0, "must be a non-negative number, got %r"),
    ("ranking_start", *_choice(RANKING_START_CHOICES)),
    ("max_points", lambda v: _is_int(v) and v >= 1, _POS_INT),
    ("budget_pct", lambda v: _is_number(v) and 0 < v <= 100,
     "must be a number in (0, 100], got %r"),
    ("rank_by", *_choice(RANK_BY_CHOICES)),
    ("stage2_minutes", _optional(_positive_number), "must be a positive number or null, got %r"),
    ("stage1", *_choice(STAGE1_CHOICES)),
    ("restart_io_form", lambda v: not isinstance(v, bool) and v in RESTART_IO_FORM_CHOICES,
     "%%r not in %s" % (list(RESTART_IO_FORM_CHOICES),)),
    ("points_by", *_choice(POINTS_BY_CHOICES)),
]
_PREDICTION_RULES = [
    ("io_target", *_choice(IO_TARGETS)),
    ("drift_band_pct", _optional(lambda v: _is_number(v) and 0 <= v <= 100),
     "must be a number in [0, 100], got %r"),
    ("measure_r_rad", lambda v: isinstance(v, bool), _BOOL),
]


def _apply_rules(section: str, values: dict, rules: list, errors: list[str]) -> None:
    for key, is_valid, message in rules:
        v = values[key]
        if not is_valid(v):
            errors.append("%s.%s: %s" % (section, key, message % (v,)))


def _check_node_count(n: int, max_nodes: int | None, errors: list[str]) -> None:
    if n <= 0:
        errors.append("budget.nodes: %d must be positive" % n)
    elif max_nodes is not None and n > max_nodes:
        errors.append("budget.nodes: %d exceeds budget.max_nodes=%d" % (n, max_nodes))


def _validate_budget(b: dict, errors: list[str]) -> None:
    _apply_rules("budget", b, _BUDGET_RULES, errors)
    max_nodes = b["max_nodes"] if _is_int(b["max_nodes"]) and b["max_nodes"] > 0 else None
    nodes = b["nodes"]
    if nodes in ("auto", "search"):
        return
    if isinstance(nodes, list):
        for n in _check_int_list("budget.nodes", nodes, errors):
            _check_node_count(n, max_nodes, errors)
    elif not _is_int(nodes):
        errors.append("budget.nodes: must be 'auto', 'search', an integer, or a list of integers, "
                      "got %r" % (nodes,))
    else:
        _check_node_count(nodes, max_nodes, errors)


def _validate_search(s: dict, errors: list[str]) -> None:
    """STRICT knobs only: each listed value must be inside the compliance catalog's legal set."""
    if s["omp"] is not None:
        vals = _check_int_list("search.omp", s["omp"], errors)
        _check_catalog_range("search.omp", vals, "per_rank_omp_threads", errors)
    if s["numtiles"] is not None:
        vals = _check_int_list("search.numtiles", s["numtiles"], errors)
        _check_catalog_range("search.numtiles", vals, "numtiles", errors)
    if s["rank_distribution"] is not None:
        if not isinstance(s["rank_distribution"], list) or not s["rank_distribution"]:
            errors.append("search.rank_distribution: must be a non-empty list")
        else:
            _check_catalog_range("search.rank_distribution", s["rank_distribution"],
                                 "srun_distribution", errors)
    _apply_rules("search", s, _SEARCH_RULES, errors)

    io = s["io"]
    _apply_rules("search.io", io, [("enabled", lambda v: isinstance(v, bool), _BOOL)], errors)
    if not isinstance(io["io_form_history"], list) or not io["io_form_history"]:
        errors.append("search.io.io_form_history: must be a non-empty list")
    else:
        vals = _check_int_list("search.io.io_form_history", io["io_form_history"], errors)
        _check_catalog_range("search.io.io_form_history", vals, "io_form_history", errors)
    if not isinstance(io["target"], list) or not io["target"]:
        errors.append("search.io.target: must be a non-empty list")
    else:
        for t in io["target"]:
            if t not in IO_TARGETS:
                errors.append("search.io.target: %r not in %s" % (t, list(IO_TARGETS)))
    if io["enabled"]:
        errors.append("declared but not yet supported by wtune tune: search.io")


def _validate_physics_gray(pg: dict, compliance: str, errors: list[str]) -> None:
    """Knobs listed while disabled are inert documentation; enabling needs compliance:
    physics-gray and is then still rejected as not yet supported (both reported)."""
    _apply_rules("physics_gray", pg, [("enabled", lambda v: isinstance(v, bool), _BOOL)], errors)
    for knob_name, knob_cfg in pg["knobs"].items():
        if pg["enabled"] and compliance != "physics-gray":
            errors.append("physics_gray.knobs.%s: requires compliance: physics-gray (got %r)"
                          % (knob_name, compliance))
        entry = wc.lookup(knob_name)
        if entry is None:
            errors.append("physics_gray.knobs.%s: not found in the compliance catalog" % knob_name)
        elif entry["classification"] != wc.PHYSICS_GRAY:
            errors.append("physics_gray.knobs.%s: catalog classification is %s, not PHYSICS-GRAY"
                          % (knob_name, entry["classification"]))
        multiply = knob_cfg.get("multiply")
        if multiply is None:
            errors.append("physics_gray.knobs.%s: missing 'multiply'" % knob_name)
        elif not isinstance(multiply, list) or not multiply:
            errors.append("physics_gray.knobs.%s.multiply: must be a non-empty list" % knob_name)
        else:
            for m in multiply:
                if not _positive_number(m):
                    errors.append("physics_gray.knobs.%s.multiply: %r must be a positive number"
                                  % (knob_name, m))
    if pg["enabled"]:
        errors.append("declared but not yet supported by wtune tune: physics_gray")


def _validate_probe_sources(p: dict, errors: list[str]) -> None:
    isrc = p["indicator_sources"]
    if not isinstance(isrc, list) or not isrc:
        errors.append("probe.indicator_sources: must be a non-empty list")
    else:
        for s_ in isrc:
            if s_ not in INDICATOR_SOURCE_CHOICES:
                errors.append("probe.indicator_sources: %r not in %s (IMERG needs NASA Earthdata "
                              "credentials and is not supported)" % (s_, list(INDICATOR_SOURCE_CHOICES)))
        if len(set(isrc)) != len(isrc):
            errors.append("probe.indicator_sources: duplicate entry")
    if not isinstance(p["sources"], list) or not p["sources"]:
        errors.append("probe.sources: must be a non-empty list")
        return
    seen_src = set()
    for s in p["sources"]:
        canon = PROBE_SOURCE_ALIASES.get(s, s)
        if canon not in PROBE_SOURCE_CHOICES:
            errors.append("probe.sources: %r not in %s (or legacy alias %s)"
                          % (s, list(PROBE_SOURCE_CHOICES), list(PROBE_SOURCE_ALIASES)))
        elif canon in seen_src:
            errors.append("probe.sources: duplicate entry %r" % (s,))
        seen_src.add(canon)


def _validate_probe_point(i: int, item: Any, errors: list[str]) -> None:
    """One probe.points entry: '+<N>m', '+<N>h', '<N>%', an ISO time, or {at, reason, source}."""
    at = item
    if isinstance(item, dict):
        if "at" not in item:
            errors.append("probe.points[%d]: mapping form requires 'at'" % i)
            return
        at = item["at"]
        extra = set(item) - {"at", "reason", "source"}
        if extra:
            errors.append("probe.points[%d]: unknown key(s) %s" % (i, sorted(extra)))
        src = item.get("source")
        if src is not None and PROBE_SOURCE_ALIASES.get(src, src) not in PROBE_SOURCE_CHOICES:
            errors.append("probe.points[%d].source: %r not in %s (or legacy alias %s)"
                          % (i, src, list(PROBE_SOURCE_CHOICES), list(PROBE_SOURCE_ALIASES)))
    if not isinstance(at, str) or not (
            _PROBE_POINT_OFFSET_RE.match(at) or _PROBE_POINT_PCT_RE.match(at)
            or _PROBE_POINT_ISO_RE.match(at)):
        errors.append(
            "probe.points[%d]: %r is not a recognized form "
            "('+<N>m', '+<N>h', '<N>%%', or 'YYYY-MM-DD_HH:MM:SS')" % (i, at))


def _validate_probe(p: dict, errors: list[str]) -> None:
    _apply_rules("probe", p, _PROBE_RULES, errors)
    _validate_probe_sources(p, errors)
    points = p["points"]
    if points == "auto":
        return
    if not isinstance(points, list) or not points:
        errors.append("probe.points: must be 'auto' or a non-empty list")
        return
    for i, item in enumerate(points):
        _validate_probe_point(i, item, errors)


def _validate_exclude(rules: list[dict], errors: list[str]) -> None:
    """Value types only; key shape was checked by _merge_exclude."""
    for i, rule in enumerate(rules):
        for k, v in rule.items():
            if k != "when" and not _is_number(v):
                errors.append("exclude[%d].%s: must be a number, got %r" % (i, k, v))


def _validate_semantics(cfg: dict, errors: list[str]) -> None:
    if cfg.get("case") and cfg.get("case_dir"):
        errors.append("case and case_dir are mutually exclusive; set only one")
    if cfg["compliance"] not in ("strict", "physics-gray"):
        errors.append("compliance: %r not in ['strict', 'physics-gray']" % (cfg["compliance"],))
    _validate_budget(cfg["budget"], errors)
    _validate_search(cfg["search"], errors)
    _validate_physics_gray(cfg["physics_gray"], cfg["compliance"], errors)
    _validate_probe(cfg["probe"], errors)
    _apply_rules("prediction", cfg["prediction"], _PREDICTION_RULES, errors)
    _validate_exclude(cfg["exclude"], errors)


def resolve_config(raw: Any) -> tuple[dict | None, list[str]]:
    """(config_dict, errors). `config_dict` is None iff `errors` is non-empty. Pure dict-in,
    dict-out -- no YAML/file IO, so it is the entry point for tests that want to exercise
    validation without PyYAML."""
    if not isinstance(raw, dict):
        return None, ["top-level config must be a mapping, got %s" % type(raw).__name__]

    errors = []
    for k in raw:
        if k not in TOP_KEYS:
            errors.append("unknown top-level key: %r" % k)

    cfg = copy.deepcopy(DEFAULTS)
    cfg["site"] = raw.get("site", cfg["site"])
    cfg["case"] = raw.get("case", cfg["case"])
    cfg["case_dir"] = raw.get("case_dir", cfg["case_dir"])
    cfg["compliance"] = raw.get("compliance", cfg["compliance"])
    cfg["budget"] = _merge_section("budget", raw.get("budget"), DEFAULTS["budget"], BUDGET_KEYS, errors)
    cfg["search"] = _merge_search(raw.get("search"), errors)
    cfg["physics_gray"] = _merge_physics_gray(raw.get("physics_gray"), errors)
    cfg["probe"] = _merge_section("probe", raw.get("probe"), DEFAULTS["probe"], PROBE_KEYS, errors)
    cfg["prediction"] = _merge_section("prediction", raw.get("prediction"), DEFAULTS["prediction"],
                                        PREDICTION_KEYS, errors)
    cfg["exclude"] = _merge_exclude(raw.get("exclude"), errors)

    _validate_semantics(cfg, errors)
    if errors:
        return None, errors
    return cfg, []


# ---------------------------------------------------------------------------
# fire-case detection for "when: fire": cases.json "max_omp": 1, or &fire ifire > 0 on any
# domain (also catches a case not registered in cases.json).
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r'[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?')


def _all_numbers(raw):
    if raw is None:
        return []
    out = []
    for m in _NUM_RE.finditer(raw):
        s = m.group(0)
        out.append(float(s) if ('.' in s or 'e' in s.lower()) else int(s))
    return out


def is_fire_case(case_json: dict | None = None, namelist_path: str | None = None) -> bool:
    if case_json and case_json.get("max_omp") == 1:
        return True
    if namelist_path and os.path.exists(namelist_path):
        try:
            import wtune_cli as _wtune_core  # root wtune.py: parse_namelist (stdlib only)
            nl = _wtune_core.parse_namelist(namelist_path)
            ifire_raw = nl.get("fire", {}).get("ifire")
            if any(v and v > 0 for v in _all_numbers(ifire_raw)):
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# shared candidate filter for every wtune_tune generation site; the anchor is exempt from the
# search restrictions (exempt_search) but not from `exclude`.
# ---------------------------------------------------------------------------
def _aspect(nx, ny):
    if not nx or not ny:
        return None
    return max(nx / float(ny), ny / float(nx))


_EXCLUDE_FIELD_RE = re.compile(r"^(omp|numtiles|nx|ny|ranks)_(gt|lt|ge|le|eq|ne)$")
_OPS = {
    "gt": lambda a, b: a is not None and a > b,
    "lt": lambda a, b: a is not None and a < b,
    "ge": lambda a, b: a is not None and a >= b,
    "le": lambda a, b: a is not None and a <= b,
    "eq": lambda a, b: a is not None and a == b,
    "ne": lambda a, b: a is not None and a != b,
}


def _exclude_rule_matches(rule, candidate, is_fire):
    when = rule.get("when")
    if when == "fire" and not is_fire:
        return False
    conds = [(k, v) for k, v in rule.items() if k != "when"]
    if not conds:
        return when is not None
    for key, threshold in conds:
        m = _EXCLUDE_FIELD_RE.match(key)
        if not m:
            continue        # already reported at validation time; ignore defensively here
        field, op = m.groups()
        if not _OPS[op](candidate.get(field), threshold):
            return False
    return True


def excluded_by_rules(candidate: dict, exclude_rules: list[dict] | None, is_fire: bool) -> bool:
    return any(_exclude_rule_matches(r, candidate, is_fire) for r in (exclude_rules or ()))


def candidate_allowed(candidate: dict, search_cfg: dict | None, exclude_rules: list[dict] | None,
                      is_fire: bool, exempt_search: bool = False) -> bool:
    """True iff `candidate` (a decomp_space/scc_sweep-shaped dict: omp/numtiles/dist/nx/ny/...)
    survives `exclude` and (unless `exempt_search`, for the anchor) the STRICT `search`
    restrictions: omp, numtiles, rank_distribution, aspect_max."""
    if excluded_by_rules(candidate, exclude_rules, is_fire):
        return False
    if exempt_search:
        return True
    search_cfg = search_cfg or {}
    omp_allowed = search_cfg.get("omp")
    if omp_allowed and candidate.get("omp") not in omp_allowed:
        return False
    nt_allowed = search_cfg.get("numtiles")
    if nt_allowed and candidate.get("numtiles") not in nt_allowed:
        return False
    dist_allowed = search_cfg.get("rank_distribution")
    if dist_allowed and candidate.get("dist") not in dist_allowed:
        return False
    aspect_max = search_cfg.get("aspect_max")
    if aspect_max is not None:
        a = _aspect(candidate.get("nx"), candidate.get("ny"))
        if a is not None and a > aspect_max:
            return False
    return True


def filter_candidates(candidates: list[dict], search_cfg: dict | None,
                      exclude_rules: list[dict] | None, is_fire: bool,
                      exempt_search: bool = False) -> list[dict]:
    return [c for c in candidates
            if candidate_allowed(c, search_cfg, exclude_rules, is_fire, exempt_search)]


# ---------------------------------------------------------------------------
# candidate-count preview (`config show`): search-space size before spending cluster time
# ---------------------------------------------------------------------------
def resolve_node_counts(budget_cfg: dict) -> list[int]:
    nodes = budget_cfg.get("nodes")
    max_nodes = budget_cfg.get("max_nodes") or DEFAULT_MAX_NODES
    if nodes in (None, "auto"):
        counts = [n for n in AUTO_NODE_COUNTS if n <= max_nodes]
        return counts or [max_nodes]
    if isinstance(nodes, list):
        return [int(n) for n in nodes]
    return [int(nodes)]


def count_legal_candidates(case_json: dict, nodes: int, cfg: dict, is_fire: bool,
                           hardware_id: str | None = None,
                           cores_per_node: int = DEFAULT_CORES_PER_NODE) -> tuple[int, int]:
    """(total, filtered) legal decomp_space candidates at `nodes` nodes, before and after
    this config's search/exclude restrictions (no anchor exemption: the preview counts what
    round0/top-up/adaptive draw from). Sets $SITE if unset (side effect of the lazy import)."""
    os.environ.setdefault("SITE", cfg.get("site") or DEFAULT_SITE)
    import decomp_space  # predictor/decomp_space.py -- stdlib only, imported lazily
    hardware_id = hardware_id or DEFAULT_HARDWARE_ID
    total_cores = nodes * cores_per_node
    candidates, axis = decomp_space.enumerate_space(case_json, total_cores=total_cores, nodes=nodes,
                                                      hardware_id=hardware_id)
    filtered = filter_candidates(candidates, cfg["search"], cfg["exclude"], is_fire, exempt_search=False)
    return axis["total"], len(filtered)


def _first_int(vals):
    vals = _all_numbers(vals) if isinstance(vals, str) else vals
    return int(vals[0]) if vals else None


def case_json_from_dir(case_dir: str) -> tuple[dict | None, str | None, str | None]:
    """(case_json, namelist_path, error): the decomp_space.enumerate_space fields
    (e_we/e_sn/e_vert[/e_we_list/e_sn_list]/max_dom) from case_dir/namelist.input, for a case not
    in campaign/cases.json (mirrors recommend.parse_case_from_namelist)."""
    case_dir = os.path.expanduser(os.path.expandvars(case_dir))
    namelist_path = os.path.join(case_dir, "namelist.input")
    if not os.path.isfile(namelist_path):
        return None, None, "no namelist.input in case_dir %r" % case_dir
    try:
        import wtune_cli as _wtune_core
        nl = _wtune_core.parse_namelist(namelist_path)
    except Exception as exc:
        return None, None, "cannot parse %s: %s" % (namelist_path, exc)
    dom = nl.get("domains", {})
    max_dom = _first_int(dom.get("max_dom")) or 1
    e_we_list = [int(v) for v in _all_numbers(dom.get("e_we"))][:max_dom]
    e_sn_list = [int(v) for v in _all_numbers(dom.get("e_sn"))][:max_dom]
    e_vert_list = [int(v) for v in _all_numbers(dom.get("e_vert"))][:max_dom]
    if not e_we_list or not e_sn_list:
        return None, None, "namelist %s has no &domains e_we/e_sn" % namelist_path
    case_json = dict(e_we=e_we_list[0], e_sn=e_sn_list[0],
                      e_vert=(e_vert_list[0] if e_vert_list else 50), max_dom=max_dom)
    if len(e_we_list) > 1 and len(e_sn_list) > 1:
        case_json["e_we_list"] = e_we_list
        case_json["e_sn_list"] = e_sn_list
    return case_json, namelist_path, None


def resolve_case_for_preview(cfg: dict) -> tuple[dict | None, str | None, str]:
    """(case_json, namelist_path, label_or_error). label_or_error is the case name/dir on
    success, or a human-readable reason case_json is None."""
    if cfg.get("case"):
        os.environ.setdefault("SITE", cfg.get("site") or DEFAULT_SITE)
        import configs as campaign_configs  # campaign/configs.py
        case_json = campaign_configs.CASES.get(cfg["case"])
        if case_json is None:
            return None, None, "unknown case %r (not in campaign/cases.json)" % cfg["case"]
        try:
            namelist_path = os.path.expanduser(os.path.expandvars(
                campaign_configs._case_namelist_path(case_json)))
        except Exception:
            namelist_path = None
        return case_json, namelist_path, cfg["case"]
    if cfg.get("case_dir"):
        case_json, namelist_path, err = case_json_from_dir(cfg["case_dir"])
        if case_json is None:
            return None, None, err
        return case_json, namelist_path, cfg["case_dir"]
    return None, None, "config has neither case nor case_dir set"


# ---------------------------------------------------------------------------
# `config init` template
# ---------------------------------------------------------------------------
_TEMPLATE = """\
# wtune tune search-space configuration -- controls what `wtune tune` is allowed to search and
# how, instead of hardcoded CLI defaults.
#
# Precedence when used via `wtune tune --config FILE`: CLI flags > this file > built-in defaults.
# Unknown keys are a validation error (a typo is caught, not silently ignored). Validate/preview
# with: wtune.py config show %(case)s.yaml

site: scc                       # only "scc" is currently supported by wtune tune
case: %(case)s                  # case name from campaign/cases.json (+ site overlay:
                                 # campaign/cases.scc.json). ALTERNATIVE: replace this key with
                                 # `case_dir: /path/to/case` (a directory with its own
                                 # namelist.input, not registered in cases.json) -- `wtune config
                                 # show/init` support it for search-space preview; `wtune tune`
                                 # itself does not run against a case_dir yet.

compliance: strict               # strict       = only the STRICT knobs below may vary (bit-identical
                                  #                decomposition/tiling/placement/I/O-format changes)
                                  # physics-gray = STRICT + the `physics_gray` section below (namelist/
                                  #                physics values that MAY change the computed solution)

budget:
  minutes: 30                    # wall-clock budget for the whole `wtune tune` run (required: here or --minutes)
  max_nodes: 13                  # hard cap -- scc's compute pool is 13 nodes (work[1-13])
  nodes: auto                    # auto | search | <int> | [<int>, ...] -- search measures the ANCHOR at
                                  # each legal node count (wtune_tune --nodes search; node_objective/
                                  # deadline_minutes/min_efficiency/node_finalists/sweep_budget_frac
                                  # keys mirror its CLI flags); auto picks ONE node count via
                                  # `nodes_policy` from wtune_case's (uncalibrated) core-count profile;
                                  # a list is previewed by `wtune config show` but not yet run by `wtune tune`
  nodes_policy: knee              # min | cap | knee (see `wtune_tune.py --nodes-policy` for the exact rule)

search:                          # STRICT-only restrictions on what `wtune tune` is allowed to try;
                                  # omitting a key (or leaving it null) means "no restriction" for that axis
  omp: [1, 2, 3, 4, 6, 8, 9, 12]         # OMP_NUM_THREADS candidates (STRICT knob: per_rank_omp_threads)
  numtiles: [1, 2, 4, 8, 12, 16, 24]     # WRF numtiles candidates (STRICT knob: numtiles, catalog range 1-64)
  rank_distribution: [block, cyclic]     # srun --distribution candidates (STRICT knob: srun_distribution)
  aspect_max: 8                          # max(nproc_x/nproc_y, nproc_y/nproc_x) allowed in a decomposition shape
  io:                                    # STRICT knob: io_form_history -- NOT YET SEARCHED by wtune_tune.py;
                                          # `enabled: true` is a hard config error, never a silent no-op
    enabled: false
    io_form_history: [2, 11, 14]         # 2=split netCDF, 11/14=ADIOS2 variants
    target: [nfs, local]                 # filesystem target wtune_predict calibrates the I/O rate against

physics_gray:                     # PHYSICS-GRAY knobs -- only legal at compliance: physics-gray;
                                   # NOT YET SEARCHED by wtune_tune.py -- `enabled: true` is a hard config error
  enabled: false
  knobs:
    radt: {multiply: [1.0, 1.5, 2.0]}       # PHYSICS-GRAY knob: radiation time step multiplier (namelist radt)
    time_step: {multiply: [1.0, 1.1, 1.2]}  # PHYSICS-GRAY knob: dynamics time step multiplier (namelist time_step)

probe:
  steps: auto                     # WRF steps run per candidate during round0/adaptive probing (--steps) --
                                   # auto = max(30, stepra+10), so a probe always contains >= 1 radiation
                                   # step; or a fixed positive integer
  confirm_reps: 2                 # replicate count for the confirm stage's top candidates
  spinup_min: 60                  # RANKING layer: minutes the anchor spin-up run advances before every
                                   # candidate starts from ITS shared restart (0 disables -> t0 start)
  ranking_start: t0                # t0 | spinup (spinup not yet wired into strict_campaign.sbatch;
                                   # selecting it prints a NOTE and runs from t0)
  points: auto                    # PREDICTION layer (winner's own full-run estimate): auto placement, or
                                   # an explicit list, e.g. ["+90m", "70%%", "2019-11-26_23:00:00",
                                   # {at: "+3h", reason: night, source: rebuild}]
  max_points: 5                   # cap on the number of prediction-layer probe points
  budget_pct: 10                  # cap the points' own measurement cost to this %% of the predicted run
  sources: [trajectory, restart, rebuild]   # preference order per point ("coldstart" also accepted,
                                             # legacy name for "rebuild")
  rank_by: t0                     # t0 | predicted (decision 0055: rank by predicted whole-run time from
                                   # ERA5-rebuilt, spun-up points; CLI --rank-by)
  restart_io_form: 14             # rank_by=predicted only: shared restart io_form, 14 (ADIOS2) or 2
  stage1: t0                      # rank_by=predicted only: t0 (broad t0 search) | pstar (P* restart)
  stage2_minutes: null            # stage1=t0 only: optional cap on stage 2 (else it always runs)

prediction:
  io_target: nfs                  # nfs | local -- calibration target for the full-run I/O estimate
  drift_band_pct: 6               # override wtune_predict's calibration drift band (percent); omit/null
                                   # to use the calibration file's own default instead
  measure_r_rad: false            # derive the radiation-step cost multiplier (r_rad) from THIS run's own
                                   # probe measurements instead of the calibration file's fixed default --
                                   # NOT YET IMPLEMENTED by wtune_predict.py; requesting it (true) is
                                   # reported in the tune report as "requested, not yet implemented",
                                   # never silently ignored

exclude:                          # hard exclusion rules, checked even against the anchor config
  - {omp_gt: 1, when: fire}      # WRF-Fire (namelist &fire ifire>0 on any domain, or cases.json
                                   # "max_omp": 1): any omp>1 crashes at the first step
"""


def render_template(case: str) -> str:
    return _TEMPLATE % {"case": case}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_errors(errors):
    for m in errors:
        print("ERROR: %s" % m, file=sys.stderr)


def cmd_init(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="wtune_config.py init")
    ap.add_argument("case")
    ap.add_argument("--out", default="wtune.yaml")
    a = ap.parse_args(argv)
    with open(a.out, "w") as f:
        f.write(render_template(a.case))
    print("wrote %s" % a.out)
    return 0


def cmd_validate(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="wtune_config.py validate")
    ap.add_argument("file")
    a = ap.parse_args(argv)
    try:
        load_config(a.file)
    except ConfigError as e:
        _print_errors(e.errors)
        return 2
    print("%s: OK" % a.file)
    return 0


def cmd_show(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="wtune_config.py show")
    ap.add_argument("file")
    ap.add_argument("--case", default=None, help="override the case in the file")
    ap.add_argument("--case-dir", default=None, dest="case_dir")
    ap.add_argument("--nodes", default=None, help="comma list of node counts (default: budget.nodes)")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.file)
    except ConfigError as e:
        _print_errors(e.errors)
        return 2

    if a.case:
        cfg["case"], cfg["case_dir"] = a.case, None
    if a.case_dir:
        cfg["case_dir"], cfg["case"] = a.case_dir, None

    print(json.dumps(cfg, indent=2, sort_keys=True))

    case_json, namelist_path, label = resolve_case_for_preview(cfg)
    if case_json is None:
        print("\n(cannot preview candidate counts: %s)" % label)
        return 0

    is_fire = is_fire_case(case_json, namelist_path)
    node_counts = ([int(x) for x in a.nodes.split(",")] if a.nodes
                    else resolve_node_counts(cfg["budget"]))
    print("\ncandidate counts for case=%s (fire_case=%s):" % (label, is_fire))
    for n in node_counts:
        total, filtered = count_legal_candidates(case_json, n, cfg, is_fire)
        print("  nodes=%-3d legal_total=%-6d legal_after_search_filter=%-6d" % (n, total, filtered))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("init", "show", "validate"):
        asked = bool(argv) and argv[0] in ("-h", "--help")
        (sys.stdout if asked else sys.stderr).write(
            "usage: wtune_config.py init CASE [--out FILE] | "
            "wtune_config.py show FILE [--case NAME|--case-dir DIR] [--nodes N,N,...] | "
            "wtune_config.py validate FILE\n")
        return 0 if asked else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "init":
        return cmd_init(rest)
    if cmd == "show":
        return cmd_show(rest)
    return cmd_validate(rest)


if __name__ == "__main__":
    raise SystemExit(main())
