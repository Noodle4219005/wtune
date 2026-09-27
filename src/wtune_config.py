#!/usr/bin/env python3
"""wtune_config.py -- YAML search-space configuration for `wtune tune`.

Loads a YAML file that controls what `wtune_tune.py` is allowed to search and how, instead of
relying on wtune_tune.py's own hardcoded CLI defaults for everything. See `examples/wtune.yaml`
for the full commented schema (also emitted by `wtune.py config init <case>`).

Precedence when wired into `wtune_tune.py --config FILE`: CLI flag > this file > built-in
defaults (`DEFAULTS` below, which mirror wtune_tune.py's own previous hardcoded argparse
defaults exactly, so a run with no --config behaves identically to before this module existed).

Two knob families are declared in the schema but NOT YET SEARCHED by wtune_tune.py
(`search.io.enabled` / `physics_gray.enabled`): turning either on is a hard validation error
(exit 2), never a silent no-op -- see `_validate_semantics`.

Stdlib only except for the YAML parser itself (PyYAML), which is imported lazily so importing
this module (e.g. for the dict-level API in tests) never requires it -- only `load_config`/
`main()`'s file-reading paths do.

CLI (forwarded from `wtune.py config ...`, same REMAINDER pattern as `tune`/`metdata`):
    wtune_config.py init  <case> [--out wtune.yaml]     write a fully commented template
    wtune_config.py show  <file> [--case NAME|--case-dir DIR] [--nodes N,N,...]
                                                          print the resolved config + candidate
                                                          counts per allowed node count
    wtune_config.py validate <file>                      load + validate only; exit 2 on error
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREDICTOR_DIR = os.path.join(_HERE, "predictor")
_CAMPAIGN_DIR = os.path.join(os.path.dirname(_HERE), "campaign")  # campaign/ stays at repo root
for _p in (_HERE, _PREDICTOR_DIR, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wtune_compliance as wc

# ---------------------------------------------------------------------------
# built-in defaults -- mirror wtune_tune.py's previous hardcoded argparse defaults / module
# constants exactly (DEFAULT_MAX_NODES=13, "auto", "knee", DEFAULT_STEPS=20, CONFIRM_REPS=2).
# Duplicated rather than imported: wtune_tune.py imports THIS module, so the reverse import would
# be circular; small stable constants are duplicated elsewhere in this codebase for the same
# reason (see wtune_tune.py's own _RUN_ID_RE docstring).
# ---------------------------------------------------------------------------
DEFAULT_SITE = "scc"
DEFAULT_MAX_NODES = 13                 # scc compute pool: work[1-13]
DEFAULT_HARDWARE_ID = "scc-gh200"
DEFAULT_CORES_PER_NODE = 72
AUTO_NODE_COUNTS = (1, 2, 4, 8, 13)     # scc_sweep.py's own DEFAULT_NODES; used for `nodes: auto`
                                         # previews only (`config show`) -- wtune_tune.py's own
                                         # --nodes auto path always picks exactly ONE node count.
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
    },
    "search": {
        "omp": None,                   # None = unrestricted (decomp_space's own omp_options)
        "numtiles": None,
        "rank_distribution": None,
        "aspect_max": None,
        "io": {"enabled": False, "io_form_history": [2, 11, 14], "target": ["nfs", "local"]},
    },
    "physics_gray": {"enabled": False, "knobs": {}},
    # 2026-09-26 probe multi-point redesign (see wtune_probe_plan.py's module docstring): steps
    # default changes from a fixed 20 to "auto" (max(30, stepra+10), >=1 radiation step always
    # inside -- the whole reason this redesign exists). spinup_min/ranking_start are the RANKING
    # layer (every round0/adaptive/confirm candidate); points/max_points/budget_pct/sources are
    # the PREDICTION layer (the winner's own full-run estimate, wtune_probe_plan.plan_points).
    # ranking_start defaults to "t0", NOT "spinup": restart-based ranking-probe start is not yet
    # wired into campaign/strict_campaign.sbatch's run loop (campaign/configs.py's edit_namelist
    # restart_from/write_restart_interval/restart_io_form params are ready; nothing submits the
    # anchor spin-up job or stages its wrfrst yet -- see this task's report). A default must not
    # name behaviour that isn't wired -- selecting "spinup" explicitly still works exactly as
    # documented (prints/reports a NOTE and runs every candidate from t0 until the wiring lands).
    "probe": {"steps": "auto", "confirm_reps": 2, "spinup_min": 60, "ranking_start": "t0",
              "points": "auto", "max_points": 5, "budget_pct": 10.0,
              "sources": ["trajectory", "restart", "rebuild"],
              # decision 0055: rank_by=predicted ranks candidates by predicted whole-run time from
              # ERA5-rebuilt points, each spun up once and shared by every candidate (wtune_tune
              # run_predicted); t0 (default) is the pre-0055 behaviour. restart_io_form is the
              # shared restart's io_form in that mode (14 ADIOS2 BP, or 2 netCDF).
              "rank_by": "t0", "restart_io_form": 14, "stage1": "t0",
              "stage2_minutes": None},
    "prediction": {"io_target": "nfs", "drift_band_pct": None, "measure_r_rad": False},
    "exclude": [],
}

TOP_KEYS = {"site", "case", "case_dir", "compliance", "budget", "search", "physics_gray",
            "probe", "prediction", "exclude"}
BUDGET_KEYS = {"minutes", "max_nodes", "nodes", "nodes_policy"}
SEARCH_KEYS = {"omp", "numtiles", "rank_distribution", "aspect_max", "io"}
SEARCH_IO_KEYS = {"enabled", "io_form_history", "target"}
PHYSICS_GRAY_KEYS = {"enabled", "knobs"}
PHYSICS_GRAY_KNOB_KEYS = {"multiply"}
PROBE_KEYS = {"steps", "confirm_reps", "spinup_min", "ranking_start", "points", "max_points",
              "budget_pct", "sources", "rank_by", "restart_io_form", "stage1",
              "stage2_minutes"}
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


def load_config(path):
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
# merge (defaults <- raw dict) + validate, at the plain-dict level (no YAML dependency --
# exercised directly by tests when PyYAML is not installed in the test env)
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


def _validate_semantics(cfg, errors):
    if cfg.get("case") and cfg.get("case_dir"):
        errors.append("case and case_dir are mutually exclusive; set only one")

    if cfg["compliance"] not in ("strict", "physics-gray"):
        errors.append("compliance: %r not in ['strict', 'physics-gray']" % (cfg["compliance"],))

    # -- budget --
    b = cfg["budget"]
    if b["minutes"] is not None and (isinstance(b["minutes"], bool) or not isinstance(b["minutes"], (int, float))
                                      or b["minutes"] <= 0):
        errors.append("budget.minutes: must be a positive number, got %r" % (b["minutes"],))
    if isinstance(b["max_nodes"], bool) or not isinstance(b["max_nodes"], int) or b["max_nodes"] <= 0:
        errors.append("budget.max_nodes: must be a positive integer, got %r" % (b["max_nodes"],))
        max_nodes = None
    else:
        max_nodes = b["max_nodes"]
    if b["nodes_policy"] not in NODES_POLICY_CHOICES:
        errors.append("budget.nodes_policy: %r not in %s" % (b["nodes_policy"], list(NODES_POLICY_CHOICES)))
    nodes = b["nodes"]
    if nodes == "auto":
        pass
    elif isinstance(nodes, list):
        for n in _check_int_list("budget.nodes", nodes, errors):
            if n <= 0:
                errors.append("budget.nodes: %d must be positive" % n)
            elif max_nodes is not None and n > max_nodes:
                errors.append("budget.nodes: %d exceeds budget.max_nodes=%d" % (n, max_nodes))
    elif isinstance(nodes, bool) or not isinstance(nodes, int):
        errors.append("budget.nodes: must be 'auto', an integer, or a list of integers, got %r" % (nodes,))
    else:
        if nodes <= 0:
            errors.append("budget.nodes: %d must be positive" % nodes)
        elif max_nodes is not None and nodes > max_nodes:
            errors.append("budget.nodes: %d exceeds budget.max_nodes=%d" % (nodes, max_nodes))

    # -- search (STRICT knobs only) --
    s = cfg["search"]
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
    if s["aspect_max"] is not None:
        if isinstance(s["aspect_max"], bool) or not isinstance(s["aspect_max"], (int, float)) \
                or s["aspect_max"] <= 0:
            errors.append("search.aspect_max: must be a positive number, got %r" % (s["aspect_max"],))

    io = s["io"]
    if not isinstance(io["enabled"], bool):
        errors.append("search.io.enabled: must be a bool, got %r" % (io["enabled"],))
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

    # -- physics_gray --
    pg = cfg["physics_gray"]
    if not isinstance(pg["enabled"], bool):
        errors.append("physics_gray.enabled: must be a bool, got %r" % (pg["enabled"],))
    for knob_name, knob_cfg in pg["knobs"].items():
        # A knob listed here while physics_gray.enabled is false is inert documentation (e.g. the
        # shipped template: compliance stays "strict" and every knob stays disabled) -- only
        # actually REQUESTING physics-gray search (enabled: true) must also declare
        # compliance: physics-gray; that combination additionally fails the unconditional
        # not-yet-supported check below, so both violations are reported together.
        if pg["enabled"] and cfg["compliance"] != "physics-gray":
            errors.append("physics_gray.knobs.%s: requires compliance: physics-gray (got %r)"
                           % (knob_name, cfg["compliance"]))
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
                if isinstance(m, bool) or not isinstance(m, (int, float)) or m <= 0:
                    errors.append("physics_gray.knobs.%s.multiply: %r must be a positive number"
                                   % (knob_name, m))
    if pg["enabled"]:
        errors.append("declared but not yet supported by wtune tune: physics_gray")

    # -- probe --
    p = cfg["probe"]
    if p["steps"] != "auto" and (isinstance(p["steps"], bool) or not isinstance(p["steps"], int)
                                  or p["steps"] <= 0):
        errors.append("probe.steps: must be a positive integer or 'auto', got %r" % (p["steps"],))
    if isinstance(p["confirm_reps"], bool) or not isinstance(p["confirm_reps"], int) \
            or p["confirm_reps"] <= 0:
        errors.append("probe.confirm_reps: must be a positive integer, got %r" % (p["confirm_reps"],))
    if isinstance(p["spinup_min"], bool) or not isinstance(p["spinup_min"], (int, float)) \
            or p["spinup_min"] < 0:
        errors.append("probe.spinup_min: must be a non-negative number, got %r" % (p["spinup_min"],))
    if p["ranking_start"] not in RANKING_START_CHOICES:
        errors.append("probe.ranking_start: %r not in %s"
                       % (p["ranking_start"], list(RANKING_START_CHOICES)))
    if isinstance(p["max_points"], bool) or not isinstance(p["max_points"], int) or p["max_points"] < 1:
        errors.append("probe.max_points: must be a positive integer, got %r" % (p["max_points"],))
    if isinstance(p["budget_pct"], bool) or not isinstance(p["budget_pct"], (int, float)) \
            or not (0 < p["budget_pct"] <= 100):
        errors.append("probe.budget_pct: must be a number in (0, 100], got %r" % (p["budget_pct"],))
    if p["rank_by"] not in RANK_BY_CHOICES:
        errors.append("probe.rank_by: %r not in %s" % (p["rank_by"], list(RANK_BY_CHOICES)))
    if p["stage2_minutes"] is not None and (isinstance(p["stage2_minutes"], bool) or not
                                            isinstance(p["stage2_minutes"], (int, float))
                                            or p["stage2_minutes"] <= 0):
        errors.append("probe.stage2_minutes: must be a positive number or null, got %r"
                       % (p["stage2_minutes"],))
    if p["stage1"] not in STAGE1_CHOICES:
        errors.append("probe.stage1: %r not in %s" % (p["stage1"], list(STAGE1_CHOICES)))
    if isinstance(p["restart_io_form"], bool) or p["restart_io_form"] not in RESTART_IO_FORM_CHOICES:
        errors.append("probe.restart_io_form: %r not in %s"
                       % (p["restart_io_form"], list(RESTART_IO_FORM_CHOICES)))
    if not isinstance(p["sources"], list) or not p["sources"]:
        errors.append("probe.sources: must be a non-empty list")
    else:
        seen_src = set()
        for s in p["sources"]:
            canon = PROBE_SOURCE_ALIASES.get(s, s)
            if canon not in PROBE_SOURCE_CHOICES:
                errors.append("probe.sources: %r not in %s (or legacy alias %s)"
                               % (s, list(PROBE_SOURCE_CHOICES), list(PROBE_SOURCE_ALIASES)))
            elif canon in seen_src:
                errors.append("probe.sources: duplicate entry %r" % (s,))
            seen_src.add(canon)
    points = p["points"]
    if points != "auto":
        if not isinstance(points, list) or not points:
            errors.append("probe.points: must be 'auto' or a non-empty list")
        else:
            for i, item in enumerate(points):
                at = item
                if isinstance(item, dict):
                    if "at" not in item:
                        errors.append("probe.points[%d]: mapping form requires 'at'" % i)
                        continue
                    at = item["at"]
                    extra = set(item) - {"at", "reason", "source"}
                    if extra:
                        errors.append("probe.points[%d]: unknown key(s) %s" % (i, sorted(extra)))
                    src = item.get("source")
                    if src is not None and PROBE_SOURCE_ALIASES.get(src, src) not in PROBE_SOURCE_CHOICES:
                        errors.append("probe.points[%d].source: %r not in %s (or legacy alias %s)"
                                       % (i, src, list(PROBE_SOURCE_CHOICES),
                                          list(PROBE_SOURCE_ALIASES)))
                if not isinstance(at, str) or not (
                        _PROBE_POINT_OFFSET_RE.match(at) or _PROBE_POINT_PCT_RE.match(at)
                        or _PROBE_POINT_ISO_RE.match(at)):
                    errors.append(
                        "probe.points[%d]: %r is not a recognized form "
                        "('+<N>m', '+<N>h', '<N>%%', or 'YYYY-MM-DD_HH:MM:SS')" % (i, at))

    # -- prediction --
    pr = cfg["prediction"]
    if pr["io_target"] not in IO_TARGETS:
        errors.append("prediction.io_target: %r not in %s" % (pr["io_target"], list(IO_TARGETS)))
    if pr["drift_band_pct"] is not None:
        v = pr["drift_band_pct"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0 <= v <= 100):
            errors.append("prediction.drift_band_pct: must be a number in [0, 100], got %r" % (v,))
    if not isinstance(pr["measure_r_rad"], bool):
        errors.append("prediction.measure_r_rad: must be a bool, got %r" % (pr["measure_r_rad"],))

    # -- exclude (key-shape already checked by _merge_exclude; check value types here) --
    for i, rule in enumerate(cfg["exclude"]):
        for k, v in rule.items():
            if k == "when":
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errors.append("exclude[%d].%s: must be a number, got %r" % (i, k, v))


def resolve_config(raw):
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
# fire-case detection: "when: fire" in an exclude rule -- cases.json's own "max_omp": 1 field
# (campaign/configs.py / predictor/experiment.py's established convention), OR the namelist's own
# &fire ifire>0 on any active domain (predictor/recommend.py's PART E convention, replicated
# read-only here so a case not yet registered in cases.json is still protected).
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


def is_fire_case(case_json=None, namelist_path=None):
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
# shared candidate filter -- the ONE function every candidate-generation site in wtune_tune.py
# calls (round0/top-up/adaptive/diverse-sampling); the anchor is exempt from the STRICT search
# restrictions (exempt_search=True) but still must obey `exclude` rules.
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


def excluded_by_rules(candidate, exclude_rules, is_fire):
    return any(_exclude_rule_matches(r, candidate, is_fire) for r in (exclude_rules or ()))


def candidate_allowed(candidate, search_cfg, exclude_rules, is_fire, exempt_search=False):
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


def filter_candidates(candidates, search_cfg, exclude_rules, is_fire, exempt_search=False):
    return [c for c in candidates
            if candidate_allowed(c, search_cfg, exclude_rules, is_fire, exempt_search)]


# ---------------------------------------------------------------------------
# candidate-count preview (`config show`) -- predictor/decomp_space.enumerate_space + the filter
# above, so the user sees the search-space size before spending cluster time.
# ---------------------------------------------------------------------------
def resolve_node_counts(budget_cfg):
    nodes = budget_cfg.get("nodes")
    max_nodes = budget_cfg.get("max_nodes") or DEFAULT_MAX_NODES
    if nodes in (None, "auto"):
        counts = [n for n in AUTO_NODE_COUNTS if n <= max_nodes]
        return counts or [max_nodes]
    if isinstance(nodes, list):
        return [int(n) for n in nodes]
    return [int(nodes)]


def count_legal_candidates(case_json, nodes, cfg, is_fire, hardware_id=None,
                            cores_per_node=DEFAULT_CORES_PER_NODE):
    """(total, filtered) legal decomp_space candidates for `case_json` at `nodes` nodes, before
    and after this config's `search`/`exclude` restrictions (the anchor is exempt from the search
    restrictions, but a preview count has no single "anchor" of its own -- it reports the fully
    unrestricted sampled-candidate count, matching what round0/top-up/adaptive actually draw
    from)."""
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


def case_json_from_dir(case_dir):
    """(case_json, namelist_path, error). Minimal decomp_space.enumerate_space-shaped case dict
    (e_we/e_sn/e_vert[/e_we_list/e_sn_list]/max_dom) read directly from case_dir/namelist.input --
    for `case_dir:` configs (a case not registered in campaign/cases.json). Mirrors
    predictor/recommend.py's parse_case_from_namelist field-by-field, but only the fields
    decomp_space.enumerate_space actually needs."""
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


def resolve_case_for_preview(cfg):
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
  nodes: auto                    # auto | <int> | [<int>, ...] -- auto picks ONE node count via
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


def render_template(case):
    return _TEMPLATE % {"case": case}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_errors(errors):
    for m in errors:
        print("ERROR: %s" % m, file=sys.stderr)


def cmd_init(argv):
    ap = argparse.ArgumentParser(prog="wtune_config.py init")
    ap.add_argument("case")
    ap.add_argument("--out", default="wtune.yaml")
    a = ap.parse_args(argv)
    with open(a.out, "w") as f:
        f.write(render_template(a.case))
    print("wrote %s" % a.out)
    return 0


def cmd_validate(argv):
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


def cmd_show(argv):
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("init", "show", "validate"):
        sys.stderr.write(
            "usage: wtune_config.py init CASE [--out FILE] | "
            "wtune_config.py show FILE [--case NAME|--case-dir DIR] [--nodes N,N,...] | "
            "wtune_config.py validate FILE\n")
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "init":
        return cmd_init(rest)
    if cmd == "show":
        return cmd_show(rest)
    return cmd_validate(rest)


if __name__ == "__main__":
    raise SystemExit(main())
