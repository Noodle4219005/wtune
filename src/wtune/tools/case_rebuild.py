#!/usr/bin/env python3
"""tools/case_rebuild.py -- SPEC-rebuild-analog.md (2026-09-24): rebuild a full runnable WRF
case from just namelist.input + a projection source, compare it against a real wrfinput/wrfbdy,
search for a real date whose weather matches an undated/fictional case, and (via
coldstart_sample.py edits) sample cold-start windows at a date-shifted analog.

REUSES, does not duplicate: tools/coldstart_sample.py (namelist.wps generation, projection
resolution, forcing chain choice/probe/fetch, namelist editing helpers, run.sbatch conventions
incl. geogrid/metgrid TBL dir links + output checks + OMP + real.exe timeout/serial fallback),
tools/metdata.py (resolve_forcing_chain, state-purpose byte-range fetch + bbox crop, via
coldstart_sample's own lazy accessors), tools/grib_state.py (target_grid,
regrid_nearest_regular), wtune/predict.py (parse_namelist, g/g_dom, sunlit_fraction),
wtune/runlength.py (via wtune.predict).

Module-level imports are STDLIB + coldstart_sample + wtune.predict ONLY (both of those are
themselves numpy-free) -- `rebuild`'s fix-span-*-namelist subcommands run inside a batch job on
scc's /usr/bin/python3, which has no numpy. numpy/netCDF4 (only on the cluster's grib pixi env,
~/wtune/envs/grib -- numpy 2.4, netCDF4 1.7, xarray, cfgrib, eccodes, requests; NO scipy/sklearn)
are imported LOCALLY inside the `compare`/`analog` functions that actually need them, same
lazy-import discipline coldstart_sample.py itself uses for metdata/grib_state.

CLI:
  case_rebuild.py rebuild plan --case-dir D [--namelist P] [--namelist-wps P |
    --geo-from-wrfinput P] --out OUT [--date-shift-to T] [--forcing DATASET] [--verify-steps N]
    [--nodes N --ranks R --omp T --nx X --ny Y] [--ncdump P] [--registry P]
  case_rebuild.py rebuild fetch --out OUT [--cache-dir C] [--grib-dir G] [--keep-archives]
  case_rebuild.py rebuild stage --out OUT [--site scc]
  case_rebuild.py rebuild finalize --out OUT
  case_rebuild.py rebuild fix-span-wps-namelist --out OUT
  case_rebuild.py rebuild fix-span-real-namelist --out OUT [--ncdump P] [--serial]
  case_rebuild.py rebuild fix-span-wrf-namelist --out OUT --nproc-x X --nproc-y Y [--numtiles T]
  case_rebuild.py compare --original W1 --rebuilt W2 [--json OUT] [threshold options]
  case_rebuild.py analog --case-dir D --out OUT [--wrfinput W] [--wrfbdy B]
    [--years 2021-2025] [--doy-window 45] [--hour same] [--top 5] [--workers 8]
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import numbers
import re
import shutil
import sys
from typing import TYPE_CHECKING, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # src/
if not __package__:                    # run by path: this directory is not an import root
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import wtune.machine
from wtune import predict as predict     # repo root -- stdlib only
from wtune.tools import coldstart_sample as cs       # tools/ -- stdlib only at module level (lazy md/gs)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike

_WRF_TIME_FMT = cs._WRF_TIME_FMT
DEFAULT_REGISTRY = os.path.join(_REPO_ROOT, "wtune_catalog_registry.json")


class RebuildError(Exception):
    """A user-facing input/modeling error, same role as coldstart_sample.ColdstartError."""


# The groups WRF's namelist reader requires to exist, even empty (a MISSING group is FATAL;
# job 986). The Registry catalog does not record this list.
_ALWAYS_GROUPS = ("time_control", "domains", "physics", "fdda", "dynamics", "bdy_control",
                  "grib2", "namelist_quilt")
_DT_PARTS = ("year", "month", "day", "hour", "minute", "second")
_IO_FORM_KEYS = ("io_form_history", "io_form_restart", "io_form_input", "io_form_boundary")
_SCHEME_KEYS = ("mp_physics", "ra_lw_physics", "ra_sw_physics", "bl_pbl_physics", "cu_physics",
                "sf_sfclay_physics", "sf_surface_physics", "physics_suite")
_DEFAULT_E_VERT = 45              # vertical levels
_DEFAULT_P_TOP_PA = 5000          # Pa
_DEFAULT_HISTORY_MIN = 60         # min
_DEFAULT_INTERVAL_S = 10800       # s (3 h forcing cadence)
_DEFAULT_RUN_S = 6 * 3600         # s
_DEFAULT_SPEC_BDY_WIDTH = 5       # WRF Registry default
_NETCDF_IO_FORM = 2               # io_form 2 = netCDF


def _dt_parts(t: datetime.datetime) -> list[tuple[str, int]]:
    """[('year', y), ..., ('second', s)] for WRF's start_*/end_* keys."""
    return [(p, getattr(t, p)) for p in _DT_PARTS]


# ===========================================================================
# R6-A: rebuild -- namelist completion + registry validation
# ===========================================================================

def load_registry(path: str = DEFAULT_REGISTRY) -> dict:
    with open(path) as f:
        return json.load(f)


def registry_valid_keys(registry: dict) -> set[str]:
    """Every valid namelist key (lowercased) across all Registry catalog groups -- flat, since a
    namelist's &group names do not always match the catalog's group keys 1:1."""
    keys = set()
    for _group, entries in registry.get("groups", {}).items():
        for e in entries:
            name = e.get("name")
            if name:
                keys.add(name.lower())
    return keys


def validate_namelist_against_registry(nl: dict, registry: dict) -> list[str]:
    """Sorted warnings for parse_namelist keys missing from the Registry catalog; never raises or
    edits. A group the catalog lacks entirely (e.g. &namelist_quilt) gets one 'group not in
    catalog' warning instead of one per key (T11)."""
    valid = registry_valid_keys(registry)
    catalog_groups = set(g.lower() for g in registry.get("groups", {}))
    warnings = []
    for group, keys in nl.items():
        if group.lower() not in catalog_groups:
            warnings.append("group '&%s' not in Registry catalog" % group)
            continue
        for key in keys:
            if key.lower() not in valid:
                warnings.append("unknown namelist key '%s' in &%s -- not in Registry catalog"
                                % (key, group))
    return sorted(warnings)


def _registry_entry(key: str, registry: dict) -> tuple[str | None, dict | None]:
    """(group, entry) of the first catalog entry named `key` (case-insensitive), or (None, None)."""
    for group, entries in registry.get("groups", {}).items():
        for e in entries:
            if e.get("name", "").lower() == key.lower():
                return group, e
    return None, None


def registry_group_for_key(key: str, registry: dict) -> str | None:
    """The catalog group that owns `key`, or None -- so an inserted key lands in its own &group
    (T11: p_top_requested appended to &time_control made real.exe FATAL)."""
    return _registry_entry(key, registry)[0]


def registry_per_domain(key: str, registry: dict, default: bool = True) -> bool:
    """Whether the catalog marks `key` max_domains-valued ('per_domain'); `default` when the
    catalog has no entry or no flag. T14: damp_opt is dimension 1 -- a broadcast list can be
    rejected by a strict namelist reader."""
    _group, e = _registry_entry(key, registry)
    if e is None:
        return default
    pd = e.get("per_domain")
    return bool(pd) if pd is not None else default


# Lives in coldstart_sample (it needs it too; case_rebuild imports coldstart_sample, not the
# reverse); aliased so `cr._namelist_values_equal` keeps working.
_namelist_values_equal = cs.namelist_values_equal


class _NamelistEditor:
    """complete_namelist's working text plus its change log ({key, old, new, reason} rows)."""

    def __init__(self, txt: str, registry: dict, max_dom: int):
        self.txt = txt
        self.changes: list[dict] = []
        self.registry = registry
        self.max_dom = max_dom

    def group_for(self, key: str, default: str = "time_control") -> str:
        return registry_group_for_key(key, self.registry) or default

    def ndom_for(self, key: str) -> int:
        """Values to write: max_dom for a catalog per-domain key, else 1 (T14)."""
        return self.max_dom if (self.max_dom > 1 and registry_per_domain(key, self.registry)) else 1

    def log(self, key: str, old, new, reason: str) -> None:
        self.changes.append({"key": key, "old": old, "new": new, "reason": reason})

    def sync(self, key: str, value, reason: str = "", group_default: str = "time_control") -> None:
        """Set `key` (insert if missing); a numerically equal spelling ('01' vs '1') is no change."""
        old = cs.get_first(self.txt, key)
        new_str = str(value)
        if old is not None and _namelist_values_equal(old.strip(), new_str):
            return
        self.txt = cs.set_list(self.txt, key, value, self.ndom_for(key),
                               new_key_group=self.group_for(key, default=group_default))
        self.log(key, old, new_str, reason)

    def fill_if_missing(self, key: str, value, reason: str = "",
                        group_default: str = "time_control") -> None:
        if cs.count_key(self.txt, key) > 0:
            return
        self.txt = cs.set_list(self.txt, key, value, self.ndom_for(key),
                               new_key_group=self.group_for(key, default=group_default))
        self.log(key, None, str(value), reason)

    def fill_values_if_missing(self, key: str, values: list, reason: str,
                               group_default: str = "time_control") -> None:
        """Like fill_if_missing, for an explicit per-domain value list."""
        if cs.count_key(self.txt, key) != 0:
            return
        self.txt = cs.set_list_values(self.txt, key, values,
                                      new_key_group=self.group_for(key, default=group_default))
        self.log(key, None, str(values), reason)


def _sync_met_and_span(ed: _NamelistEditor, facts: dict) -> int:
    """Always-sync keys from met_em/geo_em and the rebuild span; returns the run length (s)."""
    ed.sync("num_metgrid_levels", int(facts["num_metgrid_levels"]), "met_em NUM_METGRID_LEVELS")
    if facts.get("num_metgrid_soil_levels") is not None:
        ed.sync("num_metgrid_soil_levels", int(facts["num_metgrid_soil_levels"]),
                "met_em NUM_METGRID_SOIL_LEVELS")
    if facts.get("num_land_cat") is not None:
        ed.sync("num_land_cat", int(facts["num_land_cat"]), "met_em/geo_em NUM_LAND_CAT")
    ed.sync("interval_seconds", int(facts["interval_seconds"]), "forcing cadence")

    start_dt, end_dt = facts["start_dt"], facts["end_dt"]
    for k, v in _dt_parts(start_dt):
        ed.sync("start_" + k, v, "rebuild span start")
    for k, v in _dt_parts(end_dt):
        ed.sync("end_" + k, v, "rebuild span end")
    run_s = int(round((end_dt - start_dt).total_seconds()))
    ed.sync("run_days", 0, "rebuild span")
    ed.sync("run_hours", 0, "rebuild span")
    ed.sync("run_minutes", 0, "rebuild span")
    ed.sync("run_seconds", run_s, "rebuild span")
    return run_s


def _apply_vertical_grid(ed: _NamelistEditor, vertical_grid: dict) -> None:
    """Reproduce the wrfinput's own ZNW/P_TOP exactly (T8): dz* keys conflict with an explicit
    eta_levels and are removed; e_vert/eta_levels/p_top_requested always overridden."""
    for key in ("dzbot", "dzstretch_s", "dzstretch_u", "auto_levels_opt"):
        ed.txt, n_removed = remove_namelist_key(ed.txt, key)
        if n_removed:
            ed.log(key, "present", None, "conflicts with explicit eta_levels from wrfinput ZNW")
    eta_levels = list(vertical_grid["eta_levels"])
    ed.sync("e_vert", len(eta_levels), "= len(wrfinput ZNW)")
    ed.txt = cs.set_list_values(ed.txt, "eta_levels", eta_levels,
                                new_key_group=ed.group_for("eta_levels"))
    ed.log("eta_levels", None, str(eta_levels), "= wrfinput ZNW (exact vertical grid reproduction)")
    ed.sync("p_top_requested", vertical_grid["p_top_pa"], "= wrfinput P_TOP")  # plan.json key


def _fill_run_defaults(ed: _NamelistEditor, facts: dict, run_s: int) -> None:
    """Time step, nest step ratios, history/restart/io_form defaults -- never overriding."""
    dx_km = float(facts["dx_km"])
    time_step = max(1, int(6.0 * dx_km))   # s; WRF rule of thumb dt <= 6 x dx(km)
    ed.fill_if_missing("time_step", time_step, "largest int <= 6*dx_km(d01), >=1")
    if ed.max_dom > 1 and cs.count_key(ed.txt, "parent_time_step_ratio") == 0:
        ratios = [1] + [int(facts["parent_grid_ratio"][d]) for d in range(1, ed.max_dom)]
        ed.fill_values_if_missing("parent_time_step_ratio", ratios, "= parent_grid_ratio per nest")
    # WRF sums interval, _d, _h, _m and _s: any one of them present is the case's own cadence
    def has_interval(base):
        return any(cs.count_key(ed.txt, base + sfx) for sfx in ("", "_d", "_h", "_m", "_s"))
    if not has_interval("history_interval"):
        ed.fill_if_missing("history_interval", _DEFAULT_HISTORY_MIN,
                           "default history cadence (minutes)")
    ed.fill_if_missing("frames_per_outfile", 1, "default")
    ed.fill_if_missing("restart", ".false.", "default")
    run_minutes = int(round(run_s / 60.0))
    if not has_interval("restart_interval"):
        ed.fill_if_missing("restart_interval", run_minutes, "one restart at the end of the run")
    for key in _IO_FORM_KEYS:
        ed.fill_if_missing(key, _NETCDF_IO_FORM, "default netCDF io_form")


def _fill_physics_suite(ed: _NamelistEditor, original_text: str) -> None:
    """physics_suite='CONUS' when the ORIGINAL text names no &physics scheme at all; a suite with
    -1 scheme placeholders is only noted (old==new): -1 means 'suite default', not unset (T14)."""
    if not any(cs.count_key(original_text, k) > 0 for k in _SCHEME_KEYS):
        ed.txt = cs.set_list_values(ed.txt, "physics_suite", ["'CONUS'"], new_key_group="physics")
        ed.log("physics_suite", None, "'CONUS'", "no &physics scheme key present at all")
    suite_val = cs.get_first(ed.txt, "physics_suite")
    if not suite_val:
        return
    placeholder_keys = [k for k in _SCHEME_KEYS if k != "physics_suite"
                        and (cs.get_first(ed.txt, k) or "").strip().rstrip(",") == "-1"]
    if placeholder_keys:
        ed.log("physics_suite_placeholders", suite_val, suite_val,
               "physics_suite=%s with -1 placeholders for %s -- these mean 'use "
               "the suite's own default', not unset; treat an explicit scheme "
               "value in a comparison as EQUIVALENT to -1 here, not a mismatch"
               % (suite_val, ", ".join(placeholder_keys)))


def _fill_bdy_quilt_and_groups(ed: _NamelistEditor) -> None:
    """&bdy_control/&namelist_quilt defaults (same as namelist_fields_from_wrfinput_facts), then
    every still-missing required group created empty (T16, job 986)."""
    ed.fill_if_missing("spec_bdy_width", _DEFAULT_SPEC_BDY_WIDTH,
                       "default lateral boundary width", group_default="bdy_control")
    ed.fill_if_missing("spec_zone", 1, "default", group_default="bdy_control")
    ed.fill_if_missing("relax_zone", 4, "default", group_default="bdy_control")
    ed.fill_values_if_missing("specified", [".true."] + [".false."] * (ed.max_dom - 1),
                              "d01 is specified by the lateral boundary file; nests are not",
                              group_default="bdy_control")
    ed.fill_values_if_missing("nested", [".false."] + [".true."] * (ed.max_dom - 1),
                              "every domain but d01 is nested", group_default="bdy_control")
    ed.fill_if_missing("nio_tasks_per_group", 0, "default", group_default="namelist_quilt")
    ed.fill_if_missing("nio_groups", 1, "default", group_default="namelist_quilt")
    for group in _ALWAYS_GROUPS:
        if re.search(r"&%s\b" % re.escape(group), ed.txt, re.I) is None:
            ed.txt = ed.txt.rstrip("\n") + "\n\n&%s\n/\n" % group
            ed.log("&%s" % group, None, "(empty group)",
                   "WRF's namelist reader requires this group to exist even when empty")


def complete_namelist(nl_text: str, facts: dict, registry: dict | None = None,
                      vertical_grid: dict | None = None) -> tuple[str, list[dict]]:
    """Make a case namelist.input runnable for the rebuild span -> (new_text, changes).
    facts: max_dom, num_metgrid_levels, [num_metgrid_soil_levels, num_land_cat], interval_seconds,
    start_dt/end_dt (the span), dx_km (d01), parent_grid_ratio. Met/span keys are always synced,
    defaults only filled; inserted keys go to their catalog group. vertical_grid (plan.json's
    {'eta_levels', 'p_top_pa'}) overrides the vertical grid. Stdlib only: runs in the batch job."""
    max_dom = int(facts["max_dom"])
    registry = registry if registry is not None else load_registry()
    ed = _NamelistEditor(nl_text, registry, max_dom)
    run_s = _sync_met_and_span(ed, facts)
    if vertical_grid is not None:
        _apply_vertical_grid(ed, vertical_grid)
    else:
        ed.fill_if_missing("e_vert", _DEFAULT_E_VERT, "default vertical levels")
        ed.fill_if_missing("p_top_requested", _DEFAULT_P_TOP_PA, "default model top (Pa)")
    _fill_run_defaults(ed, facts, run_s)
    _fill_physics_suite(ed, nl_text)
    if re.search(r"(?im)^[^!\n]*(?<![a-z0-9_])mp_physics\s*=[^\n/!]*(?<![0-9])7(?![0-9])", ed.txt):
        ed.sync("do_radar_ref", 1, "mp_physics=7 writes refl_10cm, allocated only then",
                group_default="physics")
    _fill_bdy_quilt_and_groups(ed)
    return ed.txt, ed.changes


# ===========================================================================
# K1: namelist-from-wrfinput -- a complete namelist.input from wrfinput_d0N (+ wrfbdy_d01) alone.
# namelist_fields_from_wrfinput_facts is pure; cmd_namelist_from_wrfinput does the netCDF4 reads.
# ===========================================================================

# Per-domain grid attrs, copied as is (wrfinput's e_we/e_sn == *_GRID_DIMENSION; verified on scc).
_GRID_ATTRS = [
    ("e_we", "WEST-EAST_GRID_DIMENSION"), ("e_sn", "SOUTH-NORTH_GRID_DIMENSION"),
    ("e_vert", "BOTTOM-TOP_GRID_DIMENSION"),
    ("grid_id", "GRID_ID"), ("parent_id", "PARENT_ID"),
    ("i_parent_start", "I_PARENT_START"), ("j_parent_start", "J_PARENT_START"),
    ("parent_grid_ratio", "PARENT_GRID_RATIO"),
]
_GRID_FLOAT_ATTRS = [("dx", "DX"), ("dy", "DY")]

# Physics options present as wrfinput global attrs (radt is not -- it is a rule default).
_PHYSICS_ATTRS = [
    ("mp_physics", "MP_PHYSICS"), ("ra_lw_physics", "RA_LW_PHYSICS"),
    ("ra_sw_physics", "RA_SW_PHYSICS"), ("sf_sfclay_physics", "SF_SFCLAY_PHYSICS"),
    ("sf_surface_physics", "SF_SURFACE_PHYSICS"), ("bl_pbl_physics", "BL_PBL_PHYSICS"),
    ("cu_physics", "CU_PHYSICS"), ("diff_opt", "DIFF_OPT"), ("km_opt", "KM_OPT"),
    ("damp_opt", "DAMP_OPT"), ("hypsometric_opt", "HYPSOMETRIC_OPT"),
    ("use_theta_m", "USE_THETA_M"), ("gwd_opt", "GWD_OPT"),
    ("sf_urban_physics", "SF_URBAN_PHYSICS"),
]
_DT_FRACT_DEN = 100   # time_step_fract_den: DT exact to 2 decimals


def render_namelist_input(groups: list) -> str:
    """[(group, [(key, [v1..vN]), ...]), ...] -> namelist.input text ('key = v1, ..., vN,')."""
    lines = []
    for group, items in groups:
        lines.append("&%s" % group)
        for key, values in items:
            lines.append(" %s = %s," % (key, ", ".join(str(v) for v in values)))
        lines.append("/")
        lines.append("")
    return "\n".join(lines)


class _FieldSet:
    """namelist_fields_from_wrfinput_facts' accumulator: key values, provenance tags, notes."""

    def __init__(self, domain_facts: list[dict], registry: dict):
        self.domain_facts = domain_facts
        self.registry = registry
        self.max_dom = len(domain_facts)
        self.items: dict[str, list] = {}   # key -> values; its &group is decided at the end
        self.provenance: dict[str, str] = {}
        self.notes: list[str] = []

    def put(self, key: str, values: list, tag: str) -> None:
        self.items[key] = values
        self.provenance[key] = tag

    def ndom_vals(self, key: str, value) -> list:
        """[value] x max_dom for a catalog per-domain key, else [value] (T14)."""
        return [value] * self.max_dom if registry_per_domain(key, self.registry) else [value]

    def attr_per_domain(self, attr: str) -> list:
        vals = []
        for d in self.domain_facts:
            if attr not in d["attrs"]:
                raise RebuildError("namelist-from-wrfinput: a domain's wrfinput is missing "
                                   "global attr %s" % attr)
            vals.append(d["attrs"][attr])
        return vals


def _grid_and_physics_fields(fs: _FieldSet) -> None:
    for key, attr in _GRID_ATTRS:
        fs.put(key, [int(round(float(v))) for v in fs.attr_per_domain(attr)], "wrfinput_attr")
    for key, attr in _GRID_FLOAT_ATTRS:
        fs.put(key, [float(v) for v in fs.attr_per_domain(attr)], "wrfinput_attr")
    fs.put("parent_time_step_ratio", list(fs.items["parent_grid_ratio"]), "wrfinput_attr")
    fs.put("max_dom", [fs.max_dom], "wrfinput_attr")
    for key, attr in _PHYSICS_ATTRS:
        if any(attr not in d["attrs"] for d in fs.domain_facts):
            continue   # not every WRF build writes every option
        if registry_per_domain(key, fs.registry):
            vals = [int(round(float(d["attrs"][attr]))) for d in fs.domain_facts]
        else:   # single-valued per the catalog (e.g. damp_opt): d01's copy (T14)
            vals = [int(round(float(fs.domain_facts[0]["attrs"][attr])))]
        fs.put(key, vals, "wrfinput_attr")


def _d01_fields(fs: _FieldSet, d01: dict) -> None:
    """time_step (+ fract num/den for a non-integral DT), vertical grid, soil layers, land cats."""
    dt = float(d01["attrs"]["DT"])
    if abs(dt - round(dt)) < 1e-6:
        fs.put("time_step", [int(round(dt))], "wrfinput_attr")
    else:
        whole = int(math.floor(dt))
        num = int(round((dt - whole) * _DT_FRACT_DEN))
        fs.put("time_step", [whole], "wrfinput_attr")
        fs.put("time_step_fract_num", [num], "wrfinput_attr")
        fs.put("time_step_fract_den", [_DT_FRACT_DEN], "wrfinput_attr")
    if d01.get("znw") is not None:
        fs.put("eta_levels", list(d01["znw"]), "wrfinput_attr")
    if d01.get("p_top") is not None:
        fs.put("p_top_requested", [d01["p_top"]], "wrfinput_attr")
    if d01.get("dzs") is not None:
        fs.put("num_soil_layers", [len(d01["dzs"])], "wrfinput_attr")
    if "NUM_LAND_CAT" in d01["attrs"]:
        fs.put("num_land_cat", [int(d01["attrs"]["NUM_LAND_CAT"])], "wrfinput_attr")


def _start_fields(fs: _FieldSet, d01: dict) -> datetime.datetime:
    """start_* from SIMULATION_START_DATE (noting a mismatch with Times[0]); returns it."""
    sim_start_raw = d01["attrs"].get("SIMULATION_START_DATE")
    if not sim_start_raw:
        raise RebuildError("namelist-from-wrfinput: wrfinput_d01 has no SIMULATION_START_DATE")
    start_dt = datetime.datetime.strptime(str(sim_start_raw).rstrip("\x00"), _WRF_TIME_FMT)
    if d01.get("times"):
        t0 = d01["times"][0]
        if t0 != start_dt:
            fs.notes.append("wrfinput_d01 SIMULATION_START_DATE=%s != Times[0]=%s -- using "
                            "SIMULATION_START_DATE"
                            % (start_dt.strftime(_WRF_TIME_FMT), t0.strftime(_WRF_TIME_FMT)))
    for k, v in _dt_parts(start_dt):
        fs.put("start_" + k, [v] * fs.max_dom, "wrfinput_attr")
    return start_dt


def _span_fields(fs: _FieldSet, start_dt: datetime.datetime, wrfbdy_times: list | None,
                 hours: float | None) -> int:
    """interval_seconds (wrfbdy Times spacing) and end_*/run_* (--hours, else the wrfbdy span +
    one interval, else 6 h); returns the run length (s)."""
    if wrfbdy_times and len(wrfbdy_times) >= 2:
        interval_s = int(round((wrfbdy_times[1] - wrfbdy_times[0]).total_seconds()))
        fs.put("interval_seconds", [interval_s], "wrfbdy")
    else:
        interval_s = _DEFAULT_INTERVAL_S
        fs.put("interval_seconds", [interval_s], "rule_default")
        fs.notes.append("no wrfbdy_d01 Times to derive interval_seconds from -- rule_default 3h "
                        "(10800s)")
    if hours is not None:
        run_s = int(round(hours * 3600))
        run_prov = "given"
    elif wrfbdy_times:
        run_s = int(round((wrfbdy_times[-1] - start_dt).total_seconds())) + interval_s
        run_prov = "wrfbdy"
        fs.notes.append("--hours not given -- run length taken from wrfbdy span (last Times + one "
                        "interval) = %.2fh" % (run_s / 3600.0))
    else:
        run_s = _DEFAULT_RUN_S
        run_prov = "rule_default"
        fs.notes.append("neither --hours nor wrfbdy_d01 Times available -- run length defaulted "
                        "to rule_default 6h")
    end_dt = start_dt + datetime.timedelta(seconds=run_s)
    for k, v in _dt_parts(end_dt):
        fs.put("end_" + k, [v] * fs.max_dom, run_prov)
    fs.put("run_days", [0], run_prov)
    fs.put("run_hours", [run_s // 3600], run_prov)
    fs.put("run_minutes", [(run_s % 3600) // 60], run_prov)
    fs.put("run_seconds", [run_s % 60], run_prov)
    return run_s


def _rule_default_fields(fs: _FieldSet, dx_km: float, run_s: int,
                         wrfbdy_bdy_width: int | None) -> None:
    """radt, output/restart/io_form defaults, &bdy_control (T15(c)) and &namelist_quilt."""
    radt = max(1, int(round(dx_km)))   # min; WRF guideline 1 min per km of d01 dx
    fs.put("radt", [radt] * fs.max_dom, "rule_default")
    fs.notes.append("radt = round(dx_km) = %d min (WRF guideline: 1 min per km of d01 dx) -- "
                    "rule_default" % radt)
    fs.put("history_interval", fs.ndom_vals("history_interval", _DEFAULT_HISTORY_MIN),
           "rule_default")
    fs.put("frames_per_outfile", fs.ndom_vals("frames_per_outfile", 1), "rule_default")
    fs.put("restart", fs.ndom_vals("restart", ".false."), "rule_default")
    fs.put("restart_interval", fs.ndom_vals("restart_interval", run_s // 60), "rule_default")
    for k in _IO_FORM_KEYS:
        fs.put(k, fs.ndom_vals(k, _NETCDF_IO_FORM), "rule_default")

    if wrfbdy_bdy_width is not None:
        fs.put("spec_bdy_width", [int(wrfbdy_bdy_width)], "wrfbdy")
    else:
        fs.put("spec_bdy_width", [_DEFAULT_SPEC_BDY_WIDTH], "rule_default")
        fs.notes.append("no wrfbdy_d01 bdy_width dimension available -- spec_bdy_width "
                        "rule_default 5")
    fs.put("spec_zone", [1], "rule_default")
    fs.put("relax_zone", [4], "rule_default")
    # d01 is specified by the lateral boundary file and not nested; every nest the reverse.
    fs.put("specified", [".true."] + [".false."] * (fs.max_dom - 1), "rule_default")
    fs.put("nested", [".false."] + [".true."] * (fs.max_dom - 1), "rule_default")
    # &namelist_quilt: real WRF keys the catalog does not cover (same gap as T11).
    fs.put("nio_tasks_per_group", [0], "rule_default")
    fs.put("nio_groups", [1], "rule_default")


def _bucket_by_group(fs: _FieldSet) -> list:
    """Each key into its catalog &group (T15(a)); all _ALWAYS_GROUPS emitted, even empty (T15(b))."""
    always_groups = list(_ALWAYS_GROUPS)
    fs.notes.append("emitted groups: wtune_catalog_registry.json does not record which groups "
                    "WRF's namelist reader itself reads (only per-key group membership) -- using "
                    "the given list: %s" % ", ".join(always_groups))
    quilt_fallback_keys = ("nio_tasks_per_group", "nio_groups")
    buckets = {g: [] for g in always_groups}
    for key, values in fs.items.items():
        group = registry_group_for_key(key, fs.registry)
        if group is None:
            group = "namelist_quilt" if key in quilt_fallback_keys else "time_control"
            fs.notes.append("key %r has no Registry catalog entry -- placed in &%s by fallback"
                            % (key, group))
        if group not in buckets:
            buckets[group] = []
            fs.notes.append("key %r's catalog group &%s is not in the always-emitted group list "
                            "-- emitting &%s anyway" % (key, group, group))
        buckets[group].append((key, values))
    groups = [(g, buckets[g]) for g in always_groups]
    groups += [(g, buckets[g]) for g in sorted(buckets) if g not in always_groups]
    return groups


def namelist_fields_from_wrfinput_facts(domain_facts: list[dict], wrfbdy_times: list | None = None,
                                        hours: float | None = None, registry: dict | None = None,
                                        wrfbdy_bdy_width: int | None = None
                                        ) -> tuple[list, dict, list[str]]:
    """Pure (no netCDF4): namelist groups from per-domain wrfinput facts -> (groups, provenance,
    notes). domain_facts: [{'attrs', ['znw', 'p_top', 'dzs', 'times'] on d01}, ...], d01 first.
    provenance: {key: 'wrfinput_attr'|'wrfbdy'|'given'|'rule_default'}. Raises RebuildError on
    no facts, a missing grid attr, or no SIMULATION_START_DATE."""
    if not domain_facts:
        raise RebuildError("namelist-from-wrfinput: no domain facts given")
    registry = registry if registry is not None else load_registry()
    fs = _FieldSet(domain_facts, registry)
    d01 = domain_facts[0]
    _grid_and_physics_fields(fs)
    dx_km = float(d01["attrs"]["DX"]) / 1000.0
    _d01_fields(fs, d01)
    start_dt = _start_fields(fs, d01)
    run_s = _span_fields(fs, start_dt, wrfbdy_times, hours)
    _rule_default_fields(fs, dx_km, run_s, wrfbdy_bdy_width)
    groups = _bucket_by_group(fs)
    return groups, fs.provenance, fs.notes


def _read_wrf_times(ds):
    """[datetime, ...] from a WRF file's Times(Time,DateStrLen) char variable. Same decode
    approach as state_correction_model._read_times / state_timeline.read_times -- kept LOCAL
    (those files are owned by other concurrent agents, not imported across)."""
    out = []
    for row in ds.variables["Times"][:]:
        if hasattr(row, "tobytes"):
            s = row.tobytes().decode("utf-8", "replace")
        else:
            s = "".join(c.decode("utf-8", "replace") if isinstance(c, (bytes, bytearray))
                        else str(c) for c in row)
        s = s.split("\x00")[0]
        out.append(datetime.datetime.strptime(s, _WRF_TIME_FMT))
    return out


def _read_wrfinput_domain_facts(path, with_vertical=False):
    """netCDF4 read (K1 -- PLAN/login-time only, never in a batch job): {'attrs': {...global
    attrs...}, 'times': [...] (if Times present), and, when with_vertical (d01 only):
    'znw'/'p_top'/'dzs'}."""
    import numpy as np
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        facts = {"attrs": {a: ds.getncattr(a) for a in ds.ncattrs()}}
        if "Times" in ds.variables:
            facts["times"] = _read_wrf_times(ds)
        if with_vertical:
            if "ZNW" in ds.variables:
                facts["znw"] = [round(float(v), 6) for v in np.asarray(ds.variables["ZNW"][0])]
            if "P_TOP" in ds.variables:
                facts["p_top"] = float(np.asarray(ds.variables["P_TOP"][:]).reshape(-1)[0])
            if "DZS" in ds.variables:
                facts["dzs"] = [float(v) for v in np.asarray(ds.variables["DZS"][0])]
        return facts
    finally:
        ds.close()


def cmd_namelist_from_wrfinput(case_dir: str, out_path: str, hours: float | None = None
                               ) -> tuple[str, dict, list[str]]:
    wrfinput_names = sorted(p for p in os.listdir(case_dir) if re.match(r"^wrfinput_d\d\d$", p))
    if not wrfinput_names:
        raise RebuildError("namelist-from-wrfinput: no wrfinput_d0N in %s" % case_dir)
    domain_facts = [_read_wrfinput_domain_facts(os.path.join(case_dir, name),
                                                with_vertical=(i == 0))
                    for i, name in enumerate(wrfinput_names)]
    wrfbdy_path = os.path.join(case_dir, "wrfbdy_d01")
    wrfbdy_times = None
    wrfbdy_bdy_width = None
    if os.path.exists(wrfbdy_path):
        import netCDF4
        ds = netCDF4.Dataset(wrfbdy_path)
        try:
            wrfbdy_times = _read_wrf_times(ds)
            # the boundary width WRF actually wrote, not the Registry default 5 (T15(c))
            if "bdy_width" in ds.dimensions:
                wrfbdy_bdy_width = len(ds.dimensions["bdy_width"])
        finally:
            ds.close()

    groups, provenance, notes = namelist_fields_from_wrfinput_facts(
        domain_facts, wrfbdy_times=wrfbdy_times, hours=hours,
        wrfbdy_bdy_width=wrfbdy_bdy_width)
    text = render_namelist_input(groups)
    with open(out_path, "w") as f:
        f.write(text)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    prov_path = os.path.join(out_dir, "namelist_provenance.json")
    with open(prov_path, "w") as f:
        json.dump({"provenance": provenance, "notes": notes}, f, indent=2)
    for n in notes:
        print("NOTE: %s" % n)
    print("wrote %s (%d domain(s)) -- provenance: %s" % (out_path, len(domain_facts), prov_path))
    return text, provenance, notes


# ===========================================================================
# K2: new-domain -- d01 only, for a user who knows only where and when (no basis for a nest).
# ===========================================================================

def new_domain_fields(center_lat: float, center_lon: float, size_km: tuple[float, float],
                      dx_km: float, start_dt: datetime.datetime, hours: float, e_vert: int = 45
                      ) -> tuple[list, dict, dict, list[str]]:
    """Pure: a d01-only domain from center/size/dx/start (K2) ->
    (namelist_groups, wps_fields, provenance, notes). size_km: (W, H) km; hours: run length.
    Projection by |lat|: < 25 mercator, <= 65 lambert, else polar; e_we/e_sn = round(size/dx) + 1;
    CONUS suite (TROPICAL below 25 deg), cu_physics=0 below 4 km. provenance: 'given' only for keys
    the four inputs set directly, every rule-derived key 'rule_default'."""
    w_km, h_km = size_km
    e_we = int(round(w_km / dx_km)) + 1
    e_sn = int(round(h_km / dx_km)) + 1
    time_step = max(1, int(math.floor(6.0 * dx_km)))
    alat = abs(center_lat)
    if alat < 25.0:
        map_proj, truelat1, truelat2, stand_lon = "mercator", center_lat, None, None
    elif alat <= 65.0:
        map_proj, truelat1, truelat2, stand_lon = "lambert", center_lat, center_lat, center_lon
    else:
        map_proj, truelat1, truelat2, stand_lon = "polar", center_lat, None, None
    physics_suite = "TROPICAL" if alat < 25.0 else "CONUS"
    convection_permitting = dx_km < 4.0
    radt = max(1, int(round(dx_km)))
    dx_m = dx_km * 1000.0
    end_dt = start_dt + datetime.timedelta(hours=hours)
    run_s = int(round(hours * 3600.0))

    provenance = {}
    notes = []

    def prov(key, tag):
        provenance[key] = tag

    domains_items = [
        ("max_dom", [1]), ("e_we", [e_we]), ("e_sn", [e_sn]), ("e_vert", [e_vert]),
        ("dx", [dx_m]), ("dy", [dx_m]), ("time_step", [time_step]),
        ("grid_id", [1]), ("parent_id", [0]), ("i_parent_start", [1]), ("j_parent_start", [1]),
        ("parent_grid_ratio", [1]), ("parent_time_step_ratio", [1]),
    ]
    for k in ("max_dom", "e_we", "e_sn", "time_step", "grid_id", "parent_id",
             "i_parent_start", "j_parent_start", "parent_grid_ratio", "parent_time_step_ratio"):
        prov(k, "rule_default")
    prov("e_vert", "given" if e_vert != 45 else "rule_default")
    prov("dx", "given")
    prov("dy", "given")
    if e_vert == 45:
        notes.append("e_vert=45 (WRF's common default) -- --e-vert not given")

    time_control_items = []
    for k, v in (("year", start_dt.year), ("month", start_dt.month), ("day", start_dt.day),
                ("hour", start_dt.hour), ("minute", start_dt.minute),
                ("second", start_dt.second)):
        time_control_items.append(("start_" + k, [v]))
        prov("start_" + k, "given")
    for k, v in (("year", end_dt.year), ("month", end_dt.month), ("day", end_dt.day),
                ("hour", end_dt.hour), ("minute", end_dt.minute), ("second", end_dt.second)):
        time_control_items.append(("end_" + k, [v]))
        prov("end_" + k, "given")
    time_control_items += [("run_days", [0]), ("run_hours", [run_s // 3600]),
                           ("run_minutes", [(run_s % 3600) // 60]), ("run_seconds", [run_s % 60])]
    for k in ("run_days", "run_hours", "run_minutes", "run_seconds"):
        prov(k, "given")
    time_control_items += [
        ("interval_seconds", [10800]), ("history_interval", [60]),
        ("frames_per_outfile", [1]), ("restart", [".false."]),
        ("restart_interval", [max(1, run_s // 60)]),
    ]
    for k in ("interval_seconds", "history_interval", "frames_per_outfile", "restart",
             "restart_interval"):
        prov(k, "rule_default")
    for k in ("io_form_history", "io_form_restart", "io_form_input", "io_form_boundary"):
        time_control_items.append((k, [2]))
        prov(k, "rule_default")

    physics_items = [("physics_suite", ["'%s'" % physics_suite])]
    prov("physics_suite", "rule_default")
    if convection_permitting:
        physics_items.append(("cu_physics", [0]))
        prov("cu_physics", "rule_default")
        notes.append("cu_physics=0 (convection-permitting): dx_km=%.3f < 4 km" % dx_km)
    physics_items.append(("radt", [radt]))
    prov("radt", "rule_default")
    notes.append("radt = round(dx_km) = %d min -- rule_default" % radt)
    notes.append("projection=%s (|center_lat|=%.2f): %s" % (
        map_proj, alat,
        "< 25 -> mercator" if alat < 25.0 else
        "<= 65 -> lambert" if alat <= 65.0 else "> 65 -> polar"))

    groups = [("time_control", time_control_items), ("domains", domains_items),
             ("physics", physics_items)]

    wps_fields = {
        "max_dom": 1, "map_proj": map_proj, "ref_lat": center_lat, "ref_lon": center_lon,
        "truelat1": truelat1, "truelat2": truelat2, "stand_lon": stand_lon,
        "dx": dx_m, "dy": dx_m, "e_we": [e_we], "e_sn": [e_sn],
        "start_date": start_dt.strftime(_WRF_TIME_FMT), "end_date": end_dt.strftime(_WRF_TIME_FMT),
        "interval_seconds": 10800, "parent_id": [1], "parent_grid_ratio": [1],
        "i_parent_start": [1], "j_parent_start": [1], "geog_data_res": ["default"],
        "geog_data_path": cs.geog_data_path_default(),
    }
    prov("ref_lat", "given")
    prov("ref_lon", "given")
    prov("map_proj", "rule_default")
    prov("truelat1", "rule_default")
    if truelat2 is not None:
        prov("truelat2", "rule_default")
    if stand_lon is not None:
        prov("stand_lon", "rule_default")

    return groups, wps_fields, provenance, notes


def cmd_new_domain(center: str | None = None, size_km: str | None = None,
                   dx_km: float | str | None = None, start: str | None = None,
                   hours: float | None = None, e_vert: int = 45, out_dir: str | None = None
                   ) -> tuple[str, str, dict, list[str]]:
    """center: 'LAT,LON'. size_km: 'W,H'. start: 'YYYY-MM-DD_HH:MM:SS'. Refuses with a clear
    message if only SOME of center/size_km/dx_km/start is given (K4's front door may call this
    with any of them missing -- e.g. inventory case (e): nothing located the domain at all, or
    only some of the four)."""
    given = {"--center": center, "--size-km": size_km, "--dx-km": dx_km, "--start": start}
    present = {k: v for k, v in given.items() if v is not None}
    if present and len(present) < len(given):
        raise RebuildError(
            "new-domain: --center/--size-km/--dx-km/--start must be given ALL TOGETHER or not "
            "at all -- got %s, missing %s"
            % (", ".join(sorted(present)), ", ".join(sorted(set(given) - set(present)))))
    if not present:
        raise RebuildError("new-domain: --center/--size-km/--dx-km/--start are all required")
    if hours is None:
        raise RebuildError("new-domain: --hours is required")
    if out_dir is None:
        raise RebuildError("new-domain: --out is required")

    center_lat, center_lon = (float(x) for x in center.split(","))
    w_km, h_km = (float(x) for x in size_km.split(","))
    dx_km_f = float(dx_km)
    start_dt = datetime.datetime.strptime(start, _WRF_TIME_FMT)

    groups, wps_fields, provenance, notes = new_domain_fields(
        center_lat, center_lon, (w_km, h_km), dx_km_f, start_dt, float(hours), e_vert=e_vert)

    os.makedirs(out_dir, exist_ok=True)
    nl_text = render_namelist_input(groups)
    nl_path = os.path.join(out_dir, "namelist.input")
    with open(nl_path, "w") as f:
        f.write(nl_text)
    nlw_text = cs.render_namelist_wps(wps_fields)
    nlw_path = os.path.join(out_dir, "namelist.wps")
    with open(nlw_path, "w") as f:
        f.write(nlw_text)
    prov_path = os.path.join(out_dir, "namelist_provenance.json")
    with open(prov_path, "w") as f:
        json.dump({"provenance": provenance, "notes": notes}, f, indent=2)
    for n in notes:
        print("NOTE: %s" % n)
    print("wrote %s and %s -- provenance: %s" % (nl_path, nlw_path, prov_path))
    return nl_text, nlw_text, provenance, notes


# ===========================================================================
# K4: `complete` -- the front door. Inventories --in DIR (file presence only) and dispatches to
# K1 (namelist-from-wrfinput), K2 (new-domain), K3 (--keep-wrfinput) or the rebuild flow, in
# choose_completion_path's (a)..(f) priority order.
# ===========================================================================

def inventory_case_dir(in_dir: str) -> dict:
    """PURE filesystem check (no netCDF4): {'namelist_input': bool, 'namelist_wps': bool,
    'wrfinput_domains': [wrfinput_d01, ...] sorted, or [] if none, 'wrfbdy_d01': bool}."""
    wrfinput_domains = []
    if os.path.isdir(in_dir):
        wrfinput_domains = sorted(p for p in os.listdir(in_dir)
                                  if re.match(r"^wrfinput_d\d\d$", p))
    return {
        "namelist_input": os.path.exists(os.path.join(in_dir, "namelist.input")),
        "namelist_wps": os.path.exists(os.path.join(in_dir, "namelist.wps")),
        "wrfinput_domains": wrfinput_domains,
        "wrfbdy_d01": os.path.exists(os.path.join(in_dir, "wrfbdy_d01")),
    }


def choose_completion_path(inventory: dict, have_center_size_dx_start: bool = False) -> dict:
    """PURE decision logic (K4). `inventory`: inventory_case_dir's result.
    `have_center_size_dx_start`: True iff --center/--size-km/--dx-km/--start were ALL given
    (the caller validates all-or-nothing before this point -- see cmd_new_domain's own check,
    reused identically by cmd_complete). Returns {'path': one of 'as_is'/'k1_and_or_k3'/
    'existing_rebuild'/'k2_then_rebuild'/'refuse', 'needs_k1': bool, 'needs_k3': bool,
    'reason': str}."""
    has_namelist = inventory["namelist_input"]
    has_wrfinput = bool(inventory["wrfinput_domains"])
    has_wrfbdy = inventory["wrfbdy_d01"]
    has_projection = inventory["namelist_wps"] or has_wrfinput

    # (a) everything present -> as is (+ a t0 consistency check against the fetched analysis,
    # done by the ORCHESTRATOR via compare()/analog -- see cmd_complete).
    if has_namelist and has_wrfinput and has_wrfbdy:
        return {"path": "as_is", "needs_k1": False, "needs_k3": False,
               "reason": "namelist.input + wrfinput_d0N + wrfbdy_d01 all present -- use as is "
                         "(compared against the fetched analysis at t0)"}

    # (b)+(c): wrfinput present -> K1 if there's no namelist yet, K3 (--keep-wrfinput) if
    # there's no wrfbdy yet -- independent flags, since a case can need either, both, or (if it
    # has a namelist AND a wrfbdy already) neither of these two once wrfinput itself is present.
    needs_k1 = has_wrfinput and not has_namelist
    needs_k3 = has_wrfinput and not has_wrfbdy
    if needs_k1 or needs_k3:
        parts = []
        if needs_k1:
            parts.append("K1 (namelist-from-wrfinput: no namelist.input)")
        if needs_k3:
            parts.append("K3 --keep-wrfinput (no wrfbdy_d01)")
        return {"path": "k1_and_or_k3", "needs_k1": needs_k1, "needs_k3": needs_k3,
               "reason": "wrfinput_d0N present -- " + " + ".join(parts)}

    # (d) namelist(+projection), no wrfinput -> the existing rebuild flow (it fetches its own
    # wrfinput/wrfbdy from the analysis).
    if has_namelist and not has_wrfinput and has_projection:
        return {"path": "existing_rebuild", "needs_k1": False, "needs_k3": False,
               "reason": "namelist.input (+ projection source) present, no wrfinput_d0N -- "
                         "existing rebuild flow"}

    # (e) nothing that locates the domain except --center/--size-km/--dx-km/--start -> K2 then
    # the existing rebuild flow on K2's own output.
    if not has_namelist and not has_wrfinput and have_center_size_dx_start:
        return {"path": "k2_then_rebuild", "needs_k1": False, "needs_k3": False,
               "reason": "no namelist.input/wrfinput_d0N -- --center/--size-km/--dx-km/--start "
                         "given -- K2 (new-domain) then rebuild"}

    # (f) refuse, naming exactly what's needed.
    need = []
    if not has_namelist:
        need.append("namelist.input")
    if not has_wrfinput:
        need.append("wrfinput_d0N")
    if has_namelist and not has_projection:
        need.append("a projection source (namelist.wps, or a wrfinput_d0N to read MAP_PROJ "
                    "from)")
    if not have_center_size_dx_start:
        need.append("--center/--size-km/--dx-km/--start (all four) to build a domain from "
                    "scratch")
    return {"path": "refuse", "needs_k1": False, "needs_k3": False,
           "reason": "nothing locates the domain -- need one of: " + "; or ".join(need)}


def _print_completion_table(rows):
    print("%-42s %s" % ("item", "method"))
    print("-" * 60)
    for r in rows:
        print("%-42s %s" % (r["item"], r["method"]))


def cmd_complete(in_dir: str, out_dir: str, center: str | None = None, size_km: str | None = None,
                 dx_km: float | None = None, start: str | None = None, hours: float | None = None,
                 ml_model: str | None = None) -> dict:
    """The front door (K4). Inventories --in DIR, chooses a completion path (see
    choose_completion_path), dispatches to K1/K2/K3/the existing rebuild plan step, and writes
    OUT/completion_report.json (one row per WRF input item, method in {given, wrfinput_attr,
    wrfbdy, rule_default, analysis_rebuild, ml_state_correction, analog, missing}) plus a
    human-readable table to stdout. Like `rebuild plan`, this STOPS after writing the plan +
    instructions for the batch stage -- it does not itself run geogrid/ungrib/real.exe."""
    given = {"center": center, "size_km": size_km, "dx_km": dx_km, "start": start}
    present = {k: v for k, v in given.items() if v is not None}
    if present and len(present) < len(given):
        raise RebuildError(
            "complete: --center/--size-km/--dx-km/--start must be given ALL TOGETHER or not at "
            "all -- got %s, missing %s"
            % (", ".join(sorted(present)), ", ".join(sorted(set(given) - set(present)))))
    have_four = len(present) == len(given)

    inventory = inventory_case_dir(in_dir)
    decision = choose_completion_path(inventory, have_center_size_dx_start=have_four)
    print("complete: inventory = %s" % json.dumps(inventory))
    print("complete: path = %s -- %s" % (decision["path"], decision["reason"]))
    if decision["path"] == "refuse":
        raise RebuildError("complete: " + decision["reason"])

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    case_dir_for_plan = in_dir
    namelist_for_plan = os.path.join(in_dir, "namelist.input")

    if decision["path"] == "as_is":
        rows.append({"item": "namelist.input", "method": "given"})
        rows.append({"item": "wrfinput_d0N", "method": "given"})
        rows.append({"item": "wrfbdy_d01", "method": "given"})
        rows.append({"item": "t0 consistency vs. analysis", "method": "analysis_rebuild",
                    "note": "run `case_rebuild.py compare --original <wrfinput_d0N> --rebuilt "
                            "<a freshly fetched/rebuilt t0 wrfinput>`; on state-mismatch, "
                            "`case_rebuild.py analog` suggests an analog date"})
    elif decision["path"] == "k2_then_rebuild":
        cmd_new_domain(center=center, size_km=size_km, dx_km=dx_km, start=start, hours=hours,
                      out_dir=out_dir)
        case_dir_for_plan = out_dir
        namelist_for_plan = os.path.join(out_dir, "namelist.input")
        rows.append({"item": "domain (center/size/dx/start)", "method": "given"})
        rows.append({"item": "namelist.input/namelist.wps (K2 rules)", "method": "rule_default"})
    else:
        if decision["needs_k1"]:
            namelist_for_plan = os.path.join(out_dir, "namelist.input")
            _text, provenance, _notes = cmd_namelist_from_wrfinput(in_dir, namelist_for_plan,
                                                                    hours=hours)
            for key, tag in provenance.items():
                rows.append({"item": "namelist.%s" % key, "method": tag})
        else:
            rows.append({"item": "namelist.input", "method": "given"})
        if decision["needs_k3"]:
            rows.append({"item": "interior state (wrfinput_d0N)", "method": "given"})
            rows.append({"item": "boundaries (wrfbdy_d01)", "method": "analysis_rebuild"})
            rows.append({"item": "finalize step", "method": "missing",
                        "note": "run `case_rebuild.py rebuild finalize --out %s "
                                "--keep-wrfinput` after the batch job" % out_dir})
        elif decision["path"] == "existing_rebuild" or decision["needs_k1"]:
            rows.append({"item": "wrfinput_d0N + wrfbdy_d01", "method": "analysis_rebuild"})

    if ml_model:
        rows.append({"item": "post-real.exe state completion", "method": "ml_state_correction"})

    cmd_rebuild_plan(case_dir_for_plan, out_dir, namelist=namelist_for_plan, ml_model=ml_model)

    completion_report = {"inventory": inventory, "decision": decision, "rows": rows,
                         "plan_dir": os.path.abspath(out_dir)}
    with open(os.path.join(out_dir, "completion_report.json"), "w") as f:
        json.dump(completion_report, f, indent=2, default=str)
    _print_completion_table(rows)
    print("complete: plan written to %s -- run `rebuild fetch`/`rebuild stage` next (batch "
         "stage instructions above)" % out_dir)
    return completion_report


# ===========================================================================
# R6-A: rebuild -- span computation, plan/fetch/stage/finalize lifecycle
# ===========================================================================

def rebuild_span(nl: dict, date_shift_to: str | None = None) -> tuple:
    """(stated_start, stated_end, analog_start_or_None, span_start, span_end, run_s).
    span_start/span_end = the (possibly date-shifted) window rebuild actually fetches/runs;
    stated_start/stated_end are the case's own literal namelist.input dates."""
    stated_start, stated_end, run_s = cs.run_span(nl)
    analog_start = None
    if date_shift_to:
        analog_start = datetime.datetime.strptime(date_shift_to, _WRF_TIME_FMT)
    span_start = analog_start or stated_start
    span_end = span_start + datetime.timedelta(seconds=run_s)
    return stated_start, stated_end, analog_start, span_start, span_end, run_s


def cmd_rebuild_plan(case_dir: str, out_dir: str, namelist: str | None = None,
                     namelist_wps: str | None = None, geo_from_wrfinput: str | None = None,
                     date_shift_to: str | None = None, forcing: str | None = None,
                     verify_steps: int = 0, nodes: int = 1, ranks: int | None = None, omp: int = 1,
                     nx: int | None = None, ny: int | None = None, ncdump: str | None = None,
                     now: datetime.datetime | None = None, geog_path: str | None = None,
                     grib_dir: str | None = None, ml_model: str | None = None) -> dict:
    ranks = ranks or wtune.machine.cores_per_node()
    rows, blockers, nl = cs.inspect_case(case_dir, namelist=namelist, namelist_wps=namelist_wps,
                                         geo_from_wrfinput=geo_from_wrfinput, grib_dir=grib_dir,
                                         ncdump=ncdump, now=now, geog_path=geog_path)
    if blockers:
        raise RebuildError("rebuild plan: case is blocked:\n  - " + "\n  - ".join(blockers))

    max_dom = predict.get_max_dom(nl)
    stated_start, stated_end, analog_start, span_start, span_end, run_s = rebuild_span(
        nl, date_shift_to=date_shift_to)

    chain = cs.resolve_forcing_chain_for(span_start, grib_dir=grib_dir, forcing_pin=forcing,
                                         now=now)
    if not chain:
        raise RebuildError("rebuild plan: no forcing source covers %s (and no --grib-dir given)"
                           % span_start.strftime(_WRF_TIME_FMT))
    primary = chain[0]
    if primary.forcing_interval_h is None:
        raise RebuildError(cs._LOCAL_FORCING_UNSUPPORTED % ("rebuild plan", primary.dataset))
    interval_s = primary.forcing_interval_h * 3600
    span_start_grid = cs.align_to_interval(span_start, primary.forcing_interval_h)
    span_end_grid = cs.next_grib_time_ge(span_start_grid, interval_s, span_end)

    nlw_path = namelist_wps or os.path.join(case_dir, "namelist.wps")
    wi_path = geo_from_wrfinput or os.path.join(case_dir, "wrfinput_d01")
    geo = cs.resolve_projection(nl, nlw_path if os.path.exists(nlw_path) else None,
                                wi_path if os.path.exists(wi_path) else None, ncdump=ncdump)
    wps_fields = cs.build_namelist_wps_fields(nl, geo, span_start_grid, span_end_grid, interval_s,
                                              prefix=primary.prefix)

    dom = cs.domain_list_fields(nl, max_dom)
    dx_km = geo["dx"] / 1000.0

    # ZNW/P_TOP are read here (login node has netCDF4); a failure is recorded in plan.json, never
    # skipped silently in the batch job (T12, job 965).
    vertical_grid, vertical_grid_warning = cs.read_vertical_grid_or_warn(
        wi_path if os.path.exists(wi_path) else None)
    if vertical_grid_warning:
        print("WARNING: %s" % vertical_grid_warning)

    if not (nx and ny):
        nx, ny = near_square(ranks // max(omp, 1))
    config = {"ranks": ranks, "omp": omp, "nx": nx, "ny": ny, "numtiles": 1}
    plan = {
        "case_dir": os.path.abspath(case_dir),
        "namelist_input": os.path.abspath(namelist or os.path.join(case_dir, "namelist.input")),
        "max_dom": max_dom, "dx_km": dx_km,
        "parent_grid_ratio": dom["parent_grid_ratio"],
        "stated_start": stated_start.strftime(_WRF_TIME_FMT),
        "stated_end": stated_end.strftime(_WRF_TIME_FMT),
        "analog_start": analog_start.strftime(_WRF_TIME_FMT) if analog_start else None,
        "shift_s": (analog_start - stated_start).total_seconds() if analog_start else 0.0,
        "span_start": span_start_grid.strftime(_WRF_TIME_FMT),
        "span_end": span_end_grid.strftime(_WRF_TIME_FMT),
        "interval_seconds": interval_s,
        "forcing_chain_primary": primary.to_dict(), "forcing_pin": forcing,
        "verify_steps": int(verify_steps or 0),
        "nodes": nodes, "config": config,
        "geo": {k: v for k, v in geo.items()}, "namelist_wps_fields": wps_fields,
        # provenance only; fix-span-real-namelist applies vertical_grid (T8/T12)
        "wrfinput_path": os.path.abspath(wi_path) if os.path.exists(wi_path) else None,
        "vertical_grid": vertical_grid, "vertical_grid_warning": vertical_grid_warning,
        # K5: ML state completion, off unless --ml-model
        "ml_model": os.path.abspath(ml_model) if ml_model else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    with open(os.path.join(out_dir, "namelist.wps"), "w") as f:
        f.write(cs.render_namelist_wps(wps_fields))
    print("span: %s .. %s (%s, %dh cadence)" % (plan["span_start"], plan["span_end"],
                                                 primary.dataset, primary.forcing_interval_h))
    return plan


def cmd_rebuild_fetch(out_dir: str, cache_dir: str | None = None, grib_dir: str | None = None,
                      keep_archives: bool = False, probe_fn: Callable | None = None,
                      download_fn: Callable | None = None) -> tuple[int, dict]:
    with open(os.path.join(out_dir, "plan.json")) as f:
        plan = json.load(f)
    cache_dir = cache_dir or os.path.join(out_dir, "grib")
    os.makedirs(cache_dir, exist_ok=True)
    T = datetime.datetime.strptime(plan["span_start"], _WRF_TIME_FMT)
    T_end = datetime.datetime.strptime(plan["span_end"], _WRF_TIME_FMT)
    chain = cs.resolve_forcing_chain_for(T, grib_dir=grib_dir, forcing_pin=plan.get("forcing_pin"))
    chosen, tried = cs.choose_forcing_for_window(T, T_end, chain, probe_fn=probe_fn)
    if chosen is None:
        # the probe record is written even on failure (T4b)
        forcing = {"status": "failed", "chosen": None, "candidates_tried": tried}
        with open(os.path.join(out_dir, "forcing.json"), "w") as f:
            json.dump(forcing, f, indent=2)
        first_missing = None
        for t in tried:
            if t.get("first_missing"):
                first_missing = t["first_missing"]
                break
        detail = ("%s (final status %s)" % (first_missing["url"], first_missing["final_status"])
                  if first_missing and first_missing.get("url") else "none")
        raise RebuildError(
            "rebuild fetch: no forcing source has every needed file (tried: %s; first "
            "missing: %s)"
            % (", ".join(t["dataset"] for t in tried) or "none", detail))
    paths = cs._download_forcing_for_window(chosen, T, T_end, cache_dir,
                                            keep_archives=keep_archives, download_fn=download_fn)
    total_bytes = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    file_names = sorted(os.path.basename(p) for p in paths)
    forcing = {"chosen": chosen.to_dict(), "candidates_tried": tried, "files": file_names}
    with open(os.path.join(out_dir, "forcing.json"), "w") as f:
        json.dump(forcing, f, indent=2)
    # exact basenames for link_grib.csh: extensions differ per source (.grib/.grb/.grib2; T9)
    with open(os.path.join(out_dir, "grib_manifest.txt"), "w") as f:
        for name in file_names:
            f.write(name + "\n")
    print("rebuild fetch: %d bytes (%s, %d files)" % (total_bytes, chosen.dataset, len(paths)))
    return total_bytes, forcing


def near_square(n: int) -> tuple[int, int]:
    """(nx, ny) with nx * ny == n and nx <= ny as close as possible: real.exe's decomposition.
    1 x n put all ranks along y, more than a 126-row domain has (haitang2017, 144 ranks)."""
    nx = max(d for d in range(1, int(n ** 0.5) + 1) if n % d == 0)
    return nx, n // nx


def fit_plan_to_machine(plan: dict) -> dict:
    """plan with its real.exe/wrf.exe config shrunk to the machine's cores_per_node when the plan
    was written for more (cores_per_node lowered after `prep wps`: haitang2017 144 -> 72 asserted
    in _rebuild_job). Mutates and returns plan; a note is printed when it changes."""
    cfg = plan.get("config") or {}
    nodes = max(int(plan.get("nodes") or 1), 1)
    omp = max(int(cfg.get("omp") or 1), 1)
    cores = wtune.machine.cores_per_node()
    if cfg.get("ranks") and -(-cfg["ranks"] * omp // nodes) > cores:
        ranks = cores // omp * nodes
        nx, ny = near_square(ranks)
        print("note: plan.json was made for %d ranks; this machine has %d cores/node -> %d ranks "
              "(%dx%d)" % (cfg["ranks"], cores, ranks, nx, ny))
        cfg.update(ranks=ranks, nx=nx, ny=ny)
    return plan


def _rebuild_forcing_choice(out_dir):
    """The chosen forcing (prefix/vtable/interval) for this rebuild's span -- forcing.json if
    `fetch` already ran, else the plan-time default (mirrors
    coldstart_sample._load_window_forcing / _default_per_window_forcing)."""
    with open(os.path.join(out_dir, "plan.json")) as f:
        plan = fit_plan_to_machine(json.load(f))
    forcing_path = os.path.join(out_dir, "forcing.json")
    if os.path.exists(forcing_path):
        with open(forcing_path) as f:
            forcing = json.load(f)
        chosen = forcing.get("chosen")
        if chosen:
            return chosen, plan
    primary = plan["forcing_chain_primary"]
    return primary, plan


def cmd_rebuild_stage(out_dir: str, site: str | None = None) -> str:
    chosen, plan = _rebuild_forcing_choice(out_dir)
    wps_dir_env = os.environ.get("WPS_DIR")
    wrf_tree = os.environ.get("WRF_TREE")
    wps_out = os.path.join(out_dir, "wps")
    real_out = os.path.join(out_dir, "real")
    os.makedirs(wps_out, exist_ok=True)
    if wps_dir_env:
        for exe in ("geogrid.exe", "ungrib.exe", "metgrid.exe", "link_grib.csh"):
            src = os.path.join(wps_dir_env, exe)
            if os.path.exists(src):
                cs._symlink_one(src, os.path.join(wps_out, exe))
        for subdir in ("geogrid", "metgrid"):
            src_dir = os.path.join(wps_dir_env, subdir)
            if os.path.isdir(src_dir):
                cs._symlink_dir(src_dir, os.path.join(wps_out, subdir))
        for vt in ("Vtable.GFS", "Vtable.CFSR"):
            vt_src = os.path.join(wps_dir_env, "ungrib", "Variable_Tables", vt)
            if os.path.exists(vt_src):
                cs._symlink_one(vt_src, os.path.join(wps_out, vt))
        # Vtable.GFS_PRMSL (NCEI g3/g4, FNL) is generated from Vtable.GFS, not linked (T13)
        gfs_vt_src = os.path.join(wps_dir_env, "ungrib", "Variable_Tables", "Vtable.GFS")
        if os.path.exists(gfs_vt_src):
            prmsl_path = os.path.join(wps_out, "Vtable.GFS_PRMSL")
            try:
                with open(gfs_vt_src) as f:
                    prmsl_text = cs.generate_vtable_gfs_prmsl(f.read())
                if os.path.lexists(prmsl_path):
                    os.remove(prmsl_path)
                with open(prmsl_path, "w") as f:
                    f.write(prmsl_text)
            except cs.ColdstartError as exc:
                print("WARNING: Vtable.GFS_PRMSL generation skipped -- %s" % exc)
        default_vt = os.path.join(wps_out, "Vtable.GFS")
        if os.path.exists(default_vt):
            cs._symlink_one(default_vt, os.path.join(wps_out, "Vtable"))
    else:
        print("WARNING: $WPS_DIR not set -- wps/ exe/table symlinks skipped")
    nlw_src = os.path.join(out_dir, "namelist.wps")
    nlw_dst = os.path.join(wps_out, "namelist.wps")
    if os.path.exists(nlw_src) and os.path.abspath(nlw_src) != os.path.abspath(nlw_dst):
        shutil.copyfile(nlw_src, nlw_dst)

    if wrf_tree:
        cs._symlink_tree(os.path.join(wrf_tree, "run"), real_out)
        real_exe = os.path.join(wrf_tree, "main", "real.exe")
        if os.path.exists(real_exe):
            cs._symlink_one(real_exe, os.path.join(real_out, "real.exe"))
    else:
        os.makedirs(real_out, exist_ok=True)
        print("WARNING: $WRF_TREE not set -- real/ table symlinks skipped")

    verify_steps = int(plan.get("verify_steps") or 0)
    wrf_out = None
    if verify_steps > 0:
        wrf_out = os.path.join(out_dir, "wrf")
        if wrf_tree:
            cs._symlink_tree(os.path.join(wrf_tree, "run"), wrf_out)
            wrf_exe = os.path.join(wrf_tree, "main", "wrf.exe")
            if os.path.exists(wrf_exe):
                cs._symlink_one(wrf_exe, os.path.join(wrf_out, "wrf.exe"))
        else:
            os.makedirs(wrf_out, exist_ok=True)

    plan["_out_dir"] = os.path.abspath(out_dir)
    sbatch_text = render_rebuild_sbatch(plan, chosen, site=site)
    sbatch_path = os.path.join(out_dir, "run.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch_text)
    os.chmod(sbatch_path, 0o755)
    print("staged %s" % out_dir)
    return sbatch_path


def cmd_rebuild_finalize(out_dir: str, keep_wrfinput: bool = False) -> dict:
    """Collect OUT/case/ (copies) and write rebuild_report.json after the job.
    keep_wrfinput (K3, a case with wrfinput but no wrfbdy): keep real.exe's boundaries but the case's
    ORIGINAL wrfinput_d0N; compare() checks grid identity (RebuildError on mismatch) and its d01
    report goes into the report. RebuildError when the case lacks a wrfinput to keep."""
    with open(os.path.join(out_dir, "plan.json")) as f:
        plan = json.load(f)
    case_out = os.path.join(out_dir, "case")
    os.makedirs(case_out, exist_ok=True)

    def copy_if_exists(src, dst_name):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(case_out, dst_name))
            return True
        return False

    copy_if_exists(os.path.join(out_dir, "wps", "namelist.wps"), "namelist.wps")
    real_namelist = os.path.join(out_dir, "real", "namelist.input")
    copy_if_exists(real_namelist, "namelist.input")
    max_dom = plan["max_dom"]
    kept_wrfinput_compare = None
    kept_wrfinput_domains = []
    for d in range(1, max_dom + 1):
        dom_name = "wrfinput_d%02d" % d
        real_wrfinput = os.path.join(out_dir, "real", dom_name)
        original_wrfinput = os.path.join(plan["case_dir"], dom_name)
        if keep_wrfinput and os.path.exists(original_wrfinput) and os.path.exists(real_wrfinput):
            cmp_report = compare(original_wrfinput, real_wrfinput)   # raises on grid mismatch
            if d == 1:
                kept_wrfinput_compare = cmp_report
            kept_wrfinput_domains.append(dom_name)
            shutil.copyfile(original_wrfinput, os.path.join(case_out, dom_name))
        elif keep_wrfinput and not os.path.exists(original_wrfinput):
            raise RebuildError(
                "finalize --keep-wrfinput: %s has no %s to keep" % (plan["case_dir"], dom_name))
        else:
            copy_if_exists(real_wrfinput, dom_name)
        copy_if_exists(os.path.join(out_dir, "real", "wrflowinp_d%02d" % d),
                       "wrflowinp_d%02d" % d)
    copy_if_exists(os.path.join(out_dir, "real", "wrfbdy_d01"), "wrfbdy_d01")

    changes_path = os.path.join(out_dir, "namelist_changes.json")
    changes = json.load(open(changes_path)) if os.path.exists(changes_path) else []
    warnings_path = os.path.join(out_dir, "registry_warnings.json")
    reg_warnings = json.load(open(warnings_path)) if os.path.exists(warnings_path) else []

    rows, blockers, _nl = cs.inspect_case(plan["case_dir"], namelist=plan["namelist_input"])
    capability_notes = [note for (item, _ok, _how, note) in rows if item == "capability"]

    report = {
        "changes": changes, "registry_warnings": reg_warnings,
        "forcing_source": _rebuild_forcing_choice(out_dir)[0],
        "span_start": plan["span_start"], "span_end": plan["span_end"],
        "stated_start": plan["stated_start"], "analog_start": plan["analog_start"],
        "shift_s": plan["shift_s"], "capability_warnings": capability_notes,
        "blockers": blockers,
    }
    if keep_wrfinput:
        report["keep_wrfinput"] = {
            "domains": kept_wrfinput_domains,
            "note": ("boundaries (wrfbdy_d01%s) come from the fetched analysis; the interior "
                    "state (%s) comes from the case's own original wrfinput -- NOT real.exe's "
                    "own cold-start output"
                    % (" + wrflowinp" if any(
                        os.path.exists(os.path.join(case_out, "wrflowinp_d%02d" % d))
                        for d in range(1, max_dom + 1)) else "",
                       ", ".join(kept_wrfinput_domains))),
            "d01_vs_analysis_t0_compare": kept_wrfinput_compare,
        }
    with open(os.path.join(out_dir, "rebuild_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("finalized %s -> %s" % (out_dir, case_out))
    return report


def cmd_rebuild_fix_span_wps_namelist(out_dir: str) -> None:
    chosen, plan = _rebuild_forcing_choice(out_dir)
    T = datetime.datetime.strptime(plan["span_start"], _WRF_TIME_FMT)
    T_end = datetime.datetime.strptime(plan["span_end"], _WRF_TIME_FMT)
    fields = cs.wps_fields_for_window(plan["namelist_wps_fields"], T, T_end,
                                      prefix=chosen.get("prefix"),
                                      interval_seconds=chosen.get("forcing_interval_h", 0) * 3600
                                      or None)
    text = cs.render_namelist_wps(fields)
    out_path = os.path.join(out_dir, "wps", "namelist.wps")
    if os.path.lexists(out_path):
        os.remove(out_path)
    with open(out_path, "w") as f:
        f.write(text)
    print("wrote %s (span %s .. %s)" % (out_path, plan["span_start"], plan["span_end"]))


def cmd_rebuild_fix_span_real_namelist(out_dir: str, ncdump: str | None = None,
                                       serial: bool = False) -> None:
    chosen, plan = _rebuild_forcing_choice(out_dir)
    real_dir = os.path.join(out_dir, "real")
    met_ems = sorted(p for p in os.listdir(real_dir) if p.startswith("met_em.d01."))
    if not met_ems:
        raise RebuildError("fix-span-real-namelist: no met_em.d01.* in %s" % real_dir)
    ncdump_exe = cs.find_ncdump(ncdump)
    if not ncdump_exe:
        raise RebuildError("fix-span-real-namelist: no ncdump available")
    nlev, soil, land = cs._read_met_em_dims(os.path.join(real_dir, met_ems[0]), ncdump_exe)
    T = datetime.datetime.strptime(plan["span_start"], _WRF_TIME_FMT)
    T_end = datetime.datetime.strptime(plan["span_end"], _WRF_TIME_FMT)
    case_text = open(plan["namelist_input"]).read()

    facts = {
        "max_dom": plan["max_dom"], "num_metgrid_levels": nlev,
        "num_metgrid_soil_levels": soil, "num_land_cat": land,
        "interval_seconds": chosen.get("forcing_interval_h", plan["interval_seconds"] // 3600) * 3600,
        "start_dt": T, "end_dt": T_end, "dx_km": plan["dx_km"],
        "parent_grid_ratio": plan["parent_grid_ratio"],
    }
    # Runs in the batch job under /usr/bin/python3 (no numpy): the vertical grid comes from
    # plan.json, read at plan time (T12, job 965).
    vertical_grid = plan.get("vertical_grid")
    if vertical_grid is None and plan.get("vertical_grid_warning"):
        print("WARNING: %s (recorded at plan time)" % plan["vertical_grid_warning"])
    txt, changes = complete_namelist(case_text, facts, vertical_grid=vertical_grid)
    if serial:
        txt = cs.set_list(txt, "nproc_x", 1, 1) if cs.count_key(txt, "nproc_x") else txt
        txt = cs.set_list(txt, "nproc_y", 1, 1) if cs.count_key(txt, "nproc_y") else txt
    else:
        nx, ny = plan["config"]["nx"], plan["config"]["ny"]
        txt = cs.set_list(txt, "nproc_x", nx, 1) if cs.count_key(txt, "nproc_x") else txt
        txt = cs.set_list(txt, "nproc_y", ny, 1) if cs.count_key(txt, "nproc_y") else txt

    out_path = os.path.join(real_dir, "namelist.input")
    if os.path.lexists(out_path):
        os.remove(out_path)
    with open(out_path, "w") as f:
        f.write(txt)
    with open(os.path.join(out_dir, "namelist_changes.json"), "w") as f:
        json.dump(changes, f, indent=2)
    try:
        registry = load_registry()
        nl_parsed = predict.parse_namelist(out_path)
        reg_warnings = validate_namelist_against_registry(nl_parsed, registry)
        with open(os.path.join(out_dir, "registry_warnings.json"), "w") as f:
            json.dump(reg_warnings, f, indent=2)
    except Exception as exc:
        print("WARNING: registry validation skipped (%s)" % exc)
    print("wrote %s (%d changes%s)" % (out_path, len(changes), " serial" if serial else ""))


def cmd_rebuild_fix_span_wrf_namelist(out_dir: str, nproc_x: int, nproc_y: int,
                                      numtiles: int = 1) -> None:
    real_nl_path = os.path.join(out_dir, "real", "namelist.input")
    if not os.path.exists(real_nl_path):
        raise RebuildError("fix-span-wrf-namelist: %s does not exist yet" % real_nl_path)
    with open(os.path.join(out_dir, "plan.json")) as f:
        plan = json.load(f)
    real_text = open(real_nl_path).read()
    T = datetime.datetime.strptime(plan["span_start"], _WRF_TIME_FMT)
    dt = predict.get_timestep_seconds(predict.parse_namelist(plan["namelist_input"]))
    win_steps = int(plan.get("verify_steps") or 0)
    txt = cs.build_wrf_window_namelist_text(real_text, T, win_steps, dt, nproc_x, nproc_y,
                                            numtiles)
    out_path = os.path.join(out_dir, "wrf", "namelist.input")
    if os.path.lexists(out_path):
        os.remove(out_path)
    with open(out_path, "w") as f:
        f.write(txt)
    print("wrote %s (verify_steps=%d)" % (out_path, win_steps))


_SMOKE_EXTRA_MIN = 30              # added to the job time limit when the wrf.exe smoke run is on


@dataclasses.dataclass(frozen=True)
class _RebuildJob:
    """Everything render_rebuild_sbatch's sections need, derived once from plan + chosen forcing."""
    out_dir_abs: str | None
    nodes: int
    ranks: int
    omp: int
    nx: int
    ny: int
    ppn: int
    verify_steps: int
    prefix: str
    vtable: str
    expected_ungrib: list
    expected_met: list
    geo_em_expected: list
    time_min: int
    ml_model: str | None
    site: str
    estimate_min: int


def _rebuild_job(plan: dict, chosen: dict, site: str, estimate_min: int) -> _RebuildJob:
    out_dir_abs = plan.get("_out_dir")
    nodes = plan["nodes"]
    ranks = plan["config"]["ranks"]
    omp = plan["config"]["omp"]
    nx, ny = plan["config"]["nx"], plan["config"]["ny"]
    ppn = -(-ranks // max(nodes, 1))   # ceil, not floor: floor gave rc=213 'not enough slots' (T7)
    cores = wtune.machine.cores_per_node()
    assert ppn <= cores, ("ppn=%d exceeds CORES_PER_NODE=%d (ranks=%d nodes=%d)"
                          % (ppn, cores, ranks, nodes))
    max_dom = plan["max_dom"]
    verify_steps = int(plan.get("verify_steps") or 0)
    T = datetime.datetime.strptime(plan["span_start"], _WRF_TIME_FMT)
    T_end = datetime.datetime.strptime(plan["span_end"], _WRF_TIME_FMT)
    prefix = chosen.get("prefix", "GFS")
    interval_s = chosen.get("forcing_interval_h", plan["interval_seconds"] // 3600) * 3600
    return _RebuildJob(
        out_dir_abs=out_dir_abs, nodes=nodes, ranks=ranks, omp=omp, nx=nx, ny=ny, ppn=ppn,
        verify_steps=verify_steps,
        prefix=prefix, vtable=chosen.get("vtable", "Vtable.GFS"),
        expected_ungrib=cs.expected_wps_intermediate_files(prefix, T, T_end, interval_s),
        expected_met=cs.expected_met_em_files(max_dom, T, T_end, interval_s),
        geo_em_expected=["geo_em.d%02d.nc" % d for d in range(1, max_dom + 1)],
        time_min=int(wtune.machine.cap_seconds(60.0 * max(
            60, estimate_min * 2 + (_SMOKE_EXTRA_MIN if verify_steps else 0))) // 60),
        ml_model=plan.get("ml_model"), site=site or wtune.machine.site(),
        estimate_min=estimate_min)


def _sbatch_header(job: _RebuildJob) -> list[str]:
    lines = ["#!/usr/bin/env bash",
             "#SBATCH --job-name=case-rebuild",
             "#SBATCH --partition=%s" % wtune.machine.partition()]
    if cs.sbatch_excluded_nodes():
        lines.append("#SBATCH --exclude=%s" % ",".join(cs.sbatch_excluded_nodes()))
    if wtune.machine.account_arg():
        lines.append("#SBATCH " + wtune.machine.account_arg())
    lines += ["#SBATCH -N %d --exclusive --ntasks-per-node=%d" % (job.nodes, job.ppn),
             "#SBATCH --time=%02d:%02d:00" % (job.time_min // 60, job.time_min % 60),
             "#SBATCH --signal=B:USR1@90"]
    if job.out_dir_abs:
        lines.append("#SBATCH --output=%s/run-%%j.out" % job.out_dir_abs)
    lines += ["set -uo pipefail", "", "TOOL=%s" % cs._sh_quote(os.path.abspath(__file__))]
    if job.ml_model:   # K5: ML state completion, default off; never /usr/bin/python3 for it
        lines += ["TOOL_SCM=%s" % cs._sh_quote(cs._SCM_PATH),
                  "ML_MODEL=%s" % cs._sh_quote(os.path.abspath(job.ml_model))]
    lines += ["SITE=%s" % job.site] + cs.tree_default_lines(bool(job.ml_model))
    lines += ["source \"$CAMPAIGN/sites/$SITE.sh\"   # site_env / site_launch / site_hygiene -- "
              "reused, not reimplemented",
              "source %s" % wtune.machine.env_source(),
              "export PATH=$NETCDF/bin:$PATH",
              "ulimit -s unlimited",
              "site_env"]
    # REBUILD_DIR, not OUT, and set AFTER sourcing: campaign/sites/scc.sh assigns its own OUT
    # (T6, jobs 957/958).
    if job.out_dir_abs:
        lines.append("REBUILD_DIR=%s" % cs._sh_quote(job.out_dir_abs))
    else:
        lines.append("REBUILD_DIR=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)")
    lines.extend(cs.render_signal_trap_and_hygiene_preamble(
        job.nodes,
        extra_lines_after_alloc_nodes=["STATUS=\"$REBUILD_DIR/rebuild_status.txt\"",
                                       ": > \"$STATUS\""],
        record_label="STAGE"))
    return lines


def _sbatch_geogrid(job: _RebuildJob) -> list[str]:
    return [
        "cd \"$REBUILD_DIR/wps\"",
        "export OMP_NUM_THREADS=1",
        "CUR_WINDOW=geogrid",
        "echo \"=== geogrid.exe ($(date -u +%FT%TZ)) ===\"",
        cs._wps_launch("geogrid.exe", job.ppn),
        "rc=$?; pkill -u \"$USER\" -x geogrid.exe 2>/dev/null",
        "GEO_EM_MISSING=()",
        "for f in %s; do [ -f \"$f\" ] || GEO_EM_MISSING+=(\"$f\"); done"
        % " ".join(job.geo_em_expected),
        "if [ $rc -ne 0 ] || [ ${#GEO_EM_MISSING[@]} -gt 0 ] "
        "|| ! grep -q 'Successful completion' geogrid.log* 2>/dev/null; then",
        "  echo \"FATAL geogrid rc=$rc missing=${GEO_EM_MISSING[*]:-none}\"; exit 1",
        "fi",
        "",
    ]


def _sbatch_ungrib(job: _RebuildJob) -> list[str]:
    """link_grib.csh over exactly what `rebuild fetch` wrote (grib_manifest.txt; T9 -- FNL .grib
    and NCEI .grb are not *.grib2), then ungrib with an expected-file check."""
    vt = cs._sh_quote(job.vtable)
    return [
        "CUR_WINDOW=link_grib",
        "if [ -s \"$REBUILD_DIR/grib_manifest.txt\" ]; then",
        "  mapfile -t GRIB_FILES < <(sed \"s#^#$REBUILD_DIR/grib/#\" "
        "\"$REBUILD_DIR/grib_manifest.txt\")",
        "else",
        "  shopt -s nullglob",
        "  GRIB_FILES=(\"$REBUILD_DIR\"/grib/*.grib2 \"$REBUILD_DIR\"/grib/*.grib "
        "\"$REBUILD_DIR\"/grib/*.grb)",
        "  shopt -u nullglob",
        "fi",
        "if [ ${#GRIB_FILES[@]} -eq 0 ]; then",
        "  echo \"FATAL: no forcing files in $REBUILD_DIR/grib -- run 'case_rebuild.py "
        "rebuild fetch --out $REBUILD_DIR' first\"; exit 1",
        "fi",
        "rm -f GRIBFILE.*",
        "./link_grib.csh \"${GRIB_FILES[@]}\"",
        "[ $? -ne 0 ] && { echo \"FATAL link_grib.csh failed\"; exit 1; }",
        "python3 \"$TOOL\" rebuild fix-span-wps-namelist --out \"$REBUILD_DIR\" || exit 1",
        "if [ -e %s ]; then ln -sf %s Vtable; else echo \"FATAL: %s not staged\"; exit 1; fi"
        % (vt, vt, job.vtable),
        "rm -f \"%s:\"*" % job.prefix,
        "CUR_WINDOW=ungrib",
        "./ungrib.exe",
        "rc=$?; pkill -u \"$USER\" -x ungrib.exe 2>/dev/null",
        "UNGRIB_MISSING=()",
        "for f in %s; do [ -f \"$f\" ] || UNGRIB_MISSING+=(\"$f\"); done"
        % " ".join(cs._sh_quote(x) for x in job.expected_ungrib),
        "[ $rc -ne 0 ] || [ ${#UNGRIB_MISSING[@]} -gt 0 ] && "
        "{ echo \"FATAL ungrib rc=$rc missing=${UNGRIB_MISSING[*]:-none}\"; exit 1; }",
        "",
    ]


def _sbatch_metgrid(job: _RebuildJob) -> list[str]:
    return [
        "CUR_WINDOW=metgrid",
        "echo \"=== metgrid.exe ($(date -u +%FT%TZ)) ===\"",
        cs._wps_launch("metgrid.exe", job.ppn),
        "rc=$?; pkill -u \"$USER\" -x metgrid.exe 2>/dev/null",
        "MET_MISSING=()",
        "for f in %s; do [ -f \"$f\" ] || MET_MISSING+=(\"$f\"); done"
        % " ".join(cs._sh_quote(x) for x in job.expected_met),
        "if [ $rc -ne 0 ] || [ ${#MET_MISSING[@]} -gt 0 ] "
        "|| ! grep -q 'Successful completion' metgrid.log* 2>/dev/null; then",
        "  echo \"FATAL metgrid rc=$rc missing=${MET_MISSING[*]:-none}\"; exit 1",
        "fi",
        "",
    ]


def _sbatch_real(job: _RebuildJob) -> list[str]:
    """real.exe over the span: decomposed with a timeout, serial retry on timeout/no output.
    nocolons=.true. cases want '_' met_em names (V6 lafe)."""
    return [
        "CUR_WINDOW=real",
        "echo \"=== real.exe ($(date -u +%FT%TZ)) ===\"",
        "cd \"$REBUILD_DIR/real\"",
        "rm -f met_em.d0*.*.nc; ln -sf \"$REBUILD_DIR\"/wps/met_em.d0*.*.nc .",
        "for f in met_em.d0*.*.nc; do [ \"$f\" = \"${f//:/_}\" ] || ln -sf \"$f\" \"${f//:/_}\"; done",
        "python3 \"$TOOL\" rebuild fix-span-real-namelist --out \"$REBUILD_DIR\" || exit 1",
        "timeout %d mpirun -np %d --map-by ppr:%d:node:PE=1 --bind-to core ./real.exe"
        % (cs.REAL_TIMEOUT_S, job.ranks, job.ppn),
        "rc=$?; pkill -u \"$USER\" -x real.exe 2>/dev/null",
        "if [ $rc -eq 124 ] || [ ! -f wrfinput_d01 ] || [ ! -f wrfbdy_d01 ]; then",
        "  echo \"real.exe decomposed attempt rc=$rc -- retrying serial\" >&2",
        "  python3 \"$TOOL\" rebuild fix-span-real-namelist --out \"$REBUILD_DIR\" "
        "--serial || exit 1",
        "  timeout %d mpirun -np 1 --map-by ppr:1:node:PE=1 --bind-to core ./real.exe"
        % cs.REAL_SERIAL_TIMEOUT_S,
        "  rc=$?; pkill -u \"$USER\" -x real.exe 2>/dev/null",
        "fi",
        "[ $rc -ne 0 ] && { echo \"FATAL real.exe rc=$rc\"; exit 1; }",
        "[ -f wrfinput_d01 ] && [ -f wrfbdy_d01 ] || { echo \"FATAL: no wrfinput/wrfbdy\"; exit 1; }",
    ]


def _sbatch_ml_apply() -> list[str]:
    """K5: apply the ML state correction to real.exe's wrfinput_d01; outcome to ml_status.txt."""
    return [
        "ML_OUT=$(\"$GRIB_PY\" \"$TOOL_SCM\" apply --model \"$ML_MODEL\" "
        "--targets-from-report --wrfinput wrfinput_d01 --out wrfinput_d01.ml 2>&1)",
        "ml_rc=$?",
        "ML_TARGETS=$(printf '%s\\n' \"$ML_OUT\" | "
        "grep '^APPLIED_TARGETS=' | tail -1 | sed 's/^APPLIED_TARGETS=//')",
        "if [ $ml_rc -eq 0 ] && [ -f wrfinput_d01.ml ]; then",
        "  mv wrfinput_d01 wrfinput_d01.noml && mv wrfinput_d01.ml wrfinput_d01",
        "  echo \"ml=applied ml_targets=$ML_TARGETS\" > \"$REBUILD_DIR/ml_status.txt\"",
        "else",
        "  rm -f wrfinput_d01.ml",
        "  echo \"ml=skipped ml_targets=$ML_TARGETS\" > \"$REBUILD_DIR/ml_status.txt\"",
        "  printf '%s\\n' \"$ML_OUT\" >&2",
        "fi",
    ]


def _sbatch_wrf_smoke(job: _RebuildJob) -> list[str]:
    """wrf.exe for verify_steps through site_launch (no wrapper around wrf.exe)."""
    return [
        "",
        "echo \"=== wrf.exe smoke run: %d steps ($(date -u +%%FT%%TZ)) ===\"" % job.verify_steps,
        "CUR_WINDOW=wrf",
        "cd \"$REBUILD_DIR/wrf\"",
        "ln -sf ../real/wrfinput_d0* .; ln -sf ../real/wrfbdy_d01 .",
        "shopt -s nullglob; WRFLOWINP=(../real/wrflowinp_d0*); shopt -u nullglob",
        "[ ${#WRFLOWINP[@]} -gt 0 ] && ln -sf ../real/wrflowinp_d0* .",
        "python3 \"$TOOL\" rebuild fix-span-wrf-namelist --out \"$REBUILD_DIR\" "
        "--nproc-x %d --nproc-y %d --numtiles 1 || exit 1" % (job.nx, job.ny),
        "export OMP_NUM_THREADS=%d" % job.omp,
        "unset OMP_PLACES OMP_PROC_BIND",
        "site_launch %d %d %d block" % (job.ranks, job.ppn, job.omp),
        "timeout %d \"${SITE_LAUNCH_CMD[@]}\" > wrf.out 2>&1 &" % (job.estimate_min * 60),
        "WRF_PID=$!; wait \"$WRF_PID\"; rc=$?",
        "pkill -u \"$USER\" -x wrf.exe 2>/dev/null",
        "CUR_WINDOW=\"\"",
        "[ $rc -ne 0 ] && { echo \"FATAL wrf.exe smoke run rc=$rc\"; exit 1; }",
    ]


def render_rebuild_sbatch(plan: dict, chosen: dict, site: str | None = None,
                          estimate_min: int = 30) -> str:
    """One sbatch for the span: geogrid -> ungrib+metgrid (expected-file checks) -> real.exe
    (timeout + serial fallback) -> optional ML apply -> optional wrf.exe smoke run of
    verify_steps. Conventions reused from coldstart_sample.render_run_sbatch; no LD_PRELOAD.
    AssertionError when ranks do not fit 72 cores per node."""
    job = _rebuild_job(plan, chosen, site, estimate_min)
    lines = _sbatch_header(job)
    lines += _sbatch_geogrid(job)
    lines += _sbatch_ungrib(job)
    lines += _sbatch_metgrid(job)
    lines += _sbatch_real(job)
    if job.ml_model:
        lines += _sbatch_ml_apply()
    lines.append("CUR_WINDOW=\"\"")
    if job.verify_steps > 0:
        lines += _sbatch_wrf_smoke(job)
    lines += ["", "run_hygiene reap+census", "echo \"=== DONE ($(date -u +%FT%TZ)) ===\""]
    text = "\n".join(lines) + "\n"
    assert "LD_PRELOAD" not in text
    assert "--wrap" not in text
    return text


# ===========================================================================
# R6-B: compare -- similarity report (numpy + netCDF4, lazy-imported; grib pixi env only).
# Metrics are pure functions over arrays; read_wrfinput_fields_netcdf4 is the only file reader.
# PW always uses P+PB level differences (top half-layer dropped); the MU/C1H..C4H hybrid path is
# not implemented (unverified without a real wrfout).
# ===========================================================================

G_ACCEL = 9.81                    # m s-2
RD_CP = 287.0 / 1004.5   # WRF's own R_d/c_p (dry gas constant / specific heat at const. pressure)
P0_PA = 100000.0         # WRF's reference pressure for the Poisson (theta -> T) relation


def rmse(a: ArrayLike, b: ArrayLike, mask: ArrayLike | None = None) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = (a - b)
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.nanmean(d ** 2)))


def max_abs_diff(a: ArrayLike, b: ArrayLike, mask: ArrayLike | None = None) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.abs(a - b)
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    return float(np.nanmax(d))


def agreement_pct(a: ArrayLike, b: ArrayLike, mask: ArrayLike | None = None) -> float:
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    eq = (a == b)
    if mask is not None:
        eq = eq[np.asarray(mask, dtype=bool)]
    return float(100.0 * np.count_nonzero(eq) / eq.size) if eq.size else float("nan")


def bias(a: ArrayLike, b: ArrayLike, mask: ArrayLike | None = None) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    return float(np.nanmean(d))


def pattern_correlation(a: ArrayLike, b: ArrayLike, mask: ArrayLike | None = None) -> float:
    import numpy as np
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).ravel()
    a, b = a[valid], b[valid]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_pw(qvapor: ArrayLike, p: ArrayLike, pb: ArrayLike) -> np.ndarray:
    """qvapor, p, pb: 3D arrays [bottom_top, south_north, west_east] (mass/half levels, index 0
    = lowest model level). Returns 2D PW [south_north, west_east] in kg/m^2 -- see module
    section docstring for the dp convention (P+PB level differences; MU/C1H..C4H not
    implemented, deviation documented there)."""
    import numpy as np
    qvapor = np.asarray(qvapor, dtype=float)
    p_full = np.asarray(p, dtype=float) + np.asarray(pb, dtype=float)
    nz = p_full.shape[0]
    dp = p_full[:nz - 1] - p_full[1:nz]          # [nz-1, ny, nx], level k vs k+1
    pw = np.sum(qvapor[:nz - 1] * dp, axis=0) / G_ACCEL
    return pw


def theta_to_temp(theta_perturbation: ArrayLike, p: ArrayLike, pb: ArrayLike,
                  theta_base: float = 300.0) -> np.ndarray:
    """WRF's T variable is PERTURBATION potential temperature (theta = T_var + 300). Actual
    temperature via the Poisson relation: Temp = theta_full * (p_full/P0)^(Rd/cp)."""
    import numpy as np
    theta_full = np.asarray(theta_perturbation, dtype=float) + theta_base
    p_full = np.asarray(p, dtype=float) + np.asarray(pb, dtype=float)
    return theta_full * (p_full / P0_PA) ** RD_CP


def geopotential_height_mass_levels(ph: ArrayLike, phb: ArrayLike) -> np.ndarray:
    """PH/PHB (perturbation + base-state geopotential) are on STAGGERED w-levels
    (bottom_top_stag, nz+1 interfaces). Z = (PH+PHB)/g at each interface; averaged to the nz
    MASS-level layer centers (same convention wrf-python/NCL use) so it lines up with P/PB for
    pressure-level interpolation. Returns [bottom_top, south_north, west_east]."""
    import numpy as np
    z_stag = (np.asarray(ph, dtype=float) + np.asarray(phb, dtype=float)) / G_ACCEL
    return (z_stag[:-1] + z_stag[1:]) / 2.0


def destagger_u(u: ArrayLike) -> np.ndarray:
    """west_east_stag (nx+1) -> mass points (nx): average adjacent pairs on the LAST axis."""
    import numpy as np
    u = np.asarray(u, dtype=float)
    return 0.5 * (u[..., :-1] + u[..., 1:])


def destagger_v(v: ArrayLike) -> np.ndarray:
    """south_north_stag (ny+1) -> mass points (ny): average adjacent pairs on AXIS 1 (the
    south_north axis of a [bottom_top, south_north, west_east] array)."""
    import numpy as np
    v = np.asarray(v, dtype=float)
    return 0.5 * (v[:, :-1, :] + v[:, 1:, :])


def rotate_grid_to_earth(u: ArrayLike, v: ArrayLike, cosalpha: ArrayLike, sinalpha: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Grid-relative (u, v) -> earth-relative: u_e = u*cos - v*sin, v_e = v*cos + u*sin.
    wrfinput U/V are grid-relative, GRIB u/v earth-relative (T3). cosalpha/sinalpha: WRF COSALPHA/
    SINALPHA, 2D, broadcast over levels. Formula is wrf-python's uvmet; the sign convention is pinned
    by test_rotate_grid_to_earth_90_degrees, not by WRF source (not present here)."""
    import numpy as np
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cosalpha = np.asarray(cosalpha, dtype=float)
    sinalpha = np.asarray(sinalpha, dtype=float)
    u_e = u * cosalpha - v * sinalpha
    v_e = v * cosalpha + u * sinalpha
    return u_e, v_e


# Live in coldstart_sample (it needs them too; plan-time only, T12); aliased for `cr.` callers.
vertical_grid_from_wrfinput = cs.vertical_grid_from_wrfinput
remove_namelist_key = cs.remove_namelist_key


def target_fields_from_wrfinput(path: str) -> dict[str, np.ndarray]:
    """The analog target state from the case's OWN wrfinput_d01 (never GRIB
    at the stated date, which is the wrong weather for a relabelled case; T3) -> {hgt500, tmp850,
    ugrd850, vgrd850 (earth-relative), pwat, tmp2m} on the case grid, NaN where the level is below
    ground. RebuildError when P/PB are missing."""
    import numpy as np
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        def var(name):
            return ds.variables[name][0] if name in ds.variables else None

        p, pb = var("P"), var("PB")
        if p is None or pb is None:
            raise RebuildError("target_fields_from_wrfinput: %s has no P/PB" % path)
        p_full = np.asarray(p, dtype=float) + np.asarray(pb, dtype=float)

        out = {}
        ph, phb = var("PH"), var("PHB")
        if ph is not None and phb is not None:
            z = geopotential_height_mass_levels(ph, phb)
            out["hgt500"] = interp_pressure_level_field(z, p_full, 500.0 * 100.0)

        t = var("T")
        if t is not None:
            temp = theta_to_temp(t, p, pb)
            out["tmp850"] = interp_pressure_level_field(temp, p_full, 850.0 * 100.0)

        u, v = var("U"), var("V")
        if u is not None and v is not None:
            u_mass, v_mass = destagger_u(u), destagger_v(v)
            cosalpha, sinalpha = var("COSALPHA"), var("SINALPHA")
            if cosalpha is not None and sinalpha is not None:
                u_e, v_e = rotate_grid_to_earth(u_mass, v_mass, cosalpha, sinalpha)
            else:
                print("WARNING: %s has no COSALPHA/SINALPHA -- ugrd850/vgrd850 left "
                      "grid-relative (not rotated to earth-relative)" % path)
                u_e, v_e = u_mass, v_mass
            out["ugrd850"] = interp_pressure_level_field(u_e, p_full, 850.0 * 100.0)
            out["vgrd850"] = interp_pressure_level_field(v_e, p_full, 850.0 * 100.0)

        qvapor = var("QVAPOR")
        if qvapor is not None:
            out["pwat"] = compute_pw(qvapor, p, pb)

        t2 = var("T2")
        if t2 is not None:
            out["tmp2m"] = np.asarray(t2, dtype=float)

        return out
    finally:
        ds.close()


def interp_pressure_level_column(field_k: ArrayLike, p_k: ArrayLike, target_pa: float) -> float:
    """Linear-in-ln(p) interpolation of a SINGLE COLUMN profile `field_k` (indexed by model
    level, k=0 = lowest/highest-pressure level) at `p_k` (same units as target_pa, Pa) to
    `target_pa`. Returns NaN when target_pa is outside [min(p_k), max(p_k)] -- 'below ground'
    (target_pa > p_k[0], i.e. the surface pressure at this column is itself below the target
    level) or above the model top."""
    import numpy as np
    field_k = np.asarray(field_k, dtype=float)
    p_k = np.asarray(p_k, dtype=float)
    if target_pa > p_k[0] or target_pa < p_k[-1]:
        return float("nan")
    # p_k is DECREASING (k=0 lowest level, highest pressure) -- find the bracket [k, k+1] with
    # p_k[k] >= target_pa >= p_k[k+1].
    idx = np.searchsorted(-p_k, -target_pa) - 1
    idx = int(np.clip(idx, 0, len(p_k) - 2))
    p_hi, p_lo = p_k[idx], p_k[idx + 1]
    if p_hi == p_lo:
        return float(field_k[idx])
    frac = (math.log(target_pa) - math.log(p_hi)) / (math.log(p_lo) - math.log(p_hi))
    return float(field_k[idx] + frac * (field_k[idx + 1] - field_k[idx]))


def interp_pressure_level_field(field_3d: ArrayLike, p_3d: ArrayLike, target_pa: float
                                ) -> np.ndarray:
    """interp_pressure_level_column applied to every (j, i) column of a 3D field
    [k, south_north, west_east]. Returns a 2D array, NaN where 850 hPa-style target is below
    ground or above the model top."""
    import numpy as np
    field_3d = np.asarray(field_3d, dtype=float)
    p_3d = np.asarray(p_3d, dtype=float)
    nz, ny, nx = field_3d.shape
    out = np.full((ny, nx), np.nan)
    for j in range(ny):
        for i in range(nx):
            out[j, i] = interp_pressure_level_column(field_3d[:, j, i], p_3d[:, j, i], target_pa)
    return out


def grid_identity_check(a: dict, b: dict, tol_deg: float = 0.01) -> tuple[bool, list[str]]:
    """Ports .route1-ref/grid_identity_gate.py's own gate exactly (attrs dict with tolerance
    0.01 deg for floats/exact for others, dims exact, corner+center lat/lon within tol_deg).
    a, b: {'attrs': {...}, 'dims': {'west_east':.., 'south_north':.., 'bottom_top':..},
    'corners': {'SW':(lat,lon), 'SE':.., 'NW':.., 'NE':.., 'center':..}}.
    Returns (ok, reasons)."""
    reasons = []
    # Compare only the attributes the projection uses: an unused one holds junk (matthew,
    # Mercator: TRUELAT2 60 vs 1e20; T10). TRUELAT1: proj 1/2/3; TRUELAT2: 1; STAND_LON: 1/2/6;
    # POLE_LAT/LON: 6.
    map_proj_a, map_proj_b = a["attrs"].get("MAP_PROJ"), b["attrs"].get("MAP_PROJ")
    attr_keys = ["MAP_PROJ", "DX", "DY", "CEN_LAT", "CEN_LON", "WEST-EAST_GRID_DIMENSION",
                "SOUTH-NORTH_GRID_DIMENSION", "BOTTOM-TOP_GRID_DIMENSION"]
    if map_proj_a in (1, 2, 3) or map_proj_b in (1, 2, 3):
        attr_keys.append("TRUELAT1")
    if map_proj_a == 1 or map_proj_b == 1:
        attr_keys.append("TRUELAT2")
    if map_proj_a in (1, 2, 6) or map_proj_b in (1, 2, 6):
        attr_keys.append("STAND_LON")
    if map_proj_a == 6 or map_proj_b == 6:
        attr_keys.append("POLE_LAT")
        attr_keys.append("POLE_LON")
    for k in attr_keys:
        va, vb = a["attrs"].get(k), b["attrs"].get(k)
        # numbers.Real, not (int, float): netCDF4 returns numpy.float32 attributes, which fail an
        # isinstance(x, float) check and would fall through to exact equality (route1 gate bug).
        if isinstance(va, numbers.Real) and isinstance(vb, numbers.Real):
            same = abs(va - vb) <= tol_deg
        else:
            same = (va == vb)
        if not same:
            reasons.append("%s differs: %r vs %r" % (k, va, vb))
    for d in ("west_east", "south_north", "bottom_top"):
        if a["dims"].get(d) != b["dims"].get(d):
            reasons.append("dim %s differs: %r vs %r" % (d, a["dims"].get(d), b["dims"].get(d)))
    max_diff = 0.0
    for name in ("SW", "SE", "NW", "NE", "center"):
        la, lo = a["corners"][name]
        lb, lob = b["corners"][name]
        max_diff = max(max_diff, abs(la - lb), abs(lo - lob))
    if max_diff > tol_deg:
        reasons.append("corner/center lat/lon max diff %.6f deg > %.3f" % (max_diff, tol_deg))
    return (len(reasons) == 0), reasons


DEFAULT_STATIC_RMSE_THRESH_M = 50.0   # m, terrain RMSE
DEFAULT_LU_AGREE_THRESH_PCT = 90.0    # % of points with equal LU_INDEX
DEFAULT_CORR_Z500_THRESH = 0.9        # pattern correlations, dimensionless
DEFAULT_CORR_T850_THRESH = 0.8
DEFAULT_CORR_PW_THRESH = 0.7
# 850 hPa wind separates real from relabelled dates (real 0.955-0.994, conus12 relabelled
# 0.13/0.15; Z500/T850/PW nearly pass on it; V3)
DEFAULT_CORR_WIND850_THRESH = 0.8


def compare_verdict(terrain_rmse_m: float, lu_agreement_pct: float, corr_z500: float,
                    corr_t850: float, corr_pw: float,
                    static_rmse_thresh: float = DEFAULT_STATIC_RMSE_THRESH_M,
                    lu_agree_thresh: float = DEFAULT_LU_AGREE_THRESH_PCT,
                    corr_z500_thresh: float = DEFAULT_CORR_Z500_THRESH,
                    corr_t850_thresh: float = DEFAULT_CORR_T850_THRESH,
                    corr_pw_thresh: float = DEFAULT_CORR_PW_THRESH, corr_u850: float | None = None,
                    corr_v850: float | None = None,
                    corr_wind850_thresh: float = DEFAULT_CORR_WIND850_THRESH
                    ) -> tuple[str, bool, bool]:
    """(verdict, static_ok, state_ok). verdict in {'consistent', 'different-static-data',
    'state-mismatch'}: consistent = static_ok & state_ok; different-static-data =
    (not static_ok) & state_ok; state-mismatch = not state_ok (the stated date's real weather is
    not this case's weather -- fictional/relabelled date, or a different event) -- this
    dominates: even a static mismatch alongside a state mismatch is reported as state-mismatch,
    since that is the more actionable/severe finding."""
    static_ok = (terrain_rmse_m < static_rmse_thresh) and (lu_agreement_pct >= lu_agree_thresh)
    state_ok = (corr_z500 >= corr_z500_thresh and corr_t850 >= corr_t850_thresh
               and corr_pw >= corr_pw_thresh)
    for corr_wind in (corr_u850, corr_v850):
        if corr_wind is not None and not corr_wind >= corr_wind850_thresh:   # NaN fails
            state_ok = False
    if not state_ok:
        verdict = "state-mismatch"
    elif not static_ok:
        verdict = "different-static-data"
    else:
        verdict = "consistent"
    return verdict, static_ok, state_ok


def read_wrfinput_fields_netcdf4(path: str) -> dict:
    """Thin netCDF4 reader (lazy import) for compare()'s own use -- d01 only. Returns a dict of
    plain numpy arrays/scalars: the static fields (HGT, LANDMASK, LU_INDEX, ISLTYP, MMINLU,
    NUM_LAND_CAT, DZS), the state fields (T2, TSK, PSFC, SMOIS, QVAPOR, P, PB, T, U, V, XLAT,
    XLONG), and the grid-identity attrs/dims/corners grid_identity_check needs. NOT independently
    unit-tested here (no netCDF4 in this dev environment) -- see module docstring."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    try:
        def var(name):
            return ds.variables[name][0] if name in ds.variables else None

        ny = ds.dimensions["south_north"].size
        nx = ds.dimensions["west_east"].size
        corners = {}
        xlat, xlong = var("XLAT"), var("XLONG")
        if xlat is not None:
            pts = {"SW": (0, 0), "SE": (0, nx - 1), "NW": (ny - 1, 0), "NE": (ny - 1, nx - 1),
                   "center": (ny // 2, nx // 2)}
            for name, (j, i) in pts.items():
                corners[name] = (float(xlat[j, i]), float(xlong[j, i]))
        attrs = {}
        for k in ("MAP_PROJ", "DX", "DY", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT",
                 "CEN_LON", "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
                 "BOTTOM-TOP_GRID_DIMENSION"):
            attrs[k] = getattr(ds, k, None)
        return {
            "attrs": attrs,
            "dims": {"west_east": nx, "south_north": ny,
                    "bottom_top": ds.dimensions.get("bottom_top", ds.dimensions.get("num_metgrid_levels")).size
                    if "bottom_top" in ds.dimensions or "num_metgrid_levels" in ds.dimensions else None},
            "corners": corners,
            "HGT": var("HGT"), "LANDMASK": var("LANDMASK"), "LU_INDEX": var("LU_INDEX"),
            "ISLTYP": var("ISLTYP"),
            "MMINLU": getattr(ds, "MMINLU", None), "NUM_LAND_CAT": getattr(ds, "NUM_LAND_CAT", None),
            "DZS": var("DZS"),
            "T2": var("T2"), "TSK": var("TSK"), "PSFC": var("PSFC"),
            "SMOIS": var("SMOIS")[0] if "SMOIS" in ds.variables else None,
            "QVAPOR": var("QVAPOR"), "P": var("P"), "PB": var("PB"), "T": var("T"),
            "U": var("U"), "V": var("V"), "PH": var("PH"), "PHB": var("PHB"),
        }
    finally:
        ds.close()


def compare(original_path: str, rebuilt_path: str,
            static_rmse_thresh: float = DEFAULT_STATIC_RMSE_THRESH_M,
            lu_agree_thresh: float = DEFAULT_LU_AGREE_THRESH_PCT,
            corr_z500_thresh: float = DEFAULT_CORR_Z500_THRESH,
            corr_t850_thresh: float = DEFAULT_CORR_T850_THRESH,
            corr_pw_thresh: float = DEFAULT_CORR_PW_THRESH,
            corr_wind850_thresh: float = DEFAULT_CORR_WIND850_THRESH) -> dict:
    """R6-B orchestration: read both files (lazy netCDF4), refuse unless grid identity holds,
    compute every metric, return the full report dict."""
    import numpy as np
    a = read_wrfinput_fields_netcdf4(original_path)
    b = read_wrfinput_fields_netcdf4(rebuilt_path)
    ok, reasons = grid_identity_check(b, a)   # b=rebuilt (candidate) vs a=original (reference)
    if not ok:
        raise RebuildError("compare: grid identity check FAILED: " + "; ".join(reasons))

    land_mask = (np.asarray(a["LANDMASK"]) == 0)   # water=0 (WRF convention) -> TSK over water
    static = {
        "terrain_rmse_m": rmse(b["HGT"], a["HGT"]),
        "terrain_max_abs_diff_m": max_abs_diff(b["HGT"], a["HGT"]),
        "landmask_agreement_pct": agreement_pct(b["LANDMASK"], a["LANDMASK"]),
        "lu_index_agreement_pct": agreement_pct(b["LU_INDEX"], a["LU_INDEX"]),
        "isltyp_agreement_pct": agreement_pct(b["ISLTYP"], a["ISLTYP"]),
        "mminlu": {"original": a["MMINLU"], "rebuilt": b["MMINLU"]},
        "num_land_cat": {"original": a["NUM_LAND_CAT"], "rebuilt": b["NUM_LAND_CAT"]},
        "soil_layer_count": {"original": len(a["DZS"]) if a["DZS"] is not None else None,
                             "rebuilt": len(b["DZS"]) if b["DZS"] is not None else None},
    }

    def state_metrics(name, av, bv, mask=None):
        return {"bias": bias(bv, av, mask=mask), "rmse": rmse(bv, av, mask=mask),
                "corr": pattern_correlation(bv, av, mask=mask)}

    state = {
        "T2": state_metrics("T2", a["T2"], b["T2"]),
        "TSK": state_metrics("TSK", a["TSK"], b["TSK"], mask=land_mask),
        "PSFC": state_metrics("PSFC", a["PSFC"], b["PSFC"]),
        "SMOIS_layer1": state_metrics("SMOIS_layer1", a["SMOIS"][0], b["SMOIS"][0])
                        if a["SMOIS"] is not None and b["SMOIS"] is not None else None,
    }
    pw_a = compute_pw(a["QVAPOR"], a["P"], a["PB"]) if a["QVAPOR"] is not None else None
    pw_b = compute_pw(b["QVAPOR"], b["P"], b["PB"]) if b["QVAPOR"] is not None else None
    if pw_a is not None and pw_b is not None:
        state["PW"] = state_metrics("PW", pw_a, pw_b)

    for label, target_hpa in (("Z500", 500.0), ("T850", 850.0), ("U850", 850.0), ("V850", 850.0)):
        if a.get("P") is None or b.get("P") is None:
            continue
        pa = np.asarray(a["P"]) + np.asarray(a["PB"])
        pb_ = np.asarray(b["P"]) + np.asarray(b["PB"])
        target_pa = target_hpa * 100.0
        if label == "Z500":
            if a.get("PH") is None or a.get("PHB") is None or b.get("PH") is None \
                    or b.get("PHB") is None:
                continue
            fa = geopotential_height_mass_levels(a["PH"], a["PHB"])
            fb = geopotential_height_mass_levels(b["PH"], b["PHB"])
        else:
            field_key = {"T850": "T", "U850": "U", "V850": "V"}[label]
            fa = a.get(field_key)
            fb = b.get(field_key)
            if fa is None or fb is None:
                continue
            if field_key == "T":
                fa = theta_to_temp(fa, a["P"], a["PB"])
                fb = theta_to_temp(fb, b["P"], b["PB"])
            elif field_key == "U":   # west_east_stag -> mass points
                fa = 0.5 * (np.asarray(fa)[..., :-1] + np.asarray(fa)[..., 1:])
                fb = 0.5 * (np.asarray(fb)[..., :-1] + np.asarray(fb)[..., 1:])
            elif field_key == "V":   # south_north_stag -> mass points
                fa = 0.5 * (np.asarray(fa)[:, :-1, :] + np.asarray(fa)[:, 1:, :])
                fb = 0.5 * (np.asarray(fb)[:, :-1, :] + np.asarray(fb)[:, 1:, :])
        interp_a = interp_pressure_level_field(fa, pa, target_pa)
        interp_b = interp_pressure_level_field(fb, pb_, target_pa)
        below_ground = np.isnan(interp_a) | np.isnan(interp_b)
        state[label] = state_metrics(label, interp_a, interp_b, mask=~below_ground)

    corr_z500 = state.get("Z500", {}).get("corr", float("nan"))
    corr_t850 = state.get("T850", {}).get("corr", float("nan"))
    corr_pw = state.get("PW", {}).get("corr", float("nan"))
    corr_u850 = state.get("U850", {}).get("corr", float("nan"))
    corr_v850 = state.get("V850", {}).get("corr", float("nan"))
    verdict, static_ok, state_ok = compare_verdict(
        static["terrain_rmse_m"], static["lu_index_agreement_pct"], corr_z500, corr_t850, corr_pw,
        static_rmse_thresh=static_rmse_thresh, lu_agree_thresh=lu_agree_thresh,
        corr_z500_thresh=corr_z500_thresh, corr_t850_thresh=corr_t850_thresh,
        corr_pw_thresh=corr_pw_thresh, corr_u850=corr_u850, corr_v850=corr_v850,
        corr_wind850_thresh=corr_wind850_thresh)

    return {
        "static": static, "state": state, "verdict": verdict,
        "static_ok": static_ok, "state_ok": state_ok,
        "thresholds": {"static_rmse_thresh_m": static_rmse_thresh,
                       "lu_agree_thresh_pct": lu_agree_thresh,
                       "corr_z500_thresh": corr_z500_thresh, "corr_t850_thresh": corr_t850_thresh,
                       "corr_pw_thresh": corr_pw_thresh,
                       "corr_wind850_thresh": corr_wind850_thresh},
        "note": "PW uses the P+PB level-difference dp method (MU/C1H..C4H not implemented -- "
                "see module docstring)",
    }


# ===========================================================================
# R6-C: analog -- find a real date whose weather matches the case. Scoring math is pure; the
# per-candidate GRIB fetch is best-effort network code, not exercised offline.
# ===========================================================================

ANALOG_FIELD_KEYS = ("hgt500", "tmp850", "ugrd850", "vgrd850", "pwat", "tmp2m")


def doy_of(dt: datetime.datetime) -> int:
    return dt.timetuple().tm_yday


def candidate_dates(stated_start: datetime.datetime, years: tuple[int, int], doy_window: int,
                    hour: str | int = "same") -> list[datetime.datetime]:
    """Every date in `years` (inclusive (start_year, end_year) tuple) within +-doy_window days
    of stated_start's day-of-year, at the same UTC hour as stated_start (or an explicit int
    `hour`) -- WRAPS across Dec/Jan (e.g. stated Jan 10, window 45 -> Nov 26 .. Feb 24 of each
    requested year, since day-of-year arithmetic is anchored at Jan 1 of that requested year and
    a negative offset lands in the PREVIOUS calendar year -- exactly the wrap). Returns a
    sorted, deduplicated list of naive UTC datetimes."""
    stated_doy = doy_of(stated_start)
    hh = stated_start.hour if hour == "same" else int(hour)
    y0, y1 = years
    out = set()
    for year in range(y0, y1 + 1):
        base = datetime.datetime(year, 1, 1)
        for delta in range(-doy_window, doy_window + 1):
            d = base + datetime.timedelta(days=(stated_doy - 1 + delta))
            out.add(datetime.datetime(d.year, d.month, d.day, hh))
    return sorted(out)


def climatological_analog(stated_start: datetime.datetime, years: tuple[int, int],
                          hour: str | int = "same") -> datetime.datetime:
    """No-target analog: same day-of-year and hour in the most recent year (y1).
    Single source of this rule (cmd_analog, scenario.generators.analog). DOY 366 in a non-leap y1
    falls back to Dec 31 (T14 fixed max(candidate_dates), which returned the window's last day)."""
    y0, y1 = years
    stated_doy = doy_of(stated_start)
    hh = stated_start.hour if hour == "same" else int(hour)
    candidate = datetime.datetime(y1, 1, 1) + datetime.timedelta(days=stated_doy - 1)
    if candidate.year != y1:
        # stated DOY 366 doesn't exist in y1 (y1 is not a leap year) -- nearest real day in y1.
        candidate = datetime.datetime(y1, 12, 31)
    return datetime.datetime(candidate.year, candidate.month, candidate.day, hh)


def block_average(field2d: ArrayLike, block: int = 8, min_blocks: int = 10) -> np.ndarray:
    """Block-average to ~block x block cells, keeping at least min_blocks blocks per axis.
    NaN-aware (below-ground points are NaN; T3): nanmean per block, an all-NaN block stays NaN."""
    import numpy as np
    import warnings
    field2d = np.asarray(field2d, dtype=float)
    ny, nx = field2d.shape
    by = block if ny // block >= min_blocks else max(1, ny // min_blocks)
    bx = block if nx // block >= min_blocks else max(1, nx // min_blocks)
    ny_c, nx_c = max(1, ny // by), max(1, nx // bx)
    trimmed = field2d[:ny_c * by, :nx_c * bx]
    reshaped = trimmed.reshape(ny_c, by, nx_c, bx)
    with warnings.catch_warnings():
        # an all-NaN block correctly produces NaN via a "Mean of empty slice" RuntimeWarning --
        # expected here, not a bug to surface.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(reshaped, axis=(1, 3))


def anomaly_pattern_corr_nrmse(candidate_field: ArrayLike, target_field: ArrayLike
                               ) -> tuple[float, float]:
    """(corr, nrmse) for one field: corr = pattern correlation of (field - domain mean)
    anomalies; nrmse = RMSE(candidate, target) / std(target)."""
    import numpy as np
    c = np.asarray(candidate_field, dtype=float)
    t = np.asarray(target_field, dtype=float)
    c_anom = c - np.nanmean(c)
    t_anom = t - np.nanmean(t)
    corr = pattern_correlation(c_anom, t_anom)
    rmse_val = rmse(c, t)
    std_t = float(np.nanstd(t))
    nrmse = (rmse_val / std_t) if std_t > 0 else float("inf")
    return corr, nrmse


def field_score(corr: float, nrmse: float) -> float:
    """field score = 0.5*(1-corr) + 0.5*min(nrmse,2)/2 -- 0 is a perfect match, larger is worse."""
    if corr != corr:   # NaN
        return float("inf")
    return 0.5 * (1.0 - corr) + 0.5 * min(nrmse, 2.0) / 2.0


def score_candidate(candidate_fields: dict, target_fields: dict) -> dict:
    """candidate_fields/target_fields: {field_key: 2D array} (already regridded + block
    averaged to the same coarse grid). Returns {'total', 'per_field': {key: {'corr','nrmse',
    'score'}}}."""
    per_field = {}
    scores = []
    for key in target_fields:
        if key not in candidate_fields:
            continue
        corr, nrmse = anomaly_pattern_corr_nrmse(candidate_fields[key], target_fields[key])
        s = field_score(corr, nrmse)
        per_field[key] = {"corr": corr, "nrmse": nrmse, "score": s}
        scores.append(s)
    total = sum(scores) / len(scores) if scores else float("inf")
    return {"total": total, "per_field": per_field}


def rank_candidates(candidates_scored: list[tuple]) -> list[tuple]:
    """candidates_scored: [(date, score_dict), ...]. Returns the SAME list sorted by ascending
    total score (lower = better match) -- ties broken by date for determinism."""
    return sorted(candidates_scored, key=lambda ds: (ds[1]["total"], ds[0]))


def two_stage_select(ranked: list, top_n: int = 10) -> list:
    """The best `top_n` candidates (by t0 score) that go on to the trajectory (boundary) scoring
    stage -- SPEC item R6-C: 'only for the best 10 by the t0 score -- two-stage'."""
    return ranked[:top_n]


def combine_t0_and_trajectory(t0_total: float, trajectory_total: float) -> float:
    """total = 0.5*t0 + 0.5*trajectory when a boundary trajectory was scored."""
    return 0.5 * t0_total + 0.5 * trajectory_total


def quality_flag(t0_field_corrs: list[float]) -> str:
    """good if EVERY t0 field corr >= 0.8, fair if >= 0.6, poor otherwise (including no data or
    any NaN -- min() with a NaN present is order-dependent in plain Python since NaN comparisons
    are always False, so NaN is filtered out explicitly rather than relying on min() to surface
    it)."""
    if not t0_field_corrs:
        return "poor"
    if any(c != c for c in t0_field_corrs):   # any NaN -> can't confirm a good fit
        return "poor"
    worst = min(t0_field_corrs)
    if worst >= 0.8:
        return "good"
    if worst >= 0.6:
        return "fair"
    return "poor"


def bdy_strip_means(u: ArrayLike | None, v: ArrayLike | None, t: ArrayLike | None, side: str
                    ) -> dict[str, float]:
    """Mean U, V, T over one wrfbdy_d01 boundary strip (<VAR>_B{XS,XE,YS,YE}; the
    caller passes the already-sliced arrays for one side and time). None inputs are skipped."""
    import numpy as np
    out = {}
    for name, arr in (("U", u), ("V", v), ("T", t)):
        if arr is not None:
            out[name] = float(np.nanmean(np.asarray(arr, dtype=float)))
    return out


def cmd_analog_offline_rank(target_fields: dict, candidates: list[tuple],
                            top_n_trajectory: int = 10) -> tuple[list, list]:
    """Pure orchestration over already-fetched/regridded arrays (no network): scores every
    candidate's t0 fields, ranks them, and returns (ranked, kept_for_trajectory) -- the top
    `top_n_trajectory` by t0 score. `candidates`: [(date, {field_key: 2D array}), ...]."""
    scored = [(date, score_candidate(fields, target_fields)) for date, fields in candidates]
    ranked = rank_candidates(scored)
    kept = two_stage_select(ranked, top_n=top_n_trajectory)
    return ranked, kept


def build_analog_output(ranked: list, quality: str, trajectory_scores: dict | None = None
                        ) -> dict:
    """OUT/analog.json's own shape: ranked list, the chosen analog (rank 1), and a quality
    flag. `trajectory_scores`: optional {date: trajectory_total} for the two-stage candidates --
    when given, the FINAL total for those dates is combine_t0_and_trajectory(t0, trajectory) and
    the list is RE-ranked by that combined total."""
    entries = []
    for date, score in ranked:
        entry = {"date": date.strftime(_WRF_TIME_FMT), "t0_total": score["total"],
                 "per_field": score["per_field"]}
        if trajectory_scores and date in trajectory_scores:
            entry["trajectory_total"] = trajectory_scores[date]
            entry["total"] = combine_t0_and_trajectory(score["total"], trajectory_scores[date])
        else:
            entry["total"] = score["total"]
        entries.append(entry)
    entries.sort(key=lambda e: (e["total"], e["date"]))
    chosen = entries[0] if entries else None
    return {"ranked": entries, "chosen": chosen, "quality": quality}


def cmd_analog(case_dir: str, out_dir: str, wrfinput: str | None = None, wrfbdy: str | None = None,
               years: str = "2021-2025", doy_window: int = 45, hour: str = "same", top: int = 5,
               workers: int = 8, namelist: str | None = None) -> dict:
    """R6-A/C orchestration for the CLI: resolves the case's own start date (namelist.input),
    builds the target state (needs wrfinput; falls back to climatological analog with
    quality='no-target' when none is given -- SPEC: 'No wrfinput at all (namelist only): no
    target -> return the climatological analog (same DOY and hour, most recent year with data)'),
    fetches/regrids candidates (best-effort, real network -- not exercised offline), scores,
    writes OUT/analog.json."""
    nl = cs.load_namelist(namelist or os.path.join(case_dir, "namelist.input"))
    stated_start, _stated_end, _run_s = cs.run_span(nl)
    y0, y1 = (int(x) for x in years.split("-"))
    candidates_dt = candidate_dates(stated_start, (y0, y1), doy_window, hour=hour)

    wi_path = wrfinput or os.path.join(case_dir, "wrfinput_d01")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(wi_path):
        # No target: climatological analog (same DOY/hour, most recent year).
        chosen_dt = climatological_analog(stated_start, (y0, y1), hour=hour)
        out = {"ranked": [], "chosen":
               {"date": chosen_dt.strftime(_WRF_TIME_FMT)} if chosen_dt else None,
               "quality": "no-target",
               "trajectory": "not implemented -- t0-only match",
               "note": "no wrfinput given -- climatological analog only (same DOY/hour, most "
                       "recent year with data)"}
        with open(os.path.join(out_dir, "analog.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("analog: no target (no wrfinput) -> climatological analog %s"
              % (out["chosen"]["date"] if out["chosen"] else "none"))
        return out

    # Target-based scoring: one live GRIB fetch per candidate (grib pixi env; not tested offline).
    import concurrent.futures
    gs_mod = cs._grib_state()
    case = {"start_year": stated_start.year, "start_month": stated_start.month,
           "start_day": stated_start.day, "start_hour": stated_start.hour}
    lat2d, lon2d, _mode = gs_mod.target_grid(case, wi_path)

    only_aws_nomads = []
    skipped_years = []
    for dt in candidates_dt:
        src, reason = cs._metdata().select_source(dt)
        if src in ("aws", "nomads"):
            only_aws_nomads.append(dt)
        else:
            skipped_years.append((dt.year, src, reason))
    if skipped_years:
        print("analog: skipping dates with no cheap per-message source: %s"
              % sorted(set((y, s) for y, s, _r in skipped_years)))

    cache_dir = os.path.join(out_dir, "grib_analog")
    os.makedirs(cache_dir, exist_ok=True)

    def fetch_one(dt):
        try:
            fields = _fetch_and_regrid_candidate(dt, lat2d, lon2d, cache_dir)
        except Exception as exc:
            return dt, None, str(exc)
        return dt, fields, None

    # Target from the case's own wrfinput, never GRIB at the stated date (T3).
    try:
        target_fields = target_fields_from_wrfinput(wi_path)
    except Exception as exc:
        raise RebuildError("analog: could not build the target state from %s -- %s"
                           % (wi_path, exc))
    target_blocked = {k: block_average(v) for k, v in target_fields.items()}

    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for dt, fields, err in pool.map(fetch_one, only_aws_nomads):
            if fields is None:
                print("analog: %s fetch failed -- %s" % (dt.strftime(_WRF_TIME_FMT), err))
                continue
            candidates.append((dt, {k: block_average(v) for k, v in fields.items()}))

    ranked, kept = cmd_analog_offline_rank(target_blocked, candidates, top_n_trajectory=top)
    t0_corrs = [pf["corr"] for pf in (ranked[0][1]["per_field"].values() if ranked else [])]
    quality = quality_flag(t0_corrs)

    # Boundary-trajectory scoring is not wired up; analog.json says so (T3).
    trajectory_scores = None
    trajectory_note = "not implemented -- t0-only match"
    wb_path = wrfbdy or os.path.join(case_dir, "wrfbdy_d01")
    if os.path.exists(wb_path) and kept:
        print("analog: found %s but boundary-trajectory scoring is not implemented -- "
              "t0-only match" % wb_path)

    out = build_analog_output(ranked, quality, trajectory_scores=trajectory_scores)
    out["trajectory"] = trajectory_note
    out["candidates_screened"] = len(candidates_dt)
    out["candidates_with_cheap_source"] = len(only_aws_nomads)
    with open(os.path.join(out_dir, "analog.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("analog: chosen=%s quality=%s (ranked %d candidates)"
          % (out["chosen"]["date"] if out["chosen"] else "none", quality, len(ranked)))
    return out


def _fetch_and_regrid_candidate(dt, lat2d, lon2d, cache_dir, grib_dir=None, source="auto"):
    """One date's ANALOG_FIELD_KEYS (state-purpose byte-range fetch + bbox crop + regrid onto
    (lat2d, lon2d)). Best-effort: real network, not exercised by the offline test suite."""
    md_mod = cs._metdata()
    gs_mod = cs._grib_state()
    cycle_dt, fhour = md_mod.gfs_cycle_for(dt)
    src_choice, _reason = md_mod.select_source(dt, override=source, grib_dir=grib_dir)
    if src_choice is None:
        raise RebuildError("no cheap per-message source for %s" % dt.strftime(_WRF_TIME_FMT))
    out = {}
    for key in ANALOG_FIELD_KEYS:
        slat, slon, sval = md_mod.fetch_and_decode_state_field(
            key, cycle_dt, fhour, src_choice, cache_dir, grib_dir=grib_dir)
        slat, slon, sval = md_mod.crop_bbox(slat, slon, sval, float(lat2d.min()), float(lat2d.max()),
                                            float(lon2d.min()), float(lon2d.max()))
        out[key] = gs_mod.regrid_nearest_regular(slat, slon, sval, lat2d, lon2d)
    return out


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="case_rebuild.py", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rebuild = sub.add_parser("rebuild")
    rebuild_sub = rebuild.add_subparsers(dest="rebuild_cmd", required=True)

    sp = rebuild_sub.add_parser("plan")
    sp.add_argument("--case-dir", required=True)
    sp.add_argument("--namelist")
    sp.add_argument("--namelist-wps")
    sp.add_argument("--geo-from-wrfinput")
    sp.add_argument("--out", required=True)
    sp.add_argument("--date-shift-to")
    sp.add_argument("--forcing")
    sp.add_argument("--verify-steps", type=int, default=0)
    sp.add_argument("--nodes", type=int, default=1)
    sp.add_argument("--ranks", type=int, default=wtune.machine.cores_per_node())
    sp.add_argument("--omp", type=int, default=1)
    sp.add_argument("--nx", type=int)
    sp.add_argument("--ny", type=int)
    sp.add_argument("--ncdump")
    sp.add_argument("--geog-path")
    sp.add_argument("--grib-dir")
    sp.add_argument("--ml-model", metavar="REPORT.json", default=None,
                    help="K5 (2026-09-24): after real.exe, apply state_correction_model.py's "
                    "ML state correction (--targets-from-report) to the fresh wrfinput_d01 "
                    "before any wrf.exe smoke run. Default: off.")

    sp = rebuild_sub.add_parser("fetch")
    sp.add_argument("--out", required=True)
    sp.add_argument("--cache-dir")
    sp.add_argument("--grib-dir")
    sp.add_argument("--keep-archives", action="store_true")

    sp = rebuild_sub.add_parser("stage")
    sp.add_argument("--out", required=True)
    sp.add_argument("--site", default=wtune.machine.site())

    sp = rebuild_sub.add_parser("finalize")
    sp.add_argument("--out", required=True)
    sp.add_argument("--keep-wrfinput", action="store_true",
                    help="K3 (2026-09-24): keep the case's ORIGINAL wrfinput_d0N (interior "
                    "state) instead of real.exe's own cold-start one, while still keeping "
                    "real.exe's wrfbdy_d01/wrflowinp (boundaries) -- for a case that has "
                    "wrfinput but no wrfbdy_d01 of its own. Refuses on a grid mismatch.")

    sp = rebuild_sub.add_parser("fix-span-wps-namelist")
    sp.add_argument("--out", required=True)

    sp = rebuild_sub.add_parser("fix-span-real-namelist")
    sp.add_argument("--out", required=True)
    sp.add_argument("--ncdump")
    sp.add_argument("--serial", action="store_true")

    sp = rebuild_sub.add_parser("fix-span-wrf-namelist")
    sp.add_argument("--out", required=True)
    sp.add_argument("--nproc-x", type=int, required=True)
    sp.add_argument("--nproc-y", type=int, required=True)
    sp.add_argument("--numtiles", type=int, default=1)

    sp = sub.add_parser("namelist-from-wrfinput")
    sp.add_argument("--case-dir", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--hours", type=float, default=None)

    sp = sub.add_parser("new-domain")
    sp.add_argument("--center", required=True, metavar="LAT,LON")
    sp.add_argument("--size-km", required=True, metavar="W,H")
    sp.add_argument("--dx-km", required=True, type=float)
    sp.add_argument("--start", required=True, metavar="YYYY-MM-DD_HH:MM:SS")
    sp.add_argument("--hours", required=True, type=float)
    sp.add_argument("--e-vert", type=int, default=45)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("complete")
    sp.add_argument("--in", dest="in_dir", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--center", metavar="LAT,LON")
    sp.add_argument("--size-km", metavar="W,H")
    sp.add_argument("--dx-km", type=float)
    sp.add_argument("--start", metavar="YYYY-MM-DD_HH:MM:SS")
    sp.add_argument("--hours", type=float)
    sp.add_argument("--ml-model", metavar="REPORT.json", default=None)

    sp = sub.add_parser("compare")
    sp.add_argument("--original", required=True)
    sp.add_argument("--rebuilt", required=True)
    sp.add_argument("--json")
    sp.add_argument("--static-rmse-thresh", type=float, default=DEFAULT_STATIC_RMSE_THRESH_M)
    sp.add_argument("--lu-agree-thresh", type=float, default=DEFAULT_LU_AGREE_THRESH_PCT)
    sp.add_argument("--corr-z500-thresh", type=float, default=DEFAULT_CORR_Z500_THRESH)
    sp.add_argument("--corr-t850-thresh", type=float, default=DEFAULT_CORR_T850_THRESH)
    sp.add_argument("--corr-pw-thresh", type=float, default=DEFAULT_CORR_PW_THRESH)
    sp.add_argument("--corr-wind850-thresh", type=float, default=DEFAULT_CORR_WIND850_THRESH)

    sp = sub.add_parser("analog")
    sp.add_argument("--case-dir", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--namelist")
    sp.add_argument("--wrfinput")
    sp.add_argument("--wrfbdy")
    sp.add_argument("--years", default="2021-2025")
    sp.add_argument("--doy-window", type=int, default=45)
    sp.add_argument("--hour", default="same")
    sp.add_argument("--top", type=int, default=5)
    sp.add_argument("--workers", type=int, default=8)

    args = ap.parse_args(argv)

    try:
        if args.cmd == "rebuild":
            if args.rebuild_cmd == "plan":
                cmd_rebuild_plan(args.case_dir, args.out, namelist=args.namelist,
                                 namelist_wps=args.namelist_wps,
                                 geo_from_wrfinput=args.geo_from_wrfinput,
                                 date_shift_to=args.date_shift_to, forcing=args.forcing,
                                 verify_steps=args.verify_steps, nodes=args.nodes,
                                 ranks=args.ranks, omp=args.omp, nx=args.nx, ny=args.ny,
                                 ncdump=args.ncdump, geog_path=args.geog_path,
                                 grib_dir=args.grib_dir, ml_model=args.ml_model)
                return 0
            if args.rebuild_cmd == "fetch":
                cmd_rebuild_fetch(args.out, cache_dir=args.cache_dir, grib_dir=args.grib_dir,
                                  keep_archives=args.keep_archives)
                return 0
            if args.rebuild_cmd == "stage":
                cmd_rebuild_stage(args.out, site=args.site)
                return 0
            if args.rebuild_cmd == "finalize":
                cmd_rebuild_finalize(args.out, keep_wrfinput=args.keep_wrfinput)
                return 0
            if args.rebuild_cmd == "fix-span-wps-namelist":
                cmd_rebuild_fix_span_wps_namelist(args.out)
                return 0
            if args.rebuild_cmd == "fix-span-real-namelist":
                cmd_rebuild_fix_span_real_namelist(args.out, ncdump=args.ncdump,
                                                   serial=args.serial)
                return 0
            if args.rebuild_cmd == "fix-span-wrf-namelist":
                cmd_rebuild_fix_span_wrf_namelist(args.out, args.nproc_x, args.nproc_y,
                                                  numtiles=args.numtiles)
                return 0

        if args.cmd == "namelist-from-wrfinput":
            cmd_namelist_from_wrfinput(args.case_dir, args.out, hours=args.hours)
            return 0

        if args.cmd == "new-domain":
            cmd_new_domain(center=args.center, size_km=args.size_km, dx_km=args.dx_km,
                           start=args.start, hours=args.hours, e_vert=args.e_vert,
                           out_dir=args.out)
            return 0

        if args.cmd == "complete":
            cmd_complete(args.in_dir, args.out, center=args.center, size_km=args.size_km,
                        dx_km=args.dx_km, start=args.start, hours=args.hours,
                        ml_model=args.ml_model)
            return 0

        if args.cmd == "compare":
            result = compare(args.original, args.rebuilt,
                             static_rmse_thresh=args.static_rmse_thresh,
                             lu_agree_thresh=args.lu_agree_thresh,
                             corr_z500_thresh=args.corr_z500_thresh,
                             corr_t850_thresh=args.corr_t850_thresh,
                             corr_pw_thresh=args.corr_pw_thresh,
                             corr_wind850_thresh=args.corr_wind850_thresh)
            print(json.dumps(result, indent=2, default=str))
            if args.json:
                with open(args.json, "w") as f:
                    json.dump(result, f, indent=2, default=str)
            return 0

        if args.cmd == "analog":
            cmd_analog(args.case_dir, args.out, wrfinput=args.wrfinput, wrfbdy=args.wrfbdy,
                      years=args.years, doy_window=args.doy_window, hour=args.hour,
                      top=args.top, workers=args.workers, namelist=args.namelist)
            return 0
    except (RebuildError, cs.ColdstartError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
