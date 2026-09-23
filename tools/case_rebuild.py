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
regrid_nearest_regular), wtune_predict.py (parse_namelist, g/g_dom, sunlit_fraction),
wtune_runlength.py (via wtune_predict).

Module-level imports are STDLIB + coldstart_sample + wtune_predict ONLY (both of those are
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
import datetime
import json
import math
import os
import numbers
import re
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wtune_predict as predict     # repo root -- stdlib only
import coldstart_sample as cs       # tools/ -- stdlib only at module level (lazy md/gs)

_WRF_TIME_FMT = cs._WRF_TIME_FMT
DEFAULT_REGISTRY = os.path.join(_REPO_ROOT, "wtune_catalog_registry.json")


class RebuildError(Exception):
    """A user-facing input/modeling error, same role as coldstart_sample.ColdstartError."""


# ===========================================================================
# R6-A: rebuild -- namelist completion + registry validation
# ===========================================================================

def load_registry(path=DEFAULT_REGISTRY):
    with open(path) as f:
        return json.load(f)


def registry_valid_keys(registry):
    """Set of every valid namelist key name (lowercased), across every group in the Registry
    catalog (wtune_catalog_registry.json: {'groups': {group_name: [{'name': 'GMT', ...}, ...]}})
    -- flat/global, not scoped per &group, since a namelist's own group names don't always align
    1:1 with the Registry's internal group keys."""
    keys = set()
    for _group, entries in registry.get("groups", {}).items():
        for e in entries:
            name = e.get("name")
            if name:
                keys.add(name.lower())
    return keys


def validate_namelist_against_registry(nl, registry):
    """nl: wtune_predict.parse_namelist's result ({group: {key: [values]}}). Returns a sorted
    list of warning strings for keys not found in the Registry catalog (case-insensitive) --
    NEVER raises, NEVER deletes anything; unknown keys are reported, not touched.

    T11 (2026-09-24): a namelist group the catalog does not cover AT ALL (e.g. &namelist_quilt --
    wtune_catalog_registry.json has no 'namelist_quilt' group, even though nio_groups/
    nio_tasks_per_group are real WRF keys) must not make every one of its keys look 'unknown':
    that conflates 'this key does not exist in WRF' with 'the catalog just never covered this
    group'. Such a group gets ONE 'group not in catalog' warning instead, and its keys are not
    individually flagged."""
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


def registry_group_for_key(key, registry):
    """The Registry group (lowercase) that owns `key`, or None if the catalog has no group
    entry for it at all. Used by complete_namelist so a newly-inserted fill-if-missing/sync key
    lands in its OWN group (e_vert/p_top_requested/... -> &domains, history_interval/... ->
    &time_control, physics_suite/num_land_cat -> &physics) instead of always &time_control --
    every key complete_namelist calls this for has exactly one owning group in the catalog
    (verified against wtune_catalog_registry.json, 2026-09-24)."""
    for group, entries in registry.get("groups", {}).items():
        for e in entries:
            if e.get("name", "").lower() == key.lower():
                return group
    return None


# T12 (2026-09-24): moved to coldstart_sample.py (case_rebuild imports coldstart_sample, not the
# reverse, and coldstart_sample.apply_vertical_grid_override needs this too) -- aliased here so
# every existing `cr._namelist_values_equal` call site keeps working unchanged.
_namelist_values_equal = cs.namelist_values_equal


def complete_namelist(nl_text, facts, registry=None, vertical_grid=None):
    """(new_text, changes). Every change is {key, old, new, reason}. `facts`: max_dom,
    num_metgrid_levels, num_metgrid_soil_levels (optional), num_land_cat (optional),
    interval_seconds, start_dt, end_dt (the SPAN -- already date-shifted if applicable),
    dx_km (d01), parent_grid_ratio (list, index d = domain d+1's own ratio; index 0 unused).
    `registry`: the loaded wtune_catalog_registry.json dict (default: load_registry()) -- every
    key this function inserts is placed into ITS OWN Registry group via
    registry_group_for_key, not hardcoded to &time_control (T11, 2026-09-24: real.exe FATAL'd
    'Maybe here?: p_top_requested =' because the old code appended every missing key to
    &time_control regardless of where WRF actually expects it -- p_top_requested is &domains).

    ALWAYS SYNC (overwritten even if present, added if missing -- met_em/geo_em/span facts the
    case's own namelist.input cannot know without them): num_metgrid_levels,
    num_metgrid_soil_levels, num_land_cat, interval_seconds, start_*/end_* per domain, run_*.
    `vertical_grid` (optional, {'eta_levels': [...], 'p_top_pa': ...} -- plan.json's own schema,
    see coldstart_sample.read_vertical_grid_or_warn/T12): when given, e_vert/eta_levels/
    p_top_requested are instead ALWAYS overridden from the case's own wrfinput ZNW/P_TOP (T8,
    2026-09-24 -- see coldstart_sample.vertical_grid_from_wrfinput's docstring), and dzbot/
    dzstretch_s/dzstretch_u/auto_levels_opt are removed outright (they conflict with an explicit
    eta_levels). T12: this dict must come from plan.json (read at PLAN time, login node) --
    complete_namelist itself never imports numpy/netCDF4, so it stays safe to call from a batch
    job under /usr/bin/python3.

    FILL ONLY IF MISSING (never overrides a user value; skipped for e_vert/p_top_requested when
    `vertical_grid` is given): e_vert=45, p_top_requested=5000,
    time_step = largest int <= 6*dx_km(d01) that is >=1, parent_time_step_ratio =
    parent_grid_ratio per nest, history_interval=60 (min, only if NEITHER history_interval NOR
    history_interval_m exists at all), frames_per_outfile=1, restart=.false.,
    restart_interval=run minutes (one restart at the end), io_form_{history,restart,input,
    boundary}=2, and physics_suite='CONUS' (only if NO &physics scheme key is present at all)."""
    txt = nl_text
    text0 = nl_text
    changes = []
    max_dom = int(facts["max_dom"])
    registry = registry if registry is not None else load_registry()

    def group_for(key, default="time_control"):
        return registry_group_for_key(key, registry) or default

    def sync(key, value, ndom=1, reason=""):
        nonlocal txt
        old = cs.get_first(txt, key)
        new_str = str(value)
        # T11: numeric-equal values ('01' vs '1') are not a change, only a different string
        # spelling of the same one -- do not log it.
        if old is not None and _namelist_values_equal(old.strip(), new_str):
            return
        txt = cs.set_list(txt, key, value, ndom, new_key_group=group_for(key))
        changes.append({"key": key, "old": old, "new": new_str, "reason": reason})

    def fill_if_missing(key, value, ndom=1, reason=""):
        nonlocal txt
        if cs.count_key(txt, key) > 0:
            return
        txt = cs.set_list(txt, key, value, ndom, new_key_group=group_for(key))
        changes.append({"key": key, "old": None, "new": str(value), "reason": reason})

    # -- always sync --
    sync("num_metgrid_levels", int(facts["num_metgrid_levels"]), 1, "met_em NUM_METGRID_LEVELS")
    if facts.get("num_metgrid_soil_levels") is not None:
        sync("num_metgrid_soil_levels", int(facts["num_metgrid_soil_levels"]), 1,
            "met_em NUM_METGRID_SOIL_LEVELS")
    if facts.get("num_land_cat") is not None:
        sync("num_land_cat", int(facts["num_land_cat"]), 1, "met_em/geo_em NUM_LAND_CAT")
    sync("interval_seconds", int(facts["interval_seconds"]), 1, "forcing cadence")

    start_dt, end_dt = facts["start_dt"], facts["end_dt"]
    for k, v in (("year", start_dt.year), ("month", start_dt.month), ("day", start_dt.day),
                 ("hour", start_dt.hour), ("minute", start_dt.minute), ("second", start_dt.second)):
        sync("start_" + k, v, max_dom, "rebuild span start")
    for k, v in (("year", end_dt.year), ("month", end_dt.month), ("day", end_dt.day),
                 ("hour", end_dt.hour), ("minute", end_dt.minute), ("second", end_dt.second)):
        sync("end_" + k, v, max_dom, "rebuild span end")
    run_s = int(round((end_dt - start_dt).total_seconds()))
    sync("run_days", 0, 1, "rebuild span")
    sync("run_hours", 0, 1, "rebuild span")
    sync("run_minutes", 0, 1, "rebuild span")
    sync("run_seconds", run_s, 1, "rebuild span")

    # -- fill only if missing --
    if vertical_grid is not None:
        # T8 (2026-09-24): reproduce the case's own vertical grid EXACTLY -- see
        # vertical_grid_from_wrfinput's docstring for why dzbot/dzstretch can silently diverge
        # from what the wrfinput was actually built with. These three ALWAYS override (they are
        # not fill_if_missing), and the conflicting dz*/auto_levels_opt keys are removed.
        for key in ("dzbot", "dzstretch_s", "dzstretch_u", "auto_levels_opt"):
            txt, n_removed = remove_namelist_key(txt, key)
            if n_removed:
                changes.append({"key": key, "old": "present", "new": None,
                                "reason": "conflicts with explicit eta_levels from wrfinput ZNW"})
        eta_levels = list(vertical_grid["eta_levels"])
        old_e_vert = cs.get_first(txt, "e_vert")
        new_e_vert = str(len(eta_levels))
        if old_e_vert is None or not _namelist_values_equal(old_e_vert.strip(), new_e_vert):
            txt = cs.set_list(txt, "e_vert", len(eta_levels), max_dom,
                              new_key_group=group_for("e_vert"))
            changes.append({"key": "e_vert", "old": old_e_vert, "new": new_e_vert,
                            "reason": "= len(wrfinput ZNW)"})
        txt = cs.set_list_values(txt, "eta_levels", eta_levels,
                                 new_key_group=group_for("eta_levels"))
        changes.append({"key": "eta_levels", "old": None, "new": str(eta_levels),
                        "reason": "= wrfinput ZNW (exact vertical grid reproduction)"})
        old_ptop = cs.get_first(txt, "p_top_requested")
        # T12 (2026-09-24): 'p_top_pa', matching plan.json's own vertical_grid schema (the same
        # dict apply_vertical_grid_override/read_vertical_grid_or_warn use in coldstart_sample).
        new_ptop = str(vertical_grid["p_top_pa"])
        if old_ptop is None or not _namelist_values_equal(old_ptop.strip(), new_ptop):
            txt = cs.set_list(txt, "p_top_requested", vertical_grid["p_top_pa"], 1,
                              new_key_group=group_for("p_top_requested"))
            changes.append({"key": "p_top_requested", "old": old_ptop, "new": new_ptop,
                            "reason": "= wrfinput P_TOP"})
    else:
        fill_if_missing("e_vert", 45, max_dom, "default vertical levels")
        fill_if_missing("p_top_requested", 5000, 1, "default model top (Pa)")
    dx_km = float(facts["dx_km"])
    time_step = max(1, int(6.0 * dx_km))
    fill_if_missing("time_step", time_step, 1, "largest int <= 6*dx_km(d01), >=1")
    if max_dom > 1 and cs.count_key(txt, "parent_time_step_ratio") == 0:
        ratios = [1] + [int(facts["parent_grid_ratio"][d]) for d in range(1, max_dom)]
        txt = cs.set_list_values(txt, "parent_time_step_ratio", ratios,
                                 new_key_group=group_for("parent_time_step_ratio"))
        changes.append({"key": "parent_time_step_ratio", "old": None, "new": str(ratios),
                        "reason": "= parent_grid_ratio per nest"})
    if cs.count_key(txt, "history_interval") == 0 and cs.count_key(txt, "history_interval_m") == 0:
        fill_if_missing("history_interval", 60, max_dom, "default history cadence (minutes)")
    fill_if_missing("frames_per_outfile", 1, 1, "default")
    fill_if_missing("restart", ".false.", 1, "default")
    run_minutes = int(round(run_s / 60.0))
    fill_if_missing("restart_interval", run_minutes, 1, "one restart at the end of the run")
    for key in ("io_form_history", "io_form_restart", "io_form_input", "io_form_boundary"):
        fill_if_missing(key, 2, 1, "default netCDF io_form")

    scheme_keys = ("mp_physics", "ra_lw_physics", "ra_sw_physics", "bl_pbl_physics", "cu_physics",
                  "sf_sfclay_physics", "sf_surface_physics", "physics_suite")
    if not any(cs.count_key(text0, k) > 0 for k in scheme_keys):
        txt = cs.set_list_values(txt, "physics_suite", ["'CONUS'"], new_key_group="physics")
        changes.append({"key": "physics_suite", "old": None, "new": "'CONUS'",
                        "reason": "no &physics scheme key present at all"})

    return txt, changes


# ===========================================================================
# R6-A: rebuild -- span computation, plan/fetch/stage/finalize lifecycle
# ===========================================================================

def rebuild_span(nl, date_shift_to=None):
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


def cmd_rebuild_plan(case_dir, out_dir, namelist=None, namelist_wps=None, geo_from_wrfinput=None,
                     date_shift_to=None, forcing=None, verify_steps=0, nodes=1, ranks=72, omp=1,
                     nx=None, ny=None, ncdump=None, now=None, geog_path=None, grib_dir=None):
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

    # T12 (2026-09-24, job 965): read ZNW/P_TOP HERE (login node, netCDF4/numpy available) --
    # fix-span-real-namelist runs inside the batch job under /usr/bin/python3 and can no longer
    # do this itself. A failed read must be recorded loudly (vertical_grid: null + a warning IN
    # plan.json), not silently skipped at batch time -- that silent skip was exactly the job 965
    # bug.
    vertical_grid, vertical_grid_warning = cs.read_vertical_grid_or_warn(
        wi_path if os.path.exists(wi_path) else None)
    if vertical_grid_warning:
        print("WARNING: %s" % vertical_grid_warning)

    config = {"ranks": ranks, "omp": omp, "nx": nx or 1, "ny": ny or (ranks // max(omp, 1)),
             "numtiles": 1}
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
        # T8/T12 (2026-09-24): only when the case's own wrfinput_d01 actually exists.
        # wrfinput_path is provenance only now (fix-span-real-namelist no longer re-reads it --
        # vertical_grid below is what it actually applies).
        "wrfinput_path": os.path.abspath(wi_path) if os.path.exists(wi_path) else None,
        "vertical_grid": vertical_grid, "vertical_grid_warning": vertical_grid_warning,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    with open(os.path.join(out_dir, "namelist.wps"), "w") as f:
        f.write(cs.render_namelist_wps(wps_fields))
    print("span: %s .. %s (%s, %dh cadence)" % (plan["span_start"], plan["span_end"],
                                                 primary.dataset, primary.forcing_interval_h))
    return plan


def cmd_rebuild_fetch(out_dir, cache_dir=None, grib_dir=None, keep_archives=False, probe_fn=None,
                      download_fn=None):
    with open(os.path.join(out_dir, "plan.json")) as f:
        plan = json.load(f)
    cache_dir = cache_dir or os.path.join(out_dir, "grib")
    os.makedirs(cache_dir, exist_ok=True)
    T = datetime.datetime.strptime(plan["span_start"], _WRF_TIME_FMT)
    T_end = datetime.datetime.strptime(plan["span_end"], _WRF_TIME_FMT)
    chain = cs.resolve_forcing_chain_for(T, grib_dir=grib_dir, forcing_pin=plan.get("forcing_pin"))
    chosen, tried = cs.choose_forcing_for_window(T, T_end, chain, probe_fn=probe_fn)
    if chosen is None:
        # T4b (2026-09-24): always write the probe record, even on failure.
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
    # T9 (2026-09-24): see coldstart_sample.resolve_and_fetch_forcing's identical fix -- record
    # the exact basenames fetch wrote (FNL/.grib, ncei_g3-g4/.grb, GFS/.grib2 all differ) so
    # render_rebuild_sbatch's link_grib.csh step links exactly this list, not a *.grib2 glob.
    with open(os.path.join(out_dir, "grib_manifest.txt"), "w") as f:
        for name in file_names:
            f.write(name + "\n")
    print("rebuild fetch: %d bytes (%s, %d files)" % (total_bytes, chosen.dataset, len(paths)))
    return total_bytes, forcing


def _rebuild_forcing_choice(out_dir):
    """The chosen forcing (prefix/vtable/interval) for this rebuild's span -- forcing.json if
    `fetch` already ran, else the plan-time default (mirrors
    coldstart_sample._load_window_forcing / _default_per_window_forcing)."""
    with open(os.path.join(out_dir, "plan.json")) as f:
        plan = json.load(f)
    forcing_path = os.path.join(out_dir, "forcing.json")
    if os.path.exists(forcing_path):
        with open(forcing_path) as f:
            forcing = json.load(f)
        chosen = forcing.get("chosen")
        if chosen:
            return chosen, plan
    primary = plan["forcing_chain_primary"]
    return primary, plan


def cmd_rebuild_stage(out_dir, site="scc"):
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
        # T13 (2026-09-24): see coldstart_sample.stage_plan's identical fix -- Vtable.GFS_PRMSL
        # (NCEI g3/g4 + FNL) is GENERATED from the real Vtable.GFS, not symlinked.
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


def cmd_rebuild_finalize(out_dir):
    """Login-node, after the job: collect OUT/case/ (copies, not symlinks) of namelist.wps,
    namelist.input, wrfinput_d0N, wrfbdy_d01, wrflowinp_d0N if produced, plus rebuild_report.json
    (changes from complete_namelist, forcing source + probe record, span, shift, and inspect's
    own capability/geog warnings)."""
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
    for d in range(1, max_dom + 1):
        copy_if_exists(os.path.join(out_dir, "real", "wrfinput_d%02d" % d), "wrfinput_d%02d" % d)
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
    with open(os.path.join(out_dir, "rebuild_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("finalized %s -> %s" % (out_dir, case_out))
    return report


def cmd_rebuild_fix_span_wps_namelist(out_dir):
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


def cmd_rebuild_fix_span_real_namelist(out_dir, ncdump=None, serial=False):
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
    # T12 (2026-09-24, job 965): this subcommand runs INSIDE the batch job under /usr/bin/python3
    # (stdlib only -- no numpy), so it can no longer call vertical_grid_from_wrfinput itself (that
    # WAS the job 965 bug: 'WARNING: vertical grid fidelity (T8) skipped -- No module named
    # numpy', silently). The read happens ONCE at PLAN time (login node, netCDF4 env -- see
    # cmd_rebuild_plan) and is stored verbatim in plan.json; this just reads it back.
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


def cmd_rebuild_fix_span_wrf_namelist(out_dir, nproc_x, nproc_y, numtiles=1):
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


def render_rebuild_sbatch(plan, chosen, site="scc", estimate_min=30):
    """One sbatch: geogrid -> ungrib+metgrid for the SPAN (exact expected-file checks, reusing
    coldstart_sample.expected_wps_intermediate_files/expected_met_em_files) -> real.exe for the
    span (same timeout 600 + serial-under-timeout-1800 fallback as coldstart_sample's own
    per-window real.exe launch) -> optional wrf.exe smoke run of verify_steps (history/restart
    off). No LD_PRELOAD, no wrapper around wrf.exe; --exclude=head; geogrid/metgrid TBL dirs;
    OMP_NUM_THREADS=1 for WPS/real, =config omp for the smoke wrf.exe -- all reused conventions
    from coldstart_sample.render_run_sbatch, not reimplemented."""
    out_dir_abs = plan.get("_out_dir")
    nodes = plan["nodes"]
    ranks = plan["config"]["ranks"]
    omp = plan["config"]["omp"]
    nx, ny = plan["config"]["nx"], plan["config"]["ny"]
    # T7 (2026-09-24): ceil(ranks/nodes), not floor -- see coldstart_sample.render_run_sbatch's
    # identical fix for the observed rc=213 'not enough slots' failure mode.
    ppn = -(-ranks // max(nodes, 1))
    assert ppn <= 72, ("ppn=%d exceeds CORES_PER_NODE=72 (ranks=%d nodes=%d)"
                       % (ppn, ranks, nodes))
    real_tpn = ppn
    max_dom = plan["max_dom"]
    verify_steps = int(plan.get("verify_steps") or 0)
    T = datetime.datetime.strptime(plan["span_start"], _WRF_TIME_FMT)
    T_end = datetime.datetime.strptime(plan["span_end"], _WRF_TIME_FMT)
    prefix = chosen.get("prefix", "GFS")
    vtable = chosen.get("vtable", "Vtable.GFS")
    interval_s = chosen.get("forcing_interval_h", plan["interval_seconds"] // 3600) * 3600
    expected_ungrib = cs.expected_wps_intermediate_files(prefix, T, T_end, interval_s)
    expected_met = cs.expected_met_em_files(max_dom, T, T_end, interval_s)
    geo_em_expected = ["geo_em.d%02d.nc" % d for d in range(1, max_dom + 1)]

    time_min = max(60, estimate_min * 2 + (30 if verify_steps else 0))
    hh, mm = time_min // 60, time_min % 60

    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("#SBATCH --job-name=case-rebuild")
    lines.append("#SBATCH --partition=compute")
    lines.append("#SBATCH --exclude=head")
    lines.append("#SBATCH -N %d --exclusive --ntasks-per-node=%d" % (nodes, ppn))
    lines.append("#SBATCH --time=%02d:%02d:00" % (hh, mm))
    lines.append("#SBATCH --signal=B:USR1@90")
    if out_dir_abs:
        lines.append("#SBATCH --output=%s/run-%%j.out" % out_dir_abs)
    lines.append("set -uo pipefail")
    lines.append("")
    lines.append("TOOL=%s" % cs._sh_quote(os.path.abspath(__file__)))
    lines.append("SITE=%s" % site)
    lines.append("CAMPAIGN=${CAMPAIGN:-$HOME/wtune/campaign}")
    lines.append("source \"$CAMPAIGN/sites/$SITE.sh\"   # site_env / site_launch / site_hygiene -- "
                  "reused, not reimplemented")
    lines.append("source ~/wtune-scc/env.sh")
    lines.append("export PATH=$NETCDF/bin:$PATH")
    lines.append("ulimit -s unlimited")
    lines.append("site_env")
    # T6 (2026-09-24, jobs 957/958): REBUILD_DIR, not OUT -- campaign/sites/scc.sh (sourced just
    # above, via site_env's own env.sh too) assigns its OWN global OUT=$HOME/wtune-runs/campaign,
    # so a rebuild-output OUT= set BEFORE the source line was silently clobbered, and every
    # downstream `cd "$OUT/wps"` landed in the wrong tree ('No such file or directory'). Grepped
    # every campaign/sites/*.sh: none of them assign REBUILD_DIR, TOOL, SITE, CAMPAIGN, or
    # ALLOC_NODES -- only OUT, CORES_PER_NODE, W, HARDWARE_ID, PIXI_ENV, CONV,
    # CASES_JSON_DEFAULT. Assigned AFTER sourcing (belt-and-suspenders: even if some future site
    # file's env.sh also touched it, this assignment already runs last).
    if out_dir_abs:
        lines.append("REBUILD_DIR=%s" % cs._sh_quote(out_dir_abs))
    else:
        lines.append("REBUILD_DIR=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)")
    lines.extend(cs.render_signal_trap_and_hygiene_preamble(
        nodes,
        extra_lines_after_alloc_nodes=["STATUS=\"$REBUILD_DIR/rebuild_status.txt\"",
                                      ": > \"$STATUS\""],
        record_label="STAGE"))
    lines.append("cd \"$REBUILD_DIR/wps\"")
    lines.append("export OMP_NUM_THREADS=1")
    lines.append("CUR_WINDOW=geogrid")
    lines.append("echo \"=== geogrid.exe ($(date -u +%FT%TZ)) ===\"")
    lines.append("mpirun -np %d --map-by ppr:%d:node:PE=1 --bind-to core ./geogrid.exe" % (ppn, ppn))
    lines.append("rc=$?; pkill -u \"$USER\" -x geogrid.exe 2>/dev/null")
    lines.append("GEO_EM_MISSING=()")
    lines.append("for f in %s; do [ -f \"$f\" ] || GEO_EM_MISSING+=(\"$f\"); done"
                  % " ".join(geo_em_expected))
    lines.append("if [ $rc -ne 0 ] || [ ${#GEO_EM_MISSING[@]} -gt 0 ] "
                  "|| ! grep -q 'Successful completion' geogrid.log* 2>/dev/null; then")
    lines.append("  echo \"FATAL geogrid rc=$rc missing=${GEO_EM_MISSING[*]:-none}\"; exit 1")
    lines.append("fi")
    lines.append("")
    # T9 (2026-09-24): see coldstart_sample.render_run_sbatch's identical fix -- link exactly
    # what `rebuild fetch` wrote (grib_manifest.txt) rather than assuming a *.grib2 extension
    # (FNL/.grib, ncei_g3-g4/.grb). Fall back to a widened glob only if the manifest is missing.
    lines.append("CUR_WINDOW=link_grib")
    lines.append("if [ -s \"$REBUILD_DIR/grib_manifest.txt\" ]; then")
    lines.append("  mapfile -t GRIB_FILES < <(sed \"s#^#$REBUILD_DIR/grib/#\" "
                  "\"$REBUILD_DIR/grib_manifest.txt\")")
    lines.append("else")
    lines.append("  shopt -s nullglob")
    lines.append("  GRIB_FILES=(\"$REBUILD_DIR\"/grib/*.grib2 \"$REBUILD_DIR\"/grib/*.grib "
                  "\"$REBUILD_DIR\"/grib/*.grb)")
    lines.append("  shopt -u nullglob")
    lines.append("fi")
    lines.append("if [ ${#GRIB_FILES[@]} -eq 0 ]; then")
    lines.append("  echo \"FATAL: no forcing files in $REBUILD_DIR/grib -- run 'case_rebuild.py "
                  "rebuild fetch --out $REBUILD_DIR' first\"; exit 1")
    lines.append("fi")
    lines.append("rm -f GRIBFILE.*")
    lines.append("./link_grib.csh \"${GRIB_FILES[@]}\"")
    lines.append("[ $? -ne 0 ] && { echo \"FATAL link_grib.csh failed\"; exit 1; }")
    lines.append("python3 \"$TOOL\" rebuild fix-span-wps-namelist --out \"$REBUILD_DIR\" || exit 1")
    lines.append("if [ -e %s ]; then ln -sf %s Vtable; else echo \"FATAL: %s not staged\"; exit 1; fi"
                  % (cs._sh_quote(vtable), cs._sh_quote(vtable), vtable))
    lines.append("rm -f \"%s:\"*" % prefix)
    lines.append("CUR_WINDOW=ungrib")
    lines.append("./ungrib.exe")
    lines.append("rc=$?; pkill -u \"$USER\" -x ungrib.exe 2>/dev/null")
    lines.append("UNGRIB_MISSING=()")
    lines.append("for f in %s; do [ -f \"$f\" ] || UNGRIB_MISSING+=(\"$f\"); done"
                  % " ".join(cs._sh_quote(x) for x in expected_ungrib))
    lines.append("[ $rc -ne 0 ] || [ ${#UNGRIB_MISSING[@]} -gt 0 ] && "
                  "{ echo \"FATAL ungrib rc=$rc missing=${UNGRIB_MISSING[*]:-none}\"; exit 1; }")
    lines.append("")
    lines.append("CUR_WINDOW=metgrid")
    lines.append("echo \"=== metgrid.exe ($(date -u +%FT%TZ)) ===\"")
    lines.append("mpirun -np %d --map-by ppr:%d:node:PE=1 --bind-to core ./metgrid.exe" % (ppn, ppn))
    lines.append("rc=$?; pkill -u \"$USER\" -x metgrid.exe 2>/dev/null")
    lines.append("MET_MISSING=()")
    lines.append("for f in %s; do [ -f \"$f\" ] || MET_MISSING+=(\"$f\"); done"
                  % " ".join(cs._sh_quote(x) for x in expected_met))
    lines.append("if [ $rc -ne 0 ] || [ ${#MET_MISSING[@]} -gt 0 ] "
                  "|| ! grep -q 'Successful completion' metgrid.log* 2>/dev/null; then")
    lines.append("  echo \"FATAL metgrid rc=$rc missing=${MET_MISSING[*]:-none}\"; exit 1")
    lines.append("fi")
    lines.append("")
    lines.append("CUR_WINDOW=real")
    lines.append("echo \"=== real.exe ($(date -u +%FT%TZ)) ===\"")
    lines.append("cd \"$REBUILD_DIR/real\"")
    lines.append("rm -f met_em.d0*.*.nc; ln -sf \"$REBUILD_DIR\"/wps/met_em.d0*.*.nc .")
    lines.append("python3 \"$TOOL\" rebuild fix-span-real-namelist --out \"$REBUILD_DIR\" || exit 1")
    lines.append("timeout 600 mpirun -np %d --map-by ppr:%d:node:PE=1 --bind-to core ./real.exe"
                  % (ranks, real_tpn))
    lines.append("rc=$?; pkill -u \"$USER\" -x real.exe 2>/dev/null")
    lines.append("if [ $rc -eq 124 ] || [ ! -f wrfinput_d01 ] || [ ! -f wrfbdy_d01 ]; then")
    lines.append("  echo \"real.exe decomposed attempt rc=$rc -- retrying serial\" >&2")
    lines.append("  python3 \"$TOOL\" rebuild fix-span-real-namelist --out \"$REBUILD_DIR\" "
                  "--serial || exit 1")
    lines.append("  timeout 1800 mpirun -np 1 --map-by ppr:1:node:PE=1 --bind-to core ./real.exe")
    lines.append("  rc=$?; pkill -u \"$USER\" -x real.exe 2>/dev/null")
    lines.append("fi")
    lines.append("[ $rc -ne 0 ] && { echo \"FATAL real.exe rc=$rc\"; exit 1; }")
    lines.append("[ -f wrfinput_d01 ] && [ -f wrfbdy_d01 ] || { echo \"FATAL: no wrfinput/wrfbdy\"; exit 1; }")
    lines.append("CUR_WINDOW=\"\"")
    if verify_steps > 0:
        lines.append("")
        lines.append("echo \"=== wrf.exe smoke run: %d steps ($(date -u +%%FT%%TZ)) ===\"" % verify_steps)
        lines.append("CUR_WINDOW=wrf")
        lines.append("cd \"$REBUILD_DIR/wrf\"")
        lines.append("ln -sf ../real/wrfinput_d0* .; ln -sf ../real/wrfbdy_d01 .")
        lines.append("shopt -s nullglob; WRFLOWINP=(../real/wrflowinp_d0*); shopt -u nullglob")
        lines.append("[ ${#WRFLOWINP[@]} -gt 0 ] && ln -sf ../real/wrflowinp_d0* .")
        lines.append("python3 \"$TOOL\" rebuild fix-span-wrf-namelist --out \"$REBUILD_DIR\" "
                      "--nproc-x %d --nproc-y %d --numtiles 1 || exit 1" % (nx, ny))
        lines.append("export OMP_NUM_THREADS=%d" % omp)
        lines.append("unset OMP_PLACES OMP_PROC_BIND")
        lines.append("site_launch %d %d %d block" % (ranks, ppn, omp))
        lines.append("timeout %d \"${SITE_LAUNCH_CMD[@]}\" > wrf.out 2>&1 &" % (estimate_min * 60))
        lines.append("WRF_PID=$!; wait \"$WRF_PID\"; rc=$?")
        lines.append("pkill -u \"$USER\" -x wrf.exe 2>/dev/null")
        lines.append("CUR_WINDOW=\"\"")
        lines.append("[ $rc -ne 0 ] && { echo \"FATAL wrf.exe smoke run rc=$rc\"; exit 1; }")
    lines.append("")
    lines.append("run_hygiene reap+census")
    lines.append("echo \"=== DONE ($(date -u +%FT%TZ)) ===\"")
    text = "\n".join(lines) + "\n"
    assert "LD_PRELOAD" not in text
    assert "--wrap" not in text
    return text


# ===========================================================================
# R6-B: compare -- similarity report (needs numpy + netCDF4, lazy-imported; grib pixi env only).
#
# Every metric below is a PURE function over plain arrays -- testable without ever opening a
# netCDF file. The netCDF-reading layer (read_wrfinput_fields) is a thin, separately-lazy
# wrapper that hands those arrays to the metric functions; it is NOT independently exercised by
# the offline test suite (no netCDF4 in this dev environment -- see tools/test_case_rebuild.py's
# own skip-if-missing guard), consistent with tools/test_grib_state.py's existing convention.
#
# PW (precipitable water) deviation from the literal spec: 'sum_k QVAPOR_k * dp_k / g ... dp from
# MU/C1H..C4H if present, else from P+PB differences -- document which'. Implemented here: ALWAYS
# the P+PB (full pressure) level-difference method, dp_k = p_full[k] - p_full[k+1] for
# k=0..nz-2 (i.e. nz-1 of the nz mass levels' own spacing; the top half-layer's contribution is
# dropped, a standard and small approximation since QVAPOR near the model top is close to zero).
# The MU/C1H..C4H hybrid-sigma path was NOT implemented: that formula needs the model's own eta
# level spacing (DNW or equivalent), which is not a plain per-column array in the same sense as
# P/PB/QVAPOR, and I have no real wrfout file locally to verify the Registry's exact variable
# relationship against -- guessing it risked a silently wrong physics constant. Alternatives, in
# case the P+PB method is insufficient: (1) as implemented; (2) implement the full MU/C1H..C4H
# path once a real wrfout is available to verify DNW/C1H/C4H's exact roles against; (3) drop PW
# from the scored metric set entirely. I chose (1).
# ===========================================================================

G_ACCEL = 9.81
RD_CP = 287.0 / 1004.5   # WRF's own R_d/c_p (dry gas constant / specific heat at const. pressure)
P0_PA = 100000.0         # WRF's reference pressure for the Poisson (theta -> T) relation


def rmse(a, b, mask=None):
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = (a - b)
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.nanmean(d ** 2)))


def max_abs_diff(a, b, mask=None):
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.abs(a - b)
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    return float(np.nanmax(d))


def agreement_pct(a, b, mask=None):
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    eq = (a == b)
    if mask is not None:
        eq = eq[np.asarray(mask, dtype=bool)]
    return float(100.0 * np.count_nonzero(eq) / eq.size) if eq.size else float("nan")


def bias(a, b, mask=None):
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    return float(np.nanmean(d))


def pattern_correlation(a, b, mask=None):
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


def compute_pw(qvapor, p, pb):
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


def theta_to_temp(theta_perturbation, p, pb, theta_base=300.0):
    """WRF's T variable is PERTURBATION potential temperature (theta = T_var + 300). Actual
    temperature via the Poisson relation: Temp = theta_full * (p_full/P0)^(Rd/cp)."""
    import numpy as np
    theta_full = np.asarray(theta_perturbation, dtype=float) + theta_base
    p_full = np.asarray(p, dtype=float) + np.asarray(pb, dtype=float)
    return theta_full * (p_full / P0_PA) ** RD_CP


def geopotential_height_mass_levels(ph, phb):
    """PH/PHB (perturbation + base-state geopotential) are on STAGGERED w-levels
    (bottom_top_stag, nz+1 interfaces). Z = (PH+PHB)/g at each interface; averaged to the nz
    MASS-level layer centers (same convention wrf-python/NCL use) so it lines up with P/PB for
    pressure-level interpolation. Returns [bottom_top, south_north, west_east]."""
    import numpy as np
    z_stag = (np.asarray(ph, dtype=float) + np.asarray(phb, dtype=float)) / G_ACCEL
    return (z_stag[:-1] + z_stag[1:]) / 2.0


def destagger_u(u):
    """west_east_stag (nx+1) -> mass points (nx): average adjacent pairs on the LAST axis."""
    import numpy as np
    u = np.asarray(u, dtype=float)
    return 0.5 * (u[..., :-1] + u[..., 1:])


def destagger_v(v):
    """south_north_stag (ny+1) -> mass points (ny): average adjacent pairs on AXIS 1 (the
    south_north axis of a [bottom_top, south_north, west_east] array)."""
    import numpy as np
    v = np.asarray(v, dtype=float)
    return 0.5 * (v[:, :-1, :] + v[:, 1:, :])


def rotate_grid_to_earth(u, v, cosalpha, sinalpha):
    """Grid-relative (u, v) -> earth-relative (u_e, v_e): u_e = u*cosalpha - v*sinalpha,
    v_e = v*cosalpha + u*sinalpha (T3, 2026-09-24). wrfinput's own U/V are GRID-relative (on the
    model's Lambert/polar/etc. projection); GFS/GRIB u/v -- what analog candidates are fetched
    and scored against -- are EARTH-relative, so this rotation is required before comparing them
    at all. cosalpha/sinalpha: WRF's own COSALPHA/SINALPHA state variables ('Local cosine/sine
    of map rotation', Registry.EM_COMMON:1420-1421) -- 2D [south_north, west_east], broadcasts
    against a 3D [bottom_top, south_north, west_east] u/v.

    VERIFICATION STATUS: this is the standard NCL/wrf-python `uvmet` diagnostic's own formula.
    I could NOT independently re-derive the sign convention from this checkout's own WRF Fortran
    source: COSALPHA/SINALPHA are only DECLARED there (Registry.EM_COMMON, and read/written by
    dyn_em/module_wps_io_arw.F's WPS I/O) -- no rotation formula appears in the dyn_em/share/
    frame source actually present in this environment (grep found no use site). Per the 'stop
    and report' rule: reporting this rather than guessing further; test_rotate_grid_to_earth_90_
    degrees is the independent sanity check this implementation leans on instead of a source
    citation."""
    import numpy as np
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cosalpha = np.asarray(cosalpha, dtype=float)
    sinalpha = np.asarray(sinalpha, dtype=float)
    u_e = u * cosalpha - v * sinalpha
    v_e = v * cosalpha + u * sinalpha
    return u_e, v_e


# T12 (2026-09-24): both moved to coldstart_sample.py -- vertical_grid_from_wrfinput may now
# ONLY be called from a login-node/PLAN-time path (see its new docstring there: the batch job
# runs under /usr/bin/python3 with no numpy, which is exactly why job 965's rebuild silently
# skipped T8). coldstart_sample.build_plan needs both these too (coldstart's own real.exe
# windows hit the same matthew-style FATAL), and case_rebuild imports coldstart_sample, not the
# reverse -- aliased here so `cr.vertical_grid_from_wrfinput`/`cr.remove_namelist_key` keep
# working unchanged for every existing call site and test.
vertical_grid_from_wrfinput = cs.vertical_grid_from_wrfinput
remove_namelist_key = cs.remove_namelist_key


def target_fields_from_wrfinput(path):
    """R6-C target state (T3, 2026-09-24 fix): built DIRECTLY from the case's own wrfinput_d01,
    NEVER by fetching GRIB at the case's stated start date -- for a fictional/relabelled case
    (exactly the case `analog` exists to handle) the stated date's real weather is precisely NOT
    this case's own weather, so using it as the target would make every candidate score against
    the wrong answer. Uses the SAME functions compare() uses: hgt500 = geopotential height (PH+
    PHB, mass-level-averaged) at 500 hPa; tmp850 = theta_to_temp at 850 hPa; ugrd850/vgrd850 = U/V
    destaggered to mass points and ROTATED grid-relative -> earth-relative (rotate_grid_to_earth
    -- GRIB u/v, what candidates are scored against, are earth-relative; wrfinput's own U/V are
    not); pwat = compute_pw; tmp2m = T2 directly. All on the wrfinput's own (case) grid -- no
    regridding needed for the target itself. Points where 850/500 hPa is below ground are NaN
    (interp_pressure_level_field's own convention)."""
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


def interp_pressure_level_column(field_k, p_k, target_pa):
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


def interp_pressure_level_field(field_3d, p_3d, target_pa):
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


def grid_identity_check(a, b, tol_deg=0.01):
    """Ports .route1-ref/grid_identity_gate.py's own gate exactly (attrs dict with tolerance
    0.01 deg for floats/exact for others, dims exact, corner+center lat/lon within tol_deg).
    a, b: {'attrs': {...}, 'dims': {'west_east':.., 'south_north':.., 'bottom_top':..},
    'corners': {'SW':(lat,lon), 'SE':.., 'NW':.., 'NE':.., 'center':..}}.
    Returns (ok, reasons)."""
    reasons = []
    # T10 (2026-09-24, matthew MAP_PROJ=3 Mercator): projection-aware attribute comparison --
    # WRF only fills in the attributes a given projection actually USES; an unused one is left
    # at whatever value happened to be in the file (matthew's original wrfinput carries a
    # leftover TRUELAT2=60 from some earlier edit, the freshly rebuilt one has WRF's own "unset"
    # sentinel 1e20 -- both are meaningless for Mercator, but comparing TRUELAT2 unconditionally
    # refused an otherwise-identical grid on 'TRUELAT2 differs'). Group membership (WRF's own
    # set_ps/set_lc/set_merc/set_cyl projection setup): TRUELAT1 -> Lambert(1)/polar
    # stereo(2)/Mercator(3); TRUELAT2 -> Lambert(1) only; STAND_LON -> Lambert(1)/polar
    # stereo(2)/lat-lon-rotated(6); POLE_LAT/POLE_LON -> lat-lon-rotated(6) only. MAP_PROJ
    # itself, DX/DY, dims, CEN_LAT/CEN_LON and the corner check are unconditional -- those apply
    # to every projection.
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


DEFAULT_STATIC_RMSE_THRESH_M = 50.0
DEFAULT_LU_AGREE_THRESH_PCT = 90.0
DEFAULT_CORR_Z500_THRESH = 0.9
DEFAULT_CORR_T850_THRESH = 0.8
DEFAULT_CORR_PW_THRESH = 0.7


def compare_verdict(terrain_rmse_m, lu_agreement_pct, corr_z500, corr_t850, corr_pw,
                    static_rmse_thresh=DEFAULT_STATIC_RMSE_THRESH_M,
                    lu_agree_thresh=DEFAULT_LU_AGREE_THRESH_PCT,
                    corr_z500_thresh=DEFAULT_CORR_Z500_THRESH,
                    corr_t850_thresh=DEFAULT_CORR_T850_THRESH,
                    corr_pw_thresh=DEFAULT_CORR_PW_THRESH):
    """(verdict, static_ok, state_ok). verdict in {'consistent', 'different-static-data',
    'state-mismatch'}: consistent = static_ok & state_ok; different-static-data =
    (not static_ok) & state_ok; state-mismatch = not state_ok (the stated date's real weather is
    not this case's weather -- fictional/relabelled date, or a different event) -- this
    dominates: even a static mismatch alongside a state mismatch is reported as state-mismatch,
    since that is the more actionable/severe finding."""
    static_ok = (terrain_rmse_m < static_rmse_thresh) and (lu_agreement_pct >= lu_agree_thresh)
    state_ok = (corr_z500 >= corr_z500_thresh and corr_t850 >= corr_t850_thresh
               and corr_pw >= corr_pw_thresh)
    if not state_ok:
        verdict = "state-mismatch"
    elif not static_ok:
        verdict = "different-static-data"
    else:
        verdict = "consistent"
    return verdict, static_ok, state_ok


def read_wrfinput_fields_netcdf4(path):
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


def compare(original_path, rebuilt_path, static_rmse_thresh=DEFAULT_STATIC_RMSE_THRESH_M,
           lu_agree_thresh=DEFAULT_LU_AGREE_THRESH_PCT, corr_z500_thresh=DEFAULT_CORR_Z500_THRESH,
           corr_t850_thresh=DEFAULT_CORR_T850_THRESH, corr_pw_thresh=DEFAULT_CORR_PW_THRESH):
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
    verdict, static_ok, state_ok = compare_verdict(
        static["terrain_rmse_m"], static["lu_index_agreement_pct"], corr_z500, corr_t850, corr_pw,
        static_rmse_thresh=static_rmse_thresh, lu_agree_thresh=lu_agree_thresh,
        corr_z500_thresh=corr_z500_thresh, corr_t850_thresh=corr_t850_thresh,
        corr_pw_thresh=corr_pw_thresh)

    return {
        "static": static, "state": state, "verdict": verdict,
        "static_ok": static_ok, "state_ok": state_ok,
        "thresholds": {"static_rmse_thresh_m": static_rmse_thresh,
                       "lu_agree_thresh_pct": lu_agree_thresh,
                       "corr_z500_thresh": corr_z500_thresh, "corr_t850_thresh": corr_t850_thresh,
                       "corr_pw_thresh": corr_pw_thresh},
        "note": "PW uses the P+PB level-difference dp method (MU/C1H..C4H not implemented -- "
                "see module docstring)",
    }


# ===========================================================================
# R6-C: analog -- find a real date whose weather matches the case
#
# The scoring/candidate-date math below is PURE (numpy arrays / stdlib datetime only) and fully
# offline-testable. The actual per-candidate GRIB fetch (metdata's state-purpose byte-range +
# bbox crop, via coldstart_sample's lazy _metdata()/_grib_state() accessors, threaded with
# --workers) is a thin orchestration layer -- best-effort, NOT independently exercised by the
# offline test suite (would need real network), same disclosed-limitation pattern every prior
# round's fetch/download orchestration has used.
# ===========================================================================

ANALOG_FIELD_KEYS = ("hgt500", "tmp850", "ugrd850", "vgrd850", "pwat", "tmp2m")


def doy_of(dt):
    return dt.timetuple().tm_yday


def candidate_dates(stated_start, years, doy_window, hour="same"):
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


def block_average(field2d, block=8, min_blocks=10):
    """Block-average `field2d` down to a coarse grid (~block x block cells per block), but never
    below `min_blocks` x `min_blocks` blocks -- shrinks the block size (never below 1) when the
    nominal block size would produce fewer blocks than that floor in either dimension.
    NaN-aware (T3, 2026-09-24): 850/500 hPa points below ground are NaN (see
    target_fields_from_wrfinput/interp_pressure_level_field) and must not poison an entire
    block's average -- uses nanmean per block; a block that is ALL NaN stays NaN (there is
    nothing to average), not silently dropped or zeroed."""
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


def anomaly_pattern_corr_nrmse(candidate_field, target_field):
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


def field_score(corr, nrmse):
    """field score = 0.5*(1-corr) + 0.5*min(nrmse,2)/2 -- 0 is a perfect match, larger is worse."""
    if corr != corr:   # NaN
        return float("inf")
    return 0.5 * (1.0 - corr) + 0.5 * min(nrmse, 2.0) / 2.0


def score_candidate(candidate_fields, target_fields):
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


def rank_candidates(candidates_scored):
    """candidates_scored: [(date, score_dict), ...]. Returns the SAME list sorted by ascending
    total score (lower = better match) -- ties broken by date for determinism."""
    return sorted(candidates_scored, key=lambda ds: (ds[1]["total"], ds[0]))


def two_stage_select(ranked, top_n=10):
    """The best `top_n` candidates (by t0 score) that go on to the trajectory (boundary) scoring
    stage -- SPEC item R6-C: 'only for the best 10 by the t0 score -- two-stage'."""
    return ranked[:top_n]


def combine_t0_and_trajectory(t0_total, trajectory_total):
    """total = 0.5*t0 + 0.5*trajectory when a boundary trajectory was scored."""
    return 0.5 * t0_total + 0.5 * trajectory_total


def quality_flag(t0_field_corrs):
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


def bdy_strip_means(u, v, t, side):
    """Mean of U, V, T on one of wrfbdy_d01's 4 boundary strips. wrfbdy_d01 stores each side's
    strip SEPARATELY, named <VAR>_B<SIDE><S|E> (S=start i.e. the initial-time slab, E=end i.e.
    the tendency slab) for side in {XS (west), XE (east), YS (south), YE (north)} -- e.g.
    U_BXS/U_BXE/U_BYS/U_BYE (VERIFIED convention from the WRF Registry's boundary variable
    naming; see module docstring's stop-and-report rule if a real wrfbdy_d01 contradicts this).
    `u`/`v`/`t` here are already the SLICED strip arrays for one side/one bdy time (caller reads
    them from the file) -- this function only takes the mean."""
    import numpy as np
    out = {}
    for name, arr in (("U", u), ("V", v), ("T", t)):
        if arr is not None:
            out[name] = float(np.nanmean(np.asarray(arr, dtype=float)))
    return out


def cmd_analog_offline_rank(target_fields, candidates, top_n_trajectory=10):
    """Pure orchestration over already-fetched/regridded arrays (no network): scores every
    candidate's t0 fields, ranks them, and returns (ranked, kept_for_trajectory) -- the top
    `top_n_trajectory` by t0 score. `candidates`: [(date, {field_key: 2D array}), ...]."""
    scored = [(date, score_candidate(fields, target_fields)) for date, fields in candidates]
    ranked = rank_candidates(scored)
    kept = two_stage_select(ranked, top_n=top_n_trajectory)
    return ranked, kept


def build_analog_output(ranked, quality, trajectory_scores=None):
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


def cmd_analog(case_dir, out_dir, wrfinput=None, wrfbdy=None, years="2021-2025", doy_window=45,
               hour="same", top=5, workers=8, namelist=None):
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
        # No target: climatological analog -- same DOY/hour, most recent year with data.
        chosen_dt = max(candidates_dt) if candidates_dt else None
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

    # Target-based scoring: needs a live GRIB fetch per candidate date (grib pixi env only --
    # numpy+netCDF4+xarray+cfgrib+requests). This orchestration is BEST-EFFORT: real network,
    # not exercised by the offline test suite (same disclosed-limitation pattern every prior
    # round's fetch/download orchestration used) -- the SCORING math it feeds
    # (score_candidate/rank_candidates/two_stage_select/quality_flag/candidate_dates/
    # block_average) is separately unit-tested against synthetic arrays.
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

    # T3 (2026-09-24): the target comes from the case's OWN wrfinput_d01, NEVER from fetching
    # GRIB at the case's stated start date -- for a fictional/relabelled case that IS the wrong
    # weather by definition (that mismatch is exactly what `analog`/`compare` are for). See
    # target_fields_from_wrfinput's own docstring.
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

    # Boundary-trajectory stage: deliberately DISABLED, not merely unverified -- the wrfbdy
    # *_BXS/*_BXE/*_BYS/*_BYE/*_BTXS variable naming (bdy_strip_means) IS now verified against a
    # real conus12 wrfbdy_d01, but the trajectory scoring pass itself (fetching each of the top
    # `top` t0 candidates' own weather at every bdy interval over the run, scoring strip means,
    # combine_t0_and_trajectory) is not wired up. Leaving it off rather than shipping a
    # half-implemented stage; analog.json says so explicitly (T3, 2026-09-24) so nobody reads a
    # t0-only ranking as having used the trajectory.
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

def main(argv=None):
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
    sp.add_argument("--ranks", type=int, default=72)
    sp.add_argument("--omp", type=int, default=1)
    sp.add_argument("--nx", type=int)
    sp.add_argument("--ny", type=int)
    sp.add_argument("--ncdump")
    sp.add_argument("--geog-path")
    sp.add_argument("--grib-dir")

    sp = rebuild_sub.add_parser("fetch")
    sp.add_argument("--out", required=True)
    sp.add_argument("--cache-dir")
    sp.add_argument("--grib-dir")
    sp.add_argument("--keep-archives", action="store_true")

    sp = rebuild_sub.add_parser("stage")
    sp.add_argument("--out", required=True)
    sp.add_argument("--site", default="scc")

    sp = rebuild_sub.add_parser("finalize")
    sp.add_argument("--out", required=True)

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

    sp = sub.add_parser("compare")
    sp.add_argument("--original", required=True)
    sp.add_argument("--rebuilt", required=True)
    sp.add_argument("--json")
    sp.add_argument("--static-rmse-thresh", type=float, default=DEFAULT_STATIC_RMSE_THRESH_M)
    sp.add_argument("--lu-agree-thresh", type=float, default=DEFAULT_LU_AGREE_THRESH_PCT)
    sp.add_argument("--corr-z500-thresh", type=float, default=DEFAULT_CORR_Z500_THRESH)
    sp.add_argument("--corr-t850-thresh", type=float, default=DEFAULT_CORR_T850_THRESH)
    sp.add_argument("--corr-pw-thresh", type=float, default=DEFAULT_CORR_PW_THRESH)

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
                                 grib_dir=args.grib_dir)
                return 0
            if args.rebuild_cmd == "fetch":
                cmd_rebuild_fetch(args.out, cache_dir=args.cache_dir, grib_dir=args.grib_dir,
                                  keep_archives=args.keep_archives)
                return 0
            if args.rebuild_cmd == "stage":
                cmd_rebuild_stage(args.out, site=args.site)
                return 0
            if args.rebuild_cmd == "finalize":
                cmd_rebuild_finalize(args.out)
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

        if args.cmd == "compare":
            result = compare(args.original, args.rebuilt,
                             static_rmse_thresh=args.static_rmse_thresh,
                             lu_agree_thresh=args.lu_agree_thresh,
                             corr_z500_thresh=args.corr_z500_thresh,
                             corr_t850_thresh=args.corr_t850_thresh,
                             corr_pw_thresh=args.corr_pw_thresh)
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
