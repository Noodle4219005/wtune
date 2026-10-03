"""post/recipe.py -- `wtune post recipe RUN_DIR [RUN_DIR ...] --out DIR` (decision 0057):
detects which case type(s) this run matches (post/caseinfo.py, multi-label -- when types
overlap, every matching type's figures are produced, never just one), dispatches to each
detected type's figure set (post/recipes/*.py), deduplicates figures that only need drawing
once (the domain map), and builds one self-contained HTML report (post/report.py) whose header
lists the detected types and the exact evidence for each.

Every figure record this module's fig()/missing() helpers produce is one of:
  fig(id, path, run, domain=None, valid_time=None, extra=None)      -- a real figure
  missing(id, dataset, reason, run=None)                            -- an explicit gap, never
                                                                        a silently-dropped one
`post.recipes.*`'s own figures() functions import fig/missing from THIS module lazily (inside
their own functions), not at their top level, to avoid a circular import with this module's own
top-level `from post.recipes import ...` dispatch imports.
"""
from __future__ import annotations

import json
import os

from post import caseinfo, tools

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes", "catalog.json")


def fig(id, path, run, domain=None, valid_time=None, extra=None):
    d = dict(id=id, missing=False, path=path, run=run, domain=domain, valid_time=valid_time)
    if extra:
        d.update(extra)
    return d


def note(source, text, level="warning"):
    """A user-facing hint that is neither a figure nor a missing observation (e.g. "the track is
    on a 12 km grid"). Any recipe module may include these in the list its figures()/handler
    returns; run_recipe() pulls them out (they never reach the figure list, so they are not
    counted as missing observations) into the top-level result["notes"] list of
    {"source", "level", "text"}, which the report renders as a warning box. A figure's own
    extra["note"] stays on the figure."""
    return dict(kind="note", source=source, level=level, text=text)


def is_note(entry):
    return entry.get("kind") == "note"


def missing(id, dataset, reason, run=None):
    return dict(id=id, missing=True, dataset=dataset, reason=reason, run=run)


def load_catalog():
    with open(CATALOG_PATH) as fh:
        return json.load(fh)


def _classify_tc_path(path):
    """post/tc.py's own figure filenames (its module, not this one) -> a catalog.json id.
    Matched by substring, not an exact list, since post/tc.py is a concurrently-developed
    module this one only calls, never edits -- keeping this permissive means a filename tweak
    on that side degrades to the generic "tc_figure" caption instead of raising here."""
    name = os.path.basename(path).lower()
    if "track_map" in name:
        return "tc_track_map"
    if "track_error" in name or "tk_err" in name:
        return "tc_track_error"
    if any(s in name for s in ("amax_wind", "amslp", "vmax", "mslp", "intensity")):
        return "tc_intensity_error"
    if any(s in name for s in ("r34", "r50", "r64", "wind_radii", "radii_error")):
        return "tc_wind_radii_error"
    if "rmw" in name:
        return "tc_rmw_structure"
    return "tc_figure"


def _tc_figures(run_dirs, labels, out_dir, case_info, hurdat2_path, storm_id, bdeck_path,
                ebtrk_path, paraview, spinup_hours, track_smooth_km=None):
    from post import tc as tc_mod
    from post.recipes import tc_extra

    tc_out = os.path.join(out_dir, "tc")
    summary = tc_mod.run_tc_report(run_dirs, labels, hurdat2_path, storm_id, tc_out,
                                   ebtrk_path=ebtrk_path, bdeck_path=bdeck_path,
                                   **({} if track_smooth_km is None else {"smooth_km": track_smooth_km}))
    out = []
    tracking_note = " ".join(summary.get("notes", []))
    for path in summary["figures"]:
        kind = _classify_tc_path(path)
        out.append(fig(kind, path, ",".join(labels),
                       extra={"note": tracking_note} if tracking_note and kind == "tc_track_map"
                       else None))
    for w in summary["warnings"]:
        out.append(missing("tc_figure", "MET tc_pairs/tc_stat/tc_rmw or a b-deck", w))
    for n in summary.get("notes", []):
        out.append(note("tc", n))
    for label, st in summary.get("track_steps", {}).items():
        out.append(note("tc", tc_mod.format_track_steps(label, st), level="info"))

    for run_dir, label in zip(run_dirs, labels):
        adeck_path = summary["adecks"].get(label)
        if not adeck_path:
            continue
        domain = case_info["domains"][0]["domain"]
        out += tc_extra.figures(run_dir, label, adeck_path, domain, tc_out)

    # the 3-D render moved to the cloud set (whole domain) and tc_structure (storm-centred
    # cutaway); a tc_cloud_3d here would draw the cloud set's d01 view a second time
    return out


def _convection_figures(run_dirs, labels, out_dir, case_info, paraview, spinup_hours):
    from post.recipes import convection
    conv_out = os.path.join(out_dir, "convection")
    os.makedirs(conv_out, exist_ok=True)
    out = convection.figures(run_dirs, labels, conv_out, case_info, spinup_hours=spinup_hours)
    # the 3-D render is the cloud set's (d01 + finest nest), not repeated here
    return out


def _tc_structure_figures(run_dirs, labels, out_dir, case_info, paraview=True, every=None,
                          times=None, use_track=True):
    """post/recipes/tc_structure.py on run_dirs[0]: centres from the track post/tc.py just
    wrote (tc/track_domains_<label>.csv) when use_track, else its own smoothed-SLP minimum."""
    from post.recipes import tc_structure
    track_csv = os.path.join(out_dir, "tc", "track_domains_%s.csv" % labels[0])
    return tc_structure.figures(run_dirs[0], labels[0], os.path.join(out_dir, "tc_structure"),
                                case_info,
                                track_csv=track_csv if use_track and os.path.exists(track_csv)
                                else None, every=every, times=times, paraview=paraview)


def _cloud_figures(run_dirs, labels, out_dir, case_info, paraview=True, every=None, times=None):
    from post.recipes import cloud
    return cloud.figures(run_dirs, labels, os.path.join(out_dir, "cloud"), case_info,
                         every=every, times=times, paraview=paraview)


TYPE_DISPATCH = {
    "tc": _tc_figures,
    "convection": _convection_figures,
    "cloud": _cloud_figures,
}


def _simple_dispatch(module_name, out_subdir):
    def run(run_dirs, labels, out_dir, case_info, spinup_hours=6, cross_section=None,
           tc_centre=None, xsec_top_km=None, xsec_halfwidth_km=None, project_dir=None,
           spinup_days=None, **_ignored):
        import inspect
        mod = __import__("post.recipes.%s" % module_name, fromlist=["figures"])
        sub_out = os.path.join(out_dir, out_subdir)
        os.makedirs(sub_out, exist_ok=True)
        params = inspect.signature(mod.figures).parameters
        kwargs = {}
        if "spinup_hours" in params:
            kwargs["spinup_hours"] = spinup_hours
        if "cross_section" in params:
            kwargs["cross_section"] = cross_section
        if "tc_centre" in params:
            kwargs["tc_centre"] = tc_centre
        if "xsec_top_km" in params:
            kwargs["xsec_top_km"] = xsec_top_km
        if "xsec_halfwidth_km" in params:
            kwargs["xsec_halfwidth_km"] = xsec_halfwidth_km
        if "project_dir" in params:     # --project must reach the module from any cwd
            kwargs["project_dir"] = project_dir
        if "spinup_days" in params and spinup_days is not None:
            kwargs["spinup_days"] = spinup_days
        return guarded(module_name, mod.figures, run_dirs, labels, sub_out, case_info, **kwargs)
    return run


def guarded(name, fn, *args, **kwargs):
    """fn(*args, **kwargs); a wrfout written with a reduced history stream (iofields) that
    lacks a variable/attribute the figure set needs (post.extract.is_missing_field_error)
    becomes ONE missing entry naming it -- never an aborted report, never a raw
    AttributeError/KeyError text."""
    from post.extract import is_missing_field_error, missing_reason
    try:
        return fn(*args, **kwargs)
    except Exception as e:                                  # noqa: BLE001 -- filtered below
        if not is_missing_field_error(e):
            raise
        return [missing("%s_figures" % name, "wrfout history stream",
                        "%s -- the %s figure set needs it" % (missing_reason(e), name))]


for _name in ("precip", "synoptic", "pbl", "chem", "climate", "ensemble", "terrain", "moisture"):
    TYPE_DISPATCH[_name] = _simple_dispatch(_name, _name)
del _name


def _tc_centre_from_track(hurdat2_path, storm_id, bdeck_path, run_dir):
    """Best-track fix nearest the run's last available time, for moisture's default
    cross-section point (decision 0057 update: "剖面線預設過颶風中心 ... 或由 project.yaml
    指定"). None when no --hurdat2/--bdeck was given at all -- the moisture recipe then falls
    back to its own max-rain-point default."""
    from post import caseinfo as _caseinfo
    points = _caseinfo._best_track_points(hurdat2_path, storm_id, bdeck_path)
    if not points:
        return None
    from post.extract import list_frames
    frames = list_frames(run_dir, domain=1)
    if not frames:
        return None
    last_time = frames[-1][0]
    from datetime import datetime
    target = datetime.strptime(last_time, _caseinfo.WRF_TIME_FMT)
    nearest = min(points, key=lambda p: abs((p[0] - target).total_seconds()))
    return nearest[1], nearest[2]


def run_recipe(run_dirs, out_dir, **kwargs):
    """ADIOS2 BP history first (post.extract.prepare_history): every run's BP directories are
    converted to (or validated in) the netCDF cache -- <project>/_nc_cache, else
    <out_dir>/_nc_cache, unless the CLI/--nc-cache/$WTUNE_NC_CACHE chose already -- before any
    figure set looks at a frame. A run whose BP history cannot be converted at all (adios2
    missing) gives a report with ONE missing entry naming the fix instead of a traceback from
    whichever recipe touched a frame first; single files that failed to convert become notes.
    Then _run_recipe (the pipeline below)."""
    from post import extract
    labels = kwargs.get("labels") or [("run%d" % i) for i in range(len(run_dirs))]
    with extract.nc_cache(base=kwargs.get("project_dir") or out_dir):
        gaps, bp_notes = [], []
        for run_dir, label in zip(run_dirs, labels):
            try:
                prep = extract.prepare_history(run_dir)
            except extract.BPConversionError as e:
                gaps.append(missing("common_history_netcdf", "ADIOS2 BP history -> netCDF",
                                    str(e), run=label))
                continue
            for bp, err in prep["failed"]:
                bp_notes.append(note("history", "%s: %s not converted to netCDF, its frames "
                                     "are left out: %s" % (label, bp, err)))
        if gaps:
            os.makedirs(out_dir, exist_ok=True)
            from post.report import build_recipe_report
            case_info = dict(types=["common"], evidence={}, run_length_h=0.0,
                             d01_extent_km=(0.0, 0.0), domains=[])
            report_path = build_recipe_report(
                out_dir, case_info, gaps, run_dirs, labels, kwargs.get("spinup_hours", 6),
                load_catalog(), title=kwargs.get("title"),
                notes=[{k: n[k] for k in ("source", "level", "text")} for n in bp_notes])
            return dict(out_dir=out_dir, case_info=case_info, figures=gaps,
                        notes=bp_notes, report_path=report_path, advice=None)
        return _run_recipe(run_dirs, out_dir, pre_notes=bp_notes, **kwargs)


def _run_recipe(run_dirs, out_dir, types_override=None, ensemble=False, spinup_hours=6,
                hurdat2_path=None, storm_id=None, bdeck_path=None, ebtrk_path=None,
                labels=None, paraview=True, decomposition_only=False, title=None,
                every=None, times=None, cross_sections=None, animate=False,
                xsec_top_km=None, xsec_halfwidth_km=None, project_dir=None, spinup_days=None,
                track_smooth_km=None, pre_notes=None):
    """Full `wtune post recipe` pipeline. Returns {"out_dir", "case_info", "figures",
    "notes", "report_path"}. notes = [{"source", "level", "text"}] hints collected from any recipe
    module (note()); they are not figures and not missing observations.

    every/times (decision 0058 P3): when either is given, an additional per-valid-time full
    figure set (post/timeseries.py) plus consecutive/vs-t0 change maps (post/changemap.py) are
    produced on the finest domain of run_dirs[0], on top of the type-driven recipe figures
    above; they also set the cadence of the always-on CF pressure-level NetCDF data product
    (post/recipes/cf3d.py) and of the VAPOR bundle (post/recipes/vapor_bundle.py); last frame
    only when neither is given. animate: also builds one MP4 (post/anim.py) per time-series
    field when 2+ times were selected. cross_sections: optional {"terrain": {...},
    "moisture": {...}} line
    overrides (project.yaml's cross_sections.*, decision 0057 update). xsec_top_km/
    xsec_halfwidth_km: project.yaml's own xsec_top_km/xsec_halfwidth_km (2026-09-29 second
    lead review) -- the terrain/moisture cross-section figures' height window and
    centred-line half-width; None lets each recipe module use its own default. project_dir: the
    --project directory, handed to every recipe module whose figures() takes `project_dir`. spinup_days: likewise handed to modules whose figures() takes
    `spinup_days` (climate); None keeps that module's own default. track_smooth_km: the TC
    tracker's SLP smoothing radius (post/tc.py; None = its default 30, 0 = raw)."""
    from post.recipes import common as common_mod
    from post.recipes import vapor_bundle as vapor_bundle_mod

    os.makedirs(out_dir, exist_ok=True)
    labels = labels or [("run%d" % i) for i in range(len(run_dirs))]
    if len(labels) != len(run_dirs):
        raise ValueError("labels must have one entry per RUN_DIR")

    case_info = caseinfo.detect_types(run_dirs, types_override=types_override, ensemble=ensemble,
                                      hurdat2_path=hurdat2_path, storm_id=storm_id,
                                      bdeck_path=bdeck_path)
    types = case_info["types"]
    cross_sections = cross_sections or {}

    all_figures = list(pre_notes or [])     # note() entries from run_recipe's BP conversion
    all_figures += common_mod.figures(run_dirs, labels, out_dir, case_info,
                                      decomposition_only=decomposition_only)
    all_figures += _cf3d_data(run_dirs[0], labels[0], out_dir, case_info, every, times)
    all_figures += vapor_bundle_mod.figures(run_dirs[0], labels[0], out_dir, case_info,
                                            every=every, times=times)

    tc_centre = None
    if "moisture" in types and (hurdat2_path or bdeck_path):
        try:
            tc_centre = _tc_centre_from_track(hurdat2_path, storm_id, bdeck_path, run_dirs[0])
        except Exception:
            tc_centre = None

    for t in types:
        if t in ("common",):
            continue
        handler = TYPE_DISPATCH.get(t)
        if handler is None:
            continue
        if t == "tc":
            has_track = bool(hurdat2_path and storm_id)
            if not has_track:
                all_figures.append(missing("tc_figure", "HURDAT2/--storm",
                                           "tc was detected but --hurdat2/--storm is required "
                                           "to build the a-deck/b-deck comparison"))
            else:
                all_figures += guarded("tc", handler, run_dirs, labels, out_dir, case_info,
                                       hurdat2_path, storm_id, bdeck_path, ebtrk_path, paraview,
                                       spinup_hours, track_smooth_km=track_smooth_km)
            all_figures += guarded("tc_structure", _tc_structure_figures, run_dirs, labels,
                                   out_dir, case_info, paraview=paraview, every=every,
                                   times=times, use_track=has_track)
        elif t == "cloud":
            all_figures += guarded("cloud", handler, run_dirs, labels, out_dir, case_info,
                                   paraview=paraview, every=every, times=times)
        elif t == "convection":
            all_figures += guarded("convection", handler, run_dirs, labels, out_dir, case_info,
                                   paraview, spinup_hours)
        elif t == "terrain":
            all_figures += handler(run_dirs, labels, out_dir, case_info,
                                   spinup_hours=spinup_hours,
                                   cross_section=cross_sections.get("terrain"),
                                   xsec_top_km=xsec_top_km, xsec_halfwidth_km=xsec_halfwidth_km,
                                   project_dir=project_dir)
        elif t == "moisture":
            all_figures += handler(run_dirs, labels, out_dir, case_info,
                                   spinup_hours=spinup_hours,
                                   cross_section=cross_sections.get("moisture"),
                                   tc_centre=tc_centre,
                                   xsec_top_km=xsec_top_km, xsec_halfwidth_km=xsec_halfwidth_km,
                                   project_dir=project_dir)
        else:
            all_figures += handler(run_dirs, labels, out_dir, case_info, spinup_hours=spinup_hours,
                                   project_dir=project_dir, spinup_days=spinup_days)

    if every or times:
        all_figures += guarded("timeseries", _time_series_figures, run_dirs, labels, out_dir,
                               case_info, every, times, animate=animate)

    notes = [{k: e[k] for k in ("source", "level", "text")} for e in all_figures if is_note(e)]
    all_figures = [e for e in all_figures if not is_note(e)]

    advice = _verification_advice(run_dirs[0], out_dir, hurdat2_path, storm_id, bdeck_path, notes)

    import inspect
    from post.report import build_recipe_report
    kwargs = {"notes": notes} if "notes" in inspect.signature(build_recipe_report).parameters else {}
    if advice and advice["advice"]:
        import wtune_advise
        kwargs["advice_html"] = wtune_advise.render_html(advice)
    report_path = build_recipe_report(out_dir, case_info, all_figures, run_dirs, labels,
                                      spinup_hours, load_catalog(), title=title, **kwargs)
    return dict(out_dir=out_dir, case_info=case_info, figures=all_figures, notes=notes,
                report_path=report_path, advice=advice)


def _verification_advice(run_dir, out_dir, hurdat2_path, storm_id, bdeck_path, notes):
    """Decision 0061: verification-triggered advice (rules with a post_* clause) for the report.
    Advice-only; a failure here becomes a report note, never a failed recipe."""
    try:
        import wtune_advise
        case = wtune_advise.build_case(run_dir, hurdat2=hurdat2_path, storm=storm_id,
                                       bdeck=bdeck_path, post_tc_dir=os.path.join(out_dir, "tc"))
        advice = wtune_advise.advise(case, surface="post")
        with open(os.path.join(out_dir, "advice.json"), "w") as fh:
            json.dump(advice, fh, indent=2, ensure_ascii=False, default=str)
        return advice
    except Exception as e:                                  # noqa: BLE001
        notes.append(note("advice", "advice skipped: %s: %s" % (type(e).__name__, e),
                          level="info"))
        return None


def _cf3d_data(run_dir, run_label, out_dir, case_info, every, times):
    """Always-on CF pressure-level NetCDF data product (post/recipes/cf3d.py) for run_dirs[0],
    at the --every/--times cadence (last frame only without either). Data-file records, not
    figures; any failure becomes a missing entry, never a failed report."""
    from post.recipes import cf3d
    try:
        bdy_width = caseinfo.spec_bdy_width(run_dir)
    except caseinfo.CaseInfoError:
        bdy_width = caseinfo.DEFAULT_SPEC_BDY_WIDTH
    return cf3d.figures(run_dir, run_label, out_dir, case_info, bdy_width, every=every,
                        times=times)


def _time_series_figures(run_dirs, labels, out_dir, case_info, every, times, animate=False):
    from post import timeseries
    from post.changemap import figures_for_change_maps

    run_dir, run_label = run_dirs[0], labels[0]
    domain = case_info["domains"][-1]["domain"]
    ts_out = os.path.join(out_dir, "timeseries")
    os.makedirs(ts_out, exist_ok=True)

    include_mcape = "convection" in case_info["types"]
    out = timeseries.figures_for_time_series(run_dir, run_label, domain, ts_out, every=every,
                                             times=times, include_mcape=include_mcape)

    ok_figs = [f for f in out if not f.get("missing")]
    selected_times = sorted({f["valid_time"] for f in ok_figs})
    if selected_times:
        out += figures_for_change_maps(run_dir, run_label, domain, ts_out, selected_times)

    if animate and len(selected_times) >= 2:
        out += _time_series_animations(run_dir, run_label, domain, ts_out, ok_figs)
    return out


def _time_series_animations(run_dir, run_label, domain, ts_out, ok_figs):
    from post import anim
    from post import tools as _tools

    by_field = {}
    for f in ok_figs:
        if not f["id"].startswith("timeseries_"):
            continue
        field = f["id"][len("timeseries_"):]
        by_field.setdefault(field, []).append(f)

    out = []
    anim_dir = os.path.join(ts_out, "anim")
    os.makedirs(anim_dir, exist_ok=True)
    for field, entries in by_field.items():
        entries.sort(key=lambda f: f["valid_time"])
        pngs = [e["path"] for e in entries]
        out_path = os.path.join(anim_dir, "%s_d%02d.mp4" % (field, domain))
        try:
            written = anim.build_animation(pngs, out_path, fps=2)
            out.append(fig("anim_%s" % field, written[0], run_label, domain="d%02d" % domain))
        except (_tools.ToolNotFoundError, _tools.ToolError, ValueError) as e:
            out.append(missing("anim_%s" % field, "ffmpeg", str(e), run=run_label))
    return out
