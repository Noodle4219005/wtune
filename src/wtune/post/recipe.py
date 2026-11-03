"""post/recipe.py -- `wtune post recipe RUN_DIR... --out DIR` (decision 0057): detect the case
type(s) (post/caseinfo.py, multi-label), run each type's figure set (post/recipes/*.py), dedupe
once-only figures and build one HTML report (post/report.py). Records are fig(...) (a figure),
missing(...) (an explicit gap) and note(...) (a hint; moved to result["notes"]). Recipe modules
import fig/missing/note lazily to avoid a circular import with the dispatch imports below.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

from wtune.post import caseinfo

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes", "catalog.json")
FIGURE_EXTS = (".png", ".gif", ".svg", ".mp4")


def fig(id, path, run, domain=None, valid_time=None, extra=None) -> dict:
    d = dict(id=id, missing=False, path=path, run=run, domain=domain, valid_time=valid_time)
    if extra:
        d.update(extra)
    return d


def note(source, text, level="warning") -> dict:
    """A user-facing hint that is neither a figure nor a missing observation (e.g. "the track is
    on a 12 km grid"); run_recipe() moves these out of the figure list into result["notes"]."""
    return dict(kind="note", source=source, level=level, text=text)


def is_note(entry) -> bool:
    return entry.get("kind") == "note"


def missing(id, dataset, reason, run=None) -> dict:
    return dict(id=id, missing=True, dataset=dataset, reason=reason, run=run)


def load_catalog() -> dict:
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
    from wtune.post import tc as tc_mod
    from wtune.post.recipes import tc_extra

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


def _convection_figures(run_dirs, labels, out_dir, case_info, paraview, spinup_hours,
                        focus_box=None):
    from wtune.post.recipes import convection
    conv_out = os.path.join(out_dir, "convection")
    os.makedirs(conv_out, exist_ok=True)
    out = convection.figures(run_dirs, labels, conv_out, case_info, spinup_hours=spinup_hours,
                             focus_box=focus_box)
    # the 3-D render is the cloud set's (d01 + finest nest), not repeated here
    return out


def _tc_structure_figures(run_dirs, labels, out_dir, case_info, paraview=True, every=None,
                          times=None, use_track=True, spinup_hours=0):
    """post/recipes/tc_structure.py on run_dirs[0]: centres from the track post/tc.py just
    wrote (tc/track_domains_<label>.csv) when use_track, else its own smoothed-SLP minimum."""
    from wtune.post.recipes import tc_structure
    track_csv = os.path.join(out_dir, "tc", "track_domains_%s.csv" % labels[0])
    return tc_structure.figures(run_dirs[0], labels[0], os.path.join(out_dir, "tc_structure"),
                                case_info,
                                track_csv=track_csv if use_track and os.path.exists(track_csv)
                                else None, every=every, times=times, paraview=paraview,
                                spinup_hours=spinup_hours)


def _cloud_figures(run_dirs, labels, out_dir, case_info, paraview=True, every=None, times=None,
                   map_time=None, focus_box=None):
    from wtune.post.recipes import cloud
    return cloud.figures(run_dirs, labels, os.path.join(out_dir, "cloud"), case_info,
                         every=every, times=times, paraview=paraview, map_time=map_time,
                         focus_box=focus_box)


TYPE_DISPATCH = {
    "tc": _tc_figures,
    "convection": _convection_figures,
    "cloud": _cloud_figures,
}


def _simple_dispatch(module_name, out_subdir):
    def run(run_dirs, labels, out_dir, case_info, spinup_hours=6, cross_section=None,
           tc_centre=None, xsec_top_km=None, xsec_halfwidth_km=None, project_dir=None,
           spinup_days=None, focus_box=None, map_time=None, **_ignored):
        import inspect
        mod = __import__("wtune.post.recipes.%s" % module_name, fromlist=["figures"])
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
        if "focus_box" in params:
            kwargs["focus_box"] = focus_box
        if "map_time" in params:
            kwargs["map_time"] = map_time
        if "hovmoller_line" in params:
            kwargs["hovmoller_line"] = _ignored.get("hovmoller_line")
        return guarded(module_name, mod.figures, run_dirs, labels, sub_out, case_info, **kwargs)
    return run


def guarded(name, fn, *args, **kwargs) -> list:
    """fn(*args, **kwargs); a wrfout written with a reduced history stream (iofields) that
    lacks a variable/attribute the figure set needs (post.extract.is_missing_field_error)
    becomes ONE missing entry naming it -- never an aborted report, never a raw
    AttributeError/KeyError text."""
    from wtune.post.extract import is_missing_field_error, missing_reason
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


def _map_time(run_dir, case_info, spinup_hours, focus_box):
    """Valid time of the one-time maps: the rain peak on the finest domain (post/focus.py), or
    None (each figure set's last frame) when there is no post-spin-up rain or no frame."""
    from wtune.post.focus import rain_focus
    domain = case_info["domains"][-1]["domain"] if case_info.get("domains") else 1
    try:
        f = rain_focus(run_dir, domain, spinup_hours, box=focus_box)
    except Exception as e:                                  # noqa: BLE001
        print("post recipe: no rain peak on d%02d (%s: %s); one-time maps at the last frame"
              % (domain, type(e).__name__, e))
        return None
    if not f or f["peak_rate_mm_h"] <= 0:
        return None
    print("post recipe: one-time maps at the d%02d rain peak %s (%s)"
          % (domain, f["peak_time"], f["rule"]))
    return f["peak_time"]


def _tc_centre_from_track(hurdat2_path, storm_id, bdeck_path, run_dir):
    """Best-track fix nearest the run's last available time, for moisture's default
    cross-section point (decision 0057 update: "剖面線預設過颶風中心 ... 或由 project.yaml
    指定"). None when no --hurdat2/--bdeck was given at all -- the moisture recipe then falls
    back to its own max-rain-point default."""
    from wtune.post import caseinfo as _caseinfo
    points = _caseinfo._best_track_points(hurdat2_path, storm_id, bdeck_path)
    if not points:
        return None
    from wtune.post.extract import list_frames
    frames = list_frames(run_dir, domain=1)
    if not frames:
        return None
    last_time = frames[-1][0]
    from datetime import datetime
    target = datetime.strptime(last_time, _caseinfo.WRF_TIME_FMT)
    nearest = min(points, key=lambda p: abs((p[0] - target).total_seconds()))
    return nearest[1], nearest[2]


def run_recipe(run_dirs, out_dir, **kwargs) -> dict:
    """Convert/validate ADIOS2 BP history into the netCDF cache first (post.extract.prepare_history),
    then _run_recipe. A run whose BP history cannot be converted (adios2 missing) gives a report
    with one missing entry naming the fix; single files that failed become notes."""
    from wtune.post import extract
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
            for path, why in prep.get("bad", []):
                bp_notes.append(note("history", "%s: %s left out (%s)" % (label, path, why)))
        if gaps:
            os.makedirs(out_dir, exist_ok=True)
            from wtune.post.report import build_recipe_report
            case_info = dict(types=["common"], evidence={}, run_length_h=0.0,
                             d01_extent_km=(0.0, 0.0), domains=[])
            report_path = build_recipe_report(
                out_dir, case_info, gaps, run_dirs, labels, kwargs.get("spinup_hours", 6),
                load_catalog(), title=kwargs.get("title"),
                notes=[{k: n[k] for k in ("source", "level", "text")} for n in bp_notes])
            return dict(out_dir=out_dir, case_info=case_info, figures=gaps,
                        notes=bp_notes, report_path=report_path, advice=None)
        return _run_recipe(run_dirs, out_dir, pre_notes=bp_notes, **kwargs)


@dataclass
class _RecipeRun:
    """Inputs shared by every type runner of one `wtune post recipe` run."""
    run_dirs: list
    labels: list
    out_dir: str
    case_info: dict
    spinup_hours: float = 6
    hurdat2_path: str | None = None
    storm_id: str | None = None
    bdeck_path: str | None = None
    ebtrk_path: str | None = None
    paraview: bool = True
    every: str | None = None
    times: list | None = None
    cross_sections: dict = field(default_factory=dict)
    xsec_top_km: float | None = None
    xsec_halfwidth_km: float | None = None
    project_dir: str | None = None
    spinup_days: float | None = None
    track_smooth_km: float | None = None
    tc_centre: tuple | None = None
    focus_box: list | None = None
    map_time: str | None = None


def _args4(r: _RecipeRun) -> tuple:
    return r.run_dirs, r.labels, r.out_dir, r.case_info


def _run_tc(handler, r: _RecipeRun) -> list:
    has_track = bool(r.storm_id and (r.hurdat2_path or r.bdeck_path))
    out = []
    if not has_track:
        out.append(missing("tc_figure", "HURDAT2 or b-deck, and --storm",
                           "tc was detected but --storm with --hurdat2 or --bdeck is required "
                           "to build the a-deck/b-deck comparison"))
    else:
        out += guarded("tc", handler, *_args4(r), r.hurdat2_path, r.storm_id, r.bdeck_path,
                       r.ebtrk_path, r.paraview, r.spinup_hours,
                       track_smooth_km=r.track_smooth_km)
    out += guarded("tc_structure", _tc_structure_figures, *_args4(r), paraview=r.paraview,
                   every=r.every, times=r.times, use_track=has_track,
                   spinup_hours=r.spinup_hours)
    return out


def _run_cloud(handler, r: _RecipeRun) -> list:
    return guarded("cloud", handler, *_args4(r), paraview=r.paraview, every=r.every,
                   times=r.times, map_time=r.map_time, focus_box=r.focus_box)


def _run_convection(handler, r: _RecipeRun) -> list:
    return guarded("convection", handler, *_args4(r), r.paraview, r.spinup_hours,
                   focus_box=r.focus_box)


def _run_terrain(handler, r: _RecipeRun) -> list:
    return handler(*_args4(r), spinup_hours=r.spinup_hours,
                   cross_section=r.cross_sections.get("terrain"),
                   xsec_top_km=r.xsec_top_km, xsec_halfwidth_km=r.xsec_halfwidth_km,
                   project_dir=r.project_dir, map_time=r.map_time, focus_box=r.focus_box)


def _run_moisture(handler, r: _RecipeRun) -> list:
    return handler(*_args4(r), spinup_hours=r.spinup_hours,
                   cross_section=r.cross_sections.get("moisture"), tc_centre=r.tc_centre,
                   xsec_top_km=r.xsec_top_km, xsec_halfwidth_km=r.xsec_halfwidth_km,
                   project_dir=r.project_dir, focus_box=r.focus_box, map_time=r.map_time)


def _run_simple(handler, r: _RecipeRun) -> list:
    return handler(*_args4(r), spinup_hours=r.spinup_hours, project_dir=r.project_dir,
                   spinup_days=r.spinup_days, map_time=r.map_time)


def _run_precip(handler, r: _RecipeRun) -> list:
    return handler(*_args4(r), spinup_hours=r.spinup_hours, project_dir=r.project_dir,
                   spinup_days=r.spinup_days, map_time=r.map_time, focus_box=r.focus_box,
                   hovmoller_line=r.cross_sections.get("hovmoller"))


_TYPE_RUNNERS = {"tc": _run_tc, "cloud": _run_cloud, "convection": _run_convection,
                 "terrain": _run_terrain, "moisture": _run_moisture, "precip": _run_precip}


def _fail_soft(name, fn, *args, **kwargs) -> list:
    """fn(*args, **kwargs); any Exception (e.g. TCError "no track found", a missing external
    tool) becomes ONE missing entry for that figure set, so the report is still written with
    the figures already produced. KeyboardInterrupt/SystemExit propagate."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:                                  # noqa: BLE001
        return [missing("%s_figures" % name, "%s figure set" % name,
                        "%s: %s" % (type(e).__name__, e))]


def _post_workers(n_tasks: int) -> int:
    """Processes for the figure sets: WTUNE_POST_WORKERS, else one per CPU this job may use."""
    from wtune.post.extract import _cpu_budget
    env = os.environ.get("WTUNE_POST_WORKERS", "").strip()
    return max(1, min(int(env) if env else _cpu_budget(), n_tasks))


def _in_parallel(tasks: list) -> list:
    """[(name, fn, args, kwargs)] -> their entries, concatenated in task order. Each figure set
    only reads the history and writes its own subdirectory, so each runs in its own process; a
    worker process that dies becomes one missing entry per unfinished set."""
    workers = _post_workers(len(tasks))
    if workers <= 1:
        return [e for _name, fn, args, kw in tasks for e in fn(*args, **kw)]
    out = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("fork")) as ex:
        futures = [(name, ex.submit(fn, *args, **kw)) for name, fn, args, kw in tasks]
        for name, fut in futures:
            try:
                out += fut.result()
            except BrokenProcessPool as e:
                out.append(missing("%s_figures" % name, "%s figure set" % name,
                                   "worker process died: %s" % e))
    return out


def _one_type(t: str, r: _RecipeRun) -> list:
    """One type's figures. A worker looks the handler up by name: the dispatch closures do
    not pickle."""
    return _fail_soft(t, _TYPE_RUNNERS.get(t, _run_simple), TYPE_DISPATCH[t], r)


def _type_tasks(types: list, r: _RecipeRun) -> list:
    return [(t, _one_type, (t, r), {}) for t in types
            if t != "common" and t in TYPE_DISPATCH]


def _type_figures(types: list, r: _RecipeRun) -> list:
    """Figures of every detected type that has a handler, in detection order."""
    return _in_parallel(_type_tasks(types, r))


def _data_products(run_dir, label, out_dir, case_info, every, times) -> list:
    """The CF pressure-level file, then the VAPOR bundle (one task: their order is kept)."""
    from wtune.post.recipes import vapor_bundle as vapor_bundle_mod
    return (_cf3d_data(run_dir, label, out_dir, case_info, every, times)
            + vapor_bundle_mod.figures(run_dir, label, out_dir, case_info,
                                       every=every, times=times))


def _split_notes(entries: list) -> tuple[list, list]:
    """(notes as {source, level, text}, the remaining figure/missing entries)."""
    notes = [{k: e[k] for k in ("source", "level", "text")} for e in entries if is_note(e)]
    return notes, [e for e in entries if not is_note(e)]


def _build_report(run_dirs, labels, out_dir, case_info, figures, spinup_hours, title, notes, advice):
    import inspect
    from wtune.post.report import build_recipe_report
    kwargs = {"notes": notes} if "notes" in inspect.signature(build_recipe_report).parameters else {}
    if advice and advice["advice"]:
        import wtune.advise
        kwargs["advice_html"] = wtune.advise.render_html(advice)
    return build_recipe_report(out_dir, case_info, figures, run_dirs, labels, spinup_hours,
                               load_catalog(), title=title, **kwargs)


def _run_recipe(run_dirs, out_dir, types_override=None, ensemble=False, spinup_hours=6,
                hurdat2_path=None, storm_id=None, bdeck_path=None, ebtrk_path=None,
                labels=None, paraview=True, decomposition_only=False, title=None,
                every=None, times=None, cross_sections=None, animate=False,
                xsec_top_km=None, xsec_halfwidth_km=None, project_dir=None, spinup_days=None,
                track_smooth_km=None, pre_notes=None, focus_box=None) -> dict:
    """The `wtune post recipe` pipeline: detect types, run each type's figure set, add the CF/VAPOR
    data products and (with every/times) the time series, split notes from figures, build the HTML
    report. Returns {out_dir, case_info, figures, notes, report_path, advice}. Option meanings:
    docs/ post section (decisions 0057/0058); None keeps each recipe module's own default."""
    from wtune.post.recipes import common as common_mod

    os.makedirs(out_dir, exist_ok=True)
    labels = labels or [("run%d" % i) for i in range(len(run_dirs))]
    if len(labels) != len(run_dirs):
        raise ValueError("labels must have one entry per RUN_DIR")

    case_info = caseinfo.detect_types(run_dirs, types_override=types_override, ensemble=ensemble,
                                      hurdat2_path=hurdat2_path, storm_id=storm_id,
                                      bdeck_path=bdeck_path)
    types = case_info["types"]

    tc_centre = None
    if "moisture" in types and (hurdat2_path or bdeck_path):
        try:
            tc_centre = _tc_centre_from_track(hurdat2_path, storm_id, bdeck_path, run_dirs[0])
        except Exception:
            tc_centre = None

    run = _RecipeRun(
        run_dirs, labels, out_dir, case_info, spinup_hours=spinup_hours,
        hurdat2_path=hurdat2_path, storm_id=storm_id, bdeck_path=bdeck_path,
        ebtrk_path=ebtrk_path, paraview=paraview, every=every, times=times,
        cross_sections=cross_sections or {}, xsec_top_km=xsec_top_km,
        xsec_halfwidth_km=xsec_halfwidth_km, project_dir=project_dir, spinup_days=spinup_days,
        track_smooth_km=track_smooth_km, tc_centre=tc_centre, focus_box=focus_box,
        map_time=_map_time(run_dirs[0], case_info, spinup_hours, focus_box))
    tasks = [("common", _fail_soft, ("common", common_mod.figures, run_dirs, labels, out_dir,
                                     case_info), dict(decomposition_only=decomposition_only)),
             ("data", _data_products, (run_dirs[0], labels[0], out_dir, case_info, every, times),
              {})]
    tasks += _type_tasks(types, run)
    if "pbl" not in types:              # the pbl set draws its own radiosonde comparison
        from wtune.post.recipes import pbl as pbl_mod
        tasks.append(("soundings", _fail_soft,
                      ("common_sounding", pbl_mod.sounding_figures, run_dirs, labels,
                       os.path.join(out_dir, "soundings"), case_info),
                      dict(project_dir=project_dir, map_time=run.map_time)))
    if every or times:
        tasks.append(("timeseries", _fail_soft,
                      ("timeseries", guarded, "timeseries", _time_series_figures, run_dirs,
                       labels, out_dir, case_info, every, times), dict(animate=animate)))
    # note() entries from run_recipe's BP conversion first
    all_figures = list(pre_notes or []) + _in_parallel(tasks)

    notes, all_figures = _split_notes(all_figures)
    advice = _verification_advice(run_dirs[0], out_dir, hurdat2_path, storm_id, bdeck_path, notes)
    report_path = _build_report(run_dirs, labels, out_dir, case_info, all_figures, spinup_hours,
                                title, notes, advice)
    write_index(out_dir, run_dirs, labels, case_info, all_figures, notes)
    return dict(out_dir=out_dir, case_info=case_info, figures=all_figures, notes=notes,
                report_path=report_path, advice=advice,
                figures_dir=figures_folder(out_dir, all_figures))


INDEX_FILE = "figures.json"
INDEX_SCHEMA = "wtune-post-figures/1"


def write_index(out_dir, run_dirs, labels, case_info, figures, notes, kind="recipe") -> str:
    """out_dir/figures.json: what the recipe (kind "recipe") or `wtune post preview` (kind
    "preview") drew and could not draw, with paths relative to out_dir -- the machine-readable
    half of report.html that `wtune post brief` and `wtune post check` read (decision 0078)."""
    def rel(entry):
        e = dict(entry)
        if e.get("path") and os.path.isabs(e["path"]):
            e["path"] = os.path.relpath(e["path"], out_dir)
        return e
    doc = dict(schema=INDEX_SCHEMA, kind=kind, runs=[dict(label=lb, path=os.path.abspath(d))
                                          for d, lb in zip(run_dirs, labels)],
               case_info=case_info, figures=[rel(f) for f in figures], notes=notes)
    path = os.path.join(out_dir, INDEX_FILE)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False, default=str)
    return path


def figures_folder(out_dir, figures) -> str:
    """Every drawn figure again in one flat folder out_dir/figures/, named <type folder>-<file>
    (hard links, a copy where linking fails); rebuilt on each call. Returns the folder."""
    dest = os.path.join(out_dir, "figures")
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    taken = set()
    for f in figures:
        path = f.get("path")
        if f.get("missing") or not path or not path.endswith(FIGURE_EXTS) \
                or not os.path.isfile(path):
            continue
        parts = os.path.relpath(path, out_dir).split(os.sep)
        name = parts[-1] if len(parts) == 1 else "%s-%s" % (parts[0], parts[-1])
        stem, ext = os.path.splitext(name)
        n = 1
        while name in taken:
            n += 1
            name = "%s-%d%s" % (stem, n, ext)
        taken.add(name)
        try:
            os.link(path, os.path.join(dest, name))
        except OSError:
            shutil.copy2(path, os.path.join(dest, name))
    return dest


def _verification_advice(run_dir, out_dir, hurdat2_path, storm_id, bdeck_path, notes):
    """Decision 0061: verification-triggered advice (rules with a post_* clause) for the report.
    Advice-only; a failure here becomes a report note, never a failed recipe."""
    try:
        import wtune.advise
        case = wtune.advise.build_case(run_dir, hurdat2=hurdat2_path, storm=storm_id,
                                       bdeck=bdeck_path, post_tc_dir=os.path.join(out_dir, "tc"))
        advice = wtune.advise.advise(case, surface="post")
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
    from wtune.post.recipes import cf3d
    try:
        bdy_width = caseinfo.spec_bdy_width(run_dir)
    except caseinfo.CaseInfoError:
        bdy_width = caseinfo.DEFAULT_SPEC_BDY_WIDTH
    return cf3d.figures(run_dir, run_label, out_dir, case_info, bdy_width, every=every,
                        times=times)


def _time_series_figures(run_dirs, labels, out_dir, case_info, every, times, animate=False):
    from wtune.post import timeseries
    from wtune.post.changemap import figures_for_change_maps

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
    from wtune.post import anim
    from wtune.post import tools as _tools

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
