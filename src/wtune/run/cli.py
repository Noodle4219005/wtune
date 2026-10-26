"""`wtune run CASE_DIR ... --out OUT [--dry-run]`: parse the options, plan the run (case,
layout, wall model, inputs, segments), print the plan, then hand it to the controller.
CASE_DIR is never modified; exit codes: 0 done, 1 a segment failed, 2 nothing submitted."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import shlex
import sys
import time
from typing import Callable, Optional

import wtune.extensions
import wtune.machine
import wtune.stopping
import wtune.watch
from ._deps import _CAMPAIGN_DIR, coldstart, sweep, wp
from .common import WRF_FMT, RunError, RunStopped, _fmt, _hms, parse_hms
from .context import RunContext, RunOptions, RunPlan, Runners, default_scancel
from .controller import _make_sidecar, run_controller
from .estimate import (FALLBACK_RADIATION_FACTOR, Overheads, constant_compute_fn,
                       fallback_overheads, fallback_s_per_step, model_from_prediction)
from .geometry import Geometry, case_geometry, domain_start, resolve_end_step, step_time
from .attempts import wait_for_job
from .inputs import (InputPlan, _logging_copier, convert_inputs, convert_inputs_in_job,
                     find_project_dir, io_plan_line, plan_inputs)
from .layout import (Candidate, Config, Layout, TuneLayout, make_config, rank_mapping_notes,
                     resolve_layout, tune_candidates, tune_report_layout)
from .namelist import _aux_outputs, namelist_diff_summary, render_segment_namelist
from .plan import (RestartSetting, Segment, case_restart_interval_s, checkpoint_request,
                   checkpoint_steps_for_minutes, missing_record_files, plan_run, restart_setting,
                   restart_times)
from .report import _write_report, new_report
from .space import estimate_output_bytes, input_bytes, writable_bytes
from .staging import _missing_namelist_files, restart_files

DESCRIPTION = "wtune run -- segmented WRF executor with restart hand-offs (`wtune run`)."


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wtune run", description=DESCRIPTION)
    ap.add_argument("case_dir", nargs="?", default=None,
                    help="default: the project's case/ (then the newest tune/*/tune_report.json "
                         "is used unless --tune-report or --nodes is given)")
    ap.add_argument("--nodes", type=int, default=None,
                    help="required unless --tune-report names a winner layout (flags override it)")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--numtiles", type=int, default=None, help="default: tune report, else 1")
    ap.add_argument("--omp", type=int, default=None, help="default: tune report, else 1")
    ap.add_argument("--wall-limit", default=None, help="HH:MM:SS per segment job")
    ap.add_argument("--segment-minutes", type=int, default=None)
    ap.add_argument("--stop-at", default=None, help="YYYY-MM-DD_HH:MM:SS (on a whole time step)")
    ap.add_argument("--prediction", default=None, help="pred_rebuild.json (wtune predict-rebuild)")
    ap.add_argument("--tune-report", default=None,
                    help="predicted-mode tune_report.json: per segment the lowest-predicted "
                         "candidate is run")
    ap.add_argument("--restart-io-form", type=int, choices=(14, 2), default=14)
    ap.add_argument("--io", choices=("adios2", "netcdf"), default="adios2",
                    help="adios2 (default, 2026-10-03 decision): every stream -- history, "
                         "restart, input, boundary, aux inputs -- is ADIOS2 (io_form 14); "
                         "netCDF inputs are converted once into the project's tune cache "
                         "<project>/input/bp-cache/<case>/<key>/ (shared with wtune tune; no "
                         "project: OUT/_bp_inputs/) and linked as <name>.bp. netcdf: LEGACY "
                         "escape hatch -- the case's own io_form_* are kept (restart still per "
                         "--restart-io-form)")
    ap.add_argument("--cfl-rescue", action="store_true",
                    help="re-run a CFL/NaN-failed segment once with 0.75 dt (marks PHYSICS-GRAY). "
                         "This is the ONLY user-pre-authorized exception to decision 0056's "
                         "advisory-only rule for solution-changing knobs -- without it, a CFL/NaN "
                         "failure stops and writes advice (recommended time_step + a "
                         "segment.advice.namelist.input) instead of resubmitting anything.")
    ap.add_argument("--resume", type=int, default=1, metavar="N",
                    help="auto-resume an infrastructure-failed segment job from its latest "
                         "valid restart, at most N times per run (default 1; 0 disables)")
    ap.add_argument("--checkpoint-minutes", default="auto", metavar="M",
                    help="restart every M model minutes for --resume, when the case's own "
                         "restart_interval puts none inside a segment. Default auto: the "
                         "smallest multiple of the input record step >= run length / 8 (only "
                         "with --resume > 0); 0 writes none")
    ap.add_argument("--sidecar", action="store_true",
                    help="while a segment job runs, record each valid restart and convert "
                         "finished history frames to netCDF in the `wtune post` cache, on the "
                         "job's idle cores")
    ap.add_argument("--compress-history", action="store_true",
                    help="--sidecar, and each converted history frame replaces its BP directory "
                         "(zlib netCDF, about 4x smaller, checked against the BP first; on a "
                         "full node one core is shared at the lowest priority); of WTune's own "
                         "restarts only the newest valid set is kept, none once the run has "
                         "succeeded")
    ap.add_argument("--wrfda-forecast-only", action="store_true",
                    help="acknowledge that this WRFDA case is run as wrf.exe forecast segments "
                         "only: WTune does NOT run DA cycles (obsproc/da_wrfvar/da_update_bc). "
                         "Without it, a WRFDA case is refused.")
    ap.add_argument("--out", default=None, help="required unless --project (or $WTUNE_PROJECT "
                    "/ a project.yaml above cwd) resolves, in which case it defaults to "
                    "<project>/runs/<id>/ (decision 0058)")
    ap.add_argument("--project", default=None, help="project dir (decision 0058) -- see --out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wrf-tree", default=None, help="default $WRF_TREE (run/ files are linked "
                                                     "by the job too)")
    ap.add_argument("--env-source", default=wtune.machine.env_source())
    ap.add_argument("--campaign-dir", default=_CAMPAIGN_DIR)
    ap.add_argument("--sacct-cmd", default="sacct")
    ap.add_argument("--poll-interval", type=float, default=60.0)
    ap.add_argument("--stall-minutes", type=float, default=20.0, metavar="M",
                    help="cancel a running segment job whose rsl.error.0000 has not changed "
                         "for M minutes and stop the run (0 disables; default 20)")
    ap.add_argument("--thompson-cache", action="store_true",
                    help="link the Thompson qr_acr_q*/freezeH2O tables from the per-build cache "
                         "and add the ones a completed segment computed (saves ~70 s per "
                         "segment on scc, nothing measurable on f1-arm); without it a segment "
                         "deletes the tables WRF computed (~357 MiB)")
    ap.add_argument("--no-space-check", action="store_true",
                    help="submit even when the estimated output does not fit the free space "
                         "(GPFS quota headroom when the run dir is on GPFS)")
    return ap


def parse_options(argv: Optional[list] = None) -> RunOptions:
    return RunOptions(**vars(build_arg_parser().parse_args(argv)))


def _project_tune_report(project_dir: str, log: Callable) -> Optional[str]:
    """The newest <project>/tune/*/tune_report.json made for the project's registered case
    (none registered: the newest of all)."""
    import wtune.project
    case_id = wtune.project.project_case_id(wtune.project.load_project(project_dir))
    return wtune.project.latest_tune_report(project_dir, case_id, log)


def apply_project_defaults(opts: RunOptions, log: Callable) -> RunOptions:
    """No CASE_DIR: the project's case/. Running the project's own case with neither
    --tune-report nor --nodes: its newest tune report."""
    import wtune.project
    try:
        project_dir = wtune.project.find_project(explicit=opts.project)
    except wtune.project.ProjectError as e:
        raise RunError(str(e)) from None
    if opts.case_dir is None:
        if not project_dir or not os.path.isdir(os.path.join(project_dir, "case")):
            raise RunError("CASE_DIR is required outside a project (wtune project init --case)")
        opts = dataclasses.replace(opts, case_dir=os.path.join(project_dir, "case"))
    own_case = project_dir and os.path.realpath(opts.case_dir) == os.path.realpath(
        os.path.join(project_dir, "case"))
    if own_case and opts.tune_report is None and opts.nodes is None:
        report = _project_tune_report(project_dir, log)
        if report:
            log("tune report: %s (newest in %s)" % (report, os.path.join(project_dir, "tune")))
            opts = dataclasses.replace(opts, tune_report=report)
    return opts


# ---------------------------------------------------------------------------
# planning steps, in the order their errors and notes appear
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Case:
    """The case as read: its namelist (text and parsed), geometry and the step it ends at."""
    case_dir: str
    text: str
    nl: dict
    geo: Geometry
    n_end: int


@dataclasses.dataclass(frozen=True)
class LayoutChoice:
    """The run's layout and where it came from."""
    lay: Layout
    config: Config
    report: Optional[dict]             # the tune_report.json, when given
    winner: Optional[TuneLayout]       # its winner layout
    overridden: list                   # CLI flags that differ from the winner


def _resolve_out(opts: RunOptions) -> str:
    """--out, else <project>/runs/<id>/ (decision 0058)."""
    if opts.out:
        return os.path.abspath(opts.out)
    import wtune.project
    project_dir = wtune.project.find_project(explicit=opts.project)
    if not project_dir:
        raise RunError("--out is required (or --project DIR / $WTUNE_PROJECT for a "
                       "default runs/<id>/ -- decision 0058)")
    return wtune.project.default_out_dir(project_dir, "runs")


def case_restarts(nl: dict) -> bool:
    """restart = .true.: segment 0 starts from the case's own wrfrst_dNN_<start>."""
    return wp.g(nl, "time_control", "restart", False) is True


def restart_io_form_of_case(opts: RunOptions, case_dir: str, log: Callable) -> RunOptions:
    """A restart case: io_form_restart is the format of its wrfrst (a BP directory: 14, else
    2), since WRF reads and writes restarts with the one io_form."""
    nl = wp.parse_namelist(os.path.join(case_dir, "namelist.input"))
    if not case_restarts(nl):
        return opts
    found = restart_files(case_dir, _fmt(case_geometry(nl).start), 1)[1]
    form = 14 if found and os.path.isdir(found[0]) else 2
    if form != opts.restart_io_form:
        log("restart case: io_form_restart %d (the format of %s)" % (form, found[0] if found
                                                                      else "its wrfrst"))
    return dataclasses.replace(opts, restart_io_form=form)


def _load_case(opts: RunOptions, case_dir: str, out: str) -> Case:
    """Read the case and refuse what segments cannot handle."""
    if os.path.commonpath([out, case_dir]) == case_dir:
        raise RunError("--out must be outside CASE_DIR (the case is never modified)")
    if not opts.dry_run and os.path.isdir(os.path.join(out, "seg_0")):
        raise RunError("%s already holds a run (seg_0): choose another --out" % out)
    nml = os.path.join(case_dir, "namelist.input")
    with open(nml) as f:
        text = f.read()
    nl = wp.parse_namelist(nml)
    for key in ("rst_outname", "rst_inname"):
        if coldstart.count_key(text, key):
            raise RunError("%s is set: segments assume the default wrfrst_d<domain>_<date> "
                           "names" % key)
    geo = case_geometry(nl)
    for d in range(1, geo.max_dom):
        ds = domain_start(nl, d)
        if ds is not None and ds != geo.start:
            raise RunError("d%02d starts at %s, not with d01 (%s): staggered nest starts are "
                           "not supported" % (d + 1, _fmt(ds), _fmt(geo.start)))
    if case_restarts(nl):
        found = restart_files(case_dir, _fmt(geo.start), geo.max_dom)
        missing = ["wrfrst_d%02d_%s" % (d, _fmt(geo.start)) for d, p in found.items() if not p]
        if missing:
            raise RunError("restart = .true. but %s not in %s" % (", ".join(missing), case_dir))
    return Case(case_dir, text, nl, geo, resolve_end_step(geo, opts.stop_at))


def _choose_layout(opts: RunOptions, log: Callable) -> LayoutChoice:
    """CLI flags over --tune-report's winner; logs the report's layout line."""
    rep, winner = None, None
    if opts.tune_report:
        with open(opts.tune_report) as f:
            rep = json.load(f)
        winner = tune_report_layout(rep)
    lay, overridden = resolve_layout(opts, winner)
    if winner is not None:
        log("layout from tune report: N=%d %dx%d tiles %d omp %d (cores/node %s, %s s/step)%s"
            % (winner.nodes, winner.nx, winner.ny, winner.numtiles, winner.omp,
               winner.cores_per_node if winner.cores_per_node is not None else "?",
               "%.4f" % winner.median if winner.median is not None else "?",
               "; CLI overrides %s -> N=%d %dx%d tiles %d omp %d"
               % (" ".join(overridden), lay.nodes, lay.nx, lay.ny, lay.numtiles, lay.omp)
               if overridden else ""))
    # the rank order the winner was measured with; a layout changed by hand starts as block
    dist = winner.dist if winner is not None and not overridden and winner.dist else "block"
    config = make_config(lay.nx, lay.ny, lay.numtiles, lay.omp, lay.nodes, dist=dist)
    return LayoutChoice(lay, config, rep, winner, overridden)


def _prediction_candidate(opts: RunOptions, case: Case, choice: LayoutChoice, sample_points,
                          over: Overheads, notes: list) -> tuple[Candidate, Overheads]:
    """--prediction: the run's config with pred_rebuild.json's compute model and overheads."""
    with open(opts.prediction) as f:
        pred = json.load(f)
    comp, over = model_from_prediction(pred, case.geo, sample_points, over)
    hw, lay = pred.get("hardware") or {}, choice.lay
    if (hw.get("nx"), hw.get("ny"), hw.get("omp", 1)) != (lay.nx, lay.ny, lay.omp) \
            or (pred.get("nodes") not in (None, lay.nodes)):
        notes.append("prediction was made for %s nodes %sx%s omp %s, not this config"
                     % (pred.get("nodes"), hw.get("nx"), hw.get("ny"), hw.get("omp")))
    return Candidate(config=choice.config, compute=comp), over


def _tune_report_candidates(case: Case, choice: LayoutChoice, sample_points, candidates: list,
                            notes: list) -> bool:
    """--tune-report: add its ranking rows' candidates; a -O report without ranking points
    gives the winner's median s/step as a constant rate instead -> True (rate only)."""
    rep, winner = choice.report, choice.winner
    if rep.get("pipeline_only"):
        notes.append("-O0 tune report: its layout is used, it measured nothing")
        return False
    if any(r.get("points") for r in rep.get("ranking") or []) or not (winner and winner.median):
        cal = wp.load_calibration(wp.calibration_path())
        found = tune_candidates(rep, case.geo, sample_points, cal, choice.lay.nodes)
        if choice.overridden:          # explicit flags: that layout only, never the ranking's
            c = choice.config
            found = [x for x in found if (x.config.nx, x.config.ny, x.config.numtiles,
                                          x.config.omp) == (c.nx, c.ny, c.numtiles, c.omp)]
            if not found:
                notes.append("tune report ranks no layout matching the CLI overrides: its "
                             "predictions are not used")
        candidates += found
        return False
    if candidates:
        notes.append("tune report has no ranking points: ignored (--prediction is used)")
        return False
    candidates.append(Candidate(config=choice.config,
                                compute=constant_compute_fn(winner.median)))
    notes.append("tune report has no ranking points: winner %s median %.4f s/step "
                 "as the rate; later segments' --time from each finished segment's "
                 "observed rate" % (winner.cid, winner.median))
    if choice.overridden:
        notes.append("that median was measured for N=%d %dx%d tiles %d omp %d, not "
                     "this config" % (winner.nodes, winner.nx, winner.ny, winner.numtiles,
                                      winner.omp))
    return True


def _wall_model(opts: RunOptions, case: Case, choice: LayoutChoice,
                notes: list) -> tuple[list, Overheads, bool]:
    """(candidates, overheads, replan) from --prediction, --tune-report, else the fallback
    cell-step rate (replan: later --time from each finished segment's observed rate)."""
    sample_points = None
    if opts.prediction or opts.tune_report:
        sample_points, _src = wp.read_wrfinput_sample_points(case.case_dir)
    over = fallback_overheads(case.geo, opts.restart_io_form, opts.io)
    candidates, config, winner = [], choice.config, choice.winner
    if opts.prediction:
        cand, over = _prediction_candidate(opts, case, choice, sample_points, over, notes)
        candidates.append(cand)
    rate_only = False
    if opts.tune_report:
        rate_only = _tune_report_candidates(case, choice, sample_points, candidates, notes)
    notes += rank_mapping_notes(config, winner.cores_per_node
                                if winner is not None and not choice.overridden else None)
    replan = not candidates or rate_only
    if not candidates:
        s = fallback_s_per_step(case.nl, config.ranks, config.omp)
        candidates = [Candidate(config=config, compute=constant_compute_fn(s))]
        notes.append("no prediction: %.3f s/step fallback (cell-step model x %.1f); later "
                     "segments' --time from each finished segment's observed rate"
                     % (s, wtune.machine.measured("radiation_factor")))
    return candidates, over, replan


def _plan_segments(opts: RunOptions, case: Case, candidates: list, over: Overheads,
                   inputs: InputPlan, wall_limit_s: Optional[float],
                   notes: list) -> tuple[Optional[int], list[Segment], RestartSetting]:
    """(L, segments, restart setting); every segment start must have its input record."""
    geo, align = case.geo, inputs.align
    case_rst_s = case_restart_interval_s(case.nl)
    ckpt = checkpoint_request(opts.checkpoint_minutes, opts.resume)
    if isinstance(ckpt, int):
        ckpt = checkpoint_steps_for_minutes(geo, ckpt, align)
    L, segs = plan_run(geo, case.n_end, candidates, over, opts.segment_minutes, wall_limit_s,
                       case_restart_s=case_rst_s, align=align, notes=notes, checkpoint=ckpt)
    if inputs.bp_plan is not None:
        miss = missing_record_files(inputs.links, inputs.per_record, segs)
        if miss:
            raise RunError("per-record ADIOS2 input missing at a segment start: %s (WRF "
                           "opens <name>_<segment start>.bp)"
                           % ", ".join("seg %d %s" % m for m in miss[:6]))
    return L, segs, restart_setting(geo, L, case.n_end, case_rst_s, ckpt, align)


def _plan_notes(opts: RunOptions, case: Case, L: Optional[int], segs: list,
                rst: RestartSetting, notes: list) -> None:
    """Notes about what the plan implies for this case's own settings."""
    geo, text = case.geo, case.text
    if opts.resume > 0 and not restart_times(geo, 0, (L or case.n_end) - 1, rst.cadence_s):
        notes.append("no restart is written inside a segment (restart %s): --resume has "
                     "nothing to resume a failed job from" % rst.label)
    for key, name in _missing_namelist_files(text, case.case_dir):
        notes.append("WARNING: the namelist's %s names %s, which is not in CASE_DIR -- every "
                     "segment will lack it%s" % (
                         key, name, " (WRF 4.8 hangs at input_input when the iofields file "
                         "is missing, even with ignore_iofields_warning=.true.)"
                         if key == "iofields_filename" else ""))
    if coldstart.get_first(text, "override_restart_timers", ".false.").lower().startswith(".t"):
        notes.append("case sets override_restart_timers=.true.: set .false. in every segment "
                     "(it would re-anchor the history alarms at each segment start)")
    if segs and geo.hist[0] is None:
        notes.append("d01 writes no history")
    debug = coldstart.get_first(text, "debug_level", "0").rstrip(",").strip()
    if debug.lstrip("-").isdigit() and int(debug) > 0:
        notes.append("WARNING: case sets debug_level=%s: every rank writes per-step lines to "
                     "rsl.out.* (sc26p1, debug_level=50: ~31 MB per rank per 6 model hours), "
                     "which the output estimate does not count and which slows the run; ranks "
                     "> 0 are gzipped after each segment that succeeds, so --segment-minutes "
                     "bounds the peak; set 0 unless you are debugging" % debug)
    if coldstart.get_first(text, "write_hist_at_0h_rst", ".false.").lower().startswith(".t"):
        notes.append("case sets write_hist_at_0h_rst=.true.: every resumed segment repeats "
                     "its start history frame")


def _wall_limit_s(opts: RunOptions) -> Optional[float]:
    """--wall-limit in seconds, else the machine's longest job, else None (no limit)."""
    if opts.wall_limit:
        return parse_hms(opts.wall_limit)
    return wtune.machine.max_hours() * 3600.0 or None


def build_plan(opts: RunOptions, case_dir: str, out: str, log: Callable) -> RunPlan:
    """Everything decided before the first job; RunError (and the errors main() lists) when
    the run cannot be planned."""
    case = _load_case(opts, case_dir, out)
    wall_limit_s = _wall_limit_s(opts)
    choice = _choose_layout(opts, log)
    notes: list[str] = []
    candidates, over, replan = _wall_model(opts, case, choice, notes)
    inputs = plan_inputs(opts.io, case.text, case.nl, case.geo, case_dir, out, opts.project,
                         notes)
    L, segs, rst = _plan_segments(opts, case, candidates, over, inputs, wall_limit_s, notes)
    _plan_notes(opts, case, L, segs, rst, notes)
    nc_root = None
    if opts.sidecar or opts.compress_history:
        import wtune.sidecar
        nc_root = wtune.sidecar.post_cache_root(out, find_project_dir(opts.project))
    return RunPlan(case_dir=case_dir, out=out, text=case.text, geo=case.geo, n_end=case.n_end,
                   start_dir=case_dir if case_restarts(case.nl) else None,
                   wall_limit_s=wall_limit_s, over=over, replan=replan, inputs=inputs, L=L,
                   segments=segs, restart=rst, notes=notes, nc_root=nc_root)


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def print_plan(opts: RunOptions, plan: RunPlan, log: Callable) -> None:
    geo, align, inputs = plan.geo, plan.inputs.align, plan.inputs
    log("run %s -> %s (%d d01 steps of %gs)%s; %d segment(s)%s, restart %s, overheads: %s"
        % (_fmt(geo.start), _fmt(step_time(geo, plan.n_end)), plan.n_end, geo.dt,
           " stop-at" if plan.n_end < geo.steps else "", len(plan.segments),
           " cut on input record times (multiples of %s: %s)" % (
               _hms(align.seconds), align.label) if align and plan.L else "",
           plan.restart.label, plan.over.source))
    for s in plan.segments:
        c = s.config
        log("  seg %d  %s -> %s  %-8s %dx%d t%d omp%d  predicted %s  --time=%s"
            % (s.index, s.start, s.end, c.id, c.nx, c.ny, c.numtiles, c.omp,
               _hms(s.predicted_wall_s), s.time_limit))
    for n in plan.notes:
        log("  note: %s" % n)
    if opts.io == "netcdf":
        log("WARNING: --io netcdf is the LEGACY netCDF I/O path (ADIOS2 is the default for every "
            "WRF stream since 2026-10-03): the case's own io_form_history/input/boundary/aux are "
            "kept, restart io_form %d" % opts.restart_io_form)
    log(io_plan_line(opts.io, inputs.io_streams, opts.restart_io_form, inputs.bp_plan,
                     inputs.bp_cache, plan.case_dir, tuple(_aux_outputs(plan.text))))


def check_space(opts: RunOptions, plan: RunPlan, log: Callable, runner=None) -> None:
    """Estimated history + restart output against the space writable at --out's filesystem:
    a line when it takes half of it or more, RunError when it does not fit (unless
    --no-space-check or --dry-run)."""
    span_s = plan.n_end * plan.geo.dt
    n_rst = max(0, len(plan.segments) - 1) + (1 if plan.restart.cadence_s else 0)
    need, detail = estimate_output_bytes(plan.geo, span_s, n_rst)
    if opts.compress_history:
        need /= 2.0
    conv, fresh = input_bytes(plan.inputs.bp_plan, plan.inputs.bp_cache,
                              sweep.campaign_configs._import_wtune_bp())
    if conv or fresh:
        need += conv + fresh
        detail = ", ".join(x for x in (detail, "inputs to BP %s" % _gb(conv) if conv else "",
                                       "per-segment input copy %s" % _gb(fresh) if fresh else "")
                           if x)
    if need <= 0:
        return
    where = plan.out
    while not os.path.exists(where):
        where = os.path.dirname(where)
    free, src = writable_bytes(where, runner)
    if need < 0.5 * free:
        return
    log("output estimate %s (%s%s, %d restart set(s)); %s %s"
        % (_gb(need), detail or "no history", ", halved for --compress-history"
           if opts.compress_history else "", n_rst, src, _gb(free)))
    if need > free and not (opts.no_space_check or opts.dry_run):
        raise RunError("the run would write ~%s but only %s is writable (%s): WRF+ADIOS2 does "
                       "not fail on a full disk, it hangs or writes empty frames. Raise "
                       "history_interval, shorten with --stop-at, free space, or pass "
                       "--no-space-check" % (_gb(need), _gb(free), src))


def _gb(n: float) -> str:
    return "%.1f GB" % (n / 1e9)


def print_dry_run(opts: RunOptions, plan: RunPlan, log: Callable) -> None:
    """--dry-run: the namelist changes of the first two segments."""
    for s in plan.segments[:2]:
        t_a = dt.datetime.strptime(s.start, WRF_FMT)
        t_b = dt.datetime.strptime(s.end, WRF_FMT)
        new = render_segment_namelist(plan.text, t_a, t_b, s.index, s.config,
                                      plan.restart.set_min, opts.restart_io_form,
                                      io_streams=plan.inputs.io_streams,
                                      per_record_edits=plan.inputs.per_record_edits)
        log("namelist diff, segment %d:" % s.index)
        for ln in namelist_diff_summary(plan.text, new):
            log("  " + ln)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _run_context(opts: RunOptions, plan: RunPlan, runners: Runners,
                 sidecar_runner: Optional[Callable]) -> RunContext:
    ctx = RunContext(opts=opts, plan=plan, runners=runners, report=new_report(opts, plan),
                     wrf_tree=opts.wrf_tree or os.environ.get("WRF_TREE"),
                     bp_copier=_logging_copier(runners.log))
    case_xml = os.path.join(plan.case_dir, "adios2.xml")
    if os.path.isfile(case_xml):
        with open(case_xml) as f:
            ctx.case_adios2_xml = f.read()
    if opts.sidecar or opts.compress_history:
        ctx.sidecar = _make_sidecar(ctx, plan.nc_root, sidecar_runner)
    return ctx


def execute(ctx: RunContext) -> int:
    """Run the controller and close the report -> exit code (1: a segment failed)."""
    report, log = ctx.report, ctx.log
    path = os.path.join(ctx.plan.out, "run_report.json")
    try:
        rc = run_controller(ctx)
        report["status"] = "completed"
    except RunStopped as e:
        report["status"], report["error"] = "stopped", str(e)
        report["finished_utc"] = dt.datetime.utcnow().isoformat()
        _write_report(path, report)
        log("wtune run: %s; report %s (the segments done so far are in %s)"
            % (e, path, ctx.plan.out))
        return 130
    except (RunError, RuntimeError, OSError, coldstart.ColdstartError) as e:
        report["status"] = "failed"
        report["error"] = str(e)
        _write_report(path, report)
        print("error: %s" % e, file=sys.stderr)
        return 1
    except BaseException as e:         # a second Ctrl-C, a bug: never leave the report "running"
        report["status"] = "stopped" if isinstance(e, KeyboardInterrupt) else "failed"
        report["error"] = "%s: %s" % (type(e).__name__, e)
        _write_report(path, report)
        raise
    report["finished_utc"] = dt.datetime.utcnow().isoformat()
    _write_report(path, report)
    log("done: %d segment(s)%s; report %s" % (len(ctx.segments),
                                              " -- PHYSICS-GRAY" if report["physics_gray"] else "",
                                              path))
    return rc


def _convert_in_job(opts: RunOptions, plan: RunPlan, case_dir: str, out: str,
                    runners: Runners) -> None:
    """inputs.convert_inputs_in_job through the run's sbatch/sacct; Ctrl-C cancels the job."""
    jobs: list = []

    def submit(script):
        jobs.append(sweep.submit_job("sbatch %s" % shlex.quote(script),
                                     sbatch_runner=runners.sbatch_runner))
        return jobs[-1]

    def wait(job):
        return wait_for_job(job, runners.sacct_runner, opts.sacct_cmd, opts.poll_interval,
                            runners.sleep, runners.log)[0]
    try:
        convert_inputs_in_job(plan.inputs, plan.text, case_dir, out, submit, wait, runners.log,
                              env_source=opts.env_source, campaign_dir=opts.campaign_dir)
    except KeyboardInterrupt:
        for j in jobs:
            runners.scancel_runner(j)
        raise


def _fail(e) -> int:
    print("error: %s" % e, file=sys.stderr)
    return 2


def main(argv: Optional[list] = None, sbatch_runner: Optional[Callable] = None,
         sacct_runner: Optional[Callable] = None, snapshot_fn: Optional[Callable] = None,
         sleep: Optional[Callable] = None, log: Callable = print,
         sidecar_runner: Optional[Callable] = None, scancel_runner: Optional[Callable] = None,
         clock: Optional[Callable] = None, convert_in_job: Optional[bool] = None) -> int:
    opts = parse_options(argv)
    try:
        opts = apply_project_defaults(opts, log)
    except RunError as e:
        return _fail(e)
    case_dir = os.path.abspath(opts.case_dir)
    refusal = wtune.extensions.refusal_run(case_dir, cfl_rescue=opts.cfl_rescue,
                                           wrfda_forecast_only=opts.wrfda_forecast_only)
    if refusal:
        return _fail(refusal)
    for note in wtune.extensions.run_notes(case_dir):
        log("EXTENSION: %s" % note)
    try:
        opts = restart_io_form_of_case(opts, case_dir, log)
        out = _resolve_out(opts)
        plan = build_plan(opts, case_dir, out, log)
    except (RunError, OSError, ValueError, wp.PredictError, coldstart.ColdstartError) as e:
        return _fail(e)
    print_plan(opts, plan, log)
    try:
        check_space(opts, plan, log)
    except RunError as e:
        return _fail(e)
    if opts.dry_run:
        try:
            print_dry_run(opts, plan, log)
        except coldstart.ColdstartError as e:
            return _fail(e)
        return 0
    os.makedirs(out, exist_ok=True)
    runners = Runners(sbatch_runner=sbatch_runner or sweep.default_sbatch_runner,
                      sacct_runner=sacct_runner or sweep.default_sacct_runner,
                      snapshot_fn=snapshot_fn or wtune.watch.snapshot,
                      sleep=sleep or time.sleep, log=log,
                      scancel_runner=scancel_runner or default_scancel,
                      clock=clock or time.monotonic)
    try:
        if convert_in_job is None:      # real submission from a login node
            convert_in_job = sbatch_runner is None and not os.environ.get("SLURM_JOB_ID")
        if convert_in_job:
            _convert_in_job(opts, plan, case_dir, out, runners)
        plan.inputs = convert_inputs(plan.inputs, plan.text, case_dir, plan.segments, log)
    except RunError as e:
        return _fail(e)
    ctx = _run_context(opts, plan, runners, sidecar_runner)
    ctx.state_file = os.path.join(out, "run.state.json")
    with wtune.stopping.stop_on_signal(ctx.state_file, "run", opts.poll_interval):
        return execute(ctx)
