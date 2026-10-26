"""Pipeline phase 3, physics advice (decision 0059, opt-in with -O3): restart-compatible physics
candidates measured on the winner layout at two rebuilt points, scored by predicted speed-up
and gated for fidelity on window frames. Advisory only: nothing is applied."""
from __future__ import annotations

import dataclasses
import datetime
import glob
import math
import os
from typing import Any

from ._deps import sweep, physics_advice
from .common import ANCHOR_END_ID, compute_slots
from .context import STOP, Ctx, QueueCapExceeded
from .pipeline_state import NOT_RUN_BUDGET, MeasuredLayout, PipelineRun, _mark, _phase_record
from .points import fit_stage2, pick_points
from . import points as _points

PIPELINE_PHYSICS_MAX_CANDIDATES = 3   # physics candidates tried in phase 3 (cap)
PIPELINE_PHYSICS_POINTS = 2           # phase 3 measures t0 and the costliest predicted point
PHYSICS_RUNNER_MARKER = "physics_namelists"   # strict_campaign.sbatch must read a per-config
                                              # {case id: namelist path} map for phase 3 to run
WINDOW_FRAME_LEVEL = "mid_end"        # per-config `window_frame`: mid + end d01 frames (a restarted
                                      # run never writes a frame at its own start)
MIN_WINDOW_MIN = 2                    # model minutes the mid + end frames need


def physics_runner_supported(sbatch_script: str) -> bool:
    """True when strict_campaign.sbatch / configs.py read per-config physics namelists
    (PHYSICS_RUNNER_MARKER); today's CONFIG_FILE path carries decomposition fields only."""
    base = os.path.dirname(os.path.abspath(sweep._expand(sbatch_script)))
    for name in (os.path.basename(sbatch_script), "configs.py"):
        try:
            with open(os.path.join(base, name)) as f:
                if PHYSICS_RUNNER_MARKER in f.read():
                    return True
        except OSError:
            continue
    return False


def _run_dir(ctx: Ctx, rp: dict, cid: str, job_id: str) -> str:
    return os.path.join(ctx.ledger_dir, "%s_%s_r1_%s" % (rp["case_id"], cid, job_id))


def _window_frames(ctx: Ctx, rp: dict, cid: str, job_id: str) -> list[str]:
    """The run's kept window frames (mid + end), sorted: the gate's input list."""
    return sorted(glob.glob(os.path.join(_run_dir(ctx, rp, cid, job_id), "wrfout_d01_*.nc")))


def _delete_window_frames(ctx: Ctx, runs: list[tuple]) -> int:
    """Delete the window-frame .nc files of `runs` [(rp, cid, job)]; returns how many."""
    n = 0
    for rp, cid, jid in runs:
        for f in _window_frames(ctx, rp, cid, jid):
            try:
                os.remove(f)
                n += 1
            except OSError:
                pass
    return n


def phase2_writes_frames(st: PipelineRun) -> bool:
    """Whether phase-2 configs keep window frames: only when phase 3 can gate on them (I/O cost)."""
    return bool(st.physics and not st.extension and physics_advice is not None
                and physics_runner_supported(st.ctx.sbatch_script))


def build_physics_window_namelists(ctx: Ctx, cands: list[dict], rsub: list[dict],
                                   anchor: dict) -> tuple[dict, list[dict]]:
    """Write <out>/physics/pt_<i>/<id>/namelist.input of each point's window (shared restart to
    window end, d01 history = the window, NetCDF) for the baseline and every candidate. Returns
    (baseline config, [candidate configs]) carrying `physics_namelists` {point case id: path}.
    UNVERIFIED against the real runner (it does not read them yet)."""
    import wtune.run                  # imports wtune.tune itself: kept lazy
    api = physics_advice
    cfg = dict(ranks=anchor["ranks"], omp=anchor["omp"], nx=anchor["nx"], ny=anchor["ny"],
               numtiles=anchor["numtiles"])

    def write(rp, cid, text):
        d = os.path.join(ctx.out_dir, "physics", "pt_%d" % rp["index"], cid)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "namelist.input")
        with open(path, "w") as f:
            f.write(text)
        return path

    base_paths: dict[str, str] = {}
    cand_paths: dict[str, dict[str, str]] = {c["id"]: {} for c in cands}
    for rp in rsub:
        with open(rp["template"]) as f:
            template = f.read()
        dt = float(rp["dt"])
        seg_start = rp["T"] + datetime.timedelta(minutes=rp["spinup_min"])
        wmin = int(math.ceil(rp["steps"] * dt / 60.0))
        seg_end = seg_start + datetime.timedelta(minutes=wmin)

        def window(text):
            txt = wtune.run.render_segment_namelist(text, seg_start, seg_end, 1, cfg,
                                                    wtune.run.RESTART_OFF_MIN,
                                                    restart_io_form=ctx.restart_io_form)
            txt = wtune.run._set_one(txt, "history_interval", wmin)
            return wtune.run._set_one(txt, "io_form_history", 2)

        base_paths[rp["case_id"]] = write(rp, "ANCHOR", window(template))
        for cd in cands:
            cand_paths[cd["id"]][rp["case_id"]] = write(
                rp, cd["id"], window(api.render_candidate_namelist(template, cd)))
    base = dict(anchor, physics_namelists=base_paths)
    return base, [dict(anchor, id=cd["id"], config_source="generated",
                       physics_namelists=cand_paths[cd["id"]]) for cd in cands]


def _merge_verdicts(verdicts: list) -> dict | None:
    """One fidelity verdict per candidate: its first failing point, else the point with the
    largest worst rel_rms; None when no point could be compared."""
    vs = [v for v in verdicts if v]
    if not vs:
        return None
    fails = [v for v in vs if not v.get("passed")]
    if fails:
        return fails[0]
    return max(vs, key=lambda v: (v["worst"][0]["rel_rms"] if v.get("worst") else 0.0))


def _phase3_not_run_reason(st: PipelineRun, layouts: list[dict]) -> str | None:
    """Why phase 3 cannot run (checked in order), or None when it may."""
    ctx, plan, level_req = st.ctx, st.plan, st.level_req
    if not st.physics:
        return ("not run (physics advice is opt-in: pass --physics-advice with -O3)"
                if plan["level"] >= 3 or level_req >= 3 and plan["level"] >= 3
                else "not run (physics advice needs -O3)" if level_req < 3
                else "not run (stepped down to O%d)" % plan["level"])
    if st.extension:
        return "not run (extension case: physics parameters are not tunable)"
    if st.cap_hit():
        return "not run (%s)" % (STOP["reason"] or "max queue wait exceeded")
    if st.rebuilt is None or not layouts:
        return "not run (no rebuilt points)" if st.rebuilt is None else "not run (no layout)"
    if ctx.elapsed() >= st.limit("physics"):
        return NOT_RUN_BUDGET
    if physics_advice is None:
        return "not run (wtune.physics_advice not importable)"
    if not physics_runner_supported(ctx.sbatch_script):
        return ("not run (campaign runner has no per-config physics namelists: %s not found in "
                "strict_campaign.sbatch/configs.py)" % PHYSICS_RUNNER_MARKER)
    return None


def _phase3_plan_candidates(st: PipelineRun,
                            physics_max_candidates: int) -> tuple[str | None, list]:
    """(status or None to measure, candidates) from wtune.physics_advice.plan_candidates (<= the
    cap); planned even when the window is too short for mid + end frames."""
    ctx, ph3, cands = st.ctx, None, []
    try:
        if min(rp["steps"] * float(rp["dt"]) / 60.0 for rp in st.rebuilt) < MIN_WINDOW_MIN:
            ph3 = "not run (window < %d min of model time: mid+end frames need more)" \
                % MIN_WINDOW_MIN
        cands = physics_advice.plan_candidates(
            ctx.namelist_path, max_candidates=min(int(physics_max_candidates),
                                                  PIPELINE_PHYSICS_MAX_CANDIDATES),
            case_name=ctx.case)
        if not cands and ph3 is None:
            ph3 = "not run (no applicable physics candidate: %s)" % (
                getattr(cands, "note", "") or "none planned")
    except Exception as e:                              # noqa: BLE001
        ph3 = "failed: plan_candidates: %s: %s" % (type(e).__name__, e)
    return ph3, cands


def _pm_key(pm: Any) -> str:
    """A point measurement's key in the physics weights / speed-up tables."""
    return pm.at.isoformat()


def _phase3_weights(st: PipelineRun, api: Any, wsrc: list) -> dict:
    """Per-point weights of the speed-up from combine() (needs the prediction context), else
    equal weights, noted in pipe.physics_weights."""
    try:
        if not st.pc_ok:
            raise RuntimeError("no prediction context")
        return api.weights_from_combine(
            sorted(wsrc, key=lambda m: m.at), st.run_start, st.run_end, st.dt, st.stepra,
            calibration=st.calibration, sample_points=st.sample_points)
    except Exception as e:                      # noqa: BLE001
        weights = {_pm_key(pm): 1.0 for pm in wsrc}
        st.pipe.physics_weights = "equal (combine unavailable: %s)" % e
        return weights


def _phase3_candidate_verdicts(st: PipelineRun, api: Any, gctx: Ctx, cid: str,
                               rsub: list[dict], jobs: list, twin_e: MeasuredLayout | None
                               ) -> tuple[list, bool]:
    """(verdicts, used_twin): candidate `cid` against the baseline at every point with window
    frames, the twin being phase 2's runner-up there; writes the gate outputs under
    <out>/physics/gate."""
    vs, used_twin = [], False
    for idx, (rp, jid) in enumerate(zip(rsub, jobs)):
        b, cw = (_window_frames(gctx, rp, "ANCHOR", jid),
                 _window_frames(gctx, rp, cid, jid))
        if not (b and cw):
            continue
        twin = (_window_frames(gctx, rp, twin_e.cid, twin_e.jobs[idx])
                if twin_e and idx < len(twin_e.jobs) else None) or None
        gate_dir = os.path.join(st.ctx.out_dir, "physics", "gate")
        os.makedirs(gate_dir, exist_ok=True)
        used_twin = used_twin or bool(twin)
        vs.append(api.fidelity_verdict(
            b, cw, twin=twin, judged_fields=api.PROGNOSTIC_FIELDS,
            out_prefix=os.path.join(gate_dir, "%s_pt%d" % (cid, rp["index"]))))
    return vs, used_twin


def _phase3_score(st: PipelineRun, api: Any, gctx: Ctx, cands: list, top: list, meas: dict,
                  rsub: list[dict], jobs: list, weights: dict) -> tuple[dict, dict, bool]:
    """(speed {cid: speed-up}, fid {cid: merged verdict}, used_twin) of every measured
    candidate."""
    base_by = {_pm_key(pm): pm.s_normal for pm in meas.get("ANCHOR", [])}
    twin_e = next((e for e in st.ranked_all if e is not st.winner), None)
    used_twin = False
    speed, fid = {}, {}
    for cd in cands:
        cid = cd["id"]
        if cid not in {t["id"] for t in top} or cid not in meas:
            continue
        speed[cid] = api.predicted_speedup(
            base_by, {_pm_key(pm): pm.s_normal for pm in meas[cid]}, weights)
        vs, used = _phase3_candidate_verdicts(st, api, gctx, cid, rsub, jobs, twin_e)
        used_twin = used_twin or used
        fid[cid] = _merge_verdicts(vs)
    return speed, fid, used_twin


def _phase3_measure(st: PipelineRun, layouts: list[dict], cands: list) -> str:
    """The restart-compatible candidates on the winner layout (phase 2's, else phase 1's best)
    at t0 + the costliest point: one job per point, baseline + candidates from its shared
    restart; scheme swaps stay estimate-only rows. Returns the status (may raise); sets
    st.advice and the pipe.physics_* fields."""
    ctx, pipe, api = st.ctx, st.pipe, physics_advice
    # a skipped phase 2 must not skip phase 3: phase 1's best layout, the provisional gate and
    # weights from the phase-3 baseline itself
    w3 = st.winner or MeasuredLayout(layout=layouts[0], nodes=layouts[0]["nodes"],
                                     cores=layouts[0]["cores"], meas=None, points=None)
    lay, (n, c) = w3.layout, (w3.nodes, w3.cores)
    wm = (w3.meas or []) if w3.points in (None, [rp["index"] for rp in st.rebuilt]) else []
    rsub = pick_points(st.rebuilt, PIPELINE_PHYSICS_POINTS, st.sample_points,
                       [pm.s_normal for pm in wm] if wm else None)
    measurable = [cd for cd in cands if cd.get("restart_compatible", True)]
    base_cfg, cand_cfgs = (build_physics_window_namelists(ctx, measurable, rsub,
                                                          dict(lay["cfg"]))
                           if measurable else (dict(lay["cfg"]), []))
    base_cfg = dict(base_cfg, window_frame=WINDOW_FRAME_LEVEL)
    cand_cfgs = [dict(cf, window_frame=WINDOW_FRAME_LEVEL) for cf in cand_cfgs]
    gctx = dataclasses.replace(ctx, nodes=n, core_budget=c, budget_s=st.limit("physics"),
                               stage1_max_running=compute_slots(ctx.max_nodes, n))
    s_by = {"ANCHOR": lay["median"], **{cf["id"]: lay["median"] for cf in cand_cfgs}}
    # the 2 point jobs run in parallel
    top, _est, skip = (fit_stage2(gctx, cand_cfgs, base_cfg, rsub, s_by, None,
                                  left_s=st.limit("physics") - ctx.elapsed())
                       if cand_cfgs else ([], 0.0, None))
    if skip:
        return NOT_RUN_BUDGET + ": " + skip
    if not cand_cfgs:
        st.advice = api.advice_table(list(cands), {}, {})
        return "partial: no restart-compatible candidate -- estimates only"
    meas, _g, jobs = _points.measure_candidates_at_points(gctx, top, rsub, base_cfg, s_by_cid=s_by)
    weights = _phase3_weights(st, api, wm or meas.get("ANCHOR", []))
    speed, fid, used_twin = _phase3_score(st, api, gctx, cands, top, meas, rsub, jobs, weights)
    st.frame_runs.extend((rp, cid, jid) for cid in ["ANCHOR", ANCHOR_END_ID]
                         + [t["id"] for t in top] for rp, jid in zip(rsub, jobs))
    # measured candidates + the estimate-only scheme swaps (cost-model prior)
    st.advice = api.advice_table(list(cands), speed, {k: v for k, v in fid.items() if v})
    ph3 = "ok" if len(top) == len(cand_cfgs) else (
        "partial: %d of %d candidates fit the budget" % (len(top), len(cand_cfgs)))
    pipe.physics_points = [rp["point"].at.isoformat() for rp in rsub]
    pipe.physics_fidelity_gate = (
        "chaos-floor (phase-2 twin)" if used_twin else
        "config-gate, PROVISIONAL (no phase-2 twin: uncalibrated tolerance)")
    pipe.physics_baseline = ("phase-2 winner" if st.winner else
                             "phase-1 best layout (phase 2 not run)")
    return ph3


def _pipeline_phase3_physics(st: PipelineRun, layouts: list[dict],
                             physics_max_candidates: int) -> None:
    """Phase 3 (_phase3_not_run_reason -> _phase3_plan_candidates -> _phase3_measure); a stop or
    failure is reported in the phase record, not raised; sets pipe.physics_advice."""
    ctx = st.ctx
    m3 = _mark(ctx)
    cands: list = []
    ph3 = _phase3_not_run_reason(st, layouts)
    if ph3 is None:
        ph3, cands = _phase3_plan_candidates(st, physics_max_candidates)
    if ph3 is None:
        try:
            ph3 = _phase3_measure(st, layouts, cands)
        except QueueCapExceeded as e:
            ph3 = "stopped: %s" % e
        except Exception as e:                              # noqa: BLE001
            ph3 = "failed: %s: %s" % (type(e).__name__, e)
    st.pipe.phases.append(_phase_record("physics", m3, _mark(ctx), st.limit("physics"), ph3))
    st.cut("physics")
    st.pipe.physics_advice = st.advice
