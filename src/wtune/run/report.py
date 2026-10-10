"""run_report.json: the plain dict the run keeps (the JSON boundary; wtune_watch and
wtune_sidecar read and extend it) and its atomic writer."""
from __future__ import annotations

import datetime as dt
import json
import os

from .common import _fmt, to_plain
from .context import RunContext, RunOptions, RunPlan
from .geometry import step_time


def _write_report(path: str, report: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_plain(report), f, indent=1, default=str)
    os.replace(tmp, path)


def _save_report(ctx: RunContext) -> None:
    _write_report(os.path.join(ctx.plan.out, "run_report.json"), ctx.report)


def _io_report(opts: RunOptions, plan: RunPlan) -> dict:
    inputs = plan.inputs
    io_report = {"mode": opts.io}
    if inputs.io_streams is not None:
        io_report.update(streams=[st["stream"] for st in inputs.io_streams],
                         bp_cache=inputs.bp_cache, bp_inputs=inputs.bp_plan["links"],
                         converted=inputs.bp_plan["convert"], per_record=inputs.per_record,
                         segment_unit_s=inputs.align.seconds if inputs.align else None)
    return io_report


def new_report(opts: RunOptions, plan: RunPlan) -> dict:
    """The report of a run about to start (status running, no segments yet)."""
    rst = plan.restart
    return {"case_dir": plan.case_dir, "out": plan.out, "run_start": _fmt(plan.geo.start),
            "run_end": _fmt(step_time(plan.geo, plan.n_end)), "stop_at": opts.stop_at,
            "dt": plan.geo.dt, "restart_io_form": opts.restart_io_form,
            "io": _io_report(opts, plan), "cfl_rescue": opts.cfl_rescue,
            "prediction": opts.prediction, "tune_report": opts.tune_report,
            "overheads": plan.over, "restart_quantum_steps": plan.L,
            "restart_interval_min": (rst.set_min if rst.set_min is not None
                                     else rst.cadence_s / 60.0),
            "restart_interval_source": "case" if rst.set_min is None else "wtune",
            "restart_cadence_s": rst.cadence_s,
            "notes": plan.notes, "physics_gray": False, "resumes": 0, "decisions": [],
            "status": "running",
            "started_utc": dt.datetime.utcnow().isoformat()}
