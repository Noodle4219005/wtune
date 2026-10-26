"""`wtune prep wps` and `wtune prep real`: WPS (geogrid, ungrib, metgrid) and then real.exe for the
project's case over its whole namelist window, one Slurm job each. The forcing (GFS/FNL/CFSR by
date, or --forcing) is downloaded on the login node by `prep wps`. Work dir <project>/input/prep/;
`prep real` links the new wrfinput/wrfbdy into <project>/case/. Reuses tools/case_rebuild.py's
rebuild steps; the original case dir is never written."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from wtune.tools import case_rebuild as cr
from wtune.tools import coldstart_sample as cs
from wtune.tools import scc_sweep as sweep

REAL_OUTPUTS = ("wrfinput_d", "wrfbdy_d", "wrflowinp_d")


class PrepError(RuntimeError):
    pass


def _dirs(project: str | None) -> tuple[str, str]:
    import wtune.project
    project_dir = wtune.project.find_project(explicit=project)
    if not project_dir:
        raise PrepError("run it inside a project (wtune project init --case ...) or pass --project")
    return os.path.join(project_dir, "case"), os.path.join(project_dir, "input", "prep")


def _job(out: str, estimate_min: int = 30):
    chosen, plan = cr._rebuild_forcing_choice(out)
    plan["_out_dir"] = os.path.abspath(out)
    return cr._rebuild_job(plan, chosen, None, estimate_min)


def _write_sbatch(out: str, name: str, sections: list) -> str:
    lines = cr._sbatch_header(_job(out))
    for section in sections:
        lines += section(_job(out))
    lines += ["CUR_WINDOW=\"\"", "", "run_hygiene reap+census",
              "echo \"=== DONE ($(date -u +%FT%TZ)) ===\""]
    path = os.path.join(out, name)
    with open(path, "w") as fh:
        fh.write("\n".join(lines).replace("--job-name=case-rebuild", "--job-name=wtune-prep-"
                                          + name.split(".")[0]) + "\n")
    return path


def _submit_and_wait(sbatch_path: str, poll_s: float, sbatch_runner=None, sacct_runner=None,
                     sleep=time.sleep) -> str:
    """The job's final Slurm state. Ctrl-C cancels the job."""
    job_id = sweep.submit_job("sbatch %s" % sbatch_path,
                              sbatch_runner=sbatch_runner or sweep.default_sbatch_runner)
    print("submitted %s (%s); log %s/run-%s.out" % (job_id, sbatch_path,
                                                    os.path.dirname(sbatch_path), job_id),
          flush=True)
    try:
        while True:
            sleep(poll_s)
            state = sweep.sacct_states([job_id], sacct_runner or sweep.default_sacct_runner
                                       ).get(job_id, ("PENDING", ""))[0]
            if state not in ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING"):
                return state
    except KeyboardInterrupt:
        subprocess.run(["scancel", job_id], check=False)
        print("\ncancelled %s" % job_id, file=sys.stderr)
        raise


def prep_wps(project: str | None = None, forcing: str | None = None, poll_s: float = 30,
             **runners) -> list:
    """met_em files of the whole window in <project>/input/prep/wps/."""
    case_dir, out = _dirs(project)
    if not any(os.path.exists(os.path.join(case_dir, n)) for n in ("namelist.wps", "wrfinput_d01")):
        raise PrepError("%s has neither namelist.wps nor wrfinput_d01: WPS needs the domain's "
                        "projection" % case_dir)
    for k in ("WPS_DIR", "WRF_TREE"):
        if not os.environ.get(k):
            raise PrepError("%s is not set -- source the machine's env.sh first" % k)
    import wtune.machine
    if not os.path.isdir(wtune.machine.geog()):
        raise PrepError("WPS static data %s not found: set `geog` in machine.yaml (or WPS_GEOG)"
                        % wtune.machine.geog())
    cr.cmd_rebuild_plan(case_dir, out, forcing=forcing)
    cr.cmd_rebuild_fetch(out)            # raises when files are still missing
    cr.cmd_rebuild_stage(out)
    path = _write_sbatch(out, "wps.sbatch", [cr._sbatch_geogrid, cr._sbatch_ungrib,
                                             cr._sbatch_metgrid])
    state = _submit_and_wait(path, poll_s, **runners)
    met = sorted(glob.glob(os.path.join(out, "wps", "met_em.d*")))
    missing = [m for m in _job(out).expected_met if not os.path.exists(os.path.join(out, "wps", m))]
    if state != "COMPLETED" or missing:
        raise PrepError("WPS job %s; %d met_em file(s) missing (first: %s)"
                        % (state, len(missing), missing[0] if missing else "-"))
    print("WPS done: %d met_em file(s) in %s/wps; next: wtune prep real" % (len(met), out))
    return met


def prep_real(project: str | None = None, replace: bool = False, poll_s: float = 30,
              **runners) -> list:
    """wrfinput/wrfbdy of the whole window, linked into <project>/case/."""
    case_dir, out = _dirs(project)
    if not glob.glob(os.path.join(out, "wps", "met_em.d01.*")):
        raise PrepError("no met_em in %s/wps: run `wtune prep wps` first" % out)
    with open(os.path.join(out, "plan.json")) as fh:
        max_dom = json.load(fh)["max_dom"]
    names = ["wrfinput_d%02d" % d for d in range(1, max_dom + 1)] + ["wrfbdy_d01"]
    taken = [n for n in names if os.path.lexists(os.path.join(case_dir, n))]
    if taken and not replace:
        raise PrepError("case/ already has %s: pass --replace to use the new ones"
                        % ", ".join(taken))
    path = _write_sbatch(out, "real.sbatch", [cr._sbatch_real])
    state = _submit_and_wait(path, poll_s, **runners)
    real = os.path.join(out, "real")
    made = sorted(n for n in os.listdir(real) if n.startswith(REAL_OUTPUTS)
                  and not os.path.islink(os.path.join(real, n)))
    missing = [n for n in names if n not in made]
    if state != "COMPLETED" or missing:
        raise PrepError("real.exe job %s; missing %s (see %s/rsl.error.0000)"
                        % (state, ", ".join(missing) or "nothing", real))
    for n in made:
        dst = os.path.join(case_dir, n)
        if os.path.lexists(dst):
            os.remove(dst)              # a link in the project's case/, never the source file
        os.symlink(os.path.join(real, n), dst)
    print("real.exe done: linked %s into %s" % (", ".join(made), case_dir))
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wtune prep", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("wps", help="geogrid + ungrib + metgrid over the case's window (one job)")
    p.add_argument("--forcing", default=None, help="pin the forcing dataset (default: by date)")
    p = sub.add_parser("real", help="real.exe on the met_em of `prep wps` (one job)")
    p.add_argument("--replace", action="store_true",
                   help="replace the wrfinput/wrfbdy links already in case/")
    for p in sub.choices.values():
        p.add_argument("--project", default=None)
        p.add_argument("--poll-interval", type=float, default=30)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "wps":
            prep_wps(a.project, a.forcing, a.poll_interval)
        else:
            prep_real(a.project, a.replace, a.poll_interval)
    except (PrepError, cr.RebuildError, cs.ColdstartError, OSError, ValueError) as e:
        print("wtune prep %s: %s" % (a.cmd, e), file=sys.stderr)
        return 1
    return 0
