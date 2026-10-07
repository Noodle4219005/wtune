"""wtune_submit.py -- `wtune submit`: tune -> run -> post for the current project in one command.

Each step is its own `wtune` process, so it takes the project's settings and is recorded in
manifest.jsonl exactly as if typed by hand. The three outputs share one id:
tune/<id>/, runs/<id>/, post/<id>/."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import wtune_project

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WTUNE_BIN = os.path.join(REPO_ROOT, "bin", "wtune")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="wtune submit",
        description="tune -> run -> post for the current project (wtune project init --case). "
                    "Stops at the first failing step.")
    ap.add_argument("-O", dest="level", default="2", choices=("0", "1", "2", "3"),
                    help="tune level, as `wtune tune -O<level>` (default 2)")
    ap.add_argument("--project", default=None)
    ap.add_argument("--skip-tune", action="store_true",
                    help="no tune: run with the project's newest tune report")
    ap.add_argument("--no-post", action="store_true", help="stop after the run")
    ap.add_argument("--detach", action="store_true",
                    help="keep going after logout; output goes to <project>/submit-<id>.log")
    ap.add_argument("--dry-run", action="store_true", help="print the steps only")
    return ap


def case_window(case_dir: str):
    """(start, end, bbox) of the case's d01 from namelist.input and wrfinput_d01 -- the window
    the observation fetchers need before any wrfout exists -- or None when unreadable."""
    import datetime as dt
    import wtune_predict as wp
    try:
        nl = wp.parse_namelist(os.path.join(case_dir, "namelist.input"))
        start = wp.get_time_control_datetime(nl, "start")
        _steps, run_s, _dt, note = wp.get_d01_steps(nl)
        import netCDF4
        with netCDF4.Dataset(os.path.join(case_dir, "wrfinput_d01")) as ds:
            lat, lon = ds.variables["XLAT"][0], ds.variables["XLONG"][0]
            bbox = (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()))
    except (OSError, ValueError, KeyError, ImportError, wp.PredictError):
        return None
    if start is None or note or not run_s:
        return None
    return start, start + dt.timedelta(seconds=run_s), bbox


def fetch_steps(project_dir: str, names: list[str], window) -> list[tuple[str, list[str]]]:
    """One `wtune post fetch <name>` per observation dataset, windowed on the case."""
    start, end, bbox = window
    win = ["--end", end.strftime("%Y%m%d%H"), "--bbox", ",".join("%.3f" % v for v in bbox)]
    steps = []
    for name in names:
        cmd = [WTUNE_BIN, "post", "fetch", name, "--project", project_dir] + win
        if name != "era5":
            cmd += ["--start", start.strftime("%Y%m%d%H")]
        steps.append(("fetch " + name, cmd))
    return steps


def plan_steps(project_dir: str, out_id: str, level: str, skip_tune: bool, no_post: bool,
               fetch: list | None = None) -> list[tuple[str, list[str]]]:
    """[(step name, argv), ...] for one submit. `fetch` steps (observations) never stop it."""
    def out(sub):
        return os.path.join(project_dir, sub, out_id)

    steps = []
    run = [WTUNE_BIN, "run", "--project", project_dir, "--out", out("runs")]
    if not skip_tune:
        steps.append(("tune", [WTUNE_BIN, "tune", "-O" + level, "--project", project_dir,
                               "--out", out("tune")]))
        run += ["--tune-report", os.path.join(out("tune"), "tune_report.json")]
    steps.append(("run", run))
    if not no_post:
        steps += fetch or []
        steps.append(("post", [WTUNE_BIN, "post", "recipe", out("runs"), "--project",
                               project_dir, "--out", out("post"), "--srun"]))
    return steps


def _detach(argv: list[str], project_dir: str, out_id: str) -> int:
    log_path = os.path.join(project_dir, "submit-%s.log" % out_id)
    child = [WTUNE_BIN, "submit"] + [a for a in argv if a != "--detach"] + ["--id", out_id]
    with open(log_path, "a") as log:
        proc = subprocess.Popen(child, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True,
                                cwd=project_dir)
    print("submit running in the background (pid %d)\n  log:   %s\n  watch: wtune watch %s"
          % (proc.pid, log_path, os.path.join(project_dir, "runs", out_id)))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    ap = build_parser()
    ap.add_argument("--id", dest="out_id", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        project_dir = wtune_project.find_project(explicit=args.project)
    except wtune_project.ProjectError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    if not project_dir:
        print("error: no project here: `wtune project init DIR --case CASE_DIR`, then cd DIR "
              "(or pass --project DIR)", file=sys.stderr)
        return 2
    doc = wtune_project.load_project(project_dir)
    if not args.skip_tune and not wtune_project.project_case_id(doc):
        print("error: the project's case is not registered for tune: "
              "wtune project set --register-case", file=sys.stderr)
        return 2
    if args.skip_tune and not wtune_project.latest_tune_report(project_dir):
        print("error: --skip-tune but %s has no tune report"
              % os.path.join(project_dir, "tune"), file=sys.stderr)
        return 2

    out_id = args.out_id or wtune_project.new_output_id()
    fetch = []
    names = wtune_project.project_post_defaults(doc)["fetch"]
    if names and not args.no_post:
        window = case_window(os.path.join(project_dir, "case"))
        if window:
            fetch = fetch_steps(project_dir, names, window)
        else:
            print("note: case window unreadable (namelist.input/wrfinput_d01): observations "
                  "not fetched; post marks those figures missing")
    steps = plan_steps(project_dir, out_id, args.level, args.skip_tune, args.no_post, fetch)
    if args.dry_run:
        for name, cmd in steps:
            print("%-4s  %s" % (name, " ".join(cmd)))
        return 0
    if args.detach:
        return _detach(argv, project_dir, out_id)

    for i, (name, cmd) in enumerate(steps, 1):
        print("=== submit step %d/%d: %s ===" % (i, len(steps), name), flush=True)
        rc = subprocess.call(cmd, cwd=project_dir)
        if rc != 0 and name.startswith("fetch "):
            print("=== %s exited %d: continuing; post marks its figures missing ===" % (name, rc),
                  flush=True)
            continue
        if rc != 0:
            print("=== submit stopped: %s exited %d (outputs so far under */%s) ==="
                  % (name, rc, out_id), flush=True)
            return rc
    print("=== submit done: tune/%s runs/%s post/%s ===" % (out_id, out_id, out_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
