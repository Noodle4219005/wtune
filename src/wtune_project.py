"""wtune_project.py -- `wtune project init|info` (decision 0058): a project folder that
collects one WRF case's definition/inputs/config, every wtune tune/predict-rebuild/run
output, and every post-processing figure/report under one user-named directory, so nothing
has to be tracked by hand across separate --out flags.

Layout (decision 0058 section 1):
  project.yaml   case/site/storm/obs/time-window description
  wtune.yaml     unchanged tuner config (wtune config init writes this; not touched here)
  case/          namelist.input/namelist.output/namelist.wps copied (user edits these);
                 everything else found directly in the source case dir (wrfinput, wrfbdy,
                 met_em, ...) is symlinked, never copied -- the source case dir is read-only
                 to this tool (never opened for writing).
  input/         driving data / observations (GRIB, met_em, HURDAT2, b-deck, ERA5, ...)
  tune/          wtune tune output
  predict/       wtune predict-rebuild output
  runs/<id>/     wtune run segmented output
  post/<id>/     wtune post output (figures, tables, report.html)
  analysis/      user's own Python analysis scripts + their outputs (wtune post custom)
  manifest.jsonl one line per wtune command: time, version, argv, key input paths (+sha256
                 for files <50MB), output dir, exit code

Every project-aware subcommand discovers its project via --project DIR, else $WTUNE_PROJECT,
else walking up from cwd looking for project.yaml -- see find_project(). When no project is
found, the caller falls back to its pre-project behaviour unchanged (decision 0058 P6).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import yaml

import wtune_version

PROJECT_YAML = "project.yaml"
MANIFEST_NAME = "manifest.jsonl"
SUBDIRS = ("case", "input", "tune", "predict", "runs", "post", "analysis")
MANIFEST_HASH_LIMIT = 50 * 1024 * 1024  # 50 MB -- task spec: "sha256 for files < 50 MB"

# namelists are copied (the user edits these); everything else in a --case source dir is
# symlinked -- decision 0058 section 1.
NAMELIST_NAMES = ("namelist.input", "namelist.output", "namelist.wps")


class ProjectError(RuntimeError):
    pass


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_stamp(fmt="%Y%m%d-%H%M"):
    return datetime.now(timezone.utc).strftime(fmt)


def default_project_name(case_dir):
    case_name = os.path.basename(os.path.normpath(case_dir)) if case_dir else "wtune"
    return "%s-%s" % (case_name, _utc_stamp())


# --------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------

def find_project(start_dir=None, explicit=None):
    """Project directory resolution order (decision 0058 section 1): explicit --project DIR,
    else $WTUNE_PROJECT, else walking up from start_dir (default cwd) looking for
    project.yaml. Returns None if none of these locate a project -- callers then keep their
    pre-project behaviour unchanged (P6). Raises ProjectError only when --project/
    $WTUNE_PROJECT was given explicitly but does not actually point at a project."""
    if explicit:
        if not os.path.isfile(os.path.join(explicit, PROJECT_YAML)):
            raise ProjectError("--project %s has no %s" % (explicit, PROJECT_YAML))
        return os.path.abspath(explicit)
    env = os.environ.get("WTUNE_PROJECT")
    if env:
        if not os.path.isfile(os.path.join(env, PROJECT_YAML)):
            raise ProjectError("WTUNE_PROJECT=%s has no %s" % (env, PROJECT_YAML))
        return os.path.abspath(env)
    cur = os.path.abspath(start_dir or os.getcwd())
    while True:
        if os.path.isfile(os.path.join(cur, PROJECT_YAML)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def load_project(project_dir):
    path = os.path.join(project_dir, PROJECT_YAML)
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def project_storm(project_doc):
    """(storm_id, hurdat2_path) from a loaded project.yaml, or (None, None)."""
    storm = (project_doc or {}).get("storm") or {}
    return storm.get("id"), storm.get("hurdat2")


def project_cross_section(project_doc, kind):
    """project.yaml's optional cross_sections.<kind> override (dict with lat0/lon0/lat1/lon1),
    or None -- decision 0057 update: "也可由 project.yaml 指定"."""
    sections = (project_doc or {}).get("cross_sections") or {}
    return sections.get(kind)


def project_xsec_config(project_doc):
    """project.yaml's optional top-level xsec_top_km/xsec_halfwidth_km (2026-09-29 second
    lead review): the terrain/moisture cross-section figures' height window and
    centred-line half-width. Returns {"top_km": float|None, "halfwidth_km": float|None} --
    None for either means "let the recipe module use its own default"."""
    doc = project_doc or {}
    return dict(top_km=doc.get("xsec_top_km"), halfwidth_km=doc.get("xsec_halfwidth_km"))


def discover_runs(project_dir):
    """[(run_id, path), ...] wrfout-bearing directories: every runs/<id>/ subdirectory that
    has at least one wrfout_* frame, else (when runs/ is empty) the case/ directory itself,
    labeled "case" -- decision 0058 item 2: "runs discovered from runs/*/ or case/"."""
    runs_root = os.path.join(project_dir, "runs")
    out = []
    if os.path.isdir(runs_root):
        for name in sorted(os.listdir(runs_root)):
            d = os.path.join(runs_root, name)
            if os.path.isdir(d) and glob.glob(os.path.join(d, "wrfout_d*")):
                out.append((name, d))
    if not out:
        case_dir = os.path.join(project_dir, "case")
        if glob.glob(os.path.join(case_dir, "wrfout_d*")):
            out.append(("case", case_dir))
    return out


def new_output_id(tag=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return "%s-%s" % (stamp, tag) if tag else stamp


def default_out_dir(project_dir, subdir, tag=None):
    """post/<id>/, runs/<id>/, etc. -- id = UTC timestamp (+short tag). Does not create the
    directory; callers create it themselves, matching every existing subcommand's own
    convention of creating --out lazily."""
    return os.path.join(project_dir, subdir, new_output_id(tag))


# --------------------------------------------------------------------------------------
# manifest.jsonl (decision 0058 section 3)
# --------------------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_record(path):
    rec = dict(path=path)
    try:
        if os.path.isfile(path):
            size = os.path.getsize(path)
            rec["size_bytes"] = size
            if size < MANIFEST_HASH_LIMIT:
                rec["sha256"] = _sha256(path)
    except OSError as e:
        rec["error"] = str(e)
    return rec


def manifest_append(project_dir, argv, out_dir=None, exit_code=0, key_paths=None, extra=None):
    """Appends one line to project_dir/manifest.jsonl (decision 0058 section 3): time, wtune
    version, argv, key input paths (+sha256 for files <50MB), output dir, exit code."""
    entry = dict(
        time=_utc_now_iso(),
        wtune_version=wtune_version.__version__,
        argv=[str(a) for a in argv],
        inputs=[_path_record(p) for p in (key_paths or []) if p],
        out_dir=(os.path.relpath(out_dir, project_dir) if out_dir else None),
        exit_code=exit_code,
    )
    if extra:
        entry.update(extra)
    manifest_path = os.path.join(project_dir, MANIFEST_NAME)
    with open(manifest_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


# --------------------------------------------------------------------------------------
# `wtune project init`
# --------------------------------------------------------------------------------------

_EXAMPLE_SCRIPT = '''"""analysis/example_tc_track.py -- EXAMPLE custom analysis script, written by
`wtune project init` (decision 0058 section 2). Every script in analysis/ is run by
`wtune post custom`; this one is a template, not a required part of the pipeline -- edit or
delete it freely.

Storm-following central pressure and max 10 m wind time series: prefers an existing TC track
(post.tc / TempestExtremes output, already on disk under this project's post/ tree if
`wtune post tc` or `wtune post recipe` has run) and falls back to a per-frame SLP-minimum
search via wrf-python when no track csv is found yet, so this script produces something
even before the tc recipe has run.
"""
from post import custom_api as api


def run(ctx):
    import numpy as np

    times, pmin, vmax = [], [], []
    for run_id, run_dir in ctx.runs():
        track_csv = ctx.find_run_track_csv(run_id)
        if track_csv:
            for row in ctx.read_csv(track_csv):
                t = row.get("time") or row.get("valid_time") or row.get("valid")
                p = row.get("mslp_hpa") or row.get("pmin_hpa") or row.get("mslp")
                v = row.get("vmax_ms") or row.get("wind_kt") or row.get("vmax")
                if t is None:
                    continue
                times.append(t)
                pmin.append(float(p) if p not in (None, "") else float("nan"))
                vmax.append(float(v) if v not in (None, "") else float("nan"))
            continue
        domain = ctx.finest_domain(run_dir)
        for time_str, path in ctx.list_frames(run_dir, domain=domain):
            ds = ctx.open_wrfout(path)
            try:
                slp = np.asarray(api.wrf.getvar(ds, "slp", units="hPa"))
                wspd = np.asarray(api.wrf.getvar(ds, "uvmet10_wspd_wdir", units="m s-1")[0])
            finally:
                ds.close()
            j, i = np.unravel_index(np.argmin(slp), slp.shape)
            times.append(time_str)
            pmin.append(float(slp[j, i]))
            vmax.append(float(wspd.max()))

    if not times:
        ctx.log("example_tc_track: no runs with frames found -- nothing to plot")
        return

    fig, ax1 = api.subplots(figsize=(8, 4))
    ax1.plot(times, pmin, "-o", color="tab:blue", markersize=3)
    ax1.set_ylabel("central pressure (hPa)", color="tab:blue")
    ax1.tick_params(axis="x", rotation=45, labelsize=7)
    ax2 = ax1.twinx()
    ax2.plot(times, vmax, "-s", color="tab:red", markersize=3)
    ax2.set_ylabel("max 10 m wind (m/s)", color="tab:red")
    ax1.set_title("storm-following central pressure & max wind (example script)")
    fig.tight_layout()
    ctx.save_figure(fig, "example_tc_track.png")
    ctx.save_table("example_tc_track.csv", ["time", "pmin_hpa", "vmax_ms"],
                   list(zip(times, pmin, vmax)))
'''


def _populate_case_dir(case_dir, dest_dir):
    """Copies only namelists; symlinks everything else found directly in case_dir. Never
    opens a file under case_dir for writing -- os.symlink and shutil.copy2 (read + fresh
    write at dest) both leave the source untouched, so the original case dir's mtime/sha256
    are unchanged (decision 0058 P1)."""
    for entry in sorted(os.listdir(case_dir)):
        src = os.path.join(case_dir, entry)
        dst = os.path.join(dest_dir, entry)
        if entry in NAMELIST_NAMES and os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)


def init_project(path, case_dir=None, name=None, storm=None, hurdat2=None, site=None):
    """`wtune project init PATH [--case CASE_DIR] [--name ...] [--storm ID] [--hurdat2 F]`
    (decision 0058 P1). PATH must not already exist, or must be empty."""
    path = os.path.abspath(path)
    if os.path.exists(path) and os.listdir(path):
        raise ProjectError("%s already exists and is not empty" % path)
    os.makedirs(path, exist_ok=True)
    for sub in SUBDIRS:
        os.makedirs(os.path.join(path, sub), exist_ok=True)

    if case_dir:
        case_dir = os.path.abspath(case_dir)
        if not os.path.isdir(case_dir):
            raise ProjectError("--case %s is not a directory" % case_dir)
        _populate_case_dir(case_dir, os.path.join(path, "case"))
    if hurdat2 and not os.path.isfile(hurdat2):
        raise ProjectError("--hurdat2 %s not found" % hurdat2)

    name = name or default_project_name(case_dir)
    project_doc = dict(
        name=name,
        created=_utc_now_iso(),
        case=dict(source_dir=case_dir) if case_dir else None,
        site=site,
        storm=(dict(id=storm, hurdat2=os.path.abspath(hurdat2) if hurdat2 else None)
              if (storm or hurdat2) else None),
        obs=dict(),
        time=dict(),
        cross_sections=dict(terrain=None, moisture=None),
        xsec_top_km=None,        # terrain default 10, moisture default 16 -- override here
        xsec_halfwidth_km=None,  # default 150 for both -- override here
    )
    with open(os.path.join(path, PROJECT_YAML), "w") as fh:
        yaml.safe_dump(project_doc, fh, sort_keys=False, allow_unicode=True)

    example_path = os.path.join(path, "analysis", "example_tc_track.py")
    with open(example_path, "w") as fh:
        fh.write(_EXAMPLE_SCRIPT)

    manifest_append(path, ["wtune", "project", "init", path], out_dir=path, exit_code=0,
                    key_paths=[case_dir] if case_dir else [])
    return path


def info_project(project_dir):
    doc = load_project(project_dir)
    lines = ["project: %s" % project_dir, "name: %s" % doc.get("name"),
             "created: %s" % doc.get("created")]
    case = doc.get("case") or {}
    if case.get("source_dir"):
        lines.append("case source: %s" % case["source_dir"])
    storm = doc.get("storm") or {}
    if storm.get("id"):
        lines.append("storm: %s (hurdat2=%s)" % (storm.get("id"), storm.get("hurdat2")))
    for sub in SUBDIRS:
        d = os.path.join(project_dir, sub)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        lines.append("%-9s %d entr(y/ies)" % (sub + "/", n))
    manifest_path = os.path.join(project_dir, MANIFEST_NAME)
    if os.path.isfile(manifest_path):
        with open(manifest_path) as fh:
            n_lines = sum(1 for _ in fh)
        lines.append("manifest.jsonl: %d entr(y/ies)" % n_lines)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(prog="wtune project")
    sub = parser.add_subparsers(dest="project_command")

    p_init = sub.add_parser("init", help="create a new project folder (decision 0058)")
    p_init.add_argument("path")
    p_init.add_argument("--case", dest="case_dir", default=None,
                        help="WRF case dir; namelists copied, everything else symlinked "
                             "(the source dir is never modified)")
    p_init.add_argument("--name", default=None, help="default: <case>-<YYYYmmdd-HHMM>")
    p_init.add_argument("--storm", default=None, help="e.g. AL122005")
    p_init.add_argument("--hurdat2", default=None, help="HURDAT2 file")
    p_init.add_argument("--site", default=None, help="machine/site label")

    p_info = sub.add_parser("info", help="print the resolved project's layout/config")
    p_info.add_argument("--project", default=None)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.project_command == "init":
        try:
            path = init_project(args.path, case_dir=args.case_dir, name=args.name,
                                storm=args.storm, hurdat2=args.hurdat2, site=args.site)
        except ProjectError as e:
            print("error: %s" % e, file=sys.stderr)
            return 1
        print("created project: %s" % path)
        return 0

    if args.project_command == "info":
        try:
            project_dir = find_project(explicit=args.project)
        except ProjectError as e:
            print("error: %s" % e, file=sys.stderr)
            return 1
        if not project_dir:
            print("error: no project found (--project, $WTUNE_PROJECT, or a project.yaml "
                 "above the current directory)", file=sys.stderr)
            return 1
        print(info_project(project_dir))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
