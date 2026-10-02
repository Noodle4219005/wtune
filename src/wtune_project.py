"""wtune_project.py -- `wtune project init|info|set` (decision 0058): a project folder that
collects one WRF case's definition/inputs/config, every wtune tune/predict-rebuild/run
output, and every post-processing figure/report under one user-named directory, so nothing
has to be tracked by hand across separate --out flags.

Layout (decision 0058 section 1):
  project.yaml   case/site/storm/obs/time-window description
  wtune.yaml     unchanged tuner config (wtune config init writes this; not touched here)
  case/          namelist.input/namelist.output/namelist.wps copied (user edits these);
                 everything else found directly in the source case dir (wrfinput, wrfbdy,
                 met_em, ...) is symlinked, never copied -- the source case dir is read-only
                 to this tool (never opened for writing). Outputs of a previous run there
                 (wrfout_*, wrfrst_*, rsl.out.*, rsl.error.*, *.log, namelist.output) are not
                 linked.
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
# Outputs of a previous WRF run in the --case dir (climate-ea50 was initialised from a former run
# dir: 397 old wrfout files got linked into case/, plus rsl.*/wrfrst*/namelist.output). They are
# not case inputs -- never linked. namelist.output only when namelist.input is there (else it is
# the one namelist the case has).
_PREV_RUN_OUTPUT_PREFIXES = ("wrfout_", "wrfrst_", "rsl.out.", "rsl.error.")
_PREV_RUN_OUTPUT_NAMES = ("namelist.output",)
_PREV_RUN_OUTPUT_SUFFIXES = (".log",)


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
    has at least one wrfout_* frame -- at top level, or in its seg_<k> segment dirs (a segmented
    `wtune run`; the run is listed once, its segments are not separate runs) -- else (when
    runs/ is empty) the case/ directory itself, labeled "case" -- decision 0058 item 2: "runs
    discovered from runs/*/ or case/". init no longer links a previous run's wrfout into case/,
    so "case" falls back to project.yaml's case.source_dir when that holds wrfout_d* frames."""
    from post import extract
    runs_root = os.path.join(project_dir, "runs")
    out = []
    if os.path.isdir(runs_root):
        for name in sorted(os.listdir(runs_root)):
            d = os.path.join(runs_root, name)
            if not os.path.isdir(d):
                continue
            if glob.glob(os.path.join(d, "wrfout_d*")) or any(
                    glob.glob(os.path.join(seg, "wrfout_d*")) for seg in extract.segment_dirs(d)):
                out.append((name, d))
    if not out:
        case_dir = os.path.join(project_dir, "case")
        if glob.glob(os.path.join(case_dir, "wrfout_d*")):
            out.append(("case", case_dir))
        else:
            try:
                doc = load_project(project_dir)
            except (OSError, ValueError, yaml.YAMLError):
                doc = None
            case = doc.get("case") if isinstance(doc, dict) else None
            src = case.get("source_dir") if isinstance(case, dict) else None
            if src and glob.glob(os.path.join(src, "wrfout_d*")):
                out.append(("case", src))
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
        for time_str, path, timeidx in ctx.list_frame_refs(run_dir, domain=domain):
            ds = ctx.open_wrfout(path)
            try:
                slp = np.asarray(api.wrf.getvar(ds, "slp", timeidx=timeidx, units="hPa"))
                wspd = np.asarray(api.wrf.getvar(ds, "uvmet10_wspd_wdir", timeidx=timeidx, units="m s-1")[0])
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


def is_previous_run_output(name, entries=()):
    """True for a file a previous WRF run wrote into its run dir (wrfout_*, wrfrst_*,
    rsl.out.*, rsl.error.*, *.log, and namelist.output when namelist.input is in `entries`)."""
    if name in _PREV_RUN_OUTPUT_NAMES:
        return "namelist.input" in entries
    return name.startswith(_PREV_RUN_OUTPUT_PREFIXES) or name.endswith(_PREV_RUN_OUTPUT_SUFFIXES)


def _populate_case_dir(case_dir, dest_dir):
    """Copies only namelists; symlinks everything else found directly in case_dir except the
    outputs of a previous run (is_previous_run_output). Never opens a file under case_dir for
    writing -- os.symlink and shutil.copy2 (read + fresh write at dest) both leave the source
    untouched, so the original case dir's mtime/sha256 are unchanged (decision 0058 P1).
    Returns the number of skipped previous-run outputs."""
    entries = sorted(os.listdir(case_dir))
    skipped = 0
    for entry in entries:
        src = os.path.join(case_dir, entry)
        dst = os.path.join(dest_dir, entry)
        if is_previous_run_output(entry, entries):
            skipped += 1
        elif entry in NAMELIST_NAMES and os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
    return skipped


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
        skipped = _populate_case_dir(case_dir, os.path.join(path, "case"))
        if skipped:
            print("skipped %d output file(s) of a previous run in %s" % (skipped, case_dir))
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


def set_project(project_dir, storm=None, hurdat2=None, argv=None):
    """`wtune project set [--project DIR] [--storm ID] [--hurdat2 F]`: updates project.yaml's
    storm block in place (every other key and the key order are kept) and records the command
    in manifest.jsonl. Returns the updated storm dict."""
    if not (storm or hurdat2):
        raise ProjectError("nothing to set (give --storm and/or --hurdat2)")
    if hurdat2 and not os.path.isfile(hurdat2):
        raise ProjectError("--hurdat2 %s not found" % hurdat2)
    doc = load_project(project_dir)
    block = dict(doc.get("storm") or {})
    block.setdefault("id", None)
    block.setdefault("hurdat2", None)
    if storm:
        block["id"] = storm
    if hurdat2:
        block["hurdat2"] = os.path.abspath(hurdat2)
    doc["storm"] = block  # an existing key keeps its position; a missing one is appended
    path = os.path.join(project_dir, PROJECT_YAML)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)
    manifest_append(project_dir, argv or ["wtune", "project", "set"], out_dir=project_dir,
                    exit_code=0, key_paths=[block["hurdat2"]] if hurdat2 else [])
    return block


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

    p_set = sub.add_parser("set", help="update the resolved project's project.yaml storm block")
    p_set.add_argument("--project", default=None)
    p_set.add_argument("--storm", default=None, help="e.g. AL122005")
    p_set.add_argument("--hurdat2", default=None, help="HURDAT2 file")

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

    if args.project_command == "set":
        try:
            project_dir = find_project(explicit=args.project)
            if not project_dir:
                raise ProjectError("no project found (--project, $WTUNE_PROJECT, or a "
                                   "project.yaml above the current directory)")
            block = set_project(project_dir, storm=args.storm, hurdat2=args.hurdat2,
                                argv=["wtune", "project"] + list(argv if argv is not None
                                                                 else sys.argv[1:]))
        except ProjectError as e:
            print("error: %s" % e, file=sys.stderr)
            return 1
        print("%s: storm id=%s hurdat2=%s" % (os.path.join(project_dir, PROJECT_YAML),
                                              block.get("id"), block.get("hurdat2")))
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
