"""wtune/project.py -- `wtune project init|info|set` (decision 0058): a project folder that
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
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

import yaml

import wtune.machine
import wtune.version

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


def project_exclude_nodes(project_doc):
    """project.yaml's exclude_nodes (list of node names), [] when unset."""
    nodes = (project_doc or {}).get("exclude_nodes") or []
    if isinstance(nodes, str):
        nodes = nodes.split(",")
    return [n.strip() for n in nodes if n and n.strip()]


def project_post_defaults(project_doc):
    """project.yaml's post block: {"types", "every", "fetch"}; missing keys -> None ([] for
    fetch, the observation datasets `wtune submit` fetches before post)."""
    post = (project_doc or {}).get("post") or {}
    return dict(types=post.get("types"), every=post.get("every"), fetch=post.get("fetch") or [])


def project_case_id(project_doc):
    """The case id `wtune project init --case` registered, or None."""
    return ((project_doc or {}).get("case") or {}).get("id")


def project_case_table(project_dir):
    return os.path.join(project_dir, "cases.json")


def _base_case_table():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo, "campaign", "cases.json")


def merged_case_table(project_dir, case_id):
    """Writes <project>/.cases.merged.json -- today's campaign/cases.json plus the project's
    own case -- and returns its path. Rebuilt on every call so the shared table never goes
    stale; only the project's own entry is taken from <project>/cases.json."""
    with open(_base_case_table()) as fh:
        table = json.load(fh)
    with open(project_case_table(project_dir)) as fh:
        table["cases"][case_id] = json.load(fh)["cases"][case_id]
    path = os.path.join(project_dir, ".cases.merged.json")
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w") as fh:
        json.dump(table, fh, indent=1)
    os.replace(tmp, path)
    return path


def project_machine(project_doc):
    """project.yaml's machine block as {key: value}, only the keys it sets (wtune.machine has
    the built-in values). ProjectError on an unknown key or a bad value."""
    block = (project_doc or {}).get("machine") or {}
    if not isinstance(block, dict):
        raise ProjectError("project.yaml machine: must be a mapping")
    out = {}
    for key, value in block.items():
        where = "project.yaml machine.%s" % key
        if key not in wtune.machine.DEFAULTS:
            raise ProjectError("%s: unknown key (known: %s)"
                               % (where, ", ".join(wtune.machine.DEFAULTS)))
        if value is None:
            continue
        if key in ("cores_per_node", "max_nodes"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ProjectError("%s: %r must be a positive integer" % (where, value))
        elif key in ("partition", "account"):
            if not isinstance(value, str) or (key == "partition" and not value.strip()):
                raise ProjectError("%s: %r must be a name" % (where, value))
            value = value.strip()
        else:
            names = value.split(",") if isinstance(value, str) else value
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ProjectError("%s: %r must be a list of node names" % (where, value))
            value = [n.strip() for n in names if n.strip()]
        out[key] = value
    return out


def machine_line(project_doc):
    """'machine: cores_per_node=72 ...': the project's values over the built-in ones."""
    values = dict(wtune.machine.DEFAULTS, **project_machine(project_doc))
    values["login_nodes"] = ",".join(values["login_nodes"]) or "-"
    values["account"] = values["account"] or "-"
    return "machine: " + " ".join("%s=%s" % (k, values[k]) for k in wtune.machine.DEFAULTS)


def apply_project_env(project_dir, project_doc):
    """Exports the project settings the tuner, run and post read from the environment:
    exclude_nodes joins WTUNE_SBATCH_EXCLUDE, the machine block becomes the WTUNE_* variables
    of wtune.machine unless they are already set, and the case table with the project's case
    becomes CASES_JSON unless one is already set."""
    for key, value in project_machine(project_doc).items():
        name = wtune.machine.ENV[key]
        if os.environ.get(name) or (key == "login_nodes" and name in os.environ):
            continue                   # set by hand
        os.environ[name] = ",".join(value) if isinstance(value, list) else str(value)
    nodes = project_exclude_nodes(project_doc)
    if nodes:
        have = [n for n in os.environ.get("WTUNE_SBATCH_EXCLUDE", "").split(",") if n]
        os.environ["WTUNE_SBATCH_EXCLUDE"] = ",".join(have + [n for n in nodes if n not in have])
    case_id = project_case_id(project_doc)
    if case_id and os.path.isfile(project_case_table(project_dir)) \
            and not os.environ.get("CASES_JSON"):
        try:
            os.environ["CASES_JSON"] = merged_case_table(project_dir, case_id)
        except (OSError, ValueError, KeyError) as e:
            print("note: project case table not usable (%s: %s)" % (type(e).__name__, e),
                  file=sys.stderr)


def latest_tune_report(project_dir, case_id=None, log=None):
    """Newest <project>/tune/*/tune_report.json by mtime, or None. With case_id, only a report
    made for that case counts; each newer report skipped for another case is passed to log."""
    tune_dir = os.path.join(project_dir, "tune")
    names = os.listdir(tune_dir) if os.path.isdir(tune_dir) else []
    paths = [p for p in (os.path.join(tune_dir, n, "tune_report.json") for n in names)
             if os.path.isfile(p)]
    for path in sorted(paths, key=os.path.getmtime, reverse=True):
        if not case_id:
            return path
        try:
            with open(path) as fh:
                case = json.load(fh).get("case")
        except (OSError, ValueError, AttributeError):
            case = None
        if case == case_id:
            return path
        if log:
            log("tune report skipped: %s is for case %s, not %s" % (path, case or "?", case_id))
    return None


PHYSICS_KEYS = ("mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics",
                "sf_surface_physics", "ra_lw_physics", "ra_sw_physics", "radt", "cudt",
                "physics_suite")


def case_entry_from_namelist(case_dir):
    """A campaign/cases.json entry for an unregistered case, read from its namelist.input."""
    from wtune import predict as wp
    nl = wp.parse_namelist(os.path.join(case_dir, "namelist.input"))
    max_dom = int(wp.g(nl, "domains", "max_dom", 1))

    def per_dom(key):
        return [int(wp.g_dom(nl, "domains", key, d)) for d in range(max_dom)]

    e_we, e_sn, e_vert = per_dom("e_we"), per_dom("e_sn"), per_dom("e_vert")
    # the probe job (strict_campaign.sbatch) reads these d01 physics keys and case_family
    physics = {k: wp.g(nl, "physics", k) for k in PHYSICS_KEYS
               if wp.g(nl, "physics", k) is not None}
    return dict(
        prep_dir=case_dir, bp_dir="", restart=False, input_files=[],
        history_key="history_interval", dt=wp.g(nl, "domains", "time_step"),
        e_we=e_we[0], e_sn=e_sn[0], e_vert=e_vert[0],
        dx=wp.g(nl, "domains", "dx"), dy=wp.g(nl, "domains", "dy") or wp.g(nl, "domains", "dx"),
        max_dom=max_dom,
        e_we_list=e_we, e_sn_list=e_sn, e_vert_list=e_vert, physics=physics,
        case_family="unclassified", namelist_overrides={}, steps=64, enabled=True,
        notes="registered by wtune project init from %s" % case_dir)


def project_case_name(case_dir, namelist_path):
    """Case id for a project's case: the case dir's name (letters, digits and '-' only -- run
    ids are split on '_') plus 6 hex of sha1(source path + namelist). The measurement ledger is
    keyed by this id and shared by every project, so two different cases must never share one;
    the same case registered twice keeps its measurements."""
    name = re.sub(r"[^A-Za-z0-9]+", "-", os.path.basename(os.path.normpath(case_dir))).strip("-")
    h = hashlib.sha1(os.path.realpath(case_dir).encode())
    with open(namelist_path, "rb") as fh:
        h.update(fh.read())
    return "%s-%s" % (name or "case", h.hexdigest()[:6])


def register_project_case(project_dir, case_dir):
    """Writes the project's own entry to <project>/cases.json and returns its id."""
    namelist = os.path.join(project_dir, "case", "namelist.input")
    case_id = project_case_name(case_dir, namelist)
    table = {"cases": {case_id: case_entry_from_namelist(os.path.join(project_dir, "case"))}}
    with open(project_case_table(project_dir), "w") as fh:
        json.dump(table, fh, indent=1)
    return case_id


def discover_runs(project_dir):
    """[(run_id, path), ...] wrfout-bearing directories: every runs/<id>/ subdirectory that
    has at least one wrfout_* frame -- at top level, or in its seg_<k> segment dirs (a segmented
    `wtune run`; the run is listed once, its segments are not separate runs) -- else (when
    runs/ is empty) the case/ directory itself, labeled "case" -- decision 0058 item 2: "runs
    discovered from runs/*/ or case/". init no longer links a previous run's wrfout into case/,
    so "case" falls back to project.yaml's case.source_dir when that holds wrfout_d* frames."""
    from wtune.post import extract
    runs_root = os.path.join(project_dir, "runs")
    out = []
    if os.path.isdir(runs_root):
        for name in sorted(os.listdir(runs_root)):
            d = os.path.join(runs_root, name)
            if not os.path.isdir(d):
                continue
            if extract._wrfout_paths(d) or any(
                    extract._wrfout_paths(seg) for seg in extract.segment_dirs(d)):
                out.append((name, d))
    if not out:
        case_dir = os.path.join(project_dir, "case")
        if extract._wrfout_paths(case_dir):
            out.append(("case", case_dir))
        else:
            try:
                doc = load_project(project_dir)
            except (OSError, ValueError, yaml.YAMLError):
                doc = None
            case = doc.get("case") if isinstance(doc, dict) else None
            src = case.get("source_dir") if isinstance(case, dict) else None
            if src and extract._wrfout_paths(src):
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
        wtune_version=wtune.version.__version__,
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


POST_EVERY_MAX_RUN_DAYS = 7   # longer runs get no default --every (too many frames)
DEFAULT_FETCH = ("era5", "igra", "stage4", "stations")


def _default_post_block(case_dir):
    """post defaults for a new project: --types auto; for runs of at most
    POST_EVERY_MAX_RUN_DAYS also --every 6h and the observations the standard figures compare
    against."""
    from wtune import predict as wp
    every = None
    try:
        _steps, run_s, _dt, _note = wp.get_d01_steps(
            wp.parse_namelist(os.path.join(case_dir, "namelist.input")))
        if run_s and run_s <= POST_EVERY_MAX_RUN_DAYS * 86400:
            every = "6h"
    except (OSError, ValueError, wp.PredictError):
        pass
    return dict(types="auto", every=every, fetch=list(DEFAULT_FETCH) if every else [])


def init_project(path, case_dir=None, name=None, storm=None, hurdat2=None, site=None,
                 exclude_nodes=None):
    """`wtune project init PATH [--case CASE_DIR] [--name ...] [--storm ID] [--hurdat2 F]`
    (decision 0058 P1). PATH must not already exist, or must be empty. With --case, the case is
    registered in <project>/cases.json so `wtune tune` takes it without a case id.
    exclude_nodes defaults to $WTUNE_SBATCH_EXCLUDE."""
    path = os.path.abspath(path)
    if os.path.exists(path) and os.listdir(path):
        raise ProjectError("%s already exists and is not empty" % path)
    if case_dir:
        case_dir = os.path.abspath(case_dir)
        if not os.path.isdir(case_dir):
            raise ProjectError("--case %s is not a directory" % case_dir)
    if hurdat2 and not os.path.isfile(hurdat2):
        raise ProjectError("--hurdat2 %s not found" % hurdat2)
    os.makedirs(path, exist_ok=True)       # nothing is created before the arguments check out
    for sub in SUBDIRS:
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    if case_dir:
        skipped = _populate_case_dir(case_dir, os.path.join(path, "case"))
        if skipped:
            print("skipped %d output file(s) of a previous run in %s" % (skipped, case_dir))

    name = name or default_project_name(case_dir)
    case_id = None
    if case_dir and os.path.isfile(os.path.join(path, "case", "namelist.input")):
        try:
            case_id = register_project_case(path, case_dir)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print("note: case not registered for `wtune tune` (%s: %s)" % (type(e).__name__, e))
    if exclude_nodes is None:
        exclude_nodes = os.environ.get("WTUNE_SBATCH_EXCLUDE", "")
    project_doc = dict(
        name=name,
        created=_utc_now_iso(),
        case=dict(source_dir=case_dir, id=case_id) if case_dir else None,
        site=site,
        exclude_nodes=project_exclude_nodes(dict(exclude_nodes=exclude_nodes)),
        post=_default_post_block(os.path.join(path, "case")) if case_dir else dict(
            types="auto", every=None, fetch=[]),
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


def set_project_defaults(project_dir, exclude_nodes=None, post_every=None, post_types=None,
                         post_fetch=None, register_case=False, argv=None, machine=None):
    """`wtune project set --exclude-nodes/--post-every/--post-types/--post-fetch/
    --register-case` and the machine flags: updates those keys in place. "none" clears
    post_every and post_fetch. Returns the updated document."""
    doc = load_project(project_dir)
    if machine:
        doc["machine"] = project_machine(dict(machine=dict(doc.get("machine") or {}, **machine)))
    if register_case:
        case = dict(doc.get("case") or {})
        if not case.get("source_dir"):
            raise ProjectError("--register-case: project.yaml has no case.source_dir")
        try:
            case["id"] = register_project_case(project_dir, case["source_dir"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ProjectError("--register-case: cannot read case/namelist.input (%s: %s)"
                               % (type(e).__name__, e)) from e
        doc["case"] = case
    if exclude_nodes is not None:
        doc["exclude_nodes"] = project_exclude_nodes(dict(exclude_nodes=exclude_nodes))
    if post_every is not None or post_types is not None or post_fetch is not None:
        post = dict(doc.get("post") or {})
        if post_every is not None:
            post["every"] = None if post_every.lower() == "none" else post_every
        if post_types is not None:
            post["types"] = post_types
        if post_fetch is not None:
            post["fetch"] = ([] if post_fetch.lower() == "none" else
                             [n.strip() for n in post_fetch.split(",") if n.strip()])
        doc["post"] = post
    path = os.path.join(project_dir, PROJECT_YAML)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)
    manifest_append(project_dir, argv or ["wtune", "project", "set"], out_dir=project_dir,
                    exit_code=0)
    return doc


def info_project(project_dir):
    doc = load_project(project_dir)
    lines = ["project: %s" % project_dir, "name: %s" % doc.get("name"),
             "created: %s" % doc.get("created")]
    case = doc.get("case") or {}
    if case.get("source_dir"):
        lines.append("case source: %s" % case["source_dir"])
    if case.get("id"):
        lines.append("case id: %s (%s)" % (case["id"], project_case_table(project_dir)))
    storm = doc.get("storm") or {}
    if storm.get("id"):
        lines.append("storm: %s (hurdat2=%s)" % (storm.get("id"), storm.get("hurdat2")))
    lines.append("exclude_nodes: %s" % (",".join(project_exclude_nodes(doc)) or "-"))
    lines.append(machine_line(doc))
    post = project_post_defaults(doc)
    lines.append("post defaults: --types %s --every %s; submit fetches: %s"
                 % (post["types"] or "auto", post["every"] or "-", ",".join(post["fetch"]) or "-"))
    report = latest_tune_report(project_dir, case.get("id"))
    if report:
        lines.append("latest tune report: %s" % os.path.relpath(report, project_dir))
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
    p_init.add_argument("--exclude-nodes", default=None,
                        help="nodes every job of this project avoids, e.g. work1 "
                             "(default: $WTUNE_SBATCH_EXCLUDE)")

    p_info = sub.add_parser("info", help="print the resolved project's layout/config")
    p_info.add_argument("--project", default=None)

    p_set = sub.add_parser("set", help="update the resolved project's project.yaml")
    p_set.add_argument("--project", default=None)
    p_set.add_argument("--storm", default=None, help="e.g. AL122005")
    p_set.add_argument("--hurdat2", default=None, help="HURDAT2 file")
    p_set.add_argument("--exclude-nodes", default=None, help="comma list, '' clears")
    p_set.add_argument("--cores-per-node", type=int, default=None,
                       help="hardware cores of one compute node (built-in 72)")
    p_set.add_argument("--max-nodes", type=int, default=None,
                       help="compute nodes one job may use (built-in 13)")
    p_set.add_argument("--partition", default=None, help="Slurm partition (built-in compute)")
    p_set.add_argument("--login-nodes", default=None,
                       help="nodes no job is sent to, comma list, '' = none (built-in head)")
    p_set.add_argument("--account", default=None,
                       help="Slurm account of every job, '' = none (built-in none)")
    p_set.add_argument("--post-every", default=None, help="e.g. 6h, 1d; 'none' clears")
    p_set.add_argument("--post-types", default=None, help="'auto' or e.g. 'tc,convection'")
    p_set.add_argument("--post-fetch", default=None,
                       help="observations `wtune submit` fetches, e.g. era5,stage4; 'none' clears")
    p_set.add_argument("--register-case", action="store_true",
                       help="register case/ in <project>/cases.json for `wtune tune` "
                            "(projects made before this existed)")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.project_command == "init":
        try:
            path = init_project(args.path, case_dir=args.case_dir, name=args.name,
                                storm=args.storm, hurdat2=args.hurdat2, site=args.site,
                                exclude_nodes=args.exclude_nodes)
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
            full_argv = ["wtune", "project"] + list(argv if argv is not None else sys.argv[1:])
            defaults = (args.exclude_nodes, args.post_every, args.post_types, args.post_fetch)
            machine = {k: getattr(args, k) for k in wtune.machine.DEFAULTS
                       if getattr(args, k) is not None}
            if any(v is not None for v in defaults) or args.register_case or machine:
                doc = set_project_defaults(project_dir, *defaults, machine=machine,
                                           register_case=args.register_case, argv=full_argv)
                if machine:
                    print(machine_line(doc))
                post = project_post_defaults(doc)
                print("%s: case id=%s exclude_nodes=%s post --types %s --every %s"
                      % (os.path.join(project_dir, PROJECT_YAML), project_case_id(doc) or "-",
                         ",".join(project_exclude_nodes(doc)) or "-", post["types"] or "auto",
                         post["every"] or "-"))
                if not (args.storm or args.hurdat2):
                    return 0
            block = set_project(project_dir, storm=args.storm, hurdat2=args.hurdat2,
                                argv=full_argv)
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
        try:
            print(info_project(project_dir))
        except ProjectError as e:
            print("error: %s" % e, file=sys.stderr)
            return 1
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
