"""wtune/doctor.py -- `wtune doctor`: is this machine and project ready for tune/run/post?

One line per check: OK, WARN (a feature will be skipped) or FAIL (a step cannot run), each
FAIL/WARN with the command that fixes it. Exit status 1 when any check FAILs. `--json` prints
the same checks for an agent; `--find` also searches the disk for WRF trees, WPS, the geog data
and the libraries wrf.exe cannot load, so the paths can be set without asking anyone."""
from __future__ import annotations

import argparse
import json
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

import wtune.machine
import wtune.project

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env_dir(name: str) -> str:
    return os.path.join(REPO_ROOT, "envs", name, ".pixi")


def machine_checks() -> list[tuple[str, str, str]]:
    """[(level, what, fix), ...] for the tool chain."""
    out = []
    pixi = shutil.which("pixi") or (os.path.expanduser("~/.pixi/bin/pixi")
                                    if os.path.exists(os.path.expanduser("~/.pixi/bin/pixi"))
                                    else None)
    out.append(("OK", "pixi: %s" % pixi, "") if pixi else
               ("FAIL", "pixi not found", "curl -fsSL https://pixi.sh/install.sh | bash"))
    for env, level, why in (("wtune", "FAIL", "wtune itself"),
                            ("adios2", "FAIL", "ADIOS2 BP conversion")):
        path = _env_dir(env)
        out.append(("OK", "envs/%s ready" % env, "") if os.path.isdir(path) else
                   (level, "envs/%s not installed (%s)" % (env, why),
                    "pixi install --manifest-path %s/envs/%s/pixi.toml" % (REPO_ROOT, env)))
    for tool in ("sbatch", "srun", "sacct"):
        out.append(("OK", "%s on PATH" % tool, "") if shutil.which(tool) else
                   ("FAIL", "%s not on PATH (no Slurm here?)" % tool, "run on the login node"))
    out += path_checks()
    from wtune.post import tools
    from wtune.post.recipes import paraview3d
    for tool, what, find in (("grid_stat", "MET verification figures",
                              lambda: tools.find_postproc_tool("grid_stat")),
                             ("pvbatch", "ParaView 3-D figures", paraview3d.find_pvbatch)):
        try:
            out.append(("OK", "%s: %s" % (tool, find()), ""))
        except tools.ToolNotFoundError:
            out.append(("WARN", "%s not found: %s are skipped" % (tool, what),
                        "bin/install-postproc.sh (README 'Installation')"))
    return out


GEOG_MARKERS = ("topo_gmted2010_30s", "landuse_30s", "soiltype_top_30s",
                "modis_landuse_20class_30s")
FIND_SKIP = {".git", ".pixi", ".spack-env", "__pycache__", "node_modules", ".cache", "runs",
             "post", "tune"}


def _sourced(env_file: str, site_file: str, names) -> dict:
    """The values of `names` after sourcing the env file and then the site file in bash, `names`
    unset first so what comes back is theirs, plus `ldd_missing`: what wrf.exe cannot load."""
    script = ("unset %s; source %s >/dev/null 2>&1; CAMPAIGN=%s; [ -f %s ] && source %s "
              ">/dev/null 2>&1; for v in %s; do printf '%%s=%%s\\n' $v \"${!v}\"; done; "
              "w=$WRF_TREE/main/wrf.exe; [ -x \"$w\" ] && command -v ldd >/dev/null && "
              "ldd \"$w\" 2>/dev/null | awk '/not found/ {print \"MISSING=\" $1}'"
              % (" ".join(names), shlex.quote(env_file),
                 shlex.quote(os.path.join(REPO_ROOT, "campaign")), shlex.quote(site_file),
                 shlex.quote(site_file), " ".join(names)))
    try:
        text = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                              timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    got = {"ldd_missing": []}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key == "MISSING":
            got["ldd_missing"].append(value)
        elif key in names:
            got[key] = value
    return got


def sourced_paths() -> dict:
    """What a job sees: the machine's env file, then its site file (see _sourced)."""
    env_file = _expand(wtune.machine.env_source())
    if not os.path.isfile(env_file):
        return {}
    site_file = os.path.join(REPO_ROOT, "campaign", "sites", wtune.machine.site() + ".sh")
    return _sourced(env_file, site_file, ("WRF_TREE", "WPS_DIR", "PIXI_ENV",
                                          "WTUNE_CARTOPY_DATA"))


def _expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path)) if path else ""


def _same(a: str, b: str) -> bool:
    return bool(a) and bool(b) and os.path.realpath(a) == os.path.realpath(b)


def sinfo_partitions():
    """The partitions `sinfo` lists, None without Slurm, an empty set when sinfo fails."""
    if not shutil.which("sinfo"):
        return None
    try:
        text = subprocess.run(["sinfo", "-h", "-o", "%P"], capture_output=True, text=True,
                              timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {p.rstrip("*") for p in text.split()}


def path_checks() -> list[tuple[str, str, str]]:
    """The paths a job needs: site file, partition, env file, WRF (and the libraries wrf.exe
    loads after the env file), WPS, geog, the site's adios2 python env, Natural Earth."""
    out = []
    site = wtune.machine.site()
    sites = os.path.join(REPO_ROOT, "campaign", "sites")
    known = sorted(f[:-3] for f in os.listdir(sites) if f.endswith(".sh"))
    out.append(("OK", "site file: campaign/sites/%s.sh" % site, "") if site in known else
               ("FAIL", "no campaign/sites/%s.sh (sites: %s)" % (site, ", ".join(known)),
                "wtune project set --user --site <one of %s>, or copy the closest one to "
                "campaign/sites/%s.sh and edit it" % (",".join(known), site)))
    have = sinfo_partitions()
    if have is not None:
        want = wtune.machine.partitions()
        bad = [p for p in want if have and p not in have]
        out.append(("FAIL", "partition %s not in sinfo (%s)" % (",".join(bad), ",".join(
            sorted(have))), "wtune project set --user --partition <one of them>") if bad else
                   ("OK", "partition: %s" % ",".join(want), ""))
    env_file = _expand(wtune.machine.env_source())
    if not os.path.isfile(env_file):
        out.append(("FAIL", "env file %s does not exist (every job sources it)" % env_file,
                    "write it (WRF_TREE, WPS_DIR, the library paths of WRF's build; skill "
                    "wtune-operate 'Paths'), then wtune project set --user --env-source FILE"))
    else:
        out.append(("OK", "env file: %s" % env_file, ""))
    sourced = sourced_paths()
    wrf_tree = _expand(wtune.machine.wrf_tree())
    file_tree = sourced.get("WRF_TREE", "")
    for tree, where in ((wrf_tree, "machine wrf_tree / WRF_TREE"), (file_tree, "env file")):
        if tree and (where == "env file" or not _same(tree, file_tree)) and not os.path.isfile(os.path.join(tree, "main", "wrf.exe")):
            out.append(("FAIL", "no main/wrf.exe under %s (%s)" % (tree, where),
                        "wtune doctor --find lists WRF trees; then wtune project set --user "
                        "--wrf-tree DIR and the same WRF_TREE in %s" % env_file))
    if os.path.isfile(env_file) and not file_tree:
        out.append(("FAIL", "%s does not set WRF_TREE (tune jobs take it from there)" % env_file,
                    "add `export WRF_TREE=%s` to it" % (wrf_tree or "<tree>")))
    elif file_tree and wrf_tree and not _same(file_tree, wrf_tree):
        out.append(("FAIL", "WRF_TREE differs: %s in the env file, %s from machine.yaml or the "
                    "shell; tune jobs use the first, wtune run the second" % (file_tree, wrf_tree),
                    "make them the same: wtune project set --user --wrf-tree DIR and "
                    "`export WRF_TREE=DIR` in %s" % env_file))
    tree = file_tree or wrf_tree
    if tree and os.path.isfile(os.path.join(tree, "main", "wrf.exe")):
        for exe in ("real.exe",):
            if not os.path.isfile(os.path.join(tree, "main", exe)):
                out.append(("FAIL", "no main/%s under %s" % (exe, tree), "build WRF em_real"))
        missing = sourced.get("ldd_missing") or []
        out.append(("FAIL", "wrf.exe cannot load %s after sourcing %s" % (" ".join(missing),
                                                                         env_file),
                    "wtune doctor --find shows where they are; add those directories to "
                    "LD_LIBRARY_PATH in %s" % env_file) if missing else
                   ("OK", "wrf.exe: %s/main/wrf.exe" % tree, ""))
    wps = sourced.get("WPS_DIR") or os.environ.get("WPS_DIR", "")
    exes = ("geogrid.exe", "ungrib.exe", "metgrid.exe")
    if wps and all(os.path.exists(os.path.join(wps, e)) for e in exes):
        out.append(("OK", "WPS: %s" % wps, ""))
    else:
        out.append(("WARN", "no WPS (geogrid/ungrib/metgrid) at WPS_DIR=%s: prep wps and "
                    "predict-rebuild cannot run" % (wps or "<unset>"),
                    "wtune doctor --find lists WPS builds; `export WPS_DIR=DIR` in %s" % env_file))
    geog = _expand(wtune.machine.geog())
    if geog and os.path.isdir(geog) and any(os.path.isdir(os.path.join(geog, m))
                                            for m in GEOG_MARKERS):
        out.append(("OK", "geog: %s" % geog, ""))
    else:
        out.append(("WARN", "no WPS geog data at %s: prep wps cannot run" % (geog or "<unset>"),
                    "wtune doctor --find lists geog folders; wtune project set --user --geog DIR "
                    "(or wtune prep geog fetch)"))
    pixi_env = sourced.get("PIXI_ENV")
    if pixi_env and not os.path.isfile(os.path.join(pixi_env, "pixi.toml")):
        out.append(("FAIL", "no pixi.toml at the site's PIXI_ENV=%s: probe and conversion jobs "
                    "need a python with adios2 there" % pixi_env,
                    "`export WTUNE_PIXI_ENV=%s` in %s and in the shell that runs wtune"
                    % (os.path.join(REPO_ROOT, "envs", "adios2"), env_file)))
    elif pixi_env:
        out.append(("OK", "site python env: %s" % pixi_env, ""))
    cartopy = sourced.get("WTUNE_CARTOPY_DATA") or os.environ.get("WTUNE_CARTOPY_DATA")
    if cartopy and os.path.isdir(cartopy) and not glob.glob(os.path.join(
            cartopy, "shapefiles", "natural_earth", "physical", "ne_*_coastline.shp")):
        out.append(("WARN", "WTUNE_CARTOPY_DATA %s is a folder but has no coastline shapefile "
                    "(shapefiles/natural_earth/physical/ne_*_coastline.shp): maps on compute nodes "
                    "without internet fall back to the land mask" % cartopy,
                    "bin/fetch-natural-earth.sh, then `export WTUNE_CARTOPY_DATA=%s` in %s"
                    % (os.path.join(REPO_ROOT, "envs", "cartopy-data"), env_file)))
        return out
    out.append(("OK", "Natural Earth: %s" % cartopy, "") if cartopy and os.path.isdir(cartopy)
               else ("WARN", "WTUNE_CARTOPY_DATA %s: maps on compute nodes without internet "
                     "have no coastlines" % ("not set" if not cartopy else
                                             "is not a folder (%s)" % cartopy),
                     "bin/fetch-natural-earth.sh, then `export WTUNE_CARTOPY_DATA=%s` in %s"
                     % (os.path.join(REPO_ROOT, "envs", "cartopy-data"), env_file)))
    return out


def find_paths(roots, libs=(), seconds=120.0, depth=8) -> dict:
    """Walk `roots` (no symlinked folders, at most `depth` levels, at most `seconds`) for WRF
    trees (main/wrf.exe), WPS builds (geogrid.exe), geog folders and the files named in
    `libs`. Each list keeps the order found."""
    found = {"wrf_tree": [], "wps_dir": [], "geog": [], "libs": {name: [] for name in libs},
             "env_files": [], "stopped": False}
    t0 = time.monotonic()
    for root in roots:
        root = os.path.expanduser(root)
        base = root.rstrip(os.sep).count(os.sep)
        for here, dirs, files in os.walk(root):
            if time.monotonic() - t0 > seconds:
                found["stopped"] = True
                return found
            if here.count(os.sep) - base >= depth:
                dirs[:] = []
            dirs[:] = [d for d in dirs if d not in FIND_SKIP and not d.startswith("wrfout")]
            name = os.path.basename(here)
            if name == "main" and "wrf.exe" in files:
                found["wrf_tree"].append(os.path.dirname(here))
            if "geogrid.exe" in files and "metgrid.exe" in files:
                found["wps_dir"].append(here)
            if any(m in dirs for m in GEOG_MARKERS):
                found["geog"].append(here)
            for f in files:
                if f in found["libs"]:
                    found["libs"][f].append(here)
                elif f.startswith("env") and f.endswith(".sh") and "wtune" in here:
                    path = os.path.join(here, f)
                    if _sets_wrf_tree(path):
                        found["env_files"].append(path)
    return found


def _sets_wrf_tree(path) -> bool:
    """Whether the shell file assigns WRF_TREE, which every env file must (a name match alone
    also took atlas-f1arm-20261014/_bin/envcheck.sh)."""
    try:
        with open(path, errors="replace") as fh:
            text = fh.read(65536)
    except OSError:
        return False
    return re.search(r"(^|[\s;])(export\s+)?WRF_TREE=", text, re.M) is not None


def project_checks(project_dir: str) -> list[tuple[str, str, str]]:
    doc = wtune.project.load_project(project_dir)
    out = [("OK", "project: %s" % project_dir, "")]
    case_id = wtune.project.project_case_id(doc)
    out.append(("OK", "case registered for tune: %s" % case_id, "") if case_id else
               ("WARN", "case not registered: `wtune tune` needs a case id",
                "wtune project set --register-case"))
    nodes = wtune.project.project_exclude_nodes(doc)
    out.append(("OK", "jobs exclude: head%s" % ("," + ",".join(nodes) if nodes else ""), ""))
    post = wtune.project.project_post_defaults(doc)
    out.append(("OK", "post defaults: --types %s --every %s"
                % (post["types"] or "auto", post["every"] or "-"), ""))
    report = wtune.project.latest_tune_report(project_dir, case_id)
    out.append(("OK", "newest tune report: %s" % os.path.relpath(report, project_dir), "")
               if report else ("OK", "no tune report yet (wtune tune, or wtune submit)", ""))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wtune doctor", description=__doc__.split("\n")[0])
    ap.add_argument("--project", default=None)
    ap.add_argument("--json", action="store_true", help="the checks as JSON, for an agent")
    ap.add_argument("--find", action="store_true",
                    help="also search the disk for WRF trees, WPS, geog data and the libraries "
                         "wrf.exe cannot load")
    ap.add_argument("--find-root", action="append", default=None, metavar="DIR",
                    help="where --find looks (repeatable; default: $HOME)")
    ap.add_argument("--find-seconds", type=float, default=120.0)
    args = ap.parse_args(argv)
    checks = machine_checks()
    try:
        project_dir = wtune.project.find_project(explicit=args.project)
    except wtune.project.ProjectError as e:
        checks.append(("FAIL", str(e), ""))
        project_dir = None
    if project_dir:
        checks += project_checks(project_dir)
    else:
        checks.append(("OK", "no project here (wtune project init DIR --case CASE_DIR)", ""))
    found = None
    if args.find:
        found = find_paths(args.find_root or ["~"], sourced_paths().get("ldd_missing") or (),
                           args.find_seconds)
    status = 1 if any(level == "FAIL" for level, _w, _f in checks) else 0
    if args.json:
        print(json.dumps({"ok": status == 0,
                          "checks": [{"level": lv, "what": w, "fix": f} for lv, w, f in checks],
                          "found": found}, indent=1))
        return status
    for level, what, fix in checks:
        print("%-4s  %s%s" % (level, what, "\n      fix: %s" % fix if fix else ""))
    if found:
        print("\nfound%s:" % (" (search stopped after %.0f s)" % args.find_seconds
                              if found["stopped"] else ""))
        for key in ("wrf_tree", "wps_dir", "geog", "env_files"):
            for path in found[key]:
                print("  %-9s %s" % (key, path))
        for lib, dirs in found["libs"].items():
            for path in dirs or ["(not found)"]:
                print("  %-9s %s: %s" % ("lib", lib, path))
    return status


if __name__ == "__main__":
    sys.exit(main())
