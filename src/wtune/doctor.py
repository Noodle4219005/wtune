"""wtune/doctor.py -- `wtune doctor`: is this machine and project ready for tune/run/post?

One line per check: OK, WARN (a feature will be skipped) or FAIL (a step cannot run), each
FAIL/WARN with the command that fixes it. Exit status 1 when any check FAILs."""
from __future__ import annotations

import argparse
import os
import shutil
import sys

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
    wrf_tree = os.environ.get("WRF_TREE")
    if not wrf_tree:
        out.append(("FAIL", "WRF_TREE is not set", "source ~/wtune-scc/env.sh"))
    elif not os.path.isfile(os.path.join(wrf_tree, "main", "wrf.exe")):
        out.append(("FAIL", "no main/wrf.exe under WRF_TREE=%s" % wrf_tree, "build WRF first"))
    else:
        out.append(("OK", "wrf.exe: %s/main/wrf.exe" % wrf_tree, ""))
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
    for level, what, fix in checks:
        print("%-4s  %s%s" % (level, what, "\n      fix: %s" % fix if fix else ""))
    return 1 if any(level == "FAIL" for level, _w, _f in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
