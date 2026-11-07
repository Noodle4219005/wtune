"""Small-domain proxy: rank layouts for an N-node run by timing them on ONE node.

Each target layout (px x py ranks on N nodes, omp threads, numtiles) is reproduced on one node with
ranks/N ranks, on a box cropped from the case's wrfinput so that every rank owns the same patch as
in the target. Validated on conus25 (Grace, layout order of N=4/N=8 ledger, s/step within 2 %) and
karina (x86, hybrid and numtiles effects of a 4-node ABBA truth). It ranks; it is not a wall-time
predictor (its bias is machine dependent) and it does not see inter-node effects (cyclic
placement, network). Nested cases are refused: frozen boundaries made marshall's nest unstable.

  wtune proxy plan   CASE --nodes N --layout 18x64 [--layout 9x32:omp=2:nt=24 ...]
  wtune proxy submit CASE --nodes N --layout ... [--steps 40] [--reps 2] [--out DIR] [--dry-run]
  wtune proxy collect DIR
"""
import argparse
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys

STEP_RE = re.compile(r"Timing for main:.*on domain\s+1:\s+([0-9.]+)\s+elapsed")


# ----------------------------------------------------------------------------------- namelist
def nl_get(text, key):
    """First value of `key` (1st domain) or None; tolerant of several key=value pairs per line."""
    m = re.search(r"(?<![A-Za-z0-9_])%s\s*=\s*([^,\n/]+)" % re.escape(key), text)
    return m.group(1).strip().strip("'\"") if m else None


def nl_set(text, **kv):
    """Replace each key's whole value list (all domains) with one value (pass it without a trailing
    comma; the existing one stays); nproc_x/nproc_y/numtiles are added to &domains when absent.
    Any other absent key is an error."""
    missing = []
    for k, v in kv.items():
        pat = re.compile(r"(?<![A-Za-z0-9_])%s\s*=\s*[^=\n/]*?(?=,?\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*=|\n|/))"
                         % re.escape(k))
        new, n = pat.subn("%s = %s" % (k, v), text, count=1)
        if n:
            text = new
        else:
            missing.append(k)
    add = [k for k in missing if k in ("nproc_x", "nproc_y", "numtiles")]
    bad = [k for k in missing if k not in add]
    if bad:
        raise KeyError("not in namelist: %s" % ", ".join(bad))
    if add:
        text = re.sub(r"(&domains[^\n]*\n)", lambda m: m.group(1) + "".join(
            " %s = %s,\n" % (k, kv[k]) for k in add), text, count=1, flags=re.I)
    return text


def time_step_seconds(text):
    dt = float(nl_get(text, "time_step") or 0)
    num, den = nl_get(text, "time_step_fract_num"), nl_get(text, "time_step_fract_den")
    if num and den and float(den):
        dt += float(num) / float(den)
    return dt


# ------------------------------------------------------------------------------------ layouts
def parse_layout(spec):
    """'18x64' or '9x32:omp=2:nt=24' -> dict(px, py, omp, numtiles)."""
    head, *opts = spec.split(":")
    px, py = (int(x) for x in head.lower().split("x"))
    d = dict(px=px, py=py, omp=1, numtiles=1)
    for o in opts:
        k, v = o.split("=")
        d[{"omp": "omp", "nt": "numtiles", "numtiles": "numtiles"}[k]] = int(v)
    return d


def proxy_shape(e_we, e_sn, px, py, nodes, cores_per_node, omp=1):
    """Proxy for a px x py layout on `nodes` nodes: dict(ranks, a, b, nx, ny) -- a x b ranks on one
    node and an nx x ny mass-point box giving each rank the target's patch -- or None when the
    target cannot be reproduced on one node (ranks not divisible by nodes, box bigger than the
    domain, or more than a node's cores)."""
    mx, my = e_we - 1, e_sn - 1
    if (px * py) % nodes:
        return None
    rp = px * py // nodes
    if rp * omp > cores_per_node:
        return None
    pxm, pym = math.ceil(mx / px), math.ceil(my / py)
    best = None
    for a in range(1, rp + 1):
        if rp % a:
            continue
        b = rp // a
        nx, ny = pxm * a, pym * b
        if nx > mx or ny > my:
            continue
        score = abs(math.log((a / b) / (px / py)))
        if best is None or score < best[0]:
            best = (score, a, b, nx, ny)
    if best is None:
        return None
    _, a, b, nx, ny = best
    return dict(ranks=rp, a=a, b=b, nx=nx, ny=ny)


# --------------------------------------------------------------------------------------- crop
def crop(src_in, src_bdy, out_dir, nx, ny):
    """Centre-crop wrfinput_d01 to nx x ny mass points and synthesize wrfbdy_d01 from the crop's
    edges (values mass-coupled as real.exe writes them, zero tendencies). Fields wrfbdy carries
    but wrfinput lacks (e.g. HT_SHAD) are zero. Returns the zero-filled field names."""
    import datetime as dt
    import netCDF4 as nc
    import numpy as np
    S = nc.Dataset(src_in)
    i0 = (len(S.dimensions["west_east"]) - nx) // 2
    j0 = (len(S.dimensions["south_north"]) - ny) // 2
    size = {"west_east": nx, "west_east_stag": nx + 1, "south_north": ny, "south_north_stag": ny + 1}
    off = {"west_east": i0, "west_east_stag": i0, "south_north": j0, "south_north_stag": j0}

    def define(src, dst):
        # every variable before the first write: a classic file re-lays out its data section
        # each time a variable is defined after data exists
        for d, v in src.dimensions.items():
            dst.createDimension(d, None if v.isunlimited() else size.get(d, len(v)))
        for a in src.ncattrs():
            dst.setncattr(a, src.getncattr(a))
        for k, val in (("WEST-EAST_GRID_DIMENSION", nx + 1), ("SOUTH-NORTH_GRID_DIMENSION", ny + 1),
                       ("WEST-EAST_PATCH_END_UNSTAG", nx), ("WEST-EAST_PATCH_END_STAG", nx + 1),
                       ("SOUTH-NORTH_PATCH_END_UNSTAG", ny), ("SOUTH-NORTH_PATCH_END_STAG", ny + 1)):
            if k in src.ncattrs() or k.endswith("GRID_DIMENSION"):
                dst.setncattr(k, np.int32(val))
        out = {}
        for name, v in src.variables.items():
            ov = dst.createVariable(name, v.dtype, v.dimensions)
            for a in v.ncattrs():
                if a != "_FillValue":
                    ov.setncattr(a, v.getncattr(a))
            out[name] = ov
        dst.set_fill_off()
        return out

    O = nc.Dataset(os.path.join(out_dir, "wrfinput_d01"), "w", format="NETCDF3_64BIT_OFFSET")
    ovs = define(S, O)
    for name, v in S.variables.items():
        v.set_auto_mask(False)
        idx = tuple(slice(off[d], off[d] + size[d]) if d in size else slice(None) for d in v.dimensions)
        full = v[...]        # one sequential read, slice in memory
        ovs[name][...] = np.ascontiguousarray(full[idx]) if v.dimensions else full
    O.close()
    S.close()

    B = nc.Dataset(src_bdy)
    I = nc.Dataset(os.path.join(out_dir, "wrfinput_d01"))
    for v in I.variables.values():
        v.set_auto_mask(False)
    g = lambda n: I.variables[n][0].astype("f8")
    mu = g("MU") + g("MUB")
    mu_u = np.concatenate([mu[:, :1], 0.5 * (mu[:, 1:] + mu[:, :-1]), mu[:, -1:]], axis=1)
    mu_v = np.concatenate([mu[:1], 0.5 * (mu[1:] + mu[:-1]), mu[-1:]], axis=0)

    def couple(base, f):
        # real.exe stores lateral boundaries coupled with the total dry-air mass; MU itself is not
        f = f.astype("f8")
        if base == "MU":
            return f
        if base == "U":
            return f * mu_u / g("MAPFAC_UY")
        if base == "V":
            return f * mu_v / g("MAPFAC_VX")
        if base == "W":
            return f * mu / g("MAPFAC_MY")
        return f * mu

    bw = len(B.dimensions["bdy_width"])
    O = nc.Dataset(os.path.join(out_dir, "wrfbdy_d01"), "w", format="NETCDF3_64BIT_OFFSET")
    ovs = define(B, O)
    fmt = "%Y-%m-%d_%H:%M:%S"
    t0 = dt.datetime.strptime(nc.chartostring(I.variables["Times"][:])[0], fmt)
    t1 = t0 + dt.timedelta(seconds=int(B.getncattr("INTERVAL_SECONDS")) if "INTERVAL_SECONDS" in B.ncattrs() else 10800)
    chars = lambda ts: np.array([list(t.strftime(fmt)) for t in ts], dtype="S1")
    zero_filled, cache = set(), {}
    for name, v in B.variables.items():
        ov = ovs[name]
        if name == "Times" or "thisbdytime" in name:
            ov[0:2, :] = chars([t0, t1])
            continue
        if "nextbdytime" in name:
            ov[0:2, :] = chars([t1, t1 + (t1 - t0)])
            continue
        base, side = name.rsplit("_B", 1)
        shape = [2 if d == "Time" else size.get(d, len(B.dimensions[d])) for d in v.dimensions]
        if side.startswith("T") or base not in I.variables:
            if not side.startswith("T"):
                zero_filled.add(base)
            ov[...] = np.zeros(shape, dtype=v.dtype)
            continue
        if base not in cache:
            cache[base] = couple(base, I.variables[base][0])
        f = cache[base]
        e = {"XS": lambda: np.moveaxis(f[..., :bw], -1, 0),
             "XE": lambda: np.moveaxis(f[..., -bw:][..., ::-1], -1, 0),
             "YS": lambda: np.moveaxis(f[..., :bw, :], -2, 0),
             "YE": lambda: np.moveaxis(f[..., -bw:, :][..., ::-1, :], -2, 0)}[side]()
        ov[...] = np.broadcast_to(e[None], shape).astype(v.dtype)
    O.close()
    return sorted(zero_filled)


# -------------------------------------------------------------------------------------- plan
def make_plan(case_dir, nodes, layouts, cores_per_node, steps=40, reps=2, wrf_tree=None,
              wrf_tree_omp=None):
    text = open(os.path.join(case_dir, "namelist.input")).read()
    max_dom = int(nl_get(text, "max_dom") or 1)
    if max_dom > 1:
        raise ValueError("max_dom=%d: the proxy only handles single-domain cases" % max_dom)
    e_we, e_sn = int(nl_get(text, "e_we")), int(nl_get(text, "e_sn"))
    runs, skipped = [], []
    for spec in layouts:
        L = parse_layout(spec)
        sh = proxy_shape(e_we, e_sn, L["px"], L["py"], nodes, cores_per_node, L["omp"])
        if sh is None:
            skipped.append(spec)
            continue
        tree = wrf_tree_omp if (L["omp"] > 1 and wrf_tree_omp) else wrf_tree
        tag = "%dx%d_o%d_nt%d" % (L["px"], L["py"], L["omp"], L["numtiles"])
        runs.append(dict(tag=tag, **L, **sh, wrf_tree=tree))
    return dict(case_dir=os.path.abspath(case_dir), nodes=nodes, steps=steps, reps=reps,
                dt=time_step_seconds(text), e_we=e_we, e_sn=e_sn, runs=runs, skipped=skipped)


def check_builds(plan):
    """Refuse omp>1 on a build compiled without OpenMP (wtune.extensions.omp_cap_for_build, the
    same wrf.exe check `wtune tune` caps omp with): it would run single-threaded, silently."""
    import wtune.extensions
    errors = []
    for r in plan["runs"]:
        if r["omp"] > 1 and wtune.extensions.omp_cap_for_build(r["wrf_tree"]) == 1:
            errors.append("%s: omp=%d but %s/main/wrf.exe has no OpenMP (dmpar build)"
                          % (r["tag"], r["omp"], r["wrf_tree"]))
    return errors


def campaign_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "campaign")


def submit_plan(plan, out_dir, time_limit="01:00:00", sbatch=("sbatch",)):
    """Write <out_dir>/plan.json + proxy.sbatch and submit it; returns (job_id or None, message)."""
    os.makedirs(out_dir, exist_ok=True)
    plan_path = os.path.join(out_dir, "plan.json")
    json.dump(plan, open(plan_path, "w"), indent=1)
    sb = os.path.join(out_dir, "proxy.sbatch")
    open(sb, "w").write(render_sbatch(plan_path, sys.executable, campaign_dir(), time_limit))
    res = subprocess.run(list(sbatch) + [sb], capture_output=True, text=True)
    m = re.search(r"Submitted batch job (\d+)", res.stdout or "")
    return (m.group(1) if m else None), (res.stdout or res.stderr).strip()


# ------------------------------------------------------------------------------- the job side
def worker(plan_path):
    """Runs inside the 1-node job: crop (cached per box), namelist, site_launch, parse timing."""
    import wtune.machine
    wtune.machine.export(wtune.machine.load_user())   # not started through wtune.cli: apply the machine file
    plan = json.load(open(plan_path))
    out = os.path.dirname(os.path.abspath(plan_path))
    case = plan["case_dir"]
    base_nl = open(os.path.join(case, "namelist.input")).read()
    scratch = os.environ.get("WTUNE_PROXY_SCRATCH") or ("/dev/shm" if os.path.isdir("/dev/shm") else out)
    scratch = os.path.join(scratch, "wtune-proxy-%s" % os.environ.get("SLURM_JOB_ID", os.getpid()))
    run_seconds = int(round(plan["steps"] * plan["dt"]))
    results = open(os.path.join(out, "results.jsonl"), "a")
    try:
        for rep in range(1, plan["reps"] + 1):
            for r in plan["runs"]:
                box = os.path.join(scratch, "%dx%d" % (r["nx"], r["ny"]))
                if not os.path.exists(os.path.join(box, "wrfbdy_d01")):
                    os.makedirs(box, exist_ok=True)
                    crop(os.path.join(case, "wrfinput_d01"), os.path.join(case, "wrfbdy_d01"),
                         box, r["nx"], r["ny"])
                d = os.path.join(out, "runs", "%s-%d" % (r["tag"], rep))
                os.makedirs(d, exist_ok=True)
                run_dir = os.path.join(r["wrf_tree"], "run")
                for f in os.listdir(run_dir):
                    if f in ("namelist.input",) or f.startswith(("wrfinput", "wrfbdy", "wrfout", "wrfrst")):
                        continue
                    _link(os.path.join(run_dir, f), os.path.join(d, f))
                _link(os.path.join(r["wrf_tree"], "main", "wrf.exe"), os.path.join(d, "wrf.exe"))
                for f in ("wrfinput_d01", "wrfbdy_d01"):
                    _link(os.path.join(box, f), os.path.join(d, f))
                nl = nl_set(base_nl, e_we=r["nx"] + 1, e_sn=r["ny"] + 1,
                            run_days=0, run_hours=0, run_minutes=0, run_seconds=run_seconds,
                            nproc_x=r["a"], nproc_y=r["b"], numtiles=r["numtiles"],
                            **_quiet_output(base_nl))
                open(os.path.join(d, "namelist.input"), "w").write(nl)
                cmd = wtune.machine.site_launch_line(r["ranks"], r["ranks"], r["omp"], "block", 1)
                if not cmd:
                    raise RuntimeError("site_launch unavailable for site %s" % wtune.machine.site())
                env = dict(os.environ, OMP_NUM_THREADS=str(r["omp"]))
                env.setdefault("OMP_STACKSIZE", "1G")
                try:
                    rc = subprocess.run(shlex.split(cmd), cwd=d, env=env, stdin=subprocess.DEVNULL,
                                        stdout=open(os.path.join(d, "run.log"), "w"),
                                        stderr=subprocess.STDOUT, timeout=60 + 20 * run_seconds).returncode
                except subprocess.TimeoutExpired:
                    rc = 124
                steps = _step_times(os.path.join(d, "rsl.error.0000"))
                rec = dict(tag=r["tag"], rep=rep, rc=rc, steps=len(steps),
                           median=statistics.median(steps[1:]) if len(steps) > 1 else None,
                           px=r["px"], py=r["py"], omp=r["omp"], numtiles=r["numtiles"],
                           proxy_ranks=r["ranks"], box="%dx%d" % (r["nx"], r["ny"]))
                results.write(json.dumps(rec) + "\n")
                results.flush()
                print("RESULT %s rep %d rc %d steps %d median %s" % (
                    r["tag"], rep, rc, len(steps), "%.4f" % rec["median"] if rec["median"] else "nan"),
                    flush=True)
                if rc != 0 or len(steps) < max(10, plan["steps"] // 2):
                    print("ABORT %s rep %d: see %s" % (r["tag"], rep, d), flush=True)
                    return 4
                for f in os.listdir(d):
                    if f.startswith("wrfout"):
                        os.remove(os.path.join(d, f))
    finally:
        results.close()
        subprocess.run(["rm", "-rf", scratch])
    return 0


def _tree_name(tree):
    tree = tree.rstrip("/")
    name = os.path.basename(tree)
    return os.path.basename(os.path.dirname(tree)) if name == "WRF" else name


def _quiet_output(text):
    kv = {}
    for k, v in (("history_interval", 100000), ("history_interval_m", 100000),
                 ("restart_interval", 100000), ("restart", ".false.")):
        if nl_get(text, k) is not None:
            kv[k] = v
    return kv


def _link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def _step_times(rsl):
    try:
        return [float(m.group(1)) for line in open(rsl) for m in [STEP_RE.search(line)] if m]
    except OSError:
        return []


# ------------------------------------------------------------------------------------ sbatch
def render_sbatch(plan_path, python, campaign_dir, time_limit="01:00:00"):
    import wtune.machine
    from wtune.tools import scc_sweep
    out = os.path.dirname(os.path.abspath(plan_path))
    return "\n".join([
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=wtune-proxy",
        ("#SBATCH -p %s -N 1 --exclusive --ntasks-per-node=%d --time=%s %s"
         % (wtune.machine.partition(), wtune.machine.cores_per_node(), time_limit,
            scc_sweep.scheduler_args())).rstrip(),
        "#SBATCH --no-requeue",
        "#SBATCH --output=%s/proxy-%%j.out" % out,
        "set -uo pipefail",
        "export CAMPAIGN=%s" % shlex.quote(campaign_dir),
        'source "$CAMPAIGN/sites/%s.sh"' % wtune.machine.site(),
        "export PYTHONPATH=%s${PYTHONPATH:+:$PYTHONPATH}" % shlex.quote(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "site_env",
        "export SLURM_JOB_NUM_NODES=1",
        'echo "start $(date -Is) $(hostname)"',
        "%s -m wtune.proxy _worker %s" % (shlex.quote(python), shlex.quote(os.path.abspath(plan_path))),
        'rc=$?; echo "end $(date -Is) rc=$rc"; exit $rc']) + "\n"


# ----------------------------------------------------------------------------------- collect
def collect(out_dir):
    rows = {}
    for line in open(os.path.join(out_dir, "results.jsonl")):
        r = json.loads(line)
        if r["rc"] == 0 and r["median"]:
            rows.setdefault(r["tag"], []).append(r["median"])
    ranked = sorted(((statistics.median(v), t, v) for t, v in rows.items()))
    return [dict(tag=t, median=m, reps=v, spread=(max(v) - min(v)) / m) for m, t, v in ranked]


# --------------------------------------------------------------------------------------- CLI
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_worker"]:
        return worker(argv[1])
    import wtune.machine
    ap = argparse.ArgumentParser(prog="wtune proxy", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "submit"):
        p = sub.add_parser(name)
        p.add_argument("case_dir")
        p.add_argument("--nodes", type=int, required=True, help="node count of the TARGET run")
        p.add_argument("--layout", action="append", required=True,
                       help="PXxPY[:omp=K][:nt=T], repeatable")
        p.add_argument("--steps", type=int, default=40)
        p.add_argument("--reps", type=int, default=2)
        p.add_argument("--wrf-tree", default=None, help="WRF tree for omp=1 (default: machine's)")
        p.add_argument("--wrf-tree-omp", default=None, help="WRF tree (dmsm) for omp>1")
        if name == "submit":
            p.add_argument("--out", default=None)
            p.add_argument("--time", default="01:00:00")
            p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("collect")
    p.add_argument("out_dir")
    a = ap.parse_args(argv)

    if a.cmd == "collect":
        rows = collect(a.out_dir)
        for i, r in enumerate(rows):
            print("%2d  %-20s %.4f s/step  reps %s  spread %.1f%%" % (
                i + 1, r["tag"], r["median"], ",".join("%.4f" % x for x in r["reps"]), 100 * r["spread"]))
        return 0 if rows else 1

    tree = a.wrf_tree or wtune.machine.wrf_tree()
    plan = make_plan(a.case_dir, a.nodes, a.layout, wtune.machine.cores_per_node(), a.steps, a.reps,
                     tree, a.wrf_tree_omp)
    for r in plan["runs"]:
        print("%-20s proxy %3d ranks (%dx%d) box %dx%d  build %s" % (
            r["tag"], r["ranks"], r["a"], r["b"], r["nx"], r["ny"], _tree_name(r["wrf_tree"])))
    for s in plan["skipped"]:
        print("%-20s skipped: not reproducible on one node" % s)
    errors = check_builds(plan)
    for e in errors:
        print("ERROR " + e, file=sys.stderr)
    if a.cmd == "plan" or errors or not plan["runs"]:
        return 1 if (errors or not plan["runs"]) else 0
    out = os.path.abspath(a.out or os.path.join(a.case_dir, "..", "proxy"))
    job, msg = submit_plan(plan, out, a.time, sbatch=("true",) if a.dry_run else ("sbatch",))
    print("wrote %s" % os.path.join(out, "proxy.sbatch"))
    if a.dry_run:
        return 0
    print(msg)
    return 0 if job else 1


if __name__ == "__main__":
    sys.exit(main())
