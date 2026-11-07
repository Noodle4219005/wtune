"""Small-domain proxy: rank layouts for an N-node run by timing them on ONE node.

Each target layout (px x py ranks on N nodes, omp threads, numtiles) is reproduced on one node with
ranks/N ranks, on a box cropped from the case's wrfinput so that every rank owns the same patch as
in the target. Validated on conus25 (Grace, layout order of N=4/N=8 ledger, s/step within 2 %) and
karina (x86, hybrid and numtiles effects of a 4-node ABBA truth). It ranks; it is not a wall-time
predictor (its bias is machine dependent) and it does not see inter-node effects (cyclic
placement, network). A nested case is split into its domains, each timed as a single domain and
summed by its steps per d01 step; a fine nest can go CFL-unstable under the frozen boundary
(marshall's d02 at the larger box), which leaves that layout incomplete rather than wrong.

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


_SEG = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)")
_VAL = re.compile(r"'[^']*'|\"[^\"]*\"|[^,]+")
_BDY_KEYS = ("specified", "nested", "spec_bdy_width", "spec_zone", "relax_zone", "spec_exp")
PROXY_BDY_WIDTH = 10        # proxy lateral boundary: 1 specified + 9 relaxation rows (WRF default 1 + 4)
PROXY_SPEC_EXP = 0.33       # exponential relaxation across the wider zone (WRF default 0: linear)


def nl_values(text, key):
    """All comma-separated values of `key` (one per domain for domain arrays), as strings."""
    m = re.search(r"(?<![A-Za-z0-9_])%s\s*=\s*([^\n/]*)" % re.escape(key), text)
    if not m:
        return []
    raw = m.group(1)
    nxt = _SEG.search(raw)
    raw = raw[:nxt.start()] if nxt else raw
    return [v.strip().strip("'\"") for v in _VAL.findall(raw) if v.strip()]


def pick_column(text, d):
    """The namelist with every multi-valued key reduced to its domain-d value (the last value when
    the list is shorter), so domain d can run as a single domain."""
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        ms = list(_SEG.finditer(body))
        if not ms or body.lstrip().startswith(("&", "/", "!")):
            out.append(line)
            continue
        parts = [body[:ms[0].start()]]
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(body)
            vals = [v.strip() for v in _VAL.findall(body[m.end():end]) if v.strip()]
            if len(vals) >= 2:
                parts.append("%s%s%s, " % (m.group(1), m.group(2), vals[d - 1] if len(vals) >= d else vals[-1]))
            else:
                parts.append(body[m.start():end])
        out.append("".join(parts).rstrip() + "\n")
    return "".join(out)


def nl_force(text, **kv):
    """nl_set for the keys present; the absent ones are added to &domains (&bdy_control for
    specified/nested)."""
    present = {k: v for k, v in kv.items() if nl_get(text, k) is not None or k in ("nproc_x", "nproc_y", "numtiles")}
    text = nl_set(text, **present)
    for k, v in kv.items():
        if k in present:
            continue
        group = "bdy_control" if k in _BDY_KEYS else "domains"
        text, n = re.subn(r"(&%s[^\n]*\n)" % group, lambda m: m.group(1) + " %s = %s,\n" % (k, v), text,
                          count=1, flags=re.I)
        if not n:
            text += "\n &%s\n %s = %s,\n /\n" % (group, k, v)
    return text


def domain_layers(text):
    """Per domain (1-based): e_we, e_sn, dx, dy, dt and the number of its steps per d01 step."""
    max_dom = int(nl_get(text, "max_dom") or 1)
    col = lambda k, cast=float, default=None: [cast(x) for x in nl_values(text, k)] or default
    e_we, e_sn = col("e_we", int), col("e_sn", int)
    dx, dy = col("dx", float, [0.0]), col("dy", float, [0.0])
    parent = col("parent_id", int, [0] * max_dom)
    ratio = col("parent_time_step_ratio", int, [1] * max_dom)
    pick = lambda xs, d: xs[d - 1] if len(xs) >= d else xs[-1]
    dt = time_step_seconds(text)
    cum = {1: 1}
    layers = []
    for d in range(1, max_dom + 1):
        if d > 1:
            cum[d] = cum.get(pick(parent, d), 1) * pick(ratio, d)
        layers.append(dict(domain=d, e_we=pick(e_we, d), e_sn=pick(e_sn, d), dx=pick(dx, d), dy=pick(dy, d),
                           weight=cum[d], dt=dt / cum[d]))
    return layers


def nest_dx(layer):
    """(dx, dy) for a stand-alone run of `layer`: unchanged unless its dt exceeds WRF's
    dt <= 6*dx[km] check (a nest's dt = parent_dt / ratio can sit just on it, marshall's 0.6667 s
    at 111.1111 m), then nudged just past it. The cropped wrfinput gets the same DX/DY, since WRF
    also checks the namelist against the input file."""
    dx, dy = layer["dx"], layer["dy"]
    if layer["domain"] > 1 and dx and layer["dt"] > 6.0 * dx / 1000.0 * (1 - 1e-12):
        f = layer["dt"] / (6.0 * dx / 1000.0) * 1.0001
        return dx * f, dy * f
    return dx, dy


def single_domain_namelist(text, layer, nx, ny, a, b, numtiles, run_seconds):
    """Domain `layer` of a nested namelist as a stand-alone single-domain run of nx x ny points."""
    from fractions import Fraction
    t = pick_column(text, layer["domain"]) if layer["domain"] > 1 or int(nl_get(text, "max_dom") or 1) > 1 else text
    f = Fraction(layer["dt"]).limit_denominator(1000)
    whole, frac = int(f), f - int(f)
    dx, dy = nest_dx(layer)
    kv = dict(max_dom=1, e_we=nx + 1, e_sn=ny + 1, time_step=whole,
              time_step_fract_num=frac.numerator, time_step_fract_den=frac.denominator or 1,
              dx="%.9g" % dx, dy="%.9g" % dy,
              grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1, parent_grid_ratio=1,
              parent_time_step_ratio=1, specified=".true.", nested=".false.",
              run_days=0, run_hours=0, run_minutes=0, run_seconds=run_seconds,
              nproc_x=a, nproc_y=b, numtiles=numtiles)
    kv.update(_quiet_output(t))
    return nl_force(t, **kv)


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


def band_shape(e_we, e_sn, px, py, nodes, cores_per_node, omp=1, proxy_nodes=3):
    """Band proxy on `proxy_nodes` nodes: the full domain width and px x (proxy_nodes*h) ranks, h
    the process-grid rows one target node holds (WRF numbers ranks x-fastest, placement is block),
    so the middle node has the target's interior node: the same patches and the same on- and
    off-node neighbours. dict(ranks, a, b, nx, ny, nodes, ranks_per_node), or None when a node
    holds no whole rows or the band is not smaller than the target. proxy_nodes == nodes is the
    whole domain on the target's own nodes (frozen synthesized boundary): the reference for small N."""
    mx, my = e_we - 1, e_sn - 1
    if (px * py) % nodes or proxy_nodes > nodes:
        return None
    rp = px * py // nodes
    if rp * omp > cores_per_node:
        return None
    if proxy_nodes == nodes:
        return dict(ranks=px * py, a=px, b=py, nx=mx, ny=my, nodes=nodes, ranks_per_node=rp)
    if rp % px:
        return None
    b = proxy_nodes * (rp // px)
    ny = math.ceil(my / py) * b
    if b > py or ny > my:
        return None
    return dict(ranks=px * b, a=px, b=b, nx=mx, ny=ny, nodes=proxy_nodes, ranks_per_node=rp)


# --------------------------------------------------------------------------------------- crop
def flat_offset(hgt, nx, ny, band, steps=24):
    """(i0, j0) of the nx x ny window of terrain `hgt` (south_north x west_east) whose `band`-row
    edge strips have the smallest mean terrain slope: the frozen boundary of a crop drives the
    interior least where its edges are flat (the proxy's CFL and radiation-step failures sat on
    crop edges over steep terrain). Offsets searched on a steps x steps grid; the centre wins ties."""
    import numpy as np
    my, mx = hgt.shape
    gy, gx = np.gradient(hgt.astype("f8"))
    slope = np.hypot(gx, gy)
    ci, cj = (mx - nx) // 2, (my - ny) // 2
    best = None
    for j0 in sorted(set(np.linspace(0, my - ny, min(steps, my - ny + 1)).astype(int)) | {cj}):
        for i0 in sorted(set(np.linspace(0, mx - nx, min(steps, mx - nx + 1)).astype(int)) | {ci}):
            w = slope[j0:j0 + ny, i0:i0 + nx]
            b = min(band, nx // 2, ny // 2)
            edge = np.concatenate([w[:b].ravel(), w[-b:].ravel(), w[:, :b].ravel(), w[:, -b:].ravel()])
            key = (float(edge.mean()), abs(i0 - ci) + abs(j0 - cj))
            if best is None or key < best[0]:
                best = (key, i0, j0)
    return best[1], best[2]


def crop(src_in, src_bdy, out_dir, nx, ny, as_top=False, dxdy=None, place="centre", bdy_width=None):
    """Crop wrfinput_d01 to nx x ny mass points (`place` "centre", or "flat": flat_offset) and
    synthesize wrfbdy_d01 from the crop's edges (values mass-coupled as real.exe writes them, zero
    tendencies; `bdy_width` rows, default the source wrfbdy's). Fields wrfbdy carries but wrfinput
    lacks (e.g. HT_SHAD) are zero. Returns the zero-filled field names."""
    import datetime as dt
    import netCDF4 as nc
    import numpy as np
    S = nc.Dataset(src_in)
    if place == "flat":
        S.variables["HGT"].set_auto_mask(False)
        i0, j0 = flat_offset(S.variables["HGT"][0], nx, ny, bdy_width or 5)
    else:
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
        if as_top:      # a nest's wrfinput run as a stand-alone domain
            for k, val in (("GRID_ID", 1), ("PARENT_ID", 0), ("PARENT_GRID_RATIO", 1),
                           ("I_PARENT_START", 1), ("J_PARENT_START", 1)):
                dst.setncattr(k, np.int32(val))
            if dxdy:
                dst.setncattr("DX", np.float32(dxdy[0]))
                dst.setncattr("DY", np.float32(dxdy[1]))
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

    bw = bdy_width or len(B.dimensions["bdy_width"])
    size["bdy_width"] = bw
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
              wrf_tree_omp=None, proxy_nodes=1, place="centre", bdy_width=None):
    """One proxy run per layout and domain. A nested case is split into its domains, each run on
    its own as a single domain with the same px x py patch rule; collect() adds them up weighted by
    each domain's steps per d01 step (domain 1's timing in a nested run includes its nests).
    proxy_nodes > 1 runs band proxies (band_shape) on that many nodes instead of one."""
    text = open(os.path.join(case_dir, "namelist.input")).read()
    layers = domain_layers(text)
    runs, skipped = [], []
    for spec in layouts:
        L = parse_layout(spec)
        if proxy_nodes > 1:
            shapes = [band_shape(ly["e_we"], ly["e_sn"], L["px"], L["py"], nodes, cores_per_node, L["omp"],
                                 proxy_nodes) for ly in layers]
        else:
            shapes = [proxy_shape(ly["e_we"], ly["e_sn"], L["px"], L["py"], nodes, cores_per_node, L["omp"])
                      for ly in layers]
        if any(sh is None for sh in shapes):
            skipped.append(spec)
            continue
        tree = wrf_tree_omp if (L["omp"] > 1 and wrf_tree_omp) else wrf_tree
        tag = "%dx%d_o%d_nt%d" % (L["px"], L["py"], L["omp"], L["numtiles"])
        for ly, sh in zip(layers, shapes):
            runs.append(dict(tag=tag if len(layers) == 1 else "%s_d%d" % (tag, ly["domain"]), layout=tag,
                             domain=ly["domain"], weight=ly["weight"], dt=ly["dt"], **L, **sh, wrf_tree=tree))
    return dict(case_dir=os.path.abspath(case_dir), nodes=nodes, proxy_nodes=proxy_nodes, steps=steps,
                reps=reps, dt=time_step_seconds(text), max_dom=len(layers), layers=layers,
                place=place, bdy_width=bdy_width,
                e_we=layers[0]["e_we"], e_sn=layers[0]["e_sn"], runs=runs, skipped=skipped)


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
    nodes = plan.get("proxy_nodes", 1)
    open(sb, "w").write(render_sbatch(plan_path, sys.executable, campaign_dir(), time_limit, nodes))
    if nodes > 1 and tuple(sbatch) == ("sbatch",):
        # partitions are tiered by job size (f1-arm): try them in order as tune's probes do
        from wtune.tools import scc_sweep
        try:
            job = scc_sweep.submit_job("sbatch " + shlex.quote(sb))
            return str(job), "Submitted batch job %s" % job
        except (RuntimeError, ValueError) as e:
            return None, str(e)
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
    layers = {ly["domain"]: ly for ly in plan.get("layers") or [dict(domain=1, dt=plan["dt"], weight=1)]}
    nested = plan.get("max_dom", 1) > 1
    ok_runs = 0
    results = open(os.path.join(out, "results.jsonl"), "a")
    try:
        for rep in range(1, plan["reps"] + 1):
            for r in plan["runs"]:
                dom = r.get("domain", 1)
                run_seconds = max(1, math.ceil(plan["steps"] * r.get("dt", plan["dt"])))
                place, bw = plan.get("place", "centre"), plan.get("bdy_width")
                box = os.path.join(scratch, "d%d_%dx%d_%s_bw%s" % (dom, r["nx"], r["ny"], place, bw or "src"))
                if not os.path.exists(os.path.join(box, "wrfbdy_d01")):
                    os.makedirs(box, exist_ok=True)
                    crop(os.path.join(case, "wrfinput_d%02d" % dom), os.path.join(case, "wrfbdy_d01"),
                         box, r["nx"], r["ny"], as_top=dom > 1,
                         dxdy=nest_dx(layers[dom]) if nested and dom > 1 else None, place=place, bdy_width=bw)
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
                if nested:
                    nl = single_domain_namelist(base_nl, layers[dom], r["nx"], r["ny"], r["a"], r["b"],
                                                r["numtiles"], run_seconds)
                else:
                    nl = nl_set(base_nl, e_we=r["nx"] + 1, e_sn=r["ny"] + 1,
                                run_days=0, run_hours=0, run_minutes=0, run_seconds=run_seconds,
                                nproc_x=r["a"], nproc_y=r["b"], numtiles=r["numtiles"],
                                **_quiet_output(base_nl))
                if bw:
                    nl = nl_force(nl, spec_bdy_width=bw, spec_zone=1, relax_zone=bw - 1, spec_exp=PROXY_SPEC_EXP)
                open(os.path.join(d, "namelist.input"), "w").write(nl)
                _null_rsl(d, r["ranks"])
                cmd = wtune.machine.site_launch_line(r["ranks"], r.get("ranks_per_node", r["ranks"]), r["omp"],
                                                     "block", r.get("nodes", 1))
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
                rec = dict(tag=r["tag"], layout=r.get("layout", r["tag"]), domain=dom, weight=r.get("weight", 1),
                           rep=rep, rc=rc, steps=len(steps),
                           median=statistics.median(steps[1:]) if len(steps) > 1 else None,
                           px=r["px"], py=r["py"], omp=r["omp"], numtiles=r["numtiles"],
                           proxy_ranks=r["ranks"], box="%dx%d" % (r["nx"], r["ny"]))
                results.write(json.dumps(rec) + "\n")
                results.flush()
                print("RESULT %s rep %d rc %d steps %d median %s" % (
                    r["tag"], rep, rc, len(steps), "%.4f" % rec["median"] if rec["median"] else "nan"),
                    flush=True)
                _tidy(d, keep_logs=rc != 0)
                if rc != 0 or len(steps) < max(10, plan["steps"] // 2):
                    if ok_runs == 0:       # nothing has worked yet: a setup error, stop here
                        print("ABORT %s rep %d: see %s" % (r["tag"], rep, d), flush=True)
                        return 4
                    print("FAILED %s rep %d (layout left incomplete): see %s" % (r["tag"], rep, d), flush=True)
                    continue
                ok_runs += 1
                _tidy(d)
    finally:
        results.close()
        subprocess.run(["rm", "-rf", scratch])
    return 0


def _null_rsl(d, ranks):
    """rsl.out/rsl.error of ranks >= 1 -> /dev/null before WRF opens them, as the probes do
    (coldstart_sample): on f1-arm's 16 MiB GPFS blocks every open rsl file holds a block, 4.5 GiB
    per 144-rank proxy; only rank 0's rsl.error is read."""
    for i in range(1, ranks):
        for k in ("rsl.out", "rsl.error"):
            _link(os.devnull, os.path.join(d, "%s.%04d" % (k, i)))


def _tidy(d, keep_logs=False):
    """Drop the run's history and every rank's rsl file but rank 0's: on GPFS each small file
    holds a full block (f1-arm /work1 went over quota twice from proxy rsl files)."""
    for f in os.listdir(d):
        if f.startswith("wrfout") or (not keep_logs and f.startswith("rsl.") and not f.endswith(".0000")):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass


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
def render_sbatch(plan_path, python, campaign_dir, time_limit="01:00:00", nodes=1):
    import wtune.machine
    from wtune.tools import scc_sweep
    out = os.path.dirname(os.path.abspath(plan_path))
    return "\n".join([
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=wtune-proxy",
        ("#SBATCH -p %s -N %d --exclusive --ntasks-per-node=%d --time=%s %s"
         % (wtune.machine.partition(), nodes, wtune.machine.cores_per_node(), time_limit,
            scc_sweep.scheduler_args())).rstrip(),
        "#SBATCH --no-requeue",
        "#SBATCH --output=%s/proxy-%%j.out" % out,
        "set -uo pipefail",
        "export CAMPAIGN=%s" % shlex.quote(campaign_dir),
        'source "$CAMPAIGN/sites/%s.sh"' % wtune.machine.site(),
        "export PYTHONPATH=%s${PYTHONPATH:+:$PYTHONPATH}" % shlex.quote(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "site_env",
        "export SLURM_JOB_NUM_NODES=%d" % nodes,
        'echo "start $(date -Is) $(hostname)"',
        "%s -m wtune.proxy _worker %s" % (shlex.quote(python), shlex.quote(os.path.abspath(plan_path))),
        'rc=$?; echo "end $(date -Is) rc=$rc"; exit $rc']) + "\n"


# ----------------------------------------------------------------------------------- collect
def collect(out_dir):
    """Layouts ranked by s per d01 step: for a nested case the sum over its domains of
    weight x median (a layout with any domain missing is left out)."""
    per = {}          # layout -> rep -> domain -> (weight, median)
    domains = {}
    try:
        n_dom = json.load(open(os.path.join(out_dir, "plan.json"))).get("max_dom", 1)
    except (OSError, ValueError):
        n_dom = None
    for line in open(os.path.join(out_dir, "results.jsonl")):
        r = json.loads(line)
        lay, dom = r.get("layout", r["tag"]), r.get("domain", 1)
        domains.setdefault(lay, set()).add(dom)
        if r["rc"] == 0 and r["median"]:
            per.setdefault(lay, {}).setdefault(r["rep"], {})[dom] = (r.get("weight", 1), r["median"])
    rows = []
    for lay, reps in per.items():
        want = set(range(1, n_dom + 1)) if n_dom else domains[lay]
        scores = [sum(w * m for w, m in got.values()) for got in reps.values() if set(got) >= want]
        if scores:
            m = statistics.median(scores)
            rows.append(dict(tag=lay, median=m, reps=scores, spread=(max(scores) - min(scores)) / m))
    return sorted(rows, key=lambda r: r["median"])


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
        p.add_argument("--place", default="centre", choices=("centre", "flat"),
                       help="crop position: domain centre (default) or the flattest edges (flat_offset)")
        p.add_argument("--bdy-width", type=int, default=None,
                       help="proxy boundary rows (spec_bdy_width; relax_zone = this - 1, spec_exp %.2f); "
                            "default: the case's" % PROXY_SPEC_EXP)
        p.add_argument("--proxy-nodes", type=int, default=1,
                       help="1: one-node proxy (default); M>1: band proxy on M nodes (full width)")
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
                     tree, a.wrf_tree_omp, a.proxy_nodes, a.place, a.bdy_width)
    for r in plan["runs"]:
        print("%-20s proxy %3d ranks (%dx%d) on %d node(s) box %dx%d  build %s" % (
            r["tag"], r["ranks"], r["a"], r["b"], r.get("nodes", 1), r["nx"], r["ny"], _tree_name(r["wrf_tree"])))
    for s in plan["skipped"]:
        print("%-20s skipped: not reproducible on %d node(s)" % (s, a.proxy_nodes))
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
