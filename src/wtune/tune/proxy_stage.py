"""`wtune tune --proxy on`: pick round0's real probes with the one-node small-domain proxy.

round0 normally measures k sampled layouts at N nodes for real. With the proxy, a wider pool
(PROXY_POOL_FACTOR x k) is timed on one node each (wtune/proxy.py), the jobs spread over up to
max_nodes nodes in parallel, and only the ANCHOR plus the proxy's fastest k-1 go to the real
N-node probes. Anything the proxy cannot reproduce (nested case, no netCDF wrfinput/wrfbdy in the
case directory, 2-D tiles, a failed proxy job) falls back to round0's own selection, with a note.
"""
import math
import os

import wtune.proxy as _proxy
from . import jobs as _jobs
from .candidates import _apply_search_filter, sample_more_diverse_configs
from .context import Ctx, cores_used

PROXY_POOL_FACTOR = 3          # pool = 3 x what round0 would measure
PROXY_POOL_MAX = 24
PROXY_RUNS_PER_JOB = 4         # ~40 s each on one node -> a few minutes per job


def _spec(c: dict) -> str | None:
    """Layout spec for wtune.proxy, or None when the proxy cannot reproduce the config."""
    if str(c.get("tile_strategy") or 0) not in ("0", "") or c.get("numtiles_x") or c.get("tile_sz_x"):
        return None
    return "%dx%d:omp=%d:nt=%d" % (int(c["nx"]), int(c["ny"]), int(c.get("omp") or 1),
                                   int(c.get("numtiles") or 1))


def _tag(c: dict) -> str:
    return "%dx%d_o%d_nt%d" % (int(c["nx"]), int(c["ny"]), int(c.get("omp") or 1),
                               int(c.get("numtiles") or 1))


def select(ctx: Ctx, configs: list[dict]) -> list[dict]:
    """configs (ANCHOR first) -> the same number of configs, chosen by the proxy."""
    keep = len(configs)
    case_dir = os.path.dirname(os.path.abspath(ctx.namelist_path))
    missing = [f for f in ("wrfinput_d01", "wrfbdy_d01") if not os.path.exists(os.path.join(case_dir, f))]
    if missing or not ctx.wrf_tree:
        print("proxy: skipped (%s) -- round0 keeps its own sample"
              % ("no %s in %s" % ("/".join(missing), case_dir) if missing else "no WRF tree"))
        return configs
    extra = max(0, min(PROXY_POOL_MAX, PROXY_POOL_FACTOR * keep) - keep)
    more = _apply_search_filter(ctx, sample_more_diverse_configs(ctx, configs, extra, ctx.seed + 7)) if extra else []
    pool = configs + more
    specs = {}
    for c in pool:
        s = _spec(c)
        if s:
            specs.setdefault(_tag(c), s)
    try:
        plan = _proxy.make_plan(case_dir, ctx.nodes, list(specs.values()), cores_used(ctx), steps=ctx.steps,
                                reps=1, wrf_tree=ctx.wrf_tree)
    except (ValueError, OSError, KeyError) as e:
        print("proxy: skipped (%s) -- round0 keeps its own sample" % e)
        return configs
    errors = _proxy.check_builds(plan)
    runs = [r for r in plan["runs"] if not any(e.startswith(r["tag"] + ":") for e in errors)]
    if not runs:
        print("proxy: skipped (no layout reproducible on one node) -- round0 keeps its own sample")
        return configs
    n_jobs = max(1, min(ctx.max_nodes, math.ceil(len(runs) / PROXY_RUNS_PER_JOB)))
    out_root = os.path.join(ctx.out_dir, "proxy-round0-n%d" % ctx.nodes)
    job_ids, out_dirs = [], []
    for j in range(n_jobs):
        sub = dict(plan, runs=runs[j::n_jobs])
        d = os.path.join(out_root, "job%d" % j)
        job, msg = _proxy.submit_plan(sub, d)
        if job:
            job_ids.append(job)
            out_dirs.append(d)
            ctx.owned_jobs.append(job)
        else:
            print("proxy: submit failed for %s: %s" % (d, msg))
    if not job_ids:
        return configs
    print("proxy: %d layouts on %d one-node job(s) %s (pool %d, keeping %d for the %d-node probes)"
          % (len(runs), len(job_ids), ",".join(job_ids), len(pool), keep, ctx.nodes))
    _jobs.wait_for_jobs(ctx, job_ids, "proxy")
    timed = {}
    for d in out_dirs:
        try:
            for r in _proxy.collect(d):
                timed[r["tag"]] = r["median"]
        except OSError:
            pass
    if not timed:
        print("proxy: no usable timings -- round0 keeps its own sample")
        return configs
    anchor, rest = configs[0], pool[1:]
    seen, ranked = {_tag(anchor)}, []
    for c in sorted((c for c in rest if _tag(c) in timed), key=lambda c: timed[_tag(c)]):
        if _tag(c) not in seen:
            seen.add(_tag(c))
            ranked.append(c)
    untimed = [c for c in configs[1:] if _tag(c) not in timed and _tag(c) not in seen]
    chosen = [anchor] + (ranked + untimed)[:keep - 1]
    for i, c in enumerate([anchor] + ranked):
        t = timed.get(_tag(c))
        print("proxy: %2d %-20s %s%s" % (i, _tag(c), "%.4f s/step" % t if t else "untimed",
                                         "  -> real probe" if c in chosen else ""))
    return chosen
