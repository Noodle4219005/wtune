"""round0 by the one-node small-domain proxy (default; `--proxy off` restores the real-probe round0).

round0 would measure k sampled layouts at N nodes for real. Instead a wider pool
(PROXY_POOL_FACTOR x k) is timed on one node each (wtune/proxy.py), the jobs spread over up to
max_nodes nodes in parallel. When N > PROXY_BAND_NODES, the PROXY_BAND_KEEP fastest are re-timed as
band proxies (PROXY_BAND_NODES nodes reproducing the target's interior node) and rank by that. Only
the ANCHOR plus the proxy's PROXY_REAL_KEEP-1 fastest go to the real N-node probes (which race and
confirm as before). A nested case is split into its domains
(wtune/proxy.py). Configs the proxy cannot reproduce (2-D tiles, dist=cyclic) are not ranked; no
netCDF wrfinput/wrfbdy in the case directory, or no proxy timing at all, falls back to round0's own
selection, with a note.

At O2 the node search ranks (N, layout) jointly instead (rank_across_nodes): every legal N's pool in
one batch, and the finalists' round0 takes that ranking (ctx.proxy_ranked) without new proxy jobs.
"""
import dataclasses
import math
import os

import wtune.proxy as _proxy
from . import jobs as _jobs
from .candidates import _apply_search_filter, _sweep_anchor, sample_more_diverse_configs
from .context import Ctx, cores_used

PROXY_POOL_FACTOR = 3          # pool = 3 x what round0 would measure
PROXY_POOL_MAX = 24
PROXY_RUNS_PER_JOB = 4         # ~40 s each on one node -> a few minutes per job
PROXY_REAL_KEEP = 3            # real N-node probes after the proxy: ANCHOR + the 2 fastest
PROXY_BAND_KEEP = 4            # one-node leaders re-timed as band proxies
PROXY_BAND_NODES = 3           # band proxy size: reproduces the target's interior node
PROXY_JOINT_PER_N = 6          # rank_across_nodes: ANCHOR + 5 diverse layouts per N
PROXY_JOINT_LEADERS = 2        # rank_across_nodes: one-node leaders per N > PROXY_BAND_NODES re-timed as bands
PROXY_JOINT_FINALISTS = 2      # node search: the N with the best proxy scores tuned for real


def _spec(c: dict) -> str | None:
    """Layout spec for wtune.proxy, or None when the proxy cannot reproduce the config."""
    if str(c.get("tile_strategy") or 0) not in ("0", "") or c.get("numtiles_x") or c.get("tile_sz_x"):
        return None
    if str(c.get("dist") or "block") != "block":
        return None    # the proxy places ranks by block; a cyclic twin must not inherit its timing
    return "%dx%d:omp=%d:nt=%d" % (int(c["nx"]), int(c["ny"]), int(c.get("omp") or 1),
                                   int(c.get("numtiles") or 1))


def _tag(c: dict) -> str:
    return "%dx%d_o%d_nt%d" % (int(c["nx"]), int(c["ny"]), int(c.get("omp") or 1),
                               int(c.get("numtiles") or 1))


def _real_budget_keep(ctx, keep: int) -> int:
    """How many real N-node probes the remaining --minutes budget can still pay for (>= 1)."""
    budget = getattr(ctx, "budget_s", float("inf"))
    if not math.isfinite(budget):
        return keep
    try:
        s_est = ctx.estimate_s_per_step()[0]
        per = (getattr(ctx, "startup_s", 0.0) or 0.0) + ctx.steps * s_est
        slots = max(1, ctx.max_nodes // max(1, ctx.nodes))
        left = budget - ctx.elapsed()
    except (AttributeError, TypeError, IndexError):
        return keep
    if per <= 0:
        return keep
    return max(1, min(keep, int(left // per) * slots))


def _proxy_time_limit(ctx) -> str:
    """Slurm --time for a proxy job: an hour, or 40 % of what is left of a --minutes budget (the
    rest is for the real probes); the runs it finishes before that are ranked."""
    budget = getattr(ctx, "budget_s", float("inf"))
    left = budget - ctx.elapsed() if math.isfinite(budget) and hasattr(ctx, "elapsed") else 3600.0
    m = max(5, min(60, int(0.4 * left // 60)))
    return "00:%02d:00" % m if m < 60 else "01:00:00"


def select(ctx: Ctx, configs: list[dict]) -> list[dict]:
    """configs (ANCHOR first) -> ANCHOR + the proxy's fastest (PROXY_REAL_KEEP in all); the input
    list unchanged when the proxy cannot run."""
    k = len(configs)
    keep = min(k, PROXY_REAL_KEEP)
    if getattr(ctx, "proxy_ranked", None) is not None:
        return _select_ranked(ctx, configs, keep)
    case_dir = os.path.dirname(os.path.abspath(ctx.namelist_path))
    missing = [f for f in ("wrfinput_d01", "wrfbdy_d01") if not os.path.exists(os.path.join(case_dir, f))]
    if missing or not ctx.wrf_tree:
        print("proxy: skipped (%s) -- round0 keeps its own sample"
              % ("no %s in %s" % ("/".join(missing), case_dir) if missing else "no WRF tree"))
        return configs
    extra = max(0, min(PROXY_POOL_MAX, PROXY_POOL_FACTOR * k) - k)
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
    out_root = os.path.join(ctx.out_dir, "proxy-round0-n%d" % ctx.nodes)
    print("proxy: %d layouts on one node each (pool %d, keeping %d for the %d-node probes)"
          % (len(runs), len(pool), keep, ctx.nodes))
    timed = _run_jobs(ctx, plan, runs, out_root, "job", 1)
    if timed is None:
        return configs
    if not timed:
        print("proxy: no usable timings -- round0 keeps its own sample")
        return configs
    band = _band_round(ctx, case_dir, pool, timed, out_root)
    # timings of a proxy job the budget cut short are used as they are: partial results rank too
    anchor, rest = configs[0], pool[1:]
    seen, ranked = {_tag(anchor)}, []
    # band timings (the leaders, closer to the real N-node time) rank first, then the one-node rest
    order = lambda c: (0, band[_tag(c)]) if _tag(c) in band else (1, timed[_tag(c)])
    for c in sorted((c for c in rest if _spec(c) and _tag(c) in timed), key=order):
        if _tag(c) not in seen:
            seen.add(_tag(c))
            ranked.append(c)
    untimed = [c for c in configs[1:] if _tag(c) not in timed and _tag(c) not in seen]
    chosen = _afford(ctx, [anchor] + (ranked or untimed)[:keep - 1], anchor)
    ctx.proxy_used = True
    for i, c in enumerate([anchor] + ranked):
        t, b = timed.get(_tag(c)), band.get(_tag(c))
        print("proxy: %2d %-20s %s%s%s" % (i, _tag(c), "%.4f s/step" % t if t else "untimed",
                                           "  band %.4f" % b if b else "", "  -> real probe" if c in chosen else ""))
    return chosen


def _afford(ctx, chosen, anchor):
    afford = _real_budget_keep(ctx, len(chosen))
    if afford < len(chosen):
        # short on time: the ANCHOR goes first (the node sweep already measured it at this N),
        # then the slower of the proxy's picks
        picks = [c for c in chosen if c is not anchor] or [anchor]
        chosen = picks[:afford]
        print("proxy: budget left for %d real probe(s): %s" % (afford, ", ".join(_tag(c) for c in chosen)))
    return chosen


def _select_ranked(ctx, configs, keep):
    """select() from rank_across_nodes' ranking at this N (ctx.proxy_ranked): nothing submitted."""
    anchor = configs[0]
    score = {_tag(c): (t, f) for c, t, f in ctx.proxy_ranked}
    seen, ranked = {_tag(anchor)}, []
    for c, _t, _f in ctx.proxy_ranked:
        if _tag(c) not in seen:
            seen.add(_tag(c))
            ranked.append(c)
    untimed = [c for c in configs[1:] if _tag(c) not in seen]
    chosen = _afford(ctx, [anchor] + (ranked or untimed)[:keep - 1], anchor)
    ctx.proxy_used = True
    for i, c in enumerate([anchor] + ranked):
        t, f = score.get(_tag(c), (None, ""))
        print("proxy: %2d %-20s %s%s" % (i, _tag(c), "%.4f s/step %s" % (t, f) if t else "untimed",
                                         "  -> real probe" if c in chosen else ""))
    return chosen


def rank_across_nodes(ctx: Ctx, legal_nodes: list[int]) -> dict:
    """Joint (N, layout) proxy ranking: at every N its ANCHOR plus PROXY_JOINT_PER_N-1 diverse
    layouts, one node each, every N in one batch of jobs; then the PROXY_JOINT_LEADERS one-node
    leaders of each N > PROXY_BAND_NODES re-timed as band proxies, again one batch. Returns
    {N: [(config, s/step, "band" | "one-node"), ...]} best first, band timings ahead of one-node
    ones (one node alone reads 0-18 % low); {} when nothing was timed."""
    case_dir = os.path.dirname(os.path.abspath(ctx.namelist_path))
    pools, runs, base = {}, [], None
    for n in legal_nodes:
        cn = dataclasses.replace(ctx, nodes=n, anchor_override=None)
        anchor = _sweep_anchor(cn, n)[0]
        if anchor is None:
            continue
        more = _apply_search_filter(cn, sample_more_diverse_configs(cn, [anchor], PROXY_JOINT_PER_N - 1,
                                                                    ctx.seed + n))
        pool = {}
        for c in [anchor] + more:
            if _spec(c):
                pool.setdefault(_tag(c), c)
        plan, got = _joint_plan(cn, case_dir, n, pool, 1)
        if got:
            pools[n], runs, base = pool, runs + got, base or plan
    if not runs:
        print("proxy: joint search skipped (no layout reproducible on one node)")
        return {}
    out_root = os.path.join(ctx.out_dir, "proxy-joint")
    print("proxy: joint search over N=%s: %d run(s) on one node each" % (sorted(pools), len(runs)))
    timed = _run_jobs(ctx, dict(base, runs=runs), runs, out_root, "jobs", 1) or {}
    one = {}
    for n, pool in pools.items():
        got = {t: timed["N%d_%s" % (n, t)] for t in pool if "N%d_%s" % (n, t) in timed}
        if got:
            one[n] = got
    band = {}
    if ctx.max_nodes >= PROXY_BAND_NODES:
        bruns, bbase = [], None
        for n in sorted(one):
            if n > PROXY_BAND_NODES:
                lead = sorted(one[n], key=one[n].get)[:PROXY_JOINT_LEADERS]
                plan, got = _joint_plan(ctx, case_dir, n, {t: pools[n][t] for t in lead}, PROXY_BAND_NODES)
                bruns, bbase = bruns + got, bbase or plan
        if bruns:
            band = _run_jobs(ctx, dict(bbase, runs=bruns), bruns, out_root, "band", PROXY_BAND_NODES) or {}
    out = {}
    for n in sorted(one):
        rows = [(pools[n][t], band["N%d_%s" % (n, t)], "band") if "N%d_%s" % (n, t) in band
                else (pools[n][t], v, "one-node") for t, v in one[n].items()]
        out[n] = sorted(rows, key=lambda r: (r[2] != "band", r[1]))
        c, v, f = out[n][0]
        print("proxy: N=%-2d best %-20s %.4f s/step %-8s (%d of %d timed)"
              % (n, _tag(c), v, f, len(one[n]), len(pools[n])))
    if not out:
        print("proxy: joint search produced no usable timing")
    return out


def _joint_plan(ctx, case_dir, n, cfgs, proxy_nodes):
    """make_plan at N for {tag: config}, builds checked, every run's tag and layout prefixed
    N<n>_ so that all N share one plan (the worker reads no plan-level N). (plan, runs), or
    (None, []) when N cannot be planned."""
    try:
        plan = _proxy.make_plan(case_dir, n, [_spec(c) for c in cfgs.values()], cores_used(ctx),
                                steps=ctx.steps, reps=1, wrf_tree=ctx.wrf_tree, proxy_nodes=proxy_nodes)
    except (ValueError, OSError, KeyError) as e:
        print("proxy: N=%d skipped (%s)" % (n, e))
        return None, []
    errors = _proxy.check_builds(plan)
    return plan, [dict(r, tag="N%d_%s" % (n, r["tag"]), layout="N%d_%s" % (n, r["layout"]))
                  for r in plan["runs"] if not any(e.startswith(r["tag"] + ":") for e in errors)]


def _run_jobs(ctx, plan, runs, out_root, prefix, nodes_per_job):
    """Spread `runs` over proxy jobs of nodes_per_job nodes each (as many as max_nodes holds), wait,
    and return {layout tag: score}; None when no job could be submitted."""
    n_jobs = max(1, min(ctx.max_nodes // nodes_per_job, math.ceil(len(runs) / PROXY_RUNS_PER_JOB)))
    job_ids, out_dirs = [], []
    for j in range(n_jobs):
        d = os.path.join(out_root, "%s%d" % (prefix, j))
        job, msg = _proxy.submit_plan(dict(plan, runs=runs[j::n_jobs]), d, _proxy_time_limit(ctx))
        if job:
            job_ids.append(job)
            out_dirs.append(d)
            ctx.owned_jobs.append(job)
        else:
            print("proxy: submit failed for %s: %s" % (d, msg))
    if not job_ids:
        return None
    print("proxy: %d run(s) on %d %d-node job(s) %s" % (len(runs), len(job_ids), nodes_per_job, ",".join(job_ids)))
    _jobs.wait_for_jobs(ctx, job_ids, "proxy")
    timed = {}
    for d in out_dirs:
        try:
            for r in _proxy.collect(d):
                timed[r["tag"]] = r["median"]
        except OSError:
            pass
    return timed


def _band_round(ctx, case_dir, pool, timed, out_root):
    """Second round: the one-node proxy's PROXY_BAND_KEEP fastest are re-timed as band proxies on
    PROXY_BAND_NODES nodes (full width, whole process-grid rows per node: the target's interior
    node). On karina N=8 the band was within 1.7 % of the real time where one node was 18 % low
    (omp=4). Skipped when N is not larger than the band or max_nodes cannot hold one; {} then."""
    if ctx.nodes <= PROXY_BAND_NODES or ctx.max_nodes < PROXY_BAND_NODES:
        return {}
    leaders, seen = [], {_tag(pool[0])}     # the ANCHOR gets a real probe anyway
    for c in sorted((c for c in pool[1:] if _spec(c) and _tag(c) in timed), key=lambda c: timed[_tag(c)]):
        if _tag(c) not in seen:
            seen.add(_tag(c))
            leaders.append(c)
    try:
        plan = _proxy.make_plan(case_dir, ctx.nodes, [_spec(c) for c in leaders[:PROXY_BAND_KEEP]], cores_used(ctx),
                                steps=ctx.steps, reps=1, wrf_tree=ctx.wrf_tree, proxy_nodes=PROXY_BAND_NODES)
    except (ValueError, OSError, KeyError) as e:
        print("proxy: band round skipped (%s)" % e)
        return {}
    errors = _proxy.check_builds(plan)
    runs = [r for r in plan["runs"] if not any(e.startswith(r["tag"] + ":") for e in errors)]
    if not runs:
        print("proxy: band round skipped (no leader holds whole process-grid rows per node)")
        return {}
    return _run_jobs(ctx, plan, runs, out_root, "band", PROXY_BAND_NODES) or {}
