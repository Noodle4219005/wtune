"""round0 by the one-node small-domain proxy (default; `--proxy off` restores the real-probe round0).

round0 would measure k sampled layouts at N nodes for real. Instead a wider pool
(PROXY_POOL_FACTOR x k) is timed on one node each (wtune/proxy.py), the jobs spread over up to
max_nodes nodes in parallel. When N > PROXY_BAND_NODES, the PROXY_BAND_KEEP fastest are re-timed as
band proxies (PROXY_BAND_NODES nodes reproducing the target's interior node) and rank by that. Only
the ANCHOR plus the proxy's PROXY_REAL_KEEP-1 fastest go to the real N-node probes (which race and
confirm as before). A nested case runs as one cropped nested run, or split into its domains where
that does not fit (wtune/proxy.py). Configs the proxy cannot reproduce (2-D tiles, dist=cyclic) are not ranked; no
netCDF wrfinput/wrfbdy in the case directory, or no proxy timing at all, falls back to round0's own
selection, with a note.

At O1/O2 (the default level) the node search ranks (N, layout) jointly instead (rank_across_nodes):
every legal N's pool in one batch, and the final (N, layout) is the proxy's fastest, with no real
N-node probe; O1 re-times one leader per N, O2 PROXY_JOINT_LEADERS.
Every proxied layout is timed at wtune.proxy.PROXY_POSITIONS crop positions and scored by the
median over them; the joint search also reports whether an N's positions agree on its fastest.
"""
import dataclasses
import glob
import json
import math
import os
import re
import shutil

import wtune.proxy as _proxy
from . import jobs as _jobs
from ._deps import decomp_space
from .context import QueueCapExceeded
from .candidates import _apply_search_filter, _sweep_anchor, enumerate_full_space, sample_more_diverse_configs
from .context import Ctx, cores_used

PROXY_POOL_FACTOR = 3          # pool = 3 x what round0 would measure
PROXY_POOL_MAX = 24
PROXY_JOB_NODES_FACTOR = 4     # proxy jobs may hold up to 4 x max_nodes nodes at once: max_nodes caps the
                               # run being tuned, not the one-node tuning jobs (queue wait is not charged)
PROXY_MAX_JOBS = 28            # proxy jobs one batch submits: f1 ct112 QOS allows 32 submitted, 28 running
PROXY_POOL_OVERSAMPLE = 4      # joint pool: sample 4x and keep the first proxy-reproducible layouts
PROXY_SEED_OMP = (1, 2, 4)     # joint pool: decomp_space.seed_design shapes at these thread counts
PROXY_SEED_TILES = (12,)       # ... at the numtiles nearest these
PROXY_SEED_X_WIDE_OMP = (1, 2)  # ... plus, at these thread counts, the x-wide patch shapes (patch_x >= patch_y)
PROXY_SEED_X_WIDE_MAX = 5      # ... the most balanced of them first, up to this many per thread count
PROXY_FINALIST_MARGIN = 0.05   # objective time: a finalist slower than the proxy leader by more is dropped
                               # (band and nested crop err <= 2%, run-to-run noise 3%)
PROXY_RUNS_PER_JOB = 4         # ~40 s each on one node -> a few minutes per job
PROXY_REAL_KEEP = 4            # select(): real N-node probes after the proxy, ANCHOR + the 3 fastest
PROXY_BAND_KEEP = 4            # one-node leaders re-timed as band proxies
PROXY_BAND_NODES = 3           # band proxy size: reproduces the target's interior node
PROXY_JOINT_PER_N = 6          # rank_across_nodes: ANCHOR + 5 diverse layouts per N
PROXY_JOINT_LEADERS = 3        # rank_across_nodes: one-node leaders per N > PROXY_BAND_NODES re-timed as bands
PROXY_CLOSE_CALL = 0.03       # node pool: band leaders of one N this close to its best are re-timed once
PROXY_RETIME_LOW = 0.10       # node pool: a re-timed score this far below its one-node score is re-timed once
PROXY_JOINT_FINALISTS = 2      # node search: the N whose proxy scores the objective ranks first
PROXY_TILES_PER_GRID = 2       # joint pool: numtiles variants of one process grid among an N's first layouts
                               # (Sundowner N=4: 8x18 nt1,2,3,4,8 read 1.82-1.88 on one node, five runs alike)
PROXY_PRUNE_SLOWER = 0.25      # objective time: an N whose ANCHOR and seeds are this much slower than
                               # another N's on one node (which reads 0-18 % low) is timed no further
PROXY_NEIGHBORS = 4            # node pool: layouts next to an N's best ANCHOR / seed, timed before its samples


def proxy_node_cap(ctx) -> int:
    """Nodes the proxy jobs may hold at once: --proxy-max-nodes, else PROXY_JOB_NODES_FACTOR x max_nodes."""
    return int(getattr(ctx, "proxy_max_nodes", 0) or 0) or PROXY_JOB_NODES_FACTOR * ctx.max_nodes


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
                                reps=1, wrf_tree=ctx.wrf_tree, positions=1)
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


@dataclasses.dataclass
class Slot:
    """One (N, cores per node) the joint proxy times: `label` prefixes its run tags and names its
    directories (N4 at the full core count, N4c48 at 48)."""
    n: int
    cores: int
    label: str
    ctx: Ctx
    fidelity: str = ""             # how its leaders were re-timed: "band" or "full" (set when queued)


def rank_across_nodes(ctx: Ctx, legal_nodes: list[int], anchor_only_below: int = 0,
                      leaders: int | None = None, cores: list[int] | None = None,
                      cores_by_n: dict | None = None,
                      out_name: str = "proxy-joint", neighbors: bool = False,
                      sample: bool = True, pool_box: dict | None = None,
                      soft_budget_s: float | None = None) -> tuple[dict, dict]:
    """Joint (N, layout) proxy ranking: at every N its ANCHOR plus PROXY_JOINT_PER_N-1 diverse
    layouts, one node each; then the `leaders` (PROXY_JOINT_LEADERS) fastest one-node layouts of
    each N that can be re-timed are, so that N compare: as band proxies when N > PROXY_BAND_NODES,
    as the whole domain on N nodes when 1 < N <= PROXY_BAND_NODES. Every layout is timed at
    PROXY_POSITIONS crop positions where they can be placed (the whole domain has one). The runs
    go to one node pool (_Pool: an N's leaders start as soon as its one-node runs are timed, the
    largest N first); without one, one batch of jobs per round. At the --minutes deadline the
    runs that finished are ranked. Every N is timed at each of `cores` per node (default: the
    cores the layouts may use), or at its own list in `cores_by_n` (the caller's choice for an N
    with no legal or no fitting layout at the full count). Returns ({N:
    [(config, s/step, "band" | "full" | "one-node"), ...]} best first over all its core counts,
    re-timed ahead of one-node ones (one node alone reads 0-18 % low), {N: True when its crop
    positions agree on the fastest layout}); ({}, {}) when nothing was timed. N <
    anchor_only_below is timed with its ANCHOR alone (the objective time: an N under half the
    largest cannot be the fastest). With `neighbors` (node pool only), an N's ANCHOR and seed
    shapes run first, then up to PROXY_NEIGHBORS layouts next to the fastest of them (the grids
    on either side, numtiles on either side), then the sampled ones: at the deadline the samples
    are what is cut. `sample` False times the ANCHOR and the seed shapes only. `pool_box` ({} at
    first) keeps the node pool job for the caller's next call (its "pool": the caller closes it);
    past `soft_budget_s` of ctx.elapsed() the search stops and ranks what finished, the pool job
    left running."""
    leaders = leaders or PROXY_JOINT_LEADERS
    case_dir = os.path.dirname(os.path.abspath(ctx.namelist_path))
    full = cores_used(ctx)
    cores = cores or [full]
    slots, pools, plans, first, explore = {}, {}, {}, {}, {}
    for n in legal_nodes:
        for c in (cores_by_n or {}).get(n, cores):
            label = "N%d" % n if c == full else "N%dc%d" % (n, c)
            cn = dataclasses.replace(ctx, nodes=n, anchor_override=None, core_budget=c)
            anchor = _sweep_anchor(cn, n, cores=c)[0]
            if anchor is None:
                continue
            # oversampled: cyclic and tile-strategy samples cannot be proxied, and they used to take
            # the slots (Big Chill N=4, N=5: the ANCHOR alone was left)
            # the seed shapes first: per thread count the most balanced and the widest-in-x patch.
            # The sampled pool alone missed the fastest class on isc-synth N=8 (omp2, x-wide
            # patches: 8x56 omp2 nt14 2.71 s/step; the pick from the pool, 28x16 omp2 nt14, 3.37)
            seeds = [] if n < anchor_only_below else _seed_layouts(cn)
            more = [] if n < anchor_only_below or not sample else _apply_search_filter(
                cn, sample_more_diverse_configs(cn, [anchor] + seeds,
                                                PROXY_POOL_OVERSAMPLE * (PROXY_JOINT_PER_N - 1), ctx.seed + n))
            pool, cap, per_grid = {}, PROXY_JOINT_PER_N + len(seeds), {}
            for cfg in [anchor] + seeds + more:
                grid = (cfg["nx"], cfg["ny"], int(cfg.get("omp") or 1))
                # the neighbours of the fastest add its numtiles on either side later
                if (_spec(cfg) and len(pool) < cap and _tag(cfg) not in pool
                        and (cfg in ([anchor] + seeds) or per_grid.get(grid, 0) < PROXY_TILES_PER_GRID)):
                    pool[_tag(cfg)] = cfg
                    per_grid[grid] = per_grid.get(grid, 0) + 1
            plan, got = _joint_plan(cn, case_dir, n, pool, 1, label=label)
            if got:
                slots[label] = Slot(n, c, label, cn)
                pools[label], plans[label] = pool, (plan, got)
                if neighbors and n >= anchor_only_below:
                    first[label] = {"%s_%s" % (label, _tag(x)) for x in [anchor] + seeds}
                    explore[label] = (lambda scores, cn=cn, label=label:
                                      _neighbor_layouts(cn, scores, pools[label]))
    if not plans:
        print("proxy: joint search skipped (no layout reproducible on one node)")
        return {}, {}
    out_root = os.path.join(ctx.out_dir, out_name)
    if os.path.isdir(out_root):            # a reused --out: the last search's pool, CLOSE, results
        shutil.rmtree(out_root, ignore_errors=True)
    where = ("N=%s" % sorted({sl.n for sl in slots.values()})
             if all(sl.cores == full for sl in slots.values())
             else ", ".join(sorted(slots, key=lambda k: (slots[k].n, -slots[k].cores))))
    print("proxy: joint search over %s: %d run(s) on one node each"
          % (where, sum(len(g) for _p, g in plans.values())))
    sizes = [PROXY_BAND_NODES if sl.n > PROXY_BAND_NODES else sl.n for sl in slots.values()]
    points = _time_points(ctx, case_dir) if any(1 < z <= ctx.max_nodes for z in sizes) else []
    npool = _pool_nodes(ctx, sum(len(g) for _p, g in plans.values()))
    pool = (pool_box or {}).get("pool") or (_Pool.start(ctx, out_root, npool) if npool else None)
    if pool is not None:
        pool.ctx, pool.soft = ctx, soft_budget_s
        if pool_box is not None:
            pool_box["pool"] = pool
        one, band, one_pos, band_pos = _time_in_pool(ctx, pool, case_dir, pools, plans, out_root,
                                                     leaders, points, slots, first, explore,
                                                     keep=pool_box is not None,
                                                     prune=PROXY_PRUNE_SLOWER if anchor_only_below else None)
    else:
        one, band, one_pos, band_pos = _time_in_jobs(ctx, case_dir, pools, plans, out_root,
                                                     leaders, points, slots)
    out, same = {}, {}
    for k in sorted(one, key=lambda k: (slots[k].n, -slots[k].cores)):
        sl = slots[k]
        hf = sl.fidelity or ("band" if sl.n > PROXY_BAND_NODES else "full")
        # each row keeps its one-node score: a re-timed leader's would be lost, and the N-wide and
        # core-count comparisons need the one-node order (node_search)
        rows = [(dict(pools[k][t], proxy_one_node_s=v), band["%s_%s" % (k, t)], hf)
                if "%s_%s" % (k, t) in band else (dict(pools[k][t], proxy_one_node_s=v), v, "one-node")
                for t, v in one[k].items()]
        rows = sorted(rows, key=lambda r: (r[2] == "one-node", r[1]))
        agree = _same_fastest({t: band_pos.get("%s_%s" % (k, t), {}) for t in pools[k]},
                              {t: one_pos.get("%s_%s" % (k, t), {}) for t in pools[k]})
        cfg, v, f = rows[0]
        print("proxy: %-6s best %-20s %.4f s/step %-8s (%d of %d timed)%s"
              % (k, _tag(cfg), v, f, len(one[k]), len(pools[k]),
                 "" if agree else "  crop positions disagree on the fastest"))
        out[sl.n] = sorted(out.get(sl.n, []) + rows, key=lambda r: (r[2] == "one-node", r[1]))
        same[sl.n] = same.get(sl.n, True) and agree
    if not out:
        print("proxy: joint search produced no usable timing")
    return out, same


def cores_of(cfg: dict, n: int) -> int:
    """Cores per node a ranked layout uses: its ranks x threads over its N nodes."""
    return int(cfg["ranks"]) * int(cfg.get("omp") or 1) // max(1, n)


def _one_node_scores(label, pool, timed, one_pos, nt_used, ctx=None):
    """{layout tag: one-node s/step} of slot `label` from `timed`; a layout timed at the half-tiles
    fallback is re-timed and probed at that numtiles, under its own tag (Big Chill N=5: 7x40 omp2
    nt12 crashed, nt6 scored 2.40, the real nt12 probes hung). Renames `pool` in place."""
    if isinstance(label, int):
        label = "N%d" % label
    for t in list(pool):
        nt = nt_used.get("%s_%s" % (label, t))
        if nt and int(pool[t].get("numtiles") or 1) != nt and "%s_%s" % (label, t) in timed:
            c = dict(pool.pop(t), numtiles=nt)
            if _tag(c) in pool:
                continue                       # that numtiles is in the pool already
            if ctx is not None and c.get("id") != "ANCHOR" and not _apply_search_filter(ctx, [c]):
                timed.pop("%s_%s" % (label, t), None)
                print("proxy: %s %s ran only at numtiles=%d, outside the search: dropped" % (label, t, nt))
                continue
            pool[_tag(c)] = c
            old, new = "%s_%s" % (label, t), "%s_%s" % (label, _tag(c))
            timed[new] = timed.pop(old)
            if old in one_pos:
                one_pos[new] = one_pos.pop(old)
            print("proxy: %s %s timed at numtiles=%d (the fallback): ranked as %s" % (label, t, nt, _tag(c)))
    return {t: timed["%s_%s" % (label, t)] for t in pool if "%s_%s" % (label, t) in timed}


def _leader_plan(ctx, case_dir, n, scores, pool, leaders, points, label=None, max_size=None):
    """(size, plan, runs) re-timing N=n's `leaders` fastest distinct shapes that can be re-timed:
    a band beyond PROXY_BAND_NODES, the whole domain on n nodes at 2..PROXY_BAND_NODES (one node
    alone reads 0-18 % low; N=1's one-node run is the whole domain). A leader that cannot be
    banded (6x56 at N=6: no whole rows per node) gives way to the next; when the fastest cannot
    and N nodes fit (the cap, and `max_size`: the node pool's), the leaders are re-timed on the
    whole domain on N nodes instead. None when nothing can."""
    label = label or "N%d" % n
    size = PROXY_BAND_NODES if n > PROXY_BAND_NODES else n
    cap = min(ctx.max_nodes, proxy_node_cap(ctx))
    if size <= 1 or cap < size or not scores:
        return None
    if max_size and size < n <= min(cap, max_size) and n < 2 * PROXY_BAND_NODES:
        # the node pool holds N nodes and not two bands: the whole domain has one crop position,
        # 3 runs for a band's 6 (Sundowner N=4: the six band runs took 210 s, one at a time)
        size = n
    ranked = _distinct_shapes(sorted(scores, key=scores.get), len(scores))
    plan, got = _joint_plan(ctx, case_dir, n, {t: pool[t] for t in ranked}, size, points, label=label)
    if size < n <= min(cap, max_size or cap):
        # a leader with no whole process-grid rows per node cannot be banded, and the next took its
        # place: Sundowner N=4 at 36 cores, 8x18 (4.5 rows a node) read 1.82-1.87 on one node, the
        # fastest of three tunes, and was never re-timed. The whole domain on N nodes takes all
        banded = {r["layout"] for r in got}
        lost = [t for t in ranked[:1] if "%s_%s" % (label, t) not in banded]
        if lost:
            print("proxy: %s %s cannot be banded (no whole process-grid rows per node): its leaders "
                  "re-timed on the whole domain on %d nodes" % (label, ", ".join(lost), n))
            size = n
            plan, got = _joint_plan(ctx, case_dir, n, {t: pool[t] for t in ranked}, size, points, label=label)
    if not got:
        return None
    keep = [t for t in ranked if any(r["layout"] == "%s_%s" % (label, t) for r in got)][:leaders]
    runs = [r for r in got if r["layout"] in {"%s_%s" % (label, t) for t in keep}]
    return size, dict(plan, proxy_nodes=size), runs


def _time_in_jobs(ctx, case_dir, pools, plans, out_root, leaders, points, slots=None):
    """One batch of one-node jobs for every slot, then one batch of re-timing jobs per size: (one,
    band, one_pos, band_pos)."""
    slots = slots or {k: Slot(k, cores_used(ctx) if ctx is not None else 0, "N%d" % k, ctx)
                      for k in plans}
    base = next(iter(plans.values()))[0]
    runs = [r for _p, got in plans.values() for r in got]
    one_pos, band_pos, nt_used = {}, {}, {}
    try:
        timed = _run_jobs(ctx, dict(base, runs=runs), runs, out_root, "jobs", 1, one_pos, nt_used) or {}
        halted = None
    except QueueCapExceeded as e:
        timed, halted = e.partial or {}, e
    one = {}
    for k in sorted(pools, key=lambda k: (slots[k].n, -slots[k].cores)):
        got = _one_node_scores(slots[k].label, pools[k], timed, one_pos, nt_used, slots[k].ctx)
        if got:
            one[k] = got
    if halted is not None:
        print("proxy: %s -- ranked from the one-node runs that finished, none re-timed" % halted)
        return one, {}, one_pos, band_pos
    by_size = {}
    for k in one:
        sl = slots[k]
        lp = _leader_plan(sl.ctx, case_dir, sl.n, one[k], pools[k], leaders, points, label=sl.label)
        if lp:
            size, plan, got = lp
            sl.fidelity = "full" if size == sl.n else "band"
            # one plan per size and core count: a plan's cores launch every run in it
            prev = by_size.get((size, sl.cores), (plan, []))
            by_size[(size, sl.cores)] = (prev[0], prev[1] + got)
    # every size at once: the whole-domain runs (N <= PROXY_BAND_NODES) and the bands do not wait
    # on each other (isc-synth: 2-node then 3-node rounds ran one after the other, ~2 min each)
    handles, band = [], {}
    try:
        for (size, c), (bbase, bruns) in sorted(by_size.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
            handles.append(_submit_jobs(ctx, dict(bbase, runs=bruns, proxy_nodes=size), bruns, out_root,
                                        "band%d-%s" % (size, "" if c == max(x.cores for x in slots.values())
                                                       else "c%d-" % c), size))
    except QueueCapExceeded as e:          # stopped while waiting to submit: rank what finished
        print("proxy: %s -- ranked from the re-timing runs that finished" % e)
        band.update(e.partial or {})
        by_size = {}
    handles = [h for h in handles if h]
    if handles:
        try:
            _jobs.wait_for_jobs(ctx, [j for h in handles for j in h["live"]], "proxy")
        except QueueCapExceeded as e:
            print("proxy: %s -- ranked from the re-timing runs that finished" % e)
    for h in handles:
        band.update(_collect_jobs(h, band_pos))
    return one, band, one_pos, band_pos


def _time_in_pool(ctx, pool, case_dir, pools, plans, out_root, leaders, points, slots=None,
                  first=None, explore=None, keep=False, prune=None):
    """Every slot's one-node runs queued at once, the largest N first; a slot's leaders queued as
    soon as its one-node runs are timed, ahead of the one-node runs still waiting. Stops at the
    --minutes deadline (the pool job is cancelled) or when the pool job ends; what finished is
    ranked. Band leaders within PROXY_CLOSE_CALL of their slot's best, at a slot within
    PROXY_FINALIST_MARGIN of the best band so far, are re-timed once on other nodes: one band run
    varies 2-5 % run to run (isc-synth N=8, three tunes), more than the gap it must resolve.
    A slot in `first` ({label: its first layouts}) queues those first and its other one-node runs
    last; once the first are timed, explore[label]({tag: one-node s/step}) -> {tag: config} gives
    the layouts queued between them. With `prune`, a slot whose first layouts are that much
    slower on one node than another slot's at the same cores per node is timed no further: its
    queued runs are dropped and it is not re-timed.
    (one, band, one_pos, band_pos), keyed by slot label."""
    first, explore = first or {}, explore or {}
    slots = slots or {k: Slot(k, cores_used(ctx) if ctx is not None else 0, "N%d" % k, ctx)
                      for k in plans}
    full = max(sl.cores for sl in slots.values())
    lab = lambda k: slots[k].label                                              # noqa: E731
    one_dir = lambda k: os.path.join(out_root, "one-%s" % lab(k))               # noqa: E731
    band_dir = lambda k: os.path.join(out_root, "band-%s" % lab(k))             # noqa: E731
    prio = lambda k: "%03d%s" % (999 - slots[k].n, "" if slots[k].cores == full  # noqa: E731
                                 else "-c%03d" % (999 - slots[k].cores))
    nb_dir = lambda k: os.path.join(out_root, "one-%s-near" % lab(k))          # noqa: E731
    rest_dir = lambda k: os.path.join(out_root, "one-%s-rest" % lab(k))        # noqa: E731
    order = sorted(plans, key=lambda k: (-slots[k].n, -slots[k].cores))
    dirs_of, early = {}, []
    for k in order:
        plan, got = plans[k]
        dirs_of[k] = [one_dir(k)]
        a = [r for r in got if r["layout"] in first[k]] if k in first else got
        if k in first and a:
            # the ANCHOR and the seed shapes, then (queued once they are timed) their neighbours,
            # then the samples: "-1" < "-2" < "-3" within the slot, every slot still by N
            pool.add(one_dir(k), plan, a, 1, "1-%s-1" % prio(k))
            rest = [r for r in got if r["layout"] not in first[k]]
            if rest:
                pool.add(rest_dir(k), plan, rest, 1, "1-%s-3" % prio(k))
                dirs_of[k].append(rest_dir(k))
            early.append(k)
        else:
            pool.add(one_dir(k), plan, got, 1, "1-" + prio(k))
    one, band, one_pos, band_pos, nt_used = {}, {}, {}, {}, {}
    waiting, banding, retiming, queued = [k for k in order if k not in early], [], [], {}
    lead, pruned = {}, set()               # {cores: the fastest first-tier one-node score so far}

    def all_done(k):
        return all(pool.done(d) for d in dirs_of[k])

    def queue_near(k):
        timed = _collect_jobs(dict(out_dirs=[one_dir(k)]))
        scores = {t: timed["%s_%s" % (lab(k), t)] for t in pools[k] if "%s_%s" % (lab(k), t) in timed}
        c = slots[k].cores
        if scores and prune and c in lead and min(scores.values()) > lead[c] * (1 + prune):
            print("proxy: %s fastest of its ANCHOR and seeds %.4f s/step on one node, more than %d%% "
                  "slower than another N's %.4f: timed no further"
                  % (lab(k), min(scores.values()), round(100 * prune), lead[c]))
            pruned.add(k)
            for d in dirs_of[k][1:]:
                pool.drop(d)
            return
        if scores:
            lead[c] = min(lead.get(c, float("inf")), min(scores.values()))
        near = explore[k](scores) if k in explore and scores else {}
        # a neighbour already queued among the samples moves up to the neighbours' place
        up = pool.promote(rest_dir(k), {"%s_%s" % (lab(k), t) for t in near if t in pools[k]},
                          "1-%s-3" % prio(k), "1-%s-2" % prio(k)) if rest_dir(k) in dirs_of[k] else []
        if up:
            print("proxy: %s %s moved ahead of its other sampled layouts (next to %s)"
                  % (lab(k), ", ".join(sorted(x[len(lab(k)) + 1:] for x in up)), min(scores, key=scores.get)))
        new = {t: c for t, c in near.items() if t not in pools[k]}
        if not new:
            return
        sl = slots[k]
        plan, got = _joint_plan(sl.ctx, case_dir, sl.n, new, 1, label=sl.label)
        if got:
            pools[k].update({t: c for t, c in new.items() if any(r["layout"] == "%s_%s" % (lab(k), t) for r in got)})
            pool.add(nb_dir(k), plan, got, 1, "1-%s-2" % prio(k))
            dirs_of[k].append(nb_dir(k))
            print("proxy: %s fastest of its ANCHOR and seeds %s: %s queued ahead of its sampled layouts"
                  % (lab(k), min(scores, key=scores.get), ", ".join(sorted({r["layout"][len(lab(k)) + 1:] for r in got}))))

    def take_one(k):
        timed = _collect_jobs(dict(out_dirs=dirs_of.get(k, [one_dir(k)])), one_pos, nt_used)
        got = _one_node_scores(lab(k), pools[k], timed, one_pos, nt_used, slots[k].ctx)
        if got:
            one[k] = got
        return got

    while early or waiting or banding or retiming:
        dirs = [one_dir(k) for k in early] + [band_dir(k) for k in banding + retiming]
        if not pool.wait(lambda: any(pool.done(d) for d in dirs) or any(all_done(k) for k in waiting)):
            break
        for k in [k for k in early if pool.done(one_dir(k))]:
            early.remove(k)
            queue_near(k)
            waiting.append(k)
        for k in [k for k in waiting if all_done(k)]:
            waiting.remove(k)
            got = take_one(k)
            sl = slots[k]
            lp = _leader_plan(sl.ctx, case_dir, sl.n, got, pools[k], leaders, points,
                              label=sl.label, max_size=pool.nodes) if got and k not in pruned else None
            if lp and lp[0] <= pool.nodes:
                size, plan, bruns = lp
                sl.fidelity = "full" if size == sl.n else "band"
                pool.add(band_dir(k), plan, bruns, size, "0-" + prio(k))
                banding.append(k)
                queued[k] = lp
        for k in [k for k in banding if pool.done(band_dir(k))]:
            banding.remove(k)
            band.update(_collect_jobs(dict(out_dirs=[band_dir(k)]), band_pos))
            again = _close_calls(band, queued, k)
            low = _too_fast(band, one.get(k, {}), queued[k][2], lab(k))
            if low:
                print("proxy: %s %s re-timed %.0f%%+ below its one-node score (one node reads 0-18%% "
                      "low): re-timed twice" % (lab(k), ", ".join(sorted({
                          r.get("layout", r["tag"])[len(lab(k)) + 1:] for r in low})), PROXY_RETIME_LOW * 100))
                again = again + [r for r in low if r not in again]
            if again:
                size, plan, _r = queued[k]
                # a low reading needs two more: the median of two runs is their mean, and
                # (1.29 + 2.28) / 2 is still wrong
                plan = dict(plan, reps=2 if low else 1)
                print("proxy: %s %s within %.0f%% of each other in one band run: re-timed once"
                      % (lab(k), ", ".join(sorted({r.get("layout", r["tag"])[len(lab(k)) + 1:]
                                                   for r in again})), PROXY_CLOSE_CALL * 100))
                pool.add(band_dir(k), plan, again, size, "0-r" + prio(k), rep0=2)
                retiming.append(k)
        for k in [k for k in retiming if pool.done(band_dir(k))]:
            retiming.remove(k)
            band.update(_collect_jobs(dict(out_dirs=[band_dir(k)]), band_pos))
    if early or waiting or banding:
        print("proxy: stopped with %s not fully timed (%s): ranked from the runs that finished"
              % (_stopped_names(slots, early + waiting + banding),
                 "deadline" if ctx.elapsed() >= min(ctx.budget_s, getattr(pool, "soft", None) or ctx.budget_s)
                 else getattr(pool, "halted", None) or "the node pool job ended"))
        for k in early + waiting:
            take_one(k)
        for k in banding:
            band.update(_collect_jobs(dict(out_dirs=[band_dir(k)]), band_pos))
    for k in retiming:                     # the first runs count; finished re-runs join them
        band.update(_collect_jobs(dict(out_dirs=[band_dir(k)]), band_pos))
    if keep:
        pool.clear()                       # the caller queues more: this search's leftovers go
    else:
        pool.close(stopped=bool(early or waiting or banding or retiming))
    return one, band, one_pos, band_pos


def _stopped_names(slots, keys) -> str:
    """'N=[2, 4]' at full cores, else the slot labels."""
    labels = sorted({slots[k].label for k in keys})
    if all(re.fullmatch(r"N\d+", x) for x in labels):
        return "N=%s" % sorted({slots[k].n for k in keys})
    return ", ".join(labels)


def _too_fast(band, one_k, runs, label):
    """Slot `label`'s band / whole-domain runs (every crop position) of the layouts whose re-timed
    score is more than PROXY_RETIME_LOW below their one-node score. One node alone reads 0-18 %
    low, so such a score is a bad reading: Sundowner N=3 6x18 nt3 whole domain read 1.286 s/step
    in the pool (27 s for the run) against 2.087 on one node; the same run alone read 2.280 and
    the real run 2.265 on the same nodes."""
    bad = {"%s_%s" % (label, t) for t, v in one_k.items()
           if "%s_%s" % (label, t) in band and band["%s_%s" % (label, t)] < v * (1 - PROXY_RETIME_LOW)}
    return [r for r in runs if r.get("layout", r["tag"]) in bad]


def _close_calls(band, queued, n):
    """N=n's band runs (all its crop positions) of the layouts within PROXY_CLOSE_CALL of its
    best, when there are two or more and n is within PROXY_FINALIST_MARGIN of the best band."""
    lays = lambda m: {r.get("layout", r["tag"]) for r in queued[m][2]}      # noqa: E731
    mine = {t: band[t] for t in lays(n) if t in band}
    best_all = [min(v for t, v in band.items() if t in lays(m)) for m in queued
                if any(t in band for t in lays(m))]
    if len(mine) < 2 or min(mine.values()) > min(best_all) * (1 + PROXY_FINALIST_MARGIN):
        return []
    close = {t for t, v in mine.items() if v <= min(mine.values()) * (1 + PROXY_CLOSE_CALL)}
    return [r for r in queued[n][2] if r.get("layout", r["tag"]) in close] if len(close) > 1 else []


POOL_TIME_LIMIT_MIN = 180      # node pool's Slurm --time without a --minutes budget


def _pool_nodes(ctx, one_node_runs: int) -> int:
    """Nodes of the joint search's node pool: proxy_node_cap, no more than there are one-node
    runs (but room for a band); 0 (one job per batch) when the jobs are not submitted for real
    (tests' fake runners) or --proxy-max-nodes is below a band."""
    if getattr(ctx, "sbatch_runner", None) is not _jobs.sweep.default_sbatch_runner:
        return 0
    return min(proxy_node_cap(ctx), max(one_node_runs, PROXY_BAND_NODES))


def _pool_time_limit(ctx) -> str:
    budget = getattr(ctx, "budget_s", float("inf"))
    m = (math.ceil((budget - ctx.elapsed()) / 60.0) + 5 if math.isfinite(budget)
         else POOL_TIME_LIMIT_MIN)
    m = max(10, min(POOL_TIME_LIMIT_MIN, m))
    return "%02d:%02d:00" % (m // 60, m % 60)


class _Pool:
    """One job of `nodes` nodes serving <out_root>/pool as a queue (wtune.proxy.pool_worker):
    each queued run starts as soon as its nodes are free, so no node waits on a batch's slowest
    job and a 3-node band does not leave nodes idle that a 1-node run can use."""

    def __init__(self, ctx, root, nodes, job):
        self.ctx, self.root, self.nodes, self.job = ctx, root, nodes, job
        self.seq, self.batches, self.halted, self.soft = 0, {}, None, None

    @classmethod
    def start(cls, ctx, out_root, nodes):
        root = os.path.join(out_root, "pool")
        for d in ("tasks", "running", "done"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        # a pool the scheduler refuses is tried at half the nodes, down to max_nodes: without
        # --proxy-max-nodes the cap is 4 x max_nodes, and f1's QOS took no 32-node job
        # (QOSMaxCpuPerJobLimit), which sent isc-synth's 73 runs to 38 separate jobs
        floor = min(nodes, max(PROXY_BAND_NODES, int(getattr(ctx, "max_nodes", 0) or 0)))
        while True:
            try:
                # the tune's own --cores-per-node: a run of 72 ranks needs 72 tasks per pool node
                job, msg = _proxy.submit_pool(root, nodes, _pool_time_limit(ctx),
                                              tasks_per_node=getattr(ctx, "cores_per_node", None))
            except OSError as e:
                job, msg = None, str(e)
            if job or nodes <= floor:
                break
            print("proxy: node pool of %d node(s) not submitted (%s); trying %d"
                  % (nodes, str(msg).strip()[:100], max(floor, nodes // 2)))
            nodes = max(floor, nodes // 2)
        if not job:
            print("proxy: node pool of %d node(s) not submitted (%s); one job per batch instead"
                  % (nodes, str(msg).strip()[:100]))
            return None
        ctx.owned_jobs.append(job)
        print("proxy: node pool job %s, %d node(s)" % (job, nodes))
        return cls(ctx, root, nodes, job)

    def add(self, batch_dir, plan, runs, nodes, first, rep0=1):
        """Queue `runs` (plan.json in batch_dir), each on `nodes` nodes; tasks sort by `first`.
        rep0 > 1 queues further reps of runs already queued there."""
        os.makedirs(batch_dir, exist_ok=True)
        if rep0 == 1:
            with open(os.path.join(batch_dir, "plan.json"), "w") as f:
                json.dump(dict(plan, runs=runs), f, indent=1)
        names = []
        for rep in range(rep0, rep0 + int(plan.get("reps", 1))):
            for r in runs:
                self.seq += 1
                name = "%s-%06d.json" % (first, self.seq)
                tmp = os.path.join(self.root, name + ".tmp")
                with open(tmp, "w") as f:
                    json.dump(dict(batch=batch_dir, run=r, rep=rep, nodes=nodes), f)
                os.replace(tmp, os.path.join(self.root, "tasks", name))
                names.append(name)
        self.batches[batch_dir] = names

    def promote(self, batch_dir, layouts, old, new) -> list:
        """Rename batch_dir's tasks of `layouts` still queued from prefix `old` to `new` (the
        worker takes tasks in name order; one it took meanwhile is left). The layouts moved."""
        moved, names = set(), []
        for name in self.batches.get(batch_dir, []):
            src = os.path.join(self.root, "tasks", name)
            if name.startswith(old + "-") and os.path.exists(src):
                try:
                    lay = json.load(open(src))["run"].get("layout")
                except (OSError, ValueError, KeyError):
                    lay = None
                if lay in layouts:
                    dst = new + name[len(old):]
                    try:
                        os.replace(src, os.path.join(self.root, "tasks", dst))
                        name = dst
                        moved.add(lay)
                    except FileNotFoundError:
                        pass                   # the worker took it
            names.append(name)
        self.batches[batch_dir] = names
        return sorted(moved)

    def done(self, batch_dir):
        d = os.path.join(self.root, "done")
        return all(os.path.exists(os.path.join(d, n)) for n in self.batches.get(batch_dir, []))

    def wait(self, pred) -> bool:
        """Wait, the running budget charged while the pool runs, until pred(); False when the pool
        job ended or the deadline cancelled it first."""
        over = lambda: self.soft is not None and self.ctx.elapsed() >= self.soft   # noqa: E731
        if pred():
            return True
        if over():
            return False
        try:
            # the soft deadline ends the wait, not the job (wait_for_jobs cancels at ctx.budget_s)
            _jobs.wait_for_jobs(self.ctx, [self.job], "proxy", until=lambda: pred() or over())
        except QueueCapExceeded as e:      # --max-queue-hours or wtune stop: what finished counts
            self.halted = e
            return False
        return pred()

    def drop(self, batch_dir):
        """Remove batch_dir's tasks still queued; the ones running or done stay its batch."""
        left = []
        for name in self.batches.get(batch_dir, []):
            try:
                os.remove(os.path.join(self.root, "tasks", name))
            except OSError:
                left.append(name)          # the worker took it
        self.batches[batch_dir] = left

    def clear(self):
        """Drop the tasks still queued (running ones finish); the pool job stays for more."""
        for name in os.listdir(os.path.join(self.root, "tasks")):
            try:
                os.remove(os.path.join(self.root, "tasks", name))
            except OSError:
                pass                       # the worker took it

    def close(self, stopped=False):
        open(os.path.join(self.root, "CLOSE"), "w").close()
        if stopped:                        # cancelled worker: its runs' crops are left behind
            shutil.rmtree(os.path.join(self.root, "scratch"), ignore_errors=True)


def _same_fastest(*sources):
    """False when the first of `sources` ({layout: {crop position: score}}) with >= 2 layouts timed
    at positions 0 and 1 has a different fastest layout at each; True otherwise."""
    for src in sources:
        both = {t: p for t, p in src.items() if 0 in p and 1 in p}
        if len(both) >= 2:
            return min(both, key=lambda t: both[t][0]) == min(both, key=lambda t: both[t][1])
    return True


def _joint_plan(ctx, case_dir, n, cfgs, proxy_nodes, time_points=(), label=None):
    """make_plan at N for {tag: config}, builds checked, every run's tag and layout prefixed
    N<n>_ so that all N share one plan (the worker reads no plan-level N). (plan, runs), or
    (None, []) when N cannot be planned."""
    try:
        plan = _proxy.make_plan(case_dir, n, [_spec(c) for c in cfgs.values()], cores_used(ctx),
                                steps=ctx.steps, reps=1, wrf_tree=ctx.wrf_tree, proxy_nodes=proxy_nodes,
                                positions=_proxy.PROXY_POSITIONS if proxy_nodes > 1 else 1,
                                time_points=time_points)
    except (ValueError, OSError, KeyError) as e:
        print("proxy: N=%d skipped (%s)" % (n, e))
        return None, []
    errors = _proxy.check_builds(plan)
    label = label or "N%d" % n
    return plan, [dict(r, tag="%s_%s" % (label, r["tag"]), layout="%s_%s" % (label, r["layout"]))
                  for r in plan["runs"] if not any(e.startswith(r["tag"] + ":") for e in errors)]


def _run_jobs(ctx, plan, runs, out_root, prefix, nodes_per_job, by_pos=None, nt_used=None):
    """Spread `runs` over proxy jobs of nodes_per_job nodes each (as many as max_nodes holds), wait,
    and return {layout tag: score} (and fill by_pos with {layout tag: {crop position: score}});
    None when no job could be submitted. A layout's runs (domains, crop positions, time points)
    share one job: collect() scores a layout from one job's results."""
    h = _submit_jobs(ctx, plan, runs, out_root, prefix, nodes_per_job)
    if h is None:
        return None
    try:
        _jobs.wait_for_jobs(ctx, h["live"], "proxy")
    except QueueCapExceeded as e:          # the jobs that finished before the stop still count
        e.partial = _collect_jobs(h, by_pos, nt_used)
        raise
    return _collect_jobs(h, by_pos, nt_used)


def _distinct_shapes(tags: list[str], k: int) -> list[str]:
    """The first k of `tags` (best first) with distinct patch shapes (nx x ny, omp): numtiles
    variants of one shape rank alike, and two of them took both band slots on isc-synth N=8
    (7x64 omp2 nt12, nt6) while 16x28 omp2 nt12, one-node 2.673, was never re-timed. Tile variants
    fill in when there are fewer than k shapes."""
    shape = lambda t: t.rsplit("_nt", 1)[0]
    out, seen = [], set()
    for t in tags:
        if shape(t) not in seen:
            seen.add(shape(t))
            out.append(t)
    return (out + [t for t in tags if t not in out])[:k]


def _neighbor_space(ctx) -> list[dict]:
    """Every legal block layout at ctx.nodes x cores_used(ctx), search filter applied; [] when the
    space cannot be enumerated."""
    try:
        return enumerate_full_space(ctx)[0]
    except (KeyError, ValueError, TypeError):
        return []


def _neighbor_layouts(ctx, scores, pool) -> dict:
    """{tag: config} of up to PROXY_NEIGHBORS legal layouts next to the fastest of `scores` ({tag:
    one-node s/step}, configs in `pool`) that are not timed there yet (a neighbour in `pool` is
    returned as it is there): at its thread count, the process grids on either
    side of its (by nx) at the numtiles nearest its, and its grid at the numtiles on either side.
    Measured ahead of the sampled layouts, whose order says nothing about their speed."""
    best = pool.get(min(scores, key=scores.get)) if scores else None
    if best is None:
        return {}
    omp, nt = int(best.get("omp") or 1), int(best.get("numtiles") or 1)
    space = [c for c in _neighbor_space(ctx) if int(c.get("omp") or 1) == omp and _spec(c)]
    grids = sorted({(c["nx"], c["ny"]) for c in space} | {(best["nx"], best["ny"])})
    i = grids.index((best["nx"], best["ny"]))
    on = lambda g: [c for c in space if (c["nx"], c["ny"]) == g]          # noqa: E731
    picks = []
    for g in grids[max(0, i - 1):i] + grids[i + 1:i + 2]:
        if on(g):
            picks.append(min(on(g), key=lambda c: (abs(int(c.get("numtiles") or 1) - nt), c.get("numtiles") or 1)))
    tiles = sorted({int(c.get("numtiles") or 1) for c in on(grids[i])} | {nt})
    j = tiles.index(nt)
    for t in tiles[max(0, j - 1):j] + tiles[j + 1:j + 2]:
        picks += [c for c in on(grids[i]) if int(c.get("numtiles") or 1) == t][:1]
    out = {}
    for c in picks:
        t = _tag(c)
        if t not in scores and t not in out and len(out) < PROXY_NEIGHBORS:
            out[t] = pool[t] if t in pool else dict(c, id="B%04d" % (1 + sum(t2 not in pool for t2 in out)))
    return out


def _seed_layouts(ctx) -> list[dict]:
    """decomp_space.seed_design at ctx.nodes over PROXY_SEED_OMP (the most balanced and the
    widest-in-x patch per thread count) plus up to PROXY_SEED_X_WIDE_MAX x-wide shapes per thread
    count in PROXY_SEED_X_WIDE_OMP, numtiles nearest PROXY_SEED_TILES, as campaign configs
    (ids S0001..); [] when the space cannot be enumerated."""
    try:
        space, _axis = decomp_space.enumerate_space(ctx.case_json, ctx.nodes * cores_used(ctx), ctx.nodes,
                                                    ctx.hardware_id)
    except (KeyError, ValueError, TypeError):
        return []
    space = _apply_search_filter(ctx, [c for c in space if c.get("omp") in PROXY_SEED_OMP])
    picks = decomp_space.seed_design(space, numtiles_targets=PROXY_SEED_TILES, max_omp=max(PROXY_SEED_OMP))
    # and the x-wide shapes between those two ends: conus25 N=8's fastest omp2 patches, 14x32,
    # 16x28 and 8x56 (0.371-0.375 s/step), lie between the balanced 28x16 (0.42) and the widest
    # 4x112 and 7x64; isc-synth N=8's fastest is 8x56 omp2
    have = {(c["nx"], c["ny"], c["omp"]) for c in picks}
    aspect = lambda c: c["patch_x"] / float(c["patch_y"])
    for omp in PROXY_SEED_X_WIDE_OMP:
        group = [c for c in space if c["omp"] == omp and c.get("tile_shape", "1d_y") == "1d_y"
                 and c.get("dist", "block") == "block" and str(c.get("blocktime", "0")) == "0"
                 and c.get("patch_x") and c.get("patch_y") and c["patch_x"] >= c["patch_y"]]
        shapes = sorted({(c["nx"], c["ny"]) for c in group},
                        key=lambda xy: min(aspect(c) for c in group if (c["nx"], c["ny"]) == xy))
        for xy in shapes[:PROXY_SEED_X_WIDE_MAX]:
            if (xy[0], xy[1], omp) in have:
                continue
            shape = [c for c in group if (c["nx"], c["ny"]) == xy]
            picks.append(min(shape, key=lambda c: (abs(c["numtiles"] - PROXY_SEED_TILES[0]), c["numtiles"])))
            have.add((xy[0], xy[1], omp))
    return decomp_space.to_campaign_configs(picks, prefix="S") if picks else []


def _layout_nodes(tag: str) -> int:
    m = re.match(r"N(\d+)_", tag)
    return int(m.group(1)) if m else 0


def _submit_jobs(ctx, plan, runs, out_root, prefix, nodes_per_job):
    """The submit half of _run_jobs: {"live": job ids still to wait for, "out_dirs"}, or None."""
    groups = {}
    for r in runs:
        groups.setdefault(r.get("layout", r["tag"]), []).append(r)
    # dearest first (fewest nodes: the most cells per rank), so that when layouts must share a job
    # the dear ones run alone and the cheap ones pair up
    groups = sorted(groups.values(), key=lambda g: _layout_nodes(g[0].get("layout", g[0]["tag"])))
    # one job per layout up to the nodes there are: a nested or band run costs as much per step
    # as the real N-node run, so packing PROXY_RUNS_PER_JOB of them into a job serialises minutes
    n_jobs = max(1, min(proxy_node_cap(ctx) // nodes_per_job, len(groups),
                        PROXY_MAX_JOBS))
    # greedy, dearest first onto the least loaded job; a layout costs ~1/N (cells per rank)
    packed, load = [[] for _ in range(n_jobs)], [0.0] * n_jobs
    for g in groups:
        k = load.index(min(load))
        packed[k].append(g)
        load[k] += 1.0 / max(1, _layout_nodes(g[0].get("layout", g[0]["tag"])))
    job_ids, out_dirs, ended = [], [], []
    j = 0
    while j < n_jobs:
        d = os.path.join(out_root, "%s%d" % (prefix, j))
        job_runs = [r for g in packed[j] for r in g]
        job, msg = _proxy.submit_plan(dict(plan, runs=job_runs), d, _proxy_time_limit(ctx),
                                      tasks_per_node=getattr(ctx, "cores_per_node", None))
        if job:
            job_ids.append(job)
            out_dirs.append(d)
            ctx.owned_jobs.append(job)
            j += 1
            continue
        live = [x for x in job_ids if x not in ended]
        if not live:          # nothing of ours to wait for: the rest cannot be submitted
            print("proxy: submit failed for %s: %s" % (d, msg))
            j += 1
            continue
        # a per-user submit limit (f1 ct112 QOS: 32 submitted) is shared with every other job of
        # the user: wait for our oldest job to end, then submit the rest
        print("proxy: submit refused for %s (%s); waiting for job %s first" % (d, msg.strip()[:80], live[0]))
        try:
            _jobs.wait_for_jobs(ctx, [live[0]], "proxy")
        except QueueCapExceeded as e:      # the jobs submitted so far must not outlive the stop
            _jobs.cancel_jobs(ctx, [x for x in live[1:] if x not in ctx.cancelled_jobs])
            e.partial = _collect_jobs(dict(live=[], out_dirs=out_dirs))
            raise
        ended.append(live[0])
    if not job_ids:
        return None
    print("proxy: %d run(s) on %d %d-node job(s) %s" % (len(runs), len(job_ids), nodes_per_job, ",".join(job_ids)))
    return dict(live=[x for x in job_ids if x not in ended], out_dirs=out_dirs)


def _collect_jobs(h, by_pos=None, nt_used=None):
    timed = {}
    for d in h["out_dirs"]:
        try:
            for r in _proxy.collect(d):
                timed[r["tag"]] = r["median"]
                if nt_used is not None and r.get("numtiles"):
                    nt_used[r["tag"]] = r["numtiles"]
                if by_pos is not None:
                    by_pos[r["tag"]] = r.get("positions") or {}
        except OSError:
            pass
        _proxy.drop_runs(d)
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
                                steps=ctx.steps, reps=1, wrf_tree=ctx.wrf_tree, proxy_nodes=PROXY_BAND_NODES,
                                positions=_proxy.PROXY_POSITIONS, time_points=_time_points(ctx, case_dir))
    except (ValueError, OSError, KeyError) as e:
        print("proxy: band round skipped (%s)" % e)
        return {}
    errors = _proxy.check_builds(plan)
    runs = [r for r in plan["runs"] if not any(e.startswith(r["tag"] + ":") for e in errors)]
    if not runs:
        print("proxy: band round skipped (no leader holds whole process-grid rows per node)")
        return {}
    return _run_jobs(ctx, plan, runs, out_root, "band", PROXY_BAND_NODES) or {}


def _time_points(ctx, case_dir):
    """The leaders' time points (wtune.proxy module docstring): the first PROXY_TIME_POINTS
    completed prediction-point rebuilds of this tune, <out_dir>/rebuilt/pt_<i>/win_0/real; [] with
    a note when there are none (nothing is fetched)."""
    root = os.path.join(ctx.out_dir, "rebuilt")
    done = sorted(glob.glob(os.path.join(root, "pt_*", "namelist.template.input")),
                  key=lambda f: (len(f), f))          # pt_2 before pt_10
    points = []
    for f in done:
        d = os.path.join(os.path.dirname(f), "win_0", "real")
        why = _proxy.time_point_ok(case_dir, d)
        if why:
            print("proxy: %s is not a time point (%s)" % (d, why))
        else:
            points.append(d)
    if not points:
        print("proxy: no completed rebuilt state in %s -- leaders timed at the case's own time only" % root)
    return points[:_proxy.PROXY_TIME_POINTS]
