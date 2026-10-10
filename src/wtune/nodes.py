#!/usr/bin/env python3
"""Pure node-count selection from MEASURED seconds/step (stdlib only, no I/O).

wtune/tune/'s old `choose_nodes` picks N from an uncalibrated efficiency proxy; measured scc
scaling disagrees (large cases keep 82-87% efficiency at 8 nodes, small ones drop to 24-46%, some
are slower at 8 than at 4). This module turns per-N measurements {nodes: [s_per_step, ...]} into
a table and a choice under one of three objectives: time, cost, knee.
"""
import statistics

NOISE_FRAC = 0.03                 # run-to-run reproducibility floor used for ties / early stop
DEFAULT_MIN_EFFICIENCY = 0.7      # knee objective: parallel efficiency threshold
OBJECTIVES = ("time", "cost", "knee")


def candidate_node_counts(legal_nodes, max_nodes):
    """Sorted unique ints n with 1 <= n <= max_nodes."""
    return sorted({int(n) for n in legal_nodes if 1 <= int(n) <= int(max_nodes)})


def sweep_waves(nodes_list, max_nodes):
    """Greedy first-fit packing, ascending N, into waves whose node sums are <= max_nodes (jobs of
    one wave run concurrently). An N > max_nodes still gets a wave of its own."""
    waves = []
    for n in sorted(int(x) for x in nodes_list):
        for w in waves:
            if sum(w) + n <= max_nodes:
                w.append(n)
                break
        else:
            waves.append([n])
    return waves


def summarize(measurements):
    """{nodes: [s_per_step, ...]} -> rows sorted by nodes (empty/None lists skipped):
    nodes, median, n, spread ((max-min)/median, 0 if n<2), speedup (median at the smallest
    measured N / this median), efficiency (speedup * N_min / N)."""
    pts = sorted((int(n), list(v)) for n, v in measurements.items() if v)
    if not pts:
        return []
    n0 = pts[0][0]
    m0 = statistics.median(pts[0][1])
    rows = []
    for n, vals in pts:
        med = statistics.median(vals)
        spread = (max(vals) - min(vals)) / med if len(vals) >= 2 and med else 0.0
        speedup = m0 / med if med else 0.0
        rows.append(dict(nodes=n, median=med, n=len(vals), spread=spread, speedup=speedup,
                         efficiency=speedup * n0 / n))
    return rows


def stop_early(rows, noise=NOISE_FRAC):
    """True when the largest N measured so far did not beat the previous one by more than the
    noise floor: more nodes stopped helping, so larger N need not be measured."""
    return len(rows) >= 2 and rows[-1]["median"] > rows[-2]["median"] * (1.0 - noise)


def table(rows, *, total_steps=None, overhead_s=0.0):
    """rows plus wall_s = median*total_steps + overhead_s and node_hours = N*wall_s/3600
    (None when total_steps is None)."""
    out = []
    for r in rows:
        r = dict(r)
        if total_steps is None:
            r["wall_s"] = r["node_hours"] = None
        else:
            r["wall_s"] = r["median"] * total_steps + overhead_s
            r["node_hours"] = r["nodes"] * r["wall_s"] / 3600.0
        out.append(r)
    return out


def choose(rows, objective, *, noise=NOISE_FRAC, deadline_s=None, total_steps=None,
           overhead_s=0.0, min_efficiency=DEFAULT_MIN_EFFICIENCY):
    """(nodes, note). time: smallest N within `noise` of the minimum median. cost: min node-hours
    among N meeting deadline_s (all if None; none feasible -> fastest, said in the note), needs
    total_steps. knee: largest N with efficiency >= min_efficiency, else the smallest N."""
    if objective not in OBJECTIVES:
        raise ValueError("unknown objective %r (choices: %s)" % (objective, ", ".join(OBJECTIVES)))
    if not rows:
        raise ValueError("no measurements to choose from")
    if objective == "time":
        best = min(r["median"] for r in rows)
        r = min((r for r in rows if r["median"] <= best * (1.0 + noise)), key=lambda x: x["nodes"])
        return r["nodes"], ("objective=time: N=%d median=%.4f s/step; min is %.4f, smallest N "
                            "within %.0f%% noise of it" % (r["nodes"], r["median"], best,
                                                           noise * 100))
    if objective == "cost":
        if total_steps is None:
            raise ValueError("objective=cost requires total_steps")
        t = table(rows, total_steps=total_steps, overhead_s=overhead_s)
        feas = [r for r in t if deadline_s is None or r["wall_s"] <= deadline_s]
        if not feas:
            r = min(t, key=lambda x: (x["wall_s"], x["nodes"]))
            return r["nodes"], ("objective=cost: NO N meets deadline %.0fs; fastest N=%d "
                                "wall=%.0fs node_hours=%.2f" % (deadline_s, r["nodes"],
                                                                r["wall_s"], r["node_hours"]))
        r = min(feas, key=lambda x: (x["node_hours"], x["wall_s"]))
        dl = "no deadline" if deadline_s is None else "deadline %.0fs" % deadline_s
        return r["nodes"], ("objective=cost: N=%d wall=%.0fs node_hours=%.2f (min among %d "
                            "feasible, %s)" % (r["nodes"], r["wall_s"], r["node_hours"],
                                               len(feas), dl))
    ok = [r for r in rows if r["efficiency"] >= min_efficiency]
    if ok:
        r = max(ok, key=lambda x: x["nodes"])
        return r["nodes"], ("objective=knee: N=%d efficiency=%.2f >= %.2f (largest such N)"
                            % (r["nodes"], r["efficiency"], min_efficiency))
    r = min(rows, key=lambda x: x["nodes"])
    return r["nodes"], ("objective=knee: no N has efficiency >= %.2f; smallest N=%d"
                        % (min_efficiency, r["nodes"]))


def finalists(rows, objective, k=2, within=0.10, **choose_kwargs):
    """[chosen N, ...]: the chosen N first, then up to k-1 other N whose median is within `within`
    (relative) of the chosen N's median, closest first."""
    n, _note = choose(rows, objective, **choose_kwargs)
    by_n = {r["nodes"]: r for r in rows}
    m = by_n[n]["median"]
    others = sorted((r for r in rows if r["nodes"] != n and abs(r["median"] - m) <= within * m),
                    key=lambda r: (abs(r["median"] - m), r["nodes"]))
    return [n] + [r["nodes"] for r in others[:max(0, k - 1)]]


# ---------------------------------------------------------------------------
# coarse-to-fine node search
# ---------------------------------------------------------------------------
def seed_nodes(legal, max_nodes):
    """Geometric seeds {1, 2, 4, 8, max_nodes} intersected with the legal node counts (the
    smallest legal N when none of them is legal)."""
    legal = sorted(set(int(n) for n in legal))
    seeds = sorted({1, 2, 4, 8, int(max_nodes)} & set(legal))
    return seeds or legal[:1]


def refine_targets(best, known, legal, attempted=()):
    """Next N to measure around `best`: the nearest legal N on each side inside the bracket
    (lo, hi) = the closest `known` (measured or seeded) N below/above `best`, unless already
    attempted. Walks outwards only because the best moves (re-evaluated every round)."""
    lo = max([n for n in known if n < best], default=0)
    hi = min([n for n in known if n > best], default=None)
    below = [n for n in legal if lo < n < best]
    above = [n for n in legal if n > best and (hi is None or n < hi)]
    out = []
    for n in (below[-1:] + above[:1]):
        if n not in attempted:
            out.append(n)
    return out


def bracket(best, known):
    """(lo, hi) around `best` from the known N (hi None when nothing larger is known)."""
    return (max([n for n in known if n < best], default=0),
            min([n for n in known if n > best], default=None))


# ---------------------------------------------------------------------------
# cores-per-node dimension
# ---------------------------------------------------------------------------
def default_core_options(cores_per_node):
    """{C, C-2 (OS reserve), round(C*8/9), round(C*2/3), C/2}, descending; 72 -> 72,70,64,48,36."""
    c = int(cores_per_node)
    vals = {c, c - 2, int(round(c * 8 / 9.0)), int(round(c * 2 / 3.0)), c // 2}
    return sorted((v for v in vals if 1 <= v <= c), reverse=True)


def choose_cores(rows, noise=NOISE_FRAC):
    """(cores, note) over rows with 'cores' and 'median' at ONE fixed node count: the fastest
    median; every c within `noise` of it ties, and the tie goes to the fuller c. (At fixed N whole
    exclusive nodes are billed, so node-hours are proportional to wall time: the cost objective
    picks the same c as the time objective; knee has no efficiency across c.)"""
    if not rows:
        raise ValueError("no measurements to choose from")
    best = min(r["median"] for r in rows)
    r = max((r for r in rows if r["median"] <= best * (1.0 + noise)), key=lambda x: x["cores"])
    return r["cores"], ("cores/node=%d median=%.4f s/step; fastest is %.4f, fullest c within "
                        "%.0f%% noise of it" % (r["cores"], r["median"], best, noise * 100))
