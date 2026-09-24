#!/usr/bin/env python3
"""Build an ASYMMETRIC rank -> (threads, cpus) map: hot ranks get two cores, the rest get one.

WHY THIS IS THE ONLY REMAINING LEVER

Six ways of rebalancing WRF at a restart boundary have been measured shut: changing the
decomposition (8x56 is optimal, alternatives cost up to +17.4%), &dm_task_split (domain 1 must
span every rank), over-decomposition and core lending (+7.4%), adding nodes (forbidden by the
4-node rule), and NIC-aware rank pinning (-0.56%). WRF's decomposition is a uniform rectangle,
so patch size cannot vary per rank either.

Giving a hot rank two CORES is therefore the only way left to make ranks compute at different
speeds without touching source.

WHY THE PRIOR HYBRID SWEEP DOES NOT ALREADY CLOSE THIS

The 2026-08-23 sweep measured UNIFORM hybrid and found pure MPI fastest:
    448x1  0.939 s/step     224x2  1.059 (+12.8%)     112x4  1.804 (+92%)
Uniform 224x2 halves the rank count and makes EVERY rank pay the OpenMP tax. Asymmetric keeps
~400 ranks and only ~48 of them ever enter a parallel region. The +12.8% does not transfer,
and the asymmetric tax has never been measured.

The bar is arithmetic: perfect balance is worth at most 8.82% of wall, so the asymmetric tax
must come in under that.

WHY THE ALLOCATION IS PER NODE

wtune_restart.allocate_teams already does a greedy largest-team-first assignment and respects
NUMA capacity, but it places ranks on nodes freely. The launcher does not: `mpirun -np 400`
block-distributes ranks, so rank -> node is fixed before this code runs. Each node therefore
has its own budget -- 112 cores for its 100 ranks, i.e. at most 12 upgrades -- and if the hot
ranks happen to cluster on one node, some of them simply cannot be upgraded. That is a real
constraint of the machine, not a modelling choice, so it is enforced here rather than hidden.

NUMA: f1 COMPUTE nodes have 2 domains of 56 cores. (The 8-domains-of-14 figure is the LOGIN
node, where Sub-NUMA Clustering is on.) A 2-core team is still kept inside one domain, but the
constraint is nearly free at this width -- and the correction matters for a different reason:
"crossing NUMA" CANNOT be why 4+ threads collapsed to +92%, since four threads inside one
56-core socket never cross a domain. The likelier cause is the measured 57% intra-rank static
tile thread imbalance, which is a different problem with a different fix.

usage: make_thread_map.py <bucket_load.csv> <ranks> <nodes> <out.csv>
"""
import sys

import numpy as np

CORES_PER_NODE = 112
# f1 COMPUTE nodes have 2 NUMA domains of 56 cores. The 8-domains-of-14 figure that is easy to
# reach for is the LOGIN node (ilgn01), where Sub-NUMA Clustering is on; compute nodes have it
# off. Getting this wrong does not corrupt the map -- a 2-core team almost never straddles a
# 56-core boundary -- but it does invalidate the reasoning: "crossing NUMA" cannot explain the
# hybrid collapse at 4+ threads, because 4 threads inside one 56-core socket never cross.
DOMAINS_PER_NODE = 2
CORES_PER_DOMAIN = CORES_PER_NODE // DOMAINS_PER_NODE      # 56


def rank_loads(path, nranks):
    """Total compute seconds per rank, from the PMPI bucket profile."""
    rows = np.genfromtxt(path, delimiter=',', names=True)
    tot = np.zeros(nranks)
    for r, c in zip(rows['rank'], rows['comp_s']):
        i = int(r)
        if i < nranks:
            tot[i] += c
    return tot


def build_map(loads, nranks, nnodes):
    """-> list of (rank, nthreads, [cpus]) with node-local cpu ids.

    Block distribution: ranks [k*per_node, (k+1)*per_node) live on node k. That is what
    `mpirun -np N` does across a homogeneous allocation, and the map is meaningless if it
    disagrees with the launcher.
    """
    per_node = nranks // nnodes
    if per_node * nnodes != nranks:
        raise ValueError('%d ranks does not divide evenly over %d nodes' % (nranks, nnodes))
    upgrades = CORES_PER_NODE - per_node          # each upgrade costs one extra core
    if upgrades < 0:
        raise ValueError('%d ranks/node exceeds %d cores/node' % (per_node, CORES_PER_NODE))
    out = []
    for node in range(nnodes):
        lo = node * per_node
        local = list(range(lo, lo + per_node))
        order = sorted(local, key=lambda r: -loads[r])
        hot = set(order[:upgrades])
        # Walk the node's cores in NUMA order, never letting a 2-core team straddle a domain.
        cpu = 0
        dom_end = CORES_PER_DOMAIN
        for r in local:
            want = 2 if r in hot else 1
            if cpu + want > dom_end:              # would cross a NUMA boundary -> next domain
                cpu = dom_end
                dom_end = min(dom_end + CORES_PER_DOMAIN, CORES_PER_NODE)
            if cpu + want > CORES_PER_NODE:       # node full; the tail keeps one core each
                want = 1
            out.append((r, want, list(range(cpu, cpu + want))))
            cpu += want
    return out


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    bucket, nranks, nnodes, dest = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    loads = rank_loads(bucket, nranks)
    if loads.sum() <= 0:
        print('FATAL: no load in %s' % bucket)
        return 3
    rows = build_map(loads, nranks, nnodes)
    with open(dest, 'w') as fh:
        fh.write('rank,nthreads,cpus\n')
        for r, n, cpus in rows:
            fh.write('%d,%d,%s\n' % (r, n, '-'.join(str(c) for c in (cpus[0], cpus[-1]))
                                     if len(cpus) > 1 else str(cpus[0])))
    two = sum(1 for _, n, _ in rows if n == 2)
    used = sum(n for _, n, _ in rows)
    hot_load = np.mean([loads[r] for r, n, _ in rows if n == 2])
    all_load = float(loads.mean())
    print('%s: %d ranks, %d upgraded to 2 threads, %d cores used (of %d)'
          % (dest, len(rows), two, used, CORES_PER_NODE * nnodes))
    print('  upgraded ranks average %.1f s vs %.1f s overall  (%.2fx the mean)'
          % (hot_load, all_load, hot_load / all_load if all_load else 0))
    # Report the MAX, not max/mean. Wall time is set by the slowest rank; max/mean can move the
    # wrong way purely because halving some ranks lowers the denominator, which says nothing
    # about wall.
    adj = loads.copy()
    upgraded = {r for r, n, _ in rows if n == 2}
    for r in upgraded:
        adj[r] /= 2.0
    argmax_before = int(np.argmax(loads[:nranks]))
    print('  max rank cost:  %.1f s -> %.1f s   (%+.2f%%, IDEAL 2x speedup)'
          % (loads[:nranks].max(), adj[:nranks].max(),
             100.0 * (adj[:nranks].max() / loads[:nranks].max() - 1.0)))
    print('  the slowest rank (%d) %s upgraded'
          % (argmax_before, 'WAS' if argmax_before in upgraded else 'was NOT'))
    # The per-node budget is the binding constraint, so show how the hot set is distributed.
    per_node = nranks // nnodes
    topk = sorted(range(nranks), key=lambda r: -loads[r])[:len(upgraded)]
    spread = [sum(1 for r in topk if r // per_node == k) for k in range(nnodes)]
    print('  globally hottest %d ranks sit on nodes: %s  (each node can upgrade %d)'
          % (len(upgraded), spread, CORES_PER_NODE - per_node))
    got = len(set(topk) & upgraded)
    print('  of those, %d/%d actually got a second core -- the rest were priced out by their '
          "node's budget" % (got, len(topk)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
