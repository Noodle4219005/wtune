#!/usr/bin/env python3
"""Build a rank->host machinefile that spreads the measured hot ranks across nodes.

WHY NOT A BAND INTERLEAVE

The first attempt dealt 7-row latitude bands round-robin to the four nodes. It failed: karina's
hot region is only a few rows deep, narrower than the band, so band 0 stayed whole on node 0
and the hot-rank concentration moved only 97% -> 85.5% against a target of 25%. A partition
whose granularity is coarser than the structure it is trying to break up does not break it up.

WHY NOT A FULL ROUND-ROBIN EITHER

Dealing every rank round-robin by cost spreads the hot ranks perfectly, but it also puts every
x-neighbour on a different node. That trades contention for halo traffic -- the same trade that
measured NULL in 2026-09-07-socket-locality-is-worth-nothing. A result from that placement
would mix the two effects with opposite signs and could cancel to nothing while both are real.

SO: MINIMAL PERTURBATION

Start from the shipped block map and swap only what is needed. For each node holding more than
its share of hot ranks, swap one of its hot ranks with a cold rank on a node holding fewer.
Roughly 33 swaps out of 448 ranks for karina, so ~85% of the default map -- and therefore most
of the halo locality -- survives. Node sizes are preserved exactly because a swap moves two
ranks in opposite directions.
"""
import numpy as np

TOPFRAC = 0.10


def hot_ranks(comp, topfrac=TOPFRAC):
    """-> indices of the most expensive ranks, from a rank x bucket compute matrix.

    Bucket 0 holds initialisation and the final bucket is partial; both are dropped. The
    spin-up hour is 5-9% cheap and is not weather, so including bucket 0 tilts the ranking
    toward whatever was expensive during model start-up rather than during the segment.
    """
    used = np.where(comp.sum(axis=0) > 0)[0]
    if used.size < 3:
        return np.array([], dtype=int)
    last = int(used.max())
    tot = comp[:, 1:last].sum(axis=1)
    k = max(1, int(round(topfrac * comp.shape[0])))
    return np.argsort(tot)[-k:]


def spread_map(nranks, nnodes, hot, base=None):
    """-> node index per rank. Minimal swaps from `base` so every node holds an equal share
    of `hot`, with node sizes preserved exactly."""
    per = nranks // nnodes
    node = np.arange(nranks) // per if base is None else np.asarray(base).copy()
    hot = set(int(x) for x in hot)
    target = len(hot) / float(nnodes)
    for _ in range(nranks):                       # bounded; converges in ~len(hot) swaps
        cnt = [sum(1 for r in np.where(node == n)[0] if r in hot) for n in range(nnodes)]
        src = int(np.argmax(cnt))
        dst = int(np.argmin(cnt))
        if cnt[src] - cnt[dst] <= 1 or cnt[src] <= target:
            break
        # move the hot rank that is cheapest to move: the one nearest the destination block,
        # so the swap disturbs the least halo locality it can
        cand_hot = [r for r in np.where(node == src)[0] if r in hot]
        cand_cold = [r for r in np.where(node == dst)[0] if r not in hot]
        if not cand_hot or not cand_cold:
            break
        h = min(cand_hot, key=lambda r: abs(r - dst * per - per // 2))
        c = min(cand_cold, key=lambda r: abs(r - h))
        node[h], node[c] = dst, src
    return node


def write_machinefile(path, node_of_rank, hosts):
    with open(path, 'w') as fh:
        fh.write('\n'.join(hosts[int(n)] for n in node_of_rank) + '\n')
    return path


def summarise(node_of_rank, hot, nnodes):
    hot = set(int(x) for x in hot)
    cnt = [sum(1 for r in np.where(np.asarray(node_of_rank) == n)[0] if r in hot)
           for n in range(nnodes)]
    k = max(len(hot), 1)
    moved = int(np.sum(np.asarray(node_of_rank) != (np.arange(len(node_of_rank))
                                                    // (len(node_of_rank) // nnodes))))
    return {'hot_by_node_pct': [round(100.0 * c / k, 1) for c in cnt],
            'ranks_moved_from_default': moved,
            'moved_pct': round(100.0 * moved / len(node_of_rank), 1)}
