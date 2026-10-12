#!/usr/bin/env python3
"""Where do the hot ranks physically live, and should they be concentrated or spread?

The question as posed is "are the hot-spots pinned inside one NUMA domain, or at least one
node". The rank->hardware map makes that question answerable without running anything, and it
also shows that the answer may be the opposite of what is wanted.

THE GEOMETRY

WRF numbers ranks row-major in x (wtune_segment_train.py:114): i = rank % PX, j = rank // PX.
With PX=8, PY=56 and 112 ranks per node, node = rank // 112 covers j in [14n, 14n+14) across
ALL eight longitudes. f1 compute nodes have 2 NUMA domains of 56 cores
(knowledge/hpc/2026-09-03-f1-compute-nodes-have-2-numa-not-8.md), so a NUMA domain is a
7-row latitude band.

The dominant imbalance driver is the solar terminator, a NORTH-SOUTH line. Hot ranks
therefore share a few values of i and span every j -- which places them, by construction,
evenly across all four nodes.

WHICH RAISES THE REAL QUESTION

Concentrating hot ranks on one node would make that node the critical path while the other
three idle. For makespan the wanted property is the opposite: every node carrying the same
total. So this measures both, and reports which one the current mapping already has:

  * per-node and per-NUMA total load, and how far from equal it is
  * the same for per-rank, to separate imbalance that hardware placement could touch from
    imbalance that lives inside a rank and cannot be moved without changing cell ownership

If node totals are already flat, node-level re-placement has nothing to win, and the entire
recoverable imbalance is intra-node -- which no rankfile can reach, because a rankfile moves
ranks between cores and never changes which cells a rank owns
(knowledge/wrf/2026-08-29-wrf-bottleneck-decomposition.md:29).
"""
import os
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
from decay_gate import load_bucket_csv, windows_from_buckets, DEFAULT_WINDOW_S

RESULTS = '/home/u5727520/wrf-results'
PX, PY = 8, 56
RANKS_PER_NODE = 112
CORES_PER_NUMA = 56


def imbalance(v):
    """max/mean - 1, in percent: how much longer the slowest unit takes than the average."""
    v = np.asarray(v, float)
    m = v.mean()
    return 100.0 * (v.max() - m) / m if m > 0 else float('nan')


def report(case):
    p = os.path.join(RESULTS, case, 'bucket_load.csv')
    if not os.path.isfile(p):
        return None
    comp, bsec = load_bucket_csv(p)
    w, width = windows_from_buckets(comp, bsec, DEFAULT_WINDOW_S)
    if w.shape[1] < 3:
        return None
    nr = w.shape[0]
    if nr != PX * PY:
        return None
    node = np.arange(nr) // RANKS_PER_NODE
    numa = np.arange(nr) // CORES_PER_NUMA
    i = np.arange(nr) % PX
    tot = w.sum(axis=1)

    node_load = np.array([tot[node == n].sum() for n in range(node.max() + 1)])
    numa_load = np.array([tot[numa == n].sum() for n in range(numa.max() + 1)])
    lon_load = np.array([tot[i == k].sum() for k in range(PX)])

    # where do the hottest 10% of ranks sit? If the terminator story holds they are spread
    # evenly over nodes and concentrated in a few longitudes.
    topk = max(1, int(round(0.10 * nr)))
    hot = np.argsort(tot)[-topk:]
    hot_by_node = np.array([(node[hot] == n).sum() for n in range(node.max() + 1)])
    hot_by_lon = np.array([(i[hot] == k).sum() for k in range(PX)])
    return dict(case=case, nr=nr, nwin=w.shape[1],
                rank_imb=imbalance(tot), node_imb=imbalance(node_load),
                numa_imb=imbalance(numa_load), lon_imb=imbalance(lon_load),
                hot_by_node=hot_by_node, hot_by_lon=hot_by_lon, topk=topk)


def main():
    cases = sys.argv[1:] or ['karina_48h', 'karina_cat4', 'deep_conv_ot', 'arctic_sbl',
                             'marine_cao', 'polar_low', 'extratrop_cyc', 'edouard_landfall',
                             'atmos_river', 'lowell_cat4', 'dry_conv', 'marie_ts']
    rows = [r for r in (report(c) for c in cases) if r]
    if not rows:
        print('no usable cases')
        return 2
    print('imbalance = (max - mean)/mean of the TOTAL load of each unit, over the whole run')
    print('%-18s %6s | %8s %8s %8s %8s' % ('case', 'wins', 'per-rank', 'per-NUMA', 'per-node',
                                           'per-lon'))
    for r in rows:
        print('%-18s %6d | %7.1f%% %7.1f%% %7.1f%% %7.1f%%'
              % (r['case'], r['nwin'], r['rank_imb'], r['numa_imb'], r['node_imb'],
                 r['lon_imb']))
    print('%-18s %6s | %7.1f%% %7.1f%% %7.1f%% %7.1f%%'
          % ('MEAN', '', *[float(np.mean([r[k] for r in rows]))
                           for k in ('rank_imb', 'numa_imb', 'node_imb', 'lon_imb')]))
    print()
    print('where the hottest 10%% of ranks sit (even spread over 4 nodes would be 25%% each):')
    print('%-18s %-28s %s' % ('case', 'by node', 'by longitude column i=0..7'))
    for r in rows:
        hb = r['hot_by_node']
        print('%-18s %-28s %s'
              % (r['case'], ' '.join('%4.0f%%' % (100 * x / r['topk']) for x in hb),
                 ' '.join('%3.0f' % x for x in r['hot_by_lon'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
