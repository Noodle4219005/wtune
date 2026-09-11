#!/usr/bin/env python3
"""Is a pairing chosen from the PREVIOUS segment better than one chosen at t=0?

THE HALF OF THE PREMISE THAT WAS NEVER MEASURED

"Rebalance when the hot-spot moves substantially" has two halves. The first is established:
the per-rank pattern does move -- hot-spot identity decays 93.7% -> 54.3% over 16 model-hours
on the 48 h run. The second half has never been tested: once it has moved, does RE-CHOOSING
the pairing actually beat keeping the one computed at the start?

If a t=0 pairing is still nearly as good at hour 12, the dynamic part is worth nothing and
the old closure stands, whatever the pattern did in between. If re-choosing wins, that gap is
the value of a per-rank model, and it has a consumer again.

WHY THIS IS THE HONEST FORM OF THE QUESTION

The comparison is deliberately against the pairing from the PREVIOUS segment, not from the
target segment itself. A rebalancer standing at a restart boundary has measured the segment
it just finished; it cannot have measured the one it is about to run. The oracle arm is
carried alongside only to show how much of the achievable gain the realistic arm captures.

Billing follows pairing_sim: a core's cost is the SUM of its two halves (lending lets one
half's compute absorb the other's wait) and wall is the sum over windows of the per-core MAX.
"""
import os
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
from decay_gate import load_bucket_csv
from pairing_sim import bill, pair_extremes, pair_anticorrelated, simulate

RESULTS = '/home/u5727520/wrf-results'
NSEG = 4


def arms(comp, nseg=NSEG):
    """-> per-target-segment recovery of each pairing policy, in percent of the unpaired bill."""
    nb = comp.shape[1]
    edges = [int(round(i * nb / nseg)) for i in range(nseg + 1)]
    out = []
    for t in range(1, nseg):
        tgt = comp[:, edges[t]:edges[t + 1]]
        if tgt.shape[1] < 4 or tgt.sum() <= 0:
            continue
        seg0 = comp[:, edges[0]:edges[1]]
        prev = comp[:, edges[t - 1]:edges[t]]
        # sequential pairing = no reordering at all, the thing to beat
        seq = list(zip(range(0, comp.shape[0], 2), range(1, comp.shape[0], 2)))
        row = {'seg': t, 'base': bill(simulate(tgt, seq))}
        for lab, src in (('t0', seg0), ('prev', prev), ('oracle', tgt)):
            order = sorted(range(comp.shape[0]), key=lambda r: -src[r].sum())
            row['ext_' + lab] = bill(simulate(tgt, pair_extremes(order)))
            row['anti_' + lab] = bill(simulate(tgt, pair_anticorrelated(src)))
        out.append(row)
    return out


def main():
    cases = sys.argv[1:] or ['karina_48h', 'karina_cat4', 'deep_conv_ot', 'arctic_sbl',
                             'marine_cao', 'polar_low', 'extratrop_cyc', 'edouard_landfall']
    print('recovery vs sequential pairing, on the TARGET segment (higher = better)')
    print('%-18s %4s | %-22s | %-22s' % ('', 'seg', 'pair_extremes', 'pair_anticorrelated'))
    print('%-18s %4s | %7s %7s %7s | %7s %7s %7s'
          % ('case', '', 'from t0', 'from prev', 'oracle', 'from t0', 'from prev', 'oracle'))
    agg = {k: [] for k in ('ext_t0', 'ext_prev', 'ext_oracle',
                           'anti_t0', 'anti_prev', 'anti_oracle')}
    for c in cases:
        p = os.path.join(RESULTS, c, 'bucket_load.csv')
        if not os.path.isfile(p):
            continue
        try:
            comp, bsec = load_bucket_csv(p)
        except Exception as e:
            print('%-18s SKIP %s' % (c, e))
            continue
        used = comp.sum(axis=0) > 0
        if used.sum() < 8:
            continue
        last = int(np.where(used)[0].max())
        comp = comp[:, 1:last]                     # drop init bucket and the partial tail
        for row in arms(comp):
            vals = []
            for k in ('ext_t0', 'ext_prev', 'ext_oracle',
                      'anti_t0', 'anti_prev', 'anti_oracle'):
                rec = 100.0 * (row['base'] - row[k]) / row['base']
                vals.append(rec)
                agg[k].append(rec)
            print('%-18s %4d | %6.2f%% %6.2f%% %6.2f%% | %6.2f%% %6.2f%% %6.2f%%'
                  % (c, row['seg'], *vals))
    print()
    print('%-18s     | %6.2f%% %6.2f%% %6.2f%% | %6.2f%% %6.2f%% %6.2f%%'
          % ('MEAN', *[float(np.mean(agg[k])) if agg[k] else float('nan')
                       for k in ('ext_t0', 'ext_prev', 'ext_oracle',
                                 'anti_t0', 'anti_prev', 'anti_oracle')]))
    for fam in ('ext', 'anti'):
        t0, pv, orc = (np.array(agg[fam + '_' + s]) for s in ('t0', 'prev', 'oracle'))
        if not len(t0):
            continue
        print('\n%s: re-choosing from the previous segment is worth %+.2f pp over keeping t0'
              % (fam, float(pv.mean() - t0.mean())))
        print('   and captures %.0f%% of what the oracle pairing gets (%.2f%% vs %.2f%%)'
              % (100 * (pv.mean() - t0.mean()) / max(orc.mean() - t0.mean(), 1e-9),
                 pv.mean(), orc.mean()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
