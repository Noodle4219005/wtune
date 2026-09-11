#!/usr/bin/env python3
"""Zero-job gate for over-decomposition + core lending on HIGH-IMBALANCE cases.

WHY THIS IS BACK FROM THE DEAD

Over-decomposition with PMPI core lending was closed on arithmetic: an intrinsic floor of
+7.4% against a prize of 8.82%. Two things were wrong with that closure.

  1. The prize was CONUS's. On the same basis (billed 0.939, pooled 0.757, flat 0.664 ->
     M = 9.9%) CONUS is the FLATTEST case in this corpus. karina_cat4 measures 19.9%, exactly
     2x. Every closure argued from CONUS numbers alone carries that asterisk.
  2. The +57.2% that the closure remembers belongs to dev_Cpool_nopin_shim, an OS-float pool
     variant killed by CFS migration thrash. The LENDING shim's marginal effect in the same
     CSV is C_over4_shim 0.965 vs B_over4_noshim 1.003 = -3.8%. It fired and it recovered --
     on the flattest case in the corpus, while under-absorbing at 40-60% utilisation.

And the floor is not one number: +19.3% on icpnp331-334 (job 1043232) against +7.4% on
icpnq349/652. A 2.6x disagreement between two measurements of the same quantity, unexplained.

WHAT THIS SIMULATES

448 ranks are over-decomposed into 896 halves and paired two-per-core. A core's cost is the
SUM of its two halves' compute, because lending lets one half's compute absorb the other's
wait. Billing follows the measured synchronisation structure: wall is the SUM over windows of
the per-core MAX, not the max of any core's total.

The pairing is FIXED for the whole segment -- re-pairing per window would require migration,
which is the thing that has no cheap mechanism. Choosing it from the previous segment's
pattern is legitimate: the measured pattern decay constant is 2086 s for this case against
432 s segments.

Perfect pairing recovers exactly M by construction (every core lands on the mean), so the
question this answers is how much of M a REALISABLE fixed pairing captures, and whether what
is left after the floor is worth a job.

usage: pairing_sim.py <bucket_load.csv> [floor_pct ...]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decay_gate import load_bucket_csv


def bill(mat):
    """Wall under the measured structure: sum over windows of the per-worker max."""
    return float(mat.max(axis=0).sum())


def pair_extremes(order):
    """Largest with smallest. The classic makespan heuristic and a decent anti-correlation
    proxy when the dominant signal is the level rather than the shape."""
    return list(zip(order[:len(order) // 2], order[::-1][:len(order) // 2]))


def pair_anticorrelated(comp):
    """Greedy: repeatedly take the busiest unpaired rank and give it the partner whose time
    series correlates most negatively with it. This is the pairing the mechanism actually
    wants -- it flattens the SUM, which is what a core bills."""
    n = comp.shape[0]
    z = comp - comp.mean(axis=1, keepdims=True)
    sd = z.std(axis=1)
    sd[sd == 0] = 1e-12
    zn = z / sd[:, None]
    C = (zn @ zn.T) / comp.shape[1]
    free = set(range(n))
    order = sorted(range(n), key=lambda r: -comp[r].sum())
    pairs = []
    for r in order:
        if r not in free:
            continue
        free.discard(r)
        if not free:
            break
        cand = np.array(sorted(free))
        j = int(cand[np.argmin(C[r, cand])])
        free.discard(j)
        pairs.append((r, j))
    return pairs


def simulate(comp, pairs):
    """Each rank splits in half; a core holds one half of each of its two ranks."""
    half = comp / 2.0
    core = np.stack([half[a] + half[b] for a, b in pairs])
    return core


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    comp, bsec = load_bucket_csv(sys.argv[1])
    floors = [float(x) for x in sys.argv[2:]] or [7.4, 19.3]
    used = comp.sum(axis=0) > 0
    last = int(np.where(used)[0].max())
    core0 = comp[:, 1:last]
    nr, nb = core0.shape
    base = bill(core0)
    tot = core0.sum(axis=1)
    wall = nb * bsec
    print('%s' % os.path.basename(os.path.dirname(sys.argv[1])))
    print('  %d ranks, %d windows of %.0f s, wall %.0f s' % (nr, nb, bsec, wall))
    print('  max/mean per-rank total = %.3f      pooled prize M = %.1f%%'
          % (tot.max() / tot.mean(), 100.0 * (tot.max() - tot.mean()) / wall))
    print('  billed (sum of per-window max)  = %.1f s' % base)
    print('  floor  (sum of per-window mean) = %.1f s   <- perfect balance' % core0.mean(axis=0).sum())
    print()
    order = sorted(range(nr), key=lambda r: -tot[r])
    for name, pairs in (('extremes  (largest<->smallest)', pair_extremes(order)),
                        ('anti-corr (greedy on rho)', pair_anticorrelated(core0))):
        paired = bill(simulate(core0, pairs))
        rec = 100.0 * (base - paired) / wall
        print('  %-32s billed %.1f s   recovers %+.2f%% of wall' % (name, paired, rec))
        for f in floors:
            net = rec - f
            print('      vs floor %4.1f%%  ->  net %+.2f%%   %s'
                  % (f, net, 'WORTH A JOB' if net > 3.0 else 'dead by this floor'))
    print()
    print('READING: the gate is net > ~3%% against the floor that actually applies. The floor is')
    print('  the unresolved number (7.4 vs 19.3), which is what the two-job step measures.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
