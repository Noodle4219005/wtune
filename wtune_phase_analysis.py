#!/usr/bin/env python3
"""Settle Q2 — is the settlement tax POOLABLE, measured ALIGNMENT-FREE.

Why alignment-free: a halo exchange is Irecv*N / Isend*N / Wait*N, and N is the rank's
neighbour count (interior 4, edge 3, corner 2). So the MPI-call index DRIFTS between ranks
and "segment k = same phase on every rank" is FALSE (measured cross-rank correlation of the
per-segment series was only +0.22). Indexing by call number produces garbage.

Instead use the one call every rank makes the same number of times -- MPI_Allreduce (kind=3,
WRF's per-step CFL check) -- as a STEP delimiter, and work with per-step totals:

    C[r,s] = compute of rank r in step s        W[r,s] = wait of rank r in step s
    ranks are synchronised at the step boundary, so  C[r,s] + W[r,s] = wall(s)  for every r

Then, with no phase alignment needed anywhere:

    wall(s)          = the measured step wall
    max_r C[r,s]     = what the busiest rank must compute no matter how well slack is pooled
    mean_r C[r,s]    = the perfectly-balanced floor

    POOLING prize = (wall - max_r C)/wall   <- wall that even the BUSIEST rank spends waiting.
                                               Non-zero only if the straggler ROTATES between
                                               phases; that is exactly the "19pp" item.
    RESIDUAL      = (max_r C - mean_r C)/wall <- needs real work movement, not deferred settlement
"""
import csv, sys, collections
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/u5727520/wrf-results/alaska_winter/phase_seg.csv"
ALLREDUCE = 3


def load_per_step(path):
    """-> C[r,s], W[r,s] over steps complete on every rank."""
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0]))
    step_of = collections.defaultdict(int)
    with open(path) as f:
        rd = csv.DictReader(f)
        if "kind" not in rd.fieldnames:
            sys.exit("ERROR: phase_seg.csv has no 'kind' column - rerun with the tagging shim.")
        for row in rd:
            r = int(row["rank"]); k = int(row["kind"])
            s = step_of[r]
            cell = per[r][s]
            cell[0] += float(row["comp_s"]); cell[1] += float(row["wait_s"])
            if k == ALLREDUCE:
                step_of[r] = s + 1          # Allreduce closes the step
    ranks = sorted(per)
    common = set.intersection(*(set(per[r]) for r in ranks))
    common = sorted(s for s in common if 0 < s < max(common))   # drop partial first/last
    C = np.array([[per[r][s][0] for s in common] for r in ranks])
    W = np.array([[per[r][s][1] for s in common] for r in ranks])
    return np.array(ranks), np.array(common), C, W


def main():
    ranks, steps, C, W = load_per_step(PATH)
    R, S = C.shape
    print("=" * 76)
    print(f"per-step matrix: {R} ranks x {S} steps   (Allreduce-delimited, alignment-free)")
    print("=" * 76)

    total = C + W
    wall = np.median(total, axis=0)
    spread = (total.max(axis=0) - total.min(axis=0)) / np.maximum(wall, 1e-12)
    print(f"  sanity: C+W should be equal on every rank (they sync each step)")
    print(f"          median across-rank spread of (C+W) = {np.median(spread)*100:.1f}% of wall")
    if np.median(spread) > 0.25:
        print("          -> LARGE: step delimiting may be off; treat numbers with caution.")
    else:
        print("          -> small: step delimiting is sound.")
    print(f"  step wall: median {np.median(wall):.4f} s   (rsl 'Timing for main' comparison below)")

    maxC = C.max(axis=0); meanC = C.mean(axis=0)
    pool = (wall - maxC) / wall
    resid = (maxC - meanC) / wall
    print()
    print("=" * 76)
    print("DECISIVE ARITHMETIC (per step, then aggregated)")
    print("=" * 76)
    print(f"  wall            median {np.median(wall):8.4f} s")
    print(f"  max_r  C[r,s]   median {np.median(maxC):8.4f} s")
    print(f"  mean_r C[r,s]   median {np.median(meanC):8.4f} s")
    print()
    print(f"  >> POOLING prize (wall - maxC)/wall = {np.median(pool)*100:5.1f}%   "
          f"[p25 {np.percentile(pool,25)*100:.1f}%  p75 {np.percentile(pool,75)*100:.1f}%]")
    print(f"  >> RESIDUAL     (maxC - meanC)/wall = {np.median(resid)*100:5.1f}%   "
          f"[needs real work movement]")
    print(f"  >> total imbalance                  = {np.median(pool+resid)*100:5.1f}%")

    print()
    print("=" * 76)
    print("MECHANISM - does the busiest rank ALSO wait? (that is what makes slack poolable)")
    print("=" * 76)
    bi = C.argmax(axis=0)
    busiest_wait = W[bi, np.arange(S)]
    busiest_frac = busiest_wait / np.maximum(wall, 1e-12)
    print(f"  busiest rank's own wait: median {np.median(busiest_frac)*100:5.1f}% of wall")
    if np.median(busiest_frac) > 0.05:
        print("  -> the rank with the most total compute STILL waits a large share of the step:")
        print("     it is not the pacer in every phase => the straggler rotates => POOLING IS REAL.")
    else:
        print("  -> the busiest rank barely waits: it paces essentially every phase")
        print("     => nothing to pool; the prize is work movement only.")
    cnt = collections.Counter(bi.tolist())
    print(f"  distinct 'busiest' ranks across {S} steps: {len(cnt)}"
          f"   top: {[(int(ranks[i]), c) for i, c in cnt.most_common(4)]}")

    print()
    print("=" * 76)
    print("ANTICORRELATION - do the lightest ranks own work somewhere?")
    print("=" * 76)
    tot = C.sum(axis=1)
    order = np.argsort(tot)
    rel = C / np.maximum(C.max(axis=0, keepdims=True), 1e-12)
    print(f"  {'rank':>6} {'totC_s':>9} {'rel_med':>9} {'rel_max':>9}")
    for i in list(order[:4]) + list(order[-3:]):
        print(f"  {ranks[i]:>6} {tot[i]:>9.3f} {np.median(rel[i]):>9.3f} {rel[i].max():>9.3f}")
    print(f"  lightest rank总 compute {tot[order[0]]:.3f}s vs busiest {tot[order[-1]]:.3f}s "
          f"({tot[order[-1]]/max(tot[order[0]],1e-9):.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
