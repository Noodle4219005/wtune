#!/usr/bin/env python3
"""Tile-level microphysics cost map and its predictability (from WTIMER_TILE_PROBE output).

Input : RUNDIR/tiles.<rank>.csv   columns step,tile,its,ite,jts,jte,ticks  (one driver call per tile, rdtsc ticks)
Output: printed report + two CSVs next to OUTPREFIX:
  OUTPREFIX.tilemap.csv    rank,tile,its,ite,jts,jte,ncol,median_ms,mean_ms,cv_pct,share_of_rank
  OUTPREFIX.metrics.csv    one row per (predictor, horizon) with the tile-level metrics below

Questions answered (causal where it predicts: only steps <= t-h are used)
  1 geometry fixed?          distinct tile bounds per (rank,tile) across steps
  2 per-tile persistence     P0 / EWMA(0.7) median and p90 abs % error, horizons 1,2,4
  3 expensive-tile identity  per rank: top-1 and top-3 tile set at step t vs predicted from <= t-h
  4 concentration            share of a rank's micro cost in its top-3 of 13 tiles; hot vs cold ranks
  5 hot-rank pattern         hot/cold per-tile cost ratio distribution: all tiles heavier, or a few very heavy

usage: replay_tiles.py RUNDIR OUTPREFIX [TSC_HZ=2.0e9] [WARM=5]
"""
import collections
import csv
import glob
import re
import statistics
import sys

rd, prefix = sys.argv[1], sys.argv[2]
TSC = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0e9
WARM = int(sys.argv[4]) if len(sys.argv) > 4 else 5

cost = collections.defaultdict(dict)      # (rank, tile) -> {step: ms}
geom = collections.defaultdict(set)       # (rank, tile) -> {(its,ite,jts,jte)}
for f in glob.glob(rd + "/tiles.*.csv"):
    r = int(re.search(r"tiles\.(\d+)\.csv$", f).group(1))
    for row in csv.DictReader(open(f)):
        k = (r, int(row["tile"]))
        cost[k][int(row["step"])] = int(row["ticks"]) / TSC * 1e3
        geom[k].add((int(row["its"]), int(row["ite"]), int(row["jts"]), int(row["jte"])))
ranks = sorted({r for r, _ in cost})
tiles_of = {r: sorted(t for rr, t in cost if rr == r) for r in ranks}
steps = sorted(set.intersection(*[set(v) for v in cost.values()]))
steps = [s for s in steps if s >= WARM]
print("ranks=%d tiles/rank=%s steps=%d..%d (%d)" % (len(ranks), sorted({len(v) for v in tiles_of.values()}),
                                                   steps[0], steps[-1], len(steps)))

# 1 geometry
moving = sum(1 for g in geom.values() if len(g) > 1)
print("1 geometry: (rank,tile) with more than one bound set across steps = %d of %d" % (moving, len(geom)))


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))] if s else float("nan")


# tile map
rank_total = {r: statistics.median([sum(cost[(r, t)][s] for t in tiles_of[r]) for s in steps]) for r in ranks}
hot_cut, cold_cut = pct(list(rank_total.values()), 0.8), pct(list(rank_total.values()), 0.2)
role = {r: "hot" if rank_total[r] >= hot_cut else "cold" if rank_total[r] <= cold_cut else "mid" for r in ranks}
with open(prefix + ".tilemap.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["rank", "role", "tile", "its", "ite", "jts", "jte", "ncol", "median_ms", "mean_ms", "cv_pct", "share_of_rank"])
    for r in ranks:
        for t in tiles_of[r]:
            v = [cost[(r, t)][s] for s in steps]
            its, ite, jts, jte = sorted(geom[(r, t)])[0]
            m = statistics.mean(v)
            w.writerow([r, role[r], t, its, ite, jts, jte, (ite - its + 1) * (jte - jts + 1),
                        round(statistics.median(v), 4), round(m, 4),
                        round(statistics.pstdev(v) / m * 100, 2) if m else 0,
                        round(statistics.median(v) / rank_total[r], 4) if rank_total[r] else 0])

# 2/3 predictability
metrics = []
for name in ("P0", "EWMA0.7"):
    for h in (1, 2, 4):
        ape, top1, top3 = [], [], []
        for r in ranks:
            est = {}
            for t in tiles_of[r]:
                seq = cost[(r, t)]
                e, series = None, {}
                for s in sorted(seq):
                    e = seq[s] if (e is None or name == "P0") else 0.7 * seq[s] + 0.3 * e
                    series[s] = e
                est[t] = series
            for s in steps:
                if s - h < 1 or (s - h) not in est[tiles_of[r][0]]:
                    continue
                act = {t: cost[(r, t)][s] for t in tiles_of[r]}
                pre = {t: est[t][s - h] for t in tiles_of[r]}
                for t in tiles_of[r]:
                    if act[t] > 0:
                        ape.append(abs(pre[t] - act[t]) / act[t] * 100)
                oa = sorted(act, key=lambda t: -act[t])
                op = sorted(pre, key=lambda t: -pre[t])
                top1.append(1.0 if oa[0] == op[0] else 0.0)
                top3.append(len(set(oa[:3]) & set(op[:3])) / 3)
        metrics.append(collections.OrderedDict([("predictor", name), ("horizon", h),
                                                ("tile_medape_pct", round(pct(ape, 0.5), 2)),
                                                ("tile_p90ape_pct", round(pct(ape, 0.9), 2)),
                                                ("top1_tile_retention", round(statistics.mean(top1), 3)),
                                                ("top3_tile_retention", round(statistics.mean(top3), 3))]))
with open(prefix + ".metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(metrics[0]))
    w.writeheader()
    w.writerows(metrics)
print("2/3 predictability:")
for m in metrics:
    print("   %-8s h=%d  tile median|%%err| %6.2f  p90 %6.2f  top1-tile retention %.3f  top3 %.3f" % tuple(m.values()))

# 4 concentration
def top3_share(r):
    med = sorted((statistics.median([cost[(r, t)][s] for s in steps]) for t in tiles_of[r]), reverse=True)
    return sum(med[:3]) / sum(med)

for grp in ("hot", "mid", "cold"):
    v = [top3_share(r) for r in ranks if role[r] == grp]
    if v:
        print("4 concentration %-4s ranks=%3d  top-3 of %d tiles hold median %.1f%% (p10 %.1f%%, p90 %.1f%%) of rank micro; uniform = %.1f%%"
              % (grp, len(v), len(tiles_of[ranks[0]]), 100 * statistics.median(v), 100 * pct(v, 0.1), 100 * pct(v, 0.9),
                 300.0 / len(tiles_of[ranks[0]])))

# 5 hot-rank pattern: per-tile median cost normalised by tile columns, hot vs cold (same tile index)
cold_ranks = [r for r in ranks if role[r] == "cold"]
hot_ranks = [r for r in ranks if role[r] == "hot"]
if cold_ranks and hot_ranks:
    cold_ref = {t: statistics.median([statistics.median([cost[(r, t)][s] for s in steps]) for r in cold_ranks]) for t in tiles_of[ranks[0]]}
    ratios = []
    for r in hot_ranks:
        rr = sorted((statistics.median([cost[(r, t)][s] for s in steps]) / cold_ref[t] for t in tiles_of[r]), reverse=True)
        ratios.append(rr)
    maxr = [x[0] for x in ratios]
    minr = [x[-1] for x in ratios]
    medr = [statistics.median(x) for x in ratios]
    print("5 hot-rank per-tile cost / cold median (same tile index): max %.2fx, median %.2fx, min %.2fx (medians over %d hot ranks)"
          % (statistics.median(maxr), statistics.median(medr), statistics.median(minr), len(hot_ranks)))
    print("  -> 'few very heavy tiles' if max >> median ~ 1; 'all tiles heavier' if min is already well above 1")
