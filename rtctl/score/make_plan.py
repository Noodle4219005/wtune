#!/usr/bin/env python3
"""Offline elastic-core plan for karina 224 ranks x 4 nodes, cyclic placement.

Inputs : per-rank micro profile CSV (rank + score column), cpumap from the same srun geometry
         ("rank node first-last" per line, as written by s1_readiness.sbatch).
Outputs: <out>.plan      "hot_rank donor_rank" per slot, read by elasticd --plan
         <out>.multiprog Slurm --multi-prog config: hot ranks get OMP_NUM_THREADS=HOT_OMP,
                         every other rank inherits the job's OMP_NUM_THREADS
Rule   : per node-socket domain, hot = ranks with score >= abs threshold, hottest first, capped at
         floor(n / (1 + dph)); donors = coldest non-hot ranks in the same domain, the coldest
         donors going to the hottest ranks; each donor lends to exactly one hot rank.

usage: make_plan.py PROFILE COLUMN CPUMAP THRESHOLD DPH HOT_OMP OUTPREFIX [DOMAIN_WIDTH]
       DOMAIN_WIDTH = cores per NUMA node (ct448/ct112 icpnq nodes: 14, SNC 8x14; default 56 = socket)
"""
import collections
import csv
import sys

prof_path, col, cpumap, thr, dph, hot_omp, out = sys.argv[1:8]
width = int(sys.argv[8]) if len(sys.argv) > 8 else 56
thr, dph, hot_omp = float(thr), int(dph), int(hot_omp)
score = {int(r["rank"]): float(r[col]) for r in csv.DictReader(open(prof_path))}
dom = collections.defaultdict(list)
for line in open(cpumap):
    r, node, cpus = line.split()
    dom[(node, int(cpus.split("-")[0]) // width)].append(int(r))

pairs, hot_all, capped = [], [], 0
for key in sorted(dom):
    ranks = sorted(dom[key], key=lambda r: -score[r])
    cap = len(ranks) // (1 + dph)
    want = [r for r in ranks if score[r] >= thr]
    hot = want[:cap]
    capped += len(want) - len(hot)
    donors = [r for r in reversed(ranks) if r not in hot][: len(hot) * dph]
    for k in range(dph):
        for i, h in enumerate(hot):
            pairs.append((h, donors[k * len(hot) + i]))
    hot_all += hot

with open(out + ".plan", "w") as f:
    f.writelines("%d %d\n" % p for p in pairs)
with open(out + ".multiprog", "w") as f:
    if hot_all:
        f.write("%s env OMP_NUM_THREADS=%d ./wrf.exe\n" % (",".join(map(str, sorted(hot_all))), hot_omp))
    f.write("* ./wrf.exe\n")
unserved = max((score[r] for rs in dom.values() for r in rs if r not in hot_all), default=0)
print("domains=%d width=%d " % (len(dom), width), end="")
print("plan: hot=%d donors=%d slots=%d capped_out=%d threshold=%.2f dph=%d hottest_unserved=%.2f hottest=%.2f"
      % (len(hot_all), len({d for _, d in pairs}), len(pairs), capped, thr, dph, unserved, max(score.values())))
