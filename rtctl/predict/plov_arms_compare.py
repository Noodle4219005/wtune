#!/usr/bin/env python3
"""Cross-arm comparison for one plov_predict.sbatch job (R0..R3 with palindromic repeats).

usage: plov_arms_compare.py OUTDIR JOBID TSC_HZ CSV_OUT
Reads OUTDIR/pp_<arm>_<job>/{steps.txt,plov.<rank>.csv}. Node = rank % 4; warm steps >= 3.
Writes CSV_OUT (one row per arm x step x node) and prints per-arm and per-mode tables.
Metric definitions (per node, per step):
  max_ms/spread_ms  exit_tsc - min node entry_tsc, max and max-min over the node's ranks
  owner_fin_ms      max completion over role O ranks; donor_fin_ms over role D ranks
  wait/idle/ovh     node sums of wait_ms, idle_ms, pack_ms+commit_ms
  late_steal_ms     node sum over D ranks of (exit_tsc - own_done_tsc): time spent after own work
  work_in_ms        node sum of stolen work computed by donors
  early_frac        1 - late_steal/work_in over D ranks with work_in>0 (share of stolen work done before own work)
"""
import collections, csv, glob, random, re, statistics as st, sys

out, job, HZ, csv_out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
ARMS = ["R0", "R1", "R2", "R3", "R3b", "R2b", "R1b", "R0b"]
ms = lambda t: t / HZ * 1e3
WARM = 3
rows_out, per_arm, steptimes = [], {}, {}
for a in ARMS:
    d = "%s/pp_%s_%s" % (out, a, job)
    steptimes[a] = [float(x) for x in open(d + "/steps.txt") if x.strip()][1:]
    by = collections.defaultdict(dict)
    for f in glob.glob(d + "/plov.*.csv"):
        r = int(re.search(r"plov\.(\d+)\.csv$", f).group(1))
        for x in csv.DictReader(open(f)):
            by[int(x["step"])][r] = x
    m = collections.defaultdict(list)
    for s in sorted(by):
        if s < WARM:
            continue
        for n in range(4):
            rs = [r for r in by[s] if r % 4 == n]
            if not rs:
                continue
            e0 = min(int(by[s][r]["entry_tsc"]) for r in rs)
            comp = {r: ms(int(by[s][r]["exit_tsc"]) - e0) for r in rs}
            O = [r for r in rs if by[s][r]["role"] == "O"]
            D = [r for r in rs if by[s][r]["role"] == "D"]
            f = lambda k: sum(float(by[s][r][k]) for r in rs)
            late = sum(ms(int(by[s][r]["exit_tsc"]) - int(by[s][r]["own_done_tsc"])) for r in D)
            win = sum(float(by[s][r]["work_in_ms"]) for r in D)
            rec = dict(arm=a, step=s, node=n, max_ms=max(comp.values()),
                       spread_ms=max(comp.values()) - min(comp.values()),
                       owner_fin_ms=max((comp[r] for r in O), default=0.0),
                       donor_fin_ms=max((comp[r] for r in D), default=0.0),
                       n_owner=len(O), n_donor=len(D),
                       owner_wait_ms=f("wait_ms"), helper_idle_ms=f("idle_ms"),
                       overhead_ms=f("pack_ms") + f("commit_ms"),
                       chunks_out=int(f("chunks_out")), work_in_ms=win,
                       assigned_in_ms=f("assigned_in_ms"), late_steal_ms=late,
                       early_frac=(1 - min(1.0, late / win)) if win > 0 else "")
            rows_out.append(rec)
            for k, v in rec.items():
                if k not in ("arm", "step", "node") and v != "":
                    m[k].append(v)
    per_arm[a] = m

keys = ["max_ms", "spread_ms", "owner_fin_ms", "donor_fin_ms", "owner_wait_ms", "helper_idle_ms",
        "overhead_ms", "chunks_out", "work_in_ms", "assigned_in_ms", "late_steal_ms", "early_frac"]
with open(csv_out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["arm", "step", "node", "n_owner", "n_donor"] + keys)
    w.writeheader()
    for r in rows_out:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

print("per-arm (node-step medians; s/step mean/median of steps 2..40)")
print("arm   s_mean   s_med  " + " ".join("%9s" % k[:9] for k in keys))
for a in ARMS:
    m = per_arm[a]
    print("%-4s %.5f %.5f " % (a, st.mean(steptimes[a]), st.median(steptimes[a])) +
          " ".join("%9.3f" % st.median(m[k]) if m[k] else "%9s" % "-" for k in keys))

print("\nper-mode palindromic pair (average of rep a and rep b)")
modes = {"R0": ("R0", "R0b"), "R1": ("R1", "R1b"), "R2": ("R2", "R2b"), "R3": ("R3", "R3b")}
for md, (x, y) in modes.items():
    sm = (st.mean(steptimes[x]) + st.mean(steptimes[y])) / 2
    sd = (st.median(steptimes[x]) + st.median(steptimes[y])) / 2
    sp = (st.median(per_arm[x]["spread_ms"]) + st.median(per_arm[y]["spread_ms"])) / 2
    mx = (st.median(per_arm[x]["max_ms"]) + st.median(per_arm[y]["max_ms"])) / 2
    wt = (st.mean(per_arm[x]["owner_wait_ms"]) + st.mean(per_arm[y]["owner_wait_ms"])) / 2
    print("%s s_mean=%.5f s_med=%.5f spread_med=%.2f max_med=%.2f owner_wait_mean=%.2f rep_s_med=[%.5f %.5f]"
          % (md, sm, sd, sp, mx, wt, st.median(steptimes[x]), st.median(steptimes[y])))

random.seed(20260914)
def boot(a1, a2, b1, b2, n=4000):
    """bootstrap over steps within each run, pair-averaged: mode B minus mode A, relative median s/step"""
    res = []
    for _ in range(n):
        g = lambda v: st.median(random.choices(v, k=len(v)))
        A = (g(steptimes[a1]) + g(steptimes[a2])) / 2
        B = (g(steptimes[b1]) + g(steptimes[b2])) / 2
        res.append(B / A - 1)
    res.sort()
    return res[n // 40], res[n // 2], res[n - n // 40]
print("\nrelative median s/step vs R0 (pair-averaged, step bootstrap 95% CI; ignores between-run noise)")
for md in ("R1", "R2", "R3"):
    lo, mid, hi = boot("R0", "R0b", *modes[md])
    print("%s vs R0: %+.2f%% [%+.2f, %+.2f]" % (md, mid * 100, lo * 100, hi * 100))
print("between-run rep difference (b/a-1, median s/step): " +
      " ".join("%s=%+.2f%%" % (md, (st.median(steptimes[y]) / st.median(steptimes[x]) - 1) * 100)
               for md, (x, y) in modes.items()))
