#!/usr/bin/env python3
"""Causal offline replay of cheap per-rank microphysics cost predictors.

Data: unstaged (PLOV-free) wtimer traces of karina 901x901x45, 224 ranks, mp_physics=8, numtiles=13
(st_A_* run dirs; one CSV per rank, one row per micro call == timestep). Micro cost of a rank is set
by its patch, not by placement, so block- and cyclic-placed runs are equally valid samples.

For every trace, rank r, horizon h and step t, a predictor may use ONLY that rank's (and, for node
statistics, its node-mates') values at steps <= t-h. P4 is the one exception and is labelled: it uses
the same step of a DIFFERENT run of the same case (needs an anchor run; not causal within a run).

Predictors
  P0      last value                  v[t-h]
  P1_a    EWMA, e = a*last + (1-a)*e_prev, a in {0.3, 0.5, 0.7}
  P2      cumulative mean of steps 1..t-h
  P2w10   trailing mean of 10 steps ending at t-h
  P3      phase-aware: separate EWMA(0.7) histories for radiation steps and normal steps
  P4_anchor  same step of another run of this case (upper-bound reference)

Metrics (steps t >= WARM, over all ranks)
  mae_ms, medape_pct, p90ape_pct        rank regression
  top{5,10,20}_retention                |topK(pred) & topK(actual)| / K per step, averaged (global ranking)
  hot_precision/recall                  per node (cyclic: node = rank % 4), hot = >= node 80th percentile
  cold_precision/recall                 cold = <= node 20th percentile
  false_donor_rate                      predicted-cold ranks whose actual value exceeds the actual node mean
  excess_medrel_pct, excess_p90rel_pct  on actual-hot ranks: |pred_excess - actual_excess| / actual_excess,
  excess_mae_ms                         excess = value - node mean (pred uses predicted node mean)

usage: replay_micro.py OUT_CSV [TRACE_DIR ...]   (default: all st_A_* dirs with >= 132 rows)
"""
import collections
import glob
import math
import os
import re
import sys

ROOT = "/home/u5727520/wrf-bench"
NODES = 4
WARM = 5
HORIZONS = (1, 2, 4, 8)
KS = (5, 10, 20)


def load(d):
    micro, rad = {}, {}
    for f in glob.glob(d + "/wtimer.*.csv"):
        r = int(re.search(r"wtimer\.(\d+)\.csv$", f).group(1))
        m, rd, staged = [], [], None
        for line in open(f):
            if line.startswith("# micro_calls="):
                staged = int(re.search(r"micro_staged=(\d+)", line).group(1))
            if not line[:1].isdigit():
                continue
            p = line.split(",")
            m.append(float(p[1]))
            rd.append(float(p[2]) if p[2] else 0.0)
        if staged:
            return None, None, "staged"
        micro[r], rad[r] = m, rd
    if len(micro) != 224:
        return None, None, "ranks=%d" % len(micro)
    T = min(len(v) for v in micro.values())
    micro = {r: v[:T] for r, v in micro.items()}
    rad_steps = {t for t in range(T) if max(rad[r][t] for r in rad) > 0.1}
    return micro, rad_steps, "ok"


def ewma(v, a):
    e, out = v[0], []
    for x in v:
        e = a * x + (1 - a) * e
        out.append(e)
    return out


def predictions(v, rad_steps, h, anchor):
    T = len(v)
    cm, s = [], 0.0
    for i, x in enumerate(v):
        s += x
        cm.append(s / (i + 1))
    ew = {a: ewma(v, a) for a in (0.3, 0.5, 0.7)}
    tw = [sum(v[max(0, i - 9):i + 1]) / (i + 1 - max(0, i - 9)) for i in range(T)]
    # phase-aware histories: EWMA per class, value at each step = latest estimate for that class
    ph = {True: [], False: []}
    est = {True: None, False: None}
    for i, x in enumerate(v):
        c = i in rad_steps
        est[c] = x if est[c] is None else 0.7 * x + 0.3 * est[c]
        ph[True].append(est[True])
        ph[False].append(est[False])
    out = collections.OrderedDict()
    idx = range(WARM, T)
    out["P0"] = {t: v[t - h] for t in idx if t - h >= 1}
    for a in (0.3, 0.5, 0.7):
        out["P1_%.1f" % a] = {t: ew[a][t - h] for t in idx if t - h >= 1}
    out["P2"] = {t: cm[t - h] for t in idx if t - h >= 1}
    out["P2w10"] = {t: tw[t - h] for t in idx if t - h >= 1}
    p3 = {}
    for t in idx:
        if t - h < 1:
            continue
        c = t in rad_steps
        val = ph[c][t - h]
        p3[t] = val if val is not None else ph[not c][t - h]
    out["P3"] = p3
    if anchor is not None:
        out["P4_anchor"] = {t: anchor[t] for t in idx if t < len(anchor)}
    return out


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def evaluate(micro, rad_steps, h, anchors):
    ranks = sorted(micro)
    preds = {r: predictions(micro[r], rad_steps, h, anchors.get(r) if anchors else None) for r in ranks}
    names = list(preds[ranks[0]])
    results = []
    for name in names:
        steps = sorted(set.intersection(*[set(preds[r][name]) for r in ranks]))
        ae, ape = [], []
        ret = {k: [] for k in KS}
        hp = hr = cp = cr = 0
        hot_pred_n = hot_act_n = cold_pred_n = cold_act_n = fd = 0
        ex_rel, ex_abs = [], []
        for t in steps:
            a = {r: micro[r][t] for r in ranks}
            p = {r: preds[r][name][t] for r in ranks}
            for r in ranks:
                ae.append(abs(p[r] - a[r]))
                ape.append(abs(p[r] - a[r]) / a[r] * 100 if a[r] > 0 else 0)
            oa = sorted(ranks, key=lambda r: -a[r])
            op = sorted(ranks, key=lambda r: -p[r])
            for k in KS:
                ret[k].append(len(set(oa[:k]) & set(op[:k])) / k)
            for n in range(NODES):
                nr = [r for r in ranks if r % NODES == n]
                av = sorted(a[r] for r in nr)
                pv = sorted(p[r] for r in nr)
                a80, a20 = pct(av, 0.8), pct(av, 0.2)
                p80, p20 = pct(pv, 0.8), pct(pv, 0.2)
                amean = sum(av) / len(av)
                pmean = sum(pv) / len(pv)
                ah = {r for r in nr if a[r] >= a80}
                ph_ = {r for r in nr if p[r] >= p80}
                ac = {r for r in nr if a[r] <= a20}
                pc = {r for r in nr if p[r] <= p20}
                hp += len(ah & ph_); hot_pred_n += len(ph_); hot_act_n += len(ah)
                cp += len(ac & pc); cold_pred_n += len(pc); cold_act_n += len(ac)
                fd += sum(1 for r in pc if a[r] > amean)
                for r in ah:
                    ae_ = a[r] - amean
                    pe_ = p[r] - pmean
                    if ae_ > 0:
                        ex_rel.append(abs(pe_ - ae_) / ae_ * 100)
                        ex_abs.append(abs(pe_ - ae_))
        results.append(collections.OrderedDict([
            ("predictor", name), ("horizon", h), ("n_steps", len(steps)),
            ("mae_ms", round(1e3 * sum(ae) / len(ae), 3)),
            ("medape_pct", round(pct(ape, 0.5), 2)), ("p90ape_pct", round(pct(ape, 0.9), 2)),
            ("top5_retention", round(sum(ret[5]) / len(ret[5]), 3)),
            ("top10_retention", round(sum(ret[10]) / len(ret[10]), 3)),
            ("top20_retention", round(sum(ret[20]) / len(ret[20]), 3)),
            ("hot_precision", round(hp / hot_pred_n, 3)), ("hot_recall", round(hp / hot_act_n, 3)),
            ("cold_precision", round(cp / cold_pred_n, 3)), ("cold_recall", round(cp / cold_act_n, 3)),
            ("false_donor_rate", round(fd / cold_pred_n, 4)),
            ("excess_medrel_pct", round(pct(ex_rel, 0.5), 2)), ("excess_p90rel_pct", round(pct(ex_rel, 0.9), 2)),
            ("excess_mae_ms", round(1e3 * sum(ex_abs) / max(1, len(ex_abs)), 3)),
        ]))
    return results


def main():
    out = sys.argv[1]
    dirs = sys.argv[2:] or sorted(d for d in glob.glob(ROOT + "/st_A_*") if os.path.isdir(d))
    traces = collections.OrderedDict()
    for d in dirs:
        micro, rad_steps, status = load(d)
        if status != "ok":
            print("skip %s (%s)" % (os.path.basename(d), status))
            continue
        T = len(next(iter(micro.values())))
        if T < 132:
            print("skip %s (T=%d)" % (os.path.basename(d), T))
            continue
        traces[os.path.basename(d)] = (micro, rad_steps)
        print("trace %s T=%d rad_steps=%s" % (os.path.basename(d), T, sorted(rad_steps)[:8]))
    # anchor for P4: another trace of the same length, earliest one
    by_len = collections.defaultdict(list)
    for name, (m, _) in traces.items():
        by_len[len(next(iter(m.values())))].append(name)
    rows = []
    for name, (micro, rad_steps) in traces.items():
        T = len(next(iter(micro.values())))
        others = [o for o in by_len[T] if o != name]
        anchors = traces[others[0]][0] if others else None
        # micro on radiation steps vs neighbours (does micro itself follow the radiation cadence?)
        rs = [t for t in rad_steps if 1 <= t < T - 1]
        if rs:
            ratio = sum(sum(micro[r][t] for r in micro) / sum((micro[r][t - 1] + micro[r][t + 1]) / 2 for r in micro) for t in rs) / len(rs)
            print("  %s micro(rad step)/micro(neighbours) = %.3f" % (name, ratio))
        for h in HORIZONS:
            for res in evaluate(micro, rad_steps, h, anchors):
                row = collections.OrderedDict([("trace", name), ("T", T), ("anchor", others[0] if others else "")])
                row.update(res)
                rows.append(row)
    import csv
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s rows=%d" % (out, len(rows)))
    # compact summary: mean over traces, per predictor and horizon
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["predictor"], r["horizon"])].append(r)
    keys = ["mae_ms", "medape_pct", "p90ape_pct", "top5_retention", "top10_retention", "top20_retention",
            "hot_precision", "hot_recall", "cold_precision", "cold_recall", "false_donor_rate",
            "excess_medrel_pct", "excess_p90rel_pct", "excess_mae_ms"]
    print("\npredictor   h  " + " ".join("%10s" % k[:10] for k in keys))
    for (p, h) in sorted(agg, key=lambda x: (x[1], x[0])):
        v = agg[(p, h)]
        print("%-10s %2d  " % (p, h) + " ".join("%10.3f" % (sum(float(x[k]) for x in v) / len(v)) for k in keys))


if __name__ == "__main__":
    main()
