#!/usr/bin/env python3
"""Is the per-rank cost pattern reproducible at all? The ceiling on any predictor.

karina_cat4__mp10bl5 (Morrison + MYNN) is the worst case in the hot-spot metric: 42% where
the same weather under YSU/MYJ reaches 78-84%. Two readings are possible and they lead
opposite ways -- either the features miss what drives MYNN's cost (keep looking), or MYNN's
per-rank pattern is not reproducible run to run (stop looking; no feature can predict noise).

Same namelist, same met_em, run twice. If run-to-run agreement is high the pattern is
deterministic and the 42% is a modelling gap. If it is low, 42% is near the ceiling.
"""
import sys, numpy as np
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import wtune_segment_train as seg, wtune_oracle_train as tr

def load(path, nb_hint=None):
    rows = np.genfromtxt(path, delimiter=',', names=True)
    nb = int(rows['bucket'].max()) + 1
    comp = np.zeros((448, nb)); wait = np.zeros((448, nb))
    for r, b, c, w in zip(rows['rank'], rows['bucket'], rows['comp_s'], rows['wait_s']):
        comp[int(r), int(b)] = c; wait[int(r), int(b)] = w
    return comp, wait

c1, w1 = load(sys.argv[1]); c2, w2 = load(sys.argv[2])
nb = min(c1.shape[1], c2.shape[1])
t1 = c1[:, :nb].sum(1); t2 = c2[:, :nb].sum(1)
t1[0] = np.median(t1); t2[0] = np.median(t2)          # rank 0 is the serial-I/O outlier
print("兩次跑的 per-rank compute 秒數（前 %d 桶）" % nb)
print("  run1 %.1f..%.1f 平均 %.1f" % (t1.min(), t1.max(), t1.mean()))
print("  run2 %.1f..%.1f 平均 %.1f" % (t2.min(), t2.max(), t2.mean()))
print("\n重現性:")
print("  Spearman(run1, run2)        = %+.3f" % tr._spearman(t1, t2))
print("  Pearson                     = %+.3f" % np.corrcoef(t1, t2)[0, 1])
r1 = set(np.argsort(t1)[-45:]); r2 = set(np.argsort(t2)[-45:])
print("  最貴 45 個 rank 的重疊       = %.1f%%  (亂猜 10%%)" % (100.0 * len(r1 & r2) / 45))
print("\n判讀: 這個重疊率就是任何預測器的天花板——模型不可能比「同一個實驗自己」更一致。")
