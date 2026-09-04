#!/usr/bin/env python3
"""Retrospective closed-loop rehearsal: would the controller have made the right call?

Model, cost gate and transition controller all exist and are individually tested, but they
have never been run as one loop. Doing that live costs a pair of cluster runs; doing it
against segments already measured costs nothing and catches the integration faults that do
not depend on corpus size -- which timestamp the prediction is made from, whether the gate
gets seconds or a rate, whether the hit rate handed to the gate matches the scheme in play.

The loop, per segment boundary k -> k+1, exactly as it would run live:
  1. features from the frame at k          (available the moment WRF writes it)
  2. predict per-rank cost for k+1         (per-bl head, no geometry)
  3. project onto the remaining window     (rate x steps left)
  4. imbalance_gate_predicted(...)         (discounted by the scheme's measured hit rate)
  5. compare against the gate on the MEASURED k+1 -- the decision the controller would have
     made with hindsight

An agreement is only meaningful against a baseline: `always-hold` and `always-switch` are
reported too, because on a skewed distribution one of them can look excellent while carrying
no information at all.

usage: closed_loop.py <case>...
"""
import sys, os, glob, re, importlib.util
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
import wtune_oracle_train as tr

SEG_HOURS = 3.0
FIT = dict(depth=3, n_est=80, lr=0.10)
SWITCH_COST_S = float(os.environ.get("WTUNE_SWITCH_COST", 56.0))   # ADIOS2 restart 56s; in-place migration ~3s
HIT = {1: 0.61, 2: 0.53, 5: 0.64}   # measured per-bl hit rate, geometry excluded


def _controller():
    p = '/home/u5727520/wrf-stack/g0a/wtune-transition-controller.py'
    s = importlib.util.spec_from_file_location('ctl', p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def build(case):
    r, why = seg.build_case(case)
    if r is None:
        return None
    X, y, cs, sg, names = r
    nl = f'{seg.RESULTS}/{case}/namelist.input'
    t = open(nl).read() if os.path.exists(nl) else ''
    m = re.search(r'time_step=(\d+)', t)
    dt = int(m.group(1)) if m else 15
    keep = [i for i, q in enumerate(names) if not q.startswith('geo_')]
    return dict(case=case, X=X[:, keep], names=[names[i] for i in keep], seg=sg,
                rate=cs / (SEG_HOURS * 3600.0 / dt), bl=seg.schemes(case)['bl'],
                steps_per_seg=SEG_HOURS * 3600.0 / dt)


def main():
    ctl = _controller()
    want = sys.argv[1:]
    rows = [b for b in (build(c) for c in want) if b]
    if len(rows) < 3:
        print("need >= 3 usable cases; got %d" % len(rows)); return 2
    names = rows[0]['names']
    rows = [r for r in rows if r['names'] == names]
    n = len(rows)

    print("=== 閉環演練：預測 → 閘 → 決策，對照實測的正確決策 ===")
    print("switch 成本 %.0fs；命中率折扣 per-bl %s" % (SWITCH_COST_S, HIT))
    print("spread 收縮校正：在訓練折上量測後除掉\n")
    print("%-26s %3s %4s %10s %10s %9s %9s %s"
          % ("case", "bl", "seg", "預測可回收", "實測可回收", "預測決策", "正確決策", ""))
    agree = tot = agree_p = 0
    hold_base = switch_base = 0
    for i, r in enumerate(rows):
        # train on the other cases sharing this PBL scheme; that is the production path
        trm = np.concatenate([np.full(len(q['rate']), q['bl'] == r['bl'] and j != i)
                              for j, q in enumerate(rows)])
        Xall = np.vstack([q['X'] for q in rows])
        rate_all = np.concatenate([q['rate'] for q in rows])
        sg_all = np.concatenate([q['seg'] for q in rows])
        cid_all = np.concatenate([np.full(len(q['rate']), j) for j, q in enumerate(rows)])
        grp = sg_all + 1000 * cid_all
        sm = np.empty_like(rate_all)
        for g in np.unique(grp):
            sm[grp == g] = rate_all[grp == g].mean()
        if trm.sum() < len(r['rate']):
            trm = cid_all != i
        model = tr.gbrt_fit(Xall[trm], (rate_all / sm)[trm], **FIT)
        # Squared-error regression SHRINKS extremes toward the mean, so max(pred)-mean(pred)
        # is biased low as an estimate of recoverable seconds -- measured 0.5-0.8x of truth.
        # Stacked on top of the hit-rate discount it made the gate answer "hold" every single
        # time, matching the always-hold baseline exactly: an 85% agreement carrying no
        # information. Calibrate the shrinkage on the TRAINING rows only, then discount.
        shr = []
        for g in np.unique(grp[trm]):
            m = (grp == g) & trm
            if m.sum() < 8:
                continue
            p_ = tr.gbrt_pred(model, Xall[m]); t_ = (rate_all / sm)[m]
            ts = t_.max() - t_.mean()
            if ts > 1e-9:
                shr.append((p_.max() - p_.mean()) / ts)
        shrink = float(np.median(shr)) if shr else 1.0
        shrink = min(max(shrink, 0.2), 1.0)

        ks = sorted(np.unique(r['seg']))
        for k in ks[:-1]:
            cur = r['seg'] == k
            nxt = r['seg'] == (k + 1)
            if not nxt.any():
                continue
            # step 1-3: predict the NEXT segment from the CURRENT frame's features, scaled by
            # the level this case is running at (measured so far, as it would be live)
            shape_hat = tr.gbrt_pred(model, r['X'][cur])
            level = r['rate'][cur].mean()
            pred_s = shape_hat / shape_hat.mean() * level * r['steps_per_seg']
            # undo the shrinkage: re-inflate the spread about its own mean
            mu = pred_s.mean()
            pred_s = mu + (pred_s - mu) / shrink
            true_s = r['rate'][nxt] * r['steps_per_seg']
            g_pred = ctl.imbalance_gate_predicted(list(pred_s), SWITCH_COST_S, HIT.get(r['bl'], 0.6))
            g_true = ctl.imbalance_gate(list(true_s), SWITCH_COST_S)
            # PERSISTENCE: decide the next segment from the CURRENT one as measured. The
            # controller can always do this -- the shim reports the segment it just finished --
            # so a model has to beat it to be worth wiring in at all.
            g_pers = ctl.imbalance_gate(list(r['rate'][cur] * r['steps_per_seg']), SWITCH_COST_S)
            ok = g_pred['trigger'] == g_true['trigger']
            agree_p += (g_pers['trigger'] == g_true['trigger'])
            agree += ok; tot += 1
            hold_base += (g_true['trigger'] is False)
            switch_base += (g_true['trigger'] is True)
            print("%-26s %3d %4d %9.1fs %9.1fs %9s %9s %s"
                  % (r['case'], r['bl'], k, g_pred['recoverable_s'], g_true['recoverable_s'],
                     "switch" if g_pred['trigger'] else "hold",
                     "switch" if g_true['trigger'] else "hold", "" if ok else "  <- 不一致"))
    print("\n決策一致率 %d/%d = %.0f%%" % (agree, tot, 100.0 * agree / tot))
    print("persistence（用上一段實測）  %d/%d = %.0f%%" % (agree_p, tot, 100.0 * agree_p / tot))
    print("對照基準: always-hold %.0f%%, always-switch %.0f%%"
          % (100.0 * hold_base / tot, 100.0 * switch_base / tot))
    if max(hold_base, switch_base) / tot >= agree / tot:
        print("⚠ 未勝過恆定基準——這個一致率不代表模型有資訊")
    return 0


if __name__ == '__main__':
    sys.exit(main())
