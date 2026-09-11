#!/usr/bin/env python3
"""Leave-one-case-out: can ONE run of an unseen case pin its whole config cost curve?

cost_separability.py reported a 2.12% residual. That is an IN-SAMPLE number -- the case's own
runs helped fit the very main effects being tested against them. The operational claim is
different and stronger:

    a case nobody has run before arrives; you run it ONCE, in one configuration;
    you then predict what every OTHER configuration would have cost.

That is what the deadline actually demands (two random large cases, ~10 h, no budget for a
config sweep per case), and it is only testable out of sample. So: fit base/f(mp)/g(bl) on
five cases, hold the sixth out entirely, use exactly one of the held-out case's runs to pin
its base, and score the predictions on the runs that were never seen.

Reported for every choice of anchor run, not just a lucky one -- if the answer depends on
WHICH single run you happen to have, that is the finding, and it decides whether the anchor
has to be specified rather than left to chance.
"""
import csv
import math
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else '/home/u5727520/wrf-results/derived-data/config-cost.csv'
IOKEY = ('io_form_history', 'io_form_restart', 'history_interval')

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        if not r['variant'].startswith('mp') or not r['s_per_model_h']:
            continue
        rows.append({'grp': (r['case'],) + tuple(r.get(k) for k in IOKEY),
                     'case': r['case'], 'mp': int(r['mp_physics']),
                     'bl': int(r['bl_pbl_physics']), 'cost': float(r['s_per_model_h']),
                     'variant': r['variant']})
groups = {}
for r in rows:
    groups.setdefault(r['grp'], []).append(r)
groups = {k: v for k, v in groups.items() if len(v) >= 4}
rows = [r for g in groups.values() for r in g]


def fit(train):
    """Alternating least squares on log cost, main effects only. -> (base, fmp, gbl)."""
    lc = {id(r): math.log(r['cost']) for r in train}
    base = {r['grp']: 0.0 for r in train}
    fmp = {r['mp']: 0.0 for r in train}
    gbl = {r['bl']: 0.0 for r in train}
    for _ in range(300):
        for d, key in ((base, 'grp'), (fmp, 'mp'), (gbl, 'bl')):
            for k in d:
                sel = [r for r in train if r[key] == k]
                if sel:
                    d[k] += sum(lc[id(r)] - base[r['grp']] - fmp[r['mp']] - gbl[r['bl']]
                                for r in sel) / len(sel)
        for d in (fmp, gbl):                        # gauge: main effects sum to zero
            m = sum(d.values()) / len(d)
            for k in d:
                d[k] -= m
            for k in base:
                base[k] += m
    return base, fmp, gbl


print('=== leave-one-case-out: fit on 5 cases, anchor the 6th with ONE run ===')
print('%-18s %-9s %6s   %8s  %8s   %s' % ('held-out case', 'anchor', 'n_pred', 'MAPE', 'worst', 'worst config'))
allerr, allworst = [], []
for held in sorted({r['grp'] for r in rows}):
    train = [r for r in rows if r['grp'] != held]
    test = [r for r in rows if r['grp'] == held]
    base, fmp, gbl = fit(train)
    for anchor in test:
        if anchor['mp'] not in fmp or anchor['bl'] not in gbl:
            continue                                # unseen scheme: model cannot speak
        # one run pins the case's base; everything else is predicted from the main effects
        b = math.log(anchor['cost']) - fmp[anchor['mp']] - gbl[anchor['bl']]
        errs = []
        for r in test:
            if r is anchor or r['mp'] not in fmp or r['bl'] not in gbl:
                continue
            pred = math.exp(b + fmp[r['mp']] + gbl[r['bl']])
            errs.append((abs(pred / r['cost'] - 1) * 100, r['variant']))
        if not errs:
            continue
        mape = sum(e for e, _ in errs) / len(errs)
        worst, wv = max(errs)
        allerr.append(mape)
        allworst.append(worst)
        print('%-18s %-9s %6d   %7.2f%%  %7.2f%%   %s'
              % (held[0], anchor['variant'], len(errs), mape, worst, wv))

if allerr:
    print()
    print('  out-of-sample MAPE : mean %.2f%%   median %.2f%%   worst-anchor %.2f%%'
          % (sum(allerr) / len(allerr), sorted(allerr)[len(allerr) // 2], max(allerr)))
    print('  worst single config: %.2f%%' % max(allworst))
    print('  (in-sample residual was 2.12%%; run-to-run reproducibility is 0.68%%)')
