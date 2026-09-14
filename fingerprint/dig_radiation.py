#!/usr/bin/env python3
"""Diagnose fingerprint phase misattribution on one sampled run (job 1074426 G1: radiation/PBL undercounted).

For rank 0's master thread on its host: bin samples in 0.25 s bins, report per-bin phase counts, find the
long radiation step as the busiest-time window, and list the top symbols (with dso and classified phase)
overall and inside the bins where RADIATION samples appear or where the step is long.
usage: dig_radiation.py RUNDIR [RANK]
"""
import collections
import gzip
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fingerprint as fp  # noqa: E402
from phase_classify import classify  # noqa: E402

run = sys.argv[1]
want_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
pid_info, _hw, _hosts = fp.load_meta(os.path.join(run, 'fp'))
key = [k for k, v in pid_info.items() if v.get('rank') == want_rank][0]
host, pid = key
path = os.path.join(run, 'fp', 'perf.%s.txt.gz' % host)
print('rank', want_rank, 'host', host, 'pid', pid, 'file', path)

master, workers = [], collections.Counter()
with gzip.open(path, 'rt', errors='replace') as fh:
    for line in fh:
        m = fp.PERF_LINE_RE.match(line)
        if not m:
            continue
        p, t, _cpu, ts, _ip, sym, dso = m.groups()
        if int(p) != pid:
            continue
        if int(t) == pid:
            master.append((float(ts), sym.strip(), dso.strip()))
        else:
            workers[(sym.strip(), os.path.basename(dso.strip()), classify(sym.strip(), dso.strip()))] += 1
master.sort()
print('master samples', len(master), 'worker samples', sum(workers.values()))

att = fp.attribute_stream(master)
t0 = att[0][0]
bins = collections.defaultdict(collections.Counter)
for (t, phase, dt) in att:
    bins[int((t - t0) / 0.25)][phase] += dt
print('\n# phase seconds per phase over the window (master, time-weighted)')
tot = collections.Counter()
for (t, phase, dt) in att:
    tot[phase] += dt
for ph, s in tot.most_common():
    print('  %-15s %.3f s' % (ph, s))

print('\n# 0.25 s bins with any RADIATION/PBL time, or dominant non-DYNAMICS phase')
for b in sorted(bins):
    c = bins[b]
    if c.get('RADIATION', 0) > 0 or c.get('PBL', 0) > 0.02:
        print('  t=%6.2f %s' % (b * 0.25, ' '.join('%s=%.3f' % kv for kv in c.most_common(4))))

syms = collections.Counter()
for (t, sym, dso) in master:
    syms[(sym, os.path.basename(dso), classify(sym, dso))] += 1
print('\n# top 40 master symbols')
for (sym, dso, ph), n in syms.most_common(40):
    print('  %6d %-15s %-24s %s' % (n, ph, dso[:24], sym[:90]))
print('\n# any symbol containing rad/rrtm/ra_ or bl_/pbl/ysu (master), top 25')
pat = re.compile(r'rad|rrtm|module_ra_|module_bl_|pbl|ysu', re.I)
for (sym, dso, ph), n in [x for x in syms.most_common() if pat.search(x[0][0])][:25]:
    print('  %6d %-15s %-24s %s' % (n, ph, dso[:24], sym[:90]))
print('\n# top 15 worker symbols')
for (sym, dso, ph), n in workers.most_common(15):
    print('  %6d %-15s %-24s %s' % (n, ph, dso[:24], sym[:90]))

# longest stretch where consecutive master samples are all DYNAMICS_OTHER (candidate misclassified phase)
best, cur_start, cur_len = (0, 0, 0), None, 0
for i, (t, phase, dt) in enumerate(att):
    if phase == 'DYNAMICS_OTHER':
        if cur_start is None:
            cur_start, cur_len = i, 0
        cur_len += dt
        if cur_len > best[0]:
            best = (cur_len, cur_start, i)
    else:
        cur_start = None
print('\n# longest contiguous DYNAMICS_OTHER stretch: %.3f s at t=%.2f' % (best[0], att[best[1]][0] - t0 if att else 0))
stretch = collections.Counter()
for (t, sym, dso) in master[best[1]:best[2] + 1]:
    stretch[(sym, os.path.basename(dso))] += 1
for (sym, dso), n in stretch.most_common(15):
    print('  %6d %-24s %s' % (n, dso[:24], sym[:90]))
