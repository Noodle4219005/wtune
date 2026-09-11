#!/usr/bin/env python3
"""Harvest cost(case, config) from runs that ALREADY EXIST on disk.

WTune's target used to be the per-rank hot-spot hit rate: which ranks are expensive. Every
rebalancing actuator that could have consumed that answer is now closed by measurement
(core lending saturates at 44% absorption against 73% needed; WRF's own physics load
balancing is #if 0'd out of the source), so the answer is correct and unusable.

The target that survives is cost(case, config): what will this configuration COST, for a
configuration nobody has run. That is the one question measurement cannot answer, because
you cannot measure a config you have not run -- which is precisely what makes a model
irreplaceable there rather than merely convenient.

The enabling fact is that no new machine time is needed. bucket_load.csv already carries
wall clock: the buckets are fixed-width WALL-clock buckets, so a config that takes longer
simply fills more of them. wall = last_bucket_t_start + mean_rank(comp+wait) of that
bucket. 46 runs varying mp_physics/bl_pbl_physics/decomposition are already on disk.

Bucket accounting, which is easy to get wrong in three separate ways:
  * bucket 0 absorbs initialisation -- charge() only splits a span across ONE bucket edge,
    so a long startup lands entirely in the bucket it began in. Wall clock WANTS it;
    imbalance statistics must not.
  * the final bucket is partial. Its width is mean(comp+wait), not bsec.
  * bucket width is DERIVED from t_start_s, never assumed. Diagnostic runs use 10 s and
    hardcoding 300 s silently produces garbage -- that bug once trained a model to 8.0%,
    below the 10% random baseline.

Model span comes from the features_<date>_<hhmm>.csv frame names, not from namelist
run_hours: the namelist records what was REQUESTED, the frames record what was produced,
and a run killed by the time limit differs in exactly the way that matters here.
"""
import csv
import json
import os
import re
import sys
from datetime import datetime

RESULTS = '/home/u5727520/wrf-results'

# knobs read out of namelist.input. Everything here is either a physics choice or a shape
# that changes cost without changing the science being asked for.
KNOBS = ['mp_physics', 'bl_pbl_physics', 'cu_physics', 'ra_lw_physics', 'ra_sw_physics',
         'sf_sfclay_physics', 'sf_surface_physics', 'nproc_x', 'nproc_y', 'numtiles',
         'time_step', 'radt', 'e_we', 'e_sn', 'e_vert', 'dx',
         # I/O knobs are here because wall clock is not comparable across them, not because
         # anyone wants to tune them. Leaving history_interval out once made the I/O guard
         # below compare None against None and pass runs it should have quarantined.
         'io_form_history', 'io_form_restart', 'history_interval', 'frames_per_outfile',
         # needed by model_span_h as the second, independent source of the run length
         'run_hours', 'run_days']

# directory-name suffixes that encode a variant rather than a different case
VARIANT = re.compile(r'^(?P<base>.+?)(?:__(?P<phys>mp\d+bl\d+)|_(?P<dec>\d+x\d+)(?P<h>h\d+)?)$')
FRAME = re.compile(r'^features_(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})\.csv$')


def parse_namelist(path):
    """-> {knob: value}. Handles ' mp_physics = 8, 8, 8,' -- the d01 value is the first."""
    out = {}
    try:
        txt = open(path).read()
    except OSError:
        return out
    for k in KNOBS:
        m = re.search(r'\b%s\s*=\s*([^,\n]+)' % k, txt)
        if m:
            v = m.group(1).strip().rstrip(',')
            try:
                out[k] = float(v) if '.' in v else int(v)
            except ValueError:
                out[k] = v
    return out


def model_span_h(d, nl):
    """Model hours the run actually covered. -> (hours, source, disagreement flag)

    Two independent sources, because neither alone is trustworthy:

      frames    -- span of the features_<date>_<hhmm>.csv names. Records what was PRODUCED,
                   which is what you want when a run was killed early. But featurisation can
                   stop before the run does, and then the span is short while the run was not.
      namelist  -- run_days/run_hours. Records what was REQUESTED, so it is wrong in exactly
                   the opposite case.

    Trusting frames alone put lake_effect in the corpus at 1250.8 s/model-h -- the most
    expensive case by a wide margin and pure fiction. rsl.out.0000 showed 7200 steps over 12
    model hours with a wall time matching the harvest to 0.6%: the run was 12 h, only its
    features stopped at 3 h. Four other runs were wrong the same way, and every one of them
    became a fake outlier that the cost model then had to explain.

    So: prefer the namelist, keep the frame span, and flag any run where they disagree by more
    than 10% instead of silently picking one.
    """
    ts = []
    for f in os.listdir(d):
        m = FRAME.match(f)
        if m:
            ts.append(datetime.strptime('%s %s:%s' % m.groups(), '%Y-%m-%d %H:%M'))
    frames_h = (max(ts) - min(ts)).total_seconds() / 3600.0 if len(ts) >= 2 else None
    nl_h = None
    if nl.get('run_hours') is not None or nl.get('run_days') is not None:
        nl_h = float(nl.get('run_hours') or 0) + 24.0 * float(nl.get('run_days') or 0)
        if nl_h <= 0:
            nl_h = None
    if nl_h is None:
        return frames_h, 'frames', False
    if frames_h is None:
        return nl_h, 'namelist', False
    disagree = abs(frames_h - nl_h) / max(nl_h, 1e-9) > 0.10
    return nl_h, 'namelist', disagree


def read_buckets(path):
    """-> (comp, wait, bsec) with comp/wait as rank x bucket, bsec derived from t_start_s."""
    nr = nb = 0
    rows = []
    with open(path) as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            rk, bk = int(r['rank']), int(r['bucket'])
            rows.append((rk, bk, float(r['t_start_s']), float(r['comp_s']), float(r['wait_s'])))
            nr = max(nr, rk + 1)
            nb = max(nb, bk + 1)
    if not rows:
        return None
    comp = [[0.0] * nb for _ in range(nr)]
    wait = [[0.0] * nb for _ in range(nr)]
    starts = {}
    for rk, bk, t, c, w in rows:
        comp[rk][bk] = c
        wait[rk][bk] = w
        starts[bk] = t
    d = sorted({round(starts[b + 1] - starts[b], 6) for b in range(nb - 1) if b in starts and b + 1 in starts})
    d = [x for x in d if x > 0]
    bsec = d[len(d) // 2] if d else 300.0
    return comp, wait, bsec, starts


def summarize(name):
    d = os.path.join(RESULTS, name)
    bl = os.path.join(d, 'bucket_load.csv')
    nl = os.path.join(d, 'namelist.input')
    if not (os.path.isfile(bl) and os.path.isfile(nl)):
        return None
    got = read_buckets(bl)
    if not got:
        return None
    comp, wait, bsec, starts = got
    nr, nb = len(comp), len(comp[0])
    used = [b for b in range(nb) if sum(comp[r][b] + wait[r][b] for r in range(nr)) > 0]
    if len(used) < 2:
        return None
    last = max(used)
    tail = sum(comp[r][last] + wait[r][last] for r in range(nr)) / nr
    wall_s = starts.get(last, last * bsec) + tail

    # imbalance over the steady-state buckets only: drop bucket 0 (init) and the partial tail
    core = [b for b in used if b != 0 and b != last]
    m_pct = None
    comp_frac = None
    if core:
        per = [sum(comp[r][b] for b in core) for r in range(nr)]
        tot = [sum(comp[r][b] + wait[r][b] for b in core) for r in range(nr)]
        mean = sum(per) / nr
        if mean > 0:
            m_pct = 100.0 * (max(per) - mean) / mean
        if sum(tot) > 0:
            comp_frac = 100.0 * sum(per) / sum(tot)

    mm = VARIANT.match(name)
    base = mm.group('base') if mm else name
    variant = (mm.group('phys') or ((mm.group('dec') or '') + (mm.group('h') or ''))) if mm else ''

    knobs = parse_namelist(nl)
    mh, mh_src, mh_disagree = model_span_h(d, knobs)
    row = {'run': name, 'case': base, 'variant': variant, 'n_ranks': nr,
           'wall_s': round(wall_s, 1), 'bucket_s': bsec, 'n_buckets': len(used),
           'model_h': mh, 'model_h_source': mh_src, 'model_h_disagree': int(mh_disagree),
           'imbalance_M_pct': None if m_pct is None else round(m_pct, 2),
           'comp_frac_pct': None if comp_frac is None else round(comp_frac, 2)}
    row.update(knobs)
    if row['model_h']:
        row['s_per_model_h'] = round(wall_s / row['model_h'], 1)
    return row


def main():
    rows = []
    for name in sorted(os.listdir(RESULTS)):
        if not os.path.isdir(os.path.join(RESULTS, name)):
            continue
        try:
            r = summarize(name)
        except Exception as e:                       # a half-written run must not kill the sweep
            print('SKIP %-34s %s' % (name, e), file=sys.stderr)
            continue
        if r:
            rows.append(r)
    cols = ['run', 'case', 'variant', 'n_ranks', 'model_h', 'model_h_source',
            'model_h_disagree', 'wall_s', 's_per_model_h',
            'imbalance_M_pct', 'comp_frac_pct', 'bucket_s', 'n_buckets'] + KNOBS
    nd = [r['run'] for r in rows if r.get('model_h_disagree')]
    if nd:
        print('!! %d runs where the namelist run length and the feature-frame span disagree by '
              '>10%%.\n   model_h now comes from the namelist; the frame span is the one that '
              'was wrong before:\n   %s' % (len(nd), ' '.join(nd)))
    out = sys.argv[1] if len(sys.argv) > 1 else '/home/u5727520/wrf-results/derived-data/config-cost.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('%d runs -> %s' % (len(rows), out))

    # the comparison that the whole reformulation rests on: within one case, how much does a
    # physics config actually move the cost, and is the RANKING the same across cases?
    #
    # Grouping is by (case, I/O settings), NOT by case alone. Two runs of the same case with
    # io_form=2 and io_form=14 do not have comparable wall clock: netCDF -> ADIOS2 is worth
    # -29% on the restart write in this project's own measurement. Comparing across that
    # boundary once produced a confident "16x28 beats 8x56 by 5.6-8.7%" that a controlled
    # sweep then reversed -- the entire effect was the I/O backend. Physics knobs were checked
    # by hand and I/O knobs were not, because the question asked was "did the science change?"
    # when the question that mattered was "is the wall clock comparable?".
    IOKEY = ('io_form_history', 'io_form_restart', 'history_interval')
    by_case = {}
    for r in rows:
        if r['variant'].startswith('mp'):
            by_case.setdefault(r['case'], []).append(r)
    print('\n%-18s %-9s %8s %8s %7s  %s' % ('case', 'variant', 'wall_s', 's/model_h', 'M%', 'vs cheapest'))
    for case in sorted(by_case):
        v = [r for r in by_case[case] if r.get('s_per_model_h')]
        groups = {}
        for r in v:
            groups.setdefault(tuple(r.get(k) for k in IOKEY), []).append(r)
        if len(groups) > 1:
            print('%-18s !! MIXED I/O -- %d incomparable groups, compared separately:' % (case, len(groups)))
            for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                print('%-18s      %s = %s  ->  %s' % ('', '/'.join(IOKEY), k,
                                                      ' '.join(r['variant'] for r in g)))
        for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(g) < 2:
                print('%-18s %-9s %8.1f %8.1f %7s  (alone in its I/O group -- not compared)'
                      % (case, g[0]['variant'], g[0]['wall_s'], g[0]['s_per_model_h'],
                         g[0]['imbalance_M_pct']))
                continue
            g.sort(key=lambda r: r['s_per_model_h'])
            lo = g[0]['s_per_model_h']
            for r in g:
                print('%-18s %-9s %8.1f %8.1f %7s  %+.1f%%'
                      % (case, r['variant'], r['wall_s'], r['s_per_model_h'],
                         r['imbalance_M_pct'], 100.0 * (r['s_per_model_h'] / lo - 1)))
        print()
    json.dump(rows, open(out.replace('.csv', '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
