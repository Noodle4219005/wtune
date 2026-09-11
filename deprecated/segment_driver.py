#!/usr/bin/env python3
"""Run a WRF logical run as a chain of restart segments, measuring and deciding at each boundary.

Runs INSIDE an existing Slurm allocation and calls mpirun per segment, so the whole logical run
costs one queue wait rather than N.

WHAT THIS DOES AND DOES NOT BUY, STATED UP FRONT
------------------------------------------------
The measurement loop is real and this driver closes it. The rebalancing ACTION is not yet
worth anything, and pretending otherwise would be the easiest mistake to make here:

  * with 448 ranks pinned to 4 nodes, the only lever at a boundary is nproc_x x nproc_y;
  * 8x56 is the measured optimum for 901^2 at 3 km, and the nest shape sweep showed the
    penalty for choosing wrong runs to +17.4% -- patch cost is set by x-contiguity
    (vectorisation), not by load;
  * the wiki's own verdict is that under an absolute source freeze WRF has no external
    load-imbalance fix at all.

So the driver's immediate value is INFRASTRUCTURE: segmented execution, per-segment per-rank
measurement, and a recorded decision trace. That is what makes the first honest closed-loop
A/B possible -- fixed 8x56 for 12 h against per-segment adaptive -- which will produce a real
wall-time number instead of an offline hit rate. If that number is zero, the loop has told us
something worth knowing at last.

Restart is safe to build on: continuing from a restart is BIT-IDENTICAL to running straight
through, so segmenting cannot contaminate the science.

FOUR SILENT FAILURES THIS CODE EXISTS TO AVOID
----------------------------------------------
Every one of them has already cost a run:
  1. start_hour is NOT at the start of its line (it shares one with start_year), so an
     anchored `sed 's/^ *start_hour=.*/.../'` matches nothing and WRF hunts for a restart at
     the original time and aborts.
  2. Every start_*/end_* field is a max_domains ARRAY. A single value leaves d02 on the
     Registry default of 1993-03-13.
  3. A substituting sed for restart_interval does nothing when the key is absent, so no
     restart is ever written and the next segment cannot start.
  4. ADIOS2 restarts are DIRECTORIES. `rm -f` does not remove them.

usage (inside an allocation):
    segment_driver.py <case-dir> --total-h 12 --seg-h 3 --ranks 448 [--dry-run]
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wtune_place as wp
from decay_gate import decay_gate, load_bucket_csv, DEFAULT_WINDOW_S

SHIM = '/home/u5727520/wrf-stack/lend-microbench/libblockbar_bucket.so'
BUCKET_SEC = 10.0
NBUCKET = 800
MODEL_HIT_PCT = 46.9          # resid own, cross-case, leave-one-weather-out
# patch aspect = (nx/px)/(ny/py) = (nx*py)/(ny*px). For 901^2 the measured optimum 8x56 gives
# 56/8 = 7.0, NOT 9. The controller's comment says "~9:1" and taking that literally makes
# choose_decomp return 7x64 (aspect 9.14) -- a decomposition the sweep measured at 0.765 s/step
# against 8x56's 0.738, i.e. 3.5% SLOWER. Caught by a dry run before it cost a job.
TARGET_ASPECT = 7.0


# ────────────────────────────── namelist ──────────────────────────────
def nl_normalize(text):
    """One key per line. WRF's reader does not care about layout, and this turns every later
    get/set into a line match instead of an in-line regex that has to know what follows."""
    out = []
    for raw in text.splitlines():
        s = raw.rstrip()
        t = s.strip()
        if not t or t.startswith('&') or t == '/':
            out.append(s)
            continue
        for part in re.split(r'(?=\b[A-Za-z_]\w*\s*=)', s):
            part = part.strip()
            if part:
                out.append(' ' + part)
    return '\n'.join(out) + '\n'


def nl_get(text, key):
    m = re.search(r'^\s*%s\s*=\s*(.*?),?\s*$' % re.escape(key), text, re.M | re.I)
    return m.group(1).strip() if m else None


def nl_set(text, key, value, section='&domains'):
    """Set a key, inserting it into `section` if absent. Insertion is the point: a plain
    substitution silently does nothing for a key the template never had."""
    line = ' %s=%s,' % (key, value)
    pat = re.compile(r'^\s*%s\s*=.*$' % re.escape(key), re.M | re.I)
    if pat.search(text):
        return pat.sub(line, text, count=1)
    return re.sub(r'^(\s*%s\s*)$' % re.escape(section), r'\1\n' + line, text,
                  count=1, flags=re.M | re.I)


def max_dom(text):
    v = nl_get(text, 'max_dom')
    try:
        return int(str(v).split(',')[0])
    except Exception:
        return 1


def dup(value, n):
    """max_domains arrays: one value per domain, always. This is failure mode 2."""
    return ', '.join([str(value)] * n)


def set_window(text, t0, hours, ranks_px, ranks_py):
    """Rewrite the namelist for one segment: its clock, its restart behaviour, its decomposition."""
    nd = max_dom(text)
    t1 = t0 + dt.timedelta(hours=hours)
    for key, val in (('start_year', t0.year), ('start_month', '%02d' % t0.month),
                     ('start_day', '%02d' % t0.day), ('start_hour', '%02d' % t0.hour),
                     ('start_minute', 0), ('start_second', 0),
                     ('end_year', t1.year), ('end_month', '%02d' % t1.month),
                     ('end_day', '%02d' % t1.day), ('end_hour', '%02d' % t1.hour),
                     ('end_minute', 0), ('end_second', 0)):
        text = nl_set(text, key, dup(val, nd), '&time_control')
    text = nl_set(text, 'run_days', 0, '&time_control')
    text = nl_set(text, 'run_hours', int(hours), '&time_control')
    text = nl_set(text, 'run_minutes', 0, '&time_control')
    text = nl_set(text, 'run_seconds', 0, '&time_control')
    # restart_interval is in MINUTES and must land exactly on the boundary, or the next
    # segment has no wrfrst to start from. Failure mode 3.
    text = nl_set(text, 'restart_interval', int(hours * 60), '&time_control')
    text = nl_set(text, 'io_form_history', 14, '&time_control')
    # io_form_restart is overridable for one reason: ADIOS2 aggregates ONE SUBFILE PER NODE by
    # default (external/io_adios2/wrf_io.F90:1296, "numaggregators = 0 will set a single
    # aggregator per node"), so a restart written under one rank->node map and read under
    # another is the suspected cause of the heap corruption that killed the --place follow arm
    # at its third segment. netCDF (io_form 2) has a single placement-independent layout and is
    # the control that localises the fault.
    text = nl_set(text, 'io_form_restart',
                  int(os.environ.get('WTUNE_IO_FORM_RESTART', 14)), '&time_control')
    text = nl_set(text, 'nproc_x', ranks_px, '&domains')
    text = nl_set(text, 'nproc_y', ranks_py, '&domains')
    return text


def set_restart(text, on):
    return nl_set(text, 'restart', '.true.' if on else '.false.', '&time_control')


def slurm_hosts():
    """-> the allocation's hostnames in order, or [] outside Slurm.

    Needed because a machinefile names HOSTS, and the driver runs inside an allocation whose
    node list is only known at run time. Returning [] rather than raising keeps --place follow
    degrading to the default map instead of killing a run that is otherwise fine.
    """
    nl = os.environ.get('SLURM_JOB_NODELIST')
    if not nl:
        return []
    try:
        out = subprocess.check_output(['scontrol', 'show', 'hostnames', nl])
        return [h for h in out.decode().split() if h]
    except Exception:
        return []


# ────────────────────────────── decomposition ──────────────────────────────
MIN_PATCH_CELLS = 10          # WRF's own limit, quoted verbatim below


def legal_decomps(ranks, nx, ny):
    """px*py=ranks whose patch WRF will actually accept.

    The bound is not px<=nx. WRF refuses any patch thinner than 10 cells on either axis and
    says so itself during domain setup:

        Minimum decomposed computational patch size, either x-dir or y-dir, is 10 grid cells.
        e_sn = 901, nproc_y = 112, with cell width in y-direction = 8
        --- ERROR: Reduce the MPI rank count, or redistribute the tasks.

    px<=nx admits patches one cell wide -- ten times looser -- so this used to hand back
    shapes that FATAL in 3 seconds. Which shapes those are moves with the grid: 4x112 leaves
    1201/112 = 10 rows on 1501x1201 and is legal there, but 901/112 = 8 rows on 901^2 and is
    not. A shape table measured on one grid cannot be carried to another even to decide which
    ROWS exist.
    """
    out = []
    for px in range(1, ranks + 1):
        if ranks % px:
            continue
        py = ranks // px
        if nx // px >= MIN_PATCH_CELLS and ny // py >= MIN_PATCH_CELLS:
            out.append((px, py))
    return out


def choose_decomp(ranks, nx, ny):
    """Pick px*py=ranks whose patch aspect is closest to the measured optimum.

    Deliberately load-INDEPENDENT. The shape sweep found patch cost is set by the length of
    contiguous x runs, not by halo volume and not by where the work is: 16x28 and 28x16 have
    identical patch area and perimeter yet differ by 12.6% in wall time. Choosing shape from
    a load estimate would be optimising the wrong variable.
    """
    best = None
    for px, py in legal_decomps(ranks, nx, ny):
        aspect = (nx * py) / float(ny * px)
        score = abs(aspect - TARGET_ASPECT)
        if best is None or score < best[0]:
            best = (score, px, py)
    if best is None:
        raise ValueError('no px*py=%d fits %dx%d' % (ranks, nx, ny))
    return best[1], best[2]


# ────────────────────────────── execution ──────────────────────────────
def run(cmd, cwd, log, dry):
    if dry:
        print('    DRY: %s' % ' '.join(cmd))
        return 0
    with open(os.path.join(cwd, log), 'w') as fh:
        return subprocess.call(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)


def wrf_succeeded(case):
    p = os.path.join(case, 'rsl.error.0000')
    try:
        with open(p, errors='ignore') as fh:
            return 'SUCCESS COMPLETE WRF' in fh.read()
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case')
    ap.add_argument('--total-h', type=float, default=12.0)
    ap.add_argument('--seg-h', type=float, default=3.0)
    ap.add_argument('--ranks', type=int, default=448)
    ap.add_argument('--fixed', action='store_true',
                    help='control arm: never change the decomposition, still measure')
    ap.add_argument('--place', choices=('none', 'follow'), default='none',
                    help="none = shipped block placement, fixed for the whole run. "
                         "follow = at each boundary, rebuild the rank->node map from the "
                         "PREVIOUS segment's measured hot ranks so the placement tracks a "
                         "hot-spot that moves. The two are the arms of the experiment.")
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    case = os.path.abspath(a.case)
    out = case.replace('/wrf-bench/', '/wrf-results/')
    os.makedirs(out, exist_ok=True)
    base_nl = nl_normalize(open(os.path.join(case, 'namelist.input')).read())
    nd = max_dom(base_nl)
    nx = int(str(nl_get(base_nl, 'e_we')).split(',')[0])
    ny = int(str(nl_get(base_nl, 'e_sn')).split(',')[0])
    y = int(str(nl_get(base_nl, 'start_year')).split(',')[0])
    mo = int(str(nl_get(base_nl, 'start_month')).split(',')[0])
    d = int(str(nl_get(base_nl, 'start_day')).split(',')[0])
    h = int(str(nl_get(base_nl, 'start_hour')).split(',')[0])
    t_start = dt.datetime(y, mo, d, h)

    px, py = choose_decomp(a.ranks, nx, ny)
    nseg = int(round(a.total_h / a.seg_h))
    print('case=%s  %dx%d  max_dom=%d  %d segments x %g h  start=%s'
          % (os.path.basename(case), nx, ny, nd, nseg, a.seg_h, t_start))
    print('initial decomposition %dx%d  (mode=%s)\n'
          % (px, py, 'FIXED control' if a.fixed else 'adaptive'))

    manifest = {'case': case, 'ranks': a.ranks, 'seg_h': a.seg_h,
                'fixed': bool(a.fixed), 'segments': []}
    horizon_s = None

    for i in range(nseg):
        t0 = t_start + dt.timedelta(hours=i * a.seg_h)
        tag = 'seg%02d' % i
        print('== %s  t0=%s  decomp=%dx%d ==' % (tag, t0, px, py))

        nl = set_restart(set_window(base_nl, t0, a.seg_h, px, py), i > 0)
        if not a.dry_run:
            open(os.path.join(case, 'namelist.input'), 'w').write(nl)
        shutil.copyfile(os.path.join(case, 'namelist.input'),
                        os.path.join(out, 'namelist_%s.input' % tag)) if not a.dry_run else None

        if i == 0:
            # real.exe must be run over the FULL logical run, not the first segment. wrfbdy_d01
            # carries lateral boundary data only for the window real.exe was given, so a 3 h
            # real.exe leaves segment 1 with:
            #   ERROR: Cannot find a valid time to start the LBC during this restart
            # even though the restart file itself is written and opened correctly. Measured:
            # seg00 succeeded, seg01 died on this, and the restart was NOT the culprit.
            full = set_restart(set_window(base_nl, t_start, a.total_h, px, py), False)
            if not a.dry_run:
                open(os.path.join(case, 'namelist.input'), 'w').write(full)
            print('    real.exe over the full %g h (LBC must span the whole logical run)' % a.total_h)
            # ADIOS2 frames are DIRECTORIES; rm -f leaves them behind. Failure mode 4.
            for pat in ('wrfout_d0', 'wrfrst_d0'):
                for f in os.listdir(case) if os.path.isdir(case) else []:
                    if f.startswith(pat) and not a.dry_run:
                        p = os.path.join(case, f)
                        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
            rc = run(['mpirun', '-np', str(a.ranks), './real.exe'], case, 'real_%s.out' % tag, a.dry_run)
            for dom in range(1, nd + 1):
                f = os.path.join(case, 'wrfinput_d%02d' % dom)
                if not a.dry_run and not os.path.exists(f):
                    print('  FATAL: %s missing after real.exe (rc=%d)' % (f, rc))
                    return 4
            if not a.dry_run:
                open(os.path.join(case, 'namelist.input'), 'w').write(nl)

        env = dict(os.environ, LD_PRELOAD=SHIM,
                   BLOCKBAR_BUCKET_SEC=str(BUCKET_SEC), BLOCKBAR_NBUCKET=str(NBUCKET),
                   BLOCKBAR_BUCKET_OUT=os.path.join(out, 'bucket_%s.csv' % tag))

        # Follow the hot-spot: the placement for THIS segment comes from the segment just
        # finished. Segment 0 has nothing to learn from, so it runs the default in both arms
        # -- which also means the two arms are identical until the first boundary, and any
        # difference before it is noise, not signal.
        mpi_extra, place_note = [], None
        if a.place == 'follow' and i > 0 and not a.dry_run:
            prev = os.path.join(out, 'bucket_seg%02d.csv' % (i - 1))
            hosts = slurm_hosts()
            if os.path.exists(prev) and hosts:
                try:
                    comp, _bsec = load_bucket_csv(prev)
                    hot = wp.hot_ranks(comp)
                    node = wp.spread_map(a.ranks, len(hosts), hot)
                    mf = wp.write_machinefile(os.path.join(out, 'machinefile_%s' % tag),
                                              node, hosts)
                    mpi_extra = ['-machinefile', mf]
                    place_note = wp.summarise(node, hot, len(hosts))
                    print('    placement from %s: hot%% by node %s, %d ranks moved (%.0f%%)'
                          % (os.path.basename(prev), place_note['hot_by_node_pct'],
                             place_note['ranks_moved_from_default'], place_note['moved_pct']))
                except Exception as e:
                    print('    placement SKIPPED (%s) -- running default, arm is degraded'
                          % e)
        t_a = dt.datetime.now()
        if a.dry_run:
            print('    DRY: mpirun -np %d ./wrf.exe   (shim -> bucket_%s.csv)' % (a.ranks, tag))
            rc = 0
        else:
            with open(os.path.join(case, 'wrf_%s.out' % tag), 'w') as fh:
                rc = subprocess.call(['mpirun'] + mpi_extra + ['-np', str(a.ranks),
                                                                './wrf.exe'],
                                     cwd=case, env=env, stdout=fh, stderr=subprocess.STDOUT)
        wall = (dt.datetime.now() - t_a).total_seconds()

        ok = a.dry_run or wrf_succeeded(case)
        print('    wall=%.1f s  success=%s' % (wall, ok))
        if not ok:
            print('    STOPPING: segment did not complete; read-only diagnosis only')
            manifest['segments'].append({'tag': tag, 'state': 'FAILED', 'wall_s': wall})
            break
        if not a.dry_run:
            shutil.copyfile(os.path.join(case, 'rsl.out.0000'),
                            os.path.join(out, 'rsl_%s.txt' % tag))

        rec = {'tag': tag, 't0': str(t0), 'px': px, 'py': py,
               'wall_s': round(wall, 1), 'state': 'OK'}
        if horizon_s is None:
            horizon_s = wall          # the next boundary is one segment away

        bpath = os.path.join(out, 'bucket_%s.csv' % tag)
        if not a.dry_run and os.path.exists(bpath):
            comp, bsec = load_bucket_csv(bpath)
            g = decay_gate(comp, bsec, horizon_s, MODEL_HIT_PCT, DEFAULT_WINDOW_S)
            rec['gate'] = g
            print('    gate: %s  (%s)' % (g['source'], g['reason']))
            if g.get('tau_s'):
                print('    decay tau=%.0f s   identity@%.0fs=%.1f%%'
                      % (g['tau_s'], horizon_s, g['identity_pct'] or -1))
            tot = comp.sum(axis=1)
            if tot.sum() > 0:
                rec['imbalance'] = {'max_s': round(float(tot.max()), 1),
                                    'mean_s': round(float(tot.mean()), 1),
                                    'recoverable_s': round(float(tot.max() - tot.mean()), 1)}
                print('    per-rank: max=%.1f mean=%.1f  recoverable=%.1f s'
                      % (tot.max(), tot.mean(), tot.max() - tot.mean()))

        manifest['segments'].append(rec)
        if not a.dry_run:
            json.dump(manifest, open(os.path.join(out, 'segment_manifest.json'), 'w'), indent=2)

        # The action. Deliberately NOT driven by the load estimate -- see choose_decomp.
        if not a.fixed:
            px, py = choose_decomp(a.ranks, nx, ny)
        print()

    print('manifest -> %s' % os.path.join(out, 'segment_manifest.json'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
