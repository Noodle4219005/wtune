"""Autotuner step 2: external WRF phase fingerprinter.

Turns node-local `sampler_node.sh` output (perf.<host>.txt.gz + meta.<host>.txt)
into a per-case fingerprint.json plus diagnosis labels, by time-weighted
attribution of `perf script` samples to a WRF phase (phase_classify.classify).
This module never runs perf itself and never touches wrf.exe -- it only reads
files sampler_node.sh already wrote.

A validation mode (--wtimer-dir) compares the fingerprint's pooled phase
shares against in-process wtimer.<rank>.csv timers from a wrf_wtimer.exe run,
used only as ground truth for this one comparison.

A preflight mode (--preflight-perf-script) parses older-format `perf script
-F tid,time,period,event,ip,sym,dso` output (no pid/cpu columns) so the
classifier can be sanity-checked against perf data collected before this
sampler existed.

stdlib only. Python 3.6.
"""
import argparse
import csv
import glob
import gzip
import json
import math
import os
import re
import statistics
import sys

from phase_classify import classify

# --- tunables, echoed into the JSON so a reader does not have to guess them --

DT_CAP = 1.0  # seconds; caps the inter-sample gap attributed to one sample

MICRO_HEAVY_THRESHOLD = 0.10
RADIATION_HEAVY_THRESHOLD = 0.10
HALO_DOMINATED_THRESHOLD = 0.25
IO_HEAVY_THRESHOLD = 0.10
SPATIAL_IMBALANCE_RATIO_THRESHOLD = 1.5
SPATIAL_IMBALANCE_SPEARMAN_THRESHOLD = 0.8
OMP_TILE_BUSY_RATIO_THRESHOLD = 1.10
PREDICTABLE_PHASE_SPEARMAN_THRESHOLD = 0.9
VALIDATION_PASS_PP = 3.0
HOT_FRACTION_RATIO = 1.25
UNPARSED_FRACTION_LIMIT = 0.01

# phases attributed to actual wall time; INHERIT is always resolved away
# before it reaches any of the aggregation code below.
PHASES_TRACKED = ("MICRO", "RADIATION", "PBL", "SURFACE", "CUMULUS",
                   "DYNAMICS_OTHER", "HALO", "MPI", "IO")

# One sample line from sampler_node.sh's `perf script -F pid,tid,time,ip,sym,dso`:
#   "  1234/1235 12345.678901:  ffffffffaabbccdd sym_name+0x1a (/path/to/dso)"
# The "[cpu]" column is optional: perf only stores it with `perf record --sample-cpu`, and asking
# `perf script` for it without that makes perf print nothing at all (job 1074372).
# Tolerant of extra whitespace, a missing +0x offset, and sym == [unknown].
PERF_LINE_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+(?:\[\s*(\d+)\s*\]\s+)?([0-9]+\.[0-9]+):\s+"
    r"([0-9a-fA-Fx]+)\s+(.+?)\s+\(([^()]*)\)\s*$"
)

# g1_aggregate.py's line format, for --preflight-perf-script:
#   "perf script -F tid,time,period,event,ip,sym,dso" (no pid/cpu; possibly
# multiple events interleaved).
OLD_LINE_RE = re.compile(r"^\s*(\d+)\s+([\d.]+):\s+(\d+)\s+(\S+?):?\s+(.*)$")


# --------------------------------------------------------------------------
# small stats helpers (Python 3.6 has no statistics.quantiles)
# --------------------------------------------------------------------------

def percentile(values, pct):
    """Linear-interpolation percentile, like numpy's default."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[int(f)] * (c - k) + s[int(c)] * (k - f)


def lag1_autocorr(values):
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    num = sum((values[i] - mean) * (values[i + 1] - mean) for i in range(n - 1))
    den = sum((v - mean) ** 2 for v in values)
    if den == 0:
        return None
    return num / den


def _rank_data(values):
    """1-based ranks with average rank for ties (Spearman needs this)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rank_data(xs), _rank_data(ys))


# --------------------------------------------------------------------------
# meta.<host>.txt parsing
# --------------------------------------------------------------------------

def _parse_kv_line(line):
    parts = line.split()
    if not parts:
        return None, {}
    kv = {}
    for tok in parts[1:]:
        if '=' in tok:
            k, _, v = tok.partition('=')
            kv[k] = v
    return parts[0], kv


def load_meta(sampler_dir):
    """Returns (pid_info, host_window, hosts).

    pid_info: (host, pid) -> {'rank': int|None, 'threads': int, 'home': str}
    host_window: host -> {'start_step':, 'start_epoch':, 'stop_step':, 'stop_epoch':}
    """
    pid_info = {}
    host_window = {}
    hosts = set()
    for path in sorted(glob.glob(os.path.join(sampler_dir, 'meta.*.txt'))):
        with open(path, errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tag, kv = _parse_kv_line(line)
                if tag is None:
                    continue
                host = kv.get('host')
                if host:
                    hosts.add(host)
                if tag == 'PID' and 'pid' in kv:
                    pid = int(kv['pid'])
                    rank = kv.get('rank', '')
                    pid_info[(host, pid)] = {
                        'rank': int(rank) if rank.strip() not in ('', 'None') else None,
                        'threads': int(kv.get('threads', 0) or 0),
                        'home': kv.get('home', ''),
                    }
                elif tag == 'START' and host:
                    w = host_window.setdefault(host, {})
                    w['start_step'] = int(kv.get('step', 0) or 0)
                    w['start_epoch'] = float(kv.get('epoch', 0.0) or 0.0)
                elif tag == 'STOP' and host:
                    w = host_window.setdefault(host, {})
                    w['stop_step'] = int(kv.get('step', 0) or 0)
                    w['stop_epoch'] = float(kv.get('epoch', 0.0) or 0.0)
    return pid_info, host_window, sorted(hosts)


# --------------------------------------------------------------------------
# perf.<host>.txt.gz parsing
# --------------------------------------------------------------------------

def _host_from_perf_path(path):
    base = os.path.basename(path)
    if base.startswith('perf.') and base.endswith('.txt.gz'):
        return base[len('perf.'):-len('.txt.gz')]
    return base


def load_samples(sampler_dir, pid_info):
    """Returns (samples_by_stream, n_parsed, n_unparsed, n_ignored_pid, n_files).

    samples_by_stream: (host, pid, tid) -> list of (time, sym, dso)
    """
    samples_by_stream = {}
    n_parsed = 0
    n_unparsed = 0
    n_ignored_pid = 0
    n_files = 0
    for path in sorted(glob.glob(os.path.join(sampler_dir, 'perf.*.txt.gz'))):
        n_files += 1
        host = _host_from_perf_path(path)
        with gzip.open(path, 'rt', errors='replace') as fh:
            for line in fh:
                if not line.strip():
                    continue
                m = PERF_LINE_RE.match(line)
                if not m:
                    n_unparsed += 1
                    continue
                n_parsed += 1
                pid_s, tid_s, _cpu_s, t_s, _ip, sym, dso = m.groups()
                pid = int(pid_s)
                tid = int(tid_s)
                if (host, pid) not in pid_info:
                    n_ignored_pid += 1
                    continue
                key = (host, pid, tid)
                samples_by_stream.setdefault(key, []).append(
                    (float(t_s), sym.strip(), dso.strip()))
    return samples_by_stream, n_parsed, n_unparsed, n_ignored_pid, n_files


def estimate_freq_hz(samples_by_stream):
    """Sampling frequency is not recorded anywhere in meta.<host>.txt, so it
    is estimated from the data: 1 / median raw (uncapped) inter-sample gap,
    pooled over every stream."""
    dts = []
    for samples in samples_by_stream.values():
        times = sorted(t for t, _, _ in samples)
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            if dt > 0:
                dts.append(dt)
    if not dts:
        return None
    med = statistics.median(dts)
    return (1.0 / med) if med > 0 else None


# --------------------------------------------------------------------------
# time-weighted attribution
# --------------------------------------------------------------------------

def attribute_stream(samples, dt_cap=DT_CAP):
    """samples: iterable of (time, sym, dso), any order.

    Returns a list of (time, resolved_phase, dt) sorted by time. dt for
    samples[i] is min(t[i+1]-t[i], dt_cap); the last sample in the stream
    gets dt = median(all other dts) (0.0 if there is nothing to take a
    median of). INHERIT resolves to the stream's last non-INHERIT resolved
    phase, or DYNAMICS_OTHER if none has been seen yet.
    """
    ordered = sorted(samples, key=lambda s: s[0])
    n = len(ordered)
    resolved = []
    last_phase = None
    for t, sym, dso in ordered:
        phase = classify(sym, dso)
        if phase == 'INHERIT':
            phase = last_phase if last_phase is not None else 'DYNAMICS_OTHER'
        last_phase = phase
        resolved.append((t, phase))

    dts = []
    for i in range(n - 1):
        dt = resolved[i + 1][0] - resolved[i][0]
        if dt < 0:
            dt = 0.0
        dts.append(min(dt, dt_cap))

    out = []
    if n > 0:
        last_dt = statistics.median(dts) if dts else 0.0
        for i in range(n):
            t, phase = resolved[i]
            dt = dts[i] if i < n - 1 else last_dt
            out.append((t, phase, dt))
    return out


def segment_micro_calls(attributed):
    """attributed: (time, phase, dt) list in time order, master thread only.

    A maximal run of consecutive MICRO samples is one micro call; its
    duration is the sum of the dts in the run. Returns a list of durations
    (seconds), in call order.
    """
    calls = []
    cur = 0.0
    in_run = False
    for _t, phase, dt in attributed:
        if phase == 'MICRO':
            cur += dt
            in_run = True
        else:
            if in_run:
                calls.append(cur)
            cur = 0.0
            in_run = False
    if in_run:
        calls.append(cur)
    return calls


def build_rank_data(samples_by_stream, pid_info, dt_cap=DT_CAP):
    """Returns rank -> {'host', 'phase_time' (dict), 'total', 'micro_calls',
    'master_micro_samples', 'worker_micro_samples'}.

    The master thread of a rank is the tid equal to its pid; everything else
    on that pid is a worker thread. Phase time/total/micro_calls come only
    from the master stream (that is what wtimer measures too); worker
    threads contribute only their MICRO on-CPU sample count, for the
    thread-imbalance metric.
    """
    ranks = {}
    for (host, pid, tid), samples in samples_by_stream.items():
        info = pid_info.get((host, pid))
        if info is None or info.get('rank') is None:
            continue
        rank = info['rank']
        attributed = attribute_stream(samples, dt_cap=dt_cap)
        rd = ranks.setdefault(rank, {
            'host': host,
            'phase_time': dict.fromkeys(PHASES_TRACKED, 0.0),
            'total': 0.0,
            'micro_calls': [],
            'master_micro_samples': 0,
            'worker_micro_samples': 0,
        })
        if tid == pid:
            for _t, phase, dt in attributed:
                rd['phase_time'][phase] += dt
                rd['total'] += dt
            rd['micro_calls'] = segment_micro_calls(attributed)
            rd['master_micro_samples'] = sum(1 for _t, phase, _dt in attributed
                                              if phase == 'MICRO')
        else:
            rd['worker_micro_samples'] += sum(1 for _t, phase, _dt in attributed
                                               if phase == 'MICRO')
    return ranks


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def pooled_phase_share(ranks):
    totals = dict.fromkeys(PHASES_TRACKED, 0.0)
    grand_total = 0.0
    for rd in ranks.values():
        grand_total += rd['total']
        for phase in PHASES_TRACKED:
            totals[phase] += rd['phase_time'][phase]
    if grand_total <= 0:
        return dict.fromkeys(PHASES_TRACKED, 0.0)
    return dict((phase, totals[phase] / grand_total) for phase in PHASES_TRACKED)


def compute_rank_imbalance(ranks):
    micro_times = [rd['phase_time']['MICRO'] for rd in ranks.values()]
    if not micro_times:
        return {'max_mean': None, 'p90_p50': None, 'cv': None}
    mean = sum(micro_times) / len(micro_times)
    p50 = percentile(micro_times, 50)
    p90 = percentile(micro_times, 90)
    return {
        'max_mean': (max(micro_times) / mean) if mean > 0 else None,
        'p90_p50': (p90 / p50) if p50 else None,
        'cv': (statistics.pstdev(micro_times) / mean) if mean > 0 else 0.0,
    }


def compute_thread_imbalance(ranks, numtiles, omp_threads):
    ratios = [rd['master_micro_samples'] / float(max(rd['worker_micro_samples'], 1))
              for rd in ranks.values()]
    tiles_divisible = None
    if numtiles and omp_threads:
        tiles_divisible = (numtiles % omp_threads == 0)
    return {
        'busy_ratio_median': statistics.median(ratios) if ratios else None,
        'numtiles': numtiles,
        'omp_threads': omp_threads,
        'tiles_divisible': tiles_divisible,
    }


def compute_mpi_floor(ranks):
    shares = []
    for rd in ranks.values():
        shares.append(rd['phase_time']['MPI'] / rd['total'] if rd['total'] > 0 else 0.0)
    if not shares:
        return {'min': None, 'p10': None, 'median': None}
    return {'min': min(shares), 'p10': percentile(shares, 10), 'median': percentile(shares, 50)}


def compute_phase_predictability(ranks):
    autocorrs = []
    first_means = []
    second_means = []
    for rank in sorted(ranks):
        calls = ranks[rank]['micro_calls']
        if len(calls) >= 5:
            ac = lag1_autocorr(calls)
            if ac is not None:
                autocorrs.append(ac)
        if len(calls) >= 2:
            mid = len(calls) // 2
            first = calls[:mid]
            second = calls[mid:]
            if first and second:
                first_means.append(sum(first) / len(first))
                second_means.append(sum(second) / len(second))
    return {
        'autocorr_median': statistics.median(autocorrs) if autocorrs else None,
        'half_window_spearman': spearman(first_means, second_means),
    }


def compute_spatial(ranks, nproc_x, nproc_y):
    out = {'nproc_x': nproc_x, 'nproc_y': nproc_y, 'row_mean_micro_s': {},
           'col_mean_micro_s': {}, 'hot_fraction': None}
    if not nproc_x or not ranks:
        return out
    micro_by_rank = dict((r, rd['phase_time']['MICRO']) for r, rd in ranks.items())
    mean_micro = sum(micro_by_rank.values()) / len(micro_by_rank)
    row_vals = {}
    col_vals = {}
    for r, micro in micro_by_rank.items():
        i = r % nproc_x
        j = r // nproc_x
        row_vals.setdefault(j, []).append(micro)
        col_vals.setdefault(i, []).append(micro)
    out['row_mean_micro_s'] = dict((j, sum(v) / len(v)) for j, v in row_vals.items())
    out['col_mean_micro_s'] = dict((i, sum(v) / len(v)) for i, v in col_vals.items())
    if mean_micro > 0:
        hot = sum(1 for m in micro_by_rank.values() if m > HOT_FRACTION_RATIO * mean_micro)
        out['hot_fraction'] = hot / len(micro_by_rank)
    else:
        out['hot_fraction'] = 0.0
    return out


def compute_diagnosis(phase_share, rank_imbalance, thread_imbalance, phase_predictability):
    labels = []

    def add(label, value, threshold):
        labels.append({'label': label, 'value': value, 'threshold': threshold})

    if phase_share.get('MICRO', 0.0) >= MICRO_HEAVY_THRESHOLD:
        add('MICRO_HEAVY', phase_share['MICRO'], MICRO_HEAVY_THRESHOLD)
    if phase_share.get('RADIATION', 0.0) >= RADIATION_HEAVY_THRESHOLD:
        add('RADIATION_HEAVY', phase_share['RADIATION'], RADIATION_HEAVY_THRESHOLD)
    if phase_share.get('MPI', 0.0) >= HALO_DOMINATED_THRESHOLD:
        add('HALO_DOMINATED', phase_share['MPI'], HALO_DOMINATED_THRESHOLD)
    if phase_share.get('IO', 0.0) >= IO_HEAVY_THRESHOLD:
        add('IO_HEAVY', phase_share['IO'], IO_HEAVY_THRESHOLD)

    max_mean = rank_imbalance.get('max_mean')
    half_spear = phase_predictability.get('half_window_spearman')
    if max_mean is not None and half_spear is not None and \
            max_mean >= SPATIAL_IMBALANCE_RATIO_THRESHOLD and \
            half_spear >= SPATIAL_IMBALANCE_SPEARMAN_THRESHOLD:
        add('SPATIALLY_IMBALANCED',
            {'max_mean': max_mean, 'half_window_spearman': half_spear},
            {'max_mean': SPATIAL_IMBALANCE_RATIO_THRESHOLD,
             'half_window_spearman': SPATIAL_IMBALANCE_SPEARMAN_THRESHOLD})

    tiles_divisible = thread_imbalance.get('tiles_divisible')
    busy_ratio = thread_imbalance.get('busy_ratio_median')
    if tiles_divisible is False or (busy_ratio is not None and
                                     busy_ratio >= OMP_TILE_BUSY_RATIO_THRESHOLD):
        add('OMP_TILE_IMBALANCED',
            {'numtiles': thread_imbalance.get('numtiles'),
             'omp_threads': thread_imbalance.get('omp_threads'),
             'busy_ratio_median': busy_ratio},
            OMP_TILE_BUSY_RATIO_THRESHOLD)

    if half_spear is not None and half_spear >= PREDICTABLE_PHASE_SPEARMAN_THRESHOLD:
        add('PREDICTABLE_PHASE_COST', half_spear, PREDICTABLE_PHASE_SPEARMAN_THRESHOLD)

    return labels


# --------------------------------------------------------------------------
# namelist / rsl parsing
# --------------------------------------------------------------------------

def parse_namelist(run_dir):
    path = os.path.join(run_dir, 'namelist.input')
    nproc_x = nproc_y = numtiles = None
    try:
        with open(path, errors='replace') as fh:
            text = fh.read()
    except (IOError, OSError):
        return None, None, None
    m = re.search(r'nproc_x\s*=\s*(\d+)', text)
    if m:
        nproc_x = int(m.group(1))
    m = re.search(r'nproc_y\s*=\s*(\d+)', text)
    if m:
        nproc_y = int(m.group(1))
    m = re.search(r'numtiles\s*=\s*(\d+)', text)
    if m:
        numtiles = int(m.group(1))
    return nproc_x, nproc_y, numtiles


def parse_rsl_steps(run_dir):
    """Returns the 'Timing for main' elapsed seconds, in step order (field 9,
    i.e. index 8 after whitespace split)."""
    path = os.path.join(run_dir, 'rsl.out.0000')
    steps = []
    try:
        with open(path, errors='replace') as fh:
            for line in fh:
                if 'Timing for main' not in line:
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    try:
                        steps.append(float(parts[8]))
                    except ValueError:
                        pass
    except (IOError, OSError):
        pass
    return steps


# --------------------------------------------------------------------------
# wtimer ground truth / validation
# --------------------------------------------------------------------------

def parse_wtimer_csv(path):
    """call_index -> dict of the 9 wtimer fields. '#'-prefixed header/footer
    lines are skipped."""
    rows = {}
    with open(path, errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split(',')
            if len(fields) < 9:
                continue
            try:
                call_index = int(fields[0])
                rows[call_index] = {
                    'micro_s': float(fields[1]),
                    'radiation_s': float(fields[2]),
                    'pre_radiation_s': float(fields[3]),
                    'pbl_s': float(fields[4]),
                    'surface_s': float(fields[5]),
                    'cumulus_s': float(fields[6]),
                    'mpi_wait_s': float(fields[7]),
                    'mpi_wait_calls': int(float(fields[8])),
                }
            except ValueError:
                continue
    return rows


_WTIMER_RANK_RE = re.compile(r'wtimer\.(\d+)\.csv$')


def run_validation(wtimer_dir, run_dir, host_window, ranks, phase_share):
    """Ground truth from wtimer.<rank>.csv + rsl.out.0000, compared against
    this fingerprint's pooled phase_share. Returns None if there is not
    enough information (no host window, no wtimer files)."""
    if not host_window:
        return None
    starts = [w['start_step'] for w in host_window.values() if 'start_step' in w]
    stops = [w['stop_step'] for w in host_window.values() if 'stop_step' in w]
    if not starts or not stops:
        return None
    s_step = min(starts)
    e_step = max(stops)
    if e_step <= s_step:
        return None

    rsl_steps = parse_rsl_steps(run_dir)
    window_rsl = rsl_steps[s_step:e_step]
    total_per_rank = sum(window_rsl)

    gt_ranks = {}
    for path in sorted(glob.glob(os.path.join(wtimer_dir, 'wtimer.*.csv'))):
        m = _WTIMER_RANK_RE.search(os.path.basename(path))
        if not m:
            continue
        rank = int(m.group(1))
        rows = parse_wtimer_csv(path)
        micro = radiation = pbl = surface = cumulus = mpi = 0.0
        for idx in range(s_step, e_step):
            row = rows.get(idx)
            if row is None:
                continue
            micro += row['micro_s']
            radiation += row['radiation_s'] + row['pre_radiation_s']
            pbl += row['pbl_s']
            surface += row['surface_s']
            cumulus += row['cumulus_s']
            mpi += row['mpi_wait_s']
        dynamics_other = max(0.0, total_per_rank - micro - radiation - pbl -
                              surface - cumulus - mpi)
        gt_ranks[rank] = {
            'micro': micro, 'radiation': radiation, 'pbl': pbl, 'surface': surface,
            'cumulus': cumulus, 'mpi': mpi, 'dynamics_other': dynamics_other,
            'total': total_per_rank,
        }
    if not gt_ranks:
        return None

    grand_total = total_per_rank * len(gt_ranks)
    gt_pooled = {}
    for key in ('micro', 'radiation', 'dynamics_other', 'mpi', 'pbl', 'surface', 'cumulus'):
        s = sum(g[key] for g in gt_ranks.values())
        gt_pooled[key] = (s / grand_total) if grand_total > 0 else 0.0

    fp_pooled = {
        'micro': phase_share.get('MICRO', 0.0),
        'radiation': phase_share.get('RADIATION', 0.0),
        # wtimer only times MPI waits, so RSL halo pack/unpack (HALO) sits in its dynamics_other remainder
        'dynamics_other': phase_share.get('DYNAMICS_OTHER', 0.0) + phase_share.get('HALO', 0.0),
        'mpi': phase_share.get('MPI', 0.0),
        'pbl': phase_share.get('PBL', 0.0),
        'surface': phase_share.get('SURFACE', 0.0),
        'cumulus': phase_share.get('CUMULUS', 0.0),
    }
    diff_pp = dict((k, abs(fp_pooled[k] - gt_pooled[k]) * 100.0) for k in fp_pooled)

    per_rank_err = dict((k, []) for k in ('micro', 'radiation', 'dynamics_other', 'mpi'))
    for rank, g in gt_ranks.items():
        rd = ranks.get(rank)
        if rd is None or rd['total'] <= 0 or g['total'] <= 0:
            continue
        fp_share = {
            'micro': rd['phase_time']['MICRO'] / rd['total'],
            'radiation': rd['phase_time']['RADIATION'] / rd['total'],
            'dynamics_other': (rd['phase_time']['DYNAMICS_OTHER'] + rd['phase_time']['HALO']) / rd['total'],
            'mpi': rd['phase_time']['MPI'] / rd['total'],
        }
        gt_share = {
            'micro': g['micro'] / g['total'],
            'radiation': g['radiation'] / g['total'],
            'dynamics_other': g['dynamics_other'] / g['total'],
            'mpi': g['mpi'] / g['total'],
        }
        for k in per_rank_err:
            per_rank_err[k].append(abs(fp_share[k] - gt_share[k]) * 100.0)

    per_rank_summary = {}
    for k, vals in per_rank_err.items():
        per_rank_summary[k] = {
            'p50': percentile(vals, 50) if vals else None,
            'p90': percentile(vals, 90) if vals else None,
        }

    passed = all(diff_pp[k] <= VALIDATION_PASS_PP
                 for k in ('micro', 'radiation', 'dynamics_other', 'mpi'))
    return {
        'window': {'start_step': s_step, 'stop_step': e_step},
        'gt_share': gt_pooled,
        'fp_share': fp_pooled,
        'diff_pp': diff_pp,
        'per_rank_abs_error_pp': per_rank_summary,
        'pass': passed,
    }


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def write_outputs(outprefix, out, per_rank_list, ranks):
    with open(outprefix + '.json', 'w') as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    with open(outprefix + '_per_rank.csv', 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['rank', 'host'] + list(PHASES_TRACKED) +
                         ['micro_ms_per_call_mean', 'n_micro_calls',
                          'master_micro_samples', 'worker_micro_samples'])
        for row in per_rank_list:
            writer.writerow(
                [row['rank'], row['host']] +
                [row['phase_share'][p] for p in PHASES_TRACKED] +
                [row['micro_ms_per_call_mean'], row['n_micro_calls'],
                 row['master_micro_samples'], row['worker_micro_samples']])

    with open(outprefix + '_micro_calls.csv', 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['rank', 'call_index', 'duration_ms'])
        for rank in sorted(ranks):
            for idx, dur in enumerate(ranks[rank]['micro_calls']):
                writer.writerow([rank, idx, dur * 1000.0])


# --------------------------------------------------------------------------
# preflight mode: classify old-format perf-script text without a sampler dir
# --------------------------------------------------------------------------

def parse_preflight_stream(path):
    """Returns (samples, parsed_lines, total_lines) for one old-format
    `perf script -F tid,time,period,event,ip,sym,dso` file (plain or .gz).
    Only samples whose event name contains 'cycles' but not 'ref-cycles'
    are kept; parsed_lines/total_lines reflect the line-format match rate
    over ALL lines, independent of that event filter."""
    opener = gzip.open if path.endswith('.gz') else open
    samples = []
    total_lines = 0
    parsed_lines = 0
    with opener(path, 'rt', errors='replace') as fh:
        for line in fh:
            if not line.strip():
                continue
            total_lines += 1
            m = OLD_LINE_RE.match(line)
            if not m:
                continue
            parsed_lines += 1
            _tid, t, _period, ev, rest = m.groups()
            ev_l = ev.lower()
            if 'ref-cycles' in ev_l or 'ref_cycles' in ev_l:
                continue
            if 'cycles' not in ev_l:
                continue
            sym, _sep, dso = rest.rpartition(' (')
            sym = sym.strip() or rest.strip()
            dso = dso.rstrip(')')
            head, _sep2, tail = sym.partition(' ')
            if tail and re.fullmatch(r'[0-9a-fA-F]+', head):
                sym = tail.strip()
            samples.append((float(t), sym.strip(), dso.strip()))
    return samples, parsed_lines, total_lines


def run_preflight(paths):
    for path in paths:
        samples, parsed, total = parse_preflight_stream(path)
        attributed = attribute_stream(samples)
        phase_time = dict.fromkeys(PHASES_TRACKED, 0.0)
        phase_count = dict.fromkeys(PHASES_TRACKED, 0)
        total_time = 0.0
        for _t, phase, dt in attributed:
            phase_time[phase] += dt
            phase_count[phase] += 1
            total_time += dt
        rate = (100.0 * parsed / total) if total else 0.0
        print('PREFLIGHT file=%s lines=%d parsed=%d parse_rate=%.2f%%' %
              (path, total, parsed, rate))
        for p in PHASES_TRACKED:
            share = (phase_time[p] / total_time) if total_time > 0 else 0.0
            print('  phase=%s samples=%d share=%.4f' % (p, phase_count[p], share))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description='External WRF phase fingerprinter (autotuner step 2).')
    p.add_argument('--sampler-dir', default=None,
                    help='sampler_node.sh OUTDIR (perf.<host>.txt.gz + meta.<host>.txt)')
    p.add_argument('--run-dir', default=None,
                    help='WRF run directory (namelist.input, rsl.out.0000)')
    p.add_argument('--omp-threads', type=int, default=None)
    p.add_argument('--wtimer-dir', default=None,
                    help='enable validation mode against wtimer.<rank>.csv ground truth')
    p.add_argument('--out', default=None, help='output prefix')
    p.add_argument('--preflight-perf-script', nargs='+', default=None, metavar='FILE',
                    help='classify old-format perf-script text/.gz files, no sampler dir needed')
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.preflight_perf_script:
        return run_preflight(args.preflight_perf_script)

    if not args.sampler_dir or not args.run_dir or not args.out:
        sys.stderr.write(
            'error: --sampler-dir, --run-dir and --out are required (or use '
            '--preflight-perf-script)\n')
        return 2

    pid_info, host_window, hosts = load_meta(args.sampler_dir)
    samples_by_stream, n_parsed, n_unparsed, n_ignored, n_files = \
        load_samples(args.sampler_dir, pid_info)

    total_lines = n_parsed + n_unparsed
    if total_lines > 0 and (n_unparsed / float(total_lines)) > UNPARSED_FRACTION_LIMIT:
        sys.stderr.write(
            'error: unparsed perf-script lines %.2f%% exceed the %.1f%% limit\n' %
            (100.0 * n_unparsed / total_lines, 100.0 * UNPARSED_FRACTION_LIMIT))
        return 1

    ranks = build_rank_data(samples_by_stream, pid_info)
    nproc_x, nproc_y, numtiles = parse_namelist(args.run_dir)
    omp_threads = args.omp_threads
    if omp_threads is None:
        omp_threads = int(os.environ.get('OMP_NUM_THREADS', '1') or '1')

    phase_share = pooled_phase_share(ranks)

    per_rank_list = []
    for rank in sorted(ranks):
        rd = ranks[rank]
        share = dict((phase, (rd['phase_time'][phase] / rd['total']) if rd['total'] > 0 else 0.0)
                     for phase in PHASES_TRACKED)
        calls_ms = [c * 1000.0 for c in rd['micro_calls']]
        per_rank_list.append({
            'rank': rank,
            'host': rd['host'],
            'phase_share': share,
            'micro_ms_per_call_mean': (sum(calls_ms) / len(calls_ms)) if calls_ms else None,
            'n_micro_calls': len(rd['micro_calls']),
            'master_micro_samples': rd['master_micro_samples'],
            'worker_micro_samples': rd['worker_micro_samples'],
        })

    rank_imbalance = compute_rank_imbalance(ranks)
    thread_imbalance = compute_thread_imbalance(ranks, numtiles, omp_threads)
    mpi_floor = compute_mpi_floor(ranks)
    phase_predictability = compute_phase_predictability(ranks)
    spatial = compute_spatial(ranks, nproc_x, nproc_y)
    diagnosis = compute_diagnosis(phase_share, rank_imbalance, thread_imbalance,
                                   phase_predictability)

    validation = None
    if args.wtimer_dir:
        validation = run_validation(args.wtimer_dir, args.run_dir, host_window,
                                     ranks, phase_share)

    meta = {
        'hosts': hosts,
        'nranks': len(ranks),
        'freq_hz_estimated': estimate_freq_hz(samples_by_stream),
        'window_steps': dict((h, host_window.get(h, {})) for h in hosts),
        'n_samples': n_parsed + n_ignored,
        'n_parsed': n_parsed,
        'n_unparsed': n_unparsed,
        'n_ignored_pid': n_ignored,
        'n_perf_files': n_files,
        'omp_threads': omp_threads,
    }

    out = {
        'meta': meta,
        'phase_share': phase_share,
        'per_rank': per_rank_list,
        'rank_imbalance': rank_imbalance,
        'thread_imbalance': thread_imbalance,
        'mpi_floor': mpi_floor,
        'phase_predictability': phase_predictability,
        'spatial': spatial,
        'io_share': phase_share.get('IO', 0.0),
        'cpu_vs_memory': 'not_measured',
        'diagnosis': diagnosis,
        'thresholds': {
            'MICRO_HEAVY': MICRO_HEAVY_THRESHOLD,
            'RADIATION_HEAVY': RADIATION_HEAVY_THRESHOLD,
            'HALO_DOMINATED': HALO_DOMINATED_THRESHOLD,
            'IO_HEAVY': IO_HEAVY_THRESHOLD,
            'SPATIALLY_IMBALANCED_RATIO': SPATIAL_IMBALANCE_RATIO_THRESHOLD,
            'SPATIALLY_IMBALANCED_SPEARMAN': SPATIAL_IMBALANCE_SPEARMAN_THRESHOLD,
            'OMP_TILE_BUSY_RATIO': OMP_TILE_BUSY_RATIO_THRESHOLD,
            'PREDICTABLE_PHASE_SPEARMAN': PREDICTABLE_PHASE_SPEARMAN_THRESHOLD,
            'VALIDATION_PASS_PP': VALIDATION_PASS_PP,
        },
    }
    if validation is not None:
        out['validation'] = validation

    write_outputs(args.out, out, per_rank_list, ranks)

    labels = ','.join(d['label'] for d in diagnosis) if diagnosis else '-'
    print('FINGERPRINT micro=%.4f radiation=%.4f pbl=%.4f surface=%.4f cumulus=%.4f '
          'dynamics_other=%.4f mpi=%.4f io=%.4f labels=%s' % (
              phase_share.get('MICRO', 0.0), phase_share.get('RADIATION', 0.0),
              phase_share.get('PBL', 0.0), phase_share.get('SURFACE', 0.0),
              phase_share.get('CUMULUS', 0.0), phase_share.get('DYNAMICS_OTHER', 0.0),
              phase_share.get('MPI', 0.0), phase_share.get('IO', 0.0), labels))
    if validation is not None:
        print('VALIDATION pass=%s micro_pp=%.2f radiation_pp=%.2f dynamics_pp=%.2f mpi_pp=%.2f' % (
            validation['pass'], validation['diff_pp']['micro'],
            validation['diff_pp']['radiation'], validation['diff_pp']['dynamics_other'],
            validation['diff_pp']['mpi']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
