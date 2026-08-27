"""Stage C: empirical probe harness -- short WRF runs + ABBA to pick a config.

The analytic model (decomposition cost, hotspot, placement) only NARROWS. This
stage MEASURES: submit short WRF runs for candidate configs, parse median s/step,
gate on mechanism, and compare close candidates with ABBA (paired, same
allocation) because the cross-allocation variance is ~2.5% and real wins are thin.

Objective is MEASURED wall/step (NOT the imbalance-tax proxy epsilon, which was
shown to disagree with wall ordering). This harness only ORCHESTRATES sbatch; it
never interprets a completion message as truth -- the caller inspects rsl.

Running WRF consumes the cluster: submission requires explicit user consent per
the execution-card protocol. This module builds/parses; a separate approved step
submits.

stdlib only. Python 3.6.
"""
import json
import re
import subprocess
from pathlib import Path

RUN_SBATCH = '/home/u5727520/wrf-stack/g0a/run-wrf-decomp.sbatch'


def make_config(name, ranks, nproc_x, nproc_y, io_form=14, omp=1,
                placement_file=None, extra_env=None):
    """A probe config. placement_file = a hotspot rank->core map (over-sub)."""
    return {
        'name': name,
        'ranks': ranks,
        'nproc_x': nproc_x,
        'nproc_y': nproc_y,
        'io_form': io_form,
        'omp': omp,
        'placement_file': placement_file,
        'extra_env': extra_env or {},
    }


def submit(cfg, steps=60, nodes=4, partition='cf448', account='ACD110018',
           timelimit='00:12:00', run_sbatch=RUN_SBATCH, dry_run=True):
    """Build the sbatch env + (optionally) submit. Returns (jobid|None, cmd).

    dry_run=True returns the command WITHOUT submitting (for review / consent).
    """
    env = {
        'RUN_MODE': 'restart', 'COMPILER_FAMILY': 'intel', 'COMPILER_TAG': 'ifort',
        'BUILD': 'intel-ifortzmm-adios2', 'MODE': 'dmpar',
        'RANKS': str(cfg['ranks']), 'NPROC_X': str(cfg['nproc_x']),
        'NPROC_Y': str(cfg['nproc_y']), 'IO_FORM_RESTART': str(cfg['io_form']),
        'OMP_NUM_THREADS': str(cfg['omp']),
        'SIM_MIN': '15',
        'WRF_STACK_ROOT': '/home/u5727520/wrf-stack',
        'BENCH': '/home/u5727520/wrf-bench/v4.4_bench_conus2.5km',
    }
    if cfg.get('placement_file'):
        env['PLACEMENT_FILE'] = cfg['placement_file']   # run script converts to pinning
    env.update(cfg.get('extra_env', {}))
    envstr = ','.join('{}={}'.format(k, v) for k, v in env.items())

    ranks_per_node = max(1, cfg['ranks'] // nodes)
    sb = ['sbatch', '--parsable',
          '--nodes={}'.format(nodes),
          '--ntasks-per-node={}'.format(ranks_per_node),
          '--exclusive', '-p', partition, '-A', account, '-t', timelimit,
          '--job-name={}'.format(cfg['name']),
          '--export=ALL,{}'.format(envstr), run_sbatch]
    if dry_run:
        return None, sb
    try:
        r = subprocess.run(sb, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=60)
        jid = r.stdout.decode().strip().split(';')[0]
        return (jid if jid.isdigit() else None), sb
    except Exception as exc:
        return None, sb + ['# ERROR: {}'.format(exc)]


def parse_rsl(rsl_path):
    """Median s/step + mechanism gate from an rsl.error.0000. Returns dict."""
    p = Path(rsl_path)
    if not p.exists():
        return {'ok': False, 'reason': 'rsl missing'}
    text = p.read_text(errors='replace')
    times = [float(m) for m in re.findall(r'Timing for main[^\n]*?([0-9.]+) elapsed', text)]
    bad = re.search(r'severe|SIGSEGV|forrtl|FATAL|NaN|cfl', text, re.I)
    # mechanism: rsl reports "Ntasks in X <nx> , in Y <ny>"
    mesh = re.search(r'Ntasks in X\s+(\d+).*?in Y\s+(\d+)', text)
    if not times:
        return {'ok': False, 'reason': 'no timing lines'}
    s = sorted(times)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return {
        'ok': bad is None,
        'reason': ('mechanism error: ' + bad.group(0)) if bad else 'ok',
        'median_s': round(median, 5),
        'nsteps': n,
        'mesh': (int(mesh.group(1)), int(mesh.group(2))) if mesh else None,
    }


def abba_order(config_names, reps):
    """Yield an ABBA(-BAAB...) interleaving for `reps` paired rounds of 2 configs."""
    assert len(config_names) == 2, 'ABBA compares exactly 2 configs'
    a, b = config_names
    seq = []
    for r in range(reps):
        seq += ([a, b] if r % 2 == 0 else [b, a])
    return seq


def compare(results_a, results_b):
    """Paired comparison of two configs' median s/step lists. Returns verdict."""
    import statistics
    if not results_a or not results_b:
        return {'verdict': 'insufficient data'}
    ma = statistics.median(results_a)
    mb = statistics.median(results_b)
    diff_pct = (mb / ma - 1.0) * 100.0
    spread = (max(results_a + results_b) / min(results_a + results_b) - 1.0) * 100.0
    within_noise = abs(diff_pct) < max(2.5, spread)
    return {
        'median_a': round(ma, 5), 'median_b': round(mb, 5),
        'b_vs_a_pct': round(diff_pct, 2),
        'within_variance': within_noise,
        'verdict': ('within noise (~2.5%); not distinguishable'
                    if within_noise else
                    ('B faster' if diff_pct < 0 else 'A faster')),
    }
