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


def parse_rsl(rsl_path, require_success=True):
    """Parse an rsl.error.0000 into SEPARATED scoring objectives + an evidence gate.

    Objectives are kept distinct so a comparison must DECLARE which it judges on (a
    measured CONUS result showed io_form_history=14's win is in total wall, not the
    solver kernel -- median s/step alone mis-attributes it):
      solver_{median,mean,total}_s  -- 'Timing for main' step times (the physics/dynamics kernel)
      io_total_s / io_writes        -- 'Timing for Writing' (history/restart) time
      total_wall_lb_s               -- solver_total + io_total, a LOWER BOUND (excludes init/
                                       restart read); the harness measures the TRUE wall separately
      steps_completed               -- number of integrated steps

    status is one of: 'ok' | 'mechanism error' | 'NEED-EVIDENCE'. A run with no timing,
    or that did not reach SUCCESS COMPLETE (truncated/failed), returns NEED-EVIDENCE and
    MUST NOT be silently folded into a comparison. (median_s is retained = solver_median_s
    for back-compat.)"""
    p = Path(rsl_path)
    if not p.exists():
        return {'status': 'NEED-EVIDENCE', 'ok': False, 'reason': 'rsl missing'}
    text = p.read_text(errors='replace')
    step_times = [float(m) for m in re.findall(r'Timing for main[^\n]*?([0-9.]+) elapsed', text)]
    write_times = [float(m) for m in re.findall(r'Timing for Writing[^\n]*?([0-9.]+) elapsed', text)]
    bad = re.search(r'severe|SIGSEGV|forrtl|FATAL|\bNaN\b|cfl', text, re.I)
    mesh = re.search(r'Ntasks in X\s+(\d+).*?in Y\s+(\d+)', text)
    success = re.search(r'SUCCESS COMPLETE', text)
    if bad:
        return {'status': 'mechanism error', 'ok': False, 'reason': bad.group(0),
                'steps_completed': len(step_times)}
    if not step_times:
        return {'status': 'NEED-EVIDENCE', 'ok': False, 'reason': 'no Timing-for-main lines'}
    if require_success and not success:
        return {'status': 'NEED-EVIDENCE', 'ok': False,
                'reason': 'no SUCCESS COMPLETE (truncated/failed) -- not a valid measurement',
                'steps_completed': len(step_times)}
    s = sorted(step_times)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    solver_total = sum(step_times)
    io_total = sum(write_times)
    return {
        'status': 'ok', 'ok': True, 'reason': 'ok',
        'solver_median_s': round(median, 5),
        'solver_mean_s': round(solver_total / n, 5),
        'solver_total_s': round(solver_total, 3),
        'io_total_s': round(io_total, 3),
        'io_writes': len(write_times),
        'total_wall_lb_s': round(solver_total + io_total, 3),
        'steps_completed': n,
        'mesh': (int(mesh.group(1)), int(mesh.group(2))) if mesh else None,
        'median_s': round(median, 5),   # back-compat alias = solver_median_s
    }


def abba_order(config_names, reps):
    """Yield an ABBA(-BAAB...) interleaving for `reps` paired rounds of 2 configs."""
    assert len(config_names) == 2, 'ABBA compares exactly 2 configs'
    a, b = config_names
    seq = []
    for r in range(reps):
        seq += ([a, b] if r % 2 == 0 else [b, a])
    return seq


def compare(results_a, results_b, objective='solver_median_s', reps=None):
    """Paired comparison of two configs on a DECLARED objective (pass the per-rep values
    for that objective, e.g. solver_median_s OR total_wall_lb_s -- never mix objectives).
    Cross-allocation variance is ~2.5%, so a win must clear that AND the observed spread;
    otherwise the verdict is 'within noise', never a forced ranking. n<3 reps -> the result
    is flagged UNDER-REPLICATED (the protocol wants n=3 paired rounds)."""
    import statistics
    if not results_a or not results_b:
        return {'verdict': 'NEED-EVIDENCE', 'reason': 'missing measurements for one arm', 'objective': objective}
    ma = statistics.median(results_a)
    mb = statistics.median(results_b)
    diff_pct = (mb / ma - 1.0) * 100.0
    spread = (max(results_a + results_b) / min(results_a + results_b) - 1.0) * 100.0
    within_noise = abs(diff_pct) < max(2.5, spread)
    n = reps if reps is not None else min(len(results_a), len(results_b))
    out = {
        'objective': objective,
        'median_a': round(ma, 5), 'median_b': round(mb, 5),
        'b_vs_a_pct': round(diff_pct, 2),
        'within_variance': within_noise,
        'reps': n,
        'under_replicated': n < 3,
        'verdict': ('within noise (~2.5%); not distinguishable'
                    if within_noise else
                    ('B faster' if diff_pct < 0 else 'A faster')),
    }
    if n < 3 and not within_noise:
        out['verdict'] += ' [UNDER-REPLICATED n<3: not a released winner]'
    return out
