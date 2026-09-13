#!/usr/bin/env python3
"""Prepare a separate reviewed KF G0 A/B/C job; never build or submit jobs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from plov.prepare_isc21_node_g0 import TABLES, require, write_json
from plov.kf_node_g0_contract import (SUPPORT, arm_environment, candidate_identity,
    check_record, file_record, inputs_identity, launch_argv, validate_arm)

PYTHON = Path('/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/.pixi/envs/default/bin/python')
LIBRARY = Path('/home/u5727520/wrf-stack/intel/lib/libnetcdf.so')
WRF_RUN = Path('/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF/run')


class RunFailure(RuntimeError):
    def __init__(self, returncode, argv):
        self.returncode = returncode
        super().__init__('subprocess failed with exit code %d: %s' %
                         (returncode, ' '.join(shlex.quote(arg) for arg in argv)))


def make_run_plan(candidate, expected_hash, manifest, terminal, inputs, support, output,
                  python=PYTHON, library=LIBRARY):
    require(not output.exists() and not output.is_symlink(), 'run output already exists')
    build = candidate_identity(candidate, expected_hash, manifest, terminal)
    source = inputs_identity(inputs)
    support_records = {name: file_record(support / name) for name in SUPPORT}
    tools = {name: file_record(path) for name, path in {
        'runner': Path(__file__), 'contract': ROOT / 'plov/kf_node_g0_contract.py',
        'prepare': ROOT / 'plov/prepare_isc21_node_g0.py',
        'comparator': ROOT / 'tools/wrf_netcdf_bitwise.py', 'python': python, 'library': library}.items()}
    return {'artifact_type': 'wtune_kf_native_g0_run_plan', 'version': 1, 'status': 'plan-only',
            'build': build, 'inputs': source, 'support': support_records, 'tools': tools,
            'output': str(output.absolute()), 'resources': {'nodes': 1, 'ranks': 56, 'threads': 2,
                'partition': 'ct112', 'account': 'ACD110018', 'wall_minutes': 30, 'memory_gib': 128},
            'arms': {arm: {'argv': launch_argv(tools['python']['path'], tools['runner']['path'], build['candidate']['path']),
                          'controls': {key: value for key, value in arm_environment(arm, output / arm, {}).items()}}
                     for arm in ('A', 'B', 'C')},
            'execution_note': 'Compile/link is a separate prerequisite. No submission, input restoration, or WRF run performed.'}


def sbatch_text(plan_path, output, python=PYTHON):
    # Paths cannot become injected SBATCH directives; shell arguments are independently quoted.
    for path in (plan_path, output, python, Path(__file__)):
        require('\n' not in str(path) and '\r' not in str(path), 'newline in job path')
    argv = [str(python), str(Path(__file__).resolve()), 'execute', '--plan', str(plan_path)]
    return '\n'.join([
        '#!/bin/bash -l', '#SBATCH --job-name=wtune-kf-g0', '#SBATCH --account=ACD110018',
        '#SBATCH --partition=ct112', '#SBATCH --nodes=1', '#SBATCH --ntasks=56',
        '#SBATCH --ntasks-per-node=56', '#SBATCH --cpus-per-task=2', '#SBATCH --hint=nomultithread',
        '#SBATCH --time=00:30:00', '#SBATCH --mem=128G',
        '#SBATCH --output=' + str(output.parent / 'kf-g0-%j.out'),
        'set -euo pipefail', 'module load intel/2024_01_46',
        'export PYTHONDONTWRITEBYTECODE=1', 'export TMPDIR=' + shlex.quote(str(output.parent)),
        'export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}',
        'exec ' + ' '.join(shlex.quote(arg) for arg in argv), ''])


def rank_exec(candidate):
    def decimal(name):
        value = os.environ.get(name, '')
        require(value.isdecimal(), 'missing native rank environment: ' + name)
        return int(value)
    rank, local = decimal('SLURM_PROCID'), decimal('SLURM_LOCALID')
    require(0 <= rank < 56 and 0 <= local < 56, 'unexpected world/local rank')
    require(os.environ.get('OMP_NUM_THREADS') == '2', 'rank launch lacks OMP2')
    affinity = sorted(os.sched_getaffinity(0))
    require(len(affinity) == 2, 'srun binding must supply exactly two CPU IDs')
    record = {'world_rank': rank, 'local_rank': local, 'job': decimal('SLURM_JOB_ID'),
              'step': decimal('SLURM_STEP_ID'), 'hostname': socket.gethostname(),
              'pid': os.getpid(), 'affinity': affinity, 'omp_threads': 2,
              'tasks_per_node': os.environ.get('SLURM_STEP_TASKS_PER_NODE',
                                             os.environ.get('SLURM_TASKS_PER_NODE'))}
    write_json(Path.cwd() / ('launch.%04d.json' % rank), record)
    os.execv(str(candidate), [str(candidate)])


def prepare_case(plan, arm, root):
    directory = root / arm
    directory.mkdir(mode=0o700)
    for name in ('wrfinput_d01', 'wrfbdy_d01'):
        os.symlink(plan['inputs'][name]['path'], directory / name)
    for name in TABLES:
        with open(plan['inputs'][name]['path'], 'rb') as source, (directory / name).open('xb') as dest:
            shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
    for name, record in plan['support'].items():
        require(name in SUPPORT, 'support file not allowlisted')
        os.symlink(record['path'], directory / name)
    with open(plan['inputs']['namelist.g0']['path'], 'rb') as source, (directory / 'namelist.input').open('xb') as dest:
        shutil.copyfileobj(source, dest)
    return directory


def run_process(argv, directory, environment, logfile, timeout, execute=subprocess.run):
    with logfile.open('xb') as stream:
        result = execute(argv, cwd=directory, env=environment, stdout=stream,
                         stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RunFailure(result.returncode, argv)


def compare_outputs(plan, root, reports, run=run_process, arms=('B', 'C')):
    names = reports['A']['outputs']
    require(all(reports[arm]['outputs'] == names for arm in arms), 'WRF output filename sets differ')
    comparisons = []
    for arm in arms:
        for index, name in enumerate(names):
            report = root / ('bitwise-A-%s-%02d.json' % (arm, index))
            argv = [plan['tools']['python']['path'], plan['tools']['comparator']['path'],
                    '--library', plan['tools']['library']['path'], '--report', str(report),
                    str(root / 'A' / name), str(root / arm / name)]
            run(argv, root, arm_environment('A', root), report.with_suffix('.log'), 90)
            value = json.loads(report.read_text())
            require(value.get('result') == 'identical', 'bitwise comparator did not verify equality')
            comparisons.append(file_record(report))
    return comparisons


def execute_plan(plan, run=run_process, validator=validate_arm):
    require(plan.get('artifact_type') == 'wtune_kf_native_g0_run_plan' and plan.get('version') == 1,
            'unknown run plan')
    require(os.environ.get('SLURM_JOB_NUM_NODES') == '1' and os.environ.get('SLURM_NTASKS') == '56' and
            os.environ.get('SLURM_CPUS_PER_TASK') == '2' and os.environ.get('SLURM_JOB_ID', '').isdecimal(),
            'execute requires the exact one-node 56-rank OMP2 allocation')
    require(set(plan['arms']) == {'A', 'B', 'C'} and set(plan['support']) == set(SUPPORT),
            'fixed arm/support inventory changed')
    recovered = inputs_identity(Path(plan['inputs']['recovery.json']['path']).parent)
    require(recovered == plan['inputs'], 'input recovery provenance changed')
    for record in [*plan['support'].values(), *plan['tools'].values()]:
        check_record(record)
    build = plan['build']
    require(candidate_identity(build['candidate']['path'], build['candidate']['sha256'],
                               build['manifest']['path'], build['terminal']['path']) == build,
            'candidate provenance changed')
    root = Path(plan['output'])
    root.mkdir(mode=0o700)
    summary = {'artifact_type': 'wtune_kf_native_g0_result', 'version': 1, 'status': 'started',
               'job': os.environ['SLURM_JOB_ID'], 'plan': plan, 'arms': {}, 'comparisons': []}
    try:
        for arm in ('A', 'B', 'C'):
            check_record(build['candidate'])
            directory = prepare_case(plan, arm, root)
            env = arm_environment(arm, directory)
            controls = arm_environment(arm, directory, {})
            require(plan['arms'][arm]['controls'] == controls, 'arm controls modified')
            argv = launch_argv(plan['tools']['python']['path'], plan['tools']['runner']['path'],
                               build['candidate']['path'])
            require(plan['arms'][arm]['argv'] == argv, 'arm launch modified')
            write_json(directory / 'command.json', {'argv': argv, 'controls': controls})
            start = time.monotonic()
            run(argv, directory, env, directory / 'wrf.out', 450)
            summary['arms'][arm] = validator(directory, arm, build['profile_id'])
            summary['arms'][arm]['wall_seconds'] = time.monotonic() - start
            write_json(directory / 'validation.json', summary['arms'][arm])
            # Every arm's native validation gates the next launch. No sibling runs after failure.
            if arm != 'A':
                summary['comparisons'].extend(compare_outputs(plan, root, summary['arms'], run, (arm,)))
        for record in plan['inputs'].values():
            check_record(record)
        summary['status'] = 'verified'
    except Exception as error:
        summary.update(status='failed', error=str(error), returncode=getattr(error, 'returncode', None))
        raise
    finally:
        write_json(root / 'job-summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    rank = commands.add_parser('rank-exec')
    rank.add_argument('--candidate', required=True, type=Path)
    execute = commands.add_parser('execute')
    execute.add_argument('--plan', required=True, type=Path)
    plan = commands.add_parser('plan')
    for key in ('candidate', 'manifest', 'terminal', 'inputs', 'output', 'plan', 'sbatch'):
        plan.add_argument('--' + key, required=True, type=Path)
    plan.add_argument('--candidate-sha256', required=True)
    plan.add_argument('--support', type=Path, default=WRF_RUN)
    plan.add_argument('--python', type=Path, default=PYTHON)
    plan.add_argument('--library', type=Path, default=LIBRARY)
    args = parser.parse_args(argv)
    try:
        if args.command == 'rank-exec':
            rank_exec(args.candidate)
        elif args.command == 'execute':
            execute_plan(json.loads(args.plan.read_text()))
        else:
            require(not args.plan.exists() and not args.sbatch.exists(), 'plan/script already exists')
            value = make_run_plan(args.candidate, args.candidate_sha256, args.manifest, args.terminal,
                                  args.inputs, args.support, args.output, args.python, args.library)
            write_json(args.plan, value)
            with args.sbatch.open('x') as stream:
                stream.write(sbatch_text(args.plan.resolve(), args.output, args.python))
        return 0
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        print('FAILED/REFUSED: ' + str(error), file=sys.stderr)
        return 2
    except RunFailure as error:
        print(str(error), file=sys.stderr)
        return error.returncode if error.returncode > 0 else 128 - error.returncode


if __name__ == '__main__':
    raise SystemExit(main())
