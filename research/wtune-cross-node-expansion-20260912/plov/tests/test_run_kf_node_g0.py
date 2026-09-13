"""No native processes: synthetic identity, rank logs and fail-stop subprocess checks."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from plov import kf_node_g0_contract as contract
from plov import run_kf_node_g0 as runner
from plov.prepare_isc21_node_g0 import write_json

ARTIFACTS = Path('/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/kf/g0-harness-tests')
PROFILE = 'reviewed-g0-v1:' + '1' * 64


def fake_build(root):
    candidate, review, mapfile = root / 'candidate', root / 'review.json', root / 'link.map'
    for path in (candidate, review, mapfile):
        path.write_text('fixture-only ' + path.name)
    manifest = {'artifact_type': 'wtune_kf_build_recipe', 'version': 3, 'executed': False,
                'attestation': dict.fromkeys(contract.ATTESTATION, True), 'plan_sha256': '2' * 64,
                'node_local_profile': {'enabled': True, 'name': 'reviewed-g0-v1',
                    'proof_activation_implemented': True, 'profile_id': PROFILE,
                    'attestation': {'source_abi_verified': True, 'caller_chain_verified': True,
                                    'init_verified': True, 'concurrency_verified': False},
                    'review_record': {'path': str(review), 'sha256': contract.sha256(review)}}}
    manifest_path, native_path, terminal_path = root / 'build-manifest.json', root / 'native.json', root / 'terminal.json'
    write_json(manifest_path, manifest)
    linked, linkmap = contract.file_record(candidate), contract.file_record(mapfile)
    native = {'artifact_type': 'wtune_kf_linked_candidate_identity', 'version': 1,
              'profile_id': PROFILE, 'candidate': linked, 'link_map': linkmap,
              'build_manifest_sha256': contract.sha256(manifest_path), 'plan_sha256': manifest['plan_sha256']}
    write_json(native_path, native)
    terminal = {'status': 'complete', 'link_executed': True, 'manifest_sha256': contract.sha256(manifest_path),
                'plan_sha256': manifest['plan_sha256'], 'linked_candidate': linked, 'link_map': linkmap,
                'native_identity_record': contract.file_record(native_path)}
    write_json(terminal_path, terminal)
    return candidate, linked['sha256'], manifest_path, terminal_path


def fake_arm(directory, arm, ranks=2):
    directory.mkdir(exist_ok=True)
    (directory / 'wrf.out').write_text('native fixture stdout\n')
    (directory / 'rsl.error.0000').write_text('SUCCESS COMPLETE WRF\n')
    (directory / 'rsl.out.0000').write_text('Timing for main: fixture\n' * 20)
    for suffix in ('09:00:00', '09:05:00'):
        (directory / ('wrfout_d01_2016-08-29_' + suffix)).write_text('fake netcdf')
    for rank in range(ranks):
        write_json(directory / ('launch.%04d.json' % rank), {'world_rank': rank, 'local_rank': rank,
                   'job': 1, 'step': 0, 'pid': rank + 100, 'hostname': 'fixture-node', 'affinity': [rank * 2, rank * 2 + 1]})
        active, remote = int(arm != 'A'), int(arm == 'C')
        node = {'requested': 20, 'donated': active, 'self': active - remote, 'helper': remote,
                'committed': active, 'fallback': 20 - active, 'fatal': 0, 'shadow_checked': remote,
                'shadow_failed': 0, 'shadow_bytes': remote * 16, 'shadow_remote': remote,
                'serialized_native': 1,
                'profile': contract.profile_number(PROFILE), 'init': 99,
                'mode': 'forced-g0-owner-only' if arm == 'B' else 'forced-g0'}
        values = lambda pairs: ' '.join('%s=%s' % (key, value) for key, value in pairs.items())
        text = '# rank=%d\n# omp_max_threads=2 omp_num_threads_env=2\n' % rank
        text += '# stage_flags ' + values({key: value for key, value in contract.arm_environment(arm, directory, {}).items()
                                           if key.startswith('WTIMER_STAGE_') and key != 'WTIMER_STAGE_GRID'}) + '\n'
        text += '# wt_kf_bootstrap profile=%s configured=%d requested=%d staged=%d\n' % (PROFILE, active, active, active)
        text += '# wt_kf_node_identity pid=%d local_rank=%d grid=1 last_ktau=20 owner_remote_committed=%d\n' % (rank + 100, rank, remote)
        mode = node.pop('mode')
        text += '# wt_kf_node mode=' + mode + ' ' + values(node) + '\n'
        text += ''.join('# wt_kf_node refusal.%s=%d\n' % (name, 20 - active if name == 'single_row' else 0) for name in
                        ('config', 'abi', 'init', 'scope', 'mapper', 'omp', 'capacity', 'geometry',
                         'single_row', 'alias', 'budget', 'slot', 'prepare'))
        if arm == 'A':
            text = '\n'.join(line for line in text.splitlines() if not line.startswith('# wt_kf_node')) + '\n'
        (directory / ('wtimer.%04d.csv' % rank)).write_text(text)


class RunTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='runner-', dir=ARTIFACTS))

    def test_candidate_requires_review_terminal_link_and_exact_hash(self):
        args = fake_build(self.root)
        accepted = contract.candidate_identity(*args)
        self.assertEqual(accepted['profile_id'], PROFILE)
        with self.assertRaisesRegex(ValueError, 'explicit candidate'):
            contract.candidate_identity(args[0], '0' * 64, *args[2:])
        terminal = json.loads(args[3].read_text())
        terminal['link_executed'] = False
        args[3].write_text(json.dumps(terminal))
        with self.assertRaisesRegex(ValueError, 'terminal complete'):
            contract.candidate_identity(*args)
        terminal['link_executed'] = True
        args[3].write_text(json.dumps(terminal))
        manifest = json.loads(args[2].read_text())
        manifest['attestation']['source_abi_verified'] = False
        args[2].write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'unattested'):
            contract.candidate_identity(*args)
        with self.assertRaises(FileNotFoundError):
            contract.inputs_identity(self.root / 'missing-inputs')

    def test_environment_and_srun_are_identical_except_experiment_controls(self):
        polluted = {'PATH': '/bin', 'WTIMER_STAGE_ROWS': '99', 'WT_MYNN_NODE_G0': '1',
                    'WT_SFCLAY_NODE_G0': '1', 'WT_KF_UNKNOWN': '1', 'OMP_NUM_THREADS': '90',
                    'I_MPI_PIN_PROCESSOR_LIST': '0', 'KMP_AFFINITY': 'disabled'}
        envs = [contract.arm_environment(arm, self.root, polluted) for arm in ('A', 'B', 'C')]
        for env in envs:
            self.assertNotIn('WTIMER_STAGE_ROWS', env)
            self.assertNotIn('WT_MYNN_NODE_G0', env)
            self.assertNotIn('I_MPI_PIN_PROCESSOR_LIST', env)
            self.assertEqual(env['OMP_NUM_THREADS'], '2')
        different = {key for key in envs[0] if len({env[key] for env in envs}) > 1}
        self.assertEqual(different, {'WTIMER_STAGE_CUMULUS', 'WT_KF_NODE_G0_OWNER_ONLY', 'WT_KF_NODE_G0_SHADOW'})
        argv = contract.launch_argv('/python', '/runner', '/candidate')
        for item in ('--ntasks=56', '--cpus-per-task=2', '--distribution=block:block', '--cpu-bind=cores'):
            self.assertIn(item, argv)
        script = runner.sbatch_text(self.root / 'plan.json', self.root / 'runs')
        self.assertIn('#SBATCH --partition=ct112', script)
        self.assertIn('#SBATCH --account=ACD110018', script)
        self.assertNotIn('sbatch ', script)
        self.assertNotIn('rm ', script)

    def test_native_rank_counters_steps_and_zero_steal_are_gates(self):
        for arm in ('A', 'B', 'C'):
            fake_arm(self.root / arm, arm)
            report = contract.validate_arm(self.root / arm, arm, PROFILE, 2)
            self.assertEqual(len(report['ranks']), 2)
        path = self.root / 'C/wtimer.0000.csv'
        original = path.read_text()
        path.write_text(original.replace('shadow_checked=1', 'shadow_checked=0'))
        with self.assertRaisesRegex(ValueError, 'remote/helper/shadow'):
            contract.validate_arm(self.root / 'C', 'C', PROFILE, 2)
        path.write_text(original.replace('omp_max_threads=2', 'omp_max_threads=1'))
        with self.assertRaisesRegex(ValueError, 'OMP2'):
            contract.validate_arm(self.root / 'C', 'C', PROFILE, 2)
        path.write_text(original)
        for csv in (self.root / 'C').glob('wtimer.*.csv'):
            csv.write_text(csv.read_text().replace('self=0', 'self=1').replace('helper=1', 'helper=0')
                           .replace('owner_remote_committed=1', 'owner_remote_committed=0')
                           .replace('shadow_checked=1', 'shadow_checked=0'))
        with self.assertRaisesRegex(ValueError, 'zero steal'):
            contract.validate_arm(self.root / 'C', 'C', PROFILE, 2)
        (self.root / 'C/rsl.error.0001').write_text('FATAL CALLED\n')
        with self.assertRaisesRegex(ValueError, 'log guard'):
            contract.validate_arm(self.root / 'C', 'C', PROFILE, 2)

    def test_nonzero_subprocess_and_bitwise_fail_stop(self):
        def failed(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 17)
        with self.assertRaises(runner.RunFailure) as caught:
            runner.run_process(['mock-native'], self.root, {}, self.root / 'fail.log', 1, failed)
        self.assertEqual(caught.exception.returncode, 17)
        reports = {arm: {'outputs': ['wrfout1', 'wrfout2']} for arm in ('A', 'B', 'C')}
        plan = {'tools': {key: {'path': key} for key in ('python', 'comparator', 'library')}}
        calls = []
        def bad_compare(argv, *args):
            calls.append(argv)
            raise runner.RunFailure(1, argv)
        with self.assertRaises(runner.RunFailure):
            runner.compare_outputs(plan, self.root, reports, bad_compare)
        self.assertEqual(len(calls), 1)
        self.assertIn('--library', calls[0])

    def mock_plan(self):
        build_args = fake_build(self.root)
        build = contract.candidate_identity(*build_args)
        inputs = {}
        for name in ('wrfinput_d01', 'wrfbdy_d01', 'namelist.g0', *runner.TABLES):
            path = self.root / name
            path.write_text('fake ' + name)
            inputs[name] = contract.file_record(path)
        inputs['recovery.json'] = inputs['namelist.g0']
        tools = {key: contract.file_record(build_args[0]) for key in ('runner', 'python', 'comparator', 'library')}
        output = self.root / 'runs'
        plan = {'artifact_type': 'wtune_kf_native_g0_run_plan', 'version': 1, 'build': build,
                'inputs': inputs, 'support': dict.fromkeys(contract.SUPPORT, inputs['namelist.g0']),
                'tools': tools, 'output': str(output), 'arms': {}}
        for arm in ('A', 'B', 'C'):
            plan['arms'][arm] = {'argv': contract.launch_argv(tools['python']['path'], tools['runner']['path'], build_args[0]),
                                 'controls': contract.arm_environment(arm, output / arm, {})}
        return plan

    def test_abc_compares_all_outputs_and_ab_failure_prevents_c(self):
        plan = self.mock_plan()
        launches, comparisons = [], []
        def execute(argv, directory, env, logfile, timeout):
            if argv[0] == 'timeout':
                launches.append(directory.name)
                fake_arm(directory, directory.name)
            else:
                comparisons.append(argv[-1])
                if len(comparisons) == 2:
                    raise runner.RunFailure(1, argv)
                write_json(Path(argv[argv.index('--report') + 1]), {'result': 'identical'})
        with mock.patch.dict(os.environ, {'SLURM_JOB_NUM_NODES': '1', 'SLURM_NTASKS': '56',
                                         'SLURM_CPUS_PER_TASK': '2', 'SLURM_JOB_ID': '123'}), \
                mock.patch.object(runner, 'inputs_identity', return_value=plan['inputs']):
            with self.assertRaises(runner.RunFailure):
                runner.execute_plan(plan, execute, lambda path, arm, profile: contract.validate_arm(path, arm, profile, 2))
        self.assertEqual(launches, ['A', 'B'])
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(comparisons[0].endswith('09:00:00'))
        self.assertTrue(comparisons[1].endswith('09:05:00'))
        self.assertFalse((Path(plan['output']) / 'C').exists())

    def test_mock_complete_abc_and_table_copy_isolation(self):
        plan = self.mock_plan()
        sequence = []
        def execute(argv, directory, env, logfile, timeout):
            if argv[0] == 'timeout':
                sequence.append(directory.name)
                table = directory / runner.TABLES[0]
                self.assertFalse(table.is_symlink())
                self.assertEqual(table.read_text(), 'fake ' + runner.TABLES[0])
                table.write_text('WRF fixture rewrote private table')
                self.assertTrue((directory / 'wrfinput_d01').is_symlink())
                fake_arm(directory, directory.name)
            else:
                sequence.append('compare-' + Path(argv[-1]).parent.name)
                write_json(Path(argv[argv.index('--report') + 1]), {'result': 'identical'})
        with mock.patch.dict(os.environ, {'SLURM_JOB_NUM_NODES': '1', 'SLURM_NTASKS': '56',
                                         'SLURM_CPUS_PER_TASK': '2', 'SLURM_JOB_ID': '123'}), \
                mock.patch.object(runner, 'inputs_identity', return_value=plan['inputs']):
            result = runner.execute_plan(plan, execute,
                lambda path, arm, profile: contract.validate_arm(path, arm, profile, 2))
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(sequence, ['A', 'B', 'compare-B', 'compare-B', 'C', 'compare-C', 'compare-C'])
        self.assertEqual(len(result['comparisons']), 4)

    def test_failed_a_never_launches_b_or_c_and_preserves_summary(self):
        plan = self.mock_plan()
        inputs, output = plan['inputs'], Path(plan['output'])
        calls = []
        def failed(argv, *args):
            calls.append(argv)
            raise runner.RunFailure(17, argv)
        with mock.patch.dict(os.environ, {'SLURM_JOB_NUM_NODES': '1', 'SLURM_NTASKS': '56',
                                         'SLURM_CPUS_PER_TASK': '2', 'SLURM_JOB_ID': '123'}):
            with mock.patch.object(runner, 'inputs_identity', return_value=inputs), self.assertRaises(runner.RunFailure):
                runner.execute_plan(plan, failed)
        self.assertEqual(len(calls), 1)
        self.assertFalse((output / 'B').exists())
        self.assertFalse((output / 'C').exists())
        summary = json.loads((output / 'job-summary.json').read_text())
        self.assertEqual((summary['status'], summary['returncode']), ('failed', 17))


if __name__ == '__main__':
    unittest.main()
