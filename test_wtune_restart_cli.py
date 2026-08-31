import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import wtune
import wtune_restart


def _fake_oracle():
    """Oracle adapter with no NetCDF dependency for command-boundary tests."""
    phase_calls = []

    def extract_features(path, stride=4, lev_stride=3, **unused_options):
        return {'snapshot': path, 'stride': stride}

    def phase_cost_fields(features, doy, utc_hours, run_hours=0.0, n_time=9,
                          cu_on=False):
        phase_calls.append((doy, utc_hours, run_hours))
        return {'dynamics': 'dynamics', 'radiation': 'radiation'}

    def phase_weights(cu_on=False):
        return {'dynamics': 0.6, 'radiation': 0.4}

    def to_ranks(field, px, py):
        base = 2.0 if field == 'dynamics' else 1.0
        return [[base + row * px + col for col in range(px)] for row in range(py)]

    return types.SimpleNamespace(extract_features=extract_features,
                                 phase_cost_fields=phase_cost_fields,
                                 phase_weights=phase_weights, to_ranks=to_ranks,
                                 phase_calls=phase_calls)


class RestartWindowCliTests(unittest.TestCase):
    def _write_case(self, root, max_dom=1, tile_strategy=0):
        path = os.path.join(root, 'namelist.input')
        with open(path, 'w') as handle:
            handle.write("""&domains
 max_dom = %d,
 e_we = 41,
 e_sn = 21,
 nproc_x = 2,
 nproc_y = 2,
 numtiles = 99,
 numtiles_x = 0,
 numtiles_y = 0,
 tile_strategy = %d,
/
&time_control
 run_hours = 2,
 restart_interval = 60,
 start_year = 2024,
 start_month = 12,
 start_day = 31,
 start_hour = 23,
 start_minute = 30,
 start_second = 15,
/
&physics
 cu_physics = 0,
/
""" % (max_dom, tile_strategy))
        return path

    def _write_topology(self, root):
        path = os.path.join(root, 'topology.json')
        topology = {'nodes': [
            {'name': 'n1', 'numa_domains': [{'id': 0, 'cpus': [0, 1, 2, 3]}]},
            {'name': 'n2', 'numa_domains': [{'id': 0, 'cpus': [0, 1, 2, 3]}]},
        ], 'allowed_team_sizes': [1]}
        with open(path, 'w') as handle:
            json.dump(topology, handle)
        return path

    def _run_cli(self, argv, oracle=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, 'argv', ['wtune.py'] + argv), \
                mock.patch.dict(sys.modules, {'wtune_oracle': oracle or _fake_oracle()}), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                wtune.main()
            except SystemExit as exc:
                return exc.code, stdout.getvalue(), stderr.getvalue()
        return 0, stdout.getvalue(), stderr.getvalue()

    def test_extract_case_includes_restart_decomposition_and_start_controls(self):
        with tempfile.TemporaryDirectory() as root:
            case = wtune.extract_case(wtune.parse_namelist(self._write_case(root)))
        self.assertEqual(case['nproc_x'], 2)
        self.assertEqual(case['nproc_y'], 2)
        self.assertEqual(case['numtiles'], 99)
        self.assertEqual(case['numtiles_x'], 0)
        self.assertEqual(case['numtiles_y'], 0)
        self.assertEqual(case['tile_strategy'], 0)
        self.assertEqual((case['start_minute'], case['start_second']), (30, 15))

    def test_cli_writes_deterministic_unverified_window_artifacts_without_mutating_namelist(self):
        with tempfile.TemporaryDirectory() as root:
            namelist = self._write_case(root)
            topology = self._write_topology(root)
            snapshot = os.path.join(root, 'wrfrst_d01')
            with open(snapshot, 'w'):
                pass
            with open(namelist, 'rb') as handle:
                original = handle.read()
            first = os.path.join(root, 'first')
            second = os.path.join(root, 'second')
            common = ['restart-window', '--namelist', namelist, '--snapshot', snapshot,
                      '--topology-json', topology, '--top-k', '1']
            oracle = _fake_oracle()
            code, out, err = self._run_cli(common + ['--output-dir', first], oracle)
            self.assertEqual(code, 0, err)
            self.assertIn('UNVERIFIED', out)
            self.assertIn('actionable=false', out)
            code, unused_out, err = self._run_cli(common + ['--output-dir', second])
            self.assertEqual(code, 0, err)
            with open(namelist, 'rb') as handle:
                self.assertEqual(handle.read(), original)
            with open(os.path.join(first, 'restart-window-manifest.json')) as handle:
                manifest = json.load(handle)
            self.assertFalse(manifest['actionable'])
            self.assertEqual(manifest['strict_bitwise'], 'UNVERIFIED')
            self.assertEqual(len(manifest['windows']), 2)
            self.assertEqual(manifest['rank_counts'], [4])
            self.assertEqual(manifest['windows'][0]['start_utc'], '2024-12-31T23:30:15Z')
            self.assertAlmostEqual(oracle.phase_calls[0][1], 23.504166666666666)
            for name in ('window-000-ranks-4.map', 'window-000-ranks-4.sh',
                         'window-001-ranks-4.map', 'window-001-ranks-4.sh'):
                with open(os.path.join(first, name)) as left, \
                        open(os.path.join(second, name)) as right:
                    self.assertEqual(left.read(), right.read())

    def test_cli_rejects_missing_inputs_and_unsupported_case_before_output(self):
        with tempfile.TemporaryDirectory() as root:
            namelist = self._write_case(root)
            topology = self._write_topology(root)
            output = os.path.join(root, 'out')
            code, unused_out, err = self._run_cli(['restart-window', '--namelist', namelist,
                                                    '--topology-json', topology,
                                                    '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('snapshot', err.lower())
            self.assertFalse(os.path.exists(output))
            snapshot = os.path.join(root, 'snapshot')
            with open(snapshot, 'w'):
                pass
            code, unused_out, err = self._run_cli(['restart-window', '--namelist', namelist,
                                                    '--snapshot', snapshot,
                                                    '--topology-json', os.path.join(root, 'missing.json'),
                                                    '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('topology', err.lower())
            self.assertFalse(os.path.exists(output))
            unsupported = self._write_case(root, tile_strategy=3)
            code, unused_out, err = self._run_cli(['restart-window', '--namelist', unsupported,
                                                    '--snapshot', snapshot, '--topology-json', topology,
                                                    '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('unsupported', err.lower())
            self.assertFalse(os.path.exists(output))
            multi_domain = self._write_case(root, max_dom=2)
            code, unused_out, err = self._run_cli(['restart-window', '--namelist', multi_domain,
                                                    '--snapshot', snapshot, '--topology-json', topology,
                                                    '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('max_dom', err.lower())
            self.assertFalse(os.path.exists(output))

    def test_cli_rejects_duplicate_rank_cpu_allocations_without_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            namelist = self._write_case(root)
            topology = self._write_topology(root)
            snapshot = os.path.join(root, 'snapshot')
            with open(snapshot, 'w'):
                pass
            output = os.path.join(root, 'out')
            overlapping = [
                {'rank': 0, 'team_size': 1, 'node': 'n1', 'numa_domain': 0, 'cpus': [0]},
                {'rank': 1, 'team_size': 1, 'node': 'n1', 'numa_domain': 0, 'cpus': [0]},
                {'rank': 2, 'team_size': 1, 'node': 'n2', 'numa_domain': 0, 'cpus': [0]},
                {'rank': 3, 'team_size': 1, 'node': 'n2', 'numa_domain': 0, 'cpus': [1]},
            ]
            with mock.patch.object(wtune_restart, 'allocate_teams', return_value=overlapping):
                code, unused_out, err = self._run_cli([
                    'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                    '--topology-json', topology, '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('overlap', err.lower())
            self.assertFalse(os.path.exists(output))

    def test_cli_rejects_manifest_path_collisions_without_mutating_inputs(self):
        for collision in ('namelist', 'snapshot', 'topology'):
            with tempfile.TemporaryDirectory() as root:
                namelist = self._write_case(root)
                topology = self._write_topology(root)
                snapshot = os.path.join(root, 'snapshot')
                with open(snapshot, 'w'):
                    pass
                inputs = {'namelist': namelist, 'snapshot': snapshot,
                          'topology': topology}
                before = {}
                for key, path in inputs.items():
                    with open(path, 'rb') as handle:
                        before[key] = handle.read()
                output = os.path.join(root, 'out')
                code, unused_out, err = self._run_cli([
                    'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                    '--topology-json', topology, '--manifest-out', inputs[collision],
                    '--output-dir', output])
                self.assertNotEqual(code, 0)
                self.assertIn('collision', err.lower())
                for key, path in inputs.items():
                    with open(path, 'rb') as handle:
                        self.assertEqual(handle.read(), before[key])
                self.assertFalse(os.path.exists(output))

    def test_cli_rejects_manifest_path_equal_to_generated_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            namelist = self._write_case(root)
            topology = self._write_topology(root)
            snapshot = os.path.join(root, 'snapshot')
            with open(snapshot, 'w'):
                pass
            output = os.path.join(root, 'out')
            manifest = os.path.join(output, 'window-000-ranks-4.map')
            code, unused_out, err = self._run_cli([
                'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                '--topology-json', topology, '--manifest-out', manifest,
                '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('duplicate output path', err.lower())
            self.assertFalse(os.path.exists(output))

    def test_cli_reports_malformed_topology_without_traceback(self):
        with tempfile.TemporaryDirectory() as root:
            namelist = self._write_case(root)
            topology = os.path.join(root, 'topology.json')
            with open(topology, 'w') as handle:
                json.dump({'nodes': [None], 'allowed_team_sizes': [1]}, handle)
            snapshot = os.path.join(root, 'snapshot')
            with open(snapshot, 'w'):
                pass
            output = os.path.join(root, 'out')
            code, unused_out, err = self._run_cli([
                'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                '--topology-json', topology, '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('ERROR restart-window:', err)
            self.assertNotIn('Traceback', err)
            self.assertFalse(os.path.exists(output))

    def test_cli_publishes_generated_artifacts_before_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            namelist = self._write_case(root)
            topology = self._write_topology(root)
            snapshot = os.path.join(root, 'snapshot')
            with open(snapshot, 'w'):
                pass
            output = os.path.join(root, 'out')
            manifest = os.path.join(root, 'manifest.json')
            destinations = []
            real_replace = os.replace

            def record_replace(source, destination):
                destinations.append(destination)
                return real_replace(source, destination)

            with mock.patch.object(wtune.os, 'replace', side_effect=record_replace):
                code, unused_out, err = self._run_cli([
                    'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                    '--topology-json', topology, '--manifest-out', manifest,
                    '--output-dir', output])
            self.assertEqual(code, 0, err)
            self.assertEqual(destinations[-1], os.path.realpath(manifest))
            self.assertTrue(all(path != os.path.realpath(manifest)
                                for path in destinations[:-1]))

    def test_cli_rolls_back_partial_publication_after_late_replace_failure(self):
        with tempfile.TemporaryDirectory() as root:
            namelist = self._write_case(root)
            topology = self._write_topology(root)
            snapshot = os.path.join(root, 'snapshot')
            with open(snapshot, 'w'):
                pass
            output = os.path.join(root, 'out')
            os.makedirs(output)
            manifest = os.path.join(root, 'manifest.json')
            artifact_names = [
                'window-000-ranks-4.map', 'window-000-ranks-4.sh',
                'window-001-ranks-4.map', 'window-001-ranks-4.sh',
            ]
            originals = {manifest: 'old manifest\n'}
            originals.update((os.path.join(output, name), 'old ' + name + '\n')
                             for name in artifact_names)
            unrelated = os.path.join(output, 'keep.txt')
            for path, content in originals.items():
                with open(path, 'w') as handle:
                    handle.write(content)
            with open(unrelated, 'w') as handle:
                handle.write('keep me\n')
            real_replace = os.replace
            failed = [False]

            def fail_once_on_wrapper(source, destination):
                if destination.endswith('.sh') and not failed[0]:
                    failed[0] = True
                    raise OSError('injected late publication failure')
                return real_replace(source, destination)

            with mock.patch.object(wtune.os, 'replace',
                                   side_effect=fail_once_on_wrapper):
                code, unused_out, err = self._run_cli([
                    'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                    '--topology-json', topology, '--manifest-out', manifest,
                    '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('writing artifacts', err.lower())
            for path, content in originals.items():
                with open(path) as handle:
                    self.assertEqual(handle.read(), content)
            with open(unrelated) as handle:
                self.assertEqual(handle.read(), 'keep me\n')
            self.assertFalse(any(name.startswith('.wtune-')
                                 for name in os.listdir(output)))

    def test_cli_rejects_complete_malformed_allocations_before_rendering(self):
        valid = [{'rank': rank, 'team_size': 1, 'node': 'n1', 'numa_domain': 0,
                  'resource_key': 'n1:0', 'cpus': [rank]}
                 for rank in range(4)]
        malformed = []
        case = copy.deepcopy(valid)
        case[0]['cpus'] = []
        malformed.append(('empty CPU list', case))
        case = copy.deepcopy(valid)
        case[0]['cpus'] = [-1]
        malformed.append(('negative CPU', case))
        case = copy.deepcopy(valid)
        case[0]['team_size'] = 2
        malformed.append(('team size mismatch', case))
        case = copy.deepcopy(valid)
        del case[0]['team_size']
        malformed.append(('missing team size', case))
        case = copy.deepcopy(valid)
        case[0]['team_size'] = '1'
        malformed.append(('invalid team size', case))
        case = copy.deepcopy(valid)
        case[0]['node'] = 'unknown-node'
        malformed.append(('unknown node', case))
        case = copy.deepcopy(valid)
        case[0]['cpus'] = [99]
        malformed.append(('CPU absent from topology', case))
        for label, allocation in malformed:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as root:
                namelist = self._write_case(root)
                topology = self._write_topology(root)
                snapshot = os.path.join(root, 'snapshot')
                with open(snapshot, 'w'):
                    pass
                output = os.path.join(root, 'out')
                with mock.patch.object(wtune_restart, 'allocate_teams',
                                       return_value=allocation):
                    code, unused_out, err = self._run_cli([
                        'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                        '--topology-json', topology, '--output-dir', output])
                self.assertNotEqual(code, 0)
                self.assertIn('allocation', err.lower())
                self.assertFalse(os.path.exists(output))

    def test_cli_rejects_unselectable_rank_count_and_nonpositive_top_k(self):
        with tempfile.TemporaryDirectory() as root:
            namelist = self._write_case(root)
            topology = self._write_topology(root)
            snapshot = os.path.join(root, 'snapshot')
            with open(snapshot, 'w'):
                pass
            output = os.path.join(root, 'out')
            code, unused_out, err = self._run_cli([
                'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                '--topology-json', topology, '--rank-counts', '5',
                '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('legal', err.lower())
            self.assertFalse(os.path.exists(output))
            code, unused_out, err = self._run_cli([
                'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                '--topology-json', topology, '--top-k', '0', '--output-dir', output])
            self.assertNotEqual(code, 0)
            self.assertIn('top-k', err.lower())
            self.assertFalse(os.path.exists(output))

    def test_manifest_records_resolvable_absolute_artifact_paths(self):
        with tempfile.TemporaryDirectory() as root:
            namelist = self._write_case(root)
            topology = self._write_topology(root)
            snapshot = os.path.join(root, 'snapshot')
            with open(snapshot, 'w'):
                pass
            canonical_output = os.path.join(root, 'canonical-artifacts')
            os.makedirs(canonical_output)
            output = os.path.join(root, 'artifacts')
            os.symlink(canonical_output, output)
            manifest_path = os.path.join(output, 'manifest.json')
            code, out, err = self._run_cli([
                'restart-window', '--namelist', namelist, '--snapshot', snapshot,
                '--topology-json', topology, '--manifest-out', manifest_path,
                '--output-dir', output])
            self.assertEqual(code, 0, err)
            canonical_manifest = os.path.realpath(manifest_path)
            self.assertIn('created: {}'.format(canonical_manifest), out)
            with open(canonical_manifest) as handle:
                manifest = json.load(handle)
            for window in manifest['windows']:
                for choice in window['choices']:
                    self.assertTrue(os.path.isabs(choice['map']))
                    self.assertTrue(os.path.isabs(choice['wrapper']))
                    self.assertEqual(choice['map'], os.path.realpath(choice['map']))
                    self.assertEqual(choice['wrapper'], os.path.realpath(choice['wrapper']))
                    self.assertTrue(os.path.isfile(choice['map']))
                    self.assertTrue(os.path.isfile(choice['wrapper']))


if __name__ == '__main__':
    unittest.main()
