import json
import csv
import os
import subprocess
import sys
import tempfile
import unittest

from wtune_feedback import FeedbackController, FeedbackError, FeedbackSample, decisions_to_csv, decisions_to_json

def sample(**extra):
    value = dict(imbalance_before=2.0, imbalance_threshold=1.0,
                 migration_cost_seconds=1.0, payload_bytes=10,
                 stall_time_seconds=0.1, imbalance_after=1.0,
                 correctness_passed=True, runtime_hook_available=True,
                 manifest_compatible=True, ownership_before='a', ownership_after='b')
    value.update(extra)
    return FeedbackSample(**value)

class FeedbackTests(unittest.TestCase):
    def test_conservative_then_history_supported(self):
        c = FeedbackController(history_limit=2, max_cost_seconds=2)
        self.assertFalse(c.observe(sample()).worthy)
        d = c.observe(sample(imbalance_after=0.5))
        self.assertTrue(d.worthy)
        self.assertEqual(len(c.history), 2)

    def test_gates_and_fallback(self):
        c = FeedbackController(max_cost_seconds=1)
        for change in ({'imbalance_before': .5}, {'correctness_passed': False},
                       {'manifest_compatible': False}, {'runtime_hook_available': False},
                       {'migration_cost_seconds': 2}):
            d = c.observe(sample(**change))
            self.assertFalse(d.worthy)
            if d.triggered:
                self.assertEqual(d.fallback, 'stable-ownership')

    def test_validation_and_serialization(self):
        with self.assertRaises(FeedbackError):
            sample(payload_bytes=-1)
        d = FeedbackController().observe(sample())
        self.assertIn('reason', json.loads(decisions_to_json([d]))[0])
        self.assertIn('ownership_before', decisions_to_csv([d]))

    def test_bad_history_does_not_support_later_improvement(self):
        for failed in ({'correctness_passed': False}, {'imbalance_before': .5},
                        {'migration_cost_seconds': 3}, {'imbalance_after': 2}):
            c = FeedbackController(max_cost_seconds=2)
            c.observe(sample(**failed))
            self.assertFalse(c.observe(sample(imbalance_after=.5)).worthy)

    def test_unknown_and_non_json_rejected(self):
        with self.assertRaises(FeedbackError):
            sample(extra_field=1)
        with self.assertRaises(FeedbackError):
            sample(ownership_before={1: 'rank'})
        with self.assertRaises(FeedbackError):
            sample(ownership_after=set(['rank']))
        with self.assertRaises(FeedbackError):
            sample(ownership_after=float('inf'))

    def test_every_numeric_field_rejects_bad_values(self):
        fields = ('imbalance_before', 'imbalance_threshold', 'migration_cost_seconds',
                  'payload_bytes', 'stall_time_seconds', 'imbalance_after')
        for field in fields:
            for value in ('bad', -1, float('inf'), float('nan')):
                with self.subTest(field=field, value=repr(value)):
                    with self.assertRaises(FeedbackError):
                        sample(**{field: value})
        with self.assertRaises(FeedbackError):
            sample(payload_bytes=1.5)
        with self.assertRaises(FeedbackError):
            sample(payload_bytes=True)

    def test_boolean_ownership_and_limits_reject_bad_values(self):
        for field in ('correctness_passed', 'runtime_hook_available', 'manifest_compatible'):
            with self.assertRaises(FeedbackError):
                sample(**{field: 1})
        for value in ({'nested': {1: 'bad'}}, {'nested': [float('nan')]},
                      {'nested': object()}):
            with self.assertRaises(FeedbackError):
                sample(ownership_before=value)
        for kwargs in ({'history_limit': 0}, {'history_limit': -1},
                       {'history_limit': True}, {'history_limit': 1.0},
                       {'max_cost_seconds': -1}, {'max_cost_seconds': '1'},
                       {'max_cost_seconds': float('nan')}):
            with self.assertRaises(FeedbackError):
                FeedbackController(**kwargs)

    def test_history_eviction_removes_evidence(self):
        c = FeedbackController(history_limit=1)
        self.assertFalse(c.observe(sample(imbalance_after=.5)).worthy)
        c.observe(sample(imbalance_before=.5))
        self.assertFalse(c.observe(sample(imbalance_after=.5)).worthy)

    def test_json_csv_round_trip_columns(self):
        first = FeedbackController().observe(sample())
        payload = json.loads(decisions_to_json([first]))
        self.assertEqual(payload[0]['actionable'], False)
        rows = list(csv.DictReader(decisions_to_csv([first]).splitlines()))
        self.assertEqual(rows[0]['actionable'], 'False')
        self.assertEqual(json.loads(rows[0]['ownership_before']), 'a')

    def test_cli_publication_and_invalid_input(self):
        root = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.json')
            output_json = os.path.join(directory, 'decisions.json')
            output_csv = os.path.join(directory, 'decisions.csv')
            with open(source, 'w') as handle:
                json.dump({'samples': [sample().to_dict()]}, handle)
            command = [sys.executable, os.path.join(root, 'wtune.py'), 'feedback',
                       '--input', source, '--json-out', output_json, '--csv-out', output_csv]
            self.assertEqual(subprocess.call(command), 0)
            with open(output_json) as handle:
                self.assertFalse(json.load(handle)[0]['actionable'])
            with open(output_csv) as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ['imbalance_before', 'imbalance_threshold',
                                       'migration_cost_seconds', 'payload_bytes',
                                       'stall_time_seconds', 'imbalance_after',
                                       'correctness_passed', 'runtime_hook_available',
                                       'manifest_compatible', 'ownership_before',
                                       'ownership_after', 'triggered', 'worthy',
                                       'reason', 'fallback', 'actionable', 'advisory',
                                       'non_actionable'])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1][-3:], ['False', 'True', 'True'])
            bad = os.path.join(directory, 'bad.json')
            with open(bad, 'w') as handle:
                json.dump({'samples': [{'unknown': 1}]}, handle)
            bad_json, bad_csv = os.path.join(directory, 'bad.out'), os.path.join(directory, 'bad.csv')
            with open(bad_json, 'w') as handle:
                handle.write('old-json')
            with open(bad_csv, 'w') as handle:
                handle.write('old-csv\n')
            with open(bad_json, 'rb') as handle:
                old_json = handle.read()
            with open(bad_csv, 'rb') as handle:
                old_csv = handle.read()
            self.assertNotEqual(subprocess.call([sys.executable, os.path.join(root, 'wtune.py'), 'feedback',
                                                  '--input', bad, '--json-out', bad_json, '--csv-out', bad_csv]), 0)
            with open(bad_json, 'rb') as handle:
                self.assertEqual(handle.read(), old_json)
            with open(bad_csv, 'rb') as handle:
                self.assertEqual(handle.read(), old_csv)

if __name__ == '__main__':
    unittest.main()
