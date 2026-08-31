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
            self.assertFalse(json.load(open(output_json))[0]['actionable'])
            bad = os.path.join(directory, 'bad.json')
            with open(bad, 'w') as handle:
                json.dump({'samples': [{'unknown': 1}]}, handle)
            bad_json, bad_csv = os.path.join(directory, 'bad.out'), os.path.join(directory, 'bad.csv')
            self.assertNotEqual(subprocess.call([sys.executable, os.path.join(root, 'wtune.py'), 'feedback',
                                                  '--input', bad, '--json-out', bad_json, '--csv-out', bad_csv]), 0)
            self.assertFalse(os.path.exists(bad_json))
            self.assertFalse(os.path.exists(bad_csv))

if __name__ == '__main__':
    unittest.main()
