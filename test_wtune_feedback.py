import json
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

if __name__ == '__main__':
    unittest.main()
