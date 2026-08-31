"""Pure, advisory feedback policy for manifest-only WTune runs."""
import csv
import io
import json
import math

_FIELDS = ('imbalance_before', 'imbalance_threshold', 'migration_cost_seconds',
           'payload_bytes', 'stall_time_seconds', 'imbalance_after',
           'correctness_passed', 'runtime_hook_available',
           'manifest_compatible', 'ownership_before', 'ownership_after')

class FeedbackError(ValueError):
    pass

def _number(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeedbackError('{} must be numeric'.format(name))
    value = float(value)
    if not math.isfinite(value) or value < 0 or (integer and value != int(value)):
        raise FeedbackError('{} is invalid'.format(name))
    return int(value) if integer else value

def _boolean(value, name):
    if not isinstance(value, bool):
        raise FeedbackError('{} must be boolean'.format(name))
    return value

class FeedbackSample(object):
    def __init__(self, **kwargs):
        missing = [key for key in _FIELDS if key not in kwargs]
        if missing:
            raise FeedbackError('missing telemetry: {}'.format(','.join(missing)))
        self.imbalance_before = _number(kwargs['imbalance_before'], 'imbalance_before')
        self.imbalance_threshold = _number(kwargs['imbalance_threshold'], 'imbalance_threshold')
        self.migration_cost_seconds = _number(kwargs['migration_cost_seconds'], 'migration_cost_seconds')
        self.payload_bytes = _number(kwargs['payload_bytes'], 'payload_bytes', True)
        self.stall_time_seconds = _number(kwargs['stall_time_seconds'], 'stall_time_seconds')
        self.imbalance_after = _number(kwargs['imbalance_after'], 'imbalance_after')
        self.correctness_passed = _boolean(kwargs['correctness_passed'], 'correctness_passed')
        self.runtime_hook_available = _boolean(kwargs['runtime_hook_available'], 'runtime_hook_available')
        self.manifest_compatible = _boolean(kwargs['manifest_compatible'], 'manifest_compatible')
        if not isinstance(kwargs['ownership_before'], (str, tuple, list, dict)) or not isinstance(kwargs['ownership_after'], (str, tuple, list, dict)):
            raise FeedbackError('ownership must be a serializable value')
        self.ownership_before = kwargs['ownership_before']
        self.ownership_after = kwargs['ownership_after']

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise FeedbackError('sample must be an object')
        return cls(**value)

    def to_dict(self):
        return {key: getattr(self, key) for key in _FIELDS}

class FeedbackDecision(object):
    def __init__(self, sample, triggered, worthy, reason, fallback='none'):
        self.triggered, self.worthy = bool(triggered), bool(worthy)
        self.reason, self.fallback = reason, fallback
        self.actionable = False
        self.advisory = True
        self.non_actionable = True
        self.sample = sample

    def to_dict(self):
        data = self.sample.to_dict()
        data.update({'triggered': self.triggered, 'worthy': self.worthy,
                     'reason': self.reason, 'fallback': self.fallback,
                     'actionable': False, 'advisory': True,
                     'non_actionable': True})
        return data

class FeedbackController(object):
    def __init__(self, history_limit=8, max_cost_seconds=None):
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit <= 0:
            raise FeedbackError('history_limit must be positive integer')
        self.history_limit = history_limit
        self.max_cost_seconds = (float('inf') if max_cost_seconds is None else
                                 _number(max_cost_seconds, 'max_cost_seconds'))
        self.history = []

    def observe(self, sample):
        if not isinstance(sample, FeedbackSample):
            sample = FeedbackSample.from_dict(sample)
        triggered = sample.imbalance_before >= sample.imbalance_threshold
        fallback = 'stable-ownership' if triggered else 'none'
        if not triggered:
            reason = 'below-threshold'
        elif not sample.correctness_passed:
            reason = 'correctness-failed'
        elif not sample.manifest_compatible:
            reason = 'manifest-incompatible'
        elif not sample.runtime_hook_available:
            reason = 'runtime-hook-unavailable'
        elif sample.migration_cost_seconds > self.max_cost_seconds:
            reason = 'cost-overrun'
        elif not self.history:
            reason = 'insufficient-history'
        elif sample.imbalance_after >= sample.imbalance_before:
            reason = 'no-positive-improvement'
        else:
            reason = 'history-supported-improvement'
        worthy = reason == 'history-supported-improvement'
        decision = FeedbackDecision(sample, triggered, worthy, reason, fallback)
        self.history.append(sample)
        del self.history[:-self.history_limit]
        return decision

def decisions_to_json(decisions):
    return json.dumps([item.to_dict() for item in decisions], sort_keys=True, separators=(',', ':')) + '\n'

def decisions_to_csv(decisions):
    output = io.StringIO()
    fields = _FIELDS + ('triggered', 'worthy', 'reason', 'fallback', 'actionable', 'advisory', 'non_actionable')
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for decision in decisions:
        row = decision.to_dict()
        for key in ('ownership_before', 'ownership_after'):
            row[key] = json.dumps(row[key], sort_keys=True, separators=(',', ':'))
        writer.writerow(row)
    return output.getvalue()
