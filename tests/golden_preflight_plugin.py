"""pytest plugin for test_refactor_golden_preflight.py: records, per test of
test_wtune_preflight.py, the findings (severity/id/message/evidence/fix, in order) that every
wtune.preflight.check_* / run_checks call produced. Writes JSON to $WTUNE_GOLDEN_OUT."""
import inspect
import json
import os
import re

import pytest

from wtune import preflight as pf

_RECORDS = {}
_CURRENT = [None]
_PYT = re.compile(r"/\S*?pytest-of-[^/\s]+/pytest-\d+/")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm(text):
    return _PYT.sub("<PYT>/", text).replace(_REPO, "<REPO>")


def _dump(findings):
    return [{k: _norm(getattr(f, k)) for k in ("severity", "id", "message", "evidence", "fix")}
            for f in findings]


def _wrap_check(name, fn):
    sig = inspect.signature(fn)

    def wrapper(*a, **k):
        findings = sig.bind(*a, **k).arguments["findings"]
        n0 = len(findings)
        try:
            return fn(*a, **k)
        finally:
            _RECORDS.setdefault(_CURRENT[0], []).append({"call": name,
                                                         "findings": _dump(findings[n0:])})
    wrapper.__wrapped__ = fn
    return wrapper


def _wrap_run(fn):
    def wrapper(*a, **k):
        out = fn(*a, **k)
        _RECORDS.setdefault(_CURRENT[0], []).append({"call": "run_checks",
                                                     "findings": _dump(out)})
        return out
    return wrapper


def pytest_configure(config):
    for name in dir(pf):
        fn = getattr(pf, name)
        if name.startswith("check_") and inspect.isfunction(fn):
            setattr(pf, name, _wrap_check(name, fn))
    pf.run_checks = _wrap_run(pf.run_checks)


@pytest.fixture(autouse=True)
def _track(request):
    _CURRENT[0] = request.node.nodeid
    yield


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["WTUNE_GOLDEN_OUT"], "w") as fh:
        json.dump({"exit": int(exitstatus), "records": _RECORDS}, fh, indent=1, sort_keys=True)
        fh.write("\n")
