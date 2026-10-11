"""Golden snapshot of wtune.preflight findings (severity/id/message/evidence/fix, order
included) on the fixture cases of test_wtune_preflight.py: that suite is re-run in a subprocess
under tests/golden_preflight_plugin.py, which records what every check_*/run_checks call
produced. Pinned before the round-3 tidy-up split the long check functions.

Regenerate (only for an intended behaviour change):
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_preflight.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "fixtures", "golden", "preflight", "findings.json")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"


def _record(tmp_path):
    out = tmp_path / "rec.json"
    env = dict(os.environ, WTUNE_GOLDEN_OUT=str(out), PYTHONPATH=HERE + os.pathsep
               + os.path.join(os.path.dirname(HERE), "src"))
    env.pop("WTUNE_GOLDEN_REGEN", None)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "-p", "golden_preflight_plugin", "-o", "addopts=",
                    os.path.join(HERE, "test_wtune_preflight.py")],
                   env=env, capture_output=True, text=True, timeout=600, cwd=str(tmp_path))
    with open(out) as fh:
        return json.load(fh)


def test_preflight_findings_match_golden(tmp_path):
    got = _record(tmp_path)
    assert got["exit"] == 0
    if REGEN:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(got, fh, indent=1, sort_keys=True)
            fh.write("\n")
    with open(GOLDEN) as fh:
        want = json.load(fh)
    assert got == want
