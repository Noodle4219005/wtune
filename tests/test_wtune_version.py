"""The version must be declared once and match the changelog.

A version that lives only in a git tag is invisible to a checkout, and a changelog whose top
section drifts from the code is worse than no changelog: it is a claim nobody rechecks.
"""
import os
import re

import wtune_version

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tests/ -> repo root
CHANGELOG = os.path.join(HERE, "CHANGELOG.md")


def _top_version():
    with open(CHANGELOG) as fh:
        for line in fh:
            m = re.match(r"^## (\d+\.\d+\.\d+) ", line)
            if m:
                return m.group(1)
    return None


def test_version_is_three_numeric_parts():
    assert re.match(r"^\d+\.\d+\.\d+$", wtune_version.__version__), wtune_version.__version__


def test_changelog_top_section_matches_the_declared_version():
    assert _top_version() == wtune_version.__version__


def test_changelog_versions_are_unique_and_descending():
    with open(CHANGELOG) as fh:
        found = re.findall(r"^## (\d+\.\d+\.\d+) ", fh.read(), re.M)
    assert found, "CHANGELOG.md has no version section"
    keys = [tuple(int(p) for p in v.split(".")) for v in found]
    assert len(set(keys)) == len(keys), "duplicate version section: %s" % found
    assert keys == sorted(keys, reverse=True), "versions out of order: %s" % found


def test_cli_reports_the_same_version():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, os.path.join(HERE, "wtune.py"), "--version"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert wtune_version.__version__ in (r.stdout + r.stderr)
