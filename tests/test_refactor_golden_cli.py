#!/usr/bin/env python3
"""Characterization (golden) tests for the top-level CLI (src/wtune/cli.py, wtune/cli.py).

Pins what a user sees from the argparse front end -- the top-level and every subcommand's
--help text, the exit code and stdout/stderr of representative invalid or cheap invocations --
so a restructuring of wtune_cli.main (parser construction, dispatch) can be checked to be
behaviour-preserving. The golden file was generated from the UNCHANGED code (tag v0.8.2) with:

    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_cli.py

Every invocation runs in a fresh subprocess with a fixed terminal width (argparse wraps help to
$COLUMNS) and an empty temporary cwd, and only references fixed non-existent paths so the
captured text contains nothing run-specific.
"""

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "fixtures", "golden", "cli", "cli_snapshots.json")

NL = "/nonexistent/wtune-golden/namelist.input"
CASE = "/nonexistent/wtune-golden/case"

SUBCOMMANDS = [
    "physics", "screen", "machine-profile", "selftest", "compliance",
    "predict", "tune", "config", "metdata", "check", "calibrate", "post", "prep", "project",
    "analyze", "predict-rebuild", "watch", "run", "advise", "calc", "scenario",
]

# (case id, entry script relative to REPO, argv)
CASES = [("help-top", "src/wtune", ["--help"]),
         ("help-top-direct", "src/wtune/cli.py", ["--help"])]
CASES += [("help-" + c, "src/wtune", [c, "--help"]) for c in SUBCOMMANDS]
CASES += [
    ("no-command", "src/wtune", []),
    ("unknown-command", "src/wtune", ["bogus"]),
    ("version", "src/wtune", ["--version"]),
    ("physics-bad-compliance", "src/wtune", ["physics", "--namelist", NL, "--compliance", "bogus"]),
    ("physics-no-file", "src/wtune", ["physics", "--namelist", NL, "--no-probe"]),
    ("screen-no-file", "src/wtune", ["screen", "--namelist", NL, "--no-probe"]),
    ("machine-profile-no-file", "src/wtune", ["machine-profile", "/nonexistent/wtune-golden/m.json"]),
    ("selftest", "src/wtune", ["selftest"]),
    ("compliance-classify", "src/wtune", ["compliance", "classify", "numtiles"]),
    ("compliance-no-args", "src/wtune", ["compliance"]),
    ("predict-missing", "src/wtune", ["predict"]),
    ("predict-no-dir", "src/wtune", ["predict", CASE, "--nodes", "1"]),
    ("calc-units", "src/wtune", ["calc", "units", "kt-ms", "10"]),
    ("calc-units-direct", "src/wtune/cli.py", ["calc", "units", "kt-ms", "10"]),
    ("calc-no-args", "src/wtune", ["calc"]),
    ("advise-no-args", "src/wtune", ["advise"]),
    ("tune-no-args", "src/wtune", ["tune"]),
    ("config-no-args", "src/wtune", ["config"]),
    ("metdata-no-args", "src/wtune", ["metdata"]),
    ("check-no-args", "src/wtune", ["check"]),
    ("calibrate-no-args", "src/wtune", ["calibrate"]),
    ("post-no-args", "src/wtune", ["post"]),
    ("prep-no-args", "src/wtune", ["prep"]),
    ("project-no-args", "src/wtune", ["project"]),
    ("analyze-no-args", "src/wtune", ["analyze"]),
    ("predict-rebuild-no-args", "src/wtune", ["predict-rebuild"]),
    ("watch-no-args", "src/wtune", ["watch"]),
    ("run-no-args", "src/wtune", ["run"]),
    ("scenario-no-args", "src/wtune", ["scenario"]),
]


def _invoke(script, argv, cwd):
    env = dict(os.environ, COLUMNS="80", LC_ALL="C", PYTHONHASHSEED="0")
    env.pop("WTUNE_GOLDEN_REGEN", None)
    p = subprocess.run([sys.executable, os.path.join(REPO, script)] + argv, cwd=cwd, env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": _normalize_traceback(p.stderr)}


def _normalize_traceback(stderr):
    """An uncaught exception is pinned by its final line (type + message) only: the frames carry
    wtune/cli.py line numbers and source excerpts, which a restructuring legitimately moves."""
    if "Traceback (most recent call last):" not in stderr:
        return stderr
    head = stderr.split("Traceback (most recent call last):")[0]
    last = [ln for ln in stderr.splitlines() if ln.strip()][-1]
    return head + "Traceback (most recent call last):\n  ...\n" + last + "\n"


def _load():
    with open(GOLDEN) as fh:
        return json.load(fh)


if os.environ.get("WTUNE_GOLDEN_REGEN") == "1":
    def test_regenerate_golden(tmp_path):
        out = {}
        for cid, script, argv in CASES:
            d = tmp_path / cid
            d.mkdir()
            out[cid] = dict(_invoke(script, argv, str(d)), script=script, argv=argv)
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
            fh.write("\n")


def test_golden_covers_every_case():
    assert sorted(_load()) == sorted(cid for cid, _, _ in CASES)


@pytest.mark.parametrize("cid,script,argv", CASES, ids=[c[0] for c in CASES])
def test_cli_matches_golden(cid, script, argv, tmp_path):
    want = _load()[cid]
    got = _invoke(script, argv, str(tmp_path))
    assert (got["rc"], got["stdout"], got["stderr"]) == (want["rc"], want["stdout"], want["stderr"])
    assert os.listdir(tmp_path) == [], "invocation wrote into its cwd"
