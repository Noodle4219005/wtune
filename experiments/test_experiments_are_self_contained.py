"""The experiment harnesses must stay runnable from this checkout alone.

They live here because the scratch tree they used to live in gets rm -rf'd and reused, and an
external dependency has already vanished once: scripts/vendor/wrf-adios2-to-netcdf4/convert.py
disappeared and every MODE=physics job then produced no netCDF and therefore no fidelity verdict,
silently (smoke jobs 1076771-1076776). These tests are cheap and catch the same class of rot.
"""
import os
import py_compile
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SBATCH = ["rk_alternation.sbatch", "dt_tradeoff.sbatch", "bitrepro.sbatch"]
PY = ["segnl.py", "bitcmp.py", "summarize.py", "dt_summarize.py", "bitrms.py",
      "summarize_bitrepro.py"]


def test_every_vendored_script_is_present():
    for name in SBATCH + PY:
        assert os.path.exists(os.path.join(HERE, name)), name


def test_shell_harnesses_parse():
    for name in SBATCH:
        r = subprocess.run(["bash", "-n", os.path.join(HERE, name)],
                           capture_output=True, text=True)
        assert r.returncode == 0, "%s: %s" % (name, r.stderr)


def test_python_harnesses_compile():
    for name in PY:
        py_compile.compile(os.path.join(HERE, name), doraise=True)


def test_no_runtime_dependency_on_another_repo():
    """WTune must not reach into ~/SC26-WRF (or any sibling checkout) at run time. A path may appear
    in a comment recording where vendored code came from; it may not appear in executable code."""
    offenders = []
    for name in SBATCH + PY:
        with open(os.path.join(HERE, name)) as fh:
            for n, line in enumerate(fh, 1):
                code = line.split("#", 1)[0]
                if "SC26-WRF" in code:
                    offenders.append("%s:%d" % (name, n))
    assert not offenders, "executable reference to another checkout: %s" % offenders


def test_readme_records_a_checksum_for_every_vendored_script():
    """A vendored copy without a recorded checksum cannot be told apart from a silent divergence."""
    with open(os.path.join(HERE, "README.md")) as fh:
        readme = fh.read()
    for name in SBATCH + PY:
        assert "`%s`" % name in readme, "%s is not listed in README.md" % name
    assert len(re.findall(r"`[0-9a-f]{16}`", readme)) >= len(SBATCH + PY)
