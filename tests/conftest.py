"""Repo-wide pytest bootstrap for the src/ layout.

Puts src/ (and the subdirectories whose modules are imported as top-level names by their own
sibling code, e.g. `import decomp_search` from within src/wtune/predictor) onto sys.path, so moved
test files keep working without each one re-deriving this from its own __file__. Individual test
files' own `sys.path.insert(...)` calls (left in place; several now point at a directory that no
longer contains the target modules) are harmless: they just fail to shadow anything, and Python
falls through to the entries this file adds.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# the developer's own machine settings must not reach the tests (or the commands they spawn)
os.environ["WTUNE_MACHINE_FILE"] = os.path.join(os.sep, "nonexistent", "wtune-machine.yaml")
# nor the site's WRF tree: wtune check reads the build it names (f1-arm job 15557)
os.environ.pop("WRF_TREE", None)
# recipe figure sets in one process: tests stub them with local closures (test_post_recipe_parallel)
os.environ["WTUNE_POST_WORKERS"] = "1"


@pytest.fixture(autouse=True)
def _no_account_ledgers(monkeypatch, tmp_path_factory):
    """A tune reads the account's older ledgers (~/wtune-runs/campaign) and resolves the probes'
    WRF tree by sourcing env.sh: neither may reach a test on a cluster login node."""
    import wtune.tools.scc_sweep
    import wtune.tune.cli
    monkeypatch.delenv("WTUNE_CAMPAIGN_OUT", raising=False)
    monkeypatch.setattr(wtune.tools.scc_sweep, "HISTORY_LEDGER_DIR",
                        str(tmp_path_factory.getbasetemp() / "no-history-ledgers"))
    monkeypatch.setattr(wtune.tune.cli, "_build_id", lambda args: "")


@pytest.fixture(autouse=True)
def _no_real_slurm(monkeypatch):
    """On a cluster the node preflight would list the real nodes with sinfo and srun onto
    each of them; a test that wants a preflight passes its own runner."""
    import wtune.tune.cli
    import wtune.tune.preflight

    def absent(argv, timeout_s):
        raise FileNotFoundError(argv[0])
    monkeypatch.setattr(wtune.tune.preflight, "default_preflight_runner", absent)
    monkeypatch.setattr(wtune.tune.cli, "default_preflight_runner", absent)


def swap_wtune_module(monkeypatch, name, module):
    """Make `from wtune import <name>` yield `module`; None makes the import fail."""
    import wtune
    monkeypatch.setitem(sys.modules, "wtune." + name, module)
    if module is None:
        monkeypatch.delattr(wtune, name, raising=False)
    else:
        monkeypatch.setattr(wtune, name, module, raising=False)
