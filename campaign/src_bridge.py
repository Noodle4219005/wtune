#!/usr/bin/env python3
"""Lazy access to the repo's src/ modules from campaign/ (never at import: see configs.py).

Stdlib only, Python 3.6.
"""
import os
import sys
from types import ModuleType

CAMPAIGN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CAMPAIGN_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")


def add_src_to_path() -> None:
    """Make `import wtune_*` find this checkout's src/."""
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)


_BP_BY_PATH = {}


def _import_wtune_bp() -> ModuleType:
    """src/wtune/bp.py (the ADIOS2 converter), from $WTUNE_SRC, else this checkout's src/.

    It runs on the stdlib and calls the adios2 converters in envs/adios2's interpreter.
    $WTUNE_SRC's copy is loaded by path: `wtune` may already be this checkout's package.
    """
    src = os.environ.get("WTUNE_SRC")
    path = os.path.join(src, "wtune", "bp.py") if src else None
    if path and os.path.isfile(path) and os.path.abspath(src) != os.path.abspath(SRC_DIR):
        if path not in _BP_BY_PATH:
            import importlib.util
            spec = importlib.util.spec_from_file_location("wtune_bp_from_wtune_src", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _BP_BY_PATH[path] = mod
        return _BP_BY_PATH[path]
    add_src_to_path()
    import wtune.bp
    return wtune.bp