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


def _import_wtune_bp() -> ModuleType:
    """src/wtune_bp (the ADIOS2 converter), from $WTUNE_SRC, else this checkout's src/.

    It runs on the stdlib and calls the adios2 converters in envs/adios2's interpreter.
    """
    for src in (os.environ.get("WTUNE_SRC"), SRC_DIR):
        if src and os.path.isfile(os.path.join(src, "wtune_bp.py")):
            for p in (os.path.join(src, "tools"), src):
                if p not in sys.path:
                    sys.path.insert(0, p)
            break
    import wtune_bp
    return wtune_bp
