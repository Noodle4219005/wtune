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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def swap_wtune_module(monkeypatch, name, module):
    """Make `from wtune import <name>` yield `module`; None makes the import fail."""
    import wtune
    monkeypatch.setitem(sys.modules, "wtune." + name, module)
    if module is None:
        monkeypatch.delattr(wtune, name, raising=False)
    else:
        monkeypatch.setattr(wtune, name, module, raising=False)
