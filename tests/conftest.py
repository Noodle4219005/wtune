"""Repo-wide pytest bootstrap for the src/ layout.

Puts src/ (and the subdirectories whose modules are imported as top-level names by their own
sibling code, e.g. `import decomp_search` from within src/predictor) onto sys.path, so moved
test files keep working without each one re-deriving this from its own __file__. Individual test
files' own `sys.path.insert(...)` calls (left in place; several now point at a directory that no
longer contains the target modules) are harmless: they just fail to shadow anything, and Python
falls through to the entries this file adds.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")

for _p in (_SRC, os.path.join(_SRC, "predictor"), os.path.join(_SRC, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

del _p
