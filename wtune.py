#!/usr/bin/env python3
"""wtune.py -- thin entry point. The real CLI lives in src/wtune_cli.py; the rest of the
library/program code lives under src/ (predictor/, tools/, analysis/, wtune_*.py, ...). This
shim exists only so `python3 wtune.py ...` and bin/wtune keep working unchanged from the repo
root after the src/ layout move."""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import wtune_cli

if __name__ == "__main__":
    wtune_cli.main()
