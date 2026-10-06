#!/usr/bin/env python3
"""Script entry point `python3 src/wtune_run.py CASE_DIR ...` of the wtune_run package.

`import wtune_run` resolves to the package src/wtune_run/ (a package directory wins over a
same-named module), so this file only runs as a script; the code lives in the package.
"""
import sys

if __name__ == "__main__":
    from wtune_run.cli import main      # sys.path[0] is src/: the package directory
    sys.exit(main())
