#!/usr/bin/env python3
"""Script entry point `python3 src/wtune_tune.py <case> ...` of the wtune_tune package.

`import wtune_tune` resolves to the package src/wtune_tune/ (a package directory wins over a
same-named module), so this file only runs as a script; the code lives in the package.
"""
import sys

if __name__ == "__main__":
    from wtune_tune.cli import main      # sys.path[0] is src/: the package directory
    sys.exit(main())
