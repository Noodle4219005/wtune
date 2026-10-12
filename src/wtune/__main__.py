"""`python src/wtune ...` (bin/wtune) / `python -m wtune ...`: the wtune command line."""
import os
import sys

if not __package__:                    # run as a directory: import the package from src/
    _pkg = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _pkg]
    sys.path.insert(0, os.path.dirname(_pkg))

from wtune import cli  # noqa: E402

cli.main()
