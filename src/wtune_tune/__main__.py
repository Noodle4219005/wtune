"""`python -m wtune_tune` / `python src/wtune_tune`: the tuner CLI."""
import os
import sys

if __package__ in (None, ""):          # run as a directory: import the package from src/
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(
        os.path.abspath(__file__))]
    sys.path.insert(0, _src)

from wtune_tune.cli import main  # noqa: E402

raise SystemExit(main())
