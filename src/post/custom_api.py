"""post/custom_api.py -- the API `wtune post custom` (decision 0058 section 2) hands to every
analysis/*.py script it runs: project paths, the run list, an open_wrfout() helper, and
save_figure()/save_table() that register a script's output so post/custom.py can fold it into
one report. Stdlib + netCDF4 -- matplotlib and wrf-python are only imported lazily (via the
`wrf` proxy and subplots()), so a script that only reads CSVs never needs them installed.
"""
from __future__ import annotations

import csv
import glob
import os
import re

WRF_FRAME_RE = re.compile(r"wrfout_d(\d+)_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)$")


def subplots(*args, **kwargs):
    """matplotlib.pyplot.subplots(), Agg backend forced -- the one matplotlib entry point a
    script needs; it never has to import matplotlib itself."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(*args, **kwargs)


class _LazyWrf:
    """`from post import custom_api as api; api.wrf.getvar(...)` -- wrf-python imported only
    on first attribute access, not at post/custom_api.py import time."""

    def __getattr__(self, name):
        import wrf
        return getattr(wrf, name)


wrf = _LazyWrf()


class Context:
    """One instance per analysis/*.py script run, created by post/custom.py."""

    def __init__(self, project_dir, script_name, out_dir, run_dirs):
        self.project_dir = project_dir
        self.script_name = script_name
        self.out_dir = out_dir
        self._run_dirs = list(run_dirs)   # [(run_id, path), ...]
        self.outputs = []                 # [(kind, path, script_name), ...]
        self.logs = []

    # -- project/run discovery ---------------------------------------------------------
    def runs(self):
        """[(run_id, path), ...] -- every wrfout-bearing run directory this project has
        (wtune_project.discover_runs's own result)."""
        return list(self._run_dirs)

    def finest_domain(self, run_dir):
        from post.extract import frame_index
        doms = set(d for d, _t in frame_index(run_dir)) if os.path.isdir(run_dir) else set()
        return max(doms) if doms else 1

    def list_frames(self, run_dir, domain=1):
        from post.extract import list_frames
        return list_frames(run_dir, domain=domain)

    def open_wrfout(self, path):
        """A netCDF4.Dataset the caller must close (or use as a context manager)."""
        import netCDF4
        return netCDF4.Dataset(path)

    def find_run_track_csv(self, run_id):
        """post/tc.py's own output convention: post/<id>/tc/<run_id>*.csv under this
        project's post/ tree, most recent post/<id>/ first. None if not found."""
        post_root = os.path.join(self.project_dir, "post")
        if not os.path.isdir(post_root):
            return None
        for post_id in sorted(os.listdir(post_root), reverse=True):
            for pattern in ("%s*track*.csv" % run_id, "%s*adeck*.csv" % run_id, "%s*.csv" % run_id):
                hits = sorted(glob.glob(os.path.join(post_root, post_id, "tc", pattern)))
                if hits:
                    return hits[0]
        return None

    def read_csv(self, path):
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    # -- output registration -------------------------------------------------------------
    def save_figure(self, fig, filename, close=True):
        import matplotlib.pyplot as plt
        path = os.path.join(self.out_dir, filename)
        fig.savefig(path)
        if close:
            plt.close(fig)
        self.outputs.append(("figure", path, self.script_name))
        return path

    def save_table(self, filename, header, rows):
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        self.outputs.append(("table", path, self.script_name))
        return path

    def log(self, message):
        self.logs.append(str(message))
