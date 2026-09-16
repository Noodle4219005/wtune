#!/usr/bin/env python3
"""Does the best decomposition change as the simulation advances?

Reads per-step wall times straight out of each run's rsl.error.0000 ("Timing for main: time ... on
domain N: X elapsed seconds"), splits them into windows, and ranks the configs inside each window.
If the ranking is the same in every window, one measurement at the start of a run is enough and a
config switch at a restart boundary buys nothing; if it changes, segmented configs are worth testing.

Only domain 1 lines are used by default: a nest runs parent_time_step_ratio steps per parent step,
so mixing domains makes "step index" mean two different things.

Usage:
  phase_ranking.py --run-dir DIR [DIR ...] [--windows 3] [--domain 1] [--skip-first 1] [--tsv OUT]
"""
import argparse
import os
import re
import statistics
import sys

TIMING_RE = re.compile(
    r"Timing for main: time \S+ on domain\s+(?P<dom>\d+):\s+(?P<sec>[0-9.]+) elapsed seconds")
CONFIG_RE = re.compile(r"^(?P<case>[^_]+)_(?P<cid>.+)_r(?P<rep>\d+)_(?P<job>\d+)$")


def per_step_times(run_dir, domain=1):
    path = os.path.join(run_dir, "rsl.error.0000")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = TIMING_RE.search(line)
            if m and int(m.group("dom")) == domain:
                out.append(float(m.group("sec")))
    return out


def windows(values, n_windows, skip_first=1):
    """n_windows equal slices of `values` after dropping the first `skip_first` steps (step 1 carries
    first-touch/allocation cost -- the campaign ledger excludes it the same way)."""
    vals = values[skip_first:]
    if len(vals) < n_windows:
        return []
    size = len(vals) // n_windows
    return [vals[i * size:(i + 1) * size] for i in range(n_windows)]


def rank_by_window(runs, n_windows, domain=1, skip_first=1):
    """runs: {label: run_dir}. Returns (rows, order_per_window) where rows are
    (label, [median per window]) and order_per_window is the label order in each window."""
    rows, per_window = [], [dict() for _ in range(n_windows)]
    for label, run_dir in sorted(runs.items()):
        steps = per_step_times(run_dir, domain)
        w = windows(steps, n_windows, skip_first)
        if not w:
            continue
        meds = [statistics.median(x) for x in w]
        rows.append((label, meds, len(steps)))
        for i, m in enumerate(meds):
            per_window[i][label] = m
    order = [[lbl for lbl, _ in sorted(d.items(), key=lambda kv: kv[1])] for d in per_window]
    return rows, order


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--domain", type=int, default=1)
    ap.add_argument("--skip-first", type=int, default=1)
    ap.add_argument("--tsv")
    a = ap.parse_args(argv)

    runs = {}
    for d in a.run_dir:
        m = CONFIG_RE.match(os.path.basename(d.rstrip("/")))
        runs[m.group("cid") if m else os.path.basename(d.rstrip("/"))] = d
    rows, order = rank_by_window(runs, a.windows, a.domain, a.skip_first)
    if not rows:
        print("no per-step timings found (checked rsl.error.0000 in each --run-dir)")
        return 1

    head = ["config", "n_steps"] + ["w%d_median" % (i + 1) for i in range(a.windows)] + ["w1->wN_pct"]
    lines = ["\t".join(head)]
    for label, meds, n in sorted(rows, key=lambda r: r[1][0]):
        drift = 100 * (meds[-1] / meds[0] - 1)
        lines.append("\t".join([label, str(n)] + ["%.5f" % m for m in meds] + ["%+.1f" % drift]))
    lines.append("")
    for i, o in enumerate(order, 1):
        lines.append("window %d ranking: %s" % (i, " < ".join(o)))
    same = all(o == order[0] for o in order)
    lines.append("")
    lines.append("RANKING STABLE: %s -- %s" % (
        same,
        "one measurement is enough; a config switch at a restart boundary cannot help" if same else
        "the best config differs between windows; segmented configs are worth measuring"))
    if not same:
        best = [o[0] for o in order]
        lines.append("best per window: %s" % " -> ".join(best))
    print("\n".join(lines))
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
