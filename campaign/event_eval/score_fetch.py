"""Score agent fetches (Part 2 of the wtune-event skill) against reference conversions.

    python3 campaign/event_eval/score_fetch.py OUT_DIR --refs REF_DIR [--tasks YAML] [--csv results.csv]

OUT_DIR/<task>/project/input/obs/<name>/ is one run (run_fetch_pi.sh); REF_DIR/<name>/ holds the
reference wtune-obs/1 folder made by hand. A run passes when `obs-check` passes, the kind is
right, every requested time is there, and the values agree with the reference:
- gridded_precip: per hour, mean |diff| <= 0.05 mm at the reference cells (catches a rate left
  as mm/h is fine, a start/end time label swap is not: the hour is then missing or off);
- radar: at reference echo cells (>= 15 dBZ) 80 % within 6 dBZ (the reference is max-reduced
  to 0.02 deg), and cells the reference marks out of radar range are NaN in at least half;
- satellite_ir: at least 95 % of the reference cells valid, median |diff| <= 1 K;
- stations_extra: at least 90 % of the reference (station, time, var) rows, values within 0.11;
- rain_gauges: every reference gauge matched one to one (by id within 3 km, else the nearest
  unused gauge within 3 km), every reference hour within 0.05 mm (keyed by end time), and any
  longer row equal to the reference sum over its interval.
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
from wtune.post import agent_obs  # noqa: E402

TASKS = os.path.join(REPO, "tests", "fixtures", "fetch", "tasks.yaml")
FMT = "%Y-%m-%d %H:%M"


def _folder(run_dir, name):
    base = os.path.join(run_dir, "project", "input", "obs")
    want = os.path.join(base, name)
    if os.path.isdir(want):
        return want, ""
    found = [n for n, _d, _p in agent_obs.datasets(os.path.join(run_dir, "project"))]
    return (os.path.join(base, found[0]), "folder named %s" % found[0]) if found else (None, "")


def _sample(lat, lon, field, qlat, qlon):
    """field at the nearest cell of (lat, lon) for each query point; NaN outside the grid."""
    i = np.clip(np.searchsorted(lat, qlat), 1, len(lat) - 1)
    i -= (qlat - lat[i - 1]) < (lat[i] - qlat)
    j = np.clip(np.searchsorted(lon, qlon), 1, len(lon) - 1)
    j -= (qlon - lon[j - 1]) < (lon[j] - qlon)
    out = field[i, j]
    dy = np.abs(lat[1] - lat[0]) if len(lat) > 1 else 1
    dx = np.abs(lon[1] - lon[0]) if len(lon) > 1 else 1
    outside = (qlat < lat[0] - dy) | (qlat > lat[-1] + dy) | (qlon < lon[0] - dx) | (qlon > lon[-1] + dx)
    return np.where(outside, np.nan, out)


def _ref_points(ref_path, var, bbox):
    lat, lon, a = agent_obs.read_grid(ref_path, var)
    la = (lat >= bbox[0]) & (lat <= bbox[1])
    lo = (lon >= bbox[2]) & (lon <= bbox[3])
    LA, LO = np.meshgrid(lat[la], lon[lo], indexing="ij")
    return LA.ravel(), LO.ravel(), a[np.ix_(la, lo)].ravel()


def _grids(task, folder, ref):
    notes, ok = [], True
    t0, t1 = (datetime.strptime(x, FMT) for x in task["window"])
    cand, refi = agent_obs.grid_index(folder), agent_obs.grid_index(ref)
    ends = [t0 + timedelta(hours=h) for h in range(1, int((t1 - t0).total_seconds() // 3600) + 1)]
    got = {t for (a, t) in cand if a == 1}
    missing = [t for t in ends if t not in got]
    if missing:
        ok = False
        shifted = all((t - timedelta(hours=1)) in got for t in ends)
        notes.append("hours missing %d/%d%s" % (len(missing), len(ends),
                                                 " (labelled by period start)" if shifted else ""))
    for t in ends:
        if t not in got:
            continue
        qla, qlo, rv = _ref_points(refi[(1, t)], "precip", task["bbox"])
        lat, lon, a = agent_obs.read_grid(cand[(1, t)], "precip")
        cv = _sample(lat, lon, a, qla, qlo)
        m = np.isfinite(cv) & np.isfinite(rv)
        if m.mean() < 0.9 or np.mean(np.abs(cv[m] - rv[m])) > 0.05:
            ok = False
            notes.append("%s: mean |diff| %.3f mm on %.0f%% of cells" % (
                t.strftime("%H"), np.mean(np.abs(cv[m] - rv[m])) if m.any() else np.nan, 100 * m.mean()))
            break
    return ok, notes


def _snapshots(task, folder, ref):
    kind = task["kind"]
    var = agent_obs.SNAPSHOTS[kind][1]
    notes, ok = [], True
    cand, refi = agent_obs.snapshot_index(folder, kind), agent_obs.snapshot_index(ref, kind)
    for ts in task["times"]:
        t = datetime.strptime(ts, FMT)
        if t not in cand:
            ok = False
            notes.append("%s missing" % ts)
            continue
        qla, qlo, rv = _ref_points(refi[t], var, task["bbox"])
        lat, lon, a = agent_obs.read_grid(cand[t], var)
        cv = _sample(lat, lon, a, qla, qlo)
        if kind == "radar":
            echo = np.isfinite(rv) & (rv >= 15)
            close = np.isfinite(cv[echo]) & (np.abs(cv[echo] - rv[echo]) <= 6)
            frac = close.mean() if echo.any() else 0.0
            outside = ~np.isfinite(rv)
            masked = (~np.isfinite(cv[outside])).mean() if outside.any() else 1.0
            if frac < 0.8:
                ok = False
                notes.append("%s: %.0f%% of echo cells within 6 dBZ" % (ts, 100 * frac))
            if masked < 0.5:
                ok = False
                notes.append("%s: only %.0f%% of out-of-range cells are NaN" % (ts, 100 * masked))
        else:
            valid = np.isfinite(rv)
            have = np.isfinite(cv[valid])
            med = np.nanmedian(np.abs(cv[valid] - rv[valid])) if have.any() else np.nan
            if have.mean() < 0.95 or not med <= 1.0:
                ok = False
                notes.append("%s: %.0f%% valid, median |diff| %.2f K" % (ts, 100 * have.mean(), med))
    return ok, notes


def _stations(task, folder, ref):
    t0, t1 = (datetime.strptime(x, FMT) for x in task["window"])
    want = set(task["stations"])

    def rows(path):
        return {(r["id"], r["time"], r["var"]): r["value"] for r in agent_obs.read_extra(path)
                if r["id"] in want and t0 <= r["time"] <= t1}
    try:
        cand = rows(os.path.join(folder, agent_obs.EXTRA_FILE))
    except (OSError, ValueError) as e:
        return False, ["unreadable: %s" % e]
    refr = rows(os.path.join(ref, agent_obs.EXTRA_FILE))
    hit = [k for k in refr if k in cand]
    good = [k for k in hit if abs(cand[k] - refr[k]) <= 0.11]
    recall = len(good) / len(refr) if refr else 0.0
    notes = ["%d/%d reference rows matched, %d extra rows" % (len(good), len(refr),
                                                               len(set(cand) - set(refr)))]
    return recall >= 0.9, notes


def _gid(x):
    x = x.strip().upper()
    return (x.lstrip("0") or "0") if x.isdigit() else x


def _km(a, b):
    return 111.0 * np.hypot(a["lat"] - b["lat"], (a["lon"] - b["lon"]) * np.cos(np.radians(b["lat"])))


def _gauges(task, folder, ref):
    """Every reference gauge, matched one to one (by id within 3 km, else the nearest unused gauge
    within 3 km), must have every reference hour within 0.05 mm; a row longer than an hour (the
    figures use a gauge's longest period) must match the reference sum over its interval."""
    t0, t1 = (datetime.strptime(x, FMT) for x in task["window"])

    def read(path):
        out = {}
        for r in agent_obs.read_gauges(path):
            if t0 < r["end"] <= t1 and r["start"] >= t0 and np.isfinite(r["precip_mm"]):
                g = out.setdefault(_gid(r["id"]), dict(lat=r["lat"], lon=r["lon"], v={}))
                g["v"][(r["start"], r["end"])] = r["precip_mm"]
        return out
    try:
        cand = read(os.path.join(folder, agent_obs.GAUGE_FILE))
    except (OSError, ValueError) as e:
        return False, ["unreadable: %s" % e]
    refg = read(os.path.join(ref, agent_obs.GAUGE_FILE))
    if not refg:
        return False, ["the reference has no hourly rows in the window"]
    ok, notes, used = True, [], set()
    for gid, g in sorted(refg.items()):
        k = gid if gid in cand and _km(cand[gid], g) <= 3.0 else None
        if k is None:
            near = sorted((_km(c, g), cid) for cid, c in cand.items() if cid not in used)
            k = near[0][1] if near and near[0][0] <= 3.0 else None
        if k is None or k in used:
            ok = False
            notes.append("%s: not found" % gid)
            continue
        used.add(k)
        c = cand[k]["v"]
        hours = {e: v for (s0, e), v in g["v"].items() if e - s0 == timedelta(hours=1)}
        cand_h = {e: v for (s0, e), v in c.items() if e - s0 == timedelta(hours=1)}
        good = sum(1 for e, v in hours.items() if e in cand_h and abs(cand_h[e] - v) <= 0.05)
        if good < len(hours):
            ok = False
            notes.append("%s: %d/%d hours match" % (gid, good, len(hours)))
        for (s0, e), v in c.items():
            span = [hours.get(s0 + timedelta(hours=h + 1)) for h in range(int((e - s0).total_seconds() // 3600))]
            if len(span) > 1 and (None in span or abs(sum(span) - v) > 0.05 * len(span)):
                ok = False
                notes.append("%s: %s..%s total %.1f does not match" % (gid, s0, e, v))
                break
    notes.append("%d reference gauges, %d in the run" % (len(refg), len(cand)))
    return ok, notes


def score_run(run_dir, task, refs):
    r = dict(found=False, check=False, kind=False, values=False, passed=False, note="")
    folder, note = _folder(run_dir, task["name"])
    notes = [note] if note else []
    if not folder:
        r["note"] = "no wtune-obs/1 folder"
        return r
    r["found"] = True
    err, _w = agent_obs.check(folder)
    r["check"] = not err
    if err:
        notes.append("obs-check: " + err[0])
    prov = agent_obs.read_provenance(folder) or {}
    r["kind"] = prov.get("kind") == task["kind"]
    ref = os.path.join(refs, task["name"])
    try:
        if task["kind"] == "gridded_precip":
            ok, n = _grids(task, folder, ref)
        elif task["kind"] in agent_obs.SNAPSHOTS:
            ok, n = _snapshots(task, folder, ref)
        elif task["kind"] == "rain_gauges":
            ok, n = _gauges(task, folder, ref)
        else:
            ok, n = _stations(task, folder, ref)
    except Exception as e:  # noqa: BLE001 -- a malformed run is a failed run, not a crash
        ok, n = False, ["scoring error: %s: %s" % (type(e).__name__, e)]
    r["values"] = ok
    notes += n
    r["passed"] = r["check"] and r["kind"] and r["values"]
    r["note"] = "; ".join(notes)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    tasks = yaml.safe_load(open(a.tasks))
    rows = []
    for name, task in tasks.items():
        d = os.path.join(a.out_dir, name)
        if not os.path.isdir(d):
            continue
        r = score_run(d, task, a.refs)
        rows.append(dict(task=name, **r))
        print("%-10s %-4s check=%-5s kind=%-5s values=%-5s %s" % (
            name, "PASS" if r["passed"] else "FAIL", r["check"], r["kind"], r["values"], r["note"]))
    print("passed %d / %d" % (sum(r["passed"] for r in rows), len(rows)))
    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
