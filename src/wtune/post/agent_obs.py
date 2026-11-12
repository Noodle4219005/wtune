"""post/agent_obs.py -- observations an agent fetched (the `agent_fetch` rows of `wtune event
plan`), kept in WTune's standard format so the recipes can read them like their own data.

<project>/input/obs/<name>/ holds one dataset:

  provenance.json   format "wtune-obs/1", kind, dataset, provider, url, license, citation,
                    fetched_utc, files {relative path: sha256}
  gridded_precip:   precip_YYYYMMDDHH_NNh.nc -- the accumulation over the NN hours ending at
                    YYYYMMDDHH UTC; 1-D `lat` (degrees_north) and `lon` (degrees_east), 2-D
                    `precip(lat, lon)` in mm, missing cells NaN or _FillValue
  rain_gauges:      gauges.csv -- id,lat,lon,elev_m,start_utc,end_utc,precip_mm with times
                    "YYYY-MM-DD HH:MM" in UTC, one row per gauge and accumulation period
  radar:            refl_YYYYMMDDHHMM.nc -- composite (or lowest-tilt mosaic) reflectivity at
                    that UTC time, `refl(lat, lon)` in dBZ; no echo is a value below 0 (e.g.
                    -30), no coverage NaN
  satellite_ir:     irtb_YYYYMMDDHHMM.nc -- infrared window-channel brightness temperature
                    (~10.4-11 um), `tb(lat, lon)` in K
  stations:         stations.csv -- id,lat,lon,elev_m,time_utc,var,value; var t2 / td2 (deg C),
                    wspd10 / gust (m/s), wdir10 (deg, from north, clockwise), slp (hPa);
                    compared by post/station_compare.py (`wtune post stations`)
  stations_extra:   stations_extra.csv -- id,lat,lon,elev_m,time_utc,var,value; var `gust`
                    (m/s, peak gust) or `visibility` (m)

`check()` is what `wtune event obs-check` runs; the recipes use only datasets that pass it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone

FORMAT = "wtune-obs/1"
KINDS = ("gridded_precip", "rain_gauges", "radar", "satellite_ir", "stations_extra", "stations")
# snapshot grids: kind -> (file pattern, variable, units, valid range)
SNAPSHOTS = {
    "radar": (re.compile(r"^refl_(\d{12})\.nc$"), "refl", "dBZ", (-40.0, 85.0)),
    "satellite_ir": (re.compile(r"^irtb_(\d{12})\.nc$"), "tb", "K", (150.0, 340.0)),
}
EXTRA_FILE = "stations_extra.csv"
EXTRA_COLS = ("id", "lat", "lon", "elev_m", "time_utc", "var", "value")
EXTRA_VARS = {"gust": ("m/s", 0.0, 100.0), "visibility": ("m", 0.0, 100000.0)}
STATIONS_FILE = "stations.csv"
STATION_VARS = {"t2": ("degC", -60.0, 60.0), "td2": ("degC", -60.0, 40.0),
                "wspd10": ("m/s", 0.0, 100.0), "wdir10": ("deg", 0.0, 360.0),
                "gust": ("m/s", 0.0, 100.0), "slp": ("hPa", 850.0, 1100.0)}
PROV_REQUIRED = ("dataset", "provider", "url", "license", "fetched_utc", "files")
GRID_RE = re.compile(r"^precip_(\d{10})_(\d{2})h\.nc$")
GAUGE_FILE = "gauges.csv"
GAUGE_COLS = ("id", "lat", "lon", "elev_m", "start_utc", "end_utc", "precip_mm")
TIME_FMT = "%Y-%m-%d %H:%M"
# a rate above this in any cell or gauge is a unit or scaling error (world record ~300 mm/h)
MAX_MM_PER_H = 400.0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_provenance(ds_dir):
    try:
        with open(os.path.join(ds_dir, "provenance.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def datasets(project_dir, kind=None):
    """[(name, dir, provenance)] of the agent datasets under <project>/input/obs/, by name."""
    base = os.path.join(os.path.abspath(project_dir), "input", "obs") if project_dir else None
    if not base or not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        prov = read_provenance(d)
        if prov and prov.get("format") == FORMAT and (kind is None or prov.get("kind") == kind):
            out.append((name, d, prov))
    return out


def grid_index(ds_dir):
    """{(accum_h, valid end datetime): path} of the precip_*.nc files."""
    out = {}
    for f in os.listdir(ds_dir):
        m = GRID_RE.match(f)
        if m:
            try:
                out[(int(m.group(2)), datetime.strptime(m.group(1), "%Y%m%d%H"))] = \
                    os.path.join(ds_dir, f)
            except ValueError:
                pass
    return out


def snapshot_index(ds_dir, kind):
    """{valid datetime: path} of a radar / satellite_ir folder."""
    pat = SNAPSHOTS[kind][0]
    out = {}
    for f in os.listdir(ds_dir):
        m = pat.match(f)
        if m:
            try:
                out[datetime.strptime(m.group(1), "%Y%m%d%H%M")] = os.path.join(ds_dir, f)
            except ValueError:
                pass
    return out


def nearest(index, t, tol_min):
    """Path in {datetime: path} closest to t within tol_min minutes, else None."""
    best = min(index, key=lambda k: abs((k - t).total_seconds()), default=None)
    return index[best] if best is not None and abs((best - t).total_seconds()) <= tol_min * 60 \
        else None


def read_grid(path, var="precip"):
    """(lat ascending 1-D, lon 1-D in -180..180 order, `var` 2-D float with NaN)."""
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds.variables["lat"][:], "f8")
        lon = np.asarray(ds.variables["lon"][:], "f8")
        pr = np.ma.filled(np.ma.asarray(ds.variables[var][:], "f8"), np.nan)
    pr = pr.reshape(len(lat), len(lon))
    if lat[0] > lat[-1]:
        lat, pr = lat[::-1], pr[::-1]
    lon = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon, kind="stable")
    return lat, lon[order], pr[:, order]


def read_gauges(path):
    """[dict(id, lat, lon, elev_m, start, end, precip_mm)]; raises ValueError on a bad row."""
    rows = []
    with open(path, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh), start=2):
            try:
                rows.append(dict(id=r["id"].strip(), lat=float(r["lat"]), lon=float(r["lon"]),
                                 elev_m=float(r["elev_m"]) if r["elev_m"] not in ("", None) else 0.0,
                                 start=datetime.strptime(r["start_utc"].strip(), TIME_FMT),
                                 end=datetime.strptime(r["end_utc"].strip(), TIME_FMT),
                                 precip_mm=float(r["precip_mm"])))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError("%s line %d: %s" % (os.path.basename(path), i, e))
    return rows


def read_extra(path, blank_elev=0.0):
    """[dict(id, lat, lon, elev_m, time, var, value)]; raises ValueError on a bad row. An empty
    elev_m becomes `blank_elev`."""
    rows = []
    with open(path, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh), start=2):
            try:
                rows.append(dict(id=r["id"].strip(), lat=float(r["lat"]), lon=float(r["lon"]),
                                 elev_m=float(r["elev_m"]) if r["elev_m"] not in ("", None) else blank_elev,
                                 time=datetime.strptime(r["time_utc"].strip(), TIME_FMT),
                                 var=r["var"].strip(), value=float(r["value"])))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError("%s line %d: %s" % (os.path.basename(path), i, e))
    return rows


def source_tag(name, prov):
    """The figure `obs_sources` entry (same keys as post.sources.source_tags)."""
    tag = {k: prov.get(k, "") for k in ("dataset", "provider", "url", "license", "citation",
                                         "fetched_utc")}
    tag["name"] = name
    return tag


def check(ds_dir):
    """(errors, warnings) for one agent dataset folder."""
    err, warn = [], []
    prov = read_provenance(ds_dir)
    if prov is None:
        return ["provenance.json: missing or not JSON"], warn
    if prov.get("format") != FORMAT:
        err.append("provenance.json: format must be %r" % FORMAT)
    kind = prov.get("kind")
    if kind not in KINDS:
        err.append("provenance.json: kind %r has no reader; readers exist for %s"
                   % (kind, ", ".join(KINDS)))
    for k in PROV_REQUIRED:
        if not prov.get(k):
            err.append("provenance.json: %s is empty" % k)
    files = prov.get("files") or {}
    for rel, sha in sorted(files.items()):
        p = os.path.join(ds_dir, rel)
        if not os.path.isfile(p):
            err.append("files: %s is listed but missing" % rel)
        elif _sha256(p) != sha:
            err.append("files: %s sha256 differs from provenance.json" % rel)
    if kind == "gridded_precip":
        _check_grids(ds_dir, files, err, warn)
    elif kind == "rain_gauges":
        _check_gauges(ds_dir, files, err, warn)
    elif kind in SNAPSHOTS:
        _check_snapshots(ds_dir, kind, files, err, warn)
    elif kind == "stations_extra":
        _check_extra(ds_dir, files, err, warn)
    elif kind == "stations":
        _check_stations(ds_dir, files, err, warn)
    return err, warn


def _check_axes(f, lat, lon, err):
    import numpy as np
    if lat.min() < -90 or lat.max() > 90 or lon.min() < -180 or lon.max() > 180:
        err.append("%s: lat/lon out of range" % f)
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        err.append("%s: lat/lon must be strictly monotonic 1-D axes" % f)


def _check_snapshots(ds_dir, kind, files, err, warn):
    import numpy as np
    pat, var, units, (lo, hi) = SNAPSHOTS[kind]
    idx = snapshot_index(ds_dir, kind)
    if not idx:
        err.append("no %s_YYYYMMDDHHMM.nc file" % pat.pattern[1:pat.pattern.index("_")])
    for t, p in sorted(idx.items()):
        f = os.path.basename(p)
        if f not in files:
            err.append("%s: not listed in provenance.json files" % f)
        try:
            import netCDF4
            with netCDF4.Dataset(p) as ds:
                u = getattr(ds.variables[var], "units", "")
            lat, lon, a = read_grid(p, var)
        except (OSError, KeyError, ValueError) as e:
            err.append("%s: unreadable (%s); needs 1-D lat, lon and %s(lat, lon)" % (f, e, var))
            continue
        if u != units:
            err.append("%s: %s units %r, must be %r" % (f, var, u, units))
        _check_axes(f, lat, lon, err)
        good = a[np.isfinite(a)]
        if not good.size:
            err.append("%s: %s has no valid cell" % (f, var))
        elif good.min() < lo or good.max() > hi:
            err.append("%s: %s %.1f..%.1f is outside %g..%g %s"
                       % (f, var, good.min(), good.max(), lo, hi, units))
        elif kind == "radar":
            if np.isfinite(a).all() and max(np.ptp(lat), np.ptp(lon)) > 5:
                warn.append("%s: no cell is NaN over %.0fx%.0f deg -- are cells beyond the radars' "
                            "range set to NaN, or counted as no echo?" % (f, np.ptp(lat), np.ptp(lon)))
            if good.max() < 10:
                warn.append("%s: strongest echo %.1f dBZ -- check the dBZ decoding" % (f, good.max()))
        elif not 200 <= np.median(good) <= 300:
            warn.append("%s: median brightness temperature %.1f K -- check scale and offset"
                        % (f, np.median(good)))


def _check_extra(ds_dir, files, err, warn):
    p = os.path.join(ds_dir, EXTRA_FILE)
    if not os.path.isfile(p):
        err.append("%s: missing" % EXTRA_FILE)
        return
    if EXTRA_FILE not in files:
        err.append("%s: not listed in provenance.json files" % EXTRA_FILE)
    with open(p, newline="") as fh:
        head = next(csv.reader(fh), [])
    if tuple(h.strip() for h in head) != EXTRA_COLS:
        err.append("%s: header must be %s" % (EXTRA_FILE, ",".join(EXTRA_COLS)))
        return
    try:
        rows = read_extra(p)
    except ValueError as e:
        err.append("%s (times are '%s' UTC)" % (e, TIME_FMT.replace("%", "")))
        return
    if not rows:
        err.append("%s: no rows" % EXTRA_FILE)
    for r in rows:
        where = "%s %s %s" % (r["id"], r["time"].strftime(TIME_FMT), r["var"])
        spec = EXTRA_VARS.get(r["var"])
        if spec is None:
            err.append("%s: var must be one of %s" % (where, ", ".join(EXTRA_VARS)))
        elif not spec[1] <= r["value"] <= spec[2]:
            err.append("%s: value %g outside %g..%g %s" % (where, r["value"], spec[1], spec[2], spec[0]))
        if not (-90 <= r["lat"] <= 90 and -180 <= r["lon"] <= 360):
            err.append("%s: lat/lon out of range" % where)
        if len(err) > 20:
            err.append("... (stopped after 20 errors)")
            return
    import numpy as np
    gust = [r["value"] for r in rows if r["var"] == "gust"]
    vis = [r["value"] for r in rows if r["var"] == "visibility"]
    if gust and np.median(gust) < 2:
        warn.append("%s: median gust %.2f m/s -- gusts are reported above ~5 m/s; check the "
                    "unit (tenths of m/s, knots)" % (EXTRA_FILE, np.median(gust)))
    if vis and max(vis) < 2000:
        warn.append("%s: largest visibility %.0f m -- check the unit (km or miles vs m)"
                    % (EXTRA_FILE, max(vis)))


def read_stations(path):
    """stations.csv rows, same dict layout as read_extra; an empty elev_m is None."""
    return read_extra(path, blank_elev=None)


def _check_stations(ds_dir, files, err, warn):
    import numpy as np
    p = os.path.join(ds_dir, STATIONS_FILE)
    if not os.path.isfile(p):
        err.append("%s: missing" % STATIONS_FILE)
        return
    if STATIONS_FILE not in files:
        err.append("%s: not listed in provenance.json files" % STATIONS_FILE)
    with open(p, newline="") as fh:
        head = next(csv.reader(fh), [])
    if tuple(h.strip() for h in head) != EXTRA_COLS:
        err.append("%s: header must be %s" % (STATIONS_FILE, ",".join(EXTRA_COLS)))
        return
    try:
        rows = read_stations(p)
    except ValueError as e:
        err.append("%s (times are '%s' UTC)" % (e, TIME_FMT.replace("%", "")))
        return
    if not rows:
        err.append("%s: no rows" % STATIONS_FILE)
    by = {}
    for r in rows:
        by.setdefault(r["var"], []).append(r["value"])
    med = {v: float(np.median(x)) for v, x in by.items()}
    # common unit mistakes, reported even when the range check below also fails
    if med.get("t2", 0) > 60 or med.get("td2", 0) > 60:
        warn.append("%s: median t2/td2 above 60 -- looks like degF, not degC" % STATIONS_FILE)
    elif med.get("t2", 0) > 200:
        warn.append("%s: median t2 %.0f -- looks like K, not degC" % (STATIONS_FILE, med["t2"]))
    if med.get("wspd10", 0) > 40 or med.get("gust", 0) > 40:
        warn.append("%s: median wspd10/gust above 40 -- looks like knots (or mph), not m/s"
                    % STATIONS_FILE)
    if "slp" in med and med["slp"] < 200:
        warn.append("%s: median slp %.1f -- looks like kPa or inHg, not hPa" % (STATIONS_FILE, med["slp"]))
    if "wdir10" in by and max(by["wdir10"]) <= 2 * np.pi + 1e-9 and len(by["wdir10"]) > 20:
        warn.append("%s: wdir10 never exceeds 6.3 -- radians instead of degrees?" % STATIONS_FILE)
    no_elev = sorted({r["id"] for r in rows if r["elev_m"] is None})
    if no_elev:
        warn.append("%s: no elev_m for %s -- the model-minus-station height difference cannot be "
                    "reported" % (STATIONS_FILE, ", ".join(no_elev)))
    t2 = {(r["id"], r["time"]): r["value"] for r in rows if r["var"] == "t2"}
    n_inv = sum(1 for r in rows if r["var"] == "td2" and (r["id"], r["time"]) in t2
                and r["value"] > t2[(r["id"], r["time"])] + 1.0)
    if n_inv:
        warn.append("%s: %d td2 value(s) exceed t2 by more than 1 degC" % (STATIONS_FILE, n_inv))
    for r in rows:
        where = "%s %s %s" % (r["id"], r["time"].strftime(TIME_FMT), r["var"])
        spec = STATION_VARS.get(r["var"])
        if spec is None:
            err.append("%s: var must be one of %s" % (where, ", ".join(STATION_VARS)))
        elif not spec[1] <= r["value"] <= spec[2]:
            err.append("%s: value %g outside %g..%g %s" % (where, r["value"], spec[1], spec[2], spec[0]))
        if not (-90 <= r["lat"] <= 90 and -180 <= r["lon"] <= 360):
            err.append("%s: lat/lon out of range" % where)
        if len(err) > 20:
            err.append("... (stopped after 20 errors)")
            return


_MISSING = {"", "m", "null", "nan", "none"}
_ASOS_VARS = (("tmpf", "t2"), ("dwpf", "td2"), ("sknt", "wspd10"), ("gust", "gust"),
              ("drct", "wdir10"), ("mslp", "slp"))


def _asos_num(text):
    t = (text or "").strip()
    return None if t.lower() in _MISSING else float(t)


def from_asos(csv_path, out_dir, provenance, stations_meta=None):
    """Iowa Environmental Mesonet ASOS CSV (station, valid UTC "YYYY-MM-DD HH:MM", tmpf, dwpf,
    sknt, gust, drct, lat, lon (any column order; matched by name), optional mslp [hPa] and
    elevation [m]; missing "M", empty or
    "null") -> out_dir/stations.csv + provenance.json (kind "stations", sha256 included).
    degF -> degC, knot -> m/s (calc.kt_to_ms). lat/lon/elev come from the CSV's own columns, else
    from stations_meta {ID: (lat, lon, elev_m)}; a station without lat/lon raises ValueError;
    a missing elevation leaves elev_m empty.
    `provenance`: dict with dataset, provider, url, license (citation optional). Returns
    (stations.csv path, number of rows)."""
    from wtune import calc
    meta = {k.strip().upper(): tuple(v) for k, v in (stations_meta or {}).items()}
    pos, vals = {}, []
    with open(csv_path, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = {c.strip().lower(): c for c in (rd.fieldnames or [])}
        for need in ("station", "valid"):
            if need not in cols:
                raise ValueError("%s: no %r column" % (os.path.basename(csv_path), need))
        for i, r in enumerate(rd, start=2):
            r = {k.strip().lower(): v for k, v in r.items() if k}
            sid = r["station"].strip()
            try:
                t = datetime.strptime(r["valid"].strip()[:16], TIME_FMT)
                for col in ("lat", "lon", "elevation"):
                    v = _asos_num(r.get(col))
                    if v is not None:
                        pos.setdefault(sid, {}).setdefault(col, v)
                for col, var in _ASOS_VARS:
                    v = _asos_num(r.get(col))
                    if v is None:
                        continue
                    if col in ("tmpf", "dwpf"):
                        v = (v - 32.0) * 5.0 / 9.0
                    elif col in ("sknt", "gust"):
                        v = calc.kt_to_ms(v)
                    vals.append((sid, t, var, v))
            except ValueError as e:
                raise ValueError("%s line %d: %s" % (os.path.basename(csv_path), i, e))
    where = {}
    for sid in sorted({v[0] for v in vals}):
        got = pos.get(sid, {})
        m = meta.get(sid.upper())
        lat = got.get("lat", m[0] if m else None)
        lon = got.get("lon", m[1] if m else None)
        elev = got.get("elevation", m[2] if m and len(m) > 2 else None)
        lack = [n for n, v in (("lat", lat), ("lon", lon)) if v is None]
        if lack:
            raise ValueError("station %s: no %s in the CSV (lat/lon columns) and none in "
                             "stations_meta" % (sid, "/".join(lack)))
        where[sid] = (lat, lon, elev)
    if not vals:
        raise ValueError("%s: no usable observation" % os.path.basename(csv_path))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, STATIONS_FILE)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(EXTRA_COLS)
        for sid, t, var, v in sorted(vals, key=lambda x: (x[0], x[1], x[2])):
            lat, lon, elev = where[sid]
            w.writerow([sid, "%.5f" % lat, "%.5f" % lon, "" if elev is None else "%.1f" % elev,
                        t.strftime(TIME_FMT),
                        var, "%.3f" % v])
    prov = {"format": FORMAT, "kind": "stations", "citation": ""}
    prov.update({k: v for k, v in provenance.items() if k not in ("format", "kind", "files")})
    prov.setdefault("fetched_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    prov["files"] = {STATIONS_FILE: _sha256(out)}
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
    return out, len(vals)


def _check_grids(ds_dir, files, err, warn):
    import numpy as np
    idx = grid_index(ds_dir)
    peak = -1.0
    if not idx:
        err.append("no precip_YYYYMMDDHH_NNh.nc file")
    for (acc, valid), p in sorted(idx.items()):
        f = os.path.basename(p)
        if f not in files:
            err.append("%s: not listed in provenance.json files" % f)
        if acc < 1:
            err.append("%s: accumulation must be at least 1 h" % f)
        try:
            import netCDF4
            with netCDF4.Dataset(p) as ds:
                units = getattr(ds.variables["precip"], "units", "")
            lat, lon, pr = read_grid(p)
        except (OSError, KeyError, ValueError) as e:
            err.append("%s: unreadable (%s); needs 1-D lat, lon and precip(lat, lon)" % (f, e))
            continue
        if units != "mm":
            err.append("%s: precip units %r, must be 'mm'" % (f, units))
        _check_axes(f, lat, lon, err)
        good = pr[np.isfinite(pr)]
        if not good.size:
            err.append("%s: precip has no valid cell" % f)
        elif good.min() < 0 or good.max() > MAX_MM_PER_H * max(acc, 1):
            err.append("%s: precip %.1f..%.1f mm over %d h is outside 0..%g mm"
                       % (f, good.min(), good.max(), acc, MAX_MM_PER_H * max(acc, 1)))
        else:
            peak = max(peak, float(good.max()))
    if idx and 0 <= peak < 1.0:
        warn.append("largest accumulation in any file is %.3f mm -- possible if it did not rain, "
                    "otherwise check for a rate left as a rate or a scale applied twice" % peak)


def _check_gauges(ds_dir, files, err, warn):
    p = os.path.join(ds_dir, GAUGE_FILE)
    if not os.path.isfile(p):
        err.append("%s: missing" % GAUGE_FILE)
        return
    if GAUGE_FILE not in files:
        err.append("%s: not listed in provenance.json files" % GAUGE_FILE)
    with open(p, newline="") as fh:
        head = next(csv.reader(fh), [])
    if tuple(h.strip() for h in head) != GAUGE_COLS:
        err.append("%s: header must be %s" % (GAUGE_FILE, ",".join(GAUGE_COLS)))
        return
    try:
        rows = read_gauges(p)
    except ValueError as e:
        err.append("%s (times are '%s' UTC)" % (e, TIME_FMT.replace("%", "")))
        return
    if not rows:
        err.append("%s: no rows" % GAUGE_FILE)
    for r in rows:
        hours = (r["end"] - r["start"]).total_seconds() / 3600.0
        where = "%s %s" % (r["id"], r["end"].strftime(TIME_FMT))
        if hours <= 0 or hours != int(hours):
            err.append("%s: the period must be a whole number of hours" % where)
        elif not 0 <= r["precip_mm"] <= MAX_MM_PER_H * max(hours, 1):
            err.append("%s: precip_mm %g over %g h is outside 0..%g"
                       % (where, r["precip_mm"], hours, MAX_MM_PER_H * max(hours, 1)))
        if not (-90 <= r["lat"] <= 90 and -180 <= r["lon"] <= 360):
            err.append("%s: lat/lon out of range" % where)
        if len(err) > 20:
            err.append("... (stopped after 20 errors)")
            return
    if rows and len({(r["end"] - r["start"]) for r in rows}) > 1:
        warn.append("%s: mixed accumulation periods; each window uses the gauges whose period "
                    "matches it exactly" % GAUGE_FILE)


def usable(project_dir, kind):
    """[(name, dir, provenance)] of the datasets of `kind` that pass check(), and
    [(name, first error)] of those that do not."""
    ok, bad = [], []
    for name, d, prov in datasets(project_dir, kind):
        e, _w = check(d)
        if e:
            bad.append((name, e[0]))
        else:
            ok.append((name, d, prov))
    return ok, bad


def tile(index, start, end):
    """Paths of the files of one accumulation that tile [start, end] exactly, else []
    (the shortest such accumulation)."""
    span = (end - start).total_seconds() / 3600.0
    for acc in sorted({a for a, _ in index}):
        if span % acc:
            continue
        ends = [start + timedelta(hours=acc * (k + 1)) for k in range(int(span // acc))]
        if ends and all((acc, t) in index for t in ends):
            return [index[(acc, t)] for t in ends]
    return []


def pick_window(frame_times, spinup_hours, index, spans=(24, 12, 6, 3, 1), around=None):
    """(hours, start, end, paths) of the longest window of `spans` that starts after spin-up,
    starts and ends on model frames, and the files tile; the latest such window, or with
    `around` (datetime, the rain peak) the one centred nearest it among those containing it
    (haitang2017: the latest 24 h began after the peak). Else None."""
    if len(frame_times) < 2:
        return None
    have = set(frame_times)
    t0 = frame_times[0] + timedelta(hours=spinup_hours)
    for span in spans:
        wins = []
        for end in sorted(have, reverse=True):
            start = end - timedelta(hours=span)
            if start < t0 or start not in have:
                continue
            paths = tile(index, start, end)
            if paths:
                if around is None:
                    return span, start, end, paths
                wins.append((span, start, end, paths))
        if wins:
            inside = [w for w in wins if w[1] < around <= w[2]]
            return (min(inside, key=lambda w: abs(w[1] + (w[2] - w[1]) / 2 - around))
                    if inside else wins[0])
    return None


def sum_grids(paths):
    """(lat, lon, total mm) of files on one grid; a cell missing in any file is NaN."""
    lat, lon, total = read_grid(paths[0])
    for p in paths[1:]:
        la, lo, pr = read_grid(p)
        if la.shape != lat.shape or lo.shape != lon.shape:
            raise ValueError("%s is on another grid than %s" % (p, paths[0]))
        total = total + pr
    return lat, lon, total


def subset(lat, lon, field, bbox, margin=2.0):
    """The part of a regular grid inside bbox (lat0, lat1, lon0, lon1) plus `margin` degrees."""
    import numpy as np
    la = (lat >= bbox[0] - margin) & (lat <= bbox[1] + margin)
    lo = (lon >= bbox[2] - margin) & (lon <= bbox[3] + margin)
    return lat[la], lon[lo], field[np.ix_(la, lo)]


def regrid_snapshot(path, var, valid, griddes, out_nc, name, units, bbox=None, method="remapnn"):
    """One radar / satellite file -> CF netCDF on the model lat/lon grid `griddes` with cdo
    (`method`), variable `name`; cut to bbox + 2 degrees first. Returns out_nc."""
    from wtune.post import met_gridstat, tools
    import numpy as np
    lat, lon, a = read_grid(path, var)
    if bbox:
        lat, lon, a = subset(lat, lon, a, bbox)
    # a NaN frame one cell wide: cdo's nearest neighbour would otherwise copy the edge values
    # to every model cell outside the observation grid
    lat = np.concatenate([[2 * lat[0] - lat[1]], lat, [2 * lat[-1] - lat[-2]]])
    lon = np.concatenate([[2 * lon[0] - lon[1]], lon, [2 * lon[-1] - lon[-2]]])
    a = np.pad(a, 1, constant_values=np.nan)
    tmp = out_nc[:-3] + ".src.nc"
    met_gridstat._write_regular_nc(lat, lon, a, valid, tmp, name=name, units=units,
                                   long_name="%s from %s" % (var, os.path.basename(path)))
    tools.run([tools.find_on_path("cdo"), "-s", "%s,%s" % (method, griddes), tmp, out_nc])
    os.remove(tmp)
    met_gridstat._tag_axes(out_nc)
    return out_nc
