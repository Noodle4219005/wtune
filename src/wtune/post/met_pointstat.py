"""post/met_pointstat.py -- model-vs-surface-station verification with MET: ISD-Lite hourly
observations (post/fetch/stations.py) -> MET's own `ascii2nc` (11-column point format) -> one
`point_stat` call per model valid time against a regular lat/lon NetCDF of T2, TD2, 10 m wind
speed and MSLP (wrf-python fields moved off the curvilinear WRF grid with `cdo remapbil`).
Point-Stat does the pairing (+-30 min), bilinear interpolation to the station and the interior-
polygon mask; its CNT lines (ME = bias model-minus-obs, RMSE) give the time series and its MPR
lines the per-station mean bias for the map. The only own code is unit conversion of ISD-Lite's
fixed-width text and reading MET's tables.

Units in the MET files: temperatures K, wind speed m/s, MSLP hPa. Station elevation is NOT
corrected for (T2 at a station 50 m above its model cell is compared as is).
"""
from __future__ import annotations

import csv
import gzip
import os
from datetime import datetime, timedelta

from wtune.post import met_gridstat, sources, tools
from wtune.post.met_tc import METError, _write_config, parse_met_table

# MET var name -> (ISD-Lite column, scale to MET units, offset, model var, label, unit)
VARS = {
    # bias/RMSE are temperature DIFFERENCES: degC and K are the same number; surface
    # verification reports them in degC (decision 0064)
    "TMP": dict(col=4, hgt=2, scale=0.1, offset=273.15, model="T2", label="T2", unit="°C"),
    "DPT": dict(col=5, hgt=2, scale=0.1, offset=273.15, model="TD2", label="Td2", unit="°C"),
    "WIND": dict(col=8, hgt=10, scale=0.1, offset=0.0, model="WSPD10", label="10 m wind speed", unit="m/s"),
    "PRMSL": dict(col=6, hgt=0, scale=0.1, offset=0.0, model="MSLP", label="MSLP", unit="hPa"),
}
MAX_TIMES = 48
_TAG_FMT = "%Y%m%d%H%M"      # with minutes: sub-hourly history (HH:00, HH:30)
BOOT_REPS = 1000


class NoStations(Exception):
    """No station observations available (message = coverage or fetch-command reason)."""


def read_stations(project_dir):
    path = os.path.join(sources.dataset_dir(project_dir, "stations"), "stations.csv")
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def parse_isd_lite(text):
    """[(datetime, [12 ints])] from ISD-Lite text: `YYYY MM DD HH T Td SLP WD WS SKY P1 P6`
    (T/Td 0.1 degC, SLP 0.1 hPa, WS 0.1 m/s, precip 0.1 mm; -9999 missing)."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 12:
            continue
        try:
            vals = [int(x) for x in f[:12]]
            out.append((datetime(vals[0], vals[1], vals[2], vals[3]), vals))
        except ValueError:
            continue
    return out


def ascii2nc_lines(station, records, t_lo, t_hi):
    """MET 11-column lines (Message_Type Station_ID Valid_Time Lat Lon Elevation Var_Name Level
    Height QC Value) for one station inside [t_lo, t_hi]; missing (-9999) values skipped."""
    lines = []
    elev = station.get("elev_m") or "0"
    if elev.lower() == "nan":
        elev = "0"
    for t, vals in records:
        if not (t_lo <= t <= t_hi):
            continue
        for name, spec in VARS.items():
            raw = vals[spec["col"]]
            if raw == -9999:
                continue
            lines.append("ADPSFC %s %s %s %s %s %s -9999 %d NA %.3f" % (
                station["id"], t.strftime("%Y%m%d_%H%M%S"), station["lat"], station["lon"], elev,
                name, spec["hgt"], raw * spec["scale"] + spec["offset"]))
    return lines


def build_obs_text(project_dir, t_lo, t_hi):
    """(ascii text, n_stations_with_data) from the fetched ISD-Lite files."""
    stations = read_stations(project_dir)
    base = sources.dataset_dir(project_dir, "stations")
    lines, used = [], 0
    for st in stations:
        st_lines = []
        for year in range(t_lo.year, t_hi.year + 1):
            usaf, wban = st["id"].split("-")
            path = os.path.join(base, "%s-%s-%d.gz" % (usaf, wban, year))
            if not os.path.isfile(path):
                continue
            with gzip.open(path, "rt", errors="replace") as fh:
                st_lines += ascii2nc_lines(st, parse_isd_lite(fh.read()), t_lo, t_hi)
        if st_lines:
            used += 1
            lines += st_lines
    return "\n".join(lines) + "\n", used


def model_fields(frame_path, timeidx=0):
    """(lat, lon, {name: (arr, unit, long_name)}, valid datetime) from time index `timeidx` of
    one wrfout file (0 for a single-frame file)."""
    import netCDF4
    import numpy as np
    import wrf
    with netCDF4.Dataset(frame_path) as ds:
        lat = np.asarray(wrf.getvar(ds, "lat", timeidx=timeidx), "f8")
        lon = np.asarray(wrf.getvar(ds, "lon", timeidx=timeidx), "f8")
        t2 = np.asarray(wrf.getvar(ds, "T2", timeidx=timeidx), "f8")
        td2 = np.asarray(wrf.getvar(ds, "td2", timeidx=timeidx, units="K"), "f8")
        wspd = np.asarray(wrf.getvar(ds, "uvmet10_wspd_wdir", timeidx=timeidx,
                                     units="m s-1")[0], "f8")
        slp = np.asarray(wrf.getvar(ds, "slp", timeidx=timeidx, units="hPa"), "f8")
        valid = wrf.extract_times(ds, timeidx)
    valid = valid.astype("datetime64[s]").astype(object)
    return lat, lon, {"T2": (t2, "K", "2 m temperature"), "TD2": (td2, "K", "2 m dew point"),
                      "WSPD10": (wspd, "m s-1", "10 m wind speed"),
                      "MSLP": (slp, "hPa", "sea-level pressure")}, valid


def run_ascii2nc(text, out_dir):
    ascii2nc = tools.find_postproc_tool("ascii2nc")
    txt = os.path.join(out_dir, "stations_obs.txt")
    with open(txt, "w") as fh:
        fh.write(text)
    nc = os.path.join(out_dir, "stations_obs.nc")
    tools.run([ascii2nc, txt, nc])
    return nc


def run_point_stat(fcst_nc, obs_nc, poly, out_dir, tag):
    point_stat = tools.find_postproc_tool("point_stat")
    os.makedirs(out_dir, exist_ok=True)
    conf = _write_config(out_dir, "verif_point_stat.conf", POLY_FILE=os.path.abspath(poly))
    tools.run([point_stat, fcst_nc, obs_nc, conf, "-outdir", out_dir, "-v", "1"])
    return out_dir


def _tables(out_dir, suffix):
    rows = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(suffix):
            with open(os.path.join(out_dir, f)) as fh:
                rows += parse_met_table(fh.read())
    return rows


def verify_run(run_dir, domain, frames, project_dir, out_dir, spinup_hours):
    """Whole chain for a run. Raises NoStations when no observations are available."""
    from wtune.post.caseinfo import WRF_TIME_FMT, spec_bdy_width
    if not project_dir:
        raise NoStations("no wtune project found for %s -- station data lives under "
                         "<project>/input/obs/stations (wtune post fetch stations --project P)" % out_dir)
    stations = read_stations(project_dir)
    if not stations:
        raise NoStations("station observations not fetched yet -- run on a login node: "
                         "wtune post fetch stations --project %s --run-dir %s" % (project_dir, run_dir))
    # frames: [(time, path[, timeidx])] -- post.extract.list_frame_refs (timeidx 0 if absent)
    times = [(datetime.strptime(f[0], WRF_TIME_FMT), f[1]) for f in frames]
    tidx = {datetime.strptime(f[0], WRF_TIME_FMT): (f[2] if len(f) > 2 else 0) for f in frames}
    t0 = times[0][0] + timedelta(hours=spinup_hours)
    times = [(t, p) for t, p in times if t >= t0] or times
    stride = max(1, -(-len(times) // MAX_TIMES))
    times = times[::stride]
    work = os.path.join(out_dir, "pointstat")
    os.makedirs(work, exist_ok=True)
    text, used = build_obs_text(project_dir, times[0][0] - timedelta(hours=1),
                                times[-1][0] + timedelta(hours=1))
    if not used:
        raise NoStations("the fetched ISD-Lite stations have no valid observation between %s and %s"
                         % (times[0][0], times[-1][0]))
    obs_nc = run_ascii2nc(text, work)
    poly = met_gridstat.write_interior_polygon(times[-1][1], spec_bdy_width(run_dir),
                                               os.path.join(work, "interior.poly"))
    for t, path in times:
        tag = t.strftime(_TAG_FMT)
        lat, lon, fields, valid = model_fields(path, timeidx=tidx[t])
        fnc = met_gridstat.curvilinear_to_latlon(lat, lon, fields, valid,
                                                 os.path.join(work, "fcst_%s.nc" % tag), method="remapbil")
        run_point_stat(fnc, obs_nc, poly, os.path.join(work, "ps_%s" % tag), tag)
        os.remove(fnc)
    return summarize(work, times, out_dir, project_dir)


def summarize(work, times, out_dir, project_dir):
    """CNT rows -> per-time bias/RMSE series (csv + figure), MPR rows -> per-station mean bias
    (csv + map)."""
    import numpy as np
    from wtune.post import obs_figs
    series = {v["label"]: (v["unit"], [], [], []) for v in VARS.values()}
    rows_csv = []
    by_model = {v["model"]: v for v in VARS.values()}
    xs = []
    stn = {}
    pairs_by_time = {}
    for t, _p in times:
        d = os.path.join(work, "ps_%s" % t.strftime(_TAG_FMT))
        if not os.path.isdir(d):
            continue
        cnt = {r["FCST_VAR"]: r for r in _tables(d, "_cnt.txt") if r.get("LINE_TYPE") == "CNT"}
        got = False
        for mv, spec in by_model.items():
            r = cnt.get(mv)
            if not r:
                continue
            try:
                me, rmse, n = float(r["ME"]), float(r["RMSE"]), int(float(r["TOTAL"]))
            except (KeyError, ValueError):
                continue
            unit, b, rm, ns = series[spec["label"]]
            b.append(me); rm.append(rmse); ns.append(n)
            rows_csv.append([t.strftime("%Y-%m-%d %H:%M"), spec["label"], spec["unit"], me, rmse, n])
            got = True
        if got:
            xs.append(t)
        # only times that had at least one CNT row extend x; series lists stay aligned per var
        for r in _tables(d, "_mpr.txt"):
            if r.get("LINE_TYPE") != "MPR":
                continue
            try:
                f, o = float(r["FCST"]), float(r["OBS"])
                key = (r["OBS_SID"], r["FCST_VAR"])
                s = stn.setdefault(key, dict(lat=float(r["OBS_LAT"]), lon=float(r["OBS_LON"]), d=[]))
                s["d"].append(f - o)
                pairs_by_time.setdefault((t, r["FCST_VAR"]), []).append(f - o)
            except (KeyError, ValueError):
                continue
    if not xs:
        raise METError("point_stat produced no CNT rows -- no model/observation pairs inside the "
                       "interior polygon (stations outside the model domain?)")
    # align: each variable's series have one entry per time that had its CNT row; rebuild x per var
    per_var_x = {}
    for t, _p in times:
        d = os.path.join(work, "ps_%s" % t.strftime(_TAG_FMT))
        if os.path.isdir(d):
            have = {r["FCST_VAR"] for r in _tables(d, "_cnt.txt") if r.get("LINE_TYPE") == "CNT"}
            for mv in have:
                per_var_x.setdefault(by_model[mv]["label"], []).append(t)
    csv_ts = os.path.join(out_dir, "station_bias_rmse_timeseries.csv")
    with open(csv_ts, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["valid_time_utc", "variable", "unit", "bias_model_minus_obs", "rmse", "n_pairs"])
        w.writerows(rows_csv)
    csv_st = os.path.join(out_dir, "station_bias_by_station.csv")
    with open(csv_st, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station_id", "variable", "lat", "lon", "mean_bias", "n_times"])
        for (sid, mv), s in sorted(stn.items()):
            w.writerow([sid, mv, s["lat"], s["lon"], float(np.mean(s["d"])), len(s["d"])])
    # 95 % bootstrap CI per valid time, stations resampled (decision 0064; same idea as MET's
    # BOOT_INTERVAL percentile CIs on matched pairs)
    rng = np.random.default_rng(0)
    label_to_model = {v["label"]: m for m, v in by_model.items()}
    with_ci = {}
    for label, (unit, b, rm, ns) in series.items():
        if not b:
            continue
        ci = dict(bias_lo=[], bias_hi=[], rmse_lo=[], rmse_hi=[])
        for t in per_var_x.get(label, []):
            d = np.asarray(pairs_by_time.get((t, label_to_model[label]), []), dtype="f8")
            if d.size >= 5:
                idx = rng.integers(0, d.size, size=(BOOT_REPS, d.size))
                bs = d[idx]
                bm, br = bs.mean(axis=1), np.sqrt((bs ** 2).mean(axis=1))
                ci["bias_lo"].append(float(np.percentile(bm, 2.5)))
                ci["bias_hi"].append(float(np.percentile(bm, 97.5)))
                ci["rmse_lo"].append(float(np.percentile(br, 2.5)))
                ci["rmse_hi"].append(float(np.percentile(br, 97.5)))
            else:
                for k in ci:
                    ci[k].append(float("nan"))
        with_ci[label] = (unit, b, rm, ns, ci)
    series_png = os.path.join(out_dir, "precip_stn_series.png")
    obs_figs.bias_rmse_series(per_var_x, with_ci,
                              "model vs ISD-Lite stations, MET Point-Stat (bilinear, interior stations)",
                              series_png)
    panels = []
    for mv in ("T2", "WSPD10", "MSLP"):
        pts = [(s["lat"], s["lon"], float(np.mean(s["d"]))) for (sid, v), s in stn.items() if v == mv]
        if pts:
            b = np.array([p[2] for p in pts])
            panels.append(dict(lat=[p[0] for p in pts], lon=[p[1] for p in pts], bias=b,
                               cbar_label="mean bias, model - obs (%s)" % by_model[mv]["unit"],
                               title="%s mean bias (%s), n=%d stations" % (by_model[mv]["label"],
                                                                          by_model[mv]["unit"], len(pts)),
                               vmax=float(np.nanpercentile(np.abs(b), 95)) or 1.0))
    map_png = os.path.join(out_dir, "precip_stn_map.png")
    obs_figs.station_bias_maps(panels, "mean model - observation over the run (post-spin-up)", map_png)
    summary = {}
    for label, (unit, b, rm, ns) in series.items():
        if b:
            summary[label] = dict(unit=unit, mean_bias=float(np.mean(b)), mean_rmse=float(np.mean(rm)),
                                  n_times=len(b), n_pairs_total=int(sum(ns)))
    return dict(series_png=series_png, map_png=map_png, csv=[csv_ts, csv_st], summary=summary,
                t_first=times[0][0].strftime("%Y-%m-%d_%H:%M:%S"),
                t_last=times[-1][0].strftime("%Y-%m-%d_%H:%M:%S"))


def gauge_periods(rows, frame_times, spinup_hours, max_periods=MAX_TIMES):
    """(hours, [(start, end)]) of the longest gauge accumulation whose periods start after
    spin-up and start and end on model frames (latest max_periods of them), else (None, [])."""
    have = set(frame_times)
    t0 = frame_times[0] + timedelta(hours=spinup_hours) if frame_times else None
    by_len = {}
    for r in rows:
        if t0 and r["start"] >= t0 and r["start"] in have and r["end"] in have:
            h = (r["end"] - r["start"]).total_seconds() / 3600.0
            by_len.setdefault(h, set()).add((r["start"], r["end"]))
    if not by_len:
        return None, []
    h = max(by_len)
    return h, sorted(by_len[h])[-max_periods:]


def gauge_obs_name(hours):
    """Obs variable name of an accumulation, e.g. APCP_06. The period is in the name: Point-Stat
    rejects a point obs named plain APCP by name, and its A6 level never matches ascii2nc's
    level (stored in seconds) -- measured with MET 12.1.1 on f1-arm, job 32228."""
    return "APCP_%02d" % hours


def gauge_obs_text(rows, hours):
    """MET 11-column lines for the gauges whose period is `hours` long."""
    lines = []
    for r in rows:
        if (r["end"] - r["start"]).total_seconds() / 3600.0 != hours:
            continue
        lines.append("ADPSFC %s %s %s %s %s %s %g -9999 NA %.3f" % (
            r["id"].replace(" ", "_"), r["end"].strftime("%Y%m%d_%H%M%S"), r["lat"], r["lon"],
            r["elev_m"], gauge_obs_name(hours), hours, r["precip_mm"]))
    return "\n".join(lines) + "\n"


def verify_gauges(run_dir, frames, rows, out_dir, spinup_hours, model_accum):
    """Model accumulation vs agent-fetched rain gauges with ascii2nc + one point_stat per
    period. model_accum(ref0, ref1) -> (lat, lon, mm). Returns dict(hours, periods, pairs,
    csv) with pairs [(gauge id, lat, lon, end, model mm, gauge mm)] from the MPR lines."""
    from wtune.post.caseinfo import WRF_TIME_FMT, spec_bdy_width
    refs = {datetime.strptime(f[0], WRF_TIME_FMT): (f[1], f[2] if len(f) > 2 else 0)
            for f in frames}
    hours, periods = gauge_periods(rows, sorted(refs), spinup_hours)
    if not periods:
        raise NoStations("no gauge period starts after the %g h spin-up and starts and ends on "
                         "model output times" % spinup_hours)
    work = os.path.join(out_dir, "pointstat_gauges")
    os.makedirs(work, exist_ok=True)
    ascii2nc = tools.find_postproc_tool("ascii2nc")
    point_stat = tools.find_postproc_tool("point_stat")
    txt, obs_nc = os.path.join(work, "gauges_obs.txt"), os.path.join(work, "gauges_obs.nc")
    with open(txt, "w") as fh:
        fh.write(gauge_obs_text(rows, hours))
    tools.run([ascii2nc, txt, obs_nc])
    poly = met_gridstat.write_interior_polygon(refs[periods[-1][1]][0], spec_bdy_width(run_dir),
                                               os.path.join(work, "interior.poly"))
    pairs, cnt_rows = [], []
    for start, end in periods:
        d = os.path.join(work, "ps_%s" % end.strftime(_TAG_FMT))
        os.makedirs(d, exist_ok=True)
        lat, lon, mm = model_accum(refs[start], refs[end])
        fnc = met_gridstat.curvilinear_to_latlon(
            lat, lon, {"APCP_MODEL": (mm, "mm", "WRF accumulated precipitation")}, end,
            os.path.join(d, "fcst.nc"), method="remapcon")
        conf = _write_config(d, "verif_point_stat_apcp.conf", POLY_FILE=os.path.abspath(poly),
                             OBS_NAME=gauge_obs_name(hours))
        tools.run([point_stat, fnc, obs_nc, conf, "-outdir", d, "-v", "1"])
        os.remove(fnc)
        for r in _tables(d, "_cnt.txt"):
            if r.get("LINE_TYPE") == "CNT":
                cnt_rows.append([end.strftime("%Y-%m-%d %H:%M"), r.get("ME"), r.get("RMSE"),
                                 r.get("PR_CORR"), r.get("TOTAL")])
        for r in _tables(d, "_mpr.txt"):
            if r.get("LINE_TYPE") != "MPR":
                continue
            try:
                pairs.append((r["OBS_SID"], float(r["OBS_LAT"]), float(r["OBS_LON"]), end,
                              float(r["FCST"]), float(r["OBS"])))
            except (KeyError, ValueError):
                continue
    if not pairs:
        raise METError("point_stat paired no gauge with the model (gauges outside the interior?)")
    p_pairs = os.path.join(out_dir, "precip_gauge_pairs.csv")
    with open(p_pairs, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gauge_id", "lat", "lon", "period_end_utc", "hours", "model_mm", "gauge_mm"])
        for sid, la, lo, end, f, o in pairs:
            w.writerow([sid, la, lo, end.strftime("%Y-%m-%d %H:%M"), hours, f, o])
    p_cnt = os.path.join(out_dir, "precip_gauge_cnt.csv")
    with open(p_cnt, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["period_end_utc", "bias_model_minus_obs_mm", "rmse_mm", "pearson_r", "n_pairs"])
        w.writerows(cnt_rows)
    return dict(hours=hours, periods=periods, pairs=pairs, csv=[p_pairs, p_cnt])
