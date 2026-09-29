"""post/atcf.py -- ATCF a-deck/b-deck track-file read/write (decision 0056, 2026-09-29
update).

Why this is the one piece of new code in `wtune post tc` instead of wrapping an existing
tool: MET's tc_pairs/tc_stat/tc_rmw (wrapped in post/met_tc.py) both CONSUME ATCF-format
track files but neither one PRODUCES a model a-deck from a WRF run -- that conversion has
to originate somewhere, and no established tool performs "TempestExtremes track -> ATCF
a-deck" or "HURDAT2 best track -> ATCF b-deck". This module is that converter, kept
deliberately small: read/write the fixed comma-delimited ATCF line format (NHC ATCF
Technical Documentation; verified column-for-column against a real archive file,
tests/fixtures/post/bal122005_sample.dat, downloaded from
https://ftp.nhc.noaa.gov/atcf/archive/2005/bal122005.dat.gz), nothing else -- no track
detection (TempestExtremes/analysis.tc_track), no wind-radii math (tc_metrics), no
verification statistics (MET). Everything here is pure text I/O, no netCDF/numpy.

Line format (comma + space separated, 29 leading fields; MET's parser is delimiter-based,
not fixed-column, so exact field widths/padding don't matter as long as the values parse):

  BASIN, CY, YYYYMMDDHH, TECHNUM, TECH, TAU, LatN/S, LonE/W, VMAX, MSLP, TY, RAD,
  WINDCODE, RAD1, RAD2, RAD3, RAD4, POUTER, ROUTER, RMW, GUSTS, EYE, SUBREGION, MAXSEAS,
  INITIALS, DIR, SPEED, STORMNAME, DEPTH

LAT/LON are tenths of a degree with a trailing hemisphere letter (e.g. "246N" = 24.6N,
"833W" = 83.3W -- west/south negative, matching predictor/verify_obs.py and post/tc.py's
own HURDAT2 convention). VMAX is kt, MSLP is hPa, RAD/RAD1-4/RMW/GUSTS are kt/nmi. One
line is written per wind-radius threshold (RAD=34/50/64) once the storm has reached it, a
single RAD=0 line otherwise -- matching the real archive's own convention (see the TD-stage
lines vs. the TS/HU-stage lines in bal122005_sample.dat).
"""
from __future__ import annotations

import gzip
import os
import urllib.request
from datetime import datetime

TIME_FMT = "%Y%m%d%H"
NHC_BDECK_URL = "https://ftp.nhc.noaa.gov/atcf/archive/{year}/b{basin}{cy}{year}.dat.gz"
RAD_THRESHOLDS_KT = (34, 50, 64)


class ATCFError(RuntimeError):
    pass


def _fmt_latlon(value, pos_letter, neg_letter):
    hemi = pos_letter if value >= 0 else neg_letter
    return "%d%s" % (round(abs(value) * 10), hemi)


def _parse_latlon(token, pos_letter, neg_letter):
    token = token.strip()
    hemi = token[-1].upper()
    magnitude = int(token[:-1]) / 10.0
    if hemi == neg_letter:
        return -magnitude
    if hemi == pos_letter:
        return magnitude
    raise ATCFError("bad ATCF lat/lon token %r (expected trailing %s/%s)"
                    % (token, pos_letter, neg_letter))


def classify_ty(vmax_kt):
    """NHC intensity classification from Vmax alone (no subtropical/extratropical
    distinction -- a model track has no basis to call that)."""
    if vmax_kt is None:
        return "XX"
    if vmax_kt < 34:
        return "TD"
    if vmax_kt < 64:
        return "TS"
    return "HU"


def format_line(basin, cy, time, tech, tau_h, lat, lon, vmax_kt, mslp_hpa, ty=None,
                rad_kt=0, windcode="", radii=(0, 0, 0, 0), stormname=""):
    """One ATCF line. radii = (NE, SE, SW, NW) nmi for rad_kt (ignored when rad_kt==0)."""
    ty = ty or classify_ty(vmax_kt)
    fields = [
        basin.upper(), cy, time.strftime(TIME_FMT), "", tech.upper()[:4],
        "%d" % round(tau_h), _fmt_latlon(lat, "N", "S"), _fmt_latlon(lon, "E", "W"),
        "%d" % round(vmax_kt), "%d" % round(mslp_hpa), ty, "%d" % rad_kt, windcode,
        "%d" % round(radii[0]), "%d" % round(radii[1]), "%d" % round(radii[2]),
        "%d" % round(radii[3]), "", "", "", "", "", "", "", "", "", "",
        stormname.upper(), "",
    ]
    return ", ".join(fields)


def parse_line(line):
    """One ATCF line -> dict. Raises ATCFError on a line with fewer than the 11 mandatory
    leading fields (basin .. ty); trailing optional fields default to None/empty."""
    parts = [p.strip() for p in line.rstrip("\n").split(",")]
    if len(parts) < 11:
        raise ATCFError("not enough comma-delimited fields in ATCF line: %r" % line)
    parts += [""] * (29 - len(parts))

    def _int(tok):
        return int(tok) if tok else None

    return dict(
        basin=parts[0], cy=parts[1], time=datetime.strptime(parts[2], TIME_FMT),
        technum=parts[3], tech=parts[4], tau_h=_int(parts[5]),
        lat=_parse_latlon(parts[6], "N", "S"), lon=_parse_latlon(parts[7], "E", "W"),
        vmax_kt=_int(parts[8]), mslp_hpa=_int(parts[9]), ty=parts[10],
        rad_kt=_int(parts[11]) if len(parts) > 11 else None,
        windcode=parts[12] if len(parts) > 12 else "",
        radii=tuple(_int(parts[i]) for i in range(13, 17)) if len(parts) > 16 else None,
        rmw_nmi=_int(parts[19]) if len(parts) > 19 else None,
        stormname=parts[27] if len(parts) > 27 else "")


def parse_file(path):
    with open(path) as fh:
        return [parse_line(line) for line in fh if line.strip()]


def write_lines(rows, path, basin, cy, tech, stormname=""):
    """rows: [{"time", "tau_h", "lat", "lon", "vmax_kt", "mslp_hpa",
    "radii_kt": {34: (ne,se,sw,nw), 50: (...), 64: (...)}}, ...] (radii_kt may omit or
    zero out thresholds the storm hasn't reached). Writes one basic RAD=0 line per row
    below 34 kt, else one line per present/nonzero threshold -- see module docstring."""
    lines = []
    for row in rows:
        radii_kt = row.get("radii_kt") or {}
        # a threshold the storm has reached gets its own RAD line even if some quadrant's
        # radius came back 0 (small/asymmetric core) -- matching the real archive, which
        # keeps e.g. "60, 60, 0, 0" rows rather than omitting them (bal122005_sample.dat,
        # 2005082412Z)
        present = [th for th in RAD_THRESHOLDS_KT if row["vmax_kt"] >= th]
        if not present:
            lines.append(format_line(basin, cy, row["time"], tech, row["tau_h"], row["lat"],
                                     row["lon"], row["vmax_kt"], row["mslp_hpa"],
                                     stormname=stormname))
            continue
        for th in present:
            lines.append(format_line(basin, cy, row["time"], tech, row["tau_h"], row["lat"],
                                     row["lon"], row["vmax_kt"], row["mslp_hpa"], rad_kt=th,
                                     windcode="NEQ", radii=radii_kt.get(th, (0, 0, 0, 0)),
                                     stormname=stormname))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def fetch_bdeck(year, basin, cy, dest_path, timeout=30):
    """Downloads the NHC ATCF archive b-deck for one storm (https://ftp.nhc.noaa.gov/atcf/
    archive/<year>/b<basin><cy><year>.dat.gz) and writes the decompressed .dat to
    dest_path. Raises ATCFError on any download/decompress failure -- callers fall back to
    HURDAT2 (converted through write_lines(tech="BEST")) when this fails, per --bdeck's
    contract."""
    url = NHC_BDECK_URL.format(year=year, basin=basin.lower(), cy=cy)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            compressed = resp.read()
    except Exception as e:
        raise ATCFError("could not download b-deck from %s: %s" % (url, e))
    try:
        text = gzip.decompress(compressed)
    except OSError as e:
        raise ATCFError("b-deck at %s did not decompress as gzip: %s" % (url, e))
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    with open(dest_path, "wb") as fh:
        fh.write(text)
    return dest_path


def hurdat2_to_bdeck_rows(hurdat2_points):
    """post/tc.py's parse_hurdat2_full() output (vmax_ms/pmin_hpa/r34..r64_*_km/rmw_km,
    time as datetime) -> write_lines() row format (kt/nmi, tau_h=0 -- HURDAT2 carries no
    lead-time concept, it IS the observation)."""
    rows = []
    for p in hurdat2_points:
        vmax_kt = None if p["vmax_ms"] is None else p["vmax_ms"] / 0.514444
        radii_kt = {}
        for th in RAD_THRESHOLDS_KT:
            key = "r%d" % th
            quad = tuple((p.get("%s_%s_km" % (key, q)) or 0.0) / 1.852
                        for q in ("ne", "se", "sw", "nw"))
            if any(quad):
                radii_kt[th] = quad
        rows.append(dict(time=p["time"], tau_h=0, lat=p["lat"], lon=p["lon"],
                         vmax_kt=vmax_kt if vmax_kt is not None else 0,
                         mslp_hpa=p["pmin_hpa"] or 0, radii_kt=radii_kt))
    return rows
