#!/usr/bin/env python3
"""Tropical-cyclone vortex bogussing for rebuilt (cold-started) segments, via WRF's own tc.exe.

Phase 5 of the fast-prediction plan: a segment rebuilt from coarse driving data starts with a
hurricane that is far too weak (SC26-P2 / Katrina 2005-08-27 00Z: NAM-ANL 996 hPa vs best track
959 hPa). WRF ships `tc.exe` (main/tc_em.F), which removes the analysed vortex and inserts a
Rankine vortex balanced against the large-scale flow. This module supplies what tc.exe needs:

* `parse_hurdat2` / `fix_at` / `storms_at` -- NHC HURDAT2 best track, linearly interpolated to
  the segment's valid time (fixes are 6-hourly plus off-synoptic landfall/peak records);
* `render_tc_block` -- the `&tc` namelist group;
* `apply_tc` -- run tc.exe on one met_em file in a scratch directory and return the bogussed
  met_em (which real.exe then consumes unchanged).

tc.exe contract (read from WRF main/tc_em.F + Registry, WRF 4.x):
  input   `auxinput1_inname`  default "met_em.d<domain>.<date>" (+ ".nc" for io_form 2), at the
          namelist's d01 start date; only domain 1 is bogussed; must run on ONE MPI rank.
  output  `auxinput1_outname` default "auxinput1_d<domain>_<date>" (+ ".nc"): U/V/T/RH/GHT on the
          metgrid levels plus PMSL, PSFC and PRES(level 1) replaced -- but titled "OUTPUT FROM TC
          BOGUS" and missing ~20 metgrid fields and all FLAG_* attributes, so apply_tc merges
          only those 8 fields back into a copy of the original met_em (merge_tc_fields).
          Rank-0 stdout of this DM build goes to rsl.out.0000, not to the launcher's stdout.
  needs   isobaric metgrid levels (it reads the pressure column at grid point (1,1) as the level
          set), `vmax_ratio` set (Registry default -999 -> nonsense vortex), `rmax` set (else it
          STOPs), p_top_requested not 40000/60000. `rankine_lid` is read but unused by tc_bogus.
  vortex  v(r) = vmx*(r/rmax) inside rmax, vmx*(r/rmax)**-0.75 outside, vmx = vmax*vmax_ratio,
          full strength below 600 hPa tapering to 0 at 100 hPa; the old vortex is removed within
          300 km of the max-vorticity point found within 400 km of (latc_loc, lonc_loc).

Parameter defaults (documented, not fitted to any verification target):
  vmax  best-track 1-min 10-m wind, kt -> m/s (x 0.514444).
  rmax  best-track RMW (HURDAT2 column 21, n mi, present from 2021 on) -> m; otherwise the
        Willoughby, Darling & Rahn (2006, MWR 134:1102) climatology
        RMW[km] = 46.4 * exp(-0.0155*Vmax[m/s] + 0.0169*|lat|).
  vmax_ratio  WRF README.namelist: 0.75 for a 45 km grid, 0.9 for 15 km; linear in dx through
        those two anchors, clamped to [0.75, 1.0] (a finer grid resolves more of the peak wind).

CLI (run from src/):
  python -m tools.tc_bogus plan  --hurdat2 FILE --time YYYY-MM-DD_HH --bbox lat0,lat1,lon0,lon1 [--dx-m 12000]
  python -m tools.tc_bogus apply --met-em FILE --namelist NAMELIST --tc-block FILE --out DIR [--tc-exe PATH] [--launcher "mpirun -np 1"]
  python -m tools.tc_bogus measure --wrfinput FILE --lat LAT --lon LON [--radius-km 300]
"""
from __future__ import annotations

import argparse
import datetime
import glob
import math
import os
import re
import shlex
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

KT_TO_MS = 0.514444
NMI_TO_M = 1852.0
MAX_BOGUS = 5                      # WRF frame/module_driver_constants.F max_bogus
TROPICAL_STATUS = frozenset(("TD", "TS", "HU", "SD", "SS"))
_WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
_RADII_KEYS = tuple("r%d_%s" % (kt, q) for kt in (34, 50, 64) for q in ("ne", "se", "sw", "nw"))


class TCBogusError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# HURDAT2
# ---------------------------------------------------------------------------

def _val(tok):
    """HURDAT2 numeric field; -999 (and -99 for older pressure gaps) -> None."""
    tok = tok.strip()
    if not tok:
        return None
    v = int(tok)
    return None if v <= -99 else v


def _latlon(tok):
    tok = tok.strip()
    x = float(tok[:-1])
    return -x if tok[-1] in "SW" else x


def parse_hurdat2(path_or_lines):
    """Returns a list of storms: {"id", "name", "fixes": [fix, ...]} with fix =
    {"time": datetime, "record": str, "status": str, "lat", "lon", "vmax_kt", "pmin_mb",
     "r34_ne".."r64_nw" (n mi or None), "rmw_nmi" (None before 2021)}."""
    if isinstance(path_or_lines, str):
        with open(path_or_lines) as fh:
            lines = fh.read().splitlines()
    else:
        lines = list(path_or_lines)
    storms, cur = [], None
    for ln in lines:
        if not ln.strip():
            continue
        f = [t.strip() for t in ln.split(",")]
        if re.match(r"^[A-Z]{2}\d{6}$", f[0]):
            cur = {"id": f[0], "name": f[1], "fixes": []}
            storms.append(cur)
            continue
        if cur is None:
            raise TCBogusError("HURDAT2 data line before any header: %r" % ln[:40])
        t = datetime.datetime.strptime(f[0] + f[1].zfill(4), "%Y%m%d%H%M")
        fix = {"time": t, "record": f[2], "status": f[3], "lat": _latlon(f[4]),
               "lon": _latlon(f[5]), "vmax_kt": _val(f[6]), "pmin_mb": _val(f[7])}
        for i, k in enumerate(_RADII_KEYS):
            fix[k] = _val(f[8 + i]) if len(f) > 8 + i else None
        fix["rmw_nmi"] = _val(f[20]) if len(f) > 20 else None
        cur["fixes"].append(fix)
    for s in storms:
        s["fixes"].sort(key=lambda x: x["time"])
    return storms


def _lerp(a, b, w):
    if a is None or b is None:
        return None
    return a + (b - a) * w


def fix_at(storm, when):
    """Best-track state of `storm` at `when`: an exact fix, or a linear interpolation between
    the two bracketing fixes (lat, lon, vmax, pmin, radii, RMW; a quantity missing at either end
    stays None). Status is the earlier fix's. None when `when` is outside the track."""
    fx = storm["fixes"]
    if not fx or when < fx[0]["time"] or when > fx[-1]["time"]:
        return None
    for i, f in enumerate(fx):
        if f["time"] == when:
            out = dict(f)
            out["interpolated"] = False
            break
        if f["time"] > when:
            a, b = fx[i - 1], f
            w = (when - a["time"]).total_seconds() / (b["time"] - a["time"]).total_seconds()
            dlon = b["lon"] - a["lon"]
            if dlon > 180.0:
                dlon -= 360.0
            elif dlon < -180.0:
                dlon += 360.0
            lon = a["lon"] + dlon * w
            lon = (lon + 180.0) % 360.0 - 180.0
            out = {"time": when, "record": "", "status": a["status"],
                   "lat": _lerp(a["lat"], b["lat"], w), "lon": lon, "interpolated": True}
            for k in ("vmax_kt", "pmin_mb", "rmw_nmi") + _RADII_KEYS:
                out[k] = _lerp(a[k], b[k], w)
            break
    out["id"], out["name"] = storm["id"], storm["name"]
    return out


def in_bbox(lat, lon, bbox):
    lat0, lat1, lon0, lon1 = bbox
    if not (min(lat0, lat1) <= lat <= max(lat0, lat1)):
        return False
    if lon0 <= lon1:
        return lon0 <= lon <= lon1
    return lon >= lon0 or lon <= lon1          # bbox crossing the dateline


def storms_at(storms, when, bbox=None):
    """Interpolated fixes of every storm alive at `when` (and inside `bbox` if given)."""
    out = []
    for s in storms:
        f = fix_at(s, when)
        if f is not None and (bbox is None or in_bbox(f["lat"], f["lon"], bbox)):
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# &tc block
# ---------------------------------------------------------------------------

def default_rmax_m(vmax_ms, lat):
    """Willoughby, Darling & Rahn (2006) RMW climatology, in meters."""
    return 1000.0 * 46.4 * math.exp(-0.0155 * vmax_ms + 0.0169 * abs(lat))


def vmax_ratio_for_dx(dx_m):
    """README.namelist anchors (45 km -> 0.75, 15 km -> 0.9), linear in dx, clamped [0.75, 1]."""
    r = 0.9 + (15000.0 - float(dx_m)) * (0.15 / 30000.0)
    return max(0.75, min(1.0, r))


def bogus_params(fix, rmax_m=None):
    """(vmax m/s, rmax m, rmax_source) for one interpolated fix."""
    if fix.get("vmax_kt") is None:
        raise TCBogusError("%s %s: no best-track vmax at this time" % (fix["id"], fix["name"]))
    vmax = fix["vmax_kt"] * KT_TO_MS
    if rmax_m is not None:
        return vmax, float(rmax_m), "override"
    if fix.get("rmw_nmi") not in (None, 0):
        return vmax, fix["rmw_nmi"] * NMI_TO_M, "best-track RMW"
    return vmax, default_rmax_m(vmax, fix["lat"]), "Willoughby et al. 2006"


def _fmt_list(vals, fmt):
    return ", ".join(fmt % v for v in vals) + ","


def render_tc_block(fixes, dx_m=None, vmax_ratio=None, rmax_m=None, remove_only=False):
    """The `&tc` namelist group bogussing every fix in `fixes` (tropical/subtropical only; at
    most MAX_BOGUS). `vmax_ratio` defaults to vmax_ratio_for_dx(dx_m); `rmax_m` overrides the
    RMW for every storm (sensitivity runs only)."""
    fixes = [f for f in fixes if f.get("status") in TROPICAL_STATUS]
    if not fixes:
        raise TCBogusError("no tropical/subtropical storm to bogus")
    if len(fixes) > MAX_BOGUS:
        raise TCBogusError("%d storms > max_bogus=%d" % (len(fixes), MAX_BOGUS))
    if vmax_ratio is None:
        if dx_m is None:
            raise TCBogusError("need dx_m or an explicit vmax_ratio")
        vmax_ratio = vmax_ratio_for_dx(dx_m)
    params = [bogus_params(f, rmax_m) for f in fixes]
    lines = ["&tc",
             " insert_bogus_storm = .%s." % ("false" if remove_only else "true"),
             " remove_storm = .%s." % ("true" if remove_only else "false"),
             " num_storm = %d," % len(fixes),
             " latc_loc = " + _fmt_list([f["lat"] for f in fixes], "%.2f"),
             " lonc_loc = " + _fmt_list([f["lon"] for f in fixes], "%.2f"),
             " vmax_meters_per_second = " + _fmt_list([p[0] for p in params], "%.2f"),
             " rmax = " + _fmt_list([p[1] for p in params], "%.0f"),
             " vmax_ratio = " + _fmt_list([vmax_ratio] * len(fixes), "%.3f"),
             "/"]
    notes = ["! %s %s %s: lat %.2f lon %.2f vmax %.0f kt pmin %s mb, rmax from %s"
             % (f["id"], f["name"], f["time"].strftime("%Y-%m-%d_%H:%M"), f["lat"], f["lon"],
                f["vmax_kt"], "?" if f["pmin_mb"] is None else "%.0f" % f["pmin_mb"], p[2])
             for f, p in zip(fixes, params)]
    return "\n".join(notes + lines) + "\n"


# ---------------------------------------------------------------------------
# tc.exe
# ---------------------------------------------------------------------------

_TC_GROUP_RE = re.compile(r"^[ \t]*&tc\b.*?^[ \t]*/[ \t]*$\n?", re.I | re.M | re.S)
# the met_em fields tc_em.F rewrites (u_gc v_gc t_gc rh_gc ght_gc pslv_gc psfc p_gc)
TC_FIELDS = ("UU", "VV", "TT", "RH", "GHT", "PMSL", "PSFC", "PRES")
_MET_EM_RE = re.compile(r"met_em\.d(\d\d)\.(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)")


def met_em_date(path):
    m = _MET_EM_RE.search(os.path.basename(path))
    if not m:
        raise TCBogusError("not a met_em file name: %s" % path)
    return int(m.group(1)), datetime.datetime.strptime(m.group(2), _WRF_TIME_FMT)


def build_tc_namelist(namelist_text, tc_block, when):
    """The namelist tc.exe reads: the case/real namelist with max_dom=1, d01 start=end=`when`,
    any existing &tc group dropped and `tc_block` appended."""
    import coldstart_sample as cs            # stdlib-only at module level
    txt = _TC_GROUP_RE.sub("", namelist_text)
    txt = cs.set_list(txt, "max_dom", 1, 1, new_key_group="domains")
    for pre in ("start_", "end_"):
        for k, v in (("year", when.year), ("month", "%02d" % when.month), ("day", "%02d" % when.day),
                     ("hour", "%02d" % when.hour), ("minute", "%02d" % when.minute),
                     ("second", "%02d" % when.second)):
            txt = cs.set_list(txt, pre + k, v, 1)
    for k in ("run_days", "run_hours", "run_minutes", "run_seconds"):
        if cs.count_key(txt, k):
            txt = cs.set_list(txt, k, 0, 1)
    if not re.search(r"^\s*&tc\b", tc_block, re.I | re.M):
        raise TCBogusError("tc_block has no &tc group")
    return txt.rstrip("\n") + "\n\n" + tc_block.strip("\n") + "\n"


def merge_tc_fields(met_em, tc_out, dst, provenance=""):
    """`dst` = a copy of the metgrid file with only TC_FIELDS taken from tc.exe's output.

    tc.exe's own output is NOT a usable met_em: it is titled "OUTPUT FROM TC BOGUS" (real.exe
    rejects it as pre-v4 input unless force_use_old_data) and drops ~20 metgrid variables and
    every FLAG_* attribute (LANDSEA, LAI12M, OA*/OL*, CLONG, FLAG_SOILHGT, ...), so real.exe fed
    with it would differ from the unbogussed run for reasons unrelated to the vortex. Measured on
    SC26-P2 (scc job 1283): the fields tc.exe changes are exactly TC_FIELDS."""
    import netCDF4
    import numpy as np
    shutil.copyfile(met_em, dst)
    with netCDF4.Dataset(tc_out) as src, netCDF4.Dataset(dst, "r+") as out:
        for k in TC_FIELDS:
            if k not in src.variables or k not in out.variables:
                raise TCBogusError("merge: %s missing in %s" % (k, tc_out if k not in src.variables else met_em))
            v = np.asarray(src.variables[k][:])
            if v.shape != out.variables[k].shape:
                raise TCBogusError("merge: %s shape %s != %s" % (k, v.shape, out.variables[k].shape))
            out.variables[k][:] = v
        out.setncattr("TC_BOGUS", provenance or "tc.exe")
    return dst


def apply_tc(met_em, namelist_text, tc_block, out_dir, tc_exe, launcher=(), timeout=3600):
    """Run tc.exe on `met_em` in `out_dir`/tc_work; returns the bogussed file, written as
    `out_dir`/<same met_em basename> (the metgrid file with TC_FIELDS replaced, see
    merge_tc_fields) so real.exe picks it up unchanged."""
    dom, when = met_em_date(met_em)
    if dom != 1:
        raise TCBogusError("tc.exe only bogusses domain 1 (got d%02d)" % dom)
    work = os.path.join(out_dir, "tc_work")
    os.makedirs(work, exist_ok=True)
    stamp = when.strftime(_WRF_TIME_FMT)
    for old in (glob.glob(os.path.join(work, "auxinput1_d01_*")) + glob.glob(os.path.join(work, "met_em.d01.*"))
                + glob.glob(os.path.join(work, "rsl.*"))):
        os.remove(old)
    os.symlink(os.path.abspath(met_em), os.path.join(work, "met_em.d01.%s.nc" % stamp))
    with open(os.path.join(work, "namelist.input"), "w") as fh:
        fh.write(build_tc_namelist(namelist_text, tc_block, when))
    cmd = list(launcher) + [tc_exe]
    with open(os.path.join(work, "tc.log"), "w") as log:
        rc = subprocess.run(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
    outs = sorted(glob.glob(os.path.join(work, "auxinput1_d01_%s*" % stamp)))
    logtxt = ""
    # a DM_PARALLEL build redirects rank 0's stdout to rsl.out.0000 / rsl.error.0000
    for name in ["tc.log"] + sorted(glob.glob(os.path.join(work, "rsl.*"))):
        with open(os.path.join(work, name), errors="replace") as fh:
            logtxt += fh.read()
    n_storm = int(re.search(r"^\s*num_storm\s*=\s*(\d+)", tc_block, re.I | re.M).group(1))
    n_done = len(re.findall(r"Bogus storm number\s+\d+\s+completed", logtxt))
    if rc != 0 or not outs or n_done < n_storm:
        raise TCBogusError("tc.exe failed rc=%s outputs=%s storms_completed=%d/%d (see %s/tc.log)"
                           % (rc, outs, n_done, n_storm, work))
    dst = os.path.join(out_dir, os.path.basename(met_em))
    if os.path.abspath(dst) == os.path.abspath(met_em):
        raise TCBogusError("--out would overwrite the input met_em")
    prov = " ".join(l.strip() for l in tc_block.splitlines() if l.strip() and not l.lstrip().startswith("!"))
    return merge_tc_fields(met_em, outs[0], dst, prov)


# ---------------------------------------------------------------------------
# verification helper
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def min_slp_near(wrfinput, lat, lon, radius_km=300.0):
    """Minimum sea-level pressure (NCL wrf_slp algorithm) within `radius_km` of (lat, lon)."""
    import numpy as np
    import netCDF4
    from predictor.wrf_slp_vendored import slp_from_frame
    with netCDF4.Dataset(wrfinput) as nc:
        g = {k: np.asarray(nc.variables[k][0], dtype=np.float64)
             for k in ("P", "PB", "T", "QVAPOR", "PH", "PHB", "XLAT", "XLONG")}
    slp = slp_from_frame(g["P"], g["PB"], g["T"], g["QVAPOR"], g["PH"], g["PHB"])
    d = haversine_km(lat, lon, g["XLAT"], g["XLONG"])
    m = d <= radius_km
    if not m.any():
        raise TCBogusError("no grid point within %.0f km of %.2f,%.2f" % (radius_km, lat, lon))
    idx = np.unravel_index(np.argmin(np.where(m, slp, np.inf)), slp.shape)
    return {"min_slp_hpa": float(slp[idx]), "lat": float(g["XLAT"][idx]),
            "lon": float(g["XLONG"][idx]), "dist_km": float(d[idx])}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_time(s):
    for fmt in ("%Y-%m-%d_%H", "%Y-%m-%d_%H:%M", "%Y-%m-%d_%H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("time must be YYYY-MM-DD_HH[:MM[:SS]]: %s" % s)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tc_bogus", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan")
    sp.add_argument("--hurdat2", required=True)
    sp.add_argument("--time", required=True, type=_parse_time)
    sp.add_argument("--bbox", required=True, metavar="LAT0,LAT1,LON0,LON1")
    sp.add_argument("--dx-m", type=float, help="grid spacing for the vmax_ratio default")
    sp.add_argument("--vmax-ratio", type=float)
    sp.add_argument("--rmax-m", type=float, help="override every storm's rmax (sensitivity only)")
    sp.add_argument("--out", help="also write the &tc block here")
    sp = sub.add_parser("apply")
    sp.add_argument("--met-em", required=True)
    sp.add_argument("--namelist", required=True)
    sp.add_argument("--tc-block", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--tc-exe", default=os.path.join(os.environ.get("WRF_TREE", ""), "main", "tc.exe"))
    sp.add_argument("--launcher", default="", help='e.g. "mpirun -np 1"')
    sp.add_argument("--timeout", type=float, default=3600)
    sp = sub.add_parser("measure")
    sp.add_argument("--wrfinput", required=True)
    sp.add_argument("--lat", required=True, type=float)
    sp.add_argument("--lon", required=True, type=float)
    sp.add_argument("--radius-km", type=float, default=300.0)
    a = ap.parse_args(argv)

    if a.cmd == "plan":
        bbox = tuple(float(x) for x in a.bbox.split(","))
        if len(bbox) != 4:
            ap.error("--bbox needs 4 numbers")
        fixes = storms_at(parse_hurdat2(a.hurdat2), a.time, bbox)
        for f in fixes:
            print("%s %-12s %s %s lat %.2f lon %.2f vmax %s kt pmin %s mb rmw %s nmi%s"
                  % (f["id"], f["name"], f["time"].strftime("%Y-%m-%d_%H:%M"), f["status"],
                     f["lat"], f["lon"], f["vmax_kt"], f["pmin_mb"], f["rmw_nmi"],
                     " (interpolated)" if f["interpolated"] else ""))
        if not fixes:
            print("no storm in bbox at %s" % a.time)
            return 1
        block = render_tc_block(fixes, dx_m=a.dx_m, vmax_ratio=a.vmax_ratio, rmax_m=a.rmax_m)
        print(block, end="")
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(block)
        return 0
    if a.cmd == "apply":
        with open(a.namelist) as fh:
            nl = fh.read()
        with open(a.tc_block) as fh:
            blk = fh.read()
        os.makedirs(a.out, exist_ok=True)
        print(apply_tc(a.met_em, nl, blk, a.out, a.tc_exe, shlex.split(a.launcher), a.timeout))
        return 0
    r = min_slp_near(a.wrfinput, a.lat, a.lon, a.radius_km)
    print("min_slp_hpa=%.1f lat=%.2f lon=%.2f dist_km=%.0f" % (r["min_slp_hpa"], r["lat"], r["lon"], r["dist_km"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
