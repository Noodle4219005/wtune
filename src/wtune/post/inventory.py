"""post/inventory.py -- `wtune post inventory RUN_DIR`: domains, times, grid, variables, file format
and a missing-frame check. Grid/variables come from `ncdump -h`, lon/lat extent is cross-checked
with `cdo sinfon` (decision 0056). Only the wrfout_dNN_TIME cadence audit is self-written: no
established tool knows that WTune/WRF convention.
"""
from __future__ import annotations

import glob
import os
from datetime import datetime

from wtune.post import tools


WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"

# attribute names history_interval has been seen under across WRF builds; not all builds
# write it into wrfout at all, so this is best-effort and the inferred gap always runs too.
_HISTORY_INTERVAL_ATTRS = ("HISTORY_INTERVAL", "history_interval", "HISTORY_INTERVAL_M")


def _adios2_bp_dirs(run_dir):
    return sorted(p for p in glob.glob(os.path.join(run_dir, "*.bp")) if os.path.isdir(p))


def _domain_grid(path):
    """Grid facts for one wrfout file: `ncdump -h` for dims/global attrs (authoritative),
    `cdo sinfon` for a lon/lat extent cross-check (best-effort -- None if cdo's grid block
    doesn't parse, never fails the whole inventory over it)."""
    header = tools.parse_ncdump_header(tools.ncdump_header(path))
    attrs = header["global_attrs"]
    dims = header["dimensions"]
    e_we = attrs.get("WEST-EAST_GRID_DIMENSION") or dims.get("west_east_stag")
    e_sn = attrs.get("SOUTH-NORTH_GRID_DIMENSION") or dims.get("south_north_stag")
    e_vert = attrs.get("BOTTOM-TOP_GRID_DIMENSION") or dims.get("bottom_top_stag")
    history_interval = None
    for name in _HISTORY_INTERVAL_ATTRS:
        if name in attrs:
            history_interval = attrs[name]
            break
    try:
        cdo_grid = tools.parse_cdo_sinfon_grid(tools.cdo_sinfon(path))
    except tools.ToolNotFoundError:
        cdo_grid = None
    return dict(dx=attrs.get("DX"), dy=attrs.get("DY"), e_we=e_we, e_sn=e_sn, e_vert=e_vert,
                history_interval_attr=history_interval,
                lon_range=cdo_grid["lon_range"] if cdo_grid else None,
                lat_range=cdo_grid["lat_range"] if cdo_grid else None), header["variables"]


def _missing_frames(times):
    """Gaps in a sorted list of WRF time strings, using the modal (most common) gap as the
    expected interval -- robust to one-off irregular spacing without a namelist to consult.
    Returns (expected_interval_s, [{"after", "before", "gap_s", "n_missing_frames"}])."""
    if len(times) < 3:
        return None, []
    dts = [datetime.strptime(t, WRF_TIME_FMT) for t in times]
    gaps = [(dts[i + 1] - dts[i]).total_seconds() for i in range(len(dts) - 1)]
    counts = {}
    for g in gaps:
        counts[g] = counts.get(g, 0) + 1
    expected = max(counts, key=counts.get)
    missing = []
    for i, g in enumerate(gaps):
        if g > expected * 1.5:
            missing.append(dict(after=times[i], before=times[i + 1], gap_s=g,
                                n_missing_frames=int(round(g / expected)) - 1))
    return expected, missing


def inventory(run_dir: str) -> dict:
    """Returns a dict: format, bp_dirs (ADIOS2 note), domains -> {times, n_times, grid,
    variables, expected_interval_s, missing}."""
    from wtune.post import extract
    bp_dirs = _adios2_bp_dirs(run_dir)
    try:
        frames = extract.file_index(run_dir)    # = compare._frames, minus its CompareError
        error = None
    except extract.BPConversionError as e:  # BP history, converter unavailable
        frames, error = {}, str(e)
    domains = sorted(set(d for d, _ in frames))
    converted = sorted(p for p in frames.values() if extract.bp_origin(p))

    result = dict(run_dir=os.path.abspath(run_dir), format="netcdf" if frames else "unknown",
                  bp_dirs=bp_dirs, domains={})
    if error:
        result["format"] = "adios2_bp"
        result["note"] = error
    elif converted:
        # ADIOS2 BP history (dirs named like the netCDF file, or *.bp), read through the
        # netCDF cache post.extract keeps -- the grid/variables below come from those copies
        result["format"] = "adios2_bp" if len(converted) == len(frames) else "netcdf+adios2_bp"
        result["nc_cache"] = sorted(set(os.path.dirname(p) for p in converted))
        result["note"] = ("ADIOS2 BP history: %d file(s) read through netCDF copies in %s"
                          % (len(converted), ", ".join(result["nc_cache"])))
    elif bp_dirs and not frames:
        result["format"] = "adios2_bp"
        result["note"] = ("ADIOS2 BP directories present, no wrfout history among them -- "
                          "nothing for `wtune post` to read.")

    for dom in domains:
        times = sorted(t for d, t in frames if d == dom)
        path0 = frames[(dom, times[0])]
        grid, variables = _domain_grid(path0)
        variables = sorted(variables)
        expected, missing = _missing_frames(times)
        result["domains"]["d%02d" % dom] = dict(
            n_times=len(times), first_time=times[0], last_time=times[-1], times=times,
            grid=grid, variables=variables,
            expected_interval_s=expected if expected is not None else grid["history_interval_attr"],
            missing=missing)
    return result


def render_summary(inv: dict) -> str:
    lines = ["run_dir: %s  format: %s" % (inv["run_dir"], inv["format"])]
    if inv.get("note"):
        lines.append(inv["note"])
    if inv["bp_dirs"]:
        lines.append("ADIOS2 BP dirs: %s" % ", ".join(os.path.basename(p) for p in inv["bp_dirs"]))
    for dom, d in sorted(inv["domains"].items()):
        g = d["grid"]
        lines.append("%s: %d frames [%s .. %s]  dx=%s e_we=%s e_sn=%s e_vert=%s  %d vars"
                      % (dom, d["n_times"], d["first_time"], d["last_time"], g["dx"], g["e_we"],
                         g["e_sn"], g["e_vert"], len(d["variables"])))
        if d["missing"]:
            for m in d["missing"]:
                lines.append("  MISSING: %d frame(s) between %s and %s (gap=%.0fs, expected=%.0fs)"
                              % (m["n_missing_frames"], m["after"], m["before"], m["gap_s"],
                                 d["expected_interval_s"]))
    return "\n".join(lines)
