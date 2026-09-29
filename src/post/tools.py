"""post/tools.py -- locate and run the established external tools `wtune post` wraps
(decision 0056, 2026-09-29 update: wrap established tools, delete the self-written
substitute). Every locator raises ToolNotFoundError naming the missing tool and how to
get it (bin/install-postproc.sh, README.md 'Installation') -- `wtune post` never silently
falls back to re-implementing what the tool would have done.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess


class ToolNotFoundError(RuntimeError):
    pass


class ToolError(RuntimeError):
    pass


def find_diffwrf():
    """WRF's own netCDF differ, shipped with the WRF source tree, not conda/spack-packaged."""
    wrf_tree = os.environ.get("WRF_TREE")
    if wrf_tree:
        path = os.path.join(wrf_tree, "external", "io_netcdf", "diffwrf")
        if os.path.exists(path):
            return path
    path = shutil.which("diffwrf")
    if path:
        return path
    raise ToolNotFoundError(
        "diffwrf not found: set WRF_TREE to a built WRF source tree (needs "
        "$WRF_TREE/external/io_netcdf/diffwrf) or put diffwrf on PATH -- see README.md "
        "'Installation', layer 2 (WRF and WPS).")


def find_on_path(name, install_hint=None):
    path = shutil.which(name)
    if path:
        return path
    raise ToolNotFoundError(
        "%s not found on PATH.%s" % (name, (" " + install_hint) if install_hint else ""))


def find_postproc_tool(name):
    """MET binaries (tc_pairs, tc_stat, tc_rmw) and nccmp/pvbatch: WTUNE_POSTPROC_VIEW/bin
    (the spack view bin/install-postproc.sh builds) first, else PATH."""
    view = os.environ.get("WTUNE_POSTPROC_VIEW")
    if view:
        path = os.path.join(view, "bin", name)
        if os.path.exists(path):
            return path
    path = shutil.which(name)
    if path:
        return path
    raise ToolNotFoundError(
        "%s not found: install it with bin/install-postproc.sh (README.md 'Installation', "
        "layer 3) and export WTUNE_POSTPROC_VIEW=<repo>/envs/postproc/.spack-env/view, or "
        "put %s on PATH." % (name, name))


def run(cmd, cwd=None, check=True):
    """Runs cmd (list of str), returns (returncode, stdout, stderr). check=True raises
    ToolError with the captured output on nonzero exit."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise ToolError("%s failed (exit %d):\n%s" % (
            " ".join(cmd), proc.returncode, (proc.stderr or proc.stdout).strip()))
    return proc.returncode, proc.stdout, proc.stderr


# --------------------------------------------------------------------------------------
# ncdump -h header: shared by post/inventory.py (grid/variables) and post/compare.py
# (--vars validation), so both wrap the same tool the same way instead of each parsing it.
# --------------------------------------------------------------------------------------

_VAR_DECL_RE = re.compile(
    r"^(?:char|byte|short|int|long|float|real|double|int64|uint|uint64|string)\s+"
    r"(\w+)\s*(?:\(([^)]*)\))?\s*;")
_VAR_ATTR_RE = re.compile(r"^\w+:")
_DIM_RE = re.compile(r"^(\w+)\s*=\s*(UNLIMITED|\d+)")
_GLOBAL_ATTR_RE = re.compile(r"^:(\w+)\s*=\s*(.+?);?\s*$")


def ncdump_header(path):
    """`ncdump -h PATH` text. Raises ToolNotFoundError if ncdump is missing, ToolError if
    ncdump itself fails (e.g. not a netCDF file)."""
    ncdump = find_on_path("ncdump", install_hint="conda-forge 'netcdf4'/'libnetcdf' or "
                          "envs/wtune/pixi.toml provide it.")
    _rc, out, _err = run([ncdump, "-h", path])
    return out


def _coerce_attr(raw):
    raw = raw.strip().rstrip(",")
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    try:
        return int(raw.rstrip("Ll"))
    except ValueError:
        pass
    try:
        return float(raw.rstrip("fFdD"))
    except ValueError:
        return raw


def parse_ncdump_header(text):
    """Parses `ncdump -h` output. Returns {"dimensions": {name: size (None if UNLIMITED)},
    "variables": [name, ...] (declaration order), "global_attrs": {name: value}}."""
    dims, variables, attrs = {}, [], {}
    section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "dimensions:":
            section = "dims"
            continue
        if line == "variables:":
            section = "vars"
            continue
        if line.startswith("// global attributes"):
            section = "attrs"
            continue
        if not line or line.startswith("//") or line in ("}", "netcdf"):
            continue
        if section == "dims":
            m = _DIM_RE.match(line)
            if m:
                name, val = m.groups()
                dims[name] = None if val == "UNLIMITED" else int(val)
        elif section == "vars":
            if _VAR_ATTR_RE.match(line):
                continue
            m = _VAR_DECL_RE.match(line)
            if m:
                variables.append(m.group(1))
        elif section == "attrs":
            m = _GLOBAL_ATTR_RE.match(line)
            if m:
                attrs[m.group(1)] = _coerce_attr(m.group(2))
    return dict(dimensions=dims, variables=variables, global_attrs=attrs)


# --------------------------------------------------------------------------------------
# cdo sinfon: post/inventory.py's grid lat/lon extent cross-check.
# --------------------------------------------------------------------------------------

_CDO_GRID_POINTS_RE = re.compile(r":\s*points=\d+\s*\((\d+)x(\d+)\)")
_CDO_RANGE_RE = re.compile(r":\s*(-?[\d.eE+-]+)\s+to\s+(-?[\d.eE+-]+)")
_CDO_NEXT_ENTRY_RE = re.compile(r"^\s*\d+\s*:")


def cdo_sinfon(path):
    """`cdo sinfon PATH` text. Raises ToolNotFoundError if cdo is missing."""
    cdo = find_on_path("cdo", install_hint="conda-forge 'cdo' or envs/wtune/pixi.toml "
                       "provides it.")
    # cdo exits nonzero on the "Unsupported data type (char/string) ... skipped variable
    # Times" warning WRF's Times variable always triggers; the sinfon listing itself is
    # still complete and correct, so don't treat that as failure here.
    _rc, out, err = run([cdo, "sinfon", path], check=False)
    return out or err


def parse_cdo_sinfon_grid(text):
    """First curvilinear/lonlat grid entry's point count and lon/lat extent from `cdo
    sinfon` output (its 'Grid coordinates' section). None if no such block is found."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _CDO_GRID_POINTS_RE.search(line)
        if not m:
            continue
        nx, ny = int(m.group(1)), int(m.group(2))
        lon_range = lat_range = None
        for follow in lines[i + 1:i + 4]:
            if _CDO_NEXT_ENTRY_RE.match(follow) or "coordinate" in follow.lower():
                break
            fs = follow.strip().upper()
            rm = _CDO_RANGE_RE.search(follow)
            if not rm:
                continue
            if fs.startswith(("XLONG", "LON")):
                lon_range = (float(rm.group(1)), float(rm.group(2)))
            elif fs.startswith(("XLAT", "LAT")):
                lat_range = (float(rm.group(1)), float(rm.group(2)))
        return dict(nx=nx, ny=ny, lon_range=lon_range, lat_range=lat_range)
    return None
