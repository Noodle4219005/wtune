"""post/compare.py -- `wtune post compare RUN_A RUN_B`: verify a STRICT-classified change
did not move the answer (decision 0056). Runs WRF's own diffwrf on every common (domain,
time) wrfout frame pair and parses its differing-field table -- no self-written numpy diff
(2026-09-29 update: wrap established tools, delete the substitute). `ncdump -h` validates
a requested --vars name actually exists in the file, since diffwrf itself takes no field
filter and silently reports nothing for a name it never saw.
"""
from __future__ import annotations

import glob
import os
import re

from post import tools
from post.extract import file_index

# diffwrf table row, e.g.:
#          U    832103    3   0.8162436562E+01   0.8159945696E+01   3   0.5795E+00   0.4923E+00
_ROW_RE = re.compile(
    r"^\s*(\S+)\s+(\d+)\s+(\d+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+(\d+)\s+"
    r"([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s*$")


class CompareError(RuntimeError):
    pass


def _frames(run_dir):
    """{(domain_int, file_name_time): path} for every wrfout_dNN_TIME file directly under run_dir,
    or across the segments of a segmented run dir (post.extract.file_index). Keyed per FILE, not
    per valid time: diffwrf compares whole files, so a frames_per_outfile > 1 file is diffed
    once (all its times), not once per valid time it holds."""
    return file_index(run_dir)


def _adios2_bp_dirs(run_dir):
    return sorted(p for p in glob.glob(os.path.join(run_dir, "*.bp")) if os.path.isdir(p))


def _check_not_adios2(run_dir, frames):
    if frames or not _adios2_bp_dirs(run_dir):
        return
    raise CompareError(
        "%s has ADIOS2 BP output, no plain netCDF wrfout files -- convert first with "
        "src/tools/wrf_adios2_to_netcdf4/ before `wtune post compare`." % run_dir)


def parse_diffwrf(text):
    """diffwrf's "Field Ndifs Dims RMS(1) RMS(2) DIGITS RMSE pntwise max" table -> list of
    dicts, one per row (a field absent here had zero differing points). Fortran's `D`
    exponent marker is normalized to `E` before float() (diffwrf never emits it in
    practice, but the format string that could produces one is not ruled out)."""
    rows = []
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m or m.group(1) in ("Field",):
            continue
        field, ndifs, dims, rms1, rms2, digits, rmse, pmax = m.groups()
        rows.append(dict(
            field=field, ndifs=int(ndifs), dims=int(dims),
            rms1=float(rms1.replace("D", "E").replace("d", "e")),
            rms2=float(rms2.replace("D", "E").replace("d", "e")),
            digits=int(digits),
            rmse=float(rmse.replace("D", "E").replace("d", "e")),
            pntwise_max=float(pmax.replace("D", "E").replace("d", "e"))))
    return rows


def _validate_vars(path, variables):
    header = tools.parse_ncdump_header(tools.ncdump_header(path))
    unknown = [v for v in variables if v not in header["variables"]]
    if unknown:
        raise CompareError("--vars names not present in %s: %s" % (path, unknown))


def compare_runs(run_a, run_b, variables=None, domain=None):
    """Compares every common (domain, time) frame between two run directories via diffwrf.

    variables: restrict the reported table to these field names (validated against the
    first compared frame's `ncdump -h` header; diffwrf itself always diffs every field).
    domain: restrict to one domain number (e.g. 1 for d01) -- default: all domains.

    Returns {"identical": bool, "n_frames_compared": int, "n_different": int,
             "results": [{"domain", "time", "field", "ndifs", "dims", "rms1", "rms2",
                          "digits", "rmse", "pntwise_max"}]} -- one result row per
    differing field per compared frame (a frame with no differing fields contributes none).
    """
    diffwrf = tools.find_diffwrf()

    frames_a, frames_b = _frames(run_a), _frames(run_b)
    _check_not_adios2(run_a, frames_a)
    _check_not_adios2(run_b, frames_b)
    common = sorted(set(frames_a) & set(frames_b))
    if domain is not None:
        common = [k for k in common if k[0] == domain]

    if variables and common:
        _validate_vars(frames_a[common[0]], variables)
    wanted = set(variables) if variables else None

    results = []
    for dom, time in common:
        _rc, out, _err = tools.run([diffwrf, frames_a[(dom, time)], frames_b[(dom, time)]],
                                   check=False)
        for row in parse_diffwrf(out):
            if wanted is not None and row["field"] not in wanted:
                continue
            results.append(dict(row, domain=dom, time=time))

    n_different = len(results)
    return dict(identical=(n_different == 0 and bool(common)), n_frames_compared=len(common),
                n_different=n_different, results=results)


def render_summary(comparison):
    if not comparison["results"] and comparison["n_frames_compared"] == 0:
        return "no common (domain, time) frames found to compare"
    if comparison["identical"]:
        return "IDENTICAL (%d frame(s) compared)" % comparison["n_frames_compared"]
    lines = ["DIFFERENT (%d vars)" % comparison["n_different"]]
    for r in comparison["results"]:
        lines.append("  d%02d %s %-10s ndifs=%d rms1=%.6g rms2=%.6g digits=%d rmse=%.6g "
                     "max=%.6g" % (r["domain"], r["time"], r["field"], r["ndifs"], r["rms1"],
                                   r["rms2"], r["digits"], r["rmse"], r["pntwise_max"]))
    return "\n".join(lines)
