"""post/compare.py -- `wtune post compare RUN_A RUN_B`: verify a STRICT change did not move the
answer (decision 0056). Runs WRF's diffwrf on every common (domain, time) wrfout pair and parses its
differing-field table; `ncdump -h` checks a requested --vars name exists (diffwrf takes no field
filter and silently reports nothing for an unknown name).
"""
from __future__ import annotations

import glob
import os
import re

from post import tools
from post.extract import file_index, file_times

# diffwrf table row, e.g.:
#          U    832103    3   0.8162436562E+01   0.8159945696E+01   3   0.5795E+00   0.4923E+00
# A field that became NaN/Infinity prints that in the RMS columns (DIGITS is then junk).
_NUM = r"([-+0-9.EeDd]+|[-+]?(?:NaN|Infinity|Inf))"
_ROW_RE = re.compile(
    r"^\s*(\S+)\s+(\d+)\s+(\d+)\s+" + _NUM + r"\s+" + _NUM + r"\s+(-?\d+|\*+)\s+"
    + _NUM + r"\s+" + _NUM + r"\s*$", re.IGNORECASE)
# diffwrf exits 0 on these (WRF external/io_netcdf/diffwrf.F90) and prints no table row.
_FATAL_MARKERS = ("Big difference", "differ big time", "Error reading", "could not open",
                  "error opening")


class CompareError(RuntimeError):
    pass


def _frames(run_dir):
    """{(domain_int, file_name_time): path} for every wrfout_dNN_TIME file directly under run_dir,
    or across the segments of a segmented run dir (post.extract.file_index). Keyed per FILE, not
    per valid time: diffwrf compares whole files, so a frames_per_outfile > 1 file is diffed
    once (all its times), not once per valid time it holds. ADIOS2 BP history is diffed on its
    netCDF cache copies (post.extract); BPConversionError (no converter) -> CompareError."""
    from post.extract import BPConversionError
    try:
        return file_index(run_dir)
    except BPConversionError as e:
        raise CompareError(str(e)) from e


def _adios2_bp_dirs(run_dir):
    return sorted(p for p in glob.glob(os.path.join(run_dir, "*.bp")) if os.path.isdir(p))


def _check_not_adios2(run_dir, frames):
    if frames or not _adios2_bp_dirs(run_dir):
        return
    raise CompareError(
        "%s has ADIOS2 BP directories but no wrfout history (netCDF or BP wrfout_dNN_TIME[.bp]) "
        "-- nothing for `wtune post compare` to diff." % run_dir)


def parse_diffwrf(text: str) -> list[dict]:
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
            digits=int(digits) if digits.lstrip("-").isdigit() else 0,
            rmse=float(rmse.replace("D", "E").replace("d", "e")),
            pntwise_max=float(pmax.replace("D", "E").replace("d", "e"))))
    return rows


def _validate_vars(path, variables):
    header = tools.parse_ncdump_header(tools.ncdump_header(path))
    unknown = [v for v in variables if v not in header["variables"]]
    if unknown:
        raise CompareError("--vars names not present in %s: %s" % (path, unknown))


def _run_diffwrf(diffwrf, path_a, path_b):
    """diffwrf's stdout table text. Fails closed: a nonzero exit, a missing table header (an
    unopenable file makes diffwrf plot instead of diff) or one of its own "cannot compare"
    messages is a CompareError, never an empty (= identical) table."""
    rc, out, err = tools.run([diffwrf, path_a, path_b], check=False)
    bad = [ln.strip() for ln in out.splitlines() if any(m in ln for m in _FATAL_MARKERS)]
    if rc != 0 or bad or not re.search(r"^\s*Field\s+Ndifs\b", out, re.MULTILINE):
        detail = bad[0] if bad else ((err or out).strip().splitlines() or ["no diff table"])[-1]
        raise CompareError("diffwrf could not compare %s and %s (exit %d): %s"
                           % (path_a, path_b, rc, detail))
    return out


def compare_runs(run_a: str, run_b: str, variables=None, domain: int | None = None) -> dict:
    """Compare every common (domain, time) frame of two run dirs via diffwrf. variables restricts
    the table (validated against the first frame's `ncdump -h`); domain restricts to one domain.
    Returns {identical, n_frames_compared, n_different, only_in_a, only_in_b, results: [{domain,
    time, field, ndifs, dims, rms1, rms2, digits, rmse, pntwise_max}]}, one row per differing
    field per frame. only_in_a/only_in_b: [[domain, time], ...] present in one run only (a
    whole file, or a valid time inside a common file) -- never identical when non-empty."""
    diffwrf = tools.find_diffwrf()

    frames_a, frames_b = _frames(run_a), _frames(run_b)
    _check_not_adios2(run_a, frames_a)
    _check_not_adios2(run_b, frames_b)
    keys_a, keys_b = ({k for k in f if domain is None or k[0] == domain}
                      for f in (frames_a, frames_b))
    common = sorted(keys_a & keys_b)
    only_a, only_b = keys_a - keys_b, keys_b - keys_a
    for dom, time in common:        # diffwrf stops silently at the shorter file's last time
        ta, tb = (file_times(f[(dom, time)]) or [time] for f in (frames_a, frames_b))
        only_a |= {(dom, t) for t in ta if t not in tb}
        only_b |= {(dom, t) for t in tb if t not in ta}

    if variables and common:
        _validate_vars(frames_a[common[0]], variables)
    wanted = set(variables) if variables else None

    results = []
    for dom, time in common:
        out = _run_diffwrf(diffwrf, frames_a[(dom, time)], frames_b[(dom, time)])
        for row in parse_diffwrf(out):
            if wanted is not None and row["field"] not in wanted:
                continue
            results.append(dict(row, domain=dom, time=time))

    n_different = len(results)
    return dict(identical=(n_different == 0 and bool(common) and not only_a and not only_b),
                n_frames_compared=len(common), n_different=n_different,
                only_in_a=[list(k) for k in sorted(only_a)],
                only_in_b=[list(k) for k in sorted(only_b)], results=results)


def render_summary(comparison: dict) -> str:
    if not comparison["results"] and comparison["n_frames_compared"] == 0:
        return "no common (domain, time) frames found to compare"
    if comparison["identical"]:
        return "IDENTICAL (%d frame(s) compared)" % comparison["n_frames_compared"]
    lines = ["DIFFERENT (%d vars)" % comparison["n_different"]]
    for side in ("a", "b"):
        for dom, time in comparison.get("only_in_" + side, []):
            lines.append("  d%02d %s only in run %s" % (dom, time, side.upper()))
    for r in comparison["results"]:
        lines.append("  d%02d %s %-10s ndifs=%d rms1=%.6g rms2=%.6g digits=%d rmse=%.6g "
                     "max=%.6g" % (r["domain"], r["time"], r["field"], r["ndifs"], r["rms1"],
                                   r["rms2"], r["digits"], r["rmse"], r["pntwise_max"]))
    return "\n".join(lines)
