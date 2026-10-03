#!/usr/bin/env python3
"""wtune_bp.py -- shared ADIOS2 (WRF io_form 14) helpers: BP detection, netCDF <-> BP
conversion, a manifest-cached input converter, round-trip verification, the run-dir
adios2.xml, and the `wtune convert` CLI.

Converters
  nc -> BP : src/tools/wrf_nc2bp.py (restored from 8ba1629, validated on f1: convert job
             1071692, regression gate 1072441 -- WRF with BP5 inputs + BP5 history bit-identical
             to the netCDF path via diffwrf). Default engine BP4 (see adios2.xml below); BP5 on
             request. Multi-record inputs (wrfbdy/wrflowinp/wrffdda) convert to the same layout
             WRF writes itself, but WRF's io_adios2 reader returns BP step 0 for every field
             read (wrf_io.F90:652-660 sets the step selection only for writes), so WRF gets
             record 1's fields for every boundary/SST/FDDA record -- with WRF's own BP wrfbdy
             too (scc 2026-10-03, jobs 2778/2780). The converter's stderr WARNING says so and
             nc2bp relays it. convert_inputs therefore writes those names per record:
             cache/<name>/ is a plain directory of one-step BPs <name>_<date>.bp (WRF's own
             current_timestr), read by WRF one file per interval via multi_bdy_files / a
             dated auxinput name (per_record_namelist_edits, bp_input_links).
  BP -> nc : src/tools/wrf_adios2_to_netcdf4/convert.py (vendored upstream + patches, see its
             UPSTREAM.md). Serial path only; float/int32/string variables only (its typemap).

Both need the `adios2` Python module, which this module never imports: conda-forge adios2 2.10
conflicts with the hdf5 pinned in envs/wtune, so it lives in its own env, envs/adios2 (adios2
2.10.* nompi, matching the ADIOS2 2.10.2 library WRF is linked against). The converters run as a
subprocess of that interpreter: $WTUNE_ADIOS2_PYTHON if set (no fallback), else
envs/adios2/.pixi/envs/default/bin/python, else a `python`/`python3` on PATH that imports adios2
(adios2_python()). The child gets PYTHONPATH/PYTHONHOME/LD_LIBRARY_PATH removed so a loaded
nvhpc/spack ADIOS2 or the caller's env cannot shadow the conda libraries. Everything else here --
is_bp, manifest bookkeeping, the netCDF comparison (netCDF4 is in envs/wtune), adios2_xml, CLI
parsing -- runs in-process without adios2.

adios2.xml: WRF reads ./adios2.xml (hard-coded path, external/io_adios2/wrf_io.F90:1656) and
declares one ADIOS2 IO per file, named by the file name itself (DH%FileName, wrf_io.F90:1233-1236),
so an <io> entry must name the exact file, timestamp included -- there is no per-stream wildcard.
Restart files need BP4: WRF opens a file for read and queries `Times` without BeginStep, which
BP5's streaming read mode refuses (FATAL `adios2 error in ext_adios2_open_for_read_begin`, code 4);
with BP4 selected for the restart file on both the writing and the reading run the readback works
and is bit-identical to the netCDF restart (wiki: 2026-09-27
adios2-bp4-restart-via-adios2-xml-is-bit-identical-and-300x-faster). The same open path reads
the INPUT files, so nc2bp writes inputs as BP4 (DEFAULT_ENGINE): on scc (WRF 4.8 + ADIOS2 2.10.2
nvhpc) a BP5 wrfinput_d01.bp fails with exactly that code-4 FATAL while the BP4 conversion runs
and gives history bit-identical to netCDF inputs (SC26-P2, 2026-10-03, jobs 2734-2736; the f1
BP5-input gate 1072441 does not carry over). History is written by WRF with its default engine
and read back here by Python, so it needs no entry.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from xml.sax.saxutils import quoteattr

ADIOS2_IO_FORM = 14
DEFAULT_ENGINE = "BP4"   # WRF reads inputs like restarts: BP5 fails (see module docstring)
ENGINES = ("BP4", "BP5")
# Bump when the nc->BP layout logic changes: every cached conversion is then redone.
CONVERTER_VERSION = "wrf_nc2bp@8ba1629"
MANIFEST_NAME = "bp_manifest.json"
INPUT_NAME_RE = re.compile(r"^(wrfinput|wrfbdy|wrflowinp|wrffdda)_d\d\d$")
# Multi-record streams: converted one BP per record (see per_record_namelist_edits).
PER_RECORD_NAME_RE = re.compile(r"^(wrfbdy|wrflowinp|wrffdda)_d\d\d$")
PER_RECORD_LAYOUT = "per-record"
SINGLE_LAYOUT = "single"

_SRC = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SRC)
_NC2BP_PATH = os.path.join(_SRC, "tools", "wrf_nc2bp.py")
_BP2NC_PATH = os.path.join(_SRC, "tools", "wrf_adios2_to_netcdf4", "convert.py")
_ADIOS2_ENV = os.path.join(_REPO, "envs", "adios2")
_ADIOS2_ENV_PY = os.path.join(_ADIOS2_ENV, ".pixi", "envs", "default", "bin", "python")
_CHILD_ENV_DROP = ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")

_NC_MAGIC = (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n")


class Adios2Unavailable(RuntimeError):
    """No interpreter that can import the adios2 Python module was found."""


class Adios2ConversionError(RuntimeError):
    """A converter subprocess (adios2 interpreter) exited nonzero."""


_IMPORT_OK = {}     # interpreter path -> bool (`import adios2` result, cached per process)


def _child_env():
    return {k: v for k, v in os.environ.items() if k not in _CHILD_ENV_DROP}


def _imports_adios2(py):
    if py not in _IMPORT_OK:
        try:
            r = subprocess.run([py, "-c", "import adios2"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=_child_env(), timeout=300)
            _IMPORT_OK[py] = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _IMPORT_OK[py] = False
    return _IMPORT_OK[py]


def _adios2_candidates():
    explicit = os.environ.get("WTUNE_ADIOS2_PYTHON")
    if explicit:
        return [explicit]
    return [c for c in (_ADIOS2_ENV_PY, shutil.which("python"), shutil.which("python3")) if c]


def adios2_python():
    """Path of the interpreter the converters run under, or None (see module docstring)."""
    for py in _adios2_candidates():
        if _imports_adios2(py):
            return py
    return None


def require_adios2_python():
    """Return the adios2-capable interpreter path, or raise Adios2Unavailable naming the fix.
    Never imports adios2 into this process."""
    py = adios2_python()
    if py is not None:
        return py
    explicit = os.environ.get("WTUNE_ADIOS2_PYTHON")
    if explicit:
        why = "WTUNE_ADIOS2_PYTHON=%s cannot `import adios2`" % explicit
    else:
        why = "no Python that can `import adios2` was found (tried %s)" % ", ".join(
            [_ADIOS2_ENV_PY, "python/python3 on PATH"])
    raise Adios2Unavailable(
        "%s. The BP converters run under the separate envs/adios2 environment (adios2 2.10.* "
        "nompi, matching WRF's ADIOS2 2.10.2; it cannot live in envs/wtune because of an hdf5 "
        "conflict): run `pixi install` in %s on the cluster, or point WTUNE_ADIOS2_PYTHON at a "
        "python that imports adios2 2.10." % (why, _ADIOS2_ENV))


def _run_adios2(args, what):
    py = require_adios2_python()
    r = subprocess.run([py] + list(args), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=_child_env(), text=True, errors="replace")
    if r.returncode != 0:
        tail = "\n".join(((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-20:])
        raise Adios2ConversionError("%s failed (rc %d, %s):\n%s" % (what, r.returncode, py, tail))
    for line in (r.stderr or "").splitlines():
        if line.startswith("WARNING:"):
            print(line, file=sys.stderr)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def _has_data_subfile(entries):
    return any(re.fullmatch(r"data\.\d+", e) for e in entries)


def bp_engine(path):
    """'BP5' (md.0 + mmd.0 + data.N), 'BP4' (md.idx + data.N, no mmd.0) or None."""
    if not os.path.isdir(path):
        return None
    try:
        entries = set(os.listdir(path))
    except OSError:
        return None
    if not _has_data_subfile(entries):
        return None
    if "md.0" in entries and "mmd.0" in entries:
        return "BP5"
    if "md.idx" in entries:
        return "BP4"
    return None


def is_bp(path):
    """True for a BP4/BP5 directory as written by WRF io_form 14 or by nc2bp (symlinks
    followed). A regular file (netCDF, or a legacy BP3 single file) is False."""
    return bp_engine(path) is not None


def is_netcdf(path):
    """True for a regular file carrying a netCDF classic/64-bit/CDF5 or HDF5 signature."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return False
    return any(head.startswith(m) for m in _NC_MAGIC)


# ---------------------------------------------------------------------------
# single-file converters (atomic: write a sibling temp, then rename)
# ---------------------------------------------------------------------------
def _tmp_sibling(dst):
    d, b = os.path.split(os.path.abspath(dst))
    return os.path.join(d, ".%s.tmp-%d-%d" % (b, os.getpid(), time.monotonic_ns()))


def _rm(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.lexists(path):
        os.unlink(path)


def nc2bp(src_nc, dst_bp, engine=DEFAULT_ENGINE, split_prefix=None):
    """Convert one WRF input netCDF file to a WRF-readable BP directory. Raises
    FileExistsError if dst_bp exists; on any failure nothing is left at dst_bp.
    split_prefix (e.g. "wrfbdy_d01"): dst_bp becomes a plain directory of one-record BPs
    <split_prefix>_<date>.bp instead (record_files lists them)."""
    if engine not in ENGINES:
        raise ValueError("unsupported engine %r; use BP4 or BP5" % (engine,))
    if os.path.lexists(dst_bp):
        raise FileExistsError("output already exists: %s" % dst_bp)
    tmp = _tmp_sibling(dst_bp)
    try:
        args = [_NC2BP_PATH, os.fspath(src_nc), tmp, "--engine", engine]
        if split_prefix is not None:
            args += ["--split-prefix", split_prefix]
        _run_adios2(args, "nc2bp %s" % os.fspath(src_nc))
        os.rename(tmp, dst_bp)
    except BaseException:
        _rm(tmp)
        raise
    return dst_bp


def bp2nc(src_bp, dst_nc):
    """Convert one BP directory (WRF output or nc2bp output) to netCDF4 (all steps).
    Atomic: a failed conversion leaves no partial dst_nc (the 132 KB stub that once made
    the 1072441 gate report a conversion that had failed)."""
    if os.path.lexists(dst_nc):
        raise FileExistsError("output already exists: %s" % dst_nc)
    tmp = _tmp_sibling(dst_nc)
    try:
        # serial path (no mpirun); its \r progress bar goes to the captured child stdout
        _run_adios2([_BP2NC_PATH, "--input", os.fspath(src_bp), "--output", tmp],
                    "bp2nc %s" % os.fspath(src_bp))
        os.rename(tmp, dst_nc)
    except BaseException:
        _rm(tmp)
        raise
    return dst_nc


def is_per_record_dir(path):
    """True for a per-record container (nc2bp split_prefix output): a plain directory, not
    itself BP, holding <name>_<date>.bp BP directories."""
    return os.path.isdir(path) and not is_bp(path) and bool(record_files(path))


def record_files(container):
    """Sorted (= chronological: WRF date strings) per-record BP basenames in a container."""
    try:
        entries = os.listdir(container)
    except OSError:
        return []
    return sorted(e for e in entries
                  if e.endswith(".bp") and is_bp(os.path.join(container, e)))


def bp_input_links(name, target):
    """Run-dir symlinks {link name: target} for one converted input: <name>.bp -> target for
    a single BP; one <name>_<date>.bp -> target/<name>_<date>.bp per record for a per-record
    container (the exact names WRF opens with per_record_namelist_edits)."""
    if is_per_record_dir(target):
        return {r: os.path.join(target, r) for r in record_files(target)}
    return {name + ".bp": target}


def per_record_namelist_edits(names, max_dom=1):
    """Namelist edits that make WRF read per-record BP inputs, as
    {namelist group: {key: Fortran literal}} for the input names present (e.g. wrfbdy_d01,
    wrflowinp_d01). io_form_* are not included (unchanged: 14).

    wrfbdy  : &bdy_control multi_bdy_files = .true. and bdy_inname "wrfbdy_d<domain>_<date>.bp"
              (module_check_a_mundo.F:1324-1333 requires the "_<date>" substring;
              mediation_integrate.F:1451-1516 opens/reads/closes one file per boundary time,
              <date> = current_timestr via construct_filename2a). Run-verified, scc 2026-10-03.
    wrflowinp: auxinput4_inname "wrflowinp_d<domain>_<date>.bp" and frames_per_auxinput4 = 1
              per domain: open_aux_u (mediation_integrate.F:1613-1682) builds the dated name
              only when the stream is not open, and med_auxinput_in_closes.inc closes it only
              after frames_per_auxinput4 frames (Registry default 999999). Run-verified (B3).
    wrffdda : gfdda_inname (&fdda; copied to auxinput10_inname by module_check_a_mundo)
              "wrffdda_d<domain>_<date>.bp" and frames_per_auxinput10 = 1 -- same mechanism
              as auxinput4, NOT run-verified.
    Limitation: the files are named by the clock time of each read, so a run must start
    (cold or restart) on a record time; WRF opens <name>_<start>.bp first."""
    kinds = {PER_RECORD_NAME_RE.match(n).group(1) for n in names if PER_RECORD_NAME_RE.match(n)}
    ones = ", ".join(["1"] * max_dom)
    edits = {}
    if "wrfbdy" in kinds:
        edits.setdefault("bdy_control", {})["multi_bdy_files"] = ".true."
        edits.setdefault("time_control", {})["bdy_inname"] = '"wrfbdy_d<domain>_<date>.bp"'
    if "wrflowinp" in kinds:
        tc = edits.setdefault("time_control", {})
        tc["auxinput4_inname"] = '"wrflowinp_d<domain>_<date>.bp"'
        tc["frames_per_auxinput4"] = ones
    if "wrffdda" in kinds:
        edits.setdefault("fdda", {})["gfdda_inname"] = '"wrffdda_d<domain>_<date>.bp"'
        edits.setdefault("time_control", {})["frames_per_auxinput10"] = ones
    return edits


# ---------------------------------------------------------------------------
# cached input conversion
# ---------------------------------------------------------------------------
def sha256_file(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_input_names(case_dir):
    """Sorted wrfinput/wrfbdy/wrflowinp/wrffdda_d0N names in case_dir that are netCDF files
    (BP directories and anything else with those names are ignored)."""
    out = []
    for name in sorted(os.listdir(case_dir)):
        if INPUT_NAME_RE.match(name) and is_netcdf(os.path.join(case_dir, name)):
            out.append(name)
    return out


def _read_manifest(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"version": 1, "entries": {}}
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        raise ValueError("%s is not a bp manifest" % path)
    return data


def _write_manifest(path, data):
    tmp = _tmp_sibling(path)
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _entry_current(entry, src, st, engine, dst, layout=SINGLE_LAYOUT):
    """(current, sha) -- current means the cached BP still corresponds to src. Size+mtime
    unchanged skips hashing; otherwise the source is re-hashed and only a content change
    counts (a touch/copy with identical bytes does not force a reconversion). An entry
    without "layout" is a single-file conversion (manifests before per-record)."""
    if not entry or entry.get("layout", SINGLE_LAYOUT) != layout:
        return False, None
    if layout == PER_RECORD_LAYOUT:
        recs = entry.get("records") or []
        if not recs or record_files(dst) != sorted(recs) or is_bp(dst):
            return False, None
    elif not is_bp(dst):
        return False, None
    if (entry.get("engine") != engine or entry.get("converter_version") != CONVERTER_VERSION
            or entry.get("source") != src):
        return False, None
    if entry.get("size") == st.st_size and entry.get("mtime") == st.st_mtime:
        return True, entry.get("sha256")
    if entry.get("size") != st.st_size:
        return False, None
    sha = sha256_file(src)
    return sha == entry.get("sha256"), sha


def convert_inputs(case_dir, cache_dir, names=None, engine=DEFAULT_ENGINE):
    """Convert a case's WRF input netCDF files to BP under cache_dir, cached.

    names None -> every wrfinput/wrfbdy/wrflowinp/wrffdda_d0N netCDF file in case_dir.
    Output cache_dir/<name> (a BP directory named exactly like the WRF input, so a run dir can
    symlink wrfinput_d01 -> cache_dir/wrfinput_d01). cache_dir/bp_manifest.json records per name
    the resolved source path, size, mtime, sha256, engine, converter version and BP path; a name
    is reconverted only when one of those no longer matches (or the BP is missing).
    Multi-record names (PER_RECORD_NAME_RE: wrfbdy/wrflowinp/wrffdda) are written per record:
    cache_dir/<name> is then a plain directory of <name>_<date>.bp (manifest "layout":
    "per-record", "records": [basenames]; bp_input_links gives the run-dir links,
    per_record_namelist_edits the namelist). Other names: "layout": "single". Conversions
    are atomic and serialized by a POSIX lock (lockf: GPFS/NFS-safe across nodes) on
    cache_dir/.bp_manifest.lock.
    Returns {name: bp_path}."""
    if engine not in ENGINES:
        raise ValueError("unsupported engine %r; use BP4 or BP5" % (engine,))
    if names is None:
        names = find_input_names(case_dir)
    else:
        names = list(names)
        for n in names:
            p = os.path.join(case_dir, n)
            if not os.path.isfile(p):
                raise FileNotFoundError("no input file %s" % p)
            if not is_netcdf(p):
                raise ValueError("%s is not a netCDF file" % p)
    os.makedirs(cache_dir, exist_ok=True)
    mpath = os.path.join(cache_dir, MANIFEST_NAME)
    result = {}
    with open(os.path.join(cache_dir, ".bp_manifest.lock"), "w") as lockf:
        fcntl.lockf(lockf, fcntl.LOCK_EX)
        manifest = _read_manifest(mpath)
        entries = manifest["entries"]
        for name in names:
            src = os.path.realpath(os.path.join(case_dir, name))
            dst = os.path.join(cache_dir, name)
            st = os.stat(src)
            layout = PER_RECORD_LAYOUT if PER_RECORD_NAME_RE.match(name) else SINGLE_LAYOUT
            current, sha = _entry_current(entries.get(name), src, st, engine, dst, layout)
            if not current:
                sha = sha or sha256_file(src)
                tmp = _tmp_sibling(dst)
                try:
                    if layout == PER_RECORD_LAYOUT:
                        nc2bp(src, tmp, engine=engine, split_prefix=name)
                    else:
                        nc2bp(src, tmp, engine=engine)
                    if os.path.lexists(dst):
                        old = _tmp_sibling(dst) + ".old"
                        os.rename(dst, old)
                        os.rename(tmp, dst)
                        _rm(old)
                    else:
                        os.rename(tmp, dst)
                except BaseException:
                    _rm(tmp)
                    raise
            entries[name] = {
                "source": src, "size": st.st_size, "mtime": st.st_mtime, "sha256": sha,
                "engine": engine, "converter_version": CONVERTER_VERSION,
                "bp_path": os.path.abspath(dst), "layout": layout,
                "converted_at": (entries.get(name, {}).get("converted_at") if current
                                 else time.strftime("%Y-%m-%dT%H:%M:%S")),
            }
            if layout == PER_RECORD_LAYOUT:
                entries[name]["records"] = record_files(dst)
            # persist after every file so an interrupted batch keeps what it finished
            _write_manifest(mpath, manifest)
            result[name] = dst
    return result


# ---------------------------------------------------------------------------
# round-trip verification
# ---------------------------------------------------------------------------
def _attrs(obj):
    return {a: obj.getncattr(a) for a in obj.ncattrs() if a != "_FillValue"}


def _attr_equal(a, b):
    import numpy as np
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        return a == b
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def _compare_arrays(a, b):
    """(equal, max_abs_diff or None). Character data byte-for-byte; floats NaN-equal."""
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "SUO":
        return a.tobytes() == b.tobytes(), None
    if a.tobytes() == b.tobytes():
        return True, 0.0
    if a.dtype.kind == "f":
        eq = np.array_equal(a, b, equal_nan=True)
        both = ~(np.isnan(a) | np.isnan(b))
    else:
        eq = np.array_equal(a, b)
        both = np.ones(a.shape, bool)
    diff = np.abs(a[both].astype(np.float64) - b[both].astype(np.float64))
    return eq, (float(diff.max()) if diff.size else (0.0 if eq else float("nan")))


def compare_netcdf(left_nc, right_nc, left_record=None):
    """Compare every variable (data, dtype, shape, dimensions, attributes) and every global
    attribute and dimension length. Returns dict(ok, vars_compared, max_abs_diff, mismatches).
    left_record k: compare only record k of left's Time-dimensioned variables (Time length 1
    expected on the right) -- one file of a per-record conversion."""
    from netCDF4 import Dataset
    mismatches = []
    vars_compared = 0
    max_abs = 0.0
    with Dataset(left_nc) as L, Dataset(right_nc) as R:
        for ds in (L, R):
            ds.set_auto_mask(False)
            ds.set_auto_scale(False)
        def lshape(v):
            if left_record is not None and v.dimensions[:1] == ("Time",):
                return (1,) + tuple(v.shape[1:])
            return v.shape

        for d, dim in L.dimensions.items():
            n = 1 if left_record is not None and d == "Time" else len(dim)
            if d not in R.dimensions:
                mismatches.append("dim %s missing" % d)
            elif len(R.dimensions[d]) != n:
                mismatches.append("dim %s: %d != %d" % (d, n, len(R.dimensions[d])))
        for name in sorted(set(R.variables) - set(L.variables)):
            mismatches.append("%s: extra variable in round trip" % name)
        for name, lv in L.variables.items():
            if name not in R.variables:
                mismatches.append("%s: missing in round trip" % name)
                continue
            rv = R.variables[name]
            vars_compared += 1
            if lv.dtype != rv.dtype:
                mismatches.append("%s: dtype %s != %s" % (name, lv.dtype, rv.dtype))
                continue
            if lshape(lv) != rv.shape or lv.dimensions != rv.dimensions:
                mismatches.append("%s: shape %s%s != %s%s"
                                  % (name, lv.dimensions, lshape(lv), rv.dimensions, rv.shape))
                continue
            # one leading record at a time: bounded memory for multi-GB fields
            if left_record is not None and lv.dimensions[:1] == ("Time",):
                pairs = [((left_record,), (0,))]
            elif lv.ndim >= 2 and lv.shape[0] > 1:
                pairs = [((i,), (i,)) for i in range(lv.shape[0])]
            else:
                pairs = [(Ellipsis, Ellipsis)]
            for ls, rs in pairs:
                eq, diff = _compare_arrays(lv[ls], rv[rs])
                if diff is not None and diff == diff:
                    max_abs = max(max_abs, diff)
                if not eq:
                    mismatches.append("%s: data differs (max_abs_diff %s)" % (name, diff))
                    break
            la, ra = _attrs(lv), _attrs(rv)
            for a in la:
                if a not in ra:
                    mismatches.append("%s:%s attribute missing" % (name, a))
                elif not _attr_equal(la[a], ra[a]):
                    mismatches.append("%s:%s %r != %r" % (name, a, la[a], ra[a]))
        lg, rg = _attrs(L), _attrs(R)
        for a in lg:
            if a not in rg:
                mismatches.append("global:%s missing" % a)
            elif not _attr_equal(lg[a], rg[a]):
                mismatches.append("global:%s %r != %r" % (a, lg[a], rg[a]))
    return {"ok": not mismatches, "vars_compared": vars_compared,
            "max_abs_diff": max_abs, "mismatches": mismatches}


def roundtrip_check(src_nc, bp_path, tmp_dir):
    """nc -> BP -> nc and compare with the source (see compare_netcdf). bp_path is converted
    from src_nc first when it does not exist yet; the netCDF copy is written in tmp_dir and
    removed afterwards. Returns dict(ok, vars_compared, max_abs_diff, mismatches)."""
    if not os.path.lexists(bp_path):
        nc2bp(src_nc, bp_path)
    os.makedirs(tmp_dir, exist_ok=True)
    if is_per_record_dir(bp_path):
        return _roundtrip_per_record(src_nc, bp_path, tmp_dir)
    back = os.path.join(tmp_dir, os.path.basename(os.path.normpath(bp_path)) + ".roundtrip.nc")
    _rm(back)
    bp2nc(bp_path, back)
    try:
        return compare_netcdf(src_nc, back)
    finally:
        _rm(back)


def _roundtrip_per_record(src_nc, container, tmp_dir):
    """Record file i (chronological) must hold exactly source record i, and there must be
    one file per source record."""
    from netCDF4 import Dataset
    with Dataset(src_nc) as ds:
        n = len(ds.dimensions["Time"]) if "Time" in ds.dimensions else 1
    recs = record_files(container)
    out = {"ok": True, "vars_compared": 0, "max_abs_diff": 0.0, "mismatches": []}
    if len(recs) != n:
        out["ok"] = False
        out["mismatches"].append("%d record files for %d source records" % (len(recs), n))
        return out
    for k, rec in enumerate(recs):
        back = os.path.join(tmp_dir, rec + ".roundtrip.nc")
        _rm(back)
        bp2nc(os.path.join(container, rec), back)
        try:
            r = compare_netcdf(src_nc, back, left_record=k)
        finally:
            _rm(back)
        out["vars_compared"] += r["vars_compared"]
        out["max_abs_diff"] = max(out["max_abs_diff"], r["max_abs_diff"])
        out["mismatches"] += ["%s: %s" % (rec, m) for m in r["mismatches"]]
    out["ok"] = not out["mismatches"]
    return out


# ---------------------------------------------------------------------------
# adios2.xml
# ---------------------------------------------------------------------------
def default_engine_for(io_name):
    """BP4 for restart files (the BP5 readback failure, see module docstring); None (= no
    entry, ADIOS2 default BP5) for everything else."""
    return "BP4" if os.path.basename(io_name).startswith("wrfrst_d") else None


def restart_io_names(restart_times, max_dom):
    """ADIOS2 IO names WRF uses for the restart files: wrfrst_d<NN>_<YYYY-MM-DD_HH:MM:SS>."""
    return ["wrfrst_d%02d_%s" % (d, t) for t in restart_times for d in range(1, max_dom + 1)]


def adios2_xml(engine_by_stream=None):
    """XML text for <run_dir>/adios2.xml.

    engine_by_stream: {io_name: engine} or an iterable of io_names. io_name is the exact file
    name WRF opens (WRF names each ADIOS2 IO after the file, timestamp included; ADIOS2 matches
    names exactly -- see restart_io_names). An engine of None (or a bare name) takes
    default_engine_for(name): BP4 for wrfrst_* (restart readback needs BP4 on both the
    writing and the reading run), no entry otherwise. None -> a config with no IO entries
    (every file on the ADIOS2 default, BP5)."""
    if engine_by_stream is None:
        items = []
    elif isinstance(engine_by_stream, dict):
        items = list(engine_by_stream.items())
    else:
        items = [(n, None) for n in engine_by_stream]
    body = ""
    for name, engine in items:
        engine = engine or default_engine_for(name)
        if engine is None:
            continue
        if engine not in ENGINES:
            raise ValueError("unsupported engine %r for %s; use BP4 or BP5" % (engine, name))
        body += "  <io name=%s>\n    <engine type=\"%s\"/>\n  </io>\n" % (quoteattr(name), engine)
    return ('<?xml version="1.0"?>\n'
            "<!-- wtune: WRF names each ADIOS2 IO after its file; restart files use BP4 because\n"
            "     WRF's restart reader queries Times before BeginStep, which BP5 refuses. -->\n"
            "<adios-config>\n" + body + "</adios-config>\n")


# ---------------------------------------------------------------------------
# CLI: wtune convert nc2bp | bp2nc | inputs
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="wtune convert",
                                description="WRF netCDF <-> ADIOS2 BP (io_form %d) conversion"
                                            % ADIOS2_IO_FORM)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("nc2bp", help="netCDF input file(s) -> BP directories")
    a.add_argument("src", nargs="+")
    a.add_argument("--out", default=None,
                   help="output directory: <out>/<name>; default <src>.bp next to the source")
    a.add_argument("--engine", choices=ENGINES, default=DEFAULT_ENGINE)
    a.add_argument("--check", action="store_true",
                   help="round-trip each output back to netCDF and compare; exit 1 on mismatch")
    b = sub.add_parser("bp2nc", help="BP directories -> netCDF4 files")
    b.add_argument("src", nargs="+")
    b.add_argument("--out", default=None,
                   help="output directory (default: next to the source); name <src minus .bp>.nc")
    c = sub.add_parser("inputs", help="cached conversion of a case's wrfinput/wrfbdy/wrflowinp/"
                                      "wrffdda_d0N to BP (manifest %s)" % MANIFEST_NAME)
    c.add_argument("case_dir")
    c.add_argument("--cache", required=True, help="cache directory (outputs <cache>/<name>)")
    c.add_argument("--names", nargs="+", default=None, help="only these input names")
    c.add_argument("--engine", choices=ENGINES, default=DEFAULT_ENGINE)
    c.add_argument("--check", action="store_true",
                   help="round-trip every cached BP and compare; exit 1 on mismatch")
    return p


def nc2bp_target(src, out=None):
    base = os.path.basename(os.path.normpath(src))
    if out is None or os.path.realpath(out) == os.path.realpath(os.path.dirname(os.path.abspath(src))):
        return os.path.join(os.path.dirname(os.path.abspath(src)), base + ".bp")
    return os.path.join(out, base)


def bp2nc_target(src, out=None):
    base = os.path.basename(os.path.normpath(src))
    if base.endswith(".bp"):
        base = base[:-3]
    return os.path.join(out if out is not None else os.path.dirname(os.path.abspath(src)),
                        base + ".nc")


def _check(src, bp, scratch_parent):
    with tempfile.TemporaryDirectory(prefix=".wtune-roundtrip-", dir=scratch_parent) as td:
        r = roundtrip_check(src, bp, td)
    status = "PASS" if r["ok"] else "FAIL"
    print("CHECK %s %s: %d variables, max_abs_diff %g" % (status, bp, r["vars_compared"],
                                                          r["max_abs_diff"]))
    for m in r["mismatches"][:50]:
        print("  MISMATCH " + m)
    return r["ok"]


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        require_adios2_python()
    except Adios2Unavailable as exc:
        print("wtune convert: %s" % exc, file=sys.stderr)
        return 2
    ok = True
    if args.action == "nc2bp":
        if args.out:
            os.makedirs(args.out, exist_ok=True)
        for src in args.src:
            dst = nc2bp_target(src, args.out)
            t0 = time.time()
            nc2bp(src, dst, engine=args.engine)
            print("%s -> %s (%s, %.1f s)" % (src, dst, args.engine, time.time() - t0))
            if args.check:
                ok = _check(src, dst, os.path.dirname(os.path.abspath(dst))) and ok
    elif args.action == "bp2nc":
        if args.out:
            os.makedirs(args.out, exist_ok=True)
        for src in args.src:
            if not is_bp(src):
                print("wtune convert: %s is not a BP4/BP5 directory" % src, file=sys.stderr)
                return 2
            dst = bp2nc_target(src, args.out)
            t0 = time.time()
            bp2nc(src, dst)
            print("%s -> %s (%.1f s)" % (src, dst, time.time() - t0))
    else:
        mapping = convert_inputs(args.case_dir, args.cache, names=args.names, engine=args.engine)
        if not mapping:
            print("wtune convert: no wrfinput/wrfbdy/wrflowinp/wrffdda_d0N netCDF files in %s"
                  % args.case_dir, file=sys.stderr)
            return 2
        for name, bp in mapping.items():
            print("%s -> %s" % (name, bp))
            if args.check:
                ok = _check(os.path.join(args.case_dir, name), bp, args.cache) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
