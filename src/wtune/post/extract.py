"""post/extract.py -- `wtune post extract RUN_DIR --vars ... --out FILE`: wrf-python diagnostics as a
point time-series CSV (--points) or a gridded NetCDF box (--box); frames extracted in a process pool
(netCDF-C is not thread-safe). Supported --vars: SURFACE_FIELDS and LEVEL_FIELDS below. wrf-python
and netCDF4 are imported lazily, so `wtune post inventory`/`compare` and `wtune --help` never need
them. The --box crop stays self-written: it crops a wrf-python-derived in-memory field (decision 0056).

ADIOS2 BP history (io_form_history=14): each wrfout_dNN_<time>[.bp] BP directory is converted once
to a netCDF cache (see nc_cache) and indexed in its place, so every caller keeps using netCDF paths.
"""
from __future__ import annotations

import contextlib
import csv
import glob
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"

# rain_acc = RAINC + RAINNC; rain_rate = finite difference of rain_acc between consecutive extracted
# frames per hour (first frame empty); mcape/mcin = 2-D max CAPE / its CIN (cape_2d); pw = wrf 'pw'.
# LEVEL_FIELDS are interpolated to each --levels hPa with wrf.interplevel.
SURFACE_FIELDS = ("slp", "t2", "td2", "rh2", "wspd10", "wdir10", "rain_acc", "rain_rate",
                  "mdbz", "mcape", "mcin", "pw")
LEVEL_FIELDS = ("z", "tc", "ua", "va", "wspd")
UNITS = dict(slp="hPa", t2="K", td2="degC", rh2="%", wspd10="m/s", wdir10="deg",
             rain_acc="mm", rain_rate="mm/h", mdbz="dBZ", mcape="J/kg", mcin="J/kg",
             pw="kg/m2", z="m", tc="degC", ua="m/s", va="m/s", wspd="m/s")


class ExtractError(RuntimeError):
    pass


def _wrf():
    import wrf
    return wrf


def _netcdf4():
    import netCDF4
    return netCDF4


_SEGMENT_RE = re.compile(r"seg_(\d+)(_rescue)?(?:_r(\d+))?$")    # as wtune_watch._SEG_DIR_RE


def _is_empty_dir(path):
    """A pre-created output dir WRF has not written (`wtune run` removes them after the job)."""
    try:
        return os.path.isdir(path) and not os.listdir(path)
    except OSError:
        return False


def _wrfout_paths(d):
    """wrfout_d* entries of d, minus empty (pre-created, unwritten) directories."""
    return [p for p in glob.glob(os.path.join(d, "wrfout_d*")) if not _is_empty_dir(p)]


def segment_dirs(run_dir: str) -> list[str]:
    """The segment run dirs of a segmented `wtune run --out` dir (OUT/seg_<k>, plus its later
    attempts right after it: a CFL-rescue re-run OUT/seg_<k>_rescue, an auto-resume
    OUT/seg_<k>[_rescue]_r<n>), in segment then attempt order -- [] when run_dir holds wrfout_d*
    itself (a flat run dir) or has no seg_<int> subdirs."""
    if _wrfout_paths(run_dir):
        return []
    try:
        names = os.listdir(run_dir)
    except OSError:
        return []
    segs = []
    for name in names:
        m = _SEGMENT_RE.match(name)
        path = os.path.join(run_dir, name)
        if m and os.path.isdir(path):
            segs.append(((int(m.group(1)), bool(m.group(2)), int(m.group(3) or 0)), path))
    return [path for _key, path in sorted(segs)]


_TIMES_CACHE = {}       # {(path, mtime_ns, size): [valid_time_str, ...] or None}


def _decode_times(var):
    """WRF's Times(Time, DateStrLen) char variable -> [time_str, ...]; also accepts a 1-D
    string variable (a converted file, or netCDF4 auto-chartostring via _Encoding)."""
    import numpy as np
    raw = np.ma.getdata(var[:])
    if raw.ndim == 2 and raw.dtype.kind == "S":
        rows = [b"".join(r.tolist()) for r in raw]
    else:
        rows = list(np.atleast_1d(raw).tolist())
    return [(r.decode("ascii", "replace") if isinstance(r, bytes) else str(r)).strip("\x00 ")
            for r in rows]


def file_times(path: str) -> list[str] | None:
    """The valid times a wrfout file holds, from its own Times variable (WRF writes
    frames_per_outfile > 1 history times into one file) -- [time_str, ...] in file order, the
    list index being the file's own timeidx. None when the file is not readable netCDF or has
    no usable Times (the caller then falls back to the one file-name time). Reads only the
    Times variable; cached in-process per (path, mtime, size), so re-listing a run dir after a
    segment rewrote a file re-reads it, and an unchanged 400-file climate run is opened once."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    if key in _TIMES_CACHE:
        return _TIMES_CACHE[key]
    times = None
    if st.st_size > 0:
        try:
            netcdf4 = _netcdf4()
            with netcdf4.Dataset(path) as ds:
                if "Times" in ds.variables:
                    times = [t for t in _decode_times(ds.variables["Times"]) if t] or None
        except Exception:       # not netCDF / truncated / netCDF4 missing -> file-name time
            times = None
    _TIMES_CACHE[key] = times
    return times


_HIST_RE = re.compile(r"wrfout_d(\d+)_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)(\.bp)?$")
NC_CACHE_ENV = "WTUNE_NC_CACHE"
_NC_CACHE = {"dir": None, "base": None}
_BP_MEMO = {}       # {(bp, nc): (fast_key, signature, error_or_None)}
_BP_ORIGIN = {}     # {cached nc abs path: BP dir abs path}


class BPConversionError(ExtractError):
    """A run's ADIOS2 BP history cannot be turned into netCDF at all (the adios2 Python
    module or wtune_bp is unavailable); str() names the fix."""


# Cache layout: <root>/<run_id>/[<seg_k>/]<wrfout name without .bp>.nc; <root> is --nc-cache DIR, else
# $WTUNE_NC_CACHE, else <base>/_nc_cache (project or post output dir), else <run_dir>/_nc_cache.
# <run_id> = run dir basename + 8 hex of its realpath sha1 (every project's run is called "case").
# A sidecar <file>.nc.json holds the BP path and (name, size, mtime_ns) of its files: the cache is
# reused while that matches. A netCDF file beats a BP directory for the same time; adios2 is only
# needed to (re)convert (BPConversionError names the fix).
@contextlib.contextmanager
def nc_cache(directory: str | None = None, base: str | None = None) -> Iterator[None]:
    """Scope of the BP -> netCDF cache location. directory (`--nc-cache DIR`): the cache root
    itself. base (project dir, else the post output dir): the root becomes <base>/_nc_cache --
    applied only when no base is set yet, so the CLI's choice wins over run_recipe's default."""
    old = dict(_NC_CACHE)
    if directory:
        _NC_CACHE["dir"] = os.path.abspath(directory)
    if base and not _NC_CACHE["base"]:
        _NC_CACHE["base"] = os.path.abspath(base)
    try:
        yield
    finally:
        _NC_CACHE.clear()
        _NC_CACHE.update(old)


def nc_cache_root(run_dir: str) -> str:
    explicit = _NC_CACHE["dir"] or os.environ.get(NC_CACHE_ENV)
    if explicit:
        return os.path.abspath(explicit)
    return os.path.join(os.path.abspath(_NC_CACHE["base"] or run_dir), "_nc_cache")


def _run_id(run_dir):
    real = os.path.realpath(run_dir)
    return "%s-%s" % (os.path.basename(real) or "run",
                      hashlib.sha1(real.encode("utf-8", "surrogateescape")).hexdigest()[:8])


def cached_nc_path(run_dir: str, bp_path: str) -> str:
    """<root>/<run_id>/[<segment>/]<BP name without .bp>.nc for BP directory bp_path of
    run_dir (bp_path directly in run_dir, or in one of its segment_dirs)."""
    seg_dir = os.path.dirname(os.path.abspath(bp_path))
    parts = [nc_cache_root(run_dir), _run_id(run_dir)]
    if os.path.realpath(seg_dir) != os.path.realpath(run_dir):
        parts.append(os.path.basename(seg_dir))
    name = os.path.basename(os.path.normpath(bp_path))
    if name.endswith(".bp"):
        name = name[:-3]
    return os.path.join(*(parts + [name + ".nc"]))


def bp_signature(bp_path: str) -> list:
    """Sorted [[name, size, mtime_ns], ...] of the regular files in a BP directory (data.*,
    md.*, mmd.*, ...): changes whenever WRF (re)writes or appends to it."""
    sig = []
    with os.scandir(bp_path) as it:
        for e in it:
            if e.is_file():
                st = e.stat()
                sig.append([e.name, st.st_size, st.st_mtime_ns])
    return sorted(sig)


def _fast_key(bp_path):
    """A few stats that change whenever a BP4/BP5 directory is written (the dir itself when a
    file is created, md.0/md.idx on every step) -- lets a re-listing in the same process skip
    the full signature. None without md.0 (unknown layout: always take the full signature)."""
    key = []
    for name in ("", "md.0", "md.idx"):
        try:
            st = os.stat(os.path.join(bp_path, name) if name else bp_path)
        except OSError:
            if name == "md.0":
                return None
            continue
        key += [name, st.st_size if name else 0, st.st_mtime_ns]
    return tuple(key)


def _sidecar(nc):
    return nc + ".json"


def _cache_valid(nc, bp, sig):
    try:
        with open(_sidecar(nc)) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return False
    return (meta.get("source") == os.path.abspath(bp) and _same_files(meta.get("signature"), sig)
            and os.path.isfile(nc))


MTIME_SLACK_NS = 120 * 10**9   # NFS clients disagree on a file's mtime by seconds (scc: 18 s)


def _same_files(a, b):
    """Same names and sizes, mtimes within MTIME_SLACK_NS (a rewrite is much later than that)."""
    return (isinstance(a, list) and len(a) == len(b)
            and all(x[:2] == y[:2] and abs(x[2] - y[2]) <= MTIME_SLACK_NS for x, y in zip(a, b)))


def _tmp_tag():
    """host + pid: unique among processes sharing the cache over a cluster filesystem."""
    return "%s-%d" % (os.uname().nodename, os.getpid())


def _write_json_atomic(path, doc):
    tmp = "%s.wtune-tmp-%s" % (path, _tmp_tag())
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def _convert_job(job):
    """Worker (module-level for picklability): job = (bp2nc, bp, nc, signature). Converts into
    a temporary name next to nc, os.replace()s it, then writes the sidecar manifest."""
    convert, bp, nc, sig = job
    os.makedirs(os.path.dirname(nc), exist_ok=True)
    if os.path.lexists(_sidecar(nc)):
        os.remove(_sidecar(nc))
    tmp = "%s.wtune-tmp-%s.nc" % (nc[:-3], _tmp_tag())
    if os.path.lexists(tmp):                # a crashed earlier conversion (bp2nc refuses dst)
        os.remove(tmp)
    try:
        convert(bp, tmp)
        os.replace(tmp, nc)
    except BaseException:
        if os.path.lexists(tmp):
            os.remove(tmp)
        raise
    _write_json_atomic(_sidecar(nc), dict(source=os.path.abspath(bp), signature=sig,
                                          converter="wtune_bp.bp2nc"))
    return nc


def _cpu_budget():
    try:
        return max(1, len(os.sched_getaffinity(0)))     # a Slurm cpuset, not the whole node
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _bp2nc_or_raise(run_dir, n_files, example):
    """wtune_bp.bp2nc, after wtune.bp.require_adios2_python() -- whose Adios2Unavailable text
    names the fix (the adios2 env); BPConversionError carrying that text otherwise."""
    what = ("%s holds ADIOS2 BP history (%d file(s) not yet converted, e.g. %s) that the post "
            "tools read as netCDF copies, but the BP -> netCDF converter "
            "(src/tools/wrf_adios2_to_netcdf4/ via src/wtune_bp.py) cannot run here: "
            % (run_dir, n_files, os.path.basename(example)))
    try:
        import wtune.bp
    except Exception as e:                                  # noqa: BLE001
        raise BPConversionError(what + "cannot import wtune_bp (%s: %s)"
                                % (type(e).__name__, e)) from e
    try:
        wtune.bp.require_adios2_python()
    except (Exception, SystemExit) as e:                    # noqa: BLE001 -- Adios2Unavailable
        raise BPConversionError(what + str(e)) from e
    return wtune.bp.bp2nc


def _convert_all(run_dir, jobs):
    """jobs: [(bp, nc, signature), ...] -> {bp: None | error text}. Raises BPConversionError
    when the converter itself is unavailable."""
    convert = _bp2nc_or_raise(run_dir, len(jobs), jobs[0][0])
    work = [(convert, bp, nc, sig) for bp, nc, sig in jobs]
    n = min(len(work), _cpu_budget())
    sys.stderr.write("post: converting %d ADIOS2 BP history file(s) of %s to netCDF under %s "
                     "(%d worker(s))\n" % (len(work), run_dir,
                                           os.path.join(nc_cache_root(run_dir), _run_id(run_dir)),
                                           n))
    errors = {}
    if n <= 1:
        for job in work:
            try:
                _convert_job(job)
                errors[job[1]] = None
            except Exception as e:                          # noqa: BLE001 -- per file
                errors[job[1]] = "%s: %s" % (type(e).__name__, e)
        return errors
    with ProcessPoolExecutor(max_workers=n) as ex:
        futures = [(job[1], ex.submit(_convert_job, job)) for job in work]
        for bp, fut in futures:
            try:
                fut.result()
                errors[bp] = None
            except Exception as e:                          # noqa: BLE001 -- per file
                errors[bp] = "%s: %s" % (type(e).__name__, e)
    return errors


def _bp_to_nc(run_dir, bps):
    """({bp: cached nc}, {bp: error}) for BP directories bps of run_dir, converting what is
    missing or stale. A BP whose conversion failed is left out of the index and not retried
    until the BP directory changes."""
    mapped, failed, jobs, fast_of = {}, {}, [], {}
    for bp in bps:
        nc = cached_nc_path(run_dir, bp)
        fast = fast_of[bp] = _fast_key(bp)
        memo = _BP_MEMO.get((bp, nc))
        if memo is not None and fast is not None and memo[0] == fast and (
                memo[2] is not None or os.path.isfile(nc)):
            sig = memo[1]                   # unchanged since this process last looked
        else:
            try:
                sig = bp_signature(bp)
            except OSError as e:
                failed[bp] = "%s: %s" % (type(e).__name__, e)
                continue
            if memo is not None and memo[1] != sig:
                memo = None
        if memo is not None and memo[2] is not None:
            failed[bp] = memo[2]
        elif (memo is not None and os.path.isfile(nc)) or _cache_valid(nc, bp, sig):
            mapped[bp] = nc
            _BP_MEMO[(bp, nc)] = (fast, sig, None)
        else:
            jobs.append((bp, nc, sig))
    if jobs:
        errors = _convert_all(run_dir, jobs)
        for bp, nc, sig in jobs:
            err = errors.get(bp)
            _BP_MEMO[(bp, nc)] = (fast_of[bp], sig, err)    # the PRE-conversion state
            if err is None:
                mapped[bp] = nc
            else:
                failed[bp] = err
                sys.stderr.write("post: warning: %s not indexed, BP -> netCDF conversion "
                                 "failed: %s\n" % (bp, err))
    for bp, nc in mapped.items():
        _BP_ORIGIN[os.path.abspath(nc)] = os.path.abspath(bp)
    return mapped, failed


def bp_origin(path: str) -> str | None:
    """The ADIOS2 BP directory a cached netCDF history file was converted from, else None."""
    ap = os.path.abspath(path)
    if ap in _BP_ORIGIN:
        return _BP_ORIGIN[ap]
    try:
        with open(_sidecar(ap)) as fh:
            return json.load(fh).get("source")
    except (OSError, ValueError, AttributeError):
        return None


def _dir_entries(d):
    """[(domain, file_name_time, path, is_bp), ...] of one directory in name order; a BP
    history is a directory (wrfout_dNN_TIME or wrfout_dNN_TIME.bp). For a (domain, time) with
    both a netCDF file and a BP directory, only the netCDF file is kept."""
    try:
        with os.scandir(d) as it:
            entries = sorted((e.name, e) for e in it)
    except OSError:
        return []
    found = {}
    for name, entry in entries:
        m = _HIST_RE.match(name)
        if not m:
            continue
        path = os.path.join(d, name)
        if _is_empty_dir(path):
            continue
        try:
            is_bp = entry.is_dir()          # d_type, no extra stat; follows a symlinked BP dir
        except OSError:
            continue
        key = (int(m.group(1)), m.group(2))
        if key not in found or (found[key][3] and not is_bp):
            found[key] = (key[0], key[1], path, is_bp)
    return sorted(found.values(), key=lambda e: os.path.basename(e[2]))


def _raw_entries(run_dir):
    out = []
    for d in segment_dirs(run_dir) or [run_dir]:
        out += _dir_entries(d)
    return out


def _file_entries(run_dir):
    """[(domain, file_name_time, path), ...] for every wrfout_dNN_TIME history file, segment
    order (segment_dirs) then file-name order -- the order "later wins" is decided in. A BP
    history directory appears as its cached netCDF copy (_bp_to_nc), or not at all when its
    conversion failed."""
    raw = _raw_entries(run_dir)
    bps = [path for _d, _t, path, is_bp in raw if is_bp]
    mapped = _bp_to_nc(run_dir, bps)[0] if bps else {}
    out = []
    for dom, t, path, is_bp in raw:
        if is_bp:
            path = mapped.get(path)
            if path is None:
                continue
        out.append((dom, t, path))
    return out


def convert_frames(run_dir: str, bps) -> dict:
    """Converts (or validates the cache of) the BP history directories `bps` of run_dir ->
    {bp: error} for those that failed. Raises BPConversionError like prepare_history."""
    return _bp_to_nc(run_dir, list(bps))[1] if bps else {}


def prepare_history(run_dir: str) -> dict:
    """Converts (or validates the cache of) every BP history directory of run_dir up front.
    Returns dict(n_bp, cache, failed=[(bp, error), ...]). Raises BPConversionError when BP
    history needs converting and the converter is unavailable."""
    bps = [path for _d, _t, path, is_bp in _raw_entries(run_dir) if is_bp]
    failed = convert_frames(run_dir, bps)
    return dict(n_bp=len(bps), cache=os.path.join(nc_cache_root(run_dir), _run_id(run_dir)),
                failed=sorted(failed.items()))


def file_index(run_dir: str) -> dict:
    """{(domain, file_name_time): path} -- one entry per wrfout FILE, keyed by the time in its
    name (a later segment's file of the same name wins). For whole-file tools (diffwrf in
    post/compare.py) that compare files rather than individual valid times."""
    return {(dom, t): path for dom, t, path in _file_entries(run_dir)}


def frame_refs(run_dir: str) -> dict:
    """{(domain, valid_time): (path, timeidx)} for every history time in run_dir (all segments of a
    segmented run dir), valid times from each file's own Times variable (file_times); an unreadable
    file contributes its file-name time at timeidx 0. A time in two files maps to the later one."""
    out = {}
    for dom, name_time, path in _file_entries(run_dir):
        times = file_times(path) or [name_time]
        for idx, t in enumerate(times):
            out[(dom, t)] = (path, idx)
    return out


def frame_index(run_dir: str) -> dict:
    """{(domain, valid_time): path} -- frame_refs without the timeidx (the containing file of
    each valid time). Identical to the old file-name index for single-frame files."""
    return {k: path for k, (path, _idx) in frame_refs(run_dir).items()}


def list_frame_refs(run_dir: str, domain: int = 1) -> list[tuple]:
    """Sorted [(valid_time, path, timeidx), ...] for domain <domain> (see frame_refs)."""
    return sorted((t, path, idx) for (d, t), (path, idx) in frame_refs(run_dir).items()
                  if d == domain)


def list_frames(run_dir: str, domain: int = 1) -> list[tuple]:
    """Sorted [(valid_time, path), ...] for domain <domain> (merged across segments for a
    segmented run dir, see frame_refs). With multi-frame files several valid times share one
    path -- a caller that reads a time-varying field must use list_frame_refs/frame_ref for
    the timeidx instead of reading index 0."""
    return [(t, path) for t, path, _idx in list_frame_refs(run_dir, domain)]


def frame_ref(run_dir: str, domain: int, time_str: str) -> tuple[str, int]:
    """(path, timeidx) of valid time time_str on domain <domain> (segment-aware, Times-based,
    see frame_refs); (the flat run_dir/wrfout_dNN_<time_str> path, 0) when no file holds that
    time, so a caller's own exists()/open error still names the expected file."""
    flat = os.path.join(run_dir, "wrfout_d%02d_%s" % (int(domain), time_str))
    ref = frame_refs(run_dir).get((int(domain), time_str))
    if ref is not None:
        return ref
    return flat, 0


def frame_path(run_dir: str, domain: int, time_str: str) -> str:
    """Path of the file holding valid time time_str (frame_ref without the timeidx)."""
    return frame_ref(run_dir, domain, time_str)[0]


class MissingFields(ExtractError):
    """A wrfout history stream lacks variables/global attributes a diagnostic needs -- the
    usual cause is an iofields file that reduced the history stream (climate runs)."""

    def __init__(self, names, what=None):
        self.names = list(names)
        super().__init__("history stream lacks %s%s (iofields-reduced output?)"
                         % (", ".join(self.names), (" for " + what) if what else ""))


def require_fields(ds, variables=(), attributes=(), what=None) -> None:
    """Raise MissingFields naming every one of `variables` (netCDF variables) and `attributes`
    (global attributes) that ds lacks -- checked up front, so a recipe reports what the
    reduced history stream is missing instead of a bare AttributeError/KeyError."""
    missing = [v for v in variables if v not in ds.variables]
    attrs = set(ds.ncattrs()) if hasattr(ds, "ncattrs") else set()
    missing += [a for a in attributes if a not in attrs]
    if missing:
        raise MissingFields(missing, what)


def is_missing_field_error(exc: BaseException) -> bool:
    """True for the exceptions a reduced (iofields) history stream produces when a diagnostic
    reads a variable/attribute that is not there: MissingFields, a netCDF4/wrf-python KeyError,
    wrf.getvar's ValueError "'X' is not a valid variable name", netCDF's AttributeError
    "NetCDF: Attribute not found", wrf-python's RuntimeError "'X' not found in NetCDF file"."""
    if isinstance(exc, (MissingFields, KeyError)):
        return True
    text = str(exc)
    return ("is not a valid variable name" in text or "Attribute not found" in text
            or "not found in NetCDF" in text)


def missing_reason(exc: BaseException) -> str:
    """A human missing-entry reason for an exception raised while reading a wrfout frame:
    MissingFields keeps its own text; a netCDF/wrf-python KeyError/AttributeError/ValueError
    that names a variable or attribute becomes "history stream lacks <name> ..." rather than
    a raw exception repr."""
    if isinstance(exc, MissingFields):
        return str(exc)
    text = str(exc)
    m = (re.search(r"'([A-Za-z0-9_]+)' is not a valid variable name", text)   # wrf.getvar
         or re.search(r"'([A-Za-z0-9_]+)' not found", text))
    if m is None and isinstance(exc, KeyError) and exc.args:                # ds.variables[...]
        m = re.match(r"([A-Za-z0-9_]+)$", str(exc.args[0]))
    if m:
        return "history stream lacks %s (iofields-reduced output?)" % m.group(1)
    if "Attribute not found" in text:
        return ("history stream lacks a global attribute this diagnostic needs "
                "(iofields-reduced output?)")
    return "%s: %s" % (type(exc).__name__, text)


def accumulated_rain(ds, timeidx=0):
    """Accumulated total precipitation (mm) at timeidx as a 2-D ndarray: RAINC + RAINNC, plus
    (I_RAINC + I_RAINNC) * BUCKET_MM when the run used bucket_mm > 0 (WRF then resets
    RAINC/RAINNC every BUCKET_MM mm and counts the resets in I_RAIN*)."""
    import numpy as np
    v = ds.variables
    total = np.asarray(v["RAINC"][timeidx]) + np.asarray(v["RAINNC"][timeidx])
    bucket = float(getattr(ds, "BUCKET_MM", -1.0))
    tips = [np.asarray(v[n][timeidx], dtype="f8") for n in ("I_RAINC", "I_RAINNC") if n in v]
    if bucket > 0 and tips:
        total = total.astype("f8") + bucket * sum(tips)
    return total


def _surface_field(ds, wrf_mod, name, timeidx):
    if name == "slp":
        return wrf_mod.getvar(ds, "slp", timeidx=timeidx, units="hPa")
    if name == "t2":
        return wrf_mod.getvar(ds, "T2", timeidx=timeidx)
    if name == "td2":
        return wrf_mod.getvar(ds, "td2", timeidx=timeidx, units="degC")
    if name == "rh2":
        return wrf_mod.getvar(ds, "rh2", timeidx=timeidx)
    if name in ("wspd10", "wdir10"):
        wspd_wdir = wrf_mod.getvar(ds, "uvmet10_wspd_wdir", timeidx=timeidx, units="m s-1")
        return wspd_wdir[0] if name == "wspd10" else wspd_wdir[1]
    if name == "rain_acc":
        rainc = wrf_mod.getvar(ds, "RAINC", timeidx=timeidx)
        rainnc = wrf_mod.getvar(ds, "RAINNC", timeidx=timeidx)
        total = rainc + rainnc
        total.attrs = dict(rainc.attrs)    # xarray < 2025.11 drops attrs in arithmetic; get_cartopy needs 'projection'
        if float(getattr(ds, "BUCKET_MM", -1.0)) > 0:      # bucket_mm > 0: add the bucket tips
            total = total.astype("f8").copy(data=accumulated_rain(ds, timeidx))
        return total
    if name == "mdbz":
        return wrf_mod.getvar(ds, "mdbz", timeidx=timeidx)
    if name in ("mcape", "mcin"):
        cape2d = wrf_mod.getvar(ds, "cape_2d", timeidx=timeidx)
        return cape2d[0] if name == "mcape" else cape2d[1]
    if name == "pw":
        return wrf_mod.getvar(ds, "pw", timeidx=timeidx)
    raise ExtractError("unknown surface field %r" % name)


def _level_base_field(ds, wrf_mod, name, timeidx):
    if name == "z":
        return wrf_mod.getvar(ds, "z", timeidx=timeidx, units="m")
    if name == "tc":
        return wrf_mod.getvar(ds, "tc", timeidx=timeidx)
    if name == "ua":
        return wrf_mod.getvar(ds, "ua", timeidx=timeidx, units="m s-1")
    if name == "va":
        return wrf_mod.getvar(ds, "va", timeidx=timeidx, units="m s-1")
    if name == "wspd":
        return wrf_mod.getvar(ds, "wspd_wdir", timeidx=timeidx, units="m s-1")[0]
    raise ExtractError("unknown level field %r" % name)


def extract_frame(task: tuple) -> dict:
    """Runs in a worker process (module-level for picklability). task = (path, time_str,
    surface_vars, level_vars, levels_hpa[, timeidx]) -- timeidx (default 0) is the valid
    time's index inside a multi-frame file (frame_ref). Returns {"time": ..., fields...} with each
    surface field a 2-D ndarray and each level field a dict {level_hpa: 2-D ndarray}."""
    path, time_str, surface_vars, level_vars, levels_hpa = task[:5]
    timeidx = task[5] if len(task) > 5 else 0
    import numpy as np
    wrf_mod = _wrf()
    netcdf4 = _netcdf4()
    out = {"time": time_str}
    with netcdf4.Dataset(path) as ds:
        for name in surface_vars:
            if name == "rain_rate":
                continue           # derived after all frames are gathered
            out[name] = np.asarray(_surface_field(ds, wrf_mod, name, timeidx))
        if level_vars and levels_hpa:
            pressure = wrf_mod.getvar(ds, "pressure", timeidx=timeidx)
            for name in level_vars:
                base = _level_base_field(ds, wrf_mod, name, timeidx)
                out[name] = {lev: np.asarray(wrf_mod.interplevel(base, pressure, lev))
                            for lev in levels_hpa}
    return out


def _add_rain_rate(frames, time_fmt=WRF_TIME_FMT):
    import numpy as np
    from datetime import datetime
    if len(frames) < 2 or "rain_acc" not in frames[0]:
        for f in frames:
            f["rain_rate"] = None
        return
    frames[0]["rain_rate"] = None
    for i in range(1, len(frames)):
        dt_h = (datetime.strptime(frames[i]["time"], time_fmt)
               - datetime.strptime(frames[i - 1]["time"], time_fmt)).total_seconds() / 3600.0
        frames[i]["rain_rate"] = ((frames[i]["rain_acc"] - frames[i - 1]["rain_acc"]) / dt_h
                                  if dt_h > 0 else np.zeros_like(frames[i]["rain_acc"]))


def extract(run_dir: str, variables: list[str], out_path: str, domain: int = 1, times=None,
            levels_hpa=None, points=None, box=None, workers: int = 1) -> dict:
    """variables: list of field names (SURFACE_FIELDS + LEVEL_FIELDS). times: optional list
    of WRF time strings to restrict to (default: every frame). points: [(lat, lon, name)];
    box: (lat0, lat1, lon0, lon1). Exactly one of points/box must be given; the output
    format follows --out's extension for box (.nc) and is always CSV for points."""
    if bool(points) == bool(box):
        raise ExtractError("extract() needs exactly one of points or box")

    surface_vars = [v for v in variables if v in SURFACE_FIELDS]
    level_vars = [v for v in variables if v in LEVEL_FIELDS]
    unknown = [v for v in variables if v not in SURFACE_FIELDS and v not in LEVEL_FIELDS]
    if unknown:
        raise ExtractError("unknown --vars %s" % unknown)
    if level_vars and not levels_hpa:
        raise ExtractError("--levels required for %s" % level_vars)
    needed_surface = [v for v in surface_vars if v != "rain_rate"]
    if "rain_rate" in surface_vars and "rain_acc" not in needed_surface:
        needed_surface.append("rain_acc")

    all_frames = list_frame_refs(run_dir, domain=domain)
    if times:
        wanted = set(times)
        all_frames = [f for f in all_frames if f[0] in wanted]
    if not all_frames:
        raise ExtractError("no wrfout_d%02d_* frames found in %s" % (domain, run_dir))

    tasks = [(path, t, needed_surface, level_vars, levels_hpa or [], idx)
             for t, path, idx in all_frames]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            frames = list(ex.map(extract_frame, tasks))
    else:
        frames = [extract_frame(t) for t in tasks]
    frames.sort(key=lambda f: f["time"])

    if "rain_rate" in surface_vars:
        _add_rain_rate(frames)

    if points:
        _write_points_csv(frames, points, variables, levels_hpa or [], out_path,
                          all_frames[0][1])
        return dict(kind="points", out=out_path, n_frames=len(frames), n_points=len(points))
    _write_box_netcdf(frames, box, variables, levels_hpa or [], out_path, all_frames[0][1])
    return dict(kind="box", out=out_path, n_frames=len(frames), box=box)


def _nearest_indices(sample_path, points):
    """[(j, i, lat, lon, name), ...] for each requested point via wrf.ll_to_xy."""
    wrf_mod = _wrf()
    netcdf4 = _netcdf4()
    out = []
    with netcdf4.Dataset(sample_path) as ds:
        for lat, lon, name in points:
            xy = wrf_mod.ll_to_xy(ds, lat, lon)
            i, j = int(xy[0]), int(xy[1])
            out.append((j, i, lat, lon, name))
    return out


def _field_value(frame, var, j, i, levels_hpa):
    if var in LEVEL_FIELDS:
        return {lev: float(frame[var][lev][j, i]) for lev in levels_hpa}
    val = frame.get(var)
    if val is None:
        return None
    return float(val[j, i])


def _write_points_csv(frames, points, variables, levels_hpa, out_path, sample_path):
    idx = _nearest_indices(sample_path, points)
    columns = []
    for var in variables:
        if var in LEVEL_FIELDS:
            for lev in levels_hpa:
                columns.append("%s_%dhPa" % (var, int(lev)))
        else:
            columns.append(var)

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        for j, i, lat, lon, name in idx:
            w.writerow(["# point", name, "lat=%.4f" % lat, "lon=%.4f" % lon,
                       "grid_j=%d" % j, "grid_i=%d" % i])
            w.writerow(["time"] + ["%s[%s]" % (c, UNITS.get(c.split("_")[0], "")) for c in columns])
            for f in frames:
                row = [f["time"]]
                for var in variables:
                    if var in LEVEL_FIELDS:
                        for lev in levels_hpa:
                            row.append(f[var][lev][j, i])
                    else:
                        v = f.get(var)
                        row.append("" if v is None else float(v[j, i]))
                w.writerow(row)


def _write_box_netcdf(frames, box, variables, levels_hpa, out_path, sample_path):
    netcdf4 = _netcdf4()
    import numpy as np
    lat0, lat1, lon0, lon1 = box
    with netcdf4.Dataset(sample_path) as ds:
        lat2d = np.asarray(ds.variables["XLAT"][0])
        lon2d = np.asarray(ds.variables["XLONG"][0])
    mask = (lat2d >= min(lat0, lat1)) & (lat2d <= max(lat0, lat1)) \
        & (lon2d >= min(lon0, lon1)) & (lon2d <= max(lon0, lon1))
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ExtractError("--box %s does not intersect the domain" % (box,))
    j0, j1, i0, i1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

    with netcdf4.Dataset(out_path, "w") as out:
        out.createDimension("time", len(frames))
        out.createDimension("south_north", j1 - j0)
        out.createDimension("west_east", i1 - i0)
        tvar = out.createVariable("time", str, ("time",))
        for k, f in enumerate(frames):
            tvar[k] = f["time"]
        xlat_var = out.createVariable("XLAT", "f4", ("south_north", "west_east"))
        xlat_var[:] = lat2d[j0:j1, i0:i1]
        xlat_var.standard_name = "latitude"
        xlat_var.units = "degrees_north"
        xlong_var = out.createVariable("XLONG", "f4", ("south_north", "west_east"))
        xlong_var[:] = lon2d[j0:j1, i0:i1]
        xlong_var.standard_name = "longitude"
        xlong_var.units = "degrees_east"
        # CF "coordinates" attribute on every data variable, naming XLAT/XLONG as its 2-D
        # curvilinear grid coordinates. Without this (and the standard_name/units above), cdo
        # cannot tell XLAT/XLONG apart from an ordinary (south_north, west_east) data field --
        # `cdo sub` on two otherwise-identical extractions was found to difference XLAT/XLONG
        # right along with the requested variable, zeroing them out (post/changemap.py's own
        # change-map figures came out blank until this was added; caught on the scc P7
        # validation run, 2026-09-29).
        for var in variables:
            if var in LEVEL_FIELDS:
                for lev in levels_hpa:
                    name = "%s_%dhPa" % (var, int(lev))
                    v = out.createVariable(name, "f4", ("time", "south_north", "west_east"),
                                           zlib=True, complevel=4)
                    v.units = UNITS.get(var, "")
                    v.coordinates = "XLAT XLONG"
                    for k, f in enumerate(frames):
                        v[k] = f[var][lev][j0:j1, i0:i1]
            else:
                v = out.createVariable(var, "f4", ("time", "south_north", "west_east"),
                                       zlib=True, complevel=4)
                v.units = UNITS.get(var, "")
                v.coordinates = "XLAT XLONG"
                for k, f in enumerate(frames):
                    val = f.get(var)
                    v[k] = (np.full((j1 - j0, i1 - i0), np.nan, dtype="f4") if val is None
                            else val[j0:j1, i0:i1])
