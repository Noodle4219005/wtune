"""Disk space for a run's outputs, checked before the first job and after every job.

WRF+ADIOS2 does not fail on a full disk: it waits (the stall watch cancels the job) or, past a
GPFS soft quota, prints SUCCESS COMPLETE WRF with history frames that lost their data (f1-arm
marshall 2026-10-21: d02 frames shrank from 1.8 GB to 17 KB, run_report said completed). So the
output volume is estimated up front against the space actually writable, and each segment's
history frames are compared with the stream's largest frame afterwards."""
from __future__ import annotations

import contextlib
import glob
import gzip
import os
import pwd
import shutil
import subprocess
import sys
from typing import Callable, Optional

from wtune.preflight import _ASSUMED_3D_FIELDS, _BYTES_PER_VALUE, _fmt_bytes

MMLSQUOTA = "/usr/lpp/mmfs/bin/mmlsquota"
SHRUNK_FRAC = 0.1          # a history output under this fraction of its stream's largest: lost data


def frame_bytes(points: int) -> float:
    """One history frame of a domain with `points` = e_we*e_sn*e_vert (preflight C9's formula;
    marshall d02 505x505: 1.84 GB estimated, 1.8 GB written)."""
    return float(points) * _BYTES_PER_VALUE * _ASSUMED_3D_FIELDS


def estimate_output_bytes(geo, span_s: float, n_restart_sets: int) -> tuple[float, str]:
    """(bytes, per-domain detail): history frames over span_s seconds plus n_restart_sets
    restart sets of every domain (a restart counted as one history frame)."""
    total, detail = 0.0, []
    for d, h in enumerate(geo.hist):
        fb = frame_bytes(geo.points[d])
        frames = int((span_s - h.begin_s) // h.iv_s) + 1 if h and span_s >= h.begin_s else 0
        total += fb * (frames + n_restart_sets)
        if frames:
            detail.append("d%02d %d x %s" % (d + 1, frames, _fmt_bytes(fb)))
    return total, ", ".join(detail)


def netcdf_data_bytes(path: str) -> float:
    """Uncompressed bytes of a netCDF file's variables (its BP conversion is uncompressed: the
    2.0 GB of elevconv netCDF inputs became 7.9 GB of BP); 4x the file size when unreadable."""
    try:
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            return float(sum(v.size * v.dtype.itemsize for v in ds.variables.values()))
    except Exception:                                   # noqa: BLE001 -- an estimate only
        return 4.0 * os.path.getsize(path) if os.path.exists(path) else 0.0


def input_bytes(bp_plan: Optional[dict], cache_dir: Optional[str], bp) -> tuple[float, float]:
    """(BP conversion still to write into cache_dir, one per-attempt copy of the per-record
    inputs) for a run's input plan; (0, 0) for --io netcdf."""
    if not bp_plan:
        return 0.0, 0.0
    by_dir: dict = {}
    for p in bp_plan.get("convert") or []:
        by_dir.setdefault(os.path.dirname(p), []).append(p)
    cached = getattr(bp, "cached_inputs", None)
    convert = sum(netcdf_data_bytes(p) for d, ps in by_dir.items()
                  if cached is None or cached(d, cache_dir, [os.path.basename(p) for p in ps]) is None
                  for p in ps)
    sources = {os.path.basename(p): p for p in bp_plan.get("convert") or []}
    fresh = sum(_size(t) if os.path.exists(t) else netcdf_data_bytes(sources.get(n, ""))
                for n, t in (bp_plan.get("per_record") or {}).items())
    return convert, fresh


def _gpfs_device(path: str) -> Optional[str]:
    """The GPFS filesystem holding path (longest mount point in /proc/mounts), else None."""
    best = (None, "")
    try:
        with open("/proc/mounts") as fh:
            for line in fh:
                dev, mnt, fstype = line.split()[:3]
                if fstype == "gpfs" and (path == mnt or path.startswith(mnt.rstrip("/") + "/")) \
                        and len(mnt) > len(best[1]):
                    best = (dev, mnt)
    except OSError:
        return None
    return best[0]


def quota_headroom(path: str, runner: Optional[Callable] = None) -> Optional[float]:
    """Bytes left under the user's GPFS block quota (the soft limit, where f1-arm already
    blocks writes; the hard limit when no soft one is set) for path's filesystem, else None."""
    dev = _gpfs_device(os.path.realpath(path))
    if not dev or (runner is None and not os.access(MMLSQUOTA, os.X_OK)):
        return None
    try:
        out = (runner or _run)([MMLSQUOTA, "-u", pwd.getpwuid(os.getuid()).pw_name, "--block-size", "K", dev])
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 5 and f[0] == dev and f[1] == "USR":
            used, soft, hard = (int(x) for x in f[2:5])
            limit = soft or hard
            return (limit - used) * 1024.0 if limit else None
    return None


def _run(argv: list) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout


def writable_bytes(path: str, runner: Optional[Callable] = None) -> tuple[float, str]:
    """(bytes writable at path, where the number comes from): the smaller of the filesystem's
    free space and the GPFS quota headroom."""
    st = os.statvfs(path)
    free, src = float(st.f_bavail * st.f_frsize), "filesystem free space"
    q = quota_headroom(path, runner)
    if q is not None and q < free:
        free, src = q, "GPFS quota headroom"
    return free, src


def shrunk_outputs(seg_dir: str, names: list) -> list:
    """History outputs of the segment (wrfout_*) under SHRUNK_FRAC of the largest output of the
    same stream and kind (BP dir or netCDF file). An absent one is not counted: the planned
    names ignore history_begin and late nest starts, so absence alone proves nothing."""
    groups: dict = {}
    for n in names:
        p = os.path.join(seg_dir, n)
        if n.startswith("wrfout_") and os.path.lexists(p):
            groups.setdefault((n[:10], os.path.isdir(p)), []).append((n, _size(p)))
    out = []
    for items in groups.values():
        top = max(s for _n, s in items)
        out += [n for n, s in items if s < SHRUNK_FRAC * top]
    return sorted(out)


def gzip_rsl(seg_dir: str) -> tuple[int, int, int]:
    """gzip -1 every rsl.out.NNNN / rsl.error.NNNN of the segment but rank 0's (the progress,
    timing and FATAL readers use rank 0 or run before this); (files, bytes before, after).
    sc26p1's frozen debug_level=50 wrote 17.6 GB of rsl in 22 minutes; gzip -1 takes it 17x
    down."""
    n = before = after = 0
    for p in sorted(glob.glob(os.path.join(seg_dir, "rsl.out.*")) +
                    glob.glob(os.path.join(seg_dir, "rsl.error.*"))):
        if p.endswith((".0000", ".gz")) or os.path.islink(p) or not os.path.isfile(p):
            continue
        tmp = p + ".gz.tmp"
        try:
            with open(p, "rb") as src, gzip.open(tmp, "wb", compresslevel=1) as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            size = os.path.getsize(p)
            os.replace(tmp, p + ".gz")
            os.remove(p)
        except OSError as e:            # a full quota: the rest stay plain, the run goes on
            with contextlib.suppress(OSError):
                os.remove(tmp)
            sys.stderr.write("WARNING: rsl gzip stopped at %s: %s\n" % (os.path.basename(p), e))
            break
        before += size
        after += os.path.getsize(p + ".gz")
        n += 1
    return n, before, after


THOMPSON_TABLES = ("freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qg_mp38V1.dat", "qr_acr_qsV2.dat")


def drop_older_restarts(seg_dir: str, max_dom: int, is_bp: Callable[[str], bool],
                        keep: Optional[str] = None) -> int:
    """After a finished attempt, keep only the newest valid restart set WRF wrote into seg_dir:
    the next segment starts from it, and no resume of this attempt follows. With `keep` (a
    WRF_FMT time), keep that set instead: the failed attempt a finished resume started from.
    Bytes freed."""
    import wtune.resume
    try:
        times = wtune.resume.valid_restart_times(seg_dir, max_dom, is_bp)
    except (OSError, ValueError):
        return 0
    freed = 0
    for t in (times[:-1] if keep is None else [t for t in times if t != keep]):
        for d in range(1, max_dom + 1):
            for p in glob.glob(os.path.join(seg_dir, "wrfrst_d%02d_%s*" % (d, glob.escape(t)))):
                try:
                    if os.path.isdir(p) and not os.path.islink(p):
                        freed += sum(os.path.getsize(os.path.join(r, f))
                                     for r, _d, fs in os.walk(p) for f in fs)
                        shutil.rmtree(p)
                    else:
                        freed += 0 if os.path.islink(p) else os.path.getsize(p)
                        os.remove(p)
                except OSError:
                    pass                                    # disk only: never stops a run
    return freed


def remove_computed_tables(seg_dir: str) -> int:
    """Delete the Thompson tables WRF computed into the segment (357 MiB on f1-arm, every
    segment, without --thompson-cache); a case's own tables are links and stay. Bytes freed."""
    freed = 0
    for n in THOMPSON_TABLES:
        p = os.path.join(seg_dir, n)
        if os.path.isfile(p) and not os.path.islink(p):
            try:
                size = os.path.getsize(p)
                os.remove(p)
                freed += size
            except OSError:
                pass                                    # disk only: never stops a run
    return freed


def _size(p: str) -> int:
    """Bytes of a file or of every file under a directory; -1 when p does not exist."""
    if not os.path.lexists(p):
        return -1
    if not os.path.isdir(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(p) for f in fs)
