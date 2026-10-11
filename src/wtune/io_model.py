#!/usr/bin/env python3
"""wtune/io_model.py -- run-length I/O terms of the whole-run prediction (release 0.8.2).

wtune.probe_plan.combine predicts compute + init only; on a long run I/O dominates what it misses
(scc climate-ea50: predicted 46:14, actual 89:38; rsl decomposition in the anchors below). Terms for
one unsegmented run: history (frames x per-frame write), boundary (d01 updates), aux (sst_update
wrflowinp reads), restart (writes; all domains in one), iofields (ADIOS2 only: re-parses per I/O
call). History/restart copy wtune.run's fallback formula (not imported: circular; wtune.run
re-exports these constants, tests/test_wtune_io_model.py checks agreement). Not included: per-segment
startup (`init_s`), restart reads, auxhist streams.
"""
from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

from wtune.tools import coldstart_sample as coldstart    # tools/: wrf_interval_minutes (WRF's own _d/_h/_m/_s sum)
from wtune import predict as wp

# --- re-exported by wtune/run/ ("Fallback I/O anchors" block, fallback_restart_io_s,
# fallback_history_frame_s) ---------------------
CONUS25_RESTART_POINTS = 1500 * 1200 * 50
RESTART_IO_ANCHOR_S = {2: (324.0, 120.0), 14: (1.05, 2.3)}   # io_form: (write, read) at conus25
DORIAN_HISTORY_WRITE_S = 300.5
DORIAN_HISTORY_POINT_FRAMES = 7 * 664 * 700 * 40 + 25 * 451 * 451 * 40
HISTORY_S_PER_POINT_FRAME = DORIAN_HISTORY_WRITE_S / DORIAN_HISTORY_POINT_FRAMES
# --- end of the copy ---------------------------------------------------------------------------

# climate-ea50 (scc, 2026-10, 150x110x45, N=5 120x3) rsl: history+restart 1598 x 0.74 s, boundary
# 1584 x 0.31 s, other in-step ~260 s (mostly sst_update auxinput4 reads, charged to aux: upper bound)
CLIMATE_EA50_BOUNDARY_S = 0.31                       # per lateral boundary update
CLIMATE_EA50_BOUNDARY_PERIMETER = 2 * (150 + 110) * 45
BOUNDARY_S_PER_PERIMETER_POINT = CLIMATE_EA50_BOUNDARY_S / CLIMATE_EA50_BOUNDARY_PERIMETER
CLIMATE_EA50_AUX4_S = 260.0 / 1584                   # per auxinput4 read (upper bound, see above)
CLIMATE_EA50_AUX4_COLUMNS = 150 * 110
AUX4_S_PER_COLUMN = CLIMATE_EA50_AUX4_S / CLIMATE_EA50_AUX4_COLUMNS

# ADIOS2 (io_form 14, the wtune run default; scc rsl "Timing for Writing", 2026-10). History:
# SC26-P2 0.042 s/write at 170x130x40 (P1 0.044, netCDF 0.888/1.226) -> per point-frame. Boundary:
# climate-ea50 0.24 s/update (netCDF 0.31); P2 saw ~0.03 s, so the climate anchor is conservative.
# Restart already has its io_form 14 anchor above. Boundary and aux: climate-ea50 WITHOUT an
# iofields file (job 3144, 151x111x45, 120 ranks): 0.0216 s per boundary update, and the 0.078 s
# I/O step less boundary and the 0.0233 s history frame = 0.033 s per auxinput4 read (upper
# bound: it carries the step's bookkeeping). The earlier 0.24 s boundary anchor held one
# iofields parse (0.21 s); P2 measured ~0.03 s per update.
ADIOS2_HISTORY_S_PER_POINT_FRAME = 0.042 / (170 * 130 * 40)
ADIOS2_BOUNDARY_S_PER_PERIMETER_POINT = 0.0216 / (2 * (151 + 111) * 45)
ADIOS2_AUX4_S_PER_COLUMN = 0.033 / (151 * 111)
# iofields: seconds per get_fieldstr call (scc nvhpc, job 3134 perf: 0.21 s / 227 calls; 0.66-0.89
# ms measured, more with longer lines). machine/calibration/<site>.json "iofields_call_ms".
IOFIELDS_CALL_MS = 0.9
ADIOS2_IO_FORM = 14
DEFAULT_IO_FORMAT = "adios2"
WRF_DEFAULT_RESTART_INTERVAL_MIN = 1440.0            # Registry default when the namelist omits it

SEGMENT_NOTE = ("each extra run segment adds ~%.0f s (init_s, the predicted startup); the "
                "segment count is decided by `wtune run`, not here")


def restart_io_s(points: Sequence[int], io_form: Any) -> tuple[float, float]:
    """(write_s, read_s) of one restart of every domain (`points` = per-domain e_we x e_sn x
    e_vert), conus25 anchor; an io_form without an anchor uses netCDF's (2)."""
    w, r = RESTART_IO_ANCHOR_S.get(int(io_form), RESTART_IO_ANCHOR_S[2])
    f = sum(points) / float(CONUS25_RESTART_POINTS)
    return w * f, r * f


def normalize_io_format(io: Any) -> str:
    """'adios2' | 'netcdf' from wtune run's --io spelling or an io_form number (14 / 2);
    None -> the default (adios2)."""
    if io is None:
        return DEFAULT_IO_FORMAT
    v = str(io).strip().lower()
    if v in ("adios2", "adios", "bp", "14"):
        return "adios2"
    if v in ("netcdf", "nc", "2"):
        return "netcdf"
    raise ValueError("unknown I/O format %r (adios2 | netcdf)" % (io,))


def history_frame_s(points: Sequence[int], io_format: str = "netcdf") -> list[float]:
    """Per-domain seconds of one history frame: dorian2019 anchor (netCDF) or the SC26-P2
    ADIOS2 anchor."""
    rate = (ADIOS2_HISTORY_S_PER_POINT_FRAME if io_format == "adios2"
            else HISTORY_S_PER_POINT_FRAME)
    return [p * rate for p in points]


def boundary_update_s(e_we: int, e_sn: int, e_vert: int, io_format: str = "netcdf") -> float:
    """Seconds of one d01 lateral boundary update."""
    rate = (ADIOS2_BOUNDARY_S_PER_PERIMETER_POINT if io_format == "adios2"
            else BOUNDARY_S_PER_PERIMETER_POINT)
    return 2 * (e_we + e_sn) * e_vert * rate


def aux4_read_s(e_we: int, e_sn: int, io_format: str = "netcdf") -> float:
    """Seconds of one auxinput4 (wrflowinp) read for one domain."""
    return e_we * e_sn * (ADIOS2_AUX4_S_PER_COLUMN if io_format == "adios2"
                          else AUX4_S_PER_COLUMN)


def iofields_call_s(calibration: Optional[dict] = None) -> float:
    """Seconds per get_fieldstr call: the machine calibration's iofields_call_ms, else 0.9 ms."""
    return float((calibration or {}).get("iofields_call_ms", IOFIELDS_CALL_MS)) / 1000.0


def iofields_parses(frames: Sequence[int], frames_per_outfile: Sequence[Any],
                    n_aux: Sequence[int], n_boundary: int) -> list[int]:
    """Per-domain count of iofields parses in a run: one per history frame, one more per
    history file opened, two per auxinput read, one per boundary update (d01 only)."""
    return [n + (-(-n // max(1, int(fpo)))) + 2 * aux + (n_boundary if d == 0 else 0)
            for d, (n, fpo, aux) in enumerate(zip(frames, frames_per_outfile, n_aux))]


def iofields_parse_s(parses: Sequence[int], calls: Optional[Sequence[int]], call_s: float) -> float:
    """Seconds a run spends re-parsing iofields files: per domain parses x get_fieldstr calls
    of its file (wtune.iofields.parse_calls; 0/None = no file) x seconds per call."""
    return sum(n * (c or 0) for n, c in zip(parses, calls or [])) * call_s


def case_iofields_calls(nl: dict, case_dir: str, merge: bool = True) -> list[int]:
    """Per-domain get_fieldstr calls per parse of the case's iofields files as `wtune run`
    stages them (lines merged; merge=False: as written); 0 where a domain names no file or it
    is missing."""
    import wtune.iofields
    out = []
    for d in range(wp.get_max_dom(nl)):
        name = str(wp.g_dom(nl, "time_control", "iofields_filename", d, "") or "").strip()
        path = os.path.join(case_dir, name.replace("<domain>", "%02d" % (d + 1)))
        text = ""
        if name and name.upper() != "NONE_SPECIFIED" and os.path.isfile(path):
            with open(path) as f:
                text = f.read()
        out.append(wtune.iofields.parse_calls(wtune.iofields.merged(text) if merge else text))
    return out


def history_frames_whole_run(run_s: float, iv_s: float, begin_s: float = 0.0) -> int:
    """Frames at begin + j*iv in [0, run_s], the t=0 frame included when begin is 0 --
    wtune.run.history_frames(geo, d, 0, steps) for a whole run."""
    if not iv_s or iv_s <= 0:
        return 0

    def upto(x):
        return 0 if x < begin_s else int(math.floor((x - begin_s) / iv_s + 1e-9)) + 1
    n = upto(run_s) - upto(0.0)
    if begin_s == 0:
        n += 1
    return n


def _events(run_s: float, iv_s: float) -> int:
    """Interval events after t=0 up to and including run_s."""
    return int(math.floor(run_s / iv_s + 1e-9)) if iv_s and iv_s > 0 else 0


def _flag(v: Any) -> bool:
    return v is True or str(v).strip().strip(".").lower() in ("true", "t", "1")


def _history_term(nl: dict, points: Sequence[int], run_s: float, fmt: str) -> tuple[list[int], list[float], float]:
    """(frames per domain, seconds per frame per domain, total seconds)."""
    io_hist = wp.g(nl, "time_control", "io_form_history", 2)
    frames, frame_s = [], history_frame_s(points, fmt)
    for d in range(len(points)):
        iv = coldstart.wrf_interval_minutes(nl, "history_interval", d) * 60.0
        begin = coldstart.wrf_interval_minutes(nl, "history_begin", d) * 60.0
        frames.append(history_frames_whole_run(run_s, iv, begin) if io_hist and iv > 0 else 0)
    return frames, frame_s, sum(n * s for n, s in zip(frames, frame_s))


def _boundary_term(nl: dict, geom: Sequence[Sequence[int]], run_s: float,
                   fmt: str) -> tuple[int, float, float]:
    """(d01 boundary updates, seconds per update, total seconds)."""
    specified = wp.g(nl, "bdy_control", "specified", None)
    bdy_iv = float(wp.g(nl, "time_control", "interval_seconds", 0) or 0)
    n_bdy = _events(run_s, bdy_iv) if (specified is None or _flag(specified)) else 0
    one = boundary_update_s(*geom[0], io_format=fmt)
    return n_bdy, one, n_bdy * one


def _aux_term(nl: dict, geom: Sequence[Sequence[int]], run_s: float,
              fmt: str) -> tuple[list[int], list[float], float]:
    """(auxinput4 reads per domain, seconds per read per domain, total seconds); sst_update=1 only."""
    max_dom = len(geom)
    n_aux, one = [0] * max_dom, [aux4_read_s(a, b, fmt) for a, b, _c in geom]
    if int(wp.g(nl, "physics", "sst_update", 0) or 0) == 1:
        for d in range(max_dom):
            n_aux[d] = _events(run_s, coldstart.wrf_interval_minutes(nl, "auxinput4_interval",
                                                                      d) * 60.0)
    return n_aux, one, sum(n * s for n, s in zip(n_aux, one))


def _restart_term(nl: dict, points: Sequence[int], run_s: float,
                  fmt: str) -> tuple[float, int, float, int, float]:
    """(interval min, io_form, seconds per write, writes, total seconds)."""
    rst_min = coldstart.wrf_interval_minutes(nl, "restart_interval")
    if rst_min <= 0:
        rst_min = WRF_DEFAULT_RESTART_INTERVAL_MIN
    io_rst = (ADIOS2_IO_FORM if fmt == "adios2"
              else int(wp.g(nl, "time_control", "io_form_restart", 2) or 2))
    rst_w, _rst_r = restart_io_s(points, io_rst)
    n_rst = _events(run_s, rst_min * 60.0)
    return rst_min, io_rst, rst_w, n_rst, n_rst * rst_w


def run_length_io(nl: dict, run_s: Optional[float] = None, io: Any = "adios2",
                  iofields_calls: Optional[Sequence[int]] = None,
                  call_s: Optional[float] = None) -> dict:
    """Run-length I/O of one unsegmented run of namelist `nl` (wtune.predict.parse_namelist form);
    run_s defaults to the d01 length. Dict: total_s and history/boundary/aux/restart/iofields_s,
    the counts and per-unit costs behind each, run_s, points, io_format, source. io "netcdf"
    keeps the namelist's io_form_restart and original anchors; "adios2" (default) uses the ADIOS2
    anchors and io_form 14 restart. iofields_calls/call_s price iofields re-parsing (ADIOS2 only)."""
    fmt = normalize_io_format(io)
    if run_s is None:
        run_s = wp.get_d01_steps(nl)[1]
    run_s = float(run_s)
    max_dom = wp.get_max_dom(nl)
    geom = [wp.get_domain_geom(nl, d) for d in range(max_dom)]
    points = [int(a) * int(b) * int(c) for a, b, c in geom]

    frames, frame_s, history_s = _history_term(nl, points, run_s, fmt)
    n_bdy, bdy_one, boundary_s = _boundary_term(nl, geom, run_s, fmt)
    n_aux, aux_one, aux_s = _aux_term(nl, geom, run_s, fmt)
    rst_min, io_rst, rst_w, n_rst, restart_s = _restart_term(nl, points, run_s, fmt)

    fpo = [wp.g_dom(nl, "time_control", "frames_per_outfile", d, 1) or 1 for d in range(max_dom)]
    parses = iofields_parses(frames, fpo, n_aux, n_bdy)
    calls = list(iofields_calls or [0] * max_dom) if fmt == "adios2" else [0] * max_dom
    call_s = iofields_call_s() if call_s is None else call_s
    iofields_s = iofields_parse_s(parses, calls, call_s)

    total = history_s + boundary_s + aux_s + restart_s + iofields_s
    return dict(total_s=total, history_s=history_s, boundary_s=boundary_s, aux_s=aux_s,
                restart_s=restart_s, iofields_s=iofields_s, iofields_parses=parses,
                iofields_calls=calls, iofields_call_s=call_s,
                history_frames=frames, history_frame_s=frame_s,
                n_boundary=n_bdy, boundary_update_s=bdy_one, n_aux=n_aux, aux_read_s=aux_one,
                n_restart=n_rst, restart_write_s=rst_w, restart_interval_min=rst_min,
                io_form_restart=io_rst, run_s=run_s, points=points, io_format=fmt,
                source="wtune_io_model[%s] (history/restart: wtune_run fallback anchors, "
                       "adios2 history: SC26-P2 rsl; boundary/aux: climate-ea50 rsl; iofields: "
                       "parses x get_fieldstr calls)" % fmt)


def _s(x: float) -> str:
    return "%.0fs" % x


def breakdown_line(compute_s: float, io: dict, init_s: float) -> str:
    """'compute X + history Y + boundary Z + aux W + restart R + init I = TOTAL' (seconds);
    '+ iofields P' after restart when the run re-parses an iofields file."""
    total = compute_s + io["total_s"] + init_s
    iof = " + iofields %s" % _s(io["iofields_s"]) if io.get("iofields_s") else ""
    return ("compute %s + history %s + boundary %s + aux %s + restart %s%s + init %s = %s"
            % (_s(compute_s), _s(io["history_s"]), _s(io["boundary_s"]), _s(io["aux_s"]),
               _s(io["restart_s"]), iof, _s(init_s), _s(total)))


def segment_note(init_s: float) -> str:
    return SEGMENT_NOTE % init_s
