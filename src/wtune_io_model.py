#!/usr/bin/env python3
"""wtune_io_model.py -- run-length I/O terms of the whole-run prediction (release 0.8.2).

The -O pipeline's whole-run prediction (wtune_probe_plan.combine) is compute + init only. On a
long run the run-length I/O dominates what that misses (scc climate-ea50, 150x110x45, 13 months,
N=5: predicted 46:14, actual 89:38 summed over 14 segments; rsl decomposition: history+restart
writes 1598 x 0.74 s = 1181 s, lateral boundary processing 1584 x 0.31 s = 494 s, other in-step
overhead ~260 s, mostly the sst_update auxinput4 reads, per-segment startup 14 x ~53 s). This
module adds, for one unsegmented run of the case's own length:

  history   per domain: frames (history_interval / history_begin, the t=0 frame included -- the
            same count as wtune_run.history_frames over (0, steps]) x per-frame write seconds
  boundary  d01 lateral boundary updates floor(run_s / interval_seconds) x per-update seconds
  aux       sst_update=1: per domain floor(run_s / auxinput4_interval) wrflowinp reads x per-read
            seconds
  restart   floor(run_min / restart_interval) writes (all domains in one write) x per-write
            seconds; restart_interval absent -> WRF's Registry default 1440 min

Per-unit costs. History and restart are COPIES of wtune_run.py's fallback I/O formula
(fallback_restart_io_s / fallback_history_frame_s, measured scc anchors scaled by grid points x
levels). wtune_run.py is not imported: it pulls in the Slurm controller stack (wtune_watch,
scc_sweep) and imports wtune_tune itself, so wtune_tune importing it would be circular. Keep the
two in sync -- tests/test_wtune_io_model.py asserts they give the same numbers. Boundary and
auxinput4 have one anchor each, the climate-ea50 rsl above (single-case calibration, no
cross-check yet):
  boundary  0.31 s per update at 2 x (150 + 110) x 45 perimeter points x levels, scaled by that
            (the wrfbdy record is the 4 edges x spec_bdy_width x levels)
  aux       ~260 s / 1584 reads = 0.164 s per read at 150 x 110 columns, scaled by e_we x e_sn
            (wrflowinp carries 2-D fields); ALL of the ~260 s "other in-step" overhead is charged
            to it, so it is an upper bound

Not included: per-segment startup (the segment count is not known at tune time; each extra
segment adds ~init_s, exposed as `init_s` for wtune_run), restart reads, auxhist streams.
"""
import math

import coldstart_sample as coldstart    # tools/: wrf_interval_minutes (WRF's own _d/_h/_m/_s sum)
import wtune_predict as wp

# --- copied from wtune_run.py ("Fallback I/O anchors" block, fallback_restart_io_s,
# fallback_history_frame_s); tests/test_wtune_io_model.py asserts equality ---------------------
CONUS25_RESTART_POINTS = 1500 * 1200 * 50
RESTART_IO_ANCHOR_S = {2: (324.0, 120.0), 14: (1.05, 2.3)}   # io_form: (write, read) at conus25
DORIAN_HISTORY_WRITE_S = 300.5
DORIAN_HISTORY_POINT_FRAMES = 7 * 664 * 700 * 40 + 25 * 451 * 451 * 40
HISTORY_S_PER_POINT_FRAME = DORIAN_HISTORY_WRITE_S / DORIAN_HISTORY_POINT_FRAMES
# --- end of the copy ---------------------------------------------------------------------------

# climate-ea50 (scc, 2026-10, 150x110x45, N=5 120x3) rsl anchors
CLIMATE_EA50_BOUNDARY_S = 0.31                       # per lateral boundary update
CLIMATE_EA50_BOUNDARY_PERIMETER = 2 * (150 + 110) * 45
BOUNDARY_S_PER_PERIMETER_POINT = CLIMATE_EA50_BOUNDARY_S / CLIMATE_EA50_BOUNDARY_PERIMETER
CLIMATE_EA50_AUX4_S = 260.0 / 1584                   # per auxinput4 read (upper bound, see above)
CLIMATE_EA50_AUX4_COLUMNS = 150 * 110
AUX4_S_PER_COLUMN = CLIMATE_EA50_AUX4_S / CLIMATE_EA50_AUX4_COLUMNS
WRF_DEFAULT_RESTART_INTERVAL_MIN = 1440.0            # Registry default when the namelist omits it

SEGMENT_NOTE = ("each extra run segment adds ~%.0f s (init_s, the predicted startup); the "
                "segment count is decided by `wtune run`, not here")


def restart_io_s(points, io_form):
    """(write_s, read_s) of one restart of every domain (`points` = per-domain e_we x e_sn x
    e_vert), conus25 anchor; an io_form without an anchor uses netCDF's (2)."""
    w, r = RESTART_IO_ANCHOR_S.get(int(io_form), RESTART_IO_ANCHOR_S[2])
    f = sum(points) / float(CONUS25_RESTART_POINTS)
    return w * f, r * f


def history_frame_s(points):
    """Per-domain seconds of one history frame, dorian2019 anchor."""
    return [p * HISTORY_S_PER_POINT_FRAME for p in points]


def boundary_update_s(e_we, e_sn, e_vert):
    """Seconds of one d01 lateral boundary update."""
    return 2 * (e_we + e_sn) * e_vert * BOUNDARY_S_PER_PERIMETER_POINT


def aux4_read_s(e_we, e_sn):
    """Seconds of one auxinput4 (wrflowinp) read for one domain."""
    return e_we * e_sn * AUX4_S_PER_COLUMN


def history_frames_whole_run(run_s, iv_s, begin_s=0.0):
    """Frames at begin + j*iv in [0, run_s], the t=0 frame included when begin is 0 --
    wtune_run.history_frames(geo, d, 0, steps) for a whole run."""
    if not iv_s or iv_s <= 0:
        return 0

    def upto(x):
        return 0 if x < begin_s else int(math.floor((x - begin_s) / iv_s + 1e-9)) + 1
    n = upto(run_s) - upto(0.0)
    if begin_s == 0:
        n += 1
    return n


def _events(run_s, iv_s):
    """Interval events after t=0 up to and including run_s."""
    return int(math.floor(run_s / iv_s + 1e-9)) if iv_s and iv_s > 0 else 0


def _flag(v):
    return v is True or str(v).strip().strip(".").lower() in ("true", "t", "1")


def run_length_io(nl, run_s=None):
    """Run-length I/O of one unsegmented run of namelist `nl` ({group: {key: [values]}}, as
    wtune_predict.parse_namelist returns). run_s defaults to the d01 run length. Returns a dict:
    total_s, history_s, boundary_s, aux_s, restart_s, and the counts / per-unit costs behind each
    (history_frames, history_frame_s per domain; n_boundary, boundary_update_s; n_aux,
    aux_read_s per domain; n_restart, restart_write_s, restart_interval_min, io_form_restart),
    plus run_s, points, source."""
    if run_s is None:
        run_s = wp.get_d01_steps(nl)[1]
    run_s = float(run_s)
    max_dom = wp.get_max_dom(nl)
    geom = [wp.get_domain_geom(nl, d) for d in range(max_dom)]
    points = [int(a) * int(b) * int(c) for a, b, c in geom]

    io_hist = wp.g(nl, "time_control", "io_form_history", 2)
    frames, frame_s = [], history_frame_s(points)
    for d in range(max_dom):
        iv = coldstart.wrf_interval_minutes(nl, "history_interval", d) * 60.0
        begin = coldstart.wrf_interval_minutes(nl, "history_begin", d) * 60.0
        frames.append(history_frames_whole_run(run_s, iv, begin) if io_hist and iv > 0 else 0)
    history_s = sum(n * s for n, s in zip(frames, frame_s))

    specified = wp.g(nl, "bdy_control", "specified", None)
    bdy_iv = float(wp.g(nl, "time_control", "interval_seconds", 0) or 0)
    n_bdy = _events(run_s, bdy_iv) if (specified is None or _flag(specified)) else 0
    bdy_one = boundary_update_s(*geom[0])
    boundary_s = n_bdy * bdy_one

    n_aux, aux_one = [0] * max_dom, [aux4_read_s(a, b) for a, b, _c in geom]
    if int(wp.g(nl, "physics", "sst_update", 0) or 0) == 1:
        for d in range(max_dom):
            n_aux[d] = _events(run_s, coldstart.wrf_interval_minutes(nl, "auxinput4_interval",
                                                                      d) * 60.0)
    aux_s = sum(n * s for n, s in zip(n_aux, aux_one))

    rst_min = coldstart.wrf_interval_minutes(nl, "restart_interval")
    if rst_min <= 0:
        rst_min = WRF_DEFAULT_RESTART_INTERVAL_MIN
    io_rst = int(wp.g(nl, "time_control", "io_form_restart", 2) or 2)
    rst_w, _rst_r = restart_io_s(points, io_rst)
    n_rst = _events(run_s, rst_min * 60.0)
    restart_s = n_rst * rst_w

    total = history_s + boundary_s + aux_s + restart_s
    return dict(total_s=total, history_s=history_s, boundary_s=boundary_s, aux_s=aux_s,
                restart_s=restart_s, history_frames=frames, history_frame_s=frame_s,
                n_boundary=n_bdy, boundary_update_s=bdy_one, n_aux=n_aux, aux_read_s=aux_one,
                n_restart=n_rst, restart_write_s=rst_w, restart_interval_min=rst_min,
                io_form_restart=io_rst, run_s=run_s, points=points,
                source="wtune_io_model (history/restart: wtune_run fallback anchors; "
                       "boundary/aux: climate-ea50 rsl)")


def _s(x):
    return "%.0fs" % x


def breakdown_line(compute_s, io, init_s):
    """'compute X + history Y + boundary Z + aux W + restart R + init I = TOTAL' (seconds)."""
    total = compute_s + io["total_s"] + init_s
    return ("compute %s + history %s + boundary %s + aux %s + restart %s + init %s = %s"
            % (_s(compute_s), _s(io["history_s"]), _s(io["boundary_s"]), _s(io["aux_s"]),
               _s(io["restart_s"]), _s(init_s), _s(total)))


def segment_note(init_s):
    return SEGMENT_NOTE % init_s
