"""post/timeseries.py -- `wtune post recipe --every 6h|--times T1,T2,...` (decision 0058
section 2 + the 2026-09-29 naming update): for each requested valid time, a full 2-D figure
set (post/plotting.py's own plot_field engine -- SLP, T2, 10 m wind speed, accumulated rain,
plus MCAPE when convection was detected) written to post/<id>/fNNN_<YYYYmmddHH>Z/ with
filenames <fig>_d<domain>_fNNN_<YYYYmmddHH>Z.png. This is additive to (not a replacement for)
the type-driven recipes in post/recipes/*, which keep picking their own representative times;
--every/--times only turns on this per-time layer (decision 0058 P3).

Naming rationale (decision 0058, "隨時間輸出的圖怎麼命名" update): forecast hour (fNNN, GFS-style,
zero-padded so lexical order is time order) plus valid time with no colon (ParaView's reader
fails on a colon in the filename, confirmed on scc 2026-09-29; Windows also rejects it).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

from post.extract import list_frames

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
VALID_TIME_FMT = "%Y%m%d%H"

_INTERVAL_RE = re.compile(r"^(\d+)\s*([hHdD])$")

# the "full figure set" this module draws at each selected valid time -- decision 0058
# section 2's own examples (SLP, T2, rain, wind); mcape is added only when convection was
# detected (post/recipe.py passes include_mcape accordingly).
TIME_SERIES_FIELDS = ("slp", "t2", "wspd10", "rain_acc")


def parse_every(text):
    """'6h' / '3H' / '1d' -> timedelta. Only hour/day granularity is accepted -- decision
    0058's own examples never use anything finer."""
    m = _INTERVAL_RE.match(text.strip())
    if not m:
        raise ValueError("--every expects e.g. '6h' or '1d', got %r" % text)
    n, unit = int(m.group(1)), m.group(2).lower()
    return timedelta(hours=n) if unit == "h" else timedelta(days=n)


def _nearest_time(all_times, wanted):
    wanted = wanted.strip()
    try:
        target = datetime.strptime(wanted, WRF_TIME_FMT)
    except ValueError:
        target = datetime.strptime(wanted, VALID_TIME_FMT)
    parsed = [datetime.strptime(t, WRF_TIME_FMT) for t in all_times]
    best = min(range(len(all_times)), key=lambda i: abs((parsed[i] - target).total_seconds()))
    return all_times[best]


def select_times(all_times, every=None, times=None):
    """all_times: sorted WRF time strings (WRF_TIME_FMT) available on disk for the run's own
    reference domain. `times` (explicit WRF_TIME_FMT or 'YYYYmmddHH' strings, snapped to the
    nearest available frame) takes priority over `every` (a parse_every() timedelta or its
    string form -- every frame whose offset from all_times[0] is a multiple of it, within
    1 second, since WRF output steps rarely land on an exact multiple in float seconds
    otherwise). Returns the ordered subset of all_times actually used."""
    if not all_times:
        return []
    if times:
        wanted = [_nearest_time(all_times, t) for t in times]
        return sorted(set(wanted), key=all_times.index)
    if every:
        step = parse_every(every) if isinstance(every, str) else every
        t0 = datetime.strptime(all_times[0], WRF_TIME_FMT)
        out = []
        for t in all_times:
            offset = (datetime.strptime(t, WRF_TIME_FMT) - t0).total_seconds()
            if abs(offset % step.total_seconds()) < 1.0:
                out.append(t)
        return out
    return list(all_times)


def forecast_dir_name(t0, valid_time):
    """fNNN_<YYYYmmddHH>Z -- forecast-hour tag (fNNN) plus its valid time. NNN is the offset
    from t0 in whole hours, zero-padded to 3 digits (matches numerical-product convention,
    e.g. GFS f006 -- lexical order is then also time order)."""
    dt = datetime.strptime(valid_time, WRF_TIME_FMT)
    fhour = int(round((dt - t0).total_seconds() / 3600.0))
    return "f%03d_%sZ" % (fhour, dt.strftime(VALID_TIME_FMT))


def figures_for_time_series(run_dir, run_label, domain, out_dir, every=None, times=None,
                            include_mcape=False):
    """Writes one sub-directory per selected valid time, each a full figure set (decision
    0058 P3). Returns post/recipe.py-shaped fig()/missing() records with `valid_time` set on
    every entry and a `forecast_tag` extra, so post/report.py can group and order them for
    time-by-time navigation."""
    from post.recipe import fig, missing
    from post.plotting import plot_field
    from post import tools

    all_times = [t for t, _ in list_frames(run_dir, domain=domain)]
    selected = select_times(all_times, every=every, times=times)
    if not selected:
        return [missing("timeseries", "wrfout_d%02d_*" % domain,
                        "no frames matched --every/--times")]
    t0 = datetime.strptime(all_times[0], WRF_TIME_FMT)

    fields = list(TIME_SERIES_FIELDS)
    if include_mcape:
        fields.append("mcape")

    out = []
    for valid_time in selected:
        tag = forecast_dir_name(t0, valid_time)
        time_dir = os.path.join(out_dir, tag)
        os.makedirs(time_dir, exist_ok=True)
        for field in fields:
            fig_id = "timeseries_%s" % field
            try:
                written = plot_field(run_dir, field, time_dir, domain=domain,
                                     times=[valid_time], basemap="cartopy")
            except (tools.ToolNotFoundError, ImportError) as e:
                out.append(missing(fig_id, "wrf-python/cartopy", str(e), run=run_label))
                continue
            if not written:
                out.append(missing(fig_id, field, "no frame at %s" % valid_time, run=run_label))
                continue
            final_name = "%s_d%02d_%s.png" % (field, domain, tag)
            final_path = os.path.join(time_dir, final_name)
            if os.path.abspath(written[0]) != os.path.abspath(final_path):
                os.replace(written[0], final_path)
            out.append(fig(fig_id, final_path, run_label, domain="d%02d" % domain,
                          valid_time=valid_time, extra={"forecast_tag": tag}))
    return out
