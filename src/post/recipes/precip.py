"""post/recipes/precip.py -- the "precip" figure set (decision 0057 section 2): accumulated
precipitation (RAINC+RAINNC, `wtune post extract`'s own `rain_acc` field, drawn with
post/plotting.py's engine) is always produced (precip is always detected). The
observation-dependent figures -- model-vs-obs side-by-side (same color scale), the difference
map, ETS/frequency-bias vs. threshold, and MET Grid-Stat FSS -- all need Stage IV/MRMS/IMERG,
which this round does not fetch (task instruction: "Stage IV / IMERG / MRMS / station obs: not
fetched in this round"), so they are always reported as explicit missing-observation entries,
never silently dropped.
"""
from __future__ import annotations

import os

from post import tools
from post.extract import list_frames


def figures(run_dirs, labels, out_dir, case_info, spinup_hours=6):
    from post.recipe import fig, missing

    run_dir, run_label = run_dirs[0], labels[0]
    domain = case_info["domains"][-1]["domain"]   # finest domain, most informative rain field
    frames = list_frames(run_dir, domain=domain)
    out = []
    if not frames:
        out.append(missing("precip_accum", "wrfout_d%02d_*" % domain, "no frames found"))
    else:
        last_time = frames[-1][0]
        try:
            from post.plotting import plot_field
            written = plot_field(run_dir, "rain_acc", out_dir, domain=domain,
                                 times=[last_time], basemap="cartopy")
            out.append(fig("precip_accum", written[0], run_label, domain="d%02d" % domain,
                           valid_time=last_time,
                           extra={"note": "accumulated since run start (spin-up period "
                                          "included in the accumulation, per WRF's own "
                                          "RAINC/RAINNC convention -- spin-up excluded: "
                                          "%g h means excluded from any time-series analysis "
                                          "above, not from this cumulative total)" % spinup_hours}))
        except (tools.ToolNotFoundError, ImportError) as e:
            out.append(missing("precip_accum", "wrf-python", str(e)))

    stage4_reason = ("Stage IV / IMERG / MRMS not fetched this round -- no dated-archive "
                     "downloader implemented (task scope excludes it explicitly)")
    out.append(missing("precip_obs_compare", "Stage IV / MRMS / IMERG", stage4_reason))
    out.append(missing("precip_diff", "Stage IV / MRMS / IMERG",
                       "needs the model-vs-obs comparison above, which is missing"))
    out.append(missing("precip_ets_fbias", "MET Grid-Stat ETS/frequency bias vs. obs", stage4_reason))
    out.append(missing("precip_fss", "MET Grid-Stat FSS vs. obs", stage4_reason))
    return out
