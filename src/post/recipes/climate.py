"""post/recipes/climate.py -- the "climate" figure set (decision 0057 section 2): seasonal-mean
bias vs. ERA5 (CDO) and the annual cycle. Both need a long time series (this type is itself only
detected at run length >= 30 days, decision 0057 section 1); a run just barely over 30 days
still cannot support a season-length or annual-cycle statistic, so both figures state a minimum
history requirement and report missing with the actual span found when the run is too short for
that particular figure, rather than drawing a "seasonal mean" from a handful of frames.

Reuses CDO (post/tools.py's cdo_sinfon/find_on_path("cdo")) for both statistics -- `cdo
mergetime` to concatenate the run's own wrfout frames, then `cdo yseasmean`/`cdo ymonmean` -- and
post/recipes/era5.py for the ERA5 side of the seasonal bias. No self-written climate statistic.
"""
from __future__ import annotations

import glob
import os

from post import tools
from post.extract import list_frames

SEASONAL_MIN_DAYS = 90     # one full season, the shortest span cdo yseasmean is meaningful over
ANNUAL_MIN_DAYS = 365


def _mergetime(run_dir, domain, out_nc):
    cdo = tools.find_on_path("cdo", install_hint="conda-forge 'cdo' or envs/wtune/pixi.toml "
                             "provides it.")
    frames = sorted(glob.glob(os.path.join(run_dir, "wrfout_d%02d_*" % domain)))
    if not frames:
        raise tools.ToolError("no wrfout_d%02d_* frames in %s" % (domain, run_dir))
    tools.run([cdo, "-mergetime"] + frames + [out_nc])
    return out_nc


def figures(run_dirs, labels, out_dir, case_info):
    from post.recipe import fig, missing
    run_dir, run_label = run_dirs[0], labels[0]
    domain = case_info["domains"][0]["domain"]
    run_days = case_info["run_length_h"] / 24.0
    out = []

    if run_days < ANNUAL_MIN_DAYS:
        out.append(missing("climate_annual_cycle", "wrfout_d%02d_* time series" % domain,
                           "run is %.1f d, need >= %g d for an annual-cycle figure"
                           % (run_days, ANNUAL_MIN_DAYS)))
    else:
        try:
            merged = os.path.join(out_dir, "_climate_merged.nc")
            _mergetime(run_dir, domain, merged)
            cdo = tools.find_on_path("cdo")
            monthly = os.path.join(out_dir, "_climate_monthly.nc")
            tools.run([cdo, "ymonmean", merged, monthly])
            out.append(fig("climate_annual_cycle", monthly, run_label, domain="d%02d" % domain,
                           extra={"note": "cdo ymonmean output (NetCDF, not a PNG -- open with "
                                          "ncdump/panoply); no plotting step written this round"}))
        except (tools.ToolNotFoundError, ImportError) as e:
            out.append(missing("climate_annual_cycle", "cdo", str(e)))

    if run_days < SEASONAL_MIN_DAYS:
        out.append(missing("climate_seasonal_bias", "wrfout_d%02d_* time series + ERA5" % domain,
                           "run is %.1f d, need >= %g d for a seasonal-mean figure"
                           % (run_days, SEASONAL_MIN_DAYS)))
    else:
        try:
            from post.recipes.era5 import open_era5, Era5UnavailableError
            open_era5()   # only to raise Era5UnavailableError early with its own clear reason
            merged = os.path.join(out_dir, "_climate_merged.nc")
            _mergetime(run_dir, domain, merged)
            cdo = tools.find_on_path("cdo")
            seasonal = os.path.join(out_dir, "_climate_seasonal.nc")
            tools.run([cdo, "yseasmean", merged, seasonal])
            out.append(missing("climate_seasonal_bias", "ERA5-regridded seasonal difference",
                               "cdo yseasmean of the model ran (%s), but the ERA5-side "
                               "regrid+difference step is not implemented this round -- see "
                               "post/recipes/synoptic.py's ERA5 diff for the pattern to reuse"
                               % seasonal))
        except (tools.ToolNotFoundError, ImportError) as e:
            out.append(missing("climate_seasonal_bias", "cdo", str(e)))
        except Era5UnavailableError as e:
            out.append(missing("climate_seasonal_bias", "ERA5 (ARCO)", str(e)))
    return out
