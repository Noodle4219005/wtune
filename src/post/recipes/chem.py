"""post/recipes/chem.py -- the "chem" figure set (decision 0057 section 2): a pollutant
concentration map with a wind-flow overlay, drawn straight from whichever WRF-Chem output
species is actually present in the run's own wrfout (`ncdump -h`, post/tools.py's own header
reader -- no guessing a mechanism from chem_opt's numeric value, which only says which chemical
mechanism ran, not which of its many species were kept in history output). Station time-series
against MET Point-Stat are only produced when station observations are given; none are wired
up this round (no chem-station downloader exists in this repo), so that figure is always
reported missing with that reason.
"""
from __future__ import annotations

import os

from post import tools
from post.extract import list_frames

# checked in this order -- common WRF-Chem history species across mechanisms/aerosol schemes.
CANDIDATE_SPECIES = ("PM2_5_DRY", "o3", "co", "PM10", "so2", "no2")


def _first_present_species(path):
    header = tools.parse_ncdump_header(tools.ncdump_header(path))
    for name in CANDIDATE_SPECIES:
        if name in header["variables"]:
            return name
    return None


def plot_concentration(run_dir, domain, time_str, species, out_path, run_label):
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")
    from post.recipes.mapfig import save_map

    path = os.path.join(run_dir, "wrfout_d%02d_%s" % (domain, time_str))
    with netCDF4.Dataset(path) as ds:
        conc = np.asarray(ds.variables[species][0])
        if conc.ndim == 3:
            conc = conc[0]     # lowest model level
        u10 = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=0, units="m s-1")[0])
        v10 = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=0, units="m s-1")[1])
        lat = np.asarray(ds.variables["XLAT"][0])
        lon = np.asarray(ds.variables["XLONG"][0])
        units = getattr(ds.variables[species], "units", "")
        save_map(out_path, lat, lon,
                "%s d%02d %s (%s) + 10 m wind  %s" % (run_label, domain, species, units, time_str),
                fill_field=conc, fill_label="%s (%s)" % (species, units), fill_cmap="YlOrRd",
                u=u10, v=v10, ds=ds)
    return out_path


def figures(run_dirs, labels, out_dir, case_info):
    from post.recipe import fig, missing
    run_dir, run_label = run_dirs[0], labels[0]
    chem_domains = [d for d in case_info["domains"] if (d["chem_opt"] or 0) > 0]
    domain = (max(chem_domains, key=lambda d: d["domain"])["domain"] if chem_domains
             else case_info["domains"][-1]["domain"])
    out = []
    frames = list_frames(run_dir, domain=domain)
    if not frames:
        return [missing("chem_conc_flow", "wrfout_d%02d_*" % domain, "no frames found")]
    species = _first_present_species(frames[-1][1])
    if species is None:
        out.append(missing("chem_conc_flow", "WRF-Chem species output",
                           "none of %s present in %s -- check io_form_history/auxhist "
                           "settings for this build's species" % (CANDIDATE_SPECIES, frames[-1][1])))
    else:
        try:
            path = plot_concentration(run_dir, domain, frames[-1][0], species,
                                      os.path.join(out_dir, "chem_conc_flow.png"), run_label)
            out.append(fig("chem_conc_flow", path, run_label, domain="d%02d" % domain,
                           valid_time=frames[-1][0], extra={"species": species}))
        except (tools.ToolNotFoundError, ImportError) as e:
            out.append(missing("chem_conc_flow", "wrf-python", str(e)))

    out.append(missing("chem_station", "air-quality station observations (e.g. EPA AQS)",
                       "no station-obs downloader implemented this round"))
    return out
