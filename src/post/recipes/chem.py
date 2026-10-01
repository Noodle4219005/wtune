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
from post.extract import frame_path, list_frames

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

    path = frame_path(run_dir, domain, time_str)
    with netCDF4.Dataset(path) as ds:
        conc = np.asarray(ds.variables[species][0])
        if conc.ndim == 3:
            conc = conc[0]     # lowest model level
        u10 = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=0, units="m s-1")[0])
        v10 = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=0, units="m s-1")[1])
        lat = np.asarray(ds.variables["XLAT"][0])
        lon = np.asarray(ds.variables["XLONG"][0])
        units = getattr(ds.variables[species], "units", "")
        conc, units = concentration_units(ds, conc, units)
        from post.conventions import KT_PER_MS, ds_time_label
        save_map(out_path, lat, lon,
                "%s d%02d %s, lowest model level (%s, log scale) + 10 m wind (kt)\n%s"
                % (run_label, domain, species, units, ds_time_label(ds, time_str)),
                fill_field=np.where(conc > 0, conc, np.nan),
                fill_label="%s (%s)" % (species, units), fill_cmap="YlOrRd",
                fill_levels=log_levels(conc), fill_extend="both",
                u=u10, v=v10, barb_factor=KT_PER_MS, barb_label="wind barbs in kt", ds=ds)
    return out_path


def concentration_units(ds, conc, units):
    """Air-quality reporting units (decision 0064): gases ppmv -> ppb; aerosol mixing ratios
    ug/kg-dryair -> ug m-3 using the dry-air density 1/ALT of the lowest level."""
    import numpy as np
    u = (units or "").lower().replace(" ", "")
    if u.startswith("ppmv") or u == "ppm":
        return np.asarray(conc) * 1000.0, "ppb"
    if u.startswith("ug/kg") and "ALT" in ds.variables:
        alt = np.asarray(ds.variables["ALT"][0])
        alt = alt[0] if alt.ndim == 3 else alt
        return np.asarray(conc) / alt, "µg m$^{-3}$"
    return np.asarray(conc), units


def log_levels(field):
    """1-2-5 levels per decade spanning the field's 5th..max positive values (concentrations
    span orders of magnitude)."""
    import numpy as np
    a = np.asarray(field, dtype="f8")
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return [0.1, 0.2, 0.5, 1.0]
    lo, hi = np.percentile(a, 5), a.max()
    out = []
    for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if lo <= v * 2.5 and v <= hi * 2.5:
                out.append(v)
    return out if len(out) >= 2 else [lo, hi]


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
