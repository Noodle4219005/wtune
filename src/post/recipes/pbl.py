"""post/recipes/pbl.py -- the "pbl" figure set (decision 0057 section 2): a cross-terrain
vertical cross-section of potential temperature + wind (wrf-python `wrf.vertcross`, the NCAR
tool for exactly this diagnostic), and a single-point time-height plot of theta + PBLH.
Every field comes from wrf-python getvar (theta, ua, va, PBLH, ter) -- no self-written
diagnostic. Not exercised against real LES/sub-km data in this round (m1, the only case data
copied locally, is bl_pbl_physics=1 at dx>=4 km on every domain and does not trigger pbl at
all -- see tests/test_post_caseinfo.py); same "not yet verified against a real case" caveat
post/met_tc.py's MET templates already carry, disclosed here for the same reason.
"""
from __future__ import annotations

import os

from post import tools
from post.extract import list_frames


def _cross_section_endpoints(ds):
    """A west-east line through the domain's own centre row -- a simple, always-available
    default cross-section (no case-specific terrain feature to target automatically); --box/
    --points on `wtune post extract` remain the way to pick a specific line by hand."""
    import numpy as np
    lat = np.asarray(ds.variables["XLAT"][0])
    lon = np.asarray(ds.variables["XLONG"][0])
    j_mid = lat.shape[0] // 2
    return (float(lat[j_mid, 0]), float(lon[j_mid, 0])), (float(lat[j_mid, -1]), float(lon[j_mid, -1]))


def plot_cross_section(run_dir, domain, time_str, out_path, run_label):
    import netCDF4
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wrf_mod = __import__("wrf")

    path = os.path.join(run_dir, "wrfout_d%02d_%s" % (domain, time_str))
    with netCDF4.Dataset(path) as ds:
        z = wrf_mod.getvar(ds, "z", timeidx=0, units="m")
        theta = wrf_mod.getvar(ds, "theta", timeidx=0)
        ter = wrf_mod.getvar(ds, "ter", timeidx=0)
        start, end = _cross_section_endpoints(ds)
        start_pt = wrf_mod.CoordPair(lat=start[0], lon=start[1])
        end_pt = wrf_mod.CoordPair(lat=end[0], lon=end[1])
        theta_cross = wrf_mod.vertcross(theta, z, wrfin=ds, start_point=start_pt,
                                        end_point=end_pt, latlon=True)
        ter_line = wrf_mod.interpline(ter, wrfin=ds, start_point=start_pt, end_point=end_pt,
                                      latlon=True)

    xs = np.arange(theta_cross.shape[-1])
    ys = np.asarray(theta_cross.coords["vertical"])
    fig, ax = plt.subplots(figsize=(8, 4))
    cf = ax.contourf(xs, ys, np.asarray(theta_cross), levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="theta (K)")
    ax.fill_between(xs, 0, np.asarray(ter_line), color="0.3")
    ax.set_xlabel("cross-section point index (W->E through domain centre)")
    ax.set_ylabel("height (m)")
    ax.set_title("%s d%02d terrain cross-section: theta  %s" % (run_label, domain, time_str))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_time_height(run_dir, domain, out_dir, run_label):
    import netCDF4
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wrf_mod = __import__("wrf")

    frames = list_frames(run_dir, domain=domain)
    if len(frames) < 2:
        return None
    j_mid = i_mid = None
    theta_by_time, pblh_by_time, z_ref = [], [], None
    for time_str, path in frames:
        with netCDF4.Dataset(path) as ds:
            if j_mid is None:
                j_mid, i_mid = ds.dimensions["south_north"].size // 2, ds.dimensions["west_east"].size // 2
            z = np.asarray(wrf_mod.getvar(ds, "z", timeidx=0, units="m"))[:, j_mid, i_mid]
            theta = np.asarray(wrf_mod.getvar(ds, "theta", timeidx=0))[:, j_mid, i_mid]
            pblh = float(np.asarray(ds.variables["PBLH"][0])[j_mid, i_mid])
        if z_ref is None:
            z_ref = z
        theta_by_time.append(theta)
        pblh_by_time.append(pblh)
    theta_grid = np.array(theta_by_time).T
    fig, ax = plt.subplots(figsize=(8, 4))
    cf = ax.contourf(range(len(frames)), z_ref, theta_grid, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="theta (K)")
    ax.plot(range(len(frames)), pblh_by_time, "r-o", markersize=3, label="PBLH")
    ax.set_xlabel("frame index")
    ax.set_ylabel("height (m)")
    ax.set_title("%s d%02d time-height: theta + PBLH (domain-centre column)" % (run_label, domain))
    ax.legend(fontsize=7)
    path = os.path.join(out_dir, "pbl_time_height.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def figures(run_dirs, labels, out_dir, case_info):
    from post.recipe import fig, missing
    run_dir, run_label = run_dirs[0], labels[0]
    pbl_domains = [d for d in case_info["domains"]
                   if (d["bl_pbl_physics"] or 0) == 0 or d["dx_km"] <= 1.0]
    domain = (max(pbl_domains, key=lambda d: d["domain"])["domain"] if pbl_domains
             else case_info["domains"][-1]["domain"])
    out = []
    frames = list_frames(run_dir, domain=domain)
    if not frames:
        return [missing("pbl_cross_terrain", "wrfout_d%02d_*" % domain, "no frames found")]
    try:
        path = plot_cross_section(run_dir, domain, frames[-1][0],
                                  os.path.join(out_dir, "pbl_cross_terrain.png"), run_label)
        out.append(fig("pbl_cross_terrain", path, run_label, domain="d%02d" % domain,
                       valid_time=frames[-1][0]))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("pbl_cross_terrain", "wrf-python", str(e)))

    try:
        path = plot_time_height(run_dir, domain, out_dir, run_label)
        if path:
            out.append(fig("pbl_time_height", path, run_label, domain="d%02d" % domain,
                           valid_time="%s..%s" % (frames[0][0], frames[-1][0])))
        else:
            out.append(missing("pbl_time_height", "wrfout_d%02d_*" % domain,
                               "fewer than 2 frames available"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("pbl_time_height", "wrf-python", str(e)))
    return out
