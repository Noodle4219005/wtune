"""post/recipes/ensemble.py -- the "ensemble" figure set (decision 0057 section 2): a spaghetti
plot (one contour per member, wrf-python getvar + matplotlib), a probability-above-threshold
map (plain numpy fraction-of-members-exceeding, over the same wrf-python field each member
already computed), and MET Ensemble-Stat's rank histogram + spread-skill (skipped with a stated
reason when `ensemble_stat` is not installed -- same tools.find_postproc_tool contract as
post/met_tc.py's tc_pairs/tc_stat/tc_rmw).

Field/level/threshold default to SLP <= 1000 hPa contour (spaghetti) and precipitation
>= 25 mm (probability) -- reasonable, commonly-shown defaults for a tracked-system or
heavy-rain ensemble; a caller with a different variable of interest should draw it with
`wtune post plot`/`extract` per member instead, this figure set is the generic default view.
"""
from __future__ import annotations

import os

from post import tools
from post.extract import list_frames


def _member_field(run_dir, domain, time_str, field="slp"):
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")
    path = os.path.join(run_dir, "wrfout_d%02d_%s" % (domain, time_str))
    with netCDF4.Dataset(path) as ds:
        if field == "slp":
            var = wrf_mod.getvar(ds, "slp", timeidx=0, units="hPa")
        elif field == "rain_acc":
            var = (wrf_mod.getvar(ds, "RAINC", timeidx=0) + wrf_mod.getvar(ds, "RAINNC", timeidx=0))
        else:
            raise ValueError(field)
        lats, lons = wrf_mod.latlon_coords(var, as_np=True)
        return np.asarray(var), lats, lons


def plot_spaghetti(run_dirs, labels, domain, time_str, out_path, contour_level=1000.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        arr, lats, lons = _member_field(run_dir, domain, time_str, field="slp")
        color = plt.get_cmap("tab10")(idx % 10)
        ax.contour(lons, lats, arr, levels=[contour_level], colors=[color], linewidths=1.0)
        ax.plot([], [], color=color, label=label)   # proxy artist -- contour() has no legend label
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title("ensemble spaghetti: SLP=%g hPa contour  %s" % (contour_level, time_str))
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_probability(run_dirs, domain, time_str, out_path, threshold_mm=25.0):
    import numpy as np
    from post.recipes.mapfig import save_map

    stack, lats = None, None
    lons = None
    for run_dir in run_dirs:
        arr, lats, lons = _member_field(run_dir, domain, time_str, field="rain_acc")
        stack = arr[None, ...] if stack is None else np.concatenate([stack, arr[None, ...]], axis=0)
    prob = (stack >= threshold_mm).mean(axis=0) * 100.0
    save_map(out_path, lats, lons,
             "ensemble probability of accumulated rain >= %g mm  %s (%d members)"
             % (threshold_mm, time_str, stack.shape[0]),
             fill_field=prob, fill_label="probability (%)", fill_cmap="Blues")
    return out_path


def figures(run_dirs, labels, out_dir, case_info):
    from post.recipe import fig, missing
    if len(run_dirs) < 2:
        return [missing("ens_spaghetti", "ensemble members", "fewer than 2 run dirs given")]

    domain = case_info["domains"][-1]["domain"]
    common_times = None
    for run_dir in run_dirs:
        times = set(t for t, _ in list_frames(run_dir, domain=domain))
        common_times = times if common_times is None else (common_times & times)
    if not common_times:
        return [missing("ens_spaghetti", "wrfout_d%02d_*" % domain,
                        "no common valid time across all %d members" % len(run_dirs))]
    time_str = sorted(common_times)[-1]

    out = []
    try:
        path = plot_spaghetti(run_dirs, labels, domain, time_str,
                              os.path.join(out_dir, "ens_spaghetti.png"))
        out.append(fig("ens_spaghetti", path, ",".join(labels), domain="d%02d" % domain,
                       valid_time=time_str))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("ens_spaghetti", "wrf-python", str(e)))

    try:
        path = plot_probability(run_dirs, domain, time_str, os.path.join(out_dir, "ens_probability.png"))
        out.append(fig("ens_probability", path, ",".join(labels), domain="d%02d" % domain,
                       valid_time=time_str))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("ens_probability", "wrf-python", str(e)))

    try:
        tools.find_postproc_tool("ensemble_stat")
        out.append(missing("ens_rank_spread_skill", "MET ensemble_stat",
                           "ensemble_stat found but the obs-side wiring (rank histogram needs "
                           "an observation, spread-skill needs a verifying analysis) is not "
                           "implemented this round"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("ens_rank_spread_skill", "MET ensemble_stat", str(e)))
    return out
