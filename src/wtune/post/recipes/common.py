"""post/recipes/common.py -- the "common" figure set (decision 0057 section 2), always
produced: a domain/nest configuration map with terrain (wrf-python `terrain` + cartopy, one
call per report even with several run dirs), and a reproducibility-noise envelope (diffwrf,
`wtune post compare`'s own engine) when two or more run dirs that differ only in decomposition
are given. Domain boundaries are drawn from each nest's own XLAT/XLONG grid edge (the actual
curved boundary on the parent's map projection, not a straight lon/lat rectangle) -- no
established tool draws a WRF nest-configuration figure, so this one is self-written glue over
wrf-python's terrain diagnostic and plain numpy edge extraction.
"""
from __future__ import annotations

import os

from wtune.post import tools
from wtune.post.extract import list_frames
from wtune.post import figdpi

# what one figure of a set may raise on a reduced history stream / an out-of-range point
# (cloud.py's _ERRORS = tool/import errors + these)
FIELD_ERRORS = (KeyError, ValueError, IndexError)


def error_reason(e):
    """missing() reason for one failed figure (same rule as cloud.py's _reason)."""
    from wtune.post.extract import is_missing_field_error, missing_reason
    return missing_reason(e) if is_missing_field_error(e) else "%s: %s" % (type(e).__name__, e)


def _bucket_mm(ds):
    """BUCKET_MM when the frame carries the I_RAINC/I_RAINNC bucket counters (namelist
    bucket_mm > 0), else 0.0 -- for the CDO route; netCDF4 readers use
    post.extract.accumulated_rain."""
    bucket = float(getattr(ds, "BUCKET_MM", -1.0))
    has = "I_RAINC" in ds.variables and "I_RAINNC" in ds.variables
    return bucket if bucket > 0 and has else 0.0


def _wanted_domains(map_domains, max_dom):
    """Domain numbers of post.map_domains ('all' or ['d01', 'd03']) that exist, d01 excluded
    (its terrain is always drawn)."""
    if not map_domains:
        return []
    if "all" in map_domains:
        return list(range(2, max_dom + 1))
    return sorted({int(d[1:]) for d in map_domains} - {1})


def plot_domain_map(run_dir, max_dom, out_dir, run_label, frame_of=None, map_domains=None):
    """The domain/nest configuration map (post/domain_map.py): every domain's box in the model
    projection, its own colour and 'dNN dx km nx x ny' label, d01's extent, d01 terrain shaded.
    frame_of(dom) -> (valid time, path) or None: the file each domain is drawn from, default its
    first wrfout frame (wtune post preview passes the wrfinput files). Returns the PNG path."""
    from wtune.post import domain_map
    frame_of = frame_of or (lambda dom: (list_frames(run_dir, domain=dom) or [None])[0])
    if not frame_of(1):
        return None
    time0 = frame_of(1)[0]
    paths = {d: frame_of(d)[1] for d in range(1, max_dom + 1) if frame_of(d)}
    crs, doms, terrain = domain_map.domains_from_files(paths)
    wanted = _wanted_domains(map_domains, max_dom)
    nests = domain_map.domain_terrain(paths, wanted)
    shaded = "d01" + "".join(",d%02d" % d for d in sorted(nests))
    return domain_map.draw(
        crs, doms, os.path.join(out_dir, "common_domain_map.png"),
        "%s: %d domain(s), terrain of %s, valid %s" % (run_label, len(doms), shaded, time0),
        terrain=terrain, extra_terrain=[nests[d] for d in sorted(nests)])


def plot_domain_zooms(run_dir, max_dom, out_dir, run_label, map_domains, overlay=None,
                      overlay_time=None):
    """One terrain map zoomed to each nest of post.map_domains (common_domain_map_dNN.png).
    overlay ['wind10']: 10 m wind barbs at overlay_time (WRF time string, else the last frame).
    Returns [(dom, path, valid time, note or None)]."""
    from wtune.post import domain_map
    paths = {d: (list_frames(run_dir, domain=d) or [None])[0] for d in range(1, max_dom + 1)}
    paths = {d: p[1] for d, p in paths.items() if p}
    crs, doms, _t = domain_map.domains_from_files(paths)
    by_dom = {d["dom"]: d for d in doms}
    out = []
    for dom in _wanted_domains(map_domains, max_dom):
        if dom not in by_dom:
            out.append((dom, None, None, "d%02d has no wrfout frames" % dom))
            continue
        terr = domain_map.domain_terrain(paths, [dom])
        time0, wind, why = list_frames(run_dir, domain=dom)[0][0], None, None
        if overlay and "wind10" in overlay:
            try:
                time0, wlat, wlon, wu, wv = domain_map.wind10_at(run_dir, dom, overlay_time)
                wind = (wlat, wlon, wu, wv)
            except (ValueError, KeyError, OSError) as e:
                why = "wind10 overlay left out on d%02d: %s" % (dom, e)
        inside = [d for d in doms if d["dom"] >= dom]
        path = domain_map.draw(
            crs, inside, os.path.join(out_dir, "common_domain_map_d%02d.png" % dom),
            "%s: d%02d terrain%s, valid %s" % (run_label, dom, " + 10 m wind" if wind else "",
                                                time0),
            terrain=terr.get(dom), zoom=by_dom[dom], wind=wind)
        out.append((dom, path, time0, why))
    return out


def plot_repro_envelope(run_dirs, labels, out_dir):
    """Max per-frame diffwrf RMSE across every field, for each (run_dirs[0], run_dirs[i])
    pair, plotted vs. time -- the "reproducibility-noise envelope" (decision 0057 section 2),
    drawn straight from post/compare.py's own diffwrf-backed comparator, no re-derivation.
    Returns the PNG path, or None if fewer than 2 run dirs were given."""
    if len(run_dirs) < 2:
        return None
    from wtune.post.compare import compare_runs
    plt = _plt()

    fig, ax = plt.subplots(figsize=(7, 4))
    any_data = False
    for i in range(1, len(run_dirs)):
        result = compare_runs(run_dirs[0], run_dirs[i])
        by_time = {}
        for r in result["results"]:
            by_time.setdefault(r["time"], []).append(r["rmse"])
        if not by_time:
            continue
        times = sorted(by_time)
        envelope = [max(by_time[t]) for t in times]
        ax.plot(range(len(times)), envelope, "-o", markersize=3,
               label="%s vs %s" % (labels[0], labels[i]))
        any_data = True
    if not any_data:
        plt.close(fig)
        return None
    ax.set_xlabel("frame index")
    ax.set_ylabel("max field RMSE (diffwrf)")
    ax.set_yscale("log")
    ax.set_title("reproducibility-noise envelope")
    ax.legend(fontsize=7)
    path = os.path.join(out_dir, "common_repro_envelope.png")
    fig.tight_layout()
    fig.savefig(path, dpi=figdpi.dpi())
    plt.close(fig)
    return path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figures(run_dirs, labels, out_dir, case_info, decomposition_only=False, map_domains=None,
            overlay=None, overlay_time=None):
    """Returns the list of common-type figure records (post.recipe's fig()/missing() shape).
    decomposition_only: True when the caller asserts the extra run dirs differ only in
    decomposition (e.g. --repro-pairs) -- the envelope is only meaningful under that
    condition, so it is otherwise skipped rather than drawn on runs that also differ
    physically."""
    from wtune.post.recipe import fig, missing, note

    out = []
    run_label = labels[0]
    try:
        _, domains = _domain_physics(run_dirs[0])
        max_dom = len(domains)
        path = plot_domain_map(run_dirs[0], max_dom, out_dir, run_label,
                               **(dict(map_domains=map_domains) if map_domains else {}))
        if path:
            out.append(fig("common_domain_map", path, run_label, domain="all",
                          valid_time=list_frames(run_dirs[0], domain=1)[0][0]))
            if map_domains:
                for dom, zpath, ztime, why in plot_domain_zooms(
                        run_dirs[0], max_dom, out_dir, run_label, map_domains, overlay,
                        overlay_time):
                    if why:
                        out.append(note("map_domains", why))
                    if zpath:
                        out.append(fig("common_domain_map_zoom", zpath, run_label,
                                       domain="d%02d" % dom, valid_time=ztime))
        else:
            out.append(missing("common_domain_map", "d01 wrfout", "no wrfout_d01_* frames found"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("common_domain_map", "wrf-python/cartopy", str(e)))

    if decomposition_only and len(run_dirs) >= 2:
        from wtune.post.compare import CompareError       # diffwrf failed: compare_runs fails closed
        try:
            path = plot_repro_envelope(run_dirs, labels, out_dir)
            if path:
                out.append(fig("common_repro_envelope", path, ",".join(labels)))
            else:
                out.append(missing("common_repro_envelope", "diffwrf",
                                   "no common (domain,time) frames differed -- runs identical "
                                   "or no overlapping frames"))
        except (tools.ToolNotFoundError, ImportError, CompareError) as e:
            out.append(missing("common_repro_envelope", "diffwrf", str(e)))
    return out


def _domain_physics(run_dir):
    from wtune.post import caseinfo
    return caseinfo.domain_physics(run_dir)
