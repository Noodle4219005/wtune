"""post/recipes/cf3d.py -- the always-on CF pressure-level data product of `wtune post recipe`
(post/cf_pressure.py): Met.3D/ParaView-ready NetCDF files, listed in the report's "common"
section as data-file entries (path + size), not figures.

Which domains: the finest domain (the one the 3-D and time-series products already focus on)
plus d01 when there is a nest -- d01 gives the synoptic context the nest lacks; intermediate
nests add files without adding a view either of those two does not already give.
Which times: the --every/--times selection (same post/timeseries.select_times as the time-
series figures), or only the last frame when neither is given -- a whole run at every output
step is a data-volume decision the user should make explicitly with --every.

wrf-python/pyproj/scipy missing is a missing-entry record, never a failed report (same contract
as post/recipes/paraview3d.py's missing pvbatch).
"""
from __future__ import annotations

CATALOG_ID = "common_cf3d"


def cf_domains(case_info):
    """[1] for a single-domain run, [1, finest] when there is a nest."""
    finest = case_info["domains"][-1]["domain"] if case_info.get("domains") else 1
    return sorted({1, int(finest)})


def figures(run_dir, run_label, out_dir, case_info, bdy_width, every=None, times=None,
            command=None):
    from post.recipe import fig, missing
    try:
        from post import cf_pressure
        import wrf  # noqa: F401 -- fail here, once, rather than per frame
        import pyproj  # noqa: F401
        import scipy.ndimage  # noqa: F401
    except ImportError as e:
        return [missing(CATALOG_ID, "wrf-python/pyproj/scipy", str(e), run=run_label)]

    out = []
    for r in cf_pressure.export(run_dir, out_dir, cf_domains(case_info), bdy_width,
                                every=every, times=times, command=command):
        if r.get("error"):
            out.append(missing(CATALOG_ID, "wrfout d%02d %s" % (r["domain"], r["valid_time"] or ""),
                               r["error"], run=run_label))
            continue
        out.append(fig(CATALOG_ID, r["path"], run_label, domain="d%02d" % r["domain"],
                       valid_time=r["valid_time"],
                       extra=dict(data_file=True, size_bytes=r["size_bytes"],
                                  note="%d levels %g-%g hPa, %dx%d lat-lon at %g deg%s"
                                  % (len(r["levels"]), r["levels"][0], r["levels"][-1],
                                     r["nlat"], r["nlon"], r["dlon"],
                                     "" if r["has_qice"] else "; no QICE (CLOUD_TOTAL=QCLOUD)"))))
    return out
