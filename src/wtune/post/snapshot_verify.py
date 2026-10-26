"""post/snapshot_verify.py -- MET Grid-Stat of a model field against gridded observations valid
at single times (agent-fetched satellite IR; post/agent_obs.py), pooled over the times. Both
fields go onto the model's own lat/lon grid (to_grid FCST), like the radar comparison in
post/recipes/convection.py whose pooling it reuses."""
from __future__ import annotations

import os


def verify(picks, model_field, obs_file, names, thresholds, work, poly, widths=(1, 3, 5, 9),
           model_method="remapbil", obtype="OBS"):
    """picks: [(valid datetime, (wrfout path, timeidx), obs path)]. model_field(path, timeidx) ->
    (lat 2-D, lon 2-D, field 2-D, units). obs_file(obs path, valid, griddes, out_nc) -> out_nc
    on that grid. names: (model var, obs var). Returns dict(per_ctc, per_nbr, cts_rows, fields)
    with fields [(valid, glat, glon, model, obs)] of the matched pairs per time."""
    import numpy as np
    from wtune.post import met_gridstat
    out = dict(per_ctc=[], per_nbr=[], cts_rows=[], fields=[])
    for t, (path, ti), obs_path in picks:
        tag = t.strftime("%Y%m%d%H%M")
        lat, lon, arr, units = model_field(path, ti)
        fnc = met_gridstat.curvilinear_to_latlon(lat, lon, {names[0]: (arr, units, names[0])}, t,
                                                 os.path.join(work, "model_%s.nc" % tag),
                                                 method=model_method)
        onc = obs_file(obs_path, t, fnc[:-3] + ".griddes", os.path.join(work, "obs_%s.nc" % tag))
        res = met_gridstat.run_verification(fnc, onc, poly, os.path.join(work, "gs_%s" % tag),
                                            names[0], names[1], thresholds, widths=widths,
                                            to_grid="FCST", regrid_method="NEAREST",
                                            model="WRF", obtype=obtype)
        glat, glon, gf, go, _a, _b = met_gridstat.read_pairs_fields(res["pairs_nc"])
        if not np.any(np.isfinite(gf) & np.isfinite(go)):
            continue
        out["per_ctc"].append(met_gridstat.parse_ctc_counts(res["ctc"]))
        out["per_nbr"].append(met_gridstat.parse_nbrcnt_table(res["nbrcnt"]))
        out["cts_rows"] += [(t, r) for r in met_gridstat.parse_cts_table(res["cts"])]
        out["fields"].append((t, glat, glon, gf, go))
    return out
