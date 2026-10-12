"""post/ivt.py -- integrated vapor transport (IVT), decision 0057 update's approved
self-written exception: "wrf-python 與 MetPy 都沒有現成函式，是少數自己寫的積分（∫q·V dp／g）,
要在程式裡寫明理由並附測試". Neither wrf-python nor MetPy exposes a moisture-flux-integral
function (wrf-python has `pw` for the *scalar* precipitable-water integral only; MetPy has no
IVT at all as of this writing) -- everything else in post/recipes/moisture.py goes through
wrf-python/cartopy as usual. This module is the one place in the recipe set that integrates
by hand.

IVT = (1/g) integral over p from p_top to p_surface of q * V dp   (kg m^-1 s^-1)

g = 9.80665 m/s^2 (WMO standard gravity). Trapezoidal quadrature -- exact for any profile
linear in p, which makes it exact (not merely close) for a CONSTANT profile: this is exactly
what tests/test_post_ivt.py's analytic test checks (decision 0057 update T3).

Below-ground levels (2026-09-29 lead review fix): a pressure level that sits below the LOCAL
surface pressure -- e.g. 1000 hPa requested inside a hurricane's own centre, where psfc can be
well under 1000 hPa -- has no physical value there. wrf.interplevel() must be called with
`missing=numpy.nan` (its own default is a huge finite sentinel, ~9.97e36, NOT NaN -- see
post/recipes/moisture.py's _ivt_figure) so this module can actually detect it. The integral
below is then NaN-aware per level-pair segment, not a single numpy.trapz/trapezoid call: a
segment with either endpoint NaN contributes nothing, which is exactly integrating only the
above-ground part of the column, from its lowest valid level up to the top -- not from a
bogus zero (or sentinel) value at the surface. A column with no valid segment at all (e.g.
terrain above every requested level) returns NaN, not a false zero.
"""
from __future__ import annotations

G = 9.80665  # m s^-2, WMO standard gravity


def _nanaware_trapz(y, x, axis=0):
    """Trapezoidal integral of y over x along `axis`, skipping any segment whose either
    endpoint is NaN instead of letting it poison the whole reduction (numpy.trapz/trapezoid's
    behaviour) -- see the module docstring for why this matters for IVT specifically."""
    import numpy as np
    y = np.moveaxis(np.asarray(y, dtype=float), axis, 0)
    x = np.asarray(x, dtype=float)
    total = np.zeros(y.shape[1:], dtype=float)
    any_valid = np.zeros(y.shape[1:], dtype=bool)
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        valid = np.isfinite(y0) & np.isfinite(y1)
        total = total + np.where(valid, 0.5 * (y0 + y1) * (x[i + 1] - x[i]), 0.0)
        any_valid = any_valid | valid
    return np.where(any_valid, total, np.nan)


def ivt_components(q, u, v, pressure_pa):
    """q: specific humidity (kg/kg), u/v: earth-relative wind (m/s), pressure_pa: pressure
    (Pa) -- all 1-D, same length, any order (sorted here by pressure ascending, i.e. from the
    top of the profile down to the surface). Returns (ivt_u, ivt_v, ivt_magnitude) in
    kg m^-1 s^-1.
    """
    import numpy as np
    p = np.asarray(pressure_pa, dtype=float)
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if not (len(p) == len(q) == len(u) == len(v)):
        raise ValueError("ivt_components: q/u/v/pressure_pa must all have the same length")
    order = np.argsort(p)
    p, q, u, v = p[order], q[order], u[order], v[order]
    ivt_u = float(_nanaware_trapz(q * u, p) / G)
    ivt_v = float(_nanaware_trapz(q * v, p) / G)
    return ivt_u, ivt_v, (ivt_u ** 2 + ivt_v ** 2) ** 0.5


# standard pressure levels (hPa) the moisture recipe integrates over -- surface to 300 hPa,
# the usual IVT integration top (AMS/NOAA convention drops the stratospheric contribution,
# which is negligible for moisture transport).
IVT_LEVELS_HPA = (1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0)


def ivt_field(qvapor_levels, u_levels, v_levels, levels_hpa=IVT_LEVELS_HPA):
    """Gridded IVT: qvapor_levels/u_levels/v_levels are dicts {level_hpa: 2-D ndarray}
    (wrf.interplevel's own per-level output, one array per level -- see
    post/recipes/moisture.py). Returns (ivt_u, ivt_v, ivt_magnitude), each a 2-D ndarray,
    by applying ivt_components() at every grid cell via a vectorized trapezoidal sum along
    the level axis (equivalent to calling ivt_components() per pixel, just without the
    Python-level loop over every grid point)."""
    import numpy as np
    p_pa = np.array(sorted(float(lv) * 100.0 for lv in levels_hpa))
    order_levels = sorted(levels_hpa, key=float)
    q_stack = np.stack([np.asarray(qvapor_levels[lv]) for lv in order_levels], axis=0)
    u_stack = np.stack([np.asarray(u_levels[lv]) for lv in order_levels], axis=0)
    v_stack = np.stack([np.asarray(v_levels[lv]) for lv in order_levels], axis=0)
    ivt_u = _nanaware_trapz(q_stack * u_stack, p_pa, axis=0) / G
    ivt_v = _nanaware_trapz(q_stack * v_stack, p_pa, axis=0) / G
    return ivt_u, ivt_v, np.sqrt(ivt_u ** 2 + ivt_v ** 2)
