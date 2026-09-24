"""Analytic time-integrated solar/terminator load map for a WRF case.

The DOMINANT, verified WRF load-imbalance driver is the day/night terminator
(solar radiation; corr(compute, E-W) ~ -0.60). Unlike convection, the terminator
moves DETERMINISTICALLY (~15 deg/hour), so its full trajectory over the run is a
closed-form function of solar geometry -- computable exactly, for free, with NO
model. This module time-integrates the solar zenith angle across a coarse lat/lon
mesh over [start, start+run_hours] to produce a run-averaged radiation load map
and an E-W imbalance estimate.

It answers the "t=0 snapshot is wrong because weather moves fast" objection for
the fastest, largest driver: we integrate over time, not snapshot. The CONVECTIVE
(microphysics) hotspot, which moves chaotically, is left to the surrogate layer.

stdlib only. Python 3.6 compatible.
"""
import math

# cumulative days before month 1..12 (non-leap; leap adds <=1 day, negligible)
_CUM_DAYS = (0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def day_of_year(month, day):
    try:
        m = int(month)
        d = int(day)
    except (TypeError, ValueError):
        return None
    if m < 1 or m > 12:
        return None
    return _CUM_DAYS[m] + d


def declination_deg(doy):
    """Solar declination (deg), ~1.5 deg accuracy -- ample for an estimate."""
    return 23.44 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))


def cos_zenith(lat_deg, lon_deg, utc_hours, decl_deg):
    """cos(solar zenith). >0 means sun above horizon (daytime)."""
    lat = math.radians(lat_deg)
    dec = math.radians(decl_deg)
    # local solar time (hours); lon East positive
    lst = utc_hours + lon_deg / 15.0
    hour_angle = math.radians(15.0 * (lst - 12.0))
    return (math.sin(lat) * math.sin(dec)
            + math.cos(lat) * math.cos(dec) * math.cos(hour_angle))


def domain_bbox(case):
    """Approximate lat/lon bounding box from center + grid extent.

    Flat-earth approximation (ignores map-projection convergence): fine for a
    hotspot ESTIMATE. Returns (lat0, lat1, lon0, lon1) or None if center unknown.
    """
    lat = case.get('latitude')
    lon = case.get('longitude')
    if lat is None or lon is None:
        return None
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None
    nx = case.get('nx') or (int(case.get('e_we', 100)) - 1)
    ny = case.get('ny') or (int(case.get('e_sn', 100)) - 1)
    dx = float(case.get('dx') or 3000.0)
    dy = float(case.get('dy') or 3000.0)
    half_h_deg = (ny * dy) / 111320.0 / 2.0
    coslat = max(0.2, math.cos(math.radians(lat)))
    half_w_deg = (nx * dx) / (111320.0 * coslat) / 2.0
    return (lat - half_h_deg, lat + half_h_deg, lon - half_w_deg, lon + half_w_deg)


def solar_load_map(case, n_time=13, n_lon=24, n_lat=18):
    """Time-integrated day-fraction + mean-insolation map over the run.

    Returns a dict with the per-column day_fraction grid, E-W gradient, sweep
    classification, and an imbalance estimate -- or {'ok': False, 'reason': ...}.
    """
    bbox = domain_bbox(case)
    if bbox is None:
        return {'ok': False, 'reason': 'domain center lat/lon unknown (need ref_lat/cen_lat, cen_lon)'}
    lat0, lat1, lon0, lon1 = bbox

    doy = day_of_year(case.get('start_month'), case.get('start_day'))
    if doy is None:
        return {'ok': False, 'reason': 'start date unknown'}

    start_hr = float(case.get('start_hour') or 0)
    run_hrs = float(case.get('run_total_hours') or 0)
    if run_hrs <= 0:
        run_hrs = 1.0

    # time samples across the run (declination ~constant over hours -> once)
    if n_time < 2:
        times = [start_hr + run_hrs / 2.0]
    else:
        times = [start_hr + run_hrs * i / (n_time - 1) for i in range(n_time)]
    decl = declination_deg(doy)

    def frange(a, b, n):
        if n <= 1:
            return [(a + b) / 2.0]
        return [a + (b - a) * i / (n - 1) for i in range(n)]

    lats = frange(lat0, lat1, n_lat)
    lons = frange(lon0, lon1, n_lon)

    # per-column (indexed by lon then lat) day fraction and mean insolation
    day_frac = []          # length n_lon, each a list over lat
    insol = []
    col_lon_dayfrac = []   # (lon, mean day_frac over lat) for E-W gradient
    for lo in lons:
        col_df = []
        col_in = []
        for la in lats:
            days = 0
            ins = 0.0
            for t in times:
                cz = cos_zenith(la, lo, t, decl)
                if cz > 0:
                    days += 1
                    ins += cz
            col_df.append(days / float(len(times)))
            col_in.append(ins / float(len(times)))
        day_frac.append(col_df)
        insol.append(col_in)
        col_lon_dayfrac.append((lo, sum(col_df) / len(col_df)))

    flat_df = [v for col in day_frac for v in col]
    mean_df = sum(flat_df) / len(flat_df)
    min_df = min(flat_df)
    max_df = max(flat_df)

    # E-W gradient: Pearson corr between longitude and column-mean day_fraction
    ew_corr = _pearson([c[0] for c in col_lon_dayfrac],
                       [c[1] for c in col_lon_dayfrac])

    # sweep classification
    if max_df - min_df < 0.15:
        if mean_df > 0.8:
            sweep = 'uniform-day (whole domain sunlit most of run; low radiation imbalance)'
        elif mean_df < 0.2:
            sweep = 'uniform-night (no SW; radiation imbalance minimal, LW only)'
        else:
            sweep = 'uniform-partial (terminator outside domain; modest imbalance)'
    else:
        sweep = 'terminator-crossing (day/night split sweeps the domain; STRONG radiation E-W imbalance)'

    # crude imbalance estimate: spread of day_fraction scales the SW-driven part.
    # SW radiation is a minority of step cost; use the measured anchor that a full
    # day/night split corresponds to ~the observed radiation imbalance envelope.
    df_spread = max_df - min_df
    imbalance_pct = round(df_spread * 9.0, 1)   # scaled to the ~9% halo_wait envelope

    return {
        'ok': True,
        'bbox': {'lat0': round(lat0, 2), 'lat1': round(lat1, 2),
                 'lon0': round(lon0, 2), 'lon1': round(lon1, 2)},
        'declination_deg': round(decl, 2),
        'day_of_year': doy,
        'mean_day_fraction': round(mean_df, 3),
        'min_day_fraction': round(min_df, 3),
        'max_day_fraction': round(max_df, 3),
        'ew_corr_lon_dayfrac': round(ew_corr, 3) if ew_corr is not None else None,
        'sweep': sweep,
        'imbalance_estimate_pct': imbalance_pct,
        'day_frac_grid': day_frac,
        'lons': [round(x, 2) for x in lons],
        'lats': [round(x, 2) for x in lats],
        'n_time': len(times),
    }


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def solar_summary(case):
    """Human-readable diagnosis lines for the terminator load map."""
    m = solar_load_map(case)
    if not m.get('ok'):
        return ["  solar/terminator: cannot compute -- {}".format(m.get('reason'))]
    lines = []
    lines.append("  terminator load map (time-integrated over run, {} time samples):".format(m['n_time']))
    lines.append("    domain ~[{lat0}..{lat1}]N [{lon0}..{lon1}]E, declination {d} deg (doy {doy})".format(
        lat0=m['bbox']['lat0'], lat1=m['bbox']['lat1'], lon0=m['bbox']['lon0'],
        lon1=m['bbox']['lon1'], d=m['declination_deg'], doy=m['day_of_year']))
    lines.append("    day-fraction: mean {mean}, range [{lo}..{hi}]  E-W corr(lon,dayfrac)={c}".format(
        mean=m['mean_day_fraction'], lo=m['min_day_fraction'], hi=m['max_day_fraction'],
        c=m['ew_corr_lon_dayfrac']))
    lines.append("    regime: {}".format(m['sweep']))
    lines.append("    radiation imbalance estimate ~{}% of step (dominant driver; decomposition cannot fix)".format(
        m['imbalance_estimate_pct']))
    lines.append("    NOTE: convective (microphysics) hotspot is NOT included -- needs the surrogate layer.")
    return lines


if __name__ == '__main__':
    # tiny sanity demo: CONUS-ish winter case
    demo = {'latitude': 39.0, 'longitude': -97.0, 'nx': 1500, 'ny': 1200,
            'dx': 2500.0, 'dy': 2500.0, 'start_month': 11, 'start_day': 26,
            'start_hour': 18, 'run_total_hours': 6}
    for ln in solar_summary(demo):
        print(ln)
