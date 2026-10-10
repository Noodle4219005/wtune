"""Physics knowledge base + advisor for a WRF config autotuner.

Physics SCHEME choice is NOT a free performance knob -- changing it changes the
weather/results and fails the output-tolerance gate. This module
(a) RESOLVES physics_suite sentinels, (b) DIAGNOSES the cost/imbalance of the
CHOSEN physics, (c) ADVISES results-preserving/gated numerics only (radt,
time_step). It NEVER recommends swapping a scheme for speed.
"""

import sys

from wtune import solar as _wsolar

_MP = {
    1: {'name': 'Kessler', 'cost': 'low', 'vec': '?', 'note': 'warm-rain only'},
    2: {'name': 'Lin', 'cost': 'med', 'vec': '?', 'note': '6-species'},
    3: {'name': 'WSM3', 'cost': 'low', 'vec': '?', 'note': 'simple ice'},
    4: {'name': 'WSM5', 'cost': 'med', 'vec': '?', 'note': ''},
    6: {'name': 'WSM6', 'cost': 'med', 'vec': '?', 'note': '6-species single-moment; cheaper than Thompson'},
    8: {'name': 'Thompson', 'cost': 'high', 'vec': 'partial', 'note': 'lookup-table heavy, 6 species + 2 number-conc, branchy; the M-group hotspot; per-cell cost varies with hydrometeor presence -> storm-blob imbalance'},
    10: {'name': 'Morrison', 'cost': 'high', 'vec': '?', 'note': '2-moment, expensive'},
    16: {'name': 'WDM6', 'cost': 'high', 'vec': '?', 'note': 'double-moment warm rain'},
    18: {'name': 'NSSL', 'cost': 'high', 'vec': '?', 'note': ''},
    28: {'name': 'Thompson aerosol-aware', 'cost': 'high', 'vec': '?', 'note': 'heavier than 8'},
    38: {'name': 'Thompson-GH', 'cost': 'high', 'vec': '?', 'note': ''},
    40: {'name': 'Morrison-aero', 'cost': 'high', 'vec': '?', 'note': ''},
}

_RA_LW = {
    1: {'name': 'RRTM', 'cost': 'med', 'vec': '?', 'note': 'older band model'},
    3: {'name': 'CAM', 'cost': 'med', 'vec': '?', 'note': ''},
    4: {'name': 'RRTMG', 'cost': 'med', 'vec': 'scalar', 'note': 'called one column at a time (ncol=1) -> NO vector dim; but LW solver is NOT the hotspot: rtrnmc ~1.01% of step, ~73% of rad_lw is outside rrtmg_lw; day/night E-W imbalance driver'},
    5: {'name': 'Goddard', 'cost': 'med', 'vec': '?', 'note': ''},
    24: {'name': 'RRTMG-fast', 'cost': 'med', 'vec': '?', 'note': ''},
}

_RA_SW = {
    1: {'name': 'Dudhia', 'cost': 'low', 'vec': '?', 'note': ''},
    2: {'name': 'Goddard', 'cost': 'med', 'vec': '?', 'note': ''},
    4: {'name': 'RRTMG', 'cost': 'med', 'vec': 'scalar', 'note': 'vrtqdr_sw scalar loop; paired with lw; day/night imbalance'},
    5: {'name': 'Goddard-new', 'cost': 'med', 'vec': '?', 'note': ''},
}

_BL = {
    1: {'name': 'YSU', 'cost': 'low', 'vec': '?', 'note': 'non-local K'},
    2: {'name': 'MYJ', 'cost': 'low', 'vec': '?', 'note': 'local TKE; CONUS suite'},
    4: {'name': 'QNSE', 'cost': 'med', 'vec': '?', 'note': ''},
    5: {'name': 'MYNN2', 'cost': 'med', 'vec': '?', 'note': ''},
    6: {'name': 'MYNN3', 'cost': 'med', 'vec': '?', 'note': ''},
    7: {'name': 'ACM2', 'cost': 'low', 'vec': '?', 'note': ''},
}

_CU = {
    0: {'name': 'off', 'cost': 'none', 'vec': '?', 'note': 'convection-resolving at dx<=4km; CONUS 2.5km uses 0'},
    1: {'name': 'KF', 'cost': 'med', 'vec': '?', 'note': ''},
    2: {'name': 'BMJ', 'cost': 'low', 'vec': '?', 'note': ''},
    3: {'name': 'GF', 'cost': 'med', 'vec': '?', 'note': ''},
    4: {'name': 'scale-aware SAS', 'cost': 'med', 'vec': '?',
        'note': 'unmeasured: scale-aware mass-flux closure, assumed comparable to GF (opt 3) scale-awareness overhead'},
    5: {'name': 'Grell-3D', 'cost': 'med', 'vec': '?',
        'note': 'unmeasured: same Grell ensemble mass-flux lineage as GF (opt 3), which is a scale-aware '
                'evolution of this scheme; assumed comparable cost pending measurement'},
    6: {'name': 'Tiedtke', 'cost': 'med', 'vec': '?', 'note': ''},
    7: {'name': 'CAM Zhang-McFarlane', 'cost': 'med', 'vec': '?',
        'note': 'unmeasured: mass-flux deep-convection scheme, structurally comparable to KF/Tiedtke (opts 1, 6)'},
    10: {'name': 'KF-CuP', 'cost': 'med', 'vec': '?',
         'note': 'unmeasured: KF variant with added subgrid cloud-aerosol coupling; assumed >= KF (opt 1) cost'},
    11: {'name': 'multiscale KF', 'cost': 'med', 'vec': '?', 'note': ''},
    14: {'name': 'KIAPS SAS', 'cost': 'med', 'vec': '?',
         'note': 'unmeasured: scale-aware SAS variant, assumed comparable to scale-aware SAS (opt 4)'},
    16: {'name': 'New Tiedtke', 'cost': 'med', 'vec': '?', 'note': ''},
    93: {'name': 'Grell-Devenyi', 'cost': 'high', 'vec': '?',
         'note': 'unmeasured: ensemble multi-closure mass-flux scheme (averages many closures per column), '
                 'structurally heavier than the single-closure schemes in this table'},
    94: {'name': 'SAS', 'cost': 'low', 'vec': '?',
         'note': 'unmeasured: single-closure simplified Arakawa-Schubert, assumed comparable to BMJ (opt 2) simplicity'},
    95: {'name': 'old SAS', 'cost': 'low', 'vec': '?',
         'note': 'unmeasured: predecessor of SAS (opt 94), assumed same low-cost single-closure class'},
    96: {'name': 'New SAS', 'cost': 'low', 'vec': '?',
         'note': 'unmeasured: revised SAS closure, assumed same low-cost single-closure class as SAS (opt 94)'},
    99: {'name': 'Kain-Fritsch (original)', 'cost': 'med', 'vec': '?',
         'note': 'unmeasured: original KF formulation, assumed comparable to KF (opt 1)'},
}

_SF_SURFACE = {
    1: {'name': '5-layer thermal diffusion', 'cost': 'low', 'vec': '?', 'note': ''},
    2: {'name': 'Noah', 'cost': 'med', 'vec': '?', 'note': 'standard LSM; CONUS suite'},
    3: {'name': 'RUC', 'cost': 'med', 'vec': '?', 'note': ''},
    4: {'name': 'Noah-MP', 'cost': 'high', 'vec': '?', 'note': 'heavier, more soil/veg processes'},
    7: {'name': 'Pleim-Xiu', 'cost': 'med', 'vec': '?', 'note': ''},
    8: {'name': 'SSiB', 'cost': 'high', 'vec': '?', 'note': ''},
}

_SF_SFCLAY = {
    1: {'name': 'MM5 M-O', 'cost': 'low', 'vec': '?', 'note': ''},
    2: {'name': 'MYJ', 'cost': 'low', 'vec': '?', 'note': ''},
    4: {'name': 'QNSE', 'cost': 'low', 'vec': '?', 'note': ''},
    5: {'name': 'MYNN', 'cost': 'low', 'vec': '?', 'note': ''},
    7: {'name': 'Pleim', 'cost': 'low', 'vec': '?', 'note': ''},
    11: {'name': 'revised MM5', 'cost': 'low', 'vec': '?', 'note': ''},
}

_KB = {
    'mp_physics': _MP,
    'ra_lw_physics': _RA_LW,
    'ra_sw_physics': _RA_SW,
    'bl_pbl_physics': _BL,
    'cu_physics': _CU,
    'sf_surface_physics': _SF_SURFACE,
    'sf_sfclay_physics': _SF_SFCLAY,
}

_SUITES = {
    'conus': {
        'cu_physics': 6,
        'mp_physics': 8,
        'ra_lw_physics': 4,
        'ra_sw_physics': 4,
        'bl_pbl_physics': 2,
        'sf_sfclay_physics': 2,
        'sf_surface_physics': 2,
    },
    'tropical': {
        'cu_physics': 16,
        'mp_physics': 6,
        'ra_lw_physics': 4,
        'ra_sw_physics': 4,
        'bl_pbl_physics': 1,
        'sf_sfclay_physics': 11,
        'sf_surface_physics': 2,
    },
}

_PHYS_FIELDS = [
    'cu_physics',
    'mp_physics',
    'ra_lw_physics',
    'ra_sw_physics',
    'bl_pbl_physics',
    'sf_sfclay_physics',
    'sf_surface_physics',
]


def resolve_suite(case):
    resolved = dict(case)
    filled = []
    suite = case.get('physics_suite')
    if suite is None:
        return resolved, filled
    if isinstance(suite, str):
        suite = suite.lower()
    if suite not in _SUITES:
        for field in _PHYS_FIELDS:
            if case.get(field) == -1:
                filled.append((field, -1, 'UNRESOLVED (-1, no suite)'))
        return resolved, filled
    mapping = _SUITES[suite]
    for field in _PHYS_FIELDS:
        if case.get(field) == -1:
            val = mapping[field]
            resolved[field] = val
            filled.append((field, val, 'suite-filled'))
    return resolved, filled


def scheme_info(category, value):
    table = _KB.get(category)
    if table is not None:
        info = table.get(value)
        if info is not None:
            return info
    return {'name': 'unknown({})'.format(value), 'cost': '?', 'vec': '?', 'note': ''}


def scheme_name(category, value):
    return scheme_info(category, value)['name']


def cost_weight(category, value, weight_table):
    """Resolve (category, value) to a weight in weight_table via scheme_info()['cost'].

    A scheme absent from the _KB tables (scheme_info returns cost='?') is NOT free: it
    degrades to the 'med' weight rather than raising KeyError, and the substitution is
    surfaced on stderr -- never silent -- so profile-building degrades gracefully instead
    of crashing the whole tuning run on one legal-but-untabulated WRF option.
    """
    cost = scheme_info(category, value)['cost']
    if cost not in weight_table:
        print("WARNING cost_weight: {}={} has no cost entry in the physics knowledge base "
              "(scheme_info returned cost={!r}); defaulting to 'med' weight ({}) -- "
              "this is an unmeasured default, not a profiled cost".format(
                  category, value, cost, weight_table['med']), file=sys.stderr)
        return weight_table['med']
    return weight_table[cost]


def diagnose(case):
    lines = []
    for cat in ('mp_physics', 'ra_lw_physics', 'ra_sw_physics', 'bl_pbl_physics', 'cu_physics', 'sf_surface_physics'):
        val = case.get(cat, -1)
        info = scheme_info(cat, val)
        line = "  {}: {} {} [cost={}]".format(cat, val, info['name'], info['cost'])
        if info['note']:
            line += " " + info['note']
        lines.append(line)

    dx_km = float(case.get('dx', 3000)) / 1000.0
    cu = case.get('cu_physics', 0)
    if cu not in (0, None) and dx_km < 4.0:
        lines.append("  CORRECTNESS: cu_physics={} at dx={:g}km -- a cumulus PARAMETERISATION is physically "
                     "inappropriate below ~4km (convection is RESOLVED); it should be 0. cu=0 is BOTH more correct "
                     "AND drops the cumulus call (~few %). Results-affecting -> a correctness fix, not a blind speed knob.".format(cu, dx_km))
    elif (cu in (0, None)) and dx_km >= 10.0:
        lines.append("  CORRECTNESS: cu_physics=0 at dx={:g}km -- convection is UNDER-resolved; a cumulus scheme "
                     "(cu>0) is normally REQUIRED for correctness here (not a speed change).".format(dx_km))

    if case.get('ra_lw_physics') == 4 or case.get('ra_sw_physics') == 4:
        lines.append("  imbalance driver: RRTMG day/night terminator (E-W, corr -0.6); decomposition is a CACHE tool, cannot fix this")
    if case.get('mp_physics') in (8, 10, 28, 38, 40):
        lines.append("  imbalance driver: microphysics storm-blob (moving hotspot, phase-dependent)")
    return lines


def radt_advice(case):
    dx_km = case['dx'] / 1000.0
    rule = max(1, round(dx_km))
    current = case['radt']
    line = "radt={} min; rule-of-thumb radt~=dx_km={} min; ".format(current, rule)
    if rule <= current <= 2 * rule:
        verdict = "within typical range"
    elif current > 2 * rule:
        verdict = "COARSE (fewer radiation calls = faster but less accurate; verify tolerance)"
    else:
        verdict = "FINE (more calls than needed; radt could rise to ~{} to save time, results permitting)".format(rule)
    line += verdict
    line += " [results-affecting: fixed for a benchmark; tunable with accuracy gate for new cases]"
    return line


def timestep_advice(case):
    dx_km = case['dx'] / 1000.0
    rule_dt = 6.0 * dx_km
    current = case['time_step']
    # time_step<=0 is the "let WRF compute it" sentinel (or use_adaptive_time_step);
    # it is NOT a literal 0 s step -- do not treat it as a tunable number.
    if current is None or current <= 0:
        return ("time_step={} -> ADAPTIVE/auto (0 or use_adaptive_time_step: WRF computes dt each step). "
                "Not a fixed knob; CFL rule would be ~{:.2f}s at dx={}km if pinned.".format(current, rule_dt, dx_km))
    line = "time_step={}s; CFL rule dt~=6*dx_km={}s; ".format(current, rule_dt)
    if current <= rule_dt * 1.02 and current >= rule_dt * 0.5:
        verdict = "near recommended max"
    elif current < rule_dt * 0.5:
        verdict = "CONSERVATIVE (dt could rise toward {}s to cut step count, if CFL-stable - confirm with a short probe, no NaN/CFL)".format(rule_dt)
    else:
        verdict = "AGGRESSIVE (>CFL rule; risk of CFL/NaN - verify stability)"
    line += verdict
    line += " [tunable via short probe with HARD CFL/NaN gate]"
    return line


# ---------------------------------------------------------------------------
# Speed knobs: bit-identical (Tier 1) + gated (Tier 2). Reduce TOTAL WORK, not
# imbalance (imbalance axis is measured-exhausted). Physics from the measured
# CONUS-2.5km profile + Fable numerics review 2026-08-27.
# ---------------------------------------------------------------------------

def _acoustic_cfl(case, ns):
    """Horizontal forward-backward acoustic CFL: (cs+U)*dt_sound*sqrt(2)/dx."""
    dt = float(case.get('time_step', 15))
    dx = float(case.get('dx', 2500))
    cs, U = 350.0, 80.0   # sound speed; max advective (Nov CONUS jet aloft)
    return (cs + U) * (dt / ns) * (2 ** 0.5) / dx


def time_step_sound_advice(case):
    ns = int(case.get('time_step_sound', 4) or 4)
    if float(case.get('time_step', 15) or 0) <= 0:
        return ("time_step_sound={} -> time_step is ADAPTIVE/auto (dt<=0), so the acoustic CFL is not a fixed number; "
                "ns must stay EVEN (RK3); keep ns=4 unless a stability probe says otherwise.".format(ns))
    # odd ns is illegal for RK3: stage-2 must cover dt/2, so ns=3 forces a dt/2 acoustic pass
    cfl_cur, cfl4, cfl3 = _acoustic_cfl(case, ns), _acoustic_cfl(case, 4), _acoustic_cfl(case, 2)
    return ("time_step_sound={} -> acoustic CFL~={:.2f}. ns=4 is the FLOOR at dt={}s/dx={}km (CFL {:.2f}); "
            "ns=3 -> CFL~{:.2f} = CERTAIN CRASH (RK3 stage-2 needs even ns, forces a {:.1f}s pass); "
            "raising to 6 costs ~4% for nothing. VERDICT: KEEP ns=4 (already optimal)."
            ).format(ns, cfl_cur, int(case.get('time_step', 15)), _dxkm(case), cfl4, cfl3,
                     float(case.get('time_step', 15)) / 2.0)


def _dxkm(case):
    return round(float(case.get('dx', 2500)) / 1000.0, 1)


def _radt_cloud_bound(case, wind=12.0, ncells=3.0):
    """Cloud-optics staleness bound on radt (min): keep cloud advection under ~ncells
    between radiation calls. wind = nominal mid-trop steering speed (m/s)."""
    dx = float(case.get('dx', 2500))
    return round(ncells * dx / (wind * 60.0), 1)


def swint_advice(case):
    """Case-general: the zenith STAIRCASE only bites when the run crosses the terminator.
    Uses the solar geometry (wtune_solar) to decide, and bounds any radt increase by
    cloud-optics staleness (dx + steering wind), not just a flat 1.5x."""
    if case.get('ra_sw_physics') not in (1, 2, 3, 4, 5):
        return None
    swint = int(case.get('swint_opt', 0) or 0)
    radt = float(case.get('radt', 10))
    radt_cloud = _radt_cloud_bound(case)
    try:
        sm = _wsolar.solar_load_map(case)
    except Exception:
        sm = {'ok': False}
    if not sm.get('ok'):
        return ("swint_opt={}: pass --wrfinput (lat/lon+date) to judge terminator crossing. "
                "Rule: crossing run -> swint_opt=1 removes the zenith staircase and unlocks a radt bump; "
                "cloud-staleness caps radt at ~{} min for dx={}km.").format(swint, radt_cloud, _dxkm(case))
    crossing = 'terminator-crossing' in sm.get('sweep', '')
    spread = float(sm.get('max_day_fraction', 1) - sm.get('min_day_fraction', 1))
    if not crossing:
        return ("swint_opt: run does NOT cross the terminator (day_fraction spread {:.2f}, sweep='{}') -> zenith is "
                "near-constant, swint gives ~0 and radt is gated ONLY by cloud-optics staleness (~{} min cap at dx={}km). "
                "No terminator staircase to remove here.").format(spread, sm.get('sweep', '?'), radt_cloud, _dxkm(case))
    # terminator-crossing: swint pays off; radt headroom = min(1.5x, cloud-staleness cap)
    newr = int(min(radt * 1.5, max(radt, radt_cloud)))
    # radiation SW share of wall scales with the day_fraction spread (terminator-driven part)
    gain = 8.8 * (1.0 - radt / newr) if newr > radt else 0.0
    if swint == 0:
        base = ("swint_opt=0 + terminator-crossing (day_fraction {:.2f}->{:.2f}, spread {:.2f}) -> SW frozen for the whole "
                "radt window = zenith STAIRCASE swept across the domain. SET swint_opt=1 (~0.1% cost): removes the staircase."
                ).format(sm.get('max_day_fraction'), sm.get('min_day_fraction'), spread)
        if newr > radt:
            base += (" Then radt {}->{} min is unlocked (~{:.1f}% wall) and is MORE accurate at the terminator than the "
                     "current radt={} staircase. Cap = cloud staleness ~{} min (dx={}km, 12 m/s). TIER 2: gate vs chaos "
                     "floor; use terminator-band BIAS not point RMSE (reference itself used the staircase)."
                     ).format(int(radt), newr, gain, int(radt), radt_cloud, _dxkm(case))
        else:
            base += " radt already at the cloud-staleness cap ({} min) -> swint improves accuracy but no radt headroom.".format(radt_cloud)
        return base
    return ("swint_opt=1 already on (staircase removed). radt headroom now set by cloud-optics staleness: cap ~{} min "
            "at dx={}km/12 m/s (current {}); a bump to {} saves ~{:.1f}% if it passes the chaos-floor gate."
            ).format(radt_cloud, _dxkm(case), int(radt), newr, gain)


def numtiles_advice(case):
    nt = int(case.get('numtiles', 1) or 1)
    return ("numtiles={} -> try 2/4/8/16 (cache-blocks the tile loops, ZERO cross-tile arithmetic = bit-identical). "
            "MEASURED on CONUS 2.5km 8x56/448: numtiles=8 is -12.6% per-step vs numtiles=1 -- NOT negligible; the "
            "188x21 patch is NOT cache-resident, so tiling it matters. Sweep and pick the fastest; cannot fail a gate."
            ).format(nt)


def thompson_prestage_advice(case):
    return ("Thompson builds qr_acr_qg/qs, freezeH2O tables at init (~40-90s). PRESTAGE those .dat files in the run dir "
            "(rsl.out then says 'read' not 'computing') -> bit-identical, saves init only. Worth ~1.5-3% ONLY IF the "
            "scored window includes init; 0% if scored per-step. Confirm the scoring window first.")


def io_hygiene_advice(case):
    return ("verify restart_interval > run length (one stray restart write = minutes); debug_level=0, nwp_diagnostics=0, "
            "do_radar_ref=0, afwa_*_opt=0, no auxhist streams. Each silently costs percent-level. io_form_history=11 "
            "(PnetCDF) only if total wall is scored; io_form_boundary is a mirage (wrfbdy read once/6h).")


def _nl_list(nl_kv, key, n, cast=float, default=None):
    """Per-domain values from a namelist comma list, padded to length n (WRF repeats
    the last given value for higher domains)."""
    raw = nl_kv.get(key)
    if raw is None:
        return [default] * n
    vals = []
    for tok in str(raw).split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(cast(tok))
        except (ValueError, TypeError):
            vals.append(default)
    if not vals:
        return [default] * n
    while len(vals) < n:
        vals.append(vals[-1])
    return vals[:n]


def nesting_advice(nl_kv, resolved):
    """max_dom>1: the tuner otherwise reads ONLY domain-1, but a real nested run's cost
    is split across domains by COMPUTE WEIGHT = points x cumulative time-step ratio
    (steps_i = steps_1 x prod(parent_time_step_ratio)). Reason about WHICH domain
    dominates and that safe physics levers DIFFER per domain regime. [] for a single domain."""
    try:
        max_dom = int(float(str(nl_kv.get('max_dom', 1)).split(',')[0]))
    except (ValueError, TypeError):
        max_dom = 1
    if max_dom <= 1:
        return []
    n = max_dom
    ewe = _nl_list(nl_kv, 'e_we', n, int, 1)
    esn = _nl_list(nl_kv, 'e_sn', n, int, 1)
    gr = _nl_list(nl_kv, 'parent_grid_ratio', n, int, 1)
    tr = _nl_list(nl_kv, 'parent_time_step_ratio', n, int, 1)
    cu = _nl_list(nl_kv, 'cu_physics', n, int, 0)
    dx1 = float(resolved.get('dx', 30000.0))
    cum, cumgr = 1.0, 1.0
    weight, dxs, stepmult = [], [], []
    for i in range(n):
        cum *= (tr[i] or 1)
        cumgr *= (gr[i] or 1)
        pts = max(1, (ewe[i] or 1) - 1) * max(1, (esn[i] or 1) - 1)
        weight.append(pts * cum)
        stepmult.append(cum)
        dxs.append(dx1 / cumgr)
    tot = sum(weight) or 1.0
    frac = [w / tot for w in weight]
    dom_i = max(range(n), key=lambda i: frac[i])
    L = ["NESTING ({} domains -- WITHOUT this the tuner reasons only about domain-1):".format(n)]
    for i in range(n):
        L.append("  d{}: dx~{:.1f}km  {}x{} pts  {}x steps  -> {:.0f}% of compute  (cu_physics={})".format(
            i + 1, dxs[i] / 1000.0, ewe[i], esn[i], int(stepmult[i]), 100 * frac[i], cu[i]))
    L.append("  => domain d{} dominates ({:.0f}%): the PER-STEP levers (decomposition, numtiles, advection "
             "order, adaptive dt) pay off HERE; the same lever on a minor domain is near-wasted.".format(
                 dom_i + 1, 100 * frac[dom_i]))
    if dxs[0] / max(dxs[-1], 1e-9) >= 2.5:
        L.append("  => SCALE SEPARATION: the coarse OUTER domain (dx~{:.1f}km) safely takes radt/cudt/cu-scheme "
                 "coarsening (cloud-cap & convective-adjustment timescale both scale with dx); the fine nest "
                 "(dx~{:.1f}km) is grey-zone -> vet EACH domain's regime separately, never apply one verdict to "
                 "all. If the coarse domain is a MINOR compute fraction, its safe levers cap the total win.".format(
                     dxs[0] / 1000.0, dxs[-1] / 1000.0))
    for i in range(1, n):
        if (tr[i] or 1) > (gr[i] or 1):
            L.append("  => d{}: parent_time_step_ratio={} > grid_ratio={} -> the nest sub-steps MORE than its CFL "
                     "requires; lowering it toward the grid ratio saves nest steps (WRF-gate the nest CFL). "
                     "ratio==grid_ratio (the usual 3) has NO headroom.".format(i + 1, tr[i], gr[i]))
    return L


def numeric_speed_advice(case):
    """Ranked bit-identical (Tier 1) + gated (Tier 2) speed knobs for this case."""
    L = []
    dxkm = _dxkm(case)
    regime = "convection-permitting (fast error growth)" if dxkm <= 4 else "parameterized-convection (slower growth)"
    L.append("GATE (calibrate BEFORE judging any Tier-2 knob): at dx={}km, {} -- a Tier-2 perturbation".format(dxkm, regime))
    L.append("  grows toward the CHAOS FLOOR over the run. Establish it: reference vs a round-off-perturbed twin ->")
    L.append("  measure field (T2/RAINNC/W) divergence at run end. Tier-2 PASSES iff its diff is statistically")
    L.append("  indistinguishable from that floor AND any systematic band bias is clean. A BITWISE gate kills all of Tier 2.")
    L.append("")
    L.append("TIER 1 (bit-identical -- no accuracy gate, just measure speed):")
    L.append("  numtiles: " + numtiles_advice(case))
    if case.get('mp_physics') in (8, 28, 38):
        L.append("  thompson-tables: " + thompson_prestage_advice(case))
    L.append("  io/diag: " + io_hygiene_advice(case))
    L.append("")
    L.append("TIER 2 (gated -- needs the chaos-floor gate above):")
    sw = swint_advice(case)
    if sw:
        L.append("  swint+radt: " + sw)
    L.append("  time_step_sound: " + time_step_sound_advice(case))
    L.append("")
    L.append("DYNAMICS-BUCKET levers (the 42% chunk -- bigger than radiation; all gated, see `screen`):")
    L.append("  use_adaptive_time_step=.true. (time-mean CFL headroom + cuts step-count => only knob that also")
    L.append("    trims halo_wait); h_sca_adv_order 5->3 SCALARS-ONLY (moist-halo + scalar flux, keep momentum);")
    L.append("    non_hydrostatic=.false. ONLY at dx>=10km; diff_opt=0 smoke-tested. gate each on convective_fraction.")
    L.append("")
    L.append("COARSE-DOMAIN levers (dx>=10km, cumulus ON -- the cost bucket FLIPS: cumulus+radiation, NOT dynamics):")
    L.append("  cudt raise = the radt-analog for the CUMULUS bucket (~5-15%); cap 3*dx/wind (max 60min) -- cudt=0 is")
    L.append("    EVERY-step. radt raise: both cloud-cap AND cumulus quasi-equilibrium scale with dx, so a COARSE grid")
    L.append("    unlocks exactly what the 2.5km convective benchmark rejects. epssm 0.2 buys dt only on STEEP terrain.")
    L.append("    On a NESTED run these are safe on the COARSE outer domain only -- see the NESTING per-domain weights.")
    L.append("")
    L.append("REJECTED (with numbers, so they stay rejected): rk_ord 3->2 (RK2 stability collapse -> dt/4 -> ~3x SLOWER); "
             "moist_adv_opt 1->0 (neg-moisture clip = +10% precip bias); v_*_adv_order 3->2 (jet dispersion); "
             "bldt>0 (mis-times the terminator, saves <2%); cu_physics/ra_lw/ra_sw SCHEME swap (only ~2-4% spread & "
             "RRTMG is already efficient -- radt/cudt are the radiation/cumulus speed levers, the scheme is a SCIENCE "
             "choice); gwd_opt/bl_pbl_physics/sf_surface_physics/icloud/cu_rad_feedback (science, not speed); "
             "feedback 1->0 (changes the parent solution -- results-affecting, not free); "
             "damp_opt/smdiv (upper sponge / divergence damping = correctness, NOT a dt enabler -- epssm is the real "
             "acoustic stabilizer, promoted above to a terrain candidate).")
    return L


def menu():
    lines = []
    for cat in _PHYS_FIELDS:
        lines.append(cat + ":")
        for val in sorted(_KB[cat].keys()):
            info = _KB[cat][val]
            lines.append("  {} {} (cost={})".format(val, info['name'], info['cost']))
    return lines


if __name__ == '__main__':
    sample = {
        'physics_suite': 'conus',
        'mp_physics': -1,
        'ra_lw_physics': -1,
        'ra_sw_physics': -1,
        'bl_pbl_physics': -1,
        'sf_surface_physics': -1,
        'sf_sfclay_physics': -1,
        'cu_physics': 0,
        'radt': 10.0,
        'time_step': 20.0,
        'dx': 3000.0,
        'dy': 3000.0,
    }
    resolved, filled = resolve_suite(sample)
    print('filled:', filled)
    print('\n'.join(diagnose(resolved)))
    print(radt_advice(resolved))
    print(timestep_advice(resolved))


# ---------------------------------------------------------------------------
# Coverage audit: guarantee every SET namelist param is accounted for -- either
# actively evaluated, read as context, inactive (its scheme is off), or flagged
# as an un-evaluated gap. NOT 2062 rules; every param the user actually set is
# classified so nothing is silently ignored.
# ---------------------------------------------------------------------------

import os as _os
import json as _json

_REG_IDX = None


def _registry_index():
    """Index the WRF Registry catalog by param name (lower). Universal lookup for
    ANY param -> {group, type, default, meaning_en, io_stream}."""
    global _REG_IDX
    if _REG_IDX is None:
        path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             'wtune_catalog_registry.json')
        _REG_IDX = {}
        try:
            d = _json.load(open(path))
            for _grp, entries in d.get('groups', {}).items():
                for e in entries:
                    _REG_IDX[str(e.get('name', '')).lower()] = e
        except Exception:
            _REG_IDX = {}
    return _REG_IDX


# params for which the tuner carries DEEPER, specific advice (overlay TAG on the
# universal Registry scan -- NOT a whitelist of what gets evaluated; everything is
# evaluated via the Registry, this only marks where a specific rule adds detail).
_DEEP = {
    'physics_suite', 'mp_physics', 'ra_lw_physics', 'ra_sw_physics', 'bl_pbl_physics',
    'sf_sfclay_physics', 'sf_surface_physics', 'cu_physics', 'gwd_opt', 'sf_urban_physics',
    'slope_rad', 'topo_shading', 'num_soil_layers', 'icloud', 'swint_opt', 'o3input', 'aer_opt',
    'radt', 'cudt', 'bldt', 'time_step', 'time_step_sound', 'diff_opt', 'km_opt', 'w_damping',
    'epssm', 'non_hydrostatic', 'e_vert', 'moist_adv_opt', 'scalar_adv_opt', 'h_mom_adv_order', 'h_sca_adv_order',
    'io_form_history', 'io_form_restart', 'io_form_input', 'io_form_boundary',
    'nproc_x', 'nproc_y', 'numtiles', 'numtiles_x', 'numtiles_y',
    'history_interval', 'history_interval_m', 'restart_interval', 'frames_per_outfile',
    'nio_tasks_per_group', 'nio_groups',
}
# set-but-inactive unless the gating scheme is chosen
_SCHEME_GATED = {
    'maxens': ('cu_physics', {1, 3, 93, 99}), 'maxens2': ('cu_physics', {1, 3, 93, 99}),
    'maxens3': ('cu_physics', {1, 3, 93, 99}), 'maxiens': ('cu_physics', {1, 3, 93, 99}),
    'ensdim': ('cu_physics', {1, 3, 93, 99}),
}


def _norm(v):
    return str(v).strip().strip(',').strip().rstrip('.').lower() if v is not None else ''


# ---- relevance gating: which of the 2062 catalog params are LIVE for a config ----
# whole-group features OFF by default -> params inert unless the master switch is on
_GATED_GROUPS = {
    'fdda': ('grid_fdda', {1, 2}),
    'afwa': ('afwa_diag_opt', {1}),
    'dfi_control': ('dfi_opt', {1, 2, 3}),
    'noah_mp': ('sf_surface_physics', {4}),
}
# groups needing a special run type / off for a standard real-data forecast
_INERT_GROUPS = {'stoch', 'scm', 'tc', 'ideal', 'diags', 'grib2'}
# physics param name-prefix -> (gating scheme param, active scheme values)
_PHYS_PREFIX = {
    'nssl': ('mp_physics', {17, 18, 19, 21, 22}),
    'shinhong': ('bl_pbl_physics', {11}),
    'cu': ('cu_physics', set(range(1, 100))),
    'shcu': ('shcu_physics', {1, 2, 3}),
    'windfarm': ('windfarm_opt', {1, 2}),
    'lightning': ('lightning_option', {1, 2, 3, 11}),
    'iccg': ('lightning_option', {1, 2, 3, 11}),
    'irr': ('irrigation', {1, 2, 3}),
    'megan': ('chem_opt', set(range(1, 600))),
    'mosaic': ('chem_opt', set(range(1, 600))),
    'windturbines': ('windfarm_opt', {1, 2}),
}


def _gate_val(param, resolved, nl_kv):
    # Prefer the RESOLVED (suite-filled) scheme value: a raw namelist -1 means "let the
    # suite decide" and only `resolved` holds the real scheme, so gating on the raw -1
    # wrongly marks every cu*/scheme param inert. Fall back to the raw namelist for params
    # the resolver does not track. Take domain-1 of any multi-domain list.
    for src in (resolved.get(param), nl_kv.get(param)):
        if src is None:
            continue
        try:
            iv = int(float(str(src).strip().split(',')[0].strip()))
        except (TypeError, ValueError, IndexError):
            continue
        if iv == -1:            # unfilled suite sentinel in this source -> try the next
            continue
        return iv
    return None


def _relevance(name, e, resolved, nl_kv):
    """Classify a catalog param for THIS config: relevant / inert / plumbing."""
    grp = e.get('group', '')
    if e.get('io_stream'):
        return 'plumbing'
    if grp in _INERT_GROUPS:
        return 'inert'
    if grp in _GATED_GROUPS:
        gp, active = _GATED_GROUPS[grp]
        return 'relevant' if _gate_val(gp, resolved, nl_kv) in active else 'inert'
    if grp == 'physics':
        if name.startswith('bl_mynn') or name.startswith('mynn'):
            return 'relevant' if _gate_val('bl_pbl_physics', resolved, nl_kv) in {5, 6} else 'inert'
        for pref, (gp, active) in _PHYS_PREFIX.items():
            if name.startswith(pref):
                return 'relevant' if _gate_val(gp, resolved, nl_kv) in active else 'inert'
    return 'relevant'


def coverage_audit(nl_kv, resolved, dump_path=None):
    """FULL-catalog scan: classify ALL ~2062 Registry params for THIS config, not
    just the ones set. Every param -> SET(changed?) / DEFAULT-relevant / INERT
    (scheme/feature off) / I/O-plumbing, with a *DEEP* tag where the tuner has a
    specific rule. Returns (lines, stats); writes the full per-param table to
    dump_path if given. General: works for any WRF namelist."""
    reg = _registry_index()
    total = len(reg) or 1
    cnt = {'set': 0, 'set_changed': 0, 'relevant_default': 0, 'inert': 0, 'plumbing': 0, 'deep': 0}
    actionable, dump = [], []
    for name in sorted(reg):
        e = reg[name]
        default = e.get('default', '?')
        grp = e.get('group', '?')
        set_val = nl_kv.get(name)
        is_set = set_val is not None
        deviates = is_set and _norm(set_val) != _norm(default)
        rel = _relevance(name, e, resolved, nl_kv)
        is_deep = name in _DEEP
        if is_set:
            cnt['set'] += 1
            if deviates:
                cnt['set_changed'] += 1
        elif rel == 'inert':
            cnt['inert'] += 1
        elif rel == 'plumbing':
            cnt['plumbing'] += 1
        else:
            cnt['relevant_default'] += 1
        if is_deep:
            cnt['deep'] += 1
        if (is_set and deviates) or (is_deep and rel != 'inert'):
            state = "SET={} (def={})".format(str(set_val)[:12], default) if is_set else "default={}".format(default)
            tag = "*DEEP*" if is_deep else "      "
            actionable.append("  {:<20} {} {:<24} [{}] {}".format(name, tag, state, grp, (e.get('meaning_en', '') or '')[:40]))
        if dump_path is not None:
            dump.append("{}\t{}\t{}\t{}\t{}\t{}".format(
                name, grp, ('SET:' + str(set_val)) if is_set else 'default', default, rel, 'DEEP' if is_deep else ''))
    if dump_path is not None:
        try:
            open(dump_path, 'w').write("# name\tgroup\tstate\tdefault\trelevance\tdeep\n" + "\n".join(dump) + "\n")
        except Exception:
            pass
    accounted = cnt['set'] + cnt['relevant_default'] + cnt['inert'] + cnt['plumbing']
    head = [
        "FULL Registry-catalog scan: {} params total -- EVERY one classified for this config.".format(total),
        "  SET by namelist ...................... {} ({} changed-from-default)".format(cnt['set'], cnt['set_changed']),
        "  DEFAULT but RELEVANT to this config .. {}  (tuner keeps default unless a rule fires)".format(cnt['relevant_default']),
        "  INERT (scheme/feature OFF here) ...... {}  (correctly not tuned)".format(cnt['inert']),
        "  I/O-stream plumbing .................. {}".format(cnt['plumbing']),
        "  (accounted: {}/{}; params with a DEEP tuner rule: {}; full table -> --dump)".format(accounted, total, cnt['deep']),
        "",
        "ACTIONABLE for this config (SET-and-changed, or a live param the tuner has a specific rule for):",
    ]
    return head + (actionable or ["  (none)"]), cnt


# ---------------------------------------------------------------------------
# Stage 1 of the funnel: ANALYTIC screen. For each relevant param, does a change
# have a plausible EFFECTIVE opportunity for THIS config? Most -> None (default is
# standard/optimal, or it is a correctness choice not a speed knob). The few that
# return a candidate go on to the SpeedyWeather vet (Stage 2) then WRF (Stage 3).
# ---------------------------------------------------------------------------

def _is_les(resolved, nl_kv):
    """LES / microscale: 3D subgrid TKE or Smagorinsky (km_opt 2/3) or dx<1km. Here the
    3D diffusion (diff_opt=2/km_opt=2..3) IS the turbulence closure and the high advection
    order carries the resolved turbulent cascade -- neither is a tunable accuracy knob, so
    the 'reduce numerical work' candidates must be SUPPRESSED (removing them destroys the LES)."""
    km = ''
    if nl_kv.get('km_opt') is not None:
        km = str(nl_kv.get('km_opt')).split(',')[0].strip().rstrip('.')
    return km in ('2', '3') or float(resolved.get('dx', 2500)) < 1000.0


def _effective_opinion(name, val, e, resolved, nl_kv):
    """Return None (keep) or dict(candidate, why, tier, physics_dep, gain)."""
    v = _norm(val)
    dxkm = float(resolved.get('dx', 2500)) / 1000.0
    # --- Tier-1 bit-identical (physics-independent -> SpeedyWeather neutral, go to WRF) ---
    if name in ('io_form_history', 'io_form_restart', 'io_form_input') and v == '2':
        return dict(candidate='14 (ADIOS2)', why='netCDF write is slow; ADIOS2 same field values, faster (if I/O scored)',
                    tier='1', physics_dep=False, gain='big-if-IO-scored')
    if name == 'numtiles' and v in ('1', '0', ''):
        return dict(candidate='2/4/8', why='cache-block the tile loops, bit-identical', tier='1', physics_dep=False, gain='0-3%')
    if name in ('nproc_x', 'nproc_y'):
        return dict(candidate='cost-model #1 (run `plan`)', why='decomposition cache+vec+halo+imbalance score for THIS grid/ranks',
                    tier='1', physics_dep=False, gain='~1-3%')
    # --- Tier-2 gated, PHYSICS-dependent -> SpeedyWeather can vet cheaply ---
    if name == 'swint_opt' and v in ('0', ''):
        return dict(candidate='1', why='remove SW zenith staircase at the terminator (accuracy; unlocks radt IF cloud-cap allows)',
                    tier='2', physics_dep=True, gain='accuracy+maybe-radt')
    if name == 'radt':
        return dict(candidate='raise (e.g. +50%)', why='fewer radiation calls -> less radiation cost & de-paces day side',
                    tier='2', physics_dep=True, gain='~radiation-share')
    if name == 'time_step':
        rule = 6.0 * dxkm
        try:
            cur = float(v)
        except ValueError:
            cur = None
        if cur is None or cur <= 0:
            return None                      # time_step<=0 = adaptive/auto; not a fixed knob
        if cur < rule * 0.95:
            return dict(candidate='raise toward {:.0f}s'.format(rule), why='CFL headroom -> fewer steps',
                        tier='2', physics_dep=True, gain='~step-count')
        return None            # at CFL limit -> keep
    if name in _MP and False:
        return None
    # microphysics scheme: cheaper single-moment MIGHT help IF stratiform (physics-gated)
    if name == 'mp_physics' and (resolved.get('mp_physics') in (8, 10, 28, 38, 40)):
        return dict(candidate='WSM6 (6) -- ONLY IF stratiform', why='single-moment is cheaper; SCHEME SWAP = advise-only, gate vs reference',
                    tier='swap', physics_dep=True, gain='~microphysics-share')
    # horizontal advection order 5->3: cuts the DYNAMICS stencil (the 42% chunk) AND the halo
    # width (3->2 -> ~1/3 less halo bytes) -- a DOUBLE win. Cost = extra implicit hyperdiffusion
    # at 2-4dx that smooths small-scale structure; safe ONLY for smooth/low-convection cases.
    if name == 'h_sca_adv_order' and v in ('5', ''):
        if _is_les(resolved, nl_kv):
            return None    # LES: high advection order carries the resolved cascade -> never reduce
        return dict(candidate='3 (SCALARS ONLY; keep h_mom=5)',
                    why='3rd-order scalar advection: -33% scalar flux flops on 8 Thompson arrays + moist-halo 3->2 '
                        '(~3.5-4.5%); keeping momentum at 5 leaves updraft dynamics intact. Cost = 4th-deriv '
                        'hyperdiffusion smears 2-4dx hydrometeor gradients (a systematic precip-PDF bias)',
                    tier='2', physics_dep=True, gain='dynamics+moist-halo')
    if name == 'h_mom_adv_order' and v in ('5', ''):
        if _is_les(resolved, nl_kv):
            return None    # LES: reducing momentum advection order damps the resolved eddies
        return dict(candidate='3 (momentum -- SMOOTH cases only)',
                    why='drops mom-advection flops + the full 3D halo 3->2, but the 4th-deriv damping now hits '
                        'w/buoyancy -> weakens 5-10km updraft cores. Only smooth/low-conv cases survive',
                    tier='2', physics_dep=True, gain='dynamics+halo')
    # adaptive time step: fixed dt is sized for the WORST instant; adaptive uses the time-MEAN allowed dt
    # -> ~0.77 x mean CFL headroom. The ONLY lever that also cuts halo_wait (fewer of the 110-exchange rounds).
    # Does NOT contradict "dt at CFL limit" -- that is for FIXED dt.
    if name == 'use_adaptive_time_step' and 'true' not in v:      # off / default -> candidate to enable
        return dict(candidate='.true. (target_cfl~1.2, pair epssm=0.2 + w_damping=1 as insurance)',
                    why='fixed dt is sized for the worst instant; adaptive tracks the time-mean allowed dt '
                        '(~0.77 x CFL headroom) AND cuts step count -> the only knob that also shrinks the 28% halo_wait',
                    tier='2', physics_dep=True, gain='~CFL-headroom-time-mean')
    # non-hydrostatic: hydrostatic is cheaper (drops part of small_step) but CANNOT represent <~10km updrafts
    # -> analytic dx-gate, legitimate only at coarse dx.
    if name == 'non_hydrostatic' and 'false' not in v:           # currently non-hydrostatic (on/default)
        if dxkm >= 10:
            return dict(candidate='.false. (hydrostatic)',
                        why='hydrostatic drops part of the acoustic/small_step cost (~10-15%); VALID only at '
                            'dx>=~10km -- at finer dx it cannot represent grid-point storms',
                        tier='2', physics_dep=True, gain='~10-15%')   # physics-dependent: vet dx+regime, not a free PASS
        return None    # dx<10km -> hydrostatic invalid, never a candidate
    # horizontal diffusion: removing it saves ~4-6% but it is the ONLY horizontal mixing (MYJ is 1-D);
    # 2dx energy then accumulates in weak-flow deforming regions -> spurious convection / CFL crash.
    if name == 'diff_opt' and v.split(',')[0].strip() in ('1', '2'):
        if _is_les(resolved, nl_kv):
            return None    # LES: diff_opt=2/km_opt=2..3 IS the subgrid turbulence model, not removable
        return dict(candidate='0 -- SMOKE-TEST first (or km_opt=1)',
                    why='removes the diffusion sub-bucket (~4-6%) + a few halo calls; but it is the only horizontal '
                        'mixing -> 2dx energy accumulates in weak-flow/lee regions -> spurious convection or CFL crash',
                    tier='2', physics_dep=True, gain='~4-6%')
    # coarse-domain CUMULUS cadence: on cu-on coarse domains the cost bucket FLIPS to
    # cumulus+radiation, and cudt is the radt-analog. The cumulus tendency is slowly-varying
    # (quasi-equilibrium ~30-60 min) -> refresh it on an advective cap, not every step.
    if name == 'cudt' and int(resolved.get('cu_physics') or 0) > 0 and dxkm >= 10.0:
        cap_min = min(60.0, max(5.0, 3.0 * float(resolved.get('dx', 10000)) / (12.0 * 60.0)))
        try:
            cur_cudt = float(v.split(',')[0])
        except (ValueError, IndexError):
            cur_cudt = None
        if cur_cudt is None or cur_cudt <= 0 or cur_cudt < cap_min * 0.9:
            return dict(candidate='raise toward {:.0f} min (cudt=0 => EVERY step)'.format(cap_min),
                        why='the radt-analog for the CUMULUS bucket (~5-15% on coarse cu-on domains): the tendency '
                            'is slowly-varying (quasi-equilibrium ~30-60 min), so refreshing it on the advective cap '
                            '3*dx/wind (max 60 min) rather than every step is within tolerance for non-persistent conv',
                        tier='2', physics_dep=True, gain='~cumulus-share (5-15%)')
        return None
    # epssm: acoustic off-centering. Raising it damps vertical acoustic modes -> on STEEP
    # TERRAIN it can hold a larger time_step_sound/dt that would otherwise trip CFL. ~0 direct
    # cost; the gain is indirect and terrain-only, so terrain (not in the probe) gates it.
    if name == 'epssm':
        try:
            cur_epssm = float(v) if v else 0.1     # Fortran '.1' and '0.1' both parse
        except ValueError:
            cur_epssm = 0.1
        if cur_epssm > 0.11:                        # already off-centered -> no candidate
            return None
        return dict(candidate='0.2 (0.3 if steep terrain) -- SMOKE-TEST',
                    why='off-centers the vertically-implicit acoustic solver -> damps vertical acoustic modes; on '
                        'steep-terrain cases this can hold a larger time_step_sound/dt that would otherwise trip CFL '
                        '(INDIRECT gain, terrain-only). Cost ~0. Terrain relief is NOT in the probe -> 1h WRF smoke',
                    tier='2', physics_dep=True, gain='indirect dt-headroom (terrain)')
    # CUMULUS scheme swap: on cu-on coarse domains the cu_driver is a real cost bucket (the
    # oracle's cumulus phase). Kain-Fritsch(1)/Grell-Freitas(3,5) do CAPE closure + up/downdraft
    # profiles (dear); BMJ(2) is a bulk adjustment scheme (cheap, no explicit plume). A cheaper
    # scheme MIGHT pass for weak/shallow non-organized convection -- advise-only, gate vs reference.
    if name == 'cu_physics':
        cu = int(resolved.get('cu_physics') or 0)
        if cu in (1, 3, 5, 11, 14, 16) and dxkm >= 10.0:
            return dict(candidate='BMJ (2) -- ONLY IF weak/shallow non-organized conv',
                        why='cu_driver is a real bucket on coarse cu-on domains; KF(1)/GF(3,5) run CAPE closure + '
                            'up/downdraft profiles (dear), BMJ(2) is a cheap adjustment scheme. Cheaper MIGHT pass for '
                            'weak/shallow conv; SCHEME SWAP = advise-only, gate vs reference precip',
                        tier='swap', physics_dep=True, gain='~cumulus-share')
        return None
    # vertical levels: physics cost is ~linear in e_vert, but fewer levels REGRID the column ->
    # a different PBL/cloud vertical structure (NOT bit-identical). A correctness choice, not a
    # clean speed knob -> flag it, never silently recommend cutting.
    if name == 'e_vert':
        try:
            ev = int(str(v).split(',')[0])
        except (ValueError, IndexError):
            ev = None
        if ev and ev > 45:
            return dict(candidate='consider fewer (e.g. 35) -- RESULTS-AFFECTING, gate vs reference',
                        why='physics cost ~linear in e_vert; but reducing levels regrids the column -> different '
                            'PBL/cloud vertical structure (NOT bit-identical). Only if the case does not need the fine '
                            'vertical resolution; validate precip/PBL vs reference',
                        tier='swap', physics_dep=True, gain='~linear-in-levels')
        return None
    return None


def screen_candidates(nl_kv, resolved):
    """Stage 1: scan ALL catalog params, return the EFFECTIVE candidates (analytic)."""
    reg = _registry_index()
    cands = []
    for name in sorted(reg):
        e = reg[name]
        if _relevance(name, e, resolved, nl_kv) == 'inert':
            continue                         # scheme/feature OFF -> no effective change
        # NOTE: plumbing (io_stream) params are NOT skipped -- io_form_* are I/O-stream
        # by the Registry yet ARE the top effective knob; let _effective_opinion decide.
        val = nl_kv.get(name, e.get('default'))
        op = _effective_opinion(name, val, e, resolved, nl_kv)
        if op:
            op['name'] = name
            op['current'] = nl_kv.get(name, 'default={}'.format(e.get('default')))
            cands.append(op)
    order = {'1': 0, '2': 1, 'swap': 2}
    cands.sort(key=lambda c: order.get(c['tier'], 3))
    return cands
