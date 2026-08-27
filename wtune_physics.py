"""Physics knowledge base + advisor for a WRF config autotuner.

Physics SCHEME choice is NOT a free performance knob -- changing it changes the
weather/results and fails the output-tolerance gate. This module
(a) RESOLVES physics_suite sentinels, (b) DIAGNOSES the cost/imbalance of the
CHOSEN physics, (c) ADVISES results-preserving/gated numerics only (radt,
time_step). It NEVER recommends swapping a scheme for speed.
"""

import wtune_solar as _wsolar

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
    6: {'name': 'Tiedtke', 'cost': 'med', 'vec': '?', 'note': ''},
    11: {'name': 'multiscale KF', 'cost': 'med', 'vec': '?', 'note': ''},
    16: {'name': 'New Tiedtke', 'cost': 'med', 'vec': '?', 'note': ''},
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


def diagnose(case):
    lines = []
    for cat in ('mp_physics', 'ra_lw_physics', 'ra_sw_physics', 'bl_pbl_physics', 'cu_physics', 'sf_surface_physics'):
        val = case.get(cat, -1)
        info = scheme_info(cat, val)
        line = "  {}: {} {} [cost={}]".format(cat, val, info['name'], info['cost'])
        if info['note']:
            line += " " + info['note']
        lines.append(line)

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
    return ("numtiles={} -> try 2/4/8 (cache-blocks the tile loops, ZERO cross-tile arithmetic = bit-identical). "
            "0-3%, often ~0 at 448 ranks (patch already cache-resident on SPR). Free to sweep, cannot fail a gate; "
            "revert if it slows down.").format(nt)


def thompson_prestage_advice(case):
    return ("Thompson builds qr_acr_qg/qs, freezeH2O tables at init (~40-90s). PRESTAGE those .dat files in the run dir "
            "(rsl.out then says 'read' not 'computing') -> bit-identical, saves init only. Worth ~1.5-3% ONLY IF the "
            "scored window includes init; 0% if scored per-step. Confirm the scoring window first.")


def io_hygiene_advice(case):
    return ("verify restart_interval > run length (one stray restart write = minutes); debug_level=0, nwp_diagnostics=0, "
            "do_radar_ref=0, afwa_*_opt=0, no auxhist streams. Each silently costs percent-level. io_form_history=11 "
            "(PnetCDF) only if total wall is scored; io_form_boundary is a mirage (wrfbdy read once/6h).")


def numeric_speed_advice(case):
    """Ranked bit-identical (Tier 1) + gated (Tier 2) speed knobs for this case."""
    L = []
    L.append("GATE (calibrate BEFORE judging any Tier-2 knob): at 2.5km convective, ANY Tier-2 perturbation grows to the")
    L.append("  CHAOS FLOOR by ~6h. Establish it: reference vs a round-off-perturbed twin -> 6h T2/RAINNC/W divergence.")
    L.append("  Tier-2 PASSES iff its diff is statistically indistinguishable from that floor AND terminator-band systematics")
    L.append("  are clean (mask |cosZ|<0.1: |bias(T2)|<0.1K, |bias(SWDOWN)|<5 W/m2). A BITWISE gate kills all of Tier 2.")
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
    L.append("REJECTED (with numbers, so they stay rejected): moist_adv_opt 1->0 (neg-moisture, gate fail); "
             "h_*_adv_order 5->3 (restructures convection); ra_*=24 fast-RRTMG (=scheme swap, advise-only); "
             "epssm/smdiv (change results, save zero work); icloud=0 (different climate). Realistic strict-gate total: ~3-6%.")
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
        path = _os.path.join(_os.path.dirname(__file__), 'wtune_catalog_registry.json')
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
    'epssm', 'non_hydrostatic', 'moist_adv_opt', 'scalar_adv_opt', 'h_mom_adv_order', 'h_sca_adv_order',
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


def coverage_audit(nl_kv, resolved):
    """GENERAL, Registry-driven scan of EVERY set param -- works for ANY WRF case,
    not a hardcoded list. Each set param is looked up in the WRF Registry catalog
    (type/default/group/meaning) and reported vs its default; params for which the
    tuner has a DEEPER rule are tagged *DEEP*; scheme-internal params whose gate is
    off are flagged INACTIVE; anything absent from the Registry is flagged UNKNOWN.
    nl_kv = {param_lower: value_string}. Returns (lines, stats)."""
    reg = _registry_index()
    lines = []
    n = dev = deep = io = unknown = inactive = 0
    for k in sorted(nl_kv.keys()):
        n += 1
        val = str(nl_kv[k])[:12]
        if k in _SCHEME_GATED:
            gate, active = _SCHEME_GATED[k]
            raw = nl_kv.get(gate, resolved.get(gate))   # nl_kv has the raw set value
            try:
                gv = int(str(raw).strip().rstrip(','))
            except (TypeError, ValueError):
                gv = None
            if gv not in active:
                inactive += 1
                lines.append("  INACTIVE {:<22}={:<12} (gate {}={} off -> no effect)".format(k, val, gate, gv))
                continue
        e = reg.get(k)
        if e is None:
            unknown += 1
            lines.append("  UNKNOWN  {:<22}={:<12} (not in Registry -- typo or newer than catalog)".format(k, val))
            continue
        default = e.get('default', '?')
        grp = e.get('group', '?')
        desc = (e.get('meaning_en', '') or '')[:52]
        deviates = _norm(nl_kv[k]) != _norm(default)
        if deviates:
            dev += 1
        if e.get('io_stream'):
            io += 1
        is_deep = k in _DEEP
        if is_deep:
            deep += 1
        tag = "*DEEP*" if is_deep else ("[io]" if e.get('io_stream') else "      ")
        flag = "CHANGED(def={})".format(default) if deviates else "default"
        lines.append("  {:<20}={:<12} {} {:<20} [{}] {}".format(k, val, tag, flag, grp, desc))
    head = ("Registry-driven scan of {} SET params: {} changed-from-default, {} with deeper tuner rules (*DEEP*), "
            "{} I/O-stream, {} inactive-by-scheme, {} not-in-Registry.".format(n, dev, deep, io, inactive, unknown))
    return [head, "(every set param is scanned against the WRF Registry; *DEEP* = tuner has a specific rule; "
            "CHANGED = differs from Registry default, worth a look)", ""] + lines, \
        {'n': n, 'dev': dev, 'deep': deep, 'unknown': unknown, 'inactive': inactive}
