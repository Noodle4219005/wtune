"""Physics knowledge base + advisor for a WRF config autotuner.

Physics SCHEME choice is NOT a free performance knob -- changing it changes the
weather/results and fails the output-tolerance gate. This module
(a) RESOLVES physics_suite sentinels, (b) DIAGNOSES the cost/imbalance of the
CHOSEN physics, (c) ADVISES results-preserving/gated numerics only (radt,
time_step). It NEVER recommends swapping a scheme for speed.
"""

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
