#!/usr/bin/env python3
"""wtune/case.py -- "case understanding" profile: what WTune can infer about a case from its
INPUTS ALONE, before any measurement. This is the prior used to pick a first physics/core-count
starting point; every quantitative claim in here is labelled PRIOR/PREDICTION/UNCONFIRMED so a
later measurement can confirm or refute it, never silently treated as ground truth.

CLI: python3 wtune/cli.py profile <case_dir> [--namelist P] [--wrfinput P] [--cores-per-node 72] [--json OUT]

Reuses (does not reimplement):
  wtune/cli.py             parse_namelist, _parse_nl_kv, extract_case, enrich_case, radiation_spacing,
                        HALO (halo width constant, wtune/cli.py:27)
  wtune/physics.py      resolve_suite, diagnose, scheme_info/scheme_name, radt_advice,
                        timestep_advice, time_step_sound_advice, nesting_advice, swint_advice,
                        _nl_list (per-domain namelist list parsing)
  wtune/shape.py        MIN_PATCH, patch_dims, tiling
  predictor/decomp_space.py  legal_shapes_exact, MIN_PATCH (exact RSL patch legality, incl. nests)
  wtune/solar.py        solar_summary, domain_bbox, day_of_year
  wind_probe.py         steering() -- OPTIONAL, lazy netCDF4 import
  wtune/compliance.py   lookup, classify (legal enum values + STRICT/PHYSICS-GRAY classification)

netCDF4 is never imported at module load time. Every wrfinput-derived field is best-effort and
degrades to "unavailable, say so" when netCDF4 or the file is missing.
"""
from __future__ import annotations

import datetime
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PREDICTOR_DIR = os.path.join(_HERE, 'predictor')
if _PREDICTOR_DIR not in sys.path:
    sys.path.insert(0, _PREDICTOR_DIR)

from wtune import cli as wt                     # repo root: parse_namelist/_parse_nl_kv/extract_case/...
from wtune import physics as phys           # physics knowledge base + advisor
from wtune import shape as shape            # patch_dims/tiling (MIN_PATCH=10, cost-model constants)
from wtune import solar as wsolar           # terminator load map, domain_bbox, day_of_year
from wtune import compliance as wc          # knob catalog: legal_values + classification
import decomp_space as ds              # predictor/decomp_space.py: legal_shapes_exact (exact RSL rules)

DEFAULT_CORES_PER_NODE = 72            # GH200 Grace node default for THIS tool (independent of
                                        # wtune.py's own DEFAULT_CORES_PER_NODE=112, a different
                                        # target machine)

# ---------------------------------------------------------------------------
# Section 3: expected cost structure (PRIOR -- relative ordering only, never a wall-time claim)
# ---------------------------------------------------------------------------
# One documented weight per qualitative wtune_physics scheme_info()['cost'] label. Thompson/
# Morrison/NSSL etc. are already 'high' vs WSM's 'med' in wtune_physics._MP, so "microphysics
# heavier for Thompson than WSM" falls out of reusing that table rather than a name special-case.
COST_WEIGHT = {'none': 0.0, 'low': 1.0, 'med': 2.0, 'high': 4.0}

# Dynamics (RK3 solver + advection) has no scheme_info entry -- it always runs. This is a fixed
# generic baseline (WRF profiling folklore: dynamics is typically the single largest phase), NOT
# fit to this case; treat it as a coarse anchor for RANKING the physics phases against, not a
# calibrated number.
DYN_WEIGHT = 5.0

# ---------------------------------------------------------------------------
# Section 5: core-count priors -- strong-scaling knee model (PREDICTION, not measurement)
# ---------------------------------------------------------------------------
# per-rank work ~ px*py (patch interior cells); halo exchange ~ c * 2*H*(px+py) (perimeter cells,
# both directions, pack+MPI+unpack). H is the halo WIDTH, reusing wtune.py's own HALO=5 constant
# (wtune/cli.py:27, "WRF halo width approximation") so this file does not invent a second number for
# the same physical quantity. c is UNCALIBRATED: it assumes one halo-cell's exchange costs ~1x one
# interior cell's per-step compute (pack + MPI send/recv + unpack ~ one cell of work). Both H and c
# are printed with every prediction precisely so a later strong-scaling run can refute them.
HALO_H = wt.HALO
C_HALO = 1.0
NODE_COUNTS_TO_CHECK = (1, 2, 4, 8, 13)


def efficiency(px, py, c=C_HALO, h=HALO_H):
    """E(px,py) = px*py / (px*py + c*2*H*(px+py)) -- proxy for parallel efficiency of one rank's
    patch. 1.0 = no halo overhead; falls toward 0 as the patch shrinks toward the halo width."""
    work = float(px) * float(py)
    halo = float(c) * 2.0 * float(h) * (px + py)
    return work / (work + halo) if (work + halo) > 0 else 0.0


def compute_r_max(we_list, sn_list, min_patch=None):
    """Max legal MPI rank count at omp=1: every active domain's patch >= min_patch on both axes.

    Closed form: nx_cap = min_d(we_d // min_patch), ny_cap = min_d(sn_d // min_patch); since a
    smaller nx/ny only relaxes every domain's constraint, (nx_cap, ny_cap) is itself the largest
    legal factorisation, so R_max = nx_cap*ny_cap. Verified (not just asserted) against
    decomp_space.legal_shapes_exact -- the exact RSL-lite floor-division gate -- below. Hand-check
    (single domain 91x100, min_patch=10): nx_cap=9, ny_cap=10 -> R_max=90; 91 ranks has no legal
    shape (confirmed against legal_shapes_exact during development)."""
    min_patch = ds.MIN_PATCH if min_patch is None else min_patch
    nx_cap = min(we // min_patch for we in we_list)
    ny_cap = min(sn // min_patch for sn in sn_list)
    r_max = nx_cap * ny_cap
    if r_max <= 0:
        return 0, (0, 0)
    legal = ds.legal_shapes_exact(we_list, sn_list, r_max, min_patch)
    if (nx_cap, ny_cap) not in legal:
        # Should be unreachable given the closed-form argument above; if it ever fires, trust the
        # exact RSL rule over the closed form and fall back to the largest ranks decomp_space
        # itself accepts.
        best = 0
        for r in range(r_max, 0, -1):
            if ds.legal_shapes_exact(we_list, sn_list, r, min_patch):
                best = r
                break
        return best, (0, 0)
    return r_max, (nx_cap, ny_cap)


def least_aspect_legal(we_list, sn_list, ranks, min_patch=None):
    """The most-square (smallest |nx-ny|) shape among the shapes decomp_space.legal_shapes_exact
    actually accepts for this grid at this rank count -- None if no legal shape exists."""
    min_patch = ds.MIN_PATCH if min_patch is None else min_patch
    legal = ds.legal_shapes_exact(we_list, sn_list, ranks, min_patch)
    if not legal:
        return None
    return min(legal, key=lambda p: abs(p[0] - p[1]))


def _find_knee(threshold, dom_we, dom_sn, we_list, sn_list, cores_per_node, nodes_cap, min_patch):
    """Smallest node count (1..nodes_cap) whose least-aspect legal shape's efficiency proxy drops
    below `threshold`. (None, None) if it never drops below threshold within the legal range."""
    for n in range(1, max(1, nodes_cap) + 1):
        ranks = n * cores_per_node
        pair = least_aspect_legal(we_list, sn_list, ranks, min_patch)
        if pair is None:
            continue
        nx, ny = pair
        _, wmin = shape.patch_dims(dom_we, nx)
        _, hmin = shape.patch_dims(dom_sn, ny)
        e = efficiency(wmin, hmin)
        if e < threshold:
            return n, round(e, 4)
    return None, None


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


# ---------------------------------------------------------------------------
# Domain table (shared by identity / cost-structure / core-count sections): structured version of
# the same compute-weight arithmetic wtune_physics.nesting_advice does internally (that function
# only returns human-readable text lines, so it cannot be reused directly for structured data).
# ---------------------------------------------------------------------------

def _domain_table(nl_kv, case):
    try:
        max_dom = int(float(str(nl_kv.get('max_dom', case.get('max_dom', 1))).split(',')[0]))
    except (ValueError, TypeError):
        max_dom = 1
    n = max(1, max_dom)
    ewe = phys._nl_list(nl_kv, 'e_we', n, int, case.get('e_we', 100))
    esn = phys._nl_list(nl_kv, 'e_sn', n, int, case.get('e_sn', 100))
    evert = phys._nl_list(nl_kv, 'e_vert', n, int, case.get('e_vert', 50))
    gr = phys._nl_list(nl_kv, 'parent_grid_ratio', n, int, 1)
    tr = phys._nl_list(nl_kv, 'parent_time_step_ratio', n, int, 1)
    fb = phys._nl_list(nl_kv, 'feedback', n, int, 0)
    dx1 = float(case.get('dx', 3000.0))
    cumgr, cum = 1.0, 1.0
    doms = []
    for i in range(n):
        cumgr *= (gr[i] or 1)
        cum *= (tr[i] or 1)
        we, sn, ev = ewe[i] or 1, esn[i] or 1, evert[i] or 1
        nx, ny = max(1, we - 1), max(1, sn - 1)
        columns = nx * ny
        doms.append(dict(
            index=i + 1, e_we=we, e_sn=sn, e_vert=ev, nx=nx, ny=ny,
            dx_m=round(dx1 / cumgr, 1), dy_m=round(dx1 / cumgr, 1),
            parent_grid_ratio=gr[i], parent_time_step_ratio=tr[i], feedback=fb[i],
            step_multiplier=cum, columns=columns, cells=columns * ev,
            compute_weight_pts=columns * cum,
        ))
    tot = sum(d['compute_weight_pts'] for d in doms) or 1.0
    for d in doms:
        d['compute_share_pct'] = round(100.0 * d['compute_weight_pts'] / tot, 1)
    dom_i = max(range(n), key=lambda i: doms[i]['compute_weight_pts'])
    return doms, dom_i


# ---------------------------------------------------------------------------
# Optional wrfinput-derived fields (lazy netCDF4; never raises, never required)
# ---------------------------------------------------------------------------

def _try_wrfinput_exact(wrfinput):
    """Exact lat/lon bbox + LANDMASK land fraction straight from wrfinput, when netCDF4 is
    installed and the file is given/readable. Returns ({}, note) otherwise -- never raises."""
    if not wrfinput:
        return {}, None
    try:
        import netCDF4
    except ImportError:
        return {}, 'netCDF4 not installed -- bbox is the analytic approximation only, no land_fraction'
    try:
        nc = netCDF4.Dataset(wrfinput)
        try:
            xlat = nc.variables['XLAT'][0]
            xlong = nc.variables['XLONG'][0]
            out = {'bbox_exact': {
                'lat_min': round(float(xlat.min()), 3), 'lat_max': round(float(xlat.max()), 3),
                'lon_min': round(float(xlong.min()), 3), 'lon_max': round(float(xlong.max()), 3),
            }}
            if 'LANDMASK' in nc.variables:
                lm = nc.variables['LANDMASK'][0]
                out['land_fraction'] = round(float(lm[:].mean()), 3)
            return out, None
        finally:
            nc.close()
    except Exception as exc:
        return {}, 'wrfinput read failed ({}) -- falling back to analytic bbox'.format(exc)


# ---------------------------------------------------------------------------
# Section 1: identity
# ---------------------------------------------------------------------------

_NH_SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
              6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
_SH_FLIP = {'DJF': 'JJA', 'JJA': 'DJF', 'MAM': 'SON', 'SON': 'MAM'}


def _season(lat, month):
    if lat is None or month is None:
        return None
    s = _NH_SEASON.get(int(month))
    if s is None:
        return None
    return s if lat >= 0 else _SH_FLIP[s]


def _end_time_estimate(case):
    try:
        start = datetime.datetime(
            int(case.get('start_year') or 2020), int(case.get('start_month') or 1),
            int(case.get('start_day') or 1), int(case.get('start_hour') or 0),
            int(case.get('start_minute') or 0), int(case.get('start_second') or 0))
        end = start + datetime.timedelta(hours=float(case.get('run_total_hours') or 0))
        return end.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError, OverflowError):
        return None


def _build_identity(resolved, doms, dom_i, exact, exact_note, enrich_err):
    lat, lon = resolved.get('latitude'), resolved.get('longitude')
    doy = wsolar.day_of_year(resolved.get('start_month'), resolved.get('start_day'))
    return {
        'max_dom': len(doms),
        'domains': doms,
        'dominant_domain': dom_i + 1,
        'start': {k: resolved.get(k) for k in
                  ('start_year', 'start_month', 'start_day', 'start_hour', 'start_minute', 'start_second')},
        'end_estimate': _end_time_estimate(resolved),
        'run_total_hours': resolved.get('run_total_hours'),
        'time_step_s': resolved.get('time_step'),
        'center_lat': lat,
        'center_lon': lon,
        'bbox_analytic': wsolar.domain_bbox(resolved),
        'bbox_exact': exact.get('bbox_exact'),
        'wrfinput_note': exact_note or enrich_err,
        'day_of_year': doy,
        'season': _season(lat, resolved.get('start_month')),
        'total_columns': sum(d['columns'] for d in doms),
        'total_cells': sum(d['cells'] for d in doms),
    }


# ---------------------------------------------------------------------------
# Section 2: physics
# ---------------------------------------------------------------------------

_PHYS_CATS = ('mp_physics', 'cu_physics', 'bl_pbl_physics', 'sf_sfclay_physics',
              'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics')


def _dx_regime(dx_km):
    if dx_km <= 4.0:
        return 'convection-permitting (dx<=4km): cu_physics should normally be 0', []
    if dx_km <= 10.0:
        return 'gray zone (4-10km): scale-aware cumulus expected', [3, 11]   # Grell-Freitas, MSKF
    return 'parameterized (dx>10km): a cumulus scheme is normally required', [1, 6]  # KF, Tiedtke


def _build_physics(resolved, filled):
    dx_km = float(resolved.get('dx', 3000)) / 1000.0
    regime, suggested_cu = _dx_regime(dx_km)
    return {
        'suite': resolved.get('physics_suite'),
        'suite_filled': [{'field': f, 'value': v, 'why': w} for f, v, w in filled],
        'scheme_names': {cat: phys.scheme_name(cat, resolved.get(cat, -1)) for cat in _PHYS_CATS},
        'dx_km': round(dx_km, 3),
        'dx_regime': regime,
        'gray_zone_suggested_cu': suggested_cu,
        'notes': phys.diagnose(resolved),
        'radt_advice': phys.radt_advice(resolved),
        'timestep_advice': phys.timestep_advice(resolved),
        'sound_step_advice': phys.time_step_sound_advice(resolved),
    }


# ---------------------------------------------------------------------------
# Section 3: expected cost structure
# ---------------------------------------------------------------------------

def _build_cost_structure(resolved, doms, dom_i):
    dom = doms[dom_i]
    rad_spacing = wt.radiation_spacing(resolved)
    w = {cat: phys.cost_weight(cat, resolved.get(cat, -1), COST_WEIGHT) for cat in _PHYS_CATS}
    ra_w = (w['ra_lw_physics'] + w['ra_sw_physics']) / max(1, rad_spacing)
    columns_x_vert = dom['columns'] * dom['e_vert']
    phases = [
        ('dynamics', DYN_WEIGHT, 'baseline RK3 solver + advection (fixed generic anchor, not scheme-specific)'),
        ('microphysics', w['mp_physics'], phys.scheme_name('mp_physics', resolved.get('mp_physics', -1))),
        ('cumulus', w['cu_physics'], phys.scheme_name('cu_physics', resolved.get('cu_physics', -1))),
        ('boundary_layer', w['bl_pbl_physics'], phys.scheme_name('bl_pbl_physics', resolved.get('bl_pbl_physics', -1))),
        ('surface_layer', w['sf_sfclay_physics'], phys.scheme_name('sf_sfclay_physics', resolved.get('sf_sfclay_physics', -1))),
        ('surface', w['sf_surface_physics'], phys.scheme_name('sf_surface_physics', resolved.get('sf_surface_physics', -1))),
        ('radiation', round(ra_w, 3), 'amortized over ~{} steps/call (radt={}min, dt={}s)'.format(
            rad_spacing, resolved.get('radt'), resolved.get('time_step'))),
    ]
    ranked = sorted(
        [{'phase': p, 'weight': wt_, 'relative_cost': round(columns_x_vert * wt_, 1), 'detail': d}
         for p, wt_, d in phases],
        key=lambda r: -r['relative_cost'])
    return {
        'label': 'PRIOR -- not measured; ranks phases by weight x (dominant-domain columns x e_vert), '
                 'not a wall-time claim',
        'weight_table': COST_WEIGHT,
        'dynamics_baseline_weight': DYN_WEIGHT,
        'dominant_domain_index': dom_i + 1,
        'dominant_domain_columns_x_evert': columns_x_vert,
        'phases_ranked': ranked,
    }


# ---------------------------------------------------------------------------
# Section 4: load-imbalance drivers
# ---------------------------------------------------------------------------

def _build_imbalance(resolved, nl_kv, doms, exact):
    return {
        'label': 'PRIOR -- solar term is deterministic astronomy; nesting is geometry; the rest are qualitative flags',
        'solar_terminator': wsolar.solar_summary(resolved),
        'nesting': phys.nesting_advice(nl_kv, resolved) if len(doms) > 1 else [],
        'physics_driven_flags': [l for l in phys.diagnose(resolved) if 'imbalance driver' in l],
        'land_fraction': exact.get('land_fraction'),
        'land_fraction_note': None if 'land_fraction' in exact else 'unavailable (no wrfinput / no netCDF4 / no LANDMASK var)',
    }


# ---------------------------------------------------------------------------
# Section 5: core-count priors
# ---------------------------------------------------------------------------

def _build_core_priors(we_list, sn_list, doms, dom_i, cores_per_node):
    min_patch = ds.MIN_PATCH
    r_max, (nx_cap, ny_cap) = compute_r_max(we_list, sn_list, min_patch)
    dom = doms[dom_i]

    table = []
    for n in NODE_COUNTS_TO_CHECK:
        ranks = n * cores_per_node
        pair = least_aspect_legal(we_list, sn_list, ranks, min_patch)
        if pair is None:
            table.append({'nodes': n, 'ranks': ranks, 'legal': False})
            continue
        nx, ny = pair
        _, wmin = shape.patch_dims(dom['e_we'], nx)
        _, hmin = shape.patch_dims(dom['e_sn'], ny)
        e = efficiency(wmin, hmin)
        table.append({
            'nodes': n, 'ranks': ranks, 'legal': True, 'nproc_x': nx, 'nproc_y': ny,
            'patch_w': wmin, 'patch_h': hmin, 'efficiency_E': round(e, 4),
            'below_0.8': e < 0.8, 'below_0.6': e < 0.6,
        })

    nodes_cap = max(1, r_max // cores_per_node) if r_max else 1
    knee80_n, knee80_e = _find_knee(0.8, dom['e_we'], dom['e_sn'], we_list, sn_list, cores_per_node, nodes_cap, min_patch)
    knee60_n, knee60_e = _find_knee(0.6, dom['e_we'], dom['e_sn'], we_list, sn_list, cores_per_node, nodes_cap, min_patch)

    first_test = sorted(set(n for n in (1, 2, 4, knee80_n) if n and n <= nodes_cap)) or [1]
    check_node = first_test[min(1, len(first_test) - 1)]
    omp_candidates = []
    for d in _divisors(cores_per_node):
        ranks_per_node = cores_per_node // d
        total = ranks_per_node * check_node
        legal = bool(ds.legal_shapes_exact(we_list, sn_list, total, min_patch))
        omp_candidates.append({'omp': d, 'ranks_per_node': ranks_per_node,
                                'checked_at_nodes': check_node, 'legal': legal})

    return {
        'label': 'PREDICTION -- E(px,py)=px*py/(px*py + c*2*H*(px+py)), H={} (=wtune.py HALO), c={} '
                 '(UNCALIBRATED); a later strong-scaling run must confirm/refute these constants'.format(
                     HALO_H, C_HALO),
        'min_patch': min_patch,
        'r_max_ranks_omp1': r_max,
        'r_max_shape': {'nproc_x': nx_cap, 'nproc_y': ny_cap},
        'r_max_note': 'binding per axis independently across all {} active domain(s)'.format(len(doms)),
        'nodes_at_r_max': round(r_max / float(cores_per_node), 2) if r_max else 0.0,
        'cores_per_node': cores_per_node,
        'efficiency_table': table,
        'knee_below_0.8': {'nodes': knee80_n, 'efficiency': knee80_e},
        'knee_below_0.6': {'nodes': knee60_n, 'efficiency': knee60_e},
        'suggested_first_test_nodes': first_test,
        'suggested_omp_divisors': [c['omp'] for c in omp_candidates if c['legal']],
        'omp_candidate_detail': omp_candidates,
    }


# ---------------------------------------------------------------------------
# Section 6: physics candidates (PHYSICS-GRAY, single-axis, <=5, all UNCONFIRMED)
# ---------------------------------------------------------------------------

def _build_physics_candidates(resolved):
    cands = []

    def add(param, current, advice):
        if advice and len(cands) < 5:
            cands.append({
                'param': param, 'current': current, 'advice': advice,
                'classification': wc.classify(param), 'status': 'UNCONFIRMED',
            })

    add('radt', resolved.get('radt'), phys.radt_advice(resolved))
    add('time_step', resolved.get('time_step'), phys.timestep_advice(resolved))
    add('time_step_sound', resolved.get('time_step_sound', 4), phys.time_step_sound_advice(resolved))

    dx_km = float(resolved.get('dx', 3000)) / 1000.0
    cu = resolved.get('cu_physics')
    cu_legal = wc.lookup('cu_physics')
    cu_legal_values = cu_legal['legal_values']['values'] if cu_legal else []
    if cu not in (0, None) and dx_km < 4.0:
        add('cu_physics', cu, 'dx={:.2f}km < 4km (convection-permitting): cu_physics should be 0; '
                              'legal values {}'.format(dx_km, cu_legal_values))
    elif cu in (0, None) and 4.0 <= dx_km <= 10.0:
        add('cu_physics', cu, 'dx={:.2f}km gray zone: try a scale-aware scheme, e.g. 3 (Grell-Freitas) '
                              'or 11 (multiscale KF); legal values {}'.format(dx_km, cu_legal_values))
    elif cu in (0, None) and dx_km > 10.0:
        add('cu_physics', cu, 'dx={:.2f}km > 10km (parameterized regime): convection is under-resolved '
                              'with cu_physics=0; try e.g. 1 (KF) or 6 (Tiedtke); legal values {}'.format(
                                  dx_km, cu_legal_values))

    add('swint_opt', resolved.get('swint_opt', 0), phys.swint_advice(resolved))

    return cands[:5]


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------

PROFILE_KEYS = ('namelist', 'wrfinput', 'identity', 'physics', 'expected_cost_structure',
                'load_imbalance_drivers', 'core_count_priors', 'physics_candidates')


def build_profile(namelist_path, wrfinput_path=None, cores_per_node=DEFAULT_CORES_PER_NODE):
    nl = wt.parse_namelist(namelist_path)
    nl_kv = wt._parse_nl_kv(namelist_path)
    case = wt.extract_case(nl)
    case, enrich_err = wt.enrich_case(case, wrfinput_path)
    resolved, filled = phys.resolve_suite(case)

    doms, dom_i = _domain_table(nl_kv, resolved)
    we_list = [d['e_we'] for d in doms]
    sn_list = [d['e_sn'] for d in doms]

    exact, exact_note = _try_wrfinput_exact(wrfinput_path)

    return {
        'namelist': namelist_path,
        'wrfinput': wrfinput_path,
        'identity': _build_identity(resolved, doms, dom_i, exact, exact_note, enrich_err),
        'physics': _build_physics(resolved, filled),
        'expected_cost_structure': _build_cost_structure(resolved, doms, dom_i),
        'load_imbalance_drivers': _build_imbalance(resolved, nl_kv, doms, exact),
        'core_count_priors': _build_core_priors(we_list, sn_list, doms, dom_i, cores_per_node),
        'physics_candidates': _build_physics_candidates(resolved),
    }


def render_text(profile):
    L = []
    ident, phy, cost, imb, core = (profile['identity'], profile['physics'],
                                    profile['expected_cost_structure'],
                                    profile['load_imbalance_drivers'], profile['core_count_priors'])
    L.append("=== WTune case-understanding profile (PRIOR, no measurement) ===")
    L.append("namelist: {}".format(profile['namelist']))
    L.append("")
    L.append("IDENTITY: max_dom={} dominant=d{}  total_columns={} total_cells={}".format(
        ident['max_dom'], ident['dominant_domain'], ident['total_columns'], ident['total_cells']))
    for d in ident['domains']:
        L.append("  d{:02d}: {}x{} pts x {} lev, dx={:.0f}m, {:.1f}% compute share".format(
            d['index'], d['e_we'], d['e_sn'], d['e_vert'], d['dx_m'], d['compute_share_pct']))
    L.append("  start {} -> end~{} ({} h); center ({}, {}); season {}; doy {}".format(
        ident['start'], ident['end_estimate'], ident['run_total_hours'],
        ident['center_lat'], ident['center_lon'], ident['season'], ident['day_of_year']))
    L.append("")
    L.append("PHYSICS: suite={}  dx={}km -> {}".format(phy['suite'], phy['dx_km'], phy['dx_regime']))
    for line in phy['notes']:
        L.append(line)
    L.append("  " + phy['radt_advice'])
    L.append("  " + phy['timestep_advice'])
    L.append("")
    L.append("EXPECTED COST STRUCTURE ({})".format(cost['label']))
    for p in cost['phases_ranked']:
        L.append("  {:16s} weight={:<5} relative_cost={:<10} {}".format(
            p['phase'], p['weight'], p['relative_cost'], p['detail']))
    L.append("")
    L.append("LOAD-IMBALANCE DRIVERS")
    for line in imb['solar_terminator']:
        L.append("  " + line)
    for line in imb['nesting']:
        L.append("  " + line)
    for line in imb['physics_driven_flags']:
        L.append("  " + line)
    L.append("  land_fraction: {}".format(imb['land_fraction'] if imb['land_fraction'] is not None else imb['land_fraction_note']))
    L.append("")
    L.append("CORE-COUNT PRIORS ({})".format(core['label']))
    L.append("  R_max (omp=1) = {} ranks, shape {}x{} ({})".format(
        core['r_max_ranks_omp1'], core['r_max_shape']['nproc_x'], core['r_max_shape']['nproc_y'],
        core['r_max_note']))
    L.append("  nodes_at_R_max @ {} cores/node = {:.2f}".format(core['cores_per_node'], core['nodes_at_r_max']))
    for row in core['efficiency_table']:
        if row['legal']:
            L.append("    {:>3} nodes ({:>5} ranks): {}x{} patch {}x{}  E={:.3f}{}{}".format(
                row['nodes'], row['ranks'], row['nproc_x'], row['nproc_y'], row['patch_w'], row['patch_h'],
                row['efficiency_E'], "  <0.8" if row['below_0.8'] else "", "  <0.6" if row['below_0.6'] else ""))
        else:
            L.append("    {:>3} nodes ({:>5} ranks): NO LEGAL SHAPE".format(row['nodes'], row['ranks']))
    L.append("  knee E<0.8 at {} nodes (E={})".format(core['knee_below_0.8']['nodes'], core['knee_below_0.8']['efficiency']))
    L.append("  knee E<0.6 at {} nodes (E={})".format(core['knee_below_0.6']['nodes'], core['knee_below_0.6']['efficiency']))
    L.append("  suggest testing nodes: {}".format(core['suggested_first_test_nodes']))
    L.append("  suggest OMP thread counts (legal at {} nodes): {}".format(
        core['omp_candidate_detail'][0]['checked_at_nodes'] if core['omp_candidate_detail'] else '?',
        core['suggested_omp_divisors']))
    L.append("")
    L.append("PHYSICS CANDIDATES (PHYSICS-GRAY, single-axis, UNCONFIRMED)")
    for c in profile['physics_candidates']:
        L.append("  [{}] {} (current={}): {}".format(c['classification'], c['param'], c['current'], c['advice']))
    return L


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_case_inputs(case_dir, namelist=None, wrfinput=None):
    """Resolve (namelist_path, wrfinput_path) for a case directory: explicit args win; otherwise
    look for namelist.input and the first wrfinput_d0N in case_dir. wrfinput is optional."""
    nl = namelist or os.path.join(case_dir, 'namelist.input')
    wi = wrfinput
    if wi is None and case_dir and os.path.isdir(case_dir):
        for name in sorted(os.listdir(case_dir)):
            if name.startswith('wrfinput_d0'):
                wi = os.path.join(case_dir, name)
                break
    return nl, wi


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog='wtune_case.py', description='case-understanding profile (PRIOR, pre-measurement)')
    p.add_argument('case_dir', help='directory with namelist.input (+ optional wrfinput_d0N)')
    p.add_argument('--namelist', default=None, help='explicit namelist.input path (default: <case_dir>/namelist.input)')
    p.add_argument('--wrfinput', default=None, help='explicit wrfinput/wrfout path (default: first wrfinput_d0N in case_dir)')
    p.add_argument('--cores-per-node', type=int, default=DEFAULT_CORES_PER_NODE, help='default 72 (GH200 Grace)')
    p.add_argument('--json', default=None, help='also write the full profile as JSON to this path')
    args = p.parse_args(argv)

    namelist, wrfinput = find_case_inputs(args.case_dir, args.namelist, args.wrfinput)
    if not os.path.exists(namelist):
        print("ERROR: no namelist found at {} (pass --namelist)".format(namelist), file=sys.stderr)
        return 1

    profile = build_profile(namelist, wrfinput, args.cores_per_node)
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(profile, fh, indent=2)
        print('wrote {}'.format(args.json))
    for line in render_text(profile):
        print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
