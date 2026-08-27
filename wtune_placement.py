"""Hotspot-driven rank->core placement for over-subscribed WRF (zero-source).

Idea (user, 2026-08-27): keep EQUAL-sized patches and a STATIC decomposition,
but over-decompose (F x more ranks than cores) and assign an UNEQUAL NUMBER of
equal patches to each core so that every core's TOTAL predicted load is balanced.
A day-heavy core gets fewer patches; a night-light core gets more -- but each
core's SUM of work is ~equal, so time-sharing does not create a new bottleneck.

The load weights come from the tuner's precomputed hotspot (analytic terminator
+ SpeedyWeather convection). Because the terminator is deterministic, this static
optimum avoids the measurement-lag / migration / thrash of reactive dynamic LB
(the mode every prior AMPI/reorder test used).

HONEST MARGIN: vehicle cost ~+7.4% vs prize ~8.82% -> best case ~ -1.4% (faster),
inside the ~2.5% cross-allocation variance. A geometric E-W pairing already did
NOT clearly win; this improves the *input* (all-phase load map), not the vehicle.
Emits an MPI rank->core map; APPLYING it is zero-source (launcher placement only).

stdlib only. Python 3.6.
"""
import json
import math
from pathlib import Path

import wtune_solar as wsolar

# Per-rank load = a cost-share-weighted sum of ALL tuner hotspot sources, not
# just day/night. Weights ~ each phase's share of step cost AND its spatial
# variability (a uniform phase does not create imbalance, so it lives in W_BASE).
#   - W_BASE: dynamics + LW + PBL + surface, ~uniform across the domain.
#   - W_RAD : SW radiation, day-only -> the terminator E-W driver.
#   - W_CONV: microphysics (Thompson, M-group hotspot) -> moisture/storm driver;
#             per the physics diagnosis this is a large cost share, so it is
#             weighted comparably to radiation, NOT as an afterthought.
# Extensible: add a term here (weight, per-rank field) for any new hotspot the
# tuner learns (e.g. terrain radiation, ERA5-init convection).
W_BASE = 1.0
W_RAD = 0.35     # SW radiation surcharge on a fully-sunlit column
W_CONV = 0.45    # microphysics surcharge in a fully-active convective column


def _patch_centers(px, py, bbox):
    """lon/lat center of each rank's patch. Rank index = iy*px + ix (WRF order:
    X fastest within a Y row). Returns list indexed by rank."""
    lat0, lat1, lon0, lon1 = bbox
    centers = []
    for iy in range(py):
        cy = lat0 + (lat1 - lat0) * (iy + 0.5) / py
        for ix in range(px):
            cx = lon0 + (lon1 - lon0) * (ix + 0.5) / px
            centers.append((cx, cy))
    return centers


def _day_fraction_at(lon, lat, doy, start_hr, run_hrs, n_time=13):
    decl = wsolar.declination_deg(doy)
    if n_time < 2:
        times = [start_hr + run_hrs / 2.0]
    else:
        times = [start_hr + run_hrs * i / (n_time - 1) for i in range(n_time)]
    days = sum(1 for t in times if wsolar.cos_zenith(lat, lon, t, decl) > 0)
    return days / float(len(times))


def _conv_at(lon, lat, points, radius=3.0):
    """Nearest-ish convective value: max within `radius` deg, else 0."""
    if not points:
        return 0.0
    best = 0.0
    for p in points:
        plon = p[0] - 360.0 if p[0] > 180 else p[0]
        if abs(plon - lon) <= radius and abs(p[1] - lat) <= radius:
            if p[2] > best:
                best = p[2]
    return best


# CASE-GENERAL phase load model: each WRF phase contributes cost_weight * a
# spatial driver, but ONLY when its scheme is active in this case. So the hotspot
# set is DERIVED from the namelist, not fixed -- a different testcase (cumulus on,
# GWD on, no radiation, ...) yields a different load map automatically.
# driver in {'uniform','solar_sw','moisture','terrain','landmask'}.
# cost_weight ~ that phase's share of step cost (rough KB priors; a per-case
# profile refines them). A uniform phase creates no imbalance -> folds into base.
PHASE_LOAD_MODEL = [
    ('dynamics',     lambda c: True,                              0.30, 'uniform'),
    ('microphysics', lambda c: (c.get('mp_physics') or 0) > 0,    0.30, 'moisture'),
    ('cumulus',      lambda c: (c.get('cu_physics') or 0) > 0,    0.15, 'moisture'),
    ('radiation_sw', lambda c: (c.get('ra_sw_physics') or 0) > 0, 0.15, 'solar_sw'),
    ('radiation_lw', lambda c: (c.get('ra_lw_physics') or 0) > 0, 0.10, 'uniform'),
    ('pbl',          lambda c: (c.get('bl_pbl_physics') or 0) > 0, 0.08, 'uniform'),
    ('surface',      lambda c: (c.get('sf_surface_physics') or 0) > 0, 0.05, 'landmask'),
    ('gwd',          lambda c: (c.get('gwd_opt') or 0) > 0,       0.08, 'terrain'),
]


def active_phases(case):
    """The phases actually on for this case -> (name, weight, driver)."""
    return [(n, w, d) for (n, act, w, d) in PHASE_LOAD_MODEL if act(case)]


def rank_loads(case, px, py, surrogate_json=None):
    """Case-general per-rank load: sum over ACTIVE phases of weight*driver.

    Drivers wired: uniform, solar_sw (terminator), moisture (surrogate).
    terrain/landmask need HGT/LANDMASK from wrfinput -> currently neutral (1.0)
    with the phase still counted; wire them when a terrain/land case needs it.
    Returns (loads, note-or-None). note flags any driver that fell back to neutral.
    """
    bbox = wsolar.domain_bbox(case)
    if bbox is None:
        return None, 'domain lat/lon unknown (need --wrfinput)'
    doy = wsolar.day_of_year(case.get('start_month'), case.get('start_day'))
    if doy is None:
        return None, 'start date unknown'
    start_hr = float(case.get('start_hour') or 0)
    run_hrs = float(case.get('run_total_hours') or 6) or 6.0

    points = []
    conv_norm = 1.0
    if surrogate_json and Path(surrogate_json).exists():
        try:
            field = json.loads(Path(surrogate_json).read_text())
            points = field.get('points', [])
            mx = max((p[2] for p in points), default=0.0)
            conv_norm = mx if mx > 1e-9 else 1.0
        except Exception:
            points = []

    phases = active_phases(case)
    neutral_drivers = set()

    def driver_value(driver, lon, lat):
        if driver == 'uniform':
            return 1.0
        if driver == 'solar_sw':
            return _day_fraction_at(lon, lat, doy, start_hr, run_hrs)
        if driver == 'moisture':
            if not points:
                neutral_drivers.add('moisture(no surrogate)')
                return 0.0
            return _conv_at(lon, lat, points) / conv_norm
        if driver in ('terrain', 'landmask'):
            neutral_drivers.add(driver + '(needs HGT/LANDMASK)')
            return 0.0
        return 0.0

    centers = _patch_centers(px, py, bbox)
    loads = []
    for lon, lat in centers:
        load = 0.0
        for name, w, driver in phases:
            load += w * (1.0 if driver == 'uniform' else driver_value(driver, lon, lat))
        # uniform phases already add their flat weight; spatial phases add 0..w
        loads.append(load)
    note = None
    if neutral_drivers:
        note = 'neutral drivers (not imbalance-driving here): ' + ', '.join(sorted(neutral_drivers))
    return loads, note


def balance_assignment(loads, ncore):
    """Assign each rank (index) to a core, balancing per-core total load.

    Longest-Processing-Time (LPT) greedy: sort ranks by load desc, put each on
    the currently-lightest core. Good, simple makespan minimizer.
    Returns (rank_to_core list, core_loads list).
    """
    order = sorted(range(len(loads)), key=lambda i: -loads[i])
    core_loads = [0.0] * ncore
    rank_to_core = [0] * len(loads)
    import heapq
    heap = [(0.0, c) for c in range(ncore)]
    heapq.heapify(heap)
    for i in order:
        load, c = heapq.heappop(heap)
        rank_to_core[i] = c
        load += loads[i]
        core_loads[c] = load
        heapq.heappush(heap, (load, c))
    return rank_to_core, core_loads


def snake_order(px, py):
    """Boustrophedon (snake) row-major rank order -> spatial locality preserved.
    Rank index = iy*px + ix (WRF order). Alternate X direction each row so
    consecutive ranks in the order are physically adjacent."""
    order = []
    for iy in range(py):
        xs = range(px) if iy % 2 == 0 else range(px - 1, -1, -1)
        for ix in xs:
            order.append(iy * px + ix)
    return order


def balance_assignment_local(loads, ncore, px, py):
    """Locality-preserving: cut the snake-ordered ranks into `ncore` CONTIGUOUS
    segments of ~equal total load. A core owns one contiguous local block; its
    rank-COUNT grows where per-rank load is light (user's 1:1 day / N:1 night).
    Returns (rank_to_core, core_loads).
    """
    order = snake_order(px, py)
    total = sum(loads)
    target = total / float(ncore)
    rank_to_core = [0] * len(loads)
    core_loads = [0.0] * ncore
    cum = 0.0   # cumulative load BEFORE the current rank -> core = floor(cum/target)
    for r in order:
        c = int(cum / target)
        if c >= ncore:
            c = ncore - 1
        rank_to_core[r] = c
        core_loads[c] += loads[r]
        cum += loads[r]
    return rank_to_core, core_loads


def counts_per_core(rank_to_core, ncore):
    from collections import Counter
    cnt = Counter(rank_to_core)
    return [cnt.get(c, 0) for c in range(ncore)]


def imbalance(core_loads):
    n = len(core_loads)
    mean = sum(core_loads) / n
    if mean <= 0:
        return 0.0
    return max(core_loads) / mean - 1.0   # fractional excess of the busiest core


def plan_placement(case, px, py, ncore, surrogate_json=None):
    """Full placement plan: loads -> balanced assignment -> predicted imbalance.

    Compares the hotspot-balanced assignment against WRF's default block map
    (contiguous ranks on the same core) to predict the imbalance recovered.
    """
    loads, err = rank_loads(case, px, py, surrogate_json)
    if loads is None:
        return {'ok': False, 'reason': err}
    nranks = len(loads)
    if nranks % ncore != 0:
        # allow non-integer over-decomposition but note it
        pass
    factor = nranks / float(ncore)

    # default block map: ranks 0..k-1 on core 0, etc. (contiguous -> clustered load)
    per = int(math.ceil(nranks / float(ncore)))
    block_core_loads = [0.0] * ncore
    for i in range(nranks):
        c = min(i // per, ncore - 1)
        block_core_loads[c] += loads[i]

    rank_to_core, bal_core_loads = balance_assignment(loads, ncore)

    return {
        'ok': True,
        'nranks': nranks,
        'ncore': ncore,
        'over_decomp_factor': round(factor, 3),
        'imbalance_default_block': round(imbalance(block_core_loads), 4),
        'imbalance_hotspot_balanced': round(imbalance(bal_core_loads), 4),
        'rank_to_core': rank_to_core,
        'load_min': round(min(loads), 4),
        'load_max': round(max(loads), 4),
    }


def plan_placement_nodeaware(case, px, py, nnodes, cores_per_node,
                             surrogate_json=None):
    """Node-local balanced placement: MPI puts ranks n*rpn..(n+1)*rpn-1 on node n
    (rpn = nranks/nnodes). Balance EACH node's ranks across its own cores (0..
    cores_per_node-1), locality-preserving (contiguous cumulative-load cut). Each
    node owns full E-W rows, so within-node day/night balancing is possible.
    Returns {ok, rank_to_core (local core ids), imbalance_default, imbalance_balanced}.
    """
    loads, err = rank_loads(case, px, py, surrogate_json)
    if loads is None:
        return {'ok': False, 'reason': err}
    nranks = len(loads)
    if nranks % nnodes != 0:
        return {'ok': False, 'reason': 'nranks {} not divisible by nnodes {}'.format(nranks, nnodes)}
    rpn = nranks // nnodes
    rank_to_core = [0] * nranks
    bal_core_loads = []
    blk_core_loads = []
    per = int(math.ceil(rpn / float(cores_per_node)))
    for n in range(nnodes):
        node_ranks = list(range(n * rpn, (n + 1) * rpn))
        node_loads = [loads[r] for r in node_ranks]
        # LPT-balance WITHIN the node: inter-node comm stays local (node owns a
        # contiguous chunk), intra-node pairing (day+night on a core) is cheap
        # shared memory. Best of both -> good balance without network scatter.
        local_r2c, cl = balance_assignment(node_loads, cores_per_node)
        for j, r in enumerate(node_ranks):
            rank_to_core[r] = local_r2c[j]
        bal_core_loads += [x for x in cl if x > 0]
        # default block map within node (contiguous per-core, no load awareness)
        bcl = [0.0] * cores_per_node
        for i, r in enumerate(node_ranks):
            bcl[min(i // per, cores_per_node - 1)] += loads[r]
        blk_core_loads += [x for x in bcl if x > 0]
    return {
        'ok': True, 'nranks': nranks, 'rpn': rpn,
        'over_decomp_factor': round(nranks / float(nnodes * cores_per_node), 3),
        'imbalance_default_block': round(imbalance(blk_core_loads), 4),
        'imbalance_hotspot_balanced': round(imbalance(bal_core_loads), 4),
        'rank_to_core': rank_to_core,
    }


def plan_dynamic_schedule(case, px, py, nnodes, cores_per_node, nwin,
                          start_hour, run_hours, surrogate_json=None):
    """Time-phased placement: split the run into nwin windows, compute the
    node-aware balanced placement for EACH window's terminator position, and
    return per-rank [core_win0..core_win{nwin-1}]. blockbar7 re-pins per window.
    """
    nranks = px * py
    schedule = [[0] * nwin for _ in range(nranks)]
    residuals = []
    churn = []
    prev = None
    for w in range(nwin):
        t0 = start_hour + run_hours * w / float(nwin)
        t1 = start_hour + run_hours * (w + 1) / float(nwin)
        wc = dict(case)
        wc['start_hour'] = t0
        wc['run_total_hours'] = (t1 - t0)
        p = plan_placement_nodeaware(wc, px, py, nnodes, cores_per_node, surrogate_json)
        if not p['ok']:
            return {'ok': False, 'reason': p.get('reason')}
        r2c = p['rank_to_core']
        for r in range(nranks):
            schedule[r][w] = r2c[r]
        residuals.append(p['imbalance_hotspot_balanced'])
        if prev is not None:
            churn.append(sum(1 for a, b in zip(prev, r2c) if a != b) / float(nranks))
        prev = r2c
    return {'ok': True, 'nwin': nwin, 'schedule': schedule,
            'residual_per_window': [round(x, 4) for x in residuals],
            'churn_between_windows': [round(x, 3) for x in churn]}


def emit_schedule(schedule, path):
    """Write blockbar7 schedule: 'grank c0 c1 ... c[nwin-1]' per rank."""
    lines = ["# grank core_per_window (dynamic re-pin schedule)"]
    for r, cores in enumerate(schedule):
        lines.append("{} {}".format(r, " ".join(str(c) for c in cores)))
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def emit_rankfile(rank_to_core, path, cores_per_node=112):
    """Write an Intel-MPI/OpenMPI-style rank->core map.

    Format: one line per rank `rank <r>=<host?> slot=<core>` is launcher-specific;
    here we emit a simple `rank core` table plus an I_MPI_PIN_PROCESSOR_LIST hint.
    The run script converts this to the launcher's actual placement flag.
    """
    lines = ["# rank core  (hotspot-balanced placement; apply via launcher pinning)"]
    for r, c in enumerate(rank_to_core):
        lines.append("{} {}".format(r, c))
    Path(path).write_text("\n".join(lines) + "\n")
    return path
