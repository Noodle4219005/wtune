"""wtune_oracle.py -- Faster-than-WRF computational predictor (Stage 0).

Predicts C_WRF(x, y, phase) from a WRF snapshot's model-state fields WITHOUT running
WRF. Data-dependent phases (radiation, microphysics, cumulus, PBL) come from the actual
cloud / moisture / instability fields; structural phases (dynamics, acoustic) from grid
geometry.

Grounded in this project's MEASURED ablations (do not re-derive):
  * radiation cost is mostly coszen (day/night), NOT cloud: removing all cloud dropped
    rad_driver only 0.8% -- RRTMG McICA samples every g-point regardless of cloud.
    -> radiation field = LW(uniform) + SW(day-only); cloud is a small second order term.
  * microphysics cost is Thompson BRANCH COMPLEXITY (which condensate pathways fire),
    not condensate mass -- a mixed-phase column is far dearer than a clear one.
  * small_step / rk_tend / scalar_tend are ~uniform (1.04-1.06) -> structural, flat.
  * an imbalanced phase's profile% UNDER-states it: it also induces halo_wait on the
    ranks that finish early. So the critical (slowest) rank is what to predict.

Extended 2026-08-28 for broader case coverage (see f1-build-real-wrf-case-from-scratch,
wtune-oracle-multicase-design-and-coverage-gaps):
  * micro RH-FALLBACK: a cold-start / analysis snapshot has hydrometeors = 0, which made
    the hydrometeor-based micro field blind. When no condensate is present we fall back to
    a mid-tropospheric RH cloud proxy so micro still tracks where clouds will form.
  * CUMULUS bucket (NEW): coarse (dx>=~10km) runs use parameterized convection (cu_driver)
    whose cost the resolved-micro model never captured. Driven by theta-e instability
    (a CAPE proxy). Active only when cu_on=True.
  * PBL bucket (NEW): the boundary-layer scheme iterates more in a deep daytime unstable
    convective PBL over land. Driven by land x day-fraction x instability.
  * These new-phase amplitudes are PROVISIONAL Stage-0 estimates -- there is no cu-on /
    coarse-grid truth yet (only CONUS-Nov & Maria, both convection-resolving). Stage 2
    will replace the amplitudes with surrogates trained on standalone kernel timings.

Backward compatibility: for a convection-resolving snapshot (cu_on=False, hydrometeors
present -- CONUS 2.5km / Maria 1km) the prediction is BYTE-for-byte the previous behaviour:
micro stays hydrometeor-based, and cumulus/pbl carry zero weight (absent from PHASE_WEIGHT).
"""
import os
import numpy as np

# cu-OFF (convection-resolving) weights -- MEASURED (job 1040022 / p0-amdahl). UNCHANGED,
# validated on CONUS 2.5km & Maria 1km. dynamics dominates & is uniform; imbalance lives
# in rad + micro.
PHASE_WEIGHT = {'dynamics': 0.60, 'radiation': 0.18, 'micro': 0.12, 'surf': 0.03, 'acoustic': 0.07}

# cu-ON (coarse, parameterized convection) weights -- PROVISIONAL Stage-0 estimate (no
# cu-on truth yet). Resolved micro is cheap at coarse dx, so its share moves to the
# cu_driver (cumulus) and the daytime convective PBL. Sums ~1.0. Replace once a coarse
# cu-on case is profiled.
PHASE_WEIGHT_CU = {'dynamics': 0.55, 'radiation': 0.18, 'micro': 0.05, 'cumulus': 0.10,
                   'pbl': 0.05, 'surf': 0.03, 'acoustic': 0.04}


def phase_weights(cu_on=False):
    """Per-phase amplitude dict for the case. cu_on picks the coarse (parameterized
    convection) weight set. The application layer should ultimately derive these from the
    namelist via analytic Amdahl; these are the Stage-0 defaults."""
    return dict(PHASE_WEIGHT_CU if cu_on else PHASE_WEIGHT)


def _norm(x):
    """Robust 0..1 normalisation (1st/99th pct) for a driver field."""
    x = np.asarray(x, float)
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def _read_strided_4d(variable, lev_stride, stride):
    """Read one 4-D WRF field contiguously, then downsample in memory.

    NetCDF4/HDF5 chunk caches can amplify a multidimensional strided hyperslab into
    repeated chunk decompression. The full time slice is one contiguous logical read;
    the small strided copy is deliberately a NumPy operation outside the NetCDF layer.
    """
    return variable[0, :, :, :][::lev_stride, ::stride, ::stride].copy()


def _read_strided_3d(variable, stride):
    """Read one time-indexed 3-D WRF field contiguously, then downsample in memory."""
    return variable[0, :, :][::stride, ::stride].copy()


def _derived_moisture(ds, stride, lev_stride, qvapor=None):
    """Cheap derived drivers for the RH-fallback micro, cumulus and PBL buckets, computed
    from the T/P/QVAPOR profile. Returns {} if the profile is unavailable.
      rh_cloud : mean RH in the 850-300 hPa layer (mid-trop cloudiness proxy).
      cape     : theta_e(surface) - theta_es(500 hPa), clipped >=0 (convective instability)."""
    need = ('T', 'P', 'PB')
    if not all(v in ds.variables for v in need):
        return {}
    if qvapor is None:
        if 'QVAPOR' not in ds.variables:
            return {}
        qvapor = _read_strided_4d(ds.variables['QVAPOR'], lev_stride, stride)
    P = (np.asarray(_read_strided_4d(ds.variables['P'], lev_stride, stride), float) +
         np.asarray(_read_strided_4d(ds.variables['PB'], lev_stride, stride), float))
    theta = np.asarray(_read_strided_4d(ds.variables['T'], lev_stride, stride), float) + 300.0
    qv = np.maximum(np.asarray(qvapor, float), 1e-8)
    Temp = theta * (P / 1.0e5) ** 0.2854
    es = 611.2 * np.exp(17.67 * (Temp - 273.15) / (Temp - 29.65))
    qs = 0.622 * es / np.maximum(P - es, 1.0)
    RH = np.clip(qv / qs, 0.0, 1.2)
    mid = (P < 85000) & (P > 30000)
    with np.errstate(invalid='ignore'):
        rh_cloud = np.nanmean(np.where(mid, np.clip(RH, 0, 1), np.nan), axis=0)
    rh_cloud = np.where(np.isfinite(rh_cloud), rh_cloud, 0.0)
    # theta-e (surface parcel) vs saturated theta-e at 500 hPa. clip the Lv*q/(cp*T)
    # exponent so a warm, near-saturated low level cannot overflow exp().
    ex = np.clip(2.5e6 * qv / (1004.0 * Temp), 0, 40)
    exs = np.clip(2.5e6 * qs / (1004.0 * Temp), 0, 40)
    thetae = theta * np.exp(ex)
    thetaes = theta * np.exp(exs)
    k5 = np.argmin(np.abs(P - 50000.0), axis=0)
    J, I = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[2]), indexing='ij')
    cape = np.maximum(thetae[0] - thetaes[k5, J, I], 0.0)
    return {'rh_cloud': rh_cloud, 'cape': cape}


def extract_features(path, stride=4, lev_stride=3, include_derived=True,
                     include_surface_fluxes=True):
    """Cheap O(Nz) column features from a WRF snapshot (wrfout/wrfrst/wrfinput). Strided for
    speed at Stage 0; the full-resolution vectorised version is Stage 3.

    Returns the hydrometeor water paths + Thompson branch signature (nspecies/mixed_phase),
    the coords (XLAT/XLONG), the surface drivers (LANDMASK/HFX/PBLH), and -- when the T/P
    profile is present -- the derived moisture drivers rh_cloud/cape used by the RH-fallback
    micro, cumulus and PBL buckets. `include_derived=False` skips the T/P/PB profile when
    active hydrometeors already determine the microphysics field, but retains the clear-snapshot
    RH fallback. `include_surface_fluxes=False` omits the unused HFX/PBLH fields."""
    import wrf_frame
    ds = wrf_frame.open_frame(path)      # netCDF4 or ADIOS2 BP, same surface
    qvapor = (_read_strided_4d(ds.variables['QVAPOR'], lev_stride, stride)
              if 'QVAPOR' in ds.variables else None)

    # Mass-weighted column path [kg/m^2]:  path = sum_k q_k * (MU+MUB) * |DNW_k| / g
    # A raw level sum (the previous behaviour) is CONFOUNDED WITH THE VERTICAL CONFIG: the same
    # atmosphere yields ~2.2x more at e_vert=100 than at e_vert=45 (more terms in the sum), and
    # p_top shifts it again. A model trained across e_vert would then "discover" that high-TCWV
    # cases are slow when it had only learned e_vert. MU/MUB are 2-D and DNW is 1-D, so this
    # costs essentially no extra I/O.
    G_ACCEL = 9.81
    _mu = _wgt = None
    if all(k in ds.variables for k in ('MU', 'MUB', 'DNW')):
        _mu = (_read_strided_3d(ds.variables['MU'], stride)
               + _read_strided_3d(ds.variables['MUB'], stride))          # (ny,nx), Pa
        _dnw = np.abs(np.asarray(ds.variables['DNW'][0, :], float))      # (nz,)
        _wgt = np.array([_dnw[k:k + lev_stride].sum()
                         for k in range(0, _dnw.size, lev_stride)])      # weight per sampled level

    def colsum(v):
        arr = (qvapor if v == 'QVAPOR'
               else (_read_strided_4d(ds.variables[v], lev_stride, stride)
                     if v in ds.variables else None))
        if arr is None:
            return None
        if _mu is None or _wgt is None:
            return arr.sum(axis=0)              # fallback (e_vert-confounded); kept for old files
        w = _wgt[:arr.shape[0]]
        return (arr * w[:, None, None]).sum(axis=0) * _mu / G_ACCEL

    f = {}
    for name, var in [('TCWV', 'QVAPOR'), ('LWP', 'QCLOUD'), ('RWP', 'QRAIN'),
                      ('IWP', 'QICE'), ('SWP', 'QSNOW'), ('GWP', 'QGRAUP')]:
        c = colsum(var)
        if c is not None:
            f[name] = c
    ref = f.get('TCWV')
    # hydrometeor activity signature: how many condensate SPECIES are active per column
    # (the Thompson branch-complexity proxy). 0 = clear, 5 = full mixed-phase + graupel.
    species = [f[k] for k in ('LWP', 'RWP', 'IWP', 'SWP', 'GWP') if k in f]
    if species:
        # ABSOLUTE threshold in path units (kg/m^2). A per-case-relative threshold
        # (s > s.max()*0.01) makes the feature depend on WHICH CASE the column is in: a
        # hurricane's graupel maximum suppresses nspecies everywhere else in that same case,
        # so the feature is not a function of the column and cannot transfer across cases.
        SPECIES_MIN_PATH = 1.0e-3          # 1 g/m^2 of condensate = "species active"
        thr = [(s > SPECIES_MIN_PATH) for s in species]
        f['nspecies'] = np.sum(thr, axis=0).astype(float)
        # mixed-phase flag: BOTH liquid (qc|qr) and ice (qi|qs|qg) present -> dearest branch
        liq = (f.get('LWP', 0) > 0) | (f.get('RWP', 0) > 0)
        ice = (f.get('IWP', 0) > 0) | (f.get('SWP', 0) > 0) | (f.get('GWP', 0) > 0)
        f['mixed_phase'] = (liq & ice).astype(float)
    else:
        f['nspecies'] = np.zeros_like(ref)
        f['mixed_phase'] = np.zeros_like(ref)
    for coord in ('XLAT', 'XLONG'):
        f[coord] = (_read_strided_3d(ds.variables[coord], stride)
                    if coord in ds.variables else None)
    # surface-layer / LSM cost drivers (2D): Noah LSM runs on LAND (dear) vs water (cheap);
    # the surface-layer Monin-Obukhov similarity iterates more where fluxes are strong/unstable.
    for s2d in ('LANDMASK', 'HFX', 'PBLH'):
        if include_surface_fluxes:
            f[s2d] = (_read_strided_3d(ds.variables[s2d], stride)
                      if s2d in ds.variables else None)
    # Derived moisture/instability drivers are required for cu-on and for a clear snapshot's
    # RH fallback. A resolved-hydrometeor cu-off snapshot does not consume this profile.
    hydrometeors_active = bool(species and any(np.asarray(s).max() > 0 for s in species))
    if include_derived or not hydrometeors_active:
        f.update(_derived_moisture(ds, stride, lev_stride, qvapor=qvapor))
    ds.close()
    return f


def phase_cost_fields(feats, doy, utc_hours, run_hours=0.0, n_time=9, cu_on=False):
    """Per-cell RELATIVE cost (0..~1) for each phase, from the features + solar geometry.
    Uses the snapshot's real XLAT/XLONG for exact coszen (not a bbox approximation).
    radiation/surf use the RUN-INTEGRATED day fraction over [utc_hours, utc_hours+run_hours]
    (the terminator sweeps the domain over the run -- a single instant mis-places it).

    Phase set (each phase = one physical cost driver; add a new phase here to let the oracle
    correct for a new cost source, then give it a weight in PHASE_WEIGHT[_CU]):
      radiation : coszen day-fraction (+ small cloud-optics bump)          [always]
      micro     : Thompson branch complexity; RH-cloud fallback if clear   [always]
      cumulus   : theta-e instability (parameterized convection)           [cu_on only]
      pbl       : daytime convective PBL over land                         [always emitted; 0 weight if unused]
      surf      : Noah LSM on land                                         [always]
      dynamics  : structural (uniform, measured)                          [always]
      acoustic  : structural (uniform, measured)                          [always]
    """
    import wtune_solar as ws
    decl = ws.declination_deg(doy)
    lat, lon = feats['XLAT'], feats['XLONG']
    times = [utc_hours] if run_hours <= 0 or n_time < 2 else \
        [utc_hours + run_hours * i / (n_time - 1) for i in range(n_time)]
    dayf = np.zeros_like(lat, dtype=float)
    for t in times:
        cz = np.vectorize(lambda la, lo: ws.cos_zenith(la, lo, t, decl))(lat, lon)
        dayf += (cz > 0).astype(float)
    dayf /= len(times)                       # time-mean sunlit fraction per cell

    # radiation: LW always on (uniform) + SW day-only. small cloud-optics bump.
    cloud_bump = 0.05 * np.tanh(feats.get('LWP', 0) + feats.get('IWP', 0))
    rad = 0.45 + 0.55 * dayf + cloud_bump

    # microphysics: Thompson branch complexity when condensate is resolved; when the
    # snapshot is clear (cold start / analysis, hydrometeors == 0) fall back to a mid-trop
    # RH cloud proxy so micro still tracks where clouds will form.
    nsp = feats.get('nspecies')
    if nsp is not None and np.asarray(nsp).max() > 0:
        micro = 0.12 + 0.6 * (nsp / 5.0) + 0.28 * feats['mixed_phase']
    elif feats.get('rh_cloud') is not None:
        micro = 0.12 + 0.60 * _norm(feats['rh_cloud'])
    else:
        micro = np.full_like(rad, 0.12)

    # surf_driver: Noah LSM only runs on LAND (water = cheap slab); land alone ranked the
    # measured surf-slowest rank at 76%ile (|HFX|/PBLH did not help). land-only.
    land = feats.get('LANDMASK')
    if land is not None:
        land = np.asarray(land, float)
        surf = 0.2 + 0.8 * land
    else:
        land = None
        surf = 0.3 + 0.7 * dayf

    ones = np.ones_like(rad)
    out = {'radiation': rad, 'micro': micro, 'dynamics': ones, 'acoustic': ones, 'surf': surf}

    # cumulus (NEW): parameterized-convection cost, only meaningful when cu_physics is ON
    # (coarse dx). Driven by theta-e convective instability (CAPE proxy).
    if cu_on and feats.get('cape') is not None:
        out['cumulus'] = 0.10 + 0.90 * _norm(feats['cape'])

    # pbl (NEW): the boundary-layer scheme iterates more in a deep daytime unstable
    # convective PBL over land. Emitted always; carries weight only in PHASE_WEIGHT_CU
    # (0 for convection-resolving cases, so it never perturbs the validated cu-off result).
    inst = _norm(feats['cape']) if feats.get('cape') is not None else dayf
    land_w = land if land is not None else np.zeros_like(rad)
    out['pbl'] = 0.3 + 0.7 * np.clip(0.5 * land_w + 0.5 * dayf, 0, 1) * (0.5 + 0.5 * inst)
    return out


def to_ranks(field, px, py):
    """Aggregate a per-cell field to the px*py MPI-rank grid (mean per patch)."""
    H, W = field.shape
    out = np.zeros((py, px))
    for j in range(py):
        for i in range(px):
            out[j, i] = field[j * H // py:(j + 1) * H // py, i * W // px:(i + 1) * W // px].mean()
    return out


def predict(path, px, py, doy, utc_hours, run_hours=0.0, stride=4, cu_on=False):
    """Full Stage-0 prediction: per-rank per-phase cost + total + the critical rank.
    Returns dict with 'phase' (name->rank grid), 'total' (rank grid), 'critical_rank',
    'hotspot' (WRF rank id, row-major x + y*px). cu_on selects the coarse weight set and
    activates the cumulus bucket."""
    feats = extract_features(path, stride=stride)
    fields = phase_cost_fields(feats, doy, utc_hours, run_hours=run_hours, cu_on=cu_on)
    w = phase_weights(cu_on)
    per_phase = {}
    total = np.zeros((py, px))
    for name, fld in fields.items():
        rk = to_ranks(fld, px, py)
        rk = rk / rk.mean() if rk.mean() > 0 else rk
        weight = w.get(name, 0.0)
        per_phase[name] = rk * weight
        total += rk * weight
    j, i = np.unravel_index(np.argmax(total), total.shape)
    return {'phase': per_phase, 'total': total, 'critical_rank': j * px + i,
            'critical_yx': (int(j), int(i))}


def _legal_partitions(ranks, e_we, e_sn, min_patch=10):
    """All (px, py) with px*py == ranks whose patches meet a minimum-cell floor."""
    out = []
    for px in range(1, ranks + 1):
        if ranks % px:
            continue
        py = ranks // px
        if (e_we - 1) // px >= min_patch and (e_sn - 1) // py >= min_patch:
            out.append((px, py))
    return out


def partition_sweep(path, ranks, doy, utc_hours, run_hours=0.0, stride=4,
                    e_we=None, e_sn=None, nodes=4, numa_domains=8, cu_on=False):
    """Stage 3: extract the per-cell phase-cost fields ONCE, then score EVERY legal
    nproc_x x nproc_y and return them ranked -- all without running WRF.

    The partition wall time is FIRST-ORDER structural (cache/vector/halo/NUMA) and only
    SECOND-ORDER physics imbalance: measured on CONUS, 8x56 beats 4x112 by ~12% for
    halo/cache reasons even though 4x112 has a slightly lower micro imbalance. So the
    score is `structural_cost x physics_criticality_proxy`:
      structural  = wtune.cost_model (ghost inflation x vec x NUMA), the dominant term
      physics     = Sum_phase weight * (max_rank/mean_rank), a PROXY correction

    IMPORTANT -- physics_crit is a phase-wise CRITICALITY PROXY, NOT a measured wall
    critical path. WRF's halo is nearest-neighbour RSL_LITE (Isend/Irecv/Wait); there is
    NO global MPI_Barrier per phase (measured this project: recv-wait dominates, sends are
    eager; RSL_LITE source posts-recv->send->wait->consume). So max/mean over ranks
    over-states the recoverable wall (the imbalance is smoothed by nearest-neighbour
    coupling; the perfect-balance prize is ~8%, not the raw max/mean). Use this only to
    RANK partitions, never as an absolute wall-critical-path claim; if the ranking ever
    disagrees with a measured wall sweep, trust the measurement."""
    import wtune as _w
    feats = extract_features(path, stride=stride)
    fields = phase_cost_fields(feats, doy, utc_hours, run_hours=run_hours, cu_on=cu_on)
    w = phase_weights(cu_on)
    if e_we is None:
        e_we = feats['XLAT'].shape[1] * stride + 1
    if e_sn is None:
        e_sn = feats['XLAT'].shape[0] * stride + 1
    nx, ny = e_we - 1, e_sn - 1
    rows = []
    for px, py in _legal_partitions(ranks, e_we, e_sn):
        crit, detail = 0.0, {}
        for name, fld in fields.items():
            rk = to_ranks(fld, px, py)
            imb = rk.max() / rk.mean() if rk.mean() > 0 else 1.0
            crit += w.get(name, 0.0) * imb
            detail[name] = round(imb, 3)
        structural = _w.cost_model(nx, ny, px, py, ranks, nodes, numa_domains)
        rows.append({'px': px, 'py': py,
                     'score': round(structural * crit, 4),
                     'structural': round(structural, 4),
                     'physics_crit': round(crit, 4), 'imbalance': detail})
    rows.sort(key=lambda r: r['score'])
    return rows


# Phase weight provenance: which phase weights are MEASURED (from the validated cu-off
# ablation) vs PROVISIONAL (cu-on cumulus/pbl, no coarse-grid per-phase truth yet). The
# spatial artifact carries this so a reader never mistakes a provisional map for a measured one.
PHASE_EVIDENCE = {
    'dynamics': 'MEASURED', 'radiation': 'MEASURED', 'micro': 'MEASURED',
    'surf': 'MEASURED', 'acoustic': 'MEASURED',
    'cumulus': 'PROVISIONAL', 'pbl': 'PROVISIONAL',
}


def _downsample(field, target=24):
    """Block-mean a 2D field down to <= target x target so the JSON grid stays compact.
    Ceil-division block size so BOTH axes end up at most `target` cells."""
    H, W = field.shape
    sy = max(1, -(-H // target)); sx = max(1, -(-W // target))   # ceil(H/target), ceil(W/target)
    ny = H // sy; nx = W // sx
    return field[:ny * sy, :nx * sx].reshape(ny, sy, nx, sx).mean(axis=(1, 3))


# structural phases are ~uniform (flat weight), so they dominate every cell's absolute cost
# but explain none of the SPATIAL variation -- the hotspot "driver" is the hottest VARIABLE phase.
_STRUCTURAL_PHASES = ('dynamics', 'acoustic')


def _dominant_driver(phase_contrib):
    """The hottest non-structural phase at a cell -- what actually makes this cell a hotspot."""
    var = {n: v for n, v in phase_contrib.items() if n not in _STRUCTURAL_PHASES}
    return max(var, key=var.get) if var else max(phase_contrib, key=phase_contrib.get)


def spatial_hotspot_artifact(path, doy, utc_hours, run_hours=0.0, stride=4,
                             cu_on=False, top_k=12, grid=24):
    """DIAGNOSTIC per-cell physics load map (NOT a partition selector).

    For each phase, the per-cell RELATIVE cost weighted by its phase weight; the top-k hottest
    cells with lat/lon and per-phase breakdown; and an evidence label per phase (MEASURED vs
    PROVISIONAL). INSPECTION ONLY: WRF keeps regular rectangular patches, and partition wall
    time is first-order structural (cache/vector/halo/NUMA) and only second-order physics
    imbalance -- so this map must NEVER drive static nproc. Returns a JSON-serialisable dict;
    the full-resolution numpy fields ride under '_fields' for an optional .npz and are stripped
    before JSON serialisation by emit_spatial_artifact()."""
    feats = extract_features(path, stride=stride)
    fields = phase_cost_fields(feats, doy, utc_hours, run_hours=run_hours, cu_on=cu_on)
    w = phase_weights(cu_on)
    lat = np.asarray(feats['XLAT'], float)
    lon = np.asarray(feats['XLONG'], float)

    weighted = {}
    total = np.zeros_like(lat, dtype=float)
    for name, fld in fields.items():
        fld = np.asarray(fld, float)
        m = fld.mean()
        norm = fld / m if m > 0 else fld           # per-phase relative, mean ~1
        weighted[name] = norm * w.get(name, 0.0)
        total += weighted[name]

    # micro field provenance: hydrometeor-based (measured driver) vs RH-fallback (weaker) vs flat
    nsp = feats.get('nspecies')
    micro_mode = ('hydrometeor' if (nsp is not None and np.asarray(nsp).max() > 0)
                  else ('rh-fallback' if feats.get('rh_cloud') is not None else 'flat-floor'))

    H, W = total.shape
    flat = total.ravel()
    k = int(min(top_k, flat.size))
    hot = []
    for r in np.argsort(flat)[::-1][:k]:
        y, x = divmod(int(r), W)
        contrib = {n: round(float(weighted[n][y, x]), 4) for n in weighted}
        hot.append({
            'y': y, 'x': x,
            'lat': round(float(lat[y, x]), 3), 'lon': round(float(lon[y, x]), 3),
            'weighted_cost': round(float(total[y, x]), 4),
            'dominant_driver': _dominant_driver(contrib),   # hottest VARIABLE (non-structural) phase
            'phase_contrib': contrib,
        })

    phases = {}
    for name in fields:
        f = weighted[name]
        phases[name] = {
            'weight': round(float(w.get(name, 0.0)), 3),
            'evidence': PHASE_EVIDENCE.get(name, 'PROVISIONAL'),
            'field_stats': {'mean': round(float(f.mean()), 4),
                            'p90': round(float(np.quantile(f, 0.9)), 4),
                            'max': round(float(f.max()), 4)},
            'grid': [[round(float(v), 4) for v in row] for row in _downsample(f, grid)],
        }

    any_provisional = any(PHASE_EVIDENCE.get(n) == 'PROVISIONAL' and w.get(n, 0) > 0
                          for n in fields)
    art = {
        'artifact': 'wtune-oracle-spatial-hotspot',
        'usage': ('DIAGNOSTIC ONLY -- shows WHERE physics load concentrates. Do NOT use to '
                  'auto-select nproc_x/nproc_y: WRF uses regular patches and partition wall '
                  'time is first-order structural, not physics imbalance (see partition_sweep). '
                  'Use partition_sweep for nproc, and measurement for truth.'),
        'meta': {'case': str(path), 'doy': doy, 'utc_hours': utc_hours,
                 'run_hours': run_hours, 'cu_on': bool(cu_on), 'stride': stride,
                 'grid_shape': [int(H), int(W)], 'weight_set': 'CU' if cu_on else 'BASE',
                 'micro_mode': micro_mode},
        'confidence': {
            'level': 'MODEL-ONLY',
            'provisional_phases_active': bool(any_provisional),
            'note': ('cu-on cumulus/pbl weights are PROVISIONAL (no coarse-grid per-phase '
                     'truth yet -- keep as low-confidence)' if any_provisional else
                     'all active phase weights are from the validated cu-off ablation'),
            'micro_field': micro_mode,
        },
        'top_k': hot,
        'phases': phases,
        '_fields': dict(weighted, total=total, lat=lat, lon=lon),
    }
    return art


def emit_spatial_artifact(art, json_path=None, npz_path=None):
    """Write the artifact set with per-file atomic replacement and rollback on errors.

    A killed process can still interrupt between the two replacements, but an exception
    during publication removes newly written files and restores pre-existing targets.
    Returns a short receipt.
    """
    import json
    fields = art.pop('_fields', {})
    rcpt = []
    published = []
    backups = []
    temps = []
    failed = False

    def backup_existing(path):
        if not os.path.exists(path):
            return
        suffix = 0
        while True:
            backup = '{}.bak{}.{}'.format(path, os.getpid(), suffix)
            if not os.path.exists(backup):
                break
            suffix += 1
        os.replace(path, backup)
        backups.append((path, backup))

    def publish(path, writer, suffix):
        backup_existing(path)
        tmp = '{}.tmp{}.{}'.format(path, os.getpid(), suffix)
        temps.append(tmp)
        writer(tmp)
        os.replace(tmp, path)
        temps.remove(tmp)
        published.append(path)

    try:
        if npz_path:
            publish(npz_path,
                    lambda tmp: np.savez_compressed(
                        tmp, **{k: np.asarray(v) for k, v in fields.items()}),
                    'npz')
            rcpt.append('npz={} ({} fields)'.format(npz_path, len(fields)))
        if json_path:
            def write_json(tmp):
                with open(tmp, 'w') as fh:
                    json.dump(art, fh, indent=2, sort_keys=True)
            publish(json_path, write_json, 'json')
            rcpt.append('json={}'.format(json_path))
    except Exception:
        failed = True
        for path in reversed(published):
            try:
                os.unlink(path)
            except OSError:
                pass
        for path, backup in reversed(backups):
            if os.path.exists(backup):
                try:
                    os.replace(backup, path)
                except OSError:
                    pass
        raise
    finally:
        for tmp in temps:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if not failed:
            for unused_path, backup in backups:
                try:
                    os.unlink(backup)
                except OSError:
                    pass
        art['_fields'] = fields
    return '; '.join(rcpt) if rcpt else 'no output path given'
