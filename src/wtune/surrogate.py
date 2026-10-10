"""Pluggable surrogate layer: the CONVECTIVE (microphysics) hotspot.

The analytic terminator map (wtune_solar) handles the dominant, deterministic
radiation driver. The convective/storm hotspot moves chaotically and needs the
dynamics evolved -- that is this layer's job.

Providers (pluggable):
  - "speedy" : a fast global simplified-physics GCM (SpeedyWeather.jl) run over
               the window -> where convection concentrates and how it moves.
  - (future) "ic": read wrfinput convective indicators (case-faithful t=0).

FIDELITY CAVEAT (must be surfaced): SpeedyWeather from an idealized/prescribed
initial state gives a dynamically-plausible convection pattern, NOT this case's
exact storm. Case-accurate convection needs analysis (ERA5) initial conditions.
Use it for the SHAPE/MAGNITUDE of convective imbalance, not the exact storm cell.

This module only ORCHESTRATES + SAMPLES; the physics runs in Julia. stdlib only.
"""
import json


# ---------------------------------------------------------------------------
# SpeedyWeather physics-context probe (speedy_probe.jl) -> tuner insight.
# The probe returns DOMAIN AGGREGATES (mean/p90/max) of the fields that map to
# the tuner's knob/hotspot decisions. This turns them into advice lines and a
# refined cloud-cap steering wind.
# ---------------------------------------------------------------------------

def load_probe(path):
    """Load a speedy_probe.jl JSON; return dict or None."""
    try:
        with open(path) as f:
            d = json.load(f)
        return d if str(d.get('provider', '')).startswith('speedyweather') else None
    except Exception:
        return None


def probe_steering_wind(probe):
    """Mid-level wind (m/s) from the probe, for the radt cloud-staleness cap."""
    if not probe:
        return None
    w = probe.get('wind_midlevel_ms') or {}
    return w.get('p90') or w.get('mean')


def analyze_probe(probe, case=None):
    """Map the SpeedyWeather physics-context probe to tuner-facing insight lines.
    With `case` (dx, radt, mp_physics, cu_physics, w_damping), also emits ACTIONABLE
    knob-appropriateness advice -- advisory only, the tuner never auto-applies."""
    if not probe:
        return ["(no valid speedy probe — run: julia --project=jlenv speedy_probe.jl ...)"]
    L = ["SpeedyWeather physics context (forecast over the run; idealized IC = regime+scale, not exact storm):"]
    w = probe.get('wind_midlevel_ms') or {}
    if w.get('mean') is not None:
        L.append("  mid-level wind: mean {} p90 {} max {} m/s -> use p90 as cloud-cap steering (forecast; "
                 "cross-checks wrfinput t=0 wind)".format(w.get('mean'), w.get('p90'), w.get('max')))
    cf = probe.get('convective_fraction')
    if cf is not None:
        if cf > 0.5:
            regime = "CONVECTIVE-dominated -> moving storm-blob microphysics hotspot; cu_physics=0 correct at dx<=4km"
        elif cf < 0.2:
            regime = "STRATIFORM-dominated -> large-scale precip, microphysics more spatially uniform"
        else:
            regime = "MIXED convective+stratiform"
        L.append("  convective fraction {} -> {}".format(cf, regime))
    ct = probe.get('cloud_top') or {}
    if ct.get('mean') is not None:
        L.append("  cloud_top mean {} -> {}".format(
            ct.get('mean'),
            "cloudy: radt cloud-optics staleness MATTERS (tighter radt cap)" if (ct.get('mean') or 0) > 0
            else "clear-ish: radt cloud-staleness mild"))
    # NOTE: cos_zenith / sw_down here are the run-END instantaneous field (leapfrog
    # last time level). For a dusk run they are ~0 at 00 UTC (night) and do NOT
    # capture the terminator SWEEP -> defer the terminator to the analytic,
    # time-integrated solar model (wtune_solar), not this snapshot.
    olr = probe.get('outgoing_longwave') or {}
    if olr.get('mean') is not None:
        L.append("  OLR mean {} W/m2 (run-end) -> radiation activity level; terminator SWEEP -> use the analytic "
                 "solar model, not the instantaneous cos_zenith snapshot".format(olr.get('mean')))
    shf = probe.get('sensible_heat_flux') or {}
    if shf.get('max') is not None:
        L.append("  sensible heat flux max {} W/m2 -> surface energy activity (surf_driver load proxy)".format(shf.get('max')))

    # ---- ACTIONABLE knob-appropriateness (advisory; needs `case` for dx/radt) ----
    if case is not None and cf is not None:
        dx_km = float(case.get('dx', 2500)) / 1000.0
        cu = case.get('cu_physics', 0)
        mp = case.get('mp_physics', 8)
        cloudy = (ct.get('mean') or 0) > 0
        wd = case.get('w_damping', 1)
        L.append("ACTIONABLE knob-appropriateness (advisory -- results-affecting knobs, you decide):")
        # cu_physics: convection resolved (dx<=4km, cu=0) vs parameterized
        if cf > 0.4 and dx_km > 4 and (cu in (0, None)):
            L.append("  cu_physics={} at dx={}km with conv_frac {} -> UNDER-resolved convection: a cumulus scheme "
                     "(cu_physics>0) is likely REQUIRED for correctness (not a speed change).".format(cu, dx_km, cf))
        elif cf > 0.4 and dx_km <= 4:
            L.append("  cu_physics={} at dx={}km, conv_frac {} -> convection-resolving regime, cu=0 correct "
                     "(confirmed by the forecast).".format(cu, dx_km, cf))
        # microphysics scheme cost vs regime (advise-only, never auto-swap)
        if mp in (8, 10, 28, 38, 40) and cf < 0.15:
            L.append("  mp_physics={} (double-moment) but conv_frac {} is STRATIFORM-dominated -> a single-moment "
                     "scheme (WSM6=6) MIGHT be within tolerance & cheaper; SCHEME SWAP = advise-only, gate against a "
                     "reference before ever using for speed.".format(mp, cf))
        elif mp in (8, 10, 28, 38, 40) and cf > 0.5:
            L.append("  mp_physics={} warranted: convective regime needs the double-moment microphysics; no cheaper "
                     "scheme is within tolerance here.".format(mp))
        # w_damping: strong convective updrafts -> keep the CFL guard
        if cf > 0.4 and wd in (1, True):
            L.append("  w_damping=1 -> KEEP: convective updrafts (conv_frac {}) can trip vertical CFL; the guard costs "
                     "~0 and prevents crashes.".format(cf))
        # cloud overlap / icloud relevance for radiation
        if cloudy:
            L.append("  cloudy domain -> icloud/cldovrlp (cloud-radiation coupling & overlap) MATTER for radiation "
                     "accuracy; do NOT set icloud=0 (that is a different climate, not a speed knob).")
        L.append("  NOTE: for THIS case the forecast CONFIRMS the config is already tight (convective + cloudy + strong "
                 "jet) -> no NEW physics speed lever. A CLEAR / WEAK-convection / COARSE-grid case is where the probe "
                 "would UNLOCK a coarser radt or a cheaper scheme.")
    return L


def speedy_vet(cand, probe, resolved):
    """Stage 2 of the funnel: cheap SpeedyWeather physics vet of an analytic candidate.
    Returns (verdict, reason). Physics-independent candidates PASS straight to WRF;
    physics-dependent ones are CONFIRMED or REJECTED from the forecast BEFORE any WRF run."""
    name = cand.get('name')
    if not cand.get('physics_dep'):
        return 'PASS->WRF', 'physics-independent (I/O/decomp/tiling); SpeedyWeather neutral -> test on WRF'
    # epssm is terrain-dependent, not probe-dependent -> vet BEFORE the probe gate.
    if name == 'epssm':
        return 'PASS->WRF', ('terrain relief is not in the probe -> require a 1h WRF smoke run (watch domain-max w '
                             'and CFL trips) before trusting any dt / time_step_sound increase')
    # SpeedyWeather validity floor: it is a GLOBAL synoptic-scale spectral model. Below ~1km
    # (LES / microscale, e.g. a WRF-Fire run) it has no representation of the flow at all, so
    # its convective_fraction/wind are not a regime proxy -> do not pretend a probe would help.
    dxm = float(resolved.get('dx', 2500))
    if dxm < 1000.0:
        return 'PASS->WRF', ('dx={:.0f}m is below SpeedyWeather validity (a global synoptic spectral model has no '
                             'LES/microscale flow) -> no cheap probe signal; analytic screen + WRF smoke only').format(dxm)
    if not probe:
        return 'NEED-PROBE', 'run speedy_probe.jl to vet this cheaply before WRF'
    wind = probe_steering_wind(probe)
    cf = probe.get('convective_fraction')
    if name == 'radt':
        dx = float(resolved.get('dx', 2500))
        cap = 3.0 * dx / ((wind or 12.0) * 60.0)
        cur = float(resolved.get('radt', 10))
        if cur >= cap * 0.9:
            return 'REJECT', 'cloud-cap ~{:.1f} min at forecast wind {:.0f} m/s; radt={} already at/over cap'.format(cap, wind or 12, int(cur))
        # swint_opt COUPLING: the cloud-cap is the cloud-DRIFT limit. WITHOUT swint_opt the SW
        # zenith staircase at the terminator is a SEPARATE, tighter limit on radt -> the coarse
        # radt is only safe up to the cloud-cap once swint_opt removes that staircase.
        swint_on = int(resolved.get('swint_opt') or 0) > 0
        newr = cap * 0.9
        save = 0.18 * (1.0 - cur / newr)      # radiation ~18% of compute; frequency cut by (1 - old/new)
        couple = '' if swint_on else ' -- PAIR with swint_opt=1 (else the terminator zenith staircase caps radt tighter than the cloud-cap)'
        return 'CONFIRM', ('cloud-cap ~{:.1f} min > radt={} -> raise to ~{:.0f} min; radiation is ~18% of compute so '
                           '~{:.0%} wall from fewer rad calls; WRF-gate{}').format(cap, int(cur), newr, save, couple)
    if name == 'cudt':
        if not (int(resolved.get('cu_physics') or 0) > 0):
            return 'REJECT', 'cu_physics=0 -> cudt has no effect'
        dx = float(resolved.get('dx', 10000))
        cap_min = min(60.0, max(5.0, 3.0 * dx / ((wind or 12.0) * 60.0)))
        if cf is not None and cf > 0.6:
            return 'REJECT', ('persistent deep convection (conv_frac {}): the cumulus tendency changes fast, so a '
                              'stale tendency over cudt shifts precip -- little headroom').format(cf)
        if cf is not None and cf < 0.4:
            return 'CONFIRM', ('stratiform/intermittent (conv_frac {}): quasi-equilibrium holds -> advective cap '
                               '~{:.0f} min is safe; WRF-gate the precip').format(cf, cap_min)
        return 'PASS->WRF', 'regime unclear; cap ~{:.0f} min from 3*dx/wind -> WRF-gate it'.format(cap_min)
    if name == 'swint_opt':
        return 'CONFIRM', 'terminator run -> SW zenith staircase is real; swint removes it (accuracy, ~0.1% cost)'
    if name == 'mp_physics':
        if cf is not None and cf > 0.5:
            return 'REJECT', 'convective regime (conv_frac {}) -> double-moment needed; no cheaper scheme within tolerance'.format(cf)
        if cf is not None and cf < 0.2:
            return 'CONFIRM', 'stratiform (conv_frac {}) -> single-moment MIGHT pass; WRF-gate it'.format(cf)
        return 'PASS->WRF', 'regime unclear from probe'
    if name == 'cu_physics':
        # cheaper cumulus scheme (BMJ) vs the plume schemes (KF/GF): safe only for weak/shallow,
        # non-organized convection. Deep organized convection needs the explicit plume closure.
        if cf is not None and cf > 0.6:
            return 'REJECT', ('organized deep convection (conv_frac {}): needs the full plume closure (KF/GF); a bulk '
                              'adjustment scheme (BMJ) mislocates the convective precip').format(cf)
        if cf is not None and cf < 0.35:
            return 'CONFIRM', ('weak/shallow convection (conv_frac {}): a cheap adjustment cumulus scheme MIGHT hold '
                               '-> saves the cu_driver plume cost; WRF-gate precip vs reference').format(cf)
        return 'PASS->WRF', 'cumulus regime unclear from probe -> WRF-gate the scheme swap vs reference precip'
    if name == 'e_vert':
        return 'PASS->WRF', ('vertical level count is not represented in a global spectral probe -> WRF-gate only; '
                             'RESULTS-AFFECTING (regrids PBL/cloud vertical structure), compare precip/PBL vs reference')
    if name == 'non_hydrostatic':
        # hydrostatic assumption valid iff horizontal >> vertical scale AND convection is
        # parameterized (not grid-resolved). dx is the primary criterion; cu_physics/conv_frac
        # confirm the convection is sub-grid. This is the biggest coarse-grid lever (measured
        # -6%/normal-step on Matthew 30km) -- NOT a free physics-independent PASS.
        dx = float(resolved.get('dx', 2500))
        cu = int(resolved.get('cu_physics') or 0)
        if dx < 10000.0:
            return 'REJECT', ('dx={:.0f}m < 10km: the grid resolves deep-convective vertical acceleration -> '
                              'hydrostatic drops the w-equation and mis-places grid-point storms').format(dx)
        if cu == 0:
            return 'PASS->WRF', ('dx>={:.0f}km but cu_physics=0 (convection nominally grid-resolved) -> hydrostatic is '
                                 'borderline; WRF smoke first, watch domain-max w / precip').format(dx / 1000.0)
        return 'CONFIRM', ('dx={:.0f}m>=10km with cu_physics={} (parameterized convection): hydrostatic assumption holds '
                           '-> drops the vertically-implicit acoustic/small_step solve (~10-15% of compute; measured '
                           '~-6%/normal-step on Matthew 30km); WRF-gate for CFL/precip').format(dx, cu)
    if name == 'time_step':
        return 'PASS->WRF', 'acoustic CFL is dynamics (not in the probe); test on WRF with a hard CFL gate'
    if name == 'use_adaptive_time_step':
        # gain = time-mean CFL headroom. Persistent domain-wide convection pins the domain-max
        # w-Courant -> ~0 headroom; intermittent/frontal convection leaves headroom to harvest.
        if cf is not None and cf > 0.6:
            return 'REJECT', ('persistent convection (conv_frac {}): domain-max w-Courant pinned near the limit '
                              '-> ~0 CFL headroom to harvest').format(cf)
        if cf is not None and cf < 0.4:
            return 'CONFIRM', ('intermittent/frontal convection (conv_frac {}): time-mean allowed dt > fixed dt '
                               '-> real headroom; WRF-gate the CFL time series').format(cf)
        return 'PASS->WRF', 'CFL headroom = the dt-allowed time series; probe the domain-max Courant, then WRF-gate'
    if name == 'diff_opt':
        # removing the only horizontal mixing -> 2dx energy accumulates -> spurious convection.
        # Worst in convective + complex-terrain regimes; a smooth weak-flow case may tolerate it.
        if cf is not None and cf > 0.4:
            return 'REJECT', ('convective (conv_frac {}): with no explicit horizontal diffusion, 2dx energy '
                              'accumulates -> spurious convection / CFL crash risk').format(cf)
        return 'PASS->WRF', ('low-conv MIGHT tolerate no explicit diffusion, but terrain relief is not in the probe '
                             '-> require a 1h smoke run (watch the 2dx tail of the w-spectrum) before trusting it')
    if name in ('h_sca_adv_order', 'h_mom_adv_order'):
        # 3rd-order advection adds hyperdiffusion at 2-4dx -> smooths small-scale structure.
        # Whether that fails the gate depends on how much SMALL-SCALE signal the case has.
        if cf is not None and cf > 0.4:
            return 'REJECT', ('convective (conv_frac {}): 3rd-order hyperdiffusion smooths updraft cores -> storms '
                              'weakened/displaced, precip changes; fails a convective gate').format(cf)
        if cf is not None and cf < 0.2:
            return 'CONFIRM', ('smooth/large-scale (conv_frac {}): little 2-4dx signal to diffuse -> 3rd-order likely '
                               'within tolerance AND cuts halo bytes; worth a WRF gate').format(cf)
        return 'PASS->WRF', 'advection-order tolerance is regime-dependent; gate on WRF'
    return 'PASS->WRF', 'no specific probe signal for this param'
