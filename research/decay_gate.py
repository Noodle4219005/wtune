#!/usr/bin/env python3
"""Decide, at a segment boundary, whether to trust the MEASUREMENT or bring in the model.

The controller already has imbalance_gate: "is the imbalance big enough to pay 11.6 s for?".
This is the other half: "whose estimate of the imbalance should we act on?".

WHY A GATE IS NEEDED AT ALL

Measured across three regimes and two lags, neither source wins everywhere:

    case            lag    A raw sample   B model   C both   D residual   best
    arctic_sbl       60s       90.7          69.3     89.4      89.8      A
    arctic_sbl      240s       87.7          72.8     86.9      87.7      A
    deep_conv_ot     60s       81.5          63.0     79.1      79.7      A
    deep_conv_ot    240s       63.0          63.5     70.7      68.7      C
    karina_cat4      60s       83.2          87.0     87.3      86.3      C
    karina_cat4     240s       75.0          87.8     87.8      90.0      D

The pattern is not the LEVEL of either source, it is the GAP between them:

    A leads B by >15 pts  ->  A alone wins; combining LOSES 1-2 pts
    A and B within 13 pts ->  combining wins by +4 to +15

Classic ensemble behaviour: a source with no independent information left to contribute is
just noise. So the gate compares the two estimates and only combines when they are close.

WHY IT CAN SELF-CALIBRATE

The quality of the measurement as a predictor -- the identity path -- decays at a rate that is
wildly case-dependent. At 480 s of lag, W=40 s windows retain:

    arctic_sbl (polar stable BL)   78.4%       <- almost static
    deep_conv_ot (deep convection) 58.2%
    karina_cat4 (hurricane)        44.4%

There is therefore NO fixed re-sampling interval. But the decay does not have to be assumed:
consecutive measurement windows measure it directly. Score window k against window k+1, k+2,
... and the decay curve falls out of data the run is producing anyway.

WINDOW WIDTH IS NOT FREE

Radiation fires every 60 steps (~36 s of wall at 0.6 s/step) and the solar terminator is the
dominant imbalance in the radiation-driven regimes. A window narrower than one radiation call
misses it -- while reporting a perfectly well-behaved cost field. Measured step at W=20->40 s:

    arctic_sbl   79.5 -> 85.8  (+6.3)
    deep_conv_ot 77.8 -> 86.5  (+8.7)
    karina_cat4  88.7 -> 90.3  (+1.6)   <- storm-driven, microphysics every step, no step

Hence DEFAULT_WINDOW_S = 40.
"""
import numpy as np

DEFAULT_WINDOW_S = 40.0      # >= one radiation call; see the module docstring
GAP_WIDE = 15.0              # measurement leads model by this much -> use it alone
GAP_CLOSE = 13.0             # within this -> combine
TOPFRAC = 0.10               # hot-spot definition: the most expensive 10% of ranks


def _hit(score, truth, topk):
    a = set(np.argsort(truth)[-topk:])
    b = set(np.argsort(score)[-topk:])
    return 100.0 * len(a & b) / topk


def windows_from_buckets(comp, bsec, window_s=DEFAULT_WINDOW_S):
    """Fold a rank x bucket matrix into consecutive windows of >= window_s.

    Bucket 0 and the final bucket are dropped. The final one absorbs everything past the end
    of the run (the shim clamps), and bucket 0 absorbs initialisation: charge() only splits a
    span across ONE bucket edge, so a span crossing several is charged entirely to the bucket
    it started in -- harmless for 0.6 s timesteps, not harmless for a long startup.
    """
    used = comp.sum(axis=0) > 0
    if used.sum() < 4:
        return np.zeros((comp.shape[0], 0)), 0.0
    last = int(np.where(used)[0].max())
    core = comp[:, 1:last]
    per = max(1, int(round(window_s / max(bsec, 1e-9))))
    nw = core.shape[1] // per
    if nw < 1:
        return np.zeros((comp.shape[0], 0)), 0.0
    w = core[:, :nw * per].reshape(core.shape[0], nw, per).sum(axis=2)
    return w, per * bsec


def measure_decay(windows, width_s, topk=None):
    """-> {'lags_s': [...], 'hit': [...], 'tau_s': float|None, 'n_pairs': int}

    hit[k] is the mean, over every source position, of how well a window predicts the window
    k+1 places later. hit[0] is one window of lag. This is the identity path's quality as a
    function of staleness, measured from the run itself.
    """
    nr, nw = windows.shape
    if nw < 3:
        return {'lags_s': [], 'hit': [], 'tau_s': None, 'n_pairs': 0, 'periodic': False, 'rise_pts': 0.0}
    topk = topk or max(1, int(round(TOPFRAC * nr)))
    lags, hits, npair = [], [], 0
    # Lag range must reach past one diurnal cycle when the run is long enough. Capping at 12
    # lags was fine for 12 h runs but would hide a 24 h re-correlation entirely -- and whether
    # the decay is monotonic or periodic decides whether the controller should re-measure often
    # or wait for the phase to align.
    # Cap the lag at nw/3 so every point averages over at least two thirds of the available
    # source positions. Without this the longest lags have only a handful of pairs, the tail
    # rattles, and a periodicity test fires on noise -- measured: 12 h arctic and deep_conv
    # both reported a "+5.6 / +8.9 pt re-correlation" that is entirely the last few lags
    # having almost no samples.
    kmax = min(max(2, nw // 3), 240)
    step = max(1, kmax // 40)
    for k in range(1, kmax + 1, step):
        vals = [_hit(windows[:, t], windows[:, t + k], topk)
                for t in range(nw - k)
                if windows[:, t].sum() > 0 and windows[:, t + k].sum() > 0]
        if not vals:
            continue
        lags.append(k * width_s)
        hits.append(float(np.mean(vals)))
        npair += len(vals)
    tau = None
    # A rising tail means the pattern is re-correlating, i.e. oscillating rather than decaying.
    # Fitting an exponential to that produces a confident tau describing a function the data
    # does not follow, so detect it and refuse.
    # Measure the rise from the MINIMUM, not from the middle of the series. The first version
    # compared max(tail) against tail[0], and since the rise happens after the minimum -- which
    # sits deep inside the tail -- tail[0] was always the tail's own maximum and the test could
    # never fire. It reported periodic=False on a 48 h curve that bottoms out at 42.9% around
    # 27.5 model-h and then climbs 8 consecutive lags to 46.2%.
    periodic = False
    rise_pts = 0.0
    if len(hits) >= 6:
        h = np.array(hits)
        lo = int(h.argmin())
        if lo < len(h) - 2:
            rise_pts = float(h[lo:].max() - h[lo])
            periodic = bool(rise_pts > 2.0)
    if len(hits) >= 3 and not periodic:
        # Fit hit(lag) = 100*exp(-lag/tau) above the random floor. Least squares in log space
        # on the excess over random, which is the part that actually decays.
        y = np.maximum(np.array(hits) - 100.0 * TOPFRAC, 1e-3)
        x = np.array(lags, float)
        try:
            slope, _ = np.polyfit(x, np.log(y), 1)
            if slope < -1e-9:
                tau = float(-1.0 / slope)
        except Exception:
            tau = None
    return {'lags_s': lags, 'hit': hits, 'tau_s': tau, 'n_pairs': npair,
            'periodic': periodic, 'rise_pts': round(rise_pts, 1)}


def project_identity(decay, horizon_s):
    """Extrapolate the identity path's quality to the boundary we actually care about.

    Prefer the measured point nearest the horizon; fall back to the exponential fit; fall back
    to the last measured value. Never extrapolate a fit beyond 3x its furthest measured lag --
    the decay curves are not exponential out in the tail (arctic_sbl is nearly flat, deep_conv
    is not), and a confident number from an unconstrained fit is worse than an honest one.
    """
    if not decay['hit']:
        return None, 'no measured windows'
    lags = np.array(decay['lags_s'], float)
    hits = np.array(decay['hit'], float)
    if horizon_s <= lags.max():
        return float(np.interp(horizon_s, lags, hits)), 'interpolated from measured decay'
    tau = decay['tau_s']
    if tau and horizon_s <= 3.0 * lags.max():
        v = 100.0 * TOPFRAC + (hits[0] - 100.0 * TOPFRAC) * np.exp(-horizon_s / tau)
        return float(max(v, 100.0 * TOPFRAC)), 'exp fit, tau=%.0f s' % tau
    return float(hits[-1]), 'held at the furthest measured lag (%.0f s)' % lags[-1]


def decay_gate(comp, bsec, horizon_s, model_hit_pct, window_s=DEFAULT_WINDOW_S):
    """The decision. -> dict with 'source' in {'measurement','hybrid','model'}.

    model_hit_pct is the MODEL's expected hot-spot hit rate for this case, e.g. the offline
    cross-case number for the scheme in play. Pass the number for the scheme, not a global
    average -- they differ by 13 points across PBL schemes.
    """
    w, width = windows_from_buckets(comp, bsec, window_s)
    if w.shape[1] < 3:
        return {'source': 'model', 'reason':
                'only %d usable windows of %.0f s -- not enough to measure decay' % (w.shape[1], window_s),
                'identity_pct': None, 'model_pct': model_hit_pct}
    d = measure_decay(w, width)
    ident, how = project_identity(d, horizon_s)
    if d.get('periodic'):
        how += ' -- WARNING: the decay curve RISES in its tail, so it is oscillating, not '\
               'decaying. tau is withheld and any exponential projection past the measured ' \
               'lags is invalid.'
    if ident is None:
        return {'source': 'model', 'reason': how,
                'identity_pct': None, 'model_pct': model_hit_pct}
    gap = ident - model_hit_pct
    if gap > GAP_WIDE:
        src, why = 'measurement', ('identity %.1f%% leads model %.1f%% by %.1f > %.0f -- '
                                   'combining measured 1-2 pts WORSE' % (ident, model_hit_pct, gap, GAP_WIDE))
    elif gap < -GAP_WIDE:
        src, why = 'model', ('model %.1f%% leads identity %.1f%% by %.1f -- the sample is stale'
                             % (model_hit_pct, ident, -gap))
    else:
        src, why = 'hybrid', ('identity %.1f%% and model %.1f%% within %.0f -- '
                              'combining measured +4 to +15' % (ident, model_hit_pct, GAP_CLOSE))
    return {'source': src, 'reason': why, 'identity_pct': round(ident, 1),
            'model_pct': model_hit_pct, 'gap_pts': round(gap, 1),
            'projection': how, 'window_s': width, 'n_windows': int(w.shape[1]),
            'decay_lags_s': d['lags_s'], 'decay_hit': [round(v, 1) for v in d['hit']],
            'tau_s': round(d['tau_s'], 1) if d['tau_s'] else None,
            'periodic': d.get('periodic', False),
            'horizon_s': horizon_s}


def load_bucket_csv(path):
    """-> (comp[rank, bucket], bucket_seconds). Bucket width is DERIVED, never assumed:
    diagnostic runs use 10 s and hardcoding 300 s silently produces garbage."""
    rows = np.genfromtxt(path, delimiter=',', names=True)
    nr = int(rows['rank'].max()) + 1
    nb = int(rows['bucket'].max()) + 1
    comp = np.zeros((nr, nb))
    for r, b, c in zip(rows['rank'], rows['bucket'], rows['comp_s']):
        comp[int(r), int(b)] = c
    bsec = 300.0
    if 't_start_s' in rows.dtype.names:
        ts = np.unique(rows['t_start_s'])
        dd = np.diff(np.sort(ts))
        dd = dd[dd > 0]
        if len(dd):
            bsec = float(np.median(dd))
    return comp, bsec


if __name__ == '__main__':
    import sys, json
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: decay_gate.py <bucket_load.csv> [horizon_s] [model_hit_pct]")
        sys.exit(2)
    comp, bsec = load_bucket_csv(sys.argv[1])
    horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 432.0     # one 3 h segment at 3 km
    mhit = float(sys.argv[3]) if len(sys.argv) > 3 else 46.9         # resid own, cross-case
    print(json.dumps(decay_gate(comp, bsec, horizon, mhit), indent=2))
