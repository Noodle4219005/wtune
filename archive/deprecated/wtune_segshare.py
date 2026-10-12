#!/usr/bin/env python3
"""Segment-share model: the last measured quantity still leaking into the cost composite.

The two-stage cost model predicts

    rate[r] = scale(config) x share[segment] x shape(weather, geometry)[r]

and `scale` and `shape` are both models -- but `share` (what fraction of the run's mean cost
rate a given segment runs at) was taken from the MEASUREMENT. That is a leak: the composite
8.3% number was partly reading the answer.

What could drive share at all? Over 12 h the solar terminator sweeps 180 degrees, so the
sunlit fraction of a fixed domain changes segment to segment, and shortwave radiation only
runs where the sun is up. That is computable ANALYTICALLY from namelist.wps geometry plus the
start date -- no model run, no wrfout. Convective spin-up is the other candidate and is not
a priori knowable, so it is tested separately via the segment index and via the frame itself.

Models compared (all leave-one-case-out):
  const     share = 1                     -- the honest null; if it wins, share does not matter
  solar     analytic sunlit fraction      -- A PRIORI, needs nothing but the namelist
  frame     domain-mean of the segment's own radiation/micro columns   -- needs the frame
  segidx    linear in segment index       -- pure spin-up trend
  persist   share[k] = share[k-1]         -- what an in-flight controller can always do

usage: wtune_segshare.py [case ...]
"""
import sys, os, glob, re, math, datetime
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
import wtune_solar as ws

SEG_HOURS = 3.0
NSEG = 4
GRID = 11            # coarse domain sample; the terminator is a smooth large-scale field


def wps_geometry(case):
    """-> (ref_lat, ref_lon, half_extent_km, start_datetime) from namelist.wps, or None."""
    p = f'{seg.BENCH}/{case}/namelist.wps'
    if not os.path.exists(p):
        return None
    t = open(p).read()
    g = lambda k, cast=float: (cast(re.search(k + r'\s*=\s*([-0-9.]+)', t).group(1))
                               if re.search(k + r'\s*=\s*([-0-9.]+)', t) else None)
    lat, lon, dx, ewe = g('ref_lat'), g('ref_lon'), g('dx'), g('e_we', int)
    m = re.search(r"start_date\s*=\s*'(\d{4})-(\d{2})-(\d{2})_(\d{2})", t)
    if None in (lat, lon, dx, ewe) or not m:
        return None
    y, mo, d, h = map(int, m.groups())
    return lat, lon, (ewe - 1) * dx / 2000.0, datetime.datetime(y, mo, d, h)


def sunlit(case, k):
    """Mean sunlit fraction and mean cos(zenith) over the domain during segment k.

    Purely analytic: domain geometry from namelist.wps, sun position from the date. Nothing
    here has seen the model run, which is exactly what makes it usable before submitting.
    """
    geo = wps_geometry(case)
    if geo is None:
        return None
    lat0, lon0, half_km, t0 = geo
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(0.2, math.cos(math.radians(lat0))))
    lats = np.linspace(max(-89.5, lat0 - dlat), min(89.5, lat0 + dlat), GRID)
    lons = np.linspace(lon0 - dlon, lon0 + dlon, GRID)
    doy = t0.timetuple().tm_yday
    decl = ws.declination_deg(doy)
    frac, cz_sum, n = 0.0, 0.0, 0
    for i in range(5):                                    # 5 sub-times across the 3 h window
        utc = t0.hour + k * SEG_HOURS + SEG_HOURS * i / 4.0
        for la in lats:
            for lo in lons:
                cz = ws.cos_zenith(la, lo, utc % 24.0, decl)
                frac += 1.0 if cz > 0 else 0.0
                cz_sum += max(0.0, cz)
                n += 1
    return frac / n, cz_sum / n


def build(case):
    r, why = seg.build_case(case)
    if r is None:
        return None
    X, y, cs, sg, names = r
    nl = f'{seg.RESULTS}/{case}/namelist.input'
    txt = open(nl).read() if os.path.exists(nl) else ''
    m = re.search(r'time_step=(\d+)', txt)
    dt = int(m.group(1)) if m else 15
    rate = cs / (SEG_HOURS * 3600.0 / dt)
    ks = sorted(np.unique(sg))
    share, sun, cz, fr_rad, fr_mic = [], [], [], [], []
    for k in ks:
        msk = sg == k
        share.append(rate[msk].mean())
        s = sunlit(case, int(k))
        if s is None:
            return None
        sun.append(s[0]); cz.append(s[1])
        fr_rad.append(X[msk, names.index('radiation')].mean())
        fr_mic.append(X[msk, names.index('micro')].mean())
    share = np.array(share)
    return dict(case=case, ks=np.array(ks, float), share=share / share.mean(),
                mean_rate=rate.mean(), sun=np.array(sun), cz=np.array(cz),
                fr_rad=np.array(fr_rad), fr_mic=np.array(fr_mic))


def loco_lin(rows, key, target='share'):
    """Leave-one-case-out linear fit of share on one predictor, pooled across cases."""
    pred = []
    for i in range(len(rows)):
        Xtr = np.concatenate([r[key] for j, r in enumerate(rows) if j != i])
        ytr = np.concatenate([r[target] for j, r in enumerate(rows) if j != i])
        A = np.polyfit(Xtr, ytr, 1) if Xtr.std() > 1e-9 else [0.0, ytr.mean()]
        pred.append(np.polyval(A, rows[i][key]))
    return pred


def main():
    want = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(f'{seg.RESULTS}/*')
                                  if os.path.isdir(d) and not os.path.basename(d).startswith('_'))
    rows = [r for r in (build(c) for c in want) if r]
    if len(rows) < 4:
        print("need >= 4 cases; got %d" % len(rows)); return 2

    print("=== 實測 segment share（各段平均率 / 全程平均率）===")
    print("%-18s %7s %7s %7s %7s %8s   %s" % ("case", "seg0", "seg1", "seg2", "seg3", "極差", "日照比例 seg0..3"))
    for r in rows:
        print("%-18s %7.3f %7.3f %7.3f %7.3f %8.1f%%   %s"
              % (r['case'], *r['share'], 100 * (r['share'].max() - r['share'].min()),
                 " ".join("%.2f" % v for v in r['sun'])))
    allsh = np.concatenate([r['share'] for r in rows])
    print("\n全部 %d 個分段：share %.3f–%.3f，標準差 %.3f，平均極差 %.1f%%"
          % (len(allsh), allsh.min(), allsh.max(), allsh.std(),
             100 * np.mean([r['share'].max() - r['share'].min() for r in rows])))

    print("\n=== 預測 share（LOCO，MAE 以 share 單位，1.000 = 全程平均）===")
    truth = [r['share'] for r in rows]
    results = {}
    results['const=1'] = [np.ones_like(r['share']) for r in rows]
    results['solar 日照比例'] = loco_lin(rows, 'sun')
    results['solar cos(zenith)'] = loco_lin(rows, 'cz')
    results['frame radiation'] = loco_lin(rows, 'fr_rad')
    results['frame micro'] = loco_lin(rows, 'fr_mic')
    results['segment index'] = loco_lin(rows, 'ks')
    pers = []
    for r in rows:                                   # in-flight: reuse the previous segment
        p = np.empty_like(r['share']); p[0] = 1.0; p[1:] = r['share'][:-1]
        pers.append(p)
    results['persistence'] = pers
    for nm, pr in sorted(results.items(), key=lambda kv:
                         np.mean([np.abs(a - b).mean() for a, b in zip(kv[1], truth)])):
        e = np.mean([np.abs(a - b).mean() for a, b in zip(pr, truth)])
        print("  %-20s MAE %.4f  (%.1f%% of mean rate)" % (nm, e, 100 * e))

    print("\n=== 下游影響：把實測 share 換成預測 share，合成的絕對率誤差 ===")
    print("（先前用實測 share 的合成 MAE = 8.3%%；差額就是這個洩漏值多少）")
    for nm in ('const=1', 'solar 日照比例', 'persistence'):
        pr = results[nm]
        add = np.mean([np.abs(a - b).mean() for a, b in zip(pr, truth)])
        print("  %-20s share MAE %.4f ⇒ 合成誤差上界約 %.1f%%" % (nm, add, 100 * (0.083 + add)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
