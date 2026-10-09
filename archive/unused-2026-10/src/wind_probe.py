"""Steering-wind probe: read the case's ACTUAL mid-tropospheric wind from a
wrfinput/wrfout and derive the cloud-optics-staleness cap on radt -- so the
tuner validates its cloud-cap assumption against real data instead of a nominal
12 m/s guess.

Two cloud regimes matter for radiation staleness between radt calls:
  - LOW/MID cloud (~700 hPa) shades the SURFACE -> drives T2, the usual gate
    metric. Advects at the low-mid wind.
  - HIGH cirrus (~300-500 hPa, the jet) advects fastest -> tightest cap, but
    affects surface energy less.

cloud_cap(min) = ncells * dx / (wind * 60), ncells ~ 3 (keep cloud drift under
~3 cells between radiation calls).

Needs netCDF4 (run with a python that has it, e.g. the conda base). Emits JSON
so the stdlib-only tuner can read it, mirroring the SpeedyWeather surrogate.

Usage: python wind_probe.py --wrfinput <path> --dx <m> [--out wind.json]
"""
import sys, json


def argval(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def steering(wrfinput, dx_m, ncells=3.0):
    import netCDF4, numpy as np
    nc = netCDF4.Dataset(wrfinput)
    nz = nc.variables['U'].shape[1]

    def spd_at(k, step=8):
        U = nc.variables['U'][0, k, ::step, ::step].astype('f4')
        V = nc.variables['V'][0, k, ::step, ::step].astype('f4')
        ny = min(U.shape[0], V.shape[0]); nx = min(U.shape[1], V.shape[1])
        s = np.sqrt(U[:ny, :nx] ** 2 + V[:ny, :nx] ** 2)
        return float(np.ma.filled(s, np.nan).mean()), float(np.nanpercentile(np.ma.filled(s, np.nan), 90))

    low_k = max(1, nz // 4)         # ~700 hPa, surface-relevant cloud
    jet_k = nz // 2                 # ~500 hPa / jet, high cirrus
    low_mean, low_p90 = spd_at(low_k)
    jet_mean, jet_p90 = spd_at(jet_k)
    nc.close()

    def cap(wind):
        return round(ncells * dx_m / (wind * 60.0), 1) if wind > 0 else None

    return {
        'provider': 'wrfinput-steering',
        'nz': nz, 'dx_m': dx_m, 'ncells': ncells,
        'low_cloud': {'level': low_k, 'wind_mean': round(low_mean, 1), 'wind_p90': round(low_p90, 1),
                      'radt_cap_min': cap(low_mean)},
        'jet_cirrus': {'level': jet_k, 'wind_mean': round(jet_mean, 1), 'wind_p90': round(jet_p90, 1),
                       'radt_cap_min': cap(jet_mean)},
        'note': 'surface-relevant cap = low_cloud; worst-case (high cirrus) cap = jet_cirrus',
    }


def main():
    wi = argval('--wrfinput')
    dx = float(argval('--dx', '2500'))
    out = argval('--out')
    if not wi:
        print(json.dumps({'ok': False, 'reason': 'no --wrfinput'})); return 2
    try:
        r = steering(wi, dx)
        r['ok'] = True
    except Exception as exc:
        r = {'ok': False, 'reason': 'wind probe failed: {}'.format(exc)}
    s = json.dumps(r)
    if out:
        open(out, 'w').write(s)
        print('wrote', out)
    else:
        print(s)
    return 0 if r.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
