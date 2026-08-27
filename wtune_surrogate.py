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
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_JULIA = str(Path.home() / '.juliaup' / 'bin' / 'julia')
DEFAULT_PROJECT = str(HERE / 'jlenv')
DEFAULT_SCRIPT = str(HERE / 'speedy_hotspot.jl')


def _wrap_lon(lon):
    """Normalize longitude to [-180, 180)."""
    x = ((lon + 180.0) % 360.0) - 180.0
    return x


def run_speedy(case, out_json, trunc=31, nlayers=8, hours=None,
               julia=DEFAULT_JULIA, project=DEFAULT_PROJECT,
               script=DEFAULT_SCRIPT, srun=None, timeout=1200):
    """Run the SpeedyWeather hotspot script -> writes out_json. Returns (ok, msg).

    `srun`, if given, is a list prefix (e.g. ['srun','-p','development','-t','8'])
    so the compute runs on a node, not the login host. When None, runs inline
    (only appropriate for a tiny probe / non-login host).
    """
    if not Path(julia).exists():
        return False, 'julia not found at {}'.format(julia)
    if not Path(script).exists():
        return False, 'runner script missing: {}'.format(script)
    hrs = hours if hours is not None else int(case.get('run_total_hours') or 6)
    cmd = []
    if srun:
        cmd += list(srun)
    cmd += [julia, '--project=' + project, script,
            '--trunc', str(trunc), '--nlayers', str(nlayers),
            '--hours', str(hrs), '--out', out_json,
            '--start-month', str(int(case.get('start_month') or 1)),
            '--start-day', str(int(case.get('start_day') or 1))]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 'speedy run timed out after {}s'.format(timeout)
    except Exception as exc:
        return False, 'speedy run failed: {}'.format(exc)
    if r.returncode != 0:
        tail = r.stdout.decode('utf-8', 'replace')[-800:]
        return False, 'speedy run rc={}:\n{}'.format(r.returncode, tail)
    if not Path(out_json).exists():
        return False, 'speedy produced no output file'
    return True, 'ok'


def _iter_field_points(field):
    """Yield (lon_wrapped, lat, value) for either a point-list or a regular grid."""
    if 'points' in field:
        for p in field['points']:
            yield _wrap_lon(p[0]), p[1], p[2]
        return
    glons = [_wrap_lon(x) for x in field.get('lons', [])]
    glats = field.get('lats', [])
    conv = field.get('convection', [])
    for j, la in enumerate(glats):
        row = conv[j] if j < len(conv) else []
        for i, lo in enumerate(glons):
            if i < len(row):
                yield lo, la, row[i]


def sample_domain(field, case, n_lon=24, n_lat=18):
    """Sample the global convection field over the WRF domain bbox.

    `field` is the parsed JSON from the runner: either {points:[[lon,lat,val],..]}
    or a regular grid {lons, lats, convection[lat][lon]}.
    Returns metrics dict or {'ok': False, 'reason': ...}.
    """
    import wtune_solar as wsolar
    bbox = wsolar.domain_bbox(case)
    if bbox is None:
        return {'ok': False, 'reason': 'domain lat/lon unknown (need --wrfinput)'}
    lat0, lat1, lon0, lon1 = bbox
    lon0w, lon1w = _wrap_lon(lon0), _wrap_lon(lon1)

    in_vals = []
    max_v = None
    max_at = None
    for lo, la, v in _iter_field_points(field):
        if la < lat0 or la > lat1:
            continue
        inside = (lon0w <= lo <= lon1w) if lon0w <= lon1w else (lo >= lon0w or lo <= lon1w)
        if not inside:
            continue
        in_vals.append((lo, la, v))
        if max_v is None or v > max_v:
            max_v = v
            max_at = (lo, la)
    if not in_vals:
        return {'ok': False, 'reason': 'no surrogate grid points fell in domain (too coarse?)'}

    vals = [v for _, _, v in in_vals]
    n = len(vals)
    mean_v = sum(vals) / n
    var_v = sum((v - mean_v) ** 2 for v in vals) / n
    # convective concentration: how peaked is it (max/mean); >2 => a strong blob
    concentration = (max_v / mean_v) if mean_v > 1e-9 else 0.0
    # E-W spread of convection (does it add to the terminator's E-W imbalance?)
    return {
        'ok': True,
        'n_points': n,
        'mean_convection': round(mean_v, 4),
        'max_convection': round(max_v, 4),
        'max_at': [round(max_at[0], 2), round(max_at[1], 2)] if max_at else None,
        'concentration_max_over_mean': round(concentration, 2),
        'stdev': round(math.sqrt(var_v), 4),
        'units': field.get('units', 'convective-proxy'),
    }


def convective_summary(case, out_json):
    """Text lines describing the convective hotspot from a runner output file."""
    p = Path(out_json)
    if not p.exists():
        return ["  convective hotspot: no surrogate output at {}".format(out_json)]
    try:
        field = json.loads(p.read_text())
    except Exception as exc:
        return ["  convective hotspot: unreadable surrogate output ({})".format(exc)]
    m = sample_domain(field, case)
    lines = []
    prov = field.get('provider', 'surrogate')
    lines.append("  convective hotspot ({}, trunc {}, {} h, {} time-integrated):".format(
        prov, field.get('trunc', '?'), field.get('period_hours', '?'), field.get('units', '?')))
    if field.get('note'):
        lines.append("    CAVEAT: {}".format(field['note']))
    if not m.get('ok'):
        lines.append("    in-domain sampling failed: {}".format(m.get('reason')))
        return lines
    lines.append("    in-domain: mean {mean}, max {mx} at lon/lat {at}; concentration(max/mean)={c}".format(
        mean=m['mean_convection'], mx=m['max_convection'], at=m['max_at'],
        c=m['concentration_max_over_mean']))
    if m['concentration_max_over_mean'] >= 2.0:
        lines.append("    -> STRONG convective blob: a moving microphysics hotspot; adds to imbalance,")
        lines.append("       spatially concentrated (decomposition still cannot fix a moving hotspot).")
    else:
        lines.append("    -> diffuse convection: little extra spatial imbalance beyond the terminator.")
    return lines
