#!/usr/bin/env python3
"""wtune.py - analytical core (Stage A + Stage B) of a WRF config autotuner."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

# sibling modules (same directory as this script)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wtune_physics as phys
import wtune_knobs as knobs
import wtune_solar as wsolar
import wtune_surrogate as surr
import wtune_compliance as wc
import wtune_version
from wtune_feedback import (FeedbackController, FeedbackError, FeedbackSample,
                            decisions_to_csv, decisions_to_json)

FLOOR = 10              # minimum mass points in each direction after halo
HALO = 5                # WRF halo width approximation
KV = 50.0               # vectorization/amortization constant, calibrated to the measured
                        # CONUS 2.5km sweep so the RANKING matches wall time:
                        # 8x56 (0.9219, best) < 16x28 (0.9393) < 4x112 (1.0307, worst).
                        # KV=100 over-rewarded wide patches -> wrongly ranked 4x112 above 16x28.

DEFAULT_CORES_PER_NODE = 112
DEFAULT_NUMA_DOMAINS = 8
DEFAULT_CORES_PER_NUMA = 14
DEFAULT_NODES = 4

import os
CACHE_TMP = os.environ.get('TMPDIR') or '/tmp'


def first_value(raw):
    """Return the first scalar from a Fortran namelist value string."""
    if raw is None:
        return None
    raw = raw.strip()
    low = raw.lower()
    if low.startswith('.true.'):
        return True
    if low.startswith('.false.'):
        return False
    m = re.search(r'[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?', raw)
    if not m:
        return None
    s = m.group(0)
    if '.' in s or 'e' in s.lower():
        return float(s)
    return int(s)


def parse_namelist(path):
    """Minimal Fortran namelist reader.

    Returns {'group': {'key': raw_value_string}}.
    """
    text = Path(path).read_text()
    text = re.sub(r'!.*$', '', text, flags=re.M)

    groups = {}
    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = re.match(r'&([A-Za-z_]\w*)', line)
        if m:
            current = m.group(1).lower()
            groups.setdefault(current, [])
            rest = line[m.end():].strip()
            if '/' in rest:
                rest = rest.split('/', 1)[0]
            if rest:
                groups[current].append(rest)
            if '/' in line:
                current = None
            continue

        if '/' in line:
            before = line.split('/', 1)[0].strip()
            if current is not None and before:
                groups[current].append(before)
            current = None
            continue

        if current is not None:
            groups[current].append(line)

    parsed = {}
    assign_re = re.compile(r'([A-Za-z_]\w*)\s*=')
    for name, lines in groups.items():
        blob = ' '.join(lines)
        matches = list(assign_re.finditer(blob))
        group = {}
        for idx, m in enumerate(matches):
            key = m.group(1).lower()
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(blob)
            val = blob[start:end].strip()
            while val.endswith(','):
                val = val[:-1].rstrip()
            group[key] = val
        parsed[name] = group

    return parsed


def extract_case(nl):
    """Extract the namelist variables needed by the analytical model."""
    def scalar(group, key, default=None):
        g = nl.get(group, {})
        if key not in g:
            return default
        v = first_value(g[key])
        return default if v is None else v

    d = 'domains'
    t = 'time_control'
    p = 'physics'

    # physics_suite is a STRING ('conus'/'tropical'), not numeric -> read raw
    ps_raw = nl.get(p, {}).get('physics_suite')
    physics_suite = None
    if ps_raw:
        physics_suite = ps_raw.strip().strip("'\"").lower() or None

    e_we = int(scalar(d, 'e_we', 101))
    e_sn = int(scalar(d, 'e_sn', 101))
    nx = max(1, e_we - 1)
    ny = max(1, e_sn - 1)

    run_hours = scalar(t, 'run_hours', 0) or 0
    run_days = scalar(t, 'run_days', 0) or 0
    total_run_hours = run_hours + run_days * 24

    history_interval = scalar(t, 'history_interval_m', None)
    if history_interval is None:
        history_interval = scalar(t, 'history_interval', 60)

    lon = scalar(d, 'ref_lon', None)
    if lon is None:
        lon = scalar(d, 'stand_lon', None)
    if lon is None:
        lon = scalar(d, 'cen_lon', None)
    # leave None when absent so enrich_case() can fill it from a wrfinput header

    return {
        'nx': nx,
        'ny': ny,
        'e_we': e_we,
        'e_sn': e_sn,
        'e_vert': scalar(d, 'e_vert', 50),
        'dx': scalar(d, 'dx', 3000.0),
        'dy': scalar(d, 'dy', 3000.0),
        'time_step': scalar(d, 'time_step', 60),
        'max_dom': scalar(d, 'max_dom', 1),
        'run_total_hours': total_run_hours,
        'start_year': scalar(t, 'start_year', 2021),
        'start_month': scalar(t, 'start_month', 1),
        'start_day': scalar(t, 'start_day', 1),
        'start_hour': scalar(t, 'start_hour', 0),
        'start_minute': scalar(t, 'start_minute', 0),
        'start_second': scalar(t, 'start_second', 0),
        'history_interval': history_interval,
        'restart_interval': scalar(t, 'restart_interval', 1440),
        'nproc_x': scalar(d, 'nproc_x', None),
        'nproc_y': scalar(d, 'nproc_y', None),
        'numtiles': scalar(d, 'numtiles', None),
        'numtiles_x': scalar(d, 'numtiles_x', None),
        'numtiles_y': scalar(d, 'numtiles_y', None),
        'tile_strategy': scalar(d, 'tile_strategy', 0),
        'io_form_history': int(scalar(t, 'io_form_history', 2)),
        'io_form_restart': int(scalar(t, 'io_form_restart', 2)),
        'io_form_input': int(scalar(t, 'io_form_input', 2)),
        'io_form_boundary': int(scalar(t, 'io_form_boundary', 2)),
        # suite-fillable physics fields default to -1 (suite sentinel), NOT 0: an ABSENT
        # scheme under physics_suite='tropical'/'conus' must be filled by resolve_suite;
        # a value explicitly set in the namelist (e.g. Maria's cu_physics=0) still overrides.
        'mp_physics': int(scalar(p, 'mp_physics', -1)),
        'ra_lw_physics': int(scalar(p, 'ra_lw_physics', -1)),
        'ra_sw_physics': int(scalar(p, 'ra_sw_physics', -1)),
        'radt': scalar(p, 'radt', 15.0),
        'bl_pbl_physics': int(scalar(p, 'bl_pbl_physics', -1)),
        'sf_surface_physics': int(scalar(p, 'sf_surface_physics', -1)),
        'cu_physics': int(scalar(p, 'cu_physics', -1)),
        'sf_sfclay_physics': int(scalar(p, 'sf_sfclay_physics', -1)),
        'shcu_physics': int(scalar(p, 'shcu_physics', 0)),
        'sf_urban_physics': int(scalar(p, 'sf_urban_physics', 0)),
        'num_soil_layers': scalar(p, 'num_soil_layers', None),
        'sst_update': int(scalar(t, 'sst_update', scalar(p, 'sst_update', 0))),
        'tmn_update': int(scalar(p, 'tmn_update', 0)),
        'bucket_mm': scalar(t, 'bucket_mm', scalar(p, 'bucket_mm', -1.0)),
        'prec_acc_dt': scalar(p, 'prec_acc_dt', 0.0),
        'fractional_seaice': int(scalar(p, 'fractional_seaice', 0)),
        'o3input': scalar(p, 'o3input', None),
        # dynamics-group stability knobs
        'diff_opt': scalar('dynamics', 'diff_opt', None),
        'km_opt': scalar('dynamics', 'km_opt', None),
        'diff_6th_opt': int(scalar('dynamics', 'diff_6th_opt', 0)),
        'diff_6th_factor': scalar('dynamics', 'diff_6th_factor', 0.12),
        'w_damping': int(scalar('dynamics', 'w_damping', 0)),
        'epssm': scalar('dynamics', 'epssm', 0.1),
        'gwd_opt': int(scalar('dynamics', 'gwd_opt', 0)),
        'non_hydrostatic': first_value(nl.get('dynamics', {}).get('non_hydrostatic', '.true.')),
        # model top (Pa) for high-top rules; e_vert already captured
        'p_top_requested': scalar(d, 'p_top_requested', None),
        'latitude': scalar(d, 'ref_lat', scalar(d, 'cen_lat', None)),
        'physics_suite': physics_suite,
        'longitude': (float(lon) if lon is not None else None),
    }


def read_wrfinput_header(path):
    """Read lat/lon/date global attrs from a wrfinput/wrfout via `ncdump -h`.

    Header-only -> cheap even for a multi-GB file. Returns {} on failure.
    """
    import subprocess
    try:
        r = subprocess.run(['ncdump', '-h', path], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=120)
        text = r.stdout.decode('utf-8', 'replace')
    except Exception:
        return {}
    out = {}
    m = re.search(r':CEN_LAT\s*=\s*([-0-9.eEf]+)', text)
    if m:
        try:
            out['latitude'] = float(m.group(1).rstrip('f'))
        except ValueError:
            pass
    m = re.search(r':CEN_LON\s*=\s*([-0-9.eEf]+)', text)
    if m:
        try:
            out['longitude'] = float(m.group(1).rstrip('f'))
        except ValueError:
            pass
    m = re.search(r':START_DATE\s*=\s*"(\d+)-(\d+)-(\d+)_(\d+)', text)
    if m:
        out['start_year'] = int(m.group(1))
        out['start_month'] = int(m.group(2))
        out['start_day'] = int(m.group(3))
        out['start_hour'] = int(m.group(4))
    return out


def enrich_case(case, wrfinput):
    """Fill missing lat/lon/date in case from a wrfinput header (if given)."""
    if not wrfinput:
        return case, None
    hdr = read_wrfinput_header(wrfinput)
    if not hdr:
        return case, 'wrfinput header unreadable (ncdump failed?)'
    for k, v in hdr.items():
        if case.get(k) is None:
            case[k] = v
    return case, None


def _read_wps_center(wps_path):
    """ref_lat/ref_lon (domain centre) from a sibling namelist.wps. {} on failure."""
    try:
        txt = open(wps_path).read()
    except OSError:
        return {}
    out = {}
    for key in ('ref_lat', 'ref_lon'):
        m = re.search(r'\b' + key + r'\s*=\s*([-0-9.]+)', txt)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def _bbox_from_center(lat, lon, e_we, e_sn, dx, dy):
    """Approx lat/lon bounding box of a WRF domain from centre + grid. Projection-agnostic
    cos-lat approximation -- exact enough for a REGIME proxy (the probe needs the climate
    regime over the domain, not the map projection)."""
    half_y = (max(2, int(e_sn)) - 1) * dy / 2.0 / 111000.0
    coslat = max(0.15, math.cos(math.radians(lat)))
    half_x = (max(2, int(e_we)) - 1) * dx / 2.0 / (111000.0 * coslat)
    return (round(max(-89.0, lat - half_y), 2), round(min(89.0, lat + half_y), 2),
            round(lon - half_x, 2), round(lon + half_x, 2))


_PROBE_SCHEMA_KEYS = ('provider', 'convective_fraction', 'hours', 'trunc', 'nlayers')


def _quiet_unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _file_sha8(path):
    """First 8 hex of a file's SHA-256, or 'none' when unreadable. Folds the probe
    script's own code into the cache key so editing speedy_probe.jl invalidates."""
    try:
        with open(path, 'rb') as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:8]
    except OSError:
        return 'none'


def _speedy_pkg_id(wtune_dir):
    """The pinned SpeedyWeather build id (version + git-tree-sha1) read CHEAPLY from
    jlenv/Manifest.toml -- no julia launch. A package upgrade changes this string, so it
    goes in the cache key and a stale cached probe from an older build becomes a miss."""
    man = os.path.join(wtune_dir, 'jlenv', 'Manifest.toml')
    try:
        with open(man, 'r') as fh:
            lines = fh.read().splitlines()
    except OSError:
        return 'unknown'
    ver, sha = '', ''
    for i, ln in enumerate(lines):
        if ln.strip() == '[[deps.SpeedyWeather]]':
            for ln2 in lines[i + 1:i + 12]:
                s = ln2.strip()
                if s.startswith('[['):
                    break
                if s.startswith('version'):
                    ver = s.split('=', 1)[1].strip().strip('"')
                elif s.startswith('git-tree-sha1'):
                    sha = s.split('=', 1)[1].strip().strip('"')[:12]
            break
    return (ver + '+' + sha) if (ver or sha) else 'unknown'


def _probe_cache_valid(out_path, ident):
    """A cache file is a HIT only when it parses, carries the expected schema keys, AND its
    self-reported (hours, trunc, nlayers) match the identity we would run now. Anything else
    -- truncated, hand-edited, or from an incompatible run -- is a MISS, not silent truth.
    Returns (is_hit, why). bbox and build id need not be re-checked: they are baked into the
    filename hash, so a mismatch there is already a different file."""
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        return False, 'absent'
    try:
        with open(out_path, 'r') as fh:
            j = json.load(fh)
    except (ValueError, OSError) as exc:
        return False, 'unreadable/corrupt ({})'.format(exc)
    for k in _PROBE_SCHEMA_KEYS:
        if k not in j:
            return False, 'missing key {!r}'.format(k)
    for k in ('hours', 'trunc', 'nlayers'):
        if int(j.get(k, -1)) != int(ident[k]):
            return False, 'stale {}={} != {}'.format(k, j.get(k), ident[k])
    return True, 'ok'


def auto_probe(namelist_path, case, resolved, args):
    """Embed the SpeedyWeather probe INTO the tuner: derive the domain bbox, run
    speedy_probe.jl (cached by bbox), and return (json_path, message). The probe is a
    COMPONENT of the tuner -- screen/physics call this automatically unless --no-probe or
    an explicit --probe-json is given. Returns (None, reason) when it cannot / should not
    run (LES scale below validity floor, no lat/lon, no julia)."""
    import subprocess
    import tempfile
    dxm = float(resolved.get('dx', 2500))
    if dxm < 1000.0:      # validity floor: the global surrogate has no LES-scale flow
        return None, 'WRF-ONLY-GATE: dx={:.0f}m below SpeedyWeather validity (LES) -> probe cannot apply here (DETERMINISTIC, not a transient failure); analytic + WRF smoke only'.format(dxm)
    lat, lon = case.get('latitude'), case.get('longitude')
    if getattr(args, 'latlon', None):
        try:
            lat, lon = [float(x) for x in args.latlon.split(',')[:2]]
        except (ValueError, IndexError):
            pass
    if lat is None or lon is None:
        wps = getattr(args, 'wps', None) or os.path.join(os.path.dirname(os.path.abspath(namelist_path)), 'namelist.wps')
        c = _read_wps_center(wps)
        lat = lat if lat is not None else c.get('ref_lat')
        lon = lon if lon is not None else c.get('ref_lon')
    if lat is None or lon is None:
        return None, 'WRF-ONLY-GATE: no domain lat/lon (namelist has none; no readable namelist.wps/wrfinput) -> pass --latlon LAT,LON or --wps FILE (DETERMINISTIC input gap, not a transient failure)'
    lat0, lat1, lon0, lon1 = _bbox_from_center(lat, lon, case.get('e_we', 100), case.get('e_sn', 100),
                                               dxm, float(resolved.get('dy', dxm)))
    hours = int(getattr(args, 'probe_hours', 6) or 6)
    trunc = int(getattr(args, 'probe_trunc', 31) or 31)
    nlayers = int(getattr(args, 'probe_nlayers', 8) or 8)
    wtune_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(wtune_dir, 'speedy_probe.jl')
    # Cache IDENTITY = every input that changes the probe result, and ONLY those. This probe is
    # a CLIMATOLOGICAL global forecast from initialize!(model); it does NOT ingest WRF's initial
    # state, so the WRF case start-time / physics-suite / input checksum do NOT change it and are
    # deliberately EXCLUDED (folding them in would be false precision -- a cache miss on a field
    # that never moved the answer). What does change it: geography (bbox), integration length
    # (hours), spectral resolution (trunc, nlayers), the pinned SpeedyWeather build, and this
    # script's own code. A change in any of these yields a different filename -> automatic miss.
    ident = {
        'bbox': [round(lat0, 3), round(lat1, 3), round(lon0, 3), round(lon1, 3)],
        'hours': hours, 'trunc': trunc, 'nlayers': nlayers,
        'speedy': _speedy_pkg_id(wtune_dir), 'script_sha8': _file_sha8(script),
    }
    key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = os.environ.get('WTUNE_PROBE_CACHE', os.path.join(tempfile.gettempdir(), 'wtune_probe_cache'))
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        cache_dir = tempfile.gettempdir()
    bbox_s = 'lat[{},{}] lon[{},{}] {}h T{} L{}'.format(lat0, lat1, lon0, lon1, hours, trunc, nlayers)
    out = os.path.join(cache_dir, 'probe_{}.json'.format(key))
    hit, why = _probe_cache_valid(out, ident)
    if hit:
        return out, 'cache HIT SpeedyWeather probe ({}) [{}]'.format(bbox_s, key)
    if os.path.exists(out):   # present but not a valid hit -> say so, then overwrite atomically
        sys.stderr.write('[wtune] probe cache STALE ({}): {}; rerunning\n'.format(key, why))
    julia = os.environ.get('WTUNE_JULIA') or shutil.which('julia') or os.path.expanduser('~/.juliaup/bin/julia')
    jlenv = os.path.join(wtune_dir, 'jlenv')
    if not (os.path.exists(julia) and os.path.isdir(jlenv) and os.path.exists(script)):
        return None, 'PROBE-UNAVAILABLE: julia/jlenv/speedy_probe.jl not found -> set WTUNE_JULIA or run the probe manually'
    sys.stderr.write('[wtune] running SpeedyWeather probe ({}, {}h; first run compiles, ~1-3 min)...\n'.format(bbox_s, hours))
    sys.stderr.flush()
    # Write to a unique temp path, then os.replace() into the cache name: a hit is either the
    # whole valid file or nothing, so a killed/timed-out probe never leaves a half-written cache
    # that a later run would trust. Concurrent probes on the same key race harmlessly on rename.
    tmp = out + '.tmp.{}'.format(os.getpid())
    cmd = [julia, '--project=' + jlenv, script, '--hours', str(hours),
           '--trunc', str(trunc), '--nlayers', str(nlayers),
           '--lat0', str(lat0), '--lat1', str(lat1), '--lon0', str(lon0), '--lon1', str(lon1), '--out', tmp]
    try:
        r = subprocess.run(cmd, cwd=wtune_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=int(os.environ.get('WTUNE_PROBE_TIMEOUT', '900')))
    except subprocess.TimeoutExpired:
        _quiet_unlink(tmp)
        return None, 'PROBE-UNAVAILABLE: SpeedyWeather probe timed out (raise WTUNE_PROBE_TIMEOUT)'
    except Exception as exc:
        _quiet_unlink(tmp)
        return None, 'PROBE-UNAVAILABLE: probe failed to launch: {}'.format(exc)
    if r.returncode != 0 or not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
        _quiet_unlink(tmp)
        tail = (r.stderr.decode('utf-8', 'replace').strip().splitlines() or [''])[-1] if r.stderr else ''
        return None, 'PROBE-UNAVAILABLE: probe run failed (rc={}): {}'.format(r.returncode, tail)
    ok, why = _probe_cache_valid(tmp, ident)   # validate the fresh output before it becomes cache
    if not ok:
        _quiet_unlink(tmp)
        return None, 'PROBE-UNAVAILABLE: probe output failed schema/identity check ({})'.format(why)
    try:
        os.replace(tmp, out)
    except OSError as exc:
        _quiet_unlink(tmp)
        return None, 'PROBE-UNAVAILABLE: could not commit probe cache: {}'.format(exc)
    return out, 'ran SpeedyWeather probe ({}, {}h) [{}]'.format(bbox_s, hours, key)


def factor_pairs(n):
    """All (px, py) with px * py == n."""
    if n <= 0:
        return []
    pairs = []
    limit = int(math.sqrt(n))
    for i in range(1, limit + 1):
        if n % i == 0:
            j = n // i
            pairs.append((i, j))
            if i != j:
                pairs.append((j, i))
    return pairs


def cost_model(nx, ny, px, py, ranks, nodes, numa_domains):
    """Analytical ranking cost. Lower is better."""
    patch_x = nx / float(px)
    patch_y = ny / float(py)

    ghost_inflation = ((patch_x + 2 * HALO) * (patch_y + 2 * HALO)) / (patch_x * patch_y)
    vec_factor = 1.0 + KV / patch_x

    if nodes is not None:
        if ranks % nodes == 0:
            ranks_per_node = ranks // nodes
            numa_misalign = 1 if ranks_per_node % numa_domains != 0 else 0
        else:
            numa_misalign = 1
    else:
        numa_misalign = 0

    numa_penalty = 1.0 + 0.15 * numa_misalign
    return ghost_inflation * vec_factor * numa_penalty


def enumerate_valid(nx, ny, ranks, nodes, numa_domains):
    """Enumerate valid decompositions and score them."""
    out = []
    for px, py in factor_pairs(ranks):
        patch_x = nx / float(px)
        patch_y = ny / float(py)
        if patch_x < FLOOR or patch_y < FLOOR:
            continue
        score = cost_model(nx, ny, px, py, ranks, nodes, numa_domains)
        out.append({
            'px': px,
            'py': py,
            'patch_x': patch_x,
            'patch_y': patch_y,
            'score': score,
        })
    out.sort(key=lambda x: x['score'])
    return out


def physics_tags(case):
    tags = ['D']
    if case['mp_physics'] in (8, 28):
        tags.append('M')
    if case['ra_lw_physics'] == 4 or case['ra_sw_physics'] == 4:
        tags.append('R')
    if case['history_interval'] is not None and case['history_interval'] <= 60:
        tags.append('I')
    elif case['restart_interval'] is not None and case['restart_interval'] <= 60:
        tags.append('I')
    return tags


def radiation_spacing(case):
    radt = case.get('radt', 0.0) or 0.0
    dt = case.get('time_step', 60) or 60
    if dt <= 0:
        dt = 60
    if radt <= 0:
        return 1
    return max(1, int(round(radt * 60.0 / dt)))


def solar_estimate(case):
    utc = case.get('start_hour', 0) or 0
    lon = case.get('longitude', -120.0) or -120.0
    local = (utc + lon / 15.0) % 24.0
    if 6.0 <= local < 18.0:
        phase = 'day'
    elif (18.0 <= local < 20.0) or (4.0 <= local < 6.0):
        phase = 'terminator-crossing'
    else:
        phase = 'night'
    return {'local_hour': local, 'phase': phase}


def fixed_picks(case, allow_hybrid, numa_domains, cores_per_numa):
    picks = []
    for key in ('io_form_history', 'io_form_restart', 'io_form_input', 'io_form_boundary'):
        cur = case.get(key, 2)
        if cur != 14:
            picks.append("{} current={} -> recommend 14 (ADIOS2, ~-29% wall)".format(key, cur))
        else:
            picks.append("{} current=14 (ADIOS2) good".format(key))

    picks.append("OMP_NUM_THREADS=1 (pure MPI; WRF hybrid is +12.78% even NUMA-aligned)")

    if allow_hybrid:
        picks.append(
            "NUMA-aligned hybrid alternative: ranks_per_node={} threads={}; "
            "+12.78% floor, use only if per-rank memory forces it; "
            "needs OMP_STACKSIZE>=512M".format(numa_domains, cores_per_numa)
        )

    picks.append(
        "I_MPI_PIN_DOMAIN=core (disjoint ownership); NIC is on NUMA domain 1, "
        "domains 4-7 are NIC-remote (distance 21 vs 12)"
    )
    picks.append("numtiles=1 for pure MPI; numtiles invariant is tile SIZE not count; do not set blindly")
    return picks


def plan(args):
    cores_per_node = args.cores_per_node if args.cores_per_node else DEFAULT_CORES_PER_NODE
    numa_domains = args.numa_domains if args.numa_domains else DEFAULT_NUMA_DOMAINS
    cores_per_numa = args.cores_per_numa
    if cores_per_numa is None:
        cores_per_numa = cores_per_node // numa_domains if cores_per_node and numa_domains else DEFAULT_CORES_PER_NUMA
    smt = args.smt if args.smt else 'off'

    if args.ranks is not None:
        ranks = args.ranks
        nodes = args.nodes
    elif args.nodes is not None:
        nodes = args.nodes
        ranks = nodes * cores_per_node
    else:
        nodes = DEFAULT_NODES
        ranks = nodes * cores_per_node

    if ranks <= 0:
        print("ERROR: ranks must be positive", file=sys.stderr)
        sys.exit(1)

    try:
        nl = parse_namelist(args.namelist)
    except Exception as exc:
        print("ERROR reading namelist: {}".format(exc), file=sys.stderr)
        sys.exit(1)

    case = extract_case(nl)
    case, _enrich_err = enrich_case(case, getattr(args, 'wrfinput', None))
    resolved, filled = phys.resolve_suite(case)
    valid = enumerate_valid(case['nx'], case['ny'], ranks, nodes, numa_domains)
    shortlist = valid[:5]

    tags = physics_tags(resolved)
    rad_spacing = radiation_spacing(resolved)
    solar = solar_estimate(resolved)
    picks = fixed_picks(case, args.allow_hybrid, numa_domains, cores_per_numa)

    if args.json:
        payload = {
            'case': case,
            'hardware': {
                'cores_per_node': cores_per_node,
                'numa_domains': numa_domains,
                'cores_per_numa': cores_per_numa,
                'smt': smt,
            },
            'ranks': ranks,
            'nodes': nodes,
            'decomposition_shortlist': [
                {
                    'rank': i + 1,
                    'px': d['px'],
                    'py': d['py'],
                    'patch_x': d['patch_x'],
                    'patch_y': d['patch_y'],
                    'patch_x_rounded': round(d['patch_x']),
                    'patch_y_rounded': round(d['patch_y']),
                    'score': round(d['score'], 3),
                    'model_pick': i == 0,
                }
                for i, d in enumerate(shortlist)
            ],
            'physics_tags': tags,
            'radiation_spacing_steps': rad_spacing if 'R' in tags else None,
            'solar': solar,
            'fixed_picks': picks,
        }
        print(json.dumps(payload, indent=2))
        return

    print("=== WRF config autotuner (analytical narrow) ===")
    print("Namelist: {}".format(args.namelist))
    print("Ranks: {}".format(ranks))
    print("Nodes: {}".format(nodes if nodes is not None else "n/a"))
    print("Hardware: cores/node={}, numa_domains={}, cores/numa={}, smt={}".format(
        cores_per_node, numa_domains, cores_per_numa, smt))
    print()
    print("Domain mass points: nx={}, ny={}, nz={}, dx={}, dy={}, dt={}, max_dom={}".format(
        case['nx'], case['ny'], case['e_vert'], case['dx'], case['dy'],
        case['time_step'], case['max_dom']))
    print("Run length: {} h; history_interval={} min; restart_interval={} min".format(
        case['run_total_hours'], case['history_interval'], case['restart_interval']))

    print()
    print("PHYSICS DIAGNOSIS + NUMERICS ADVICE (Stage A; report only, does not drive decomposition)")
    if resolved.get('physics_suite'):
        print("  physics_suite = '{}'".format(resolved['physics_suite']))
        if filled:
            print("  suite-filled (-1 -> default): {}".format(
                ", ".join("{}={}".format(f[0], f[1]) for f in filled)))
    print("  Tags: {}  (D=dynamics M=microphysics R=radiation I=io)".format(", ".join(tags)))
    for line in phys.diagnose(resolved):
        print(line)
    if 'R' in tags:
        print("  radiation cadence: every {} steps (radt={} min, dt={} s)".format(
            rad_spacing, resolved.get('radt'), resolved.get('time_step')))
    print("  numerics(radt): {}".format(phys.radt_advice(resolved)))
    print("  numerics(dt):   {}".format(phys.timestep_advice(resolved)))
    print()
    print("PHYSICS HOTSPOT -- terminator load map (analytic, time-integrated; dominant driver)")
    for line in wsolar.solar_summary(resolved):
        print(line)
    sj = getattr(args, 'surrogate_json', None)
    if sj:
        for line in surr.convective_summary(resolved, sj):
            print(line)

    # knob advisor: surface only the actionable (non-OK) findings here; full via `knobs`
    findings = knobs.advise(resolved)
    flagged = [f for f in findings if f.get('verdict') in ('MISMATCH', 'SUGGEST')]
    print()
    print("KNOB ADVISOR -- {} findings ({} actionable); full list: wtune.py knobs --namelist ...".format(
        len(findings), len(flagged)))
    for f in flagged[:8]:
        print("  [{:8s}] {:20s} cur={} -> {}".format(
            f.get('verdict'), f.get('param'), f.get('current'), f.get('recommended')))
    if len(flagged) > 8:
        print("  ... {} more (see `knobs`)".format(len(flagged) - 8))

    print()
    print("FIXED BEST-PRACTICE PICKS (report only)")
    for p in picks:
        print("  - {}".format(p))

    print()
    print("STAGE B DECOMPOSITION SHORTLIST (top 5 valid by model score)")
    if not shortlist:
        print("  No valid decompositions meet patch floor >= {}".format(FLOOR))
    else:
        print("  Valid decompositions meeting patch floor: {}".format(len(valid)))
        print("  {}  {:<10} {:<12} {}".format("Rank", "px x py", "patch", "score"))
        for i, d in enumerate(shortlist):
            marker = "*" if i == 0 else " "
            print(" {}{:<3} {:<10} {:<12} {:.3f}".format(
                marker, i + 1,
                "{} x {}".format(d['px'], d['py']),
                "{} x {}".format(round(d['patch_x']), round(d['patch_y'])),
                d['score']))
        print("  * = model #1 pick (lowest analytical score)")
        print("  NOTE: model narrows the candidate list; it does not predict wall time.")


def profile_cmd(args):
    """Case-understanding profile: hooks wtune_case.py's build_profile/render_text in."""
    import wtune_case as wcase
    namelist, wrfinput = wcase.find_case_inputs(args.case_dir, args.namelist, args.wrfinput)
    if not os.path.exists(namelist):
        print("ERROR: no namelist found at {} (pass --namelist)".format(namelist), file=sys.stderr)
        return 1
    cores_per_node = args.cores_per_node or wcase.DEFAULT_CORES_PER_NODE
    try:
        profile = wcase.build_profile(namelist, wrfinput, cores_per_node)
    except Exception as exc:
        print("ERROR building profile: {}".format(exc), file=sys.stderr)
        return 1
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(profile, fh, indent=2)
        print('wrote {}'.format(args.json))
    for line in wcase.render_text(profile):
        print(line)
    return 0


def physics_cmd(args):
    """Physics diagnosis + numerics advice + (optional) scheme menu."""
    try:
        nl = parse_namelist(args.namelist)
    except Exception as exc:
        print("ERROR reading namelist: {}".format(exc), file=sys.stderr)
        sys.exit(1)
    case = extract_case(nl)
    case, _ = enrich_case(case, getattr(args, 'wrfinput', None))
    resolved, filled = phys.resolve_suite(case)

    print("=== WRF physics diagnosis + numerics advice ===")
    print("Namelist: {}".format(args.namelist))
    if resolved.get('physics_suite'):
        print("physics_suite = '{}'".format(resolved['physics_suite']))
        if filled:
            print("suite-filled (-1 -> default): {}".format(
                ", ".join("{}={}".format(f[0], f[1]) for f in filled)))
    print()
    print("CHOSEN PHYSICS (science is fixed; tuner diagnoses cost, never swaps for speed):")
    for line in phys.diagnose(resolved):
        print(line)
    print()
    nest = phys.nesting_advice(_parse_nl_kv(args.namelist), resolved)
    if nest:
        for line in nest:
            print(line)
        print()
    print("RESULTS-AFFECTING NUMERICS (gated tuning only):")
    level = getattr(args, 'compliance', None)
    hidden = 0
    radt_cls = wc.classify('radt')
    if level is None or wc.admits(level, radt_cls):
        print("  [{}] radt: {}".format(radt_cls, phys.radt_advice(resolved)))
    else:
        hidden += 1
    dt_cls = wc.classify('time_step')
    if level is None or wc.admits(level, dt_cls):
        print("  [{}] dt:   {}".format(dt_cls, phys.timestep_advice(resolved)))
    else:
        hidden += 1
    if level is not None and hidden:
        print("  ({} item(s) hidden as unclassified/disallowed at --compliance {})".format(hidden, level))
    print()
    print("SPEED KNOBS -- reduce TOTAL WORK (bit-identical Tier 1 + gated Tier 2):")
    for line in phys.numeric_speed_advice(resolved):
        print("  " + line if line else "")
    print()
    # The SpeedyWeather probe is PART of the tuner: auto-run it (cached) unless the user
    # passed --probe-json or --no-probe.
    import wtune_surrogate as wsurr
    probe = None
    if getattr(args, 'probe_json', None):
        probe = wsurr.load_probe(args.probe_json)
    elif not getattr(args, 'no_probe', False):
        ppath, note = auto_probe(args.namelist, case, resolved, args)
        print("SpeedyWeather probe: {}".format(note))
        if ppath:
            probe = wsurr.load_probe(ppath)
        print()
    if probe is not None:
        print("SPEEDYWEATHER SELF-VALIDATION (fast forecast checks the tuner's physics assumptions):")
        for line in wsurr.analyze_probe(probe, resolved):
            print("  " + line)
        wind = wsurr.probe_steering_wind(probe)
        if wind:
            nominal = phys._radt_cloud_bound(resolved)
            refined = phys._radt_cloud_bound(resolved, wind=wind)
            print("  -> cloud-cap CHECK: nominal 12 m/s gave {} min; forecast wind {:.0f} m/s gives {} min. "
                  "radt={} {} the validated cap.".format(
                      nominal, wind, refined, int(resolved.get('radt', 10)),
                      "is AT/UNDER" if float(resolved.get('radt', 10)) <= refined else "EXCEEDS"))
        print()
    print("PHYSICS HOTSPOT -- terminator load map (pass --wrfinput for lat/lon):")
    for line in wsolar.solar_summary(resolved):
        print(line)

    if args.menu:
        print()
        print("SCHEME MENU (advisory — for DESIGNING a new case; you pick the science, costs shown):")
        for line in phys.menu():
            print(line)


def hotspot_cmd(args):
    """Physics load map: analytic terminator (always) + convective surrogate (optional)."""
    try:
        nl = parse_namelist(args.namelist)
    except Exception as exc:
        print("ERROR reading namelist: {}".format(exc), file=sys.stderr)
        sys.exit(1)
    case = extract_case(nl)
    case, enrich_err = enrich_case(case, getattr(args, 'wrfinput', None))
    resolved, _ = phys.resolve_suite(case)

    print("=== WRF physics load-map / hotspot ===")
    print("Namelist: {}".format(args.namelist))
    if enrich_err:
        print("  (wrfinput: {})".format(enrich_err))
    print()
    print("RADIATION driver -- terminator (analytic, time-integrated):")
    for line in wsolar.solar_summary(resolved):
        print(line)

    sj = getattr(args, 'surrogate_json', None)
    if getattr(args, 'run_surrogate', False):
        sj = sj or (str(Path(CACHE_TMP) / 'wtune_speedy.json') if CACHE_TMP else 'wtune_speedy.json')
        srun = args.srun.split() if getattr(args, 'srun', None) else None
        print()
        print("CONVECTIVE driver -- running SpeedyWeather surrogate (trunc {}, {} h)...".format(
            args.trunc, resolved.get('run_total_hours')))
        ok, msg = surr.run_speedy(resolved, sj, trunc=args.trunc, nlayers=args.nlayers,
                                  srun=srun)
        if not ok:
            print("  surrogate run FAILED: {}".format(msg))
            sj = None
    print()
    print("CONVECTIVE driver -- storm/microphysics hotspot (surrogate):")
    if sj:
        for line in surr.convective_summary(resolved, sj):
            print(line)
    else:
        print("  (no surrogate output; pass --surrogate-json PATH or --run-surrogate)")


def oracle_map_cmd(args):
    """DIAGNOSTIC spatial physics-load map from the Stage-0 Oracle: per-cell radiation/micro/
    cumulus/pbl/surface cost + top-k hotspots + per-phase evidence, written machine-readable.
    Needs a real snapshot (wrfinput/wrfout) for the model-state fields. INSPECTION ONLY -- it
    does NOT pick nproc (that is partition_sweep; WRF keeps regular patches)."""
    import datetime as _dt
    try:
        nl = parse_namelist(args.namelist)
    except Exception as exc:
        print("ERROR reading namelist: {}".format(exc), file=sys.stderr)
        sys.exit(1)
    case = extract_case(nl)
    case, enrich_err = enrich_case(case, getattr(args, 'wrfinput', None))
    resolved, _ = phys.resolve_suite(case)
    snapshot = getattr(args, 'snapshot', None) or getattr(args, 'wrfinput', None)
    if not snapshot or not os.path.exists(snapshot):
        print("ERROR: need a NetCDF snapshot with model-state fields -- pass --snapshot "
              "wrfout_d01_... (or --wrfinput). Given: {!r}".format(snapshot), file=sys.stderr)
        sys.exit(1)
    try:
        import wtune_oracle as orc
    except Exception as exc:
        print("ERROR: cannot import wtune_oracle (needs numpy + netCDF reader): {}".format(exc),
              file=sys.stderr)
        sys.exit(1)

    y = int(resolved.get('start_year', 2021)); mo = int(resolved.get('start_month', 1))
    da = int(resolved.get('start_day', 1)); hr = int(resolved.get('start_hour', 0))
    doy = _dt.date(y, mo, da).timetuple().tm_yday
    run_hours = float(resolved.get('run_total_hours', 0) or 0)
    cu_on = int(resolved.get('cu_physics', 0) or 0) > 0

    print("=== WRF Oracle spatial hotspot (DIAGNOSTIC -- not a partition selector) ===")
    print("snapshot: {}  doy={} utc={}h run={}h cu_on={}".format(snapshot, doy, hr, run_hours, cu_on))
    if enrich_err:
        print("  (wrfinput: {})".format(enrich_err))
    try:
        art = orc.spatial_hotspot_artifact(snapshot, doy, hr, run_hours=run_hours,
                                           stride=args.stride, cu_on=cu_on,
                                           top_k=args.top_k, grid=args.grid)
    except Exception as exc:
        print("ERROR building artifact: {}".format(exc), file=sys.stderr)
        sys.exit(1)

    if cu_on:
        print("  weight set: CU (PROVISIONAL cumulus/pbl -- no coarse per-phase truth yet)")
    print("  micro field: {}".format(art['meta']['micro_mode']))
    print("  top hotspots (weighted cost, lat/lon; driver = hottest variable phase):")
    for h in art['top_k'][:min(5, len(art['top_k']))]:
        print("    ({:.2f},{:.2f}) cost={:.3f} driver={}".format(
            h['lat'], h['lon'], h['weighted_cost'], h['dominant_driver']))
    jp = getattr(args, 'json_out', None)
    npzp = getattr(args, 'npz_out', None)
    if jp or npzp:
        rcpt = orc.emit_spatial_artifact(art, json_path=jp, npz_path=npzp)
        print("  wrote: {}".format(rcpt))
    else:
        print("  (pass --json-out PATH and/or --npz-out PATH to persist the full artifact)")


def knobs_cmd(args):
    """Full knob advisor: input-derived recommendations + mismatch flags, grouped."""
    try:
        nl = parse_namelist(args.namelist)
    except Exception as exc:
        print("ERROR reading namelist: {}".format(exc), file=sys.stderr)
        sys.exit(1)
    case = extract_case(nl)
    case, _ = enrich_case(case, getattr(args, 'wrfinput', None))
    resolved, _ = phys.resolve_suite(case)

    grouped = knobs.advise_grouped(resolved)
    print("=== WRF knob advisor (input-derived) ===")
    print("Namelist: {}".format(args.namelist))
    print("Legend: [CORRECTNESS]=code-enforced/silent-override  [RESULTS]/[best-practice]=results-affecting guidance")
    level = getattr(args, 'compliance', None)
    total = 0
    hidden = 0
    for cat in ('correctness-setup', 'resolution(dx)', 'stability', 'run-length', 'terrain-latitude'):
        items = grouped.get(cat, [])
        if not args.all:
            items = [f for f in items if f.get('verdict') in ('MISMATCH', 'SUGGEST')]
        rows = []
        for f in items:
            cls = wc.classify(f.get('param', ''))
            if level is not None and not wc.admits(level, cls):
                hidden += 1
                continue
            rows.append((f, cls))
        if not rows:
            continue
        print()
        print("[{}]".format(cat))
        for f, cls in rows:
            total += 1
            enf = 'enforced' if f.get('enforced') else 'best-practice'
            print("  [{:8s}] [{}] {:22s} cur={!s:12s} rec={!s:24s} ({}/{})".format(
                f.get('verdict', '?'), cls, f.get('param', '?'), f.get('current'),
                f.get('recommended'), f.get('tag', '?'), enf))
            if f.get('reason'):
                print("             {}".format(f['reason']))
    if total == 0:
        print("\n  no actionable findings (use --all to see OK checks).")
    if level is not None and hidden:
        print("\n  ({} item(s) hidden as unclassified/disallowed at --compliance {})".format(hidden, level))


def _load_catalog():
    p = Path(__file__).resolve().parent / 'wtune_catalog_registry.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def catalog_cmd(args):
    """List the complete Registry-derived parameter catalog."""
    cat = _load_catalog()
    if cat is None:
        print("catalog not found; run: python3 gen_registry_catalog.py <Registry_dir> "
              "wtune_catalog_registry.json", file=sys.stderr)
        sys.exit(1)
    groups = cat['groups']
    names = [args.group] if args.group else sorted(
        groups, key=lambda g: -len(groups[g]))
    shown = 0
    print("=== WRF parameter catalog (Registry-derived, forecast-relevant) ===")
    print("source: {} | total {} params".format(cat['meta']['source'], cat['total_params']))
    for g in names:
        entries = groups.get(g, [])
        if not args.include_io:
            entries = [e for e in entries if not e.get('io_stream')]
        if args.grep:
            entries = [e for e in entries if args.grep.lower() in e['name'].lower()]
        if not entries:
            continue
        print("\n&{} ({} params{})".format(
            g, len(entries), "" if args.include_io else ", I/O plumbing hidden"))
        for e in entries:
            pd = '(max_dom)' if e.get('per_domain') else ''
            meaning = e.get('meaning_en') or ''
            print("  {:26s} {:9s} default={!s:8s} {} {}".format(
                e['name'], e['type'], e['default'], pd, meaning[:48]))
            shown += 1
    print("\n({} params shown)".format(shown))


def selftest():
    nx, ny, ranks = 1500, 1200, 448
    nodes, numa_domains = 4, 8

    valid = enumerate_valid(nx, ny, ranks, nodes, numa_domains)
    rank = {(d['px'], d['py']): i for i, d in enumerate(valid)}
    bottom3 = {(d['px'], d['py']) for d in valid[-3:]}

    # Calibrated against the MEASURED CONUS 2.5km sweep (median s/step):
    #   8x56 (0.9219, best) < 16x28 (0.9393) < 4x112 (1.0307, worst).
    # The model must reproduce that ORDER, not just put 8x56 near the top.
    ok1 = bool(valid) and (valid[0]['px'], valid[0]['py']) == (8, 56)      # 8x56 is the #1 pick
    ok2 = rank.get((16, 28), 999) < rank.get((4, 112), -1)                 # 16x28 ranks ABOVE 4x112
    ok3 = (56, 8) in bottom3 and (112, 4) in bottom3                       # transposes are worst

    print("CONUS 2.5km selftest (nx={}, ny={}, ranks={})".format(nx, ny, ranks))
    print("Valid decompositions:")
    for d in valid:
        print("  {}x{}  patch {}x{}  score {:.3f}".format(
            d['px'], d['py'], round(d['patch_x']), round(d['patch_y']), d['score']))

    ok = True
    for label, cond in [("8x56 is the #1 pick (measured best)", ok1),
                        ("16x28 ranks above 4x112 (measured order)", ok2),
                        ("56x8,112x4 in bottom-3 (transposes worst)", ok3)]:
        print("  {}: {}".format(label, "PASS" if cond else "FAIL"))
        ok = ok and cond

    compliance_errs = wc.validate()
    print("compliance catalog: {} entries, {} errors".format(len(wc.CATALOG), len(compliance_errs)))
    for m in compliance_errs:
        print("  ERROR " + m)
    ok = ok and not compliance_errs

    if ok:
        print("ALL PASS")
        return 0
    print("SELFTEST FAILED", file=sys.stderr)
    return 1


def audit_cmd(args):
    """Registry-driven coverage scan of EVERY set param -- general, any WRF namelist."""
    import re
    try:
        text = open(args.namelist).read()
    except Exception as exc:
        print("ERROR reading namelist: {}".format(exc), file=sys.stderr)
        sys.exit(1)
    nl_kv = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('!') or s.startswith('&') or s == '/':
            continue
        m = re.match(r'([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$', s)
        if m:
            nl_kv[m.group(1).lower()] = m.group(2).strip().rstrip(',').strip()
    resolved, _ = phys.resolve_suite(extract_case(parse_namelist(args.namelist)))
    print("=== wtune AUDIT -- Registry-driven coverage scan (general; works for any WRF namelist) ===")
    print("namelist:", args.namelist)
    print()
    lines, stats = phys.coverage_audit(nl_kv, resolved)
    for ln in lines:
        print(ln)


def _parse_nl_kv(path):
    import re
    nl_kv = {}
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('!') or s.startswith('&') or s == '/':
            continue
        m = re.match(r'([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$', s)
        if m:
            nl_kv[m.group(1).lower()] = m.group(2).strip().rstrip(',').strip()
    return nl_kv


# Golden-regression: params with a REAL measured-WRF result (an objective + a case).
# A CONFIRM/PASS never upgrades to WRF-VALIDATED on its own -- only a measured run does.
# A vet REJECT for the current case still wins over a known coarse-only result.
_KNOWN_WRF_VALIDATED = {
    'io_form_history': ('total_wall', '-30%', 'CONUS 2.5km'),
    'io_form_restart': ('total_wall', '-29%', 'CONUS 2.5km'),
    'io_form_input':   ('total_wall', 'same-field-values', 'CONUS 2.5km'),
    'nproc_x':         ('solver', '-1~3% (8x56)', 'CONUS 2.5km'),
    'nproc_y':         ('solver', '-1~3% (8x56)', 'CONUS 2.5km'),
    'non_hydrostatic': ('solver+radt', '-6%/step (30km); -15% combined (15km/448)', 'Matthew/Large coarse'),
}


def _evidence_level(name, verdict):
    """Map a screen candidate + its SpeedyWeather verdict to a traceable evidence level.
    CLASSIFIED < MODEL-ONLY < PROBE-ADVISORY < WRF-VALIDATED ; REJECTED / NEED-TRUTH are terminal.
    A CONFIRM means the REGIME/risk is confirmed cheaply -- NOT that performance is proven."""
    if verdict == 'REJECT':
        return 'REJECTED'
    if name in _KNOWN_WRF_VALIDATED:
        return 'WRF-VALIDATED'
    if verdict == 'CONFIRM':
        return 'PROBE-ADVISORY'
    if verdict == 'NEED-PROBE':
        return 'NEED-TRUTH'
    return 'MODEL-ONLY'            # PASS->WRF or analytic-only


def build_manifest(namelist, resolved, probe_src, cands, vets):
    """Candidate-level evidence manifest (JSON-serialisable). Separates 'classified' from
    'WRF-validated' so a reader cannot mistake a cheap CONFIRM for a measured win."""
    import json as _json
    items = []
    for c in cands:
        verdict, reason = vets[c['name']]
        lvl = _evidence_level(c['name'], verdict)
        item = {
            'param': c['name'],
            'current': str(c['current']),
            'candidate': c['candidate'],
            'tier': c['tier'],
            'objective': c.get('gain', 'unspecified'),
            'speedyweather_verdict': verdict,
            'evidence_level': lvl,
            'classification': wc.classify(c['name']),
            'why': c['why'],
        }
        if lvl == 'WRF-VALIDATED':
            obj, res, case = _KNOWN_WRF_VALIDATED[c['name']]
            item['wrf_measured'] = {'objective': obj, 'result': res, 'case': case}
        items.append(item)
    return _json.dumps({
        'namelist': namelist,
        'dx_m': resolved.get('dx'),
        'cu_physics': resolved.get('cu_physics'),
        'probe': probe_src,
        'evidence_legend': ['CLASSIFIED', 'MODEL-ONLY', 'PROBE-ADVISORY', 'WRF-VALIDATED', 'REJECTED', 'NEED-TRUTH'],
        'note': 'CONFIRM = regime/risk confirmed cheaply, NOT performance proven; WRF-VALIDATED = a measured run only.',
        'candidates': items,
    }, indent=2)


def screen_cmd(args):
    """3-stage autotuner funnel: analytic screen (all 2062) -> SpeedyWeather cheap
    physics vet -> the survivors go to WRF empirical test."""
    import wtune_surrogate as wsurr
    nl_kv = _parse_nl_kv(args.namelist)
    case = extract_case(parse_namelist(args.namelist))
    case, _ = enrich_case(case, getattr(args, 'wrfinput', None))
    resolved, _ = phys.resolve_suite(case)
    # The SpeedyWeather probe is PART of the tuner: run it automatically here (cached by
    # bbox) unless the user passed an explicit --probe-json or --no-probe.
    probe, probe_src = None, None
    if getattr(args, 'probe_json', None):
        probe, probe_src = wsurr.load_probe(args.probe_json), args.probe_json
    elif not getattr(args, 'no_probe', False):
        ppath, probe_note = auto_probe(args.namelist, case, resolved, args)
        if ppath:
            probe, probe_src = wsurr.load_probe(ppath), '{}  [{}]'.format(ppath, probe_note)
        else:
            probe_src = 'AUTO-PROBE SKIPPED: {}'.format(probe_note)
    print("=== wtune SCREEN -- autotuner funnel: all params -> a few effective, without running 2062 ===")
    print("namelist:", args.namelist)
    print("probe:", probe_src or "(--no-probe; physics candidates cannot be vetted)")
    if probe is not None:
        print()
        for line in wsurr.analyze_probe(probe, resolved):
            print(line)
    print()
    _, cnt = phys.coverage_audit(nl_kv, resolved)
    print("STAGE 0 -- full catalog classified: {} total = {} set / {} relevant-default / {} inert / {} plumbing".format(
        (cnt['set'] + cnt['relevant_default'] + cnt['inert'] + cnt['plumbing']),
        cnt['set'], cnt['relevant_default'], cnt['inert'], cnt['plumbing']))
    nest = phys.nesting_advice(nl_kv, resolved)
    if nest:
        print()
        for line in nest:
            print(line)
    cands = phys.screen_candidates(nl_kv, resolved)
    level = getattr(args, 'compliance', None)
    hidden = 0
    if level is not None:
        kept = []
        for c in cands:
            if wc.admits(level, wc.classify(c['name'])):
                kept.append(c)
            else:
                hidden += 1
        cands = kept
    print("\nSTAGE 1 -- analytic effective candidates ({}) [everything else: default optimal / correctness / inert]:".format(len(cands)))
    for c in cands:
        print("  T{:<4} [{:13s}] {:<15} {:<16} -> {:<26} {}".format(
            c['tier'], wc.classify(c['name']), c['name'], str(c['current'])[:16], c['candidate'], c['why']))
    if level is not None and hidden:
        print("  ({} candidate(s) hidden as unclassified/disallowed at --compliance {})".format(hidden, level))
    print("\nSTAGE 2 -- SpeedyWeather cheap physics vet (rejects physics-infeasible BEFORE any WRF run):")
    survivors = []
    vets = {}
    for c in cands:
        verdict, reason = wsurr.speedy_vet(c, probe, resolved)
        vets[c['name']] = (verdict, reason)
        print("  {:<15} [{:13s}] {:<11} {}  [{}]".format(
            c['name'], wc.classify(c['name']), verdict, reason, _evidence_level(c['name'], verdict)))
        if verdict in ('PASS->WRF', 'CONFIRM'):
            survivors.append((c, verdict))
    if getattr(args, 'json', False):
        print("\n=== EVIDENCE MANIFEST (JSON) ===")
        print(build_manifest(args.namelist, resolved, probe_src, cands, vets))
        return
    print("\nSTAGE 3 -- survivors -> WRF empirical ABBA (the 'few effective' to actually test):")
    for c, v in sorted(survivors, key=lambda x: {'1': 0, '2': 1, 'swap': 2}.get(x[0]['tier'], 3)):
        print("  [T{}] [{:13s}] {:<15} -> {:<26} ({})".format(c['tier'], wc.classify(c['name']), c['name'], c['candidate'], v))
    print("\nmeasured winners so far: io_form_history=14 (-30% wall, if I/O scored), 8x56 decomposition (-1~3%).")
    print("the rest are candidates -- run them through wtune_probe (Stage C) with the pre-registered gate.")


def _restart_error(message):
    print("ERROR restart-window: {}".format(message), file=sys.stderr)
    return 1


def _restart_rank_counts(value, case):
    if value:
        try:
            counts = [int(part.strip()) for part in value.split(',')]
        except ValueError:
            raise ValueError("rank counts must be comma-separated positive integers")
        if not counts or any(count < 1 for count in counts):
            raise ValueError("rank counts must be comma-separated positive integers")
        return list(dict.fromkeys(counts))
    px, py = case.get('nproc_x'), case.get('nproc_y')
    if px is None or py is None or int(px) < 1 or int(py) < 1:
        raise ValueError("pass --rank-counts when namelist nproc_x/nproc_y are unavailable")
    return [int(px) * int(py)]


def _restart_tile_config(case):
    config = {'strategy': int(case.get('tile_strategy', 0) or 0)}
    for key in ('numtiles', 'numtiles_x', 'numtiles_y'):
        value = case.get(key)
        if value is not None and int(value) > 0:
            config[key] = int(value)
    return config


def _restart_loads(oracle, fields, weights, px, py):
    loads = None
    for phase, weight in weights.items():
        if phase not in fields:
            continue
        grid = oracle.to_ranks(fields[phase], px, py)
        values = [float(value) for row in grid for value in row]
        if loads is None:
            loads = [0.0] * len(values)
        if len(values) != len(loads):
            raise ValueError("oracle rank field shape is inconsistent")
        for index, value in enumerate(values):
            loads[index] += float(weight) * value
    if not loads:
        raise ValueError("oracle returned no weighted phase fields")
    return loads


def _restart_path_key(path):
    return os.path.realpath(os.path.abspath(path))


def _validate_restart_paths(inputs, manifest_path, output_dir, artifacts=None):
    input_keys = set(_restart_path_key(path) for path in inputs)
    if _restart_path_key(output_dir) in input_keys:
        raise ValueError("input/output path collision: {}".format(output_dir))
    output_keys = set()
    for path in [manifest_path] + list(artifacts or []):
        key = _restart_path_key(path)
        if key in input_keys:
            raise ValueError("input/output path collision: {}".format(path))
        if key in output_keys:
            raise ValueError("duplicate output path: {}".format(path))
        output_keys.add(key)


def _validate_restart_allocations(allocations, rank_count, topology):
    resources = {}
    for node in topology['nodes']:
        for domain in node['numa_domains']:
            resources[(node['name'], domain['id'])] = set(domain['cpus'])
    allowed_sizes = set(topology['allowed_team_sizes'])
    ranks, cpus = set(), set()
    if not isinstance(allocations, list) or len(allocations) != rank_count:
        raise ValueError("allocation must contain exactly one entry per rank")
    for allocation in allocations:
        if not isinstance(allocation, dict):
            raise ValueError("allocation entry is malformed")
        rank = allocation.get('rank')
        node = allocation.get('node')
        domain = allocation.get('numa_domain')
        team_size = allocation.get('team_size')
        assigned = allocation.get('cpus')
        if type(rank) is not int or rank in ranks:
            raise ValueError("allocation rank overlap or invalid rank")
        if not isinstance(node, str) or not node or type(domain) is not int:
            raise ValueError("allocation node/domain map is malformed")
        if type(team_size) is not int or team_size < 1 or team_size not in allowed_sizes:
            raise ValueError("allocation team_size is invalid")
        if not isinstance(assigned, list) or not assigned or len(assigned) != team_size:
            raise ValueError("allocation team_size does not match CPUs")
        resource = (node, domain)
        if resource not in resources:
            raise ValueError("allocation references an unknown node or NUMA domain")
        resource_key = allocation.get('resource_key')
        expected_key = "%s:%s" % resource
        if resource_key is not None and resource_key != expected_key:
            raise ValueError("allocation resource_key does not match node/domain")
        if any(type(cpu) is not int or cpu < 0 for cpu in assigned):
            raise ValueError("allocation node/cpu map is malformed")
        if len(set(assigned)) != len(assigned):
            raise ValueError("allocation CPU list contains duplicates")
        ranks.add(rank)
        for cpu in assigned:
            key = (node, cpu)
            if key in cpus:
                raise ValueError("allocation node-local CPU overlap")
            if cpu not in resources[resource]:
                raise ValueError("allocation CPU is absent from topology")
            cpus.add(key)
    if ranks != set(range(rank_count)):
        raise ValueError("allocation ranks must be exactly 0..N-1")


def _stage_write(path, text, executable=False):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    descriptor, temporary = tempfile.mkstemp(prefix='.wtune-', dir=directory)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(text)
        if executable:
            os.chmod(temporary, 0o755)
        return temporary
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _stage_backup(path):
    descriptor, temporary = tempfile.mkstemp(
        prefix='.wtune-backup-', dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(descriptor, 'wb') as destination, open(path, 'rb') as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                destination.write(block)
        os.chmod(temporary, os.stat(path).st_mode & 0o777)
        return temporary
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _remove_staged(paths):
    for path in paths:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def _publish_restart_artifacts(outputs, manifest_path, manifest_text):
    entries = list(outputs) + [(manifest_path, manifest_text, False)]
    staged, backups, published = [], {}, []
    try:
        for path, text, executable in entries:
            staged.append([path, _stage_write(path, text, executable), executable])
        for path, unused_temporary, unused_executable in staged:
            if os.path.exists(path):
                backups[path] = _stage_backup(path)
        for entry in staged:
            path, temporary, unused_executable = entry
            os.replace(temporary, path)
            entry[1] = None
            published.append(path)
    except Exception:
        for path in reversed(published):
            backup = backups.pop(path, None)
            try:
                if backup is None:
                    os.unlink(path)
                else:
                    os.replace(backup, path)
            except OSError:
                pass
        raise
    finally:
        _remove_staged([entry[1] for entry in staged])
        _remove_staged(list(backups.values()))


def restart_window_cmd(args):
    # DEPRECATED 2026-09-11: restart-boundary rebalancing is closed (a restart
    # costs ~56 s, the recoverable imbalance is 7-17 s; PLOV rebalances in place
    # instead, see plov/README.md). The modules live under deprecated/ and are
    # imported only when this command is actually invoked.
    print("WARNING restart-window: deprecated path (see deprecated/README.md); "
          "PLOV is the live rebalancer", file=sys.stderr)
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'deprecated'))
    from wtune_migration_config import (MigrationConfigError, from_environment,
                                        manifest_metadata, validate_case)
    output_dir = _restart_path_key(args.output_dir)
    manifest_path = (_restart_path_key(args.manifest_out)
                     if args.manifest_out else
                     _restart_path_key(os.path.join(
                         output_dir, 'restart-window-manifest.json')))
    if args.top_k <= 0:
        return _restart_error("top-k must be positive")
    if args.stride <= 0:
        return _restart_error("stride must be positive")
    try:
        nl = parse_namelist(args.namelist)
        case = extract_case(nl)
    except Exception as exc:
        return _restart_error("reading namelist: {}".format(exc))
    try:
        migration_config = from_environment()
        validate_case(migration_config, case)
    except MigrationConfigError as exc:
        return _restart_error(str(exc))
    try:
        _validate_restart_paths((args.namelist, args.snapshot, args.topology_json),
                                manifest_path, output_dir)
    except ValueError as exc:
        return _restart_error(str(exc))
    snapshot = args.snapshot
    try:
        with open(snapshot, 'rb'):
            pass
    except (IOError, OSError) as exc:
        return _restart_error("snapshot is not readable: {}".format(exc))
    try:
        with open(args.topology_json) as handle:
            topology_raw = json.load(handle)
        import wtune_restart as restart
        topology = restart.validate_topology(topology_raw)
    except (IOError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return _restart_error("topology is not readable or valid: {}".format(exc))
    try:
        rank_counts = _restart_rank_counts(args.rank_counts, case)
        tile_config = _restart_tile_config(case)
        restart.normalize_tile_config(tile_config, case['ny'])
    except ValueError as exc:
        return _restart_error(str(exc))
    resolved, _ = phys.resolve_suite(case)
    cu_on = int(resolved.get('cu_physics', 0) or 0) > 0
    try:
        import wtune_oracle as oracle
        features = oracle.extract_features(snapshot, stride=args.stride,
                                           include_derived=cu_on,
                                           include_surface_fluxes=False)
    except Exception as exc:
        return _restart_error("snapshot feature extraction failed: {}".format(exc))

    try:
        import datetime as _dt
        stamp = _dt.datetime(int(resolved['start_year']), int(resolved['start_month']),
                             int(resolved['start_day']), int(resolved['start_hour']),
                             int(resolved.get('start_minute', 0)),
                             int(resolved.get('start_second', 0)))
        windows = restart.build_windows(resolved['run_total_hours'], resolved['restart_interval'],
                                        start_utc=stamp, window_hours=args.window_hours,
                                        run_hours_override=args.run_hours)
    except (TypeError, ValueError) as exc:
        return _restart_error("restart windows: {}".format(exc))

    weights = oracle.phase_weights(cu_on=cu_on)
    outputs, manifest_windows = [], []
    try:
        for window in windows:
            fields = oracle.phase_cost_fields(features, window['day_of_year'],
                                              (float(resolved.get('start_hour', 0)) +
                                               float(resolved.get('start_minute', 0)) / 60.0 +
                                               float(resolved.get('start_second', 0)) / 3600.0 +
                                               window['start_hour']),
                                              run_hours=window['end_hour'] - window['start_hour'],
                                              cu_on=cu_on)
            choices, rejected = [], []
            for rank_count in rank_counts:
                candidates = restart.enumerate_candidates(rank_count, case['nx'], case['ny'], topology,
                                                          local_patch_y=case['ny'])
                ranked = []
                for candidate in candidates:
                    loads = _restart_loads(oracle, fields, weights, candidate['px'], candidate['py'])
                    ranked.append((restart.score_candidate(candidate, loads), loads))
                ranked.sort(key=lambda item: (item[0]['score'], item[0]['px'], item[0]['py']))
                selected = 0
                for candidate, loads in ranked:
                    if selected >= args.top_k:
                        break
                    try:
                        allocations = restart.allocate_teams(
                            loads, topology['allowed_team_sizes'], topology)
                    except ValueError as exc:
                        rejected.append({'rank_count': rank_count, 'px': candidate['px'],
                                         'py': candidate['py'], 'reason': str(exc)})
                        continue
                    _validate_restart_allocations(allocations, rank_count, topology)
                    config = restart.normalize_tile_config(tile_config, candidate['min_patch_y'])
                    spec = {'config': config, 'window': window, 'candidate': candidate,
                            'allocations': allocations, 'local_patch_y': candidate['min_patch_y'],
                            'proxy_cost': None, 'transition_cost': 'UNKNOWN', 'restart_cost': None}
                    suffix = 'window-{0:03d}-ranks-{1}'.format(window['id'], rank_count)
                    if selected:
                        suffix += '-candidate-{}'.format(selected)
                    map_path = _restart_path_key(os.path.join(output_dir, suffix + '.map'))
                    wrapper_path = _restart_path_key(os.path.join(output_dir, suffix + '.sh'))
                    outputs.append((map_path, restart.render_map(spec), False))
                    outputs.append((wrapper_path, restart.render_wrapper(spec), True))
                    choices.append({'rank_count': rank_count, 'candidate': candidate,
                                    'allocations': allocations, 'map': map_path,
                                    'wrapper': wrapper_path})
                    selected += 1
                if not candidates:
                    return _restart_error("rank count {} has no legal rectangular patches".format(rank_count))
                if not selected:
                    return _restart_error("rank count {} has no selectable allocated candidate".format(rank_count))
            manifest_windows.append(dict(window, choices=choices, rejected=rejected))
    except (TypeError, ValueError, KeyError) as exc:
        return _restart_error("candidate scoring/allocation failed: {}".format(exc))

    try:
        _validate_restart_paths((args.namelist, args.snapshot, args.topology_json), manifest_path,
                                output_dir, [item[0] for item in outputs])
    except ValueError as exc:
        return _restart_error(str(exc))
    manifest = restart.render_manifest({
        'config': restart.normalize_tile_config(tile_config, case['ny']),
        'local_patch_y': case['ny'], 'rank_counts': rank_counts, 'windows': manifest_windows,
        'proxy_cost': None, 'transition_cost': 'UNKNOWN', 'restart_cost': None,
    })
    manifest['migration'] = manifest_metadata(migration_config)
    try:
        _publish_restart_artifacts(
            outputs, manifest_path,
            json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n')
    except (IOError, OSError) as exc:
        return _restart_error("writing artifacts: {}".format(exc))
    print("created: {}".format(manifest_path))
    for path, unused_text, unused_executable in outputs:
        print("created: {}".format(path))
    print("UNVERIFIED: changed rank/shape/team/CPU maps; actionable=false")
    return 0

def feedback_cmd(args):
    try:
        with open(args.input) as handle:
            document = json.load(handle)
        if not isinstance(document, dict) or not isinstance(document.get('samples'), list):
            raise FeedbackError('input must contain a samples array')
        limits = document.get('limits', {})
        if not isinstance(limits, dict):
            raise FeedbackError('limits must be an object')
        controller = FeedbackController(**limits)
        decisions = [controller.observe(FeedbackSample.from_dict(item))
                     for item in document['samples']]
        json_text, csv_text = decisions_to_json(decisions), decisions_to_csv(decisions)
        _validate_restart_paths((args.input,), args.json_out, os.path.dirname(args.json_out) or '.',
                                [args.csv_out])
        _publish_restart_artifacts([(args.csv_out, csv_text, False)], args.json_out, json_text)
    except (IOError, OSError, ValueError, TypeError, FeedbackError, json.JSONDecodeError) as exc:
        return _restart_error('feedback input/output: {}'.format(exc))
    print('created: {}'.format(args.json_out))
    print('created: {}'.format(args.csv_out))
    print('manifest-only: actionable=false')
    return 0


def main():
    parser = argparse.ArgumentParser(prog='wtune.py', description='WRF config autotuner analytical core')
    parser.add_argument('--version', action='version',
                        version='wtune %s' % wtune_version.__version__)
    sub = parser.add_subparsers(dest='command')

    p_plan = sub.add_parser('plan', help='generate ranked decomposition shortlist')
    p_plan.add_argument('--namelist', required=True, help='path to namelist.input')
    p_plan.add_argument('--ranks', type=int, default=None, help='total MPI ranks')
    p_plan.add_argument('--nodes', type=int, default=None, help='number of nodes')
    p_plan.add_argument('--cores-per-node', type=int, default=None, help='default 112')
    p_plan.add_argument('--numa-domains', type=int, default=None, help='default 8')
    p_plan.add_argument('--cores-per-numa', type=int, default=None, help='default derived or 14')
    p_plan.add_argument('--smt', choices=['on', 'off'], default='off', help='default off')
    p_plan.add_argument('--json', action='store_true', help='emit JSON')
    p_plan.add_argument('--allow-hybrid', action='store_true', help='report hybrid alternative')
    p_plan.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h)')
    p_plan.add_argument('--surrogate-json', default=None, help='precomputed SpeedyWeather hotspot JSON to include')

    p_hot = sub.add_parser('hotspot', help='physics load map: analytic terminator + convective surrogate')
    p_hot.add_argument('--namelist', required=True, help='path to namelist.input')
    p_hot.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date')
    p_hot.add_argument('--surrogate-json', default=None, help='use a precomputed SpeedyWeather hotspot JSON')
    p_hot.add_argument('--run-surrogate', action='store_true', help='run SpeedyWeather now (slow: JIT + model)')
    p_hot.add_argument('--trunc', type=int, default=31, help='SpeedyWeather spectral truncation (default 31)')
    p_hot.add_argument('--nlayers', type=int, default=8, help='SpeedyWeather vertical layers (default 8)')
    p_hot.add_argument('--srun', default=None, help='srun prefix to run the surrogate on a compute node')

    p_phys = sub.add_parser('physics', help='physics diagnosis + numerics advice + scheme menu')
    p_phys.add_argument('--namelist', required=True, help='path to namelist.input')
    p_phys.add_argument('--menu', action='store_true', help='also print the advisory scheme menu')
    p_phys.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h); also sets the probe bbox')
    p_phys.add_argument('--probe-json', default=None, help='use THIS speedy_probe.jl JSON instead of auto-running the probe')
    p_phys.add_argument('--wps', default=None, help='namelist.wps for the domain centre (default: sibling of --namelist)')
    p_phys.add_argument('--latlon', default=None, help='override domain-centre "LAT,LON" for the probe bbox')
    p_phys.add_argument('--probe-hours', type=int, default=6, help='SpeedyWeather probe forecast length (default 6)')
    p_phys.add_argument('--probe-trunc', type=int, default=31, help='SpeedyWeather spectral truncation (default 31); part of the probe cache key')
    p_phys.add_argument('--probe-nlayers', type=int, default=8, help='SpeedyWeather vertical layers (default 8); part of the probe cache key')
    p_phys.add_argument('--no-probe', action='store_true', help='skip the built-in SpeedyWeather probe')
    p_phys.add_argument('--compliance', choices=sorted(wc.LEVELS), default=None,
                        help='keep only findings admitted at this compliance level (decision 0050); default: show all, labelled')

    p_knobs = sub.add_parser('knobs', help='full knob advisor: input-derived param recommendations + mismatch flags')
    p_knobs.add_argument('--namelist', required=True, help='path to namelist.input')
    p_knobs.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h)')
    p_knobs.add_argument('--all', action='store_true', help='show OK findings too, not just actionable')
    p_knobs.add_argument('--compliance', choices=sorted(wc.LEVELS), default=None,
                         help='keep only findings admitted at this compliance level (decision 0050); default: show all, labelled')

    p_cat = sub.add_parser('catalog', help='list the complete Registry-derived parameter catalog')
    p_cat.add_argument('--group', default=None, help='filter to one namelist group (e.g. physics)')
    p_cat.add_argument('--grep', default=None, help='filter by substring in param name')
    p_cat.add_argument('--include-io', action='store_true', help='include per-stream I/O plumbing params')

    p_audit = sub.add_parser('audit', help='Registry-driven scan of EVERY (2062) catalog param (coverage guarantee, any WRF case)')
    p_audit.add_argument('--namelist', required=True, help='path to namelist.input')

    p_screen = sub.add_parser('screen', help='autotuner funnel: analytic screen (all params) -> SpeedyWeather vet -> effective few')
    p_screen.add_argument('--namelist', required=True, help='path to namelist.input')
    p_screen.add_argument('--probe-json', default=None, help='use THIS speedy_probe.jl JSON instead of auto-running the probe')
    p_screen.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h); also sets the probe bbox')
    p_screen.add_argument('--wps', default=None, help='namelist.wps for the domain centre (default: sibling of --namelist)')
    p_screen.add_argument('--latlon', default=None, help='override domain-centre "LAT,LON" for the probe bbox')
    p_screen.add_argument('--probe-hours', type=int, default=6, help='SpeedyWeather probe forecast length (default 6)')
    p_screen.add_argument('--probe-trunc', type=int, default=31, help='SpeedyWeather spectral truncation (default 31); part of the probe cache key')
    p_screen.add_argument('--probe-nlayers', type=int, default=8, help='SpeedyWeather vertical layers (default 8); part of the probe cache key')
    p_screen.add_argument('--no-probe', action='store_true', help='skip the built-in SpeedyWeather probe (physics candidates stay NEED-PROBE)')
    p_screen.add_argument('--json', action='store_true', help='emit a candidate-level JSON evidence manifest (classification/evidence_level/objective) after the vet')
    p_screen.add_argument('--compliance', choices=sorted(wc.LEVELS), default=None,
                          help='keep only candidates admitted at this compliance level (decision 0050); default: show all, labelled')

    p_profile = sub.add_parser('profile', help='case-understanding profile (PRIOR, pre-measurement): identity/physics/cost-structure/imbalance/core-count priors')
    p_profile.add_argument('case_dir', help='directory with namelist.input (+ optional wrfinput_d0N)')
    p_profile.add_argument('--namelist', default=None, help='explicit namelist.input path (default: <case_dir>/namelist.input)')
    p_profile.add_argument('--wrfinput', default=None, help='explicit wrfinput/wrfout path (default: first wrfinput_d0N in case_dir)')
    p_profile.add_argument('--cores-per-node', type=int, default=None, help='default 72 (GH200 Grace)')
    p_profile.add_argument('--json', default=None, help='also write the full profile as JSON to this path')

    p_omap = sub.add_parser('oracle-map', help='DIAGNOSTIC spatial physics-load map (per-cell phase cost + hotspots); NOT a partition selector')
    p_omap.add_argument('--namelist', required=True, help='path to namelist.input (for date + cu_physics + run length)')
    p_omap.add_argument('--snapshot', default=None, help='wrfout/wrfinput NetCDF with model-state fields (defaults to --wrfinput)')
    p_omap.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h); also the snapshot fallback')
    p_omap.add_argument('--stride', type=int, default=4, help='spatial subsample stride for the model-state read (default 4)')
    p_omap.add_argument('--top-k', dest='top_k', type=int, default=12, help='number of hottest cells to list (default 12)')
    p_omap.add_argument('--grid', type=int, default=24, help='max side of the per-phase JSON grid (block-mean downsample, default 24)')
    p_omap.add_argument('--json-out', dest='json_out', default=None, help='write the artifact JSON here (compact grids + top-k)')
    p_omap.add_argument('--npz-out', dest='npz_out', default=None, help='write the full-resolution per-phase fields here as .npz')

    p_restart = sub.add_parser('restart-window', help='UNVERIFIED restart-boundary decomposition/team hypotheses')
    p_restart.add_argument('--namelist', required=True, help='path to namelist.input (never modified)')
    p_restart.add_argument('--snapshot', required=True, help='readable wrfout/wrfrst snapshot for oracle fields')
    p_restart.add_argument('--topology-json', required=True, help='topology JSON with node-local CPU lists')
    p_restart.add_argument('--rank-counts', '--ranks', dest='rank_counts', default=None,
                           help='comma-separated MPI rank counts (default nproc_x*nproc_y)')
    p_restart.add_argument('--window-hours', type=float, default=None, help='restart advisor window size in hours')
    p_restart.add_argument('--run-hours', type=float, default=None, help='override namelist run duration in hours')
    p_restart.add_argument('--stride', type=int, default=4, help='snapshot spatial subsample stride (default 4)')
    p_restart.add_argument('--top-k', dest='top_k', type=int, default=1, help='candidates per rank count/window')
    p_restart.add_argument('--output-dir', default='restart-window-artifacts', help='directory for rank maps and wrappers')
    p_restart.add_argument('--manifest-out', default=None, help='overall canonical JSON manifest path')

    p_feedback = sub.add_parser('feedback', help='manifest-only advisory feedback decisions')
    p_feedback.add_argument('--input', required=True, help='JSON samples and controller limits')
    p_feedback.add_argument('--json-out', required=True, help='decision JSON output')
    p_feedback.add_argument('--csv-out', required=True, help='canonical telemetry/decision CSV output')

    sub.add_parser('selftest', help='run CONUS calibration self-test')

    p_compliance = sub.add_parser('compliance', help='compliance model + knob catalog (decision 0050): validate/list/classify')
    p_compliance.add_argument('compliance_args', nargs=argparse.REMAINDER,
                              help='forwarded to wtune_compliance.py (validate | list [--compliance LEVEL] [--json] | classify NAME...)')

    args = parser.parse_args()

    if args.command == 'plan':
        plan(args)
    elif args.command == 'physics':
        physics_cmd(args)
    elif args.command == 'profile':
        status = profile_cmd(args)
        if status:
            sys.exit(status)
    elif args.command == 'hotspot':
        hotspot_cmd(args)
    elif args.command == 'oracle-map':
        oracle_map_cmd(args)
    elif args.command == 'restart-window':
        status = restart_window_cmd(args)
        if status:
            sys.exit(status)
    elif args.command == 'feedback':
        status = feedback_cmd(args)
        if status:
            sys.exit(status)
    elif args.command == 'knobs':
        knobs_cmd(args)
    elif args.command == 'catalog':
        catalog_cmd(args)
    elif args.command == 'audit':
        audit_cmd(args)
    elif args.command == 'screen':
        screen_cmd(args)
    elif args.command == 'selftest':
        sys.exit(selftest())
    elif args.command == 'compliance':
        sys.exit(wc._main(args.compliance_args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
