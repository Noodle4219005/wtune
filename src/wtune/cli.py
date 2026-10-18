#!/usr/bin/env python3
"""wtune/cli.py - analytical core (Stage A + Stage B) of a WRF config autotuner."""

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

# sibling modules (same directory as this script)
if not __package__:                    # run by path: src/wtune itself is not an import root
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wtune import physics as phys
from wtune import solar as wsolar
from wtune import compliance as wc
import wtune.machine
import wtune.version
from wtune import runlength as runlen         # resolve_run_seconds: single-sourced run_* keys vs.
                                          # start/end date-span resolver (dependency-free)

FLOOR = 10              # minimum mass points in each direction after halo
HALO = 5                # WRF halo width approximation
KV = wtune.machine.MEASURED["vec_kv"]   # built-in; vectorization/amortization constant,
                        # calibrated to the measured
                        # CONUS 2.5km sweep so the RANKING matches wall time:
                        # 8x56 (0.9219, best) < 16x28 (0.9393) < 4x112 (1.0307, worst).
                        # KV=100 over-rewarded wide patches -> wrongly ranked 4x112 above 16x28.


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
    # A group ends at a slash standing alone at the end of a line (after the line start, a
    # comma or blank), not at any slash: namelist.output writes string defaults unquoted,
    # e.g. `PATH_TO_FILES = ~/WRFV3/,` inside &physics, which used to end the group early.
    end_re = re.compile(r'(^|[\s,])/\s*$')

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = re.match(r'&([A-Za-z_]\w*)', line)
        if m:
            current = m.group(1).lower()
            groups.setdefault(current, [])
            rest = line[m.end():].strip()
            ends = end_re.search(rest)
            if ends:
                rest = rest[:ends.start()].strip()
            if rest:
                groups[current].append(rest)
            if ends:
                current = None
            continue

        ends = end_re.search(line)
        if ends:
            before = line[:ends.start()].strip()
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
    run_minutes = scalar(t, 'run_minutes', 0) or 0
    run_seconds = scalar(t, 'run_seconds', 0) or 0

    import datetime as _dt

    def _time_control_datetime(prefix):
        """d01 <prefix>_year/month/day/hour/minute/second (first value) -> datetime, or None when
        year/month/day are not all present. Local to run-length resolution only: the output
        dict's own 'start_year' etc. fields below default to a synthetic 2021-01-01 for OTHER
        consumers (solar/radiation calc) and must not leak into the run-length date-span
        fallback, which needs to know when a date is genuinely absent."""
        y = scalar(t, prefix + '_year', None)
        mo = scalar(t, prefix + '_month', None)
        dy = scalar(t, prefix + '_day', None)
        if y is None or mo is None or dy is None:
            return None
        h = scalar(t, prefix + '_hour', 0) or 0
        mi = scalar(t, prefix + '_minute', 0) or 0
        se = scalar(t, prefix + '_second', 0) or 0
        return _dt.datetime(int(y), int(mo), int(dy), int(h), int(mi), int(se))

    # run_total_hours: run_days/run_hours/run_minutes/run_seconds when their sum is > 0 (this
    # used to silently DROP run_minutes/run_seconds, e.g. isc21's run_minutes=30 -> 0); otherwise
    # the start_date -> end_date span (this case, e.g. lafe's all-four-zero shape, used to have
    # no fallback at all and also come out 0). Falls back to 0.0 ("unset", extract_case's own
    # existing convention for every other missing field) rather than raising, since extract_case
    # is called from contexts that never need run length at all (physics-suite resolution, pure
    # domain-geometry queries) and must keep returning a best-effort dict.
    try:
        total_run_hours = runlen.resolve_run_seconds(
            run_days, run_hours, run_minutes, run_seconds,
            _time_control_datetime('start'), _time_control_datetime('end')) / 3600.0
    except ValueError:
        total_run_hours = 0.0

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
    wtune_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    vec_factor = 1.0 + wtune.machine.measured("vec_kv") / patch_x

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


def radiation_spacing(case):
    radt = case.get('radt', 0.0) or 0.0
    dt = case.get('time_step', 60) or 60
    if dt <= 0:
        dt = 60
    if radt <= 0:
        return 1
    return max(1, int(round(radt * 60.0 / dt)))


def machine_profile_cmd(args):
    """Hook into machine/profile.py: print a human summary of a merged
    machine profile JSON (produced by machine/run_machine_profile.sh).
    Loaded via importlib (not sys.path + import) so machine/profile.py
    never shadows the stdlib 'profile' module for the rest of the process.
    """
    import importlib.util
    mp_path = Path(__file__).resolve().parent.parent.parent / 'machine' / 'profile.py'
    if not mp_path.exists():
        print("ERROR: {} not found".format(mp_path), file=sys.stderr)
        return 1
    spec = importlib.util.spec_from_file_location('wtune_machine_profile', mp_path)
    mprofile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mprofile)
    try:
        doc = json.loads(Path(args.json_path).read_text())
    except (IOError, OSError, ValueError) as exc:
        print("ERROR reading {}: {}".format(args.json_path, exc), file=sys.stderr)
        return 1
    for line in mprofile.summarize(doc):
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
    _print_extension_notes(nl, args.namelist)
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
    from wtune import surrogate as wsurr
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
    from wtune import surrogate as wsurr
    try:
        nl_kv = _parse_nl_kv(args.namelist)
        case = extract_case(parse_namelist(args.namelist))
    except Exception as exc:
        print("ERROR reading namelist: {}".format(exc), file=sys.stderr)
        return 1
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


def _print_extension_notes(nl, namelist_path):
    """One support line per detected extension (plan/physics/knobs): grid/rank-only policy."""
    import wtune.extensions
    try:
        import wtune.predict
        pnl = wtune.predict.parse_namelist(namelist_path)
    except Exception:
        return
    run_dir = os.path.dirname(os.path.abspath(namelist_path))
    dets = [d for d in wtune.extensions.detect(pnl, run_dir) if d['active']]
    for d in dets:
        print('EXTENSION: ' + wtune.extensions.support_line(d))
    if any(d['id'] == 'wrfda' for d in dets):
        print('EXTENSION: ' + wtune.extensions.DA_ADVISORY_BANNER)
    for _uid, label, msg, hits in wtune.extensions.detect_unsupported(run_dir):
        print('EXTENSION: ' + wtune.extensions.unsupported_line(label, msg, hits))


# ---------------------------------------------------------------- command-line front end
#
# main() = _build_parser() + dispatch. Subcommands come in two kinds:
#   * in-process: implemented in this module (plan, physics, ...), parsed here, dispatched
#     through _IN_PROCESS_COMMANDS;
#   * forwarded: the subcommand takes its whole argv as one REMAINDER positional and hands it
#     to another module's main(argv) (_FORWARDED_COMMANDS), which owns its own parser.
# Subparsers are registered in the order `wtune/cli.py --help` lists them.

_COMPLIANCE_HELP = ('keep only %s admitted at this compliance level (decision 0050); '
                    'default: show all, labelled')
_PROBE_WRFINPUT_HELP = 'wrfinput/wrfout for lat/lon/date (via ncdump -h); also sets the probe bbox'
_PROBE_JSON_HELP = 'use THIS speedy_probe.jl JSON instead of auto-running the probe'


def _add_compliance_arg(p, what):
    p.add_argument('--compliance', choices=sorted(wc.LEVELS), default=None,
                   help=_COMPLIANCE_HELP % what)


def _add_probe_window_args(p):
    """Domain-centre + SpeedyWeather probe settings shared by `physics` and `screen`."""
    p.add_argument('--wps', default=None, help='namelist.wps for the domain centre (default: sibling of --namelist)')
    p.add_argument('--latlon', default=None, help='override domain-centre "LAT,LON" for the probe bbox')
    p.add_argument('--probe-hours', type=int, default=6, help='SpeedyWeather probe forecast length (default 6)')
    p.add_argument('--probe-trunc', type=int, default=31, help='SpeedyWeather spectral truncation (default 31); part of the probe cache key')
    p.add_argument('--probe-nlayers', type=int, default=8, help='SpeedyWeather vertical layers (default 8); part of the probe cache key')


def _add_physics_parser(sub):
    p = sub.add_parser('physics', help='physics diagnosis + numerics advice + scheme menu')
    p.add_argument('--namelist', required=True, help='path to namelist.input')
    p.add_argument('--menu', action='store_true', help='also print the advisory scheme menu')
    p.add_argument('--wrfinput', default=None, help=_PROBE_WRFINPUT_HELP)
    p.add_argument('--probe-json', default=None, help=_PROBE_JSON_HELP)
    _add_probe_window_args(p)
    p.add_argument('--no-probe', action='store_true', help='skip the built-in SpeedyWeather probe')
    _add_compliance_arg(p, 'findings')


def _add_screen_parser(sub):
    p = sub.add_parser('screen', help='autotuner funnel: analytic screen (all params) -> SpeedyWeather vet -> effective few')
    p.add_argument('--namelist', required=True, help='path to namelist.input')
    p.add_argument('--probe-json', default=None, help=_PROBE_JSON_HELP)
    p.add_argument('--wrfinput', default=None, help=_PROBE_WRFINPUT_HELP)
    _add_probe_window_args(p)
    p.add_argument('--no-probe', action='store_true', help='skip the built-in SpeedyWeather probe (physics candidates stay NEED-PROBE)')
    p.add_argument('--json', action='store_true', help='emit a candidate-level JSON evidence manifest (classification/evidence_level/objective) after the vet')
    _add_compliance_arg(p, 'candidates')


def _add_machine_profile_parser(sub):
    p = sub.add_parser('machine-profile', help='print a human summary of a machine/run_machine_profile.sh JSON (MEASURED machine speed, not the case profile)')
    p.add_argument('json_path', help='machine_profile_<host>_<date>.json produced by machine/run_machine_profile.sh')


def _add_selftest_parser(sub):
    sub.add_parser('selftest', help='run CONUS calibration self-test')


def _add_compliance_parser(sub):
    p = sub.add_parser('compliance', help='compliance model + knob catalog (decision 0050): validate/list/classify')
    p.add_argument('compliance_args', nargs=argparse.REMAINDER,
                   help='forwarded to wtune/compliance.py (validate | list [--compliance LEVEL] [--json] | classify NAME...)')


def _add_predict_parser(sub):
    import wtune.predict
    wtune.predict.add_subparser(sub)


# In-process subcommand builders, in --help order (`predict` is defined by wtune.predict).
_IN_PROCESS_PARSERS = (
    _add_physics_parser, _add_screen_parser, _add_machine_profile_parser,
    _add_selftest_parser, _add_compliance_parser, _add_predict_parser,
)

# Forwarded subcommands, in --help order: (name, REMAINDER dest, help, script the help says the
# argv goes to, module imported to run it). The dest is visible: it names the positional in the
# subcommand's usage line. `prep` has no module entry: see _forwarded_module.
_FORWARDED_COMMANDS = (
    ('submit', 'submit_args', 'tune -> run -> post for the current project in one command '
     '(wtune/submit.py)', 'wtune/submit.py', 'wtune.submit'),
    ('doctor', 'doctor_args', 'check that this machine and project are ready (wtune/doctor.py)',
     'wtune/doctor.py', 'wtune.doctor'),
    ('tune', 'tune_args', 'budgeted on-cluster tuner (scc): probe, screen, confirm, report',
     'wtune/tune/', 'wtune.tune'),
    ('config', 'config_args', 'wtune tune search-space YAML config (wtune/config.py): init | show | validate',
     'wtune/config.py', 'wtune.config'),
    ('metdata', 'metdata_args', 'GRIB source selection/availability/download (tools/metdata.py): find | check | fetch',
     'tools/metdata.py', 'wtune.tools.metdata'),
    ('check', 'check_args', 'pre-submission WRF run_dir sanity checker (wtune/preflight.py): time windows, iofields, physics data files, decomposition, disk',
     'wtune/preflight.py', 'wtune.preflight'),
    ('calibrate', 'calibrate_args', 'machine calibration for a new cluster (wtune/calibrate.py plan | fit)',
     'wtune/calibrate.py', 'wtune.calibrate'),
    ('post', 'post_args', 'post-processing/analysis, advisory only (decision 0056; post/cli.py): '
                          'inventory | extract | plot | sounding | tc | compare | report | recipe | custom',
     'post/cli.py', 'wtune.post.cli'),
    ('prep', 'prep_args', 'optional input-data preparation, user-invoked only (decision 0063; wtune/geog.py): '
                          'geog list | fetch | import-dem | verify | bounds | srtm-fetch',
     'wtune/geog.py', None),
    ('project', 'project_args', 'project folder holding case/input/config/runs/post '
                                '(decision 0058; wtune/project.py): init | info | set',
     'wtune/project.py', 'wtune.project'),
    ('analyze', 'analyze_args', 'TC diagnostics/tracking via wrf-python + TempestExtremes (analysis/cli.py): diag | tc-track',
     'analysis/cli.py', 'wtune.analysis.cli'),
    ('predict-rebuild', 'prb_args', 'whole-run wall-time prediction from rebuilt segments measured in parallel on the target config (wtune/rebuild_predict.py)',
     'wtune/rebuild_predict.py', 'wtune.rebuild_predict'),
    ('watch', 'watch_args', 'read-only monitor of a WRF run dir: progress, remaining time, hang/nest-stop/failure warnings (wtune/watch.py)',
     'wtune/watch.py', 'wtune.watch'),
    ('run', 'run_args', 'segmented WRF execution with restart hand-offs, per-segment config and opt-in CFL rescue (wtune/run/)',
     'wtune/run/', 'wtune.run'),
    ('convert', 'convert_args', 'WRF netCDF <-> ADIOS2 BP (io_form 14) conversion (wtune/bp.py): '
     'nc2bp | bp2nc | inputs (cached, manifest) [--check round trip]',
     'wtune/bp.py', 'wtune.bp'),
    ('advise', 'advise_args', 'advice-only: detect case conditions and suggest WRF modes/workflows with cost, evidence and side effects (decision 0061; wtune/advise.py)',
     'wtune/advise.py', 'wtune.advise'),
    ('calc', 'calc_args', 'rules of thumb and unit conversions for namelist parameters, plus `calc check <namelist.input>` (decision 0062; wtune/calc.py)',
     'wtune/calc.py', 'wtune.calc'),
    ('scenario', 'scenario_args', 'pluggable scenario generators for fictional-date WRF forcing (scenario/cli.py): list | generate | inspect',
     'scenario/cli.py', 'wtune.scenario.cli'),
)
_FORWARDED_DEST = {name: dest for name, dest, _help, _target, _module in _FORWARDED_COMMANDS}
_FORWARDED_MODULE = {name: module for name, _dest, _help, _target, module in _FORWARDED_COMMANDS}


def _build_parser():
    parser = argparse.ArgumentParser(
        prog='wtune', description='WRF config autotuner analytical core',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='common workflow:\n'
               '  wtune project init DIR --case CASE_DIR   new project (registers the case)\n'
               '  cd DIR; wtune doctor                     is everything installed?\n'
               '  wtune submit                             tune -> run -> post in one go\n'
               '  wtune tune -O2 / wtune run / wtune post recipe --srun   the same, step by step\n'
               '  wtune watch runs/<id>                    progress of a running run')
    parser.add_argument('--version', action='version',
                        version='wtune %s' % wtune.version.__version__)
    sub = parser.add_subparsers(dest='command')
    for add_parser in _IN_PROCESS_PARSERS:
        add_parser(sub)
    for name, dest, help_text, target, _module in _FORWARDED_COMMANDS:
        p = sub.add_parser(name, help=help_text)
        p.add_argument(dest, nargs=argparse.REMAINDER, help='forwarded to ' + target)
    return parser


def _forwarded_module(args):
    """Import the module a forwarded subcommand hands its argv to -- lazily, so only the running
    subcommand's module (and its dependencies) loads.

    `prep` keeps a literal import statement: decision 0063 keeps wtune.geog out of every other
    code path, and test_wtune_geog pins that it is imported only inside the 'prep' branch."""
    if args.command == 'prep':
        import wtune.geog
        return wtune.geog
    return importlib.import_module(_FORWARDED_MODULE[args.command])


def _predict_cmd(args):
    import wtune.predict
    return wtune.predict.predict_cmd(args)


# In-process subcommand -> (handler(args), always_exit). The handler's return value is the exit
# status: with always_exit the process exits with it even when it is 0/None; otherwise it exits
# only on a nonzero status and main() returns normally (the plan/physics/... handlers return
# None and report failure by calling sys.exit themselves). Handlers call the module-level
# function by name at call time, so monkeypatching wtune.cli.<function> still takes effect.
_IN_PROCESS_COMMANDS = {
    'physics': (lambda args: physics_cmd(args), False),
    'machine-profile': (lambda args: machine_profile_cmd(args), False),
    'screen': (lambda args: screen_cmd(args), False),
    'selftest': (lambda args: selftest(), True),
    'compliance': (lambda args: wc._main(args.compliance_args), True),
    'predict': (lambda args: _predict_cmd(args), True),
}


# Forwarded commands that take project.yaml's exclude_nodes and case table, and the subset
# whose full argv goes to manifest.jsonl here (post and project record their own).
_PROJECT_AWARE = ('tune', 'run', 'post', 'watch', 'check', 'convert', 'advise', 'submit',
                  'doctor')
_RECORDED = ('tune', 'run', 'submit')


def _project_for(argv):
    """(project_dir, project_doc) a forwarded argv runs in, from --project, $WTUNE_PROJECT or
    a project.yaml above cwd; (None, None) when none resolves."""
    import wtune.project
    explicit = None
    for i, a in enumerate(argv):
        if a == '--project' and i + 1 < len(argv):
            explicit = argv[i + 1]
        elif a.startswith('--project='):
            explicit = a.split('=', 1)[1]
    try:
        project_dir = wtune.project.find_project(explicit=explicit)
    except wtune.project.ProjectError:
        return None, None          # the subcommand reports a bad --project itself
    return project_dir, (wtune.project.load_project(project_dir) if project_dir else None)


def _run_forwarded(args):
    import wtune.machine
    import wtune.project
    from wtune.predictor import features
    argv = getattr(args, _FORWARDED_DEST[args.command])
    project_dir = None
    try:
        if args.command in _PROJECT_AWARE:
            project_dir, doc = _project_for(argv)
            if project_dir:
                wtune.project.apply_project_env(project_dir, doc)
        if args.command != "project":           # `project set` stays usable to repair them
            wtune.machine.export(wtune.machine.load_user())     # after the project: it wins
            wtune.machine.max_nodes()
            features.hardware(wtune.machine.hardware_id())
    except (wtune.project.ProjectError, ValueError) as e:    # MachineError is a ValueError
        print("error: %s" % e, file=sys.stderr)
        return 2
    try:
        status = _forwarded_module(args).main(argv)
    except SystemExit as e:
        status = e.code
    if project_dir and args.command in _RECORDED and not {'-h', '--help'} & set(argv):
        import wtune.project
        code = (status or 0) if isinstance(status, (int, type(None))) else 1
        wtune.project.manifest_append(project_dir, ['wtune', args.command] + list(argv),
                                      exit_code=code)
    return status


def main():
    argv = sys.argv[1:]
    # argparse's REMAINDER only starts at the first positional, so `wtune run --dry-run` would
    # fail at this level: hand a forwarded command its argv untouched.
    if argv and argv[0] in _FORWARDED_DEST:
        args = argparse.Namespace(command=argv[0], **{_FORWARDED_DEST[argv[0]]: argv[1:]})
        sys.exit(_run_forwarded(args))
    parser = _build_parser()
    args = parser.parse_args()
    if args.command in _FORWARDED_DEST:
        sys.exit(_run_forwarded(args))
    if args.command not in _IN_PROCESS_COMMANDS:
        parser.print_help()
        sys.exit(1)
    handler, always_exit = _IN_PROCESS_COMMANDS[args.command]
    status = handler(args)
    if always_exit or status:
        sys.exit(status)


if __name__ == '__main__':
    main()
