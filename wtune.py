#!/usr/bin/env python3
"""wtune.py - analytical core (Stage A + Stage B) of a WRF config autotuner."""

import argparse
import json
import math
import re
import sys
from pathlib import Path

# sibling modules (same directory as this script)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wtune_physics as phys
import wtune_knobs as knobs
import wtune_solar as wsolar
import wtune_surrogate as surr

FLOOR = 10              # minimum mass points in each direction after halo
HALO = 5                # WRF halo width approximation
KV = 100.0              # vectorization/amortization constant

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
        'history_interval': history_interval,
        'restart_interval': scalar(t, 'restart_interval', 1440),
        'io_form_history': int(scalar(t, 'io_form_history', 2)),
        'io_form_restart': int(scalar(t, 'io_form_restart', 2)),
        'io_form_input': int(scalar(t, 'io_form_input', 2)),
        'io_form_boundary': int(scalar(t, 'io_form_boundary', 2)),
        'mp_physics': int(scalar(p, 'mp_physics', 0)),
        'ra_lw_physics': int(scalar(p, 'ra_lw_physics', 0)),
        'ra_sw_physics': int(scalar(p, 'ra_sw_physics', 0)),
        'radt': scalar(p, 'radt', 15.0),
        'bl_pbl_physics': int(scalar(p, 'bl_pbl_physics', 0)),
        'sf_surface_physics': int(scalar(p, 'sf_surface_physics', 0)),
        'cu_physics': int(scalar(p, 'cu_physics', 0)),
        'sf_sfclay_physics': int(scalar(p, 'sf_sfclay_physics', 0)),
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
    print("RESULTS-AFFECTING NUMERICS (gated tuning only):")
    print("  radt: {}".format(phys.radt_advice(resolved)))
    print("  dt:   {}".format(phys.timestep_advice(resolved)))
    print()
    print("SPEED KNOBS -- reduce TOTAL WORK (bit-identical Tier 1 + gated Tier 2):")
    for line in phys.numeric_speed_advice(resolved):
        print("  " + line if line else "")
    print()
    if getattr(args, 'probe_json', None):
        import wtune_surrogate as wsurr
        probe = wsurr.load_probe(args.probe_json)
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
    total = 0
    for cat in ('correctness-setup', 'resolution(dx)', 'stability', 'run-length', 'terrain-latitude'):
        items = grouped.get(cat, [])
        if not args.all:
            items = [f for f in items if f.get('verdict') in ('MISMATCH', 'SUGGEST')]
        if not items:
            continue
        print()
        print("[{}]".format(cat))
        for f in items:
            total += 1
            enf = 'enforced' if f.get('enforced') else 'best-practice'
            print("  [{:8s}] {:22s} cur={!s:12s} rec={!s:24s} ({}/{})".format(
                f.get('verdict', '?'), f.get('param', '?'), f.get('current'),
                f.get('recommended'), f.get('tag', '?'), enf))
            if f.get('reason'):
                print("             {}".format(f['reason']))
    if total == 0:
        print("\n  no actionable findings (use --all to see OK checks).")


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
    top3 = {(d['px'], d['py']) for d in valid[:3]}
    bottom3 = {(d['px'], d['py']) for d in valid[-3:]}

    ok1 = (8, 56) in top3
    ok2 = (56, 8) in bottom3 and (112, 4) in bottom3

    print("CONUS 2.5km selftest (nx={}, ny={}, ranks={})".format(nx, ny, ranks))
    print("Valid decompositions:")
    for d in valid:
        print("  {}x{}  patch {}x{}  score {:.3f}".format(
            d['px'], d['py'], round(d['patch_x']), round(d['patch_y']), d['score']))
    print("Top 3 by score: {}".format(", ".join("{}x{}".format(px, py) for px, py in sorted(top3))))
    print("Bottom 3 by score: {}".format(", ".join("{}x{}".format(px, py) for px, py in sorted(bottom3))))

    ok = True
    if ok1:
        print("8x56 in top-3: PASS")
    else:
        print("8x56 in top-3: FAIL")
        ok = False

    if ok2:
        print("56x8,112x4 in bottom-3: PASS")
    else:
        print("56x8,112x4 in bottom-3: FAIL")
        ok = False

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
    resolved, _ = phys.resolve_suite(parse_namelist(args.namelist))
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


def screen_cmd(args):
    """3-stage autotuner funnel: analytic screen (all 2062) -> SpeedyWeather cheap
    physics vet -> the survivors go to WRF empirical test."""
    import wtune_surrogate as wsurr
    nl_kv = _parse_nl_kv(args.namelist)
    resolved, _ = phys.resolve_suite(parse_namelist(args.namelist))
    probe = wsurr.load_probe(args.probe_json) if getattr(args, 'probe_json', None) else None
    print("=== wtune SCREEN -- autotuner funnel: all params -> a few effective, without running 2062 ===")
    print("namelist:", args.namelist, "| probe:", args.probe_json or "(none; physics candidates cannot be vetted -> run speedy_probe.jl)")
    print()
    _, cnt = phys.coverage_audit(nl_kv, resolved)
    print("STAGE 0 -- full catalog classified: {} total = {} set / {} relevant-default / {} inert / {} plumbing".format(
        (cnt['set'] + cnt['relevant_default'] + cnt['inert'] + cnt['plumbing']),
        cnt['set'], cnt['relevant_default'], cnt['inert'], cnt['plumbing']))
    cands = phys.screen_candidates(nl_kv, resolved)
    print("\nSTAGE 1 -- analytic effective candidates ({}) [everything else: default optimal / correctness / inert]:".format(len(cands)))
    for c in cands:
        print("  T{:<4} {:<15} {:<16} -> {:<26} {}".format(c['tier'], c['name'], str(c['current'])[:16], c['candidate'], c['why']))
    print("\nSTAGE 2 -- SpeedyWeather cheap physics vet (rejects physics-infeasible BEFORE any WRF run):")
    survivors = []
    for c in cands:
        verdict, reason = wsurr.speedy_vet(c, probe, resolved)
        print("  {:<15} {:<11} {}".format(c['name'], verdict, reason))
        if verdict in ('PASS->WRF', 'CONFIRM'):
            survivors.append((c, verdict))
    print("\nSTAGE 3 -- survivors -> WRF empirical ABBA (the 'few effective' to actually test):")
    for c, v in sorted(survivors, key=lambda x: {'1': 0, '2': 1, 'swap': 2}.get(x[0]['tier'], 3)):
        print("  [T{}] {:<15} -> {:<26} ({})".format(c['tier'], c['name'], c['candidate'], v))
    print("\nmeasured winners so far: io_form_history=14 (-30% wall, if I/O scored), 8x56 decomposition (-1~3%).")
    print("the rest are candidates -- run them through wtune_probe (Stage C) with the pre-registered gate.")


def main():
    parser = argparse.ArgumentParser(prog='wtune.py', description='WRF config autotuner analytical core')
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
    p_phys.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h)')
    p_phys.add_argument('--probe-json', default=None, help='speedy_probe.jl JSON: SpeedyWeather physics-context self-validation')

    p_knobs = sub.add_parser('knobs', help='full knob advisor: input-derived param recommendations + mismatch flags')
    p_knobs.add_argument('--namelist', required=True, help='path to namelist.input')
    p_knobs.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h)')
    p_knobs.add_argument('--all', action='store_true', help='show OK findings too, not just actionable')

    p_cat = sub.add_parser('catalog', help='list the complete Registry-derived parameter catalog')
    p_cat.add_argument('--group', default=None, help='filter to one namelist group (e.g. physics)')
    p_cat.add_argument('--grep', default=None, help='filter by substring in param name')
    p_cat.add_argument('--include-io', action='store_true', help='include per-stream I/O plumbing params')

    p_audit = sub.add_parser('audit', help='Registry-driven scan of EVERY (2062) catalog param (coverage guarantee, any WRF case)')
    p_audit.add_argument('--namelist', required=True, help='path to namelist.input')

    p_screen = sub.add_parser('screen', help='autotuner funnel: analytic screen (all params) -> SpeedyWeather vet -> effective few')
    p_screen.add_argument('--namelist', required=True, help='path to namelist.input')
    p_screen.add_argument('--probe-json', default=None, help='speedy_probe.jl JSON (Stage-2 physics vet)')

    sub.add_parser('selftest', help='run CONUS calibration self-test')

    args = parser.parse_args()

    if args.command == 'plan':
        plan(args)
    elif args.command == 'physics':
        physics_cmd(args)
    elif args.command == 'hotspot':
        hotspot_cmd(args)
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
