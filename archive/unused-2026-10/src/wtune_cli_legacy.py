# The nine retired `wtune` commands (plan, hotspot, oracle-map, restart-window, feedback, knobs,
# catalog, audit, profile) as they were in src/wtune_cli.py at 18a9b9f, removed verbatim.
# Not wired to anything; shared helpers (parse_namelist, auto_probe, ...) remain in src/wtune_cli.py.
import tempfile
import wtune_knobs as knobs
import wtune_surrogate as surr
from wtune_feedback import (FeedbackController, FeedbackError, FeedbackSample,
                            decisions_to_csv, decisions_to_json)

DEFAULT_CORES_PER_NODE = 112
DEFAULT_NUMA_DOMAINS = 8
DEFAULT_CORES_PER_NUMA = 14
DEFAULT_NODES = 4

CACHE_TMP = os.environ.get('TMPDIR') or '/tmp'


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

    if not args.json:
        _print_extension_notes(nl, args.namelist)
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


def _restart_error(message):
    print("ERROR restart-window: {}".format(message), file=sys.stderr)
    return 1


def _restart_rank_counts(value, case):
    if value:
        try:
            counts = [int(part.strip()) for part in value.split(',')]
        except ValueError:
            raise ValueError("rank counts must be comma-separated positive integers") from None
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
    print("WARNING restart-window: deprecated path (see archive/deprecated/README.md); "
          "PLOV is the live rebalancer", file=sys.stderr)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'archive' / 'deprecated'))
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


def _add_plan_parser(sub):
    p = sub.add_parser('plan', help='generate ranked decomposition shortlist')
    p.add_argument('--namelist', required=True, help='path to namelist.input')
    p.add_argument('--ranks', type=int, default=None, help='total MPI ranks')
    p.add_argument('--nodes', type=int, default=None, help='number of nodes')
    p.add_argument('--cores-per-node', type=int, default=None, help='default 112')
    p.add_argument('--numa-domains', type=int, default=None, help='default 8')
    p.add_argument('--cores-per-numa', type=int, default=None, help='default derived or 14')
    p.add_argument('--smt', choices=['on', 'off'], default='off', help='default off')
    p.add_argument('--json', action='store_true', help='emit JSON')
    p.add_argument('--allow-hybrid', action='store_true', help='report hybrid alternative')
    p.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h)')
    p.add_argument('--surrogate-json', default=None, help='precomputed SpeedyWeather hotspot JSON to include')


def _add_hotspot_parser(sub):
    p = sub.add_parser('hotspot', help='physics load map: analytic terminator + convective surrogate')
    p.add_argument('--namelist', required=True, help='path to namelist.input')
    p.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date')
    p.add_argument('--surrogate-json', default=None, help='use a precomputed SpeedyWeather hotspot JSON')
    p.add_argument('--run-surrogate', action='store_true', help='run SpeedyWeather now (slow: JIT + model)')
    p.add_argument('--trunc', type=int, default=31, help='SpeedyWeather spectral truncation (default 31)')
    p.add_argument('--nlayers', type=int, default=8, help='SpeedyWeather vertical layers (default 8)')
    p.add_argument('--srun', default=None, help='srun prefix to run the surrogate on a compute node')


def _add_knobs_parser(sub):
    p = sub.add_parser('knobs', help='full knob advisor: input-derived param recommendations + mismatch flags')
    p.add_argument('--namelist', required=True, help='path to namelist.input')
    p.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h)')
    p.add_argument('--all', action='store_true', help='show OK findings too, not just actionable')
    _add_compliance_arg(p, 'findings')


def _add_catalog_parser(sub):
    p = sub.add_parser('catalog', help='list the complete Registry-derived parameter catalog')
    p.add_argument('--group', default=None, help='filter to one namelist group (e.g. physics)')
    p.add_argument('--grep', default=None, help='filter by substring in param name')
    p.add_argument('--include-io', action='store_true', help='include per-stream I/O plumbing params')


def _add_audit_parser(sub):
    p = sub.add_parser('audit', help='Registry-driven scan of EVERY (2062) catalog param (coverage guarantee, any WRF case)')
    p.add_argument('--namelist', required=True, help='path to namelist.input')


def _add_profile_parser(sub):
    p = sub.add_parser('profile', help='case-understanding profile (PRIOR, pre-measurement): identity/physics/cost-structure/imbalance/core-count priors')
    p.add_argument('case_dir', help='directory with namelist.input (+ optional wrfinput_d0N)')
    p.add_argument('--namelist', default=None, help='explicit namelist.input path (default: <case_dir>/namelist.input)')
    p.add_argument('--wrfinput', default=None, help='explicit wrfinput/wrfout path (default: first wrfinput_d0N in case_dir)')
    p.add_argument('--cores-per-node', type=int, default=None, help='default 72 (GH200 Grace)')
    p.add_argument('--json', default=None, help='also write the full profile as JSON to this path')


def _add_oracle_map_parser(sub):
    p = sub.add_parser('oracle-map', help='DIAGNOSTIC spatial physics-load map (per-cell phase cost + hotspots); NOT a partition selector')
    p.add_argument('--namelist', required=True, help='path to namelist.input (for date + cu_physics + run length)')
    p.add_argument('--snapshot', default=None, help='wrfout/wrfinput NetCDF with model-state fields (defaults to --wrfinput)')
    p.add_argument('--wrfinput', default=None, help='wrfinput/wrfout for lat/lon/date (via ncdump -h); also the snapshot fallback')
    p.add_argument('--stride', type=int, default=4, help='spatial subsample stride for the model-state read (default 4)')
    p.add_argument('--top-k', dest='top_k', type=int, default=12, help='number of hottest cells to list (default 12)')
    p.add_argument('--grid', type=int, default=24, help='max side of the per-phase JSON grid (block-mean downsample, default 24)')
    p.add_argument('--json-out', dest='json_out', default=None, help='write the artifact JSON here (compact grids + top-k)')
    p.add_argument('--npz-out', dest='npz_out', default=None, help='write the full-resolution per-phase fields here as .npz')


def _add_restart_window_parser(sub):
    p = sub.add_parser('restart-window', help='UNVERIFIED restart-boundary decomposition/team hypotheses')
    p.add_argument('--namelist', required=True, help='path to namelist.input (never modified)')
    p.add_argument('--snapshot', required=True, help='readable wrfout/wrfrst snapshot for oracle fields')
    p.add_argument('--topology-json', required=True, help='topology JSON with node-local CPU lists')
    p.add_argument('--rank-counts', '--ranks', dest='rank_counts', default=None,
                   help='comma-separated MPI rank counts (default nproc_x*nproc_y)')
    p.add_argument('--window-hours', type=float, default=None, help='restart advisor window size in hours')
    p.add_argument('--run-hours', type=float, default=None, help='override namelist run duration in hours')
    p.add_argument('--stride', type=int, default=4, help='snapshot spatial subsample stride (default 4)')
    p.add_argument('--top-k', dest='top_k', type=int, default=1, help='candidates per rank count/window')
    p.add_argument('--output-dir', default='restart-window-artifacts', help='directory for rank maps and wrappers')
    p.add_argument('--manifest-out', default=None, help='overall canonical JSON manifest path')


def _add_feedback_parser(sub):
    p = sub.add_parser('feedback', help='manifest-only advisory feedback decisions')
    p.add_argument('--input', required=True, help='JSON samples and controller limits')
    p.add_argument('--json-out', required=True, help='decision JSON output')
    p.add_argument('--csv-out', required=True, help='canonical telemetry/decision CSV output')


# _IN_PROCESS_PARSERS as it was:
_IN_PROCESS_PARSERS = (
    _add_plan_parser, _add_hotspot_parser, _add_physics_parser, _add_knobs_parser,
    _add_catalog_parser, _add_audit_parser, _add_screen_parser, _add_profile_parser,
    _add_machine_profile_parser, _add_oracle_map_parser, _add_restart_window_parser,
    _add_feedback_parser, _add_selftest_parser, _add_compliance_parser, _add_predict_parser,
)

# Entries removed from _IN_PROCESS_COMMANDS:
_IN_PROCESS_COMMANDS = {
    'plan': (lambda args: plan(args), False),
    'profile': (lambda args: profile_cmd(args), False),
    'hotspot': (lambda args: hotspot_cmd(args), False),
    'oracle-map': (lambda args: oracle_map_cmd(args), False),
    'restart-window': (lambda args: restart_window_cmd(args), False),
    'feedback': (lambda args: feedback_cmd(args), False),
    'knobs': (lambda args: knobs_cmd(args), False),
    'catalog': (lambda args: catalog_cmd(args), False),
    'audit': (lambda args: audit_cmd(args), False),
}
