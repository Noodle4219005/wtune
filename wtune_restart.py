"""Pure, deterministic restart-window advisor primitives (Python 3.6)."""
import datetime as _dt
import base64
import copy
import json


def _utc(value):
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.timezone.utc)
        return value.astimezone(_dt.timezone.utc)
    text = value[:-1] if value.endswith("Z") else value
    if "+" in text[10:] or text.count("-") > 2:
        parsed = _dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z")
    else:
        parsed = _dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
    return _utc(parsed)


def build_windows(run_total_hours, restart_interval, start_utc="2000-01-01T00:00:00Z",
                  window_hours=None, run_hours_override=None):
    total = float(run_hours_override if run_hours_override is not None else run_total_hours)
    size = float(window_hours if window_hours is not None else restart_interval) / (60.0 if window_hours is None else 1.0)
    if total <= 0 or size <= 0:
        raise ValueError("run duration and window size must be positive")
    start = _utc(start_utc)
    out, hour = [], 0.0
    while hour < total:
        end = min(total, hour + size)
        stamp = start + _dt.timedelta(hours=hour)
        out.append({"id": len(out), "start_hour": hour, "end_hour": end,
                    "start_utc": stamp.isoformat().replace("+00:00", "Z"),
                    "day_of_year": stamp.timetuple().tm_yday})
        hour = end
    return out


def validate_topology(topology):
    if not isinstance(topology, dict) or not topology.get("nodes"):
        raise ValueError("topology requires non-empty nodes")
    sizes = topology.get("allowed_team_sizes")
    if not sizes or any(type(x) is not int or x < 1 for x in sizes):
        raise ValueError("allowed_team_sizes must contain positive integers")
    result = copy.deepcopy(topology)
    for node in result["nodes"]:
        if not node.get("name") or not node.get("numa_domains"):
            raise ValueError("each node requires name and numa_domains")
        ids, cpus_seen = set(), set()
        for domain in node["numa_domains"]:
            ident, cpus = domain.get("id"), domain.get("cpus")
            if type(ident) is not int or ident in ids:
                raise ValueError("NUMA/domain ids must be unique integers")
            if not isinstance(cpus, list) or not cpus or any(type(c) is not int for c in cpus):
                raise ValueError("each NUMA domain requires integer CPUs")
            if len(set(cpus)) != len(cpus) or cpus_seen.intersection(cpus):
                raise ValueError("CPU lists must be disjoint within each node")
            ids.add(ident); cpus_seen.update(cpus)
    return result


def _domain_capacities(topology):
    return [len(d["cpus"]) for n in topology["nodes"] for d in n["numa_domains"]]


def score_candidate(candidate, predicted_loads=None):
    structural = abs(candidate["px"] - candidate["py"]) + (1.0 / candidate["min_patch_y"])
    criticality = 0.0
    if predicted_loads:
        criticality = max(predicted_loads) / float(sum(predicted_loads) or 1)
    candidate = dict(candidate)
    candidate.update(structural_score=structural, criticality_score=criticality,
                     score=structural + criticality)
    return candidate


def enumerate_candidates(rank_count, patch_x, patch_y, topology, local_patch_y=None,
                         predicted_loads=None):
    validate_topology(topology)
    if rank_count < 1:
        return []
    if local_patch_y is None:
        raise ValueError("local_patch_y is required")
    sizes = sorted(set(topology["allowed_team_sizes"]))
    result = []
    for px in range(1, rank_count + 1):
        if rank_count % px:
            continue
        py = rank_count // px
        lx, ly = patch_x // px, local_patch_y // py
        if lx < 10 or ly < 10:
            continue
        if not any(all(cap >= rank_count * size for cap in _domain_capacities(topology))
                   for size in sizes):
            # A candidate may use several domains, but each rank's team must fit one domain.
            if rank_count > sum(max(cap // min(sizes), 0) for cap in _domain_capacities(topology)):
                continue
        team = min((s for s in sizes if s <= max(_domain_capacities(topology))), default=0)
        if not team:
            continue
        result.append(score_candidate({"rank_count": rank_count, "px": px, "py": py,
                                       "team_size": team, "min_patch_x": lx,
                                       "min_patch_y": ly}, predicted_loads))
    return sorted(result, key=lambda c: (c["score"], c["px"], c["py"]))


def allocate_teams(predicted_loads, legal_team_sizes, topology):
    topology = validate_topology(topology)
    sizes = sorted(set(legal_team_sizes))
    if not sizes or any(type(s) is not int or s < 1 for s in sizes):
        raise ValueError("no legal team sizes")
    allowed = set(topology["allowed_team_sizes"])
    if not set(sizes).issubset(allowed):
        raise ValueError("legal team sizes must be allowed by topology")
    domains = [(n["name"], d["id"], list(d["cpus"]))
               for n in topology["nodes"] for d in n["numa_domains"]]
    if len(predicted_loads) > sum(len(c) // min(sizes) for _, _, c in domains):
        raise ValueError("NUMA capacity exceeded")
    allocations = []
    for rank, load in sorted(enumerate(predicted_loads), key=lambda x: (-x[1], x[0])):
        chosen = next((i for i, (_, _, cpus) in enumerate(domains)
                       if len(cpus) >= min(sizes)), None)
        if chosen is None:
            raise ValueError("NUMA capacity exceeded")
        name, domain, cpus = domains[chosen]
        team = max(s for s in sizes if s <= len(cpus))
        selected, domains[chosen] = cpus[:team], (name, domain, cpus[team:])
        allocations.append({"rank": rank, "team_size": team, "node": name,
                            "resource_key": "%s:%s" % (name, domain),
                            "numa_domain": domain, "cpus": selected,
                            "predicted_load": load, "predicted_speedup": 0.0})
    return sorted(allocations, key=lambda x: x["rank"])


def normalize_tile_config(config, local_patch_y=None):
    cfg = dict(config)
    if cfg.get("strategy") == 3 or "numtiles_x" in cfg or "numtiles_y" in cfg:
        raise ValueError("strategy 3 and explicit numtiles_x/y are unsupported")
    if cfg.get("strategy", 0) == 0 and "numtiles" in cfg:
        if local_patch_y is None:
            raise ValueError("local_patch_y is required for scalar numtiles")
        cfg["numtiles"] = min(int(cfg["numtiles"]), int(local_patch_y))
    return cfg


def _canonical(spec):
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def render_manifest(spec):
    out = dict(spec)
    config = dict(out.get("config", {}))
    local_y = out.get("local_patch_y", config.get("local_patch_y"))
    config = normalize_tile_config(config, local_y)
    out["config"] = config
    out.update(strict_bitwise="UNVERIFIED", scientific_tolerance="UNVERIFIED", actionable=False)
    for key in ("proxy_cost", "transition_cost", "restart_cost"):
        out.setdefault(key, None)
    return out


def render_map(spec):
    return "# rank map\n" + _canonical(render_manifest(spec)) + "\n"


def render_wrapper(spec):
    manifest = render_manifest(spec)
    payload = base64.b64encode(_canonical(manifest).encode("ascii")).decode("ascii")
    helper = ("import base64, json, os, sys\n"
              "def fail(message):\n    raise SystemExit('wtune wrapper: ' + message)\n"
              "rank_text = next((os.environ[name] for name in "
              "('PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'PMIX_RANK') "
              "if name in os.environ), None)\n"
              "if rank_text is None: fail('rank is not set')\n"
              "try:\n    rank = int(rank_text)\nexcept (TypeError, ValueError):\n    fail('rank is not an integer')\n"
              "try:\n    data = json.loads(base64.b64decode(os.environ['WTUNE_PAYLOAD']).decode('ascii'))\n    allocations = data['allocations']\nexcept (KeyError, TypeError, ValueError, UnicodeDecodeError):\n    fail('invalid embedded payload')\n"
              "if not isinstance(allocations, list): fail('allocations must be a list')\n"
              "matches = [a for a in allocations if isinstance(a, dict) and type(a.get('rank')) is int and a['rank'] == rank]\n"
              "if len(matches) != 1: fail('rank must have exactly one allocation')\n"
              "allocation = matches[0]\n"
              "team_size, cpus = allocation.get('team_size'), allocation.get('cpus')\n"
              "if type(team_size) is not int or team_size < 1 or not isinstance(cpus, list) or not cpus or len(cpus) != team_size or any(type(cpu) is not int or cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus): fail('allocation types are invalid')\n"
              "os.environ['OMP_NUM_THREADS'] = str(team_size)\n"
              "os.environ['OMP_PLACES'] = '{%s}' % ','.join(str(cpu) for cpu in cpus)\n"
              "os.environ['OMP_PROC_BIND'] = 'close'\n"
              "if hasattr(os, 'sched_setaffinity'):\n    try: os.sched_setaffinity(0, cpus)\n    except (OSError, ValueError): pass\n"
              "if len(sys.argv) < 2: fail('command is missing')\n"
              "os.execvpe(sys.argv[1], sys.argv[1:], os.environ)\n")
    helper_payload = base64.b64encode(helper.encode("ascii")).decode("ascii")
    return ("#!/bin/sh\nset -eu\nexport WTUNE_PAYLOAD='%s'\n"
            "exec python3 -c 'import base64; exec(compile(base64.b64decode(\"%s\"), \"<wtune-wrapper>\", \"exec\"))' \"$@\"\n" %
            (payload, helper_payload))
