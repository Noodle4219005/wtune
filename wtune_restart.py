"""Pure, deterministic restart-window advisor primitives."""
import datetime as _dt


def _utc(value):
    if isinstance(value, _dt.datetime):
        return value.replace(tzinfo=value.tzinfo or _dt.timezone.utc)
    text = value.replace("Z", "")
    return _dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=_dt.timezone.utc)


def build_windows(run_total_hours, restart_interval, start_utc="2000-01-01T00:00:00Z",
                  window_hours=None, run_hours_override=None):
    total = float(run_hours_override if run_hours_override is not None else run_total_hours)
    size = float(window_hours if window_hours is not None else restart_interval) / (60.0 if window_hours is None else 1.0)
    if total <= 0 or size <= 0:
        raise ValueError("run duration and window size must be positive")
    start = _utc(start_utc)
    out = []
    hour = 0.0
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
    if not sizes or any(not isinstance(x, int) or x < 1 for x in sizes):
        raise ValueError("allowed_team_sizes must contain positive integers")
    for node in topology["nodes"]:
        if not node.get("name") or not node.get("numa_domains"):
            raise ValueError("each node requires name and numa_domains")
        for domain in node["numa_domains"]:
            cpus = domain.get("cpus")
            if not isinstance(cpus, list) or not cpus or len(set(cpus)) != len(cpus):
                raise ValueError("each NUMA domain requires unique CPU list")
    return topology


def enumerate_candidates(rank_count, patch_x, patch_y, topology):
    validate_topology(topology)
    capacity = sum(len(d["cpus"]) for n in topology["nodes"] for d in n["numa_domains"])
    sizes = sorted(set(topology["allowed_team_sizes"]))
    if rank_count < 1 or rank_count * min(sizes) > capacity:
        return []
    result = []
    for px in range(1, rank_count + 1):
        if rank_count % px:
            continue
        py = rank_count // px
        if patch_x // px < 10 or patch_y // py < 10:
            continue
        if any(rank_count * team > capacity for team in sizes[:1]):
            continue
        result.append({"rank_count": rank_count, "px": px, "py": py,
                       "min_patch_x": patch_x // px, "min_patch_y": patch_y // py})
    return result


def allocate_teams(predicted_loads, legal_team_sizes, topology):
    validate_topology(topology)
    domains = [(n["name"], d["id"], len(d["cpus"]))
               for n in topology["nodes"] for d in n["numa_domains"]]
    sizes = sorted(set(legal_team_sizes))
    if not sizes:
        raise ValueError("no legal team sizes")
    allocations = []
    remaining = list(domains)
    for rank, load in sorted(enumerate(predicted_loads), key=lambda x: (-x[1], x[0])):
        chosen = next((i for i, (_, _, cap) in enumerate(remaining) if max(sizes) <= cap), None)
        if chosen is None:
            chosen = next((i for i, (_, _, cap) in enumerate(remaining) if min(sizes) <= cap), None)
        if chosen is None:
            raise ValueError("NUMA capacity exceeded")
        name, domain, cap = remaining[chosen]
        team = max(s for s in sizes if s <= cap) if not allocations else min(s for s in sizes if s <= cap)
        allocations.append({"rank": rank, "team_size": team, "node": name,
                            "numa_domain": domain, "predicted_load": load,
                            "predicted_speedup": 0.0})
        remaining[chosen] = (name, domain, cap - team)
    return sorted(allocations, key=lambda x: x["rank"])


def normalize_tile_config(config):
    cfg = dict(config)
    if cfg.get("strategy") == 3 or "numtiles_x" in cfg or "numtiles_y" in cfg:
        raise ValueError("strategy 3 and explicit numtiles_x/y are unsupported")
    if cfg.get("strategy", 0) == 0 and "numtiles" in cfg and "rank_count" in cfg:
        cfg["numtiles"] = min(int(cfg["numtiles"]), int(cfg["rank_count"]))
    return cfg


def render_manifest(spec):
    out = dict(spec)
    config = dict(out.get("config", {}))
    if "rank_count" not in config and out.get("candidates"):
        config["rank_count"] = max(c["rank_count"] for c in out["candidates"])
    out["config"] = normalize_tile_config(config)
    out["verdict"] = "PENDING"
    out["scientific_verdict"] = "UNVERIFIED"
    out.setdefault("proxy_cost", None)
    out.setdefault("transition_cost", "UNKNOWN")
    out.setdefault("restart_cost", None)
    return out


def render_map(spec):
    return "# rank map\nverdict=PENDING\nscientific_verdict=UNVERIFIED\n" + repr(spec)


def render_wrapper(spec):
    return "#!/bin/sh\n# verdict=PENDING scientific_verdict=UNVERIFIED\n" + repr(spec) + "\n"
