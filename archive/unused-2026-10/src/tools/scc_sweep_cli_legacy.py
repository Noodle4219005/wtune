# The standalone plan/submit/status CLI of src/tools/scc_sweep.py (superseded by `wtune tune`),
# cut verbatim at 33a1ef7. Not importable as is (helpers and constants stayed in scc_sweep.py).


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s):
    return [int(x) for x in parse_list(s)]


def default_cases():
    """Every campaign/cases.json id (merged with cases.scc.json via SITE=scc) that is enabled
    and whose (possibly $HOME/~-relative) prep_dir exists on this machine."""
    out = []
    for cid in sorted(campaign_configs.CASES):
        cj = campaign_configs.CASES[cid]
        if not cj.get("enabled", True):
            continue
        prep_dir = _expand(cj.get("prep_dir") or "")
        if prep_dir and os.path.isdir(prep_dir):
            out.append(cid)
    return out


def estimate_time(n_configs, reps, steps, s_est):
    """Per-job wall-clock estimate and per-run TIMEOUT_S watchdog.

    per_run = STARTUP_S + steps*s_est; total = n_configs * reps * per_run (REPS multiplies each
    config's run time -- the task's one-line formula omits this, but a REPS=2 job genuinely runs
    every config twice; the archived experiment engine's _cost_estimate uses the identical
    `reps * (...)` shape). (total/3600)*1.5 + 10 min, capped at 12 h, is the --time request;
    TIMEOUT_S = max(600, 3*per_run) is the per-run watchdog, independent of REPS/cap."""
    per_run_s = STARTUP_S + steps * s_est
    total_s = n_configs * reps * per_run_s
    job_hours = total_s / 3600.0
    padded_hours = job_hours * TIME_SAFETY_FACTOR + TIME_PAD_MIN / 60.0
    capped_hours = min(padded_hours, TIME_CAP_HOURS)
    timeout_s = max(TIMEOUT_MIN_S, int(math.ceil(TIMEOUT_MULTIPLIER * per_run_s)))
    return dict(job_hours=capped_hours, uncapped_hours=padded_hours, total_run_s=total_s,
                time_limit=campaign_experiment._format_hms(capped_hours * 60.0), timeout_s=timeout_s)


def read_manifest(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


def manifest_lookup(rows):
    return {(r["case"], int(r["nodes"])): r for r in rows}


def build_plan_for_pair(case, nodes, args):
    case_json = resolve_case_json(case)
    built = build_configs(case, case_json, nodes, args.k, args.seed, args.hardware_id,
                           args.cores_per_node)
    out_dir = os.path.join(args.root, case, "n%d" % nodes)
    configs_path = os.path.join(out_dir, "configs.json")
    plan = dict(case=case, nodes=nodes, out_dir=out_dir, configs_path=configs_path, **built)
    if built["skip_reason"]:
        return plan
    s_est, s_source, n_rows = estimate_s_per_step(case, nodes, args.ledger_dir, args.hardware_id)
    time_est = estimate_time(len(built["configs"]), args.reps, args.steps, s_est)
    plan.update(s_est=s_est, s_source=s_source, s_rows=n_rows, **time_est)
    plan["node_hours"] = nodes * time_est["job_hours"]
    return plan


def summary_line(p):
    if p["skip_reason"]:
        return "%s N=%-2d SKIP: %s" % (p["case"], p["nodes"], p["skip_reason"])
    return ("%s N=%-2d cores=%-4d configs=%-3d (%d anchor + %d sampled, space=%d) "
            "s_est=%.3fs/step[%s,n=%d] time=%s timeout=%ds node-hours=%.2f"
            % (p["case"], p["nodes"], p["total_cores"], len(p["configs"]), p["n_anchor"],
               p["n_sampled"], p["space_size"], p["s_est"], p["s_source"], p["s_rows"],
               p["time_limit"], p["timeout_s"], p["node_hours"]))


def cmd_plan(args):
    args.root = _expand(args.root)
    args.ledger_dir = _expand(args.ledger_dir)
    cases = args.cases if args.cases is not None else default_cases()
    total_node_hours = 0.0
    for case in cases:
        for nodes in args.nodes:
            p = build_plan_for_pair(case, nodes, args)
            print(summary_line(p))
            if p["skip_reason"]:
                continue
            if not args.dry_run:
                write_configs_json(p["configs_path"], p["configs"])
            total_node_hours += p["node_hours"]
    print("TOTAL estimated node-hours: %.2f" % total_node_hours)
    return 0


def cmd_submit(args):
    args.root = _expand(args.root)
    args.ledger_dir = _expand(args.ledger_dir)
    cases = args.cases if args.cases is not None else default_cases()

    manifest_path = os.path.join(args.root, "manifest.csv")
    by_key = manifest_lookup(read_manifest(manifest_path))

    existing_ids = [r["job_id"] for r in by_key.values()
                    if r.get("state") == "submitted" and r.get("job_id")]
    states = sacct_states(existing_ids, sacct_runner=args.sacct_runner, sacct_cmd=args.sacct_cmd)
    in_flight = sum(int(r["nodes"]) for r in by_key.values()
                     if r.get("state") == "submitted"
                     and states.get(r.get("job_id"), ("UNKNOWN", ""))[0] in IN_FLIGHT_STATES)

    stopped = False
    remaining = []
    total_node_hours = 0.0
    for case in cases:
        for nodes in args.nodes:
            key = (case, nodes)
            if key in by_key and not args.force:
                print("%s N=%-2d SKIP: already submitted (job_id=%s, state=%s)"
                      % (case, nodes, by_key[key].get("job_id"), by_key[key].get("state")))
                continue
            if stopped:
                remaining.append(key)
                continue

            p = build_plan_for_pair(case, nodes, args)
            print(summary_line(p))
            if p["skip_reason"]:
                continue

            if in_flight + nodes > args.max_nodes_in_flight:
                # Skip, don't stop: stopping here made a 13-node pair block every smaller pair
                # behind it while 1 node was busy (scc feeder stalled for hours after severe2023 N=13).
                print("%s N=%-2d: DEFER -- would exceed --max-nodes-in-flight=%d "
                      "(in_flight=%d, this job needs %d); trying smaller pairs"
                      % (case, nodes, args.max_nodes_in_flight, in_flight, nodes))
                remaining.append(key)
                continue

            cmd_str = build_sbatch_command(
                case, nodes, args.cores_per_node, p["time_limit"], p["timeout_s"], args.reps,
                args.steps, os.path.abspath(p["configs_path"]), p["out_dir"],
                env_source=args.env_source, sbatch_script=args.sbatch_script,
                campaign_dir=campaign_dir_for(args.sbatch_script))

            if args.dry_run:
                print("DRY-RUN: %s" % cmd_str)
                total_node_hours += p["node_hours"]
                continue

            write_configs_json(p["configs_path"], p["configs"])
            os.makedirs(p["out_dir"], exist_ok=True)
            job_id = submit_job(cmd_str, sbatch_runner=args.sbatch_runner)
            print("%s N=%-2d: submitted job %s" % (case, nodes, job_id))
            by_key[key] = dict(case=case, nodes=str(nodes), job_id=job_id,
                                configs_path=os.path.abspath(p["configs_path"]),
                                n_configs=str(len(p["configs"])), submitted_at=_utcnow(),
                                state="submitted")
            write_manifest(manifest_path, list(by_key.values()))
            in_flight += nodes
            total_node_hours += p["node_hours"]

    print("TOTAL estimated node-hours (this invocation): %.2f" % total_node_hours)
    if remaining:
        print("NOT SUBMITTED (%d pairs) -- re-run submit later:" % len(remaining))
        for case, nodes in remaining:
            print("  %s N=%d" % (case, nodes))
    return 0


def cmd_status(args):
    args.root = _expand(args.root)
    args.ledger_dir = _expand(args.ledger_dir)
    manifest_path = os.path.join(args.root, "manifest.csv")
    rows = read_manifest(manifest_path)
    if not rows:
        print("no manifest at %s -- nothing submitted yet" % manifest_path)
        return 0

    job_ids = [r.get("job_id") for r in rows if r.get("job_id")]
    states = sacct_states(job_ids, sacct_runner=args.sacct_runner, sacct_cmd=args.sacct_cmd)
    for r in rows:
        case, nodes, job_id = r["case"], r["nodes"], r.get("job_id", "")
        state, elapsed = states.get(job_id, ("UNKNOWN", ""))
        ledger_path = os.path.join(args.ledger_dir, "ledger_%s_%s.csv" % (case, job_id))
        ok = fail = 0
        if os.path.exists(ledger_path):
            for lr in _read_ledger_rows(ledger_path):
                if _to_float(lr.get("s_per_step_median")) is not None:
                    ok += 1
                else:
                    fail += 1
        print("%s N=%-2s job=%-8s state=%-10s elapsed=%-10s configs=%-3s ok=%d fail=%d"
              % (case, nodes, job_id, state, elapsed, r.get("n_configs", ""), ok, fail))
    return 0


def _add_common_args(sp):
    sp.add_argument("--cases", type=parse_list, default=None,
                     help="comma list; default: every enabled cases.json/cases.scc.json case "
                          "whose prep_dir exists on this machine")
    sp.add_argument("--nodes", type=parse_int_list, default=list(DEFAULT_NODES))
    sp.add_argument("--k", type=int, default=DEFAULT_K)
    sp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sp.add_argument("--reps", type=int, default=DEFAULT_REPS)
    sp.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    sp.add_argument("--root", default=DEFAULT_ROOT)
    sp.add_argument("--dry-run", action="store_true", dest="dry_run")
    sp.add_argument("--ledger-dir", default=DEFAULT_LEDGER_DIR)
    sp.add_argument("--hardware-id", default=DEFAULT_HARDWARE_ID)
    sp.add_argument("--cores-per-node", type=int, default=DEFAULT_CORES_PER_NODE)


def build_arg_parser():
    p = argparse.ArgumentParser(prog="scc_sweep.py")
    sub = p.add_subparsers(dest="command", required=True)

    sp_plan = sub.add_parser("plan")
    _add_common_args(sp_plan)

    sp_submit = sub.add_parser("submit")
    _add_common_args(sp_submit)
    sp_submit.add_argument("--max-nodes-in-flight", type=int, default=DEFAULT_MAX_NODES_IN_FLIGHT,
                            dest="max_nodes_in_flight")
    sp_submit.add_argument("--force", action="store_true")
    sp_submit.add_argument("--env-source", default=DEFAULT_ENV_SOURCE, dest="env_source")
    sp_submit.add_argument("--sbatch-script", default=DEFAULT_SBATCH_SCRIPT, dest="sbatch_script")
    sp_submit.add_argument("--sacct-cmd", default="sacct", dest="sacct_cmd")

    sp_status = sub.add_parser("status")
    sp_status.add_argument("--root", default=DEFAULT_ROOT)
    sp_status.add_argument("--ledger-dir", default=DEFAULT_LEDGER_DIR)
    sp_status.add_argument("--sacct-cmd", default="sacct", dest="sacct_cmd")

    p.set_defaults(sbatch_runner=default_sbatch_runner, sacct_runner=default_sacct_runner)
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "submit":
        return cmd_submit(args)
    if args.command == "status":
        return cmd_status(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
