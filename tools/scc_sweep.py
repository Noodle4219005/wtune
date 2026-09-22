#!/usr/bin/env python3
"""tools/scc_sweep.py -- drives a large decomposition measurement sweep on the scc cluster
(GH200, 72 cores/node, 13 compute nodes work[1-13]; login node `head` excluded) to feed a
run-time prediction model.

For each (case, node_count) this generates ANCHOR + K sampled decomposition configs via
predictor/decomp_space.py's exact WRF legality gate (legal_shapes_exact/rsl patch rules) and
predictor/experiment.py's own generated-space baseline rule (_generated_baseline/
_max_legal_ranks, imported and reused, NOT re-derived), then submits ONE Slurm job running
wtune's existing campaign/strict_campaign.sbatch runner over that config set
(CONFIG_FILE=<configs.json>, MODE=timing).

ANCHOR (one per case,N, always included so node counts stay comparable): omp=1, the maximum
legal rank count <= N*cores_per_node, the least-aspect legal shape at that rank count, numtiles=1,
dist=block. This is predictor/experiment.py's GENERATED_BASELINE_RULE (_generated_baseline) with
`dist` overridden from 'cyclic' to 'block' post-hoc -- `dist` does not affect shape/tile-geometry
resolution (enumerate_space enumerates both 'block' and 'cyclic' for the identical
ranks/omp/nx/ny/numtiles candidate), so this is exactly the 'block' twin of the same anchor
point, not a second implementation of the legality/aspect logic.

Subcommands:
  plan    <common args>            compute + print the per-(case,N) config set, time estimate
                                    and total node-hours; writes configs.json unless --dry-run;
                                    never touches the manifest or submits anything.
  submit  <common args> [--force]  same as plan, then (unless --dry-run) sbatch's the job,
                                    respecting --max-nodes-in-flight and manifest idempotency.
  status  [--root R] [--ledger-dir D] [--sacct-cmd C]
                                    join <root>/manifest.csv against sacct state and this case's
                                    campaign ledger (ok/failed row counts).

CASE_FILE/CONFIG_FILE mechanics reused verbatim: campaign/strict_campaign.sbatch's own
CONFIG_FILE loop (campaign/configs.py load_generated_configs/cmd_case_generated) accepts either
a bare JSON list or {"configs": [...]} -- the latter is what we write here, matching
predictor/decomp_space.py's own `sample`/`to_campaign_configs` output shape.

Config generation imports predictor/decomp_space.py and predictor/experiment.py DIRECTLY (both
are plain-stdlib-importable -- verified: neither needs numpy/sklearn/pixi; only campaign/configs.py
and predictor/features.py, both stdlib) instead of shelling out to the pixi-wrapped CLI the
campaign historically used for this. SITE=scc is set here (before importing campaign/configs.py)
so campaign_configs.CASES carries the campaign/cases.scc.json overlay (prep_dir/bp_dir path
overrides; case geometry -- e_we/e_sn/e_vert -- is identical across sites and not overlaid).

Manifest: <root>/manifest.csv (case,nodes,job_id,configs_path,n_configs,submitted_at,state).
`submit` looks up (case,nodes) there and skips an already-submitted pair unless --force (which
resubmits and overwrites that row -- see submit_job's docstring for the one accepted imprecision
this introduces in the in-flight node budget).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_PREDICTOR_DIR = os.path.join(_REPO_ROOT, "predictor")
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
for _p in (_PREDICTOR_DIR, _REPO_ROOT, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must precede the first `import configs` anywhere in the process (campaign/configs.py reads
# this env var at MODULE level: `SITE = os.environ.get("SITE") or "f1"`) -- setdefault so an
# operator's own SITE=scc export (or a test's monkeypatch before import) is not clobbered.
os.environ.setdefault("SITE", "scc")

import configs as campaign_configs          # campaign/configs.py: CASES (scc-overlaid), stdlib only
import decomp_space                         # predictor/decomp_space.py: enumerate_space/sample_design/
                                             # to_campaign_configs/legal_shapes_exact, stdlib only
import experiment as campaign_experiment    # predictor/experiment.py: _generated_baseline/
                                             # _max_legal_ranks/_format_hms (reused, not copied),
                                             # stdlib only

DEFAULT_HARDWARE_ID = "scc-gh200"
DEFAULT_CORES_PER_NODE = 72
DEFAULT_NODES = [1, 2, 4, 8, 13]
DEFAULT_K = 12
DEFAULT_SEED = 1
DEFAULT_REPS = 1
DEFAULT_STEPS = 30
DEFAULT_ROOT = "$HOME/wtune-runs/sweep"
DEFAULT_LEDGER_DIR = "$HOME/wtune-runs/campaign"          # campaign/sites/scc.sh's own $OUT
DEFAULT_MAX_NODES_IN_FLIGHT = 13                           # == total compute nodes, work[1-13]
DEFAULT_ENV_SOURCE = "$HOME/wtune-scc/env.sh"
DEFAULT_SBATCH_SCRIPT = "$HOME/wtune/campaign/strict_campaign.sbatch"

STARTUP_S = 120.0                  # per-run startup overhead assumed for the time-limit estimate
CONSERVATIVE_S_PER_STEP = 10.0     # fallback when no ledger row exists yet for this case on scc
TIME_SAFETY_FACTOR = 1.5
TIME_PAD_MIN = 10.0
TIME_CAP_HOURS = 12.0
TIMEOUT_MULTIPLIER = 3.0
TIMEOUT_MIN_S = 600

MANIFEST_FIELDS = ["case", "nodes", "job_id", "configs_path", "n_configs", "submitted_at", "state"]
IN_FLIGHT_STATES = {"PENDING", "RUNNING"}   # sacct State values counted against --max-nodes-in-flight

_JOBID_RE = re.compile(r"Submitted batch job (\d+)")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _expand(path):
    return os.path.expanduser(os.path.expandvars(path))


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s):
    return [int(x) for x in parse_list(s)]


# ---------------------------------------------------------------------------
# case selection
# ---------------------------------------------------------------------------
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


def resolve_case_json(case_id):
    cj = campaign_configs.CASES.get(case_id)
    if cj is None:
        raise SystemExit("unknown case %r (not in campaign/cases.json)" % case_id)
    return cj


# ---------------------------------------------------------------------------
# config generation: ANCHOR + K sampled, per (case, nodes)
# ---------------------------------------------------------------------------
def _identity(c):
    return (int(c["ranks"]), int(c["omp"]), int(c["nx"]), int(c["ny"]), int(c["numtiles"]),
            str(c["dist"]), str(c["blocktime"]))


def build_anchor(case_json, total_cores, nodes, hardware_id):
    """omp=1, max legal ranks <= total_cores, least-aspect legal shape, numtiles=1, dist=block --
    see module docstring for why this is _generated_baseline() plus a post-hoc dist override
    rather than a re-derivation. Returns None when no legal MPI decomposition exists at omp=1 for
    any rank count in 1..total_cores (predictor/experiment.py's own failure mode there)."""
    try:
        base = campaign_experiment._generated_baseline(case_json, total_cores, nodes, hardware_id)
    except SystemExit:
        return None
    return dict(
        id="ANCHOR", ranks=base["ranks"], omp=base["omp"], nx=base["nx"], ny=base["ny"],
        numtiles=base["numtiles"], dist="block", blocktime=base["blocktime"],
        patch_x=base.get("patch_x"), patch_y=base.get("patch_y"),
        numtiles_x=base.get("numtiles_x", 0), numtiles_y=base.get("numtiles_y", 0),
        tile_sz_x=base.get("tile_sz_x", 0), tile_sz_y=base.get("tile_sz_y", 0),
        tile_strategy=base.get("tile_strategy", 0), tile_shape="1d_y",
        config_source="anchor",
    )


def build_sampled(case_json, total_cores, nodes, hardware_id, k, seed):
    """K stratified-sampled generated configs (decomp_space.sample_design, must_include=None --
    the '--no-must-include' convention: STRICT C01-C16 templates are anchored at f1's 448/224/112
    core counts and never resolve at scc's 72*N core counts anyway). Returns (configs, space_size)
    -- space_size is enumerate_space's exact legal-candidate count, 0 when nothing is legal at any
    omp value whose exact-fit ranks (total_cores // omp) divides evenly."""
    candidates, axis = decomp_space.enumerate_space(
        case_json, total_cores=total_cores, nodes=nodes, hardware_id=hardware_id)
    space_size = axis["total"]
    if space_size == 0 or k <= 0:
        return [], space_size
    chosen, _coverage = decomp_space.sample_design(candidates, k, seed, must_include=None)
    return decomp_space.to_campaign_configs(chosen, prefix="G"), space_size


def build_configs(case, case_json, nodes, k, seed, hardware_id, cores_per_node):
    """ANCHOR + K sampled configs for one (case, nodes), deduped by decomposition identity
    (anchor wins ties). `skip_reason` is set iff the merged list is empty -- i.e. even the
    downward-scanning anchor found no legal decomposition at all, not just the exact-fit sampled
    space (those can differ: the sampled space only tries EXACT ranks=total_cores//omp)."""
    total_cores = nodes * cores_per_node
    anchor = build_anchor(case_json, total_cores, nodes, hardware_id)
    sampled, space_size = build_sampled(case_json, total_cores, nodes, hardware_id, k, seed)

    merged, seen = [], set()
    if anchor is not None:
        merged.append(anchor)
        seen.add(_identity(anchor))
    for c in sampled:
        ident = _identity(c)
        if ident in seen:
            continue
        seen.add(ident)
        merged.append(c)

    skip_reason = None
    if not merged:
        skip_reason = ("no legal WRF decomposition for case=%r at %d cores (N=%d x %d "
                        "cores/node): floor(extent/nproc) >= 10 fails on every domain for every "
                        "rank count 1..%d" % (case, total_cores, nodes, cores_per_node, total_cores))

    n_anchor = 1 if anchor is not None else 0
    return dict(configs=merged, total_cores=total_cores, space_size=space_size,
                n_anchor=n_anchor, n_sampled=len(merged) - n_anchor, skip_reason=skip_reason)


def write_configs_json(path, configs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(configs=configs), f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# s/step estimate (ideal core-count scaling from any prior scc ledger row for this case) and
# time-limit / TIMEOUT_S estimate
# ---------------------------------------------------------------------------
def _read_ledger_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def estimate_s_per_step(case, nodes, ledger_dir, hardware_id=DEFAULT_HARDWARE_ID,
                         fallback=CONSERVATIVE_S_PER_STEP):
    """Median of (measured s_per_step_median * n_nodes_measured / nodes) over every ledger row
    for `case` on `hardware_id`, scanning ledger_dir/ledger_<case>_*.csv -- ideal linear
    core-count scaling (72*n_measured/(72*nodes), the 72 cancels). `fallback` (source
    'fallback_10s') when no matching row exists at all. Returns (s_est, source, n_rows_used)."""
    scaled = []
    for path in sorted(glob.glob(os.path.join(ledger_dir, "ledger_%s_*.csv" % case))):
        try:
            rows = _read_ledger_rows(path)
        except OSError:
            continue
        for r in rows:
            if r.get("case_base") != case or r.get("hardware_id") != hardware_id:
                continue
            s = _to_float(r.get("s_per_step_median"))
            n = _to_float(r.get("n_nodes"))
            if s is None or s <= 0 or n is None or n <= 0:
                continue
            scaled.append(s * n / nodes)
    if scaled:
        return statistics.median(scaled), "ledger", len(scaled)
    return fallback, "fallback_10s", 0


def estimate_time(n_configs, reps, steps, s_est):
    """Per-job wall-clock estimate and per-run TIMEOUT_S watchdog.

    per_run = STARTUP_S + steps*s_est; total = n_configs * reps * per_run (REPS multiplies each
    config's run time -- the task's one-line formula omits this, but a REPS=2 job genuinely runs
    every config twice; predictor/experiment.py's own _cost_estimate uses the identical
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


# ---------------------------------------------------------------------------
# manifest IO
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# sacct / sbatch, both behind injectable runner functions (tests substitute fakes)
# ---------------------------------------------------------------------------
def default_sacct_runner(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def sacct_states(job_ids, sacct_runner=default_sacct_runner, sacct_cmd="sacct"):
    """{job_id: (state, elapsed)} via ONE batched sacct call (-X: no job-step rows, -n: no
    header, -P: pipe-delimited)."""
    job_ids = [str(j) for j in job_ids if j]
    if not job_ids:
        return {}
    cmd = [sacct_cmd, "-j", ",".join(job_ids), "-X", "-n", "-P", "--format=JobID,State,Elapsed"]
    out = sacct_runner(cmd)
    states = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jid = parts[0].strip().split(".")[0]
        state = parts[1].strip()
        states[jid] = ((state.split() or [state])[0], parts[2].strip())
    return states


def default_sbatch_runner(cmd_str):
    out = subprocess.run(["bash", "-lc", cmd_str], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError("sbatch failed (rc=%d): %s"
                            % (out.returncode, (out.stderr or out.stdout).strip()))
    return out.stdout


def build_sbatch_command(case, nodes, cores_per_node, time_limit, timeout_s, reps, steps,
                          configs_path, out_dir, env_source=DEFAULT_ENV_SOURCE,
                          sbatch_script=DEFAULT_SBATCH_SCRIPT):
    total_cores = nodes * cores_per_node
    export = ("ALL,SITE=scc,WTUNE_TOTAL_CORES=%d,CASE=%s,MODE=timing,REPS=%d,STEPS=%d,"
              "TIMEOUT_S=%d,CONFIG_FILE=%s" % (total_cores, case, reps, steps, timeout_s, configs_path))
    sbatch = ("sbatch -p compute -N %d --ntasks-per-node=%d --exclusive --exclude=head "
              "--time=%s --job-name=sw-%s-n%d --output=%s/sweep-%%j.out --export=%s %s"
              % (nodes, cores_per_node, time_limit, case, nodes, out_dir, export,
                 _expand(sbatch_script)))
    return "source %s; %s" % (_expand(env_source), sbatch)


def submit_job(cmd_str, sbatch_runner=default_sbatch_runner):
    """Runs `cmd_str` (already includes the `source env.sh;` prefix) and parses the job id out
    of sbatch's "Submitted batch job N" stdout line. --force re-running an in-flight job's
    (case,N) slot is the one accepted imprecision in the --max-nodes-in-flight budget: the OLD
    job's nodes stay counted in the run's initial in-flight baseline (taken once, from the
    manifest, before the submit loop starts) even after the manifest row is overwritten --
    conservative (under-states available budget) rather than over-subscribing the cluster."""
    out = sbatch_runner(cmd_str)
    m = _JOBID_RE.search(out)
    if not m:
        raise RuntimeError("could not parse job id from sbatch output: %r" % (out,))
    return m.group(1)


# ---------------------------------------------------------------------------
# per-(case, nodes) plan
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
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
                print("%s N=%-2d: STOP -- would exceed --max-nodes-in-flight=%d "
                      "(in_flight=%d, this job needs %d)"
                      % (case, nodes, args.max_nodes_in_flight, in_flight, nodes))
                stopped = True
                remaining.append(key)
                continue

            cmd_str = build_sbatch_command(
                case, nodes, args.cores_per_node, p["time_limit"], p["timeout_s"], args.reps,
                args.steps, os.path.abspath(p["configs_path"]), p["out_dir"],
                env_source=args.env_source, sbatch_script=args.sbatch_script)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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
