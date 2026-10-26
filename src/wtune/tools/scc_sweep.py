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

The standalone plan/submit/status command line this file used to carry (and its manifest.csv)
is in archive/unused-2026-10/src/tools/scc_sweep_cli_legacy.py; what is left is the library
`wtune tune` / `wtune run` import.

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
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # src/
_TRUE_ROOT = os.path.dirname(_REPO_ROOT)  # repo root
_CAMPAIGN_DIR = os.path.join(_TRUE_ROOT, "campaign")
if not __package__:                    # run by path: this directory is not an import root
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_REPO_ROOT, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must precede the first `import configs` anywhere in the process (campaign/configs.py reads
# this env var at MODULE level: `SITE = os.environ.get("SITE") or "f1"`) -- setdefault so an
# operator's own SITE=scc export (or a test's monkeypatch before import) is not clobbered.
import wtune.machine

os.environ.setdefault("SITE", wtune.machine.site())

import configs as campaign_configs          # campaign/configs.py: CASES (scc-overlaid), stdlib only
from wtune.predictor import decomp_space                         # predictor/decomp_space.py: enumerate_space/sample_design/
                                             # to_campaign_configs/legal_shapes_exact, stdlib only
from wtune.predictor import experiment as campaign_experiment    # predictor/experiment.py: _generated_baseline/
                                             # _max_legal_ranks/_format_hms (reused, not copied),
                                             # stdlib only

DEFAULT_K = 12
DEFAULT_SEED = 1
DEFAULT_REPS = 1
DEFAULT_STEPS = 30
DEFAULT_ROOT = "$HOME/wtune-runs/sweep"
DEFAULT_LEDGER_DIR = "$HOME/wtune-runs/campaign"          # campaign/sites/scc.sh's own $OUT
CAMPAIGN_OUT_ENV = "WTUNE_CAMPAIGN_OUT"                   # the site files' $OUT when set
HISTORY_LEDGER_DIR = DEFAULT_LEDGER_DIR                   # where tunes before <project>/probes wrote
# The campaign/ of THIS source tree (resolved from __file__), not a fixed "$HOME/wtune/campaign":
# on scc ~/wtune is a stale checkout (no RANKING_START, no write_adios2_restart_xml, no \037 TSV
# fix) while the live code runs from ~/wtune-src, so the old hardcoded default silently paired
# new Python with an old campaign script (2026-09-27). --sbatch-script still overrides it.
DEFAULT_SBATCH_SCRIPT = os.path.join(_CAMPAIGN_DIR, "strict_campaign.sbatch")

STARTUP_S = wtune.machine.MEASURED["startup_s"]            # built-in values; the code reads
CONSERVATIVE_S_PER_STEP = wtune.machine.MEASURED["s_per_step"]    # wtune.machine.measured
TIME_SAFETY_FACTOR = 1.5
TIME_PAD_MIN = 10.0
TIME_CAP_HOURS = 12.0             # of one probe job; time_cap_hours() adds the machine's cap
TIMEOUT_MULTIPLIER = 3.0
TIMEOUT_MIN_S = 600

MANIFEST_FIELDS = ["case", "nodes", "job_id", "configs_path", "n_configs", "submitted_at", "state"]
# `sacct -j` alone searches all of slurmdbd and answers with the newest job of that number. On
# a cluster whose job ids started over (f1-arm) that is another user's old job until the new
# one is recorded: a job just submitted read as FAILED.
SACCT_SINCE = ("-S", "now-30days")
IN_FLIGHT_STATES = {"PENDING", "RUNNING"}   # sacct State values counted against --max-nodes-in-flight

_JOBID_RE = re.compile(r"Submitted batch job (\d+)")
_SBATCH_RE = re.compile(r"(?<![\w./-])sbatch(?: -p \S+)?(?=\s)")    # the command, not *.sbatch
_SUBMIT_LIMIT_RE = re.compile(r"MaxSubmitJob")    # per-user count of queued jobs, not this job
SUBMIT_LIMIT_WAIT_S = 60
SUBMIT_LIMIT_MAX_WAIT_S = 6 * 3600


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


# ---------------------------------------------------------------------------
# case selection
# ---------------------------------------------------------------------------
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


def default_ledger_dir():
    """Where probes write their ledgers when nothing else says: $WTUNE_CAMPAIGN_OUT, else
    DEFAULT_LEDGER_DIR."""
    return _expand(os.environ.get(CAMPAIGN_OUT_ENV) or DEFAULT_LEDGER_DIR)


def history_ledger_dirs(ledger_dir):
    """[HISTORY_LEDGER_DIR] when ledger_dir is another directory, else []: a project's probes
    write into the project, earlier tunes of the same case id wrote into the shared one."""
    shared = _expand(HISTORY_LEDGER_DIR)
    return [] if os.path.realpath(shared) == os.path.realpath(_expand(ledger_dir)) else [shared]


def job_wrf_tree(env_source, campaign_dir, site=None):
    """The $W a probe job will hash: the site file's, after the submit shell's `source env.sh`
    (build_sbatch_command); None when it cannot be resolved."""
    if not campaign_dir:
        return None
    script = ('source %s >/dev/null 2>&1; CAMPAIGN=%s; source "$CAMPAIGN/sites/%s.sh" '
              '>/dev/null 2>&1; printf %%s "$W"'
              % (shlex.quote(_expand(env_source or wtune.machine.env_source())),
                 shlex.quote(campaign_dir), site or wtune.machine.site()))
    try:
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def wrf_build_id(wrf_tree):
    """sha1 of <wrf_tree>/main/wrf.exe's contents, first 16 hex (coldstart_sample.build_id_expr,
    the probes' build= ledger token); '' when unreadable."""
    import hashlib
    h = hashlib.sha1()
    try:
        with open(os.path.realpath(os.path.join(_expand(wrf_tree or ""), "main", "wrf.exe")),
                  "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()[:16]


def row_build_id(row):
    """The build= token of a ledger row's notes, '' for a row written before it existed."""
    for tok in (row.get("notes") or "").split():
        if tok.startswith("build="):
            return tok[len("build="):]
    return ""


def estimate_s_per_step(case, nodes, ledger_dir, hardware_id=None,
                         fallback=None, build_id=None, extra_dirs=()):
    """Median of (measured s_per_step_median * n_nodes_measured / nodes) over the ledger rows
    for `case` on `hardware_id` at the node count nearest `nodes` (log scale; the larger on a
    tie), scanning ledger_<case>_*.csv in ledger_dir and extra_dirs -- ideal linear scaling
    only across that one gap (over all N it was 4x off: matthew 0.061 vs 0.016). With
    build_id, only that build's rows ('ledger'); when it has none, the rows that carry no build id ('ledger_unknown_build'); another build's rows
    never (a tuned-flag build is not the vanilla one). `fallback` (source 'fallback_10s') when
    no matching row exists at all. Returns (s_est, source, n_rows_used).

    ledger_dir is expanded here (not just by CLI-level callers) so an unexpanded default like
    "$HOME/wtune-runs/campaign" resolves correctly regardless of what the caller passes."""
    fallback = wtune.machine.measured("s_per_step") if fallback is None else fallback
    hardware_id = hardware_id or wtune.machine.hardware_id()
    same, unknown = [], []
    seen = set()
    for d in [_expand(ledger_dir)] + [_expand(x) for x in extra_dirs]:
        for path in sorted(glob.glob(os.path.join(d, "ledger_%s_*.csv" % case))):
            if os.path.basename(path) in seen:
                continue
            seen.add(os.path.basename(path))
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
                b = row_build_id(r)
                if not build_id or b == build_id:
                    same.append((n, s))
                elif not b:
                    unknown.append((n, s))
    for pool, source in ((same, "ledger"), (unknown, "ledger_unknown_build")):
        if pool:
            near = min({n for n, _s in pool}, key=lambda n: (abs(math.log(n / nodes)), -n))
            vals = [s * n / nodes for n, s in pool if n == near]
            return statistics.median(vals), source, len(vals)
    return fallback, "fallback_10s", 0


# ---------------------------------------------------------------------------
# sacct / sbatch, both behind injectable runner functions (tests substitute fakes)
# ---------------------------------------------------------------------------
def default_sacct_runner(cmd):
    """sacct's stdout; "" when sacct failed (rc != 0), so no job gets a state from a broken call
    and callers see every job as unreported (UNKNOWN) rather than finished."""
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return out.stdout if out.returncode == 0 else ""


def sacct_states(job_ids, sacct_runner=default_sacct_runner, sacct_cmd="sacct"):
    """{job_id: (state, elapsed)} via ONE batched sacct call (-X: no job-step rows, -n: no
    header, -P: pipe-delimited)."""
    job_ids = [str(j) for j in job_ids if j]
    if not job_ids:
        return {}
    cmd = [sacct_cmd, "-j", ",".join(job_ids), *SACCT_SINCE, "-X", "-n", "-P",
           "--format=JobID,State,Elapsed"]
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


def campaign_dir_for(sbatch_script):
    """The campaign/ directory an sbatch script belongs to (its own dirname), or None when that
    directory is not a campaign checkout (no configs.py next to it -- e.g. a script copied
    elsewhere). strict_campaign.sbatch cannot find its own path once Slurm has spooled it, and
    otherwise falls back to CAMPAIGN=$HOME/wtune/campaign -- which pairs the submitted script with
    a possibly different tree's configs.py/sites/node_hygiene.sh. Exporting CAMPAIGN=<this> keeps
    script and helpers from the SAME tree."""
    d = os.path.dirname(os.path.abspath(_expand(sbatch_script)))
    return d if os.path.isfile(os.path.join(d, "configs.py")) else None


PREFLIGHT_EXCLUDE_ENV = "WTUNE_PREFLIGHT_EXCLUDE"   # set by the controller itself (wtune tune's
                                                      # node health pre-flight): nodes that failed it


def time_cap_hours():
    """TIME_CAP_HOURS, or the machine's max_hours when that is lower."""
    return wtune.machine.cap_seconds(TIME_CAP_HOURS * 3600.0) / 3600.0


def excluded_nodes():
    """Ordered, de-duplicated node names every sbatch of this tool chain must avoid: the login
    nodes (wtune.machine, scc: head),
    the user's opt-in WTUNE_SBATCH_EXCLUDE, and the controller's own WTUNE_PREFLIGHT_EXCLUDE (nodes
    the health pre-flight found unusable). ONE helper -- exclude_arg() and
    tools/coldstart_sample.render_run_sbatch both read it."""
    out = wtune.machine.login_nodes()
    for var in ("WTUNE_SBATCH_EXCLUDE", PREFLIGHT_EXCLUDE_ENV):
        for n in os.environ.get(var, "").split(","):
            n = n.strip()
            if n and n not in out:
                out.append(n)
    return out


def exclude_arg():
    """'--exclude=head[,<WTUNE_SBATCH_EXCLUDE nodes>][,<pre-flight failures>]' -- the same opt-in
    extra exclusion tools/coldstart_sample.render_run_sbatch honours (scc work5, 2026-09-27:
    non-Slurm root load slowed wrf.exe 5x). '' when there is nothing to exclude."""
    nodes = excluded_nodes()
    return "--exclude=" + ",".join(nodes) if nodes else ""


def scheduler_args():
    """exclude_arg() and the machine's --account, space-joined; '' when neither applies."""
    return " ".join(a for a in (exclude_arg(), wtune.machine.account_arg()) if a)


def build_sbatch_command(case, nodes, cores_per_node, time_limit, timeout_s, reps, steps,
                          configs_path, out_dir, env_source=None,
                          sbatch_script=DEFAULT_SBATCH_SCRIPT, campaign_dir=None, extra_export=None):
    """campaign_dir (optional): appended as CAMPAIGN=<dir> (see campaign_dir_for). extra_export
    (optional): [(KEY, VALUE), ...] appended verbatim after it (CONFIG_FILE stays last) (decision 0055 ranking context:
    CASES_JSON/RANKING_START/SPINUP_MINUTES/...); a VALUE containing ',' is refused -- sbatch
    --export would split it. Both default to None, which reproduces the old command exactly."""
    total_cores = nodes * cores_per_node
    export = ("ALL,SITE=%s,WTUNE_TOTAL_CORES=%d,CASE=%s,MODE=timing,REPS=%d,STEPS=%d,"
              "TIMEOUT_S=%d" % (wtune.machine.site(), total_cores, case, reps, steps, timeout_s))
    if campaign_dir:
        export += ",CAMPAIGN=%s" % campaign_dir
    for key, value in (extra_export or []):
        if "," in str(value):
            raise ValueError("--export value %s=%r contains ',' which sbatch --export would split"
                             % (key, value))
        export += ",%s=%s" % (key, value)
    export += ",CONFIG_FILE=%s" % configs_path      # kept LAST, exactly as before
    sbatch = ("sbatch -p %s -N %d --ntasks-per-node=%d --exclusive %s "
              "--time=%s --job-name=sw-%s-n%d --output=%s/sweep-%%j.out --export=%s %s"
              % (wtune.machine.partition(), nodes, cores_per_node, scheduler_args(), time_limit,
                 case, nodes, out_dir, export, _expand(sbatch_script)))
    return "source %s; %s" % (_expand(env_source or wtune.machine.env_source()), sbatch)


def submit_job(cmd_str, sbatch_runner=default_sbatch_runner, sleep=time.sleep):
    """`_submit_once`, retried every minute (up to 6 h) while a refusal is the per-user limit on
    queued jobs (QOSMaxSubmitJobPerUserLimit): that limit is about the user's other jobs, e.g. a
    second tune sharing the QOS, so the job fits once one of them ends."""
    waited = 0
    while True:
        try:
            return _submit_once(cmd_str, sbatch_runner)
        except RuntimeError as e:
            if not _SUBMIT_LIMIT_RE.search(str(e)) or waited >= SUBMIT_LIMIT_MAX_WAIT_S:
                raise
            if not waited:
                print("submit: per-user job limit reached; retrying every %d s"
                      % SUBMIT_LIMIT_WAIT_S, file=sys.stderr)
        sleep(SUBMIT_LIMIT_WAIT_S)
        waited += SUBMIT_LIMIT_WAIT_S


def _submit_once(cmd_str, sbatch_runner):
    """Runs `cmd_str` (already includes the `source env.sh;` prefix) and parses the job id out
    of sbatch's "Submitted batch job N" stdout line. --force re-running an in-flight job's
    (case,N) slot is the one accepted imprecision in the --max-nodes-in-flight budget: the OLD
    job's nodes stay counted in the run's initial in-flight baseline (taken once, from the
    manifest, before the submit loop starts) even after the manifest row is overwritten --
    conservative (under-states available budget) rather than over-subscribing the cluster.
    With several partitions each is tried in order (`-p` on the command line beats the
    script's own); RuntimeError naming every refusal when none takes the job."""
    parts = wtune.machine.partitions()
    if len(parts) == 1:
        out = sbatch_runner(cmd_str)
    else:
        refused = []
        for p in parts:
            cmd, n = _SBATCH_RE.subn("sbatch -p " + p, cmd_str, count=1)
            if not n:
                raise ValueError("no sbatch in %r" % (cmd_str,))
            try:
                out = sbatch_runner(cmd)
                break
            except RuntimeError as e:
                refused.append("%s: %s" % (p, e))
        else:
            raise RuntimeError("no partition took the job -- " + "; ".join(refused))
    m = _JOBID_RE.search(out)
    if not m:
        raise RuntimeError("could not parse job id from sbatch output: %r" % (out,))
    return m.group(1)
