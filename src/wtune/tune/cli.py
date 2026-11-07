"""`wtune tune` command line: argument parsing, option resolution (CLI flag > --config > built-in
default), the run context, the mode dispatch and main()."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import sys
from typing import Any, Callable

import wtune.config
import wtune.extensions
import wtune.machine
import wtune.nodes
import wtune.stopping
from ._deps import _HERE, sweep
from .context import (CONFIRM_REPS, DEFAULT_KAPPA,
                      DEFAULT_POLL_INTERVAL_S, DEFAULT_SEED, DEFAULT_STEPS, Ctx, RealClock,
                      default_ledger_reader, default_scancel_runner, resolve_auto_probe_steps)
from .estimates import DEFAULT_LEVEL, LEVELS
from .jobs import _final_cancel_safety_net
from .node_search import DEFAULT_SWEEP_BUDGET_FRAC, NODES_POLICIES, choose_nodes
from .physics_phase import PIPELINE_PHYSICS_MAX_CANDIDATES
from .pipeline import PIPELINE_MAX_QUEUE_HOURS, PRECISION_DEFAULT
from .prediction import PRECISION_FLOOR_MIN
from .preflight import default_preflight_runner, node_preflight
from .report import _emit_report
from .context import STOP, StopRequested
from . import context as _context
from . import jobs as _jobs
from . import node_search as _node_search
from . import pipeline as _pipeline
from . import report as _report
from . import rounds as _rounds


@dataclasses.dataclass
class _RunOptions:
    """run()'s options resolved from the CLI and --config (CLI flag > file > built-in default),
    before the case is resolved."""
    config_path: Any
    cfg: dict
    case: str
    minutes: float | None
    level: int | None
    nodes_raw: Any
    nodes_policy: str
    max_nodes_requested: int
    steps_cfg: Any
    confirm_reps: int
    restart_io_form: Any


@dataclasses.dataclass
class _NodeChoice:
    """How run() picks the node count: the mode (pipeline "full" or None, node_search), the
    --nodes search options, and the fixed/auto/search node count with its note."""
    pipeline: str | None
    node_search: bool
    node_objective: str
    deadline_minutes: float | None
    min_efficiency: float
    node_finalists: int
    sweep_budget_frac: float
    node_seeds: str
    hotspot_fill: bool
    core_options: list | None
    total_steps: int | None
    nodes: int
    nodes_note: str
    proxy: bool = False


def _precision_arg(text: str) -> str:
    """--precision: "N%" (of the run's wall time) or minutes; argparse rejects anything else."""
    t = text.strip()
    try:
        value = float(t[:-1] if t.endswith("%") else t)
    except ValueError:
        raise argparse.ArgumentTypeError("expected N%% or minutes, got %r" % text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0, got %r" % text)
    return t


def _cfg_key_explicit(config_path: str | None, section: str, key: str) -> bool:
    """True when the --config YAML itself sets <section>.<key> (not just the built-in default)."""
    if not config_path:
        return False
    try:
        yaml = wtune.config._load_yaml_module()
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return key in (raw.get(section) or {})
    except Exception:
        return False


def run(args: argparse.Namespace, sbatch_runner: Callable | None = None,
        sacct_runner: Callable | None = None, scancel_runner: Callable | None = None,
        ledger_reader: Callable | None = None, proposer: Callable | None = None,
        clock: Any = None, preflight_runner: Callable | None = None) -> dict:
    """`wtune tune` for parsed `args`: resolve the options, the case, the node pre-flight and
    the node count, build the Ctx and run the mode -- -O0, -O1..3, --nodes search or
    a fixed node count -- then print the report and write <out>/tune_report.json. Returns the
    report; SystemExit for invalid options. The runners and clock default to the real ones."""
    STOP["reason"] = None
    o = _resolve_run_options(args)
    case_json, namelist_path, profile = _context.resolve_case(o.case, args.cores_per_node)
    is_fire, ext_notes, steps, ranking_start_note = _run_case_inputs(o, case_json, namelist_path)

    _use_project_bp_cache(args)                     # the cache the space check looks at
    if o.level != 0 and not getattr(args, "no_space_check", False):
        _jobs.check_input_space(o.case)             # before the node preflight's probes too
    max_nodes = min(int(o.max_nodes_requested), wtune.machine.max_nodes())
    node_health, preflight, max_nodes = _run_node_preflight(args, o.level, case_json, max_nodes,
                                                            preflight_runner)
    ns = _resolve_node_choice(args, o, profile, max_nodes, namelist_path)
    out_dir = _run_out_dir(args, o.case)
    ctx = _build_ctx(args, o, ns, case_json, namelist_path, is_fire, steps, ranking_start_note,
                     max_nodes, out_dir, sbatch_runner, sacct_runner, scancel_runner,
                     ledger_reader, proposer, clock)
    ctx.node_health = node_health
    ctx.start_time = ctx.clock.now()

    print("case=%s nodes=%s (%s) budget=%s max_nodes=%d out=%s"
          % (o.case, "search" if ns.node_search else ns.nodes, ns.nodes_note,
             ("%.1f min" % o.minutes) if o.minutes is not None else "none", max_nodes, out_dir))

    s_est, _src, _n = ctx.estimate_s_per_step()

    ctx.state_file = os.path.join(out_dir, "tune.state.json")
    with _campaign_out(ctx.ledger_dir), _stop_on_signal(ctx):
        try:
            report = _run_mode(ctx, profile, args, o, ns, ext_notes, preflight, s_est)
        except StopRequested as e:
            os.remove(ctx.state_file)
            raise SystemExit("wtune tune: %s before anything was measured -- no report" % e) \
                from None
    if STOP["reason"]:
        report["stopped"] = STOP["reason"]
    _emit_report(report, out_dir, node_health)
    if STOP["reason"]:
        print("wtune tune: %s -- %s/tune_report.json holds the best of what was measured; "
              "`wtune run` uses it" % (STOP["reason"], out_dir))
    return report


def _stop_on_signal(ctx: Ctx):
    """Ctrl-C (SIGINT) or `wtune stop` (SIGTERM): stop at the next poll -- cancel this tune's
    jobs, keep what was measured, write the report (wtune.stopping)."""
    return wtune.stopping.stop_on_signal(ctx.state_file, "tune", ctx.poll_interval)


def _run_mode(ctx: Ctx, profile: dict, args: argparse.Namespace, o: _RunOptions, ns: _NodeChoice,
              ext_notes: list, preflight: Any, s_est: float) -> dict:
    """The report of the chosen mode: -O0, the pipeline, --nodes search or a fixed node count."""
    if o.level == 0:                       # no jobs: nothing to cancel
        return _pipeline.run_level0(ctx, profile, ns.total_steps, o.nodes_policy)
    elif ns.pipeline == "full":
        report = _run_pipeline_mode(ctx, profile, args, o, ns, ext_notes, preflight)
        _final_cancel_safety_net(ctx)
    elif ns.node_search:
        report = _node_search.run_node_search(ctx, profile, objective=ns.node_objective,
                                              deadline_s=(ns.deadline_minutes * 60.0
                                                          if ns.deadline_minutes is not None
                                                          else None),
                                              min_efficiency=ns.min_efficiency,
                                              k_finalists=ns.node_finalists,
                                              total_steps=ns.total_steps,
                                              nodes_policy=o.nodes_policy,
                                              sweep_budget_frac=ns.sweep_budget_frac,
                                              core_options=ns.core_options)
        _final_cancel_safety_net(ctx)
    else:
        all_results, confirm_results, anchor_cid = _rounds.run_tuning_at(ctx, s_est)
        _final_cancel_safety_net(ctx)
        report = _report.build_report(ctx, ns.nodes, ns.nodes_note, all_results, confirm_results,
                                      anchor_cid)
    return report


def _build_ctx(args: argparse.Namespace, o: _RunOptions, ns: _NodeChoice, case_json: dict,
               namelist_path: str, is_fire: bool, steps: int, ranking_start_note: str | None,
               max_nodes: int, out_dir: str, sbatch_runner: Callable | None,
               sacct_runner: Callable | None, scancel_runner: Callable | None,
               ledger_reader: Callable | None, proposer: Callable | None, clock: Any) -> Ctx:
    """The run's Ctx; missing runners / clock default to the real Slurm / time ones."""
    sbatch_runner = sbatch_runner or sweep.default_sbatch_runner
    sacct_runner = sacct_runner or sweep.default_sacct_runner
    scancel_runner = scancel_runner or default_scancel_runner
    clock = clock or RealClock()
    ledger_dir = _ledger_dir(args)
    ledger_reader = ledger_reader or (lambda case, job_ids: default_ledger_reader(
        ledger_dir, case, job_ids))
    extension_omp_cap = wtune.extensions.omp_cap_for_namelist(namelist_path)
    tree = _probe_tree(args)
    build_omp_cap = wtune.extensions.omp_cap_for_build(tree)
    if build_omp_cap:
        print("wtune tune: %s/main/wrf.exe has no OpenMP (dmpar build): omp capped at 1" % tree)
    case_caps = [c for c in (extension_omp_cap, build_omp_cap) if c]
    return Ctx(case=o.case, case_json=case_json, namelist_path=namelist_path, nodes=ns.nodes,
               cores_per_node=args.cores_per_node, hardware_id=args.hardware_id, steps=steps,
               seed=args.seed, kappa=args.kappa,
               budget_s=(o.minutes * 60.0 if o.minutes is not None else float("inf")),
               max_nodes=max_nodes,
               out_dir=out_dir, ledger_dir=ledger_dir, env_source=args.env_source,
               sbatch_script=args.sbatch_script, sacct_cmd=args.sacct_cmd,
               poll_interval=args.poll_interval, sbatch_runner=sbatch_runner,
               sacct_runner=sacct_runner, scancel_runner=scancel_runner,
               ledger_reader=ledger_reader, proposer=proposer, clock=clock,
               build_id=_build_id(tree),
               history_dirs=sweep.history_ledger_dirs(ledger_dir),
               search_cfg=o.cfg["search"], exclude_rules=o.cfg["exclude"], is_fire=is_fire,
               omp_cap=wtune.machine.omp_cap(min(case_caps) if case_caps else None),
               extension_omp_cap=extension_omp_cap,
               chem_e_sn=wtune.extensions.chem_e_sn_for_namelist(namelist_path),
               confirm_reps=o.confirm_reps, prediction_cfg=o.cfg["prediction"],
               probe_cfg=o.cfg["probe"], ranking_start_note=ranking_start_note,
               restart_io_form=int(o.restart_io_form), node_seeds=ns.node_seeds,
               hotspot_fill=ns.hotspot_fill, proxy=bool(getattr(ns, "proxy", False)), wrf_tree=tree or "")


def _run_config(args: argparse.Namespace) -> tuple[str | None, dict]:
    """(config_path, cfg) of --config; without one, resolve_config({})'s built-in defaults (which
    reproduce the pre---config argparse defaults). SystemExit listing every violation."""
    config_path = getattr(args, "config", None)
    if config_path:
        try:
            cfg = wtune.config.load_config(config_path)
        except wtune.config.ConfigError as e:
            raise SystemExit("--config %s is invalid:\n%s"
                             % (config_path, "\n".join("  - %s" % m for m in e.errors))) from e
    else:
        cfg, _errs = wtune.config.resolve_config({})
    return config_path, cfg


def _run_case_name(args: argparse.Namespace, cfg: dict) -> str:
    """The case id: positional, else --config's 'case', else the project's registered case;
    SystemExit when none is given (a case_dir-only config cannot be tuned yet)."""
    case = args.case if args.case is not None else cfg["case"]
    if case is None and not cfg.get("case_dir"):
        import wtune.project
        project_dir = wtune.project.find_project(explicit=getattr(args, "project", None))
        if project_dir:
            case = wtune.project.project_case_id(wtune.project.load_project(project_dir))
    if case is None:
        if cfg.get("case_dir"):
            raise SystemExit(
                "config declares case_dir=%r, but wtune tune cannot run a tune against an "
                "arbitrary case_dir yet (only `wtune config show`/`config init` preview it) -- "
                "pass a registered case name positionally, or set 'case' in --config instead"
                % cfg["case_dir"])
        raise SystemExit("case is required: pass it positionally, set 'case' in --config, or "
                         "run inside a project made with `wtune project init --case DIR`")
    return case


def _run_nodes_raw(args: argparse.Namespace, cfg: dict, config_path: str | None) -> Any:
    """The --nodes request (N, "auto" or "search"), default "search". SystemExit for a
    budget.nodes list of several counts."""
    nodes_raw = args.nodes if args.nodes is not None else cfg["budget"]["nodes"]
    if (args.nodes is None and nodes_raw == "auto"
            and not _cfg_key_explicit(config_path, "budget", "nodes")):
        nodes_raw = "search"      # default: MEASURE s/step across node counts (node sweep)
    if isinstance(nodes_raw, list):
        if len(nodes_raw) != 1:
            raise SystemExit(
                "budget.nodes as a list of multiple node counts (%r) is not yet run by `wtune "
                "tune` (its pipeline targets one node count per run) -- use `wtune config show` "
                "to preview candidate counts across node counts, or pass a single --nodes value"
                % (nodes_raw,))
        nodes_raw = nodes_raw[0]
    return nodes_raw


def _run_probe_options(args: argparse.Namespace, cfg: dict) -> Any:
    """restart_io_form; writes --points-by / --max-points into cfg["probe"]. SystemExit for
    --config's retired probe.rank_by: predicted."""
    if cfg["probe"].get("rank_by") == "predicted":
        raise SystemExit("probe.rank_by: predicted was retired (archive/unused-2026-10); remove "
                         "the key or set it to t0")
    if getattr(args, "points_by", None) is not None:
        cfg["probe"]["points_by"] = args.points_by
    if getattr(args, "max_points", None) is not None:
        cfg["probe"]["max_points"] = args.max_points
    return getattr(args, "restart_io_form", None) or cfg["probe"].get("restart_io_form") or 14


def _resolve_run_options(args: argparse.Namespace) -> _RunOptions:
    """Everything decidable before the case is touched: --config, case, --minutes / level
    (--pipeline full = O3, where --minutes is an optional cap), nodes, probe options.
    SystemExit for missing or invalid options, in run()'s historical order."""
    config_path, cfg = _run_config(args)
    case = _run_case_name(args, cfg)
    minutes = args.minutes if args.minutes is not None else cfg["budget"]["minutes"]
    level = getattr(args, "level", None)
    if level is None and getattr(args, "pipeline", None) == "full":
        level = 3
    if minutes is None and level is None:
        raise SystemExit("--minutes is required: pass it on the command line or set "
                         "budget.minutes in --config")
    nodes_raw = _run_nodes_raw(args, cfg, config_path)
    nodes_policy = args.nodes_policy if args.nodes_policy is not None else cfg["budget"]["nodes_policy"]
    max_nodes_requested = args.max_nodes if args.max_nodes is not None else cfg["budget"]["max_nodes"]
    steps_cfg = args.steps if args.steps is not None else cfg["probe"]["steps"]
    if steps_cfg != "auto":
        cfg["probe"]["sweep_steps"] = None       # a fixed probe length holds for the sweeps too
    confirm_reps = args.confirm_reps if args.confirm_reps is not None else cfg["probe"]["confirm_reps"]
    restart_io_form = _run_probe_options(args, cfg)
    return _RunOptions(config_path=config_path, cfg=cfg, case=case, minutes=minutes, level=level,
                       nodes_raw=nodes_raw, nodes_policy=nodes_policy,
                       max_nodes_requested=max_nodes_requested, steps_cfg=steps_cfg,
                       confirm_reps=confirm_reps, restart_io_form=restart_io_form)


def _run_case_inputs(o: _RunOptions, case_json: dict,
                     namelist_path: str) -> tuple[bool, list, int, str | None]:
    """(is_fire, extension notes, probe steps, ranking_start note) of the resolved case: the
    extension gate (SystemExit when it refuses), probe.steps 'auto' and the not-yet-wired
    probe.ranking_start=spinup note, printed."""
    is_fire = wtune.config.is_fire_case(case_json, namelist_path)
    _ext_notes, _ext_refusal = wtune.extensions.tune_gate(namelist_path)
    if _ext_refusal:
        raise SystemExit(_ext_refusal)
    for _n in _ext_notes:
        print("EXTENSION: %s" % _n)

    if o.steps_cfg == "auto":
        steps, steps_note = resolve_auto_probe_steps(namelist_path)
    else:
        steps, steps_note = o.steps_cfg, None
    if steps_note:
        print("NOTE: %s" % steps_note)
    ranking_start_note = None
    if o.cfg["probe"].get("ranking_start") == "spinup":
        # the sbatch run-loop hand-off of a shared restart is not wired: say so, never ignore it
        ranking_start_note = (
            "probe.ranking_start=spinup requested but restart-based ranking-probe start is NOT "
            "YET WIRED to the campaign submission pipeline (campaign/configs.py's edit_namelist "
            "restart_from/write_restart_interval/restart_io_form params exist; the sbatch run-"
            "loop wiring that would use them does not yet) -- every candidate is still probed "
            "from t0, exactly as before probe.ranking_start existed.")
        print("NOTE: %s" % ranking_start_note)
    return is_fire, _ext_notes, steps, ranking_start_note


def _run_node_preflight(args: argparse.Namespace, level: int | None, case_json: dict,
                        max_nodes: int, preflight_runner: Callable | None
                        ) -> tuple[dict | None, Callable | None, int]:
    """(start check, re-check callable, max_nodes capped at the healthy count); skipped (None,
    None, max_nodes) by --no-node-preflight and for -O0. Controller time, never budget."""
    if (getattr(args, "no_node_preflight", False) or level == 0      # O0 runs no jobs
            or not wtune.machine.node_preflight()):
        return None, None, max_nodes
    _src_dir = os.path.dirname(sweep.campaign_dir_for(args.sbatch_script) or _HERE)
    _wrf = os.path.join(sweep._expand(wtune.machine.wrf_tree()),
                        "main", "wrf.exe")
    _requested = max_nodes

    def preflight(when="start"):
        return node_preflight(preflight_runner or default_preflight_runner,
                              case_dir=sweep._expand(case_json["prep_dir"]), src_dir=_src_dir,
                              wrf_exe=_wrf, requested=_requested, when=when)
    node_health = preflight("start")
    return node_health, preflight, node_health["max_nodes_effective"]


def _parse_core_options(args: argparse.Namespace) -> list[int] | None:
    """--cores-per-node-options as a descending list of distinct ints (None when not given:
    wtune.nodes.default_core_options). SystemExit for a non-integer or a value outside
    1..--cores-per-node."""
    core_options_raw = getattr(args, "cores_per_node_options", None)
    core_options = None
    if core_options_raw:
        try:
            core_options = sorted({int(v) for v in core_options_raw.split(",") if v.strip()},
                                  reverse=True)
        except ValueError as e:
            raise SystemExit("--cores-per-node-options must be a comma list of integers, got %r"
                             % core_options_raw) from e
        if not core_options or any(v < 1 or v > args.cores_per_node for v in core_options):
            raise SystemExit("--cores-per-node-options values must be in 1..%d (--cores-per-node), "
                             "got %r" % (args.cores_per_node, core_options_raw))
    return core_options


def _resolve_node_choice(args: argparse.Namespace, o: _RunOptions, profile: dict,
                         max_nodes: int, namelist_path: str) -> _NodeChoice:
    """The mode (-O / --pipeline: its own node search), the --nodes search options (CLI flag >
    --config budget.<name> > default), the total step count and the node count. Prints the
    --nodes auto warning; SystemExit for incompatible options."""
    nodes_raw = o.nodes_raw
    pipeline = "full" if o.level is not None else None
    if pipeline == "full":
        if args.nodes not in (None, "search"):
            raise SystemExit("-O/--pipeline runs its own node search: do not combine it with "
                             "--nodes N/auto or --rank-by predicted")
        nodes_raw = "search"
    node_search = (nodes_raw == "search")

    def _nopt(name, default):          # CLI flag > --config budget.<name> > built-in default
        v = getattr(args, name, None)
        if v is None:
            v = o.cfg["budget"].get(name)
        return default if v is None else v
    node_objective = _nopt("node_objective", "time")
    deadline_minutes = _nopt("deadline_minutes", None)
    min_efficiency = _nopt("min_efficiency", wtune.nodes.DEFAULT_MIN_EFFICIENCY)
    node_finalists = max(1, int(_nopt("node_finalists", 2)))
    sweep_budget_frac = float(_nopt("sweep_budget_frac", DEFAULT_SWEEP_BUDGET_FRAC))
    node_seeds = _nopt("node_seeds", "geometric")
    hotspot_fill = _nopt("hotspot_fill", True)
    hotspot_fill = hotspot_fill if isinstance(hotspot_fill, bool) else hotspot_fill == "on"
    proxy = _nopt("proxy", True)
    proxy = proxy if isinstance(proxy, bool) else proxy == "on"
    core_options = _parse_core_options(args)
    total_steps = _context._total_steps(namelist_path) if node_search else None
    adaptive = _context.adaptive_steps_warning(namelist_path)
    if adaptive:
        print("WARNING: " + adaptive)
    if node_search and node_objective == "cost" and total_steps is None:
        raise SystemExit("--node-objective cost needs the case's total step count (run length / "
                         "time_step from the namelist), which could not be determined")
    if node_search:
        nodes, nodes_note = 1, "node search (measured)"
    elif nodes_raw in ("auto", None):
        nodes, nodes_note, nodes_warning = choose_nodes(profile, max_nodes, o.nodes_policy)
        print("WARNING: %s" % nodes_warning)
    else:
        nodes = int(nodes_raw)
        nodes_note = "user-specified (cap=%d)" % max_nodes
        if nodes > max_nodes:
            raise SystemExit("--nodes %d exceeds --max-nodes %d" % (nodes, max_nodes))
    return _NodeChoice(pipeline=pipeline, node_search=node_search, node_objective=node_objective,
                       deadline_minutes=deadline_minutes, min_efficiency=min_efficiency,
                       node_finalists=node_finalists, sweep_budget_frac=sweep_budget_frac,
                       node_seeds=node_seeds, hotspot_fill=hotspot_fill, core_options=core_options,
                       total_steps=total_steps, nodes=nodes, nodes_note=nodes_note, proxy=proxy)


def _run_out_dir(args: argparse.Namespace, case: str) -> str:
    """--out, else <project>/tune/<id>/ (decision 0058), else $HOME/wtune-runs/tune/<case>;
    created."""
    if args.out:
        out_dir = sweep._expand(args.out)
    else:
        import wtune.project
        _project_dir = wtune.project.find_project(explicit=getattr(args, "project", None))
        out_dir = (wtune.project.default_out_dir(_project_dir, "tune") if _project_dir
                   else sweep._expand("$HOME/wtune-runs/tune/%s" % case))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _ledger_dir(args: argparse.Namespace) -> str:
    """--ledger-dir, else <project>/probes inside a project, else sweep.default_ledger_dir()."""
    import wtune.project
    ledger_dir = args.ledger_dir
    if not ledger_dir:
        project_dir = wtune.project.find_project(explicit=getattr(args, "project", None))
        ledger_dir = (os.path.join(project_dir, "probes") if project_dir
                      else sweep.default_ledger_dir())
    return sweep._expand(ledger_dir)


def _probe_tree(args: argparse.Namespace) -> str:
    """The WRF tree the probe jobs will run: the site file's $W, else machine.wrf_tree()."""
    return (sweep.job_wrf_tree(args.env_source, sweep.campaign_dir_for(args.sbatch_script))
            or wtune.machine.wrf_tree())


def _build_id(tree: str) -> str:
    """The WRF build the probe jobs will run (sweep.wrf_build_id of `tree`); '' when there is no
    wrf.exe there, and the ledger estimates then use every build's rows."""
    build_id = sweep.wrf_build_id(tree)
    print("build %s (%s/main/wrf.exe)" % (build_id, tree) if build_id else
          "wtune tune: no wrf.exe under %s -- the ledger estimates use every build's rows" % tree)
    return build_id


@contextlib.contextmanager
def _campaign_out(ledger_dir: str):
    """WTUNE_CAMPAIGN_OUT=ledger_dir while the tune runs, so its probe jobs (sbatch --export=ALL)
    write where it reads (the site files' $OUT follows it); restored afterwards."""
    old = os.environ.get(sweep.CAMPAIGN_OUT_ENV)
    os.environ[sweep.CAMPAIGN_OUT_ENV] = ledger_dir
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(sweep.CAMPAIGN_OUT_ENV, None)
        else:
            os.environ[sweep.CAMPAIGN_OUT_ENV] = old


def _use_project_bp_cache(args: argparse.Namespace) -> None:
    """Inside a project the tune's BP input cache is <project>/input/bp-cache/ (decision 0058),
    not $HOME/wtune-runs/bp-cache; an explicit WTUNE_BP_CACHE still wins. The jobs inherit it
    through sbatch --export=ALL."""
    import wtune.project
    project_dir = wtune.project.find_project(explicit=getattr(args, "project", None))
    if project_dir:
        os.environ.setdefault("WTUNE_BP_CACHE", os.path.join(project_dir, "input", "bp-cache"))


def _run_pipeline_mode(ctx: Ctx, profile: dict, args: argparse.Namespace, o: _RunOptions,
                       ns: _NodeChoice, ext_notes: list, preflight: Callable | None) -> dict:
    """-O1..3 / --pipeline full: run_pipeline_full with the resolved options (not yet emitted)."""
    return _pipeline.run_pipeline_full(
        ctx, profile, minutes=o.minutes, level=o.level, total_steps=ns.total_steps,
        objective=ns.node_objective,
        deadline_s=(ns.deadline_minutes * 60.0 if ns.deadline_minutes is not None else None),
        min_efficiency=ns.min_efficiency, k_finalists=ns.node_finalists,
        nodes_policy=o.nodes_policy,
        sweep_budget_frac=ns.sweep_budget_frac, core_options=ns.core_options,
        extension=bool(ext_notes or ctx.extension_omp_cap or ctx.chem_e_sn),
        physics=bool(getattr(args, "physics_advice", False)),
        physics_max_candidates=int(getattr(args, "physics_max_candidates", None)
                                   or PIPELINE_PHYSICS_MAX_CANDIDATES),
        recheck=preflight, stop_if_short=bool(getattr(args, "stop_if_short", False)),
        precision_min=(args.precision if getattr(args, "precision", None) is not None
                       else PRECISION_DEFAULT),
        max_queue_hours=(getattr(args, "max_queue_hours", None)
                         if getattr(args, "max_queue_hours", None) is not None
                         else PIPELINE_MAX_QUEUE_HOURS),
        keep_frames=bool(getattr(args, "keep_frames", False)))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wtune tune")
    p.add_argument("case", nargs="?", default=None,
                    help="registered campaign/cases.json case id; required here or via "
                         "--config's 'case' key")
    p.add_argument("--config", default=None,
                    help="wtune config YAML file: search-space restrictions + budget/probe/"
                         "prediction defaults (wtune/config.py); CLI flags given explicitly still "
                         "override it, which overrides this module's own built-in defaults")
    p.add_argument("--site", default=None,
                    help="campaign/sites/<site>.sh (default: the machine's, built-in scc)")
    p.add_argument("--minutes", type=float, default=None,
                    help="required here or via --config's budget.minutes")
    p.add_argument("--nodes", default=None,
                    help="N = fixed node count; 'search' (DEFAULT) = measure the ANCHOR at each "
                         "legal node count, choose N by --node-objective, then tune at the "
                         "finalist N; 'auto' = uncalibrated efficiency proxy (--nodes-policy), "
                         "no measurement. (a --config budget.nodes also overrides the "
                         "default)")
    p.add_argument("--node-objective", choices=wtune.nodes.OBJECTIVES, default=None,
                    dest="node_objective",
                    help="--nodes search: time = fastest (smallest N within noise), cost = fewest "
                         "node-hours (within --deadline-minutes), knee = largest N with "
                         "efficiency >= --min-efficiency (default: time)")
    p.add_argument("--deadline-minutes", type=float, default=None, dest="deadline_minutes",
                    help="--node-objective cost: whole-run wall-time limit (minutes)")
    p.add_argument("--min-efficiency", type=float, default=None, dest="min_efficiency",
                    help="--node-objective knee: efficiency threshold (default %.2f)"
                         % wtune.nodes.DEFAULT_MIN_EFFICIENCY)
    p.add_argument("--sweep-budget-frac", type=float, default=None, dest="sweep_budget_frac",
                    help="--nodes search: share of --minutes the node sweep may use; once spent, "
                         "larger N are skipped ('sweep budget cap') so the finalists keep budget "
                         "(default %.1f, or --config's budget.sweep_budget_frac)"
                         % DEFAULT_SWEEP_BUDGET_FRAC)
    p.add_argument("--proxy", default=None, choices=("on", "off"), dest="proxy",
                    help="round0: time a 3x wider pool of layouts on one node each (small-domain "
                         "proxy, `wtune proxy`; a nested case per domain), re-time the 4 fastest as "
                         "3-node band proxies when N > 3, and send only the ANCHOR plus the 2 fastest "
                         "to the real N-node probes; needs netCDF wrfinput/wrfbdy, else round0 keeps "
                         "its real probes (default on, or --config's budget.proxy)")
    p.add_argument("--hotspot-fill", default=None, choices=("on", "off"), dest="hotspot_fill",
                    help="--nodes search: measure untried N near the best one in the nodes a "
                         "sweep wave leaves idle, when no longer than the wave (default on, or "
                         "--config's budget.hotspot_fill)")
    p.add_argument("--node-seeds", default=None, choices=("geometric", "all"), dest="node_seeds",
                    help="--nodes search: 'geometric' (DEFAULT) = 1, 2, 4, ... and max_nodes, "
                         "early stop upward, then refine; 'all' = every legal N, no early stop "
                         "(or --config's budget.node_seeds)")
    p.add_argument("--cores-per-node-options", default=None, dest="cores_per_node_options",
                    help="--nodes search: comma list of cores per node to measure at the finalist "
                         "node counts (the rest of the node idles). Default derived from the "
                         "hardware core count C: C, C-2, round(C*8/9), round(C*2/3), C/2 "
                         "(72 -> 72,70,64,48,36); layouts that are not legal are skipped. "
                         "'--cores-per-node-options 72' disables the dimension")
    p.add_argument("--pipeline", choices=("full",), default=None,
                    help="full = -O3 (kept for compatibility): one pass -- layout search (node, "
                         "cores/node, ranks, threads, decomposition, tiles), whole-run "
                         "prediction from rebuilt points, physics advice (decision 0059)")
    p.add_argument("--max-queue-hours", type=float, default=None, dest="max_queue_hours",
                    help="--pipeline full: safety cap on the accumulated queue-only wait (hours, "
                         "default %g); queue wait is not charged to --minutes, but past this the "
                         "run stops and reports" % PIPELINE_MAX_QUEUE_HOURS)
    p.add_argument("--no-space-check", action="store_true", dest="no_space_check",
                    help="submit even when converting the inputs to BP does not fit the space "
                         "writable at the BP cache (GPFS quota or free space)")
    p.add_argument("--no-node-preflight", action="store_true", dest="no_node_preflight",
                    help="skip the node health pre-flight (sinfo states + one 1-minute srun probe "
                         "per node checking $HOME, the case dir, wtune-src and wrf.exe); by "
                         "default failing nodes are excluded from every sbatch and max_nodes is "
                         "capped at the healthy node count")
    for lv in (0, 1, 2, 3):
        p.add_argument("-O%d" % lv, action="store_const", const=lv, dest="level",
                        help="effort level %d: %s (layout search effort + physics advice; the "
                             "prediction points come from the case type)" % (lv,
                                                                            LEVELS[lv]["method"]))
    p.add_argument("--precision", type=_precision_arg, default=None, dest="precision",
                    help="whole-run wall time needed to +-N%%%% of the run (e.g. 10%%%%) or +-MIN "
                         "minutes (e.g. 15); default %s, at least %g min: the case type decides "
                         "how many prediction points that takes "
                         "(wtune.cost_drivers.plan_points_for_case)"
                         % (PRECISION_DEFAULT.replace("%", "%%"), PRECISION_FLOOR_MIN))
    p.add_argument("--level", type=int, choices=(0, 1, 2, 3), default=None, dest="level",
                    help="same as -O<level>; selects the one-pass pipeline (default level when "
                         "only --pipeline full is given: 3). The estimated tuning time of every "
                         "level is printed before anything runs; --minutes is an optional hard cap")
    p.add_argument("--stop-if-short", action="store_true", dest="stop_if_short",
                    help="-O levels: when the whole run is shorter than the estimated tuning time, "
                         "step down to the highest level whose estimate fits (never below O1). "
                         "OFF by default: the layout search is not cut short")
    p.add_argument("--physics-advice", action="store_true", dest="physics_advice",
                    help="-O3: also run phase 3, the physics advice (opt-in, not part of the "
                         "default run)")
    p.add_argument("--keep-frames", action="store_true", dest="keep_frames",
                    help="--pipeline full: keep the phase-2/3 window-frame .nc files after the "
                         "fidelity gate (default: delete them, keeping the gate json/csv)")
    p.add_argument("--physics-max-candidates", type=int, default=None,
                    dest="physics_max_candidates",
                    help="--pipeline full: physics candidates tried in phase 3 (default %d)"
                         % PIPELINE_PHYSICS_MAX_CANDIDATES)
    p.add_argument("--node-finalists", type=int, default=None, dest="node_finalists",
                    help="--nodes search: how many node counts get the full round0/adaptive/"
                         "confirm search (default 2)")
    p.add_argument("--nodes-policy", choices=sorted(NODES_POLICIES), default=None,
                    dest="nodes_policy",
                    help="how --nodes auto picks a node count from the (uncalibrated) profile: "
                         "min=smallest legal, cap=largest legal within --max-nodes, "
                         "knee=stop doubling once marginal speedup drops below 1.5x "
                         "(default: 'knee', or --config's budget.nodes_policy)")
    p.add_argument("--max-nodes", type=int, default=None, dest="max_nodes",
                    help="default: %d (or --config's budget.max_nodes)" % wtune.machine.max_nodes())
    p.add_argument("--out", default=None)
    p.add_argument("--project", default=None, help="project dir (decision 0058): when given "
                   "(or via $WTUNE_PROJECT / a project.yaml above cwd) and --out is not, "
                   "defaults --out to <project>/tune/<id>/ instead of $HOME/wtune-runs/tune/"
                   "<case>. Does not otherwise change this command's behaviour.")
    p.add_argument("--steps", type=int, default=None,
                    help="default: %d (or --config's probe.steps)" % DEFAULT_STEPS)
    p.add_argument("--confirm-reps", type=int, default=None, dest="confirm_reps",
                    help="default: %d (or --config's probe.confirm_reps)" % CONFIRM_REPS)
    p.add_argument("--restart-io-form", type=int, choices=(14, 2), default=None,
                    dest="restart_io_form",
                    help="io_form of the point jobs' shared restart (RANKING_RESTART_IO_FORM); "
                         "default 14 (ADIOS2), or --config's probe.restart_io_form")
    p.add_argument("--max-points", type=int, default=None, dest="max_points",
                    help="probe.max_points override")
    p.add_argument("--points-by", choices=("time", "cost-indicators"), default=None,
                    dest="points_by",
                    help="probe.points_by override (decision 0060): place the prediction points by "
                         "sun/time (default) or by a weather cost proxy (wtune.cost_drivers; "
                         "sources from probe.indicator_sources)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    p.add_argument("--cores-per-node", type=int, default=wtune.machine.cores_per_node(),
                    dest="cores_per_node")
    p.add_argument("--hardware-id", default=wtune.machine.hardware_id(), dest="hardware_id")
    p.add_argument("--env-source", default=wtune.machine.env_source(), dest="env_source")
    p.add_argument("--sbatch-script", default=sweep.DEFAULT_SBATCH_SCRIPT, dest="sbatch_script")
    p.add_argument("--ledger-dir", default=None, dest="ledger_dir",
                   help="where the probes write their ledger CSVs and run directories (default: "
                        "<project>/probes in a project, else $WTUNE_CAMPAIGN_OUT or %s)"
                        % sweep.DEFAULT_LEDGER_DIR)
    p.add_argument("--sacct-cmd", default="sacct", dest="sacct_cmd")
    p.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S,
                    dest="poll_interval")
    return p


def main(argv: list[str] | None = None) -> int:
    """`wtune tune` / `python3 src/wtune/tune/`: 2 for a site without campaign/sites/<site>.sh,
    1 for an invalid option (message on stderr), else 0. With no level, pipeline or --nodes the
    run is the default -O2 layout search."""
    # line-buffered: under nohup stdout is a block-buffered file and the progress log stayed empty
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    args = build_arg_parser().parse_args(argv)
    args.site = args.site or wtune.machine.site()
    if not os.path.isfile(os.path.join(sweep._CAMPAIGN_DIR, "sites", args.site + ".sh")):
        print("ERROR: --site %s: there is no campaign/sites/%s.sh" % (args.site, args.site),
              file=sys.stderr)
        return 2
    os.environ[wtune.machine.ENV["site"]] = args.site
    if (getattr(args, "level", None) is None and getattr(args, "pipeline", None) is None
            and args.nodes is None):
        args.level = DEFAULT_LEVEL            # the default run: -O2
    os.environ.setdefault("SITE", args.site)
    try:
        run(args)
    except SystemExit as e:
        if e.code not in (0, None):
            print("ERROR: %s" % e, file=sys.stderr)
            return 1
        return 0
    return 0
