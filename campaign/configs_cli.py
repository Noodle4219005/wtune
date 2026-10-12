#!/usr/bin/env python3
"""The `configs.py <subcommand>` command line strict_campaign.sbatch and nc2bp.sbatch call.

Handlers only parse arguments, call the domain modules and print: the sbatch parses their
stdout with `read` / `eval`, so every printed byte is an interface. The case and template
tables come in as a Campaign (configs.py owns them). Stdlib only, Python 3.6.
"""
import argparse
import datetime
import json
import os
import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from bp_inputs import bp_cache_dir, link_plan_into_cache, resolve_bp_inputs
from case_tables import Case, load_generated_configs
from decomp_layout import (_GENERATED_TILE_FIELDS, build_order, config_extras_rows,
                           resolve_point_plan)
from namelist_files import format_namelist_stage, merge_iofields, stage_namelist_files
from namelist_text import case_start_datetime
from physics_templates import applicable_template_ids
from run_reports import post_spinup_steps, quota_free_gib, step_stats
from src_bridge import _import_wtune_bp, add_src_to_path


class Campaign(NamedTuple):
    """The tables configs.py read at import, and its functions bound to them."""
    cases: Dict[str, Case]
    case_order: Sequence[str]
    cost_guess: Dict[Tuple[int, int, int], Tuple[float, float]]
    physics_templates: Dict[str, Dict[str, Any]]
    physics_order: Sequence[str]
    resolve_case: Callable[[Case], tuple]                 # case -> (kept, skipped, aliases)
    resolve_physics_config: Callable[[str, str], tuple]   # (case id, template id)
    edit_namelist: Callable[..., str]


NODES = 4                       # the STRICT templates' allocation (f1: 4 x 112 cores)
COST_PER_NODE_HOUR = 22.4       # 112 cores x 0.2 per core-hour
SETUP_S_PER_RUN = 15.0          # staging + srun start around each run (a guess, like COST_GUESS)

_CASE_COLUMNS = ("config_id", "ranks", "omp", "nproc_x", "nproc_y", "numtiles", "dist",
                 "blocktime", "patch_x", "patch_y")

_COMPLIANCE_KNOBS = (
    "mpi_ranks", "OMP_NUM_THREADS", "nproc_x", "nproc_y",
    "numtiles", "srun_distribution", "KMP_BLOCKTIME",
)

USAGE = (
    "usage: configs.py CASE_ID | --all | order CASE_ID [--reps N] | "
    "edit-namelist IN OUT --nproc-x X --nproc-y Y --numtiles N "
    "--run-minutes M [--max-dom N] [--history-mode off|interval|end_d01] "
    "[--history-minutes N] [--set KEY=VALUE ...] [--delete-key KEY ...] "
    "[--nest-history-on] [--ranks N] [--no-adios2] "
    "[--numtiles-x N] [--numtiles-y N] [--tile-sz-x N] [--tile-sz-y N] "
    "[--tile-strategy N] [--physics-template Pxx] | "
    "step-stats STEPS_FILE | "
    "physics-list CASE_ID [--json] | "
    "quota-free-gib [--filesystem home1] (reads mmlsquota text on stdin) | "
    "link-namelist-files NAMELIST RUN_DIR SEARCH_DIR... | "
    "stage-bp-inputs CASE_ID NAMELIST SRC_DIR [--cache-dir DIR] [--check]\n"
    "CONFIG_FILE=path.json env (predictor/decomp_space.py to_campaign_configs/sample "
    "output): CASE_ID and order CASE_ID measure that generated config set instead of "
    "the fixed C01-C16 STRICT templates.\n"
)


def _shq(v: Any) -> str:
    """`v` as one single-quoted shell word."""
    return "'%s'" % str(v).replace("'", "'\\''")


def _tsv(row: Sequence[Any]) -> str:
    return "\t".join("" if v is None else str(v) for v in row)


def _enabled_case(cases: Dict[str, Case], case_id: str) -> Tuple[Optional[Case], int]:
    """(case, 0), or (None, exit code) after saying why on stderr: 2 unknown, 1 disabled."""
    case = cases.get(case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % case_id)
        return None, 2
    if not case.get("enabled", True):
        sys.stderr.write("case %s is disabled: %s\n"
                         % (case_id, case.get("reason", "(no reason recorded)")))
        return None, 1
    return case, 0


# --------------------------------------------------------------------------- STRICT table
def assert_strict_compliance() -> None:
    """Every knob the campaign varies must be STRICT (wtune.compliance, decision 0050)."""
    add_src_to_path()
    try:
        import wtune.compliance
    except Exception as e:
        sys.stderr.write(
            "WARNING: could not import wtune.compliance (%s: %s); skipping the "
            "STRICT-compliance assertion for this campaign's varied knobs\n"
            % (type(e).__name__, e)
        )
        return
    bad = [(n, wtune.compliance.classify(n)) for n in _COMPLIANCE_KNOBS]
    bad = [(n, c) for n, c in bad if c != wtune.compliance.STRICT]
    assert not bad, "campaign varies non-STRICT knob(s): %s" % (bad,)


def cmd_case(campaign: Campaign, case_id: str) -> int:
    """TSV of the case's kept STRICT configs; SKIP / ALIAS lines on stderr."""
    case, rc = _enabled_case(campaign.cases, case_id)
    if case is None:
        return rc
    kept, skipped, aliases = campaign.resolve_case(case)
    print("\t".join(_CASE_COLUMNS))
    for cfg in kept:
        print("\t".join(str(x) for x in (
            cfg["id"], cfg["ranks"], cfg["omp"], cfg["nx"], cfg["ny"], cfg["numtiles"],
            cfg["dist"], cfg["blocktime"], cfg["patch_x"], cfg["patch_y"],
        )))
    for tid, reason in skipped:
        sys.stderr.write("SKIP %s: %s\n" % (tid, reason))
    for kid, dropped in aliases.items():
        sys.stderr.write("ALIAS %s -> %s\n" % (kid, ",".join(dropped)))
    return 0


def cmd_case_generated(campaign: Campaign, case_id: str, config_file: str) -> int:
    """cmd_case for a CONFIG_FILE set: the same 10 columns + tile geometry + config_source,
    every config in file order (decomp_space.py already deduplicated and ordered it)."""
    case, rc = _enabled_case(campaign.cases, case_id)
    if case is None:
        return rc
    configs_list = load_generated_configs(config_file)
    print("\t".join(_CASE_COLUMNS + _GENERATED_TILE_FIELDS + ("config_source",)))
    for cfg in configs_list:
        row = [cfg.get("id"), cfg.get("ranks"), cfg.get("omp"), cfg.get("nx"), cfg.get("ny"),
               cfg.get("numtiles"), cfg.get("dist"), cfg.get("blocktime"),
               cfg.get("patch_x"), cfg.get("patch_y")]
        row += [cfg.get(k, 0) for k in _GENERATED_TILE_FIELDS]
        row.append(cfg.get("config_source", "generated"))
        print(_tsv(row))
    return 0


def cmd_config_extras(case_id: str, config_file: str) -> int:
    """TSV config_id, physics_namelist, window_frame per CONFIG_FILE config, in file order."""
    for row in config_extras_rows(load_generated_configs(config_file), case_id):
        print(_tsv(row))
    return 0


def cmd_order(campaign: Campaign, case_id: str, reps: Any,
              config_file: Optional[str] = None) -> int:
    """The run order (one config id per line) of the case's STRICT or generated set."""
    case = campaign.cases.get(case_id)
    if case is None or not case.get("enabled", True):
        sys.stderr.write("case %s unknown or disabled\n" % case_id)
        return 2
    if config_file:
        ids = [cfg["id"] for cfg in load_generated_configs(config_file)]
    else:
        kept, _skipped, _aliases = campaign.resolve_case(case)
        ids = [c["id"] for c in kept]
    for cid in build_order(ids, reps):
        print(cid)
    return 0


def _case_cost_row(campaign: Campaign, cid: str, reps: int = 2) -> Optional[Tuple[int, float, str]]:
    """(runs, node-hours, TSV row) of one case's STRICT campaign; None (said on stderr) when
    the case is missing, disabled or has no cost guess."""
    case = campaign.cases.get(cid)
    if case is None:
        sys.stderr.write("%s: missing from cases.json\n" % cid)
        return None
    if not case.get("enabled", True):
        sys.stderr.write("%s: DISABLED (%s)\n" % (cid, case.get("reason", "no reason recorded")))
        return None
    kept, skipped, _aliases = campaign.resolve_case(case)
    runs = len(kept) * reps
    key = (case["e_we"], case["e_sn"], case["e_vert"])
    if key not in campaign.cost_guess:
        sys.stderr.write("%s: no cost guess for grid %s\n" % (cid, key))
        return None
    s_per_step, init_io = campaign.cost_guess[key]
    per_run_s = init_io + case["steps"] * s_per_step
    wall_h = runs * (per_run_s + SETUP_S_PER_RUN) / 3600.0
    node_h = wall_h * NODES
    return runs, node_h, "%s\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.2f" % (
        cid, len(kept), len(skipped), reps, runs, wall_h, node_h, node_h * COST_PER_NODE_HOUR)


def cmd_all(campaign: Campaign) -> int:
    """Cost estimate of the whole STRICT campaign, one TSV row per case and a TOTAL line."""
    print("case\tn_configs\tn_skipped\treps\truns\test_wall_h\tnode_hours\tcost")
    grand_runs = 0
    grand_node_h = 0.0
    for cid in campaign.case_order:
        row = _case_cost_row(campaign, cid)
        if row is None:
            continue
        runs, node_h, line = row
        grand_runs += runs
        grand_node_h += node_h
        print(line)
    grand_wall_h = grand_node_h / float(NODES)
    print("TOTAL\truns=%d\test_wall_h=%.3f\tnode_hours=%.3f\tcost=%.2f  (GUESS: s_per_step/init_io "
          "are task-spec estimates, not measurements)"
          % (grand_runs, grand_wall_h, grand_node_h, grand_node_h * COST_PER_NODE_HOUR))
    return 0


def cmd_restart_time(campaign: Campaign, case_id: str, spinup_minutes: Any) -> int:
    """Print t0 + spinup_minutes as YYYY-MM-DD_HH:MM:SS: the shared restart's time, with t0
    read from the case's own namelist template (the file edit_namelist edits)."""
    case = campaign.cases.get(case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % case_id)
        return 2
    namelist_template = case.get("namelist_template") or (case["prep_dir"] + "/namelist.input")
    with open(namelist_template) as f:
        text = f.read()
    start = case_start_datetime(text)
    restart_time = start + datetime.timedelta(minutes=float(spinup_minutes))
    print(restart_time.strftime("%Y-%m-%d_%H:%M:%S"))
    return 0


# --------------------------------------------------------------------------- edit-namelist
# (flags, add_argument keywords) of `configs.py edit-namelist`, after IN and OUT
_EDIT_NAMELIST_ARGS = (
    (("--nproc-x",), dict(type=int, required=True)),
    (("--nproc-y",), dict(type=int, required=True)),
    (("--numtiles",), dict(type=int, required=True)),
    (("--run-minutes",), dict(type=int, required=True)),
    (("--max-dom",), dict(
        type=int, default=1,
        help="number of active domains; controls how many per-domain "
             "values are written for numtiles and history_interval*")),
    (("--history-mode",), dict(
        choices=["off", "interval", "end_d01", "mid_end_d01"], default="off",
        help="'off' (default, timing runs): no domain writes history. "
             "'interval' (PHYSICS-GRAY fidelity replicates): d01 writes a "
             "frame every --history-minutes, starting at t=0; requires "
             "--history-minutes. 'end_d01': d01 writes only its final "
             "frame (correctness runs); nested domains always get 0. "
             "'mid_end_d01': d01 writes two frames, at run_minutes//2 and at the "
             "end (window-frame fidelity-gate input).")),
    (("--history-minutes",), dict(
        type=int, default=None, help="required when --history-mode interval is given")),
    (("--physics-template",), dict(
        default=None, metavar="Pxx",
        help="apply a physics_templates.json template id (decision "
             "0051 PHYSICS-GRAY data campaign) on top of every other "
             "edit, including ADIOS2, e.g. --physics-template P02")),
    (("--physics-case",), dict(
        default=None, metavar="CASE",
        help="cases.json case id whose extra_edits (per-case paired "
             "edits) are applied with --physics-template")),
    (("--set",), dict(
        action="append", default=[], metavar="KEY=VALUE",
        help="repeatable; forces namelist scalar KEY to VALUE after "
             "every other edit except ADIOS2, e.g. --set max_dom=2 "
             "--set restart=.false.")),
    (("--delete-key",), dict(
        action="append", default=[], dest="delete_keys", metavar="KEY",
        help="repeatable; delete every line assigning this namelist key "
             "(case-insensitive, any group), e.g. --delete-key opt_run")),
    (("--nest-history-on",), dict(
        action="store_true",
        help="keep history-interval writes for nested domains (default: "
             "nested domains get 0, d01 gets the run length)")),
    (("--no-adios2",), dict(
        dest="adios2", action="store_false", default=True,
        help="disable ADIOS2 I/O settings (default: on). Legacy behaviour: "
             "io_form_history forced to 2, nothing else about I/O touched.")),
    (("--ranks",), dict(
        type=int, default=None,
        help="MPI rank count; required unless --no-adios2 is given "
             "(adios2_numaggregators MUST equal the rank count)")),
    (("--numtiles-x",), dict(
        type=int, default=None, metavar="N",
        help="&domains numtiles_x (WTune D1 decomposition-space work); "
             "not written at all when omitted")),
    (("--numtiles-y",), dict(
        type=int, default=None, metavar="N",
        help="&domains numtiles_y; not written at all when omitted")),
    (("--tile-sz-x",), dict(
        type=int, default=None, metavar="N",
        help="&domains tile_sz_x (overrides numtiles entirely, "
             "frame/module_tiles.F:353); not written at all when omitted")),
    (("--tile-sz-y",), dict(
        type=int, default=None, metavar="N",
        help="&domains tile_sz_y; not written at all when omitted")),
    (("--tile-strategy",), dict(
        type=int, default=None, metavar="N",
        help="&domains tile_strategy (TILE_NONE=0 TILE_X=1 TILE_Y=2 TILE_XY=3, "
             "module_machine.F:11); not written at all when omitted")),
    (("--restart-from",), dict(
        default=None, metavar="YYYY-MM-DD_HH:MM:SS",
        help="WTune probe multi-point (2026-09-26): this is a RANKING-PROBE run "
             "that starts from an existing WRF restart file at this time instead "
             "of the case's own t0 -- every domain's start_* is overridden to it "
             "and restart is forced to .true. (see edit_namelist's docstring). "
             "Not written at all when omitted (default: unchanged behaviour).")),
    (("--restart-io-form",), dict(
        type=int, default=None, metavar="N",
        help="overrides io_form_restart on EITHER a --restart-from run or a "
             "--write-restart-interval run, applied after --no-adios2/ADIOS2's own "
             "default (e.g. 2 for netCDF, decomposition-independent -- unlike "
             "ADIOS2's io_form 14). Not written at all when omitted.")),
    (("--write-restart-interval",), dict(
        type=int, default=None, metavar="MINUTES",
        help="the ANCHOR spin-up run that WRITES the shared restart file: sets "
             "restart_interval to exactly this value (restart itself is left "
             "'.false.' -- this run still starts at t0). Not written at all when "
             "omitted.")),
    (("--run-seconds",), dict(
        type=int, default=0, metavar="S",
        help="exact sub-minute tail added to --run-minutes (POINT_FUSED / "
             "WINDOW_EXACT: window = STEPS*dt, not rounded up to whole minutes). "
             "Only with --history-mode off. Not written when 0 (default).")),
)


def _edit_namelist_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="configs.py edit-namelist")
    ap.add_argument("infile")
    ap.add_argument("outfile")
    for flags, keywords in _EDIT_NAMELIST_ARGS:
        ap.add_argument(*flags, **keywords)
    return ap


def _parse_edit_namelist(argv: Sequence[str]) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """(parsed arguments, edit_namelist's keyword arguments); exits 2 on a bad combination."""
    ap = _edit_namelist_parser()
    a = ap.parse_args(argv)
    if a.run_seconds and a.history_mode != "off":
        ap.error("--run-seconds requires --history-mode off")
    if a.adios2 and a.ranks is None:
        ap.error("--ranks is required unless --no-adios2 is given")
    if a.history_mode == "interval" and a.history_minutes is None:
        ap.error("--history-minutes is required when --history-mode interval is given")
    restart_from = None
    if a.restart_from is not None:
        try:
            restart_from = datetime.datetime.strptime(a.restart_from, "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            ap.error("--restart-from must be YYYY-MM-DD_HH:MM:SS, got %r" % a.restart_from)
    overrides = {}
    for kv in a.set:
        key, sep, value = kv.partition("=")
        if not sep:
            ap.error("--set expects KEY=VALUE, got %r" % kv)
        overrides[key] = value
    return a, dict(
        max_dom=a.max_dom, overrides=overrides, delete_keys=a.delete_keys,
        nest_history_on=a.nest_history_on, history_mode=a.history_mode, adios2=a.adios2,
        ranks=a.ranks, history_minutes=a.history_minutes, physics_template=a.physics_template,
        physics_case=a.physics_case, numtiles_x=a.numtiles_x, numtiles_y=a.numtiles_y,
        tile_sz_x=a.tile_sz_x, tile_sz_y=a.tile_sz_y, tile_strategy=a.tile_strategy,
        restart_from=restart_from, restart_io_form=a.restart_io_form,
        write_restart_interval=a.write_restart_interval, run_seconds=a.run_seconds)


def cmd_edit_namelist(campaign: Campaign, argv: Sequence[str]) -> int:
    """edit-namelist IN OUT ...: write edit_namelist(IN's text) to OUT."""
    a, keywords = _parse_edit_namelist(argv)
    with open(a.infile) as f:
        text = f.read()
    new_text = campaign.edit_namelist(text, a.nproc_x, a.nproc_y, a.numtiles, a.run_minutes,
                                      **keywords)
    with open(a.outfile, "w") as f:
        f.write(new_text)
    return 0


# --------------------------------------------------------------------------- physics
def _describe_edits(summary: Sequence[Dict[str, Any]]) -> str:
    """'key: old -> new; ...' for physics-list's human-readable report."""
    edit_strs = []
    for e in summary:
        if e["op"] == "delete":
            edit_strs.append("%s: DELETED" % (e["key"],))
        else:
            edit_strs.append("%s: %s -> %s" % (
                e["key"], ",".join(e["old"]), ",".join(e["new"])))
    return "; ".join(edit_strs)


def cmd_physics_list(campaign: Campaign, argv: Sequence[str]) -> int:
    """physics-list CASE_ID [--json]: every template's edits or SKIP reason for the case."""
    ap = argparse.ArgumentParser(prog="configs.py physics-list")
    ap.add_argument("case_id")
    ap.add_argument("--json", action="store_true",
                     help="print applicable non-P00 template ids as a JSON list "
                          "(PHYSICS_TEMPLATE_ORDER order) instead of the "
                          "human-readable per-template report")
    a = ap.parse_args(argv)
    if a.case_id not in campaign.cases:
        sys.stderr.write("unknown case %r\n" % (a.case_id,))
        return 2
    if a.json:
        print(json.dumps(applicable_template_ids(a.case_id, campaign.physics_order,
                                                 campaign.resolve_physics_config)))
        return 0
    for tid in campaign.physics_order:
        family = campaign.physics_templates[tid].get("family", "-")
        summary, reason = campaign.resolve_physics_config(a.case_id, tid)
        if summary is None:
            print("%s (%s) SKIP: %s" % (tid, family, reason))
        elif not summary:
            print("%s (%s) APPLICABLE: (no edits -- reference template)" % (tid, family))
        else:
            print("%s (%s) APPLICABLE: %s" % (tid, family, _describe_edits(summary)))
    return 0


# --------------------------------------------------------------------------- run reports
def cmd_step_stats(argv: Sequence[str]) -> int:
    """step-stats STEPS_FILE (one step time per line): 'med= mean= n= excl= nall='."""
    ap = argparse.ArgumentParser(prog="configs.py step-stats")
    ap.add_argument("steps_file")
    a = ap.parse_args(argv)
    try:
        with open(a.steps_file) as f:
            values = [float(x) for x in f if x.strip()]
    except IOError:
        values = []
    stats = step_stats(values)

    def fmt(v):
        return "%.5f" % v if v is not None else ""

    print("med=%s mean=%s n=%d excl=%d nall=%d" % (
        fmt(stats["median"]), fmt(stats["mean_all"]), stats["n_measured"],
        stats["n_excluded"], stats["n_all"]))
    return 0


def cmd_fused_steps(argv: Sequence[str]) -> int:
    """fused-steps RSL_FILE --spin-seconds S --dt DT: the step times after the spin-up."""
    ap = argparse.ArgumentParser(prog="configs.py fused-steps")
    ap.add_argument("rsl_file")
    ap.add_argument("--spin-seconds", type=float, required=True)
    ap.add_argument("--dt", type=float, required=True)
    a = ap.parse_args(argv)
    try:
        with open(a.rsl_file, "rb") as f:
            text = f.read().decode("ascii", "ignore")
    except IOError:
        text = ""
    for v in post_spinup_steps(text, a.spin_seconds, a.dt):
        print("%s" % v)
    return 0


def cmd_point_plan(argv: Sequence[str]) -> int:
    """point-plan --fused 0|1 --anchor-end ''|0|1 [--window-frame 0|1] ID...: shell
    assignments POINT_ORDER=(...) POINT_FUSED_ON= POINT_ANCHOR_END= POINT_NOTE='..'."""
    ap = argparse.ArgumentParser(prog="configs.py point-plan")
    ap.add_argument("--fused", default="0")
    ap.add_argument("--anchor-end", default="")
    ap.add_argument("--window-frame", default="0")
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args(argv)
    ae = None if a.anchor_end == "" else (a.anchor_end != "0")
    r = resolve_point_plan(a.ids, a.fused == "1", ae, a.window_frame == "1")
    print("POINT_ORDER=(%s)" % " ".join("'%s'" % c for c in r["order"]))
    print("POINT_FUSED_ON=%d" % (1 if r["fused"] else 0))
    print("POINT_ANCHOR_END=%d" % (1 if r["anchor_end"] else 0))
    print("POINT_NOTE='%s'" % r["note"].replace("'", ""))
    return 0


def cmd_quota_free_gib(argv: Sequence[str]) -> int:
    """quota-free-gib [--filesystem FS] < mmlsquota text: free GiB ('%.3f'), or NA + rc 1."""
    ap = argparse.ArgumentParser(prog="configs.py quota-free-gib")
    ap.add_argument("--filesystem", default="home1")
    a = ap.parse_args(argv)
    text = sys.stdin.read()
    free_gib = quota_free_gib(text, filesystem=a.filesystem)
    if free_gib is None:
        print("NA")
        return 1
    print("%.3f" % free_gib)
    return 0


# --------------------------------------------------------------------------- staging
def cmd_link_namelist_files(argv: Sequence[str]) -> int:
    """link-namelist-files NAMELIST RUN_DIR SEARCH_DIR...: stage the files the final namelist
    names, then merge its iofields files. rc 0 even with a file missing (`wtune check` gates)."""
    if len(argv) < 2:
        sys.stderr.write("usage: configs.py link-namelist-files NAMELIST RUN_DIR SEARCH_DIR...\n")
        return 2
    with open(argv[0]) as f:
        text = f.read()
    res = stage_namelist_files(text, argv[1], argv[2:])
    for line in format_namelist_stage(res, argv[2:]) + merge_iofields(text, argv[1]):
        (sys.stderr if line.startswith("WARNING") else sys.stdout).write(line + "\n")
    return 0


def _print_stage_result(res: Dict[str, Any], cache: str, aux: Sequence[str], src_dir: str,
                        check: bool) -> None:
    """The shell assignments strict_campaign.sbatch / nc2bp.sbatch `eval`."""
    per_record = sorted(n for n, t in res["per_record"].items() if os.path.isdir(t))
    src = os.path.realpath(src_dir)
    print("BP_DIR=%s" % _shq(cache))
    print("WTUNE_AUX_BP_STREAMS=%s" % _shq(",".join(aux)))
    print("WTUNE_PER_RECORD_BP=%s" % _shq(" ".join(per_record)))
    print("BP_CONVERTED=%s" % _shq(" ".join(
        os.path.basename(p) for p in res["convert"]
        if os.path.realpath(os.path.dirname(p)) == src and not check)))


def _check_only_aux(res: Dict[str, Any], src_dir: str, err: Callable[[str], Any]
                    ) -> Optional[List[str]]:
    """--check: the aux streams already BP, or None when wrfinput/wrfbdy are not all BP."""
    for p in res["convert"]:
        err("WARNING: %s is netCDF only (no .bp next to it) -- its stream stays netCDF" % p)
    if {"input", "boundary"} & set(res["needs_convert"]):
        err("FATAL: --check: wrfinput/wrfbdy are not all BP in %s" % src_dir)
        return None
    return [s for s in res["ready"]
            if s not in ("input", "boundary") and s not in res["needs_convert"]]


def _stage_bp_inputs_locked(a: argparse.Namespace, text: str, cache: str,
                            err: Callable[[str], Any]) -> int:
    """stage-bp-inputs once the cache lock is held: resolve (convert unless --check), link
    into the cache, print the assignments. rc 1 without wrfinput/wrfbdy as BP."""
    res = resolve_bp_inputs(text, a.src_dir, cache, convert=not a.check, log=err)
    for stream, name in res["missing"]:
        err("WARNING: %s input %s found nowhere in %s (neither netCDF nor .bp)"
            % (stream, name, a.src_dir))
    if not {"input", "boundary"} <= set(res["ready"]):
        err("FATAL: wrfinput/wrfbdy not available as ADIOS2 inputs for %s" % a.case_id)
        return 1
    if a.check:
        aux = _check_only_aux(res, a.src_dir, err)
        if aux is None:
            return 1
    else:
        link_plan_into_cache(res, cache, err)
        aux = [s for s in res["ready"] if s not in ("input", "boundary")]
    for target in res["single_multirecord"]:
        err("WARNING: %s is a single multi-record BP: WRF 4.8's ADIOS2 reader returns its FIRST "
            "record for every boundary/SST/FDDA read -- convert the netCDF per record instead "
            "(wtune convert inputs)" % target)
    _print_stage_result(res, cache, aux, a.src_dir, a.check)
    return 0


def cmd_stage_bp_inputs(argv: Sequence[str]) -> int:
    """stage-bp-inputs CASE_ID NAMELIST SRC_DIR [--cache-dir DIR] [--check]: make DIR (default
    bp_cache_dir) hold every input of the namelist as BP, converting netCDF (not with --check).

    stdout: BP_DIR=, WTUNE_AUX_BP_STREAMS=, WTUNE_PER_RECORD_BP=, BP_CONVERTED= for `eval`;
    the log goes to stderr. rc 1 when wrfinput/wrfbdy resolve nowhere or conversion fails.
    """
    ap = argparse.ArgumentParser(prog="configs.py stage-bp-inputs")
    ap.add_argument("case_id")
    ap.add_argument("namelist")
    ap.add_argument("src_dir")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    with open(a.namelist) as f:
        text = f.read()
    cache = os.path.abspath(a.cache_dir or bp_cache_dir(a.case_id, a.src_dir))

    def err(message):
        return sys.stderr.write(message + "\n")

    lockf = None
    try:
        if not a.check:
            # probe jobs of one case start together: one converts, the others wait here
            # (lockf is NFS/GPFS-safe across nodes), then find the manifest current
            import fcntl
            # open_lock: survives the cross-node mkdir race (scc jobs 2884/2885)
            lockf = _import_wtune_bp().open_lock(os.path.join(cache, ".stage.lock"))
            fcntl.lockf(lockf, fcntl.LOCK_EX)
        return _stage_bp_inputs_locked(a, text, cache, err)
    except Exception as e:                     # ImportError/RuntimeError/OSError from the converter
        err("FATAL: ADIOS2 input staging failed for %s: %s: %s" % (a.case_id, type(e).__name__, e))
        return 1
    finally:
        if lockf is not None:
            lockf.close()                      # closing the fd releases the lockf lock


# --------------------------------------------------------------------------- dispatch
def _restart_time(campaign: Campaign, argv: Sequence[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: configs.py restart-time CASE_ID SPINUP_MINUTES\n")
        return 2
    return cmd_restart_time(campaign, argv[0], argv[1])


def _config_extras(campaign: Campaign, argv: Sequence[str]) -> int:
    config_file = os.environ.get("CONFIG_FILE")
    if len(argv) != 1 or not config_file:
        sys.stderr.write("usage: CONFIG_FILE=path.json configs.py config-extras CASE_ID\n")
        return 2
    return cmd_config_extras(argv[0], config_file)


def _order(campaign: Campaign, argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="configs.py order")
    ap.add_argument("case_id")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args(argv)
    return cmd_order(campaign, a.case_id, a.reps, config_file=os.environ.get("CONFIG_FILE"))


def _case_table(campaign: Campaign, case_id: str) -> int:
    config_file = os.environ.get("CONFIG_FILE")
    if config_file:
        return cmd_case_generated(campaign, case_id, config_file)
    return cmd_case(campaign, case_id)


Handler = Callable[[Campaign, Sequence[str]], int]


def _plain(handler: Callable[[Sequence[str]], int]) -> Handler:
    """A handler that needs none of the campaign's tables."""
    return lambda campaign, argv: handler(argv)


# subcommand -> handler(campaign, arguments after the subcommand)
COMMANDS: Dict[str, Handler] = {
    "edit-namelist": cmd_edit_namelist,
    "step-stats": _plain(cmd_step_stats),
    "fused-steps": _plain(cmd_fused_steps),
    "point-plan": _plain(cmd_point_plan),
    "quota-free-gib": _plain(cmd_quota_free_gib),
    "link-namelist-files": _plain(cmd_link_namelist_files),
    "stage-bp-inputs": _plain(cmd_stage_bp_inputs),
    "physics-list": cmd_physics_list,
    "restart-time": _restart_time,
    "config-extras": _config_extras,
}
# The commands that read the STRICT template table: the compliance assertion runs first.
STRICT_COMMANDS: Dict[str, Handler] = {
    "--all": lambda campaign, argv: cmd_all(campaign),
    "order": _order,
}


def main(campaign: Campaign, argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch `configs.py ARGS`; anything that is not a subcommand is a CASE_ID."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.stderr.write(USAGE)
        return 2
    handler = COMMANDS.get(argv[0])
    if handler is not None:
        return handler(campaign, argv[1:])
    assert_strict_compliance()
    handler = STRICT_COMMANDS.get(argv[0])
    if handler is not None:
        return handler(campaign, argv[1:])
    return _case_table(campaign, argv[0])
