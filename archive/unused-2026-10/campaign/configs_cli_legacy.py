"""configs.py subcommands only the removed MODE=physics block of strict_campaign.sbatch called:
decomp-pair, physics-order, physics-describe, fidelity-tokens. Verbatim as at 30572ce; not
wired to anything, not importable as is."""

# ---- from campaign/configs_cli.py
# role + the columns cmd_case_generated prints, minus patch_x/patch_y (unused after the pair)
_DECOMP_PAIR_COLUMNS = ("role", "config_id", "ranks", "omp", "nproc_x", "nproc_y", "numtiles",
                        "dist", "blocktime") + _GENERATED_TILE_FIELDS + ("config_source",)


# ---- from campaign/configs_cli.py
def cmd_decomp_pair(campaign: Campaign, argv: Sequence[str]) -> int:
    """Validate and print MODE=physics's reference/twin pair: rc 0 + a 2-row TSV, or rc 2 +
    one 'DECOMP-PAIR ERROR: <reason>' line. The sbatch's fail-fast gate before any srun."""
    ap = argparse.ArgumentParser(prog="configs.py decomp-pair")
    ap.add_argument("case_id")
    ap.add_argument("decomp")
    ap.add_argument("twin_decomp")
    ap.add_argument("--config-file", default=None,
                    help="generated config set (predictor/decomp_space.py sample|seed output); "
                         "default: the CONFIG_FILE env var, and unset means the STRICT templates")
    a = ap.parse_args(argv)
    config_file = a.config_file or os.environ.get("CONFIG_FILE")
    ref, twin, reason = campaign.resolve_decomp_pair(a.case_id, a.decomp, a.twin_decomp,
                                                     config_file)
    if reason:
        sys.stderr.write("DECOMP-PAIR ERROR: %s\n" % reason)
        return 2
    print("\t".join(_DECOMP_PAIR_COLUMNS))
    for role, cfg in (("reference", ref), ("twin", twin)):
        row = [role, cfg.get("id"), cfg.get("ranks"), cfg.get("omp"), cfg.get("nx"), cfg.get("ny"),
               cfg.get("numtiles"), cfg.get("dist"), cfg.get("blocktime")]
        row += [cfg.get(k, 0) for k in _GENERATED_TILE_FIELDS]
        row.append(cfg.get("config_source", "strict_template"))
        print(_tsv(row))
    return 0


# ---- from campaign/configs_cli.py
def cmd_physics_order(campaign: Campaign, argv: Sequence[str]) -> int:
    """physics-order CASE_ID: the run tags (P00_ref, P00_twin, templates) in run order."""
    ap = argparse.ArgumentParser(prog="configs.py physics-order")
    ap.add_argument("case_id")
    ap.add_argument("--pconfigs", default=None,
                     help="'+'-separated physics template ids (default: every "
                          "template applicable to this case, in "
                          "PHYSICS_TEMPLATE_ORDER, excluding P00)")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args(argv)
    if a.case_id not in campaign.cases:
        sys.stderr.write("unknown case %r\n" % (a.case_id,))
        return 2
    if a.pconfigs:
        pconfigs = [p for p in a.pconfigs.split("+") if p]
    else:
        pconfigs = applicable_template_ids(a.case_id, campaign.physics_order,
                                           campaign.resolve_physics_config)
    for tag in build_order(["P00_ref", "P00_twin"] + pconfigs, a.reps):
        print(tag)
    return 0


# ---- from campaign/configs_cli.py
def cmd_physics_describe(campaign: Campaign, argv: Sequence[str]) -> int:
    """physics-describe CASE_ID TEMPLATE_ID: one TSV line for the pair's ledger row --
    family, edits, probe, time_step and the nine physics columns -- or 'SKIP<TAB>reason'
    when the template does not apply (callers check field 1 first)."""
    ap = argparse.ArgumentParser(prog="configs.py physics-describe")
    ap.add_argument("case_id")
    ap.add_argument("template_id")
    ap.add_argument("--screen-dir", default=DEFAULT_SCREEN_DIR)
    a = ap.parse_args(argv)
    case = campaign.cases.get(a.case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % (a.case_id,))
        return 2
    template = campaign.physics_templates.get(a.template_id)
    if template is None:
        sys.stderr.write("unknown physics template %r\n" % (a.template_id,))
        return 2
    summary, reason = campaign.resolve_physics_config(a.case_id, a.template_id)
    if summary is None:
        print("SKIP\t%s" % (reason,))
        return 0
    family = template.get("family", "-")
    edits_str = format_physics_edits(summary)
    edited_keys = [e["key"] for e in summary if e["op"] != "delete"]
    probe_str = probe_verdicts_for_case(a.case_id, edited_keys, screen_dir=a.screen_dir)
    cols = physics_ledger_columns(case, summary)
    print("\t".join(describe_field(x) for x in [
        family, edits_str, probe_str,
        cols["time_step"], cols["mp_physics"], cols["cu_physics"],
        cols["bl_pbl_physics"], cols["sf_sfclay_physics"], cols["sf_surface_physics"],
        cols["ra_lw_physics"], cols["ra_sw_physics"], cols["radt"], cols["cudt"],
    ]))
    return 0


# ---- from campaign/configs_cli.py
def cmd_fidelity_tokens(argv: Sequence[str]) -> int:
    """fidelity-tokens FIDELITY_JSON: the line the sbatch reads into FID_LABEL FID_ND
    FID_FLOOR FID_DECIDING; rc 2 and no output when the JSON is missing or unreadable."""
    ap = argparse.ArgumentParser(prog="configs.py fidelity-tokens")
    ap.add_argument("fidelity_json")
    a = ap.parse_args(argv)
    try:
        with open(a.fidelity_json) as f:
            payload = json.load(f)
    except (IOError, OSError, ValueError) as e:
        sys.stderr.write("cannot read fidelity JSON %s: %s\n" % (a.fidelity_json, e))
        return 2
    print(" ".join(fidelity_notes_tokens(payload)))
    return 0


# ---- from campaign/configs.py
def cmd_decomp_pair(argv: Sequence[str]) -> int:
    return _cli.cmd_decomp_pair(_campaign(), argv)


# ---- from campaign/configs.py
def cmd_physics_order(argv: Sequence[str]) -> int:
    return _cli.cmd_physics_order(_campaign(), argv)


# ---- from campaign/configs.py
def cmd_physics_describe(argv: Sequence[str]) -> int:
    return _cli.cmd_physics_describe(_campaign(), argv)


# ---- from campaign/configs_cli.py: USAGE lines, dispatch entries, Campaign field
#     "decomp-pair CASE_ID DECOMP TWIN_DECOMP [--config-file FILE.json] | "
#     "physics-order CASE_ID [--pconfigs P01+P02+...] [--reps N] | "
#     "physics-describe CASE_ID TEMPLATE_ID | "
#     "fidelity-tokens FIDELITY_JSON | "
# COMMANDS:        "physics-order": cmd_physics_order, "physics-describe": cmd_physics_describe,
#                  "fidelity-tokens": _plain(cmd_fidelity_tokens),
# STRICT_COMMANDS: "decomp-pair": cmd_decomp_pair,
# Campaign:        resolve_decomp_pair: Callable[..., tuple]   # (case, decomp, twin, config_file)
