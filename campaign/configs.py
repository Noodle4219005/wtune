#!/usr/bin/env python3
"""STRICT-configuration campaign generator (decision 0050): `python3 configs.py <cmd>` for
strict_campaign.sbatch / nc2bp.sbatch, and the `configs` module the predictor and tuner import.

The composition root: it reads the environment ONCE, at import (SITE, CASES_JSON,
WTUNE_TOTAL_CORES), binds the resulting tables to the pure functions of the modules next to
it, and re-exports every name. Stdlib only, Python 3.6 (f1's bare python3 runs it).
"""
# Python 3.6: no dataclasses, no walrus, no `from __future__ import annotations` -- here and in
# every module below. Layers, lowest first (a module imports only from the lines above it, and
# none imports this file):
#   namelist_text  case_tables  run_reports  src_bridge
#   decomp_layout  namelist_streams  physics_templates
#   namelist_adios2  namelist_files  bp_inputs
#   namelist_edit
#   configs_cli
import argparse  # noqa: F401 -- with json, math, re, statistics: names this module always had
import datetime
import json  # noqa: F401
import math  # noqa: F401
import os
import re  # noqa: F401
import statistics  # noqa: F401
import sys
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CAMPAIGN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CAMPAIGN_DIR)
if CAMPAIGN_DIR not in sys.path:        # the modules next to this file, however it was loaded
    sys.path.append(CAMPAIGN_DIR)

import bp_inputs as _bp  # noqa: E402
import case_tables as _case_tables  # noqa: E402
import configs_cli as _cli  # noqa: E402
import decomp_layout as _layout  # noqa: E402
import namelist_edit as _edit  # noqa: E402
import physics_templates as _physics  # noqa: E402
from bp_inputs import (  # noqa: E402,F401
    FRESH_COPY_PREFIX, _link_idempotent, _src_path, bp_cache_dir, resolve_bp_inputs)
from case_tables import (  # noqa: E402,F401
    CASE_ORDER, COST_GUESS, _active_domain_extents, _case_effective_max_dom,
    _case_namelist_path, case_frozen_keys, load_generated_configs)
from configs_cli import (  # noqa: E402,F401
    _COMPLIANCE_KNOBS, _shq, _stage_bp_inputs_locked,
    assert_strict_compliance, cmd_config_extras)
from configs_cli import cmd_fused_steps as _cmd_fused_steps  # noqa: E402,F401
from configs_cli import cmd_link_namelist_files as _cmd_link_namelist_files  # noqa: E402,F401
from configs_cli import cmd_point_plan as _cmd_point_plan  # noqa: E402,F401
from configs_cli import cmd_quota_free_gib as _cmd_quota_free_gib  # noqa: E402,F401
from configs_cli import cmd_stage_bp_inputs as _cmd_stage_bp_inputs  # noqa: E402,F401
from configs_cli import cmd_step_stats as _cmd_step_stats  # noqa: E402,F401
from decomp_layout import (  # noqa: E402,F401
    DECOMP_IDENTITY_FIELDS, MIN_PATCH, TEMPLATES, _GENERATED_TILE_FIELDS, _SHAPE_INDEX,
    _decomp_identity, _fallback_score, build_order, config_extras_rows, dedupe, describe_decomp,
    factor_pairs, legal_shapes, patch_dims, resolve_point_plan, window_frame_level,
    wrf_patch_legal)
from namelist_adios2 import (  # noqa: E402,F401
    ADIOS2_BLOSC_COMPRESSOR, ADIOS2_COMPRESSION_ENABLE, _apply_adios2,
    _per_record_netcdf_fallback)
from namelist_edit import (  # noqa: E402,F401
    HISTORY_NEVER_MINUTES, HISTORY_REQUIRING_DIAGNOSTICS, _apply_restart_from,
    _history_key_and_value, _history_requiring_diagnostics_on, _set_history_interval,
    _set_history_mode)
from namelist_files import (  # noqa: E402,F401
    format_namelist_stage, merge_iofields, stage_namelist_files)
from namelist_streams import (  # noqa: E402,F401
    ADIOS2_IO_FORM, PER_RECORD_STREAMS, WRF_DATE_FMT, _NML_REF_SKIP_AUXINPUT,
    _PER_RECORD_NAME_RE, adios2_input_streams, adios2_stream_edits, namelist_file_refs,
    per_record_has, per_record_streams)
from namelist_text import (  # noqa: E402,F401
    _NML_NEXT_KEY_RE, _delete_key_lines, _find_namelist_key, _get_scalar_first, _has_key,
    _insert_after_domains, _insert_after_section_header, _namelist_ints,
    _namelist_string_values, _replace_list_line, _set_namelist_value,
    _set_or_insert_time_control_scalar, _set_scalar, case_start_datetime)
from physics_templates import (  # noqa: E402,F401
    DEFAULT_SCREEN_DIR, PHYSICS_TEMPLATES_JSON_PATH, _LEDGER_PHYSICS_KEYS, _PROBE_VERDICT_RE,
    _set_values_for_domains, apply_physics_edit, apply_physics_template, format_physics_edits,
    load_physics_templates, parse_probe_verdicts, physics_ledger_columns, physics_tag_template,
    physics_template_applicability, probe_verdicts_for_case, template_frozen_key_conflicts)
from run_reports import (  # noqa: E402,F401
    FIDELITY_TOKEN_NA, _TIMING_D01_RE, describe_field, fidelity_max_over_fields,
    fidelity_notes_tokens, post_spinup_steps, quota_free_gib, step_stats)
from src_bridge import _import_wtune_bp, add_src_to_path  # noqa: E402,F401

Case = Dict[str, Any]
Config = Dict[str, Any]

# --------------------------------------------------------------------------- import-time state
# CASES_JSON: an alternate case file with the same {"cases": {...}} schema (a smoke run).
CASES_JSON_PATH = os.environ.get("CASES_JSON") or os.path.join(CAMPAIGN_DIR, "cases.json")
# SITE (decision 0050 step 8): anything but "f1" overlays campaign/cases.<site>.json.
SITE = os.environ.get("SITE") or "f1"
# f1: 4 nodes x 112 cores. TEMPLATES is anchored at 448 and NOT rescaled: with another total
# every template is skipped, and the caller generates a set (predictor/decomp_space.py).
TOTAL_CORES = int(os.environ.get("WTUNE_TOTAL_CORES", 448))


def load_cases(path: str = CASES_JSON_PATH, site: Optional[str] = None) -> Dict[str, Case]:
    """cases.json, overlaid by campaign/cases.<site>.json; site None = this process's SITE."""
    return _case_tables.load_cases(path, SITE if site is None else site, CAMPAIGN_DIR)


CASES = load_cases()
PHYSICS_TEMPLATES = load_physics_templates()
PHYSICS_TEMPLATE_ORDER = sorted(PHYSICS_TEMPLATES)      # ids are zero-padded: P00 < P01 < ...

_DECOMP = None
_DECOMP_TRIED = False


def _load_decomp() -> Optional[ModuleType]:
    """src/wtune_decomp (it needs Python >= 3.7) or None; tried once, a failure is cached too."""
    global _DECOMP, _DECOMP_TRIED
    if _DECOMP_TRIED:
        return _DECOMP
    _DECOMP_TRIED = True
    add_src_to_path()
    try:
        import wtune_decomp
        _DECOMP = wtune_decomp
    except Exception as e:
        sys.stderr.write(
            "NOTE: wtune_decomp unavailable (%s: %s); using fallback shape scorer\n"
            % (type(e).__name__, e)
        )
        _DECOMP = None
    return _DECOMP


# --------------------------------------------------------------------------- bound functions
def ranked_shapes(e_we: int, e_sn: int, e_vert: int, ranks: int, omp: int,
                  nodes: int = 4) -> Tuple[List[Tuple[int, int]], str]:
    """(legal (nx, ny) pairs best first, 'wtune_decomp' | 'fallback')."""
    return _layout.ranked_shapes(e_we, e_sn, e_vert, ranks, omp, nodes, load_decomp=_load_decomp)


def resolve_config(case: Case, template: Config) -> Tuple[Optional[Config], Optional[str]]:
    """One template on one case -> (config, None) or (None, reason); see decomp_layout."""
    return _layout.resolve_config(case, template, TOTAL_CORES, _load_decomp)


def resolve_case(case: Case) -> Tuple[List[Config], List[Tuple[str, str]], Dict[str, List[str]]]:
    """Every TEMPLATES row on one case -> (kept, skipped, aliases)."""
    return _layout.resolve_case(case, TEMPLATES, TOTAL_CORES, _load_decomp)


def available_decomps(case_id: str, config_file: Optional[str] = None,
                      cases: Optional[Dict[str, Case]] = None) -> tuple:
    """(ordered ids, {id: config}, None) or (None, None, reason); see decomp_layout."""
    return _layout.available_decomps(case_id, CASES if cases is None else cases, resolve_case,
                                     config_file)


def resolve_decomp_pair(case_id: str, decomp_id: Any, twin_id: Any,
                        config_file: Optional[str] = None,
                        cases: Optional[Dict[str, Case]] = None) -> tuple:
    """MODE=physics's (reference, twin, None) or (None, None, reason); see decomp_layout."""
    return _layout.resolve_decomp_pair(case_id, decomp_id, twin_id,
                                       CASES if cases is None else cases, resolve_case,
                                       config_file)


def resolve_physics_config(case_id: str, template_id: str,
                           cases: Optional[Dict[str, Case]] = None,
                           templates: Optional[Dict[str, Any]] = None) -> tuple:
    """(edit summary, None) or (None, reason) of one template on a case's own namelist."""
    return _physics.resolve_physics_config(
        case_id, template_id, CASES if cases is None else cases,
        PHYSICS_TEMPLATES if templates is None else templates)


def edit_namelist(text: str, nproc_x: int, nproc_y: int, numtiles: int, run_minutes: int,
                  max_dom: int = 1, overrides: Optional[Mapping[str, Any]] = None,
                  delete_keys: Optional[Iterable[str]] = None, nest_history_on: bool = False,
                  history_mode: str = "off", adios2: bool = True, ranks: Optional[int] = None,
                  history_minutes: Optional[int] = None, physics_template: Optional[str] = None,
                  physics_case: Optional[str] = None,
                  numtiles_x: Optional[int] = None, numtiles_y: Optional[int] = None,
                  tile_sz_x: Optional[int] = None, tile_sz_y: Optional[int] = None,
                  tile_strategy: Optional[int] = None,
                  restart_from: Optional[datetime.datetime] = None,
                  restart_io_form: Optional[int] = None,
                  write_restart_interval: Optional[int] = None,
                  run_seconds: Optional[int] = None) -> str:
    """Apply every campaign edit to a namelist text; physics_template is a PHYSICS_TEMPLATES
    id. The steps and their order are namelist_edit.edit_namelist's."""
    return _edit.edit_namelist(
        text, nproc_x, nproc_y, numtiles, run_minutes, max_dom=max_dom, overrides=overrides,
        delete_keys=delete_keys, nest_history_on=nest_history_on, history_mode=history_mode,
        adios2=adios2, ranks=ranks, history_minutes=history_minutes,
        physics_template=physics_template, physics_case=physics_case, numtiles_x=numtiles_x,
        numtiles_y=numtiles_y, tile_sz_x=tile_sz_x, tile_sz_y=tile_sz_y,
        tile_strategy=tile_strategy, restart_from=restart_from, restart_io_form=restart_io_form,
        write_restart_interval=write_restart_interval, run_seconds=run_seconds,
        physics_templates=PHYSICS_TEMPLATES)


def _case_lookup(cases_json: Optional[str], site: Optional[str]) -> _bp.CaseLookup:
    """case id -> case of `cases_json` (default: this process's case file) for `site`."""
    return lambda case_id: load_cases(cases_json or CASES_JSON_PATH, site)[case_id]


def _case_bp_plan(case_id: str, cases_json: Optional[str] = None, site: str = "scc",
                  bp: Any = None) -> Tuple[Dict[str, Any], str, List[str]]:
    """(BP input plan, cache dir, stale names) of a case, read-only. KeyError: no such case."""
    return _bp._case_bp_plan(case_id, _case_lookup(cases_json, site), bp)


def bp_cache_status(case_id: str, cases_json: Optional[str] = None, site: str = "scc",
                    bp: Any = None) -> Tuple[str, str]:
    """("current" | "stale" | "none", detail) of a case's BP input cache."""
    return _bp.bp_cache_status(case_id, _case_lookup(cases_json, site), bp)


def per_record_sources(case_id: str, cases_json: Optional[str] = None, site: str = "scc",
                       bp: Any = None) -> Tuple[Dict[str, str], Optional[str]]:
    """({name: per-record dir} a probe of the case reads, None) or ({}, reason)."""
    return _bp.per_record_sources(case_id, _case_lookup(cases_json, site), bp)


def fresh_probe_inputs(case_id: str, dest_dir: str, copies: Any,
                       cases_json: Optional[str] = None, site: str = "scc",
                       bp: Any = None) -> Optional[str]:
    """dest_dir holding `copies` fresh per-record input copies, or None (WARNING on stderr)."""
    return _bp.fresh_probe_inputs(case_id, dest_dir, copies, _case_lookup(cases_json, site), bp)


# --------------------------------------------------------------------------- command line
def _campaign() -> _cli.Campaign:
    """This module's tables and bound functions as they are NOW."""
    return _cli.Campaign(
        cases=CASES, case_order=CASE_ORDER, cost_guess=COST_GUESS,
        physics_templates=PHYSICS_TEMPLATES, physics_order=PHYSICS_TEMPLATE_ORDER,
        resolve_case=resolve_case,
        resolve_physics_config=resolve_physics_config, edit_namelist=edit_namelist)


def cmd_case(case_id: str) -> int:
    return _cli.cmd_case(_campaign(), case_id)


def cmd_case_generated(case_id: str, config_file: str) -> int:
    return _cli.cmd_case_generated(_campaign(), case_id, config_file)


def cmd_restart_time(case_id: str, spinup_minutes: Any) -> int:
    return _cli.cmd_restart_time(_campaign(), case_id, spinup_minutes)


def cmd_order(case_id: str, reps: Any, config_file: Optional[str] = None) -> int:
    return _cli.cmd_order(_campaign(), case_id, reps, config_file=config_file)


def cmd_all() -> int:
    return _cli.cmd_all(_campaign())


def _cmd_edit_namelist(argv: Sequence[str]) -> int:
    return _cli.cmd_edit_namelist(_campaign(), argv)


def cmd_physics_list(argv: Sequence[str]) -> int:
    return _cli.cmd_physics_list(_campaign(), argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`configs.py ARGS`: see configs_cli.USAGE and its dispatch table."""
    return _cli.main(_campaign(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
