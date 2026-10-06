#!/usr/bin/env python3
"""The campaign's ADIOS2 I/O edits of a WRF namelist: io_form 14 on every stream, BP input
names, per-record inputs, the &namelist_adios2 group, quilting off.

What is available as BP comes from the environment strict_campaign.sbatch exports after
`configs.py stage-bp-inputs` (WTUNE_*); the edit itself is a pure function of (text, inputs).
Stdlib only, Python 3.6.
"""
import datetime
import os
import re
import sys
from typing import Dict, List, Mapping, NamedTuple, Optional, Set

from namelist_streams import (ADIOS2_IO_FORM, PER_RECORD_STREAMS, adios2_input_streams,
                              adios2_stream_edits, per_record_streams)
from namelist_text import (_delete_key_lines, _has_key, _header_eol, _set_namelist_value,
                           _set_or_insert_time_control_scalar)
from src_bridge import _import_wtune_bp

# Written into every run's &namelist_adios2 instead of inheriting the Registry defaults
# (.true. / "lz4"): zstd writes faster and 8% smaller than lz4 (jobs 1037956-64, 1082176).
# This WRF build compresses only a run's FIRST write stream (wrf_io.F90:1283, job 1082176);
# tools/bp_compression_check.py detects it.
ADIOS2_COMPRESSION_ENABLE = ".true."
ADIOS2_BLOSC_COMPRESSOR = "zstd"

RESTART_NEVER_MINUTES = 1000000     # restart_interval beyond any campaign run: none written


class BpInputs(NamedTuple):
    """Which inputs a run may read as ADIOS2 BP (strict_campaign.sbatch's WTUNE_* exports)."""
    netcdf_inputs: bool             # WTUNE_INPUT_FORMAT=netcdf: the legacy escape
    aux_ok: Set[str]                # aux streams whose files exist as <name>.bp
    per_record: str                 # input names held one BP per record
    bp_dir: Optional[str]           # where those record directories are


def bp_inputs_from_env(environ: Mapping[str, str]) -> BpInputs:
    return BpInputs(
        netcdf_inputs=environ.get("WTUNE_INPUT_FORMAT", "adios2") == "netcdf",
        aux_ok=set(x for x in environ.get("WTUNE_AUX_BP_STREAMS", "").split(",") if x),
        per_record=environ.get("WTUNE_PER_RECORD_BP", ""),
        bp_dir=environ.get("WTUNE_BP_DIR") or None)


def _force_io_forms(text: str, netcdf_inputs: bool) -> str:
    """io_form 14 for history/restart (and the inputs unless netCDF), no restart writes, and
    the old input_inname / bdy_inname lines dropped."""
    for key in ("io_form_input", "io_form_boundary", "io_form_history", "io_form_restart"):
        value = 2 if netcdf_inputs and key in ("io_form_input", "io_form_boundary") else 14
        text = _set_or_insert_time_control_scalar(text, key, value)
    text = _set_or_insert_time_control_scalar(text, "restart_interval", RESTART_NEVER_MINUTES)
    text = _delete_key_lines(text, "input_inname")
    return _delete_key_lines(text, "bdy_inname")


def _switch_aux_streams(text: str, aux_ok: Set[str], per_record: Set[str]) -> str:
    """Aux streams available as <name>.bp read ADIOS2; the others keep the case's settings."""
    for st in adios2_input_streams(text):
        name = st["stream"]
        if name not in aux_ok or name in ("input", "boundary") or name in per_record:
            continue
        for key, value, group in adios2_stream_edits(st):
            text = _set_or_insert_time_control_scalar(text, key, value, group)
    return text


def _read_per_record(text: str, streams: Dict[str, List[str]], max_dom: int) -> str:
    """Per-record streams: wtune_bp's dated innames / multi_bdy_files, and io_form 14."""
    if not streams:
        return text
    names = [n for s in streams.values() for n in s]
    edits = _import_wtune_bp().per_record_namelist_edits(names, max_dom)
    for group, kv in edits.items():
        for key, value in kv.items():
            text = _set_namelist_value(text, key, value, group)
    for stream in streams:
        io_key, group = PER_RECORD_STREAMS[stream][1:3]
        text = _set_namelist_value(text, io_key, str(ADIOS2_IO_FORM), group)
    return text


def _per_record_netcdf_fallback(text: str, stream: str, max_dom: int) -> str:
    """Make `stream` read its netCDF file: io_form 2, the default inname, and a previous
    pass's per-record keys undone (multi_bdy_files .false., frames_per_auxinputN 999999)."""
    _kind, io_key, group, inname_key, default, frames_key = PER_RECORD_STREAMS[stream]
    text = _set_namelist_value(text, io_key, "2", group)
    if stream == "boundary":
        text = _delete_key_lines(text, inname_key)
        if _has_key(text, "multi_bdy_files"):
            text = _set_namelist_value(text, "multi_bdy_files", ".false.", "bdy_control")
    else:
        text = _set_namelist_value(text, inname_key, '"%s"' % default, group)
        if _has_key(text, frames_key):
            text = _set_namelist_value(text, frames_key, ", ".join(["999999"] * max_dom),
                                       "time_control")
    return text


def _switch_inputs(text: str, eol: int, max_dom: int, start: Optional[datetime.datetime],
                   inputs: BpInputs) -> str:
    """Point every input stream at its BP file; `eol` ends the &time_control header line.

    WRF 4.8's ADIOS2 reader never selects a BP step, so multi-record inputs are one BP per
    record, named by the CLOCK TIME of the read: a run starting off a record time reads that
    stream from netCDF instead (a NOTE on stderr). Idempotent over a previous pass.
    """
    pr_ok, pr_fallback = per_record_streams(inputs.per_record, start, text, inputs.bp_dir,
                                            aux_ok=inputs.aux_ok)
    new_line = ' input_inname="wrfinput_d<domain>.bp",'
    if "boundary" not in pr_ok and "boundary" not in pr_fallback:
        new_line += ' bdy_inname="wrfbdy_d<domain>.bp",'
    text = text[: eol + 1] + new_line + "\n" + text[eol + 1 :]
    text = _switch_aux_streams(text, inputs.aux_ok, set(pr_ok) | set(pr_fallback))
    text = _read_per_record(text, pr_ok, max_dom)
    for stream, missing in sorted(pr_fallback.items()):
        text = _per_record_netcdf_fallback(text, stream, max_dom)
        sys.stderr.write("NOTE: ADIOS2 per-record %s input has no record at the run start "
                         "(%s) -- this run reads that stream from netCDF instead\n"
                         % (stream, ", ".join(missing)))
    return text


def _write_adios2_group(text: str, ranks: int) -> str:
    """Replace &namelist_adios2. adios2_numaggregators MUST equal the rank count: WRF's
    default with --distribution=cyclic wrote BP that cannot be read back (job 1072441)."""
    text = re.sub(r"(?ms)^&namelist_adios2\b.*?^/[ \t]*\n?", "", text)
    return text.rstrip("\n") + (
        "\n&namelist_adios2\n"
        " adios2_numaggregators=%d,\n"
        " adios2_compression_enable=%s,\n"
        ' adios2_blosc_compressor="%s",\n'
        "/\n" % (ranks, ADIOS2_COMPRESSION_ENABLE, ADIOS2_BLOSC_COMPRESSOR))


def _quilting_off(text: str) -> str:
    """Replace &namelist_quilt: I/O server ranks break nproc_x*nproc_y and hang WRF (LAFE
    template, smoke job 1074562)."""
    text = re.sub(r"(?msi)^[ \t]*&namelist_quilt\b.*?^[ \t]*/[ \t]*\n?", "", text)
    return text.rstrip("\n") + "\n&namelist_quilt\n nio_tasks_per_group=0,\n nio_groups=1,\n/\n"


def apply_adios2(text: str, ranks: Optional[int], max_dom: int,
                 start: Optional[datetime.datetime], inputs: BpInputs) -> str:
    """Force ADIOS2 I/O (2026-09-14 decision; recipe of regress_bp.sbatch, job 1072441).

    start: when the run starts, for per-record inputs (default: the namelist's own start).
    """
    if ranks is None:
        raise ValueError("ranks is required when adios2=True (for adios2_numaggregators)")
    text = _force_io_forms(text, inputs.netcdf_inputs)
    eol = _header_eol(text, "&time_control")
    if not inputs.netcdf_inputs:
        text = _switch_inputs(text, eol, max_dom, start, inputs)
    return _quilting_off(_write_adios2_group(text, ranks))


def _apply_adios2(text: str, ranks: Optional[int], max_dom: int,
                  start: Optional[datetime.datetime] = None) -> str:
    """apply_adios2 with the inputs the WTUNE_* environment declares."""
    return apply_adios2(text, ranks, max_dom, start, bp_inputs_from_env(os.environ))
