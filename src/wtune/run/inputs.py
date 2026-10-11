"""ADIOS2 inputs of a run (--io adios2, the default since 2026-10-03): which streams switch
to io_form 14, where their BP copies live (<project>/input/bp, else OUT/_bp_inputs), which
are one BP per record (WRF 4.8's reader returns record 1 of a multi-record BP), and the
conversion that happens once before segment 0 is submitted."""
from __future__ import annotations

import dataclasses
import os
from typing import Callable, Optional

from ._deps import sweep
from .common import ADIOS2_IO_FORM, RunError, _PER_RECORD_KIND_RE
from .geometry import Geometry
from .plan import RecordAlignment, input_record_alignment, missing_record_files


@dataclasses.dataclass(frozen=True)
class InputPlan:
    """How a run reads its inputs; the defaults are --io netcdf (the case's own io_form_*)."""
    io_streams: Optional[list] = None          # configs.adios2_input_streams entries switched
    bp_plan: Optional[dict] = None             # configs.resolve_bp_inputs' result
    bp_cache: Optional[str] = None
    per_record: list = dataclasses.field(default_factory=list)   # sorted input names
    per_record_edits: Optional[dict] = None    # {group: {key: literal}} namelist edits
    align: Optional[RecordAlignment] = None
    bp_xml: Optional[str] = None               # wtune_bp.adios2_xml(), once converted

    @property
    def links(self) -> Optional[dict]:
        """{run-dir link name: BP} of every resolved input."""
        return self.bp_plan["links"] if self.bp_plan else None

    @property
    def fresh_sources(self) -> Optional[dict]:
        """{name: per-record container} copied afresh for every attempt."""
        return {n: self.bp_plan["per_record"][n] for n in self.per_record} or None


def find_project_dir(project: Optional[str] = None) -> Optional[str]:
    """The project dir (--project, $WTUNE_PROJECT, a project.yaml above cwd) or None."""
    import wtune.project
    try:
        return wtune.project.find_project(explicit=project)
    except wtune.project.ProjectError as e:
        raise RunError(str(e))


def bp_cache_dir(out: str, project: Optional[str] = None) -> str:
    """BP input cache: <project>/input/bp/ when a project resolves (decision 0058), else
    OUT/_bp_inputs/."""
    project_dir = find_project_dir(project)
    if project_dir:
        return os.path.join(os.path.abspath(project_dir), "input", "bp")
    return os.path.join(out, "_bp_inputs")


def adios2_streams_or_raise(bp_plan: dict, case_dir: str, notes: list) -> list:
    """Streams a run switches to ADIOS2 (configs.resolve_bp_inputs' plan): every ready one;
    wrfinput/wrfbdy must be ready (RunError otherwise); an aux stream with a missing file keeps
    the case's own settings, with a note."""
    missing: dict[str, list[str]] = {}
    for stream, name in bp_plan["missing"]:
        missing.setdefault(stream, []).append(name)
    for stream in ("input", "boundary"):
        if stream in missing:
            raise RunError("ADIOS2 I/O: %s not found in %s (neither netCDF nor .bp) -- the "
                           "legacy --io netcdf path does not need it either"
                           % (", ".join(missing[stream]), case_dir))
    for stream, names in missing.items():
        notes.append("WARNING: aux input stream %s keeps the case's own (netCDF) io_form: %s "
                     "not found in CASE_DIR" % (stream, ", ".join(names)))
    return [st for st in bp_plan["streams"] if st["stream"] in bp_plan["ready"]]


def _per_record_of(bp_plan: dict, io_streams: Optional[list]) -> list[str]:
    """Sorted per-record input names (resolve_bp_inputs' "per_record") of the streams this run
    switches to ADIOS2 -- a stream left on netCDF (an aux file missing) has none."""
    cc = sweep.campaign_configs
    kind_stream = {v[0]: s for s, v in cc.PER_RECORD_STREAMS.items()}
    ready = set(st["stream"] for st in io_streams or ())
    return sorted(n for n in (bp_plan.get("per_record") or {})
                  if _PER_RECORD_KIND_RE.match(n)
                  and kind_stream[_PER_RECORD_KIND_RE.match(n).group(1)] in ready)


def io_plan_line(io: str, io_streams: Optional[list], restart_io_form: int,
                 bp_plan: Optional[dict], bp_cache: Optional[str], case_dir: str) -> str:
    """The one-line I/O plan, e.g. 'I/O: ADIOS2 (history/restart/input/boundary/aux4), inputs
    from <cache> (3 netCDF -> BP4, 1 existing BP)'."""
    if io_streams is None:
        return "I/O: LEGACY netCDF (--io netcdf): case io_form_* kept, restart io_form %d" % (
            restart_io_form)
    parts = ["history"] + (["restart"] if int(restart_io_form) == ADIOS2_IO_FORM else []) + [
        st["label"] for st in io_streams]
    n_conv, n_bp = len(bp_plan["convert"]), len(bp_plan["existing"])
    src = bp_cache if n_conv else case_dir
    pr = _per_record_of(bp_plan, io_streams)
    return "I/O: ADIOS2 (%s)%s, inputs from %s (%d netCDF -> BP4, %d existing BP)%s" % (
        "/".join(parts), "" if int(restart_io_form) == ADIOS2_IO_FORM
        else "; restart netCDF (--restart-io-form 2)", src, n_conv, n_bp,
        "; one BP per record: %s" % " ".join(pr) if pr else "")


def plan_inputs(io: str, text: str, nl: dict, geo: Geometry, case_dir: str, out: str,
                project: Optional[str], notes: list) -> InputPlan:
    """The input plan of a run, nothing converted yet; warnings go into `notes`."""
    if io != "adios2":
        return InputPlan()
    cc = sweep.campaign_configs
    bp_cache = bp_cache_dir(out, project)
    bp_plan = cc.resolve_bp_inputs(text, case_dir, bp_cache, convert=False)
    io_streams = adios2_streams_or_raise(bp_plan, case_dir, notes)
    per_record = _per_record_of(bp_plan, io_streams)
    edits, align = None, None
    if per_record:
        edits = cc._import_wtune_bp().per_record_namelist_edits(per_record, geo.max_dom)
        align = input_record_alignment(nl, geo, per_record)
    for t in bp_plan["single_multirecord"]:
        notes.append("WARNING: %s is a single multi-record BP: WRF 4.8's ADIOS2 reader "
                     "returns its FIRST record for every read -- remove it so the "
                     "netCDF is converted per record" % t)
    return InputPlan(io_streams=io_streams, bp_plan=bp_plan, bp_cache=bp_cache,
                     per_record=per_record, per_record_edits=edits, align=align)


def convert_inputs(inputs: InputPlan, text: str, case_dir: str, segments: list,
                   log: Callable) -> InputPlan:
    """Convert the netCDF inputs into the BP cache (--io adios2) and re-check the plan against
    what is now on disk: the same per-record set, a record file at every segment start.
    RunError: nothing may be submitted."""
    if inputs.io_streams is None:
        return inputs
    cc = sweep.campaign_configs
    try:
        bp_plan = cc.resolve_bp_inputs(text, case_dir, inputs.bp_cache, convert=True, log=log)
        bp_xml = cc._import_wtune_bp().adios2_xml()
    except Exception as e:           # converter / adios2 bindings: any failure stops here
        raise RunError("ADIOS2 input staging failed (%s: %s) -- fix the converter env, or "
                       "rerun with the legacy --io netcdf" % (type(e).__name__, e))
    got = _per_record_of(bp_plan, inputs.io_streams)
    miss = missing_record_files(bp_plan["links"], got, segments)
    if got != inputs.per_record or miss:
        raise RunError("per-record ADIOS2 inputs %s (planned %s)%s -- nothing submitted"
                       % (" ".join(got) or "none", " ".join(inputs.per_record) or "none",
                          "; missing at segment starts: " + ", ".join(
                              "seg %d %s" % m for m in miss[:6]) if miss else ""))
    return dataclasses.replace(inputs, bp_plan=bp_plan, bp_xml=bp_xml)


def _logging_copier(log: Callable) -> Callable:
    """wtune_bp.fresh_input_copy that logs reflink vs full copy (one line per segment attempt)."""
    def copy(sources, dest):
        stats = {}
        out = sweep.campaign_configs._import_wtune_bp().fresh_input_copy(sources, dest,
                                                                         stats=stats)
        log("  fresh per-record inputs -> %s (%s)" % (dest, "reflink" if stats.get("reflink")
                                                       else "cp -a %.1f MB" % (
                                                           stats.get("bytes", 0) / 1e6)))
        return out
    return copy
