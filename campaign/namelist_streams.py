#!/usr/bin/env python3
"""Which files a WRF namelist makes wrf.exe open: the files it names, its input streams, and
the streams staged one ADIOS2 BP per record. Reads the namelist text (and, for per-record
inputs, whether a record directory exists). Stdlib only, Python 3.6.
"""
import datetime
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from namelist_text import _namelist_ints, _namelist_string_values, case_start_datetime

Stream = Dict[str, Any]
FileRef = Tuple[str, str, bool]

_NML_REF_SKIP_AUXINPUT = (1, 4)   # 1 = met_em (real.exe only); 4 = wrflowinp, via sst_update
ADIOS2_IO_FORM = 14
WRF_DATE_FMT = "%Y-%m-%d_%H:%M:%S"

# Multi-record inputs converted one BP per record (wtune.bp.PER_RECORD_NAME_RE): stream ->
# (file kind, io_form key, group, inname key, netCDF default inname, frames_per key or None).
PER_RECORD_STREAMS = {
    "boundary": ("wrfbdy", "io_form_boundary", "time_control", "bdy_inname",
                 "wrfbdy_d<domain>", None),
    "auxinput4": ("wrflowinp", "io_form_auxinput4", "time_control", "auxinput4_inname",
                  "wrflowinp_d<domain>", "frames_per_auxinput4"),
    "gfdda": ("wrffdda", "io_form_gfdda", "fdda", "gfdda_inname", "wrffdda_d<domain>",
              "frames_per_auxinput10"),
}
_PER_RECORD_NAME_RE = re.compile(r"^(wrfbdy|wrflowinp|wrffdda)_d\d\d$")

# (flag key, stream / inname-key prefix, default inname) of the two FDDA input streams
_FDDA_STREAMS = (("grid_fdda", "gfdda", "wrffdda_d<domain>"),
                 ("grid_sfdda", "sgfdda", "wrfsfdda_d<domain>"))


def _active_domains(text: str) -> List[int]:
    max_dom = (_namelist_ints(text, "max_dom") or [1])[0] or 1
    return list(range(1, max_dom + 1))


def _for_domains(name: str, doms: Iterable[int]) -> List[str]:
    """`name` with '<domain>' expanded per domain (WRF writes the 2-digit domain id)."""
    return [name.replace("<domain>", "%02d" % d) for d in doms] if "<domain>" in name else [name]


def _sst_update_on(text: str) -> bool:
    return (_namelist_ints(text, "sst_update") or [0])[0] == 1


def _aux_stream_numbers(text: str) -> List[int]:
    """N of every auxinputN_inname the namelist sets, minus the ones handled elsewhere."""
    found = re.findall(r"(?i)(?<![A-Za-z0-9_])auxinput(\d+)_inname", text)
    return [n for n in sorted(set(int(x) for x in found)) if n not in _NML_REF_SKIP_AUXINPUT]


# --------------------------------------------------------------------------- files named
def namelist_file_refs(text: str) -> List[FileRef]:
    """[(key, name, required)] files wrf.exe opens because the namelist names them.

    Required: iofields_filename, wrflowinp (sst_update=1), wrffdda / wrfsfdda (grid_[s]fdda>0).
    Staged when present: other auxinputN_inname, OBS_DOMAIN<d>* (obs_nudge_opt=1), tslist.
    '<domain>' is resolved per active domain; '<date>' is left in (a glob pattern).
    """
    doms = _active_domains(text)
    refs = []

    def per_dom(values, d):          # WRF's per-domain arrays: an absent entry keeps the default
        return values[d - 1] if values and d - 1 < len(values) else None

    def add(key, names, required):
        for nm in names:
            if nm and (key, nm, required) not in refs:
                refs.append((key, nm, required))

    io = _namelist_string_values(text, "iofields_filename") or []
    for d in doms:
        v = per_dom(io, d)
        if v and v.strip().upper() != "NONE_SPECIFIED":
            add("iofields_filename", [v.strip().replace("<domain>", "%02d" % d)], True)
    if _sst_update_on(text):
        nm = (_namelist_string_values(text, "auxinput4_inname") or ["wrflowinp_d<domain>"])[0]
        add("auxinput4_inname", _for_domains(nm.strip(), doms), True)
    for flag, stream, default in _FDDA_STREAMS:
        key = stream + "_inname"
        on = _namelist_ints(text, flag)
        nm = (_namelist_string_values(text, key) or [default])[0].strip()
        for d in doms:
            if (per_dom(on, d) or 0) > 0:
                add(key, [nm.replace("<domain>", "%02d" % d)], True)
    for n in _aux_stream_numbers(text):
        v = _namelist_string_values(text, "auxinput%d_inname" % n)
        if v:
            add("auxinput%d_inname" % n, _for_domains(v[0].strip(), doms), False)
    obs = _namelist_ints(text, "obs_nudge_opt")
    for d in doms:
        if per_dom(obs, d) == 1:
            add("obs_nudge_opt", ["OBS_DOMAIN%d*" % d], False)
    add("tslist", ["tslist"], False)
    return refs


# --------------------------------------------------------------------------- input streams
def _inname_template(text: str, key: str, default: str) -> str:
    """The stream's file-name template as the namelist sets it (else `default`), without .bp."""
    v = _namelist_string_values(text, key)
    t = v[0].strip() if v and v[0].strip() else default
    return t[:-3] if t.endswith(".bp") else t


def _stream(stream: str, label: str, io_key: str, group: str, inname_key: str, template: str,
            doms: Sequence[int],
            required: Optional[Callable[[List[str]], List[str]]] = None) -> Stream:
    names = _for_domains(template, doms)
    return {"stream": stream, "label": label, "io_key": io_key, "group": group,
            "inname_key": inname_key, "template": template,
            "inname": os.path.basename(template) + ".bp", "names": names,
            "required": names if required is None else required(names)}


def adios2_input_streams(text: str) -> List[Stream]:
    """The WRF input streams this namelist reads, in order: input (d01 required), boundary
    (d01 only), auxinput4 (sst_update=1), gfdda / sgfdda (domains with grid_[s]fdda > 0),
    auxinputN (N != 1, 4, inname set and io_form_auxinputN > 0). Each is {stream, label,
    io_key, group, inname_key, template (no .bp), inname ("<basename(template)>.bp"), names
    (per domain; may hold <date>), required}.
    """
    doms = _active_domains(text)
    out = [_stream("input", "input", "io_form_input", "time_control", "input_inname",
                   _inname_template(text, "input_inname", "wrfinput_d<domain>"), doms,
                   lambda n: n[:1]),
           _stream("boundary", "boundary", "io_form_boundary", "time_control", "bdy_inname",
                   _inname_template(text, "bdy_inname", "wrfbdy_d<domain>"), [1])]
    if _sst_update_on(text):
        out.append(_stream("auxinput4", "aux4", "io_form_auxinput4", "time_control",
                           "auxinput4_inname",
                           _inname_template(text, "auxinput4_inname", "wrflowinp_d<domain>"),
                           doms))
    for flag, stream, default in _FDDA_STREAMS:
        on = _namelist_ints(text, flag)
        ds = [d for d in doms if d - 1 < len(on) and on[d - 1] > 0]
        if ds:
            out.append(_stream(stream, stream, "io_form_" + stream, "fdda", stream + "_inname",
                               _inname_template(text, stream + "_inname", default), ds))
    for n in _aux_stream_numbers(text):
        if not (_namelist_ints(text, "io_form_auxinput%d" % n) or [0])[0]:
            continue
        template = _inname_template(text, "auxinput%d_inname" % n, "")
        if template:
            out.append(_stream("auxinput%d" % n, "aux%d" % n, "io_form_auxinput%d" % n,
                               "time_control", "auxinput%d_inname" % n, template, doms))
    return out


def adios2_stream_edits(stream: Stream) -> List[Tuple[str, Any, str]]:
    """[(key, value, group)] that make WRF read `stream` as ADIOS2: io_form 14 and the inname
    with .bp (io_adios2 opens the inname exactly as given, wrf_io.F90)."""
    return [(stream["io_key"], ADIOS2_IO_FORM, stream["group"]),
            (stream["inname_key"], '"%s"' % stream["inname"], stream["group"])]


# --------------------------------------------------------------------------- per-record inputs
def per_record_has(bp_dir: Optional[str], name: str, when: str) -> bool:
    """True when bp_dir holds the record <name>_<when>.bp WRF opens at `when` (a WRF date):
    inside <bp_dir>/<name>[.bp]/ or as a flat link <bp_dir>/<record>."""
    rec = "%s_%s.bp" % (name, when)
    if not bp_dir:
        return False
    return any(os.path.isdir(os.path.join(bp_dir, *parts))
               for parts in ((name, rec), (name + ".bp", rec), (rec,)))


def per_record_streams(names: Union[str, Iterable[str], None],
                       start: Optional[datetime.datetime], text: str, bp_dir: Optional[str],
                       aux_ok: Optional[Set[str]] = None
                       ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Split per-record input names (WTUNE_PER_RECORD_BP) by stream ->
    ({stream: names} readable from the run start, {stream: [missing record files]}).

    A stream is readable only when EVERY name has <name>_<start>.bp (one inname serves all
    domains); start defaults to the namelist's own. Aux streams outside aux_ok are dropped.
    """
    if isinstance(names, str):
        names = [n for n in re.split(r"[\s,]+", names) if n]
    kind_stream = {v[0]: k for k, v in PER_RECORD_STREAMS.items()}
    by_stream = {}
    for n in names or []:
        m = _PER_RECORD_NAME_RE.match(n)
        if m:
            by_stream.setdefault(kind_stream[m.group(1)], []).append(n)
    if aux_ok is not None:
        by_stream = {s: v for s, v in by_stream.items() if s == "boundary" or s in aux_ok}
    if not by_stream:
        return {}, {}
    when = (start or case_start_datetime(text)).strftime(WRF_DATE_FMT)
    ok, fallback = {}, {}
    for stream, ns in by_stream.items():
        missing = ["%s_%s.bp" % (n, when) for n in ns if not per_record_has(bp_dir, n, when)]
        if missing:
            fallback[stream] = missing
        else:
            ok[stream] = ns
    return ok, fallback
