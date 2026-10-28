#!/usr/bin/env python3
"""WRF namelist.input as text: find, set, insert and delete assignments in place.

Every edit keeps the template's own layout (packed lines, '=' spacing, comments), because
the campaign edits real case namelists it does not own. Pure functions of the text.
Stdlib only, Python 3.6.
"""
import datetime
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_NOT_IN_NAME = r"(?<![A-Za-z0-9_])"        # the key is not the tail of a longer name
_NML_NEXT_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_%]*[ \t]*=")


# --------------------------------------------------------------------------- set / delete
def _set_scalar(text: str, key: str, value: Any) -> str:
    """Set the first assignment of `key`, keeping its '=' spacing; unchanged when absent."""
    pattern = re.compile(r"\b%s\b([ \t]*=[ \t]*)[^,\n]*" % re.escape(key))
    new_text, _n = pattern.subn(lambda m: "%s%s%s" % (key, m.group(1), value), text, count=1)
    return new_text


def _delete_key_lines(text: str, key: str, ignore_case: bool = False) -> str:
    """Delete every whole line whose first token is `key =` (a key packed mid-line stays)."""
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(r"(?m)^[ \t]*%s[ \t]*=.*\n?" % re.escape(key), flags)
    return pattern.sub("", text)


def _header_eol(text: str, section: str) -> int:
    """Index of the newline ending the first line that holds `section`; ValueError if none."""
    idx = text.find(section)
    if idx == -1:
        raise ValueError("no %s section found in namelist" % (section,))
    eol = text.find("\n", idx)
    return len(text) if eol == -1 else eol


def _insert_after_section_header(text: str, section: str, line: str) -> str:
    """Insert `line` (ending in a newline) right after the first line holding `section`."""
    eol = _header_eol(text, section)
    return text[: eol + 1] + line + text[eol + 1 :]


def _insert_after_domains(text: str, nproc_x: int, nproc_y: int, numtiles: int, max_dom: int = 1,
                          extra: Optional[Iterable[Tuple[str, int]]] = None) -> str:
    """Add one `nproc_x=, nproc_y=, numtiles=[, extra key=value...]` line under &domains.

    All are Registry dimension-1 keys: written once whatever max_dom is (a per-domain numtiles
    list aborts with "ERROR reading namelist domains", smoke jobs 1074506-1074508).
    """
    eol = _header_eol(text, "&domains")
    parts = ["nproc_x=%d" % nproc_x, "nproc_y=%d" % nproc_y, "numtiles=%d" % numtiles]
    for key, value in (extra or []):
        parts.append("%s=%d" % (key, value))
    new_line = " " + ", ".join(parts) + ",\n"
    return text[: eol + 1] + new_line + text[eol + 1 :]


def _replace_list_line(text: str, key: str, values: Sequence[Any]) -> str:
    """Replace every whole-line assignment of `key` by one comma list under &time_control."""
    text = _delete_key_lines(text, key)
    eol = _header_eol(text, "&time_control")
    new_line = " %s = %s,\n" % (key, ", ".join(str(v) for v in values))
    return text[: eol + 1] + new_line + text[eol + 1 :]


def _set_or_insert_time_control_scalar(text: str, key: str, value: Any,
                                       group: str = "time_control") -> str:
    """Set `key` wherever it is assigned, else insert it under &`group`.

    Every non-comment occurrence is rewritten, mid-line ones too: inserting next to karina's
    packed "io_form_input=2," left WRF on netCDF (smoke job 1074653).
    """
    occ = re.compile(r"(?m)^(?![ \t]*!)(.*?)%s%s([ \t]*=[ \t]*)[^,\n]*"
                     % (_NOT_IN_NAME, re.escape(key)))
    if occ.search(text):      # at most one match per line: a plain sub rewrites them all
        return occ.sub(lambda m: "%s%s%s%s" % (m.group(1), key, m.group(2), value), text)
    m = re.search(r"(?im)^[ \t]*&%s\b" % re.escape(group), text)
    if m is None:
        raise ValueError("no &%s section found in namelist" % group)
    eol = text.find("\n", m.start())
    if eol == -1:
        eol = len(text)
    return text[: eol + 1] + " %s=%s,\n" % (key, value) + text[eol + 1 :]


def _value_list_end(rest: str) -> Optional[int]:
    """Where the value list starting `rest` ends: the next `key =`, '!' or '/' outside quotes."""
    quote = None
    for j, c in enumerate(rest):
        if quote:
            quote = None if c == quote else quote
        elif c in "'\"":
            quote = c
        elif c in "!/":
            return j
        elif ((c.isalpha() or c == "_") and (j == 0 or rest[j - 1] in " \t,")
              and _NML_NEXT_KEY_RE.match(rest, j)):
            return j
    return None


def _set_namelist_value(text: str, key: str, literal: str, group: str) -> str:
    """Replace the WHOLE value list of every assignment of `key` by the Fortran `literal`.

    A packed line keeps its other keys. An unassigned key is inserted under &`group`; a
    missing group is appended holding just this key.
    """
    rhs = literal.strip().rstrip(",").rstrip() + ","
    key_re = re.compile(r"%s%s[ \t]*=[ \t]*" % (_NOT_IN_NAME, re.escape(key)), re.I)
    lines = text.split("\n")
    hit = False
    for i, ln in enumerate(lines):
        m = None if ln.lstrip().startswith("!") else key_re.search(ln)
        if m is None:
            continue
        rest = ln[m.end():]
        end = _value_list_end(rest)
        tail = rest[end:] if end is not None else ""
        lines[i] = ln[:m.end()] + rhs + ((" " + tail.lstrip()) if tail.strip() else "")
        hit = True
    if hit:
        return "\n".join(lines)
    m = re.search(r"(?im)^[ \t]*&%s\b[^\n]*\n" % re.escape(group), text)
    if m is not None:
        return text[:m.end()] + " %s = %s\n" % (key, rhs) + text[m.end():]
    return text.rstrip("\n") + "\n&%s\n %s = %s\n/\n" % (group, key, rhs)


# --------------------------------------------------------------------------- read
def _get_scalar_first(text: str, key: str, default: Optional[str] = None) -> Optional[str]:
    """First value of `key` (a comma list's first entry), stripped; `default` when absent."""
    m = re.search(r"\b%s\b[ \t]*=[ \t]*([^,\n]*)" % re.escape(key), text)
    if not m:
        return default
    return m.group(1).strip()


def case_start_datetime(text: str) -> datetime.datetime:
    """Domain 1's start_year..start_second (hour/minute/second default to 0)."""
    year = int(_get_scalar_first(text, "start_year"))
    month = int(_get_scalar_first(text, "start_month"))
    day = int(_get_scalar_first(text, "start_day"))
    hour = int(_get_scalar_first(text, "start_hour", "0") or 0)
    minute = int(_get_scalar_first(text, "start_minute", "0") or 0)
    second = int(_get_scalar_first(text, "start_second", "0") or 0)
    return datetime.datetime(year, month, day, hour, minute, second)


def _own_values(scan: str) -> Tuple[List[str], int]:
    """(value tokens, chars consumed) of the comma list opening `scan`.

    Stops at a token holding '=': that is the next key packed onto the same line.
    """
    values = []
    consumed = 0
    pos = 0
    while True:
        comma = scan.find(",", pos)
        tok = scan[pos:comma] if comma != -1 else scan[pos:]
        if "=" in tok:
            break
        stripped = tok.strip()
        if stripped:
            values.append(stripped)
        consumed = (comma + 1) if comma != -1 else len(scan)
        if comma == -1:
            break
        pos = comma + 1
    return values, consumed


def _find_namelist_key(text: str, key: str) -> Optional[Dict[str, Any]]:
    """First non-comment assignment of `key`, anywhere on its line (case-sensitive), or None.

    Returns {key_start, value_start, value_end, values}: text[value_start:value_end] is exactly
    this key's own comma list, so it can be replaced without touching a neighbour packed onto
    the same line. Only for keys whose values are never quoted strings.
    """
    pattern = re.compile(r"(?m)^(?![ \t]*!)(?:.*?)%s(%s[ \t]*=[ \t]*)"
                         % (_NOT_IN_NAME, re.escape(key)))
    m = pattern.search(text)
    if not m:
        return None
    value_start = m.end()
    line_end = text.find("\n", value_start)
    if line_end == -1:
        line_end = len(text)
    rest = text[value_start:line_end]
    bang = rest.find("!")
    values, consumed = _own_values(rest if bang == -1 else rest[:bang])
    return dict(key_start=m.start(1), value_start=value_start,
                value_end=value_start + consumed, values=values)


def _value_tokens(rest: str) -> List[str]:
    """Value tokens opening `rest`, quotes stripped; ends at an unquoted '!' or the next key."""
    vals, i, n = [], 0, len(rest)
    while i < n:
        c = rest[i]
        if c in " \t,":
            i += 1
        elif c in "'\"":
            j = rest.find(c, i + 1)
            j = n if j == -1 else j
            vals.append(rest[i + 1:j])
            i = j + 1
        elif c == "!":
            break
        else:
            j = i
            while j < n and rest[j] not in " \t,!":
                j += 1
            tok = rest[i:j]
            if "=" in tok:
                break
            vals.append(tok)
            i = j
    return vals


def _namelist_string_values(text: str, key: str) -> Optional[List[str]]:
    """Value tokens of the LAST non-comment assignment of `key` (any case); None if unassigned."""
    pat = re.compile(r"%s%s[ \t]*=[ \t]*" % (_NOT_IN_NAME, re.escape(key)), re.I)
    found = None
    for line in text.splitlines():
        if line.lstrip().startswith("!"):
            continue
        m = None
        for m in pat.finditer(line):
            pass
        if m is not None:
            found = _value_tokens(line[m.end():])
    return found


def _namelist_ints(text: str, key: str) -> List[int]:
    """`key`'s values as ints ('2*5' counts as 5, a non-number as 0); [] if unassigned."""
    out = []
    for v in _namelist_string_values(text, key) or []:
        try:
            out.append(int(v.split("*")[-1]))
        except ValueError:
            out.append(0)
    return out


def _force_gsfcgce_radar_ref(text: str) -> str:
    """do_radar_ref = 1 when a domain uses mp_physics 7. WRF's gsfcgce 4-ice reads and writes
    refl_10cm over the whole tile (module_mp_gsfcgce_4ice_nuwrf.f90:446,505) but the Registry
    allocates it only under compute_radar_ref (module_check_a_mundo.F:3528): without this the
    first step overruns the heap and crashes at random (f1-arm haitang2017, 6 of 9 launches)."""
    if 7 not in _namelist_ints(text, "mp_physics") or _namelist_ints(text, "do_radar_ref") == [1]:
        return text
    return _set_namelist_value(text, "do_radar_ref", "1", "physics")


def _has_key(text: str, key: str) -> bool:
    return any(re.search(r"%s%s[ \t]*=" % (_NOT_IN_NAME, re.escape(key)), ln, re.I)
               for ln in text.splitlines() if not ln.lstrip().startswith("!"))
