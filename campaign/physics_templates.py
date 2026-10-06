#!/usr/bin/env python3
"""PHYSICS-GRAY physics templates (decision 0051): apply physics_templates.json edits to a
namelist, decide which templates a case may run, and format the result for the ledger.

The ops ('set', 'mul', 'delete') are specified in physics_templates.json's '_comment'.
Stdlib only, Python 3.6.
"""
import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from case_tables import Case, _case_effective_max_dom, _case_namelist_path, case_frozen_keys
from namelist_text import _find_namelist_key, _insert_after_section_header

Edit = Dict[str, Any]
Template = Dict[str, Any]
Summary = List[Dict[str, Any]]          # [{key, op, old, new}] in the order applied

PHYSICS_TEMPLATES_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "physics_templates.json")
DEFAULT_SCREEN_DIR = "/home/u5727520/wrf-bench/wscore/physics"

# ledger columns a template edit can change (time_step is its own column)
_LEDGER_PHYSICS_KEYS = (
    "time_step", "mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics",
    "sf_surface_physics", "ra_lw_physics", "ra_sw_physics", "radt", "cudt",
)

_PROBE_VERDICT_RE = re.compile(
    r"(?m)^[ \t]*(?P<knob>\S+)[ \t]+\[(?:PHYSICS-GRAY|STRICT)[ \t]*\][ \t]+(?P<verdict>\S+)"
)


def load_physics_templates(path: str = PHYSICS_TEMPLATES_JSON_PATH) -> Dict[str, Template]:
    with open(path) as f:
        data = json.load(f)
    return data["templates"]


# --------------------------------------------------------------------------- one edit
def _domain_values_from_map(key: str, vbd: Dict[Any, Any], expected: int) -> List[Any]:
    """A {1-based domain: value} map (JSON keys are strings) as a d01-first list."""
    index = {}
    for k, v in vbd.items():
        try:
            index[int(k)] = v
        except (TypeError, ValueError):
            raise ValueError(
                "physics template: key %r values_by_domain has a non-integer domain index %r "
                "(keys are 1-based domain numbers)" % (key, k))
    outside = sorted(d for d in index if d < 1 or d > expected)
    if outside:
        raise ValueError(
            "physics template: key %r values_by_domain names domain(s) %s outside this case's "
            "%d active domain(s)" % (key, outside, expected))
    missing = [d for d in range(1, expected + 1) if d not in index]
    if missing:
        raise ValueError(
            "physics template: key %r values_by_domain is missing domain(s) %s of this case's "
            "%d active domain(s)" % (key, missing, expected))
    return [index[d] for d in range(1, expected + 1)]


def _set_values_for_domains(edit: Edit, n: int, max_dom: Optional[int]) -> List[str]:
    """The n per-domain strings a 'set' edit writes: its 'value' repeated, or 'values_by_domain'
    (a d01-first list or a {domain: value} map; a nest need not take its parent's value).

    A length or index that does not match the case's domains RAISES: padding or truncating
    is how a nest ends up at a value nobody chose.
    """
    vbd = edit.get("values_by_domain")
    if vbd is None:
        return [str(edit["value"])] * n
    key = edit["key"]
    expected = int(max_dom) if max_dom else n
    if isinstance(vbd, dict):
        ordered = _domain_values_from_map(key, vbd, expected)
    else:
        ordered = list(vbd)
        if len(ordered) != expected:
            raise ValueError(
                "physics template: key %r values_by_domain has %d value(s) for this case's %d "
                "active domain(s)" % (key, len(ordered), expected))
    if len(ordered) != n:
        raise ValueError(
            "physics template: key %r values_by_domain has %d value(s) but only %d namelist "
            "entr(ies) are active for this key -- add the missing per-domain entries to the "
            "namelist first" % (key, len(ordered), n))
    return [str(v) for v in ordered]


def _delete_assignment(text: str, key: str) -> str:
    """Remove `key`'s own "key=value," span (a packed neighbour stays), and its line if that
    leaves the line blank."""
    found = _find_namelist_key(text, key)
    if found is None:
        raise ValueError("physics template: key %r not found for 'delete'" % (key,))
    new_text = text[:found["key_start"]] + text[found["value_end"]:]
    line_start = new_text.rfind("\n", 0, found["key_start"]) + 1
    line_end = new_text.find("\n", found["key_start"])
    if line_end == -1:
        line_end = len(new_text)
    if new_text[line_start:line_end].strip() == "":
        new_text = new_text[:line_start] + new_text[line_end + 1:]
    return new_text


def _insert_assignment(text: str, edit: Edit, max_dom: Optional[int]) -> Tuple[str, List[str]]:
    """'set' on an absent key: a new line under the edit's section -> (text, values written).

    per_domain (implied by values_by_domain) writes every active domain, or the nests would
    keep the Registry default while only d01 changes.
    """
    per_domain = edit.get("per_domain") or edit.get("values_by_domain") is not None
    n_insert = max_dom if (per_domain and max_dom) else 1
    new_active = _set_values_for_domains(edit, n_insert, max_dom if per_domain else None)
    line = " %s = %s,\n" % (edit["key"], ", ".join(new_active))
    section = edit.get("section", "&physics")
    return _insert_after_section_header(text, section, line), new_active


def _set_values(edit: Edit, active_old: List[str], max_dom: Optional[int]) -> List[str]:
    return _set_values_for_domains(edit, len(active_old), max_dom)


def _mul_values(edit: Edit, active_old: List[str], max_dom: Optional[int]) -> List[str]:
    """Each active value times edit['factor'], optionally rounded to int and clamped."""
    key = edit["key"]
    factor = edit["factor"]
    round_int = edit.get("round") == "int"
    mn, mx = edit.get("min"), edit.get("max")
    active_new = []
    for v in active_old:
        try:
            native = float(v)
        except ValueError:
            raise ValueError("physics template: key %r value %r is not numeric" % (key, v))
        if native == -1:
            raise ValueError(
                "physics template: cannot 'mul' key %r whose native value is -1 "
                "(physics_suite default) -- use 'set' to override a suite key "
                "instead" % (key,)
            )
        new_v = native * factor
        new_v = int(round(new_v)) if round_int else new_v
        if mn is not None:
            new_v = max(new_v, mn)
        if mx is not None:
            new_v = min(new_v, mx)
        active_new.append(str(new_v))
    return active_new


_VALUE_OPS = {"set": _set_values, "mul": _mul_values}   # op -> new active-domain values


def apply_physics_edit(text: str, edit: Edit, max_dom: Optional[int]
                       ) -> Tuple[str, Optional[List[str]], Optional[List[str]]]:
    """Apply one template edit -> (text, old active-domain values, new ones); (text, None,
    None) for 'delete'. ValueError names the key/op when the edit cannot be applied."""
    op = edit["op"]
    key = edit["key"]
    if op == "delete":
        return _delete_assignment(text, key), None, None
    found = _find_namelist_key(text, key)
    if found is None:
        if op != "set":
            raise ValueError("physics template: key %r not found for op %r" % (key, op))
        new_text, new_active = _insert_assignment(text, edit, max_dom)
        return new_text, [], new_active
    values = found["values"]
    n_active = min(len(values), max_dom) if max_dom else len(values)
    if n_active == 0:
        raise ValueError("physics template: key %r has no value(s) to edit" % (key,))
    active_old = values[:n_active]
    new_values = _VALUE_OPS.get(op)
    if new_values is None:
        raise ValueError("unknown physics edit op %r" % (op,))
    active_new = new_values(edit, active_old, max_dom)
    new_values_str = ", ".join(active_new + values[n_active:]) + ","
    new_text = text[:found["value_start"]] + new_values_str + text[found["value_end"]:]
    return new_text, active_old, active_new


# --------------------------------------------------------------------------- one template
def _template_edits(template: Optional[Template], case_id: Optional[str]) -> List[Edit]:
    """The template's edits followed by its extra_edits for this case."""
    edits = list((template or {}).get("edits") or [])
    if case_id is not None:
        edits += list(((template or {}).get("extra_edits") or {}).get(case_id, []))
    return edits


def template_frozen_key_conflicts(template: Optional[Template],
                                  frozen_keys: Optional[Iterable[str]],
                                  case_id: Optional[str] = None) -> List[str]:
    """The frozen keys `template` would write (edits + this case's extra_edits), in edit order."""
    frozen = {str(k).strip().lower() for k in frozen_keys or ()}
    if not frozen:
        return []
    hit = []
    for e in _template_edits(template, case_id):
        k = str(e.get("key", "")).lower()
        if k in frozen and k not in hit:
            hit.append(k)
    return hit


def apply_physics_template(text: str, template: Template, max_dom: Optional[int],
                           case_id: Optional[str] = None,
                           frozen_keys: Optional[Iterable[str]] = None) -> Tuple[str, Summary]:
    """Apply the template's edits, then its extra_edits[case_id] -> (text, summary).

    A template that would write a task-mandated frozen key is refused before any edit.
    ValueError from an edit propagates: do not use a partially applied text.
    """
    summary = []
    edits = list(template.get("edits", []))
    if case_id is not None:
        edits += list((template.get("extra_edits") or {}).get(case_id, []))
    conflicts = template_frozen_key_conflicts(template, frozen_keys, case_id=case_id)
    if conflicts:
        raise ValueError(
            "physics template: refusing to edit task-mandated frozen key(s) %s for case %s -- "
            "cases.json lists them in 'frozen_keys', i.e. the task fixes their value and changing "
            "one is a rules violation, not a tuning trade-off"
            % (", ".join(conflicts), case_id if case_id is not None else "<unnamed>"))
    for edit in edits:
        text, old, new = apply_physics_edit(text, edit, max_dom)
        summary.append(dict(key=edit["key"], op=edit["op"], old=old, new=new))
    return text, summary


def physics_template_applicability(case_id: str, template: Template,
                                   frozen_keys: Optional[Iterable[str]] = None
                                   ) -> Tuple[bool, Optional[str]]:
    """(applicable, reason) from the template's applies_to / skip_cases and the case's frozen
    keys. The edits are not attempted: an applicable template can still fail to resolve."""
    applies_to = template.get("applies_to")
    if applies_to is not None and case_id not in applies_to:
        return False, "not in applies_to %s" % (applies_to,)
    skip_reason = (template.get("skip_cases") or {}).get(case_id)
    if skip_reason is not None:
        return False, skip_reason
    conflicts = template_frozen_key_conflicts(template, frozen_keys, case_id=case_id)
    if conflicts:
        return False, ("edits task-mandated frozen key(s) %s for case %s: the task FIXES their "
                       "value, so changing one is a rules violation, not a tuning trade-off"
                       % (", ".join(conflicts), case_id))
    return True, None


def resolve_physics_config(case_id: str, template_id: str, cases: Dict[str, Case],
                           templates: Dict[str, Template]
                           ) -> Tuple[Optional[Summary], Optional[str]]:
    """Try one template on the case's OWN namelist (the file edit-namelist edits).

    (summary, None) -- [] for a template without edits (P00) -- or (None, why it does not
    apply or resolve).
    """
    case = cases.get(case_id)
    if case is None:
        return None, "unknown case %r" % (case_id,)
    template = templates.get(template_id)
    if template is None:
        return None, "unknown physics template %r" % (template_id,)
    frozen = case_frozen_keys(case)
    ok, reason = physics_template_applicability(case_id, template, frozen_keys=frozen)
    if not ok:
        return None, reason
    path = _case_namelist_path(case)
    try:
        with open(path) as f:
            text = f.read()
    except IOError as e:
        return None, "cannot read namelist template %s: %s" % (path, e)
    max_dom = _case_effective_max_dom(case)
    try:
        _new_text, summary = apply_physics_template(text, template, max_dom, case_id=case_id,
                                                    frozen_keys=frozen)
    except ValueError as e:
        return None, str(e)
    return summary, None


def applicable_template_ids(case_id: str, template_order: Sequence[str],
                            resolve: Callable[[str, str], Tuple[Optional[Summary], Any]]
                            ) -> List[str]:
    """The non-P00 template ids that resolve for the case, in template_order."""
    return [tid for tid in template_order
            if tid != "P00" and resolve(case_id, tid)[0] is not None]


# --------------------------------------------------------------------------- ledger fields
def format_physics_edits(summary: Summary) -> str:
    """The ledger notes 'edits=' value: ';'-joined key=new tokens, domains '/'-joined, no
    spaces. '' for a template without edits."""
    parts = []
    for e in summary:
        if e["op"] == "delete":
            parts.append("%s=DELETED" % (e["key"],))
        else:
            parts.append("%s=%s" % (e["key"], "/".join(e["new"])))
    return ";".join(parts)


def parse_probe_verdicts(screen_text: str) -> Dict[str, str]:
    """{knob: verdict} of the '<knob> [PHYSICS-GRAY ] VERDICT' / '[STRICT ]' lines of a wtune
    screen report (job 1076060); a later line for a knob wins."""
    out = {}
    for m in _PROBE_VERDICT_RE.finditer(screen_text):
        out[m.group("knob")] = m.group("verdict")
    return out


def probe_verdicts_for_case(case_id: str, keys: Sequence[str],
                            screen_dir: str = DEFAULT_SCREEN_DIR) -> str:
    """The notes 'probe=<knob>:<verdict>,...' value from screen_dir/screen_<case_id>.txt.

    'probe=none' without keys or without the file; a knob the file lacks reads NOTFOUND.
    """
    if not keys:
        return "probe=none"
    path = os.path.join(screen_dir, "screen_%s.txt" % (case_id,))
    if not os.path.isfile(path):
        return "probe=none"
    with open(path) as f:
        verdicts = parse_probe_verdicts(f.read())
    parts = ["%s:%s" % (k, verdicts.get(k, "NOTFOUND")) for k in keys]
    return "probe=" + ",".join(parts)


def physics_tag_template(tag: str) -> Tuple[str, str]:
    """(template id, role) of a physics run tag: P00_ref / P00_twin / a candidate's own id."""
    if tag == "P00_ref":
        return "P00", "reference"
    if tag == "P00_twin":
        return "P00", "twin"
    return tag, "candidate"


def physics_ledger_columns(case: Case, summary: Summary) -> Dict[str, Any]:
    """{ledger column: value}: the case's native d01 values (case['dt'], case['physics']),
    with each edited key replaced by its new d01 value."""
    cols = {"time_step": case["dt"]}
    phys = case.get("physics", {})
    for k in _LEDGER_PHYSICS_KEYS:
        if k == "time_step":
            continue
        cols[k] = phys.get(k, "")
    for e in summary:
        key = e["key"]
        if key not in _LEDGER_PHYSICS_KEYS or e["op"] == "delete":
            continue
        cols[key] = e["new"][0]
    return cols
