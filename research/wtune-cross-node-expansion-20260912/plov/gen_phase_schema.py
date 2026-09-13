#!/usr/bin/env python3
"""Emit a conservative, audit-only manifest for one compiled Fortran routine."""
import argparse
import hashlib
import json
import os
import re
import tempfile


class SchemaError(ValueError):
    pass


def _segments(line):
    """Split free-form statements, recognizing Fortran quotes and comments."""
    segments, current, quote = [], [], None
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            current.append(char)
            if char == quote:
                if index + 1 < len(line) and line[index + 1] == quote:
                    current.append(line[index + 1])
                    index += 1
                else:
                    quote = None
        elif char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == "!":
            break
        elif char == ";":
            segments.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    if quote:
        raise SchemaError("unmatched quote")
    segments.append("".join(current))
    return segments


def _statements(lines):
    """Join free-form continuations; this deliberately is not a Fortran parser."""
    out, current = [], ""
    for line in lines:
        for segment in _segments(line):
            text = segment.strip()
            if not text:
                continue
            if text.startswith("&"):
                text = text[1:].lstrip()
            continued = text.endswith("&")
            current += text[:-1].rstrip() if continued else text
            if not continued:
                out.append(current.strip())
                current = ""
            else:
                current += " "
    if current:
        raise SchemaError("unterminated free-form continuation")
    return out


def _split_top_level(text):
    parts, current, depth = [], [], 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise SchemaError("unbalanced parentheses")
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if depth:
        raise SchemaError("unbalanced parentheses")
    parts.append("".join(current).strip())
    return parts


def _kind(type_spec):
    match = re.fullmatch(r"\s*(?:real|integer|logical)\s*(?:\*\s*(\d+)|\(\s*(?:kind\s*=\s*)?([^,)]+)\s*\))?\s*", type_spec, re.I)
    if not match or not (match.group(1) or match.group(2)):
        return None, "missing_kind"
    value = (match.group(1) or match.group(2)).strip().lower()
    return value, None if value.isdigit() else "unresolved_kind"


def _dimensions(text):
    values = _split_top_level(text)
    return [value.strip() for value in values]


def _shape_reasons(dimensions):
    reasons = []
    for dimension in dimensions:
        compact = re.sub(r"\s+", "", dimension)
        if compact == "..":
            reasons.append("assumed_rank")
        elif compact == "*":
            reasons.append("assumed_size")
        elif ":" in compact:
            pieces = compact.split(":")
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                reasons.append("assumed_shape")
            elif pieces[1] == "*":
                reasons.append("assumed_size")
    return reasons


def _attributes(tokens):
    result = {"intent": None, "optional": False, "dimensions": [], "reasons": []}
    for token in tokens:
        lower = token.strip().lower()
        intent = re.fullmatch(r"intent\s*\(\s*(inout|in|out)\s*\)", lower)
        dimension = re.fullmatch(r"dimension\s*\((.*)\)", token.strip(), re.I)
        if intent:
            result["intent"] = intent.group(1)
        elif lower == "optional":
            result["optional"] = True
        elif dimension:
            result["dimensions"] = _dimensions(dimension.group(1))
        elif lower.startswith("codimension"):
            result["reasons"].append("coarray")
        elif lower in {"pointer", "allocatable", "value", "character"}:
            result["reasons"].append(lower)
        elif lower == "external" or lower.startswith("procedure"):
            result["reasons"].append("callable")
        else:
            result["reasons"].append("unsupported_attribute")
    result["reasons"].extend(_shape_reasons(result["dimensions"]))
    return result


def _declaration(stmt):
    if "::" in stmt:
        spec, variables = (piece.strip() for piece in stmt.split("::", 1))
    else:
        legacy = re.match(r"\s*((?:real|integer|logical)\s*(?:\([^)]*\)|\*\s*\d+)?)\s+(.+)$", stmt, re.I)
        if not legacy:
            return []
        spec, variables = legacy.group(1), legacy.group(2)
    base = re.match(r"(real|integer|logical|character|type\s*\(|class\s*\()", spec, re.I)
    if not base:
        return []
    raw_type = base.group(1).lower()
    intrinsic = raw_type if raw_type in {"real", "integer", "logical"} else None
    specification_tokens = _split_top_level(spec)
    attributes = _attributes(specification_tokens[1:])
    kind, kind_reason = _kind(specification_tokens[0]) if intrinsic else (None, None)
    rows = []
    try:
        declared_variables = _split_top_level(variables)
    except SchemaError:
        # A non-dummy specification outside this tool's grammar must not make
        # an otherwise auditable interface disappear.  Dummies not recovered
        # below are explicitly reported as missing declarations.
        return []
    for variable in declared_variables:
        match = re.fullmatch(r"([a-z]\w*)\s*(?:\((.*)\))?", variable, re.I)
        if not match:
            continue
        name, per_var = match.group(1).lower(), match.group(2)
        dimensions = _dimensions(per_var) if per_var is not None else attributes["dimensions"]
        reasons = []
        if intrinsic is None:
            if raw_type.startswith(("type", "class")):
                reasons.append("derived_type")
            elif raw_type == "character":
                reasons.append("character")
            else:
                reasons.append("unsupported_type")
        if kind_reason:
            reasons.append(kind_reason)
        reasons.extend(attributes["reasons"])
        reasons.extend(_shape_reasons(dimensions) if per_var is not None else [])
        rows.append((name, {"type": intrinsic or raw_type, "kind": kind,
                            "intent": attributes["intent"], "optional": attributes["optional"],
                            "dimensions": dimensions,
                            "unsupported_reasons": reasons}))
    return rows


def _standalone_attributes(stmt):
    match = re.match(r"\s*(value|optional|pointer|allocatable|external|procedure\s*(?:\([^)]*\))?|intent\s*\([^)]*\)|dimension|codimension)(?=\s|:|$)\s*(?:::\s*)?(.*)$", stmt, re.I)
    if not match:
        return []
    attribute, variables = match.group(1), match.group(2).strip()
    if not variables:
        return []
    result = []
    for variable in _split_top_level(variables):
        name = re.match(r"([a-z]\w*)\s*(?:\((.*)\)|\[(.*)\])?$", variable.strip(), re.I)
        if not name:
            continue
        token = attribute
        if attribute.lower() == "dimension":
            token += "(" + (name.group(2) or "") + ")"
        if attribute.lower() == "codimension":
            token += "[" + (name.group(3) or "") + "]"
        result.append((name.group(1).lower(), _attributes([token])))
    return result


def extract(source_path, procedure):
    if not source_path.lower().endswith(".f90"):
        raise SchemaError("source must be a compiled .f90 file")
    raw = open(source_path, "rb").read()
    text = raw.decode("utf-8")
    if re.search(r"^\s*#", text, re.M):
        raise SchemaError("CPP directives are not accepted; provide compiled .f90 source")
    if re.search(r"^\s*!(?:dec\$|dir\$)\s*attributes\b", text, re.I | re.M):
        raise SchemaError("compiler ABI directive is unsupported")
    statements = _statements(text.splitlines())
    wanted = procedure.lower()
    starts = [i for i, stmt in enumerate(statements)
              if re.match(r"subroutine\s+" + re.escape(wanted) + r"\b", stmt, re.I)]
    if len(starts) != 1:
        raise SchemaError("procedure %s is %s" % (procedure, "missing" if not starts else "ambiguous"))
    header = statements[starts[0]]
    match = re.match(r"subroutine\s+\w+\s*\((.*)\)\s*$", header, re.I)
    if not match:
        raise SchemaError("unsupported procedure header")
    names = [value.strip().lower() for value in _split_top_level(match.group(1))]
    if not names or any(not re.fullmatch(r"[a-z]\w*", value, re.I) for value in names):
        raise SchemaError("unsupported dummy-argument syntax")
    if len(set(names)) != len(names):
        raise SchemaError("duplicate dummy argument")
    declared, standalone = {}, {}
    interface_depth, in_specification, found_end = 0, True, False
    for stmt in statements[starts[0] + 1:]:
        if in_specification and re.match(r"(?:abstract\s+)?interface\b", stmt, re.I):
            interface_depth += 1
            continue
        if in_specification and re.match(r"end\s*interface\b", stmt, re.I):
            if not interface_depth:
                raise SchemaError("unmatched end interface")
            interface_depth -= 1
            continue
        if interface_depth:
            continue
        if re.fullmatch(r"end\s+subroutine\s+" + re.escape(wanted), stmt, re.I):
            found_end = True
            break
        if re.match(r"contains\b", stmt, re.I):
            in_specification = False
            continue
        if not in_specification:
            continue
        try:
            parsed = _declaration(stmt)
        except SchemaError:
            parsed = []
        for name, entry in parsed:
            if name in names:
                declared.setdefault(name, []).append(entry)
        try:
            attributes = _standalone_attributes(stmt)
        except SchemaError:
            attributes = []
        for name, entry in attributes:
            if name in names:
                standalone.setdefault(name, []).append(entry)
    if interface_depth:
        raise SchemaError("unterminated interface")
    if not found_end:
        raise SchemaError("missing matched end subroutine " + procedure)
    arguments = []
    for index, name in enumerate(names):
        entries = declared.get(name, [])
        if not entries:
            entry = {"type": None, "kind": None, "intent": None, "optional": False,
                     "dimensions": [], "unsupported_reasons": ["missing_declaration"]}
        elif len(entries) != 1:
            entry = dict(entries[0])
            entry["unsupported_reasons"] = entry["unsupported_reasons"] + ["ambiguous_declaration"]
        else:
            entry = entries[0]
        for attribute in standalone.get(name, []):
            if attribute["intent"]:
                if entry["intent"] and entry["intent"] != attribute["intent"]:
                    entry["unsupported_reasons"] = entry["unsupported_reasons"] + ["ambiguous_intent"]
                else:
                    entry["intent"] = attribute["intent"]
            entry["optional"] = entry["optional"] or attribute["optional"]
            if attribute["dimensions"]:
                if entry["dimensions"] and entry["dimensions"] != attribute["dimensions"]:
                    entry["unsupported_reasons"] = entry["unsupported_reasons"] + ["ambiguous_dimensions"]
                else:
                    entry["dimensions"] = attribute["dimensions"]
            entry["unsupported_reasons"] = entry["unsupported_reasons"] + attribute["reasons"]
        entry = {"index": index, "name": name, **entry}
        entry["supported"] = not entry["unsupported_reasons"]
        arguments.append(entry)
    return {"schema_version": 1, "procedure": procedure,
            "source": {"path": os.path.abspath(source_path),
                       "sha256": hashlib.sha256(raw).hexdigest(),
                       "compiled_f90": True},
            "safe_supported": all(arg["supported"] for arg in arguments),
            "arguments": arguments}


def write_atomic(path, content):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, temporary = tempfile.mkstemp(prefix=".phase-schema-", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--procedure", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        manifest = extract(args.source, args.procedure)
    except (OSError, UnicodeDecodeError, SchemaError) as error:
        parser.error(str(error))
    write_atomic(args.output, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
