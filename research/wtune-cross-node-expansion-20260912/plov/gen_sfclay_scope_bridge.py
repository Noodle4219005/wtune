#!/usr/bin/env python3
"""Emit an audited 84-reference first-RK scope wrapper; not native ABI certification."""
import argparse
import hashlib
import json
from pathlib import Path
import re

SOURCE_SHA256 = "4668213aaec35d72f7bdbd2473d810b4538370f8aadf5979ef940f315be9c43b"
DEFAULT_SYMBOL = "module_first_rk_step_part1_mp_first_rk_step_part1_"
NAMES = tuple("""grid config_flags moist moist_tend chem chem_tend tracer tracer_tend
scalar scalar_tend fdda3d fdda2d aerod ru_tendf rv_tendf rw_tendf t_tendf
ph_tendf mu_tendf tke_tend adapt_step_flag curr_secs curr_mins2 psim psih
gz1oz0 chklowq cu_act_flag hol th_phy pi_phy p_phy t_phy dz8w p8w t8w
ids ide jds jde kds kde ims ime jms jme kms kme ips ipe jps jpe kps kpe
imsx imex jmsx jmex kmsx kmex ipsx ipex jpsx jpex kpsx kpex
imsy imey jmsy jmey kmsy kmey ipsy ipey jpsy jpey kpsy kpey
k_start k_end f_flux aerocu restart_flag feedback_is_ready""".split())


def expected_profile():
    entries = {name: dict(name=name, type="integer", intent="in", shape=[], optional=False)
               for name in NAMES}
    def set_fields(names, kind, intent="in", shape=(), optional=False):
        for name in names.split():
            entries[name] = dict(name=name, type=kind, intent=intent,
                                 shape=list(shape), optional=optional)
    set_fields("grid", "type(domain)", "inout")
    set_fields("config_flags", "type(grid_config_rec_type)")
    set_fields("adapt_step_flag", "logical")
    set_fields("curr_secs curr_mins2", "real")
    for name in "moist chem tracer scalar fdda3d fdda2d aerod aerocu".split():
        shape = ("ims:ime", "1:1" if name == "fdda2d" else "kms:kme", "jms:jme", "num_" + name)
        set_fields(name, "real", "inout", shape, name == "aerocu")
        if name in ("moist", "chem", "tracer", "scalar"):
            set_fields(name + "_tend", "real", "inout", shape)
    set_fields("ru_tendf rv_tendf rw_tendf t_tendf ph_tendf tke_tend th_phy pi_phy p_phy t_phy dz8w p8w t8w",
               "real", "inout", ("ims:ime", "kms:kme", "jms:jme"))
    set_fields("mu_tendf psim psih gz1oz0 chklowq hol", "real", "inout", ("ims:ime", "jms:jme"))
    set_fields("cu_act_flag", "logical", "inout", ("ims:ime", "jms:jme"))
    set_fields("f_flux restart_flag feedback_is_ready", "logical", optional=True)
    return [entries[name] for name in NAMES]


def profile_digest(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


PROFILE_SHA256 = "545d198388690ca64d46e4aa8a0fabf565d4e176bf02dd93f72c2ab9fb0de7cf"
if profile_digest(expected_profile()) != PROFILE_SHA256:
    raise RuntimeError("declaration profile SHA-256 pin differs from audited profile")


def comma_parts(text):
    depth, start, out = 0, 0, []
    for i, char in enumerate(text):
        if char == "(": depth += 1
        elif char == ")": depth -= 1
        elif char == "," and depth == 0:
            out.append(text[start:i].strip()); start = i + 1
        if depth < 0: raise ValueError("unbalanced parentheses")
    if depth: raise ValueError("unclosed parentheses")
    out.append(text[start:].strip())
    return out


def statements(text):
    pending = ""
    for raw in text.splitlines():
        # The audited specification has no quoted strings. Stop before execution.
        line = raw.split("!", 1)[0].strip()
        if not line: continue
        if ";" in line: raise ValueError("unresolved semicolon statement")
        if line.startswith("&"): line = line[1:].lstrip()
        continuation = line.endswith("&")
        if continuation: line = line[:-1]
        pending += " " + line
        if not continuation:
            yield pending.strip().lower()
            pending = ""
    if pending: raise ValueError("truncated continued signature or declaration")


def audit_text(text):
    """Complete signature and per-formal declaration proof, independent of file hash."""
    stream = iter(statements(text))
    names = None
    for statement in stream:
        if statement.startswith("subroutine first_rk_step_part1"):
            match = re.fullmatch(r"subroutine\s+first_rk_step_part1\s*\((.*)\)", statement)
            if not match: raise ValueError("missing complete first-RK signature closing parenthesis")
            names = tuple(comma_parts(match[1]))
            break
    if names != NAMES or len(names) != 84 or len(set(names)) != 84:
        raise ValueError("complete ordered 84-formal signature differs from audited profile")
    found, implicit_none, reached_execution = {}, False, False
    for statement in stream:
        if statement == "implicit none": implicit_none = True
        if re.match(r"call\s+get_ijk_from_subgrid\b", statement):
            reached_execution = True
            break
        if "::" not in statement: continue
        left, right = statement.split("::", 1)
        entities = comma_parts(right)
        targets = [item for item in entities if re.match(r"[a-z_]\w*", item) and
                   re.match(r"[a-z_]\w*", item)[0] in names]
        if not targets: continue
        attrs = [re.sub(r"\s+", "", x) for x in comma_parts(left)]
        kind, rest = attrs[0], attrs[1:]
        if kind not in ("real", "integer", "logical", "type(domain)", "type(grid_config_rec_type)"):
            raise ValueError("unsupported formal type: " + kind)
        intent, shape, optional = None, [], False
        seen = set()
        for attr in rest:
            key = attr.split("(")[0]
            if key in seen: raise ValueError("duplicate declaration attribute")
            seen.add(key)
            if re.fullmatch(r"intent\((in|out|inout)\)", attr): intent = attr[7:-1]
            elif attr == "optional": optional = True
            elif attr.startswith("dimension(") and attr.endswith(")"):
                shape = comma_parts(attr[10:-1])
                bound = r"(?:[a-z_]\w*|[0-9]+)"
                if not shape or any(not re.fullmatch(bound + r"(?::" + bound + r")?", x) for x in shape):
                    raise ValueError("unproven explicit-shape extent")
            else: raise ValueError("unsupported formal attribute: " + attr)
        if not intent: raise ValueError("missing formal INTENT")
        for name in targets:
            if not re.fullmatch(r"[a-z_]\w*", name): raise ValueError("unsupported entity declaration")
            if name in found: raise ValueError("duplicate formal declaration")
            if kind.startswith("type(") and (name, kind) not in (
                    ("grid", "type(domain)"), ("config_flags", "type(grid_config_rec_type)")):
                raise ValueError("derived-reference whitelist mismatch")
            found[name] = dict(name=name, type=kind, intent=intent, shape=shape, optional=optional)
    if not implicit_none or not reached_execution or set(found) != set(names):
        raise ValueError("incomplete specification or missing formal declarations")
    profile = [found[name] for name in names]
    if profile != expected_profile() or profile_digest(profile) != PROFILE_SHA256:
        raise ValueError("formal declaration profile differs from audited profile")
    return profile


def generate(source, symbol=DEFAULT_SYMBOL):
    data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("compiled source SHA-256 differs from audited first-RK source")
    audit_text(data.decode())
    if not re.fullmatch(r"[A-Za-z_]\w*", symbol, flags=re.ASCII):
        raise ValueError("symbol must be a C identifier")
    formals = ", ".join("void *a%d" % i for i in range(84))
    actuals = ", ".join("a%d" % i for i in range(84))
    pointer_actuals = ", ".join("args[%d]" % i for i in range(84))
    return "\n".join((
        "/* Generated compiler-managed first-RK scope: 84 raw references.",
        " * compiled_source_sha256=" + SOURCE_SHA256,
        " * declaration_profile_sha256=" + PROFILE_SHA256,
        " * This audited prototype is not certification of a native WRF binary ABI. */",
        '#include <stdint.h>',
        '#if !defined(WT_MYNN_NODE_LOCAL) || defined(WT_SFCLAY_NODE_LOCAL)',
        '#include "wt_sfclay_live.h"',
        '#endif',
        '#ifdef WT_MYNN_NODE_LOCAL',
        '#include "wt_mynn_node_live.h"',
        '#endif',
        "extern void __real_%s(%s);" % (symbol, formals),
        "void __wrap_%s(%s) {" % (symbol, formals),
        '#if !defined(WT_MYNN_NODE_LOCAL) || defined(WT_SFCLAY_NODE_LOCAL)',
        "    uint64_t token = wt_sfclay_first_rk_begin(a0, a1);",
        '#endif',
        '#ifdef WT_MYNN_NODE_LOCAL',
        "    uint64_t mynn_token = wt_mynn_first_rk_begin(a0, a1);",
        '#endif',
        "    __real_%s(%s);" % (symbol, actuals),
        '#ifdef WT_MYNN_NODE_LOCAL',
        "    wt_mynn_first_rk_end(mynn_token);",
        '#endif',
        '#if !defined(WT_MYNN_NODE_LOCAL) || defined(WT_SFCLAY_NODE_LOCAL)',
        "    wt_sfclay_first_rk_end(token);",
        '#endif',
        "}",
        "#ifdef WT_SFCLAY_SCOPE_TEST_HELPERS",
        "extern void %s(%s);" % (symbol, formals),
        "void wt_sfclay_scope_test_call(void **args) { %s(%s); }" % (symbol, pointer_actuals),
        "#endif", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = generate(args.source, args.symbol)
        Path(args.output).write_text(result)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
