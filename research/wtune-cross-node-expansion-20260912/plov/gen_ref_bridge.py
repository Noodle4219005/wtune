#!/usr/bin/env python3
"""Generate a compiler-managed, pointer-only GNU --wrap reference bridge."""
import argparse
import re

from gen_phase_schema import SchemaError, extract, write_atomic


IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
ALLOWED_REASONS = {"missing_kind", "unresolved_kind"}
MAX_ARGUMENTS = 4096


def checked_identifier(value, label):
    if not IDENTIFIER.fullmatch(value):
        raise SchemaError(label + " must be a C identifier")
    return value


def generate(source, procedure, symbol, dispatch):
    checked_identifier(procedure, "procedure")
    symbol = checked_identifier(symbol, "symbol")
    dispatch = checked_identifier(dispatch, "dispatch")
    manifest = extract(source, procedure)
    arguments = manifest["arguments"]
    if not arguments or len(arguments) > MAX_ARGUMENTS:
        raise SchemaError("argument count must be between 1 and %d" % MAX_ARGUMENTS)
    for argument in arguments:
        if argument["type"] not in {"real", "integer", "logical"}:
            raise SchemaError("unsupported argument %s type" % argument["name"])
        if set(argument["unsupported_reasons"]) - ALLOWED_REASONS:
            raise SchemaError("unsupported argument %s: %s" %
                              (argument["name"], ",".join(argument["unsupported_reasons"])))
    n = len(arguments)
    formals = ", ".join("void *a%d" % i for i in range(n))
    forwarded = ", ".join("args[%d]" % i for i in range(n))
    local = ", ".join("a%d" % i for i in range(n))
    return "\n".join([
        "/* Generated pointer-only bridge; source_sha256=%s procedure=%s. */" %
        (manifest["source"]["sha256"], procedure),
        "#include <stddef.h>",
        "extern void __real_%s(%s);" % (symbol, formals),
        "extern void %s(void **args, void (*invoke)(void **), size_t nargs);" % dispatch,
        "static void wt_invoke_%s(void **args) { __real_%s(%s); }" %
        (symbol, symbol, forwarded),
        "void __wrap_%s(%s) {" % (symbol, formals),
        "  void *args[%d] = {%s};" % (n, local),
        "  %s(args, wt_invoke_%s, %d);" % (dispatch, symbol, n),
        "}",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--procedure", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--dispatch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = generate(args.source, args.procedure, args.symbol, args.dispatch)
    except (OSError, UnicodeDecodeError, SchemaError) as error:
        parser.error(str(error))
    write_atomic(args.output, output)


if __name__ == "__main__":
    main()
