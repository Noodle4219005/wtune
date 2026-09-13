#!/usr/bin/env python3
"""Generate the GNU pointer-reference test bridge for audited SFCLAY."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SOURCE_SHA256 = "dfd7aae5ef1f0bd979246480af5276a1ee956bc216ec8dc6e82eb963ded761dc"
SCHEMA_SHA256 = "348941418e6ec617a0f87a6de4f3a5e201b3646a8919d29ed4f92ffabc654c50"
NARGS = 84
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_identifier(value):
    if not IDENTIFIER.fullmatch(value):
        raise ValueError("symbol must be a C identifier")
    return value


def generate(source, schema, symbol):
    if digest(source) != SOURCE_SHA256:
        raise ValueError("source SHA-256 does not match audited SFCLAY")
    if digest(schema) != SCHEMA_SHA256:
        raise ValueError("schema SHA-256 does not match audited SFCLAY")
    arguments = json.loads(Path(schema).read_text()).get("arguments", [])
    if len(arguments) != NARGS or [item.get("index") for item in arguments] != list(range(1, NARGS + 1)):
        raise ValueError("schema must contain the ordered audited 84 arguments")
    symbol = checked_identifier(symbol)
    formals = ", ".join("void *a%d" % i for i in range(NARGS))
    argv = ", ".join("a%d" % i for i in range(NARGS))
    forwarded = ", ".join("args[%d]" % i for i in range(NARGS))
    return "\n".join((
        "/* Generated compiler-managed bridge: source_sha256=%s schema_sha256=%s. */" %
        (SOURCE_SHA256, SCHEMA_SHA256),
        "#include <stddef.h>",
        "#ifdef WT_SFCLAY_BRIDGE_TEST_HELPERS",
        "#include <stdatomic.h>",
        "#endif",
        "extern void %s(%s);" % (symbol, formals),
        "extern void __real_%s(%s);" % (symbol, formals),
        "extern void wt_sfclay_bridge_dispatch(void **, size_t, void (*)(void **));",
        "static void wt_sfclay_real(void **args) { __real_%s(%s); }" % (symbol, forwarded),
        "#ifdef WT_SFCLAY_BRIDGE_TEST_HELPERS",
        "static _Atomic size_t wt_sfclay_wrapped_calls;",
        "void wt_sfclay_bridge_real(void **args) { wt_sfclay_real(args); }",
        "void wt_sfclay_bridge_call(void **args) { %s(%s); }" % (symbol, forwarded),
        "size_t wt_sfclay_bridge_wrapped_calls(void) { return atomic_load_explicit(&wt_sfclay_wrapped_calls, memory_order_relaxed); }",
        "#endif",
        "void __wrap_%s(%s) {" % (symbol, formals),
        "  void *args[%d] = {%s};" % (NARGS, argv),
        "#ifdef WT_SFCLAY_BRIDGE_TEST_HELPERS",
        "  atomic_fetch_add_explicit(&wt_sfclay_wrapped_calls, 1, memory_order_relaxed);",
        "#endif",
        "  wt_sfclay_bridge_dispatch(args, %d, wt_sfclay_real);" % NARGS,
        "}", ""))


def atomic_write(path, contents):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        atomic_write(args.output, generate(args.source, args.schema, args.symbol))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
