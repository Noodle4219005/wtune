#!/usr/bin/env python3
"""Generate the checked 71-reference RK scalar schema header from its proof."""
import argparse, hashlib, json
from pathlib import Path

SCHEMA_SHA256 = "18cd06a09574f03f2d7a9bb8284fb7aebe47a63f810064073635589288e1785a"
SOURCE_SHA256 = "32c1b72af58dbefd0115481352529c35ecaa9ddfd45252d829c2a48d25af0944"
KIND = {"integer": "WT_RK_INTEGER", "logical": "WT_RK_LOGICAL", "real": "WT_RK_REAL",
        "derived:grid_config_rec_type": "WT_RK_OPAQUE"}
SHAPE = {"scalar": "WT_RK_SHAPE_SCALAR", "(kms:kme)": "WT_RK_SHAPE_VECTOR",
         "(ims:ime,jms:jme)": "WT_RK_SHAPE_PLANE",
         "(ims:ime,kms:kme,jms:jme)": "WT_RK_SHAPE_VOLUME",
         "(ims:ime,kms:kme,jms:jme,scs:sce)": "WT_RK_SHAPE_SPECIES_VOLUME"}
WRITABLE = {"wwE", "wwI", "scalar_tends", "advect_tend", "h_tendency", "z_tendency", "RQVFTEN"}

def load(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCHEMA_SHA256: raise ValueError("immutable RK schema SHA-256 differs")
    doc = json.loads(raw)
    if doc.get("argument_count") != 71 or doc.get("source", {}).get("sha256") != SOURCE_SHA256: raise ValueError("RK schema source/profile mismatch")
    args = doc.get("arguments")
    if not isinstance(args, list) or len(args) != 71: raise ValueError("RK schema must contain 71 arguments")
    for i, arg in enumerate(args, 1):
        if set(arg) != {"index","name","type","kind","intent","optional","dimensions"} or arg["index"] != i:
            raise ValueError("RK schema argument ordering/profile is invalid")
        if arg["type"] not in KIND or arg["dimensions"] not in SHAPE or arg["kind"] != "default" or arg["optional"]:
            raise ValueError("RK schema has unsupported kind, shape, or optional formal")
    if args[2]["name"] != "config_flags" or args[2]["type"] != "derived:grid_config_rec_type": raise ValueError("opaque config position changed")
    return args

def emit(args):
    lines = ["#ifndef WT_RK_SCHEMA_H", "#define WT_RK_SCHEMA_H", "", "#include <stddef.h>", "",
             "#define WT_RK_NARG 71", "#define WT_RK_SCHEMA_SHA256 \"" + SCHEMA_SHA256 + "\"",
             "enum wt_rk_type { WT_RK_INTEGER, WT_RK_LOGICAL, WT_RK_REAL, WT_RK_OPAQUE };",
             "enum wt_rk_shape { WT_RK_SHAPE_SCALAR, WT_RK_SHAPE_VECTOR, WT_RK_SHAPE_PLANE, WT_RK_SHAPE_VOLUME, WT_RK_SHAPE_SPECIES_VOLUME };",
             "struct wt_rk_arg_metadata { const char *name; unsigned char type, shape, writable; };", "",
             "enum wt_rk_arg {"]
    lines += ["    WT_RK_%s = %d," % (a["name"].upper(), i) for i,a in enumerate(args)]
    lines += ["};", "", "#define WT_RK_SCHEMA_ARGS(X) \\"]
    for i,a in enumerate(args):
        tail = " " + chr(92) if i + 1 < len(args) else ""
        lines.append(" X(%s, \"%s\", %s, %s, %d)%s" % (a["name"].upper(), a["name"], KIND[a["type"]], SHAPE[a["dimensions"]], int(a["name"] in WRITABLE), tail))
    lines += ["", "extern const struct wt_rk_arg_metadata wt_rk_args[WT_RK_NARG];", "#endif", ""]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--schema", required=True); p.add_argument("--output", required=True); a=p.parse_args()
    try: Path(a.output).write_text(emit(load(a.schema)))
    except (OSError, ValueError, json.JSONDecodeError) as e: p.error(str(e))
if __name__ == "__main__": main()
