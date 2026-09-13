#!/usr/bin/env python3
"""Render the SFCLAY transaction metadata from the frozen audit JSON."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sfclay_schema.json"
TARGET = ROOT / "wt_sfclay_schema.h"
AUDIT_SHA256 = "348941418e6ec617a0f87a6de4f3a5e201b3646a8919d29ed4f92ffabc654c50"


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUDIT_SHA256:
        raise SystemExit("sfclay schema audit SHA-256 mismatch")
    data = json.loads(raw)
    args = data.get("arguments")
    if data.get("argument_count") != 84 or not isinstance(args, list) or len(args) != 84:
        raise SystemExit("expected exactly 84 audited arguments")
    shapes = {"scalar": "WT_SFCLAY_SHAPE_SCALAR", "full_2d": "WT_SFCLAY_SHAPE_2D",
              "full_3d": "WT_SFCLAY_SHAPE_3D"}
    directions = {"IN": "WT_SFCLAY_READONLY", "INOUT": "WT_SFCLAY_WRITABLE",
                  "OUT": "WT_SFCLAY_WRITABLE"}
    if any(a.get("index") != i + 1 or a.get("width_bytes") != 4 or
           a.get("shape") not in shapes or a.get("direction") not in directions
           for i, a in enumerate(args)):
        raise SystemExit("audit ordering, width, shape, or direction invalid")
    lines = [
        "#ifndef WT_SFCLAY_SCHEMA_H", "#define WT_SFCLAY_SCHEMA_H", "",
        "#define WT_SFCLAY_NARG 84", "#define WT_SFCLAY_AUDIT_SHA256 \"" + AUDIT_SHA256 + "\"", "",
        "enum wt_sfclay_shape { WT_SFCLAY_SHAPE_SCALAR, WT_SFCLAY_SHAPE_2D, WT_SFCLAY_SHAPE_3D };",
        "enum wt_sfclay_access { WT_SFCLAY_READONLY, WT_SFCLAY_WRITABLE };",
        "static const unsigned char wt_sfclay_schema_shape[WT_SFCLAY_NARG] = {",
        "  " + ", ".join(shapes[a["shape"]] for a in args), "};",
        "static const unsigned char wt_sfclay_schema_access[WT_SFCLAY_NARG] = {",
        "  " + ", ".join(directions[a["direction"]] for a in args), "};",
        "static const unsigned char wt_sfclay_schema_optional[WT_SFCLAY_NARG] = {",
        "  " + ", ".join("1" if a.get("optional") else "0" for a in args), "};",
        "static const char *const wt_sfclay_schema_name[WT_SFCLAY_NARG] = {",
        "  " + ", ".join(json.dumps(a["formal"]) for a in args), "};", "", "#endif", "",
    ]
    TARGET.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
