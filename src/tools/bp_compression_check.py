#!/usr/bin/env python3
"""Report whether an ADIOS2 BP directory's variables actually carry a
compression operator, so a campaign can ASSERT compression instead of hoping.

Why this exists
---------------
WRF's ADIOS2 backend treats a failed compression setup as non-fatal. In
`external/io_adios2/wrf_io.F90` every failure on the compression path is a
`wrf_debug(WARN, ...)` followed by `return`, so a run whose blosc operator was
never attached writes fully uncompressed BP and says nothing about it. Two
separate mechanisms have already been caught doing exactly that:

  * the global-operator-name collision (wiki note
    `adios2-compression-only-reaches-the-first-stream-to-open`, 2026-08-20):
    `'Compressor'` is registered on the shared `adios` object but guarded
    per-DataHandle, so the second stream to open got no operator at all --
    a measured wrfrst stored 33.32 GiB against 33.11 GiB of logical data.
  * in the WTune build
    (`/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF`), `wrf_io.F90:1283-1285`
    assigns `DH%blosc_compressor` only on the branch that DEFINES the operator.
    A later stream inquires the operator successfully, so `%valid` is true, but
    its `DH%blosc_compressor` is blank and the codec if/elseif chain at
    `wrf_io.F90:2497-2508` matches nothing -- `adios2_add_operation` is never
    called. Measured on conus12 (job 1082176): frame 1 stored 166.7 MB with
    `adios2_blosc_compressor="zstd"`, frames 2 and 3 stored 595,875,392 bytes
    each, byte-for-byte the uncompressed control.

How the check works
-------------------
Neither `bpls` nor the BP metadata spells the codec out: on this ADIOS2
(2.10.2, `+blosc2`) a `strings` sweep of `md.idx`/`md.0`/`mmd.0` finds zero
occurrences of "blosc", "zstd" or "lz4" whether or not the variables are
compressed, and `bpls -l -a --hidden_attrs` and `bpls -D` carry no
operator/transform keyword either. So the detectable signal is the one that
matters operationally anyway: stored bytes against logical bytes.

  logical bytes  -- parsed out of `bpls -l`: for every VARIABLE line,
                    prod(shape) * sizeof(type). (Attribute lines are skipped.)
  stored bytes   -- the size of the BP directory's `data.*` subfiles, which is
                    what the filesystem actually paid for.

On an uncompressed conus12 wrfout the two agree to 0.06% (595,501,316 logical
against 595,875,392 stored; the excess is BP block-header overhead), so a
threshold anywhere above ~1.05 separates "compressed" from "not compressed"
with a wide margin. Measured conus12 ratios: zstd 3.57x, lz4 3.29x, off 1.00x.

Usage
-----
    bp_compression_check.py wrfout_d01_2019-11-26_12:00:00 [more BP dirs ...]
    bp_compression_check.py --json rundir/wrfout_d01_*
    bp_compression_check.py --min-ratio 2.0 rundir/wrfout_d01_*

Exit status is 0 only when EVERY BP directory examined has a ratio at or above
--min-ratio, so a campaign can put this straight after its WRF srun.
"""
import argparse
import json
import os
import re
import subprocess
import sys

DEFAULT_BPLS = "/home/u5727520/wrf-stack/intel/bin/bpls"

# Ratio at or above which a BP directory counts as compressed. An uncompressed
# BP lands at 0.999 (stored is fractionally LARGER than logical because of BP
# block headers), and the weakest codec measured lands at 3.29, so 1.05 is a
# deliberately loose floor rather than a tuned one.
DEFAULT_MIN_RATIO = 1.05

# `bpls -l` type names -> bytes per element. `string` variables are skipped
# rather than sized: their on-disk length is not implied by the shape.
TYPE_SIZES = {
    "char": 1,
    "int8_t": 1,
    "uint8_t": 1,
    "int16_t": 2,
    "uint16_t": 2,
    "int32_t": 4,
    "uint32_t": 4,
    "int64_t": 8,
    "uint64_t": 8,
    "float": 4,
    "double": 8,
    "complex": 8,
    "double complex": 16,
}

# A variable line looks like
#     "  float    QVAPOR    {49, 299, 424} = 1e-06 / 0.0173528"
# An attribute line carries the literal " attr " before the "=" and is skipped
#     "  string   QVAPOR/units    attr   = {"kg kg-1"}"
_VAR_LINE = re.compile(r"^\s+(\S+(?:\s+complex)?)\s+(\S+)\s+\{([^}]*)\}\s*(?:=.*)?$")


def parse_bpls_listing(text):
    """Parse `bpls -l` output into a list of (name, type, shape) variables.

    Attribute lines are dropped. Scalars print as `{1}` or with no braces at
    all; the brace-less form is dropped too (it contributes nothing measurable
    and never dominates a WRF output).
    """
    out = []
    for line in text.splitlines():
        if " attr " in line:
            continue
        m = _VAR_LINE.match(line)
        if not m:
            continue
        vtype, name, dims = m.group(1), m.group(2), m.group(3)
        shape = [int(tok) for tok in dims.replace(" ", "").split(",") if tok]
        out.append((name, vtype, shape))
    return out


def logical_bytes(variables):
    """Uncompressed size implied by the variable shapes and element types.

    Returns (total_bytes, skipped) where `skipped` lists (name, type) for every
    variable whose type has no fixed element size (`string`) or is unknown.
    """
    total = 0
    skipped = []
    for name, vtype, shape in variables:
        size = TYPE_SIZES.get(vtype)
        if size is None:
            skipped.append((name, vtype))
            continue
        n = size
        for dim in shape:
            n *= dim
        total += n
    return total, skipped


def stored_bytes(bp_dir):
    """Bytes the filesystem actually holds for this BP directory's payload.

    Only `data.*` subfiles count: `md.*`/`mmd.*` are metadata and their size
    moves with the operator declaration, which would contaminate the ratio.
    """
    total = 0
    for entry in os.listdir(bp_dir):
        if entry.startswith("data."):
            total += os.path.getsize(os.path.join(bp_dir, entry))
    return total


def run_bpls(bp_dir, bpls=DEFAULT_BPLS):
    """Return `bpls -l` text for bp_dir, or raise RuntimeError."""
    exe = bpls if os.path.exists(bpls) else "bpls"
    try:
        proc = subprocess.Popen([exe, "-l", bp_dir],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()
    except OSError as exc:
        raise RuntimeError("cannot run %s: %s" % (exe, exc))
    if proc.returncode != 0:
        raise RuntimeError("%s -l %s failed (rc=%d): %s"
                           % (exe, bp_dir, proc.returncode,
                              err.decode("utf-8", "replace").strip()))
    return out.decode("utf-8", "replace")


def check_bp_dir(bp_dir, min_ratio=DEFAULT_MIN_RATIO, bpls=DEFAULT_BPLS):
    """Examine one BP directory. Returns a dict; never raises for a bad ratio."""
    result = {"path": bp_dir, "compressed": False, "ratio": None,
              "logical_bytes": None, "stored_bytes": None,
              "n_variables": None, "skipped_types": [], "error": None}
    if not os.path.isdir(bp_dir):
        result["error"] = "not a directory"
        return result
    try:
        listing = run_bpls(bp_dir, bpls=bpls)
    except RuntimeError as exc:
        result["error"] = str(exc)
        return result
    variables = parse_bpls_listing(listing)
    logical, skipped = logical_bytes(variables)
    stored = stored_bytes(bp_dir)
    result["n_variables"] = len(variables)
    result["logical_bytes"] = logical
    result["stored_bytes"] = stored
    result["skipped_types"] = sorted(set(t for _, t in skipped))
    if stored == 0:
        result["error"] = "no data.* subfiles found"
        return result
    if logical == 0:
        result["error"] = "bpls listed no sizeable variables"
        return result
    ratio = float(logical) / float(stored)
    result["ratio"] = ratio
    result["compressed"] = ratio >= min_ratio
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="bp_compression_check.py",
        description="Report whether ADIOS2 BP directories are actually compressed.")
    ap.add_argument("bp_dirs", nargs="+", metavar="BP_DIR")
    ap.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO,
                    help="logical/stored ratio at or above which a BP counts as "
                         "compressed (default %(default)s; uncompressed measures "
                         "1.00, lz4 3.29, zstd 3.57 on conus12 wrfout)")
    ap.add_argument("--bpls", default=DEFAULT_BPLS,
                    help="bpls executable (default %(default)s; falls back to "
                         "whatever is on PATH if that path does not exist)")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON object per line instead of a table")
    a = ap.parse_args(argv)

    failures = 0
    for bp_dir in a.bp_dirs:
        r = check_bp_dir(bp_dir.rstrip("/"), min_ratio=a.min_ratio, bpls=a.bpls)
        if a.json:
            print(json.dumps(r, sort_keys=True))
        elif r["error"]:
            print("ERROR      %s: %s" % (r["path"], r["error"]))
        else:
            print("%-10s %s  ratio=%.2fx  logical=%d  stored=%d  vars=%d"
                  % ("COMPRESSED" if r["compressed"] else "UNCOMPRESSED",
                     r["path"], r["ratio"], r["logical_bytes"],
                     r["stored_bytes"], r["n_variables"]))
        if r["error"] or not r["compressed"]:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
