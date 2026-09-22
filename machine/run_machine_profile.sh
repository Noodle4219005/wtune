#!/usr/bin/env bash
# run_machine_profile.sh -- build and run every machine/bench/* benchmark
# on the allocated node(s), then collect the results into
# machine_profile_<host>_<date>.json via machine/profile.py.
#
# Usable directly as (or from) a Slurm batch script -- it is ordinary
# bash with no embedded #SBATCH directives, so the caller supplies its
# own sbatch options and this stays reusable across allocation shapes:
#
#   sbatch -N 2 --ntasks-per-node=1 --wrap "FFLAGS='-O3 -march=native' \
#     /path/to/machine/run_machine_profile.sh"
#
# Compiler/MPI/launcher selection (env, all optional):
#   CC, FC, MPICC        default gcc, gfortran, mpicc
#   CFLAGS, FFLAGS        default -O3 -- pass WRF's OWN build flags here so
#                         the `kernels` numbers reflect WRF's real codegen,
#                         not a generic -O3 guess
#   MPIRUN                default mpirun
#   MPIRUN_ARGS           extra launcher args, e.g. "--bind-to core"
#
# Bench sizing/output (env, all optional -- see machine/bench/*.c|f90
# headers for what each knob means):
#   MP_FREQ_SECONDS, MP_STREAM_ITERS, MP_STREAM_LLC_BYTES, MP_OMP_SECONDS,
#   MP_KERNELS_NX/NY/NZ/NCALLS, MP_IO_DIR/SIZE_MB/WRITERS/BLOCK_KB,
#   MP_HALO_PATCHES/EVERT/NFIELDS/REPS/PINGPONG_MAX_BYTES, MP_HALO_INTRA_NP
#   MACHINE_PROFILE_OUT_DIR   where the merged JSON is written (default .)
#   MACHINE_PROFILE_TMP       scratch dir for build + raw output
#                             (default: a fresh mktemp -d, removed on exit
#                             unless MACHINE_PROFILE_KEEP_TMP=1)
#   PYTHON                    default python3

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/bench"

CC=${CC:-gcc}
FC=${FC:-gfortran}
MPICC=${MPICC:-mpicc}
CFLAGS=${CFLAGS:--O3}
FFLAGS=${FFLAGS:--O3}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_ARGS=${MPIRUN_ARGS:-}
PYTHON=${PYTHON:-python3}
OUT_DIR=${MACHINE_PROFILE_OUT_DIR:-.}
TMP_DIR=${MACHINE_PROFILE_TMP:-$(mktemp -d)}
KEEP_TMP=${MACHINE_PROFILE_KEEP_TMP:-0}

cleanup() { [[ "$KEEP_TMP" = "1" ]] || rm -rf "$TMP_DIR"; }
trap cleanup EXIT

mkdir -p "$TMP_DIR/build" "$TMP_DIR/raw" "$OUT_DIR"

HOST=$(hostname -s 2>/dev/null || hostname)
DATE=$(date +%F)
NODE_COUNT=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
CORES_PER_NODE=${SLURM_CPUS_ON_NODE:-$(nproc --all 2>/dev/null || echo 4)}

echo "== machine profile: host=$HOST date=$DATE nodes=$NODE_COUNT cores/node=$CORES_PER_NODE ==" >&2

# --- build -----------------------------------------------------------------
build_fail=()
try_build() {
    local label="$1"; shift
    if "$@" >"$TMP_DIR/build/$label.log" 2>&1; then
        echo "built $label" >&2
    else
        build_fail+=("$label")
        echo "BUILD FAILED: $label ($* ) -- see $TMP_DIR/build/$label.log" >&2
    fi
}

try_build freq    "$CC" $CFLAGS -fopenmp -Wall "$BENCH_DIR/freq.c"    -o "$TMP_DIR/build/freq"    -lm
try_build stream  "$CC" $CFLAGS -fopenmp -Wall "$BENCH_DIR/stream.c"  -o "$TMP_DIR/build/stream"  -lm
try_build omp     "$CC" $CFLAGS -fopenmp -Wall "$BENCH_DIR/omp.c"     -o "$TMP_DIR/build/omp"     -lm
try_build io      "$CC" $CFLAGS          -Wall "$BENCH_DIR/io.c"      -o "$TMP_DIR/build/io"
try_build kernels "$FC" $FFLAGS -fopenmp -Wall -J "$TMP_DIR/build" "$BENCH_DIR/kernels.f90" -o "$TMP_DIR/build/kernels"
try_build halo    "$MPICC" $CFLAGS -Wall "$BENCH_DIR/halo.c" -o "$TMP_DIR/build/halo" -lm

built() {
    for f in "${build_fail[@]}"; do [[ "$f" == "$1" ]] && return 1; done
    return 0
}

# --- metadata ----------------------------------------------------------------
META_JSON="$TMP_DIR/metadata.json"
CC="$CC" FC="$FC" MPIRUN="$MPIRUN" "$PYTHON" - "$META_JSON" "$NODE_COUNT" "$CORES_PER_NODE" <<'PYEOF'
import json, os, subprocess, sys

out_path, node_count, cores_per_node = sys.argv[1:4]

def cmd(*args, timeout=10):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception as exc:
        return "unavailable: {}".format(exc)

def first_line(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text

meta = {
    "uname": cmd("uname", "-a"),
    "lscpu": cmd("lscpu"),
    "cc_version": first_line(cmd(os.environ.get("CC", "gcc"), "--version")),
    "fc_version": first_line(cmd(os.environ.get("FC", "gfortran"), "--version")),
    "mpi_version": first_line(cmd(os.environ.get("MPIRUN", "mpirun"), "--version")),
    "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
    "node_count": int(node_count),
    "ncores_per_node": int(cores_per_node) if cores_per_node else None,
}
with open(out_path, "w") as f:
    json.dump(meta, f, indent=2)
PYEOF

# --- run ---------------------------------------------------------------------
RAW="$TMP_DIR/raw"
run_and_capture() {  # run_and_capture <bench-label> <output-file> <cmd...>
    local label="$1" outfile="$2"; shift 2
    if "$@" >"$outfile" 2>"$TMP_DIR/raw/$label.run.log"; then
        echo "ran $label -> $outfile" >&2
    else
        echo "RUN FAILED: $label -- see $TMP_DIR/raw/$label.run.log" >&2
        rm -f "$outfile"
    fi
}

if built freq; then
    run_and_capture freq "$RAW/freq.json" \
        "$TMP_DIR/build/freq" --seconds "${MP_FREQ_SECONDS:-1.0}"
else
    echo "SKIP freq: build failed" >&2
fi

if built stream; then
    stream_args=(--iters "${MP_STREAM_ITERS:-20}")
    [[ -n "${MP_STREAM_LLC_BYTES:-}" ]] && stream_args+=(--llc-bytes "$MP_STREAM_LLC_BYTES")
    run_and_capture stream "$RAW/stream.json" "$TMP_DIR/build/stream" "${stream_args[@]}"
else
    echo "SKIP stream: build failed" >&2
fi

if built omp; then
    run_and_capture omp "$RAW/omp.json" \
        "$TMP_DIR/build/omp" --seconds "${MP_OMP_SECONDS:-0.2}"
else
    echo "SKIP omp: build failed" >&2
fi

if built kernels; then
    run_and_capture kernels "$RAW/kernels.json" \
        "$TMP_DIR/build/kernels" \
        --nx "${MP_KERNELS_NX:-300}" --ny "${MP_KERNELS_NY:-300}" \
        --nz "${MP_KERNELS_NZ:-64}" --ncalls "${MP_KERNELS_NCALLS:-20}" \
        --threads-full "$CORES_PER_NODE"
else
    echo "SKIP kernels: build failed (need gfortran)" >&2
fi

if built io; then
    run_and_capture io "$RAW/io.json" \
        "$TMP_DIR/build/io" --dir "${MP_IO_DIR:-$OUT_DIR}" \
        --size-mb "${MP_IO_SIZE_MB:-1024}" --writers "${MP_IO_WRITERS:-$CORES_PER_NODE}" \
        --block-kb "${MP_IO_BLOCK_KB:-1024}"
else
    echo "SKIP io: build failed" >&2
fi

if built halo; then
    halo_args=(--evert "${MP_HALO_EVERT:-33}" --nfields "${MP_HALO_NFIELDS:-4}" \
                --reps "${MP_HALO_REPS:-20}" --patches "${MP_HALO_PATCHES:-20,50,100,200}" \
                --pingpong-max-bytes "${MP_HALO_PINGPONG_MAX_BYTES:-1048576}")
    intra_np=${MP_HALO_INTRA_NP:-$CORES_PER_NODE}
    run_and_capture halo_intra "$RAW/halo_intra.json" \
        $MPIRUN -np "$intra_np" $MPIRUN_ARGS "$TMP_DIR/build/halo" --tag intra "${halo_args[@]}"
    if [[ "$NODE_COUNT" -gt 1 ]]; then
        run_and_capture halo_inter "$RAW/halo_inter.json" \
            $MPIRUN -np "$NODE_COUNT" --map-by node $MPIRUN_ARGS \
            "$TMP_DIR/build/halo" --tag inter "${halo_args[@]}"
    else
        echo "SKIP halo inter-node: NODE_COUNT=1 (single-node allocation/local test)" >&2
    fi
else
    echo "SKIP halo: build failed (need mpicc)" >&2
fi

# --- merge + summarize ---------------------------------------------------
BENCH_FILES=()
for f in "$RAW"/*.json; do
    [[ -f "$f" ]] && BENCH_FILES+=("$f")
done

OUT_FILE="$OUT_DIR/machine_profile_${HOST}_${DATE}.json"
if [[ "${#BENCH_FILES[@]}" -eq 0 ]]; then
    echo "no bench produced output; nothing to merge" >&2
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/profile.py" merge "${BENCH_FILES[@]}" \
    --metadata "$META_JSON" --host "$HOST" --date "$DATE" --out "$OUT_FILE"
MERGE_RC=$?

echo >&2
"$PYTHON" "$SCRIPT_DIR/profile.py" summary "$OUT_FILE" >&2
echo >&2
echo "wrote $OUT_FILE" >&2

exit "$MERGE_RC"
