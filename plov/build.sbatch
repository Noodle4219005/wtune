#!/usr/bin/env bash
#SBATCH --job-name=wtimerbuild
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/u5727520/wrf-stack/wtimer/build-%j.out
# TREE selects which WRF tree to relink. dmsm has -qopenmp live; dmpar has it compiled out.
# Sources are compiled straight from the git checkout (T = wtune/plov); objects go to O.
# There is no second copy of the sources any more: what git shows is what gets linked.
set -uo pipefail
W=${TREE:-/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF}
T=${SRC:-/home/u5727520/wrf-stack/wtune/plov}
O=${WTIMER_OBJ_DIR:-/home/u5727520/wrf-stack/wtimer/obj}
OUTPUT=${WTIMER_OUTPUT:-/home/u5727520/wrf-stack/wtimer/wrf_wtimer.exe}
WT_PYTHON=${WTIMER_PYTHON:-python3}
# Refuse unsafe destinations before any compilation. The frozen tree is input only.
"$WT_PYTHON" - "$W" "$T" "$O" "$OUTPUT" <<'PY' || exit 1
from pathlib import Path
import sys
wrf, source, objects, output = map(Path, sys.argv[1:])
if not all(path.is_absolute() for path in (wrf, source, objects, output)):
    raise SystemExit("FATAL: TREE, SRC and output paths must be absolute")
frozen = wrf.resolve(strict=True)
for path in (objects, output):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise SystemExit("FATAL: object/output paths must not contain symlinks")
    try:
        path.resolve(strict=False).relative_to(frozen)
    except ValueError:
        pass
    else:
        raise SystemExit("FATAL: object/output path is inside frozen WRF tree")
for path in (output, Path(str(output) + ".link.json"), Path(str(output) + ".build")):
    if path.exists() or path.is_symlink():
        raise SystemExit("FATAL: output/provenance already exists: " + str(path))
if output.parent != objects and not output.parent.is_dir():
    raise SystemExit("FATAL: executable output parent does not exist")
PY
mkdir -p "$O" || exit 1
cd "$O" || exit 1
echo "sources: $T  commit: $(git -C "$T" rev-parse --short HEAD 2>/dev/null)  dirty: [$(git -C "$T" status --short -- . 2>/dev/null | tr '\n' ' ')]"
module load intel/2024_01_46 2>/dev/null || true
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

SYMS="module_microphysics_driver_mp_microphysics_driver_
module_radiation_driver_mp_radiation_driver_
module_radiation_driver_mp_pre_radiation_driver_
module_pbl_driver_mp_pbl_driver_
module_surface_driver_mp_surface_driver_
module_cumulus_driver_mp_cumulus_driver_
for_allocate_handle"
WRAPS=(); for s in $SYMS; do WRAPS+=(--wrap-symbol "$s"); done

echo "=== 1. assemble + compile the timer ==="
W=${TREE:-/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF}
CC=$(command -v icx || command -v gcc)
echo "using CC=$CC"
$CC -O2 -c "$T/wtimer.S" -o "$O/wtimer_asm.o" || { echo "FATAL asm"; exit 2; }
# Strip the comment BEFORE looking for -qopenmp: the dmpar trees carry the flag commented out
# as "OMP = # -qopenmp -fpp -auto", and matching the raw line compiled wtimer.c with OpenMP
# against a tree that links no OpenMP runtime -> undefined omp_get_max_threads.
OMPF=$(sed -n 's/^OMP *= *//p' "$W/configure.wrf" | head -1 | sed 's/#.*//' | grep -q -- -qopenmp && echo -qopenmp); echo "  wtimer.c OpenMP flag: [${OMPF:-none}]"; $CC -O2 $OMPF -c "$T/wtimer.c" -o "$O/wtimer_c.o"   || { echo "FATAL cc";  exit 2; }
MPICC=$(command -v mpiicx || command -v mpiicc || echo "$CC -I${I_MPI_ROOT:-/opt/intel}/include")
echo "  MPI cc: $MPICC"
$MPICC -O2 -c "$T/wtmpi.c" -o "$O/wtmpi_c.o" || { echo "FATAL mpicc"; exit 2; }
$CC -O2 $OMPF -c "$T/wtlewi.c" -o "$O/wtlewi_c.o" || { echo "FATAL lewi cc"; exit 2; }
$CC -O2 -c "$T/wt_call.S" -o "$O/wt_call.o"          || { echo "FATAL wt_call"; exit 2; }
$CC -O2 -I"$T" -c "$T/wt_micro.c" -o "$O/wt_micro.o" || { echo "FATAL wt_micro"; exit 2; }
$CC -O2 -I"$T" -c "$T/wt_rad.c" -o "$O/wt_rad.o"     || { echo "FATAL wt_rad"; exit 2; }
$CC -O2 -c "$T/wt_alloc.c" -o "$O/wt_alloc.o"          || { echo "FATAL wt_alloc"; exit 2; }
nm "$O/wtmpi_c.o" | grep -E " T MPI_" | sed 's/^/    /'
NM=$(nm "$O/wtmpi_c.o" | grep -c " T MPI_")
[ "$NM" -eq 7 ] || { echo "FATAL: $NM/7 MPI wrappers defined"; exit 3; }
nm "$O/wtimer_asm.o" | grep -E " T __wrap_" | sed 's/^/  /'
NW=$(nm "$O/wtimer_asm.o" | grep -c " T __wrap_")
[ "$NW" -eq 6 ] || { echo "FATAL: $NW/6 wrappers defined"; exit 3; }

echo "=== 2. link wrappers to a separate executable (frozen tree read-only) ==="
echo "  tree=$W  output=$OUTPUT"
OBJECTS=(); for object in wtimer_asm.o wtimer_c.o wtmpi_c.o wtlewi_c.o wt_call.o wt_micro.o wt_rad.o wt_alloc.o; do
  OBJECTS+=(--wrapper-object "$O/$object")
done
LINE=(); if [ -n "${WTIMER_LINK_LINE:-}" ]; then LINE=(--line "$WTIMER_LINK_LINE"); fi
"$WT_PYTHON" "$T/prepare_native_link.py" --compile-log "$W/compile.log" \
  --link-cwd "$W/main" --frozen-root "$W" --output "$OUTPUT" \
  --manifest "$OUTPUT.link.json" "${OBJECTS[@]}" "${WRAPS[@]}" \
  --extra-library rt "${LINE[@]}" || { echo "FATAL: link preparation refused"; exit 4; }
# Execute the reviewed argv directly, with no shell expansion and no make/rebuild.
"$WT_PYTHON" - "$OUTPUT.link.json" <<'PY' || exit 4
import json
from pathlib import Path
import subprocess
import sys
with open(sys.argv[1]) as stream:
    manifest = json.load(stream)
output = Path(manifest["output"])
if output.exists() or output.is_symlink():
    raise SystemExit("FATAL: link output appeared after preparation")
subprocess.run(manifest["argv"], cwd=manifest["link_cwd"], check=True)
PY
[ -s "$OUTPUT" ] || { echo "FATAL: link produced nothing"; exit 4; }
ls -la "$OUTPUT"
# provenance stamp next to the binary: which sources, which commit, which tree
{ echo "built=$(date -Is) job=${SLURM_JOB_ID:-local} tree=$W src=$T commit=$(git -C "$T" rev-parse HEAD 2>/dev/null)"
  git -C "$T" status --short -- . 2>/dev/null | sed 's/^/dirty: /'
  (cd "$T" && sha256sum wtimer.S wt_call.S *.c *.h); } > "$OUTPUT.build"

echo "=== 3. G0 DISPATCH GATE ==="
# Delegated to gate0.sh. The version that used to live here reported a false FAIL twice: it
# disassembled module_em_mp_solve_em_ (solve_em is not in a module) and matched "call " when
# this objdump prints "callq". gate0.sh proves it can read before it reports.
NA=$(objdump -d "$OUTPUT" | grep -c "call.*<__wrap_for_allocate_handle>")
echo "  for_allocate_handle wrapped call sites: $NA (allocs_8.o alone has 510)"
[ "$NA" -gt 500 ] || { echo "FATAL: allocator wrap did not fire"; exit 4; }
exec "$T/gate0.sh" "$OUTPUT"
