#!/usr/bin/env bash
#SBATCH --job-name=wtimerbuild
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/u5727520/wrf-stack/wtimer/build-%j.out
# TREE selects which WRF tree to relink. dmsm has -qopenmp live; dmpar has it compiled out.
set -uo pipefail
W=${TREE:-/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF}
T=/home/u5727520/wrf-stack/wtimer
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
WRAPS=""; for s in $SYMS; do WRAPS="$WRAPS -Wl,--wrap=$s"; done

echo "=== 1. assemble + compile the timer ==="
W=${TREE:-/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF}
CC=$(command -v icx || command -v gcc)
echo "using CC=$CC"
$CC -O2 -c "$T/wtimer.S" -o "$T/wtimer_asm.o" || { echo "FATAL asm"; exit 2; }
# Strip the comment BEFORE looking for -qopenmp: the dmpar trees carry the flag commented out
# as "OMP = # -qopenmp -fpp -auto", and matching the raw line compiled wtimer.c with OpenMP
# against a tree that links no OpenMP runtime -> undefined omp_get_max_threads.
OMPF=$(sed -n 's/^OMP *= *//p' "$W/configure.wrf" | head -1 | sed 's/#.*//' | grep -q -- -qopenmp && echo -qopenmp); echo "  wtimer.c OpenMP flag: [${OMPF:-none}]"; $CC -O2 $OMPF -c "$T/wtimer.c" -o "$T/wtimer_c.o"   || { echo "FATAL cc";  exit 2; }
MPICC=$(command -v mpiicx || command -v mpiicc || echo "$CC -I${I_MPI_ROOT:-/opt/intel}/include")
echo "  MPI cc: $MPICC"
$MPICC -O2 -c "$T/wtmpi.c" -o "$T/wtmpi_c.o" || { echo "FATAL mpicc"; exit 2; }
$CC -O2 $OMPF -c "$T/wtlewi.c" -o "$T/wtlewi_c.o" || { echo "FATAL lewi cc"; exit 2; }
$CC -O2 -c "$T/wt_call.S" -o "$T/wt_call.o"          || { echo "FATAL wt_call"; exit 2; }
$CC -O2 -I"$T" -c "$T/wt_micro.c" -o "$T/wt_micro.o" || { echo "FATAL wt_micro"; exit 2; }
$CC -O2 -I"$T" -c "$T/wt_rad.c" -o "$T/wt_rad.o"     || { echo "FATAL wt_rad"; exit 2; }
$CC -O2 -c "$T/wt_alloc.c" -o "$T/wt_alloc.o"          || { echo "FATAL wt_alloc"; exit 2; }
nm "$T/wtmpi_c.o" | grep -E " T MPI_" | sed 's/^/    /'
NM=$(nm "$T/wtmpi_c.o" | grep -c " T MPI_")
[ "$NM" -eq 7 ] || { echo "FATAL: $NM/7 MPI wrappers defined"; exit 3; }
nm "$T/wtimer_asm.o" | grep -E " T __wrap_" | sed 's/^/  /'
NW=$(nm "$T/wtimer_asm.o" | grep -c " T __wrap_")
[ "$NW" -eq 6 ] || { echo "FATAL: $NW/6 wrappers defined"; exit 3; }

LDL=$(sed -n 's/^LDFLAGS_LOCAL *= *//p' "$W/configure.wrf" | head -1 | sed 's/#.*//' | xargs)
echo "=== 2. relink wrf.exe with the wrappers (tree config untouched) ==="
echo "  tree=$W"; echo "  LDFLAGS_LOCAL from that tree: [$LDL]"
cd "$W/main" || exit 1
cp -p wrf.exe wrf.exe.orig
make SOLVER=em WRF_SRC_ROOT_DIR="$W" em_wrf \
  LDFLAGS_LOCAL="$LDL" \
  LIB_LOCAL="$WRAPS -Wl,--start-group $T/wtimer_asm.o $T/wtimer_c.o $T/wtmpi_c.o $T/wtlewi_c.o $T/wt_call.o $T/wt_micro.o $T/wt_rad.o $T/wt_alloc.o $W/main/libwrflib.a -lrt -Wl,--end-group" 2>&1 | tail -14
[ -s wrf.exe ] || { echo "FATAL: link produced nothing"; mv -f wrf.exe.orig wrf.exe; exit 4; }
if cmp -s wrf.exe wrf.exe.orig; then echo "FATAL: binary unchanged, link did not rerun"; exit 5; fi
mv wrf.exe wrf_wtimer.exe
mv wrf.exe.orig wrf.exe
ls -la wrf.exe wrf_wtimer.exe

echo "=== 3. G0 DISPATCH GATE ==="
# Delegated to gate0.sh. The version that used to live here reported a false FAIL twice: it
# disassembled module_em_mp_solve_em_ (solve_em is not in a module) and matched "call " when
# this objdump prints "callq". gate0.sh proves it can read before it reports.
NA=$(objdump -d "$W/main/wrf_wtimer.exe" | grep -c "call.*<__wrap_for_allocate_handle>")
echo "  for_allocate_handle wrapped call sites: $NA (allocs_8.o alone has 510)"
[ "$NA" -gt 500 ] || { echo "FATAL: allocator wrap did not fire"; exit 4; }
exec /home/u5727520/wrf-stack/wtimer/gate0.sh "$W/main/wrf_wtimer.exe"
