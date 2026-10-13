#!/usr/bin/env bash
# build-wrf-extensions.sh -- build WRF v4.8.0 extension binaries on scc with nvhpc.
#
#   bin/build-wrf-extensions.sh submit <ext>...   # login node: prep + sbatch each (one job per ext)
#   bin/build-wrf-extensions.sh prep   <ext>      # login node: copy pristine tree, fix csh shebangs
#   sbatch --exclude=head ... bin/build-wrf-extensions.sh build <ext>   # compute node (what `submit` runs)
#
# <ext>: movenest | chem | hydro | wrfda
#
# Each ext builds in its OWN copy of the vendored source ($WRF_SRC, default
# ~/SC26-WRF/src/WRF = upstream v4.8.0 + the two ADIOS2 patches, i.e. what the production
# build nvhpc-fast-adios2-dmsm was made from) at
#   $WRF_STACK_ROOT/wrf/nvhpc-<ext>-<mode>/WRF        (default root ~/wrf-stack)
# Production builds are never touched. Source files are never edited: the only tree edits are
# (1) interpreter path of csh scripts (/bin/csh does not exist on scc) and
# (2) generated configure.wrf: FCOPTIM line replaced with the production one.
# configure.wrf.delta (diff vs the production configure.wrf) is written next to it.
#
# Per-ext settings (mode = dmpar|dmsm, configure nesting answer, env):
#   movenest  dmsm   nesting=3 (-DMOVE_NESTS -DVORTEX_CENTER)          target em_real
#   chem      dmsm   WRF_CHEM=1 WRF_KPP=1 (KPP=0 via CHEM_KPP=0)        target em_real
#   hydro     dmsm   WRF_HYDRO=1                                        target em_real
#   wrfda     dmpar  ./configure wrfda                                  target all_wrfvar
# chem/hydro use dmsm like production: with the production FCOPTIM, nvfortran 26.5 dies in fort2 (signal 11)
# on module_dm.F etc. in dmpar (no -mp) builds, while every dm+sm build compiled. Run with 1 thread/tile.
# wrfda uses dmpar (WRFDA has no OpenMP). Override with BUILD_MODE=. ADIOS2/PnetCDF stay on wherever the extension tolerates them
# (USE_ADIOS2=0 drops ADIOS2 for that build).
#
# Other env: BUILD_CORES (make -j, default 16), WALL (sbatch --time, default 01:30:00; not TIME: GNU `time` reads $TIME as its output format),
#            CPUS (default 32), CHEM_KPP (default 1), CRTM (wrfda; default unset = WRFDA default)
# COMPILER=gnu (wrfda only): nvfortran 26.5 rejects the GNU `isnan` in da_read_obs_ncgoesabi.inc, so WRFDA
#   may be built with gcc/gfortran (user-approved exception; every other ext stays nvhpc). Its MPI/netCDF/HDF5
#   must be built by the same gcc (Fortran .mod files are compiler-specific): a separate spack env
#   $WRFDA_GNU_ENV (default ~/spack-envs/wrfda-gnu) is created and installed inside the build job.
#   Tree: $WRF_STACK_ROOT/wrf/gcc-wrfda-dmpar/WRF. Use WALL=03:00:00 (spack deps + WRFDA).
set -eu   # no pipefail: head-terminated pipelines SIGPIPE upstream (see wiki wps-460 note)

EXT=${2:-}
PHASE=${1:-}
SELF=$(readlink -f "${BASH_SOURCE[0]}")
WRF_STACK_ROOT=${WRF_STACK_ROOT:-$HOME/wrf-stack}
WRF_SRC=${WRF_SRC:-$HOME/SC26-WRF/src/WRF}
PROD=${PROD:-$WRF_STACK_ROOT/wrf/nvhpc-fast-adios2-dmsm/WRF}
TCSH=${TCSH:-/nfs/software/tcsh/6.24.16/bin/tcsh}
BUILD_CORES=${BUILD_CORES:-16}
FCOPTIM_FLAGS='-O3 -fast -Mvect=noaltcode -Mprefetch=distance:8 -tp=neoverse-v2 -Mstack_arrays'
COMPILER=${COMPILER:-nvhpc}
WRFDA_GNU_ENV=${WRFDA_GNU_ENV:-$HOME/spack-envs/wrfda-gnu}
if [ "$COMPILER" = gnu ]; then
  # the configure stanza's -mcpu=native fails in the armv9 assembler on scc
  FCOPTIM_FLAGS='-O2 -mtune=neoverse-v2 -funroll-loops'
  MENU_LABEL='GCC (gfortran/gcc): Aarch64'; TREE_PREFIX=gcc
elif [ "$COMPILER" = nvhpc ]; then
  MENU_LABEL='NVHPC (nvfortran/nvc)'; TREE_PREFIX=nvhpc
else echo "COMPILER must be nvhpc or gnu"; exit 2; fi

usage() { sed -n '2,12p' "$SELF"; exit 2; }

ext_settings() {   # sets MODE NEST TARGET CONFIG_ARG
  case "$1" in
    movenest) MODE=${BUILD_MODE:-dmsm};  NEST=3; TARGET=em_real;      CONFIG_ARG="" ;;
    chem)     MODE=${BUILD_MODE:-dmsm} ; NEST=1; TARGET=em_real;      CONFIG_ARG="" ;;
    hydro)    MODE=${BUILD_MODE:-dmsm} ; NEST=1; TARGET=em_real;      CONFIG_ARG="" ;;
    wrfda)    MODE=${BUILD_MODE:-dmpar}; NEST=1; TARGET=all_wrfvar;   CONFIG_ARG="wrfda" ;;
    *) usage ;;
  esac
  if [ "$COMPILER" = gnu ] && [ "$1" != wrfda ]; then echo "COMPILER=gnu is allowed for wrfda only"; exit 2; fi
  WRFDIR=$WRF_STACK_ROOT/wrf/$TREE_PREFIX-$1-$MODE/WRF
}

case "$PHASE" in
submit)
  shift
  [ $# -ge 1 ] || usage
  mkdir -p "$WRF_STACK_ROOT/logs"
  for e in "$@"; do
    bash "$SELF" prep "$e"
    login=${WTUNE_LOGIN_NODES-head}      # the machine settings wtune itself uses, scc built in
    sbatch --parsable ${login:+--exclude=$login} ${WTUNE_PARTITION:+-p $WTUNE_PARTITION} \
      ${WTUNE_ACCOUNT:+--account=$WTUNE_ACCOUNT} --job-name="wrfext-$e" --nodes=1 --ntasks=1 \
      --cpus-per-task="${CPUS:-32}" --mem=48G --time="${WALL:-01:30:00}" \
      --output="$WRF_STACK_ROOT/logs/wrfext-$e-%j.out" \
      --export=ALL "$SELF" build "$e"
  done
  exit 0 ;;
prep)
  ext_settings "$EXT"
  mkdir -p "$(dirname "$WRFDIR")"
  if [ -f "$WRFDIR/configure" ]; then echo "already present: $WRFDIR"; else
    rsync -a --exclude=.git "$WRF_SRC"/ "$WRFDIR"/     # ~380 MB, login-node-safe
  fi
  # scc has no /bin/csh. Rewrite the interpreter path only (compile, clean, KPP, WRFDA scripts).
  ( cd "$WRFDIR" && grep -rlI -E '^#! ?/bin/csh' . 2>/dev/null | while read -r f; do
      sed -i -E "1s|^#! ?/bin/csh|#!$TCSH|" "$f"; done )
  echo "prep OK: $WRFDIR"
  exit 0 ;;
build) ;;
*) usage ;;
esac

# ---------------- build phase (compute node) ----------------
ext_settings "$EXT"
if [ "$COMPILER" = gnu ]; then
  # gcc-built MPI + netCDF-C/Fortran + HDF5 in a separate spack env; its view merges C and Fortran netCDF
  # into one prefix, as WRF's configure expects. Only this new env is added to the spack tree.
  # shellcheck disable=SC1091
  set +u; source "${SPACK_SETUP:-$HOME/load_spack.sh}" >/dev/null 2>&1; set -u
  # flags must precede %gcc: written after it they attach to the compiler node and concretization fails
  # ("Only external, or concrete, compilers are allowed"). Rewritten every run so the spec list is exact.
  # ^zlib-ng: the external zlib is registered at prefix /usr but lives in /usr/lib/aarch64-linux-gnu,
  # which pmix's configure does not search ("zlib library (z) in /usr... not found").
  # ^libevent~openssl: libevent's configure cannot find openssl.pc; MPI only needs its event loop.
  mkdir -p "$WRFDA_GNU_ENV"
  cat > "$WRFDA_GNU_ENV/spack.yaml" <<'YAML'
spack:
  specs:
  - openmpi@5.0.10 +fortran cflags=-std=gnu17 %gcc@15.2.0 ^zlib-ng ^libevent~openssl
  - hdf5@1.14.6 ~mpi +fortran cflags=-std=gnu17 %gcc@15.2.0
  - netcdf-c@4.9.2 ~mpi ~blosc cflags=-std=gnu17 %gcc@15.2.0
  - netcdf-fortran@4.6.2 %gcc@15.2.0
  view: true
  concretizer:
    unify: true
YAML
  spack -e "$WRFDA_GNU_ENV" concretize -f
  spack -e "$WRFDA_GNU_ENV" install -j "${CPUS:-32}"
  VIEW=$WRFDA_GNU_ENV/.spack-env/view
  export NETCDF=$VIEW HDF5=$VIEW PATH=$VIEW/bin:$PATH LD_LIBRARY_PATH=$VIEW/lib:$VIEW/lib64:${LD_LIBRARY_PATH:-}
  unset PNETCDF ADIOS2 ADIOS2_DIR
  command -v mpif90 >/dev/null || { echo "mpif90 missing in $VIEW"; exit 1; }
else
# shellcheck disable=SC1091
source "${WTUNE_ENV_SOURCE:-$HOME/wtune-scc/env.sh}"   # nvhpc + MPI + netcdf/hdf5/pnetcdf/adios2 paths
export NETCDF PNETCDF HDF5
if [ "${USE_ADIOS2:-1}" = 0 ]; then unset ADIOS2 ADIOS2_DIR; else export ADIOS2=$ADIOS2_DIR; fi
fi
for p in "${NETCDF:-}" "${PNETCDF:-}" "${HDF5:-}" "${ADIOS2:-}"; do
  [ -n "$p" ] || continue
  export LIBRARY_PATH=$p/lib:$p/lib64:${LIBRARY_PATH:-}   # link-time lib64 fallback
  export PATH=$p/bin:$PATH
done
export OMP_STACKSIZE=1G
ulimit -s unlimited 2>/dev/null || true
[ -d "$WRFDIR" ] || { echo "run prep first: $WRFDIR"; exit 1; }
cd "$WRFDIR"
mkdir -p .tools/bin; ln -sf "$TCSH" .tools/bin/csh; export PATH=$WRFDIR/.tools/bin:$PATH

# per-extension environment (must be set for BOTH configure and compile)
case "$EXT" in
  chem)
    export WRF_CHEM=1 EM_CORE=1 NMM_CORE=0
    if [ "${CHEM_KPP:-1}" = 1 ]; then
      export WRF_KPP=1
      # KPP needs flex + yacc; none on scc's system. spack has flex/bison (built with gcc, only used
      # as build tools). yacc = bison -y via wrapper. csh -> tcsh symlink for scripts calling `csh`.
      # shellcheck disable=SC1091
      set +u; source "${SPACK_SETUP:-$HOME/load_spack.sh}" >/dev/null 2>&1; set -u
      # two identical flex/bison specs are installed, so `spack location -i` is ambiguous: take the first prefix
      FLEXP=$(spack find --format "{prefix}" flex | head -1); BISONP=$(spack find --format "{prefix}" bison | head -1)
      printf '#!/bin/sh\nexec %s/bin/bison -y "$@"\n' "$BISONP" > "$WRFDIR/.tools/bin/yacc"
      chmod +x "$WRFDIR/.tools/bin/yacc"
      export PATH=$WRFDIR/.tools/bin:$FLEXP/bin:$BISONP/bin:$PATH
      export LD_LIBRARY_PATH=$FLEXP/lib:$LD_LIBRARY_PATH   # kpp binary needs libfl.so.2 at run time
      export FLEX=$FLEXP/bin/flex FLEX_LIB_DIR=$FLEXP/lib YACC="$WRFDIR/.tools/bin/yacc -d"
    else
      export WRF_KPP=0
    fi ;;
  hydro) export WRF_HYDRO=1 ;;
  wrfda)
    [ -n "${CRTM:-}" ] && export CRTM
    # scc's `ln` is uutils coreutils 0.8: "ln -sf a/* b/* ." refuses to overwrite a link created in the same
    # invocation, and var/build's `make links` has two da_residual.inc. GNU ln overwrites. Shim: link one
    # source at a time so later duplicates overwrite (PATH change only, no source edit).
    REAL_LN=$(PATH=/usr/bin:/bin command -v ln)
    cat > "$WRFDIR/.tools/bin/ln" <<'SHIM'
#!/bin/bash
fl=(); src=()
for a in "$@"; do case "$a" in -*) fl+=("$a");; *) src+=("$a");; esac; done
n=${#src[@]}
if [ "$n" -gt 2 ]; then
  d=${src[$((n-1))]}
  for ((i=0;i<n-1;i++)); do REAL_LN_PLACEHOLDER "${fl[@]}" "${src[i]}" "$d" || exit $?; done
else exec REAL_LN_PLACEHOLDER "$@"; fi
SHIM
    sed -i "s|REAL_LN_PLACEHOLDER|$REAL_LN|g" "$WRFDIR/.tools/bin/ln"
    chmod +x "$WRFDIR/.tools/bin/ln"
    rm -f var/build/links ;;
esac
# `spack` may prepend its own dirs; keep nvhpc first
[ "$COMPILER" = nvhpc ] && export PATH=$N/compilers/bin:$N/comm_libs/mpi/bin:$PATH

if [ "${SKIP_CONFIGURE:-0}" != 1 ]; then   # SKIP_CONFIGURE=1 = resume compile in an existing tree
# ---- configure (menu option picked by label, like SC26-WRF/scripts/build-wrf.sbatch) ----
MENU_TAG=dmpar; [ "$MODE" = dmsm ] && MENU_TAG='dm\+sm'
LINE=$(timeout 10 ./configure $CONFIG_ARG < /dev/null 2>&1 | grep -m 1 -F "$MENU_LABEL" || true)
OPT=$(printf '%s\n' "$LINE" | grep -oE "[0-9]+\.[[:space:]]*\($MENU_TAG\)" | grep -oE '^[0-9]+' || true)
[ -n "$OPT" ] || { echo "$MENU_LABEL $MODE option not found in ./configure $CONFIG_ARG menu"; exit 1; }
echo "$MENU_LABEL $MODE = option $OPT, nesting answer $NEST"
# extra newlines: some configures (wrfda) ask more than one follow-up question
timeout 300 bash -c "printf '%s\n%s\n1\n1\n1\n' '$OPT' '$NEST' | ./configure $CONFIG_ARG" > configure.log 2>&1 || true
[ -s configure.wrf ] || { echo "configure.wrf missing; see $WRFDIR/configure.log"; exit 1; }
sed -i "s/^FCOPTIM[[:space:]]*=.*/FCOPTIM        = $FCOPTIM_FLAGS/" configure.wrf
# gcc 15 defaults to C23, which breaks WRF's K&R-style C; pin gnu17 for C only
[ "$COMPILER" = gnu ] && sed -i 's/^\(CFLAGS_LOCAL[[:space:]]*=.*\)$/\1 -std=gnu17/' configure.wrf
# WRFDA needs 8-byte default reals (RWORDSIZE=8). The aarch64 GCC stanza leaves PROMOTION empty, while the
# x86 gfortran stanzas carry -fdefault-real-8 (enabled for WRFDA) and NVHPC uses -r8: without it
# da_etkf.f fails "passed REAL(4) to REAL(8)" on the LAPACK calls. Mirror the x86 gfortran setting.
[ "$COMPILER" = gnu ] && sed -i 's/^PROMOTION[[:space:]]*=.*/PROMOTION       = -fdefault-real-8/' configure.wrf
# record the delta against the production configure.wrf (the only edit above is FCOPTIM)
diff <(grep -v '^#' "$PROD/configure.wrf") <(grep -v '^#' configure.wrf) > configure.wrf.delta || true
case "$EXT" in
  movenest) grep -q -- '-DMOVE_NESTS -DVORTEX_CENTER' configure.wrf || { echo "nesting flags missing"; exit 1; } ;;
  chem)     grep -q '^WRF_CHEM[[:space:]]*=[[:space:]]*1' configure.wrf || { echo "WRF_CHEM not 1"; exit 1; } ;;
  hydro)    grep -q -- '-DWRF_HYDRO' configure.wrf || { echo "-DWRF_HYDRO missing"; exit 1; } ;;
esac
echo "--- configure.wrf.delta vs production (lines: $(wc -l < configure.wrf.delta)) ---"
head -40 configure.wrf.delta
fi

# WRF-Hydro's own (legacy) configure must run once: it writes hydro/macros and hydro/Makefile.comm, which
# WRF's ./compile expects (without them: "No rule to make target Makefile.comm", then "cannot find -lHYDRO").
if [ "$EXT" = hydro ] && [ ! -f hydro/Makefile.comm ]; then
  ( cd hydro && ./configure nvfort ) > hydro-configure.log 2>&1 || { echo "hydro configure failed, see hydro-configure.log"; exit 1; }
  [ -f hydro/Makefile.comm ] && [ -f hydro/macros ] || { echo "hydro configure did not write Makefile.comm/macros"; exit 1; }
fi

# ---- compile ----
JOBS=$BUILD_CORES; [ "$EXT" = wrfda ] && JOBS=8   # WRFDA/phys makefiles race at high -j
# phys/Makefile does not declare every module dependency, so -j can lose a race ("Unable to open MODULE
# file"). The .mod exists on the next pass: re-run compile (incremental) up to 3 times until the exes exist.
for try in 1 2 3; do
  ./compile -j "$JOBS" "$TARGET" 2>&1 | tee -a compile.log || true
  case "$EXT" in wrfda) [ -x var/da/da_wrfvar.exe ] && break ;; *) [ -x main/wrf.exe ] && [ -x main/real.exe ] && break ;; esac
  echo "compile pass $try did not produce the executables; retrying"
done
case "$EXT" in
  wrfda) EXES="var/da/da_wrfvar.exe" ;;
  *)     EXES="main/wrf.exe main/real.exe" ;;
esac
rc=0
for x in $EXES; do
  if [ -x "$x" ]; then ls -l "$x"; else echo "MISSING: $x"; rc=1; fi
done
[ $rc = 0 ] && echo "OK: $EXT built in $WRFDIR" || echo "FAILED: $EXT, see $WRFDIR/compile.log"
exit $rc
