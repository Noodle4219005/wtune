#!/bin/bash -l
#SBATCH --job-name=wtune-kf-preproof-r1
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/native-review/preproof-%j.out

# One-use proof-disabled build and link trace. It must never run WRF.
set -euo pipefail
umask 077
ROOT=/home/u5727520/wrf-stack/wtune-kf-reviewed-recipe-20260913-r1/preproof
WTUNE=/home/u5727520/wrf-stack/wtune
FROZEN=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
SOURCE=$WTUNE/artifacts/20260913-node-local-physics/native-review
RECIPE=$ROOT/recipe
INPUTS=$ROOT/inputs
RAW=$ROOT/raw
CONFIG=$INPUTS/unattested-preproof-config.json
PREPARER=$WTUNE/plov/prepare_kf_build.py
EXECUTOR=$WTUNE/plov/execute_build_recipe.py
REVIEW_HELPER=$WTUNE/plov/kf_review_record.py
VERIFY=$SOURCE/verify-native-preproof.py
PROSPECTIVE=$SOURCE/prospective-inputs.json
PY=/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/.pixi/envs/default/bin/python3.11
CC=/pkg/compiler/intel/2024/mpi/2021.11/bin/mpiicx
FC=/pkg/compiler/intel/2024/mpi/2021.11/bin/mpiifx
ICX=/pkg/compiler/intel/2024/compiler/2024.0/bin/icx
IFX=/pkg/compiler/intel/2024/compiler/2024.0/bin/ifx

test ! -e "$ROOT" && test ! -L "$ROOT"
mkdir -p "$INPUTS" "$RAW" "$ROOT/shared-scratch"
export TMPDIR=$ROOT/shared-scratch TMP=$ROOT/shared-scratch TEMP=$ROOT/shared-scratch
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1

sha256sum --check --strict <<'SHA256'
ce5c909b89e6be3dd5d448d1a63a33c26f15f06764308615c07f96a8eda1e74e  /home/u5727520/wrf-stack/wtune/plov/prepare_kf_build.py
477680eb02afd8a9022db8d528044d1455908c7c4424c669db9ab3269474dcb0  /home/u5727520/wrf-stack/wtune/plov/execute_build_recipe.py
d7a8e523368fe7f184999c1911c505014b29525b5c28c1dbb7035d2688907c49  /home/u5727520/wrf-stack/wtune/plov/kf_review_record.py
6c83106d0c391a4697777f702572380090b70ac799eb927734fe22ccca3499fb  /home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/native-review/prospective-inputs.json
386afb31f2865c146814e93ee0aac65293d412c4a109f4cb1e5ecb1868fbd1f6  /home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/native-review/verify-native-preproof.py
fbe4f2627403660852b7d4bd5a9113ae124a420c8aca65a8ac3979e25d6899b8  /pkg/compiler/intel/2024/mpi/2021.11/bin/mpiicx
e9a7ea3f988763f98e7e7f99fcffd45bc15462ba9c052ba7dc5d50d8f826fbc9  /pkg/compiler/intel/2024/mpi/2021.11/bin/mpiifx
7ac617bec9e9a97d6c273e7db26760fab4d0998ac7abc299fec20972f64c5d77  /pkg/compiler/intel/2024/compiler/2024.0/bin/icx
f85213d5b151273368ba5f5c40fe87115bd925504b95a7d213cd10e40a99b489  /pkg/compiler/intel/2024/compiler/2024.0/bin/ifx
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  /pkg/compiler/intel/2024/compiler/2024.0/bin/icx.cfg
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  /pkg/compiler/intel/2024/compiler/2024.0/bin/ifx.cfg
SHA256

module load intel/2024_01_46
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
unset I_MPI_CC I_MPI_CXX I_MPI_F77 I_MPI_F90 I_MPI_FC MPICH_CC MPICH_CXX MPICH_F77 MPICH_F90 MPICH_FC
unset I_MPI_COMPILER_CONFIG_DIR I_MPI_CC_PROFILE I_MPI_FC_PROFILE I_MPI_F90_PROFILE I_MPI_LINK
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH INCLUDE LIBRARY_PATH COMPILER_PATH GCC_EXEC_PREFIX
unset IFORTCFG IFXCFG ICXCFG ICC_CFG ICPXCFG
test "$(readlink -f "$(command -v icx)")" = "$ICX"
test "$(readlink -f "$(command -v ifx)")" = "$IFX"
test "${I_MPI_ROOT-}" = /pkg/compiler/intel/2024/mpi/2021.11
{
  for name in PATH LD_LIBRARY_PATH I_MPI_ROOT I_MPI_CC I_MPI_CXX I_MPI_F77 I_MPI_F90 I_MPI_FC \
    MPICH_CC MPICH_CXX MPICH_F77 MPICH_F90 MPICH_FC I_MPI_COMPILER_CONFIG_DIR \
    I_MPI_CC_PROFILE I_MPI_FC_PROFILE I_MPI_F90_PROFILE I_MPI_LINK IFORTCFG IFXCFG ICXCFG \
    ICC_CFG ICPXCFG CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH INCLUDE \
    LIBRARY_PATH COMPILER_PATH GCC_EXEC_PREFIX; do
    if [[ -v $name ]]; then printf '%s=%q\n' "$name" "${!name}"; else printf '%s=<UNSET>\n' "$name"; fi
  done
} > "$RAW/build-environment.txt"
"$ICX" --version > "$RAW/icx-version.txt"
"$IFX" --version > "$RAW/ifx-version.txt"
"$CC" -show > "$RAW/mpiicx-show.txt"
"$FC" -show > "$RAW/mpiifx-show.txt"
sha256sum "$CC" "$FC" "$ICX" "$IFX" \
  /pkg/compiler/intel/2024/compiler/2024.0/bin/icx.cfg \
  /pkg/compiler/intel/2024/compiler/2024.0/bin/ifx.cfg > "$RAW/compiler-sha256.txt"

"$PY" - "$PROSPECTIVE" "$CONFIG" <<'PY'
import json,sys
source,target=sys.argv[1:]
data=json.load(open(source))
config=data['candidate_config']
config['node_local_profile']={'name':'unattested-v1'}
with open(target,'x') as stream:
    json.dump(config,stream,indent=2,sort_keys=True); stream.write('\n')
PY

"$PY" "$VERIFY" --prospective "$PROSPECTIVE" --preflight > "$RAW/prospective-preflight-before.json"
"$PY" "$PREPARER" --wtune-root "$WTUNE" --frozen-root "$FROZEN" \
  --cc "$CC" --fc "$FC" --python "$PY" --compiler-config "$CONFIG" \
  --output-dir "$RECIPE" > "$RAW/prepare.json"
MANIFEST_SHA=$(sha256sum "$RECIPE/build-manifest.json" | cut -d' ' -f1)
"$PY" "$EXECUTOR" --manifest "$RECIPE/build-manifest.json" \
  --expected-sha256 "$MANIFEST_SHA" --timeout-seconds 60 --execute > "$RAW/compile-status.json"
"$PY" - "$RAW/compile-status.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['status']=='complete' and d['link_executed'] is False
PY

"$PY" "$VERIFY" --prospective "$PROSPECTIVE" --preflight > "$RAW/prospective-preflight-after.json"
# Revalidate the exact reviewed direct-token contract immediately before link.
"$PY" - "$PROSPECTIVE" "$REVIEW_HELPER" <<'PY'
import json,sys,types
record,helper_path=sys.argv[1:]
m=types.ModuleType('review_helper'); m.__file__=helper_path
exec(compile(open(helper_path,'rb').read(),helper_path,'exec'),m.__dict__)
d=json.load(open(record)); m._library_contract(d['lanes']['libraries'])
PY

# Use the fresh 25-object proof-disabled build because r3 lacks the seven node-local
# units. The only changes from the future reviewed argv are proof macros/profile text.
"$PY" - "$RECIPE/link-manifest.json" "$FC" "$RECIPE/wrf-wtune-unattested.exe" "$RECIPE/wrf-wtune-unattested.map" > "$RAW/link-argv.json" <<'PY'
import json,sys
manifest,fc,output,link_map=sys.argv[1:]
d=json.load(open(manifest)); argv=d['argv']; argv[0]=fc
argv[argv.index('-o')+1]=output
argv[1:1]=['-Wl,-t','-Wl,-Map,'+link_map]
print(json.dumps(argv))
PY
"$PY" - "$RAW/link-argv.json" "$RECIPE" "$RAW/link-trace.stdout" "$RAW/link-trace.stderr" <<'PY'
import json,subprocess,sys
argv_path,cwd,out,err=sys.argv[1:]
argv=json.load(open(argv_path))
with open(out,'x') as stdout,open(err,'x') as stderr:
    result=subprocess.run(argv,cwd=cwd,stdout=stdout,stderr=stderr,timeout=120)
if result.returncode: raise SystemExit(result.returncode)
PY

readelf -d "$RECIPE/wrf-wtune-unattested.exe" > "$RAW/candidate-dynamic.txt"
readelf -l "$RECIPE/wrf-wtune-unattested.exe" > "$RAW/candidate-program-headers.txt"
INTERP=$(sed -n 's/.*Requesting program interpreter: \(.*\)]/\1/p' "$RAW/candidate-program-headers.txt")
case "$INTERP" in /*) ;; *) echo 'missing absolute PT_INTERP' >&2; exit 1;; esac
INTERP=$(readlink -f -- "$INTERP")
test -f "$INTERP" && test ! -L "$INTERP"
sha256sum "$INTERP" > "$RAW/loader-sha256.txt"
env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}" I_MPI_ROOT="${I_MPI_ROOT-}" \
  "$INTERP" --list "$RECIPE/wrf-wtune-unattested.exe" > "$RAW/loader-list.txt"
sha256sum "$RECIPE/wrf-wtune-unattested.exe" "$RECIPE/wrf-wtune-unattested.map" \
  "$RAW/link-argv.json" "$RAW/link-trace.stdout" "$RAW/link-trace.stderr" \
  "$RAW/candidate-dynamic.txt" "$RAW/candidate-program-headers.txt" \
  "$RAW/loader-list.txt" > "$RAW/preproof-output-sha256.txt"

# Terminal inspection is deliberately strict: the human review still decides whether
# these native facts justify compiler_module_abi and library_closure.
"$PY" "$VERIFY" --prospective "$PROSPECTIVE" \
  --recipe "$RECIPE" --raw "$RAW" --output "$RAW/native-closure.json"
sha256sum --check --strict "$RAW/compiler-sha256.txt"
sha256sum --check --strict "$RAW/loader-sha256.txt"
sha256sum --check --strict <<'SHA256'
ce5c909b89e6be3dd5d448d1a63a33c26f15f06764308615c07f96a8eda1e74e  /home/u5727520/wrf-stack/wtune/plov/prepare_kf_build.py
477680eb02afd8a9022db8d528044d1455908c7c4424c669db9ab3269474dcb0  /home/u5727520/wrf-stack/wtune/plov/execute_build_recipe.py
d7a8e523368fe7f184999c1911c505014b29525b5c28c1dbb7035d2688907c49  /home/u5727520/wrf-stack/wtune/plov/kf_review_record.py
6c83106d0c391a4697777f702572380090b70ac799eb927734fe22ccca3499fb  /home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/native-review/prospective-inputs.json
386afb31f2865c146814e93ee0aac65293d412c4a109f4cb1e5ecb1868fbd1f6  /home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/native-review/verify-native-preproof.py
SHA256
test ! -s "$RAW/link-trace.stderr"
echo 'PREPROOF COMPLETE; WRF WAS NOT RUN'
