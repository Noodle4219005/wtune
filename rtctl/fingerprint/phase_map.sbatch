#!/usr/bin/env bash
#SBATCH --job-name=phasemap
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/fingerprint/phasemap-%j.out
#
# Static half of the external WRF phase fingerprinter, for the pristine production wrf.exe. Read-only: no WRF run.
#  - address range -> symbol -> phase CSV for the executable (text symbols)
#  - whether absolute sample IPs can be used directly (ELF type EXEC vs DYN/PIE)
#  - which phases live in shared libraries (MPI, OpenMP, libm) and must be mapped via /proc/<pid>/maps at run time
#  - whether user callchains are feasible: frame-pointer prologue share in hot physics code, .eh_frame presence
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
EXE=$W/main/wrf.exe
O=/home/u5727520/wrf-bench/wscore/fingerprint/phasemap_$SLURM_JOB_ID
mkdir -p "$O"
module load intel/2024_01_46 2>/dev/null || true
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

echo "exe sha256=$(sha256sum "$EXE" | cut -c1-16)"
readelf -h "$EXE" | grep -E "Type:"
readelf -S -W "$EXE" | awk '$2 ~ /^\.(text|eh_frame|eh_frame_hdr|debug_frame|debug_info|symtab)$/ {print "  section", $2, "size", $6}'
echo "shared libraries:"; ldd "$EXE" | awk '{print "  " $1}' | head -40

nm -n -S --defined-only "$EXE" | awk 'NF == 4 && $3 ~ /[tTwW]/' | awk -v OFS=, '
  BEGIN { print "start_hex,size_hex,symbol,phase" }
  { n = tolower($4); p = "OTHER"
    if      (n ~ /mp_gt_driver|module_mp_thompson/)                       p = "MICRO_THOMPSON"
    else if (n ~ /microphysics_driver|microphysics_zero_out/)             p = "MICRO_DRIVER"
    else if (n ~ /module_mp_/)                                            p = "MICRO_OTHER_SCHEME"
    else if (n ~ /radiation_driver|module_ra_/)                           p = "RADIATION"
    else if (n ~ /surface_driver|module_sf_/)                             p = "SURFACE"
    else if (n ~ /pbl_driver|module_bl_/)                                 p = "PBL"
    else if (n ~ /cumulus_driver|module_cu_/)                             p = "CUMULUS"
    else if (n ~ /module_small_step|advance_uv|advance_w|advance_mu_t|sumflux|small_step/) p = "DYN_SMALL_STEP"
    else if (n ~ /module_advect|rk_tendency|rk_scalar_tend|rk_update|module_big_step/)     p = "DYN_RK_TEND"
    else if (n ~ /module_diffusion|first_rk_step|module_em_/)             p = "DYN_OTHER"
    else if (n ~ /rsl_lite|halo_em|period_bdy|module_comm_dm|module_dm_/) p = "HALO_COMM"
    else if (n ~ /^solve_em_$/)                                           p = "SOLVE_EM"
    else if (n ~ /ext_ncd|ext_adios2|wrf_ext_|output_wrf|med_hist|module_io|wrf_put_|wrf_get_|collect_/) p = "IO"
    else if (n ~ /^__kmp|^__kmpc|^kmp_|^omp_/)                            p = "OPENMP_RT"
    print $1, $2, $4, p }' > "$O/wrf_exe_phase_map.csv"
echo "symbols and text bytes per phase:"
awk -F, 'NR > 1 {c[$4]++; b[$4] += strtonum("0x" $2)} END {for (k in c) printf "  %-20s symbols=%6d bytes=%10d\n", k, c[k], b[k]}' "$O/wrf_exe_phase_map.csv" | sort
echo "key anchors:"
grep -iE ",(solve_em_|module_microphysics_driver_mp_microphysics_driver_|module_mp_thompson_mp_mp_gt_driver_|module_radiation_driver_mp_radiation_driver_|module_first_rk_step_part1_mp_first_rk_step_part1_|module_first_rk_step_part2_mp_first_rk_step_part2_)," "$O/wrf_exe_phase_map.csv" | sed 's/^/  /'

echo "frame-pointer prologue share (push rbp; mov rbp,rsp) in physics/dynamics symbols:"
for pat in module_mp_thompson module_microphysics_driver module_small_step module_advect; do
  tot=0; fp=0
  while IFS=, read -r st sz sym ph; do
    s=$((16#$st)); e=$((s + 16#$sz)); [ "$sz" = 0 ] && continue
    first=$(objdump -d --no-show-raw-insn -M intel --start-address=$s --stop-address=$((s + 16)) "$EXE" | awk '/^ +[0-9a-f]+:/{sub(/^ +[0-9a-f]+:\t*/, ""); print}' | head -2 | tr '\n' ';')
    tot=$((tot + 1)); case "$first" in *"push   rbp"*"mov    rbp,rsp"*) fp=$((fp + 1));; esac
  done < <(grep -i ",[^,]*$pat" "$O/wrf_exe_phase_map.csv")
  echo "  $pat functions=$tot with_fp_prologue=$fp"
done
sha256sum "$O/wrf_exe_phase_map.csv"; wc -l < "$O/wrf_exe_phase_map.csv"
echo "=== done"
