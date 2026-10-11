"""wtune/preflight_tables.py -- static tables for wtune.preflight's C3 check: the data files each
physics scheme opens at init (WRF source anchors in the comments). Pure data, no logic."""

# scheme id -> required run_dir files, verified 2026-09-24 against
# /nfs/home/nkli/wrf-stack/wrf/nvhpc-fast-adios2-dmsm/WRF/run/README.physics_files
_RA_LW_FILES = {
    1: ["RRTM_DATA"],                              # RRTM
    3: ["CAM_ABS_DATA", "CAM_AEROPT_DATA"],        # CAM
    4: ["RRTMG_LW_DATA"],                          # RRTMG
    14: ["RRTMG_LW_DATA"],                         # RRTMG-KIAPS (module_ra_rrtmg_lwk.F opens it)
    24: ["RRTMG_LW_DATA"],                         # RRTMG fast (module_ra_rrtmg_lwf.F opens it)
}
_RA_SW_FILES = {
    3: ["CAM_ABS_DATA", "CAM_AEROPT_DATA"],        # CAM
    4: ["RRTMG_SW_DATA"],                          # RRTMG
    14: ["RRTMG_SW_DATA"],                         # RRTMG-KIAPS
    24: ["RRTMG_SW_DATA"],                         # RRTMG fast
}
# Files needed when EITHER ra_lw_physics or ra_sw_physics selects the scheme -- checked once per
# domain (not once per lw/sw) so lw=sw=5 does not double-report. Goddard: init_goddardrad is
# called from both the lw and sw branches of ra_init (phys/module_physics_init.F).
_RA_EITHER_FILES = {
    5: ["BROADBAND_CLOUD_GODDARD.bin"],            # Goddard
}
# GFDL (99): gfdletainit (lw or sw) reads co2 transmission tables, but WHICH ones depends on
# co2tf (Registry default 1): module_ra_gfdleta.F:141-145 -- co2tf==1 -> CO2O3 opens
# tr49t85/tr49t67/tr67t85; otherwise CONRAD opens co2_trans. Requiring all four would
# false-FATAL a run_dir that only stages the set its co2tf actually reads.
_GFDL_SCHEME = 99   # ra_lw/sw_physics id
_GFDL_CO2O3_FILES = ["tr49t67", "tr49t85", "tr67t85"]
_GFDL_CONRAD_FILES = ["co2_trans"]
# ozone*.formatted: read by oznini on domain 1 only (module_physics_init.F:2330, EM_CORE) when
# o3input==2 (Registry default 2) -- and share/module_check_a_mundo.F forces o3input=0 unless
# domain-1 ra_lw or ra_sw is an RRTMG-family scheme (4, 14, 24). CAM's own camradinit
# (module_ra_cam.F:1019) calls oznini UNCONDITIONALLY, independent of o3input.
_RRTMG_FAMILY = {4, 14, 24}
_CAM_SCHEME = 3     # ra_lw_physics id
_OZONE_FILES = ["ozone.formatted", "ozone_lat.formatted", "ozone_plev.formatted"]
# aerosol*.formatted: read ONLY by aerosol_in (module_physics_init.F:2340-2345, 5476-5507),
# gated on aer_opt==1 (Tegen climatology) and domain 1, for any radiation scheme. Not read by
# CAM (its aerosol_init opens no *.formatted file) and not read for aer_opt!=1.
_AEROSOL_FILES = ["aerosol.formatted", "aerosol_lat.formatted", "aerosol_lon.formatted",
                  "aerosol_plev.formatted"]
_ECLIPSE_FILE = "eclipse_besselian_elements.dat"   # module_ra_eclipse.F:373-377 wrf_error_fatal

_LSM_FILES = {
    1: ["LANDUSE.TBL"],                                                     # slab (5-layer)
    2: ["LANDUSE.TBL", "VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"],       # Noah
    3: ["LANDUSE.TBL", "VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"],       # RUC
    5: ["LANDUSE.TBL", "CLM_ALB_ICE_DFS_DATA", "CLM_ALB_ICE_DRC_DATA",      # CLM
        "CLM_ASM_ICE_DFS_DATA", "CLM_ASM_ICE_DRC_DATA", "CLM_DRDSDT0_DATA",
        "CLM_EXT_ICE_DFS_DATA", "CLM_EXT_ICE_DRC_DATA", "CLM_KAPPA_DATA", "CLM_TAU_DATA"],
    7: ["LANDUSE.TBL", "VEGPARM.TBL"],                                      # Pleim-Xiu
    8: ["LANDUSE.TBL"],                                                     # SSiB
}
_LSM_NAMES = {1: "slab", 2: "Noah", 3: "RUC", 5: "CLM", 7: "Pleim-Xiu", 8: "SSiB"}
# Noah-MP snicar_*.nc are only read by the SNICAR snow-albedo option -- deliberately NOT required.

# mp_physics -> files the WRF 4.8 OPEN statements actually read: Ferrier 5/95 only
# ETAMPNEW_DATA.expanded_rain (module_mp_fer_hires.F:2969); Thompson CCN only for aerosol-aware
# 28/38 (module_mp_thompson.F:1013); full SBM 32 (fast SBM 30 reads other files, omitted);
# Jensen 55 (module_mp_jensen_ishmael.F:119).
_ETAMPNEW_FILES = ["ETAMPNEW_DATA.expanded_rain"]
_FULL_SBM_FILES = ["bulkdens.asc_s_0_03_0_9", "bulkradii.asc_s_0_03_0_9", "capacity.asc",
                   "coeff_p.asc", "coeff_q.asc", "constants.asc", "kernels.asc_s_0_03_0_9",
                   "kernels_z.asc", "masses.asc", "termvels.asc"]
_MP_FILES = {
    5: _ETAMPNEW_FILES,
    95: _ETAMPNEW_FILES,
    28: ["CCN_ACTIVATE.BIN"],
    38: ["CCN_ACTIVATE.BIN"],
    32: _FULL_SBM_FILES,
    55: ["ishmael-gamma-tab.bin", "ishmael-qi-qc.bin", "ishmael-qi-qr.bin"],
}
# P3: table file names carry a version suffix that drifts between WRF releases (README names
# ...-3momI_v5.1.7 / -5.3-2momI, scc's tree ships p3_lookupTable_1.dat-v5.4_2momI/_3momI and
# p3_lookupTable_2.dat-v5.3), so match by PREFIX. Table 1 is read by every P3 option; table 2
# only when nCat>1 (module_mp_p3.F:474-478), i.e. P3_2CATEGORY (52) only.
_P3_TABLE1_MP = {50, 51, 52, 53}
_P3_TABLE2_MP = {52}
_P3_TABLE1_PREFIX = "p3_lookupTable_1.dat"
_P3_TABLE2_PREFIX = "p3_lookupTable_2.dat"

# sf_urban_physics 1/2/3: urban_param_init (module_sf_urban.F:2997-3022) opens URBPARM.TBL when
# use_wudapt_lcz==0 and URBPARM_LCZ.TBL INSTEAD (not in addition) when use_wudapt_lcz==1.
_URBAN_SCHEMES = {1, 2, 3}
# chem_opt values whose chemistry reads HLC.TBL (README.physics_files).
_HLC_CHEM_OPTS = {111, 112, 114, 201, 202}
# Noah-MP (4): LANDUSE.TBL/VEGPARM.TBL/SOILPARM.TBL/GENPARM.TBL are NOT used by Noah-MP itself,
# but its own table (MPTABLE.TBL in older WRF, renamed NoahmpTable.TBL from ~4.x on -- confirmed
# on scc's WRF tree, which ships NoahmpTable.TBL and no MPTABLE.TBL) is required; accept either
# name so this doesn't false-FATAL across WRF versions.
_NOAHMP_TABLE_NAMES = ("NoahmpTable.TBL", "MPTABLE.TBL")

# ghg_input=1 (default) reads CAMtr_volume_mixing_ratio at init for every non-idealized run and
# FATALs without it for every lw scheme except CAM (3), which has a built-in table
# (module_physics_init.F:1010-1021, module_ra_clWRF_support.F:187-201).
_CAMTR_EXEMPT_RA_LW = {3}

_THOMPSON_MP = {8, 28, 38}
_THOMPSON_DAT_FILES = ["freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat"]

# The buggy staging tool's extension whitelist (missed the extension-less CAMtr file, sc26p2-n4);
# evidence only, unused by this checker.
