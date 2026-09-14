"""Autotuner step 2: classify one perf-script sample (sym, dso) into a WRF phase.

Pure lookup table, no I/O. Used by fingerprint.py to turn raw `perf script`
samples into one of the labels in PHASES. Rule tables are module-level
constants so tests can exercise them directly instead of poking at strings
embedded in a function body.

stdlib only. Python 3.6.
"""
import os

PHASES = ("MICRO", "RADIATION", "PBL", "SURFACE", "CUMULUS", "DYNAMICS_OTHER",
          "HALO", "MPI", "IO", "INHERIT")

# dso-basename rules, case-insensitive substring match, checked in this order
# before anything else.
MPI_DSO_SUBSTRINGS = ("libmpi",)

INHERIT_DSO_SUBSTRINGS = ("libiomp5", "libgomp", "libimf", "libsvml", "libirc",
                           "libintlc")
INHERIT_DSO_PREFIXES = ("libm.so", "libm-")

IO_DSO_SUBSTRINGS = ("netcdf", "hdf5", "pnetcdf", "adios2", "libz.so", "libsz",
                      "blosc", "zstd", "bz2")

# lowercased-sym rules for samples inside wrf.exe / wrf_wtimer*.exe, checked in this order; first match wins.
# Each rule has substring needles and Fortran-module prefixes ("<prefix>..._mp_" at the start of the symbol).
# Job 1074426 G1 showed why prefixes are needed: the hot radiation kernels are RRTMG/McICA modules
# (rrtmg_sw_*, rrtmg_lw_*, mcica_subcol_gen_*) and the YSU kernel is bl_ysu_mp_*, none of which contain
# "module_ra_"/"module_bl_", so radiation (12.1% true share) was measured as 0.15%.
WRF_PHASE_SYM_RULES = (
    ("MICRO", ("module_mp_", "microphysics_driver"), ("mp_",)),
    ("RADIATION", ("module_ra_", "radiation_driver", "pre_radiation", "rrtmg_", "rrtm_", "mcica_subcol"),
     ("ra_",)),
    ("PBL", ("module_bl_", "pbl_driver"), ("bl_",)),
    ("SURFACE", ("module_sf_", "surface_driver", "module_lsm", "noahmp"), ("sf_",)),
    ("CUMULUS", ("module_cu_", "cumulus_driver"), ("cu_",)),
    ("IO", ("ext_ncd", "ext_adios2", "wrf_ext_", "output_wrf", "med_hist",
            "module_io", "wrf_put_", "wrf_get_", "collect_"), ()),
    # RSL_LITE halo pack/unpack and exchange helpers: communication work outside libmpi
    ("HALO", ("f_pack_", "f_unpack_", "rsl_lite_"), ()),
)

# statically linked compiler/Fortran runtime helpers inside wrf.exe (memcpy/memset, allocation): attributed to
# whatever phase called them, like shared-library frames
WRF_RUNTIME_SYM_PREFIXES = ("__intel_", "_intel_fast_", "for_", "__libm_", "__svml_")

# in the wtimer ground-truth binary only: these sym prefixes are wrapper/
# instrumentation code, not WRF physics, so they inherit like any other
# non-WRF frame.
WTIMER_WRAPPER_SYM_PREFIXES = ("__wrap_", "wt_")


def _basename_lower(dso):
    return os.path.basename(dso or "").lower()


def _is_wrf_dso(dso_base):
    if dso_base == "wrf.exe":
        return True
    return dso_base.startswith("wrf_wtimer") and dso_base.endswith(".exe")


def _is_wtimer_dso(dso_base):
    return dso_base.startswith("wrf_wtimer") and dso_base.endswith(".exe")


def classify(sym, dso):
    """Classify one (sym, dso) perf-script sample into one of PHASES."""
    dso_base = _basename_lower(dso)
    sym_l = (sym or "").lower()

    if any(s in dso_base for s in MPI_DSO_SUBSTRINGS):
        return "MPI"
    if any(s in dso_base for s in INHERIT_DSO_SUBSTRINGS) or \
            dso_base.startswith(INHERIT_DSO_PREFIXES):
        return "INHERIT"
    if any(s in dso_base for s in IO_DSO_SUBSTRINGS):
        return "IO"

    if not _is_wrf_dso(dso_base):
        # libc, ld-linux, [unknown], [vdso], anything else not wrf.exe
        return "INHERIT"

    if sym_l.startswith(WRF_RUNTIME_SYM_PREFIXES):
        return "INHERIT"

    for phase, needles, module_prefixes in WRF_PHASE_SYM_RULES:
        if any(n in sym_l for n in needles):
            return phase
        if "_mp_" in sym_l and sym_l.startswith(module_prefixes):
            return phase

    if _is_wtimer_dso(dso_base) and sym_l.startswith(WTIMER_WRAPPER_SYM_PREFIXES):
        return "INHERIT"

    # '[unknown]' sym inside wrf.exe, or anything else unmatched
    return "DYNAMICS_OTHER"
