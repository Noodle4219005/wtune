"""WRF extensions -- WRF-Chem, WRF-Hydro, WRFDA, moving nest: detection, refusal, cost factors.

Policy (user decision 2026-09-29, decision 0056): for these four extensions WTune tunes ONLY
grid/rank-level STRICT knobs (MPI ranks, nproc_x/nproc_y, tiles, OpenMP threads, I/O settings).
Every extension-specific parameter (chem_opt, chemdt, photdt, emissions, hydro routing
timesteps, AGGFACTRT, DA iterations/outer loops/B-matrix options, vortex_interval and the other
vortex-following parameters, ...) is left byte-for-byte as given and is not even suggested. They
are catalogued PHYSICS-GRAY / do-not-touch in wtune/compliance.py so assert_strict_only rejects
any change.

Stdlib only (wtune_compliance imports the key tables from here). Namelists are the nested
{group: {key: [scalar, ...]}} dicts of wtune_predict.parse_namelist; `load_namelist` imports it
lazily.

Build detection: `scan_binary` looks for symbol-name substrings in wrf.exe. BUILD_SIGNATURES are
UNVERIFIED against a real chem/hydro/move-nest build (none exists on scc as of 2026-09-29); the
control symbols guard against a stripped binary, which yields None ("cannot tell"), never a
false "not built with".
"""
import json
import mmap
import os
import re

# ---------------------------------------------------------------------------
# extension-specific parameters (all do-not-touch); lowercase, as parse_namelist emits them.
# Trailing '*' = prefix wildcard (wtune_compliance.lookup semantics).
# ---------------------------------------------------------------------------
CHEM_KEYS = (
    'chem_opt', 'chemdt', 'photdt', 'bioemdt', 'chem_in_opt', 'phot_opt', 'kemit',
    'emiss_inpt_opt', 'emiss_opt', 'emiss_opt_vol', 'io_style_emissions', 'emiss_ash_hgt',
    'gas_bc_opt', 'gas_ic_opt', 'aer_bc_opt', 'aer_ic_opt', 'have_bcs_chem', 'have_bcs_upper',
    'gaschem_onoff', 'aerchem_onoff', 'wetscav_onoff', 'cldchem_onoff', 'vertmix_onoff',
    'chem_conv_tr', 'conv_tr_wetscav', 'conv_tr_aqchem', 'biomass_burn_opt', 'plumerisefire_frq',
    'aer_ra_feedback', 'aer_op_opt', 'opt_pars_out', 'diagnostic_chem', 'bio_emiss_opt',
    'ne_area', 'dust_opt', 'dmsemis_opt', 'seas_opt', 'depo_fact', 'gas_drydep_opt',
    'aer_drydep_opt', 'dust_alpha', 'dust_gamma', 'dust_schme', 'dust_*', 'emiss_*',
    'chem_*',
)
HYDRO_KEYS = (
    'aggfactrt', 'dtrt_ch', 'dtrt_ter', 'dxrt', 'subrtswcrt', 'ovrtswcrt', 'chanrtswcrt',
    'gwbaseswcrt', 'rt_option', 'channel_option', 'gw_restart', 'gwbasmskfil', 'gwstrmfil',
    'udmp_opt', 'sys_cpl', 'geo_static_flnm', 'restart_file', 'rst_typ', 'rst_bi_in',
    'rst_bi_out', 'rst_dt', 'chrtout_domain', 'chrtout_grid', 'lsmout_domain', 'rtout_domain',
    'chanobs_domain', 'output_gw', 'outlake', 'frxst_pts_out', 'output_channelbdy_hydro',
    'io_config_outputs', 'io_form_outputs', 't0outputflag', 'order_to_write',
    'route_lake_f', 'route_link_f', 'route_chan_f', 'route_topo_f',
)
WRFDA_KEYS = (
    'var4d', 'var4d_lbc', 'var4d_bin', 'var4d_bin_rad', 'multi_inc', 'max_ext_its', 'ntmax',
    'eps', 'orthonorm_gradient', 'cv_options', 'cv_options_hum', 'be_options', 'as1', 'as2',
    'as3', 'as4', 'as5', 'len_scaling1', 'len_scaling2', 'len_scaling3', 'len_scaling4',
    'len_scaling5', 'je_factor', 'write_increments', 'check_max_iv', 'time_window_min',
    'time_window_max', 'analysis_type', 'fg_format', 'cloud_cv_options', 'analysis_date',
    'use_synopobs', 'use_shipsobs', 'use_metarobs', 'use_soundobs', 'use_pilotobs',
    'use_airepobs', 'use_geoamvobs', 'use_polaramvobs', 'use_bogusobs', 'use_buoyobs',
    'use_profilerobs', 'use_satemobs', 'use_gpspwobs', 'use_gpsrefobs', 'use_qscatobs',
    'use_radarobs', 'use_radar_rv', 'use_radar_rf', 'use_airsretobs', 'use_rad',
    'use_ssmiretrievalobs', 'use_hsbobs', 'use_obs_errfac', 'use_ground_gps',
)
MOVENEST_KEYS = (
    'num_moves', 'move_id', 'move_interval', 'move_cd_x', 'move_cd_y', 'time_to_move',
    'vortex_interval', 'max_vortex_speed', 'corral_dist', 'track_level',
)
# keys whose presence (any value) marks a vortex-following nest; num_moves>0 marks preset moves
_VORTEX_KEYS = ('vortex_interval', 'max_vortex_speed', 'corral_dist', 'track_level', 'time_to_move')

EXT_IDS = ('chem', 'hydro', 'wrfda', 'movenest')
EXT_KEYS = {'chem': CHEM_KEYS, 'hydro': HYDRO_KEYS, 'wrfda': WRFDA_KEYS, 'movenest': MOVENEST_KEYS}
EXT_GROUPS = {'chem': ('chem',), 'hydro': ('hydro', 'hydro_nlist', 'wrf_hydro'), 'movenest': ()}

# how to describe each extension to the user (exact strings used by `wtune check`)
LABEL = {'chem': 'WRF-Chem', 'hydro': 'WRF-Hydro', 'wrfda': 'WRFDA', 'movenest': 'Moving nest'}
UNTOUCHED = {
    'chem': 'chem_opt/chemdt/photdt/emissions',
    'hydro': 'AGGFACTRT/DTRT_CH/DTRT_TER/routing switches',
    'wrfda': 'the DA settings (&wrfvar* groups: iterations, outer loops, B-matrix and observation options)',
    'movenest': 'vortex_interval/max_vortex_speed/corral_dist/track_level/time_to_move/num_moves/move_*',
}
# dmpar-only extension builds on scc (build agent, 2026-09-29): no OpenMP, so omp must stay 1.
# Moving nest is dm+sm and keeps OpenMP.
NO_OPENMP = ('chem', 'hydro', 'wrfda')
CALIBRATION_TEXT = {
    'chem': 'time prediction for this extension is UNCALIBRATED until calibration runs exist',
    'hydro': 'time prediction for this extension is UNCALIBRATED until calibration runs exist',
    'movenest': ('time prediction adds a MEASURED +6.0% MOVE_NESTS build-overhead constant '
                 '(all-four-wrf-nesting-modes-measured; moving itself, about -0.6%, is not '
                 'modelled) and is otherwise UNCALIBRATED for this extension'),
    'wrfda': ('WTune does not run or orchestrate DA cycles (obsproc -> da_wrfvar.exe -> '
              'da_update_bc -> forecast) and has no DA time model; the forecast (wrf.exe) part '
              'uses the normal model. da_wrfvar.exe rank/decomposition recommendations are '
              'ADVISORY ONLY and are never applied automatically (a bit-identical check does '
              'not hold for DA minimisation: reduction order changes with decomposition)'),
}
REBUILD_CANNOT = {
    'chem': 'the chemical initial/boundary/emission state',
    'hydro': 'the routing/groundwater state (Fulldom, hydro restart files)',
    'wrfda': 'the analysis/DA-cycle state',
    'movenest': 'the nest position history',
}
CFL_COUPLED = {
    'chem': 'chemdt/photdt are tied to time_step',
    'hydro': 'the routing timesteps DTRT_CH/DTRT_TER are tied to time_step',
    'movenest': 'vortex_interval and the nest-move schedule are tied to time_step',
}

# ---------------------------------------------------------------------------
# unsupported families (file evidence in the run dir)
# ---------------------------------------------------------------------------
# An executable alone is evidence only when no ARW build ships it. ndown.exe is built and linked
# into run/ by every em_real compile, and `wtune run` links all of run/ into the run dir, so
# ndown is reported only on its data (wrfndi_d0*). NMM / MPAS executables never appear in an ARW
# run/ and stay evidence on their own. WRF-Hydro standalone exes are ignored next to wrf.exe
# (a WRF atmosphere means the coupled setup, not standalone).
UNSUPPORTED_FILES = (
    ('ndown', 'ndown', (), ('wrfndi_d02', 'wrfndi_d01'),
     'one-way ndown.exe nesting is not modelled: rank/decomposition advice and the time model '
     'assume a wrf.exe run driven by real.exe output'),
    ('hydro-standalone', 'WRF-Hydro standalone',
     ('wrf_hydro.exe', 'wrf_hydro_NoahMP.exe', 'wrf_hydro_noahmp.exe'), ('namelist.hrldas',),
     'stand-alone WRF-Hydro (namelist.hrldas + hydro.namelist without a WRF atmosphere) has no '
     'WTune support; only WRF-coupled WRF-Hydro (wrf.exe) is supported, at grid/rank level'),
    ('nmm', 'WRF-NMM', ('real_nmm.exe', 'wrf_nmm.exe'), (),
     'WRF-NMM is not supported (ARW only)'),
    ('mpas', 'MPAS', ('atmosphere_model', 'init_atmosphere_model'), ('namelist.atmosphere',),
     'MPAS is not supported (WRF ARW only)'),
)

# ---------------------------------------------------------------------------
# build signatures -- UNVERIFIED (see module docstring)
# ---------------------------------------------------------------------------
# chem/hydro entries stay UNVERIFIED until their binaries are built (production has 0 hits for them).
BUILD_SIGNATURES = {
    'chem': (b'chem_driver', b'module_chem_utilities'),
    'hydro': (b'wrf_drv_hydro', b'module_hydro_drv', b'hydro_drv'),
    # med_nest_move: 18 nm symbols / 19 strings in the scc MOVE_NESTS wrf.exe, 0 / 0 in production
    # (agent E, nm/strings, VERIFIED). 'vortex_center' is a false positive (a
    # COMPUTE_VORTEX_CENTER_ALARM string sits in BOTH binaries) and must never be used.
    'movenest': (b'med_nest_move',),
}
BUILD_CONTROL = (b'solve_em', b'start_domain')

# PROVISIONAL (one case, exploratory; every cell one execution): WRF-Chem GOCART (chem_opt=300, nvhpc
# dm+sm) on Katrina d01 169x129 HANGS at the first post-t=0 history write when the smallest j-patch is
# narrow (8x9: j 14-15, 6x12: j 11), passes at j >= 21 (6x6, 12x6); nproc_y 7 and 8 untested and other
# domain sizes unconfirmed. Narrow i-patches are fine. chem_opt=0 at 72 ranks is fine.
CHEM_MIN_J_PATCH = 21
CHEM_J_PATCH_NOTE = ('knowledge/failed-ideas/2026-09-30-wrf-chem-gocart-hangs-at-small-j-patch-'
                     'decomposition.md')

_COEFF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'wtune_extension_coefficients.json')


# ---------------------------------------------------------------------------
# namelist helpers
# ---------------------------------------------------------------------------
def load_namelist(path):
    from wtune import predict as wtune_predict
    return wtune_predict.parse_namelist(path)


def _vals(nl, key, groups=None):
    """All values of `key` in the listed groups (default: every group), flattened."""
    out = []
    for gname, grp in (nl or {}).items():
        if groups is not None and gname not in groups:
            continue
        v = grp.get(key)
        if v:
            out.extend(v)
    return out


def _has_key(nl, key):
    return any(key in grp for grp in (nl or {}).values())


def _num_list(vals):
    out = []
    for v in vals:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(v)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
    return out


def extension_group_names(nl):
    """Names of the namelist groups that belong wholly to an extension (do-not-touch)."""
    names = []
    for gname in (nl or {}):
        if gname in EXT_GROUPS['chem'] + EXT_GROUPS['hydro'] or gname.startswith('wrfvar'):
            names.append(gname)
    return sorted(names)


def extension_keys_changed(orig_nl, final_nl):
    """Group-level guard: [(group, key)] whose value differs between two parsed namelists inside an
    extension-owned group, or whose key matches an extension key anywhere. Empty = untouched."""
    bad = []
    ext_groups = set(extension_group_names(orig_nl)) | set(extension_group_names(final_nl))
    all_keys = {}
    for e in EXT_IDS:
        for k in EXT_KEYS[e]:
            all_keys[k] = e
    for gname in sorted(set(orig_nl or {}) | set(final_nl or {})):
        og, fg = (orig_nl or {}).get(gname, {}), (final_nl or {}).get(gname, {})
        for k in sorted(set(og) | set(fg)):
            if og.get(k) == fg.get(k):
                continue
            if gname in ext_groups or _is_ext_key(k):
                bad.append((gname, k))
    return bad


def _is_ext_key(k):
    for e in EXT_IDS:
        for pat in EXT_KEYS[e]:
            if pat.endswith('*'):
                if k.startswith(pat[:-1]):
                    return True
            elif k == pat:
                return True
    return False


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def _hydro_namelist_path(run_dir):
    if not run_dir:
        return None
    for n in ('hydro.namelist', 'HYDRO.namelist'):
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            return p
    return None


def detect(nl, run_dir=None):
    """Extensions the case declares. Returns a list of dicts
    {id, label, active, evidence:[str], detail:{}} in EXT_IDS order (only detected ones).
    `active` is False when the namelist only carries the group without switching the feature on
    (e.g. &chem with chem_opt=0); build mismatch is only an error for active extensions."""
    out = []

    ev = []
    chem_opts = _num_list(_vals(nl, 'chem_opt'))
    active = any(v > 0 for v in chem_opts)
    if 'chem' in (nl or {}):
        ev.append('&chem group present')
    if chem_opts:
        ev.append('chem_opt=%s' % ','.join(str(int(v)) for v in chem_opts))
    if ev:
        out.append(dict(id='chem', label=LABEL['chem'], active=active, evidence=ev,
                        detail={'chem_opt': int(max(chem_opts)) if chem_opts else 0}))

    ev = []
    hp = _hydro_namelist_path(run_dir)
    if hp:
        ev.append('hydro.namelist in run dir (%s)' % hp)
    for gname in EXT_GROUPS['hydro']:
        if gname in (nl or {}):
            ev.append('&%s group in namelist.input' % gname)
    if ev:
        out.append(dict(id='hydro', label=LABEL['hydro'], active=True, evidence=ev,
                        detail={'hydro_namelist': hp}))

    ev = []
    wgroups = sorted(g for g in (nl or {}) if g.startswith('wrfvar'))
    if wgroups:
        ev.append('&%s group(s) in namelist.input' % ','.join(wgroups))
    da_exe = None
    if run_dir:
        p = os.path.join(run_dir, 'da_wrfvar.exe')
        if os.path.lexists(p):
            da_exe = p
            ev.append('da_wrfvar.exe in run dir')
    if ev:
        out.append(dict(id='wrfda', label=LABEL['wrfda'], active=True, evidence=ev,
                        detail={'da_wrfvar_exe': da_exe, 'wrfvar_groups': wgroups}))

    ev = []
    vort = [k for k in _VORTEX_KEYS if _has_key(nl, k)]
    if vort:
        ev.append('vortex-following keys set: %s' % ', '.join(vort))
    nm = _num_list(_vals(nl, 'num_moves'))
    if any(v > 0 for v in nm):
        ev.append('num_moves=%d (preset moves)' % int(max(nm)))
    if ev:
        out.append(dict(id='movenest', label=LABEL['movenest'], active=True, evidence=ev,
                        detail={'vortex_keys': vort, 'num_moves': int(max(nm)) if nm else 0}))
    return out


def detect_unsupported(run_dir):
    """[(id, label, message, evidence)] for ndown / Hydro-standalone / NMM / MPAS run dirs."""
    out = []
    if not run_dir or not os.path.isdir(run_dir):
        return out
    for uid, label, exes, files, msg in UNSUPPORTED_FILES:
        hits = [n for n in exes + files if os.path.lexists(os.path.join(run_dir, n))]
        if uid == 'hydro-standalone':
            # namelist.hrldas alongside a WRF namelist.input is the COUPLED setup's leftover; only
            # flag stand-alone when there is no wrf.exe atmosphere (no wrf.exe, and no
            # namelist.input unless a stand-alone exe is present)
            if os.path.lexists(os.path.join(run_dir, 'wrf.exe')):
                hits = []
            elif os.path.exists(os.path.join(run_dir, 'namelist.input')) and \
                    not any(n in exes for n in hits):
                hits = []
        if hits:
            out.append((uid, label, msg, hits))
    return out


# ---------------------------------------------------------------------------
# build capability (wrf.exe)
# ---------------------------------------------------------------------------
def find_wrf_exe(run_dir, explicit=None):
    cands = [explicit] if explicit else []
    if run_dir:
        cands.append(os.path.join(run_dir, 'wrf.exe'))
    tree = os.environ.get('WRF_TREE')
    if tree:
        cands += [os.path.join(tree, 'run', 'wrf.exe'), os.path.join(tree, 'main', 'wrf.exe')]
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.realpath(c)
    return None


def _nm_symbols(path, timeout=120):
    """Text of `nm` output for the binary, or '' when nm is missing/fails/finds no symbols."""
    import shutil
    import subprocess
    exe = shutil.which('nm')
    if not exe:
        return ''
    try:
        r = subprocess.run([exe, path], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ''
    return r.stdout if r.returncode == 0 else ''


def scan_binary(path):
    """{'chem': True|False|None, 'hydro': ..., 'movenest': ...} from symbol substrings. None for
    every capability when the binary has no control symbol (stripped) or cannot be read."""
    caps = {k: None for k in BUILD_SIGNATURES}
    nm = _nm_symbols(path)
    if nm:  # preferred: real symbol table (a stripped binary yields none -> byte scan below)
        if not any(c.decode() in nm for c in BUILD_CONTROL):
            return caps
        return {cap: any(sig.decode() in nm for sig in sigs) for cap, sigs in BUILD_SIGNATURES.items()}
    try:
        with open(path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                if not any(mm.find(c) >= 0 for c in BUILD_CONTROL):
                    return caps
                for cap, sigs in BUILD_SIGNATURES.items():
                    caps[cap] = any(mm.find(s) >= 0 for s in sigs)
            finally:
                mm.close()
    except (OSError, ValueError):
        pass
    return caps


def wrf_auto_mesh(ranks):
    """(nproc_x, nproc_y) WRF picks when neither is set: MPASPECT (external/RSL_LITE/module_dm.F:98-
    133) -- the factor pair closest to square, ties to the smaller M, with M <= N."""
    best, mini = (1, ranks), 2 * ranks
    for m in range(1, ranks + 1):
        if ranks % m == 0 and abs(m - ranks // m) < mini:
            mini, best = abs(m - ranks // m), (m, ranks // m)
    return best


def chem_min_j_patch(e_sn_list, nproc_y):
    """Smallest j-patch (rows) over the domains: floor((e_sn-1)/nproc_y)."""
    return min((int(e) - 1) // int(nproc_y) for e in e_sn_list)


def chem_j_patch_reason(e_sn_list, nproc_y, nproc_x=None):
    """None when the split is acceptable for an active WRF-Chem case, else the printed reason."""
    if not e_sn_list or not nproc_y or nproc_y < 1:
        return None
    j = chem_min_j_patch(e_sn_list, nproc_y)
    if j >= CHEM_MIN_J_PATCH:
        return None
    return ('rejected %sx%s: WRF-Chem smallest j-patch %d rows < %d (CHEM_MIN_J_PATCH, PROVISIONAL: '
            'one case, exploratory) -- chem on with a narrow j-patch hung at the first post-t=0 '
            'history write (%s)'
            % (nproc_x if nproc_x else '?', nproc_y, j, CHEM_MIN_J_PATCH, CHEM_J_PATCH_NOTE))


def chem_e_sn_list(nl, max_dom=None):
    """e_sn of every active domain when WRF-Chem is active, else None."""
    if not any(d['id'] == 'chem' and d['active'] for d in detect(nl, None)):
        return None
    vals = [int(v) for v in _num_list(_vals(nl, 'e_sn', ('domains',)))]
    md = max_dom or (int(_num_list(_vals(nl, 'max_dom', ('domains',)))[0]) if _vals(nl, 'max_dom', ('domains',)) else 1)
    return vals[:md] or None


def chem_e_sn_for_namelist(namelist_path):
    try:
        return chem_e_sn_list(load_namelist(namelist_path))
    except Exception:
        return None


def chem_auto_split_finding(nl, ranks=None, nproc_x=None, nproc_y=None):
    """`wtune check` line when the (possibly automatic) split of an active WRF-Chem case has a
    too-narrow j-patch; None otherwise."""
    e_sn = chem_e_sn_list(nl)
    if not e_sn:
        return None
    nx = nproc_x if nproc_x else (_num_list(_vals(nl, 'nproc_x', ('domains',))) or [-1])[0]
    ny = nproc_y if nproc_y else (_num_list(_vals(nl, 'nproc_y', ('domains',))) or [-1])[0]
    nx, ny = int(nx), int(ny)
    how = 'namelist/--nproc'
    if ny < 1 and nx >= 1 and ranks:
        ny = ranks // nx
    elif nx < 1 and ny >= 1 and ranks:
        nx = ranks // ny
    elif nx < 1 and ny < 1:
        if not ranks:
            return None
        nx, ny = wrf_auto_mesh(int(ranks))
        how = 'WRF automatic split (MPASPECT) for %d ranks' % ranks
    r = chem_j_patch_reason(e_sn, ny, nx)
    return ('%s -- split from %s' % (r, how)) if r else None


def build_mismatches(dets, caps, exe):
    """FATAL messages for ACTIVE extensions the binary provably lacks."""
    msgs = []
    if not caps or not exe:
        return msgs
    how = {'chem': 'WRF_CHEM=1 / ./compile em_real with a chem build',
           'hydro': 'WRF_HYDRO=1', 'movenest': 'a moving-nest build (-DMOVE_NESTS, ./configure nesting option 2/3)'}
    for d in dets:
        if d['id'] in caps and d['active'] and caps[d['id']] is False:
            msgs.append('build mismatch: the namelist asks for %s (%s) but %s has no %s symbol '
                        '(%s not found); rebuild WRF with %s, or remove the %s settings'
                        % (d['label'], '; '.join(d['evidence']), exe, d['label'],
                           '/'.join(s.decode() for s in BUILD_SIGNATURES[d['id']]),
                           how[d['id']], d['label']))
    return msgs


# ---------------------------------------------------------------------------
# user-facing messages
# ---------------------------------------------------------------------------
def support_line(det):
    if det['id'] in NO_OPENMP:
        scope = ('OpenMP not supported by %s; tuning ranks/decomposition/tiles/I-O only'
                 % det['label'])
    else:
        scope = 'WTune will tune ranks/decomposition/tiles/OpenMP/I-O only'
    extra = (' Decomposition constraint (PROVISIONAL, one case): splits with a j-patch < %d rows '
             'are never proposed (WRF-Chem hung at narrow j-patches).' % CHEM_MIN_J_PATCH
             if det['id'] == 'chem' else '')
    return ('%s detected (%s): %s; %s are left exactly as given; %s.%s'
            % (det['label'], '; '.join(det['evidence']), scope, UNTOUCHED[det['id']],
               CALIBRATION_TEXT[det['id']], extra))


def omp_cap(nl, run_dir=None):
    """1 when an ACTIVE extension has no OpenMP support (WRF-Chem/Hydro/WRFDA builds are dmpar
    only), else None (no extension cap)."""
    return 1 if any(d['active'] and d['id'] in NO_OPENMP for d in detect(nl, run_dir)) else None


def omp_cap_for_namelist(namelist_path):
    try:
        return omp_cap(load_namelist(namelist_path), os.path.dirname(os.path.abspath(namelist_path)))
    except Exception:
        return None


def unsupported_line(label, msg, hits):
    return 'UNSUPPORTED %s detected (%s): %s. WTune will not tune this case.' % (
        label, ', '.join(hits), msg)


DA_ADVISORY_BANNER = ('ADVISORY ONLY (WRFDA): da_wrfvar.exe rank/decomposition recommendations '
                      'are printed for you and are never applied automatically; a bit-identical '
                      'check does not hold for DA minimisation.')


def check_report(namelist_path, run_dir, wrf_exe=None, ranks=None, nproc_x=None, nproc_y=None):
    """(lines, exit_code) for `wtune check`. Never raises. exit 2 on build mismatch, 1 on
    unsupported families, else 0."""
    lines, code = [], 0
    try:
        nl = load_namelist(namelist_path) if namelist_path and os.path.exists(namelist_path) else {}
    except Exception as exc:  # a parser bug must not hide the other checks
        return ['INFO EXT internal error reading %s: %r' % (namelist_path, exc)], 0
    dets = detect(nl, run_dir)
    for d in dets:
        note = '' if d['active'] else ' [group present but inactive]'
        lines.append('INFO EXT-%s %s%s' % (d['id'].upper(), support_line(d), note))
    for uid, label, msg, hits in detect_unsupported(run_dir):
        lines.append('WARN EXT-UNSUPPORTED-%s %s' % (uid.upper(), unsupported_line(label, msg, hits)))
        code = max(code, 1)
    if any(d['id'] == 'wrfda' for d in dets):
        lines.append('INFO EXT-WRFDA ' + DA_ADVISORY_BANNER)
    f = chem_auto_split_finding(nl, ranks, nproc_x, nproc_y)
    if f:
        lines.append('WARN EXT-CHEM-JPATCH ' + f)
        code = max(code, 1)
    exe = find_wrf_exe(run_dir, wrf_exe)
    if dets and exe:
        caps = scan_binary(exe)
        for m in build_mismatches(dets, caps, exe):
            lines.append('FATAL EXT-BUILD ' + m)
            code = 2
        if all(v is None for v in caps.values()):
            lines.append('INFO EXT-BUILD could not determine extension build capability of %s '
                         '(stripped binary or unreadable); namelist-only detection used' % exe)
    elif dets:
        lines.append('INFO EXT-BUILD no wrf.exe found (run dir wrf.exe, $WRF_TREE/run): build '
                     'capability not checked')
    return lines, code


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def _case_detections(case_dir):
    nml = os.path.join(case_dir, 'namelist.input')
    nl = {}
    if os.path.exists(nml):
        try:
            nl = load_namelist(nml)
        except Exception:
            nl = {}
    return detect(nl, case_dir), nl


def refusal_predict_rebuild(case_dir):
    """Message (str) if predict-rebuild must refuse this case, else None."""
    dets, _nl = _case_detections(case_dir)
    uns = detect_unsupported(case_dir)
    if uns:
        return ('predict-rebuild refused: %s' % unsupported_line(*uns[0][1:]))
    active = [d for d in dets if d['active']]
    if not active:
        return None
    labels = ' + '.join(d['label'] for d in active)
    what = '; '.join('%s: %s' % (d['label'], REBUILD_CANNOT[d['id']]) for d in active)
    return ('predict-rebuild refused: %s case (%s). predict-rebuild re-creates the initial state '
            'at several time points from ERA5/GFS with real.exe and cannot reconstruct %s. '
            'Use `wtune predict` (probe-based; the extension cost factor is shown with its '
            'measured/placeholder status) instead.'
            % (labels, '; '.join(d['evidence'][0] for d in active), what))


def refusal_run(case_dir, cfl_rescue=False, wrfda_forecast_only=False):
    """Message (str) if `wtune run` must refuse this case, else None."""
    dets, _nl = _case_detections(case_dir)
    uns = detect_unsupported(case_dir)
    if uns:
        return 'wtune run refused: %s' % unsupported_line(*uns[0][1:])
    by_id = {d['id']: d for d in dets if d['active']}
    if 'wrfda' in by_id and not wrfda_forecast_only:
        return ('wtune run refused: WRFDA case (%s). WTune does not run or orchestrate DA cycles '
                '(obsproc -> da_wrfvar.exe -> da_update_bc -> forecast); `wtune run` only '
                'segments wrf.exe forecasts. Run the DA cycle with your own workflow; to run '
                'only the forecast segments here, add --wrfda-forecast-only.'
                % '; '.join(by_id['wrfda']['evidence']))
    if cfl_rescue:
        hit = [by_id[k] for k in ('movenest', 'chem', 'hydro') if k in by_id]
        if hit:
            why = '; '.join('%s: %s' % (d['label'], CFL_COUPLED[d['id']]) for d in hit)
            return ('wtune run refused: --cfl-rescue would change time_step, which is not '
                    'allowed on %s cases (%s). Extension parameters are never touched; drop '
                    '--cfl-rescue and lower time_step yourself if the run hits CFL.'
                    % (' + '.join(d['label'] for d in hit), why))
    return None


def cfl_coupling_warnings(case_dir, old_dt, new_dt):
    """Warning lines for the advice-only CFL path: the advised time_step may break the coupling of
    extension parameters that WTune will not change. [] for plain ARW cases or no advice."""
    if new_dt is None:
        return []
    dets, nl = _case_detections(case_dir)
    by_id = {d['id']: d for d in dets if d['active']}
    out = []

    def ratio(period_s):
        return 'divisible' if period_s and abs((period_s / new_dt) - round(period_s / new_dt)) < 1e-9 \
            and round(period_s / new_dt) >= 1 else 'NOT an integer multiple of the advised time_step'

    if 'chem' in by_id:
        parts = []
        for k in ('chemdt', 'photdt'):
            v = _num_list(_vals(nl, k))
            if v:
                parts.append('%s=%g min (%s)' % (k, v[0], ratio(v[0] * 60.0)))
        out.append('chem: ' + (', '.join(parts) if parts else 'chemdt/photdt not set') + ' vs time_step')
    if 'hydro' in by_id:
        kv = _read_kv(by_id['hydro']['detail']['hydro_namelist']) if by_id['hydro']['detail'].get('hydro_namelist') else {}
        parts = ['%s=%g s (%s)' % (k.upper(), kv[k], ratio(kv[k])) for k in ('dtrt_ter', 'dtrt_ch') if k in kv]
        out.append('hydro: ' + (', '.join(parts) if parts else 'DTRT_TER/DTRT_CH not readable') + ' vs time_step')
    if 'movenest' in by_id:
        v = _num_list(_vals(nl, 'vortex_interval'))
        part = 'vortex_interval=%g min (%s); ' % (v[0], ratio(v[0] * 60.0)) if v else ''
        out.append('moving nest: %stime_to_move/move schedule are in steps/minutes tied to time_step' % part)
    return ['EXTENSION WARNING: advised time_step=%s s (case %g s) may break extension coupling -- %s. '
            'WTune will NOT change these; you must check them yourself before using the advised time_step.'
            % (new_dt, old_dt, o) for o in out]


def tune_gate(namelist_path):
    """(notes, refusal) for wtune tune/recommend on a namelist: one support line per active
    extension (tuning proceeds on the STRICT grid/rank/tile/OpenMP/I-O knob set only) and a
    refusal message for unsupported families found next to the namelist."""
    run_dir = os.path.dirname(os.path.abspath(namelist_path))
    try:
        nl = load_namelist(namelist_path)
    except Exception:
        return [], None
    dets = [d for d in detect(nl, run_dir) if d['active']]
    notes = [support_line(d) for d in dets]
    if any(d['id'] == 'wrfda' for d in dets):
        notes.append(DA_ADVISORY_BANNER)
    uns = detect_unsupported(run_dir)
    refusal = ('tune refused: ' + unsupported_line(*uns[0][1:])) if uns else None
    return notes, refusal


def run_notes(case_dir):
    """Informational lines for `wtune run` on extension cases (never refuse)."""
    dets, _nl = _case_detections(case_dir)
    notes = [support_line(d) for d in dets if d['active']]
    if any(d['id'] == 'movenest' and d['active'] for d in dets):
        notes.append('Moving nest: segments restart from wrfrst files; verify nest position '
                     'continuity across segment boundaries (untested).')
    return notes


# ---------------------------------------------------------------------------
# cost factors
# ---------------------------------------------------------------------------
def load_coefficients(calibration=None):
    with open(_COEFF_PATH) as f:
        coeffs = json.load(f)
    env = os.environ.get('WTUNE_EXTENSION_COEFFS')
    over = []
    if env and os.path.exists(env):
        with open(env) as f:
            over.append(json.load(f))
    if calibration and isinstance(calibration.get('extension_coefficients'), dict):
        over.append(calibration['extension_coefficients'])
    for o in over:
        for k, v in o.items():
            if isinstance(v, dict) and isinstance(coeffs.get(k), dict):
                coeffs[k].update(v)
            else:
                coeffs[k] = v
    return coeffs


def _read_kv(path):
    """Loose key=value scan (case-insensitive) of a Fortran-style file such as hydro.namelist."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = re.sub(r'!.*$', '', line)
                m = re.match(r'\s*([A-Za-z_]\w*)\s*=\s*([-+0-9.eEdD]+)', line)
                if m:
                    try:
                        out[m.group(1).lower()] = float(m.group(2).lower().replace('d', 'e'))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def cost_factors(nl, dt_s, run_dir=None, calibration=None):
    """Per-extension multiplicative compute factors for the detected ACTIVE extensions:
    [{id, label, multiplier, status: measured|placeholder|none, uncertainty_pct, terms, source}]."""
    coeffs = load_coefficients(calibration)
    out = []
    for d in detect(nl, run_dir):
        if not d['active']:
            continue
        c = coeffs.get(d['id'], {})
        f = dict(id=d['id'], label=d['label'], status=c.get('status', 'placeholder'),
                 uncertainty_pct=float(c.get('uncertainty_pct', 0.0)), source=c.get('source', ''),
                 multiplier=1.0, terms={})
        if d['id'] == 'movenest':
            f['terms'] = {'build_constant': c.get('const_frac', 0.0)}
        elif d['id'] == 'chem':
            opt = str(int(d['detail'].get('chem_opt', 0)))
            n_tr = c.get('n_tracers_by_chem_opt', {}).get(opt, c.get('default_n_tracers', 0))
            chemdt = _num_list(_vals(nl, 'chemdt'))
            step_frac = 1.0
            if chemdt and chemdt[0] > 0 and dt_s:
                step_frac = min(1.0, float(dt_s) / (chemdt[0] * 60.0))
            f['terms'] = {'tracers': c.get('frac_per_tracer', 0.0) * n_tr,
                          'chem_step': c.get('frac_per_chem_step', 0.0) * step_frac}
            f['n_tracers'] = n_tr
            f['n_tracers_source'] = 'chem_opt table' if opt in c.get('n_tracers_by_chem_opt', {}) \
                else 'default placeholder'
        elif d['id'] == 'hydro':
            kv = _read_kv(d['detail'].get('hydro_namelist')) if d['detail'].get('hydro_namelist') else {}
            agg = kv.get('aggfactrt', c.get('default_aggfactrt', 4))
            dtrt = kv.get('dtrt_ter') or dt_s or 1.0
            ratio = max(1.0, (float(dt_s) / dtrt) if dt_s else 1.0)
            f['terms'] = {'routing_grid': c.get('frac_per_routing_cell_ratio', 0.0) * agg * agg * ratio}
            f['aggfactrt'] = agg
        f['multiplier'] = 1.0 + sum(f['terms'].values())
        out.append(f)
    return out


# probe sources whose s/step was measured on THIS case's own binary+namelist: the extension cost
# is already inside that number, so applying the factor again would double count
_EMBEDDED_SOURCES = ('measured', 'probe_rsl', 'extrapolated', 'no_radiation')


def apply_cost_factors(compute, nl, dt_s, run_dir=None, calibration=None):
    """Scale a wtune_predict.predict_compute() result by the extension factors when it came from
    the no-probe fallback (calibrated on plain ARW runs). Returns the same dict, mutated, with
    'extension_factors' (always present when an extension is active, else absent) and widened
    noise_pct/noise_abs_s. Probe-based results are left numerically unchanged."""
    facts = cost_factors(nl, dt_s, run_dir, calibration)
    if not facts:
        return compute
    applies = compute.get('source') not in _EMBEDDED_SOURCES
    mult, widen2 = 1.0, 0.0
    for f in facts:
        f['applied'] = applies
        if applies:
            mult *= f['multiplier']
            widen2 += f['uncertainty_pct'] ** 2
        else:
            f['note'] = ('not applied: s/step probe (source=%s) was measured on this extension '
                         'case, so its cost is already included' % compute.get('source'))
    compute['extension_factors'] = facts
    compute['extension_multiplier'] = mult
    if applies:
        for k in ('s_normal', 'rad_extra_s', 'compute_total_s'):
            compute['%s_base' % k] = compute[k]
            compute[k] = compute[k] * mult
        noise = (compute['noise_pct'] ** 2 + widen2) ** 0.5
        compute['noise_pct_base'] = compute['noise_pct']
        compute['noise_pct'] = noise
        compute['noise_abs_s'] = compute['compute_total_s'] * noise / 100.0
        compute['extension_uncertainty_pct'] = widen2 ** 0.5
    else:
        compute['extension_uncertainty_pct'] = 0.0
    return compute


def render_factor_lines(compute):
    """Text lines for wtune_predict.render_text; empty when no extension is active."""
    out = []
    if any(f.get('applied') for f in compute.get('extension_factors') or ()):
        out.append('  (s/step and compute totals above already include the applied extension '
                   'factors; base s/step x%.3f)' % compute.get('extension_multiplier', 1.0))
    for f in compute.get('extension_factors') or ():
        out.append('  extension factor [%s]: x%.3f  status=%s  uncertainty +-%.0f%%  %s'
                   % (f['label'], f['multiplier'], f['status'].upper(), f['uncertainty_pct'],
                      'APPLIED' if f.get('applied') else 'NOT APPLIED'))
        if f.get('note'):
            out.append('    ' + f['note'])
        if f['status'] == 'placeholder':
            out.append('    UNCALIBRATED placeholder coefficients (wtune_extension_coefficients.json); '
                       'range widened accordingly')
        elif f['status'] == 'none':
            out.append('    no cost model for this extension')
    return out
