"""Compliance model and machine-readable knob catalog for WTune (decision 0050). Stdlib only.

Three classes, never ranked together:
  STRICT        WRF source unchanged, normal build, simulation semantics unchanged
                (decomposition, tiling, placement, binding, OpenMP/MPI runtime, compiler flags, I/O layout).
  PHYSICS-GRAY  WRF source unchanged, official namelist/physics only, but the numerical/physical solution may
                change (scheme swaps, radt, cudt, hydrostatic, ...). Recommend and test behind a fidelity gate;
                passing a WTune gate is NOT an official ruling (report as PHYSICS-GRAY / UNCONFIRMED).
  RESEARCH      changes the execution flow of the binary/application (PLOV, --wrap, LD_PRELOAD, owner/helper
                work migration). Upper bound / oracle only.

Every catalog entry carries the fields in REQUIRED_FIELDS. numerics_change is three-valued:
  none      output expected bit-identical to the reference        -> gate 'diffwrf'
  roundoff  same algorithm, floating-point roundoff may differ     -> gate 'roundoff_gate' (tight field tolerance)
  solution  the computed solution legitimately changes             -> gate 'fidelity_gate'

usage:
  python3 wtune_compliance.py validate
  python3 wtune_compliance.py list [--compliance strict|physics-gray|research] [--json]
  python3 wtune_compliance.py classify NAME [NAME ...]
"""
import argparse
import json
import sys

STRICT = 'STRICT'
PHYSICS_GRAY = 'PHYSICS-GRAY'
RESEARCH = 'RESEARCH'
UNCLASSIFIED = 'UNCLASSIFIED'
CLASSES = (STRICT, PHYSICS_GRAY, RESEARCH)

# --compliance level -> classes it admits (each level includes the stricter ones)
LEVELS = {
    'strict': (STRICT,),
    'physics-gray': (STRICT, PHYSICS_GRAY),
    'research': (STRICT, PHYSICS_GRAY, RESEARCH),
}

NUMERICS = ('none', 'roundoff', 'solution')
GATE_FOR_NUMERICS = {'none': 'diffwrf', 'roundoff': 'roundoff_gate', 'solution': 'fidelity_gate'}
KINDS = ('namelist', 'launcher', 'env', 'build', 'io', 'os', 'interposition')
STATUSES = ('gain', 'null', 'closed', 'unmeasured', 'required')

REQUIRED_FIELDS = ('name', 'kind', 'classification', 'source_change', 'numerics_change',
                   'execution_interposition', 'requires_correctness_gate', 'applicable_phases',
                   'hardware_requirements', 'status', 'evidence')


def _k(name, kind, cls, numerics, phases, hw=(), status='unmeasured', evidence='', interposition=False,
       source_change=False):
    return {
        'name': name,
        'kind': kind,
        'classification': cls,
        'source_change': source_change,
        'numerics_change': numerics,
        'execution_interposition': interposition,
        'requires_correctness_gate': GATE_FOR_NUMERICS[numerics],
        'applicable_phases': list(phases),
        'hardware_requirements': list(hw),
        'status': status,
        'evidence': evidence,
    }


ALL = ('all',)
_S, _G, _R = STRICT, PHYSICS_GRAY, RESEARCH

CATALOG = [
    # ---- STRICT: WRF native work geometry ----
    _k('nproc_x', 'namelist', _S, 'roundoff', ALL, status='gain',
       evidence='8x56 -1~3% solver on CONUS 2.5km [[wrf-conus25-tuning-conclusion]]; bitwise across decompositions unverified'),
    _k('nproc_y', 'namelist', _S, 'roundoff', ALL, status='gain',
       evidence='paired with nproc_x [[wrf-conus25-tuning-conclusion]]'),
    _k('numtiles', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), status='gain',
       evidence='even numtiles -1.4% on karina 224x2, diffwrf 0 [[even-numtiles-12-or-14-is-1-4-percent-faster-than-13-on-224x2-bit-identical-pristine-and-tile-size-is-not-the-reason]]'),
    _k('numtiles_x', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), status='null',
       evidence='2D tile mixer +15~23% (hybrid 56x8) [[wrf-openmp-tile-static-mixer-hybrid]]'),
    _k('numtiles_y', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), status='null',
       evidence='paired with numtiles_x [[wrf-openmp-tile-static-mixer-hybrid]]'),
    _k('tile_sz_x', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), evidence='unmeasured'),
    _k('tile_sz_y', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), evidence='unmeasured'),
    _k('mpi_ranks', 'launcher', _S, 'roundoff', ALL, status='gain',
       evidence='hybrid 224x2 with correct numtiles beats 448x1 by 5.7% (wtune README); bitwise across rank counts unverified'),
    # ---- STRICT: runtime / placement ----
    _k('OMP_NUM_THREADS', 'env', _S, 'none', ALL, status='gain',
       evidence='224x2 hybrid; uniform OMP=4 on 2 cores +8.1~9.6%, diffwrf 0 [[no-wrap-elastic-core-loses-on-real-wrf-best-e-is-plus-2-percent-vs-vanilla-and-plov-stays-5-percent-ahead]]'),
    _k('per_rank_omp_threads', 'launcher', _S, 'none', ('microphysics',), hw=('omp_threads>1',), status='closed',
       evidence='cost-weighted 1-5 cores/rank +7.9% [[the-micro-thread-controller-is-built-it-halves-the-imbalance-and-loses-7-9-percent]]'),
    _k('srun_distribution', 'launcher', _S, 'none', ALL, status='gain',
       evidence='cyclic -2.2%, diffwrf empty [[cyclic-rank-placement-plus-adaptive-node-local-stealing-is-7-7-percent-faster-per-step-bit-identical-with-no-source-change]]'),
    _k('rank_mapping', 'launcher', _S, 'none', ALL, status='null',
       evidence='profile-guided LPT hostfile equalises node micro to 1%, s/step unchanged [[profile-guided-lpt-placement-equalises-node-micro-to-1-percent-but-s-step-is-unchanged-the-micro-rebalance-lever-is-at-its-ceiling]]'),
    _k('cpu_bind', 'launcher', _S, 'none', ALL, status='gain',
       evidence='keep tight per-rank masks; --cpu-bind=sockets +15.7% (OMP=2) [[no-wrap-elastic-core-loses-on-real-wrf-best-e-is-plus-2-percent-vs-vanilla-and-plov-stays-5-percent-ahead]]'),
    _k('numa_bind', 'os', _S, 'none', ALL, hw=('numa',), status='null',
       evidence='hot ranks not NUMA/bandwidth bound [[hot-wrf-ranks-are-slower-only-because-they-execute-more-instructions-not-cache-bandwidth-numa-or-frequency]]'),
    _k('OMP_PLACES', 'env', _S, 'none', ALL, evidence='unmeasured'),
    _k('OMP_PROC_BIND', 'env', _S, 'none', ALL, status='gain',
       evidence='false in production; true + OMP=4 on 2 cores doubles region time [[external-re-pin-moves-a-running-openmp-worker-in-0-2-ms-and-cuts-the-remaining-region-move-not-widen-is-the-actuator]]'),
    _k('KMP_BLOCKTIME', 'env', _S, 'none', ALL, hw=('intel_openmp',), status='gain',
       evidence='0 is -2.8% on 224x2 ABAB [[kmp-blocktime-zero-is-worth-2-8-percent-on-hybrid-wrf]]'),
    _k('OMP_WAIT_POLICY', 'env', _S, 'none', ALL, evidence='unmeasured'),
    _k('OMP_STACKSIZE', 'env', _S, 'none', ALL, hw=('omp_threads>1',), status='required',
       evidence='hybrid WRF segfaults with default worker stack [[wrf-openmp-tile-static-mixer-hybrid]]'),
    _k('OMP_SCHEDULE', 'env', _S, 'none', ('microphysics',), status='closed',
       evidence='no effect on the static-init micro loop [[wrf-micro-tile-loop-is-static-init-and-no-runtime-env-or-compiler-flag-changes-iteration-ownership-only-a-source-schedule-clause-does]]'),
    _k('KMP_SCHEDULE', 'env', _S, 'none', ('microphysics',), hw=('intel_openmp',), status='closed',
       evidence='static,greedy/balanced do not change ownership [[wrf-micro-tile-loop-is-static-init-and-no-runtime-env-or-compiler-flag-changes-iteration-ownership-only-a-source-schedule-clause-does]]'),
    _k('elastic_core_repin', 'os', _S, 'none', ('microphysics',), status='closed',
       evidence='external re-pin / core lending best +2.2% [[no-wrap-elastic-core-loses-on-real-wrf-best-e-is-plus-2-percent-vs-vanilla-and-plov-stays-5-percent-ahead]]'),
    # ---- STRICT: MPI runtime (loaded per detected implementation) ----
    _k('I_MPI_*', 'env', _S, 'roundoff', ('halo', 'mpi'), hw=('intel_mpi',),
       evidence='collective algorithm selection may change reduction order; unmeasured'),
    _k('OMPI_MCA_*', 'env', _S, 'roundoff', ('halo', 'mpi'), hw=('openmpi_or_hpcx',), evidence='unmeasured'),
    _k('UCX_*', 'env', _S, 'roundoff', ('halo', 'mpi'), hw=('ucx',), evidence='unmeasured'),
    _k('MPICH_*', 'env', _S, 'roundoff', ('halo', 'mpi'), hw=('mpich',), evidence='unmeasured'),
    # ---- STRICT: I/O (output requirements unchanged) ----
    _k('io_form_history', 'io', _S, 'none', ('io',), hw=('adios2_build',), status='gain',
       evidence='ADIOS2 (14) -30% total wall on CONUS 2.5km, same field values'),
    _k('io_form_restart', 'io', _S, 'none', ('io',), hw=('adios2_build',), status='gain',
       evidence='ADIOS2 -29% total wall on CONUS 2.5km'),
    _k('io_form_input', 'io', _S, 'none', ('io',), hw=('adios2_build',), status='gain',
       evidence='same field values on CONUS 2.5km'),
    _k('nio_tasks_per_group', 'io', _S, 'none', ('io',), evidence='unmeasured'),
    _k('nio_groups', 'io', _S, 'none', ('io',), evidence='unmeasured'),
    _k('adios2_aggregators', 'io', _S, 'none', ('io',), hw=('adios2_build',), evidence='unmeasured'),
    # ---- STRICT: compiler (normal source/build semantics) ----
    _k('compiler_opt_level', 'build', _S, 'roundoff', ALL, evidence='unmeasured'),
    _k('compiler_arch_target', 'build', _S, 'roundoff', ALL, evidence='unmeasured'),
    _k('compiler_vectorization', 'build', _S, 'roundoff', ALL, evidence='unmeasured'),
    _k('compiler_ipo_lto', 'build', _S, 'roundoff', ALL, evidence='only where rules allow; unmeasured'),
    # ---- PHYSICS-GRAY: official physics / numerics options ----
    _k('mp_physics', 'namelist', _G, 'solution', ('microphysics',), evidence='scheme swap, advise-only behind fidelity gate'),
    _k('cu_physics', 'namelist', _G, 'solution', ('cumulus',), evidence='dx-dependent; never switched off for speed alone'),
    _k('bl_pbl_physics', 'namelist', _G, 'solution', ('pbl',), evidence='unmeasured'),
    _k('sf_sfclay_physics', 'namelist', _G, 'solution', ('surface',), evidence='must pair with PBL scheme'),
    _k('sf_surface_physics', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured'),
    _k('sf_urban_physics', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured'),
    _k('ra_lw_physics', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured'),
    _k('ra_sw_physics', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured'),
    _k('physics_suite', 'namelist', _G, 'solution', ALL, evidence='official suites only'),
    _k('radt', 'namelist', _G, 'solution', ('radiation',), status='gain',
       evidence='hydrostatic + radt -15.0% wall on Large 15km [[session-wrf-imbalance-closed-coarse-axis-won]]'),
    _k('cudt', 'namelist', _G, 'solution', ('cumulus',), evidence='unmeasured'),
    _k('non_hydrostatic', 'namelist', _G, 'solution', ('dynamics',), status='gain',
       evidence='-6%/step Matthew 30km; -15% combined with radt [[session-wrf-imbalance-closed-coarse-axis-won]]'),
    _k('time_step', 'namelist', _G, 'solution', ('dynamics',), evidence='CFL headroom; unmeasured'),
    _k('use_adaptive_time_step', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured'),
    _k('h_sca_adv_order', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured'),
    _k('h_mom_adv_order', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured'),
    _k('diff_opt', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured'),
    _k('km_opt', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured'),
    _k('diff_6th_opt', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured'),
    _k('epssm', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured'),
    _k('w_damping', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured'),
    _k('gwd_opt', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured'),
    _k('e_vert', 'namelist', _G, 'solution', ALL, evidence='needs re-initialisation; unmeasured'),
    _k('swint_opt', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured'),
    _k('slope_rad', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured'),
    _k('topo_shading', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured'),
    _k('o3input', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured'),
    _k('num_soil_layers', 'namelist', _G, 'solution', ('surface',), evidence='must match the LSM'),
    _k('sst_update', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured'),
    _k('tmn_update', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured'),
    _k('fractional_seaice', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured'),
    _k('bucket_mm', 'namelist', _G, 'solution', ('io',), evidence='changes stored accumulation fields'),
    _k('prec_acc_dt', 'namelist', _G, 'solution', ('io',), evidence='adds/changes precipitation diagnostics'),
    # ---- RESEARCH: execution-flow interposition ----
    _k('plov_micro_staging', 'interposition', _R, 'none', ('microphysics',), status='gain', interposition=True,
       evidence='oracle: cyclic + adaptive stealing -13.0% over 400 steps, bit-identical [[plov-production-config-holds-over-400-steps-minus-13-percent-and-the-gain-grows-as-the-storm-develops-bit-identical]]'),
    _k('plov_rad_staging', 'interposition', _R, 'none', ('radiation',), status='gain', interposition=True,
       evidence='-0.6% net on top of micro staging (wtune README)'),
    _k('plov_predictive', 'interposition', _R, 'none', ('microphysics',), status='closed', interposition=True,
       evidence='gate failed [[predictive-plov-uses-cold-slack-from-phase-start-but-moves-less-work-and-widens-node-spread-gate-failed]]'),
    _k('link_wrap', 'interposition', _R, 'none', ALL, interposition=True,
       evidence='-Wl,--wrap driver trampolines; mechanism for PLOV'),
    _k('ld_preload_shim', 'interposition', _R, 'none', ('mpi',), interposition=True,
       evidence='PMPI / LD_PRELOAD profilers and shims'),
]


def validate(catalog=None):
    """Return a list of human-readable consistency errors (empty = valid)."""
    catalog = CATALOG if catalog is None else catalog
    errs, seen = [], set()
    for i, e in enumerate(catalog):
        tag = e.get('name', '#%d' % i)
        missing = [f for f in REQUIRED_FIELDS if f not in e]
        if missing:
            errs.append('%s: missing fields %s' % (tag, ','.join(missing)))
            continue
        if tag in seen:
            errs.append('%s: duplicate name' % tag)
        seen.add(tag)
        cls, num = e['classification'], e['numerics_change']
        if cls not in CLASSES:
            errs.append('%s: classification %r not in %s' % (tag, cls, CLASSES))
        if num not in NUMERICS:
            errs.append('%s: numerics_change %r not in %s' % (tag, num, NUMERICS))
        if e['kind'] not in KINDS:
            errs.append('%s: kind %r not in %s' % (tag, e['kind'], KINDS))
        if e['status'] not in STATUSES:
            errs.append('%s: status %r not in %s' % (tag, e['status'], STATUSES))
        for f in ('source_change', 'execution_interposition'):
            if not isinstance(e[f], bool):
                errs.append('%s: %s must be bool' % (tag, f))
        for f in ('applicable_phases', 'hardware_requirements'):
            if not isinstance(e[f], list):
                errs.append('%s: %s must be a list' % (tag, f))
        if not e['applicable_phases']:
            errs.append('%s: applicable_phases is empty' % tag)
        if e['execution_interposition'] and cls != RESEARCH:
            errs.append('%s: execution_interposition requires RESEARCH, got %s' % (tag, cls))
        if e['source_change'] and cls != RESEARCH:
            errs.append('%s: source_change requires RESEARCH, got %s' % (tag, cls))
        if cls == STRICT and num == 'solution':
            errs.append('%s: STRICT cannot change the solution (numerics_change=solution)' % tag)
        if cls == PHYSICS_GRAY and num != 'solution':
            errs.append('%s: PHYSICS-GRAY must declare numerics_change=solution, got %s' % (tag, num))
        if num in GATE_FOR_NUMERICS and e['requires_correctness_gate'] != GATE_FOR_NUMERICS[num]:
            errs.append('%s: gate %r does not match numerics_change=%s (expected %r)'
                        % (tag, e['requires_correctness_gate'], num, GATE_FOR_NUMERICS[num]))
    return errs


_BY_NAME = {e['name']: e for e in CATALOG}
_RANK = {STRICT: 0, PHYSICS_GRAY: 1, RESEARCH: 2, UNCLASSIFIED: 3}


def lookup(name):
    """Catalog entry for a knob name, or None. Wildcard entries such as 'I_MPI_*' match by prefix."""
    if name in _BY_NAME:
        return _BY_NAME[name]
    for key, e in _BY_NAME.items():
        if key.endswith('*') and name.startswith(key[:-1]):
            return e
    return None


def classify(name):
    """Classification of a knob name. Combined names like 'slope_rad/topo_shading' take the least strict part;
    names not in the catalog are UNCLASSIFIED."""
    parts = [p.strip() for p in str(name).split('/') if p.strip()] or [str(name)]
    worst = STRICT
    for p in parts:
        e = lookup(p)
        c = e['classification'] if e else UNCLASSIFIED
        if _RANK[c] > _RANK[worst]:
            worst = c
    return worst


def admits(level, classification):
    """True if a --compliance level admits a classification. UNCLASSIFIED is never admitted."""
    if level not in LEVELS:
        raise ValueError('unknown compliance level %r (choose from %s)' % (level, ', '.join(LEVELS)))
    return classification in LEVELS[level]


def filter_names(names, level):
    return [n for n in names if admits(level, classify(n))]


def entries(level='research'):
    return [e for e in CATALOG if admits(level, e['classification'])]


def _main(argv=None):
    ap = argparse.ArgumentParser(description='WTune compliance model and knob catalog (decision 0050)')
    sub = ap.add_subparsers(dest='cmd')
    sub.add_parser('validate', help='check catalog consistency; exit 1 on any error')
    lp = sub.add_parser('list', help='list catalog entries admitted by a compliance level')
    lp.add_argument('--compliance', choices=sorted(LEVELS), default='research')
    lp.add_argument('--json', action='store_true')
    cp = sub.add_parser('classify', help='classify knob names')
    cp.add_argument('names', nargs='+')
    args = ap.parse_args(argv)
    if args.cmd == 'validate':
        errs = validate()
        for m in errs:
            print('ERROR ' + m)
        n = {c: sum(1 for e in CATALOG if e['classification'] == c) for c in CLASSES}
        print('catalog entries=%d STRICT=%d PHYSICS-GRAY=%d RESEARCH=%d errors=%d'
              % (len(CATALOG), n[STRICT], n[PHYSICS_GRAY], n[RESEARCH], len(errs)))
        return 1 if errs else 0
    if args.cmd == 'list':
        rows = entries(args.compliance)
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            for e in rows:
                print('%-13s %-24s %-13s numerics=%-8s gate=%-13s status=%-10s phases=%s'
                      % (e['classification'], e['name'], e['kind'], e['numerics_change'],
                         e['requires_correctness_gate'], e['status'], ','.join(e['applicable_phases'])))
        return 0
    if args.cmd == 'classify':
        for nm in args.names:
            print('%s %s' % (nm, classify(nm)))
        return 0
    ap.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(_main())
