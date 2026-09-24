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
import re
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
                   'hardware_requirements', 'status', 'evidence', 'legal_values')

# legal_values schema (dict, one per catalog entry): the knob's legal value space and provenance.
#   type         'enum' | 'int' | 'real' | 'bool' | 'string'
#   values       non-empty list, required iff type == 'enum'
#   min, max     number or None (int/real only; min <= max when both set)
#   constraints  list[str] -- cross-field / case-dependent rules, e.g. pairing requirements
#   source       exact provenance: 'Registry.EM_COMMON:<line>', 'module_check_a_mundo.F:<line>',
#                repo 'file.py:<line>', or a locally-queryable tool/doc ('srun(1)', 'ld(1)', ...)
#   verified     True only if values/range were actually read from the file/tool cited in `source`
#                during this catalog's construction; otherwise False with 'range not verified from
#                local source' recorded in constraints
LEGAL_FIELDS = ('type', 'values', 'min', 'max', 'constraints', 'source', 'verified')
LEGAL_TYPES = ('enum', 'int', 'real', 'bool', 'string')

_SLUG_RE = re.compile(r'\[\[[a-z0-9-]+\]\]')  # wiki slugs are lowercase kebab-case


def _legal(type_, values=(), min=None, max=None, constraints=(), source='', verified=True):
    """Build a legal_values dict. Never invent option numbers: only pass verified=True when
    `values`/`min`/`max` were actually read from the file or tool named in `source`."""
    return {
        'type': type_,
        'values': list(values),
        'min': min,
        'max': max,
        'constraints': list(constraints),
        'source': source,
        'verified': bool(verified),
    }


def _unverified(type_='string', values=(), min=None, max=None, constraints=(),
                source='not available in this repo/.wrfref'):
    """The honest fallback: widest plausible type, no confirmed bound, verified=False."""
    return _legal(type_, values=values, min=min, max=max, source=source, verified=False,
                  constraints=['range not verified from local source'] + list(constraints))


def _k(name, kind, cls, numerics, phases, hw=(), status='unmeasured', evidence='', interposition=False,
       source_change=False, legal=None):
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
        'legal_values': legal if legal is not None else _unverified(),
    }


ALL = ('all',)
_S, _G, _R = STRICT, PHYSICS_GRAY, RESEARCH

CATALOG = [
    # ---- STRICT: WRF native work geometry ----
    _k('nproc_x', 'namelist', _S, 'roundoff', ALL, status='gain',
       evidence='8x56 -1~3% solver on CONUS 2.5km [[wrf-conus25-tuning-conclusion]]; bitwise across decompositions unverified',
       legal=_legal('int', min=-1, max=None, source='Registry.EM_COMMON:2354; module_check_a_mundo.F:3014-3029',
                    constraints=['-1 = unset/auto sentinel (Registry.EM_COMMON:2354); 0 and values <= -2 are '
                                 'illegal; explicit values must be >= 1',
                                 'nproc_x*nproc_y == mpi_ranks when both are set',
                                 '(e_we-1)/nproc_x >= 10 per rank required '
                                 '[[wrf-refuses-any-patch-thinner-than-10-cells]] (module_check_a_mundo.F:3014-3029)'])),
    _k('nproc_y', 'namelist', _S, 'roundoff', ALL, status='gain',
       evidence='paired with nproc_x [[wrf-conus25-tuning-conclusion]]',
       legal=_legal('int', min=-1, max=None, source='Registry.EM_COMMON:2355; module_check_a_mundo.F:3014-3029',
                    constraints=['-1 = unset/auto sentinel (Registry.EM_COMMON:2355); 0 and values <= -2 are '
                                 'illegal; explicit values must be >= 1',
                                 'nproc_x*nproc_y == mpi_ranks when both are set',
                                 '(e_sn-1)/nproc_y >= 10 per rank required '
                                 '[[wrf-refuses-any-patch-thinner-than-10-cells]] (module_check_a_mundo.F:3014-3029)'])),
    _k('numtiles', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), status='gain',
       evidence='even numtiles -1.4% on karina 224x2, diffwrf 0 [[even-numtiles-12-or-14-is-1-4-percent-faster-than-13-on-224x2-bit-identical-pristine-and-tile-size-is-not-the-reason]]',
       legal=_legal('int', min=1, max=64, source='Registry.EM_COMMON:2349; predictor/decomp_space.py:89',
                    constraints=['upper bound 64 is a WTune-imposed practical cap (predictor/decomp_space.py:89), '
                                 'not a WRF hard limit'])),
    _k('numtiles_x', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), status='null',
       evidence='2D tile mixer +15~23% (hybrid 56x8) [[wrf-openmp-tile-static-mixer-hybrid]]',
       legal=_legal('int', min=0, max=64, source='Registry.EM_COMMON:2351; predictor/decomp_space.py:89,236',
                    constraints=['0 = unused, defers to numtiles (Registry.EM_COMMON:2351)',
                                 'effective minimum is 1 when set (predictor/decomp_space.py:236)'])),
    _k('numtiles_y', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), status='null',
       evidence='paired with numtiles_x [[wrf-openmp-tile-static-mixer-hybrid]]',
       legal=_legal('int', min=0, max=64, source='Registry.EM_COMMON:2352; predictor/decomp_space.py:89,236',
                    constraints=['0 = unused, defers to numtiles (Registry.EM_COMMON:2352)',
                                 'effective minimum is 1 when set (predictor/decomp_space.py:236)'])),
    _k('tile_sz_x', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), evidence='unmeasured',
       legal=_legal('int', min=0, max=None, source='Registry.EM_COMMON:2347',
                    constraints=['0 = unused; WRF derives tile size from numtiles instead '
                                 '(frame/module_tiles.F set_tiles2, see predictor/decomp_space.py:12)'])),
    _k('tile_sz_y', 'namelist', _S, 'none', ALL, hw=('omp_threads>1',), evidence='unmeasured',
       legal=_legal('int', min=0, max=None, source='Registry.EM_COMMON:2348',
                    constraints=['0 = unused; WRF derives tile size from numtiles instead '
                                 '(frame/module_tiles.F set_tiles2, see predictor/decomp_space.py:12)'])),
    _k('mpi_ranks', 'launcher', _S, 'roundoff', ALL, status='gain',
       evidence='hybrid 224x2 with correct numtiles beats 448x1 by 5.7% (wtune README); bitwise across rank counts '
                'unverified [[wrf-conus25-tuning-conclusion]]',
       legal=_unverified('int', min=1, source='srun(1)',
                         constraints=['nproc_x*nproc_y == mpi_ranks when nproc_x/nproc_y are both set (>0) '
                                      '(Registry.EM_COMMON:2354-2355)'])),
    # ---- STRICT: runtime / placement ----
    _k('OMP_NUM_THREADS', 'env', _S, 'none', ALL, status='gain',
       evidence='224x2 hybrid; uniform OMP=4 on 2 cores +8.1~9.6%, diffwrf 0 [[no-wrap-elastic-core-loses-on-real-wrf-best-e-is-plus-2-percent-vs-vanilla-and-plov-stays-5-percent-ahead]]',
       legal=_legal('int', min=1, max=None,
                    source='libgomp info manual, node OMP_NUM_THREADS (OpenMP v4.5 sec 4.2); plov/production.env:6',
                    constraints=['may be a comma-separated list of positive integers for nested parallelism levels'])),
    _k('per_rank_omp_threads', 'launcher', _S, 'none', ('microphysics',), hw=('omp_threads>1',), status='closed',
       evidence='cost-weighted 1-5 cores/rank +7.9% [[the-micro-thread-controller-is-built-it-halves-the-imbalance-and-loses-7-9-percent]]',
       legal=_unverified('int', min=1, source='srun(1)',
                         constraints=["this catalog's own evidence tested cost-weighted 1-5 cores/rank "
                                      '(status closed, +7.9% regression)',
                                      'WRF-Fire (ifire>0): omp must be 1 -- crashes at first step '
                                      'otherwise [[wtune-one-node-decomposition-search-beats-40-random-'
                                      'configs-on-matthew-and-severe2023]]'])),
    _k('srun_distribution', 'launcher', _S, 'none', ALL, status='gain',
       evidence='cyclic -2.2%, diffwrf empty [[cyclic-rank-placement-plus-adaptive-node-local-stealing-is-7-7-percent-faster-per-step-bit-identical-with-no-source-change]]',
       legal=_legal('enum', values=['cyclic', 'block'], source='campaign/configs.py:95-110',
                    constraints=['srun(1) --distribution supports additional forms (e.g. plane, block:cyclic, '
                                 'arbitrary) not exercised in this repo; srun not installed locally to confirm '
                                 'the full grammar'])),
    _k('rank_mapping', 'launcher', _S, 'none', ALL, status='null',
       evidence='profile-guided LPT hostfile equalises node micro to 1%, s/step unchanged [[profile-guided-lpt-placement-equalises-node-micro-to-1-percent-but-s-step-is-unchanged-the-micro-rebalance-lever-is-at-its-ceiling]]',
       legal=_unverified('string', source='srun(1) --distribution=arbitrary + hostfile',
                         constraints=['implemented as a generated MPI rankfile, see the profile-guided LPT evidence'])),
    _k('cpu_bind', 'launcher', _S, 'none', ALL, status='gain',
       evidence='keep tight per-rank masks; --cpu-bind=sockets +15.7% (OMP=2) [[no-wrap-elastic-core-loses-on-real-wrf-best-e-is-plus-2-percent-vs-vanilla-and-plov-stays-5-percent-ahead]]',
       legal=_unverified('string', source='srun(1) --cpu-bind',
                         constraints=["this catalog's own evidence tested \"--cpu-bind=sockets\" only"])),
    _k('numa_bind', 'os', _S, 'none', ALL, hw=('numa',), status='null',
       evidence='hot ranks not NUMA/bandwidth bound [[hot-wrf-ranks-are-slower-only-because-they-execute-more-instructions-not-cache-bandwidth-numa-or-frequency]]',
       legal=_legal('enum', values=['interleave', 'membind', 'cpunodebind', 'physcpubind', 'localalloc',
                                     'preferred', 'preferred-many', 'strict'],
                    source='numactl(1)')),
    _k('OMP_PLACES', 'env', _S, 'none', ALL, evidence='unmeasured',
       legal=_legal('string', source='libgomp info manual, node OMP_PLACES (OpenMP v4.5 sec 4.5)',
                    constraints=['abstract names: threads, cores, sockets, ll_caches, numa_domains (optionally '
                                 'with a count in parentheses); or an explicit comma-separated list of '
                                 'thread-number sets in {}'])),
    _k('OMP_PROC_BIND', 'env', _S, 'none', ALL, status='gain',
       evidence='false in production; true + OMP=4 on 2 cores doubles region time [[external-re-pin-moves-a-running-openmp-worker-in-0-2-ms-and-cuts-the-remaining-region-move-not-widen-is-the-actuator]]',
       legal=_legal('enum', values=['true', 'false', 'primary', 'master', 'close', 'spread'],
                    source='libgomp info manual, node OMP_PROC_BIND (OpenMP v4.5 sec 4.4); plov/production.env:6',
                    constraints=['MASTER deprecated since OpenMP 5.1 in favor of PRIMARY'])),
    _k('KMP_BLOCKTIME', 'env', _S, 'none', ALL, hw=('intel_openmp',), status='gain',
       evidence='0 is -2.8% on 224x2 ABAB [[kmp-blocktime-zero-is-worth-2-8-percent-on-hybrid-wrf]]',
       legal=_legal('int', min=0, max=None, source='plov/production.env:5',
                    constraints=["0 tested as -2.8% on 224x2 (this catalog's own evidence)",
                                 'Intel OpenMP docs for the full legal range (e.g. "infinite") not available '
                                 'locally (icc/ifort not installed)'])),
    _k('OMP_WAIT_POLICY', 'env', _S, 'none', ALL, evidence='unmeasured',
       legal=_legal('enum', values=['ACTIVE', 'PASSIVE'],
                    source='libgomp info manual, node OMP_WAIT_POLICY (OpenMP v4.5 sec 4.8)')),
    _k('OMP_STACKSIZE', 'env', _S, 'none', ALL, hw=('omp_threads>1',), status='required',
       evidence='hybrid WRF segfaults with default worker stack [[wrf-openmp-tile-static-mixer-hybrid]]',
       legal=_legal('string',
                    source='libgomp info manual, node OMP_STACKSIZE (OpenMP v4.5 sec 4.7); plov/production.env:6',
                    constraints=['form: <positive number>[B|K|M|G] (kilobytes if unsuffixed); 512M used in '
                                 'production'])),
    _k('OMP_SCHEDULE', 'env', _S, 'none', ('microphysics',), status='closed',
       evidence='no effect on the static-init micro loop [[wrf-micro-tile-loop-is-static-init-and-no-runtime-env-or-compiler-flag-changes-iteration-ownership-only-a-source-schedule-clause-does]]',
       legal=_legal('string',
                    source='libgomp info manual, node OMP_SCHEDULE (OpenMP v4.5 sec 2.7.1.1, 4.1)',
                    constraints=['form: type[,chunk]; type in {static,dynamic,guided,auto}; chunk must be a '
                                 'positive integer'])),
    _k('KMP_SCHEDULE', 'env', _S, 'none', ('microphysics',), hw=('intel_openmp',), status='closed',
       evidence='static,greedy/balanced do not change ownership [[wrf-micro-tile-loop-is-static-init-and-no-runtime-env-or-compiler-flag-changes-iteration-ownership-only-a-source-schedule-clause-does]]',
       legal=_unverified('string', source='Intel OpenMP runtime docs (not installed locally)',
                         constraints=["this catalog's own evidence tested 'static,greedy' and 'static,balanced' "
                                      'as literal values (status closed, no effect)'])),
    _k('elastic_core_repin', 'os', _S, 'none', ('microphysics',), status='closed',
       evidence='external re-pin / core lending best +2.2% [[no-wrap-elastic-core-loses-on-real-wrf-best-e-is-plus-2-percent-vs-vanilla-and-plov-stays-5-percent-ahead]]',
       legal=_unverified('bool', source='taskset(1)/cgroups',
                         constraints=['mechanism is external re-pin/core lending, not a documented '
                                      'single-variable enum'])),
    # ---- STRICT: MPI runtime (loaded per detected implementation) ----
    _k('I_MPI_*', 'env', _S, 'roundoff', ('halo', 'mpi'), hw=('intel_mpi',),
       evidence='unmeasured: collective algorithm selection may change reduction order',
       legal=_unverified('string', source='Intel MPI docs (not installed locally)',
                         constraints=['prefix for arbitrary I_MPI_<PARAM> triples'])),
    _k('OMPI_MCA_*', 'env', _S, 'roundoff', ('halo', 'mpi'), hw=('openmpi_or_hpcx',), evidence='unmeasured',
       legal=_legal('string', source='ompi_info(1)',
                    constraints=['prefix for arbitrary OMPI_MCA_<framework>_<component>_<param> triples; '
                                 'enumerate the current build\'s legal params with '
                                 '`ompi_info --param all all --level 9` (873 params on this host, Open MPI 5.0.10)',
                                 'may change MPI collective algorithm selection and reduction order'])),
    _k('UCX_*', 'env', _S, 'roundoff', ('halo', 'mpi'), hw=('ucx',), evidence='unmeasured',
       legal=_legal('string', source='ucx_info(1)',
                    constraints=['prefix for arbitrary UCX_<COMPONENT>_<PARAM> triples; enumerate with '
                                 '`ucx_info -f` (openucx 1.20.1 installed on this host)',
                                 'may change transport/collective algorithm selection and reduction order'])),
    _k('MPICH_*', 'env', _S, 'roundoff', ('halo', 'mpi'), hw=('mpich',), evidence='unmeasured',
       legal=_unverified('string', source='MPICH docs (not installed locally)')),
    # ---- STRICT: I/O (output requirements unchanged) ----
    _k('io_form_history', 'io', _S, 'none', ('io',), hw=('adios2_build',), status='gain',
       evidence='ADIOS2 (14) -30% total wall on CONUS 2.5km, same field values [[wrf-formal-runs-adios2-backend]]',
       legal=_legal('enum', values=[2, 11, 14],
                    source='wtune_physics.py:319-320,671; campaign/configs.py:594,1117',
                    constraints=['confirmed-legal subset only; WRF defines additional io_form codes (e.g. '
                                 '1=binary, 4=PHDF5, 5/10=GRIB) not locally verified -- treat unlisted values as '
                                 'unverified, not illegal'])),
    _k('io_form_restart', 'io', _S, 'none', ('io',), hw=('adios2_build',), status='gain',
       evidence='ADIOS2 -29% total wall on CONUS 2.5km [[wrf-formal-runs-adios2-backend]]',
       legal=_legal('enum', values=[2, 14],
                    source='wtune_physics.py:671; campaign/configs.py:594,1074',
                    constraints=['confirmed-legal subset only; see io_form_history for the same caveat'])),
    _k('io_form_input', 'io', _S, 'none', ('io',), hw=('adios2_build',),
       evidence='unmeasured wall-time (read once at cold start); correctness: same field values on CONUS 2.5km'
                ' [[wrf-conus25-tuning-conclusion]]',
       legal=_legal('enum', values=[2, 14],
                    source='wtune_physics.py:671; campaign/configs.py:1074',
                    constraints=['confirmed-legal subset only; see io_form_history for the same caveat'])),
    _k('nio_tasks_per_group', 'io', _S, 'none', ('io',), evidence='unmeasured',
       legal=_legal('int', min=0, max=None, source='campaign/configs.py:639-644',
                    constraints=['nio_tasks_per_group=0 with nio_groups=1 disables I/O quilting '
                                 '(campaign/configs.py:644)'])),
    _k('nio_groups', 'io', _S, 'none', ('io',), evidence='unmeasured',
       legal=_legal('int', min=1, max=None, source='campaign/configs.py:639-644',
                    constraints=['nio_groups=1 is the disabled/default state paired with nio_tasks_per_group=0 '
                                 '(campaign/configs.py:644)'])),
    _k('adios2_aggregators', 'io', _S, 'none', ('io',), hw=('adios2_build',), evidence='unmeasured',
       legal=_legal('int', min=0, max=None,
                    source='campaign/configs.py:600-601,635; deprecated/segment_driver.py:135-136',
                    constraints=['0 = one aggregator per node (default) (external/io_adios2/wrf_io.F90:1296, '
                                 'cited at deprecated/segment_driver.py:135)',
                                 'MUST equal the MPI rank count when explicitly set, else BP corruption under '
                                 '--distribution=cyclic (campaign/configs.py:600-601)'])),
    # ---- STRICT: compiler (normal source/build semantics) ----
    _k('compiler_opt_level', 'build', _S, 'roundoff', ALL, evidence='unmeasured',
       legal=_unverified('string', source="compiler docs (this project's WRF builds use Intel ifort/icc per "
                                           'wiki note wrf-stack-build-recipe; not installed locally to query)')),
    _k('compiler_arch_target', 'build', _S, 'roundoff', ALL, evidence='unmeasured',
       legal=_unverified('string', source="compiler docs (this project's WRF builds use Intel ifort/icc per "
                                           'wiki note wrf-stack-build-recipe; not installed locally to query)')),
    _k('compiler_vectorization', 'build', _S, 'roundoff', ALL, evidence='unmeasured',
       legal=_unverified('string', source="compiler docs (this project's WRF builds use Intel ifort/icc per "
                                           'wiki note wrf-stack-build-recipe; not installed locally to query)')),
    _k('compiler_ipo_lto', 'build', _S, 'roundoff', ALL, evidence='unmeasured: only where compliance rules allow varying build flags',
       legal=_unverified('bool', source='compiler docs (icc -ipo / ifort -ipo / gcc -flto; Intel toolchain not '
                                         'installed locally)')),
    # ---- PHYSICS-GRAY: official physics / numerics options ----
    _k('mp_physics', 'namelist', _G, 'solution', ('microphysics',),
       evidence='unmeasured: scheme swap, advise-only behind fidelity gate',
       legal=_legal('enum',
                    values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 24, 26, 27, 28, 29, 38, 40,
                            50, 51, 52, 53, 55, 56, 88, 95, 96, 97, 106],
                    source='Registry.EM_COMMON:3069-3105',
                    constraints=['ARW-legal option set; no unconditional ARW exclusion found in '
                                 'module_check_a_mundo.F (only cu_physics=4 is explicitly barred for ARW)',
                                 'mp_physics=12 (milbrandt3mom) is commented out/disabled (Registry.EM_COMMON:3083)',
                                 'must be equal across all domains, auto-corrected to the innermost domain value '
                                 '(module_check_a_mundo.F:787-802)'])),
    _k('cu_physics', 'namelist', _G, 'solution', ('cumulus',),
       evidence='unmeasured: dx-dependent; never switched off for speed alone',
       legal=_legal('enum', values=[0, 1, 2, 3, 5, 6, 7, 10, 11, 14, 16, 93, 94, 95, 96, 99],
                    source='Registry.EM_COMMON:3254-3270; module_check_a_mundo.F:671-681',
                    constraints=['cu_physics=4 (scalesasscheme) is fatal for ARW; cu_physics=95 (osasscheme) is '
                                 'the suggested replacement (module_check_a_mundo.F:671-681)',
                                 'cu_physics=3 (gflscheme/GF) is fatal only when built with a compiler lacking an '
                                 'intrinsic gamma function (#if NO_GAMMA_SUPPORT, module_check_a_mundo.F:944-960) '
                                 '-- a build-time condition, not a blanket ARW exclusion, so it stays in the set',
                                 'must be equal across all domains, or zero (module_check_a_mundo.F:929-940)'])),
    _k('bl_pbl_physics', 'namelist', _G, 'solution', ('pbl',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16, 17, 99],
                    source='Registry.EM_COMMON:3227-3241')),
    _k('sf_sfclay_physics', 'namelist', _G, 'solution', ('surface',),
       evidence='unmeasured: must pair with PBL scheme',
       legal=_legal('enum', values=[0, 1, 2, 3, 4, 5, 7, 10, 89, 91], source='Registry.EM_COMMON:3191-3200')),
    _k('sf_surface_physics', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1, 2, 3, 4, 5, 6, 7, 8], source='Registry.EM_COMMON:3208-3216')),
    _k('sf_urban_physics', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1, 2, 3], source='Registry.EM_COMMON:2515,3203-3205',
                    constraints=['0=no, 1=Noah_UCM, 2=BEP_UCM (Registry.EM_COMMON:2515 description); 3=BEP_BEM '
                                 'has a package line but no rconfig description text (Registry.EM_COMMON:3205)'])),
    _k('ra_lw_physics', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured',
       legal=_legal('enum', values=[-1, 1, 3, 4, 5, 7, 14, 24, 31, 99],
                    source='Registry.EM_COMMON:2469,3170-3178',
                    constraints=['-1 = unset; only auto-resolved to 4 (RRTMG_LWSCHEME) inside physics_suite '
                                 "conus/tropical (module_check_a_mundo.F:3173,3190); otherwise it's the "
                                 'namelist default and legality elsewhere is not verified locally'])),
    _k('ra_sw_physics', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured',
       legal=_legal('enum', values=[-1, 1, 2, 3, 4, 5, 7, 14, 24, 99],
                    source='Registry.EM_COMMON:2470,3180-3188',
                    constraints=['-1 = unset; only auto-resolved to 4 (RRTMG_SWSCHEME) inside physics_suite '
                                 "conus/tropical (module_check_a_mundo.F:3174,3191); otherwise it's the "
                                 'namelist default and legality elsewhere is not verified locally'])),
    _k('physics_suite', 'namelist', _G, 'solution', ALL, evidence='unmeasured: official suites only',
       legal=_legal('enum', values=['none', 'conus', 'tropical'],
                    source='module_check_a_mundo.F:3143,3162-3200',
                    constraints=['case-insensitive (compared lowercase); any other value is fatal '
                                 '(module_check_a_mundo.F:3199-3200 CASE DEFAULT)'])),
    _k('radt', 'namelist', _G, 'solution', ('radiation',), status='gain',
       evidence='hydrostatic + radt -15.0% wall on Large 15km [[session-wrf-imbalance-closed-coarse-axis-won]]',
       legal=_legal('real', min=None, max=None, source='Registry.EM_COMMON:2473')),
    _k('cudt', 'namelist', _G, 'solution', ('cumulus',), evidence='unmeasured',
       legal=_legal('real', min=None, max=None, source='Registry.EM_COMMON:2523')),
    _k('non_hydrostatic', 'namelist', _G, 'solution', ('dynamics',), status='gain',
       evidence='-6%/step Matthew 30km; -15% combined with radt [[session-wrf-imbalance-closed-coarse-axis-won]]',
       legal=_legal('bool', source='Registry.EM_COMMON:2919')),
    _k('time_step', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured: CFL headroom',
       legal=_legal('int', min=1, max=None, source='Registry.EM_COMMON:2262; module_check_a_mundo.F:886-892',
                    constraints=['-1 default is fatal once use_wps_input==0 (module_check_a_mundo.F:886-892); '
                                 'must be a positive integer'])),
    _k('use_adaptive_time_step', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured',
       legal=_legal('bool', source='Registry.EM_COMMON:2280')),
    _k('h_sca_adv_order', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured',
       legal=_legal('int', min=None, max=None, source='Registry.EM_COMMON:2924')),
    _k('h_mom_adv_order', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured',
       legal=_legal('int', min=None, max=None, source='Registry.EM_COMMON:2922')),
    _k('diff_opt', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured',
       legal=_legal('int', min=None, max=None, source='Registry.EM_COMMON:2891')),
    _k('km_opt', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured',
       legal=_legal('int', min=None, max=None,
                    source='Registry.EM_COMMON:2895,3249; module_check_a_mundo.F:356-364',
                    constraints=['km_opt=5 (sms_3dtke) requires bl_pbl_physics=0, else fatal '
                                 '(module_check_a_mundo.F:356-364)',
                                 'other legal option numbers not enumerated locally'])),
    _k('diff_6th_opt', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured',
       legal=_legal('int', min=None, max=None, source='Registry.EM_COMMON:2909')),
    _k('epssm', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured',
       legal=_legal('real', min=None, max=None, source='Registry.EM_COMMON:2918')),
    _k('w_damping', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured',
       legal=_legal('int', min=None, max=None, source='Registry.EM_COMMON:2887')),
    _k('gwd_opt', 'namelist', _G, 'solution', ('dynamics',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1, 3], source='Registry.EM_COMMON:3316-3318')),
    _k('e_vert', 'namelist', _G, 'solution', ALL, evidence='unmeasured: needs re-initialisation',
       legal=_legal('int', min=None, max=None, source='Registry.EM_COMMON:2290',
                    constraints=['must be identical, or integer-refinement-compatible, across nested domains '
                                 '(module_check_a_mundo.F:2651-2696)'])),
    _k('swint_opt', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1, 2], source='Registry.EM_COMMON:2611')),
    _k('slope_rad', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1], source='Registry.EM_COMMON:2660')),
    _k('topo_shading', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1], source='Registry.EM_COMMON:2661')),
    _k('o3input', 'namelist', _G, 'solution', ('radiation',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 2], source='Registry.EM_COMMON:2609',
                    constraints=['Registry.EM_COMMON:2609 only documents 0 (original) and 2 (CAM ozone); no '
                                 'value 1 is described'])),
    _k('num_soil_layers', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured: must match the LSM',
       legal=_legal('enum', values=[2, 3, 4, 5, 6, 9, 10], source='module_check_a_mundo.F:3587-3612',
                    constraints=['forced to match sf_surface_physics: NOLSMSCHEME/SLABSCHEME->5, '
                                 'LSMSCHEME/NOAHMPSCHEME/CTSMSCHEME->4, RUCLSMSCHEME->6 or 9 (user choice), '
                                 'PXLSMSCHEME->2, CLMSCHEME->10, SSIBSCHEME->3 (module_check_a_mundo.F:3587-3612)'])),
    _k('sst_update', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1], source='Registry.EM_COMMON:2589')),
    _k('tmn_update', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1], source='Registry.EM_COMMON:2593')),
    _k('fractional_seaice', 'namelist', _G, 'solution', ('surface',), evidence='unmeasured',
       legal=_legal('enum', values=[0, 1], source='Registry.EM_COMMON:2664')),
    _k('bucket_mm', 'namelist', _G, 'solution', ('io',),
       evidence='unmeasured: changes stored accumulation fields',
       legal=_legal('real', min=None, max=None, source='Registry.EM_COMMON:2673',
                    constraints=['-1 = inactive (Registry.EM_COMMON:2673)'])),
    _k('prec_acc_dt', 'namelist', _G, 'solution', ('io',),
       evidence='unmeasured: adds/changes precipitation diagnostics',
       legal=_legal('real', min=None, max=None, source='Registry.EM_COMMON:2676',
                    constraints=['units: minutes (Registry.EM_COMMON:2676)'])),
    # ---- RESEARCH: execution-flow interposition ----
    _k('plov_micro_staging', 'interposition', _R, 'none', ('microphysics',), status='gain', interposition=True,
       evidence='oracle: cyclic + adaptive stealing -13.0% over 400 steps, bit-identical [[plov-production-config-holds-over-400-steps-minus-13-percent-and-the-gain-grows-as-the-storm-develops-bit-identical]]',
       legal=_legal('bool', source='plov/production.env:8-9',
                    constraints=['configured via WTIMER_STAGE_ROWS/SHARE/ADAPT/CHUNKS/MAX_PCT env vars '
                                 '(plov/production.env:8-9); production values ROWS=4 SHARE=1 ADAPT=1 CHUNKS=6 '
                                 'MAX_PCT=75'])),
    _k('plov_rad_staging', 'interposition', _R, 'none', ('radiation',), status='gain', interposition=True,
       evidence='-0.6% net on top of micro staging (wtune README) [[wtune-abandon-cross-node-and-cumulus-staging-ship-node-local-micro-rad-only]]',
       legal=_legal('bool', source='plov/production.env:12-13',
                    constraints=['configured via WTIMER_RAD_ROWS/BUF/ISPLIT/FULLJ/SHARE env vars '
                                 '(plov/production.env:13); optional, left commented (disabled) in the '
                                 'production recipe pending an adoption ruling'])),
    _k('plov_predictive', 'interposition', _R, 'none', ('microphysics',), status='closed', interposition=True,
       evidence='gate failed [[predictive-plov-uses-cold-slack-from-phase-start-but-moves-less-work-and-widens-node-spread-gate-failed]]',
       legal=_unverified('bool', source='not available in this working tree',
                         constraints=['implementation lives on remote branch plov-predictive-record, not '
                                      'present in this working tree'])),
    _k('link_wrap', 'interposition', _R, 'none', ALL, interposition=True,
       evidence='unmeasured: -Wl,--wrap driver trampolines; mechanism for PLOV',
       legal=_legal('string', source='ld(1) --wrap',
                    constraints=['value is a linker flag -Wl,--wrap,<symbol>; symbol must be an undefined '
                                 'reference, resolved to __wrap_<symbol> (ld(1))'])),
    _k('ld_preload_shim', 'interposition', _R, 'none', ('mpi',), interposition=True,
       evidence='unmeasured: PMPI / LD_PRELOAD profilers and shims',
       legal=_legal('string', source='ld.so(8) LD_PRELOAD',
                    constraints=['colon/whitespace-separated list of ELF shared object paths preloaded before '
                                 'other libraries (ld.so(8))'])),
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
        evidence = e.get('evidence', '')
        if e['status'] == 'unmeasured':
            if not evidence.startswith('unmeasured'):
                errs.append("%s: status 'unmeasured' requires evidence to start with 'unmeasured'" % tag)
        elif not _SLUG_RE.search(evidence):
            errs.append("%s: status %r requires evidence to contain at least one [[slug]]" % (tag, e['status']))
        lv = e['legal_values']
        if not isinstance(lv, dict):
            errs.append('%s: legal_values must be a dict' % tag)
        else:
            missing_lv = [f for f in LEGAL_FIELDS if f not in lv]
            if missing_lv:
                errs.append('%s: legal_values missing fields %s' % (tag, ','.join(missing_lv)))
            else:
                if lv['type'] not in LEGAL_TYPES:
                    errs.append('%s: legal_values type %r not in %s' % (tag, lv['type'], LEGAL_TYPES))
                if lv['type'] == 'enum' and not lv['values']:
                    errs.append('%s: legal_values type enum requires non-empty values' % tag)
                if not isinstance(lv['constraints'], list):
                    errs.append('%s: legal_values constraints must be a list' % tag)
                if not isinstance(lv['verified'], bool):
                    errs.append('%s: legal_values verified must be a bool' % tag)
                if not lv['source']:
                    errs.append('%s: legal_values source must be non-empty' % tag)
                mn, mx = lv.get('min'), lv.get('max')
                if mn is not None and mx is not None and mn > mx:
                    errs.append('%s: legal_values min > max (%r > %r)' % (tag, mn, mx))
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
