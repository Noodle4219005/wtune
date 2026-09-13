# Redistribution runtime development

These modules implement the reusable C runtime beneath future WRF phase
adapters. They are not linked into the production WRF wrapper build yet.
The existing microphysics/radiation implementation remains separate.

| Component | Contract |
| --- | --- |
| `gen_phase_schema.py` | Conservative compiled `.f90` interface audit; unknown ABI stays unsupported. |
| `gen_ref_bridge.py` | Generates a C `--wrap` bridge for audited pointer-only arguments; compiler handles the calling convention. |
| `wt_cumulus.c` | KF CPS owner-local transaction with explicit capacities, snapshots and validation before writes. |
| `wt_sfclay.c`, `gen_sfclay_schema.py` | Audited 84-argument surface transaction with caller-provided capacities, preflight before copies, preserved OUT backing and one-shot validated commit. |
| `gen_kf_bridge.py`, `wt_kf_bridge.c` | Source-hash/order-checked 79-argument CPS bridge; source audit alone is not native object ABI verification. |
| `wt_kf_dispatch.c` | Runtime OMP/capacity gates, exactly one native call, fatal validation failure after invocation. |
| `wt_kf_capacity.c` | Per-call capacities with pointer identity and live generation; optional fully attested automatic-array class. |
| `wt_kf_live.c` | Outer-driver scope publication, actual OMP query, exclusive staging, counters and scope recheck before commit. |
| `wt_kf_native.c`, `wt_grid_capture.f90` | Compiler-matched grid capture and immutable grid selection; native module ABI must be attested. |
| `wt_kf_bootstrap.c` | Build-proof gates, strict resource budgets, initialization and actual staging diagnostics. |
| `wt_live_spans.c`, `wt_grid_spans.f90` | Generation-scoped typed capacity from current contiguous field pointers; unregistered fields remain unknown. |
| `wt_mpi_channel.c` | Private communicator, bounded owned buffers, nonblocking progress, deadline poisoning with pending requests retained. |
| `wt_mpi_remote_owner.c`, `wt_mpi_remote_worker.c`, `wt_mpi_remote_common.c` | Caller-driven asynchronous MPI transport composed with policy and ledger; trusted endpoints and at-most-once callbacks. |
| `wt_wire.c` | Bounded canonical little-endian frame codec; CRC detects accidental corruption. |
| `wt_policy.c` | Caller-supplied topology/cost estimates, deterministic gain gate and private reservation accounting. |
| `wt_stream.c` | Framed I/O over an existing nonblocking stream socket, with deadlines and terminal channel poisoning. |
| `wt_ledger.c`, `wt_ledger_validate.c` | Bounded owner/worker deduplication, cached replies and at-most-once callbacks during a process lifetime. |
| `wt_remote.c` | Synchronous policy/reservation/ledger/stream exchange with a trusted endpoint binding. |

All contexts require caller serialization and trusted callbacks. Payload
capacities must describe real backing storage. The wire codec does not establish
phase closure, field lifetime, peer authentication, or crash recovery.
The caller must supply canonical payload bytes and compatible schema/ABI values.
An execution failure never authorizes retrying the original physics callback.

The remote fixture uses real local `fork`/`socketpair` processes. It does not
demonstrate MPI progress, cross-node WRF execution, native state equivalence,
or performance gains. PBL, surface, dynamics and halo still require native
adapters and full-state validation. In particular, the allocation table in
`wt_alloc.c` lacks deallocation tracking and cannot serve as a universal
live-capacity provider.

## Tests with Pixi

Use a Pixi environment containing Python 3.11, a C11 compiler and, for the
sanitized checks, its matching sanitizer runtimes. Python test launchers honor
`CC`; several C fixtures also accept `CFLAGS`. The complete suite additionally
requires a modern Fortran compiler with `IS_CONTIGUOUS` and an MPI development
toolchain. The live-span fixture uses a substitute `module_domain_type`, not
the native WRF module. The MPI fixture honors `MPICC`/`MPIEXEC` and currently
uses MPICH's explicit local `-launcher fork -n 2`; it never requests Slurm.

From the WTune repository root:

```sh
pixi run --manifest-path /path/to/dev/pixi.toml python -m unittest discover -s plov/tests -p 'test_*.py' -v
```

The development environment used for the recorded Linux verification is
`/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/pixi.toml`.
This local path is an example, not a runtime dependency. No hostname, rank
count, NUMA layout, WRF case name or cluster allocation is required by the
portable core tests. The reference-bridge fixture requires GNU-compatible
linker `--wrap`; native Fortran ABI verification is separate.

To link the remote core into a standalone application, compile these translation
units together: `wt_wire.c`, `wt_policy.c`, `wt_stream.c`, `wt_ledger.c`,
`wt_ledger_validate.c`, and `wt_remote.c`, with `plov/` on the include path.
Do not omit the ledger validation translation unit.

## Optional cumulus scope hooks

`wtimer.S` defines outer-driver entry/exit hook calls only when compiled with
`WT_CUMULUS_LIVE_SPANS`. With the macro absent, its preprocessed assembly stays
unchanged. The hooks preserve the existing Intel x86-64 argument trampoline;
they do not make its static slots recursive or safe for simultaneous outer
driver calls. A scope must survive the complete internal OpenMP team and end
only after the outer driver returns. Nonlocal exits are outside this contract.

For a live-scope build, compile only the generated `wt_kf_bridge.c` translation
unit with `-Dwt_kf_bridge_dispatch=wt_kf_live_dispatch`; link the live, capacity,
dispatcher, KF transaction, native capture, bootstrap and span modules. Compile
`wtimer.c` with the same scope macro: its constructor initializes the bootstrap
using `WTIMER_STAGE_CUMULUS` and the existing `WTIMER_STAGE_GRID` parser.
Grid zero selects all grids; positive values select an exact native grid ID.
The native adapter supplies the capture callback and actual OMP query. These
link instructions do not attest a native WRF build or authorize its flags.
The `caller_chain_verified` option defaults to zero: enabling source-proven
automatic-array capacities requires the complete caller/bounds-helper object,
data-order, ABI and relocation evidence documented in `wt_kf_capacity.h`.

The bootstrap accepts build-time `WT_KF_BUILD_SOURCE_ABI_VERIFIED` and
`WT_KF_BUILD_CALLER_CHAIN_VERIFIED` only as boolean macros (both default zero).
The second requires the first. `WT_KF_BUILD_PROFILE_ID` records the audited
build identity and defaults to `unattested`; setting an environment variable
cannot assert these proofs. Resource environment values `WT_KF_MAX_SPANS`,
`WT_KF_MAX_NODES` and `WT_KF_BYTE_BUDGET` default to 4096, 8192 and 268435456
bytes respectively. Explicit values must be positive decimal integers fitting
`size_t`; malformed values refuse configuration. The first startup outcome is
immutable. CSV comment diagnostics separate configuration from actual calls,
staged transactions and fallback reasons. Effective cumulus is `enabled` only
after at least one transaction was staged; valid configuration with no calls
remains `unchecked`. This does not certify remote execution or every scheme.

`prepare_native_link.py` produces a reviewable command manifest from a historical
successful Fortran link line and existing wrapper objects. It never executes
the linker. Its output and execution directory must be outside the frozen WRF
tree, it refuses overwrites and unsupported flags, and hashes direct inputs.
Implicit libraries remain explicitly unattested. `build.sbatch` now compiles
into an external object directory and links directly to a new external output;
it preserves the eight legacy compile units and does not include the KF units.

`prepare_kf_build.py` writes a draft build recipe from explicit source roots,
compiler executables and a compiler-config JSON. The JSON supplies baseline
compile units and wrap symbols; baseline completeness is a separate review.
The recipe adds the ten KF compile units, generates the audited bridge, places
all object/module/generated outputs outside both source roots, hashes bounded
inputs and records dependency ordering. It does not execute any step. Attestation
macros are fixed to zero in this draft; a successfully generated recipe is not
an enabled or ABI-verified native build.

`execute_build_recipe.py` validates the manifest SHA, canonical plan, pinned
inputs and exact bounded step graph by default. Explicit `--execute` creates a
single-use marker and runs the compile recipe; a failure or timeout stops later
steps and prohibits replay. Its last step writes the link manifest and does not
invoke that manifest's linker command. Native execution requires a separately
authorized operation and a matching compiler/module environment.

The MPI remote adapter uses the channel, wire, policy and ledger modules. All
participants must enter channel create/destroy collectively with the documented
parent/thread preconditions. Caller code progresses endpoints explicitly; the
adapter contains no waiting loop or WRF service point. Callback reentry into
adapter operations is refused. Unknown delivery retains admission credits until
collective shutdown; a failed credit release after channel close retains the
owner object for explicit local `finish_closed` handling.

The stored deadline is checked immediately before user commit callback admission.
A trusted matching result that expires at this final gate returns
`WT_MPI_REMOTE_DEADLINE_EXPIRED_RESULT_KNOWN`: no user commit, terminal failed
ledger and released credit. Callback completion is not deadline-bounded and
cannot be rolled back. No timeout permits local physics replay. Four local
two-rank MPI fixtures cover progress, corruption, deadline, callback reentry and
credit retention; native WRF service points and cross-node execution remain
unimplemented.

SFCLAY uses an audited 84-reference schema, guarded transaction core, live
capacity provider, ordinary Fortran configuration query, and an independent
live-scope runtime. The generated bridge connects through
`wt_sfclay_dispatch.c`, which also wraps the one-LOGICAL-reference initializer.
Its reader/writer lease covers both staged and fallback calls. Configuration
proofs default false; tests supplying mock grid metadata do not attest WRF
objects or caller lifetimes. The actual Fortran integration fixture runs 73
physical cases through this live path with full backing-byte comparisons.
The node-live path now connects the first-RK/surface hooks and startup
configuration. A reviewed native build and WRF execution remain required.
The live module requires `-pthread`; its compile and link flags must both retain
that dependency. Default bridge builds omit all test helper exports.

MYNN's source schema records 132 ordered references and 47 optional arguments.
`gen_mynn_schema.py --check` verifies the locked generated metadata;
`wt_mynn_layout` checks conditional presence, canonical logical values, checked
extents and actual capacities before producing a complete descriptor. Widths
in the source schema remain unverified. Some declared OUT fields are read by
the native driver, so OUT never permits omitting copy-in. The current layout
envelope requires `tke_budget == 0`. The batch/node adapter now executes
continuation calls on another local process and commits on the owner. Its GNU
regression covers real initialization followed by two continuation calls and
compares all 132 argument backings. Native WRF admission remains unverified.

Dynamics currently has an audited 71-reference `rk_scalar_tend` boundary and a
limited middle-RK, second-order, interior-tile closure. Its derived config stays
opaque to C. Configuration-query work does not implement a dynamics transaction,
remote execution or halo ownership transfer. Other branches require their own
capacity, write-footprint and side-effect proofs.

## Node-local physics extension

The current extension targets KF, MYNN and original SFCLAY (scheme 91).
It uses `wt_node_local` shared-memory slots and the phase-aware
`wt_kf_node_shm` mapper. Each request retains the original active I range,
vertical bounds and array alignment; donation changes only the J-row batch.
Wire data contains values and offsets, never another process's pointers.
The owner alone validates and commits output. A helper must be a different
local rank with matching phase, build profile, grid, actual timestep, branch
and scientific initialization identity.

`WTIMER_STAGE_CUMULUS`, `WTIMER_STAGE_PBL` and `WTIMER_STAGE_SURFACE` select
the requested phases. Build-time source/caller/state proofs still default
false. Setting a stage flag does not supply those proofs. The node-live build
macros are `WT_KF_NODE_LOCAL`, `WT_MYNN_NODE_LOCAL` and
`WT_SFCLAY_NODE_LOCAL`. The corresponding `WT_*_NODE_G0=1` modes are forced
correctness experiments; adaptive profitability has not been enabled for
these phases. GNU kernel and fork/OpenMP tests do not certify Intel WRF.

MYNN and SFCLAY share the single generated first-RK wrapper because their
inner physics drivers do not carry the owning grid pointer. The scope calls
begin in SFCLAY/MYNN order and end in reverse order; MYNN-only builds omit
the SFCLAY dependency. Local scope tokens control lifetimes and must not be
used as cross-rank timestep identities.

Qualification requires identical A/B/C inputs, decomposition, placement,
thread count and I/O settings: A runs the native path, B executes owner-only
transactions, and C allows helper transactions. B must report real self
execution and commits; C must report remote commits with matching helper
execution counts. Both require bitwise state comparisons, followed by the
existing micro/radiation combination regression. A run with only fallbacks
does not qualify. Performance qualification is separate and must account for
packing, validation, allocation, commit and helper work; correctness shadow
calculations must be disabled for that measurement.

Each phase provides two strict boolean controls, used with its
`WT_<PHASE>_NODE_G0=1` flag (`PHASE` is `KF`, `MYNN` or `SFCLAY`):
`WT_<PHASE>_NODE_G0_OWNER_ONLY=1` suppresses helper scans while preserving the
owner transaction path, and `WT_<PHASE>_NODE_G0_SHADOW=1` checks remote results
against a private native reference before commit. Shadow failure is fatal
and leaves owner output untouched. `shadow_checked`, `shadow_failed` and
`shadow_bytes` report this extra correctness work separately. Keep shadow
off when measuring performance.

The KF G0 harness (`prepare_isc21_node_g0.py`, `run_kf_node_g0.py`) prepares
isolated ISC21 inputs and runs A/B/C at 56 ranks with two OpenMP threads.
Input preparation defaults to a plan; extraction requires the pinned plan.
The runner requires a reviewed linked-candidate identity and stops on a
failed run, missing rank evidence, zero remote work or any bitwise mismatch.
It never builds the executable. `tools/wrf_netcdf_bitwise.py` compares typed
NetCDF data bytes and metadata, reporting its two excluded provenance
attributes (`history`, `_NCProperties`); unsupported data models fail closed.
These tools have local regression coverage; the native G0 remains pending.

KF build preparation keeps `unattested-v1` as the default node profile.
An explicit `reviewed-g0-v1` profile requires a pinned human review record
covering source ABI, callers, baseline objects, compiler/modules, libraries
and initialization state. Hashes preserve the reviewed inputs; they do not
prove those claims. The reviewed recipe uses manifest version 3 and permits
`execute_build_recipe.py --execute --link`; unreviewed recipes still stop at
link preparation. Successful linking records the candidate and map hashes
for G0 to verify. Native CPS calls remain serialized while concurrency is
unproven. A linked candidate still needs the real WRF correctness gate.
