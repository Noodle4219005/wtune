# Cross-node / KF cumulus / MYNN / SFCLAY expansion — archived, unfinished, not shipped

Archived 2026-09-13. This directory holds the prototype code and native-compile
recipes from the 2026-09-12/13 attempt to extend WTune's node-local micro/radiation
staging (the shipped, gate-passed mechanism) into:

- cross-**node** work stealing (`wt_mpi_channel/remote_owner/remote_worker`,
  `wt_ledger`, `wt_wire`, `wt_policy`, `wt_stream`),
- KF/cumulus staging (`wt_cumulus.*`, `wt_kf_*`, the `gen_kf_*` schema generators,
  `kf_cps_schema.json`, the native-compile recipes referenced from
  `wtune-kf-native-recipe-20260912/` and `-20260913-r2/-r3` outside this repo),
- PBL/MYNN and surface/SFCLAY node-local staging (`wt_mynn*`, `wt_sfclay*`).

**Why it's archived instead of shipped:**

1. **Cumulus is architecturally blocked, not just unfinished.** `module_cumulus_driver.f90:821-835`
   calls `HALO_CUP_G3_IN_sub(grid, local_communicator, ...)` — a live MPI halo collective —
   inside the `KFETASCHEME` branch. The micro/radiation two-pass wrap-and-steal pattern
   assumes an MPI-clean driver call; cumulus does not have one. `KF_eta_PARA` is also
   defined in the same object as its caller, so it has no undefined reference for
   `--wrap` to intercept. See [[wtune-per-phase-stealing-flag-surface]] (2nd 2026-09-12
   update) and the wiki decision this archive corresponds to.
2. **Nothing here ever passed a native G0/bitwise gate.** Every native-compile attempt
   (job 1073389 and its r2/r3 successors) failed before producing a link manifest —
   one root cause found was `/tmp` not being shared between f1's login and compute
   nodes (see the wiki note on that). No KF/MYNN/SFCLAY object has ever been linked
   into a WRF executable, let alone run and diffed.
3. **Cross-node transport is test-only.** `wt_mpi_channel`/`wt_mpi_remote_*` have C-level
   unit/sanitizer tests but are never called from `wtimer.c`/`wtmpi.c` — no WRF physics
   or halo call path resolves to them.

**Correction (2026-09-13, previous version of this file overstated this):** the native
compile recipe did NOT always fail before producing a link manifest. Job **1073389** (r1)
and its r2 successor failed before compiling anything; but r3, **job 1073533, COMPLETED**
(20260913-r3/terminal-review.md): all 18 objects compiled with the Intel toolchain and a
real `recipe/link-manifest.json` was produced. The card was scoped to stop there —
`link_executed=false` is the card's designed boundary, not a failure. What genuinely has
never happened is running the linker itself, producing a `wrf-wtune.exe`, or any
ABI/caller-closure attestation (`baseline_closure_verified`/`caller_chain_verified`/
`compiler_module_abi_verified`/`library_closure_verified`/`source_abi_verified` are all
still `false`) — and, as before, no run against real WRF at all.

**Decision**: keep this as reference/salvage material (the wire/ledger/policy protocol
core and the KF/MYNN/SFCLAY schema-generation approach may still be reusable if a future
attempt targets microphysics/radiation cross-node transport instead of cumulus), but do
not present any of it as a working WTune capability. The shipped, verified WTune surface
is node-local micro/radiation staging only (`plov/wt_micro.c`, `plov/wt_rad.c`,
`plov/wtimer.c`'s `WTIMER_STAGE_*` flags), gated by `plov/regress_bp.sbatch`
(job 1072441: 224-rank bit-identical, `refused=0`).

## Layout

- `plov/` — everything untracked under the live `plov/` tree that this archive covers,
  minus the four kept production/regression scripts (`regress_bp.sbatch`,
  `atlas_isc21_d01*.sbatch`, `atlas_maria_d01_bp.sbatch`) and `tests/test_gen_argmap_safety.py`,
  which stayed in the live tree because they exercise only the shipped flag surface.
- `plov/tests/` — the corresponding unit/sanitizer tests (KF, MYNN, SFCLAY, MPI channel/remote,
  ledger/wire/policy/stream, native-link recipe tooling).

None of this is wired into the WTune build (`build.sbatch` does not compile any of it),
and none of it should be treated as evidence that WTune can rebalance cumulus, PBL,
surface, or cross-node work.

## Update 2026-09-13: the KF/MYNN/SFCLAY node-local track is a *different* claim than "architecturally blocked"

The `HALO_CUP_G3_IN_sub` finding above is about wrapping the *whole* `cumulus_driver`
call the way micro/radiation are wrapped (two-pass, driver-call-granularity steal).
It is **not** a blocker for the narrower approach recorded separately in
[[wtune-node-local-physics-implementation]]: wrapping `KF_eta_CPS` itself (a confirmed
cross-object `--wrap`-able symbol — `nm -u module_cumulus_driver.o` shows
`U module_cu_kfeta_mp_kf_eta_cps_`) for **node-local only** helper compute, at a call
site that executes *after* the driver's halo exchange in the tile loop. That track has
substantial fixture-level evidence (GNU-compiled differential tests, bitwise-identical
to direct computation, ASan/UBSan-clean, real Fortran driver state comparisons for
MYNN/SFCLAY too) — it is archived here **only because it has never been linked into or
run against the actual Intel-built WRF binary on the cluster**, the same bar micro/rad
had to clear via job 1072441. It is not dead the way cumulus's whole-driver wrap or the
cross-node transport are; a future attempt does not need to re-litigate an architectural
blocker, only get past native compile/link (where r1/r2/r3 all failed before producing
a link manifest) and then run the real regression gate.
