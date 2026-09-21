# WTune production wiring status — read-only inventory

## Conclusion

The r3 manifest is a **compile recipe** with `executed=false`; it lists loose
objects and ends at a later link-preparation step.  The inspected r2 attempt
stopped before a link manifest.  For this r2/r3 extension evidence, object
inclusion in r3 is planned build inclusion, never evidence that a WRF
executable calls the adapter.  This report does not make a claim about earlier
WTune or Claude native-link work outside its bounded evidence.

`plov/Makefile` was unavailable in the indexed scope (jCodeMunch returned
`File not found`); this does not establish filesystem absence.  The grounded
build inventory is consequently the r3 config/manifest.  Narrow history of
the requested indexed entry files has only `1aec2f2` (PLOV resync), with no
conclusion about unavailable Makefile history.

| Phase | Actual called adapter / interception | Build inclusion | Gate / default | Missing production integration |
| --- | --- | --- | --- | --- |
| Microphysics | `wtimer.S:113-130` routes the wrapped driver to `wt_micro_call`; it either calls the real driver or enters staged logic (`wt_micro.c:1312-1346`). | r3 declares `wtimer.S`, `wtimer.c`, and `wt_micro.c` (`inputs/...r3.json:4-10`); manifest compile step is `build-manifest.json:873-885`. | `WTIMER_STAGE_MICRO` defaults 1 (`wtimer.c:350-354`), but grid, row, and geometry checks can call the real driver unstaged (`wt_micro.c:1317-1344`). | No completed native link/run proves that this wrapper reaches a WRF binary. |
| Radiation | `wtimer.S:133-150` routes the radiation driver to `wt_rad_call`; pre-radiation remains the pass-through timer wrapper (`wtimer.S:152`). | r3 includes `wt_rad.c` (`inputs/...r3.json:10`; manifest `:894-906`). | `WTIMER_STAGE_RAD` defaults 0 (`wtimer.c:351-354`); even when requested, grid/rows/self-check refusal falls back to `rad_real_call` (`wt_rad.c:1347-1393`). | No completed native link/run; a flag alone does not establish staged radiation. |
| Cumulus / KF | The pass-through cumulus wrapper is `WTWRAP(...,5)` (`wtimer.S:155`).  With `WT_CUMULUS_LIVE_SPANS`, it brackets the live-grid hooks (`wtimer.S:67-75`, `:91-95`); constructor startup is conditional (`wtimer.c:360-361`). | r3 plans `wt_kf_capacity`, `dispatch`, `live`, `native`, `bootstrap`, and generated bridge objects (`build-manifest.json:975-1069`, `:1193-1207`). | `WTIMER_STAGE_CUMULUS` defaults 0 (`wtimer.c:352`); absent the compile macro, wtimer reports requested cumulus as refused (`wtimer.c:255-275`). | r3 is unexecuted; this table makes no claim that its KF objects are linked into a WRF executable. |
| PBL | Only the generic timing trampoline wraps the PBL driver (`wtimer.S:153`), then calls the real symbol. | Only `wtimer.S`/`wtimer.c` are planned in r3; no PBL-specific adapter appears in the r3 object list. | `WTIMER_STAGE_PBL` defaults 0 (`wtimer.c:352`) and requested staging is explicitly reported refused (`wtimer.c:255-275`). | No PBL transaction/adapter invocation or native link. |
| Surface | Only the generic timing trampoline wraps surface (`wtimer.S:154`), then calls the real symbol. | Only timer objects are planned; no SF scope/native object is in the r3 object list. | `WTIMER_STAGE_SURFACE` defaults 0 (`wtimer.c:353`) and requested staging is explicitly refused (`wtimer.c:255-275`). | **Pending:** SF native/bootstrap is being rewritten; its internals were not reviewed here.  No r3 live surface adapter claim. |
| Dynamics | No dynamics symbol occurs in the six wrapped symbols: micro, radiation, pre-radiation, PBL, surface, cumulus (`wtimer.c:28-31`; `wtimer.S:151-155`). | No `wt_rk*` object is listed in r3’s 18-object compile plan (`build-manifest.json:1173-1207`). | No production dynamics gate in the requested entry files. | No dynamics interception, adapter call, or build inclusion. |
| Halo / MPI waits | `wtmpi.c:35-56` defines PMPI forwarding wrappers for Wait/Waitall/Waitany and several collectives; it records waits and calls node-local LEWI begin/end. | r3 plans `wtmpi.c` and `wtlewi.c` (`inputs/...r3.json:6-7`; manifest `:809-843`). | There is no feature-enable flag in `wtmpi.c`; runtime effect still requires these objects to be placed before MPI in a completed native link.  `wt_lewi_init` returns without configuration if `WTIMER_LEWI` is absent (`wtlewi.c:94-101`). | No completed native link establishes that WRF’s halo calls resolve to these PMPI definitions. |
| Cross-node remote work | No call from `wtimer.c` or `wtmpi.c` to `wt_mpi_channel_*`/`wt_mpi_remote_*` was found. | No `wt_mpi_channel.c` or `wt_mpi_remote_*` object occurs in r3’s compile list (`build-manifest.json:1173-1207`). | No production gate/default in the requested wiring path. | Cross-node transport is **test-only evidence**: `tests/test_wt_mpi_channel.py:25-26` compiles a channel test and remote tests construct owner/worker instances (`tests/test_wt_mpi_commit_deadline.c:125-191`).  This is not production remote invocation. |

## Build and native boundary evidence

- r3 config names timer, PMPI, LEWI, microphysics, and radiation objects at
  `inputs/wtune-baseline-build-config-20260913-r3.json:4-10`; the manifest
  preserves them as compile steps and wrapper-object inputs
  (`recipe/build-manifest.json:765-906`, `:1173-1207`).
- The r3 draft’s `executed=false` is local recipe metadata, not a completed
  compile or link.  No claim in this table relies on it becoming an executable.
- Inspected r2 terminal evidence records `FAILED`, no `link-manifest.json`,
  and `link_executed=false`
  (`wtune-kf-native-recipe-20260913-r2/terminal-review.md:5-16`).  This
  establishes the absence of a completed link for that inspected r2 attempt,
  without re-opening the module-closure diagnosis.
- `wtmpi.c` is an instrumentation/LEWI boundary, not the generic remote
  transport subsystem.  The latter has independent MPI tests, whose passing
  behavior would still not establish a call from a WRF physics or halo path.

## Scope limits

This was a static, read-only source/history/artifact inventory.  It did not
invoke a compiler, link, run WRF, submit/query a cluster, modify WRF or WTune
source, or inspect the changing SF native/bootstrap internals.  A future
native link and run needs separate evidence for symbol resolution, gate state,
and actual adapter invocation.
