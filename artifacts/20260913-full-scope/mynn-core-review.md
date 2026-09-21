# MYNN core transaction correctness review

Verdict: **ACCEPT after focused fix**. The reviewed transaction satisfies the listed safety invariants under the documented owner-local prototype constraints. The initial review found a P1 structural-reentry handle-loss counterexample; the final source reorders the reentry gate before any output-slot access and adds a direct sole-handle RED/GREEN regression with exact cleanup accounting.

Reviewed source:

- Final `plov/wt_mynn.c` SHA-256 `4983405403ba41a53fb4f6ade51dd4ebe97b870ff05e5b9bfb440456abdffb1c` (261 lines)
- Final `plov/wt_mynn.h` SHA-256 `307ccf3eccec5f422eaa285691956da0ad11c10f90a618925226bdd05ede14e3`
- `plov/wt_mynn_layout.h`, `plov/wt_mynn_layout.c`, and generated `plov/wt_mynn_schema.h`
- `plov/mynn_schema.json`
- Final `plov/tests/test_mynn.c` SHA-256 `036071f83d4493b68978ea8293a2f22b37d076d3259be8cc4b4def05af88d13a`, plus `plov/tests/test_mynn.py`
- `knowledge/wrf/2026-09-12-wtune-mynn-driver-boundary.md`

This is a source/invariant review. I did not build, run WRF, invoke native MYNN, or claim numerical equivalence. The implementation worker recorded the actual old-order RED as exit 134 at the sole-handle preservation assertion, then the final ordinary plus ASan/UBSan suite passed (`Ran 2 tests ... OK`, 2.481 s). I did not rerun it because this assignment prohibited new temporary writes/builds. Native evidence and exact commands are preserved in the shared `mynn-reentry-fix.md` artifact.

## Resolved P1: nested prepare erased the sole outer handle and leaked a poisoned transaction

Evidence:

- `plov/wt_mynn.c:140` clears `*out` before `plov/wt_mynn.c:141` calls `reentry()`.
- `reentry()` at `plov/wt_mynn.c:62-68` correctly poisons the TLS `entered_transaction`, but it cannot restore the caller's output slot.
- During the callback, a caller can invoke `wt_mynn_prepare(..., &transaction)` where `transaction` is its sole handle to the currently ENTERED object. That output slot does not alias the new `argv`, capacity table, config, or any owner interval, so the documented preconditions do not reject it.
- The call first assigns `transaction = NULL`, then detects reentry and returns `WT_MYNN_STATE`. The outer `wt_mynn_execute_once` still has its local `tx`, observes the poison at `plov/wt_mynn.c:216`, and returns failure. Once it returns, the caller has no pointer with which to call `wt_mynn_destroy`; the allocation and all clones/snapshots remain live indefinitely.
- The current structural-reentry fixture at `plov/tests/test_mynn.c:126-133` uses a separate local `nested` output slot and asserts that it becomes NULL. It proves rejection before bad `argv` reads/allocation, but cannot expose loss of the outer handle. The reentrant-destroy fixture at `plov/tests/test_mynn.c:121` proves no immediate free, not eventual reachability of the allocation after nested prepare.

Applied fix:

1. Final `wt_mynn_prepare` calls `reentry()` at `plov/wt_mynn.c:143` before its first `out` access at `:144`. Structural reentry therefore preserves the handle while ordinary non-reentrant failures still clear it.
2. `plov/wt_mynn.h:25-26` now states that distinction explicitly.
3. Final callback mode 14 at `plov/tests/test_mynn.c:119-127` passes `&transaction` directly, preserves and compares the unique outer handle, uses pointer-one `argv` and capacities as a fail-fast no-read witness, checks zero new allocations, and observes outer FAILED. `state_and_corruption` at `plov/tests/test_mynn.c:230-237` then proves no premature free, no replay, and exactly one free on successful destroy.

The copied old-order implementation failed this exact fixture at the handle-preservation assertion with exit 134. The final implementation passed both ordinary and sanitizer configurations. No residual source finding remains in this path.

The API should still document that an ordinary, non-reentrant prepare output slot must not contain the sole handle to another live transaction. That is a caller-lifetime clarification; it does not affect the repaired TLS reentry path.

## Invariants that passed source review

- **Complete preflight before scalar reads:** `plov/wt_mynn_layout.c:99-110` checks every present/absent pointer, minimum capacity, nonwrapping interval, and alignment before the scalar-copy loop at `:111-116`. Config/ABI gates precede both. All 48 scalar schema entries are non-optional, so the scalar loop cannot dereference an allowed absent scalar. Layout/branch validation completes before transaction allocation at `plov/wt_mynn.c:143-155`.
- **Checked total heap budget:** `regions_for` starts with `sizeof(wt_mynn_tx)` at `plov/wt_mynn.c:76`; for every deduplicated union it adds clone bytes, snapshot bytes, and two 32-byte guards with checked additions at `:104-109`. `wt_mynn_prepare` rejects over-budget before its single `malloc(total)` at `:153-156`. Metadata, owner/capacity snapshots, layout, and region descriptors are inline in `wt_mynn_tx` (`:17-27`). Fixed preparation and callback tables are stack objects, as documented.
- **Overlap semantics:** intervals are ordered by start at `plov/wt_mynn.c:86-91`; only strict overlap (`begin < previous.end`) merges at `:94`; mixed writable/read-only overlap returns `WT_MYNN_ALIAS` at `:96`; equal-permission chains extend one union at `:97`; touching intervals take the separate-region path at `:98`. Staged pointers preserve each formal's offset into its union at `:184-195`. Commit iterates each union once at `:238-241`.
- **Snapshots, guards, and staleness:** the owner pointer and capacity tables are value-snapshotted inline at `plov/wt_mynn.c:159-168`; every union's complete bytes are copied to snapshot and clone at `:172-181`. `validate` checks all 132 pointer/capacity entries, both clone guards, pristine/read-only clone bytes, and every owner byte at `:112-127`. Layout assigns inactive present optionals a four-byte non-writable view at `plov/wt_mynn_layout.c:140-158`, so they receive the same snapshot/read-only/guard checks.
- **Execute once and terminal failures:** entry requires PREPARED and a valid callback at `plov/wt_mynn.c:206-210`; the state becomes ENTERED before the callback at `:212-215`; every post-return structural, rebind, guard, read-only, or stale error poisons FAILED at `:216-223`; only a fully validated return becomes EXECUTED at `:224`. Invalid replay/commit paths poison or preserve an already COMMITTED terminal state through `poison` at `:55-59`. There is no internal fallback or second callback call.
- **All transaction validation before owner commit writes:** `wt_mynn_commit` performs the complete `validate(tx, 0)` at `plov/wt_mynn.c:235-237`, then and only then copies writable unions at `:238-241`. This is owner-local validation, not concurrent atomicity, matching the header's limitation.
- **Callback pointer rebinding:** the callback receives a stack copy of all 132 staged pointers at `plov/wt_mynn.c:211-215`; every entry, including absent NULL entries, is compared with the original staged table before any success at `:218-220`.
- **TLS affinity and monotonic nonce:** `current_nonce` at `plov/wt_mynn.c:35-48` assigns nonzero values from an atomic monotonic counter, permanently saturates before wrap/reuse, and caches the nonce in TLS. `owned` checks this immutable nonce before mutable transaction state at `:50-53`. Prepare, execute, commit, and destroy now all call `reentry()` before touching new call data (`:143`, `:203`, `:230`, `:248`).
- **No UAF on reentrant destroy:** destroy calls `reentry()` first at `plov/wt_mynn.c:248`; while a callback is active this poisons the TLS outer transaction and returns without reaching `free` at `:253`. The outer execute retains its local object pointer, clears callback state, and returns terminal failure safely. Ordinary successful destroy makes further pointer use a documented caller violation.
- **Schema agreement and boundary:** the current generated `wt_mynn_schema.h` contains 132 metadata entries. I independently compared, entry by entry, name, type, rank, optional flag, intent, access condition, and write condition with all 132 JSON entries; there were zero mismatches. Its embedded schema SHA is `73225add0c60a4c5ac0da76a67d913a89fd7ad7e641f3cfdc01079ae0724133b`, matching the accepted boundary note. `plov/mynn_schema.json:5` still says `source_schema_only_native_abi_unverified`, and its `byte_width` values remain null. The implementation/header also explicitly denies native or numerical admission, so this review does not widen the prior source-closure decision.

## Additional falsifiers worth retaining after the P1 fix

These are coverage improvements, not additional source findings:

- Regenerate `wt_mynn_schema.h` from `mynn_schema.json` in a test-controlled output and compare it byte-for-byte (or compare all 132 semantic records plus the actual JSON digest) before compiling the transaction fixture. The reviewed copies agree, but `tests/test_mynn.py:13-29` presently compiles the existing header and the C fixture consumes that same metadata, so future schema/header drift could be self-consistent inside the test.
- Extend the exact-budget test at `plov/tests/test_mynn.c:152-165` to compute the expected request from `sizeof(wt_mynn_tx)` via an exposed accounting helper or diagnostic rather than learning the accepted size from a preceding successful allocation. The source accounting is correct, but the current learned-size test would not independently detect a newly added budgeted metadata field if both runs include it.
