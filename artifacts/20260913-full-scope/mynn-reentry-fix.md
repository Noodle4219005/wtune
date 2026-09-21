# MYNN structural reentry fix

## Result

PASS. `wt_mynn_prepare` now checks structural reentry before touching `*out`. Ordinary failure still clears `*out`; structural reentry preserves the caller's outer transaction handle so the callback can observe `FAILED` and the caller can destroy it. No layout or transaction-core changes were made.

## RED case and fix

The callback's nested prepare case (`plov/tests/test_mynn.c`, mode 14) now passes `&transaction` directly (the unique outer handle), saves `before_handle` only for the assertion, invokes nested prepare with invalid pointer-one arguments, and asserts: `WT_MYNN_STATE`, unchanged `transaction == before_handle`, zero additional malloc calls, outer state `WT_MYNN_FAILED`, and later destroy succeeds. The corruption loop also now asserts `free_calls == freed + 1` after every owned failed transaction is destroyed. Ordinary failure cases retaining a separate output slot remain in the existing preparation-refusal coverage.

An actual RED reproducer was run from a copied fixture under `shared-scratch/red-reentry-20260913b`; only the copied `wt_mynn.c` had the old two-line order (`*out = NULL` before `reentry`). Exact result: process exit `134` (SIGABRT), first assertion `/plov/tests/test_mynn.c:124`: `transaction == before_handle && malloc_calls == before` failed. Production files were not changed for the RED run.

Header contract now says ordinary prepare failures clear `*out`, while structural reentry preserves `*out` for outer cleanup.

## Verification

Exact command (ordinary and ASan/UBSan tests are both defined by `test_mynn.py`):

```sh
R=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope
mkdir -p "$R/shared-scratch" "$R/pixi-cache"
export TMPDIR="$R/shared-scratch" TMP="$R/shared-scratch" TEMP="$R/shared-scratch"
export PIXI_HOME="$R/pixi-cache" XDG_CACHE_HOME="$R/pixi-cache"
cd /home/u5727520/wrf-stack/wtune
/home/u5727520/.pixi/bin/pixi run --manifest-path \
  /home/u5727520/wrf-stack/wtune-runtime-dev-20260912/pixi.toml \
  python plov/tests/test_mynn.py
```

Result: `Ran 2 tests in 2.481s ... OK`; both ordinary and sanitized tests passed after the fix, including the exact post-destroy free-count assertion. `git diff --check` was attempted but cannot inspect this external project from the wiki repository CWD; no source-tree lint claim is made.

File hashes after the fix:

```text
wt_mynn.c       4983405403ba41a53fb4f6ade51dd4ebe97b870ff05e5b9bfb440456abdffb1c
wt_mynn.h       307ccf3eccec5f422eaa285691956da0ad11c10f90a618925226bdd05ede14e3
tests/test_mynn.c 036071f83d4493b68978ea8293a2f22b37d076d3259be8cc4b4def05af88d13a
```

Only the assigned three source/test files were changed. No WRF, native build, cluster, submit, or retry operation was performed.
