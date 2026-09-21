# SFCLAY native adapter/bootstrap focused acceptance review

## Verdict: FAIL pending sanitizer-harness repair

I found no concrete production-source counterexample in the bounded native adapter/bootstrap delta. The fail-closed proof gates, immutable installation state, rollback, actual OpenMP callback, SF/KF selector separation, strict budgets, terminal startup, staged-state semantics, CSV counters, and inherited post-native no-replay path are coherent. Acceptance is blocked by one concrete test-evidence defect: the two claimed ASan/UBSan runtime suites do not make UBSan diagnostics fail the test.

Reviewed read-only from fresh jCodeMunch `index_file` results:

- `plov/wt_sfclay_native.c`, `plov/wt_sfclay_native.h`
- `plov/wt_sfclay_bootstrap.c`, `plov/wt_sfclay_bootstrap.h`
- `plov/tests/test_sfclay_native.py`, `plov/tests/test_sfclay_bootstrap.py`
- `plov/tests/sfclay_native_bootstrap_fixture.c`
- `plov/tests/sfclay_bootstrap_fixture.c`
- `plov/tests/sfclay_native_live_fixture.c`
- accepted `wt_sfclay_live.c` path and shared SF source-proof/review artifacts where the new adapter delegates behavior

I did not edit source, compile, link, run a fixture, query a cluster, or inspect native WRF execution.

## P1 acceptance evidence: recoverable UBSan errors can pass both suites

`test_sfclay_native.py:11-17` and `test_sfclay_bootstrap.py:12-18` compile with `-fsanitize=address,undefined`, but neither adds `-fno-sanitize-recover=undefined`. Their environments set only `ASAN_OPTIONS=detect_leaks=0:abort_on_error=1` (`test_sfclay_native.py:22-30`; `test_sfclay_bootstrap.py:23-31`) and do not set `UBSAN_OPTIONS=halt_on_error=1`. Both runtime helpers merely assert that stderr does not contain the substring `Sanitizer` (`test_sfclay_native.py:60-67`; `test_sfclay_bootstrap.py:72-82`).

Concrete falsifier: insert or link a fixture operation that triggers recoverable signed-integer-overflow UB and then returns zero. GCC UBSan normally emits a line containing `runtime error:` and continues. That text does not contain `Sanitizer`; the subprocess return code remains zero, so both current assertions pass. The implementation receipt's statement that “ASan/UBSan errors remain fatal” is therefore not established by these test definitions.

Minimal fix in both Python test modules:

1. Add `-fno-sanitize-recover=undefined` to the compile flags, and set `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` in the subprocess environment.
2. Reject `runtime error:` explicitly, or require empty stderr when the toolchain/runtime emits no benign diagnostics.
3. Rerun the same eight test methods under the strict compiler. A one-time sanitizer canary that intentionally triggers recoverable UB and is required to exit nonzero would directly prove the harness setting; it need not become production code.

This is an evidence blocker, not evidence that the reviewed C code currently executes UB.

## Source invariants that pass focused review

- **Proof before opaque/native-pointer access.** `wt_sfclay_native_configure` validates canonical proof booleans, requires all three proofs, positive bounds/budget, and a nonnegative selector before publishing callbacks (`wt_sfclay_native.c:66-86`). `wt_sfclay_native_capture` checks grid/registry/output/bound/state before invoking either ordinary-Fortran helper (`:34-47`). The actual-live proof-zero fixture passes opaque `args==(void **)1`, gets one fallback, and requires zero identity/capture/query calls (`sfclay_native_live_fixture.c:161-172`). The inherited live dispatcher performs `base_reason` before `current_scope` or argv-entry access (`wt_sfclay_live.c:228-237`).
- **Immutable successful configuration and failed-install rollback.** A CAS admits only `UNCONFIGURED -> INSTALLING`; successful live configuration release-publishes `IMMUTABLE`, and a rejected installation clears the selected grid then release-restores `UNCONFIGURED` (`wt_sfclay_native.c:86-110`). Once immutable, another configure cannot alter callbacks, selector, or budgets. The adapter fixture exercises reject-then-success and successful duplicate refusal (`sfclay_native_bootstrap_fixture.c:99-128`). Static `selected_grid` is written before the release publication and only read after an acquire state load, so the stated immutable lifecycle supplies its synchronization.
- **Actual OpenMP and live scope.** The installed live config uses the real `omp_in_parallel` function pointer, not an environment/policy boolean (`wt_sfclay_native.c:88-101`). The adapter fixture asserts exact pointer identity and invokes it inside a two-thread OpenMP region (`sfclay_native_bootstrap_fixture.c:78-89, 129-135`). The actual-live fixture calls the production live runtime from inside an OpenMP region and observes fallback with one original native call and zero staging (`sfclay_native_live_fixture.c:133-149, 173-184`).
- **SF selector is independent from KF.** SF owns its own immutable `selected_grid` (`wt_sfclay_native.c:21-22, 47-50, 86-110`). The fixture configures KF selector 7 and SF selector 2 simultaneously, proves each accepts its own grid and SF rejects grid 7 (`sfclay_native_bootstrap_fixture.c:136-146`). Selector zero still requires a positive native identity and admits all positive identities (`:121-128`).
- **Strict budgets and proof-only build activation.** Bootstrap proofs default to zero and reject noncanonical compile-time values; caller/initializer proofs cannot be enabled over a zero source proof (`wt_sfclay_bootstrap.c:7-39`). Environment variables affect only three decimal budgets. `positive_decimal` accepts digits only, rejects empty/zero/sign/whitespace/trailing data, and checks `size_t` multiply/add overflow (`:59-79`). `wt_sfclay_bootstrap_start` checks request, selector, budgets, profile, and each proof before its sole native configure call (`:106-158`). Tests cover every invalid grammar and exact forwarded values (`test_sfclay_bootstrap.py:84-122`; `sfclay_bootstrap_fixture.c:61-100`).
- **Terminal startup and truthful effectiveness state.** The first start sets `boot.started` before every refusal/configure path; later calls return the original `configured` result without rereading environment or changing stored selector/budgets (`wt_sfclay_bootstrap.c:106-158`). Fixtures cover duplicate success and failed-native non-retry (`sfclay_bootstrap_fixture.c:101-116`). State is zero unless configured, `-1` while actual live `staged==0`, and `1` only after the live staged counter is positive (`wt_sfclay_bootstrap.c:160-189`). Requested-only and fallback-only counters do not claim effectiveness; tests cover zero requests, fallback, and staged (`sfclay_bootstrap_fixture.c:117-134`). The actual live fixture separately proves the production counter moves to exactly one only after clone execution and commit (`sfclay_native_live_fixture.c:196-209`; `wt_sfclay_live.c:264-276`).
- **Comment-prefixed CSV and counter types.** Dump handles NULL, prefixes every record with `# sfclay_bootstrap,`, quotes/doubles profile text, replaces controls/DEL, uses `%zu` for `size_t` and `%lu` for `unsigned long`, asserts the ten-reason table count, and emits requested/staged/fallback/fatal plus every reason counter (`wt_sfclay_bootstrap.c:191-267`). The Python fixture parses every line as CSV, verifies a comma-bearing profile, all counters/reasons, NULL safety, and control-character line-injection resistance (`test_sfclay_bootstrap.py:133-151`).
- **No post-native replay.** The adapter merely installs the already accepted live callbacks. In the live dispatcher, native clone execution occurs once; every later scope/token/span/init/query/transaction failure calls the fatal path and cannot reach fallback (`wt_sfclay_live.c:264-284`). The new selected-staged fixture observes three consistent queries, exactly one native call, one staged result, writable clone commits, and unchanged read-only owners (`sfclay_native_live_fixture.c:185-209`). The broader previously accepted live suite supplies the post-native drift/guard/readonly/owner fatal falsifiers; the newly reviewed fixture does not duplicate those cases.

## Remaining uncertainty and bounded non-claims

- The scoped fixtures use C seams for the ordinary-Fortran grid identity, span capture, and native query. They prove call shape and local runtime wiring, not the frozen Intel WRF derived-type/default-kind ABI, real grid traversal, source/object hashes, relocation manifest, or production startup call site.
- The bootstrap fixture stubs live counters, while the native-live fixture separately exercises the real counter. Source composition is direct, but there is no single fixture that starts through bootstrap, stages through the real live runtime, and then observes bootstrap state 1.
- ASan/UBSan is not a race detector. Bootstrap start is documented for one serialized startup caller; this review makes no concurrent-start/state-dump guarantee.
- Active OpenMP falls back. No MPI, remote execution, OpenMP-positive staging, native WRF enablement, or numerical equivalence is admitted.

After the P1 harness fix and a green rerun, the bounded local source delta is suitable for ACCEPT without another broad source redesign; a focused verification of the two test-file changes and native eight-test output is sufficient.

## Final focused delta: ACCEPT

The sanitizer-harness blocker is closed. Fresh jCodeMunch indexes show both test modules now compile with `-fno-sanitize-recover=undefined`, set `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`, and reject the GCC diagnostic substring `runtime error:` in every successful runtime case (`test_sfclay_native.py:11-18, 22-31, 61-69`; `test_sfclay_bootstrap.py:12-19, 23-32, 74-84`). These three independent protections make the original recoverable-UB counterexample fail the suite.

The appended implementation receipt records a volatile `INT_MAX + 1` witness built and run with the same strict/sanitizer options. It emitted `runtime error: signed integer overflow` and exited 1. The corrected eight-test run then passed in 5.294 s. I inspected those retained artifact facts but did not rerun either command in this no-test review.

The lead independently reported the same focused checks after this review: all eight tests passed in 5.698 s, and the retained witness exited 1 with the signed-overflow `runtime error:` and stack trace (`artifacts/20260913-full-scope/root-check/sf-native-bootstrap-fatal-ubsan.txt`).

Only the two Python harnesses changed, with final hashes:

```text
075dc48538a9ebe0b0f50195a1ad1852eda2357b4cda0d420c4f5503407dbe49  plov/tests/test_sfclay_native.py
8dc8c57b9e253acde1e7d64b5f15f9c22b23830e54a132f021fc25616a7991ab  plov/tests/test_sfclay_bootstrap.py
```

The reviewed production files remain byte-identical:

```text
ad41f1fa3f18adc2e68d5154587c2156e6e080fd60fa7ab36b874ed51804d929  plov/wt_sfclay_native.c
09b5533555ee0a67f086a00f89e889e42bf07267c1c2e4f8c9a7f23cd1f3da58  plov/wt_sfclay_native.h
bd192a89f3d5e8f714ea4278eb0fb427c95d66ad6ecb1cd1ef8f757cc6242d95  plov/wt_sfclay_bootstrap.c
eb6bf8ba0bdd689c522f7dd94b9f7dce03f41178158005c0d31ae7136b70c453  plov/wt_sfclay_bootstrap.h
```

Final verdict for the bounded local native adapter/bootstrap contract is **ACCEPT**. The native WRF/Intel ABI, production startup and wrapper manifests, MPI/remote behavior, and race detection remain outside this acceptance exactly as stated above.
