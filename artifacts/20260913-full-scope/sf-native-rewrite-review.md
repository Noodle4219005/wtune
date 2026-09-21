# SFCLAY native adapter/bootstrap rewrite review (2026-09-13)

## Result

GREEN for the bounded local GNU runtime contract.  The native adapter and
explicit bootstrap now compile with GCC 16 under strict warnings plus
ASan/UBSan, and all 8 Python test methods pass.  No cluster, Intel compiler,
native WRF build/link, runner, WT timer, production enablement, or WRF source
was queried or changed.

## RED before implementation

After adding the executable bootstrap fixture and before changing the four
implementation/header files, this exact command reached GCC and failed:

```bash
env TMPDIR=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite TMP=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite TEMP=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite PIXI_CACHE_DIR=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite/pixi-cache /home/u5727520/.pixi/bin/pixi run --manifest-path /home/u5727520/wrf-stack/wtune-runtime-dev-20260912/pixi.toml env CC=/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/.pixi/envs/default/bin/x86_64-conda-linux-gnu-gcc python -m unittest plov.tests.test_sfclay_bootstrap -v
```

Native wall time was 0.9997 s, exit 1.  GCC rejected the fixture because the
snapshot type, state/get-state API, and reason enum did not exist.  It also
independently rejected `wt_sfclay_bootstrap_dump` because `%u` received the
`unsigned long` staged counter.  An earlier shell form failed before GCC
because `bash -lc` discarded the Pixi Python path; that invocation is not
counted as the RED.

## GREEN verification

Final command:

```bash
env TMPDIR=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite TMP=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite TEMP=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite PIXI_CACHE_DIR=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite/pixi-cache /home/u5727520/.pixi/bin/pixi run --manifest-path /home/u5727520/wrf-stack/wtune-runtime-dev-20260912/pixi.toml env CC=/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/.pixi/envs/default/bin/x86_64-conda-linux-gnu-gcc python -m unittest plov.tests.test_sfclay_bootstrap plov.tests.test_sfclay_native -v
```

Result: exit 0, 8 tests PASS, unittest runtime 4.528 s, process wall time
5.7024 s.  Every compile and executable subprocess has an explicit timeout.
Both test modules use:

```text
-std=c11 -Wall -Wextra -Werror -Wpedantic -Wformat=2
-Wstrict-prototypes -Wmissing-prototypes -Wshadow -Wconversion
-Wsign-conversion -g -O1 -fsanitize=address,undefined
-fno-omit-frame-pointer
```

The native suite also uses `-fopenmp -pthread`.  ASan leak detection is
disabled for the OpenMP runtime, while ASan/UBSan errors remain fatal.

## Runtime evidence covered

- The bounded adapter fixture verifies exact live-config fields and callback
  identities (`wt_sfclay_native_capture`, actual `omp_in_parallel`, and
  `wt_sfclay_native_query`), proof/size/selector rejection before helper calls,
  failed-install rollback, immutable duplicate rejection, selector 0 with a
  positive native identity, exact positive selection, and independent KF and
  SF selectors.
- The live fixture links actual `wt_sfclay_live.c`,
  `wt_sfclay_init_lease.c`, `wt_sfclay_capacity.c`, `wt_sfclay.c`,
  `wt_live_spans.c`, the native adapter, and the production C native-query
  shim.  Its only behavior seams are the three ordinary Fortran externals:
  grid identity, grid span capture, and native query.  It proves proof-zero
  fallback with opaque `(void **)1` and zero helper calls, actual OpenMP-region
  fallback, selected-grid miss, selected-grid match, initialized READY state,
  exactly one staged native call, three consistent native queries, and every
  writable owner array receiving the clone result while readonly arrays remain
  unchanged.
- Bootstrap builds cover default proof zero, source-only, source+caller, all
  proofs, noncanonical proof rejection, and source-proof closure rejection.
  Runtime cases cover request-disabled, each missing proof, strict positive
  decimal budgets (empty, zero, whitespace, sign, trailing data, and overflow),
  exact budget forwarding, terminal duplicate/failed start, configured with
  zero requests, post-fallback, actual staged effectiveness, NULL dump, all
  live counters/reasons, quoted profile CSV, and control-character sanitation.

All implementation and fixture files are below 500 lines; the largest is
`wt_sfclay_bootstrap.c` at 267 lines.  jCodeMunch initially returned the prior
37/33-line cache even after `register_edit`; explicit `index_file` refreshed
all nine owned files, after which it reported the new 110-line native source.

## SHA-256

```text
09b5533555ee0a67f086a00f89e889e42bf07267c1c2e4f8c9a7f23cd1f3da58  plov/wt_sfclay_native.h
ad41f1fa3f18adc2e68d5154587c2156e6e080fd60fa7ab36b874ed51804d929  plov/wt_sfclay_native.c
eb6bf8ba0bdd689c522f7dd94b9f7dce03f41178158005c0d31ae7136b70c453  plov/wt_sfclay_bootstrap.h
bd192a89f3d5e8f714ea4278eb0fb427c95d66ad6ecb1cd1ef8f757cc6242d95  plov/wt_sfclay_bootstrap.c
ad67168868f7da636b5e1c156d2b885ca812cb00308b4949756f089436a9e696  plov/tests/test_sfclay_native.py
49a16463cd295265fac6cfa8b509eda23422117bdfbefc7113b2c1fb74eda80e  plov/tests/test_sfclay_bootstrap.py
662015504f96c0363cd388bca2cfa58180cf59d0037de80a3987f5d26e1a0800  plov/tests/sfclay_native_bootstrap_fixture.c
f94ca2b27da3dcbc14090805e6658fb261f3badd77d4c03e1eede814c052fcef  plov/tests/sfclay_bootstrap_fixture.c
512f8587d18f3794902ce015d61ce0f4b688a46923905464ea8b7133c79ca5db  plov/tests/sfclay_native_live_fixture.c
```

## Limits

These tests establish the local C state machine, callback wiring, ordinary
Fortran-call shape, and integration with the accepted live runtime.  They do
not attest the frozen Intel/WRF module ABI, source/object proof hashes, the real
WRF grid traversal helper, linker relocation manifest, production startup
call site, MPI/remote execution, or thread races.  ASan/UBSan is not a race
detector.  Production activation must remain proof-zero until those external
artifacts are independently validated.

## Correction 2026-09-13: UBSan recovery was not previously fatal

The earlier statement above that “ASan/UBSan errors remain fatal” is
retracted for the original 8-test run. That run used GCC's default UBSan
recovery, set only `ASAN_OPTIONS`, and did not reject the diagnostic substring
`runtime error:`. A recoverable UBSan error could therefore have printed to
stderr and still satisfied the original harness. This was a test-evidence
defect; the focused review found no corresponding production C counterexample.

Only the two Python harnesses were changed. Both now add:

```text
-fno-sanitize-recover=undefined
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
```

Every successful runtime case also asserts that stderr contains neither
`Sanitizer` nor `runtime error:`.

### Fatal-UBSan witness

A bounded one-time witness was compiled from standard input with the same GCC
16 strict warning, ASan/UBSan, nonrecovering UBSan, and frame-pointer flags:

```bash
env TMPDIR=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite TMP=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite TEMP=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite PIXI_CACHE_DIR=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite/pixi-cache /home/u5727520/.pixi/bin/pixi run --manifest-path /home/u5727520/wrf-stack/wtune-runtime-dev-20260912/pixi.toml sh -c '/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/.pixi/envs/default/bin/x86_64-conda-linux-gnu-gcc -std=c11 -Wall -Wextra -Werror -Wpedantic -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -Wshadow -Wconversion -Wsign-conversion -g -O1 -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer -x c -o /home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite/ubsan-fatal-witness -'
```

The stdin program adds volatile `INT_MAX` and 1. Running it with the exact
harness options:

```bash
env ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 TMPDIR=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite TMP=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite TEMP=/home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite /home/u5727520/wrf-stack/wtune/artifacts/20260913-full-scope/sf-native-rewrite/ubsan-fatal-witness
```

returned exit 1 in 0.8030 s and emitted:

```text
<stdin>:6:20: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
```

The retained witness executable SHA-256 is
`b8fa1bcea0c129d696f2f4c6eb8726dcffbd9a681d5b6d7dd13f9b4a56849d4e`.
There is no permanent witness source or new repository test.

### Corrected GREEN run

The same combined command documented in `GREEN verification` was rerun after
the harness correction. Result: exit 0, 8 tests PASS, unittest runtime
5.294 s, process wall time 6.3423 s. These results, together with the fatal
witness, supersede the earlier sanitizer-evidence claim.

Corrected test hashes:

```text
075dc48538a9ebe0b0f50195a1ad1852eda2357b4cda0d420c4f5503407dbe49  plov/tests/test_sfclay_native.py
8dc8c57b9e253acde1e7d64b5f15f9c22b23830e54a132f021fc25616a7991ab  plov/tests/test_sfclay_bootstrap.py
```

The C/header/runtime files and their hashes were unchanged by this correction.
