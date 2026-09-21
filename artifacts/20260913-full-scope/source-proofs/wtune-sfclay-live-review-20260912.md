# SFCLAY live rewrite concurrency review

## Verdict: ACCEPT

The P1 owner-affinity defect is fixed.  I found no concrete counterexample in
the bounded delta.  The initialization lease, cancellation lifetime, fallback
count, post-native revalidation, and no-replay paths continue to implement the
supplied invariants coherently.

## Resolved P1 finding

Successful configuration now assigns a monotonically nonzero nonce to both
immutable process state and the configuring thread's TLS immediately before
publishing `configured` (`plov/wt_sfclay_live.c:36-63`).  `is_owner` checks the
TLS nonce for nonzero equality before evaluating `pthread_equal`
(`plov/wt_sfclay_live.c:76-82`).  The only raw `pthread_equal(owner, ...)` call
in the indexed `plov/` tree is inside this guarded helper.  `base_reason`,
`current_scope`, and `surface_begin` all use the helper
(`plov/wt_sfclay_live.c:85-95`, `:128-133`, `:190-200`).  Scope teardown clears
grid, config, result, and scope state but preserves the nonce
(`plov/wt_sfclay_live.c:172-188`).

Therefore a replacement thread begins with zero-initialized TLS and is rejected
before the code evaluates an expired or recycled `pthread_t`.  The targeted
glibc falsifier records a short-lived configuring thread's handle bytes, joins
it, proves that the replacement received the same handle representation, then
requires one native fallback with zero OMP/query/capture callbacks and zero
stages (`plov/tests/test_sfclay_live.c:193-215`).  This is the concrete old-code
failure mode and the repaired-code pass condition.

## Audited properties that pass inspection

- Configuration publishes copied immutable state only at quiescence and does
  not reset initialization state (`plov/wt_sfclay_live.c:36-63`).  The tests
  exercise rejected configure while active and init-before-configure
  (`plov/tests/test_sfclay_live.c:226-231`).
- Proof-zero and disabled admission precede all OMP/native-query/capture calls
  and any `argv[i]` read (`plov/wt_sfclay_live.c:85-95`, `:228-233`).  Native
  fallback obtains the reader lease before calling the original arguments once
  (`plov/wt_sfclay_live.c:274-283`).  Tests use an intentionally opaque
  `args == (void **)1` and require zero callbacks for the proof failures
  (`plov/tests/test_sfclay_live.c:83-100`, `:252-256`).
- OMP admits exactly zero (`plov/wt_sfclay_live.c:97-105`); native admission uses
  the provider's `actual_scheme == expected_scheme`, not literal 91, and checks
  status/fractional/grid (`plov/wt_sfclay_live.c:107-118`).  Test scheme 7 stages,
  while positive and negative OMP and fractional cases fall back
  (`plov/tests/test_sfclay_live.c:15`, `:244-249`).
- A direct surface call without first-RK receives a paired disabled token and
  falls back (`plov/wt_sfclay_live.c:190-209`,
  `plov/tests/test_sfclay_live.c:252-254`).  Same-wrapper, control-callback, and
  dispatch/init/reader reentry reach structural fatal checks
  (`plov/wt_sfclay_live.c:136-138`, `:191-192`, `:226-227`;
  `plov/wt_sfclay_init_lease.c:85-87`).
- The eligible lock order is staging gate, reader lease, span/capacity, then
  transaction (`plov/wt_sfclay_live.c:243-263`).  The initializer holds the
  lease mutex only while changing lease state, releases it around the native
  initializer, and gives waiting writers preference over all new readers
  (`plov/wt_sfclay_init_lease.c:45-61`, `:83-112`).  No reviewed path holding a
  span/provider lock acquires the lease mutex.
- Cancellation is disabled at the first activity depth and restored at the
  final leave (`plov/wt_sfclay_init_lease.c:18-43`).  Dispatch releases reader,
  staging, and TLS dispatch state before its final leave
  (`plov/wt_sfclay_live.c:279-284`); outer/surface endpoints clear their gates
  and tokens before leaving (`plov/wt_sfclay_live.c:172-209`).
- After clone execution, scope/token/span/init checks run, the native query is
  repeated, and scope/token/span/init are checked again after that callback and
  before commit (`plov/wt_sfclay_live.c:264-272`).  `wt_sfclay_commit` rejects
  guard overwrite, original-owner mutation, and readonly-clone mutation before
  copying writable results (`plov/wt_sfclay.c:121-133`).  Every such failure
  aborts without replay.  The test matrix records 20 fatal cases and asserts
  the exact number of native executions from stdout
  (`plov/tests/test_sfclay_live.py:10-13`, `:36-43`).
- Reader/writer state is mutex-protected; config is published through the
  `configured` atomic; counters and gates are atomic; mutable scope state is
  TLS.  Within the stated serial owner-local model I found no concrete data
  race in these objects.  ASan/UBSan is not a race detector, so the reported
  53-case run alone is not evidence for this conclusion.

## Necessary test evidence now present

1. Writer preference holds R1, queues W, then starts R2 and proves R2 remains
   out until W enters and finishes (`plov/tests/test_sfclay_live.c:128-192`).
   The coordination flags are protected by the fixture mutex and the wrapped
   internal condition wait proves that each contender actually blocked.
2. The same deterministic harness pauses inside the native initializer and
   proves both unconfigured and configured-disabled native fallbacks remain out
   until READY is published (`plov/tests/test_sfclay_live.c:139-192`).
3. `scheme-mismatch` returns native-query success and `status == OK` while
   making `actual_scheme != expected_scheme`; it requires exactly one fallback,
   one query, and zero captures (`plov/tests/test_sfclay_live.c:51-56`,
   `:250-276`).
4. `reused-owner` is the terminated-owner/glibc-handle-reuse falsifier described
   above (`plov/tests/test_sfclay_live.c:193-220`).

The 53-case manifest comprises 33 normal and 20 fatal subprocesses
(`plov/tests/test_sfclay_live.py:9-13`).  The suite also covers init on a
non-owner thread, the original
LOGICAL pointer, proof-zero reader exclusion, direct-surface fallback,
pre/post-native drift, guard/readonly/owner staleness, token/span corruption,
and no replay.  The reported GCC 16 ASan/UBSan run was green; I inspected its
test definitions but did not independently run compilers or tests in this
read-only review.

## Scope and provenance

Review scope was limited to the live module, initialization lease, test pair,
and narrow transaction/capacity/live-span primitives.  This verdict makes no
MPI, WRF activation, wrapper-manifest, native ABI, remote execution, or OMP>0
claim.

Repository HEAD observed: `bbefeb155af430cc5c229ad1536b1d2746495c27`

Reviewed SHA-256:

```
4a57352cad8efc9b2c3ec2d186214c0966221020e95c6a5252f130de8fc6c9e0  wtune-sfclay-live-invariants-20260912.md
528bfa8a9e25935242c2a191ff4d286034d80b551d054f36dd6af93cae825a88  plov/wt_sfclay_live.h
99742f6c4ac2947c652866c325a06638d06d8ce315a47e2b19af10b08c08ebb6  plov/wt_sfclay_live.c
4af5ec2a73cb89e62744ab1819363c562d6f151d9d7a7707717ca90b06b68fde  plov/wt_sfclay_live_internal.h
767457b17f7865e44449dcf65819aaba841ba36633fc9c2eec5561c63ff33ac9  plov/wt_sfclay_init_lease.c
4b25d630a2499fdd8f37e8527b8c491690313adc2f30f4172b4d1e9fd6e43b44  plov/tests/test_sfclay_live.c
399b6e39cdfd49f3b06228eb88787a8a37a707654636da871f6df959ed1d4ada  plov/tests/test_sfclay_live.py
e6d3005e7d6c9b393dd10ecc62a52c00414bc8070e5898b29cf1c3fdc3b62a46  plov/wt_sfclay.c
517458a2cba87db52d8efb9463dc6ef3334005ca614d7b9c73ca9eecc252102d  plov/wt_sfclay_capacity.c
3bf882b9bddc6eccdba9867ba7cf24274e3d316e40e94a6f707433efc73e0f69  plov/wt_live_spans.c
```
