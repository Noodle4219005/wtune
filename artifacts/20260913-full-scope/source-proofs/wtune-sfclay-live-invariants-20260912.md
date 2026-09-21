# SFCLAY live admission, initialization lease, and context state machine

## Scope and decision

Implement a serial owner-local synchronous adapter first; this design makes no node thread-pool or remote-execution claim. It may stage only a direct SFCLAY call proven by `/tmp/u5727520/wtune-sfclay-native-provider-proof-20260912.md` (SHA-256 `108cb46a7528fa06b035f98f035219c6e90e2a66bb7d91c500f60b2db3c5d7cc`): native config/getters agree, `actual_scheme == expected_scheme` from the successful native query, `fractional_seaice=0`, the audited first-RK/surface activation is current, `num_tiles=1`, OMP query is exactly zero, initialization has completed, and capacities are valid for the same epoch. Runtime admission must consume `native_result.status` and `native_result.expected_scheme`; it must not embed the currently compiled numeric scheme value as a second authority. Every ordinary failure before native execution invokes the original SFCLAY exactly once. Once SFCLAY has run on clones, every failure is fatal; never run the original arguments.

The implementation should be a new `wt_sfclay_live.[ch]` module. Reuse the patterns, not the globals, of `wt_kf_live.c`: immutable configure (`:25-40`), outer span publication (`:48-96`), early OMP gating and generation validation (`:129-198`). Use existing `wt_sfclay_native_query`, `wt_sfclay_capacity_init_attested`, and the prepare/clone/commit core (`wt_sfclay.c:76-137`).

## Minimal API

```c
struct wt_sfclay_live_config {
  int enabled, source_abi_verified, caller_chain_verified;
  int initializer_relocation_verified;
  size_t max_spans, max_nodes, byte_budget;
  int (*capture)(void *grid, wt_live_spans *, size_t max_nodes, int *tiles);
  int (*omp_query)(void); /* only 0 admits */
  int (*native_query)(void *grid, void *cfg,
                      struct wt_sfclay_native_result *); /* injectable fixture */
};
int wt_sfclay_live_configure(const struct wt_sfclay_live_config *);
uint64_t wt_sfclay_first_rk_begin(void *grid, void *cfg);
void wt_sfclay_first_rk_end(uint64_t token);
uint64_t wt_sfclay_surface_begin(void);
void wt_sfclay_surface_end(uint64_t token);
void wt_sfclay_live_dispatch(void **argv, void (*real)(void **), size_t nargs);
void wt_sfclay_init_dispatch(int *allowed_to_read, void (*real)(int *));
void wt_sfclay_live_get_counters(struct wt_sfclay_live_counters *);
```

Production config supplies `wt_sfclay_native_query`; tests inject a C callback. All proof flags default false and are immutable after one quiescent configure. Configure captures the eligible owner `pthread_t`. It must not reset initialization state: `sfclayinit` can legitimately run before configure. The initializer wrapper is active even when live staging is unconfigured or disabled.

This module introduces pthread mutex/condition/cancellation calls. Every future SFCLAY recipe must compile the live module with `-pthread` and preserve `-pthread` in the later link manifest. The currently pending KF compile card and its pinned manifest remain unchanged.

The first-RK wrapper needs a separate generated-signature wrapper or audited stack/TLS-preserving trampoline. It encloses existing physics wrappers and must not use `WTWRAP`'s shared static `wt_a0..5`, saved-register, or return slots. The surface hook may use its existing wrapper only in a later pinned-manifest revision; do not mutate or reinterpret a pending authorized recipe.

## State

Process state:

* configuration: `UNCONFIGURED -> INSTALLING -> IMMUTABLE`;
* one `wt_live_spans` registry and monotonically nonzero outer generation;
* eligible-owner scope gate and synchronous staging gate;
* initialization lease: mutex/condition, `readers`, `writer_waiting`, `writer_active`, `init_state={UNINITIALIZED,INITIALIZING,READY}`, and nonzero `init_generation`;
* atomic counters by pre-invoke fallback reason.

TLS state is `{outer_token,surface_token,grid,cfg,query_result,span_generation,outer_reason,outer_depth,surface_depth,dispatch_depth,control_depth,init_depth,reader_depth}`. Tokens monotonically change; begin/end must match exactly. A disabled outer/surface activation still gets a pairing token and reason, allowing the real native call while preventing stale context reuse.

## Transitions

`first_rk_begin(grid,cfg)` rejects same-thread recursion as structural corruption. On the configured owner it queries OMP before dereferencing either pointer. If config/proofs/callbacks/native query fail, it installs a disabled TLS scope. Otherwise it claims the sole eligible scope gate, calls the injected native query under `control_depth`, captures a bounded grid span epoch, requires `tiles==1`, and publishes that generation only to the matching TLS scope. It returns a token in both cases. The wrapper invokes real first-RK exactly once. `first_rk_end(token)` requires no active surface/dispatch, clears publication before ending spans, releases the gate, and clears every TLS pointer. A different-thread first-RK gets a disabled context and remains native.

`surface_begin()` creates either an eligible activation (paired eligible first-RK, owner, current generation) or a disabled activation carrying the reason. A legitimate direct surface-driver call with no first-RK context gets a disabled surface token and its inner SFCLAY calls fall back natively; absence of first-RK is not structural fatal. Nested/reentrant surface activation or a mismatched existing token is fatal because a one-slot context cannot be restored safely. The real surface routine is always invoked once; `surface_end(token)` requires matching token and no dispatch, then clears the activation.

`sfclay_live_dispatch` first rejects null bridge inputs, wrong `nargs`, callback reentry, or same-wrapper recursion as structural fatal errors. If configuration is not IMMUTABLE/enabled, either proof is false, owner/scope is ineligible, or the relevant callback pointer is null, it invokes no callback and dereferences no argv entry: it takes the reader lease and forwards original SFCLAY once. Only after those gates does it query OMP, still before reading shape scalars or provider metadata. Missing, negative, or nonzero OMP is an ordinary pre-invoke fallback. Re-query native config from the stored live `grid,cfg` and require it to match the outer result; this makes config drift a fallback before native. Calls arising from the fractional-sea-ice wrapper therefore remain two separate native fallbacks.

For an otherwise eligible call, claim the staging gate, then acquire an initialization reader lease. Under that lease require READY and snapshot `init_generation`; otherwise invoke the original once. Require current outer/surface tokens and span generation, then call:

```c
wt_sfclay_capacity_init_attested(&cap, argv, spans, span_generation,
    config.source_abi_verified,
    config.caller_chain_verified && dynamic_chain_is_current);
wt_sfclay_prepare(argv, cap.bytes, config.byte_budget, &tx);
real(wt_sfclay_cloned_argv(tx));              /* native_started = true */
/* Recheck tokens, span generation, init generation, guards/staleness. */
wt_sfclay_commit(tx);
```

Do not force `caller_chain_verified=1`: the source-proven capacity exemption in `wt_sfclay_capacity.c:31-47` is available only when both immutable attestation and the live first-RK/surface chain are true. Capacity/prepare/readiness failure before `native_started` calls original SFCLAY once under the reader lease. After `native_started`, a failed generation check, guard, readonly validation, stale-owner check, or commit aborts. Destroy the transaction and release reader/staging state only on a valid path.

Unconfigured, disabled, or `source_abi_verified=0` operation still acquires an initialization reader lease and calls original SFCLAY exactly once. If initialization is currently active it waits; if state is UNINITIALIZED it calls native without staging, preserving the original program's ordering rather than inventing readiness.

`sfclay_init_dispatch(allowed,real)` is independent of configure, proof flags, owner thread, and OMP. Disable thread cancellation, reject same-thread callback/reader/init reentry, acquire a writer-preferred initialization lease, publish INITIALIZING, and call `real(allowed)` exactly once with the original LOGICAL pointer. On return increment generation, publish READY with release semantics, clear writer state, wake waiters, and restore cancellation. Calls from another thread are serialized and never skipped. Multiple sequential initializations are valid. A reader holds its logical lease across either native fallback or clone execution and commit, so a writer cannot start after the READY check.

## Guard order and progress

Eligible staged order is `eligible-scope gate (already represented by TLS) -> staging gate -> init reader lease -> span/provider -> transaction`. Native fallback takes only the init reader lease. Initializer takes only the init writer lease. The init mutex protects counters/state only and is never held while calling OMP/native-query/capture callbacks, span functions, allocation, native SFCLAY, native initializer, or commit. No path holding a span/provider lock may acquire the init mutex. These rules eliminate a lock cycle and let an initializer on another thread wait for a reader, then proceed.

Use writer preference (`writer_waiting` blocks new readers) to prevent init starvation. Condvar waits and native calls have no safe timeout or cancellation guarantee. Disable pthread cancellation from before lease acquisition until all logical leases/gates are released. A stuck initializer blocks SFCLAY; it must not trigger native fallback racing the tables. Signals/longjmp across wrappers and asynchronous cancellation are unsupported process-fatal conditions.

## Falsifiers

Ordinary pre-native fallback: disabled/unconfigured/proof false; non-owner thread; OMP unavailable/nonzero; native-query failure or drift; scheme/fractional mismatch; no eligible outer/surface token; tiles other than one; absent/current-epoch span; initializer UNINITIALIZED after acquiring the lease; capacity, budget, overlap, or prepare rejection.

Structural fatal even before native: begin/end token mismatch; same-wrapper or callback reentry; nested surface/first-RK on the eligible owner; staging-gate contention that should be impossible under owner affinity; TLS depth corruption; init called by a thread already holding a reader; synchronization failure. Fatal after native: any outer/span/init generation change, guard overwrite, readonly clone mutation, owner mutation, or commit failure. No fatal path may call native again.

Acceptance falsifiers for tests: init on a non-owner thread is not forwarded exactly once; SFCLAY overlaps table writes; `source_abi_verified=0` stages or bypasses the reader lease; fractional sea ice stages either nested call; a stale surface/outer token stages; caller-chain exemption occurs with either proof false; unsupported OMP does anything except one native fallback; or post-native failure re-executes original arguments.
