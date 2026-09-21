# RK owner-local transaction contract review

## Verdict: PASS — actionable bounded design

The accepted RK layout is sufficient to drive a MYNN-style owner-local transaction, provided the transaction adds a real post-native selector query and a caller-integrated scope check. This contract admits only the proved serial, single-species, middle-RK, order-two interior-tile branch. It does not implement or claim remote execution, halo movement, boundary tiles, OpenMP-active calls, first/final RK steps, moisture, other advection orders, IEVA, convection, diffusion, or full dynamics support.

The narrow `git log -S wt_rk_layout_build` and `git log -S wt_mynn_prepare` searches returned no tracked history in this working tree. The current accepted sources and the shared `rk-layout-review.md`, `mynn-core-review.md`, and `source-proofs/wtune-rk-scalar-local-closure-20260912.md` are therefore the applicable prior art. No source, WRF tree, build, compiler, or cluster state was changed.

## Minimal public API

```c
typedef struct wt_rk_tx wt_rk_tx;

enum wt_rk_status {
    WT_RK_OK = 0, WT_RK_ARGUMENT, WT_RK_LAYOUT, WT_RK_BUDGET,
    WT_RK_ALIAS, WT_RK_ALLOCATION, WT_RK_STATE, WT_RK_SCOPE,
    WT_RK_STALE, WT_RK_REBIND, WT_RK_GUARD, WT_RK_READONLY,
    WT_RK_QUERY, WT_RK_OVERFLOW
};

enum wt_rk_state {
    WT_RK_PREPARED, WT_RK_ENTERED, WT_RK_EXECUTED,
    WT_RK_COMMITTED, WT_RK_FAILED
};

typedef int  (*wt_rk_query_fn)(void *opaque_cfg, int rk_step,
                               struct wt_rk_native_result *out);
typedef void (*wt_rk_native_fn)(void *argv[WT_RK_NARG]);
typedef int  (*wt_rk_scope_check_fn)(const void *scope_cookie,
                                     const void *config_identity);

struct wt_rk_tx_config {
    const struct wt_rk_layout_config *layout_config;
    int queried_rk_step;              /* value used for both pre-query snapshots */
    wt_rk_query_fn query;             /* exact attested native-query endpoint */
    wt_rk_scope_check_fn scope_check; /* samples actual OMP/lease state */
    const void *scope_cookie;         /* stable until destroy */
    size_t total_budget;              /* complete transaction heap budget */
};

enum wt_rk_status wt_rk_prepare(
    void *const argv[WT_RK_NARG], const size_t capacities[WT_RK_NARG],
    const struct wt_rk_tx_config *config, wt_rk_tx **out);
enum wt_rk_status wt_rk_execute_once(wt_rk_tx *, wt_rk_native_fn);
enum wt_rk_status wt_rk_commit(wt_rk_tx *);
enum wt_rk_status wt_rk_destroy(wt_rk_tx *);
enum wt_rk_state  wt_rk_get_state(const wt_rk_tx *);
int wt_rk_entered_once(const wt_rk_tx *); /* caller's fallback/fatal discriminator */
```

`layout_config->query_before` and `query_after` are two distinct, stable, independently obtained **pre-entry** observations made with `queried_rk_step` under the held scope. The transaction calls `wt_rk_layout_build`, verifies the layout-copied `RK_STEP` equals `queried_rk_step`, and stores `layout.selectors` by value. Those two layout inputs cannot serve as the actual post-native observation. `wt_rk_execute_once` must invoke `query(tx->owners[WT_RK_CONFIG_FLAGS], stored_rk_step, &post)` after the native callback and compare all 17 fields against the stored selector with `wt_rk_selector_equal`. This closes the stale-snapshot ambiguity without changing the accepted layout API (`wt_rk_layout.h:10-23`; `wt_rk_layout.c:84-131`).

The scope callback is mandatory. At the real wrapper it must return true only when `omp_in_parallel()==0`, the same nonblocking exclusive owner/scratch lease remains held, and the same read lease covers both the original `grid_config_rec_type` and the `model_config_rec` state read by `nl_get_time_step`. It is called before layout preparation, immediately before native entry, after native return, after the post query, and immediately before commit. The existing booleans in `wt_rk_layout_config` remain admission records; they are not substitutes for these actual checks (`wt_rk_layout.c:69-73, 93-101`). A transient global mutation and restoration can evade selector equality, so equality is detection at boundaries, while the lease is the correctness invariant.

## Complete 71-formal ownership map

The schema fixes 71 ordered, non-optional references (`rk_scalar_schema.json:1-79`; generated indices in `wt_rk_schema.h:11-82`). The transaction accounts for every one:

| Class | Count | Formals | Transaction treatment |
|---|---:|---|---|
| Opaque derived config | 1 | `config_flags` / `argv[2]` | Never sized, copied, decoded, or placed in an alias region. Capacity must be zero and pointer-identical to `config_identity`. The exact original pointer is forwarded to native and every query under the scope lease. Caller attests that the unknown derived object is disjoint from all intrinsic owners. |
| Intrinsic scalars | 33 | `scs`, `sce`, `tenddec`, `rk_step`, `dt_step`, `moist_step`, `rdx`, `rdy`, `khdif`, `kvdif`, `diff_6th_opt`, `diff_6th_factor`, `adv_opt`, `mix2_off`, `mix6_off`, and `ids` through `kte` | Full four-byte views are cloned and remain read-only, including selectors inactive in this branch. |
| Vertical vectors | 9 | `c1h`, `c2h`, `c1f`, `c2f`, `base`, `fnm`, `fnp`, `rdn`, `rdnw` | Each complete `4*extent(kms:kme)` backing is cloned read-only. |
| Horizontal planes | 10 | `mut`, `mub`, `mu_old`, `msfux`, `msfuy`, `msfvx`, `msfvx_inv`, `msfvy`, `msftx`, `msfty` | Each complete `4*extent(ims:ime)*extent(jms:jme)` backing is cloned read-only. |
| Ordinary volumes | 15 | `ru`, `rv`, `ww`, `wwE`, `wwI`, `u_old`, `v_old`, `alt`, `advect_tend`, `h_tendency`, `z_tendency`, `RQVFTEN`, `xkmhd`, `phb`, `ph` | Complete three-dimensional backing is cloned. `wwE`, `wwI`, `advect_tend`, `h_tendency`, `z_tendency`, and `RQVFTEN` are writable; the other nine are read-only. |
| Species volumes | 3 | `scalar_old`, `scalar`, `scalar_tends` | `scs==sce` proves one three-dimensional backing for each. `scalar_tends` is writable; the other two are read-only. |

Thus all 70 intrinsic views are copied in full, including `INTENT(OUT)` and branch-inactive formals; exactly seven are writable (`wt_rk_schema.h:84-157`). The accepted layout computes the distinct vector, plane, and volume byte sizes with checked products and proves species extent one (`wt_rk_layout.c:53-82, 118-128`).

## Owner graph, capacity, and budget invariants

1. `wt_rk_prepare` checks structural reentry before touching `out`, new argv, capacities, configuration, query objects, or allocation state. Ordinary failure clears `*out`; structural reentry preserves it so `&outer_tx` cannot erase the only destroy handle.
2. `wt_rk_layout_build` runs first and retains its documented non-alias requirement for `out`. It preflights every intrinsic pointer, minimum capacity, four-byte alignment, and nonwrapping endpoint before any scalar read (`wt_rk_layout.c:84-115`). Transaction code must not read an owner before layout succeeds.
3. After layout succeeds, snapshot all 71 pointer-table entries and all 71 capacities. The caller-owned argv and capacity tables, every original intrinsic owner, config object, and scope cookie remain valid and unchanged until destroy. The two pre-query objects need remain stable only through `wt_rk_prepare`; the transaction stores their validated selector value and retains no borrowed query-result pointer.
4. For each of the 70 intrinsic views, form the half-open interval `[address,address+required_bytes)`, rejecting `UINTPTR_MAX` overflow. Sort by start. Strictly overlapping views with equal writability merge transitively into one union; mixed read-only/writable overlap returns `WT_RK_ALIAS`; equality at an endpoint is touching and stays separate. Preserve every formal's offset into its union clone. `argv[2]` bypasses this graph and is staged as the original pointer.
5. Before allocation or owner copying, compute a positive, checked total: transaction metadata, 71 saved owners, 71 saved capacities, staged argv, layout/selectors/scope state, all region descriptors, two full copies per unique union, and two fixed guards per union. Apply `total_budget` to that full heap total. The layout's `byte_budget` remains the separate cap on the sum of formal backing requirements. Use one allocation; fixed 71-entry preparation and callback tables may remain bounded stack storage.
6. Snapshot every union's full owner bytes once, copy snapshot to clone, and guard the clone on both sides. OUT and inactive views are not exceptions. Store no borrowed query-result pointer in the transaction; store the validated selector value.

These are the accepted MYNN union/accounting rules at `wt_mynn.c:76-110, 130-197`, specialized to 70 intrinsic owners and one opaque forwarded reference.

## Execution, validation, commit, and lifetime

The state machine is `PREPARED -> ENTERED -> EXECUTED -> COMMITTED`, with `FAILED` terminal. A nonzero monotonic TLS owner nonce binds every object to its creating thread. Foreign-thread calls return `WT_RK_STATE` without reading or changing mutable transaction state. Every public operation first checks the shared RK TLS reentry marker; when an RK transaction is ENTERED, any nested prepare/execute/commit/destroy poisons the outer object before reading the nested call's data or allocating.

Before ENTERED, validate the live scope, saved argv entries, saved capacities, every owner against its snapshot, pristine clones, and guards. Only then copy the 71 staged pointers to a bounded temporary argv, set ENTERED/TLS, and invoke native exactly once. Temporary entries 0..1 and 3..70 point to their preserved offsets in union clones; entry 2 is the exact original opaque config pointer. The callback must return normally, retain nothing, and leave every temporary pointer unchanged.

Keep the TLS marker active through pointer-rebind checks, scope checks, the fresh post-native query, guard checks, clone checks, and owner-staleness checks. Validate all 71 temporary pointers, both saved tables, every guard, all 63 read-only intrinsic clone views/unions, all original owner bytes, exact config identity, scope continuity, query success/canonical representation, and all 17 selector fields before changing state to EXECUTED. Any error after ENTERED poisons FAILED.

`wt_rk_commit` rechecks every table, owner snapshot, guard, read-only clone, post-query-valid flag, config identity, and the actual scope before its first owner write. It then performs one write per writable alias union and no write for read-only unions, and sets COMMITTED. This is an all-validation-before-write pass under exclusive ownership; it does not promise hardware atomicity or safety with concurrent readers. Replaying execute, committing twice, or executing after any failure is rejected terminally.

`wt_rk_entered_once` is the wrapper's required discriminator. A prepare failure or an execute failure before ENTERED permits destruction/release followed by exactly one call to the original native routine. Once it reports true, every failure is fatal: no native fallback, retry, replay, alternate branch, or partial commit. Reentrant destroy while ENTERED poisons but does not free; the outer frame retains its local handle and destroys after callback unwinds. No API accepts a pointer after successful destroy. No async callback, cancellation, `longjmp`, retained argv/query pointer, or owner/config/scope lifetime shorter than destroy is supported. These semantics follow the proven MYNN transition and validation ordering (`wt_mynn.c:112-128, 199-255`) while adding the RK post query and scope checks.

## Unsupported-branch gates and falsifiers

Admission must fail before native entry unless every existing layout gate holds: enabled; source ABI and caller closure attested; positive budgets; canonical logical bits; distinct valid/equal pre-query objects; `scs==sce`; `moist_step==false`; `2<=rk_step<rk_ord`; horizontal and vertical scalar order two; nonempty domain/memory/tile bounds; one-cell interior horizontal halo; and the proved `ktf+1<=kte<=kme` vertical backing (`wt_rk_layout.c:93-128`). The actual scope must also report OpenMP inactive and the exclusive owner/scratch plus config/global read leases held. The API has no remote target, halo buffer, MPI service, boundary mode, or generic dynamics branch selector, so those requests are rejected rather than silently mapped to this transaction.

Minimum deterministic tests:

1. Compare generated schema/header semantics exactly and assert the 1+33+9+10+15+3 partition, all 71 indices, and exactly seven writable names.
2. For scalar/vector/plane/volume/species capacities, test exact capacity and capacity-minus-one; pointer-end wrap, multiplication/total-budget overflow, allocation failure, exact full budget, and one-byte-short budget must perform zero allocation/copy/native calls on the rejected path.
3. Test equal-permission exact/partial/transitive overlap with preserved offsets and one union commit; mixed-permission overlap refusal; touching intervals remaining distinct; and opaque config exact-equality/disjointness rejection.
4. Prove all 70 intrinsic backings, including scalars, inactive inputs, OUT views, and the single-species backings, are snapshotted. Mutate each saved argv/cap entry and owner class. Corrupt each guard side and a read-only clone. Every failure must produce zero owner writes.
5. In the callback, assert entries 0..1 and 3..70 use clones while entry 2 is the original `config_identity`. Rebind every temporary pointer in turn, including entry 2, and require terminal `WT_RK_REBIND`.
6. Independently create two pre-query objects. Reject same-pointer, invalid representation, query failure, queried-step/layout-step mismatch, and one-at-a-time drift of every one of the 17 fields in the actual post query. A changed `time_step` specifically falsifies the global-state identity.
7. Drive the real scope check with `omp_in_parallel()!=0`, failed try-lock, lost config/global lease before entry, and scope loss after entry. Pre-entry cases allow one original call; post-entry cases permit zero fallback.
8. Exercise every branch/bounds inequality on both sides: first/final RK step, moist true, `scs!=sce`, each non-two order, horizontal boundary/halo failures, vertical `kts+1`, `ktf+1`, and `kte>kme`. Add explicit remote/halo requests at the wrapper and require pre-entry refusal.
9. Test execute/commit replay, wrong-thread operations, nested prepare with `out=&outer`, bad nested argv/no-read and zero-allocation witnesses, nested execute/commit, reentrant destroy/no-free, eventual exactly-one free, and post-destroy caller prohibition.
10. Same-compiler Fortran fixture: construct native config/global time step, run the exact linked routine directly and through the transaction on byte-identical order-two interior data, exercise `ktf+1==kte`, assert native entry count one, compare all writable union bytes, and verify every read-only byte/redzone unchanged. This establishes only the admitted local branch.

## Remaining native proof gaps and activation blockers

- No same-compiler direct-versus-transaction native fixture has yet established local equivalence, real default INTEGER/LOGICAL/REAL width and true-bit representation, query/helper linkage, or `ktf+1==kte` bounds behavior. `wt_rk_native_query.f90:19-49` checks kinds and reads all selectors plus `nl_get_time_step`, but that is not a transaction proof.
- Real caller capacities for every complete backing span and disjointness/lifetime of the opaque derived config still need call-site evidence. The schema explicitly declines to infer byte ABI (`rk_scalar_schema.json:5`).
- The accepted source proof places the scalar call in an OpenMP tile loop with shared tendency/scratch arrays. Unless the actual invocation observes OpenMP inactive and acquires the owner/scratch lease, this design rejects it. A configured flag claiming serial execution is insufficient.
- There is no implemented lease covering both `config_flags` and the global `model_config_rec%time_step`; repeated query equality cannot replace it. The lease constructor/checker and its call-site lifetime must be independently reviewed.
- Source/object/compiler/default-kind hashes and the ordinary-Fortran query endpoint still require one native build/link attestation. The current closure is source-limited.
- No branch proof exists for boundary, moisture, first/final RK, alternate advection, IEVA, convection, diffusion, active OpenMP, remote execution, halo transfer, MPI progress, or full dynamics. Those remain hard refusals and cannot inherit this transaction's acceptance.

These gaps block LIVE/native claims; they do not leave the bounded transaction API ambiguous. Implementing the contract plus the listed C/mock tests would be the next local step, followed by the single native fixture before any activation decision.
