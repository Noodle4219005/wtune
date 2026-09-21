# MYNN owner-local core admission review — 2026-09-12

## Decision

**GO for a bounded serial owner-local transaction-core prototype around the complete external `module_bl_mynnedmf_driver_mp_mynnedmf_driver_` invocation.** This is not LIVE admission yet. LIVE remains closed until the separately owned 132-formal schema, optional-argument ABI, alias, extent/capacity, caller-chain, and linked-object attestations pass, followed by the native differential fixture below. This decision neither admits process transport nor parallel execution.

Frozen sources reviewed:

- `phys/module_bl_mynnedmf_driver.f90`, SHA-256 `9b339c1bb9e318086113b99e40f691d731639adbe57b579cf98046bf3f2dcfde`
- `phys/module_bl_mynnedmf.f90`, SHA-256 `a93d0c0e453738c16751525aec04403ec129b735b577e5f399b1edfd2129937b`
- `phys/module_bl_mynnedmf_common.f90`, SHA-256 `960a7370de0f33a85bc948eaf13e2a9811f6553450ac111e8f6839ea8f132cd8`
- `share/module_model_constants.f90`, SHA-256 `c387a51c2992da393551ac63291c79afbf6b78efbca2045a18cf9c94381f9954`

Prior evidence: `wtune-mynn-boundary-20260912.md` SHA-256 `9c82823cc9cc15a91f33de65b3126db5039e91ec79352499b4564c2d3e0540aa`; `wtune-mynn-kind-save-proof-20260912.md` SHA-256 `fd5537491fb711e2229f9169328e8a964a4f56f3266939b71bb80d5afab4354c`.

## State-closure proof

1. **The five implicit-`SAVE` locals do not form a hidden write set.** `mym_initialize` declares initialized locals `pmz=1., phh=1., flt=0., fltv=0., flq=0.` at `module_bl_mynnedmf.f90:1163-1164`; Fortran gives these saved lifetime. In the complete routine (`:1124-1304`), `pmz` is only read at `:1196,1204,1258,1261`, `phh` only at `:1204,1261`, and `flt/fltv/flq` are forwarded at `:1232`. The only callee, `mym_length`, declares all three `INTENT(IN)` at `:1582`. There is no assignment, pointer association, or escape. Their state is constant across calls, so it need not be cloned or locked.

2. **Chemistry is unreachable in this frozen driver.** The external driver declares `mix_chem=.false.` and `enh_mix=.false.` as parameters at `module_bl_mynnedmf_driver.f90:207-208`, passes them to the internal core at `:548`, and the only `mynn_mix_chem` call is guarded by `if (mix_chem)` at `module_bl_mynnedmf.f90:964-984`. Chemistry packing is likewise guarded in the mass-flux closure (for example `:6396-6398` and `:6909-6913`). `mynn_mix_chem`'s `integer,SAVE :: icall` at `:5574` has no other occurrence in the frozen file: it is neither read nor written. It therefore contributes no state even if that routine were called, but admission relies on the stronger compile-time-unreachable proof.

3. **No mutable module variable was found in the native closure.** `module_bl_mynnedmf_driver` enters `CONTAINS` immediately at `module_bl_mynnedmf_driver.f90:9-11`; its work arrays, `errmsg`, and `errflg` are invocation locals. `module_bl_mynnedmf` exposes only the parameters `debug_code`, `idbg`, and `jdbg` before `CONTAINS` (`module_bl_mynnedmf.f90:136-147`). `module_bl_mynnedmf_common` has a module-wide `SAVE` at `module_bl_mynnedmf_common.f90:15`, but every object it declares is `PARAMETER` (`:32-143`). Its imported `module_model_constants` values are also `PARAMETER` (`module_model_constants.f90:9-60,73-104`). Thus the audited module write set is entirely caller storage plus automatic/local scratch.

4. **Observed I/O and allocation are excluded from the first envelope.** Driver debug output is dead under local `logical,parameter :: debug=.false.` (`module_bl_mynnedmf_driver.f90:249`; guarded uses include `:258-262,578-580,712-714`). Core debug output is dead under `debug_code=.false.` (`module_bl_mynnedmf.f90:142`) or helper-local `debug_mf=0` (`:6228`) and `debug_dd=0` (`:7766`). The driver allocates five TKE-budget work arrays only for `tke_budget==1` (`module_bl_mynnedmf_driver.f90:276-282`) and deallocates them at `:701-707`. The first prototype must require `tke_budget==0`; this removes allocator failure/state from its admitted branch. This restriction can be relaxed only with a separate native failure/side-effect contract.

5. **The transaction boundary must remain the complete external driver.** It performs column packing, calls `mynnedmf_pre_run`, then the internal `mynnedmf`, then `mynnedmf_post_run` (`module_bl_mynnedmf_driver.f90:496-577`), and finally scatters outputs (`:582-698`). The internal CHARACTER `errmsg` never crosses the external ABI. Interposing only the internal column call would omit pre/post conversions and driver scatter semantics.

## Required admission invariants

- Exact source hashes above, generated 132-formal schema hash, `kind_phys` width, compiler/ABI manifest, symbol ownership, and linked object hashes all match the reviewed endpoint. A source hash alone is insufficient.
- Invocation is owner-local and serial: one caller thread, `omp_in_parallel()==0`, no callback/reentry, no concurrent access to any owner span, and one native driver execution. This does not establish general MYNN thread safety.
- Supported branch requires `tke_budget==0`; the caller-chain attestation excludes the outer `pert_mynn && multi_perturb==1` path. The runtime scheme/config query must select MYNN and all branch-driving scalars must be included in the immutable work identity.
- Every actual has a proved live capacity for its complete formal extent, including optional presence. Aliases are represented as one owner interval with all formal views reconstructed at their exact offsets. Unknown capacity, inconsistent overlap, absent required actual, arithmetic overflow, or schema mismatch causes fallback **before** native entry.
- Preparation clones the complete admitted owner set and preserves alias topology; execution calls the native external driver once on clones; validation then commits each writable owner once. Read-only owners must compare unchanged. Any error after native entry is fatal for this invocation; it must never trigger native fallback or a retry.
- `mix_chem=.false.` is an attested build property, not a caller-selectable promise. A future source/object where it is mutable or true is a distinct schema/closure and is rejected.

## Minimal interface sketch

```c
typedef struct wt_mynn_schema wt_mynn_schema;   /* generated elsewhere: 132 formals */
typedef struct wt_mynn_actuals wt_mynn_actuals; /* pointer, presence, proved capacity */
typedef struct wt_mynn_txn wt_mynn_txn;

enum wt_mynn_admit {
  WT_MYNN_ADMIT = 0,       /* safe to invoke staged native exactly once */
  WT_MYNN_FALLBACK,        /* only before native entry */
  WT_MYNN_FATAL            /* invariant failure, or any post-entry failure */
};

enum wt_mynn_admit wt_mynn_prepare(const wt_mynn_schema *,
                                   const wt_mynn_actuals *,
                                   const struct wt_mynn_runtime_proof *,
                                   wt_mynn_txn **);
enum wt_mynn_admit wt_mynn_execute_once(wt_mynn_txn *,
                                        const struct wt_mynn_native_bridge *);
enum wt_mynn_admit wt_mynn_validate_and_commit(wt_mynn_txn *);
void wt_mynn_discard_preinvoke(wt_mynn_txn *); /* legal only before entry */
```

The bridge is generated from the accepted schema and calls the real 132-reference Fortran entry. It must not use a generic x86-64 register-call shim. State transitions are `NEW -> PREPARED -> ENTERED -> VALIDATED -> COMMITTED`; only `NEW/PREPARED` may return `WT_MYNN_FALLBACK`.

## Minimal native fixture before LIVE

Use the same compiled Fortran driver object and generated bridge intended for LIVE. In separate fresh processes, run (A) the native driver directly and (B) prepare/execute/commit, from byte-identical valid WRF-derived actuals under the admitted branch. Execute a two-call sequence in each process so call-history divergence from a missed saved write is observable. Compare every writable owner byte after each call and assert every read-only owner is unchanged; preserve aliases, optional-presence combinations used by the real caller, exact bounds, and red zones. Assert native-entry count is exactly one in B and that each injected pre-entry rejection calls the direct native path exactly once. Inject a post-entry validation failure and assert fatal termination/status with zero fallback calls. Run under memory/Fortran bounds checking. A single synthetic column with arbitrary values is insufficient; at least one captured valid nested-grid and one parent-grid call state must exercise the real loop/bound branches.

## Falsifiers

- Any write/escape of `pmz/phh/flt/fltv/flq`, or a changed `mym_length` intent, invalidates the no-lock proof.
- Reachable `mix_chem`, a use of `icall`, any new mutable module/SAVE object, reachable debug/runtime I/O, or admitted `tke_budget==1` invalidates this closure.
- A helper resolving outside the attested object closure, compiler-generated optional/temporary semantics differing from the bridge, or a linked-object hash mismatch closes admission.
- Any capacity/alias ambiguity, read-only byte change, red-zone write, baseline/staged byte mismatch, call-history divergence, duplicate native entry, or post-entry fallback is a hard failure.

This review closes only the hidden-state question. It does not replace the Astra-owned ABI/schema inventory or prove caller capacities.
