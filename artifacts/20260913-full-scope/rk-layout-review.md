# RK scalar layout consequential correctness review

Verdict: **ACCEPT after focused delta**. The revised layout fixes the previously rejected vector-byte, signed-bound, empty-tile, query-boolean, selector-comparison, aliased-query, and output-contract defects. Its shape and branch arithmetic matches the immutable 71-formal schema and the frozen native declarations. The final delta makes the before/after selector comparison non-vacuous and documents the required output disjointness.

Reviewed read-only through fresh jCodeMunch indexes:

- `plov/rk_scalar_schema.json`
- `plov/gen_rk_schema.py`
- `plov/wt_rk_schema.h`
- `plov/wt_rk_layout.h` and `plov/wt_rk_layout.c`
- `plov/wt_rk_native_query.h`
- `plov/tests/test_rk_layout.c` and `plov/tests/test_rk_layout.py`
- frozen `dyn_em/module_em.f90:1096-1198`
- `artifacts/20260913-full-scope/source-proofs/wtune-rk-scalar-local-closure-20260912.md`

I did not modify source, invoke a compiler, create temporary files, query a cluster, or claim native/numerical equivalence.

## P1: aliased query pointers erase the before observation

The closure requires a selector query before preparation and another after native execution, under the same config read lease, with all 17 fields equal. `wt_rk_layout_build` checks only that both pointers are non-null at `plov/wt_rk_layout.c:102`. It later validates and compares the pointed-to values at `:116`, then copies `*query_before` into the layout at `:117`.

There is no requirement that `query_before` and `query_after` name distinct stable snapshots. The positive fixture explicitly sets both to the same object at `plov/tests/test_rk_layout.c:17`:

```c
cfg.query_before = cfg.query_after = &query;
```

Concrete counterexample: the caller queries into `q` before native, native/config state changes, and the caller queries after native into the same `q`. Passing `query_before=&q` and `query_after=&q` makes every field compare equal to itself, so layout returns OK using only the after value. The original before value no longer exists. Canonical boolean checks and fieldwise comparison cannot detect this loss.

Minimal fix:

1. Require `query_before != query_after` and document that both point to distinct, stable result objects for the duration of the call under the config read lease.
2. After complete intrinsic pointer/capacity preflight and scalar reads, copy each query once to local `struct wt_rk_native_result before, after`; run `query_valid` and `wt_rk_selector_equal` on those locals; store `before` in `plan.selectors`. This avoids repeated reads from caller-owned mutable structs.
3. Change the positive fixture to two distinct equal query objects. Add a same-pointer negative that expects `WT_RK_LAYOUT_QUERY` and zeroed output, retain the one-field mismatch negative, and verify later mutation of either caller query object cannot alter `out.selectors`.

The API still cannot prove that the caller actually invoked the native helper twice; that remains an attested lifecycle responsibility. Distinct immutable snapshots are the minimum representation that makes the equality gate meaningful.

## P2: the eager zero-on-failure write has no output non-alias contract

`wt_rk_layout_build` calls `memset(out, 0, sizeof *out)` at `plov/wt_rk_layout.c:97` before it validates or reads the config, query objects, pointer table, capacity table, or intrinsic owners. `plov/wt_rk_layout.h:1-20` declares the API but gives no requirement that `out` be disjoint from those inputs.

An overlapping, sufficiently sized caller allocation can therefore be zeroed before validation, corrupting the very config/query/table/owner bytes the function is supposed to preflight. This also makes the promised zeroed failure result incompatible with unrestricted aliasing.

Minimal fix: add an explicit header contract that `out` points to a valid standalone `struct wt_rk_layout` object and does not overlap `config`, `argv`, `capacities`, either query result, `config_identity`, or any intrinsic owner interval. If overlap is intended to be rejected dynamically, delay all output writes, build entirely in the local `plan`, preflight ranges, reject overlap, and only then write a zero or successful result. A documented non-alias precondition is the smaller change and matches the established MYNN layout contract.

## Invariants that pass source review

- **Schema/native signature:** the JSON contains exactly 71 ordered, non-optional formals. The generated header matches names, types, shapes, and the seven conservative writable formals. Frozen `module_em.f90:1096-1179` confirms all nine vector formals (`c1h`, `c2h`, `c1f`, `c2f`, `base`, `fnm`, `fnp`, `rdn`, `rdnw`) are independently declared `DIMENSION(kms:kme)`. They therefore legitimately share the checked `4*extent(kms,kme)` byte formula; this is source evidence, not an assumption that every rank-one formal always has the same range.
- **Shape bytes:** `dimensions` computes `ni`, `nj`, and `nk` from `ims:ime`, `jms:jme`, and `kms:kme` at `plov/wt_rk_layout.c:61-66`. It checks every multiplication. `bytes_for_shape` at `:76-83` maps scalar to 4 bytes, vector to `4*nk`, plane to `4*ni*nj`, volume to `4*ni*nj*nk`, and species-volume to volume. The last mapping is correct only because the branch gate proves `scs==sce` at `:118`, so the species extent is exactly one.
- **Signed bounds and empty extents:** all 18 bound scalars are copied from 32-bit owner data into `int64_t` locals before `+1`, `-1`, or `-2` arithmetic (`:52-61`). Their full difference fits in int64. Domain, memory, and tile ranges are explicitly nonempty at `:62`; `extent` rejects inverted ranges at `:27-34`. The earlier INT32 overflow and inverted-tile paths are gone.
- **Exact branch closure:** `:118-120` enforces `scs==sce`, `moist_step==false`, `2<=rk_step<rk_ord`, and horizontal/vertical scalar order 2. `:62-65` enforces one-cell interior horizontal halo and memory backing. `ktf=min(kte,kde-1)` plus `kts+1<=ktf`, `ktf+1<=kte`, and `kte<=kme` at `:64-65` exactly covers the order-two `vflux(k+1)` upper read while keeping the tile vertically nonempty.
- **Complete intrinsic preflight before scalar reads:** after config/query/schema pointer gates, `:105-113` checks every intrinsic actual is non-null, at least four bytes, nonwrapping through its reported minimum capacity, and four-byte aligned. The opaque config formal is separately required to be pointer-identical to `config_identity` with capacity zero at `:103`; it is never dereferenced, sized, copied, or included in byte accounting. Only after the full loop does `:114-115` read scalar bytes and enforce canonical native LOGICAL bits.
- **Configuration booleans and lease gates:** `config_valid` at `:69-74` requires exact 0/1 representation for activation, ABI/closure attestations, read lease, actual OpenMP state, exclusive ownership, and allocation-descriptor presence; LOGICAL true is restricted to 1 or `UINT32_MAX`. `:98-101` then requires enabled, both proof gates, active read lease, inactive OpenMP, and exclusive owner. A declared allocation descriptor without a positive byte budget fails at `:101`.
- **Query representation and complete identity:** `query_valid` at `:36-41` requires status OK, positive `rk_ord` and advection orders, plus canonical 0/1 values for `zadvect_implicit` and all eight topology flags. `wt_rk_selector_equal` at `:43-50` compares all 17 fields: status; RK order; implicit flag; both advection orders; `cu_physics`; `diff_opt`; `bl_pbl_physics`; `time_step`; and all eight topology selectors. Struct assignment snapshots field values into the resulting layout; the P1 concerns loss of the distinct prior observation before that assignment.
- **Capacity, address, and total arithmetic:** after dimensions are known, `:123-125` computes every intrinsic formal's complete declared backing bytes, rechecks capacity and pointer-end nonwrap for that full size, checks total `SIZE_MAX` addition, and applies the optional byte budget. No owner is accessed beyond the initial four-byte scalar reads in this descriptor builder.
- **Fieldwise writability:** the generator's writable set is exactly `wwE`, `wwI`, `scalar_tends`, `advect_tend`, `h_tendency`, `z_tendency`, and `RQVFTEN` (`gen_rk_schema.py:13`). This matches the native declarations plus the closure proof for `WW_SPLIT`; all other formals remain read-only in the metadata.
- **Failure output:** with the required output non-alias precondition added, the local zero plan and eager zeroing ensure every ordinary validation failure leaves `out.ready==0`; only the final assignment at `:126` publishes a ready layout.

## Missing falsifiers after the blocking fix

These are test gaps, not additional current-source mismatches:

1. `tests/test_rk_layout.py:5-10` generates a header but only checks three strings; it never compares the generated bytes with checked-in `plov/wt_rk_schema.h`. Add an exact comparison. Otherwise a stale header with unchanged names but a wrong non-config type/shape/writable field can remain self-consistent with the C fixture; `schema_ok` at `wt_rk_layout.c:14-20` validates all names but only the config formal's type/shape.
2. Add capacity-minus-one positives/negatives for vector, plane, volume, and species-volume independently; current coverage exercises RU and C1H only.
3. Add a synthetic `UINTPTR_MAX` endpoint failure and a dimension product that reaches `WT_RK_LAYOUT_OVERFLOW` while satisfying branch inequalities. Current INT_MIN/INT_MAX cases are useful but are allowed to fail earlier as BOUNDS and do not prove multiplication overflow handling.
4. Flip each of the 17 query fields independently, and feed every boolean field a noncanonical value. Current tests cover `time_step` mismatch and `specified=2`; source review confirms all fields, but a table-driven test would protect the fieldwise contract.
5. Add one negative for every closure inequality: both sides of each horizontal halo constraint, `kts+1>ktf`, `ktf+1>kte`, `kte>kme`, first/final RK steps, moist true, and both non-two advection orders.

This review admits only the intrinsic layout descriptor after the P1/P2 fixes. It does not attest default Fortran kind widths, `grid_config_rec_type` ABI, query helper linkage, transaction aliasing/commit behavior, native equivalence, halo movement, OpenMP lifecycle, or production activation.

## Final focused delta review

**ACCEPT. No residual finding in the assigned delta.**

- `plov/wt_rk_layout.h:14-16` now requires two distinct, stable selector observations under the caller's config read lease.
- `plov/wt_rk_layout.h:20-21` now states that `out` is a valid writable layout object disjoint from the config, pointer/capacity tables, both query snapshots, and every intrinsic owner. This resolves the zero-before-validation alias ambiguity as an explicit caller precondition.
- `plov/wt_rk_layout.c:98-107` completes the full minimum-capacity, pointer-end, alignment, and presence preflight for all intrinsic owners before it reads any scalar or dereferences a query result.
- `plov/wt_rk_layout.c:108-115` reads and validates scalar values first, rejects identical query pointers, copies each distinct query object exactly once into local `before`/`after` structs, canonical-validates both local copies, compares all 17 fields, and stores the `before` copy by value. Later caller mutation cannot change `plan.selectors`.
- `plov/tests/test_rk_layout.c:12-18` constructs two separate equal query objects for the positive case. `:39` retains the distinct-object field-drift negative, and `:40` adds the same-pointer rejection with zeroed output. These cases directly falsify the original vacuous comparison while preserving the earlier branch and representation checks.

This focused review did not rerun the compiler-backed test suite, per the no-compile/no-temporary-write assignment. The lead is collecting that independent execution evidence. Acceptance here is for the exact reviewed source delta and does not widen the boundary limitations above.
