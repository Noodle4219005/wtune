# RK scalar checked-layout review — 2026-09-13

Implemented only the bounded owner-local **layout admission** layer. It does
not call native code, allocate clone storage, copy/interpret `config_flags`,
start a transaction, or enable production execution.

`rk_scalar_schema.json` is an exact copy of the compiled-ABI proof and the
generator refuses any byte change from SHA-256
`18cd06a09574f03f2d7a9bb8284fb7aebe47a63f810064073635589288e1785a`. It also
pins `module_em.f90` SHA-256
`32c1b72af58dbefd0115481352529c35ecaa9ddfd45252d829c2a48d25af0944` and
generates complete 71-formal metadata/header.

Admission first validates enablement, source ABI, caller closure, config read
lease, actual non-OpenMP state, exclusive ownership, logical representation,
and budget-descriptor policy. It then requires the opaque identity snapshot to
equal `argv[2]`, retain capacity zero, and leaves that pointer untouched. All
other 70 references receive non-null/alignment/address-overflow/minimum
capacity checks before any scalar read. The scalar branch requires one species,
false `moist_step`, middle RK, order-two horizontal/vertical selectors, and
the proved interior/vertical `ktf` inequalities. Full declared backing bytes
use checked `size_t` products. `wwE`, `wwI`, and the five proved output owners
are explicitly writable; no missing-INTENT heuristic is used.

Verification used Pixi GCC 16.2.0 with C11, `-Wall -Wextra -Werror`, ASan and
UBSan:

```text
PYTHONPATH=/home/u5727520/wrf-stack/wtune \
CC=/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/.pixi/envs/default/bin/x86_64-conda-linux-gnu-gcc \
pixi run python -m unittest plov.tests.test_rk_layout -v
```

Result: 2 tests passed in 0.653 s. The fixture covers positive
multidimensional full-backing layout, writable-owner marking, disabled ABI
gate, late bad capacity before scalar reads, opaque NULL/pointer-1 and selector
identity refusal, one-species/middle-RK/order selectors, interior halo
rejection, before/after selector drift, and `INT_MAX` arithmetic refusal.
Generated-schema testing confirms all 71 entries including `RQVFTEN` and
species-volume shape, then rejects a changed proof.

This is not native WRF/Intel ABI certification. Native query/Fortran adapters,
transaction/commit logic, remote operation, callback wiring, and production
enablement remain outside this ownership boundary.

## Update 2026-09-13 — root review corrections

The initial review correctly rejected the first version. Its vector case used
`volume / plane`, which is an element count (`nk`) rather than bytes. The
corrected shape helper maps every generated `WT_RK_SHAPE_VECTOR` formal to
`4 * (kme-kms+1)`: `c1h`, `c2h`, `c1f`, `c2f`, `base`, `fnm`, `fnp`, `rdn`,
and `rdnw`. Plane and volume cases retain their separately checked byte
products; the one-species volume uses the same complete volume only after
`scs == sce` is proven.

The C fixture was extended before the correction. It first failed on the
vector `c1h == 24` assertion, then required capacities of only `nk` bytes to
be rejected. It also adds `INT_MIN+1`/`INT_MAX`, inverted I/J/K tile bounds,
invalid allocation-descriptor and query booleans, selector drift, and zeroed
output checks. Bounds now promote every source integer to `int64_t` before
performing `+1`/`-1`/`-2`; no signed `int` endpoint arithmetic remains.
`wt_rk_selector_equal` compares all 17 defined snapshot fields explicitly,
not struct padding.

Fresh GCC 16.2.0 ASan/UBSan evidence after these corrections: 2 tests passed
in 1.137 s. The source evidence remains the immutable schema plus
`wtune-rk-scalar-local-closure-20260912.md`; no native execution occurred.

## Update 2026-09-13 — independent selector observations

The layout now rejects identical `query_before`/`query_after` addresses. After
the scalar preflight it copies each distinct snapshot into separate local
objects, validates and compares those local copies field by field, and stores
the copied before snapshot. This prevents a native query from overwriting a
single caller object and making before/after comparison vacuous. The public
header documents that both observations must be independent and stable under
the config read lease, and that `out` must be a non-overlapping writable layout
object because it is zeroed before any input is read.

The fixture's successful path now uses two equal snapshots. It adds the RED
case where the same pointer is passed twice, plus per-field before/after drift
rejection. Fresh GCC 16.2.0 C11 `-Wall -Wextra -Werror` ASan/UBSan evidence:
2 tests passed in 1.191 s.
