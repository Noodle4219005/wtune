# `rk_scalar_tend` serial owner-local closure — 2026-09-12

## Decision

**GO for a bounded owner-local prototype on one interior scalar tile.** The admitted operation snapshots complete proved backing spans, reconstructs all aliases in private storage, invokes the native 71-argument routine once, validates, and commits the writable-owner union once. It neither changes WRF ownership nor performs a halo exchange. Remote execution, boundary tiles, final/first RK substeps, IEVA, convection, diffusion, other advection orders, and OpenMP-active calls remain closed.

The ABI authority is `/tmp/u5727520/wtune-rk-scalar-compiled-abi-20260912.json`, SHA-256 `18cd06a09574f03f2d7a9bb8284fb7aebe47a63f810064073635589288e1785a` (71 arguments). Relevant frozen hashes are: `module_em.f90` `32c1b72a…f0944`; `module_advect_em.f90` `8ce34b00…965a`; `module_ieva_em.f90` `3e1e8077…1dbc`; `module_big_step_utilities_em.f90` `36003375…013`; `solve_em.f90` `e3462d09…c554`; `frame/yy0.f90` `bcb6df7f…bcd`; `module_model_constants.f90` `c387a51c…9954`. Linked-object and compiler manifests must carry full hashes before LIVE.

## Exact initial branch

Admission requires all of the following:

```text
scs == sce
moist_step == .false.
2 <= rk_step < query.rk_ord
query.h_sca_adv_order == 2
query.v_sca_adv_order == 2
its >= ids+1; ite <= ide-2
jts >= jds+1; jte <= jde-2
ims <= its-1; ite+1 <= ime
jms <= jts-1; jte+1 <= jme
kms <= kts; kts+1 <= min(kte,kde-1)
min(kte,kde-1)+1 <= kte <= kme
omp_in_parallel() == 0 and an exclusive owner/scratch lease is held
```

`rk_scalar_tend` computes `ieva=CHK_IEVA(config_flags,rk_step)` and selects special advection only when `rk_step==rk_order` (`module_em.f90:1199-1218,1260-1345`). The middle-step inequality therefore selects the ordinary `advect_scalar` arm and proves IEVA false regardless of `zadvect_implicit`; it also bypasses all `rk_step==1` second/sixth-order diffusion (`:1380-1439`). `moist_step=.false.` proves the `set_tend(RQVFTEN,...)` convection branch dead (`:1368-1378`). The compiled scalar caller already passes one species and `.false.` (`solve_em.f90:3043-3078`).

The interior inequalities make `specified`, `nested`, `periodic_x`, `open_xs/open_xe/open_ys/open_ye`, and `polar` boundary adjustments observationally inactive even though the native routine may read them. In the order-two arm, source ranges and neighbor reads are explicit at `module_advect_em.f90:3921-3978`; the later open-boundary corrections require `its==ids`, `ite==ide`, `jts==jds`, or `jte==jde` (`:3996-4071`), all excluded. Order-two vertical flux reads `field(k-1)` and then consumes `vflux(k+1)` through `ktf=min(kte,kde-1)` (`:4228-4246`). The last vertical inequality is mandatory because local `vflux` is declared only `kts:kte` (`:2977`); it is a direct bounds proof, not an assumed WRF convention.

`adv_opt`, `tenddec`, `diff_6th_opt/factor`, `mix2_off`, `mix6_off`, `diff_opt`, `bl_pbl_physics`, `cu_physics`, and all topology selectors remain part of the immutable selector snapshot/work identity even when this envelope proves their effect dead.

## Reachable state and side effects

- `prandtl` is the frozen `PARAMETER 1./3.` (`share/module_model_constants.f90:86`). `khdq/kvdq` are computed from it at `module_em.f90:1199-1201`, but diffusion is unreachable. There is no mutable state here.
- `CHK_IEVA` only reads `rk_ord` and `zadvect_implicit` and returns a logical (`module_ieva_em.f90:13-44`). With IEVA false, `WW_SPLIT` copies `ww` to `wwE`, zeros `wwI`, and touches `[its-1:min(ite,ide-1)+1, kts:kte, jts-1:min(jte,jde-1)+1]` (`:102-112,294-304`). Its implicit-SAVE `print_flag` (`:85-86`) is read but not written; debug I/O and the write to `print_flag` are guarded by IEVA (`:125-152`).
- Each species iteration calls `zero_tend` on `advect_tend`, `h_tendency`, and `z_tendency`; it writes precisely the tile box (`module_big_step_utilities_em.f90:4442-4477`). `scs==sce` eliminates multi-species scratch reuse from this first proof.
- `nl_get_time_step` is called twice (`module_em.f90:1228,1244`). The generated getter only assigns its output from `model_config_rec%time_step` (`frame/yy0.f90:476-483`). Order-two advection never uses `time_step`; nevertheless the configure lifecycle must forbid concurrent mutation.
- `advect_scalar` declares `config_flags INTENT(IN)`, input fields `INTENT(IN)`, and only `tendency INTENT(INOUT)` (`module_advect_em.f90:2921-2977`). Its order-two arms contain no calls. The only external error path is `wrf_error_fatal3` for unsupported orders (`:3987-3993,4247-4251`), excluded by the query.
- The reachable call graph contains no MPI call, halo exchange, allocator, file I/O, or mutable module write. Imported `module_dm` is not referenced on this path. Automatic `vflux/fqx/fqy` are invocation-local. This claim is limited to the branch and pinned objects above.

Potentially writable formal owners are the alias union containing `wwE`, `wwI`, `scalar_tends`, `advect_tend`, `h_tendency`, `z_tendency`, and `RQVFTEN`. The last two schema-declared outputs may receive no source store on some paths; treating every OUT/INOUT owner as writable is conservative. All other intrinsic owners must remain byte-identical. No owner is shared with a sibling tile during the lease: the compiled caller places this call inside an OpenMP tile loop (`solve_em.f90:3043-3052`) and uses shared scratch arrays, so any active parallel region or failed try-lock causes pre-invoke fallback exactly once.

## Opaque configuration and minimal API

The C core must never size, copy, decode, or synthesize `grid_config_rec_type`. An ABI-locked ordinary Fortran query/helper receives the original native config reference and emits intrinsic selector scalars:

```c
struct wt_rk_query {
  int status, rk_ord, zadvect_implicit;
  int h_sca_adv_order, v_sca_adv_order;
  int cu_physics, diff_opt, bl_pbl_physics, time_step;
  unsigned specified, nested, periodic_x, polar;
  unsigned open_xs, open_xe, open_ys, open_ye;
};

int wt_rk_query_native(const void *opaque_config, int rk_step,
                       struct wt_rk_query *out); /* ABI-locked adapter */
int wt_rk_prepare(const struct wt_rk_query *, const struct wt_rk_actuals *,
                  const struct wt_rk_limits *, struct wt_rk_txn **);
int wt_rk_execute_once(struct wt_rk_txn *, const void *opaque_config,
                       const struct wt_rk_native_bridge *);
int wt_rk_validate_commit(struct wt_rk_txn *);
```

The original config reference is forwarded unchanged to native execution under a caller-provided configure/read lease; query before prepare and after native must match. This does not make the derived value C-interoperable. The adapter, type module, compiler, source, and linked objects are one attested native endpoint. Absent lifecycle proof, null query, query disagreement, or config-address lifetime failure means fallback before native.

For every intrinsic actual, capacity must independently prove the complete declared backing extent from the ABI JSON, not merely the tile footprint or formal shape. Checked arithmetic forms a deduplicated owner graph and rejects capacity shortfall, alias conflict, or a configured total-clone-byte cap. Snapshot and clone every unique owner in full; preserve formal offsets; execute once; verify readonly owners; commit each writable owner once under the same exclusive lease. Pre-entry failure invokes original native exactly once. Any failure after entry is fatal with no retry, cancellation, or fallback.

## Falsifiers and native fixture

Admission closes if any gate above fails; if the query/linked hash changes; if config or global time-step may mutate during execution; if any capacity is unknown; or if an alias crosses a proved owner. It also closes for `rk_step==1`, `rk_step==rk_ord`, `moist_step`, `scs!=sce`, non-order-two advection, insufficient halo/top backing, active OpenMP/sibling access, or any remote request. Those branches require separate closure proofs.

The minimum fixture is a same-compiler Fortran harness that constructs `grid_config_rec_type` and initializes `model_config_rec%time_step`, then calls the same linked `rk_scalar_tend` object directly versus through the transaction on byte-identical order-two interior-tile data satisfying the inequalities. Compare all writable owner bytes, assert all readonly owner bytes/redzones unchanged, selector query unchanged, and native entry count one. Negative cases must cover each gate, one overlapping owner, capacity-minus-one, active OpenMP, and injected post-entry validation failure (fatal, zero fallback). A bounds-checked run must specifically exercise `ktf+1==kte`; failure falsifies the vertical proof. This fixture proves local equivalence only, not numerical quality, cross-node execution, or a halo redistribution scheme.
