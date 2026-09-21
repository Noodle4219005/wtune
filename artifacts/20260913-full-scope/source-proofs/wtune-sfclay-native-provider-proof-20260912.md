# SFCLAY native configuration and capacity-provider proof (2026-09-12)

## Decision

**GO only for the direct, owner-local SFCLAY path** when a native outer-scope query proves, for the current grid invocation, `sf_sfclay_physics == SFCLAYSCHEME == 91` and `fractional_seaice == 0`; the initialization state is READY; the call is on the admitted owner thread with no active multi-thread OpenMP region; and every non-source-proven array has a live span for the same outer-call epoch. Any failed check falls back before invoking staged SFCLAY.

`fractional_seaice == 1` is deliberately ineligible. The compiled surface driver branches to `SFCLAY_SEAICE_WRAPPER` (`phys/module_surface_driver.f90:1983-2005`), whose closure calls SFCLAY once for land/frozen water (`:6286`) and again for open water (`:6373`) with wrapper-local state. The direct 84-argument call exists only in the `ELSE` branch (`:2006`). Interposing those two nested calls as independent work would change the wrapper transaction and lacks capacity/lifetime proof for its private arrays.

## Native configuration source and helper

The actual call configuration comes from the `config_flags` snapshot passed to `first_rk_step_part1`: the compiled call to `surface_driver` supplies `FRACTIONAL_SEAICE=config_flags%fractional_seaice` and `SF_SFCLAY_PHYSICS=config_flags%sf_sfclay_physics` (`dyn_em/module_first_rk_step_part1.f90:395-410`). The nearest grid-owning outer boundary is `first_rk_step_part1(grid,config_flags,...)` (`:7-35`, declarations `:60-61`); `solve_em` passes those as its first two actuals (`solve_em.f90:949-970`). `surface_driver` itself does not have a grid first argument.

Use one same-toolchain plain-Fortran helper, called only with those two live native pointers:

```fortran
subroutine wt_sfclay_native_query(grid, cfg, grid_id, actual_scheme, &
                                  actual_fractional, expected_scheme, ok)
  use module_domain, only: domain
  use module_configure, only: grid_config_rec_type
  use module_driver_constants, only: max_domains
  use module_state_description, only: SFCLAYSCHEME
  type(domain), intent(in) :: grid
  type(grid_config_rec_type), intent(in) :: cfg
  integer, intent(out) :: grid_id, actual_scheme, actual_fractional
  integer, intent(out) :: expected_scheme, ok
  integer :: getter_scheme, getter_fractional
  grid_id = grid%id
  actual_scheme = 0
  actual_fractional = 0
  expected_scheme = SFCLAYSCHEME
  ok = 0
  if (grid_id < 1 .or. grid_id > max_domains) return
  actual_scheme = cfg%sf_sfclay_physics
  actual_fractional = cfg%fractional_seaice
  call nl_get_sf_sfclay_physics(grid_id, getter_scheme)
  call nl_get_fractional_seaice(grid_id, getter_fractional)
  if (actual_scheme /= getter_scheme) return
  if (actual_fractional /= getter_fractional) return
  if (actual_scheme /= SFCLAYSCHEME .or. actual_fractional /= 0) return
  ok = 1
end
```

The getter cross-check prevents a stale or mismatched `config_flags` snapshot from being treated as the active model configuration. `nl_get_sf_sfclay_physics` is exactly two default-INTEGER references and indexes `model_config_rec%sf_sfclay_physics(id_id)` (`frame/yy0.f90:1799-1806`); `nl_get_fractional_seaice` has the same ABI and returns the global scalar (`frame/yy1.f90:1008-1015`). The backing fields are respectively `INTEGER(max_domains)` and scalar `INTEGER` (`frame/module_configure.f90:452`, `:631`); `max_domains` is 11 (`frame/module_driver_constants.f90:47`). The selected constant is 91 (`frame/module_state_description.f90:112-120`).

This helper is intentionally not `BIND(C)`: WRF `domain` and `grid_config_rec_type` are not C-interoperable. Its C caller treats the first two arguments as opaque pointers and all five outputs as pointers to 4-byte integers. It must be compiled with the frozen Intel Fortran module files and ABI flags (`-i4 -real-size 32`) and included in the object manifest. A C struct view of either derived type is forbidden.

The outer scope hook must establish thread-local `{grid_id, config, epoch}` before calling the real `first_rk_step_part1`, and clear it on every return. Native evidence shows a cross-object relocation suitable for `--wrap`: `solve_em.o` has undefined `module_first_rk_step_part1_mp_first_rk_step_part1_`, while `module_first_rk_step_part1.o` defines it. A missing/stale TLS context at SFCLAY entry is an immediate pre-invoke fallback. Because the outer routine has a large Fortran ABI, its wrapper must be generated from the frozen signature or be an audited machine-specific register-preserving trampoline; an unprototyped C forwarder is inadmissible.

## Capacity and lifetime classes

For the ordered 84-entry schema in `/tmp/u5727520/wtune-sfclay-schema-audit-20260912.json` (SHA-256 `348941418e6ec617a0f87a6de4f3a5e201b3646a8919d29ed4f92ffabc654c50`):

* `U3D=u_phytmp` and `V3D=v_phytmp` are surface-driver automatic arrays with exact capacity `4*(ime-ims+1)*(kme-kms+1)*(jme-jms+1)` bytes. `QGH`, `CHS`, `CHS2`, `CQS2`, and `CPM` are automatic 2-D arrays, each exactly `4*(ime-ims+1)*(jme-jms+1)` bytes (`phys/module_surface_driver.f90:1192-1203`). `LAKEMASK` has the same exact 2-D capacity (`:1382-1385`). These eight capacities are source-proven without allocator metadata, using checked nonnegative multiplication. Their lifetime epoch is the current dynamic activation of `surface_driver`; no descriptor, pointer, or late result may survive its return.
* `PSIM`, `PSIH`, `GZ1OZ0`, and `DZ8W` are `solve_em` automatic arrays with explicit `grid%sm*:grid%em*` bounds (`dyn_em/solve_em.f90:267-275`) and remain live across the nested first-rk/surface/SFCLAY call. Their capacity may be source-proven only when the frozen `solve_em -> first_rk -> surface_driver -> SFCLAY` actual mapping and object manifest match.
* `T3D` ultimately comes from `grid%t_phy` (`solve_em.f90:949-965`), `P3D` from `grid%p_hyd`, and `QV3D` from the `moist(...,P_QV)` slab; the remaining forwarded grid fields likewise require a live native span/lifetime provider. Formal explicit shape proves the bytes SFCLAY may access, not the backing allocation. Current `wt_alloc_size` exact-base records have no deallocation/lifetime epoch and cannot alone admit them; the QV interior slab is also not an exact allocation base.

The minimal span API is an outer-epoch provider: `begin(grid,cfg)->epoch`, `lookup(epoch,actual_ptr,required_bytes)->{base,available_bytes,grid_id}`, and `end(epoch)`. A lookup succeeds only for a pointer range presently owned by that exact grid and epoch. The eight surface automatics and four solve automatics use the source-proven class above; all other arrays use live spans. Aliases are validated by the transaction core before cloning.

## SFCLAY table readiness

`sfclayinit` has exactly one default-LOGICAL argument by reference; it does not read it and writes all `PSIMTB(0:1000)` and `PSIHTB(0:1000)` elements (`phys/module_sf_sfclay.F:954-969`). Physics initialization selects `config_flags%sf_sfclay_physics` and calls it only in `CASE(SFCLAYSCHEME)` (`phys/module_physics_init.F:3316-3321`). `start_em` calls `phy_init(grid%id,config_flags,...)` inside its tile initialization loop (`dyn_em/start_em.F:1027-1034`). Artifact evidence is cross-object: `module_physics_init.o` has undefined `module_sf_sfclay_mp_sfclayinit_`; `module_sf_sfclay.o` defines it at `0x25fc0`. Therefore a one-argument wrapper can call the real initializer and publish readiness.

Use states `(UNINITIALIZED | READY) -> INITIALIZING -> READY` (or `POISONED` on reentry/invariant failure). Store `INITIALIZING` before invoking the real initializer; call it exactly once for every native invocation; publish `READY` with release semantics only after it returns. SFCLAY loads READY with acquire semantics. This is sufficient only under the admitted single-owner/no-concurrent-OpenMP contract. A general multithreaded mode would need a reader/writer lease, because another initializer could begin after a one-time READY check. Multiple sequential grid initializations are allowed; they rewrite the same tables and republish READY.

## Attestation and falsifiers

Pin at least these frozen hashes: `solve_em.f90` `e3462d09aa7da52ec798838b3d097dfd5209a8c9cdaa2d380629189d2201c554` and object `90b640673916ec8f5f4d23025ca81011fc6e33b47969ca71590fe800a191803a`; `module_first_rk_step_part1.f90` `4668213aaec35d72f7bdbd2473d810b4538370f8aadf5979ef940f315be9c43b` and object `544e29d35ca7e7971597849d0a845e0cfe16ab65ecaaf54bb9a54d69ff2b77f8`; `module_surface_driver.f90` `2101f00b02f4ad509d337dfb21eb63887bf57b66946d27900f50c215a5351aad` and object `a3996ce6b919c447c79535ebbdcaa099366cb6c0c6bb69458ac5e50ac85647bb`; `module_physics_init.F` `df49fe763e3458f5c01a6e29504fd77bb2419d267527a89092ae9623fb664ff1` and object `1ff2e3213c1ac501306980d17f40b1897eb2691ab090f36c73b73c56147f5b60`; `module_sf_sfclay.o` `53cb9c483995fe5fa799526735ca5006633b7cf73f1275c1bf5fe1a3add5831c`; getter objects `550d56fcf8b03e306b95f726054309ab51e0a2b95ab68445ad35a1b89f76fd8d` and `0263a3b100eb5f39e61bacc3563e853469fe7aef143eff5618e6216acdfe6529`. The frozen profile directories are not Git worktrees, so object/source/build-command hashes are the usable native attestation.

Falsify admission on any: scheme constant/value mismatch; fractional sea ice other than zero; invalid grid id; getter/config snapshot mismatch; absent, nested, or stale TLS epoch; pointer not covered by the correct source-proven class or live span; overflow/invalid bounds; initializer not READY; initializer concurrent with a reader; active multi-thread OpenMP; object/source/compiler/module-manifest mismatch; or any post-invoke validation failure (fatal, never fallback/re-execute).
