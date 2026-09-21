# KF reviewed-G0 prospective claim matrix — 2026-09-13

This is input to a human scientific review. It is not the `wtune_kf_reviewed_g0_attestation`
record and cannot activate `reviewed-g0-v1`. The prospective pinned scientific closure is
`reviewed-g0-v1:ab0b10b6f0e36a18ef2ff0ce3eacbafcfc36392fa9d17d0e738370e3793c3414`.
The exact 50-source, 43-module, 10-baseline-input, and 21-token ledgers are in
`prospective-inputs.json`.

| Claim | State | Reviewable evidence and conclusion boundary |
|---|---|---|
| `source_abi` | reviewable after human review | `kf_cps_schema.json` records 79 ordered intrinsic by-reference arguments at frozen source SHA `9533f860…`; `module_cu_kfeta.f90:29-161` has explicit-shape arrays and no `CHARACTER`, `VALUE`, or derived-type dummy. This does not assert native WRF equivalence. |
| `caller_chain` | reviewable after human review | `native-attestation-raw.json` plus frozen `solve_em.f90:275-281,442,949`, `module_first_rk_step_part1.f90:700`, `module_cumulus_driver.f90:1533`, `module_domain.F:243-282`, and `model_data_order.inc:7` establish the bounded production chain and the four automatic-actual bounds. The explicit input scan finds one CPS undefined reference, in `module_cumulus_driver.o`. |
| `baseline_closure` | reviewable after human review | The r3 link manifest has ten pre-existing direct `.o/.a` inputs. The prior audit shows six loose objects byte-match their unique `libwrflib.a` members and pins the archive at SHA `af01a793…`. Future generated objects remain measured outputs. |
| `compiler_module_abi` | **pending native preproof** | r3 successfully consumed the same module directories with the same wrapper and flags, and recorded Intel 2024.0.2. The reviewed compiler lane hashes only the MPI wrapper scripts. Those wrappers select bare `icx`/`ifx` through `PATH` and accept override variables, so the underlying compiler ELF/config files are not presently rehashed at every reviewed boundary. |
| `library_closure` | **pending native preproof** | The helper resolves every ordered direct token, including `/usr/lib64` selections for `m/z/rt`, and pins first candidates plus shadows. This does not identify archive-member selection, compiler-injected libraries, GNU `libm.so` script inputs, or the future candidate's recursive `DT_NEEDED` closure. No linker `-t` trace exists for the 25-object node-local set. |
| `node_init_static` | reviewable after human review | Frozen source lines 18-22 and 3171-3303 identify and initialize seven saved LUT objects. `intel-object-nm.txt:227-244` establishes their Intel symbol sizes. `wt_kf_init_bridge.c:43-77` hashes all seven after the real 35-argument init under a reader/writer lease. Reviewed preparation now emits `WT_KF_BUILD_INIT_TABLES_VERIFIED=1` for that unit only. |
| `concurrency` | `not_verified` by design | `lastdt` is output-dead in the audited path, but Fortran I/O and all native side effects are not proven thread-safe. `WT_KF_BUILD_CONCURRENT_CPS_VERIFIED=0`; `wt_kf_node_live.c:55-64,324` serializes helper, head, owner, and fallback native CPS entries. |

The helper contract cannot emit a real review record while either pending claim remains. Hashing
these files preserves reviewed inputs; it does not turn the pending facts into scientific truth.

## Exact next operation: native preproof only

Use `native-preproof.sbatch`. It requires fresh proof-disabled node-local objects, rather than the
existing r3 outputs: r3 has 18 objects and omits `wt_kf_batch`, `wt_kf_batch_codec`,
`wt_node_local`, `wt_kf_node`, `wt_kf_node_live`, `wt_kf_node_shm`, and `wt_kf_init_bridge`.
The fresh `unattested-v1` recipe produces the complete 25-object link surface while keeping every
scientific proof macro off. The preproof candidate must never be used for WRF.

Execution card:

- Target: one new root,
  `/home/u5727520/wrf-stack/wtune-kf-reviewed-recipe-20260913-r1/preproof`; the script refuses an
  existing path. Frozen WRF, r3, and the final reviewed `recipe/` stay read-only.
- Resource: account `ACD110018`, partition `ct112`, 1 node, 1 task, 4 CPUs, 4 GiB,
  10-minute limit.
- Cost: `1 job x 1 node x 0.167 h = 0.167 node-hours = 3.74`, wall-clock at most 10 minutes,
  disk peak about 0.3 GiB.
- Submission command after current-turn user consent:
  `sbatch --parsable /home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/native-review/native-preproof.sbatch`.
- Step 1 pins and records MPI wrapper, resolved `icx`/`ifx`, empty compiler config, version,
  wrapper `-show`, and an explicit allowlist of build variables after compiler overrides are cleared.
- Step 2 generates an exact `unattested-v1` config from `prospective-inputs.json`, then the existing
  one-shot executor compiles 25 objects and prepares the link manifest.
- Step 3 reconstructs that manifest exactly, changes only the separate proof output path, and adds
  GNU linker trace/map options. It performs one native link; it does not launch WRF.
- Step 4 extracts and hashes the candidate's `PT_INTERP`, invokes that loader with `--list` under the
  recorded G0 `LD_LIBRARY_PATH` without entering WRF, and makes `verify-native-preproof.py` check
  exact argv, executor-recorded object hashes, all direct first-candidate resolutions in the linker
  trace, the baseline `DT_NEEDED` set, and the loader-selected recursive `DT_NEEDED` closure.
- Terminal evidence: `preproof/raw/native-closure.json`, `link-trace.stdout`,
  `wrf-wtune-unattested.map`, compiler identity files, and scheduler output. Stop on the first
  failure. No retry, WRF run, account query, or follow-on build belongs to this card.
- Pass: compiler/config hashes remain fixed; every one of the 25 object hashes matches the executor
  ledger; traced direct selections equal the prospective ordered contract; the candidate retains
  every baseline `DT_NEEDED` name; and every recursively encountered `DT_NEEDED` name has one
  unambiguous loader-selected canonical path and SHA-256. Fail on any mismatch, missing selection,
  conflicting loader mapping, nonempty link stderr, or changed pinned input. This covers static
  link inputs and startup `DT_NEEDED`; it does not inspect libraries opened later with `dlopen`.

Passing the preproof supplies facts for human review of the two pending claims. It does not itself
activate reviewed-G0, validate native ABI semantics, or validate WRF outputs.
