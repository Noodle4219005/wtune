# PLOV — physics-level in-place load rebalancing for WRF, no source change

PLOV is the part of wtune that actually moves wall time. It rebalances the two most
imbalanced physics drivers (microphysics, radiation) **inside** each time step, without
patching a single WRF source file: the drivers are interposed at link time with
`-Wl,--wrap=`, the wrapper stages the last few j-rows of the owner's patch into a
`/dev/shm` slot, and idle ranks on the same node steal those rows. Output is
bit-identical to unmodified WRF (`diffwrf` empty on every run, 224 ranks).

Measured on f1/NCHC, 4 nodes x 56 ranks x 2 threads, intel-adios2-dmsm tree,
1500x1500 CONUS-type case (2026-09-10/11):

| what | s/step | delta | bit-identical | job |
|---|---:|---:|---|---|
| A: unmodified placement (block), nothing staged | 0.3349 | — | (reference) | 1070876 |
| `srun --distribution=cyclic` only | 0.2947 vs 0.3013 | -2.2% | yes | (cyclic note) |
| `KMP_BLOCKTIME=0` only (224x2, ABAB) | 0.3188 vs 0.3280 | -2.8% | yes | (blocktime note) |
| **Ec: cyclic + micro staging rows=4 + adaptive node-local stealing (adapt75)** | **0.2913** | **-13.0%** (400 steps) | yes | 1070876 |
| ER: Ec + radiation staging rows=4 + row stealing | 0.2925 (all steps); rad steps -0.15..-0.23 s each, sum -1.06 s / 6 | -0.6% net | yes | 1070876 |

The gain grows as the storm develops (per-100-step: -10.2%, -12.4%, -12.7%, -12.0%),
the hottest rank's microphysics drops 30.7 s -> 16.9 s over 400 steps, and the feedback
controller does not drift. ER is measured and safe but its adoption is still the user's
call (radiation is 1/60 of the steps, so the net is small).

## Files (synced from `/home/u5727520/wrf-stack/wtimer/`, 2026-09-11)

| file | role |
|---|---|
| `wtimer.S`, `wt_call.S` | `__wrap_*` trampolines: capture the 402/422 driver arguments into `wt_argv` and re-enter the real driver with a shrunk `JME` (pass 1) and on the staged rows (pass 2) |
| `wtimer.c` | per-rank timers, `WTIMER_DIR` CSV writer, `WTIMER_CPU`/`WTIMER_PH` thread-map hooks (legacy) |
| `wt_micro.c` | microphysics stager: row staging into `/dev/shm`, node-local work stealing (CAS PUBLISHED->CLAIMED, single writer), adaptive R feedback `R += 0.5*NJ*(t_me - t_mean)/t_me` |
| `wt_rad.c` | radiation stager: same scheme; the argument whitelist is audited per Registry *package* because WRF passes size-1 dummies for inactive package fields (pointer/PRESENT checks are useless) |
| `wt_argmap.h`, `wt_argmap_rad.h`, `gen_argmap.py` | generated argument maps (index -> name, rank, staged?) for the two drivers; regenerate with `gen_argmap.py` when the WRF build or namelist packages change |
| `wt_alloc.c` | `for_allocate_handle` wrap so pass-2 temporaries land in the shm slot |
| `wtmpi.c` | PMPI shims (rank/node discovery, node-local barrier) |
| `wtlewi.c` | DLB/LeWI core-lending experiment — linked for symbol compatibility, **deprecated** (closed: no gain over KMP_BLOCKTIME=0) |
| `build.sbatch` | relinks `wrf_wtimer.exe` against `libwrflib.a` with the `--wrap` list; `TREE=` selects the WRF tree |
| `gate0.sh` | G0 dispatch gate: proves via `objdump` that the call sites really go through `__wrap_*` (two earlier versions returned a clean-looking 0 for "could not read") |
| `stage.sbatch`, `stage_rad.sbatch`, `stage_r3.sbatch` | A/B harnesses used for the measurements above; `stage_r3.sbatch` is the 400-step A / Ec / ER production check |
| `production.env` | the exact environment of Ec (and optionally ER) |

Every `.c/.S/.h` here is a byte copy of the wtimer working tree at sync time (sha256
prefixes: wt_rad.c `8691a4165e81`, wt_micro.c `c385e86430bd`, wtimer.c `7add64d9aa0e`).
`build.sbatch`/`stage*.sbatch` still point `T=` at `/home/u5727520/wrf-stack/wtimer`
and `W=` at the adios2 tree; change both when building from this copy.

## How to run it

1. Build once per WRF tree: `sbatch build.sbatch` (1 node, ~5 min), then
   `bash gate0.sh <tree>/main/wrf_wtimer.exe` must report every driver call site wrapped
   and 0 unwrapped. **Always check the exe mtime is newer than every `.c`** — a stale
   binary has produced a convincing null result before.
2. Source `production.env`, run with `srun --distribution=cyclic -n 224 --ntasks-per-node=56 -c 2`.
3. Verify: `diffwrf` between the PLOV run and an unmodified run must list no differing
   fields; per-rank `wtimer.<rank>.csv` footers report staged/refused/share_out/share_in
   and wait_us so a mechanism that silently did nothing is visible (staged == 0 is a FAIL,
   not a fast run).

## What it does not do (measured, closed)

- Cross-node stealing: the remaining imbalance is between nodes (node means 0.97-1.26),
  but cross-node radiation stealing is bounded at <=1.3% and micro is at its arithmetic
  ceiling (hottest-node micro equalised to 1% by LPT placement changed s/step by 0).
- The MPI floor is 37 ms/step (13 halo closures x 2.8 ms) and is source-locked.
- Rows > 4 lose: each extra driver entry costs 13-16 ms, which is why rows=8/31 were
  slower than rows=4 in the radiation sweep.
- Rad-step timing noise across jobs is +-0.2 s; only same-job arms are comparable.

See the wiki notes under `knowledge/wrf/2026-09-1*-plov-*` and
`knowledge/decisions/0046-why-rebalancing-fails-on-wrf-six-structural-reasons.md`.
