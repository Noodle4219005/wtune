# experiments/ — the harnesses that produced WTune's conclusions

These ran outside the repo, under `/home/u5727520/wrf-bench/wscore/`, and are vendored here because
that tree is scratch space: a run directory gets `rm -rf`'d and reused, and an external script this
repo depended on has already vanished once — `scripts/vendor/wrf-adios2-to-netcdf4/convert.py`
disappeared and every MODE=physics job silently produced no netCDF and therefore no fidelity at all
(smoke jobs 1076771–1076776). WTune has to be runnable from its own checkout, so the harness that
generated a conclusion lives next to the code that acts on it.

Copied 2026-09-17 with their sha256 prefixes recorded, so a later divergence is visible:

| file | sha256 (first 16) | what it answers |
|---|---|---|
| `rk_alternation.sbatch` | `5fc13eea3ad48712` | Six arms in one allocation: does alternating `rk_ord` across restart boundaries land between pure RK3 and pure RK2 in cost AND drift? Carries the bit-identity control (a restart chain must reproduce a single call exactly) and the chaos-floor twin. |
| `segnl.py` | `8f9bdac94bbb4636` | The segment namelist pass (clock, `restart`, `restart_interval`, `rk_ord`, optional `--time-step`) with read-back verification. List-valued keys are chunk-parsed — a regex that stopped at the first comma silently put a nest on the wrong date. |
| `bitcmp.py` | `0ba7d30271968af2` | Exact, no-tolerance netCDF comparison. PASS/FAIL over every variable. |
| `summarize.py` | `a87cad9e193dc20d` | The arm table for `rk_alternation.sbatch`. |
| `dt_tradeoff.sbatch` | `97f8cb804f0b5c1a` | The RK order × `time_step` Pareto: is there an operating point that is both cheaper and closer to a converged reference? |
| `dt_summarize.py` | `03647efd7144eedc` | Fits `solver = a + b·steps` and the per-field error exponent `p` in `error ∝ dt^p`. |
| `bitrepro.sbatch` | `c6c1039b9a1ef125` | Nine arms off one base, each differing in exactly ONE decomposition dimension, to find which one breaks bit-for-bit reproducibility. Includes a determinism control (same config twice). |
| `bitrms.py` | `87bdc30c2723bfcf` | Per-field RMS between two runs, for the arms that are not bit-identical. |
| `summarize_bitrepro.py` | `96e9647a1f2953f1` | The arm table for `bitrepro.sbatch`. |

## The results these produced (full write-ups in the wiki's experiments notes)

- **Only `nproc_x` breaks bit-identity** (job 1081872). `nproc_y`, `numtiles`, OpenMP threads, MPI rank
  count and rank placement are all numerically neutral, because WRF's default tiling splits in J only,
  so `nproc_x` is the sole knob that changes the innermost loop trip count.
  Consequence for `fidelity_gate.py`: **a twin that does not change `nproc_x` has a chaos floor of
  exactly zero**, and `WITHIN-FLOOR` then degenerates into "bit-identical to the reference".
  `rk_alternation.sbatch` has this defect for karina (`dNX=8` and `dTNX=8`); conus12's pair is fine.
- **Segmented `rk_ord` alternation is closed** (job 1081239, karina 24 h): the alternating arm's drift
  was 99.9% of pure RK2's while it kept only half the speed-up. Error does not heal, and once the
  chaotic divergence saturates there is nothing left to recover.
- **RK2 with a shorter `time_step` buys accuracy at equal cost, never speed** (job 1081883, conus12):
  RK2 @ dt=60 matches RK3 @ dt=72 in cost within the ±0.3% timing noise and lands 15% closer to a
  converged reference. Measured RK2/RK3 per-step cost is **0.892**, not the 2/3 a stage count suggests —
  RK3 runs `1 + nss/2 + nss` acoustic sub-steps against RK2's `nss/2 + nss` (7 vs 6 at nss=4).
  The error is not purely second order: T/U/V scale as dt² but W and QVAPOR scale as dt¹, the
  signature of the first-order physics/dynamics split, and W dominates the headline metric.

## Running them

Each script documents its env surface in its own header. They expect the campaign's pixi env
(`/home/u5727520/wrf-stack/wtune-predictor-env`) and write under
`/home/u5727520/wrf-bench/wscore/<experiment>/`; that output root is the one path still assumed, and
it is scratch by design — the evidence worth keeping is copied into `data/wtune/` in the wiki.
