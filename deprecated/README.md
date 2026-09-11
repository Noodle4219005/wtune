# deprecated/ — optimizations that were measured and found null, or superseded

Nothing here is wired into `wtune.py` any more except `restart-window`, which lazy-imports
from this directory and prints a deprecation warning. Everything stays runnable
(`PYTHONPATH=..:. python deprecated/<script>.py`, tests via `pytest deprecated/`) so the
numbers in the wiki can be reproduced; nothing here should be extended.

Moved on 2026-09-11. Reasons, with the wiki note that closed each line
(`knowledge/...` in `my-llm-wiki`):

## Restart-boundary migration (the "transition controller" family)

`wtune_migration.py`, `wtune_block_migration.py`, `wtune_runtime_migration.py`,
`wtune_migration_config.py`, `wtune_dynamic_provider.py`, `wtune_restart.py` and their
tests `test_wtune_{migration,block_migration,runtime_migration,migration_config,dynamic_provider,restart,restart_cli}.py`.

Closed: a WRF restart write+read costs ~56 s on the 4-node case while the imbalance a
re-decomposition could recover is 7-17 s per window, so the transition can never pay for
itself within the 4-node cap. The consumer-transition card was also blocked at the staging
permission step before Slurm (handoff `2026-09-01-wtune-restart-consumer-transition`).
PLOV rebalances in place instead, every step, at zero restart cost.
See `knowledge/decisions/0046-why-rebalancing-fails-on-wrf-six-structural-reasons.md`.

## Rank placement / thread maps / core lending

`wtune_placement.py`, `wtune_place.py`, `wtune_rankfile.py`, `make_thread_map.py`,
`segment_driver.py`, `segment_length_sweep.py`, `shims/blockbar6.c`, `shims/blockbar7.c`
(and `plov/wtlewi.c`, still linked for symbol compatibility).

Closed: profile-guided LPT placement equalises node microphysics to 1% but s/step does not
move (`knowledge/wrf/2026-09-11-profile-guided-lpt-placement-equalises-node-micro-to-1-percent-but-s-step-is-unchanged...`);
x-pair placement and the block-barrier shims were null; DLB/LeWI core lending gave
nothing beyond `KMP_BLOCKTIME=0` alone (-2.8%), which is now just an env line in
`plov/production.env`. The one placement result that survived — `srun --distribution=cyclic`
(-2.2%) — needs no code, so it lives in `plov/production.env`.

## Per-rank hotspot model and its studies

`wtune_hotspot.py`, `wtune_models.py`, `wtune_accuracy.py`, `wtune_arch.py`,
`scheme_head.py`, `scheme_head_mp.py`, `stencil_scale.py`, `geo_portability.py`,
`wtune_crosscase_blk.py`, `wtune_crosscase_train.py`, `hotspot_improve.py`,
`hotspot_regime_residual.py`, `closed_loop.py`, `wtune_segshare.py`.

Deprecated by decision 0045 (2026-09-07): accuracy was fine (46.9% cross-case hit rate vs
10% chance) but no actuator could act on "which rank is hot". A 2026-09-08 banner
withdrew that deprecation on the argument that restart-boundary rebalancing would be the
consumer; that consumer is now closed (above), and PLOV's feedback controller measures
the per-call time it needs directly, so the prediction has no consumer again. The
banners are kept in the files as history, with a superseding note on top.
The *architecture* (recomputable lookup + GBRT depth-3 residual) survives in
`../wtune_cost_model.py`, which stays live; the shared libraries
`wtune_oracle.py`, `wtune_oracle_train.py`, `wtune_segment_train.py` also stay live
because the runtime predictor imports them.

## Running the tests here

```
cd wtune && PYTHONPATH=.:deprecated python -m pytest deprecated/ -q
```
`deprecated/conftest.py` puts both directories on `sys.path` for pytest.
