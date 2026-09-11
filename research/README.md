# research/ — one-off studies that answered a question and produced a wiki note

These are analysis scripts, not tools: each one reads existing run artifacts
(per-rank `wtimer.*.csv`, wrfout frames, the harvested cost table) and answers one
question. They are kept so the number in the corresponding wiki note can be
regenerated. None is imported by `wtune.py`; nothing here is a production path.

Run with `PYTHONPATH=..:. python research/<script>.py --help` (they import the live
libraries `wtune_oracle_train`, `wtune_segment_train`, and each other).

| script | question it settled |
|---|---|
| `imbalance_census.py` | which cases rebalancing could ever help (two measured axes, both required) |
| `ceiling_census.py` | the imbalance ceiling per case — the upper bound any rebalancer can reach |
| `hotspot_locality.py` | where hot ranks physically live; concentrate or spread |
| `pairing_sim.py` | zero-job gate for over-decomposition + core lending (put the ceiling at 12-17% on high-M cases; the actuators then lost it in overhead) |
| `repair_value.py` | is a pairing from the previous segment better than one chosen at t=0 |
| `probe_length.py` | how long a run must be before the per-rank cost pattern is known |
| `repeat_check.py` | is the per-rank cost pattern reproducible at all (ceiling on any predictor) |
| `sample_decay.py`, `decay_gate.py` | how fast a per-rank measurement goes stale; measure-vs-model decision at a segment boundary |
| `rebill_time.py` | per-rank cost re-billed by time instead of call index |
| `segment_cost_predict.py`, `residual_drift.py`, `drift_is_case_or_config.py` | simulate-a-little-predict-a-lot; whether drift is a case or a config property |
| `harvest_config_cost.py` | harvest cost(case, config) from runs already on disk (feeds `../wtune_cost_model.py`) |
| `cost_model_v2.py`, `cost_v2_diagnose.py`, `cost_looco.py`, `cost_separability.py` | cost(case, config) model studies; per-PBL head REFUTED on a diverse corpus (2026-09-05), separability holds |
| `saturation.py` | how many cases per regime the model needs |
| `qke_check.py` | does MYNN's prognostic turbulence close the hotspot gap (within-case) |
| `wtune_phase_analysis.py`, `wtune_replay_sim.py` | settlement-tax poolability and the "19pp cross-phase slack pooling" replay |
