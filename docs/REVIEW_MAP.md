# WTune 審查地圖

這份文件告訴審查者：**哪些程式要逐行讀、哪些可以快速掃過、哪些可以交給別人**。
以 `wtune submit -O2`（project init → doctor/check → tune → run → 下載觀測 → post）這條標準流程為主。

## 這份地圖怎麼來的

1. 在 scc 上用 SC26-P1 跑一次完整的 `wtune submit -O2`，另加 `doctor`、`check`、`advise`、`calc`、`watch`，
   記錄每一個被呼叫到的 Python 函式（commit 8bdcd3c）。
2. 被呼叫到的函式逐一分成 A、B 兩類（三位審查代理分頭讀碼，主控彙整）。
3. 沒被呼叫到、但屬於同一條路徑的錯誤處理（續跑、CFL 救援、中斷、量測失敗）人工補回 A。

| 層 | 意義 | 審法 | 規模 |
|---|---|---|---:|
| **A 決策邏輯** | 會改變「WRF 實際怎麼跑」或「WTune 報出什麼數字／選哪個版面」 | 逐行 | 約 14,600 行 |
| **B 檔案處理與流程** | 建目錄、連結、轉檔、參數解析、列印、送 job、輪詢 | 快速掃過 | 約 8,800 行 |
| **C 選用模組** | 這次沒進入、但屬於功能（颱風／對流／climate 圖組、`-O3` 提案器…） | 可交給第二位審查者 | 見文末 |
| **歸檔** | 沒有任何指令會走到 | 不審 | `archive/unused-2026-10/` |

A 層分三塊：調校與量測約 4,850 行、執行與專案約 4,990 行、後處理約 4,770 行。
其中**真正會改變 WRF 怎麼跑**的（`wtune_run/`、續跑、sbatch、成功判定、submit、project）只有約 1,900 行，建議從那裡和第 1 部分開始。

追蹤量測的限制：
- 只記 Python。三支 shell（`campaign/strict_campaign.sbatch`、`node_hygiene.sh`、`sites/scc.sh`）是人工讀的。
- 量測之後才加的函式不在紀錄裡（例如 `extract.accumulated_rain`、`wtune_run/controller._run_segments`），已人工補上。
- 乾淨跑完一次不會經過錯誤分支；這些分支列在各檔的「沒執行到但屬於這條路徑」。
- 函式名稱是穩定的定位方式（`grep -n "def <name>"`）；行數是寫這份文件時的大小，會隨修改變動。

---

## 第 1 部分：調校與量測（s/step 怎麼來、贏家怎麼選）

閱讀順序即流程順序：probe job 量測 → controller 解析 → 選節點數與版面 → 預測整段時間 → 報告。

### 1.1 量測在哪裡產生

| 檔案 | A：逐行讀 | 它決定什麼 |
|---|---|---|
| `campaign/strict_campaign.sbatch` | 啟動與收尾那一段（`timeout … mpirun`、取 `SUCCESS COMPLETE WRF`、抓 `Timing for main` 的 d01 行、`step-stats`） | **s/step 唯一的出生地**：RC=0 且成功才算；否則中位數留空 |
| 同上 | `run_node_hygiene`、`write_interrupted_row` 與 trap、`check_disk_guard`、`configure_omp_env`、rep 順序（正反交替）、`ntasks_per_node` | 清理沒跑成 → `node_dirty=unknown`；被中斷的 run 不給 s/step；`OMP_PLACES/PROC_BIND` 只在 omp>1 時設 |
| `campaign/run_reports.py` | `step_stats` | 丟掉第 1 步；大於 3 倍中位數的步當輻射步排除；其餘取中位數 |
| `campaign/sites/scc.sh` | `site_launch`、`site_env`、`CORES_PER_NODE=72` | `mpirun --map-by ppr:<tpn>:node:PE=<omp> --bind-to core`；rank 數除不盡節點數時 tpn 進位 |
| `campaign/node_hygiene.sh` | `reap`、`census`、`foreign_load` | 殺自己殘留的 `wrf.exe`；CPU 忙碌 >20% 記 `NODE_DIRTY`；別人的程序 pcpu>20 記 `FOREIGN_LOAD` |
| `campaign/namelist_edit.py` | `edit_namelist`、`_set_decomposition`、`_set_run_length`、`_history_off` 一族、`_apply_case_overrides` | probe 實際跑的 namelist：nproc_x/y、numtiles、跑幾步、量測時不寫 history |
| `campaign/namelist_adios2.py`、`namelist_streams.py` | `apply_adios2`、`_force_io_forms`、`_write_adios2_group`、`adios2_input_streams`、`per_record_streams` | io_form 14、`numaggregators = ranks`、哪些輸入讀 BP |

B（掃過）：`strict_campaign.sbatch` 的參數解析、連結檔案、BP 轉檔、ledger 寫出；`nc2bp.sbatch`；`configs.py`、`bp_inputs.py`、`case_tables.py`、`namelist_files.py`、`namelist_text.py`。
`MODE=physics`、`MODE=correctness`、沒有 `CONFIG_FILE` 的固定 C01-C16 分支已於 2026-10-10 移出（原檔 `archive/unused-2026-10/campaign/strict_campaign_full.sbatch`）；腳本剩 1213 行，只收 `MODE=timing`。window-frame 區塊仍在（-O3 與第二階段的 frame 會用）。
沒執行到但屬於這條路徑：共用 spin-up restart、restart 編輯失敗的處理、`RESTART_CHECK_FAILED`。

### 1.2 候選版面從哪裡來

| 檔案 | A | 它決定什麼 |
|---|---|---|
| `src/predictor/decomp_space.py` | `legal_shapes_exact`、`rsl_patch_extents`、`enumerate_space`、`default_omp_options`、`numtiles_candidates`、`sample_design`、tile 幾何一族 | 合法切法（每個 domain 每個 patch ≥ 10 格）、候選空間、分層抽樣 |
| `src/predictor/experiment.py` | `_max_legal_ranks`、`_generated_baseline` | ANCHOR：omp=1、最大合法 rank 數、長寬比最小 |
| `src/tools/scc_sweep.py` | `estimate_s_per_step`、`build_anchor`、`build_sampled`、`build_configs`、`_identity`、`excluded_nodes`／`exclude_arg`、`build_sbatch_command`、檔頭常數 | 候選去重、`--exclude` 清單（head + `WTUNE_SBATCH_EXCLUDE` + 預檢失敗節點）、時限估計常數 |
| `src/wtune_tune/candidates.py` | `aggregate_by_cid`、`_sweep_anchor`、`_hybrid_anchor`、`_sweep_omp_options`、`_apply_search_filter` | 失敗或 ≤0 的量測丟掉；各 rep 取中位數；每個 N 的 ANCHOR |

### 1.3 節點數與贏家

| 檔案 | A | 它決定什麼 |
|---|---|---|
| `src/wtune_nodes.py`（全檔 178 行） | `choose`、`stop_early`、`summarize`、`finalists`、`seed_nodes`、`refine_targets`、`choose_cores` | **3% 雜訊規則**：取「在最小值 3% 內的最小 N」；core 數平手取較滿的 |
| `src/wtune_tune/node_search.py` | `run_node_search`、`run_node_sweep`、`_measure_anchor_wave`、`run_core_sweep`、`_tune_finalists`、`build_layout_table` | 掃哪些 N、何時提早停、最後的 (N, 每節點核數, 版面) |
| `src/wtune_tune/rounds.py` | `run_round0`、`_round0_size`、`_configs_for_budget`、`run_tuning_at` | finalist 上量幾個候選 |
| `src/wtune_tune/common.py` | `_cluster_rank`、`would_exceed_budget` | 0.5% 內視為同一群 |
| `src/wtune_tune/report.py` | `build_report`、`final_winner`、`local_param_table`、`_assert_report_is_strict_only` | BEST 與 `report["winner"]`（`wtune run` 讀的就是它） |
| `src/wtune_tune/budget.py`、`jobs.py`、`preflight.py` | `estimate_run_time`、`measure_startup_s`、`_wait_running_budget`、`wait_for_jobs`、`node_preflight` | 預算怎麼計、何時取消 job、哪些節點被排除 |

沒執行到但屬於這條路徑：`_proxy_fallback_report`（什麼都沒量到時）、`evaluate_job_gates`、`cancel_jobs`、`_in_flight`、`sample_more_diverse_configs`。

### 1.4 整段時間的預測

| 檔案 | A | 它決定什麼 |
|---|---|---|
| `src/wtune_tune/pipeline.py` | `run_pipeline_full`、`_pipeline_phase1_layout`、`_pipeline_on_sweep`、`_prediction_from_t0_anchor`、`_pipeline_prediction`、`_pipeline_finish`、`_pipeline_cold_estimate` | 等級怎麼降（模擬比調校短就降級）、預測來源 |
| `src/wtune_tune/prediction.py` | `WholeRunPrediction.add_run_length_io`、`time_suggestion_s`、`plan_prediction_points`、`cold_start_prediction` | T = 計算 + 啟動 + I/O；建議的 `--time` |
| `src/wtune_predict.py` | `build_prediction`、`predict_compute`、`predict_radiation_extra`、`measure_rad_from_rsl`、`parse_rsl_d01_step_seconds`、`resolve_probe`、`classify_steps`、`predict_init`、`get_d01_steps` | 預測值與區間；輻射步怎麼分類 |
| `src/wtune_io_model.py`、`wtune_runlength.py` | `run_length_io` 及其各項、`resolve_run_seconds` | history／restart 的 I/O 時間、iofields 的重複解析成本 |
| `src/wtune_probe_plan.py`、`wtune_cost_drivers.py` | `plan_points`、`plan_points_for_case`、`first_global_step_from_rsl`、`measurement_from_steps_file` | 量哪幾個模式時刻、區間寬度 |

B（掃過）：`wtune_predict.py` 的 namelist parser 與 ledger 讀取、`render_text`；`report.py` 的列印函式；`context.py`、`cli.py` 的參數組裝。

---

## 第 2 部分：執行（真正會改變 WRF 怎麼跑的部分，約 1,900 行）

建議照這個順序讀：

1. `src/wtune_submit.py`：`plan_steps`、`case_window`、`fetch_steps`、`main`
   — tune → run → 下載 → post 的指令；下載失敗繼續、其餘失敗即停。
2. `src/wtune_project.py`：`find_project`、`apply_project_env`、`merged_case_table`、`latest_tune_report`、
   `project_case_name`、`case_entry_from_namelist`、`register_project_case`、`init_project`、`_default_post_block`
   — 哪些節點被排除、case 表怎麼合併、用哪一份 tune 報告、case id 怎麼產生。
3. `src/wtune_run/layout.py`：`tune_report_layout`、`resolve_layout`、`make_config` — 實際執行的版面（命令列旗標蓋過報告）。
4. `src/wtune_run/plan.py`（全檔都是 A）：`plan_run`、`restart_setting`、`auto_checkpoint_steps`、`checkpoint_request`、
   `assign_segments`、`input_record_alignment` — 分幾段、restart／checkpoint 落在哪些時刻。
5. `src/wtune_run/estimate.py`：`time_limit_s`、`segment_wall`、`history_wall`、`restart_write_steps`
   與檔頭常數（`TIME_FACTOR=1.5`、`TIME_PAD_S=600`、`TIME_ROUND_S=300`）— 每個 job 的 `--time`。
6. `src/wtune_run/namelist.py`：`render_segment_namelist`、`_set_window_and_layout`、`_set_restart_keys`、`_set_io_keys`
   — 對 namelist 的全部改動，以及「不准改到白名單以外的 key」的檢查。
7. `src/wtune_run/staging.py`：`render_adios2_restart_xml`、`merge_adios2_xml`、`output_dir_names`、`restart_files`。
8. `src/wtune_run/sbatch.py`（全檔 70 行）：`_header_lines`、`_launch_lines`、`render_segment_sbatch`
   — `--exclude`、`--signal=B:USR1@90`、`--no-requeue`、launcher 與綁核。
   連同 `src/tools/coldstart_sample.py` 的 `render_signal_trap_and_hygiene_preamble`（trap、kill、reap、`INTERRUPTED`）。
9. `src/wtune_run/attempts.py`：`_run_attempt`（成功 = watch 判 done 且 restart 完整）、`wait_for_job`、`_submit_attempt`。
10. `src/wtune_run/controller.py`：`_run_segments`、`_next_attempt`、`_interrupted`、`_resume_attempt`、`resume_time_limit`、
    `rescue_time_step`、`_cfl_rescue_attempt`、`_update_from_observed`
    — 失敗怎麼分類、從哪個 restart 續跑。**`--cfl-rescue` 是整個 WTune 唯一會改變模擬解的地方。**
11. `src/wtune_resume.py`（全檔約 80 行，乾淨的 run 不會執行）：`classify_failure`、`resume_point`、
    `valid_restart_times`、`restart_complete`，以及 `INFRA_STATES`／`MODEL_NEEDLES` 常數。
12. `src/wtune_watch.py`：`snapshot`、`_set_status`、`read_rsl`、`parse_d01_steps`、`_domain_progress`、`_check_cfl`、`exit_code`
    — done／failed／running 的定義（`wtune run` 靠它判定成功）。其餘是剩餘時間估計。
13. `src/wtune_run/cli.py`：`apply_project_defaults`、`_load_case`、`_choose_layout`、`_tune_report_candidates`、`build_plan`、`execute`、`main`。
14. `src/wtune_bp.py`：`per_record_namelist_edits`、`adios2_xml`、`_entry_current`、`cached_inputs`（其餘 26 個函式是 B）。

B（掃過）：`wtune_run/inputs.py` 的連結與複製、`common.py`、`context.py`、`report.py`；`wtune_bp.py` 的轉檔、manifest、lock；`bin/wtune`。
`src/wtune_cli.py` 只需要看 `main`、`_run_forwarded`、`_project_for` 與指令表。

### 送出前的檢查與建議（不改變 WRF 怎麼跑，但決定 WTune 對使用者說什麼）

| 檔案 | A | 內容 |
|---|---|---|
| `src/wtune_preflight.py` | `run_checks`、`exit_code_for`、`check_c1…check_c11` 及其 helper | 什麼情況判 FATAL／WARN：時間窗、iofields、物理資料檔、輸入一致性、切法、dt/dx、磁碟、wrfinput 數值 |
| `src/wtune_extensions.py` | `detect`、`refusal_run`、`tune_gate`、`omp_cap`、`cost_factors` | Chem／Hydro／WRFDA／moving nest 的偵測、拒跑條件、成本係數 |
| `src/wtune_calc.py` | `check` 與 13 個 `_chk_*`、公式函式 | 經驗法則（dt、radt、nest 比例…） |
| `src/wtune_advise.py` | `advise`、`det_case_types`、`build_case` | 建議規則引擎 |
| `src/wtune_compliance.py` | `assert_strict_only`、`classify` | STRICT 以外的 namelist 變動一律拒絕 |
| `src/wtune_config.py`、`wtune_case.py`、`wtune_physics.py`、`wtune_doctor.py` | `resolve_config`、`candidate_allowed`、`compute_r_max`、`diagnose`、`machine_checks` | 設定檔驗證、case profile、環境檢查 |

---

## 第 3 部分：後處理（一行寫錯就會給出錯的科學結果的地方）

1. `src/post/extract.py`：`segment_dirs`、`file_times`、`_file_entries`、`frame_refs`、`list_frame_refs`
   — **每張圖的時刻都從這裡來**（分段合併、同一時刻後段蓋前段）。
   `cached_nc_path`、`bp_signature`、`_cache_valid`、`_same_files`、`_bp_to_nc` — 轉檔快取會不會用到舊檔。
   `_surface_field`、`accumulated_rain`、`extract_frame` — 單位、雨量（含 bucket）、等壓面內插。
2. `src/post/caseinfo.py`：`detect_types`、`detect_terrain`、`detect_tc`、`detect_cloud`、`run_length_hours` — 自動判定個案類型的全部門檻。
3. `src/post/timeseries.py`、`conventions.py`：`select_times`、`figures_for_time_series`、`time_label` — `--every` 選哪些時刻、圖上的時間標籤。
4. `src/post/recipes/precip.py`：`stage4_index`、`pick_window`、`_model_accum`、`_stage4_figures`、`station_verification`。
5. `src/post/met_gridstat.py` 與 `src/post/met/*.conf.tmpl`：`run_verification`、`curvilinear_to_latlon`、`write_interior_polygon`、`coverage_polygon`、`parse_cts_table` — 每個驗證分數的網格、遮罩、門檻。
6. `src/post/met_pointstat.py`：`parse_isd_lite`、`build_obs_text`、`model_fields`、`verify_run`、`summarize` — 測站配對、單位換算、時間窗。
7. `src/post/recipes/synoptic.py`、`era5.py`：`_post_spinup_time`、`_read_shared_fields`、`cyclonic_avo`、各 `_*_chart`、`z500_at`。
8. `src/post/ivt.py`、`condensate.py`、`recipes/moisture.py`、`recipes/terrain.py`、`recipes/cloud.py`、`recipes/xsection.py`、`cf_pressure.py`
   — 積分、剖面幾何、地面以下的處理、垂直內插。
9. `src/post/compare.py`（這次沒進入，但屬於你要審的）：逐位元比對工具，驗收靠它。
10. `src/post/sources.py`：`download`、`http_get`、`resolve_window`；`src/post/fetch/{era5,stage4,stations}.py` 的 `run` — 下載完整性、抓哪個時間窗。
11. `src/post/recipe.py`：`guarded`、`_run_recipe`、`_type_figures` — 哪些錯誤變成「缺」。

B（掃過）：`report.py`、`tools.py`、`basemap.py`、`plotting.py` 的繪圖、色階、標籤；`cli.py` 的參數；`vapor_bundle.py`、`paraview3d.py`、`xwrf_convert.py` 的 3D 輸出（只影響視覺）。

---

## C 層：可以交給第二位審查者的選用模組

| 模組 | 行數 | 什麼時候才會用到 |
|---|---:|---|
| `src/post/tc.py`、`recipes/tc_structure.py`、`recipes/tc_extra.py`、`tc_metrics.py`、`met_tc.py`、`atcf.py` | 約 2,600 | 颱風個案 |
| `src/post/recipes/convection.py`、`radar_composite.py`、`fetch/mrms.py`、`fetch/nexrad.py` | 約 1,000 | 對流個案、雷達比對 |
| `src/post/recipes/climate.py`、`fetch/era5_monthly.py`、`fetch/gpcp_monthly.py` | 約 860 | 長積分 |
| `src/post/recipes/pbl.py`、`sounding.py`、`fetch/igra.py` | 約 1,160 | 邊界層、探空 |
| `src/post/recipes/ensemble.py`、`recipes/chem.py` | 約 290 | ensemble、WRF-Chem |
| `src/predictor/decomp_search.py`、`models.py`、`features.py`、`ledger.py`、`targets.py`、`benchmark.py` | 2,977 | `-O3` 的 CatBoost 提案器 |
| `src/wtune_tune/points.py`、`physics_phase.py`、`src/tools/case_rebuild.py`、`coldstart_sample.py` 其餘部分 | 約 6,500 | 重建量測點、physics 建議 |
| `src/wtune_geog.py`、`wtune_calibrate.py`、`src/scenario/`、`src/analysis/`、`src/tools/metdata.py` | 約 6,500 | `prep`、`calibrate`、`scenario`、`analyze`、`metdata` 指令 |

---

## 審的時候請特別留意（已知、尚未處理）

1. **`-O2` 不會拒絕髒節點上的量測。** probe job 把 `node_dirty`／`foreign_load` 寫進 ledger，
   但 t0 搜尋與 node sweep 都不檢查這兩個欄位（原本只有已退役的 predicted 模式會看）。
   節點上有別人的程序在搶核時，量測照樣被採用。自己殘留的 `wrf.exe` 會在 job 開頭被清掉。
2. **報告印的 launcher 行可能和實際不同。** `report.local_param_table` 用 `ranks // nodes`，`site_launch` 是進位。
3. **兩個雜訊門檻。** 選 N／核數／贏家用 3%（`wtune_nodes.NOISE_FRAC`）；`_cluster_rank` 用 0.5%（`predictor/targets.NOISE`）。
4. **輻射步有兩種定義。** `step_stats` 用「大於 3 倍中位數」；`measure_rad_from_rsl` 用輻射排程的相位。
5. **probe 步數有兩個預設。** `context.DEFAULT_STEPS=20` 與 auto 的 `max(30, stepra+10)`。
6. **等級會自動降。** 給了 `--minutes` 之後，模擬比調校短就從 O3 降到 O2 再降到 O1；降級後選出的版面可能不同（SC26-P1：O1 選 N=4，O2 選 N=5）。
7. **IVT 的濕度變數。** `recipes/moisture._ivt_figure` 傳入 `QVAPOR`（混合比），`ivt.py` 的說明寫比濕；兩者差約 1–2%，待確認。
8. **下載的既有檔案。** `sources.download` 對「已存在、但沒有記錄 sha」的檔案只檢查大小 > 0。
