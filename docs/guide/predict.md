# 提交前檢查與時間預測

```bash
wtune check case/ --ranks 448
```

送出之前先檢查這個個案會不會一開始就失敗。

## wtune check

它在 WRF 的啟動目錄上檢查：時間窗、iofields 檔、物理方案需要的資料表、decomposition 是否合法、磁碟空間，以及 `wrfinput` 的內容。

| 輸出 | 意思 | 該做什麼 |
|---|---|---|
| `FATAL` | 這個設定或輸入會讓執行失敗 | 先修好再送 |
| `WARN` | 可以跑，但有值得注意的地方 | 看過再決定 |

`wrfinput` 的檢查包含：土壤溫度是否在合理範圍、土壤濕度是否超出 0 到 1、主要變數有沒有 NaN 或 Inf。

```bash
wtune check case/ --ranks 448 --json check.json                # 另存 JSON
wtune check case/ --nproc-x 16 --nproc-y 28                     # 檢查特定切法
wtune check case/ --wrf-run-dir $WRF_TREE/run                   # 列出 WRF run/ 有、個案目錄沒有的檔案
```

偵測到 WRF 擴展（moving nest、WRF-Chem、WRF-Hydro、WRFDA）時會各多印一行支援說明，見 [WRF 擴展](extensions.md)。

## wtune predict-rebuild：整段要跑多久

```bash
wtune predict-rebuild case/ --nodes 4 --nx 16 --ny 28 --numtiles 4 --omp 2 --dry-run
wtune predict-rebuild case/ --nodes 4 --nx 16 --ny 28 --numtiles 4 --omp 2
```

它回答的問題是：用這個配置跑完整段模擬，`sbatch --time` 該給多少。

```mermaid
flowchart LR
    T[模擬時間軸] --> P[挑最多 5 個時間點]
    P --> R[每個點：用 ERA5 重建初始場<br/>WPS → real.exe → 短 spin-up]
    R --> M[在目標配置上量幾十步]
    M --> C[合成整段的計算時間<br/>含日夜輻射的差異]
    C --> O[pred_rebuild.json<br/>與建議的 --time]
```

每個時間點是一個獨立的 Slurm job，全部同時送出。I/O 是實際量測的：最後一個點會用個案自己的設定各寫一個歷史 frame 和 restart。

輸出在 `predict/<id>/pred_rebuild.json`，並印出建議的 `--time`（已含 I/O 與餘裕）。它需要 `WPS_DIR`、`WRF_TREE` 和 `case/wrfinput_d01`。

### 準確度

scc 上的實測（計算加初始化的時間，對同一配置的完整執行）：

| 個案 | 預測誤差 |
|---|---:|
| conus25，6 小時 | +0.98% |
| maria1km，3 小時 | −0.04% |
| zhengzhou，30 小時、三層巢狀 | −5.78% |
| lafe，24 小時 | +0.16% |

同一個執行在 scc 上隔天重跑本來就有 ±2–4% 的差異。

### 限制

- **對流尖峰會低估。** 短 spin-up 長不出組織化的對流，那個時段的成本會估得偏低；不要把預測值當上限。
- **不支援 adaptive time step。**
- **WRF-Chem、WRF-Hydro、WRFDA、moving nest 一律拒絕。**
- 只取模擬起點一個時間點會錯過日夜輻射的差異（lafe −10.5%、conus25 −9.6%），所以預設取多個點。

### 把預測用起來

```bash
wtune run --prediction predict/<id>/pred_rebuild.json --wall-limit 04:00:00
wtune watch runs/<id> --prediction predict/<id>/pred_rebuild.json
```

`run` 用它決定每一段切多長、每個 job 的 `--time` 給多少；`watch` 用它估剩餘時間。

!!! note "一般情況不需要這一步"
    `wtune tune` 的報告已經帶有時間估計，`wtune run` 會直接使用。`predict-rebuild` 是給「不調校、只想知道某個固定配置要跑多久」的情況。
