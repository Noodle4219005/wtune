# 6. post

```bash
wtune post recipe --srun
```

在 project 內執行。它讀 `runs/*/` 的輸出，依個案類型畫一組標準圖，和觀測資料比較，全部收進 `post/<id>/report.html`。

所有圖另外集中在 `post/<id>/figures/`，檔名是 `<類型>-<原檔名>`（例如 `tc-track_map.png`、`synoptic-syn_slp_thickness.png`），可以直接整個資料夾瀏覽。這裡是 hard link，不另外佔空間；各類型的子資料夾仍保留圖旁邊的資料檔（csv、dat）。

`wtune post` 底下所有指令都只讀模式輸出、只產生圖和表：不改 namelist，不送模式的 job。

!!! tip "`--srun` 是什麼"
    出圖很吃記憶體和 CPU，不要在登入節點上做。任何 `wtune post` 子指令加上 `--srun`，就會包成一個獨佔計算節點的 job step 來執行（預設上限 120 分鐘，`--srun-minutes N` 可改），並避開 `head` 和專案設定的節點。

各圖組（共同圖、資料產品、每種個案類型、時間序列）彼此獨立，會同時在不同程序裡畫，報告裡的順序不變；颱風追蹤的 `DetectNodes` 也會把時間切成幾段同時跑。程序數預設是 job 可用的 CPU 數，記憶體不夠時用 `WTUNE_POST_WORKERS` 調小（`1` 就是逐組依序畫）。

## 個案類型

`--types auto`（預設）會自動判斷。類型是**多標籤**：同時符合幾種，就把每一種的圖都畫出來。

| 類型 | 自動判斷的條件 | 主要的圖 |
|---|---|---|
| `common` | 一律包含 | 區域與巢狀域地圖 |
| `tc` | 最佳路徑資料顯示風暴在 d01 內（需要 `--storm` 與 `--hurdat2`，或 `--bdeck`） | 路徑、強度誤差、風速半徑 |
| `convection` | 有任一個域 dx ≤ 4 km 且 `cu_physics=0` | 最大回波、CAPE／CIN、上升氣流螺旋度、Skew-T |
| `precip` | 一律包含 | 累積降水，以及對 Stage IV 的比較 |
| `synoptic` | d01 範圍 ≥ 1500 km 或模擬 ≥ 24 小時 | 500 hPa 高度與渦度、海平面氣壓、850 hPa、250 hPa 噴流 |
| `pbl` | `bl_pbl_physics=0` 或 dx ≤ 1 km | 剖面、時間–高度圖、對探空的比較 |
| `terrain` | 地形起伏 ≥ 1000 m 或最高點 ≥ 1500 m | 地形與風場、跨山剖面 |
| `moisture` | 偵測到 precip、convection 或 tc | 可降水量、水氣通量（IVT） |
| `chem` | `chem_opt>0` | 濃度與風場 |
| `climate` | 模擬 ≥ 30 天 | 月平均、季平均、對 ERA5 與 GPCP 的偏差 |

自己指定類型就跳過自動判斷：

```bash
wtune post recipe --types tc,convection --srun
```

## 跑之前先看：preview

```bash
wtune post preview          # 在 project 內；不需要 runs/
```

還沒跑 WRF 就能做，幾秒鐘，不送 job。它用專案裡已有的資料回答兩件事：

- **跑完會畫哪些圖**：依 namelist 和 `wrfinput` 偵測個案類型（和 `recipe` 用同一套規則），把每種圖和它回答的問題列在 `post/preview/expected_figures.csv` 與 `preview.html`。
- **大概會長什麼樣**：現在就能畫的圖放在 `post/preview/figures/`——
    - 地形與網格（`common_domain_map.png`）
    - 起始時刻的海平面氣壓與 10 m 風（`synoptic-syn_slp_wind10_f000.png`，來自 `wrfinput`）
    - 颱風個案的最佳路徑（`tc-track_map.png`）
    - 模擬時段的 Stage IV 觀測雨量（`precip-precip_stage4_total.png`）
    - 已下載的 ERA5 500 hPa 高度

- **圖的數值**：每張圖畫的數值另存成 netCDF，放在 `post/preview/data/`，檔名和 `figures/` 裡的圖相同、副檔名換成 `.nc`。例如 `synoptic-syn_slp_wind10_f000.nc` 有 `slp`（hPa）、`u10`、`v10`（m/s）和 2-D 的 `lat`、`lon`；`common_domain_map.nc` 每個 domain 一個 group（`d01`、`d02`…），各有地形 `terrain`。最佳路徑沒有網格，數據是 `tc/best_track.dat`（ATCF 格式）。用 `ncdump -h` 或 xarray 讀即可。

`preview.html` 另外列出 tune 預估的執行時間，每張圖下面也註明數據檔。

這些圖都不是模式結果，而是初始場和觀測：模式中途會算出什麼，跑之前無法知道。跑完後拿 `post/<id>/figures/` 的同名圖（例如兩邊的 `tc-track_map.png`）來比，差異就是模式自己的表現。觀測要先用 `wtune post fetch` 下載，沒下載的那幾張會在「略過」裡說明。

## 時間序列與動畫

```bash
wtune post recipe --every 6h --srun            # 每 6 小時一組完整的圖
wtune post recipe --every 6h --animate --srun  # 另外每個欄位做一支 MP4
wtune post recipe --times 2005082700,2005082712 --srun
```

圖放在 `post/<id>/timeseries/`，檔名像 `slp_d02_f018_2005082718Z.png`：欄位、域、預報時數、有效時間。

## 觀測比較

```mermaid
flowchart LR
    F[wtune post fetch] -->|input/obs/&lt;name&gt;/<br/>附 provenance.json| R[wtune post recipe]
    W[runs/&lt;id&gt;/ 的 wrfout] --> R
    R --> H[report.html<br/>每張圖註明資料來源與授權]
```

`wtune submit` 會自動抓 `project.yaml` 裡 `post.fetch` 列的資料。手動抓：

```bash
wtune post fetch --list                 # 有哪些資料集、各自涵蓋的時間
wtune post fetch stage4 --project .
wtune post fetch nexrad --project . --run-dir runs/<id>/
```

| 資料集 | 內容 | 限制 |
|---|---|---|
| `stage4` | NCEP Stage IV 降水 | 2001 年起，僅美國本土 |
| `nexrad` | NEXRAD 雷達體積掃描 | — |
| `mrms` | MRMS 合成回波與降水估計 | 2020-10-14 起 |
| `stations` | ISD-Lite 地面測站 | 點觀測 |
| `igra` | IGRA2 探空 | 00Z 與 12Z |
| `era5`、`era5_monthly` | ERA5 再分析 | — |
| `gpcp_monthly` | GPCP 月降水 | 2.5 度 |

抓資料要在能連外網的機器上執行。每個資料集都會寫一份 `provenance.json`（提供者、網址、授權、抓取時間、引用格式）。

比較範圍一律排除模式的側邊界區。降水比對另外限制在 NEXRAD 站 230 km 以內，因為 Stage IV 在外海是 0 而不是缺值。

### agent 找來的觀測

沒有 fetch 模組的資料（美國以外的格點雨量、各國雨量站）可以由 agent 依 [event](event.md) 的流程找來，轉成標準格式 `wtune-obs/1` 放在 `input/obs/<name>/`。`post recipe` 會自動使用通過 `wtune event obs-check` 的資料夾：

| kind | 檔案 | 畫出的圖 |
|---|---|---|
| `gridded_precip` | `precip_YYYYMMDDHH_NNh.nc`（到該時刻為止 NN 小時的累積，`precip` mm） | `precip_agent_*`：和 Stage IV 同樣的 Grid-Stat 四張圖，在該資料集自己的格網上比 |
| `rain_gauges` | `gauges.csv`（`id,lat,lon,elev_m,start_utc,end_utc,precip_mm`） | `precip_gauge_*`：MET Point-Stat 的散布圖和每站偏差地圖 |
| `radar` | `refl_YYYYMMDDHHMM.nc`（`refl` dBZ；無回波填負值，無涵蓋填 NaN） | `conv_radar_*`、`conv_fss`：沒有 MRMS／NEXRAD 時改用它，在模式網格上比 |
| `satellite_ir` | `irtb_YYYYMMDDHHMM.nc`（紅外窗區亮溫 `tb`，K） | `cloud_ir_*`：模式 OLR 換算亮溫對衛星，冷雲頂 ≤240／220／210 K 的分數與 FSS |
| `stations_extra` | `stations_extra.csv`（`id,lat,lon,elev_m,time_utc,var,value`，`var` 是 `gust` m/s 或 `visibility` m） | `stn_gust_*`、`stn_vis_*`：Point-Stat 散布圖與每站偏差地圖 |

所有格點檔都是 1-D `lat`、`lon`（度），時間一律 UTC。WRF 不輸出陣風與能見度，模式端用標準診斷：陣風有 `WSPD10MAX`（`nwp_diagnostics = 1`）就用，否則用 NCEP UPP 的 CALGUST；能見度用 Stoelinga–Warner（1999），模式與觀測都截在 16.1 km。衛星比較的模式值是 OLR 換算的寬頻亮溫，在高雲上比窗區頻道偏暖。

```bash
wtune event obs-check --project .        # 檢查 input/obs/ 下所有 wtune-obs/1 資料夾
```

有 error 的資料夾不會被使用，報告頂端的 notes 會寫出第一個錯誤。`obs-check` 也會對常見的轉換錯誤發出 warning（整個資料夾都沒有超過 1 mm 的雨、大範圍雷達沒有任何缺值、陣風中位數低於 2 m/s 等），這些多半是單位、縮放或雷達範圍沒處理好，要確認後再用。圖下的來源標示取自該資料夾的 `provenance.json`。

## 報告

打開 `post/<id>/report.html`，從上往下看：

1. **頂端的 notes 框**：WTune 繞過了哪些問題（路徑追蹤退回父域、觀測沒涵蓋、離線、缺工具）。
2. **偵測到的類型與依據**，例如 `convection: d02 dx=4 km, cu_physics=0`。
3. **各類型的圖**，每張圖有說明：這張圖回答什麼問題、怎麼讀、常見的誤讀。
4. **缺少的圖**與原因。
5. **資料來源與授權**。

畫不出來的圖不會被默默略過，一定列在「缺少」裡並附原因。

## 其他 subcommand

```bash
wtune post inventory runs/<id>/ --json
wtune post extract runs/<id>/ --vars slp,t2,wspd10 --points "25.4,-74.6,eye" --out points.csv
wtune post plot runs/<id>/ --field slp --overlay wind --out figs/
wtune post sounding runs/<id>/ --point 25.4,-74.6 --time 2005-08-27_00:00:00 --out skewt.png
wtune post compare runs/a/ runs/b/ --vars T2,slp
```

| 指令 | 用途 | 底層工具 |
|---|---|---|
| `inventory` | 有哪些域、時間、變數；有沒有缺 frame | `ncdump`、CDO |
| `extract` | 抽出點位時間序列或區域 netCDF | wrf-python |
| `plot` | 單一欄位的地圖 | wrf-python、cartopy |
| `sounding` | Skew-T 與 CAPE／CIN | MetPy |
| `compare` | 逐欄位比對兩次執行的結果 | WRF 的 `diffwrf` |
| `tc` | 颱風路徑追蹤與誤差 | TempestExtremes、MET |
| `custom` | 執行 `analysis/*.py` 裡你自己的腳本 | — |

!!! note "用 compare 檢查兩次執行的差異"
    ```bash
    wtune post compare runs/before/ runs/after/
    ```
    兩次執行只差在 tile 數、OpenMP 執行緒數或 rank 擺法時，`diffwrf` 應該回報完全相同。改了 decomposition 或 rank 數時，可能出現浮點捨入等級的差異，原因見 [compliance](../guide/compliance.md)。

## custom

`wtune project init` 在 `analysis/` 放了一個範例。每個腳本定義一個 `run(ctx)`：

```python
def run(ctx):
    for run_id, run_dir in ctx.runs():
        domain = ctx.finest_domain(run_dir)
        for time_str, path, timeidx in ctx.list_frame_refs(run_dir, domain=domain):
            ds = ctx.open_wrfout(path)
            ...
    ctx.save_figure(fig, "my_figure.png")
    ctx.save_table("my_table.csv", ["time", "value"], rows)
```

```bash
wtune post custom --srun
```

所有腳本的圖和表會收成一份 HTML。某個腳本丟出例外時只會多一筆錯誤記錄，其他腳本照跑。

教學到這裡結束。接下來可以看 [compliance](../guide/compliance.md)。
