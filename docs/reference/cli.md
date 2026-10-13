# 指令總表

```bash
wtune --help
wtune <指令> --help
```

每個指令的完整旗標以 `--help` 為準。這一頁是索引。

## 主要流程

| 指令 | 作用 | 會送 job | 教學 |
|---|---|:-:|---|
| `wtune project init\|info\|set` | 建立、檢視、修改 project 資料夾 | 否 | [project](../tutorial/project.md) |
| `wtune doctor` | 檢查機器與專案是否準備好 | 否 | [安裝](../tutorial/install.md) |
| `wtune submit` | 調校 → 執行 → 後處理，一次做完 | 是 | [submit](../tutorial/quickstart.md) |
| `wtune tune` | 實測找最快的配置 | 是 | [tune](../tutorial/tune.md) |
| `wtune run` | 分段執行，restart 交接，失敗續跑 | 是 | [run 與 watch](../tutorial/run.md) |
| `wtune watch` | 唯讀監看一個執行目錄 | 否 | [run 與 watch](../tutorial/run.md) |
| `wtune post ...` | 後處理與分析 | 加 `--srun` 時 | [post](../tutorial/post.md) |

## 檢查與預測

| 指令 | 作用 | 會送 job |
|---|---|:-:|
| `wtune check` | 提交前檢查啟動目錄 | 否 |
| `wtune predict` | 不執行，估計各階段與整段的時間 | 否 |
| `wtune predict-rebuild` | 在目標配置上量測重建的時間點，預測整段時間 | 是 |
| `wtune config init\|show\|validate` | 管理調校的搜尋空間 YAML | 否 |

## 建議

| 指令 | 作用 |
|---|---|
| `wtune advise` | 偵測個案條件，建議 WRF 的模式與流程 |
| `wtune calc` | 參數的經驗法則與單位換算；`calc check` 檢查整份 namelist |
| `wtune physics` | 物理診斷、數值建議、方案選單 |
| `wtune compliance validate\|list\|classify` | 設定的分類目錄 |

## 資料準備與轉檔

| 指令 | 作用 |
|---|---|
| `wtune convert nc2bp\|bp2nc\|inputs` | WRF netCDF 與 ADIOS2 BP 互轉 |
| `wtune metdata find\|check\|fetch` | GRIB 驅動資料的來源、可用性、下載 |
| `wtune prep geog ...` | 選用的高解析度地理資料與 DEM 匯入 |
| `wtune scenario list\|generate\|inspect` | 產生虛構日期的 WRF 驅動資料 |

## 機器校正

| 指令 | 作用 |
|---|---|
| `wtune calibrate plan\|fit` | 在新叢集上量測並產生校正檔 |
| `wtune machine-profile` | 印出機器量測結果的摘要 |

## post 的 subcommand

| 子指令 | 作用 |
|---|---|
| `recipe` | 依個案類型出標準圖組與 HTML 報告 |
| `fetch`、`fetch-era5` | 下載觀測與再分析資料 |
| `tc` | 颱風路徑追蹤與誤差 |
| `inventory` | 域、時間、網格、變數、缺 frame 的摘要 |
| `extract` | 抽出點位 CSV 或區域 netCDF |
| `plot` | 單一欄位的地圖 |
| `sounding` | Skew-T |
| `compare` | 用 `diffwrf` 比對兩次執行 |
| `report` | 把一個目錄的圖表收成一份 HTML |
| `custom` | 執行 project 內 `analysis/*.py` |
| `cf3d` | 輸出氣壓層、經緯度網格的 CF NetCDF |
| `vapor-bundle` | 產生可複製到工作站的 VAPOR 資料包 |

## 其他

| 指令 | 作用 |
|---|---|
| `wtune analyze diag\|tc-track` | 颱風診斷與追蹤 |
| `wtune screen` | 參數篩選流程 |
| `wtune selftest` | 校正自我測試 |
| `wtune --version` | 版本 |
