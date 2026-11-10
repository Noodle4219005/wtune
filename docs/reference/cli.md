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
| `wtune stop` | 讓正在跑的 tune 或 run 收尾：取消它自己的 job，寫出報告 | 否 | [tune](../tutorial/tune.md)、[run](../tutorial/run.md) |
| `wtune run` | 分段執行，restart 交接，失敗續跑 | 是 | [run 與 watch](../tutorial/run.md) |
| `wtune watch` | 唯讀監看一個執行目錄 | 否 | [run 與 watch](../tutorial/run.md) |
| `wtune post ...` | 後處理與分析 | 加 `--srun` 時 | [post](../tutorial/post.md) |
| `wtune event init\|vocab\|check\|plan` | 把 `event.txt` 的事件描述轉成檢查過的 `event.yaml`，列出要用的觀測資料 | 否 | [event](../tutorial/event.md) |
| `wtune event obs-check\|obs-convert` | 檢查 `input/obs/` 下的 `wtune-obs/1` 資料夾；把 ASOS CSV 轉成測站資料集 | 否 | [event](../tutorial/event.md) |
| `wtune proxy plan\|submit\|collect` | 用 1 個節點上的小區域代理，替 N 節點的執行排出切法名次 | `submit` 會 | [tune](../tutorial/tune.md) |

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
| `wtune prep wps`、`wtune prep real` | 用 WPS 與 real.exe 產生 `wrfinput`／`wrfbdy`（見 [project](../tutorial/project.md)） |
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
| `preview` | 還沒跑 WRF 前：列出會畫的圖，先畫初始場與觀測 |
| `fetch` | 下載觀測與再分析資料 |
| `fetch-era5` | deprecated：`fetch era5` 的別名，請改用 `fetch era5` |
| `tc` | 颱風路徑追蹤與誤差 |
| `inventory` | 域、時間、網格、變數、缺 frame 的摘要 |
| `extract` | 抽出點位 CSV 或區域 netCDF |
| `plot` | 單一欄位的地圖 |
| `sounding` | Skew-T |
| `compare` | 用 `diffwrf` 比對兩次執行 |
| `report` | 把一個目錄的圖表收成一份 HTML |
| `custom` | 執行 project 內 `analysis/*.py` |
| `brief` | 把一個 `recipe` 輸出目錄整理成給審圖者讀的 `brief.json` |
| `stations` | 模式與測站觀測的逐站比對 |
| `check` | 檢查 `review.json` 與 `assessment.md`，全部通過才回傳 0 |
| `cf3d` | 輸出氣壓層、經緯度網格的 CF NetCDF |
| `vapor-bundle` | 產生可複製到工作站的 VAPOR 資料包 |

## 其他

| 指令 | 作用 |
|---|---|
| `wtune analyze diag\|tc-track` | 颱風診斷與追蹤 |
| `wtune screen` | 參數篩選流程 |
| `wtune selftest` | 校正自我測試 |
| `wtune --version` | 版本 |
