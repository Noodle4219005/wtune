# 環境變數

```bash
source ~/wtune-scc/env.sh       # scc：一次設好 WRF 與編譯器相關的變數
```

這一頁列出使用者會用到的環境變數。

## 必要

| 變數 | 內容 | 誰需要 |
|---|---|---|
| `WRF_TREE` | WRF 的根目錄，底下有 `main/wrf.exe`、`main/real.exe`、`run/`。也可以寫成 `machine.wrf_tree` | `tune`、`run`、`post compare` |
| `WPS_DIR` | WPS 的根目錄 | `predict-rebuild` |

## 專案與叢集

| 變數 | 內容 | 預設 |
|---|---|---|
| `WTUNE_PROJECT` | project 資料夾。指令列的 `--project` 優先 | 從目前目錄往上找 `project.yaml` |
| `WTUNE_SBATCH_EXCLUDE` | 所有 job 額外避開的節點，逗號分隔。登入節點一律避開 | 無 |
| `WTUNE_CORES_PER_NODE` | 一個計算節點的硬體核心數 | `72` |
| `WTUNE_MAX_NODES` | 一個 job 最多用幾個節點 | `13` |
| `WTUNE_PARTITION` | Slurm partition | `compute` |
| `WTUNE_LOGIN_NODES` | 不送工作的節點，逗號分隔。設成空字串代表沒有 | `head` |
| `WTUNE_ACCOUNT` | Slurm 的計畫帳號 | 不指定 |
| `WTUNE_HARDWARE_ID` | 這種節點在量測紀錄裡的名字 | `scc-gh200` |
| `WTUNE_NUMA_PER_NODE` | 一個節點有幾個 NUMA 區 | `1` |
| `WTUNE_CALIBRATION` | 這台機器的校準檔（絕對路徑） | `machine/calibration/scc.json` |
| `WTUNE_SITE` | 用哪一份 `campaign/sites/<site>.sh` | `scc` |
| `WTUNE_ENV_SOURCE` | 每個工作先 source 的環境腳本 | `$HOME/wtune-scc/env.sh` |
| `WPS_GEOG` | WPS 的靜態地理資料 | `/nfs/data/wrf/WPS_GEOG/` |
| `WTUNE_MACHINE_FILE` | 使用者設定檔的位置 | `~/.config/wtune/machine.yaml` |

```bash
export WTUNE_SBATCH_EXCLUDE=work1
```

`wtune project init` 沒給 `--exclude-nodes` 時，會把這個變數的值寫進 `project.yaml`。

從 `WTUNE_CORES_PER_NODE` 到 `WPS_GEOG` 都是機器規格，平常寫在使用者設定檔或 `project.yaml` 的 [machine](../tutorial/project.md#machine) 區塊就好。手動設了環境變數時，環境變數優先。

## 後處理

| 變數 | 內容 | 預設 |
|---|---|---|
| `WTUNE_POSTPROC_VIEW` | MET 等工具的 spack view | 無；找不到時相關的圖列為缺少 |
| `WTUNE_PARAVIEW_BIN` | `pvbatch` 所在目錄 | `envs/paraview/.pixi/envs/default/bin` |
| `WTUNE_CARTOPY_DATA` | Natural Earth 底圖資料 | 無；找不到時改畫 LANDMASK 等值線 |
| `WTUNE_NC_CACHE` | ADIOS2 歷史輸出轉成 netCDF 的快取目錄 | project 或輸出目錄下的 `_nc_cache` |
| `WTUNE_HURDAT2` | HURDAT2 最佳路徑檔（`wtune advise` 使用） | 無 |
| `WTUNE_PYART_PATH` | Py-ART 的備用安裝目錄 | 使用 pixi 環境內的版本 |

## 進階

| 變數 | 內容 |
|---|---|
| `WTUNE_EXTENSION_COEFFS` | 自訂 WRF 擴展的時間係數 JSON，見 [WRF 擴展](../guide/extensions.md) |

!!! note "帳號與金鑰"
    某個資料來源需要帳號時，一律只從環境變數讀取，不會寫進檔案或記錄。
