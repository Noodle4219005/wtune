# 環境變數

```bash
source ~/wtune-scc/env.sh       # scc：一次設好 WRF 與編譯器相關的變數
```

這一頁列出使用者會用到的環境變數。

## 必要

| 變數 | 內容 | 誰需要 |
|---|---|---|
| `WRF_TREE` | WRF 的根目錄，底下有 `main/wrf.exe`、`main/real.exe`、`run/` | `tune`、`run`、`post compare` |
| `WPS_DIR` | WPS 的根目錄 | `predict-rebuild` |

## 專案與叢集

| 變數 | 內容 | 預設 |
|---|---|---|
| `WTUNE_PROJECT` | project 資料夾。指令列的 `--project` 優先 | 從目前目錄往上找 `project.yaml` |
| `WTUNE_SBATCH_EXCLUDE` | 所有 job 額外避開的節點，逗號分隔。`head` 一律避開 | 無 |

```bash
export WTUNE_SBATCH_EXCLUDE=work1
```

`wtune project init` 沒給 `--exclude-nodes` 時，會把這個變數的值寫進 `project.yaml`。

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
