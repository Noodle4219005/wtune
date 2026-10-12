# WRF 擴展

```bash
wtune check case/
```

偵測到 moving nest、WRF-Chem、WRF-Hydro、WRFDA 時，`check` 會各印一行支援說明。

對這些個案，WTune **只調整網格與 rank 的配置**（rank 數、decomposition、tile 數、OpenMP 執行緒數、I/O）。擴展自己的參數一律照原樣不動。

## 各擴展的支援範圍

| 擴展 | WTune 做什麼 | WTune 拒絕什麼 |
|---|---|---|
| moving nest | 網格與 rank 調校；時間估計加 6.0%（實測值） | `predict-rebuild`、`--cfl-rescue` |
| WRF-Chem | 網格與 rank 調校；OpenMP 執行緒數上限 1；y 方向最小 patch 須 ≥ 21 格 | `predict-rebuild`、`--cfl-rescue` |
| WRF-Hydro | 網格與 rank 調校；OpenMP 執行緒數上限 1 | `predict-rebuild`、`--cfl-rescue`；不支援 standalone Hydro |
| WRFDA | 預報段照一般 WRF 調校；`da_wrfvar.exe` 的配置只給建議；OpenMP 執行緒數上限 1 | `predict-rebuild`；DA cycling |

不支援：ndown、NMM、MPAS。

## WRFDA 要多一個旗標

```bash
wtune run --wrfda-forecast-only
```

WTune 只跑 `wrf.exe` 的預報段，不跑 `obsproc`、`da_wrfvar`、`da_update_bc`。這個旗標是要你確認知道這件事；不加的話 WRFDA 個案會被拒絕。

`da_wrfvar.exe` 的 rank 數與切法不會自動套用，因為最小化過程的歸約順序隨切法改變，結果不會逐位元相同。

## 哪些數字是暫定的

| 項目 | 狀態 |
|---|---|
| moving nest 的時間係數 +6.0% | 實測 |
| Chem、Hydro 的時間係數 | 暫定值，誤差範圍 ±100%，尚未校正 |
| Chem 的最小 patch 門檻 21 | 來自單一個案、每個配置只跑一次，屬暫定 |

時間係數只在該個案沒有實際量測時才套用。要用自己的係數：設定 `$WTUNE_EXTENSION_COEFFS` 指向你的 JSON 檔。

## 建置（scc）

```bash
bin/build-wrf-extensions.sh submit movenest
bin/build-wrf-extensions.sh submit chem
bin/build-wrf-extensions.sh submit hydro
bin/build-wrf-extensions.sh submit wrfda
```

- WRFDA 用 gcc 建置，其餘用 nvhpc。
- Chem 與 Hydro 要建成 dm+sm，並以 `OMP_NUM_THREADS=1` 執行。
- WRF-Chem 目前只有不含 KPP 的版本，可跑 GOCART 等非 KPP 的化學機制。

!!! warning "WRF-Chem 會偶發卡住"
    每個 job 都要設時間上限（`--wall-limit`），逾時就重新提交，不要空等。
