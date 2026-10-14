# Vendored: WRF-ADIOS2-to-NetCDF4

WRF 官方文件 [`doc/README.adios2`](https://github.com/wrf-model/WRF/blob/06d4240ae989cc3e50af412bb472df3d9048783c/doc/README.adios2#L27-L29)
指定的 ADIOS2 → NetCDF4 反向轉換工具,vendor 進來後補上相容性 patch(見下方)。

| | |
|---|---|
| upstream | https://github.com/MichaelLaufer/WRF-ADIOS2-to-NetCDF4 |
| commit | `602c85bc75cc39fae22a3e6ca3d684d96b90606f`（2023-12-09） |
| 授權 | MIT,見同目錄 `LICENSE` |

## 這個 repo 加的 patch(2026-08-07)

實際在競賽叢集上跑起來後發現 upstream 版本用的 `adios2.open(file, "rra")`
是已經從現行 `adios2` Python 套件移除的舊 API(現在只剩 `Stream`/
`FileReader`)。`convert.py` 的 `open_files`/`r_metadata`/`r_w_data_serial`
已改寫成 `adios2.FileReader`,行為對照 upstream 原邏輯(逐 step 隨機讀取,
scalar/array 的 netCDF 賦值判斷準則不變),並用真實壓縮過的 wrfout 資料端到端
驗證過(`compare-wrfout-adios2.py --bitexact` 對原始 BP 目錄、以及對另一趟
`ENABLE_ADIOS2_ZSTD_IO=false` 的 netCDF4 基準,都是 `PASS (bit-exact, 11
variables)`)。

MPI 平行路徑(`--parallel`、`r_w_data_parallel`)**沒有**跟著改,呼叫時會直接
丟 `NotImplementedError`——upstream 自己的 README 就說平行版沒有效能優勢,
沒有測試環境去驗證改寫後的正確性,不值得冒風險保留一份看起來能跑、實際上
沒人驗證過的程式碼。要轉檔一律用預設的 serial 模式(不要加 `mpirun`)。

## 這個 repo 加的 patch(2026-08-12)

第一次拿官方 CONUS 2.5km benchmark 的完整輸出(6 小時 cold start、
`frames_per_outfile` 大於 1,所以 7 個 history frame 全在同一個 BP 目錄裡,
18.9 GB)去轉,才發現上面那次驗證用的資料量太小,遮住了兩個問題:

**1. 輸出沒有壓縮。** upstream 的 `create_nc_vars` 建變數時不給任何 filter,
所以轉出來的是未壓縮 NetCDF4。但 WRF 自己的 `io_form_history=2` 寫出來的
wrfout **是壓縮過的**(從本 benchmark 的實際輸出讀出來是 zlib level 2 +
shuffle),2.5km 一個 frame 邏輯資料量 8.45 GB、落地只有 2.79 GB。少了壓縮,
7 個 frame 要 59 GB(而不是 19 GB),在 100 GB 的 user quota 上轉到一半就以
`RuntimeError: NetCDF: HDF error` 死掉。`create_nc_vars` 改成比照 WRF 自己的
設定加上 `zlib=True, complevel=2, shuffle=True`;char 型別與 scalar 變數維持
原樣(netCDF4 不接受 scalar 上的這些 filter,而 WRF 的 `Times` 本來就沒壓縮)。

**2. 沒辦法只轉一個 frame。** 官方 benchmark 附的參考解是單一 frame(只有
終點時刻 `2019-11-27_00:00:00`),所以任何拿它當基準的比對都只需要一個 frame;
而逐變數取 `mean` 的比對(`compare-wrfout-adios2.py --tolerance`,以及 PR #20
的 `tools/verify_wrf.py` correctness gate)餵進 7 個 frame 的話,`t=0` 那個恆為
0 的差異會把平均值稀釋掉。新增 `--step N`
(0-based,負數從尾端數)只轉指定的那一個 frame 成 1-frame 檔,
`r_w_data_serial` 的 step 迴圈改成「輸入 step 的清單 + 它在清單裡的位置當
輸出 Time index」。不給 `--step` 時清單就是 `range(num_steps)`、
輸出 index 等於輸入 step,行為與 upstream 相同。

驗證:同一個 BP 目錄用 `--step -1` 轉出的 frame,與另一趟
`ENABLE_ADIOS2_ZSTD_IO=false` 的 netCDF run 的同時刻 wrfout,
`compare-wrfout-adios2.py --bitexact` 判 `PASS (bit-exact, 11 variables)`。

細節見 `convert.py` 檔頭註解,以及 `docs/io-adios2-zstd.md` 的
「Converting ADIOS2 output back to NetCDF4」一節。

## 這個 repo 加的 patch(2026-10-14)

**每個 frame 一個 chunk。** `Time` 是固定長度維度,netCDF-C 的預設 chunk 沿
`Time` 跨很多 frame(121 frame 時是 41 個一組);逐 frame 寫入時每寫一張就要把
整個 chunk 解壓再壓一次。`create_nc_vars` 對第一維是 `Time` 的壓縮變數改用
`chunksizes = (1, 其餘維度全長)`,和 WRF 自己寫 wrfout 的分法相同。數值不變。

量測(本機,`(T, 40, 136, 196)` float32,zlib 2 + shuffle,逐 frame 寫):
41 frame 26.2 s → 2.0 s;121 frame 166 s → 5.9 s(3.1 → 87 MB/s)。實際案例:
yagi d02 一個 BP 裡 121 frame,收尾轉檔作業(含比對)佔一個 144 核節點 46 分鐘。

## 用途

把 `io_form_history=14`（ADIOS2 + zstd,見 `docs/io-adios2-zstd.md`）寫出的
BP4/BP5 目錄轉回標準 NetCDF4 檔案,給不吃 ADIOS2 格式的下游工具(ncl、
wrf-python、既有後處理流程)用。轉換保留原始 wrfout 的 dimension、
global/variable attributes,輸出檔案結構跟 `io_form_history=2` 直接寫出的
wrfout 相同。

## 用法

見 repo 根目錄 `docs/io-adios2-zstd.md`。

## 重新取得 upstream 時

```bash
curl -sL https://raw.githubusercontent.com/MichaelLaufer/WRF-ADIOS2-to-NetCDF4/<commit>/convert.py -o convert.py
curl -sL https://raw.githubusercontent.com/MichaelLaufer/WRF-ADIOS2-to-NetCDF4/<commit>/LICENSE -o LICENSE
```
