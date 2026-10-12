# 5. 執行與監看 (run / watch)

```bash
wtune run --dry-run     # 先看會怎麼切
wtune run               # 送出
```

在 project 內執行。它會採用最新的 `tune/*/tune_report.json` 裡的 WINNER 配置，把整段模擬送進 Slurm，輸出放在 `runs/<id>/`。

## 為什麼要分段

一個很長的 job 容易撞到時間上限，中途節點故障也會全部重來。`wtune run` 把模擬切成數段，每段是一個獨立的 Slurm job，段與段之間用 restart 檔交接。

```mermaid
flowchart LR
    S0[seg_0<br/>job A] -->|wrfrst| S1[seg_1<br/>job B]
    S1 -->|wrfrst| S2[seg_2<br/>job C]
    S1 -.->|節點故障| R[seg_1_r1<br/>從最近的 restart 續跑]
    R -->|wrfrst| S2
```

```bash
wtune run --wall-limit 04:00:00        # 每個 job 最多 4 小時，WTune 自己決定切幾段
wtune run --segment-minutes 180        # 每段固定 180 分鐘的模擬時間
```

兩個都不給就是一段跑完。

!!! note "分段不改變結果"
    每一段都寫在同一個 restart 間隔的格點上，歷史輸出的時間點不會因為分段而偏移。實測：`dorian2019` 的 30 + 30 分鐘 restart 接力和一次跑完相比，d01 與 d02 各 202 個變數逐位元相同。

## 自己指定配置

不想用 tune 的結果時：

```bash
wtune run --nodes 4 --nx 16 --ny 28 --numtiles 4 --omp 2
```

`--nodes`、`--nx`、`--ny` 三個要嘛來自 tune 報告，要嘛全部自己給。指定某一份報告：`--tune-report tune/<id>/tune_report.json`。

## 失敗時會怎樣

```mermaid
flowchart TD
    F[一段 job 結束] --> Q{成功？}
    Q -->|是| N[下一段]
    Q -->|否| T{失敗原因}
    T -->|節點故障、逾時、被搶佔| A[自動從最近的有效 restart 續跑<br/>預設最多 1 次]
    T -->|CFL 或 NaN| C{有 --cfl-rescue？}
    C -->|沒有| STOP1[停下來<br/>寫出建議的 time_step]
    C -->|有| RES[同一段用較小的 dt 重跑一次<br/>標記 PHYSICS-GRAY]
    T -->|其他：segfault、FATAL| STOP2[停下來並報錯]
```

| 情況 | WTune 的行為 | 相關旗標 |
|---|---|---|
| 節點故障、逾時、被搶佔 | 從最近一個有效的 restart 自動續跑 | `--resume N`（預設 1，0 關閉） |
| 個案自己沒寫 restart | 自動補上檢查點 | `--checkpoint-minutes M`（預設 `auto`） |
| CFL 或 NaN | **停下來**，寫出建議的 `time_step` 與 `segment.advice.namelist.input`，不重送 | — |
| CFL 或 NaN，且你事先授權 | 同一段以約 0.75 倍的 `time_step` 重跑一次，之後的段回到原本的值 | `--cfl-rescue` |
| 其他錯誤 | 停下來並報錯，不會默默重試 | — |

!!! warning "`--cfl-rescue` 會改變結果"
    縮小 `time_step` 會改變數值解，所以用了它的 run 會在報告裡標記 `physics_gray: true`。這是 WTune 唯一會替你改「影響結果的設定」的情況，而且要你明確加上旗標才會發生。

`runs/<id>/run_report.json` 在每一段結束後都會重寫，記錄每次嘗試的結果。

## I/O 格式

預設所有 I/O 都走 ADIOS2（`io_form 14`）：歷史輸出、restart、輸入、邊界條件。個案原本的 netCDF 輸入會轉換一次，放在 `<project>/input/bp/` 重複使用。

要保留個案自己的 `io_form_*` 設定：`--io netcdf`（舊路徑，會印 WARNING）。

`--sidecar`：job 執行期間，用 WRF 沒用到的核心驗證每組 restart，並把寫完的歷史輸出轉成 netCDF 放進後處理的快取。

## 監看：wtune watch

```bash
wtune watch runs/<id>              # 每 60 秒更新
wtune watch runs/<id> --once       # 只印一次
```

`watch` 完全唯讀：只讀 `namelist.input` 和 `rsl.error.0000`／`rsl.out.0000`，不寫檔、不送訊號。它顯示進度、剩餘時間（計算與寫檔分開列），以及這些警告：

| 警告 | 意思 | exit status |
|---|---|---|
| `HANG_INIT` | 初始化之後很久沒有出現 d01 的 Timing 行（預設 600 秒） | 2 |
| `HANG` | 執行中途停止推進 | 2 |
| `NEST_STOPPED`、`NEST_END_EARLY` | 巢狀域停了，或比預期早結束 | 2 |
| `FAILED` | 執行失敗 | 3 |

有時間預測檔時，加上 `--prediction predict/<id>/pred_rebuild.json`，剩餘時間會改用預測曲線乘上目前的實際與預測比值。

下一步：[6. 後處理](post.md)。
