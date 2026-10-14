# 4. tune

```bash
wtune tune -O2
```

在 project 內執行。它會送出一批短的 Slurm job，實際量測不同配置的每步耗時（s/step），選出最快的，寫成 `tune/<id>/tune_report.json`。

## 搜尋目標

同一個個案、同一份 namelist，下面這些怎麼配都不會改變演算法和物理設定，但速度可以差很多：

| 項目 | 報告裡的欄位 | 意思 |
|---|---|---|
| 節點數 | `N` | 用幾個節點 |
| 每節點核心數 | `c/n` | 每個節點實際用幾顆核心，其餘閒置 |
| MPI rank 數 | `ranks` | 總共幾個 MPI 行程 |
| OpenMP 執行緒數 | `omp` | 每個 rank 幾條執行緒 |
| decomposition | `nx x ny` | 區域在 x、y 方向各切成幾塊 |
| tile 數 | `tiles` | 每個 patch 再分成幾個 tile |

```mermaid
flowchart LR
    D[整個模擬區域] -->|MPI：nx × ny| P[patch<br/>一個 rank 一塊]
    P -->|OpenMP：numtiles| T[tile<br/>一條執行緒算一塊]
```

每節點不一定要把核心用滿：有時少用幾顆核心反而比較快，所以 `c/n` 也是搜尋的一個維度。

## 等級 -O0 到 -O3

| 等級 | 做什麼 | 什麼時候用 |
|---|---|---|
| `-O0` | 不送 job：用模型估計時間，用代理指標挑配置 | 只想先看個大概 |
| `-O1` | 一輪量測，節點數取 1、2、4、8 與可用的最大值 | 短的個案 |
| `-O2` | `-O1` 加上細部搜尋、每節點核心數、在最佳節點數上抽樣量測 | 一般情況（`wtune submit` 的預設） |
| `-O3` | 完整搜尋：所有合法的節點數、核心數、rank × 執行緒、切法、tile 數，再重複量測確認 | 很長的正式執行，調校成本攤得掉 |

直接下 `wtune tune` 不帶等級時是 `-O3`。

開始之前，WTune 會先印出**每個等級估計要花多久**，再開始送 job。

```bash
wtune tune -O2 --minutes 90        # 加一個硬上限：整個調校最多 90 分鐘
wtune tune -O3 --stop-if-short     # 整段模擬比調校還短時，自動降到做得完的等級
```

!!! note "排隊時間不算在預算裡"
    `--minutes` 只計算量測本身。累積的排隊等待另有上限 `--max-queue-hours`（預設 6 小時），超過就停下來回報。

細部搜尋一輪一輪進行：每輪由模型提出一批候選，平均分給空著的節點同時量測，所以一輪的時間大約是一個候選的量測時間。出現下面兩種情況就提早結束，直接進入確認量測，不必跑滿 6 輪：

- 節點數：多加節點沒有快超過 3%，就不再送更多節點。
- 配置：目前最快的配置領先第二名超過 3%，而且這一輪沒有找到更快的。

3% 是量測重複性（`machine.measured.noise_frac`），換機器時可以在 [machine](project.md#machine) 區塊改。

## 節點檢查

調校前 WTune 會檢查每個節點：`sinfo` 的狀態，加上每個節點一個 1 分鐘的 `srun` 探測（家目錄、個案目錄、WTune 原始碼、`wrf.exe` 是否都讀得到）。沒通過的節點會從所有 job 排除，可用節點數也跟著下修。

要跳過這個檢查：`--no-node-preflight`。

## 報告

結果同時印在終端機和 `tune/<id>/tune_report.json`。先找這幾行：

```text
--- measured layouts (37; best first; the full table is in tune_report.json node_selection.layouts) ---
   N   c/n  ranks  omp   nx x ny tiles    median_s  spread  reps  source
   5    48     80    3      8x10     3      0.0482    0.4%     2  ...
 ...
WINNER:    N=5 c/n=48 ranks=80 omp=3 8x10 tiles=3 0.0482 s/step
RUNNER-UP: N=...
winner vs runner-up: 1.2% apart -> NOT distinguishable from noise: either is fine (noise floor 3%)
```

（表格列數、第二名與差距的數字是示意；`WINNER` 那一行是 `sc26p1` 在 scc 上的實測結果。）

| 行 | 怎麼讀 |
|---|---|
| `WINNER` | 選中的配置。`wtune run` 會自動採用它 |
| `RUNNER-UP` | 第二名 |
| `winner vs runner-up` | 兩者差距有沒有超過量測雜訊。寫 `NOT distinguishable from noise` 時，兩個都可以用 |
| `NOT TRIED` | 沒有量測的配置，依原因分組 |
| `not run N=...` | 被跳過的節點數與原因（例如預算用完） |

!!! warning "dirty node"
    看到這一行時，這次的排名不可信：

    ```text
    WARNING: the winner's number comes from a dirty node (...); the ranking may be wrong -- clean the node and tune again
    ```

    意思是量測時那個節點上有別的行程在吃 CPU（常見原因是先前被砍的 job 留下的 `wrf.exe`）。清掉之後重新調校。

## 節點數

預設的目標是「最快」。想省資源時改目標：

```bash
wtune tune --node-objective time                         # 最快；雜訊範圍內取節點最少的（預設）
wtune tune --node-objective cost --deadline-minutes 240  # 在 240 分鐘內跑完的前提下，node-hours 最少
wtune tune --node-objective knee --min-efficiency 0.7    # 平行效率仍有 0.7 以上的最大節點數
```

固定節點數、不搜尋：

```bash
wtune tune --nodes 4
```

## 不更動的項目

調校只改上面那六項。報告產生前有一道檢查：把報告改過的每個 namelist key 和個案原本的 namelist 比對，只要有一個不屬於 STRICT 類別（見 [compliance](../guide/compliance.md)）就直接報錯。

加上 `--physics-advice`（僅 `-O3`）會另外跑一段物理方案的建議，那一段只產生建議，不會套用。

下一步：[5. run 與 watch](run.md)。
