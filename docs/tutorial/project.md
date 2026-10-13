# 3. project

```bash
wtune project info
```

一個 project 資料夾收納同一個個案的輸入、設定、每次執行和所有圖。上面這個指令印出它目前的內容。

## 目錄結構

```text
mycase/
  project.yaml      個案、站點、要避開的節點、機器規格、後處理預設值、風暴資訊
  cases.json        這個專案的個案登記（wtune tune 用）
  case/             namelist 的副本，其餘檔案是 symlink
  input/            驅動資料與觀測（era5/、obs/<name>/、bp/）
  tune/<id>/        wtune tune 的輸出
  predict/<id>/     wtune predict-rebuild 的輸出
  runs/<id>/        wtune run 的輸出
  post/<id>/        wtune post 的輸出（圖、表、report.html）
  analysis/         你自己的分析腳本（wtune post custom 會執行）
  manifest.jsonl    每個 wtune 指令一行記錄
```

```mermaid
flowchart TD
    SRC[原始個案資料夾<br/>永遠唯讀] -->|namelist 複製<br/>其餘 symlink| CASE[case/]
    CASE --> TUNE[tune/&lt;id&gt;/]
    TUNE -->|tune_report.json| RUNS[runs/&lt;id&gt;/]
    RUNS -->|wrfout| POST[post/&lt;id&gt;/]
    INPUT[input/obs/] --> POST
    TUNE & RUNS & POST -.->|每個指令一行| MAN[manifest.jsonl]
```

!!! note "原始個案不會被動到"
    `case/` 裡的 `namelist.input` 是副本，可以自由編輯。`wrfinput`、`wrfbdy`、`met_em` 等是 symlink，WTune 不會開啟原始資料夾來寫入。

## project 的搜尋順序

依序是：

1. 指令上的 `--project DIR`
2. 環境變數 `$WTUNE_PROJECT`
3. 從目前目錄往上找 `project.yaml`

所以最省事的做法是 `cd` 進 project 再下指令。在 project 內，`--out` 預設是對應子目錄下的 `<id>/`，要處理的 run 預設是 `runs/*/`。

在 project 之外，`--out` 和 run 目錄都必須自己給。

## project.yaml

`wtune project init` 產生的內容：

```yaml
name: mycase
created: '2026-10-12T03:13:05Z'
case:
  source_dir: /path/to/case_dir
  id: mycase-b8e603
site: scc
exclude_nodes:
- work1
post:
  types: auto
  every: 6h
  fetch:
  - era5
  - igra
  - stage4
  - stations
storm: null
```

| 欄位 | 作用 |
|---|---|
| `case.id` | `wtune tune` 用的個案代號 |
| `site` | 機器的標籤，只是記錄，不影響執行 |
| `exclude_nodes` | 這個專案所有 job 都避開的節點（另外一律避開 `machine.login_nodes`） |
| `post.types` | 後處理的個案類型，`auto` 代表自動判斷 |
| `post.every` | 每隔多久出一組完整的圖 |
| `post.fetch` | `wtune submit` 會自動去抓的觀測資料 |
| `storm` | 颱風代號與 HURDAT2 檔，颱風個案才需要 |

## machine

叢集的規格。內建值是 scc 的，`wtune project init` 不會寫這個區塊。換一台機器時才加，只寫要改的 key：

```yaml
machine:
  cores_per_node: 64       # 一個計算節點的硬體核心數，內建 72
  max_nodes: 20            # 一個 job 最多用幾個節點，內建 13
  partition: batch         # Slurm partition，內建 compute
  login_nodes: [login1]    # 不送工作的節點，內建 [head]；[] 代表沒有
  account: myproject       # Slurm 的計畫帳號，內建不指定
  hardware_id: mybox       # 這種節點在量測紀錄裡的名字，內建 scc-gh200
  numa_per_node: 2         # 一個節點有幾個 NUMA 區，內建 1
  calibration: /path/to/mybox.json   # 這台機器的校準檔，內建用 scc 的
  site: mybox              # 用 campaign/sites/mybox.sh 啟動工作，內建 scc
  env_source: ~/mybox-env.sh         # 每個工作先 source 的環境腳本，內建 $HOME/wtune-scc/env.sh
  wrf_tree: ~/wrf/WRF      # WRF 的根目錄，底下有 main/wrf.exe
  geog: /data/WPS_GEOG     # WPS 的靜態地理資料
```

| key | 影響 |
|---|---|
| `cores_per_node` | 每個 job 向 Slurm 要的 `--ntasks-per-node`。配置的每節點 rank 數 × 執行緒數不能超過它 |
| `max_nodes` | `wtune tune` 搜尋節點數的上限。`--max-nodes` 和 `wtune.yaml` 的 `budget.max_nodes` 只能再往下調 |
| `partition` | 每個 `sbatch` 和收尾轉檔 `srun` 用的 partition |
| `login_nodes` | 每個 job 一律排除的節點，和 `exclude_nodes` 合併 |
| `account` | 每個 `sbatch`、`srun` 加上 `--account=`。依計畫計費的機器不給會被拒絕 |
| `hardware_id` | 每筆量測都記這個名字，預測模型用它分辨機器。WTune 內建認得 `scc-gh200`、`f1-spr-icpnq`、`f1-spr-icpnp`；其他機器要自己取一個名字 |
| `numa_per_node` | 決定可以試哪些 OpenMP 執行緒數：只試能整除「每個 NUMA 區核心數」的值，執行緒才不會跨區 |
| `calibration` | `wtune tune`、`wtune run`、`wtune check` 估算執行時間用的校準檔，由 `wtune calibrate fit` 產生。要寫絕對路徑 |
| `site` | 選用哪一份 `campaign/sites/<site>.sh`（MPI 怎麼啟動、節點怎麼清理）和 `campaign/cases.<site>.json`。那個檔不存在時 `wtune tune` 會直接報錯 |
| `env_source` | 每個工作開頭 source 的腳本，負責載入編譯器、MPI 和函式庫 |
| `wrf_tree` | 等同環境變數 `WRF_TREE` |
| `geog` | 等同環境變數 `WPS_GEOG`，重建個案時 geogrid 會用到 |

新機器至少要給 `cores_per_node`、`hardware_id`、`numa_per_node`。沿用 `scc-gh200` 這個名字卻改了核心數會直接報錯，避免別台機器的量測混進 scc 的紀錄。

`cores_per_node` 是節點的大小，不是每節點實際用幾核。實際用幾核由配置決定，`wtune tune` 會自己試少用幾核的配置。

優先順序：

```mermaid
flowchart LR
    A[手動設的環境變數] -->|蓋過| B[project.yaml 的 machine] -->|蓋過| U[使用者設定檔] -->|蓋過| C[內建值]
```

### 使用者設定檔

機器的規格每個專案都一樣，所以可以只寫一次，放在家目錄：

```bash
wtune project set --user --cores-per-node 144 --partition arm-dev --account myproject
```

這會寫進 `~/.config/wtune/machine.yaml`，內容和上面的 `machine:` 區塊相同，只是少一層縮排：

```yaml
cores_per_node: 144
partition: arm-dev
account: myproject
```

這台機器上的每個專案、以及在專案外下的指令都會讀它。要換位置就設環境變數 `WTUNE_MACHINE_FILE`。專案的 `machine:` 只需要寫和這份不同的地方。

`tune`、`run`、`post`、`submit` 都讀這裡，所以設一次就夠。打錯 key 或給了不合理的值會直接報錯，例如：

```text
error: project.yaml machine.cores_per_node: 'many' must be a positive integer
```

!!! note "目前只在 scc 上驗證過"
    這些設定在 f1-arm 上只確認過排程器接受產生的工作，還沒有在 scc 以外實際跑完一次 WRF。MPI 的啟動方式和節點清理寫在 `campaign/sites/<site>.sh`，換機器時要準備一份，再把 `site` 指過去。

## project set

不用手改 YAML，用 `wtune project set`：

```bash
wtune project set --exclude-nodes work1,work7
wtune project set --cores-per-node 64 --max-nodes 20
wtune project set --partition batch --login-nodes login1 --account myproject
wtune project set --hardware-id mybox --numa-per-node 2 --calibration ~/mybox.json
wtune project set --site mybox --env-source ~/mybox-env.sh --wrf-tree ~/wrf/WRF --geog /data/WPS_GEOG
wtune project set --user --partition batch      # 寫進使用者設定檔，不動這個專案
wtune project set --post-every 1d
wtune project set --post-types tc,convection
wtune project set --post-fetch era5,stage4
wtune project set --storm AL122005 --hurdat2 hurdat2-1851-2024.txt
```

清除一個設定：`--exclude-nodes ''`、`--post-every none`、`--post-fetch none`。`--login-nodes ''` 代表這台機器沒有要排除的登入節點。要讓 `machine` 的某個 key 回到內建值，把那一行從 `project.yaml` 刪掉。

比較舊的 project 沒有登記個案時：

```bash
wtune project set --register-case
```

## manifest.jsonl

每個 `wtune` 指令在這裡留一行：時間、WTune 版本、完整的 argv、主要輸入檔的路徑與 sha256（小於 50 MB 的檔案）、輸出目錄、exit code。

```bash
tail -3 manifest.jsonl
```

下一步：[4. tune](tune.md)。
