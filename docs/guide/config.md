# config

```bash
wtune config init conus25 --out wtune.yaml
wtune config validate wtune.yaml
wtune config show wtune.yaml
wtune tune --config wtune.yaml
```

`wtune.yaml` 控制 `wtune tune` 可以搜尋哪些值。不寫這個檔也能調校；需要限制範圍時才用。

## 優先順序

```mermaid
flowchart LR
    A[指令列旗標] -->|蓋過| B[wtune.yaml] -->|蓋過| C[內建預設值]
```

未知的 key 會直接報錯，打錯字不會被默默忽略。

## 常用 key

```yaml
site: scc
case: conus25
compliance: strict

budget:
  minutes: 60            # 整個調校的時間上限
  max_nodes: 13          # 最多用幾個節點；不寫就用機器的上限
  nodes: search          # auto | search | 固定的整數
  node_objective: time   # time | cost | knee
  node_seeds: geometric  # geometric | all
  hotspot_fill: true     # 一波沒用滿的節點補量最佳節點數附近

search:
  omp: [1, 2, 4, 8]                    # 允許的 OpenMP 執行緒數
  numtiles: [1, 4, 8, 16]              # 允許的 tile 數
  rank_distribution: [block, cyclic]   # rank 在節點間的擺法
  aspect_max: 8                        # nx 與 ny 的比例上限

probe:
  steps: auto            # 每個候選配置量幾步
  sweep_steps: [10, 40]  # 節點與核心數掃描的步數範圍
  sweep_reps: 1          # 掃描每點量幾次；接近最快的會再量一次
  confirm_reps: 2        # 前幾名重複量幾次
```

`search` 底下省略某個 key，代表那個維度不設限。

採樣點怎麼選、每個 key 怎麼影響採樣，見 [tune](../tutorial/tune.md#wtuneyaml)。

| 區塊 | 控制什麼 |
|---|---|
| `budget` | 時間與節點的上限、節點數怎麼選 |
| `search` | 每個維度允許的值 |
| `probe` | 量測的步數與重複次數、時間預測點的數量 |
| `prediction` | 整段時間估計的校正目標 |
| `exclude` | 一定要排除的組合 |

完整的 key 與註解在 `examples/wtune.yaml`。

!!! note "機器規格不在這個檔"
    每節點核心數、節點數上限、partition、登入節點寫在 `project.yaml` 的 [machine](../tutorial/project.md#machine) 區塊。`budget.max_nodes` 超過機器的上限時，以機器的上限為準。

## 尚未搜尋的區塊

下面兩個區塊在範例檔裡有，但 `wtune tune` 還沒實作；把 `enabled` 設成 `true` 會直接報設定錯誤，不會假裝有做：

- `search.io`（歷史輸出的 `io_form`）
- `physics_gray`（`radt`、`time_step` 的倍數）

## 排除規則範例

```yaml
exclude:
  - {omp_gt: 1, when: fire}
```

WRF-Fire 個案只要 OpenMP 執行緒數大於 1 就會在第一步當掉，所以這條規則對所有候選配置（包含基準配置）都生效。
