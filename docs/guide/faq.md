# 常見問題與已知限制

## 出問題時先做什麼

```bash
wtune doctor
```

它會指出缺了什麼，並給修正指令。接著看對應的記錄：

| 卡在哪一步 | 看哪裡 |
|---|---|
| `wtune submit` | `<project>/submit-<id>.log` |
| `wtune tune` | `tune/<id>/` 與終端機輸出的最後幾行 |
| `wtune run` | `runs/<id>/run_report.json`、`runs/<id>/seg_<k>/rsl.error.0000` |
| `wtune post` | `report.html` 頂端的 notes 框 |

## 常見問題

??? question "`wtune submit` 說 the project's case is not registered for tune"
    這個 project 是在個案登記功能出現之前建立的。執行：

    ```bash
    wtune project set --register-case
    ```

??? question "`wtune run` 因為 CFL 或 NaN 停下來了"
    這是預設行為。看失敗那一段旁邊的 `segment.advice.namelist.input`，裡面有建議的 `time_step`。

    兩個選擇：自己把 `case/namelist.input` 的 `time_step` 改小再重跑；或加上 `--cfl-rescue`，讓 WTune 把失敗的那一段用較小的步長重跑一次。後者會讓這次執行標記為 PHYSICS-GRAY。

??? question "調校報告出現 dirty node 警告"
    量測時節點上有別的行程在佔 CPU，通常是先前逾時的 job 留下的 `wrf.exe`。清掉之後重新調校，否則排名不可信。`campaign/node_hygiene.sh` 是清理與普查用的腳本。

??? question "`wtune watch` 顯示 HANG_INIT"
    初始化之後超過 600 秒沒有出現第一個時間步。先確認 job 還在跑，再看 `rsl.error.0000` 的最後幾行。已知有個案在特定切法下固定卡在初始化（`sc26p2` 的 17×13，7 次都卡住），這時換一個 decomposition。

??? question "報告裡有圖被列為缺少"
    每一張缺少的圖都附原因，常見的是：沒抓觀測資料（照報告印出的 `wtune post fetch` 指令去抓）、沒裝 MET 或 ParaView（`bin/install-postproc.sh`）、觀測資料沒涵蓋那個時間或區域。

??? question "我只想跑，不想調校"
    ```bash
    wtune run --nodes 4 --nx 16 --ny 28 --numtiles 4 --omp 2
    ```

??? question "調校花的時間比正式執行還久"
    用比較低的等級（`-O1`），或加 `--stop-if-short` 讓 WTune 自動降到做得完的等級。

## 已知限制

**時間預測**

- `predict-rebuild` 不支援 adaptive time step。
- 對流旺盛的時段會低估成本：短 spin-up 長不出組織化的對流。不要把預測值當成上限。

**支援範圍**

- `wtune tune` 目前只支援 `scc` 站點。
- ndown、NMM、MPAS 不支援。
- WRF 擴展的限制見 [WRF 擴展](extensions.md)。

**I/O**

- 預設全部使用 ADIOS2。
- VAPOR 本身在 x86_64 或 macOS 工作站上執行；`wtune post vapor-bundle` 產生的是可以複製過去用的資料包。

**叢集**

- 每個 job 都會避開登入節點 `head`。其他要避開的節點用 `wtune project set --exclude-nodes` 或 `$WTUNE_SBATCH_EXCLUDE` 指定。
- job 被砍或逾時之後，節點上可能留下空轉的 `wrf.exe`，會嚴重拖慢之後在同一節點上的量測。
