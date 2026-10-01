# wtune — profile-guided, compliance-aware WRF performance and physics configuration autotuner (decision 0050)

> 第一次使用？直接跳到 [教學：從零開始用 WTune](#教學從零開始用-wtune)。

WTune = profile-guided, compliance-aware WRF performance and physics configuration autotuner
(decision 0050). The full autotuner pipeline — profile → diagnosis → ceiling → candidates →
experiments → correctness gate → compliance classification → selection — is being built in
steps. Landed so far: **step 1** (the compliance model: `wtune_compliance.py`, `wtune.py
compliance`, `--compliance` on `knobs`/`physics`/`screen`), **step 2** (the phase
fingerprinter), **steps 4-5** (the configuration predictor, the STRICT and PHYSICS-GRAY
campaigns, the fidelity gate) and **research step D1** (the decomposition space and its
active search). What is not yet built is the automatic end-to-end loop: the pieces below are
run individually, and the selection at the end is always the user's. The one exception is
opt-in: `wtune run --tune-report` (section 6) applies a predicted-mode tune report's
per-segment choice, and only when the user passes that report.

`wtune` does four things to a WRF run, **none of which modifies a WRF source file**:

1. **Research oracle (RESEARCH)** — `plov/`: link-time `--wrap` interposition of the
   microphysics and radiation drivers, row staging into `/dev/shm`, node-local work stealing
   with a feedback controller. **-13.0% s/step over 400 steps, bit-identical** (job 1070876).
   This is an upper bound on execution-flow interposition, not a STRICT-tuning result — see
   Compliance model below.
2. **Pick the launch configuration** — `wtune.py plan/physics/knobs/screen`: decomposition
   shape, tiles, physics-consistent knobs, `srun --distribution=cyclic`, `KMP_BLOCKTIME=0`.
   Recommendations carry a STRICT or PHYSICS-GRAY compliance label; filter with `--compliance`.
3. **Gate result-affecting knobs** — `wtune_config_gate.py`: `radt`, `non_hydrostatic`,
   `cudt`, … legitimately change the answer, so they are compared field-by-field against a
   reference `wrfout` with a tolerance *calibrated* from a known-acceptable pair. This is the
   fidelity gate that PHYSICS-GRAY knobs must pass; bit-identity (`diffwrf`) is the gate for
   everything else.
4. **Predict runtime** — `wtune_runtime_predict.py` / `wtune_cost_model.py`: total runtime to
   ~5% (nested LOO MAE 1.07 s) from a `wrfout` snapshot; cost(case, config) as a recomputable
   lookup plus a GBRT residual.

Everything that was measured and found null lives in `deprecated/` (kept runnable, not
extended) and every one-off study in `research/`. See the layout at the end.

---

## 教學：從零開始用 WTune

這一節是動手教學：從建立 project 資料夾，到跑完後出報告。各指令的完整旗標與實測數字在後面的第 1–7 節；本節只給最短可用路徑。以下指令都寫成 `wtune ...`，實際一律透過 `bin/wtune` 啟動（見「Installation」）。

### 1. WTune 是什麼、分成哪幾段

WTune 不改 WRF 原始碼，只做三件事：跑之前挑好啟動設定與預測時間、跑的時候分段執行並監看、跑完用既有工具出圖並和觀測比較。

```
 ┌──────────── 跑之前 ────────────┐   ┌─── 跑的時候 ───┐   ┌─────────── 跑完 ───────────┐
 check / plan → predict → tune   →   run / watch      →   post recipe / fetch / 報告
 檢查設定        預測時間   找最快設定    分段執行、監看         出圖、和觀測比、整理資料
       ↑                                                          │
       └────────── project 資料夾：輸入、設定、run、圖全部放在一起 ──────┘
```

**核心規則：** WTune 只會自動調整不改變結果的設定（result-neutral）：MPI decomposition、tiles、OpenMP 執行緒數、I/O。物理方案、time step、垂直層數、nesting 一律只給建議（ADVICE ONLY），不會替你改。唯一事先授權的例外是 `wtune run --cfl-rescue`（CFL/NaN 失敗時用 0.75 倍 dt 重跑一次，並標記為 PHYSICS-GRAY）。細節見下方「Compliance model (decision 0050)」。

### 2. 安裝

照下方「Installation」做即可（pixi 環境 + 視需要的 WRF/WPS 路徑 + 後處理工具）。只想預測與調校，裝第 1、2 層就夠；要用 `wtune post` 的完整功能，另外需要 MET（TC 誤差、Grid-Stat、Point-Stat）、ParaView（3-D 雲渲染）與 Py-ART（NEXRAD 組合反射率，已在 pixi 環境內）。缺工具時指令會明確指出缺哪個並指向 `bin/install-postproc.sh`，不會靜默跳過。

```bash
bin/wtune --version
```
看到版本號代表 pixi 環境可用。

### 3. 建立 project 資料夾

一個 project 資料夾收納同一個個案的輸入、設定、每次 run 與所有圖。之後的子指令會自動找到它（`--project DIR`，否則 `$WTUNE_PROJECT`，否則從目前目錄往上找 `project.yaml`）。

```bash
wtune project init ~/projects/katrina-20260928 \
    --case /path/to/katrina_case --name katrina-20260928 \
    --storm AL122005 --hurdat2 hurdat2-1851-2024.txt --site scc
cd ~/projects/katrina-20260928
wtune project info
```
`project info` 會列出各子目錄內容與 `manifest.jsonl` 的筆數。`--case` 的來源目錄永遠只讀：namelist 會被複製（可自由編輯），wrfinput／wrfbdy／met_em 等以 symlink 連過來。

```
katrina-20260928/
  project.yaml      # 個案、站點、風暴、觀測、時間窗描述
  wtune.yaml        # tune 的搜尋空間設定（wtune config init 產生）
  case/             # namelist 副本 + 其餘檔案的 symlink
  input/            # 驅動資料與觀測（ERA5、HURDAT2、obs/<name>/ ...）
  tune/             # wtune tune 輸出
  predict/          # wtune predict-rebuild 輸出
  runs/<id>/        # wtune run 輸出
  post/<id>/        # wtune post 輸出（圖、表、report.html）
  analysis/         # 自己的 Python 分析腳本（wtune post custom）
  manifest.jsonl    # 每個 wtune 指令一行：時間、版本、argv、輸入 sha256、輸出目錄、exit code
```

在 project 內，`--out` 預設為各子目錄下的 `<id>/`，RUN_DIR 預設為 `runs/*/`（沒有則用 `case/`）。以下各指令都假設你在 project 內。

### 4. 跑之前

**4.1 `wtune check`：提交前檢查。** 在 WRF 啟動目錄上檢查時間窗、iofields、物理方案所需資料檔、decomposition、磁碟，以及 wrfinput 的土壤溫度／濕度／NaN。

```bash
wtune check case/ --ranks 448 --json check.json
```
看輸出的 FATAL／WARN 行：FATAL 代表這個 wrfinput 或設定會讓 run 失敗，先修再往下走。若偵測到 WRF 擴展，還會多印一行支援說明（moving nest、WRF-Chem、WRF-Hydro、WRFDA 各一行，內容見第 7 節）。

**4.2 `wtune plan`：decomposition 候選。** 由 namelist 與機器拓樸排出 MPI decomposition 短名單。

```bash
wtune plan --namelist case/namelist.input --nodes 4 --json
```
看排名最前的 `nproc_x × nproc_y`；它只是候選，要不要採用由 tune 實測決定。

**4.3 `wtune predict-rebuild`：整段 run 的牆鐘時間預測。** 在目標設定上，用 ERA5 重建數個時間點，各跑 WPS + real.exe + 短 spin-up + 若干量測步，再合成整段時間。這會對叢集送 Slurm job，先用 `--dry-run` 看計畫。

```bash
wtune predict-rebuild case/ --nodes 4 --nx 16 --ny 28 --numtiles 4 --omp 2 --dry-run
wtune predict-rebuild case/ --nodes 4 --nx 16 --ny 28 --numtiles 4 --omp 2
```
輸出在 `predict/<id>/pred_rebuild.json`，並印出建議的 `--time`（含 I/O 與餘裕）。限制：對對流尖峰時段會低估（短 spin-up 長不出組織化對流，見 elevconv 的實測）；不支援 adaptive time step；WRF-Chem／Hydro／WRFDA／moving nest 一律拒絕。

**4.4 `wtune tune`：在時間預算內找最快設定。** 在叢集上做 probe → screen → confirm，只搜尋 STRICT（不改結果）的旋鈕。`case` 是 `campaign/cases.json` 註冊的 case id。

```bash
wtune tune conus25 --minutes 60 --project .
```
`--minutes` 是整個 tune 的牆鐘預算（沒給時必須由 `--config` 的 `budget.minutes` 提供）。

*節點數搜尋（0.7.0-dev 新增，尚未在叢集上驗證）：* `--nodes` 的預設值現在是 `search`：在每個合法節點數上量測 ANCHOR 設定，依目標選出節點數，再在 finalist 節點數上做完整搜尋。

```bash
wtune tune conus25 --minutes 90 --nodes search --node-objective time
wtune tune conus25 --minutes 90 --nodes search --node-objective cost --deadline-minutes 240
wtune tune conus25 --minutes 90 --nodes search --node-objective knee --min-efficiency 0.7 \
    --node-finalists 2 --sweep-budget-frac 0.4
```
- `--node-objective time|cost|knee`：`time` = 最快（雜訊內取最小 N，預設）；`cost` = node-hours 最少，需搭配 `--deadline-minutes`（整段 run 的牆鐘上限）；`knee` = 效率仍 ≥ `--min-efficiency`（預設 0.70）的最大 N。
- `--node-finalists`（預設 2）：幾個節點數進入完整搜尋；`--sweep-budget-frac`（預設 0.4）：節點掃描最多用掉預算的比例，用完後較大的 N 被跳過。

固定節點數用 `--nodes N`；不量測、以未校準的效率代理挑一個節點數用 `--nodes auto`（搭配 `--nodes-policy min|cap|knee`）。

*依預測整段時間排名：*

```bash
wtune tune conus25 --minutes 120 --rank-by predicted --project .
```
`--rank-by predicted` 以 ERA5 重建、spin-up 後的共享 restart，依預測整段 compute+init 排名，而不是只看 t0 的 s/step；此模式的 `--nodes` 預設為 `auto`。

*讀 `tune_report.json`*（在 `tune/<id>/`，同樣內容也會印在 stdout）：
1. 找最佳設定：decomposition、numtiles、OMP、`--distribution`，以及它對 ANCHOR 的相對時間。
2. 看 `node_selection`：`--nodes search` 時會記錄採用的目標、選中的節點數與理由；節點表列出每個 N 的 median s/step、重複次數、spread、speedup、efficiency 與 node-hours，被跳過的 N 附原因。
3. 對照三個目標（time／cost／knee）各自會選的 N；若 time 與 knee 相差很大，代表加節點的邊際效益已經很低。

**4.5 `wtune physics`：物理與數值建議（僅建議）。**

```bash
wtune physics --namelist case/namelist.input --menu --compliance strict --no-probe
```
輸出是診斷與 scheme 選單，每條帶 STRICT／PHYSICS-GRAY 標籤；WTune 不會替你改 namelist。`--no-probe` 略過 SpeedyWeather 對流 probe（較快，但少了對流負載地圖）。

**4.6 用 YAML 設定 tune。** `wtune config` 管理 tune 的搜尋空間；優先序為 CLI 旗標 > YAML > 內建預設，未知 key 是錯誤（打錯字不會被默默忽略）。

```bash
wtune config init conus25 --out wtune.yaml
wtune config validate wtune.yaml
wtune config show wtune.yaml
wtune tune --config wtune.yaml
```
```yaml
site: scc
case: conus25
compliance: strict          # strict | physics-gray
budget:
  minutes: 60
  max_nodes: 13
  nodes: search             # auto | search | <int>
  node_objective: time      # time | cost | knee
search:
  omp: [1, 2, 4, 8]
  numtiles: [1, 4, 8, 16]
  rank_distribution: [block, cyclic]
probe:
  rank_by: t0               # t0 | predicted
```
`validate` 通過、`show` 印出展開後的搜尋空間即可。完整的 key 與註解見 `examples/wtune.yaml`；`node_objective` 等節點搜尋 key 的名稱以該檔與 `wtune config validate` 為準。

### 5. 跑的時候

**`wtune run`：分段執行。** 把整段 run 切成數個 segment，以 restart 交接，每段各送一個 Slurm job，避免單一長 job 撞時間上限。

```bash
wtune run case/ --nodes 4 --nx 16 --ny 28 --numtiles 4 --omp 2 \
    --wall-limit 04:00:00 --segment-minutes 180 --prediction predict/<id>/pred_rebuild.json --dry-run
wtune run case/ --nodes 4 --nx 16 --ny 28 --numtiles 4 --omp 2 \
    --wall-limit 04:00:00 --segment-minutes 180
```
先用 `--dry-run` 確認 segment 切法；輸出在 `runs/<id>/`。若有預測模式的 tune 報告，可用 `--tune-report tune/<id>/tune_report.json` 讓每段採用預測最快的候選（需明確傳入才會套用）。

CFL/NaN 失敗時預設**停下來**，並寫出建議的 `time_step` 與 `segment.advice.namelist.input`，不會自動重送。加上 `--cfl-rescue` 才會以 0.75 倍 dt 重跑該 segment 一次（標記 PHYSICS-GRAY）；這是唯一被預先授權的例外，且不適用於 Chem、Hydro、moving nest。

**`wtune watch`：唯讀監看。**

```bash
wtune watch runs/<id>/ --prediction predict/<id>/pred_rebuild.json --interval 60
wtune watch runs/<id>/ --once
```
看進度、剩餘時間，以及 HANG_INIT（初始化後長時間沒有 d01 Timing 行）、巢狀域停止與失敗警告。`--once` 只印一次快照後離開。

### 6. 跑完：後處理

`wtune post` 全部是唯讀建議：只出圖與表，不改 namelist、不送 job。

**6.1 一鍵出報告：`wtune post recipe`。**

```bash
wtune post recipe --project . --types auto
wtune post recipe --project . --types auto --every 6h --animate
wtune post recipe --project . --types tc,convection --no-paraview
```
`--types auto` 依個案內容自動判斷類型；個案類型是**多標籤**（例如同時是 `tc` 與 `convection`），`common` 一律會包含。`--every 6h` 每 6 小時一組完整圖，放在 `post/<id>/timeseries/`；`--animate` 另做每個欄位的 MP4（至少兩個時間，需要 ffmpeg）；`--no-paraview` 略過 3-D 雲渲染。結果在 `post/<id>/report.html`，頂端的 notes 框會列出繞過的問題（追蹤 fallback、缺觀測、離線抓取、缺工具）。

**6.2 抓觀測與再分析資料：`wtune post fetch`。** 要在能連外網的節點上執行（scc 的計算節點可連外；離線時 `recipe` 會印出缺哪些資料與對應的 fetch 指令）。

```bash
wtune post fetch --list
wtune post fetch stage4 --project .
wtune post fetch nexrad --project . --run-dir runs/<id>/
wtune post fetch-era5 --project .
```
`--list` 列出模組（`era5`、`era5_monthly`、`gpcp_monthly`、`igra`、`mrms`、`nexrad`、`stage4`、`stations`）與涵蓋時間。每個模組寫入 `input/obs/<name>/` 並附 `provenance.json`（提供者、URL、授權、抓取時間、引用格式）；`fetch-era5` 等同 `fetch era5`。

**6.3 和觀測比較。** `recipe` 抓到資料後會自動比：降水對 Stage IV、對流對 NEXRAD（Py-ART）或 MRMS 組合反射率（MET Grid-Stat）、測站 T2／風（Point-Stat）、探空對 IGRA2、氣候類個案對 ERA5 月平均與 GPCP。比較範圍排除邊界 spec_bdy_width。報告內有「資料來源與授權」一節，每張圖的圖說也註明來源與授權。

**6.4 颱風：`wtune post tc`。**

```bash
wtune post tc runs/a/ runs/b/ --labels ctrl,strict \
    --hurdat2 hurdat2-1851-2024.txt --storm AL122005 --track-smooth-km 30
wtune post report post/<id>/ --title "Katrina: ctrl vs strict"
```
用 TempestExtremes 追蹤，b-deck 優先序為 `--bdeck`、NHC 下載、HURDAT2。位置取自經 CDO 平滑的 SLP（`--track-smooth-km`，預設 30；0 = 原始最低點，大眼時會在 mesovortex 間跳動），MSLP 與 Vmax 取自未平滑的場。注意追蹤警告：退回父域的時間點、網格粗於約 5 km 且無巢狀域、風暴離開 nest；另有 `track steps` 行報告步長中位數與曲折度。沒有 MET 時只會產生 a-deck、`quadrant_<label>.csv`、`obs.json`、`track_map.png`，誤差圖會被跳過並警告。

**6.5 其他工具。**

```bash
wtune post inventory runs/<id>/ --json
wtune post extract runs/<id>/ --vars slp,t2,wspd10 --points "25.4,-74.6,eye" --out points.csv
wtune post plot runs/<id>/ --field slp --overlay wind --out figs/
wtune post sounding runs/<id>/ --point 25.4,-74.6 --time 2005-08-27_00:00:00 --out skewt.png
wtune post compare runs/a/ runs/b/ --vars T2,slp
wtune post custom --project .
```
- `inventory`：網格、變數、缺格檢查。
- `extract`：用 wrf-python 抽點位時序或區域 netCDF。
- `plot`：單一欄位地圖，預設 cartopy 底圖。
- `sounding`：MetPy Skew-T 與 CAPE/CIN。
- `compare`：呼叫 `diffwrf` 比對共同的 frame；用來驗證 STRICT 變更沒有動到答案（要看 `diffwrf` 的位元一致結果）。
- `custom`：執行 `analysis/*.py`，輸出進 `post/<id>/`；`project init` 會放一個範例腳本。

### 7. WRF 擴展

`wtune check` 會偵測 moving nest、WRF-Chem、WRF-Hydro、WRFDA。對這些個案 WTune 只調整網格與 rank 配置（ranks、decomposition、tiles、OpenMP、I/O），擴展自身的參數一律照原樣不動。

| 擴展 | WTune 做什麼 | WTune 拒絕什麼 |
|---|---|---|
| moving nest | 網格／rank 調校；時間預測係數 +6.0%（唯一實測過的） | `predict-rebuild`、`--cfl-rescue` |
| WRF-Chem | 網格／rank；OMP 上限 1；decomposition guard：最小 j-patch 須 ≥ 21 | `predict-rebuild`、`--cfl-rescue` |
| WRF-Hydro | 網格／rank；OMP 上限 1 | `predict-rebuild`、`--cfl-rescue`；standalone Hydro 不支援 |
| WRFDA | forecast 段照一般 WRF 調校；`da_wrfvar.exe` 的 rank／decomposition 只給建議、不自動套用（最小化的歸約順序隨切法改變，結果不會逐位元相同）；OMP 上限 1 | `predict-rebuild`；`wtune run` 需 `--wrfda-forecast-only`（不跑 obsproc／da_wrfvar／da_update_bc，DA cycling 被拒） |

- Chem 與 Hydro 的時間預測係數只是暫定值（±100% 區間），未校準；僅在該個案沒有 probe 量測時才套用，可用 `$WTUNE_EXTENSION_COEFFS` 覆蓋。
- Chem 的 j-patch 門檻 21 來自單一個案、每格只跑一次的探索性結果（GOCART 在 j-patch 過窄時於第一次歷史寫出 hang），屬暫定。
- ndown、NMM、MPAS 不支援。
- 在 scc 上建置：`bin/build-wrf-extensions.sh submit movenest|chem|hydro|wrfda`（經 `sbatch --exclude=head`）。WRFDA 是唯一用 gcc 建置的（`COMPILER=gnu`），其餘用 nvhpc；Chem／Hydro 要建 dm+sm 並以 `OMP_NUM_THREADS=1` 執行。WRF-Chem 目前只有不含 KPP 的 build（`CHEM_KPP=0`，可跑 GOCART 等非 KPP 機制）：nvfortran 26.5 在 `-O3 -fast` 下編不完 `module_kpp_mozart_mosaic_4bin_Integr.f90`。

### 8. 常見問題與已知限制

- **adaptive time step：** `predict-rebuild` 不支援（`time_step` 為 0 或 `use_adaptive_time_step`）；`wtune physics` 只給建議。
- **對流尖峰低估：** 預測用短 spin-up 的 ERA5 重建，長不出組織化對流，尖峰時段的成本會低估；整段預測誤差因此在對流個案偏樂觀，不要把它當成上限。
- **WRF-Chem 間歇 hang：** 會偶發卡住。每個 job 都設時間上限（`--wall-limit`），逾時後重新提交，不要空等。
- **scc 計算節點可連外網：** `post fetch` 與 ERA5 下載可以直接在計算節點上跑。
- **逾時後殘留 wrf.exe：** job 被砍後可能留下空轉的 `wrf.exe` 孤兒程序，量測前先檢查並清理該節點（見 `campaign/node_hygiene.sh`），否則會污染計時。

---

## Compliance model (decision 0050)

Invariant, unchanged from before: **no WRF `.F`/`.c`/`.h` is ever edited.**

Every knob/lever WTune touches is classified into exactly one of three classes
(`wtune_compliance.py`); **the three classes are never ranked together**:

- **STRICT** — WRF source unchanged, normal build, simulation semantics unchanged
  (decomposition, tiling, placement, binding, OpenMP/MPI runtime, compiler flags, I/O layout).
- **PHYSICS-GRAY** — WRF source unchanged, official namelist/physics only, but the
  numerical/physical solution may change (scheme swaps, `radt`, `cudt`, hydrostatic, ...).
  Recommend and test behind a fidelity gate; **passing a WTune fidelity gate is NOT an
  official ruling** — PHYSICS-GRAY results are reported as **UNCONFIRMED**.
- **RESEARCH** — changes the execution flow of the binary/application (PLOV, `--wrap`,
  `LD_PRELOAD`, owner/helper work migration). Upper bound / oracle only. `-Wl,--wrap`,
  `LD_PRELOAD`, and PMPI shims are **RESEARCH — not allowed in STRICT tuning.**

Every catalog entry (`wtune_compliance.CATALOG`) carries: `name`, `kind`, `classification`,
`source_change`, `numerics_change`, `execution_interposition`, `requires_correctness_gate`,
`applicable_phases`, `hardware_requirements`, `status`, `evidence`. `numerics_change` maps to
the correctness gate that class must clear:

| `numerics_change` | meaning | `requires_correctness_gate` |
|---|---|---|
| `none` | output expected bit-identical to the reference | `diffwrf` |
| `roundoff` | same algorithm, floating-point roundoff may differ | `roundoff_gate` (tight field tolerance) |
| `solution` | the computed solution legitimately changes | `fidelity_gate` |

`plov/gate0.sh` proves from `objdump` that the driver call sites go through the `__wrap_*`
trampolines; `diffwrf` against an unmodified run proves the output is unchanged — together
these are the RESEARCH-path proofs for PLOV.

```bash
python3 wtune.py compliance list --compliance strict
python3 wtune.py knobs --namelist namelist.input --compliance strict
```

### What WTune changes automatically vs. only recommends (decision 0056)

**Auto-applies (no separate approval per run): STRICT only.** `wtune tune`'s final recommended
configuration (`wtune_tune.build_report`) is guarded by `wtune_compliance.assert_strict_only`,
which diffs every namelist key the report changed against the case's own namelist and raises
`ComplianceError` if any changed key classifies as anything other than STRICT (measurement-only
probe-window keys — `run_*`, `start_*`, `end_*` — are exempt, since narrowing a probe window does
not change the solution the full run produces).

**Advisory only: PHYSICS-GRAY and RESEARCH.** `wtune run` never silently changes a solution
knob. Without `--cfl-rescue`, a CFL/NaN segment failure stops the run and writes the recommended
`time_step` next to the failed segment as `segment.advice.namelist.input` (never applied,
never resubmitted) plus a log line explaining why the change is PHYSICS-GRAY.
`--cfl-rescue` is the **only** user-pre-authorized exception: it re-runs the same failed segment
once at `0.75x` its `time_step`, and still marks the run's report `physics_gray: true`.
`wtune post`'s own subcommands (below) never write a namelist or submit anything — they only
read wrfout files and produce figures/tables from what already ran.

---

## 1. PLOV — the live rebalancer (`plov/`)

The load imbalance in WRF is in the physics (convective microphysics, the solar terminator in
radiation), and it moves every step. Nothing that acts at restart boundaries or through rank
placement can follow it (`deprecated/README.md`). PLOV acts inside the step:

- the `--wrap` trampoline captures the driver's 402/422 arguments, re-enters the real driver
  with the owner's `JME` shrunk by R rows (pass 1), then runs the R staged rows from a
  `/dev/shm` slot (pass 2) — either itself or an idle rank on the same node that claimed
  them (CAS PUBLISHED->CLAIMED, single writer, so the result is bit-identical);
- R adapts per rank from measured driver time, `R += 0.5*NJ*(t_me - t_mean)/t_me`, chunked
  in 6, capped at 75% of the patch ("adapt75"); no prediction is needed;
- `srun --distribution=cyclic` spreads the hot region over the four nodes first (-2.2% on
  its own), because stealing is node-local.

| arm (f1, 4 nodes x 56 x 2 threads, 1500x1500, 400 steps) | s/step | delta | bit-identical |
|---|---:|---:|---|
| A: block placement, unmodified | 0.3349 | — | reference |
| **Ec: cyclic + micro rows=4 adapt75** | **0.2913** | **-13.0%** | yes |
| ER: Ec + radiation rows=4 + row stealing | 0.2925 | rad steps -0.15..-0.23 s each (6/6 negative, sum -1.06 s); -0.6% net | yes |

Plus, independent of PLOV: `KMP_BLOCKTIME=0` on 224x2 is **-2.8%** (ABAB), and the hybrid
224x2 with the right `numtiles` beats 448x1 by 5.7%.

Per-step budget after PLOV (0.292 s): dynamics ~143 ms, MPI floor 37 ms (13 halo closures x
2.8 ms, source-locked), radiation ~46 ms averaged, microphysics ~34 ms (at its arithmetic
ceiling), residual imbalance ~18 ms (between nodes; cross-node stealing is bounded at
<=1.3%), PBL+surface ~11 ms. Whether ER (radiation) becomes production is still an open
decision; it is measured and safe.

Build, run, verify: `plov/README.md`; the exact environment: `plov/production.env`.

## 2. Advisory layer — `wtune.py`

```bash
python3 wtune.py plan    --namelist namelist.input [--ranks N] [--wrfinput wrfinput_d01]
python3 wtune.py physics --namelist namelist.input [--menu] [--probe-json speedy_probe.json]
python3 wtune.py knobs   --namelist namelist.input [--all]
python3 wtune.py screen  --namelist namelist.input     # 3-stage funnel to the effective few
python3 wtune.py audit   --namelist namelist.input     # classify all ~2062 catalog params
python3 wtune.py catalog [--group physics] [--grep mp_]
python3 wtune.py hotspot --namelist namelist.input --wrfinput ... [--run-surrogate]
python3 wtune.py feedback ...                          # PLOV controller replay / decisions CSV
python3 wtune.py selftest
```

- **Decomposition** — cache · vectorization · halo · NUMA score over the legal
  `nproc_x x nproc_y` shapes; re-derives the measured CONUS 2.5 km winner (`8x56`) without
  running the sweep (`wtune_decomp.py`, `wtune_shape.py` for the patch geometry).
- **Physics suite + knobs** — 20 input-derived rules over a Registry-generated catalog of
  ~2062 parameters (`num_soil_layers`<->LSM, `radt`≈dx, PBL<->sfclay pairing, `cu_physics` vs
  dx …). Knobs are labelled by what they can change: `code_enforced` (WRF overrides them),
  performance-only (`numtiles`, decomposition, I/O), and **RESULTS-affecting numerics**
  (`radt`, `non_hydrostatic`, `cudt`, …) which `wtune.py` prints under "gated tuning only".
- **Hotspot preview** — analytic solar terminator (`wtune_solar.py`) plus the SpeedyWeather
  convective surrogate (`wtune_surrogate.py` + `speedy_hotspot.jl`; always run the probe).
  Used to *explain* a case; PLOV does not consume it.
- **Transition controller** — `wtune_transition_controller.py` (copied from `../g0a/`):
  zero-measurement initial config and node-scaling projection (4N->8N measured -44.6%,
  projection error 1%; blocked by the 4-node rule). Its restart-boundary imbalance gate is
  deprecated; the 2/40 selftest failures are both in that branch.

Lat/lon/date come from a `wrfinput`/`wrfout` header via `ncdump -h` when the namelist lacks
them.

## 3. Calibration gate for result-affecting knobs — `wtune_config_gate.py`

Config-layer changes such as `radt` 10->15 or `non_hydrostatic` are worth up to -15% on a
15 km case but change the fields, so bit-identity cannot judge them. The gate compares a
candidate `wrfout` with a reference field by field (`max|d|`, RMS of the difference, RMS
relative to the field) and passes when every field is within a tolerance that was
**calibrated from a pair the user already accepts** (two runs of the accepted config, or
reference vs an accepted change):

```bash
PY="pixi run --manifest-path ../wtune-pixi-env-20260830/pixi.toml python"   # numpy + netCDF4
$PY wtune_config_gate.py calibrate --ref accepted_A/wrfout_d01_... --cand accepted_B/wrfout_d01_...
#   -> prints the per-field table and a suggested --tol-rel (max rel_rms x safety 3)
$PY wtune_config_gate.py check --ref ref/wrfout_d01_... --cand cand/wrfout_d01_... --tol-rel <that> --json gate.json
#   -> GATE PASS / GATE FAIL <fields>, exit 0/1
```

One variable is read at a time (files are ~800 MB), the last time record by default. Run it
on a compute node like everything else. A bit-identical pair reports every field at 0 and
passes at any tolerance.

## 4. Prediction layer — runtime and cost

Features come from a `wrfout` snapshot aggregated per rank patch (`wtune_oracle.py`;
`featurize_frames.py` reduces a 1-2 GB frame to 448 x 17 and deletes it; `wtune_segment_train.py`
aligns segmented features to segmented labels).

| tool | predicts | validation |
|---|---|---|
| `wtune_runtime_predict.py` | **total runtime**, case level | nested LOO **MAE 1.07 s (~5.2%)**, 1.9x better than the mean; best single feature = radiation load |
| `wtune_runtime12.py` | same on the 12 km corpus | — |
| `wtune_cost_model.py` | cost(case, config) = f(mp) + g(bl) + B(weather) | separable; the per-PBL head is **REFUTED on a diverse corpus (2026-09-05)** and stays opt-in via `WTUNE_PER_BL_HEAD` |

Per-rank prediction is a scale problem, not a model problem: at 12 km/36 ranks the per-rank
spread is at the profiling noise floor; at 3 km/448 ranks it is learnable (within-case
Spearman +0.44..+0.85). But no live actuator consumes "which rank is hot" — PLOV measures
it — so the per-rank hotspot model itself is in `deprecated/` (decision 0045).

### Cold-start sampling (`tools/coldstart_sample.py`) — unknown-case prediction

For a case with only a `namelist.input` (+ a projection source) and NO wrfinput/wrfbdy/restart
files, `wtune_predict.py`'s probe-based model has nothing to measure from. `coldstart_sample.py`
closes that gap: it picks K "characteristic point" sampling times inside the run, fetches only
the GFS GRIB times those windows need, stages ONE sbatch running geogrid -> ungrib -> metgrid ->
real.exe (one per window) -> wrf.exe (one short cold-start window per sampling time, all at the
configuration being predicted), then scores the windows and prints a whole-run wall-time
prediction. Each window is an independent cold start from GFS state at time `T_k` (not a
restart — an unknown case has none). Measured on the cluster (conus12, 3 times x paired with
true restarts): cold-start cost equals restart cost within 0.35% once the first ~1 simulated
hour is discarded, so every window budgets a spin-up discard before it starts measuring.

    python3 src/tools/coldstart_sample.py inspect --case-dir D [--namelist-wps P | --geo-from-wrfinput P] [--grib-dir G]
    python3 src/tools/coldstart_sample.py plan --case-dir D --out PLAN_DIR --k 4 --nodes N --ranks R --omp T --nx X --ny Y
    python3 src/tools/coldstart_sample.py fetch --plan PLAN_DIR         # login node; needs network
    python3 src/tools/coldstart_sample.py stage --plan PLAN_DIR --site scc
    sbatch PLAN_DIR/run.sbatch
    python3 src/tools/coldstart_sample.py score --plan PLAN_DIR --json PLAN_DIR/score.json

`inspect` names every blocker explicitly (no projection source; a run time older than the GFS
0.25deg archive's 2015-01-15 start with no `--grib-dir`; `ifire>0` — WRF-SFIRE unsupported; a
nest whose start time differs from d01's; `sf_lake_physics>0` with no `lake_depth` dataset under
geog; `sf_urban_physics` 2/3 — BEP/BEM need urban parameter data geog doesn't carry; `chem_opt>0`
— WRF-Chem emissions; `obs_nudge_opt>0` — observation files) and, separately, what is
completable but "filled differently" than the case's own namelist asked for (hydrometeors/number
concentrations start at zero — always, from GFS via Vtable.GFS — which is *why* cold-start
windows discard a spin-up hour; `mp_physics=28` with no aerosol climatology wired in; a
`num_land_cat` mismatch against whichever landuse dataset the geog directory actually has;
`sf_surface_physics=3` with `num_soil_layers!=4`; `grid_fdda>0` nudging files only covering the
sampled windows). The geog capability check reads the real `geog_data_path` directory listing
(`--geog-path`, else `$WPS_GEOG`/`/nfs/data/wrf/WPS_GEOG/`) rather than a hardcoded dataset list,
so it degrades to "skipped" (never a false blocker) on a site whose mirror can't be listed, and
still works on a site whose geog mirror differs from scc's. Projection (namelist.input carries
none) comes from `--namelist-wps`/`D/namelist.wps` first, else `--geo-from-wrfinput`/
`D/wrfinput_d01`'s global attributes — generalised from the proven-on-cluster, conus12-hardcoded
`.route1-ref/make_namelist_wps.py` recipe, dropping its nested-case MOAD_CEN_LAT refusal (d01's
own MOAD_CEN_LAT always equals CEN_LAT, so that check never distinguished a real nest; nest
placement comes from namelist.input's own per-domain lists instead). Namelist edits reuse the
same packed-line-safe `set_list`/`count_key` approach `.route1-ref/gate2_stage_window.py` proved
on the cluster (key-count invariants; a `history_interval*` key that doesn't already exist is
never added; `fix-real-namelist`'s `num_land_cat` overwrite from met_em is required by real.exe
and stays, but now prints old->new instead of changing it silently). `ungrib`/`metgrid` run once
PER WINDOW (`fix-wps-namelist` narrows namelist.wps `&share` start/end to that window's own
`[t, t_end]` first — running them once against the whole run's span makes metgrid walk every
interval and die on the first GRIB time `fetch` never downloaded), and a window that fails at
any step is recorded `ok=0` in `windows_status.txt` and does not abort the remaining windows;
`score` skips `ok=0` windows and errors only if none are usable. `run.sbatch` sources
`campaign/sites/scc.sh` for `site_env`/`site_launch`/`site_hygiene` and
`campaign/node_hygiene.sh` for orphan reap/dirty-node census rather than reimplementing either —
`--exclude=head`, `--signal=B:USR1@90` (USR1/TERM/INT kill the in-flight run, reap, record
`INTERRUPTED`, and exit 3), no `LD_PRELOAD`, no wrapper around `wrf.exe`. `tools/metdata.py` and
`tools/grib_state.py` (both need numpy, which scc's compute-node `python3` doesn't have) are
imported lazily and only by `inspect`/`plan --state`/`fetch` — every compute-node subcommand
(`fix-wps-namelist`, `fix-real-namelist`, `fix-wrf-namelist`, `stage`, `score`) never touches
them. `score` classifies radiation steps from each window's own data
(`classify_steps_detected`), fits a day/night radiation ratio (`fit_day_night_r_rad`), and
linearly interpolates each window's measured step cost across the full run between window
measured-part midpoints; I/O and init come from `wtune_predict.build_prediction` against
`PLAN_DIR/win_0/real` (the generated wrfinput), never the original case_dir, which in an unknown
case has none and in a held-out validation case would be leakage.

Tests: `tools/test_coldstart_sample.py` (65 cases, offline/no network — namelist.wps field
derivation including per-window time narrowing, nested-domain list copying and sbatch linking,
inspect blockers and capability-gap warnings driven by a real tmp geog listing, window selection
on a synthetic 72h run, the `dt=72/radt=15 -> stepra=13` window-size arithmetic, wrf-window
namelist key-count invariants, score's spin-up exclusion + exact trapezoid interpolation +
`ok=0` window skipping + case_dir non-leakage, the USR1-trap/TOOL-path/GRIB-fallback shape of
the generated sbatch text, and that every compute-node code path stays importable with numpy
blocked).

## 5. Measurement — where the time goes

- **Per-rank driver timers** come free with PLOV: `wtimer.<rank>.csv` under `WTIMER_DIR`
  (per-call microphysics/radiation/PBL/surface/cumulus time, staged/refused/shared counters,
  wait time). A run whose counters read `staged 0` did nothing, however fast it looks.
- **PMPI shims** (RESEARCH; `profilers/blockbar_prof.c`, `blockbar_phase.c`, `LD_PRELOAD`):
  per-rank wait fraction and per-segment compute/wait, because APS and `I_MPI_STATS` both
  hang at 448 ranks. `wtune_probe.py` is the ABBA submit/parse harness.

Three profiling traps, all hit: launch with `srun --mpi=pmi2` (plain `srun` leaves Intel
MPI as `comm_size=1` singletons; diagnostic `ls rsl.out.* | wc -l`); never align phases by
MPI-call index across ranks (neighbour count differs per rank; Fortran collectives cannot be
intercepted); bound windows by wall time, not call count.

### Phase fingerprinter (autotuner step 2, validated; for calibration / second-pass profile runs only)

`fingerprint/` observes an unmodified `wrf.exe` from outside: `sampler_node.sh`
attaches `perf record` (user-mode cycles sampling) to its pids, and
`fingerprint.py` time-weights the resulting `perf script` samples into per-phase
shares (MICRO/RADIATION/PBL/SURFACE/CUMULUS/DYNAMICS_OTHER/MPI/IO), a
`fingerprint.json`, and diagnosis labels. No `--wrap`, no `LD_PRELOAD`.

    python3 fingerprint/fingerprint.py --sampler-dir OUTDIR --run-dir RUNDIR \
      [--omp-threads N] [--wtimer-dir DIR] --out OUTPREFIX

`--wtimer-dir` compares pooled shares against in-process `wtimer.<rank>.csv`
timers (ground truth only). Validation job 1074484 (karina 224x2, 4 nodes, 100 Hz,
`POLL_S=5`): out-of-sample share error micro 0.71 / radiation 0.30 / dynamics 0.96 /
MPI 1.87 pp, diffwrf identical; overhead 1.40% (gate was <=1%). 2026-09-14 user ruling:
overhead accepted because the fingerprinter is used only for second-pass profile runs,
never on production timing runs.

### Configuration predictor and STRICT data campaign

`campaign/strict_campaign.sbatch` measures the 16 STRICT templates of `campaign/configs.py`
per case (all-ADIOS2 I/O, palindromic replicates) into `ledger_<case>_<job>.csv`;
`predictor/run_benchmark.sbatch` builds the ledger (`predictor/ledger.py`) and runs the
grouped-CV benchmark (`predictor/benchmark.py`). First cf448 campaign: jobs 1075293-1075304;
benchmark 1075654 -- leave-one-case-out median regret catboost_pairwise 0.4% vs naive (always
C01) 15.2% over 6 cases, but conus12 and leave-one-hardware-out do not generalise. The
predictor outputs a ranked shortlist; the final configuration is the user's decision.

`predictor/recommend.py` is the user-facing shortlist command (STRICT and PHYSICS-GRAY); see
`predictor/README.md` for flags, evidence warnings, and status rules. It has two objective
modes, `--objective perf` (default) and `--objective accuracy` (`OBJECTIVES` at
`predictor/recommend.py:1502`): perf-first ranks on cost subject to a fidelity constraint,
acc-first ranks on error and reports what the accuracy costs. Both emit a Pareto front over
(cost, error) rather than a single answer.

**The one invariant to preserve when touching physics templates**: the repo has *two* template
engines and they must agree. The dict engine
(`predictor/physics.py:physics_candidates_for_case`) reads the `physics` dict from
`campaign/cases.json` and never opens a namelist; the text engine
(`campaign/configs.py:resolve_physics_config`) edits the real `namelist.input` that WRF runs.
`predictor/tests/test_physics.py:596` asserts they agree on applicability and on every changed
key's target value across six real cases and every template. The known asymmetry is documented
there at `:640-643` — the dict engine cannot know the native value of a namelist-only knob
(`swint_opt`, `aer_opt`, `use_aero_icbc`), so the *from* value is compared only for
`physics.CHANGE_FIELDS`.

### I/O: a guarantee, not a tuning axis

All WRF I/O goes through ADIOS2 (`io_form 14`). The codec is not searched: `campaign/configs.py`
writes `adios2_compression_enable=.true.` and `adios2_blosc_compressor="zstd"` into every run's
`&namelist_adios2`, because zstd wins on both axes measured (write 0.468 s vs lz4 0.496 s vs
1.014 s uncompressed; 166.7 MB vs lz4 181.1 MB vs 595.9 MB for the same frame).
`adios2_numaggregators` **must** equal the MPI rank count — values at or below the node count are
silently ignored.

`tools/bp_compression_check.py` asserts that compression actually applied, from the ratio of
logical to stored bytes, because this ADIOS2 build carries no codec marker in BP metadata at all.
It is needed: in this WRF build only the *first* ADIOS2 write stream of a run is compressed
(`external/io_adios2/wrf_io.F90:1283-1285` assigns the codec name only on the branch that defines
the operator), so `--history-mode interval` silently writes frames 2+ uncompressed. That is a WRF
source bug, not a namelist one, and it is not patched here.

### PHYSICS-GRAY data campaign and fidelity gate (decision 0051)

- `campaign/physics_templates.json`: 19 fixed templates in three families, plus an `excluded`
  list with reasons, derived from a WRF 4.8 source compatibility survey. Each template carries a
  `direction` field (`cost` or `accuracy`) that ties it to the objective modes below.
  - **P00-P06 — physics** (`direction: cost`). P00 reference; P01 `swint_opt`; P02 `radt`×2+`swint_opt`;
    P03 `time_step`×1.2; P04 WSM6; P05 PBL+sfclay swap; P06 RRTM/Dudhia.
  - **D01-D07 — `&dynamics` cost levers** (`direction: cost`): advection options, `rk_ord`=2,
    output diagnostics off, `bldt`, one-way nesting, `time_step_sound`.
  - **A01-A05 — accuracy mirrors** (`direction: accuracy`): the same knobs pushed the other way, so
    `recommend.py --objective accuracy` has candidates to rank.
- `strict_campaign.sbatch MODE=physics` (env `PCONFIGS`, `DECOMP`, `TWIN_DECOMP`, `SIM_MINUTES`, `HIST_MINUTES`, `CONFIG_FILE`) runs three kinds of run:
  - reference P00 at DECOMP;
  - a same-physics twin at TWIN_DECOMP;
  - each candidate at DECOMP.
  `CONFIG_FILE` is the only way to run a case that has no template of its own — it supplies the
  decomposition pair directly (`campaign/configs.py decomp-pair`); see
  `campaign/strict_campaign.sbatch:38-46`.
- Replicate 1 writes interval history (fidelity), replicate 2 is timing only. Every candidate is scored by `fidelity_gate.py`.
- `fidelity_gate.py --reference --twin --candidate --family --out-prefix` computes, per field:
  - the chaos-floor ratio RMS(ref,cand)/RMS(ref,twin)
  - the normalized distance
  - bias and precipitation accumulation difference

  It gives a provisional, overridable label WITHIN-FLOOR / SMALL-DRIFT / LARGE-DRIFT / INVALID. The label describes model self-drift, not observed skill; acceptability is the user's call.

  **Read `floor_zero=True` as "the floor was not measured", not "the candidate is fine."** Only
  `nproc_x` changes WRF's arithmetic (the default tiling splits in J alone), so a twin that keeps
  the reference's `nproc_x` has a chaos floor of exactly zero; WITHIN-FLOOR then degenerates into
  "bit-identical to the reference" and every perturbation reads as LARGE-DRIFT.

### Decomposition space and active search (research step D1)

`predictor/decomp_space.py` enumerates the full legal stock-WRF decomposition and tiling space per case with WRF's exact patch and tile rules (`enumerate`), draws a stratified ground-truth sample that `strict_campaign.sbatch CONFIG_FILE=...` can measure (`sample`), and emits a cold-start design of two shapes per thread count — most balanced and widest-in-x — for a case with no measurements at all (`seed`).

`predictor/decomp_search.py` is the D3 active search: a bootstrap CatBoost ensemble with LCB acquisition proposes each round's batch, stopping when the remaining predicted gain falls inside the timing noise floor. `campaign/decomp_search_campaign.sbatch` runs seed → rounds → confirm → rank in one allocation; `campaign/phase_ranking.py` ranks per window from `rsl.error.0000` and settles the verdict with a co-winner rule inside the noise band, and `campaign/decomp_summary.py` groups results by the decomposition WRF actually used.

### Judging accuracy against observations, not against another model run

`predictor/verify_obs.py` is the only path in the repo that scores a configuration against something outside WRF. Two modes: `station` compares nearest-gridpoint T2 against an hourly observation CSV (RMSE, bias); `track` locates the storm centre from SLP and compares it with an HURDAT2 best track (great-circle km, `pmin` hPa, `vmax` m/s). `case` runs whichever the case declares. Everything else in this repo — including the fidelity gate — measures model self-drift, so this is what makes `--objective accuracy` mean anything.

`predictor/stability.py` replicates WRF's acoustic sub-step formula (`dyn_em/solve_em.F:451-464`) and reports the acoustic CFL margin before a run, which is what makes the D07 `time_step_sound` template safe to propose per case. Its documented limit: acoustic CFL does **not** explain the P03 `time_step`×1.2 crashes.

## 6. Whole-run prediction, monitoring and segmented execution (0.2.0)

Four subcommands of `bin/wtune` (`wtune.py <cmd>`), plus preflight rules. None edits a WRF
source file or injects anything; none writes inside `CASE_DIR`. Every sbatch they generate
carries `--exclude=head[,$WTUNE_SBATCH_EXCLUDE]`. All evidence below is scc (72 ranks/node),
2026-09-27; errors are against a truth run of the same case and configuration.

### `wtune predict-rebuild` — whole-run wall time from rebuilt segments (decision 0054)

```bash
wtune predict-rebuild CASE_DIR --nodes N --nx X --ny Y [--numtiles T] [--omp O]
    [--forcing era5|auto] [--spinup-min 20] [--max-points K] [--measure-steps N|short]
    [--rain-snow] [--no-io-probe] [--out DIR] [--dry-run]
```

Places up to K points on the run's timeline (default 5; on the forcing's hourly grid,
`T + spin-up <= P`), rebuilds each from ERA5 with cloud water/ice (ARCO-ERA5 written straight to
WPS intermediate, no ungrib; `--forcing auto` = GFS chain with a 60-min spin-up), runs WPS +
real.exe + a 20-min spin-up + `max(30, 2*stepra)` measured d01 steps **on the target
decomposition**, one Slurm job per point, all submitted at once, then combines the points into
compute+init (`wtune_probe_plan.combine`: per-point normal-step cost, day/night radiation ratio).
`OUT/pred_rebuild.json` is written before anything is compared with it. Needs the site env
(`WPS_DIR`, `WRF_TREE`) and `CASE_DIR/wrfinput_d01` (geometry and ERA5 crop box).

**I/O is measured, not modelled** (`--io-probe`, default on): the last point's segment also
writes one history frame per domain (+ one restart if the case writes restarts) with the case's
own `io_form`/`iofields`; I/O = Σ frames × timed write per domain. The printed **recommended
`--time`** = range_high + 2 × I/O + margin (10% of range_high, clamped to 2–5 min), rounded up to
1 min below 2 h, else 5 min. Factor 2: history writes are 21–48% of wall on scc and the same
volume varies up to 2.4× there.

| case | run, config | predicted compute+init | truth | error | same-day drift-corrected | driver wall |
|---|---|---:|---:|---:|---:|---|
| conus25 | 6 h, 4 nodes 12×24 nt 8 | 1800.9 s | 1783.4 s | +0.98% | not measured | 19:40 (jobs 1309–1313; 51% of the 38:40 run — just run it) |
| maria1km | 3 h, 4 nodes 16×18 | 4268.3 s | 4270 s (job 977) | −0.04% | −2.19% (job 1321) | jobs 1325–1327 in parallel |
| zhengzhou | 30 h, 3 domains, 4 nodes 16×18 | 15970.2 s | 16949.8 s (job 1020) | −5.78% (−5.67% rerun) | ≈ −2% (job 1320) | 2244 s (mostly queue); 804.7 s = 4.5% of wall with 3 points |
| lafe | 24 h, 1000×1000, 8 nodes 24×24 | 13637.8 s | 13616.1 s (job 975) | +0.16% | +0.41% (job 1346) | 32:41 (jobs 1343–1345, serialised on 12 free nodes) |

- The same run on scc varies ±2–4% between days, in both directions (zhengzhou 3.4–4.1% faster,
  maria1km 2.2% slower). Validate a prediction against a same-day control on the original input.
- One point at t0 misses the diurnal radiation: lafe −10.5%, conus25 −9.6%.
- `--spinup-min 10 --measure-steps short` (`max(30, stepra+10)` steps, radiation flagged by
  known phase `n % stepra == 1`) is opt-in. It was picked in-sample on zhengzhou (627 s = 3.5% of
  wall); out-of-sample it moves the prediction −1.11% (dorian2019, one clean pair) and +0.38%
  (lafe, driver 829 → 687 s) against the default window.

`--time` check T1 (truth wall ≤ recommended ≤ 1.5 × wall):

| case | measured I/O | truth Σ writes | I/O error | recommended `--time` | ÷ truth wall |
|---|---:|---:|---:|---:|---:|
| dorian2019 (6 h) | 299.1 s | 300.5 s | −0.5% | 38:00 | 1.33 (1718.0 s) |
| maria1km | 389.2 s | 391 s | −0.5% | 1:37:00 | 1.25 (4661.5 s) |
| zhengzhou | 905.2 s | 877.8 s | +3.1% | 5:25:00 | 1.09 (17827.6 s) |

**Not yet validated:** the first rule (fixed 5-min margin, 5-min rounding) failed T1 on dorian
(45:00 = 1.57×); the scaled margin was chosen after seeing that, so the passing column is
in-sample. zhengzhou's compute range_high (17387.4 s) is *below* its real wall — only the 2× I/O
covers it, so do not lower that factor. The out-of-sample check on elevconv2022 (15 h, 8 nodes,
writes 21% of wall) passed the `--time` check (5:20:00 = **1.25×** the 15377 s wall) and the I/O
probe (3134.9 vs 3260.2 s, −3.8%); compute+init was −6.8% (in range). The short window vs the
old window on the same day: −1.14%. **Known limit:** elevconv's per-step cost rises 44% between
05 and 11 h as convection develops; the rebuilt point at 07:35 measured ~15% below the truth at
that hour (20 min of spin-up from ERA5 does not grow organised convection) and the 10 h peak lay
between points. Typhoon vortex bogussing (`tools/tc_bogus.py`) failed its criteria and is opt-in.

### `wtune watch` — read-only run monitor

```bash
wtune watch RUNDIR [--interval 60] [--once] [--json] [--prediction pred_rebuild.json]
    [--init-limit 600] [--hang-factor 10] [--hang-min 300] [--replay-fraction F[,F...]]
```

Reads only `namelist.input` and `rsl.error.0000`/`rsl.out.0000`; never writes, never signals.
Reports progress, remaining **compute and writes separately** (plus their sum), and warnings;
exit 3 on FAILED, 2 on HANG / HANG_INIT / NEST_STOPPED / NEST_END_EARLY, else 0 (CFL is
informational).

- A d01 step that absorbed `Timing for Writing` lines has those seconds subtracted; compute rate
  = mean compute over a window aligned to the radiation/history LCM period. Remaining writes =
  per-stream median seconds per frame × frames still due per the namelist (history, restart,
  auxhist, per domain); a stream not written yet uses `pred_rebuild.json`'s measured io block.
- `--prediction`: remaining compute = k × predicted compute over the remaining steps, k =
  observed ÷ predicted so far — the prediction supplies the trend, the run its level.
- HANG_INIT caught the known silent hang (elevconv, missing iofields file); NEST_END_EARLY caught
  yagi truth 1038 (nest `end_*` not updated, d01 kept advancing).

Replay on scc truth logs at 25% progress, error of estimated vs actual remaining (wtune 3ee3808):

| case | rate only | with `--prediction` |
|---|---:|---:|
| zhengzhou / dorian2019 / lafe | −0.7% / −1.1% / −3.3% | −1.8% / −1.2% / +0.6% |
| maria1km / conus25-g0014 | −2.9% / −3.8% | −1.7% / −6.4% |
| marshall / isc21 (30 min) / c34 (truncated log) | −6.2% / +0.4% / −0.7% | — |
| elevconv (576 ranks, out of sample) | **−15.6%** | **−13.6%** (fails: the prediction itself misses the convective peak) |

This is the third attempt at a pre-registered ≤10% criterion (first: 3 of 9 failed; second
dropped write-absorbing steps as outliers, marshall −53.5%). The with-prediction column is
in-sample: those predictions were made against the same truths. Rate-only cannot see a rising
per-step cost (elevconv, convection developing) — use `--prediction` for long runs. Zero false
positives across healthy runs was part of the criterion and has not been reported separately.
Limits: rank 0's rsl only, so an error printed on another rank is not seen; files are reopened
and re-stat'ed every poll because a long-lived NFS handle can serve stale content, and HANG is
only as fresh as the login node's view of the rsl mtime.

### `wtune run` — segmented execution with restart hand-offs

```bash
wtune run CASE_DIR --nodes N --nx X --ny Y [--numtiles T] [--omp O]
    [--wall-limit HH:MM:SS | --segment-minutes M] [--stop-at YYYY-MM-DD_HH:MM:SS]
    [--prediction pred_rebuild.json] [--tune-report tune_report.json]
    [--restart-io-form 14|2] [--cfl-rescue] --out OUT [--dry-run]
```

A login-node controller: segment k runs in `OUT/seg_k` as its own Slurm job (inputs symlinked;
`--out` may not be inside `CASE_DIR`), is inspected with `watch`, and hands off to k+1 through
its restart files. `OUT/run_report.json` is rewritten after every attempt; any failure other
than the rescue case below stops with an error, never a silent retry.

- **One restart quantum L for the whole run.** Every segment is an L block from the run start
  (the last may be shorter) and `restart_interval = L` in every segment. `--segment-minutes M`
  sets L = M; `--wall-limit W` picks the largest L for which every segment's predicted wall
  (compute + init + restart read/write + history frames inside it) ≤ 0.8 W; neither = one
  segment. Segment `--time` = 1.5 × predicted wall + 10 min (5-min steps), capped at W; after
  each segment its observed overhead replaces the prediction in later `--time`s, never the grid.
- **`override_restart_timers` is never set here**, so WRF restores every alarm's phase from the
  restart file. With it (first R1 attempt), d01's 60-min history alarm re-anchored at the 00:30
  boundary and the 01:00 frame was never written. The tuner's candidates need the opposite (next
  section): they must *not* inherit the spin-up's restart interval.
- Every active domain's `start_*`/`end_*` is set per segment (yagi 1038: a nest whose `end_*` was
  left alone stopped early). Restarts default to ADIOS2 BP4 via `adios2.xml` (io_form 14).
- `--tune-report`: each segment takes the candidate a predicted-mode tune report predicts
  cheapest over that segment's interval (default: one config throughout). A switch that changes
  `nproc_x` changes roundoff, so such a run is not bit-identical to a single-config run.
  On hardware (dorian2019, 30 + 30 min, 16×18 then 12×24 nt 8 from the first config's ADIOS2
  restart, jobs 1531 + 1534): both segments completed with the intended configs. The 01:00 output
  differs from a 16×18 single run by T max 1.9 K / RMS 0.007 K — exactly as much as a 12×24 run
  from t0 differs from a 16×18 run (job 1535). The switch adds no error beyond choosing a
  decomposition: on this build decompositions are **not** bitwise (roundoff grows in convection).
- `--cfl-rescue` (off by default, **PHYSICS-GRAY**): a segment that failed with a CFL line or NaN
  is rerun once per run from the same restart with `time_step` = the largest integer in
  [0.5, 0.75] × dt dividing the segment length and L; later segments return to the case's dt.
  On hardware (dorian2019 copy, dt 120 s): WRF printed `points exceeded v_cfl` from step 2, the
  segment was rerun once at 90 s and flagged PHYSICS-GRAY, that also exceeded CFL, and the run
  stopped with an error (jobs 1537/1538) — detection, one rescue and a clean stop all work. dt 30 s
  and 45 s ran cleanly on dorian's first 30 min, so one 0.75× step cannot rescue a time step that
  is far off; that conservatism is deliberate.

Evidence (dorian2019 first hour, 4 nodes 16×18): **R1** 30 + 30 min chain (jobs 1405 + 1410) vs
one-shot (job 1392): 01:00 `wrfout` for d01 and d02, **202 variables each, bit-identical**, no
NaN. **R3** `--dry-run` prints the plan and submits nothing. **R2** (each segment's wall inside
its predicted range): first attempt failed — predicted 478/578/548 s vs measured 326/198/168 s,
because the prediction had no io block and restart/history fell back to constants; fallbacks now
scale from measured scc anchors (conus25 restart, dorian2019 history) by grid points × levels.
Rerun with an io-block prediction (`r2_chain`, jobs 1409 + 1420): predicted 197.1 / 173.1 s vs measured 201 / 164 s (**+2.0% / −5.3%**; segment 1's estimate updated from segment 0's observed rate) — **R2 passes**.

### `wtune tune --rank-by predicted` — rank by predicted whole-run time (decision 0055, opt-in)

```bash
wtune tune CASE --site scc --minutes 60 --nodes 4 --max-nodes 12 --rank-by predicted \
    [--max-points 3] [--max-wall-minutes M] [--restart-io-form 14|2] --out DIR
```

The default (`t0`) is unchanged. In predicted mode:

1. Plan points as `predict-rebuild` does (ERA5; spin-up = smallest whole minute ≥ 20 on the
   time-step grid, dt 36 s → 21 min) and rebuild every point once (real_only, 1 node each, in
   parallel — real.exe output does not depend on the candidates' decomposition).
2. Stage 1 at P* (the point with the largest sunlit fraction): each Slurm job spins up once and
   every candidate in it starts from that shared restart; ANCHOR runs first and ANCHOREND last in
   every job, and each run is normalised by its own job's anchors. Jobs with foreign load or a
   run that did not start from the restart are excluded.
3. Stage 2: top 3 + ANCHOR/ANCHOREND at every point in parallel, one `combine()` per candidate;
   the smallest predicted compute+init wins. Gates raise on: restart not used, |ANCHOREND ÷
   ANCHOR − 1| > 1%, foreign load, wrong first global step.

Candidate runs set `override_restart_timers = .true.`: gate G0 found that with
`restart_interval = 1000000` WRF still inherited the spin-up's 20-min restart interval and wrote
a restart inside every measured window (netCDF 299 s on a radiation step, r ≈ 5 → 121; ADIOS2
0.6 s). `run_seconds` is zeroed next to `run_minutes`, because WRF adds all four `run_*` keys.

Budget: `--minutes` counts **running** time only (queue time is reported separately; hard wall
cap `--max-wall-minutes`, default 3 × budget). Before anything is submitted a feasibility check
computes the least running budget (rebuild prep + one stage-1 job of ANCHOR + 1 candidate +
ANCHOREND + stage 2 for that one candidate) and refuses with that number. A stage-1 job starts
only if elapsed + its own estimate + the stage-2 reserve fits; stage-1 jobs are trimmed to fit
(floor ANCHOR + 1 + ANCHOREND); pending stage-1 jobs at the cutoff are cancelled and reported as
not measured; stage 2 drops its lowest candidates until it fits, or is skipped with the reason.
All estimates use one cost model: per run = startup + restart read + steps × s/step ×
(1 + ⌈steps/stepra⌉ (r_rad − 1)/steps), r_rad 3.93 (conus25, 82 steps: 1.107; the earlier blanket
×1.5 overstated it ~1.36×), with per-run overhead re-measured after the first wave.
`--max-points` defaults to 3 in this mode.

| gate / attempt | result |
|---|---|
| **G0** restart hand-off across rank counts (jobs 1371 io 14, 1372 io 2; conus25, 4 nodes, 82 steps after 20 min) | **pass.** ADIOS2 restart read at 288/144/72 ranks; candidate ÷ ANCHOR differs between io forms by ≤ 0.38% (gate 2%); in-allocation drift −0.18% / −0.11%; first global step 81 in all 10 runs. Refutes the code comment that io_form 14 restarts are decomposition-dependent. |
| e2e 1 (conus25-pred1) | **failed on budget**: queue time consumed the 60 min (6 free nodes, five 4-node rebuild jobs ran one at a time, ~22 min); stage-2 jobs 1393–1397 cancelled while PENDING |
| e2e 2 (conus25-pred2) | **failed**: rebuild took 7 min, then every stage-1 slot was skipped (7:04 + 22:26–26:11 per job + 37:39 reserve > 60 min) — nothing measured |
| e2e 3 (conus25-pred3) | **incomplete**: stage 1 and stage 2 ran one job at a time on a busy cluster; the stage-2 reserve had assumed 3 concurrent point jobs, and the running budget cut stage 2 midway → stage 2 now runs to completion once started, concurrency from observation |
| e2e 4 (conus25-pred4, stage 1 at P*, 90 min) | **G1 pass** (6 ranking jobs, anchor drift ≤ 0.96%), **G2 pass**: winner G0001 predicted 1997 s vs same-day full run 1982.1 s (**+0.75%**, job 1462). **G3 fail**: G0001 is 10.6% slower than the t0 tuner's G0014 (1791.8 s, same job and nodes) — stage 1 at P* afforded only 3 candidates and G0014 was never in the pool |
| e2e 5 | stopped during stage 1 (same narrow-pool problem; superseded by the budget split) |
| e2e 6 (conus25-pred6, stage 1 = t0 search, `--minutes 30` for stage 1 only) | stage 1 30.0 min + stage 2 58.0 min (announced 78.6) + 2.1 min, queue 2:33. Winner **G0001** again (predicted 33:19). Within the same candidate pool the t0 ranking and the whole-run ranking picked the **same** config; G3 vs G0014 still fails on pool breadth (one 4-node job at a time → ~6 candidates in 30 min) |

**Conclusion (decision 0055):** `--rank-by predicted` stays **opt-in**; `t0` is the default. Its
measured value is an accurate whole-run prediction of the winner (+0.75%); on conus25 it did not
change the choice — consistent with 0.1.0's finding that the best decomposition does not move
with simulated length. Recommended pipeline: `wtune tune` (t0) → `wtune predict-rebuild` on the
winner (1 config × 3 points, ~15 min, same prediction quality) rather than stage 2 (5 configs ×
3 points, ~58 min). A case whose top t0 candidates are close and whose cost regime changes over
the run is what would be needed to show a ranking benefit.

Node hygiene (every command in this section): `WTUNE_SBATCH_EXCLUDE=work5[,...]` adds nodes to
`--exclude=head`.
On 2026-09-27 scc work5 carried a root VS Code server outside Slurm (91% of one core) plus an
`rm -rf`, which slowed a 576-rank `wrf.exe` from 1.25 to 6.3 s/step; `--exclusive` does not stop
that. `campaign/node_hygiene.sh`'s census now prints `FOREIGN_LOAD` for any process not owned by
the job's user with lifetime %CPU > 20 and elapsed ≥ 30 s (kernel threads ignored).

### `wtune check` — physics tables and wrfinput sanity

- **C3 physics tables follow WRF 4.8's own `OPEN` statements**, not `README.physics_files`,
  which over-lists; each rule cites its source line. E.g. Ferrier (5/95) opens only
  `ETAMPNEW_DATA.expanded_rain`; `CCN_ACTIVATE.BIN` only for aerosol-aware Thompson (28/38); P3
  table 1 matched by prefix, table 2 only for nCat > 1 (52); full SBM (32) but not fast SBM (30,
  a different file set); ozone on d01 for RRTMG with `o3input=2` and on any CAM domain;
  `aerosol*.formatted` only for `aer_opt=1` on d01; `URBPARM_LCZ.TBL` *replaces* `URBPARM.TBL`
  when `use_wudapt_lcz=1`; GFDL (co2tf-dependent), Goddard, eclipse, slab/RUC/CLM/PX/SSiB,
  `HLC.TBL` for chem.
- **C10 wrfinput sanity uses land/water masks.** TSLB on land only: exactly 0 is real.exe's
  "undefined" fill → WARN; non-zero outside [200, 350] K → FATAL above 0.1% of the domain's land
  TSLB values, WARN below. SST on water only (exact 0 skipped: inland lakes); SMOIS < 0 → WARN,
  > 1 → FATAL; TSK everywhere; any NaN/Inf in T/U/V/QVAPOR/TSLB/SMOIS/PSFC → FATAL. The 0.1% line:
  dorian2019 d02, 20/(14806×4) = 0.034%, ran 1 h with no NaN (job 1392) → WARN; tw3km-2031 case_I,
  13186/(37588×4) = 8.8%, went NaN by hour 1 while printing SUCCESS → FATAL. On scc: 11 healthy
  truth dirs zero FATAL, case_I FATAL, case_A clean. The first C10 was false-FATAL on zhengzhou
  and dorian2019, both of which ran.
- **No "small patch hangs" rule.** The hypothesis (sc26p2 17×13 hangs at init with 10-cell
  patches) is refuted: 102 successful ledger runs had a minimum patch ≤ 10 cells (matthew 9×10
  and severe2023 10×10 at 9 cells, 97 and 109 runs; sc26p2 2×12 at 10 cells, 30 runs). Only
  sc26p2 17×13 hangs (7 of 7). The guard is per-candidate timeouts plus `watch` HANG_INIT.

### WRF extensions (0.6.0)

`wtune check` prints one support line per detected extension: WRF-Chem, WRF-Hydro, WRFDA and
moving nest. For these WTune tunes **grid and rank layout only** (ranks, decomposition, tiles,
OpenMP threads, I/O) and leaves every extension parameter exactly as given (the compliance
catalog marks them do-not-touch).

- **OpenMP is capped at 1** for Chem, Hydro and WRFDA (build them dm+sm and run with
  `OMP_NUM_THREADS=1`; a dmpar-only Chem/Hydro build crashes nvfortran 26.5).
- **Cost factors are uncalibrated except moving nest.** Moving nest +6.0 % is measured; Chem
  and Hydro are placeholders with a +/-100 % band until calibrated
  (`src/wtune_extension_coefficients.json`, override with `$WTUNE_EXTENSION_COEFFS`). A factor is
  applied only when no probe measurement exists for the case.
- **Refusals:** `predict-rebuild` on any of the four; `wtune run` on WRFDA cycling
  (`--wrfda-forecast-only` runs forecast segments); `--cfl-rescue` on Chem, Hydro or moving nest;
  ndown, NMM, MPAS and standalone Hydro are unsupported. A build that lacks the extension the
  namelist asks for is caught (moving nest via the `med_nest_move` symbol).
- CFL advice prints coupling warnings for `chemdt`/`photdt`, `DTRT_TER`/`DTRT_CH`,
  `vortex_interval`.
- `bin/build-wrf-extensions.sh submit movenest|chem|hydro|wrfda` builds them through
  `sbatch --exclude=head` (`SKIP_CONFIGURE=1`, `TIME=`, `BUILD_MODE`, `USE_ADIOS2`,
  `CHEM_KPP`); the WRFDA build is still pending.

## 7. Post-processing and analysis (`wtune post`, decision 0056)

Advisory only — no autotuning, and never writes a namelist or submits anything. Every
subcommand wraps an established tool instead of re-implementing it (2026-09-29 update:
`post/tools.py` locates each one via `WRF_TREE`/`PATH`/`WTUNE_POSTPROC_VIEW` and fails with a
clear message naming the missing tool + `bin/install-postproc.sh` below, never a silent
self-written fallback). Heavy imports are all lazy, inside each subcommand's own function, so
`wtune --help` and `wtune post inventory`/`compare` never need wrf-python/matplotlib/cartopy
installed.

| subcommand | wraps |
|---|---|
| `inventory` | `ncdump -h` (grid/variables), `cdo sinfon` (lon/lat extent cross-check); the missing-frame cadence check is the only self-written piece (WRF/WTune-specific, no tool audits it) |
| `extract` | wrf-python (the NCAR tool for WRF diagnostics — slp/mcape/level interpolation etc.) |
| `plot` | wrf-python (`wrf.get_cartopy`/`wrf.latlon_coords`) + cartopy Natural Earth coastlines/borders; falls back to a LANDMASK contour (and prints that it did) only when the Natural Earth cache is missing, never on a cartopy import failure |
| `sounding` | MetPy (`SkewT`, `surface_based_cape_cin`, `parcel_profile`, `shade_cape`/`shade_cin`) |
| `compare` | `diffwrf` (WRF's own netCDF differ, `$WRF_TREE/external/io_netcdf/diffwrf`) |
| `tc` | TempestExtremes (tracking, unchanged), MET `tc_pairs`/`tc_stat` (track/intensity/wind-radii error vs. an ATCF b-deck), MET `tc_rmw` (azimuthal-mean radius-height structure, on a `cdo remapbil`-regridded frame) |
| `report` | stdlib only (glue: collects the other subcommands' PNGs/CSVs into one HTML) |

```bash
wtune post inventory RUN_DIR [--json]
wtune post extract RUN_DIR --vars slp,t2,wspd10 --points "25.4,-74.6,eye" --out points.csv
wtune post extract RUN_DIR --vars z,tc --levels 850,500 --box 20,30,-80,-70 --out box.nc
wtune post plot RUN_DIR --field slp --overlay wind --out figs/          # --basemap cartopy is the default
wtune post sounding RUN_DIR --point 25.4,-74.6 --time 2005-08-27_00:00:00 --out skewt.png
wtune post compare RUN_A RUN_B --vars T2,slp     # diffwrf on every common frame; verifies a STRICT change did not move the answer
wtune post report OUT_DIR --title "case X"       # collect OUT_DIR's PNGs/CSVs into one HTML
```

`wtune post tc` — TempestExtremes tracks the run (Vmax/MSLP come straight from its own track,
no self-written re-derivation), `post/tc_metrics.py`'s quadrant_radii() computes the model's own
34/50/64 kt wind radii (the one surviving self-written metric — see its module docstring for why:
no tool derives a model's OWN quadrant radii from a WRF wind field), and `post/atcf.py` writes
the result as an ATCF a-deck (the one justified new module — no tool converts a TempestExtremes
track to ATCF; verified column-for-column against a real NHC archive file in
`tests/fixtures/post/bal122005_sample.dat`). The b-deck is `--bdeck FILE`, else downloaded from
the NHC ATCF archive, else (last resort) HURDAT2 converted through the same ATCF writer. MET
`tc_pairs`/`tc_stat`/`tc_rmw` then compare a-deck vs. b-deck:

```bash
wtune post tc RUN_A RUN_B --labels ctrl,strict --hurdat2 hurdat2-1851-2024.txt \
    --storm AL122005 --bdeck bal122005.dat --ebtrk ebtrk_al_1988_2023.txt --out tc_report/
wtune post report tc_report/ --title "Katrina 2005: ctrl vs. strict decomposition"
```

Always written (needs only TempestExtremes + wrf-python): `adeck_<label>.dat`,
`quadrant_<label>.csv`, `obs.json` (parsed HURDAT2 + EBTRK), `track_map.png` (drawn from the
a-deck/b-deck files directly). Written only if MET is installed: `tc_pairs.tcst`,
`tc_stat_track.stat`/`tc_stat_wind.stat`, and the error-vs-lead and radius-height figures drawn
from them — a missing MET tool is reported in the command's `warning:` output, not a hard
failure, so the a-deck/track-map part of the report is still usable without MET.

MET's exact `tc_pairs`/`tc_stat`/`tc_rmw` config syntax in `src/post/met/*.conf.tmpl` is based on
MET 12.1.1's documented config structure but **has not been run against a real binary** — MET
was still building (spack job 1575) when this was written. Diff the templates against
`$WTUNE_POSTPROC_VIEW/share/met/config/TC*Config_default` once installed and fix any mismatch.

### `wtune post recipe` — case-type-driven standard figure set (decision 0057)

```bash
wtune post recipe RUN_DIR [RUN_DIR ...] --out DIR [--types auto|tc,convection,...] \
    [--ensemble] [--spinup-hours 6] [--hurdat2 F --storm ID | --bdeck F] [--ebtrk F] \
    [--labels a,b] [--repro-pairs] [--no-paraview] [--title "..."]
```

Detects which case type(s) the run matches — **overlapping types are never narrowed to one; every
matching type's figures are produced** — and writes one self-contained HTML report
(`post/report.py`'s `build_recipe_report`) whose header lists the detected types and the exact
evidence for each (e.g. `convection: d02 dx=4 km, cu_physics=0`).

| type | auto-detect condition | reads from |
|---|---|---|
| `common` (always) | — | domain/nest map (wrf-python `terrain` + cartopy), reproducibility-noise envelope (diffwrf, only with `--repro-pairs` and >=2 RUN_DIRs) |
| `tc` | a HURDAT2/b-deck best-track fix falls inside d01 during the run window (needs `--hurdat2`/`--storm` or `--bdeck`; without either, tc is never auto-detected) | `wtune post tc`'s own pipeline + a new 10 m tangential-wind radius-time plot |
| `convection` | any domain dx<=4 km with `cu_physics=0` | max reflectivity, MCAPE/MCIN, 2-5 km updraft helicity time-max (wrf-python `uhel`), a representative Skew-T (`wtune post sounding`) |
| `precip` | always | accumulated RAINC+RAINNC; obs-dependent figures (diff/ETS/FSS) are reported **missing** (Stage IV/MRMS/IMERG not fetched this round) |
| `synoptic` | d01 extent >= 1500 km or run length >= 24 h | 500 hPa Z+vorticity, SLP+1000-500 hPa thickness, 850 hPa theta-e+wind, 250 hPa jet, T2+10 m wind, an ERA5 z500 diff map (ARCO, `post/recipes/era5.py`) |
| `pbl` | `bl_pbl_physics=0` or dx<=1 km | a cross-terrain theta+wind cross-section (`wrf.vertcross`), a time-height theta+PBLH plot |
| `chem` | `chem_opt>0` | a concentration+wind map for whichever WRF-Chem species is present in the file |
| `climate` | run length >= 30 d | `cdo ymonmean`/`yseasmean` (the ERA5 side of the seasonal bias is not implemented this round) |
| `ensemble` | >=2 RUN_DIRs with `--ensemble` | a spaghetti plot, a probability-above-threshold map; MET `ensemble_stat` rank histogram/spread-skill needs obs wiring not implemented this round |

Every figure is captioned from `src/post/recipes/catalog.json` (zh + en: the question it answers,
how to read it, common misreadings) and stamped with run/domain/valid time and the excluded
spin-up length. `wtune post recipe --types tc` (or any explicit list) narrows the report to
exactly `common` + the given type(s), skipping auto-detection.

ParaView 3-D rendering (tc/convection): one wrfout frame is converted to two small CF NetCDFs by
`post/xwrf_convert.py` (xWRF `.postprocess()` + wrf-python's own `z` for a physical height field,
since the NetCDF's own `z` dimension is WRF's eta level index), then `post/paraview/render_cloud.py`
is run under `pvbatch --force-offscreen-rendering` (located via `WTUNE_PARAVIEW_BIN`, default
`envs/paraview/.pixi/envs/default/bin`) to contour a QCLOUD+QICE+QSNOW isosurface over a terrain
surface. **Not exercised against a real `pvbatch` this round** (ParaView is not installed locally)
— when `pvbatch` is missing, the figure is reported missing with that reason, same contract as a
missing MET tool. `--no-paraview` skips it outright.

**Known deviation from decision 0057** (flagged for review): the synoptic ERA5 z500 diff figure's
RMSE/bias is computed directly in numpy after a `scipy.interpolate.griddata` regrid onto ERA5's
grid, not through MET Grid-Stat as decision 0057 names — MET is not installed locally this round,
and an unverified Grid-Stat config would be no more trustworthy than the numpy path; see
`post/recipes/synoptic.py`'s module docstring. `gcsfs`/`zarr` are in `envs/wtune/pixi.toml`
(2026-09-29), so this figure now actually opens the anonymous public
`gs://gcp-public-data-arco-era5` zarr store (verified locally and on scc's login node) instead of
being reported missing.

Two more case types, added 2026-09-29 (decision 0057 update):

| type | auto-detect condition | reads from |
|---|---|---|
| `terrain` | any domain's HGT relief (max-min) >= 1000 m, or peak >= 1500 m | terrain+10 m/700 hPa wind map, a theta+w cross-section through the peak along the domain-mean 700 hPa wind (grid-index-direction approximation; `project.yaml`'s `cross_sections.terrain` overrides it), accumulated-rain-over-terrain-contours, a time-height theta diagram at the domain's lowest point. Lee-side station T2/RH/wind and MET Point-Stat are reported missing (no station list fetched this round) |
| `moisture` | precip, convection, or tc detected (i.e. effectively always on, since precip always is) | PW map + its 6 h `cdo sub` change, IVT (integrated vapor transport) + 850 hPa wind, an RH+theta_e cross-section through the TC centre or max-rain point (`cross_sections.moisture` override), a specific-humidity time-height diagram |

IVT is this package's one self-written integral (`post/ivt.py`) — neither wrf-python nor MetPy
exposes a moisture-flux integral (wrf-python's `pw` is the scalar precipitable-water integral
only). `IVT = (1/g) * integral of q*V dp`, via `numpy.trapezoid` — exact for a piecewise-linear
profile, so exact (not merely close) for the constant synthetic profile
`tests/test_post_ivt.py` checks against the closed form.

### `wtune post fetch` — observation and reanalysis data, and comparisons against it (0.6.0)

```bash
wtune post fetch --list                                   # modules, help, and what each needs
wtune post fetch stage4 --project P                       # one module into <project>/input/obs/<name>/
wtune post fetch nexrad --project P --run-dir R           # window/domain taken from R's wrfout
wtune post fetch-era5 --project P                         # alias of `fetch era5`
```

| module | source | limits |
|---|---|---|
| `stage4` | IEM archive of NCEP Stage IV | 2001 onward, CONUS only |
| `nexrad` | `unidata-nexrad-level2` (AWS) | 2005 volumes carry no radar position; taken from NCEI HOMR |
| `mrms` | `noaa-mrms-pds` (AWS) | from 2020-10-14 only |
| `stations` | ISD-Lite | point observations, converted with MET `ascii2nc` |
| `igra` | IGRA2 radiosondes | used by the `pbl` recipe |
| `era5`, `era5_monthly` | ARCO ERA5 (monthly is aggregated from the hourly store) | `synoptic` z500 diff, `climate` |
| `gpcp_monthly` | GPCP v2.3 | `climate` precipitation bias |

Every module writes `<project>/input/obs/<name>/provenance.json` (provider, URL, license, fetch
time, citation). `wtune post recipe` prints the fetchable datasets it found missing, with the
exact `wtune post fetch` command, when the machine is offline. Credentials, if a source ever
needs one, come only from environment variables. Compute nodes on scc do have internet.

Observation comparisons in the recipes, all restricted to the model interior (`spec_bdy_width`
excluded): `precip` is scored against Stage IV, `convection` against a NEXRAD (Py-ART) or MRMS
composite, both with MET Grid-Stat (CTC/CTS, FSS); station T2/wind through `ascii2nc` +
Point-Stat; `pbl` compares WRF with IGRA2 radiosondes (Skew-T overlay with the surface-based
parcel and CAPE/CIN shading, bulk-Richardson PBL height after Seibert et al. 2000 / Vogelezang
& Holtslag 1996, T/Td/wind-speed RMSE). Precip is also limited to within 230 km of a NEXRAD
site, because Stage IV is 0, not missing, offshore. The report's "資料來源與授權" section and each
figure's caption give source, provider, URL, license, fetch time and citation.

`ensemble` draws spaghetti contours at the highest member minimum + 10 hPa and probabilities in
discrete k/N bins with the boundary zone blanked. `climate` draws the annual cycle for four boxes
and seasonal bias maps against ERA5 monthly and GPCP (day-weighted seasons); the spin-up is
excluded with `--climate-spinup-days N`.

**Notes and warnings.** Anything the recipe had to work around (a tracking fallback, an
uncovered obs window, an offline fetch, a missing tool) is collected as a note, printed as
`warning: ...` and shown in a notes box at the top of `report.html`; a figure that could not be
made is listed as missing with its reason.

**Track quality.** TC tracking runs per output time on the finest domain whose interior contains
the storm, seeded from d01 and falling back to the parent with a recorded reason (per-time
domain in `track_domains_<label>.csv`). The position comes from CDO-smoothed SLP
(`--track-smooth-km`, default 30; 0 = raw minimum); MSLP and Vmax come from the raw fields.
Warnings cover fallback times, a grid coarser than about 5 km with no nest, and a storm leaving
the nest. A built-in line reports `track steps <run> (after 6 h spin-up)`: median step against
best-track speed, tortuosity, max/median step.

Time-series maps use one fixed scale per field across all times (true min/max for
slp/t2/wspd10/pw, pooled 2-98 % for mcape, rain from 0); change maps use symmetric levels with a
white zero bin.

### `wtune project` — one folder for a case's inputs, config, runs and figures (decision 0058)

```bash
wtune project init PATH [--case CASE_DIR] [--name NAME] [--storm AL122005] [--hurdat2 F] [--site scc]
wtune project info [--project PATH]
```

```
PATH/
  project.yaml        # case/site/storm/obs/time description + cross_sections.{terrain,moisture} overrides
  wtune.yaml           # unchanged tuner config (wtune config init)
  case/                 # namelist.input/output/wps COPIED (edit freely); everything else (wrfinput,
                         # wrfbdy, met_em, ...) SYMLINKED -- CASE_DIR is never opened for writing
  input/                # driving data / observations
  tune/  predict/  runs/<id>/  post/<id>/    # every subcommand's own output
  analysis/             # your own Python scripts (wtune post custom); project init drops in an example
  manifest.jsonl        # one line per command: time, wtune version, argv, key input paths (+sha256 <50MB), out dir, exit code
```

`wtune post recipe|tc|plot|custom` resolve a project via `--project DIR`, else `$WTUNE_PROJECT`,
else walking up from the current directory for `project.yaml`. Inside a project: `--out` defaults
to `post/<id>/` (`id` = a UTC timestamp + short tag), RUN_DIRs default to every `runs/*/` with a
`wrfout_*` frame (else `case/` itself), and `--hurdat2`/`--storm` default to `project.yaml`'s
`storm:` block. **Outside a project, nothing changes** — every `--out`/positional RUN_DIR stays
required exactly as before (decision 0058 P6).

**Time series** — `wtune post recipe --every 6h` (or `--times T1,T2,...`) adds a full
figure set (SLP, T2, 10 m wind, rain, +MCAPE if convection was detected) per selected valid
time, in `post/<id>/timeseries/fNNN_<YYYYmmddHH>Z/`, named:

```
<field>_d<domain>_fNNN_<YYYYmmddHH>Z.png        e.g. slp_d02_f018_2005082718Z.png
```

`fNNN` is the forecast hour (zero-padded, GFS-style, so lexical order is time order); the valid
time has no colon (ParaView's own reader fails on one — confirmed on scc, 2026-09-29). The
report's "Time series" section groups these by forecast tag with prev/next links.

**Change maps** — consecutive and vs-t0 differences for SLP/T2/PW/MCAPE, plus a period rain
accumulation, all computed with `cdo sub` on NetCDF extracted through the existing `wtune post
extract` engine (`post/changemap.py`) — never re-derived in Python. Named
`<field>-chg<N>h_d<domain>_fNNN_<validZ>.png`, e.g. `pw-chg6h_d02_f018_2005082718Z.png`.

**Animations** — `--animate` builds one ffmpeg MP4 per time-series field (`post/anim.py`, `ffmpeg`
added to `envs/wtune/pixi.toml`), referenced from the report by relative path next to
`report.html` (not base64-embedded — MP4s can be large).

**Custom analysis** — `wtune post custom` runs every `analysis/*.py` script in the project. Each
gets a `post.custom_api.Context` (`ctx.runs()`, `ctx.list_frames()`, `ctx.open_wrfout()`,
`ctx.find_run_track_csv()`, `ctx.save_figure()`/`ctx.save_table()`) and its own top-level
`run(ctx)` is the entry point; whatever it registers is folded into one HTML report. A script
that raises gets one error entry instead of aborting the batch. `wtune project init` writes
`analysis/example_tc_track.py` (storm-following central pressure & max wind, from an existing
track csv if available, else a per-frame SLP-minimum fallback) as a labelled example, not a
required part of the pipeline.

## Measured results

| item | result | conditions |
|---|---|---|
| **PLOV Ec** (cyclic + micro staging + adaptive stealing) [RESEARCH] | **-13.0% s/step, bit-identical** | 400 steps, 4 nodes, job 1070876 |
| PLOV ER (+ radiation staging) [RESEARCH] | -0.15..-0.23 s per rad step, -0.6% net | same job; adoption pending |
| `srun --distribution=cyclic` | -2.2% | 4 nodes, no staging |
| `KMP_BLOCKTIME=0` | -2.8% | 224x2 ABAB |
| hybrid 224x2 + correct `numtiles` vs 448x1 | -5.7% (-6.4% with blocktime) | same node set |
| `numtiles = patch/251` | -1.72% | same-node A/B |
| `8x56` decomposition | shape win, zero code | 448 ranks / 4 nodes |
| 4N->8N node scaling | -44.6%, projection error 1% | blocked by the 4-node rule |
| total-runtime prediction | MAE 1.07 s (~5.2%) | case-level nested LOO |
| config layer (`radt`, `non_hydrostatic`, `cudt`) [PHYSICS-GRAY] | up to -15% | 15 km case; result-affecting, behind the calibration gate; UNCONFIRMED until an official ruling |
| `wtune predict-rebuild` compute+init | +0.98% / −0.04% / −5.78% / +0.16% | conus25 / maria1km / zhengzhou / lafe on scc; ≈ ±2% after same-day drift correction |
| measured I/O probe | −0.5% / −0.5% / +3.1% of truth Σ writes | dorian2019 / maria1km / zhengzhou |
| ADIOS2 BP4 restart vs netCDF | write 1.05 s vs 323.5 s, read 2.3-2.5 s vs 120 s, bit-identical 60 min later; readable across 288/144/72 ranks | conus25, jobs 1277/1288/1289, 1371/1372 |
| `wtune run` 30 + 30 min restart chain vs one-shot | 202 + 202 variables bit-identical | dorian2019 d01+d02, jobs 1405+1410 vs 1392 |

## Measured-closed — do not re-propose

| direction | why it is closed |
|---|---|
| restart-boundary rebalancing / migration | restart ~56 s > 7-17 s recoverable per window; PLOV does it in place. That restart cost is netCDF: an ADIOS2 BP4 restart is 1.05 s write / 2.3-2.5 s read (conus25, jobs 1277/1288/1289), so `wtune run --tune-report` switches config at restart boundaries — whether a switch recovers more than it costs is not measured |
| rank placement (LPT, x-pair, per-window re-pin) | equalises node micro to 1%, s/step unchanged |
| DLB/LeWI core lending | nothing beyond `KMP_BLOCKTIME=0`; static OpenMP cannot add threads mid-region |
| per-rank hotspot prediction as an actuator input | no consumer; PLOV measures |
| rows > 4 in PLOV, cross-node stealing | each extra driver entry costs 13-16 ms; cross-node rad bounded at 1.3% |
| breaking the halo barrier externally | 13 closures x 2.8 ms = 37 ms/step floor, every lever needs RSL_LITE/solve_em source |
| the 29% settlement tax | levers are in solve_em; priced and parked |
| frequency / power shifting | cluster pinned at 3.0 GHz; 112->28 cores gave +0.2% |
| AMPI / over-decomposition | multi-week port, stopped at Phase 2 |
| reduced precision | WRF is already single precision |
| more nodes | 4-node cap is a rule |

## Installation

Three layers; install only the ones you use.

**1. Core — every `wtune` command (pixi).** One environment for the predictor, tuner,
rebuild, analysis and the Python half of post-processing: wrf-python, xWRF, MetPy, cartopy,
matplotlib, CDO, NCO, TempestExtremes, netCDF4/xarray, CatBoost, and Py-ART (`arm_pyart`,
`<2`, with `xradar>=0.9` and `dask` pinned beside it: an older xradar drags xarray down to
2024.9, and conda-forge's arm_pyart omits the `dask` it imports) for the NEXRAD composite reflectivity of the
convection observation comparison.

```bash
curl -fsSL https://pixi.sh/install.sh | bash          # once per user; installs ~/.pixi/bin/pixi
pixi install --manifest-path envs/wtune/pixi.toml     # linux-64 and linux-aarch64
bin/wtune --version                                   # always launch through bin/wtune
```

`radar_composite.py` imports `pyart` from the environment; if it is missing it falls back to a
directory named by `WTUNE_PYART_PATH` (`uv pip install --no-deps --target DIR 'arm_pyart<2'
lazy_loader`).

**2. WRF and WPS — built separately.** WTune never builds WRF. Point it at an existing tree:
`WRF_TREE` (its `run/`, `main/wrf.exe`, `main/real.exe`, and
`external/io_netcdf/diffwrf`, which `wtune post compare` uses) and `WPS_DIR` (for
`predict-rebuild`). Site settings live in `campaign/sites/<site>.sh`.

**3. Tools outside envs/wtune.** `bin/install-postproc.sh` installs them and then checks that
each one runs.

| tool | where from | used for |
|---|---|---|
| MET 12.1.1 (TC-Pairs, TC-Stat, TC-RMW, Grid-Stat, Point-Stat, Ensemble-Stat) | spack, `envs/postproc/spack.yaml` (not on conda-forge for linux-aarch64) | track, intensity and quadrant wind-radii errors against an ATCF b-deck; storm-relative azimuthal structure; gridded and point verification scores |
| nccmp | spack, same environment | generic NetCDF comparison (diffwrf stays the WRF-specific one) |
| ParaView 6.1.1 | pixi, `envs/paraview/pixi.toml` (conda-forge Qt6 + EGL build) | headless `pvbatch` rendering of xWRF-converted output; offscreen rendering uses the node's EGL |

```bash
source <spack>/share/spack/setup-env.sh
sbatch --exclude=head bin/install-postproc.sbatch     # scc: one compute node (the ParaView check needs its EGL)
bin/install-postproc.sh --check                        # re-check an existing install
export WTUNE_POSTPROC_VIEW=$PWD/envs/postproc/.spack-env/view
export WTUNE_PARAVIEW_BIN=$PWD/envs/paraview/.pixi/envs/default/bin
```

What it took on scc (GH200, Ubuntu 26.04, GCC 15.2), recorded in `envs/postproc/spack.yaml`:
MPI and curl are kept out of MET's tree (the site's `/usr` externals have headers on the login
node only, not on compute nodes), `netcdf-c ~blosc` avoids c-blosc 1.21.6, which does not
compile under GCC 15's default C23. ParaView comes from conda-forge because spack's
`+python` needs a spack-built Python and python@3.14.5 does not compile with GCC 15; that
build links `libhdf5.so.310` without pinning it, hence `hdf5 = "1.14.*"` in its manifest.
VAPOR 3.9.3 is not installed: its spack build links the site's header-less `/usr` curl
external, and conda-forge has no linux-aarch64 build.

Map backgrounds come from Natural Earth. Compute nodes often have no internet, so fetch the
shapefiles once on a login node (about 73 MB):

```bash
bin/fetch-natural-earth.sh                             # -> envs/cartopy-data
export WTUNE_CARTOPY_DATA=$PWD/envs/cartopy-data
```

Status (2026-09-29): `wtune post` reads `WTUNE_POSTPROC_VIEW` and `WTUNE_CARTOPY_DATA`
(`WTUNE_PARAVIEW_BIN` is for the 3-D figures of the figure recipes, decision 0057, not yet
wired) (decision 0056, 2026-09-29 update: every `wtune post` subcommand wraps
an established tool, see section 7). On scc, `fetch-natural-earth.sh` has been run and
ParaView renders offscreen on a compute node; MET's install is job 1582, and `wtune post
tc`'s MET steps are unverified against a real binary until it completes.

## Requirements

- Python 3.6+ (stdlib only) for `wtune.py` and the transition controller; everything else
  runs in the pixi environment above (see Installation).
- `ncdump` for headers; Julia + SpeedyWeather.jl for the convective probe
  (`julia --project=src/jlenv -e 'using Pkg; Pkg.instantiate()'`).
- PLOV: Intel compilers + Intel MPI and a WRF `dm+sm` build (`-qopenmp` live) with
  `libwrflib.a`; see `plov/build.sbatch`.

## Layout

The repo root is user-facing only; everything else lives under one of these directories
(2026-09-24 `src/` layout refactor — pure move + import/path fix, no behaviour change).

| path | status | contents |
|---|---|---|
| `wtune.py` | live | thin entry point: puts `src/` on `sys.path` and calls `wtune_cli.main()` |
| `bin/wtune` | live | runs `wtune.py` under the `envs/wtune` pixi env |
| `src/wtune_cli.py` | live | the ~1800-line CLI body (formerly `wtune.py` itself) |
| `src/wtune_physics.py`, `wtune_knobs.py`, `wtune_solar.py`, `wtune_surrogate.py`, `speedy_probe.jl`, `speedy_hotspot.jl`, `jlenv/`, `wind_probe.py`, `wtune_decomp.py`, `wtune_shape.py`, `gen_registry_catalog.py`, `wtune_catalog_registry.json` | live | advisory layer |
| `src/wtune_config_gate.py` (+ test) | live | calibration gate for result-affecting knobs |
| `src/wtune_transition_controller.py` | live (gate branch deprecated) | zero-measurement config + node-scaling projection |
| `src/wtune_feedback.py` (+ test), `wtune_probe.py` (+ test) | live | PLOV controller replay; ABBA probe harness |
| `src/wtune_rebuild_predict.py`, `wtune_watch.py`, `wtune_run.py`, `wtune_tune.py`, `wtune_preflight.py`, `tools/coldstart_sample.py`, `tools/era5_intermediate.py`, `tools/rescore_rebuild.py`, `tools/tc_bogus.py` | live | `predict-rebuild`, `watch`, `run`, `tune`, `check` and their helpers (section 6) |
| `src/wtune_oracle.py`, `wtune_oracle_train.py`, `wtune_segment_train.py`, `wrf_frame.py`, `featurize_frames.py`, `wtune_runtime_predict.py`, `wtune_runtime12.py`, `wtune_cost_model.py` | live | prediction layer |
| `campaign/` | **live, production** (unmoved) | the measurement side: `strict_campaign.sbatch`, `configs.py`, `cases.json`, `physics_templates.json`, the decomposition-search campaign and its rankers |
| `machine/` | **live, production** (unmoved) | hardware calibration data and probes (`machine/calibration/*.json`, `machine/bench/`) |
| `src/predictor/` | **live, production** | the decision side: ledger, benchmark, `recommend.py`, decomposition space and search, `stability.py`, `verify_obs.py` (`src/predictor/README.md`) |
| `src/fidelity_gate.py` (+ test) | live | chaos-floor scoring of a candidate against a reference and a twin |
| `src/fingerprint/` | live | out-of-process phase fingerprinter (autotuner step 2) -- moved into `src/` late in the refactor after `tests/test_wtune_fingerprint.py` proved it is imported, not dead |
| `src/tools/` | live | `bp_compression_check.py`, `run_tests.sbatch`, the ADIOS2→netCDF4 converter, `predict_validate/` |
| `src/analysis/` | live (not yet wired/tested) | wrf-python + TempestExtremes wrappers |
| `src/post/` | live | `wtune post` (decision 0056): inventory/extract/plot/sounding/tc/compare/report, advisory only, all heavy imports lazy |
| `tests/` | live | every test (`tests/`, `tests/tools/`, `tests/predictor/` incl. fixtures) + `tests/conftest.py`, which puts `src/`, `src/predictor`, `src/tools` on `sys.path` |
| `results/derived-data/` | data (unmoved) | canonical processed CSV/JSON behind every reported number |
| `examples/`, `docs/` | unmoved | example configs; docs |
| `archive/` | **not imported by any live code or test** — see below | `plov/`, `research/`, `experiments/`, `profilers/`, `deprecated/` |

`archive/` holds code confirmed (by grep for real `import`/`sys.path`/`subprocess`
references, not just comment mentions) to have zero coupling to this repo's Python import
graph: `archive/plov/` (wrap trampolines, stagers, build/gate/harness scripts —
**its own `README.md` calls it "live, production"; it is a standalone C/asm rebalancer
built and deployed independently of the wtune Python tooling, not something this Python
refactor's "imported by live code or a test" test can certify as dead in a deployment
sense**), `archive/experiments/` (harnesses behind reported numbers, vendored with sha256
provenance — same caveat, see `archive/experiments/README.md`), `archive/research/`
(one-off studies behind wiki notes), `archive/profilers/` (PMPI `LD_PRELOAD` profilers),
`archive/deprecated/` (migration/restart, placement, hotspot model and their tests).
`fingerprint/` looked like a candidate too but is not archived (see above).

Running the suite: `python3 -m pytest -q tests/` (pixi env at `envs/wtune/`, or
`uv run --with pytest --with numpy --with netCDF4 --with scikit-learn --with catboost
python -m pytest -q tests/`) — **1744 passed / 9 skipped / 4 known pre-existing failures** (2026-09-29, post decision 0056 tool-wrapping refactor)
(`tests/predictor/test_decomp_search.py::test_real_lafe_round2_incumbent_is_now_on_the_campaign_reference_scale`,
`tests/predictor/test_physics.py::test_template_engine_consistency_across_real_case_namelists`,
`tests/predictor/test_physics.py::test_physics_candidates_from_namelist_matches_text_engine_for_karina`,
`tests/test_wtune_config.py::test_examples_wtune_yaml_matches_the_rendered_template_and_validates` --
found during this update, unrelated to it: `examples/wtune.yaml` has drifted from
`wtune_config.render_template("conus25")`'s current output; neither file was touched by decision
0056's `src/post/` work).
`src/tools/run_tests.sbatch` runs the whole suite on a compute node the same way it always
did (`predictor/tests` needs the predictor pixi env; running it in the other one turns 18
skips into 19 lightgbm failures that look like a regression). The pieces, if you need one
directly: `pytest tests/test_wtune_*.py tests/test_fidelity_gate.py`, `pytest campaign
tests/tools archive/experiments`, `pytest tests/predictor` (predictor env), `pytest
archive/deprecated/` (archived), `python3 wtune.py selftest` (also runs
`wtune_compliance.validate()`), `python3 src/wtune_transition_controller.py selftest`
(38/40; both failures are in the deprecated imbalance-gate branch).

## Status

Developed against CONUS 2.5 km, a 12-case 3 km multi-regime corpus, and a 1500x1500 4-node
production case on f1/NCHC. The wall-time work is now in PLOV (-13%, bit-identical, no
source change); the remaining per-step budget is dynamics + a source-locked MPI floor, so
further gains come from the config layer behind the calibration gate, not from rebalancing.
