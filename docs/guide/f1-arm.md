# f1-arm 操作手冊

在 f1-arm 上自己從頭跑 WTune 需要的東西都在這頁：檔案放哪、怎麼更新 WTune、怎麼調校、怎麼看進度、怎麼停、會撞到哪些限制。WRF 怎麼編見 [在 f1-arm 編 WRF](build-wrf-f1-arm.md)。

## 機器與排程限制

| 項目 | 值 |
|---|---|
| 節點 | aarch64 Grace（Neoverse-V2），每節點 144 核、2 個 NUMA domain，WTune 估記憶體用 218 GiB |
| 登入節點 | `nlgn01`；計算節點沒有外網 |
| 計費 | 依核心小時：1 節點小時 = 144 × 0.2 = 28.8 |
| 家目錄配額 | GPFS `home2`：soft 100G、hard 200G |
| MPI | Open MPI 5.0.10，沒有 UCX，跨節點走 TCP（IPoIB） |

| partition | 每個 job 的核心數 | 每人同時在跑 | 每人可送（含排隊） | 時間上限 |
|---|---|---|---|---|
| `arm144` | ≤ 144（1 節點） | 10 | 12 | 2 天 |
| `arm576` | 145–576（2–4 節點） | 4 | 8 | 2 天 |
| `arm1440` | 577–1440（5–10 節點） | 2 | 4 | 2 天 |
| `arm-dev` | ≤ 1440 | 1 | 2 | 8 小時 |

（`sacctmgr show qos p_arm144,p_arm576,p_arm1440,p_arm-dev`，2026-10-18。）

`machine.yaml` 的 `partition` 寫 `arm144,arm576,arm1440`。WTune 依序試，`sbatch` 被拒就換下一個，所以每個 job 自動落到符合節點數的那一個。5 節點以上的 job 只能進 `arm1440`，每人最多兩個同時跑、四個排隊，多節點的掃描常常在這裡排很久。

## 檔案在哪裡

| 路徑 | 是什麼 | 誰讀它 |
|---|---|---|
| `~/wtune-adios2/` | WTune 程式；`.deployed-commit` 記錄是哪個 commit | 你下的每個 `wtune` 指令 |
| `~/wtune-adios2/envs/wtune/.pixi/` | WTune 的 Python 環境（pixi） | `bin/wtune` |
| `~/wtune-envs/campaign/` | 有 `adios2` Python 綁定的 pixi 環境（BP 轉 netCDF） | 設 `WTUNE_ADIOS2_PYTHON` 時 |
| `~/wtune-f1arm/env.sh` | 執行環境：函式庫 view、`WRF_TREE`、`OMP_STACKSIZE`、地圖資料 | 每個 WTune job（`machine.env_source`） |
| `~/.config/wtune/machine.yaml` | 機器設定 | 所有專案 |
| `~/wtune-adios2/campaign/sites/f1arm.sh` | 怎麼啟動 MPI：`ulimit -s unlimited`、IPoIB 網段、`mpirun --map-by ppr:..:numa` | 每個 WTune job |
| `~/wrf-stack/nvhpc-arm/` | netCDF、HDF5、ADIOS2、Open MPI 的 view | `env.sh` |
| `~/wrf-env/nvhpc-arm/wrf/*/WRF` | 編好的 WRF 樹 | `WRF_TREE` |
| `~/wtune-cases/<case>/` | 個案輸入 | `wtune project init --case` |
| `~/wtune-projects/` | 專案 | 你 |

### env.sh

```bash
V=$HOME/wrf-stack/nvhpc-arm
export PATH=$V/bin:$HOME/.pixi/bin:$PATH
export LD_LIBRARY_PATH=$V/lib:$V/lib64:${LD_LIBRARY_PATH:-}
export NETCDF=$V HDF5=$V PNETCDF=$V ADIOS2_DIR=$V
export WRF_TREE=$HOME/wrf-env/nvhpc-arm/wrf/vanilla-v480-adios2-dmpar/WRF
export WPS_DIR=$HOME/wrf-env/nvhpc-arm/wps/WPS-4.6.0    # WPS 4.6.0，可讀 GRIB2
export WTUNE_ADIOS2_PYTHON=$HOME/wtune-envs/campaign/.pixi/envs/default/bin/python   # 有 adios2 綁定的 python
export OMP_STACKSIZE=1G
export WTUNE_CARTOPY_DATA=$HOME/wtune-f1arm/cartopy-data   # 計算節點沒網路，post 的底圖放本機
```

### machine.yaml

```yaml
cores_per_node: 144
numa_per_node: 2
max_nodes: 8
partition: arm144,arm576,arm1440
login_nodes: []
account: <你的計畫代號>
hardware_id: f1-grace
site: f1arm
env_source: $HOME/wtune-f1arm/env.sh
wrf_tree: $HOME/wrf-env/nvhpc-arm/wrf/vanilla-v480-adios2-dmpar/WRF
geog: /work2/u5727520/wrf-newcase/WPS_GEOG_LOW_RES
max_hours: 8.0
max_omp: 1
node_preflight: false
node_memory_gib: 218.0
copy_probe_inputs: false
```

| 欄位 | 為什麼是這個值 |
|---|---|
| `max_nodes: 8` | 測試時自訂的上限；`arm1440` 最多 10 節點 |
| `max_omp: 1` | WRF 編成 dmpar（沒有 OpenMP） |
| `node_preflight: false` | 跳過每個節點 1 分鐘的 `srun` 探測，省排隊時間 |
| `copy_probe_inputs: false` | 不把輸入複製到每個探測目錄，省家目錄配額 |
| `geog` | `wtune prep wps` 的 WPS 靜態資料。f1-arm 上只有低解析度版（1.5 G）；完整的 `geog_high_res_mandatory` 解開約 29 GB，home2 與 work2 的配額都放不下。所以個案要附 `namelist.wps` 並設 `geog_data_res = 'lowres'` |

換 WRF 樹時 `env.sh` 的 `WRF_TREE` 和這裡的 `wrf_tree` 要一起改。只 `export WRF_TREE` 不夠：`wtune run` 會用它，但 tune 的量測 job 送出前會先 `source` env 檔，用的是檔案裡的 `WRF_TREE`。要讓某次 tune 用別的樹，就另存一份 env 檔，再 `export WTUNE_ENV_SOURCE=<那份檔>`。`~/wtune-f1arm/env-tuned.sh` 就是這樣留下來的：它指向舊的 tuned 樹，給需要和舊量測比較的驗證用。

## 更新 WTune

WTune 的 GitHub repo 是私有的，f1-arm 上拉不下來，所以從本機把某個 commit 打包送過去：

```bash
# 本機，在 wtune 的 git 目錄
C=$(git rev-parse --short devel/nk)
git archive --format=tar $C | ssh f1-arm "tar -x -C ~/wtune-adios2 && echo $C > ~/wtune-adios2/.deployed-commit"
ssh f1-arm "cd ~/wtune-adios2 && ~/.pixi/bin/pixi install --frozen --manifest-path envs/wtune/pixi.toml"
```

- 有 tune 正在跑時不要部署：Python 有些模組是用到才載入，新舊版本會混在一起。
- `tar -x` 只會新增和覆蓋檔案，上游刪掉的檔案會留著。要完全乾淨就解到新目錄，再重裝 pixi 環境。
- 確認：`cat ~/wtune-adios2/.deployed-commit`。

## 後處理工具（MET、ParaView）

`wtune post` 的 MET 驗證圖和 ParaView 3-D 圖要另外裝，裝在 `~/wtune-adios2/envs/` 底下，`wtune` 自己找得到，不用設環境變數。重新部署不會刪掉它們。

```bash
# 登入節點：ParaView（conda-forge，2.3G），並先把 MET 的原始碼抓下來（計算節點沒網路）
cd ~/wtune-adios2
~/.pixi/bin/pixi install --manifest-path envs/paraview/pixi.toml
source ~/spack/share/spack/setup-env.sh
spack -e envs/postproc config add "packages:all:target:[armv8.5a]"
spack -e envs/postproc concretize -f
spack -e envs/postproc fetch -D
# 計算節點編 MET（約 10 分鐘），最後一步檢查每個工具
sbatch -A <計畫代號> -p arm-dev -n 1 -c 32 --mem=60G -t 1:00:00 --wrap '
  export PATH=$HOME/wrf-env/nvhpc-arm/bootstrap-prefix/bin:$PATH
  source ~/spack/share/spack/setup-env.sh; cd ~/wtune-adios2
  spack -e envs/postproc install -j 32 && bin/install-postproc.sh --check'
```

- `target armv8.5a`：GCC 15.2 的 assembler 不認得 spack 給 `neoverse_v2` 的 `-march=armv9-a+...+memtag+profile`，每個套件都會編不過。
- `bootstrap-prefix/bin`：f1-arm 沒有系統的 `patch`，spack 套 patch 時會失敗；這裡有一份另外編好的 GNU patch。
- 只要 1/4 個節點，`arm-dev` 通常馬上排到；要整個節點的 job 常要排到隔天。
- `wtune doctor` 的 `grid_stat`、`pvbatch` 兩行變成 `OK` 就是裝好了。ParaView 在沒有 GPU 的節點上也能 offscreen 出圖。

## 每次登入

```bash
source ~/wtune-f1arm/env.sh
export PATH=$HOME/wtune-adios2/bin:$PATH
wtune doctor
```

`wtune doctor` 沒有 `FAIL` 再往下。

## 建專案

```bash
wtune project init ~/wtune-projects/yagi --case ~/wtune-cases/yagi
cd ~/wtune-projects/yagi
```

颱風個案加 `--storm <ATCF id> --hurdat2 ~/wtune-cases/hurdat2.txt`，後處理才會畫路徑比較。f1-arm 沒有登入節點要排除，不用 `--exclude-nodes`。

## 調校

```bash
cd ~/wtune-projects/yagi
setsid nohup wtune tune -O2 --minutes 45 --max-queue-hours 10 > tune.log 2>&1 < /dev/null &
```

- `-O2 --minutes 45`：量測時間（job 實際在跑的時間）最多 45 分鐘。排隊不算。
- `--max-queue-hours 10`：累積「我們的 job 一個都沒在跑」的時間到 10 小時就停，用已量到最快的節點數交出 ANCHOR 設定。預設 6 小時。
- `setsid nohup ... &`：tune 本身在登入節點上等 job，要好幾個小時；這樣登出也不會中斷。
- 結果寫在 `tune/<id>/tune_report.json`，怎麼讀見 [tune](../tutorial/tune.md) 的「報告」一節。`wtune run` 會自動用最新那份的 `WINNER`。

## 看進度

```bash
tail -f tune.log
squeue --me -o "%.9i %.24j %.8T %.10M %.4D %.9P %r"
```

`tune.log` 每 20 秒印一行：

```text
[node-sweep] [#-----------------------]   2%  elapsed=1:00 / budget=45:00  (running budget; wall 6:18:45, queue-only 6:16:40)
```

| 欄位 | 意思 |
|---|---|
| `elapsed` / `budget` | 已用掉的量測時間 / `--minutes` |
| `wall` | 從開始到現在的實際時間 |
| `queue-only` | 我們的 job 都在排隊的累積時間，碰到 `--max-queue-hours` 就停 |

WTune 送的量測 job 都叫 `sw-<case-id>-n<N>`，例如 `sw-yagi-3f06a1-n5`。`<case-id>` 在 `tune.log` 開頭那行 `case=...`。

查已結束的 job 一定要帶 `-S`：這台的 job 編號重新起算過，`sacct -j <id>` 不帶 `-S` 可能查到 2024 年同號的別人的 job。

```bash
sacct -S now-7days -X --format=JobID,JobName%24,State,Start,Elapsed,NNodes
```

!!! warning "`squeue --start` 不準"
    2026-10-17 它預估 2 節點的 job 要到隔天 10:48Z 才開始，實際 19:25Z 就跑完，差了 15 小時以上。不要用它決定要不要取消。

## 停止

tune 在背景跑時：

```bash
cd ~/wtune-projects/yagi
wtune stop --dry-run      # 先看會做什麼
wtune stop
wtune run                 # 用剛寫出的 tune_report.json
```

`wtune stop` 找到這個專案正在跑的 tune，請它收尾，通常半分鐘內完成：

- 取消它自己的 job。
- 用已量到的最快配置寫出 `tune_report.json`。

在前景跑時，按 Ctrl-C 效果相同。不要用 `scancel -u $USER`，那會連其他工作一起取消。

## 家目錄配額

```bash
/usr/lpp/mmfs/bin/mmlsquota -u $USER --block-size auto
```

GPFS 看的是 `blocks` 加 `in_doubt`：兩者加起來超過 100G，所有寫入都會失敗，而且出現在程式寫檔的錯誤裡，不是 `No space left on device`。每個 job 跑的時候 `in_doubt` 約多 1G／節點。大個案跑之前先清掉舊的 `tune/<id>/` 和 `runs/<id>/`。

2026-10-19 以前，tune 把轉好的 BP 輸入放在專案外的 `~/wtune-runs/bp-cache/`（當時 21G）；之後在專案內跑的 tune 改放 `<project>/input/bp-cache/`。舊的那份只剩專案外的 tune 會用到，確認不再需要就可以刪。

## 已知限制

| 現象 | 原因 | 影響 |
|---|---|---|
| 多節點比單節點慢（conus12：1、2、4 節點 0.173、0.207、0.462 s/step） | Open MPI 沒有 UCX，跨節點走 TCP | 小網格常選 N=1–2，這是正確的 |
| 5 節點以上的掃描排很久 | `arm1440` 每人 2 跑／4 送，叢集忙 | 用 `--max-queue-hours` 收尾 |
| 整段模擬時間的 I/O 預測不準 | 沒有 f1-arm 的 calibration，退回 `machine/calibration/scc.json` | 只影響時間預測，不影響選哪個設定 |
| 每個輸出檔印一次 `adios2 error ... error code 4` | ADIOS2 沒有編進 Blosc2，壓縮加不上去 | 輸出照寫、不壓縮 |
| 舊的量測不能直接和現在比 | 2026-10-19 以前用 `nvhpc-adios2-tuned-dmpar`（含團隊的 `wrf_io.F90` patch、多了 `-tp=grace -Mstack_arrays -Mfprelaxed`），之後預設是 vanilla | 要比就用 `env-tuned.sh`，見上面 machine.yaml 那段 |
