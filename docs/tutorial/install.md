# 1. 安裝

```bash
curl -fsSL https://pixi.sh/install.sh | bash            # 每個使用者一次
pixi install --manifest-path envs/wtune/pixi.toml       # WTune 本體
pixi install --manifest-path envs/adios2/pixi.toml      # netCDF 與 ADIOS2 互轉
bin/wtune --version
```

看到版本號就代表可以用了。之後所有指令都透過 `bin/wtune` 啟動；文件裡寫的 `wtune ...` 都是指它。

!!! tip "把 bin/ 加進 PATH"
    ```bash
    export PATH=/path/to/wtune/bin:$PATH
    ```

## 三層安裝

| 層 | 內容 | 什麼時候需要 |
|---|---|---|
| 1. 核心 | `envs/wtune`、`envs/adios2`（pixi） | 一定要 |
| 2. WRF 與 WPS | 你自己編好的 WRF；WTune 不編 WRF | `tune`、`run` |
| 3. 後處理工具 | MET、ParaView、Natural Earth 底圖 | `post` 的觀測比對與 3-D 圖 |

### 第 2 層：WRF 路徑

```bash
export WRF_TREE=/path/to/WRF        # 底下要有 main/wrf.exe、main/real.exe、run/
export WPS_DIR=/path/to/WPS         # 只有 predict-rebuild 需要
```

在 scc 上這些都在 `~/wtune-scc/env.sh` 裡：

```bash
source ~/wtune-scc/env.sh
```

站點相關的設定（每節點核心數、怎麼啟動 MPI）放在 `campaign/sites/<site>.sh`。

### 第 3 層：後處理工具

```bash
source <spack>/share/spack/setup-env.sh
sbatch --exclude=head bin/install-postproc.sbatch       # 在計算節點上裝
bin/install-postproc.sh --check                          # 檢查已安裝的版本
bin/fetch-natural-earth.sh                               # 地圖底圖，約 73 MB

export WTUNE_POSTPROC_VIEW=$PWD/envs/postproc/.spack-env/view
export WTUNE_PARAVIEW_BIN=$PWD/envs/paraview/.pixi/envs/default/bin
export WTUNE_CARTOPY_DATA=$PWD/envs/cartopy-data
```

| 工具 | 用在哪裡 | 沒裝會怎樣 |
|---|---|---|
| MET 12.1.1 | 颱風路徑與強度誤差、Grid-Stat、Point-Stat | 對應的圖在報告裡列為缺少，並寫明原因 |
| ParaView 6.1.1 | 3-D 雲的渲染 | 同上 |
| Natural Earth | 地圖的海岸線與國界 | 改畫 LANDMASK 等值線，並印出說明 |

缺工具時指令會指出缺哪一個，不會靜默跳過。

## doctor

```bash
wtune doctor
```

每一項檢查印一行。`FAIL` 代表某個步驟不能跑，`WARN` 代表某個功能會被跳過，兩者都會附上修正指令：

```text
OK    pixi: /home/you/.pixi/bin/pixi
OK    envs/wtune ready
FAIL  envs/adios2 not installed (ADIOS2 BP conversion)
      fix: pixi install --manifest-path /path/to/wtune/envs/adios2/pixi.toml
FAIL  WRF_TREE is not set
      fix: source ~/wtune-scc/env.sh
WARN  pvbatch not found: ParaView 3-D figures are skipped
      fix: bin/install-postproc.sh (README 'Installation')
```

照 `fix:` 那一行做，重跑 `wtune doctor`，直到沒有 `FAIL`。有任何 `FAIL` 時指令的 exit status 是 1。

下一步：[2. submit](quickstart.md)。
