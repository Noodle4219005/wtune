# WTune

WTune 讓一個 WRF 個案跑得更快、跑得完、跑完有圖，而且**不改 WRF 原始碼**。

```bash
wtune project init ~/projects/mycase --case /path/to/case_dir
cd ~/projects/mycase
wtune doctor      # 機器和專案準備好了嗎
wtune submit      # 調校 → 執行 → 後處理，一次做完
```

## 做什麼

| 階段 | 指令 | 做什麼 | 產出 |
|---|---|---|---|
| 跑之前 | `wtune tune` | 在叢集上實測，找最快的節點數、MPI 切法、OpenMP 執行緒數、tile 數 | `tune/<id>/tune_report.json` |
| 跑的時候 | `wtune run` | 把整段模擬切成數個 Slurm job，用 restart 交接；失敗會續跑 | `runs/<id>/` |
| 跑完 | `wtune post recipe` | 依個案類型出標準圖組，並和觀測比較 | `post/<id>/report.html` |

完整指令表在 [`docs/reference/cli.md`](docs/reference/cli.md)。

## 不改什麼

WTune 只動三樣東西：namelist、啟動方式（`mpirun` 參數）、環境變數。它不修改 WRF 原始檔，也不用 `LD_PRELOAD`、`--wrap`、PMPI 介入 `wrf.exe`。

自動套用的只有不改變模擬結果的設定：節點數、MPI 切法、tile 數、OpenMP 執行緒數、I/O 格式。物理方案、`time_step`、垂直層數這類會改變答案的設定只給建議。規則見 [`docs/guide/compliance.md`](docs/guide/compliance.md)。

## 安裝

```bash
curl -fsSL https://pixi.sh/install.sh | bash            # 每個使用者一次
pixi install --manifest-path envs/wtune/pixi.toml       # WTune 本體
pixi install --manifest-path envs/adios2/pixi.toml      # netCDF 與 ADIOS2 互轉
export PATH=$PWD/bin:$PATH                              # 一律透過 bin/wtune 啟動
wtune --version
```

- WRF 要自己編好：`WRF_TREE` 指到 WRF 樹（`main/wrf.exe`、`run/`），`predict-rebuild` 另外需要 `WPS_DIR`。
- 站點設定（環境、MPI 啟動方式）在 `campaign/sites/<site>.sh`：目前有 `scc`、`f1`、`f1arm`。
- 機器設定（partition、每節點核心數、節點上限）在 `~/.config/wtune/machine.yaml`，見 [`docs/tutorial/project.md`](docs/tutorial/project.md)。
- 後處理工具（MET、ParaView、Natural Earth 底圖）是選用的，見 [`docs/tutorial/install.md`](docs/tutorial/install.md)。

## 文件

```bash
pixi run --manifest-path envs/docs/pixi.toml serve      # MkDocs，原始檔在 docs/
```

| 想做的事 | 從這裡開始 |
|---|---|
| 第一次使用 | `docs/tutorial/`：安裝 → submit → project → tune → run → post |
| 調整搜尋空間、限制調校時間 | [`docs/guide/config.md`](docs/guide/config.md)、[`docs/tutorial/tune.md`](docs/tutorial/tune.md) |
| 已知限制 | [`docs/guide/faq.md`](docs/guide/faq.md) |
| 審查程式碼 | [`docs/REVIEW_MAP.md`](docs/REVIEW_MAP.md)、[`docs/CODE_STYLE.md`](docs/CODE_STYLE.md) |
| 版本變更 | [`CHANGELOG.md`](CHANGELOG.md)；版本號在 `src/wtune/version.py` |

## 目錄

| 路徑 | 內容 |
|---|---|
| `bin/` | `wtune` 啟動器、後處理工具的安裝腳本 |
| `src/wtune/` | 套件本體：`cli.py`、`tune/`、`run/`、`post/`、`predictor/`、`tools/` |
| `campaign/` | 探測用的 sbatch、個案設定、站點檔 |
| `machine/` | 硬體校正資料與量測腳本 |
| `envs/` | pixi 與 spack 環境 |
| `tests/` | 測試 |
| `docs/` | MkDocs 文件 |
| `examples/` | 設定檔範例 |
| `results/derived-data/` | 報告數字背後的處理後資料 |
| `archive/` | 已退役、沒有現行程式匯入的程式碼：`plov/`、`research/`、`experiments/`、`profilers/`、`deprecated/`、`unused-2026-10/` |

## 測試

```bash
PATH=$PWD/envs/wtune/.pixi/envs/default/bin:$PATH python -m pytest -q tests campaign
```

在有 Slurm 的登入節點上，部分測試會真的呼叫 `sinfo`、`srun`；完整測試請在計算節點或沒有 Slurm 的機器上跑。
