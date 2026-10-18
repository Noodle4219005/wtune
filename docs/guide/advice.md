# calc、advise、physics、prep geog

```bash
wtune calc check case/namelist.input
```

把所有適用的經驗法則套到你的 namelist 上，逐條回報是否符合。這一頁的指令都**只印建議，不改任何檔案、不送 job**。

| 指令 | 回答什麼問題 |
|---|---|
| `wtune calc` | 這個參數通常該設多少？依據是什麼？ |
| `wtune advise` | 這個個案適合用哪些 WRF 的模式或流程？ |
| `wtune physics` | 物理方案與數值設定有沒有不搭的地方？ |
| `wtune prep geog` | 需要更高解析度的地形資料時怎麼準備？ |

## calc

```bash
wtune calc dt --dx-km 3
```

```text
- time_step recommended: 18 s（terrain=typical）
    公式: time_step [s] ~= 6 * dx[km]
    來源等級: wrf-doc（WRF 官方文件）  出處: run/README.namelist:155-156
- time_step steep terrain / high latitude: [12, 15] s
    公式: time_step [s] ~= (4..5) * dx[km] for steep terrain / high latitude / strong jets
    來源等級: rule-of-thumb（經驗值）
```

每一條都附公式、來源等級與出處。來源等級分清楚「WRF 官方文件寫的」和「社群慣例」，後者不要當成定論。

檢查整份 namelist：

```bash
wtune calc check case/namelist.input
```

```text
[OK 符合] d01 dt.six_dx: time_step/dx = 6.00 s/km，未超過建議的 6
[OK 符合] d01 nest.dim_legal: e_we=100, e_sn=100：(e-1) 可被 ratio 1 整除
[OK 符合] d01 bdy.spec_width: spec_bdy_width 5 = spec_zone 1 + relax_zone 4
```

可用的主題：

| 主題 | 內容 |
|---|---|
| `dt`、`cfl`、`acoustic` | 時間步長、Courant 數、聲波子步 |
| `radt` | 輻射呼叫間隔 |
| `nest`、`domain`、`bdy` | 巢狀比例與位置、區域大小與投影、邊界區 |
| `vert` | 模式頂、層厚、阻尼 |
| `physics`、`les` | 依 dx 建議積雲方案；LES 與灰色區的設定 |
| `geog` | 地理資料解析度與 dx 的搭配 |
| `decomp`、`io`、`run` | 最多能用幾個 rank、輸出檔大小、步數與估計時間 |
| `nudge`、`movenest`、`longrun`、`spinup` | nudging、移動巢狀、長時間模擬的 SST 更新、spin-up |
| `units`、`scales` | 單位換算；Coriolis、Rossby 半徑等尺度 |

每個主題的參數用 `wtune calc <主題> --help` 查。加 `--json` 取得機器可讀的輸出。

## advise

```bash
wtune advise .                          # 在 project 內
wtune advise case/ --need chem
wtune advise . --storm KATRINA2005 --hurdat2 hurdat2-1851-2024.txt
```

它偵測個案的條件，列出可以考慮的 WRF 模式與流程（WRFDA、nudging、moving nest、`tc.exe`、WRF-Chem、WRF-Hydro 等），每一項附上代價、依據與副作用。

`--need chem|hydro|surge` 用來宣告你有這個需求，建議會跟著調整。

## physics

```bash
wtune physics --namelist case/namelist.input --menu --compliance strict --no-probe
```

輸出是診斷加上方案選單，每一條帶 STRICT 或 PHYSICS-GRAY 的標籤。

- `--menu`：另外印出可選的方案清單。
- `--no-probe`：略過對流探測（比較快，但少了對流負載的分布資訊）。
- `--compliance`：只看某一類的建議。

## prep geog

這是選用的輸入資料準備，只在你下指令時才會做。

```bash
wtune prep geog list                    # WPS_GEOG 底下有哪些資料集
wtune prep geog bounds ...              # 區域的經緯度範圍與需要的 SRTM 圖磚
wtune prep geog srtm-fetch ...          # 下載 SRTM 1 角秒圖磚
wtune prep geog import-dem ...          # GeoTIFF 或 SRTM .hgt 轉成 WPS 格式
wtune prep geog verify ...              # 解碼 WPS 資料集，和來源比對
```

各子指令的參數用 `wtune prep geog <子指令> --help` 查。
