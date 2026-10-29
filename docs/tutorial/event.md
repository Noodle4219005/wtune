# 7. event：用一段話描述個案

```bash
wtune event init event.txt          # 產生 event.yaml 樣板
wtune event check event.yaml        # 檢查
wtune event plan event.yaml --run-dir runs/<id>
```

在 case 或 project 資料夾放一個 `event.txt`，用自己的話寫：模擬的是什麼事件、想比什麼。例如：

```
2021 年 7 月河南鄭州的極端暴雨，20 日下午一小時雨量破 200 mm。
主要想看 24 小時累積雨量和逐時雨量的位置對不對，也看地面風。
我有一份 CMA 測站逐時雨量 ~/obs/cma_rain.csv。
```

agent（Claude Code、pi 等）依 `skills/wtune-event/SKILL.md` 把它轉成 `event.yaml`，WTune 再檢查它、決定要用哪些觀測資料。這三步各有分工：

| 步驟 | 誰做 | 做什麼 |
|---|---|---|
| `event.txt` | 你 | 用一段話寫事件和想比較的東西 |
| `event.yaml` | agent | 每個值都附上 `from:`，引用 `event.txt` 的原句 |
| `check`、`plan` | WTune | 檢查有沒有編造內容；決定每項觀測資料從哪裡來 |

## check 擋什麼

- `from:` 不是 `event.txt` 裡的原句（agent 自己補的內容）。
- 類型、比較項目、觀測種類不在 `wtune event vocab` 的清單裡。
- 風暴編號不在原句裡。加 `--hurdat2 FILE` 會用風暴名字和年份查出編號。
- `event.txt` 改過以後，`source_sha256` 對不上：要重新轉一次。
- 颱風個案沒寫是哪個風暴，也沒在 `questions` 問。
- 類型有 `idealized`（理想化），卻又要和觀測比（`verify`、`obs_requests`、`local_obs`）：理想化個案沒有觀測可比。LES 或高解析度本身不算理想化。

## plan 的狀態

| 狀態 | 意思 |
|---|---|
| `module` | 有 fetch 模組，`wtune post fetch` 會在登入節點下載 |
| `agent_fetch` | WTune 有比較方法但沒有下載模組，由 agent 找資料、轉成標準格式、記錄來源 |
| `local` | 你自己的檔案 |
| `unsupported` | WTune 沒有這種比較（例如 AOD、通量、和論文的圖比） |
| `model_only` | 只畫模式本身的圖，不需要觀測 |

美國本土以外的個案，雨量和雷達都會是 `agent_fetch`（Stage IV、NEXRAD 只涵蓋美國）。

## agent 找來的資料

`agent_fetch` 的資料由 agent 轉成標準格式 `wtune-obs/1`，放在 `input/obs/<name>/`，再用 `wtune event obs-check` 檢查。格點雨量、雨量站、雷達、衛星紅外、測站陣風與能見度通過檢查後，`post recipe` 會拿來比較，格式見 [post](post.md#agent)。非 NHC 海域的最佳路徑則寫成 ATCF b-deck，用 `post recipe --bdeck` 指定。
