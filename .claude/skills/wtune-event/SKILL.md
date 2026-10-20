---
name: wtune-event
description: Turn a user's plain-language event.txt (what weather event a WRF case simulates and what they want checked) into a validated event.yaml for WTune, then follow the observation plan, finding data WTune cannot fetch itself. Use when a case folder has an event.txt, or the user describes the event a run is about.
---

# wtune-event

The user writes `event.txt` in their own words. You turn it into `event.yaml`. WTune checks
your file and decides which observations to use. You never decide that yourself.

## Part 1: translate event.txt -> event.yaml

1. `wtune event init event.txt` (writes `event.yaml` with `source_sha256` filled in).
2. `wtune event vocab` (the only allowed values for `types`, `verify`, and obs `kind`).
3. Read `event.txt` once, then fill `event.yaml` by the rules below.
4. `wtune event check event.yaml`. Fix every `error:` and run it again. Stop after 3 tries
   and report the remaining errors.
5. Reply to the user in the language of event.txt: 2-4 sentences saying what you understood,
   then `questions` if any.

### Rules

- Every item has `from:`, copied **character for character** from event.txt (a phrase is
  enough). If you cannot point to a phrase, the item does not go in.
- Never add a storm id, date, place or dataset the text does not state.
- `storms` is only for tropical cyclones (hurricane, typhoon, tropical storm). Fires,
  derechos, floods and fronts are not storms. `name`: the storm's English name as used by
  best-track archives (write 莫拉克 as Morakot); `year` if stated; `id` only if the text has it.
- `verify` = things to compare **with observations**; one item per quantity the text names,
  using the exact vocab word: hourly rain `precip_1h`, 24 h / daily rain `precip_24h`,
  multi-day or storm total `precip_total`, `precip_gauge` only when the text says to compare
  at gauges / rain stations; highest / lowest temperature `t2_max` / `t2_min`.
- `diagnose` = figures from the model alone (cross-sections, profiles of one field,
  helicity, structure, "does the model show X"). Do not put a model-only figure in `verify`.
- `obs_requests` only for a **specific provider or network** the user names that WTune does
  not use by default (e.g. a national weather service's radar, an agency's gauges). WTune's
  defaults: ISD surface stations, Stage IV, NEXRAD, ERA5, IGRA soundings, NHC / HURDAT2 best
  track, GPCP. "Compare with stations", "with the best track", "with ERA5" are `verify`
  items only, never `obs_requests`.
- A file or folder the user has goes in `local_obs` with its `path` exactly as written.
- `types`: every type the text supports, usually 1-3. Each type needs a phrase naming that
  phenomenon: a place name or a mechanism is not one (Tokyo is not `urban`; a warm front is
  not `extratropical_cyclone`).
- `idealized` only when the text says idealized, or names no real date and place. LES or a
  fine grid is not idealized, and a case compared with observations never is (`check`
  rejects `idealized` together with `verify`, `obs_requests` or `local_obs`).
- `questions` only when (a) the event or its type cannot be identified, (b) the text asks
  you to choose something (a day, a case), or (c) it asks for a comparison no obs kind covers
  (e.g. against a paper figure). Never ask which dataset to use (WTune's plan decides) or for
  dates and times (WTune reads them from the run). Write questions in the user's language.

### Example

event.txt:
```
2013 年 6 月加拿大 Alberta 的洪水，暖濕氣流撞上洛磯山脈東坡，兩天下了 200 mm 以上。
想看兩天累積雨量的分布對不對，還有山坡上的上升運動剖面。雨量站資料在 /data/ab_gauges.csv。
```
event.yaml (after `init`, which wrote the first three lines):
```yaml
schema: wtune-event/1
source: event.txt
source_sha256: <written by init>
event: {value: Alberta flood June 2013, from: "2013 年 6 月加拿大 Alberta 的洪水"}
types:
  - {value: heavy_rain, from: "兩天下了 200 mm 以上"}
  - {value: orographic, from: "暖濕氣流撞上洛磯山脈東坡"}
storms: []
region: {value: Alberta, Canadian Rockies east slope, from: "加拿大 Alberta"}
verify:
  - {value: precip_total, from: "兩天累積雨量的分布"}
diagnose:
  - {value: cross-section of vertical motion over the slope, from: "山坡上的上升運動剖面"}
obs_requests: []
local_obs:
  - {path: /data/ab_gauges.csv, kind: rain_gauges, from: "雨量站資料在 /data/ab_gauges.csv"}
questions: []
```

## Part 2: the observation plan

`wtune event plan event.yaml --run-dir runs/<id>` prints one row per need:

| status | what you do |
|---|---|
| `module` | nothing; `wtune post fetch <source>` on the login node gets it |
| `agent_fetch` | find the data (below) |
| `local` | convert the user's file if it is not in the standard format |
| `unsupported` | tell the user WTune has no comparison for it; do not build one |
| `model_only` | nothing to fetch |

### Finding data for `agent_fetch`

1. Find the official source (the agency or data centre), not a mirror or a paper figure.
2. Read its licence and access rules. If it needs a key, read it from an environment
   variable; never write a key into a file, a command line you print, or the wiki.
3. Download on the login node (compute nodes have no internet) into
   `<project>/input/obs/<name>/raw/`.
4. Write the conversion as a script `<project>/input/obs/<name>/convert.py`, keep it with
   the data, and convert to the standard format of the row's `kind` (below). Times in **UTC**
   (Taiwan and China local time is UTC+8).
5. Write `<project>/input/obs/<name>/provenance.json`:
   `{"format": "wtune-obs/1", "kind": ..., "dataset", "provider", "url", "license",
   "citation", "fetched_utc", "files": {converted file: sha256}}`.
6. `wtune event obs-check <project>/input/obs/<name>`; fix every `error:`. `wtune post recipe`
   uses only folders that pass.
7. Report what you got, what you could not get, and why.

Standard formats (grids: 1-D `lat`, `lon` in degrees, the variable on `(lat, lon)`, missing
cells NaN; times UTC). A best track outside the NHC basins is an ATCF b-deck file passed to
`wtune post recipe --bdeck`. Any other kind: keep the raw data and provenance, and say no
comparison reads it.

| kind | file | content |
|---|---|---|
| `gridded_precip` | `precip_YYYYMMDDHH_NNh.nc` | accumulation over the NN hours **ending** at YYYYMMDDHH; `precip`, `units = "mm"`. One file per native period (e.g. hourly); WTune adds them up |
| `rain_gauges` | `gauges.csv` | header `id,lat,lon,elev_m,start_utc,end_utc,precip_mm`; times `YYYY-MM-DD HH:MM`; one row per gauge and period (whole hours) |
| `radar` | `refl_YYYYMMDDHHMM.nc` | composite or lowest-tilt mosaic at that time; `refl`, `units = "dBZ"`; no echo a negative value, no coverage NaN (mask beyond the radars' range) |
| `satellite_ir` | `irtb_YYYYMMDDHHMM.nc` | IR window (~10.4-11 um) brightness temperature; `tb`, `units = "K"` |
| `stations_extra` | `stations_extra.csv` | header `id,lat,lon,elev_m,time_utc,var,value`; `var` is `gust` (m/s) or `visibility` (m) |

A rate (mm/h) is not an accumulation: multiply by the period length. Check that a period's
time stamp is its start or its end in the product's documentation before naming the file.

Stop and ask the user when the source needs an account, an application or a payment, or
when the licence forbids copying the data.
