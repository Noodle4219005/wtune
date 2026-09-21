# KF 原生 compiler／library 單次探查卡

狀態：approval_pending。尚未查 cluster、atd 或提交作業；目前沒有背景監看。

目的：補齊 compiler/module 與 linker/loader 實際選擇證據，交下一階段 reviewed build 審核。不是 WRF 數值驗收，不能據此宣稱 stealing 已在 WRF 成功。

- f1／ACD110018／ct112，1 node、1 task、4 CPUs、4 GiB、10 分鐘上限。
- `1 job × 1 node × 0.166667 h = 0.166667 node-hours = 3.74`（整節點保守上限）；預計運算1–3分鐘，排隊未知；新增磁碟峰值預留1 GiB。
- 唯一作業目錄：`/home/u5727520/wrf-stack/wtune-kf-reviewed-recipe-20260913-r1/preproof`；存在即拒絕。既有 WRF、r3、G0 inputs 不變。
- 腳本：本目錄 `native-preproof.sbatch`，SHA256 `368913a343e7205a1d7340df157db33407db6b9e748974ec3aa55634f935314e`。
- 前置：本地 builder tests、preproof verifier tests、bash syntax 已通過；25-object source/config ledger 已固定。需本回合明確 cluster/account 同意，並確認 atd 可用，才提交一次。

工作圖：核對固定 compiler/source/config → 產生 proof-disabled recipe → 編譯25個objects → 依固定argv連結一次、保存link trace/map → 在與G0相同LD_LIBRARY_PATH下執行已hash的PT_INTERP `--list`（不進WRF main）→ 核對executor object ledger與實際DT_NEEDED選擇 → 交 `preproof/raw/native-closure.json`。不包含動態 `dlopen` 插件的認證。

PASS：Slurm COMPLETED/ExitCode0、完整25objects與ledger一致、direct library選擇符合固定contract、loader mapping可解析全部DT_NEEDED、candidate/map/compiler/loader hashes齊全、`PREPROOF COMPLETE; WRF WAS NOT RUN`。原生事實仍須審核，不能自動把claims設verified。任一步失敗停止、保留證據，無retry、WRF run、後續build或自動取消。

授權範圍包括下列單次提交、exact-job觀察與結案sacct對帳。以下命令尚未執行：

```bash
set -euo pipefail
wt_review=/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/native-review
wt_python=/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/.pixi/envs/default/bin/python3.11
wt_monitor=/home/u5727520/wrf-stack/wtune/plov/monitor_native_job.py
rtk proxy /usr/bin/systemctl is-active --quiet atd
test -x /usr/bin/at
test -x /usr/bin/squeue
test -x /usr/bin/sacct
test ! -e /home/u5727520/wrf-stack/wtune-kf-reviewed-recipe-20260913-r1/preproof
printf '%s  %s\n' 368913a343e7205a1d7340df157db33407db6b9e748974ec3aa55634f935314e "$wt_review/native-preproof.sbatch" | rtk proxy sha256sum --check --strict
printf '%s  %s\n' 8419fa14d2ce5baf9fb5b4685d973f1ac563d5cd3efd90e165cf64c24e1125b5 "$wt_monitor" | rtk proxy sha256sum --check --strict
wt_job=$(rtk proxy sbatch --parsable "$wt_review/native-preproof.sbatch")
[[ "$wt_job" =~ ^[1-9][0-9]*$ ]]
rtk proxy mkdir -m 700 "$wt_review/monitor-$wt_job"
wt_deadline=$(rtk proxy date -u -d '+24 hours' +%Y-%m-%dT%H:%M:%SZ)
rtk proxy env PYTHONDONTWRITEBYTECODE=1 "$wt_python" "$wt_monitor" start \
  --job-id "$wt_job" --operation 'KF native compiler/link preproof, one job only' \
  --artifact-dir "$wt_review/monitor-$wt_job" --native-log "$wt_review/preproof-$wt_job.out" \
  --pass-criterion 'COMPLETED/0; 25 object hashes, native-closure.json, compiler and loader hashes, no WRF entry' \
  --dependent-action 'Review compiler and library claims; do not automatically submit any build or WRF run' \
  --time-critical-reason 'This ten-minute probe gates the current native build review; terminal evidence needs prompt verification' \
  --deadline "$wt_deadline" --at /usr/bin/at --squeue /usr/bin/squeue --sacct /usr/bin/sacct
```

首次callback為提交後約60秒；具日誌的RUNNING至少四筆且觀察span≥240秒才通過startup，之後每5分鐘（本次10分鐘probe的terminal依賴屬time-critical）。提早terminal必須記startup未完整觀察，不捏造通過。at服務或投遞失敗需記blocked、最後觀察與下一人工檢查時間；不宣稱通知目前chat，無busy-wait替代。24小時monitor期限只停止監看，不取消job。Root須核對原生輸出才標verified。

結案對帳命令（僅本job，另依wiki要求核本批累計）：`sacct -j JOB -X --format=JobID,JobName,State,ExitCode,Elapsed,NNodes,NCPUS`；`sacct -u "$USER" -S SUBMISSION_DATE --format=JobID,JobName,State,Elapsed,NNodes,NCPUS -X`。以sacct實際時間與資源報成本，不以squeue推斷沒有重複。
