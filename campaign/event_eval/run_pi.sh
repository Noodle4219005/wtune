#!/usr/bin/env bash
# One pi translation per event.txt: run_pi.sh OUT_DIR MODEL THINKING [case ...]
# e.g. run_pi.sh /tmp/ev minimax/MiniMax-M3.1-Flash low
# EVENTS=tests/fixtures/events-holdout picks the held-out set (default tests/fixtures/events).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
out=$1 model=$2 thinking=$3; shift 3
events="$repo/${EVENTS:-tests/fixtures/events}"
cases=("$@")
[ ${#cases[@]} -eq 0 ] && cases=($(ls -d "$events"/*/ | xargs -n1 basename))
export PATH="$repo/bin:$PATH"
prompt='The file event.txt in the current directory describes a WRF case. Follow the wtune-event skill, Part 1 only: produce event.yaml in the current directory and make `wtune event check event.yaml` pass. Do not fetch any data. When done, reply with your 2-4 sentence summary and any questions.'
for c in "${cases[@]}"; do
  d="$out/$c"; mkdir -p "$d"; cp "$events/$c/event.txt" "$d/"; rm -f "$d/event.yaml"
  start=$(date +%s.%N)
  ( cd "$d" && timeout 600 pi -p --no-session --no-context-files --no-skills \
      --skill "$repo/skills/wtune-event" --model "$model" --thinking "$thinking" \
      --tools read,bash,edit,write --mode json "$prompt" > events.jsonl 2> stderr.txt )
  rc=$?
  usage=$(python3 "$here/usage.py" "$d/events.jsonl" "$d/reply.txt")
  # case rc seconds turns input_tokens output_tokens cost_usd
  echo "$c rc=$rc $(echo "$(date +%s.%N) - $start" | bc | cut -d. -f1)s $usage" | tee -a "$out/timing.txt"
done
