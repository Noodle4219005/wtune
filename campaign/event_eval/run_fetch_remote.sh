#!/usr/bin/env bash
# Fetch eval on a machine without user namespaces (f1-arm login node): run_fetch_pi.sh's
# sandbox replaced by permissions. Reads the provider's API key from the first line of stdin
# into the variable KEY_VAR names (default MINIMAX_API_KEY), never from a file or argv.
#   run_fetch_remote.sh BASE OUT MODEL THINKING TASKS_YAML [task ...]
# BASE holds pi/ (npm install @earendil-works/pi-coding-agent), pi-home/ (empty agent dir) and
# wtune/ (git archive of the repo without results/, envs/wtune/.pixi -> a built env).
# While it runs, LOCK (colon-separated) are chmod 000 and every finished run folder is too, so
# no run can read reference conversions or an earlier run; all are restored on exit.
set -uo pipefail
base=$1 out=$2 model=$3 thinking=$4 tasks_yaml=$5; shift 5
IFS= read -r key
export "${KEY_VAR:-MINIMAX_API_KEY}=$key"
unset key
py="$base/wtune/envs/wtune/.pixi/envs/default/bin/python"
tasks=("$@")
[ ${#tasks[@]} -eq 0 ] && tasks=($("$py" -c "import sys,yaml;print(' '.join(yaml.safe_load(open(sys.argv[1]))))" "$tasks_yaml"))
mkdir -p "$base/wtune/sbxbin" "$out"
printf '#!/bin/sh\nexec %s %s/wtune/src/wtune "$@"\n' "$py" "$base" > "$base/wtune/sbxbin/wtune"
chmod +x "$base/wtune/sbxbin/wtune"
locked=()
restore() { for p in "${locked[@]}"; do chmod "${p%%:*}" "${p#*:}"; done; }
trap restore EXIT
lock_one() { [ -e "$1" ] && locked+=("$(stat -c %a "$1"):$1") && chmod 000 "$1"; }
IFS=: read -ra lock <<< "${LOCK:-}"
for p in "${lock[@]}"; do lock_one "$p"; done
export PATH="$base/wtune/sbxbin:$(dirname "$py"):$HOME/.pixi/bin:/usr/bin:/bin"
export PI_CODING_AGENT_DIR="$base/pi-home"
for t in "${tasks[@]}"; do
  d="$out/$t"; rm -rf "$d"; mkdir -p "$d/project/input/obs" "$d/tmp"
  desc=$("$py" - "$tasks_yaml" "$t" <<'PY'
import sys, yaml
t = yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]
lines = ["kind: %s" % t["kind"], "need: %s" % t["need"], "source: %s" % t["source"],
         "region (lat0, lat1, lon0, lon1): %s" % t["bbox"] if "bbox" in t else None,
         "window (UTC): %s to %s" % tuple(t["window"]) if "window" in t else None,
         "valid times (UTC): %s" % ", ".join(t["times"]) if "times" in t else None,
         "stations (USAF-WBAN): %s" % ", ".join(t["stations"]) if "stations" in t else None,
         "folder: project/input/obs/%s" % t["name"]]
print("\n".join(l for l in lines if l))
PY
)
  prompt="You are working in a WTune project (./project). \`wtune event plan\` gave this agent_fetch row:

$desc

Follow the wtune-event skill, Part 2: find the data at the official source, download it, write project/input/obs/<folder>/convert.py, convert to the standard format of the kind, write provenance.json, and make \`wtune event obs-check project/input/obs/<folder>\` pass. Cover exactly the region and times above. Do not ask questions; if something blocks you, write it in REPORT.md. End with a short report."
  start=$(date +%s)
  ( cd "$d" && TMPDIR="$d/tmp" timeout 1800 "$base/pi/node_modules/.bin/pi" -p --no-session \
      --no-context-files --no-skills --skill "$base/wtune/skills/wtune-event" \
      --model "$model" --thinking "$thinking" --tools read,bash,edit,write --mode json \
      "$prompt" > events.jsonl 2> stderr.txt )
  rc=$?
  echo "$t rc=$rc $(( $(date +%s) - start ))s" >> "$out/timing.txt"
  lock_one "$d"
done
