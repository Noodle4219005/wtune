#!/usr/bin/env bash
# One pi run per fetch task (Part 2 of the wtune-event skill):
#   run_fetch_pi.sh OUT_DIR MODEL THINKING [task ...]
# Each run gets an empty <OUT>/<task>/project/input/obs/ and must leave a wtune-obs/1 folder
# that passes `wtune event obs-check`. python3 is the wtune environment (netCDF4, numpy, PIL).
# The agent runs in a bubblewrap sandbox: /home and /tmp are empty except pi (~/.pi without its
# sessions), node (~/.nvm), a copy of the repo without results/ at /work/wtune, and its own
# run folder at /work/run -- so it cannot read reference conversions or notes on this machine.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
out=$1 model=$2 thinking=$3; shift 3
tasks_yaml="$repo/tests/fixtures/fetch/tasks.yaml"
tasks=("$@")
[ ${#tasks[@]} -eq 0 ] && tasks=($(python3 -c "import sys,yaml;print(' '.join(yaml.safe_load(open(sys.argv[1]))))" "$tasks_yaml"))
clean="$out/_wtune"
if [ ! -d "$clean" ]; then
  mkdir -p "$clean"
  git -C "$repo" archive HEAD -- . ':!results' | tar -x -C "$clean"
  mkdir -p "$clean/envs/wtune/.pixi" "$clean/sbxbin"
  # bin/wtune goes through pixi, which wants to write its own state: call the env's python
  printf '#!/bin/sh\nexec /work/wtune/envs/wtune/.pixi/envs/default/bin/python /work/wtune/src/wtune "$@"\n' \
    > "$clean/sbxbin/wtune"
  chmod +x "$clean/sbxbin/wtune"
fi
export PATH="$repo/envs/wtune/.pixi/envs/default/bin:$PATH"
sandbox() {   # sandbox RUN_DIR CMD...
  local run=$1; shift
  bwrap --ro-bind /usr /usr --ro-bind /etc /etc --symlink usr/bin /bin --symlink usr/lib /lib \
    --symlink usr/lib /lib64 --symlink usr/bin /sbin --proc /proc --dev /dev \
    --tmpfs /tmp --tmpfs /home --tmpfs /var --tmpfs /run --ro-bind /etc/resolv.conf /etc/resolv.conf \
    --bind "$HOME/.pi" "$HOME/.pi" --tmpfs "$HOME/.pi/agent/sessions" \
    --tmpfs "$HOME/.pi/agent/missions" --ro-bind "$HOME/.nvm" "$HOME/.nvm" \
    --ro-bind "$clean" /work/wtune \
    --ro-bind "$repo/envs/wtune/.pixi" /work/wtune/envs/wtune/.pixi \
    --bind "$run" /work/run --chdir /work/run --share-net --unshare-pid --die-with-parent \
    --setenv HOME "$HOME" --setenv TMPDIR /work/run/tmp \
    --setenv PATH "/work/wtune/sbxbin:/work/wtune/envs/wtune/.pixi/envs/default/bin:$(dirname "$(command -v node)"):$HOME/.pi/agent/bin:/usr/bin" \
    "$@"
}
for t in "${tasks[@]}"; do
  d="$out/$t"; rm -rf "$d"; mkdir -p "$d/project/input/obs"
  desc=$(python3 - "$tasks_yaml" "$t" <<'PY'
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
  start=$(date +%s.%N)
  mkdir -p "$d/tmp"
  sandbox "$d" timeout 1800 pi -p --no-session --no-context-files --no-skills \
      --skill /work/wtune/.claude/skills/wtune-event --model "$model" --thinking "$thinking" \
      --tools read,bash,edit,write --mode json "$prompt" > "$d/events.jsonl" 2> "$d/stderr.txt"
  rc=$?
  usage=$(python3 "$here/usage.py" "$d/events.jsonl" "$d/reply.txt")
  echo "$t rc=$rc $(echo "$(date +%s.%N) - $start" | bc | cut -d. -f1)s $usage" | tee -a "$out/timing.txt"
done
