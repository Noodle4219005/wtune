#!/usr/bin/env python3
"""External affinity actuator for elastic_bench (mechanism probe).

Runs as a separate process. Finds the benchmark by /proc comm, reads its current
Cpus_allowed_list (narrow set) and the donor task's cores from DONOR_FILE. On each
target region it waits TRIGGER_MS after "start k", acts on the benchmark's TIDs with
sched_setaffinity, reads the masks back, samples /proc/<tid>/stat 'processor' every 1 ms
until the region ends, then restores the narrow mask on every TID.

Modes:
  widen  every TID gets narrow | donor          (kernel may or may not migrate: load balancer)
  move   the last len(donor) worker TIDs are pinned one-to-one onto the donor cores,
         the rest keep narrow                    (current CPU leaves the mask -> kernel
                                                  migrates that thread synchronously)
Regions: odd regions >= 3 are acted on, even regions are the in-run control; --all acts
on every region >= 3.

usage: expander.py STATUSFILE DONOR_FILE NREGIONS TRIGGER_MS MODE [--all]
"""
import os, sys, time
from pathlib import Path

status_path, donor_path, nregions, trigger_ms, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
every = "--all" in sys.argv
assert mode in ("widen", "move")


def cpulist(s):
    out = set()
    for part in s.strip().split(","):
        if not part: continue
        if "-" in part:
            a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def find_bench():
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try:
            if (p / "comm").read_text().strip() == "elastic_bench" and p.stat().st_uid == os.getuid():
                return int(p.name)
        except OSError:
            continue
    return None


def allowed(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return cpulist(line.split(":", 1)[1])
    return set()


def processor(tid):
    try:
        raw = Path(f"/proc/{tid}/stat").read_text()
    except OSError:
        return -1
    return int(raw[raw.rfind(")") + 1:].split()[36])


def read_status():
    try:
        w, k = Path(status_path).read_text().split()
        return w, int(k)
    except (OSError, ValueError):
        return None, -1


def set_all(tids, masks):
    errs, back = {}, {}
    for tid in tids:
        try:
            os.sched_setaffinity(tid, masks[tid])
        except OSError as e:
            errs[tid] = f"{e.errno}:{e.strerror}"
        try:
            back[tid] = sorted(os.sched_getaffinity(tid))
        except OSError as e:
            back[tid] = str(e)
    return errs, back


t_dead = time.monotonic() + 60
pid = None
while pid is None and time.monotonic() < t_dead:
    pid = find_bench(); time.sleep(0.005)
while not Path(donor_path).exists() and time.monotonic() < t_dead:
    time.sleep(0.005)
if pid is None:
    print("expander: bench not found", flush=True); sys.exit(1)

tids = sorted(int(t.name) for t in Path(f"/proc/{pid}/task").iterdir())
narrow = allowed(pid)
donor = sorted(cpulist(Path(donor_path).read_text()))
wide = narrow | set(donor)
print(f"expander: mode {mode} pid {pid} tids {tids} narrow {sorted(narrow)} donor {donor}", flush=True)

t_end = time.monotonic() + 40  # hard cap: a hang must never eat the job
seen = -1
while time.monotonic() < t_end:
    w, k = read_status()
    if k >= nregions - 1 and w == "end":
        break
    if w == "start" and k != seen:
        seen = k
        tids = sorted(int(t.name) for t in Path(f"/proc/{pid}/task").iterdir())
        if k < 3 or (not every and k % 2 == 0):
            continue
        time.sleep(trigger_ms / 1e3)
        if mode == "widen":
            masks = {t: wide for t in tids}
        else:
            masks = {t: set(narrow) for t in tids}
            for t, c in zip(tids[-len(donor):], donor):
                masks[t] = {c}
        t_set = time.monotonic()
        errs, back = set_all(tids, masks)
        t_after = time.monotonic()
        print(f"region {k} ACT {mode} at +{trigger_ms:.1f}ms set_took_us {(t_after-t_set)*1e6:.0f} errs {errs} readback {back}", flush=True)
        samples = []
        while time.monotonic() < t_end:
            w2, k2 = read_status()
            if (w2 == "end" and k2 >= k) or (w2 == "start" and k2 > k):
                break
            samples.append((round((time.monotonic() - t_set) * 1e3, 2), [processor(t) for t in tids]))
            time.sleep(0.001)
        set_all(tids, {t: set(narrow) for t in tids})
        on_donor = [s for s in samples if any(c in donor for c in s[1])]
        first = on_donor[0][0] if on_donor else None
        print(f"region {k} samples {len(samples)} on_donor {len(on_donor)} first_donor_sample_ms {first} trace {samples[:10]}", flush=True)
    time.sleep(0.0002)
set_all(tids, {t: set(narrow) for t in tids})
print("expander: done", flush=True)
