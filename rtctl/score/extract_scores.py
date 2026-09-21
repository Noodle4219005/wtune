#!/usr/bin/env python3
"""Canonical per-arm dataset for the no-wrap elastic-core WRF score campaign (2026-09-13).

Reads the batch stdout of each score job (RESULT / DIFF lines) and each controller arm's
ctl.<node>.out (TOTAL / ERRNO lines), normalizes every arm's median s/step to the mean of the
V13a/V13b arms of the SAME job, and writes one CSV row per valid arm.

usage: extract_scores.py OUT_CSV [COPY_CSV]
"""
import csv
import glob
import os
import re
import sys

ROOT = "/home/u5727520/wrf-bench/wscore"
JOBS = [  # job, stdout, arm-dir prefix, stage, plan (None when no controller plan)
    ("1073854", "s1-1073854.out", "", "s1_readiness", ""),
    ("1073865", "s2-1073865.out", "s2_", "s2_socket_plan", "plan_dph2"),
    ("1073873", "s2-1073873.out", "s2_", "s3a_numa14", "plan14_d1_t16_omp3"),
    ("1073874", "s2-1073874.out", "s2_", "s3b_numa14", "plan14_d2_t19"),
]
FIELDS = ["job", "stage", "plan", "arm", "exe", "omp_default", "numtiles", "bind", "plov", "ctl", "ok", "rc",
          "steps", "med_s_step", "med_skip1_s_step", "mean_skip1_s_step", "v_mean_same_job", "rel_to_v",
          "diffwrf_differing_fields", "ctl_slots", "ctl_lends", "ctl_collides", "ctl_hotends", "ctl_lent_s",
          "ctl_errs", "ctl_errno", "ctl_poll_cost_mean_us", "valid"]


def kv(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


rows = []
for job, out, prefix, stage, plan in JOBS:
    path = os.path.join(ROOT, out)
    if not os.path.exists(path):
        print("missing", path)
        continue
    res, diff = {}, {}
    for line in open(path):
        if line.startswith("RESULT "):
            d = kv(line)
            res[d["tag"]] = d
        m = re.match(r"DIFF (\S+) vs V13a .*differing_fields=(\d+)", line)
        if m:
            diff[m.group(1)] = int(m.group(2))
    vs = [float(res[t]["med"]) for t in ("V13a", "V13b") if t in res and res[t].get("ok") == "1"]
    vmean = sum(vs) / len(vs) if vs else float("nan")
    for tag, d in res.items():
        ok = d.get("ok") == "1" and int(d.get("steps", "0")) == 40
        ctl = d.get("ctl", "none")
        tot = {"slots": 0, "lends": 0, "collides": 0, "hotends": 0, "lent_s": 0.0, "errs": 0}
        costs, errnos = [], {}
        for f in glob.glob(os.path.join(ROOT, "%s%s_%s" % (prefix, tag, job), "ctl.*.out")):
            for line in open(f):
                if line.startswith("TOTAL "):
                    t = kv(line)
                    for k in ("slots", "lends", "collides", "hotends", "errs"):
                        tot[k] += int(t[k])
                    tot["lent_s"] += float(t["lent_s"])
                    costs.append(float(t["poll_cost_mean_us"]))
                elif line.startswith("ERRNO "):
                    for name, n in re.findall(r" ([A-Za-z][A-Za-z ]*?)=(\d+)", line.split(" ", 2)[-1]):
                        errnos[name.strip()] = errnos.get(name.strip(), 0) + int(n)
        med = float(d["med"])
        rows.append({
            "job": job, "stage": stage, "plan": plan if ctl != "none" or tag.startswith("EA") else "",
            "arm": tag, "exe": d.get("exe", "wrf_wtimer.exe" if d.get("plov") == "1" else "wrf.exe"),
            "omp_default": d.get("omp"), "numtiles": d.get("numtiles"), "bind": d.get("bind", "default"),
            "plov": d.get("plov"), "ctl": ctl, "ok": d.get("ok"), "rc": d.get("rc"), "steps": d.get("steps"),
            "med_s_step": med, "med_skip1_s_step": float(d["med_skip1"]), "mean_skip1_s_step": float(d["mean_skip1"]),
            "v_mean_same_job": round(vmean, 5), "rel_to_v": round(med / vmean, 4) if ok else "",
            "diffwrf_differing_fields": diff.get(tag, "" if tag == "V13a" else "missing"),
            "ctl_slots": tot["slots"] or "", "ctl_lends": tot["lends"] or "", "ctl_collides": tot["collides"] or "",
            "ctl_hotends": tot["hotends"] or "", "ctl_lent_s": round(tot["lent_s"], 2) if tot["slots"] else "",
            "ctl_errs": tot["errs"] if tot["slots"] else "", "ctl_errno": ";".join("%s:%d" % e for e in errnos.items()),
            "ctl_poll_cost_mean_us": round(sum(costs) / len(costs), 1) if costs and max(costs) > 0 else "",
            "valid": int(ok),
        })

for dst in sys.argv[1:]:
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", dst, len(rows), "rows")
for r in rows:
    print("%-8s %-15s %-5s med=%.5f rel=%s diff=%s valid=%s lent_s=%s" % (
        r["job"], r["stage"], r["arm"], r["med_s_step"], r["rel_to_v"], r["diffwrf_differing_fields"], r["valid"], r["ctl_lent_s"]))
