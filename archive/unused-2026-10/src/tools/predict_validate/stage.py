"""Stage one full-length validation run dir per case for the wtune predict test.

Copies the clean probe run dir (inputs, tables, wrf.exe) from the strong-scaling campaign,
replaces namelist.input with the case template, and changes only:
  - the decomposition (nproc_x/nproc_y/numtiles) to the probed ANCHOR config,
  - io_form_history/io_form_restart -> 2 (netCDF): the only I/O regime predict is calibrated for,
  - restart_interval -> run_minutes/4, so the ground-truth run leaves restarts at 25/50/75%
    that the K=4 restart-sampling estimator can start short windows from.

Every key is rewritten wherever it appears, including several keys packed on one line
(karina's template has `nproc_x=8, nproc_y=28, numtiles=13,`); a line-anchored edit missed those
and left two conflicting values. A key is appended only if absent, and the result is checked for
duplicates.
"""
import csv
import json
import os
import re
import shutil
import sys

W = os.path.expanduser("~/wtune")
sys.path.insert(0, W)
import wtune_cli as wt  # noqa: E402  (single-source run length via extract_case -> wtune_runlength)

cj = json.load(open(W + "/campaign/cases.scc.json"))
cases = cj.get("cases", cj)
PROBES = {"matthew": 755, "severe2023": 759, "sc26p2": 757, "sc26p2narr": 758,
          "conus12": 748, "karina": 751, "c34": 747, "sc26p1": 756}


def setkey(text, key, value):
    pat = re.compile(r"(?<![A-Za-z0-9_])(%s\s*=\s*)[^,\n/]*" % re.escape(key), re.I)
    new, n = pat.subn(lambda m: m.group(1) + str(value), text)
    if n:
        return new
    group = "&domains" if key.startswith(("nproc", "numtiles")) else "&time_control"
    return re.sub(r"(%s[^\n]*\n)" % group, r"\g<1> %s = %s,\n" % (key, value), text, count=1, flags=re.I)


def count_key(text, key):
    body = "\n".join(l.split("!")[0] for l in text.splitlines())   # ignore Fortran comments
    return len(re.findall(r"(?<![A-Za-z0-9_])%s\s*=" % re.escape(key), body, re.I))


def main():
    only = set(sys.argv[1:]) or set(PROBES)
    for case, job in PROBES.items():
        if case not in only:
            continue
        c = cases[case]
        pd = os.path.expandvars(c.get("prep_dir", ""))
        tmpl = os.path.expandvars(c.get("namelist_template", "") or os.path.join(pd, "namelist.input"))
        led = os.path.expanduser("~/wtune-runs/campaign/ledger_%s_%d.csv" % (case, job))
        r = next(csv.DictReader(open(led)))
        src = os.path.expanduser("~/wtune-runs/campaign/%s_ANCHOR_r1_%d" % (case, job))
        dst = os.path.abspath(case)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
        for f in os.listdir(dst):
            if f.startswith(("rsl.", "wrfout", "wrfrst", "auxhist")) or f == "wrf.out":
                os.remove(os.path.join(dst, f))

        run_min = int(round(float(wt.extract_case(wt.parse_namelist(tmpl))["run_total_hours"]) * 60))
        rint = max(1, run_min // 4)
        t = open(tmpl).read()
        edits = (("nproc_x", r["nproc_x"]), ("nproc_y", r["nproc_y"]), ("numtiles", 1),
                 ("io_form_history", 2), ("io_form_restart", 2),
                 ("restart_interval", rint), ("restart", ".false."))
        for k, v in edits:
            t = setkey(t, k, v)
        dup = {k: count_key(t, k) for k, _ in edits if count_key(t, k) != 1}
        if dup:
            sys.exit("STAGE FAIL %s: keys not present exactly once: %s" % (case, dup))
        open(os.path.join(dst, "namelist.input"), "w").write(t)

        for m in re.findall(r"iofields_filename\s*=\s*([^\n]+)", t, re.I):
            for fn in set(re.findall(r"['\"]([^'\"]+)['\"]", m)):
                s = os.path.join(pd, fn)
                if os.path.exists(s) and not os.path.exists(os.path.join(dst, fn)):
                    shutil.copy(s, os.path.join(dst, fn))

        meta = dict(case=case, probe_job=job, probe_ledger=led, prep_dir=pd, template=tmpl,
                    ranks=r["n_ranks"], nx=r["nproc_x"], ny=r["nproc_y"],
                    run_minutes=run_min, restart_interval=rint)
        json.dump(meta, open(os.path.join(dst, "stage.json"), "w"))
        print("staged %-11s %sx%s run=%d min restart_interval=%d min" % (case, r["nproc_x"], r["nproc_y"], run_min, rint))


if __name__ == "__main__":
    main()
