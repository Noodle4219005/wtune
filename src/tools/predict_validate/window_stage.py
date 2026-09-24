"""Stage the K=4 sampling windows for one case of the wtune predict validation.

For CASE (staged + ground-truth run by stage.py / truth.sbatch) this creates four run dirs under
<case>/win_<k>/, each a copy of the ground-truth run dir with a namelist that runs a short window:

  win_0   cold start at t0 from wrfinput (same as the probe)
  win_1..3 restart=.true. from the ground-truth run's own restarts at 25 / 50 / 75 % of the run

Every window runs WINDOW_STEPS+1 steps (step 1 excluded when measuring), writes no history and
no restarts, and keeps the ground-truth decomposition, physics and I/O forms. Per-domain list keys
(start_year = 2019, 2019,) are rewritten in every entry, not just the first.
"""
import datetime as dt
import glob
import json
import os
import re
import shutil
import sys

WINDOW_STEPS = 30


def set_list(text, key, value, ndom):
    """Set every per-domain entry of `key` to `value` (ndom entries).

    Replaces only this key's value list -- stopping at the next `name=` on the same line -- so
    packed lines (karina: `restart=.false., restart_interval=25,`) keep their other keys.
    """
    rhs = ", ".join([str(value)] * ndom) + ","
    keyre = r"(?<![A-Za-z0-9_])(%s\s*=\s*)" % re.escape(key)
    pat = re.compile(keyre + r"((?:[^,=\n/!]*,)+)(?=\s*(?:[A-Za-z_][A-Za-z0-9_%]*\s*=|$|/|!))", re.I | re.M)
    new, n = pat.subn(lambda m: m.group(1) + rhs, text, count=1)
    if n:
        return new
    pat2 = re.compile(keyre + r"([^,=\n/!]*?)\s*$", re.I | re.M)   # lone value, no trailing comma
    new, n = pat2.subn(lambda m: m.group(1) + rhs, text, count=1)
    if n:
        return new
    return re.sub(r"(&time_control[^\n]*\n)", r"\g<1> %s = %s\n" % (key, rhs), text, count=1, flags=re.I)


def count_key(text, key):
    body = "\n".join(l.split("!")[0] for l in text.splitlines())
    return len(re.findall(r"(?<![A-Za-z0-9_])%s\s*=" % re.escape(key), body, re.I))


def get_first(text, key, default=None):
    m = re.search(r"(?<![A-Za-z0-9_])%s\s*=\s*([^,\n/]+)" % re.escape(key), text, re.I)
    return m.group(1).strip() if m else default


def restart_times(run_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "wrfrst_d01_*"))):
        stamp = os.path.basename(f)[len("wrfrst_d01_"):]
        stamp = stamp.replace("_", " ", 1).replace("_", ":")
        out.append((dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S"), f))
    return out


def main(case):
    base = os.path.abspath(case)
    meta = json.load(open(os.path.join(base, "stage.json")))
    text0 = open(os.path.join(base, "namelist.input")).read()
    ndom = int(get_first(text0, "max_dom", 1))
    tstep = float(get_first(text0, "time_step"))
    num = float(get_first(text0, "time_step_fract_num", 0) or 0)
    den = float(get_first(text0, "time_step_fract_den", 1) or 1)
    dt_s = tstep + (num / den if den else 0)
    # WRF defaults start_minute/start_second to 0 when absent (matthew omits start_second)
    start0 = dt.datetime(*[int(get_first(text0, "start_" + k, 0 if k in ("minute", "second") else None)) for k in ("year", "month", "day", "hour", "minute", "second")])
    run_min = meta["run_minutes"]

    rst = restart_times(base)
    targets = [start0 + dt.timedelta(minutes=run_min * q / 4.0) for q in (1, 2, 3)]
    picks = []
    for t in targets:
        match = [p for p in rst if p[0] == t]
        if not match:
            sys.exit("WINDOW FAIL %s: no d01 restart at %s (have %s)" % (case, t, [str(p[0]) for p in rst]))
        picks.append(match[0][0])

    win_seconds = int(round((WINDOW_STEPS + 1) * dt_s))
    plan = [("win_0", start0, False)] + [("win_%d" % (i + 1), t, True) for i, t in enumerate(picks)]
    out = []
    for name, t, is_restart in plan:
        d = os.path.join(base, name)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for f in os.listdir(base):
            src = os.path.join(base, f)
            if f.startswith(("win_", "rsl.", "truth", "prediction")) or f in ("wrf.out", "namelist.input"):
                continue
            if f.startswith("wrfrst_"):
                continue
            os.symlink(os.path.realpath(src), os.path.join(d, f))
        if is_restart:  # link only this window's restart files (all domains)
            tag = t.strftime("%Y-%m-%d_%H:%M:%S")
            for f in glob.glob(os.path.join(base, "wrfrst_d0*_*")):
                if os.path.basename(f).endswith(tag) or os.path.basename(f).endswith(tag.replace(":", "_")):
                    os.symlink(os.path.realpath(f), os.path.join(d, os.path.basename(f)))
        txt = text0
        for k, v in (("year", t.year), ("month", "%02d" % t.month), ("day", "%02d" % t.day),
                     ("hour", "%02d" % t.hour), ("minute", "%02d" % t.minute), ("second", "%02d" % t.second)):
            txt = set_list(txt, "start_" + k, v, ndom)
        end = t + dt.timedelta(seconds=win_seconds)
        for k, v in (("year", end.year), ("month", "%02d" % end.month), ("day", "%02d" % end.day),
                     ("hour", "%02d" % end.hour), ("minute", "%02d" % end.minute), ("second", "%02d" % end.second)):
            txt = set_list(txt, "end_" + k, v, ndom)
        for k, v in (("run_days", 0), ("run_hours", 0), ("run_minutes", 0), ("run_seconds", win_seconds)):
            txt = set_list(txt, k, v, 1)
        txt = set_list(txt, "restart", ".true." if is_restart else ".false.", 1)
        txt = set_list(txt, "restart_interval", 999999, 1)
        txt = set_list(txt, "history_interval", 999999, ndom)
        keys = ["start_" + k for k in ("year", "month", "day", "hour", "minute", "second")] + \
               ["end_" + k for k in ("year", "month", "day", "hour", "minute", "second")] + \
               ["run_days", "run_hours", "run_minutes", "run_seconds", "restart", "restart_interval",
                "history_interval", "nproc_x", "nproc_y", "io_form_history", "io_form_restart"]
        bad = {k: count_key(txt, k) for k in keys if count_key(txt, k) != 1}
        if bad:
            sys.exit("WINDOW FAIL %s/%s: keys not present exactly once: %s" % (case, name, bad))
        lost = [k for k in re.findall(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*=", text0)
                if count_key(text0, k) and not count_key(txt, k)]
        if lost:
            sys.exit("WINDOW FAIL %s/%s: edit dropped keys %s" % (case, name, sorted(set(lost))))
        open(os.path.join(d, "namelist.input"), "w").write(txt)
        out.append(dict(window=name, start=str(t), restart=is_restart, steps=WINDOW_STEPS + 1, dt_s=dt_s))
    json.dump(dict(case=case, windows=out, ndom=ndom, dt_s=dt_s), open(os.path.join(base, "windows.json"), "w"), indent=1)
    for w in out:
        print("staged %-11s %s start=%s restart=%s" % (case, w["window"], w["start"], w["restart"]))


if __name__ == "__main__":
    for c in sys.argv[1:]:
        main(c)
