#!/usr/bin/env python3
"""Segment-specific namelist post-pass for the rk_ord alternation experiment.

campaign/configs.py's `edit-namelist` already produces a correct ADIOS2 / decomposition /
history namelist, but it deliberately knows nothing about restart chaining: `_apply_adios2`
forces restart_interval=1000000 (restarts OFF) as its LAST edit, so a `--set
restart_interval=...` passed to it is silently overwritten.  This script runs AFTER it and
owns exactly the four things a segment needs:

    * the segment's clock          (start_* / end_*)
    * whether it is a restart      (restart=.true./.false.)
    * whether it writes one        (restart_interval)
    * rk_ord                       (&dynamics, absent from every template -> INSERTED)

Failure modes deliberately guarded against (each one has already cost a run somewhere in
this tree):

  1. Keys share lines in hand-written namelists, so a `^ *key=` anchored substitution can
     match nothing and silently do nothing.  Every edit here matches the key ANYWHERE on a
     line and the script re-reads the file afterwards and refuses to exit 0 unless every
     key it was asked to set actually reads back with the requested value.
  2. A substituting-only sed does nothing when the key is ABSENT.  rk_ord is absent from
     every template in this repo, so set_key() inserts into the named section instead.
  3. start_*/end_* are max_domains arrays in WRF.  conus12 is max_dom=1 so a scalar is
     correct here; --max-dom replicates the value if that ever changes.
"""
import argparse
import datetime as dt
import re
import sys

WRF_TIME = "%Y-%m-%d_%H:%M:%S"
NEVER_MIN = 1000000          # restart_interval that never elapses (same value configs.py uses)


# Split a line at the start of every `name =` assignment.  Done with finditer rather than a
# zero-width re.split: the system python here is 3.6, and splitting on an empty match is a
# hard ValueError before 3.7.  ''.join(_chunks(line)) == line exactly, so rewriting one
# chunk leaves every other character on that line untouched -- including the leading
# indentation, which lands in chunk 0.
_KEYSTART = re.compile(r"\b[A-Za-z_]\w*\s*=")
_ASSIGN = re.compile(r"(\s*([A-Za-z_]\w*)\s*=\s*)(.*)$", re.S)


def _chunks(line):
    bounds = [0] + [m.start() for m in _KEYSTART.finditer(line)] + [len(line)]
    return [line[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]


def _walk(text, key, rewrite):
    """Find the first chunk assigning `key`; hand (prefix, old_value) to rewrite().

    Chunk-based rather than one regex because of two WRF-namelist facts:
      * several keys share one line, so a ^-anchored pattern matches nothing at all;
      * a value is a max_domains LIST ("2023, 2023"), so a pattern that stops at the first
        comma rewrites only the first element and leaves the rest appended -- turning
        `start_year = 2023, 2023,` into `start_year = 2023, 2023, 2023,` and silently
        handing d02 a different start date than d01.  That is a nest-only failure, which
        is exactly the kind that survives a single-domain dry run.
    """
    out, hit = [], None
    for line in text.splitlines():
        if hit is None:
            parts = _chunks(line)
            for idx, part in enumerate(parts):
                m = _ASSIGN.match(part)
                if m and m.group(2).lower() == key.lower():
                    hit = m.group(3)
                    new = rewrite(m.group(1), m.group(3))
                    if new is not None:
                        parts[idx] = new
                        line = "".join(parts)
                    break
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), hit


def get_key(text, key):
    _text, value = _walk(text, key, lambda _p, _v: None)
    return value.strip().rstrip(",").strip() if value is not None else None


def set_key(text, key, value, section):
    """Set `key` to `value` (the WHOLE value list), inserting under `section` when absent."""
    def rewrite(prefix, old):
        # Keep the old value's trailing comma AND its trailing whitespace, so an edited
        # namelist is byte-identical to the previous generator's output everywhere except
        # the value itself -- that is what makes "does this still reproduce job NNNN"
        # answerable with `diff` instead of with judgement.
        m = re.match(r"^(.*?)(,?)(\s*)$", old, re.S)
        return prefix + str(value) + m.group(2) + m.group(3)

    new_text, hit = _walk(text, key, rewrite)
    if hit is not None:
        return new_text
    hdr = re.compile(r"^(\s*%s\s*)$" % re.escape(section), re.M | re.I)
    if not hdr.search(text):
        raise SystemExit("segnl: section %s not found, cannot insert %s" % (section, key))
    return hdr.sub(lambda m: m.group(1) + "\n %s = %s," % (key, value), text, count=1)


def dup(value, n):
    return ", ".join([str(value)] * n) if n > 1 else str(value)


def show_start(path):
    """Print the template's domain-1 start time as YYYY-MM-DD_HH:MM:SS.

    The logical run's t=0 is a property of the CASE, not of this script, so it is read from
    the case's own namelist template rather than hard-coded per case.
    """
    with open(path) as fh:
        text = fh.read()

    def first(key, default="0"):
        v = get_key(text, key)
        return (v.split(",")[0].strip() if v else default) or default

    print("%04d-%02d-%02d_%02d:%02d:%02d" % (
        int(first("start_year")), int(first("start_month")), int(first("start_day")),
        int(first("start_hour")), int(first("start_minute")), int(first("start_second"))))
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "show-start":
        return show_start(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("namelist", help="namelist.input to edit IN PLACE")
    ap.add_argument("--base-start", required=True, metavar="YYYY-MM-DD_HH:MM:SS",
                    help="t=0 of the whole LOGICAL run (not of this segment)")
    ap.add_argument("--offset-min", type=int, required=True,
                    help="this segment's start, in minutes after --base-start")
    ap.add_argument("--run-min", type=int, required=True, help="this segment's length in minutes")
    ap.add_argument("--rk-ord", type=int, required=True, choices=(2, 3))
    ap.add_argument("--restart", choices=("true", "false"), required=True,
                    help="true: continue from wrfrst at this segment's start time")
    ap.add_argument("--restart-interval-min", type=int, required=True,
                    help="minutes; 0 means 'never write a restart' (mapped to %d)" % NEVER_MIN)
    ap.add_argument("--max-dom", type=int, default=1)
    # OPTIONAL and default None: when omitted this script does not touch time_step at all,
    # so every existing caller (rk_alternation.sbatch) produces byte-identical namelists.
    # Only the dt/accuracy trade-off driver passes it.
    ap.add_argument("--time-step", type=int, default=None,
                    help="&domains time_step in seconds; omitted = leave the template's value alone")
    a = ap.parse_args()

    t0 = dt.datetime.strptime(a.base_start, WRF_TIME) + dt.timedelta(minutes=a.offset_min)
    t1 = t0 + dt.timedelta(minutes=a.run_min)
    nd = a.max_dom

    with open(a.namelist) as fh:
        text = fh.read()

    want = {}
    for key, val in (("start_year", t0.year), ("start_month", "%02d" % t0.month),
                     ("start_day", "%02d" % t0.day), ("start_hour", "%02d" % t0.hour),
                     ("start_minute", "%02d" % t0.minute), ("start_second", "%02d" % t0.second),
                     ("end_year", t1.year), ("end_month", "%02d" % t1.month),
                     ("end_day", "%02d" % t1.day), ("end_hour", "%02d" % t1.hour),
                     ("end_minute", "%02d" % t1.minute), ("end_second", "%02d" % t1.second)):
        want[key] = ("&time_control", dup(val, nd))
    # run_* : configs.py already set run_minutes to this segment's length, but be explicit --
    # WRF takes the domain-1 stop time from run_* and ignores end_* for max_dom=1.
    want["run_days"] = ("&time_control", "0")
    want["run_hours"] = ("&time_control", "0")
    want["run_minutes"] = ("&time_control", str(a.run_min))
    want["run_seconds"] = ("&time_control", "0")
    want["restart"] = ("&time_control", ".true." if a.restart == "true" else ".false.")
    want["restart_interval"] = ("&time_control",
                                str(a.restart_interval_min if a.restart_interval_min > 0
                                    else NEVER_MIN))
    want["rk_ord"] = ("&dynamics", str(a.rk_ord))
    if a.time_step is not None:
        # time_step is a max_domains array in WRF; dup() replicates it for a nest.  The
        # fractional part must be zeroed too, or a template carrying time_step_fract_num/=0
        # silently gives a dt that is not the integer requested.
        want["time_step"] = ("&domains", dup(a.time_step, nd))
        want["time_step_fract_num"] = ("&domains", dup(0, nd))
        want["time_step_fract_den"] = ("&domains", dup(1, nd))

    for key, (section, value) in want.items():
        text = set_key(text, key, value, section)

    with open(a.namelist, "w") as fh:
        fh.write(text)

    # Read back and verify: a silently-missed substitution is the whole reason this file
    # exists, so never report success without checking.
    with open(a.namelist) as fh:
        back = fh.read()
    bad = []

    def norm(v):
        # compare as an element LIST, so a nest that got "2023, 2023, 2023" (one element
        # too many) fails here instead of running with d02 on the wrong date
        return [x.strip() for x in str(v).split(",") if x.strip()]

    for key, (_section, value) in want.items():
        got = get_key(back, key)
        if got is None or norm(got) != norm(value):
            bad.append("%s: want %r got %r" % (key, value, got))
    if bad:
        print("segnl: VERIFY FAILED\n  " + "\n  ".join(bad), file=sys.stderr)
        return 3
    print("segnl ok: start=%s end=%s run_min=%d restart=%s rst_int=%s rk_ord=%d dt=%s"
          % (t0.strftime(WRF_TIME), t1.strftime(WRF_TIME), a.run_min,
             want["restart"][1], want["restart_interval"][1], a.rk_ord,
             a.time_step if a.time_step is not None else "(template)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
