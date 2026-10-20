"""Score event.yaml translations against tests/fixtures/events/expected.yaml.

    python3 campaign/event_eval/score.py OUT_DIR [--expected FILE] [--csv results.csv]

OUT_DIR/<case>/event.yaml (+ event.txt) is one translation per case. A case passes when
`wtune event check` passes, every required item is present, no item outside required+allowed
appears, the storms and local paths match, and `questions` is non-empty exactly when expected.
A required entry "a|b" is met by either value.
"""
import argparse
import csv
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXPECTED = os.path.join(REPO, "tests", "fixtures", "events", "expected.yaml")
WTUNE = os.path.join(REPO, "bin", "wtune")


def _values(doc, key, field="value"):
    return {i.get(field) for i in (doc.get(key) or []) if isinstance(i, dict)}


def score_case(case_dir, exp):
    path = os.path.join(case_dir, "event.yaml")
    r = dict(check=False, missing=[], extra=[], ask_ok=False, passed=False, note="")
    if not os.path.exists(path):
        r["note"] = "no event.yaml"
        return r
    p = subprocess.run([WTUNE, "event", "check", path], capture_output=True, text=True)
    r["check"] = p.returncode == 0
    if not r["check"]:
        r["note"] = "; ".join(l for l in p.stdout.splitlines() if l.startswith("error"))[:300]
    try:
        doc = yaml.safe_load(open(path)) or {}
    except yaml.YAMLError as e:
        r["note"] = "yaml: %s" % e
        return r
    for key, field in (("types", "value"), ("verify", "value"), ("obs_requests", "kind")):
        want = exp.get(key) or {"required": [], "allowed": []}
        got = _values(doc, key, field)
        r["missing"] += ["%s:%s" % (key, x) for x in want["required"]
                         if not set(x.split("|")) & got]
        ok = {v for x in want["required"] for v in x.split("|")} | set(want.get("allowed") or [])
        r["extra"] += ["%s:%s" % (key, x) for x in sorted(got - ok, key=str)]
    names = " ".join(str(s.get("name", "")) for s in doc.get("storms") or [] if isinstance(s, dict))
    r["missing"] += ["storm:%s" % n for n in exp.get("storms") or []
                     if not any(a.lower() in names.lower() for a in n.split("|"))]
    if not exp.get("storms") and doc.get("storms"):
        r["extra"].append("storms:%s" % names)
    paths = _values(doc, "local_obs", "path")
    r["missing"] += ["local:%s" % x for x in exp.get("local_obs") or [] if x not in paths]
    if exp.get("diagnose") and not doc.get("diagnose"):
        r["missing"].append("diagnose")
    r["ask_ok"] = bool(doc.get("questions")) == bool(exp.get("ask"))
    r["passed"] = r["check"] and not r["missing"] and not r["extra"] and r["ask_ok"]
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--expected", default=EXPECTED)
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    expected = yaml.safe_load(open(a.expected))
    rows = []
    for case, exp in expected.items():
        r = score_case(os.path.join(a.out_dir, case), exp)
        rows.append(dict(case=case, **{k: (";".join(v) if isinstance(v, list) else v)
                                       for k, v in r.items()}))
        print("%-20s %-4s check=%-5s ask=%-5s missing=%s extra=%s %s"
              % (case, "PASS" if r["passed"] else "FAIL", r["check"], r["ask_ok"],
                 ",".join(r["missing"]) or "-", ",".join(r["extra"]) or "-", r["note"]))
    n = sum(r["passed"] for r in rows)
    print("passed %d / %d" % (n, len(rows)))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
