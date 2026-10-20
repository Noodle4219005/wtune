"""Rescore every eval round with the current score.py and check, into one CSV:

    python3 campaign/event_eval/build_csv.py RUNS_DIR OUT.csv

RUNS_DIR holds one run_pi.sh output folder per round and set (names in RUNS)."""
import csv, os, sys
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(W, "campaign", "event_eval"))
import yaml, score
S = sys.argv[1]
EXP = {"main": os.path.join(W, "tests/fixtures/events/expected.yaml"),
       "holdout": os.path.join(W, "tests/fixtures/events-holdout/expected.yaml")}
RUNS = [(1, "main", "", "ev-mm-low"), (2, "main", "", "ev-mm-low-r2"),
        (2, "holdout", "", "ev-mm-low-r2-holdout"), (3, "main", "", "r3-main"),
        (3, "holdout", "a", "r3-hold-a"), (3, "holdout", "b", "r3-hold-b"),
        (3, "holdout", "c", "r3-hold-c"), (4, "main", "", "r4-main"),
        (4, "holdout", "a", "r4-hold-a"), (4, "holdout", "b", "r4-hold-b"),
        (4, "holdout", "c", "r4-hold-c")]
cols = ["round", "set", "repeat", "case", "passed", "check", "ask_ok", "fail", "missing",
        "extra", "seconds", "turns", "input_tokens", "output_tokens", "cost_usd"]
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=cols)
w.writeheader()
for rnd, st, rep, d in RUNS:
    exp = yaml.safe_load(open(EXP[st]))
    tim = {}
    for line in open(os.path.join(S, d, "timing.txt")):
        f = line.split()
        tim[f[0]] = f[2:]
    for case, e in exp.items():
        r = score.score_case(os.path.join(S, d, case), e)
        t = tim.get(case, [])
        row = dict(round=rnd, set=st, repeat=rep, case=case, passed=int(r["passed"]),
                   check=int(r["check"]), ask_ok=int(r["ask_ok"]), fail=";".join(r["fail"]),
                   missing=";".join(r["missing"]), extra=";".join(r["extra"]),
                   seconds=t[0].rstrip("s") if t else "")
        if len(t) == 5:
            row.update(turns=t[1], input_tokens=t[2], output_tokens=t[3], cost_usd=t[4])
        w.writerow(row)
