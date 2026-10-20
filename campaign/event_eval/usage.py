"""Sum token usage and cost of one `pi --mode json` run: usage.py EVENTS.jsonl REPLY.txt
-> prints "turns input output cost_usd" and writes the last assistant text to REPLY.txt."""
import json
import sys

turns = inp = out = 0
cost = 0.0
reply = ""
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    m = d.get("message") or {}
    if d.get("type") != "message_end" or m.get("role") != "assistant":
        continue
    u = m.get("usage") or {}
    turns += 1
    inp += u.get("input", 0) + u.get("cacheRead", 0)
    out += u.get("output", 0)
    cost += (u.get("cost") or {}).get("total", 0.0)
    text = "".join(c.get("text", "") for c in m.get("content") or [] if c.get("type") == "text")
    if text.strip():
        reply = text
open(sys.argv[2], "w").write(reply + "\n")
print("%d %d %d %.6f" % (turns, inp, out, cost))
