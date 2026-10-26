import re, sys
def waves(ns, cap=8):
    w = []
    for n in sorted(ns):
        for x in w:
            if sum(x) + n <= cap: x.append(n); break
        else: w.append([n])
    return w
def sec(t):
    p = [int(x) for x in t.split(":")]; s = 0
    for x in p: s = s * 60 + x
    return s
blocks = open(sys.argv[1]).read().split("=== ")[1:]
print("case | sweep running | waves | s/wave | untried | untried waves | need | 15% of left | early")
for b in blocks:
    lines = b.splitlines(); name = lines[0].split("/")[0]
    sent, nolegal, nwaves, inwave, el = set(), set(), 0, False, 0
    for l in lines[1:]:
        m = re.match(r"node sweep: nodes-n(\d+) (submitted|skipped)", l)
        if m:
            sent.add(int(m.group(1)))
            if not inwave: nwaves += 1; inwave = True
            continue
        if l.startswith("P "):
            el = sec(l[2:]); inwave = False; continue
        inwave = False
        m = re.match(r"node sweep: N=(\d+) skipped", l)
        if m: nolegal.add(int(m.group(1)))
    untried = [n for n in range(1, 9) if n not in sent | nolegal]
    ws = el / max(1, nwaves); uw = len(waves(untried)); need = uw * ws; left = 2700 - el
    print("%s | %d:%02d | %d | %.0f | %s | %d | %d:%02d | %d:%02d | %s" % (
        name, el // 60, el % 60, nwaves, ws, untried, uw, need // 60, need % 60,
        int(.15 * left) // 60, int(.15 * left) % 60, "yes" if untried and need <= .15 * left else ("-" if not untried else "no")))
