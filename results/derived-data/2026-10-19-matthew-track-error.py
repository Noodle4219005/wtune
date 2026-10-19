# Track/intensity error of a WTune a-deck against the HURDAT2 best track (ATCF, 34 kt rows).
# usage: python3 -I 2026-10-19-matthew-track-error.py adeck_<run>.dat bal142016.dat > out.csv
import sys, math, datetime as dt


def ll(s):
    v = int(s[:-1]) / 10.0
    return -v if s[-1] in "SW" else v


def read(p, tech=None):
    out = {}
    for line in open(p):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 10 or (f[11] if len(f) > 11 else "34") not in ("34", ""):
            continue
        if tech and f[4] != tech:
            continue
        t0 = dt.datetime.strptime(f[2], "%Y%m%d%H") + dt.timedelta(hours=int(f[5] or 0))
        out.setdefault(t0, (ll(f[6]), ll(f[7]), int(f[8]), int(f[9])))
    return out


a, b = read(sys.argv[1]), read(sys.argv[2], "BEST")
print("valid_utc,lead_h,model_lat,model_lon,best_lat,best_lon,err_km,model_vmax_kt,best_vmax_kt,"
      "model_mslp_hpa,best_mslp_hpa")
t0 = min(a)
for t in sorted(a):
    if t not in b:
        continue
    (la, lo, va, pa), (lb, lob, vb, pb) = a[t], b[t]
    p1, p2, dl = math.radians(la), math.radians(lb), math.radians(lob - lo)
    d = 6371.0 * math.acos(min(1, math.sin(p1) * math.sin(p2)
                               + math.cos(p1) * math.cos(p2) * math.cos(dl)))
    print("%s,%d,%.1f,%.1f,%.1f,%.1f,%.0f,%d,%d,%d,%d" % (
        t.strftime("%Y-%m-%dT%HZ"), (t - t0).total_seconds() // 3600, la, lo, lb, lob, d,
        va, vb, pa, pb))
