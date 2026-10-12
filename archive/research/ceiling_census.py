import glob, os, sys
sys.path.insert(0,'/home/u5727520/wrf-stack/wtune')
import numpy as np
from decay_gate import load_bucket_csv
R='/home/u5727520/wrf-results'
rows=[]
for p in glob.glob(os.path.join(R,'*','bucket_load.csv')):
    c=os.path.basename(os.path.dirname(p))
    try: comp,bsec=load_bucket_csv(p)
    except Exception: continue
    used=comp.sum(axis=0)>0
    if used.sum()<3: continue
    last=int(np.where(used)[0].max()); core=comp[:,1:last]
    tot=core.sum(axis=1)
    if tot.sum()<=0 or core.shape[1]<2: continue
    span=float(core.sum(axis=0).max())          # wall proxy: busiest rank's elapsed compute+
    wall=core.shape[1]*bsec
    rows.append((c, tot.max()/tot.mean(), 100.0*(tot.max()-tot.mean())/wall, wall))
rows.sort(key=lambda r:-r[2])
print("%-28s %9s %14s %9s" % ("case","max/mean","LOWER-BOUND","wall_s"))
print("%-28s %9s %14s %9s" % ("","","recoverable %",""))
for c,r,pct,w in rows[:14]:
    flag = "  <-- above the 8.82% CONUS ceiling" if pct>8.82 else ""
    print("%-28s %9.3f %13.1f%% %9.0f%s" % (c,r,pct,w,flag))
above=[r for r in rows if r[2]>8.82]
print("\n%d of %d cases exceed 8.82%%   median=%.1f%%  max=%.1f%%"
      % (len(above), len(rows), np.median([r[2] for r in rows]), max(r[2] for r in rows)))
