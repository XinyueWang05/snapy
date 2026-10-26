import csv, sys, numpy as np
for tree in ("base","fix"):
  for n in sys.argv[1:]:
    try: rows=list(csv.DictReader(open(f"runs/{tree}/{n}.csv")))
    except Exception: continue
    t=np.array([float(r['t']) for r in rows])/86400; w=np.array([float(r['rms_w']) for r in rows])
    out=[]
    for td in (0.1,0.25,0.5,1,1.5,2,2.5,2.75,3):
      k=np.argmin(abs(t-td))
      if abs(t[k]-td)<0.02: out.append(f"{td}:{w[k]:.2e}")
    print(tree,n,len(rows)," ".join(out))
    for a,b in ((0.5,1.5),(1.5,2.5),(2,3),(2.5,3)):
      s=(t>=a)&(t<=b)&(w>0)
      if s.sum()>3: print("   fit %.1f-%.1f d: %+.2f /d"%(a,b,np.polyfit(t[s],np.log(w[s]),1)[0]))
