"""Clean sampling-noise estimate for the beta1 'sensor': WITHOUT-replacement
subsampling at 90% of N (jackknife-style) — avoids the duplicate-point artifact that
biased the with-replacement bootstrap low. For each representative static cloud, draw
0.9*N distinct points B times, recompute beta1(filt), beta1_raw, survival; report
mean/std/95% CI at 0.9N plus the full-N point estimate. The std is an honest
sampling-noise scale for the reported beta1.
"""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.home()/"papers/mathematical-life/actopo/src"))
from actopo import FROZEN_V5, measure

ACT=Path.home()/"papers/mathematical-life/experiments/activations_v3"
OUT=Path.home()/"papers/mathematical-life/experiments/results/bootstrap_beta1_sub.json"
B=100; FRAC=0.9
TARGETS=["Qwen3.5-2B-base","Qwen3.5-4B-base","Qwen3.5-9B-base","Qwen3.5-9B-Inst",
         "Qwen2.5-0.5B-base","Qwen3-1.7B-base"]
MODES=["reasoning","hallucination"]

def files():
    out=[]
    for t in TARGETS:
        for m in MODES:
            g=sorted(ACT.glob(f"{t}_*_{m}.npy"))
            if g: out.append((f"{t}_{m}", g[0]))
    return out

def sub(X,n,rng):
    b1=[];raw=[];surv=[]
    for _ in range(B):
        idx=rng.choice(len(X),size=n,replace=False)
        r=measure(X[idx].astype(np.float32),FROZEN_V5)
        b1.append(r.beta1);raw.append(r.beta1_raw);surv.append(r.survival_rate)
    def st(a):
        a=np.array(a,float)
        return {"mean":float(a.mean()),"std":float(a.std()),
                "ci":[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]}
    return {"beta1":st(b1),"beta1_raw":st(raw),"surv":st(surv)}

res=json.load(open(OUT)) if OUT.exists() else {}
rng=np.random.default_rng(1)
for name,f in files():
    if name in res: continue
    X=np.load(f); n=int(round(FRAC*len(X)))
    pt=measure(X.astype(np.float32),FROZEN_V5)
    rec={"n_full":len(X),"n_sub":n,
         "point":{"beta1":pt.beta1,"beta1_raw":pt.beta1_raw,"surv":pt.survival_rate},
         "sub90":sub(X,n,rng)}
    res[name]=rec
    bf=rec["sub90"]["beta1"]
    print(f"{name:34s} point b1={pt.beta1:3d}  90%-sub {bf['mean']:.1f}±{bf['std']:.1f} CI{[round(c,1) for c in bf['ci']]}",flush=True)
    json.dump(res,open(OUT,"w"),indent=1)
print("DONE",flush=True)
