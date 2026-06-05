"""Bootstrap the beta1 'sensor' — quantify sampling variance of the topology readout.
For representative static clouds, resample prompts B times and recompute beta1(filtered),
beta1_raw, survival via the canonical actopo.measure. Report mean/std/95% CI.
Two regimes:
  - FULL-N, with-replacement bootstrap -> sensor variance of the headline (1319-pt) numbers.
  - n=300 subsample (without replacement) -> sampling-noise FLOOR at the dynamic-arm cloud size,
    so we can say what fraction of the dynamic step-to-step jitter is just sampling.
CPU, resumable. Runs in parallel with the GPU PR job + modesep job.
"""
import sys,json,glob,re
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.home()/"papers/mathematical-life/actopo/src"))
from actopo import FROZEN_V5, measure

ACT=Path.home()/"papers/mathematical-life/experiments/activations_v3"
OUT=Path.home()/"papers/mathematical-life/experiments/results/bootstrap_beta1.json"
B=100
# representative subset (substring match against *_reasoning.npy / *_hallucination.npy)
TARGETS=["Qwen3.5-2B-base","Qwen3.5-4B-base","Qwen3.5-9B-base","Qwen3.5-9B-Inst",
         "Qwen2.5-0.5B-base","Qwen3-1.7B-base"]
MODES=["reasoning","hallucination"]

def files():
    out=[]
    for t in TARGETS:
        for mode in MODES:
            g=sorted(ACT.glob(f"{t}_*_{mode}.npy"))
            if g: out.append((f"{t}_{mode}", g[0]))
    return out

def boot(X, n, replace, rng):
    b1=[];raw=[];surv=[]
    for _ in range(B):
        idx=rng.choice(len(X), size=n, replace=replace)
        r=measure(X[idx].astype(np.float32), FROZEN_V5)
        b1.append(r.beta1); raw.append(r.beta1_raw); surv.append(r.survival_rate)
    def stat(a):
        a=np.array(a,float)
        return {"mean":float(a.mean()),"std":float(a.std()),
                "ci":[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]}
    return {"beta1":stat(b1),"beta1_raw":stat(raw),"surv":stat(surv)}

res=json.load(open(OUT)) if OUT.exists() else {}
rng=np.random.default_rng(0)
for name,f in files():
    if name in res: continue
    X=np.load(f)
    point=measure(X.astype(np.float32),FROZEN_V5)   # the canonical point estimate
    rec={"n_full":len(X),
         "point":{"beta1":point.beta1,"beta1_raw":point.beta1_raw,"surv":point.survival_rate},
         "bootstrap_fullN":boot(X,len(X),True,rng)}
    if len(X)>=300:
        rec["subsample_n300"]=boot(X,300,False,rng)
    res[name]=rec
    p=rec["point"];bf=rec["bootstrap_fullN"]["beta1"];s3=rec.get("subsample_n300",{}).get("beta1",{})
    print(f"{name:34s} point b1={p['beta1']:3d} | fullN {bf['mean']:.1f}±{bf['std']:.1f} CI{bf['ci']} | n300 {s3.get('mean',float('nan')):.1f}±{s3.get('std',float('nan')):.1f}",flush=True)
    json.dump(res,open(OUT,"w"),indent=1)
print("DONE",flush=True)
