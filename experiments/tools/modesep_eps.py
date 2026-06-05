"""Mode-separation epsilon-robustness (CPU). The surviving positive pillar is that
Reasoning and Hallucination occupy SEPARATE topological regions (union test). We
showed beta1-COLLAPSE is threshold-fragile; here we test whether the SEPARATION
verdict is threshold-ROBUST across betti_eps_frac. For every model with both
reasoning & hallucination clouds, run the canonical actopo union_test (same_basis)
at a sweep of eps fractions and record the verdict + overlap.
"""
import sys,json,glob,re,dataclasses
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.home()/"papers/mathematical-life/actopo/src"))
import actopo
from actopo import FROZEN_V5, union_test

ACT=Path.home()/"papers/mathematical-life/experiments/activations_v3"
OUT=Path.home()/"papers/mathematical-life/experiments/results/modesep_eps.json"
FRACS=[0.01,0.02,0.03,0.05,0.08,0.10]

# discover model prefixes that have BOTH *_reasoning.npy and *_hallucination.npy
pairs=[]
for rf in sorted(ACT.glob("*_reasoning.npy")):
    hf=Path(str(rf).replace("_reasoning.npy","_hallucination.npy"))
    if hf.exists():
        prefix=rf.name.replace("_reasoning.npy","")
        pairs.append((prefix,rf,hf))
print(f"{len(pairs)} R/H within-basis pairs",flush=True)

res=json.load(open(OUT)) if OUT.exists() else {}
for prefix,rf,hf in pairs:
    if prefix in res: continue
    R=np.load(rf).astype(np.float32); H=np.load(hf).astype(np.float32)
    row={}
    for frac in FRACS:
        cfg=dataclasses.replace(FROZEN_V5,betti_eps_frac=frac)
        try:
            u=union_test(R,H,cfg,same_basis=True)
            row[str(frac)]={"verdict":u.get("verdict"),"A":u.get("A_beta1"),"B":u.get("B_beta1"),
                            "U":u.get("U_beta1"),"overlap_pct":u.get("overlap_pct")}
        except Exception as e:
            row[str(frac)]={"error":str(e)[:80]}
    res[prefix]=row
    verds=" ".join(f"{f}:{row[str(f)].get('verdict','ERR')}" for f in FRACS)
    print(f"{prefix:32s} {verds}",flush=True)
    json.dump(res,open(OUT,"w"),indent=1)
print("DONE",flush=True)
