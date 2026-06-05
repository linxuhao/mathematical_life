import json, sys, numpy as np, time
sys.path.insert(0, "/home/linxuhao/papers/mathematical-life/actopo/src")
from actopo import FROZEN_V5, measure
Z="/home/linxuhao/papers/mathematical-life/experiments/activations_v3/Qwen__Qwen3.5-2B_L12_tokenwise.npz"
z=np.load(Z); acts=z["acts"].astype("float32"); lens=z["lens"]
offs=np.concatenate([[0],np.cumsum(lens)]); ci=set(int(x) for x in z["correct_idx"])
rows=[]; t0=time.time()
for i in range(len(lens)):
    t=acts[offs[i]:offs[i+1]]
    if t.shape[0]<10: continue
    m=measure(t,FROZEN_V5)
    rows.append({"idx":i,"correct":i in ci,"beta1":m.beta1,"raw":m.beta1_raw,"surv":m.survival_rate,"ntok":int(t.shape[0])})
    if i%150==0: print(f"{i}/{len(lens)} {time.time()-t0:.0f}s",flush=True)
json.dump(rows,open("/home/linxuhao/papers/mathematical-life/experiments/results/tokenwise_reanalysis.json","w"))
c=np.array([r["beta1"] for r in rows if r["correct"]],float)
ic=np.array([r["beta1"] for r in rows if not r["correct"]],float)
def auc(pos,neg):
    allv=np.concatenate([pos,neg]); order=allv.argsort(kind="mergesort")
    ranks=np.empty(len(allv)); ranks[order]=np.arange(1,len(allv)+1)
    R=ranks[:len(pos)].sum(); return (R-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
d=(c.mean()-ic.mean())/np.sqrt((c.var()+ic.var())/2)
print(f"=== CORRECT n={len(c)} mean={c.mean():.2f}±{c.std():.1f} | INCORRECT n={len(ic)} mean={ic.mean():.2f}±{ic.std():.1f}")
print(f"=== Cohen d={d:.3f}  AUC P(correct>incorrect)={auc(c,ic):.3f}  (0.5=null)")
