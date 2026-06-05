#!/usr/bin/env python3
"""Chain Union: progressively union sub-domain activations to verify old 2569pt beta1."""
import sys, json, numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances
from ripser import ripser

EPS = 0.03

def betti(pts, eps_pred_override=None):
    eps_max = float(np.max(pairwise_distances(pts)))
    eps_pred = eps_pred_override if eps_pred_override is not None else EPS * eps_max
    dgms = ripser(pts, maxdim=1)['dgms']
    b1 = sum(1 for d in dgms[1] if (d[1]-d[0]) > eps_pred)
    return b1, eps_pred, eps_max

act_dir = Path(sys.argv[1])
model = sys.argv[2]
L2 = int(sys.argv[3])
out_path = sys.argv[4] if len(sys.argv) > 4 else f"results/smoke_chain_{model}.json"

subsets = ["reasoning", "bbh_ff", "bbh_nav", "bbh_ts", "math500"]

print(f"Chain Union: {model} L{L2}")
print(f"{'Step':<20} {'beta1':>6} {'eps_pred':>10} {'inc':>6}")
print("-" * 50)

cumulative = None
prev_b1 = 0
results = {}
indiv_sum = 0
ref_eps_pred = None  # frozen from first point cloud

for i, sub in enumerate(subsets):
    npy = act_dir / f"{model}_L{L2}_{sub}.npy"
    pts = np.load(npy).astype(np.float32)
    
    b1_indiv, eps_indiv, eps_max_indiv = betti(pts)
    results[f"indiv_{sub}"] = {"beta1": b1_indiv, "eps_pred": round(eps_indiv, 4), "n": len(pts)}
    indiv_sum += b1_indiv
    
    # Freeze ref_eps_pred from first subset
    if ref_eps_pred is None:
        ref_eps_pred = eps_indiv
    
    cumulative = pts if cumulative is None else np.concatenate([cumulative, pts], axis=0)
    b1u, epsu, eps_max_u = betti(cumulative, eps_pred_override=ref_eps_pred)
    inc = b1u - prev_b1
    
    step_key = f"step{i+1}_{sub}"
    results[step_key] = {"beta1": b1u, "eps_pred": round(ref_eps_pred, 4), "eps_max_u": round(eps_max_u, 4), "n": len(cumulative), "increment": inc}
    
    print(f"  +{sub:<16}  {b1u:>6}  (ref_eps={ref_eps_pred:.4f})  {inc:>+5}")
    prev_b1 = b1u

final_b1 = results[f"step5_math500"]["beta1"]
interface = final_b1 - indiv_sum

print("-" * 50)
print(f"  Sum individual:  {indiv_sum}")
print(f"  Full union:      {final_b1}")
print(f"  Interface loops: {interface} (should be >0 for base, ~0 for inst)")
print(f"  Old 2569pt:      ~169 (expected if same prompt distribution)")

out = {"model": model, "L2": L2, "indiv_sum": indiv_sum,
       "full_union": final_b1, "interface_loops": interface,
       "old_2569pt_expected": 169, "results": results}
Path(out_path).parent.mkdir(exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved: {out_path}")
