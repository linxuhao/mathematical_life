#!/usr/bin/env python3
"""Fair tripartite test with equal sample sizes via downsampling."""
import json, numpy as np, os
from sklearn.metrics import pairwise_distances
os.environ['OMP_NUM_THREADS'] = '4'
from ripser import ripser
from persim import bottleneck
np.random.seed(42)

acts = np.load('activations_v3/Qwen3.5-2B-Inst_L14_reasoning.npy').astype(np.float32)
h_acts = np.load('activations_v3/Qwen3.5-2B-Inst_L14_hallucination.npy').astype(np.float32)
with open('activations_v3/Qwen__Qwen3.5-2B_L12_5shot_labels.json') as f:
    labels = json.load(f)

r_true = acts[labels['correct_idx']]
r_flawed_full = acts[labels['incorrect_idx']]

N = 200
r_t = r_true[:N]
r_f = r_flawed_full[np.random.choice(len(r_flawed_full), N, replace=False)]
h_s = h_acts[np.random.choice(len(h_acts), N, replace=False)]

def betti(pts):
    em = float(np.max(pairwise_distances(pts)))
    ep = 0.03 * em
    dgms = ripser(pts, maxdim=1)['dgms']
    lt = dgms[1][:,1] - dgms[1][:,0]
    lt = lt[lt > 0]
    return {
        'b1': int(np.sum(lt > ep)),
        'raw': len(lt),
        'surv': int(np.sum(lt > 0.01 * em)),
        'ep': round(ep, 4),
        'h1': dgms[1].tolist(),
    }

rt = betti(r_t)
rf = betti(r_f)
hh = betti(h_s)

sr = lambda d: d['surv'] / d['raw'] * 100 if d['raw'] > 0 else 0
print(f'R_true (n={N}):    b1={rt["b1"]:>3} raw={rt["raw"]:>3} surv={rt["surv"]:>3} ({sr(rt):.1f}%)')
print(f'R_flawed (n={N}):  b1={rf["b1"]:>3} raw={rf["raw"]:>3} surv={rf["surv"]:>3} ({sr(rf):.1f}%)')
print(f'H (n={N}):         b1={hh["b1"]:>3} raw={hh["raw"]:>3} surv={hh["surv"]:>3} ({sr(hh):.1f}%)')

# Union
ep = rt['ep']
def union_test(pa, pb):
    pu = np.concatenate([pa, pb], axis=0)
    dgms = ripser(pu, maxdim=1)['dgms']
    lt = dgms[1][:,1] - dgms[1][:,0]
    lt = lt[lt > 0]
    return int(np.sum(lt > ep))

u1 = union_test(r_t, r_f)
u2 = union_test(r_t, h_s)

def verdict(u, ba, bb):
    if u >= 0.8 * (ba + bb): return 'SEPARATE'
    if u <= 1.2 * max(ba, bb): return 'SHARED'
    return 'MIXED'

v1 = verdict(u1, rt['b1'], rf['b1'])
v2 = verdict(u2, rt['b1'], hh['b1'])
print(f'\nR_true U R_flawed: U={u1} sum={rt["b1"]+rf["b1"]} -> {v1}')
print(f'R_true U H:        U={u2} sum={rt["b1"]+hh["b1"]} -> {v2}')

def filt(diag, eps):
    arr = np.array([[b, d] for b, d in diag if (d - b) > eps])
    return arr if len(arr) > 0 else np.zeros((0, 2))

bd1 = bottleneck(filt(rt['h1'], rt['ep']), filt(rf['h1'], rf['ep']))
bd2 = bottleneck(filt(rt['h1'], rt['ep']), filt(hh['h1'], hh['ep']))
print(f'\nBottleneck R_true<->R_flawed: {bd1:.4f}')
print(f'Bottleneck R_true<->H:        {bd2:.4f}')
print(f'PHI(R_flawed/H) = {sr(rf)/sr(hh):.2f}' if sr(hh) > 0 else 'PHI = N/A')
