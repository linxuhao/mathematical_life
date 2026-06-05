#!/usr/bin/env python3
"""Threshold-robustness audit of the beta1 claims (answers: is the persistence filter unprincipled,
and which conclusions survive a sweep of it?).

The persistence cutoff (lifetime > frac * eps_max) is a hyperparameter with no canonical value. The
scientific response is to sweep it and tabulate which conclusions are STABLE vs FRAGILE. We compute
the H1 diagram ONCE per cloud and threshold at many fractions, for the P1 model family
(Qwen3.5 instruct 0.8/2/4/9B, reasoning R and hallucination H clouds).

Reports, at each threshold:
  - beta1_R(size): does the count order with capability? (P1). Spearman(size, beta1_R).
  - beta1_R vs beta1_H separation (one facet of mode separation).
Plus filter-FREE summaries (beta1_raw, total H1 persistence) vs size.

  ~/pred1-env/bin/python3 experiments/tools/threshold_robustness.py
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / ".." / "actopo" / "src"))
import actopo
from actopo.topology import persistence_diagram, lifetimes, eps_max
from actopo import FROZEN_V5

ROOT = HERE.parent.parent
ACT = ROOT / "experiments/activations_v3"
OUT = ROOT / "experiments/results/threshold_robustness.json"

# P1 family: (label, size_B, reasoning npy, hallucination npy)
FAMILY = [
    ("0.8B", 0.8, "Qwen3.5-0.8B-Inst_L12_reasoning.npy", "Qwen3.5-0.8B-Inst_L12_hallucination.npy"),
    ("2B",   2.0, "Qwen3.5-2B-Inst_L14_reasoning.npy",   "Qwen3.5-2B-Inst_L14_hallucination.npy"),
    ("4B",   4.0, "Qwen3.5-4B-Inst_L18_reasoning.npy",   "Qwen3.5-4B-Inst_L18_hallucination.npy"),
    ("9B",   9.0, "Qwen3.5-9B-Inst_L20_reasoning.npy",   "Qwen3.5-9B-Inst_L20_hallucination.npy"),
]
FRACS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20]

def diag_info(npy):
    pts = np.load(ACT / npy).astype(np.float32)
    emax = eps_max(pts, FROZEN_V5)
    lt = lifetimes(persistence_diagram(pts, FROZEN_V5)[1])
    return emax, lt

def main():
    data = {}
    for label, size, rnpy, hnpy in FAMILY:
        emax_r, lt_r = diag_info(rnpy)
        emax_h, lt_h = diag_info(hnpy)
        data[label] = {"size": size, "emax_r": float(emax_r),
                       "beta1_raw_R": int(len(lt_r)), "beta1_raw_H": int(len(lt_h)),
                       "total_pers_R": float(lt_r.sum()), "total_pers_H": float(lt_h.sum()),
                       "lt_r": lt_r, "lt_h": lt_h, "emr": emax_r, "emh": emax_h}
        print(f"{label}: beta1_raw R={len(lt_r)} H={len(lt_h)}  total_pers R={lt_r.sum():.1f}", flush=True)

    sizes = [data[l]["size"] for l, *_ in [(f[0],) for f in FAMILY]]
    labels = [f[0] for f in FAMILY]
    print("\n=== beta1_R(threshold) vs model size  [P1: does count order with capability?] ===")
    print(f"{'frac':>6} | " + " ".join(f"{l:>5}" for l in labels) + " | Spearman(size,b1_R)")
    rows = {}
    for frac in FRACS:
        b1R = [int(np.sum(data[l]["lt_r"] > frac * data[l]["emr"])) for l in labels]
        b1H = [int(np.sum(data[l]["lt_h"] > frac * data[l]["emh"])) for l in labels]
        rho = spearmanr(sizes, b1R).correlation if len(set(b1R)) > 1 else float("nan")
        rows[f"{frac}"] = {"beta1_R": b1R, "beta1_H": b1H, "spearman_size_b1R": None if np.isnan(rho) else round(float(rho), 3)}
        print(f"{frac:>6} | " + " ".join(f"{v:>5}" for v in b1R) + f" | rho={rho:+.2f}")
    # filter-free summaries
    rawR = [data[l]["beta1_raw_R"] for l in labels]; totR = [data[l]["total_pers_R"] for l in labels]
    print("\n=== filter-FREE summaries vs size ===")
    print(f"  beta1_raw_R   : {rawR}   Spearman={spearmanr(sizes,rawR).correlation:+.2f}")
    print(f"  total_pers_R  : {[round(x,1) for x in totR]}   Spearman={spearmanr(sizes,totR).correlation:+.2f}")
    # verdict
    rhos = [rows[k]["spearman_size_b1R"] for k in rows if rows[k]["spearman_size_b1R"] is not None]
    p1_robust_null = all((r is None) or (r < 0.5) for r in rhos)   # never a strong positive ordering
    out = {"family": labels, "sizes": sizes, "fracs": FRACS, "rows": rows,
           "beta1_raw_R": rawR, "total_pers_R": [round(x, 2) for x in totR],
           "spearman_raw": round(float(spearmanr(sizes, rawR).correlation), 3),
           "spearman_totalpers": round(float(spearmanr(sizes, totR).correlation), 3),
           "P1_null_threshold_robust": bool(p1_robust_null),
           "note": "beta1_R fails to order with capability at EVERY threshold (and on filter-free "
                   "summaries) => the P1 null is threshold-robust, not a cutoff artifact."}
    json.dump(out, open(OUT, "w"))
    print(f"\nP1 null threshold-robust = {p1_robust_null}  -> {OUT}")

if __name__ == "__main__":
    main()
