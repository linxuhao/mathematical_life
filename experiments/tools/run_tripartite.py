#!/usr/bin/env python3
"""Full tripartite pipeline: label → split → Union + bottleneck + persistence.
One command after the labeling is done.

Usage:
  python tools/run_tripartite.py \
      --activations activations_v3/Qwen3.5-2B-Inst_L14_reasoning.npy \
      --labels activations_v3/Qwen__Qwen3.5-2B_L14_reasoning_labels.json \
      --hallucination activations_v3/Qwen3.5-2B-Inst_L14_hallucination.npy \
      -o results/tripartite_v1.json
"""
import json, argparse, time, os
from pathlib import Path
import numpy as np

EPS_FRAC = 0.03

def betti_from_points(points):
    from sklearn.metrics import pairwise_distances
    eps_max = float(np.max(pairwise_distances(points)))
    eps_pred = EPS_FRAC * eps_max

    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms = ripser(points, maxdim=1)['dgms']
    h1 = dgms[1]
    lifetimes = h1[:, 1] - h1[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    beta1_raw = len(lifetimes)
    beta1 = int(np.sum(lifetimes > eps_pred))
    surv_001 = int(np.sum(lifetimes > 0.01 * eps_max))

    return {
        'beta1_raw': beta1_raw, 'beta1': beta1,
        'eps_max': round(eps_max, 4), 'eps_pred': round(eps_pred, 4),
        'surv_001': surv_001,
        'surv_rate': round(surv_001/beta1_raw*100, 1) if beta1_raw > 0 else 0,
        'h1_diagram': h1.tolist(),
        'n_points': points.shape[0],
        'dim': points.shape[1],
    }

def union_test(points_a, points_b, eps_ref):
    points_u = np.concatenate([points_a, points_b], axis=0)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms = ripser(points_u, maxdim=1)['dgms']
    h1 = dgms[1]
    lifetimes = h1[:, 1] - h1[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    bu = int(np.sum(lifetimes > eps_ref))

    ba = int(np.sum((r_true['h1_diagram'][:,1]-r_true['h1_diagram'][:,0])[r_true['h1_diagram'][:,1]-r_true['h1_diagram'][:,0]>0] > eps_ref) if 'r_true' in dir() else ...)
    # Actually just recompute from diagram
    return None  # placeholder

# Let me just rewrite this properly in the main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations', required=True, help='Existing reasoning.npy (1319 prompts)')
    parser.add_argument('--labels', required=True, help='Labels from label_gsm8k.py')
    parser.add_argument('--hallucination', required=True, help='Hallucination .npy')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    print("=== TRIPARTITE TEST ===")
    print()

    # Load and split
    print("--- Loading data ---")
    acts = np.load(args.activations).astype(np.float32)
    with open(args.labels) as f:
        labels = json.load(f)
    h_acts = np.load(args.hallucination).astype(np.float32)

    correct_idx = labels['correct_idx']
    incorrect_idx = labels['incorrect_idx']
    print(f"Total: {acts.shape[0]}, Correct: {len(correct_idx)}, Incorrect: {len(incorrect_idx)}, H: {h_acts.shape[0]}")
    print(f"Accuracy: {labels['accuracy']:.1f}%")

    # Split
    r_true = acts[correct_idx]
    r_flawed = acts[incorrect_idx]

    # Compute individual betti
    print("\n--- Individual persistence ---")
    t0 = time.time()
    rt = betti_from_points(r_true)
    print(f"  R_true:    n={rt['n_points']}, raw={rt['beta1_raw']}, β₁={rt['beta1']}, surv@0.01={rt['surv_001']} ({rt['surv_rate']}%)")
    rf = betti_from_points(r_flawed)
    print(f"  R_flawed:  n={rf['n_points']}, raw={rf['beta1_raw']}, β₁={rf['beta1']}, surv@0.01={rf['surv_001']} ({rf['surv_rate']}%)")
    hh = betti_from_points(h_acts)
    print(f"  H:         n={hh['n_points']}, raw={hh['beta1_raw']}, β₁={hh['beta1']}, surv@0.01={hh['surv_001']} ({hh['surv_rate']}%)")

    # PHI
    phi = round(rf['surv_rate'] / hh['surv_rate'], 2) if hh['surv_rate'] > 0 else None
    print(f"\n  PHI(R_flawed/H) = {phi}")

    # Union tests
    print("\n--- Union tests ---")
    eps_ref = rt['eps_pred']
    print(f"  eps_ref (from R_true): {eps_ref}")

    # R_true ∪ R_flawed
    points_u1 = np.concatenate([r_true, r_flawed], axis=0)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms_u1 = ripser(points_u1, maxdim=1)['dgms']
    h1_u1 = dgms_u1[1]
    lt_u1 = h1_u1[:, 1] - h1_u1[:, 0]
    lt_u1 = lt_u1[lt_u1 > 0]
    bu1 = int(np.sum(lt_u1 > eps_ref))

    inter1 = rt['beta1'] + rf['beta1'] - bu1
    ov1 = round(inter1 / max(rt['beta1'], rf['beta1']) * 100, 1) if max(rt['beta1'], rf['beta1']) > 0 else 0
    if bu1 >= 0.8 * (rt['beta1'] + rf['beta1']): v1 = 'SEPARATE'
    elif bu1 <= 1.2 * max(rt['beta1'], rf['beta1']): v1 = 'SHARED'
    else: v1 = 'MIXED'
    print(f"  R_true ∪ R_flawed: β₁(U)={bu1} (sum={rt['beta1']+rf['beta1']}) → {v1} overlap={ov1}%")

    # R_true ∪ H
    points_u2 = np.concatenate([r_true, h_acts], axis=0)
    dgms_u2 = ripser(points_u2, maxdim=1)['dgms']
    h1_u2 = dgms_u2[1]
    lt_u2 = h1_u2[:, 1] - h1_u2[:, 0]
    lt_u2 = lt_u2[lt_u2 > 0]
    bu2 = int(np.sum(lt_u2 > eps_ref))

    inter2 = rt['beta1'] + hh['beta1'] - bu2
    ov2 = round(inter2 / max(rt['beta1'], hh['beta1']) * 100, 1) if max(rt['beta1'], hh['beta1']) > 0 else 0
    if bu2 >= 0.8 * (rt['beta1'] + hh['beta1']): v2 = 'SEPARATE'
    elif bu2 <= 1.2 * max(rt['beta1'], hh['beta1']): v2 = 'SHARED'
    else: v2 = 'MIXED'
    print(f"  R_true ∪ H:        β₁(U)={bu2} (sum={rt['beta1']+hh['beta1']}) → {v2} overlap={ov2}%")

    # Bottleneck
    print("\n--- Bottleneck distances ---")
    from persim import bottleneck
    def filt_dgm(diag_list, eps):
        arr = np.array([[b,d] for b,d in diag_list if (d-b) > eps])
        return arr if len(arr) > 0 else np.zeros((0,2))

    dgm_rt = np.array(rt['h1_diagram'])
    dgm_rf = np.array(rf['h1_diagram'])
    dgm_h = np.array(hh['h1_diagram'])

    bd1 = bottleneck(filt_dgm(rt['h1_diagram'], rt['eps_pred']),
                     filt_dgm(rf['h1_diagram'], rf['eps_pred']))
    bd2 = bottleneck(filt_dgm(rt['h1_diagram'], rt['eps_pred']),
                     filt_dgm(hh['h1_diagram'], hh['eps_pred']))
    print(f"  R_true ↔ R_flawed: {bd1:.4f}")
    print(f"  R_true ↔ H:        {bd2:.4f}")

    # Verdict
    print("\n=== VERDICT ===")
    predictions = []
    if v1 == 'SHARED' and bd1 < 0.5:
        predictions.append("✅ R_flawed NESTED in R_true — flawed reasoning is same manifold, localized error")
    elif v1 == 'SHARED':
        predictions.append("⚠️  R_flawed shares region but barcode differs — partial structural divergence")
    else:
        predictions.append(f"❌ R_flawed {v1} from R_true — different manifold (unexpected)")

    if v2 == 'SEPARATE' and bd2 > 0.5:
        predictions.append("✅ H is SEPARATE from R_true — hallucination is different topological entity")
    elif v2 == 'SEPARATE':
        predictions.append("⚠️  H separate region but similar barcode — coordinate rotation?")
    else:
        predictions.append(f"❌ H overlaps R_true — unexpected")

    if rf['surv_rate'] < rt['surv_rate']:
        predictions.append(f"✅ R_flawed ({rf['surv_rate']}%) less persistent than R_true ({rt['surv_rate']}%) — flawed reasoning is fragile")
    else:
        predictions.append("⚠️  Persistence order unexpected")

    for p in predictions:
        print(f"  {p}")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.0f}s")

    # Save
    result = {
        'r_true': {k:v for k,v in rt.items() if k != 'h1_diagram'},
        'r_flawed': {k:v for k,v in rf.items() if k != 'h1_diagram'},
        'hallucination': {k:v for k,v in hh.items() if k != 'h1_diagram'},
        'union_Rtrue_Rflawed': {'verdict': v1, 'U_beta1': bu1, 'sum': rt['beta1']+rf['beta1'],
                                'intersection': inter1, 'overlap_pct': ov1},
        'union_Rtrue_H': {'verdict': v2, 'U_beta1': bu2, 'sum': rt['beta1']+hh['beta1'],
                          'intersection': inter2, 'overlap_pct': ov2},
        'bottleneck_Rtrue_Rflawed': round(float(bd1), 6),
        'bottleneck_Rtrue_H': round(float(bd2), 6),
        'phi_Rflawed_H': phi,
        'predictions': predictions,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {args.output}")

if __name__ == '__main__':
    main()
