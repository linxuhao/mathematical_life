#!/usr/bin/env python3
"""Tripartite Test: R_true vs R_flawed vs H topological comparison.
Requires extract_split_gsm8k.py output first.

Usage:
  python tools/tripartite_test.py \
      --correct activations_v3/Qwen__Qwen3.5-2B-Inst_L14_reasoning_correct.npy \
      --incorrect activations_v3/Qwen__Qwen3.5-2B-Inst_L14_reasoning_incorrect.npy \
      --hallucination activations_v3/Qwen3.5-2B-Inst_L14_hallucination.npy \
      -o results/tripartite_v1.json
"""
import os, json, argparse, time
from pathlib import Path
import numpy as np

EPS_FRAC = 0.03

def compute_all(npy_path, label):
    """Compute full persistence data for a .npy file."""
    t0 = time.time()
    points = np.load(npy_path).astype(np.float32)
    if np.isnan(points).any():
        raise ValueError(f"NaN in {npy_path}")

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

    # Survival at ε=0.01×ε_max
    surv_001 = int(np.sum(lifetimes > 0.01 * eps_max))

    dt = time.time() - t0

    return {
        'label': label,
        'file': str(npy_path),
        'shape': list(points.shape),
        'eps_max': round(eps_max, 4),
        'eps_pred': round(eps_pred, 4),
        'beta1_raw': beta1_raw,
        'beta1': beta1,
        'survival_001': surv_001,
        'survival_rate_001': round(surv_001 / beta1_raw * 100, 1) if beta1_raw > 0 else 0,
        'h1_diagram': h1.tolist(),
        'time_s': round(dt, 1),
    }

def union_test(result_a, result_b, npy_a, npy_b):
    """Run Union test between two point clouds (same model, same basis)."""
    t0 = time.time()
    points_a = np.load(npy_a).astype(np.float32)
    points_b = np.load(npy_b).astype(np.float32)
    points_u = np.concatenate([points_a, points_b], axis=0)

    # Use eps_pred from A as reference (Type A — same model)
    eps_ref = result_a['eps_pred']

    from sklearn.metrics import pairwise_distances
    eps_max_u = float(np.max(pairwise_distances(points_u)))

    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms_u = ripser(points_u, maxdim=1)['dgms']
    h1_u = dgms_u[1]
    lifetimes_u = h1_u[:, 1] - h1_u[:, 0]
    lifetimes_u = lifetimes_u[lifetimes_u > 0]
    bu = int(np.sum(lifetimes_u > eps_ref))

    ba = result_a['beta1']
    bb = result_b['beta1']
    intersection = ba + bb - bu
    overlap_pct = round(intersection / max(ba, bb) * 100, 1) if max(ba, bb) > 0 else 0

    # Verdict
    if bu >= 0.8 * (ba + bb):
        verdict = 'SEPARATE'
    elif bu <= 1.2 * max(ba, bb):
        verdict = 'SHARED'
    else:
        verdict = 'MIXED'

    dt = time.time() - t0
    return {
        'A_beta1': ba, 'B_beta1': bb, 'U_beta1': bu,
        'sum': ba + bb, 'intersection': intersection,
        'overlap_pct': overlap_pct, 'verdict': verdict,
        'eps_ref': round(eps_ref, 4),
        'time_s': round(dt, 1),
    }

def bottleneck_dist(diag_a, diag_b, eps_a, eps_b):
    """Bottleneck distance between two H1 diagrams."""
    from persim import bottleneck

    def filt(diag, eps):
        arr = np.array([[b, d] for b, d in diag if (d - b) > eps])
        return arr if len(arr) > 0 else np.zeros((0, 2))

    dgm_a = filt(diag_a, eps_a)
    dgm_b = filt(diag_b, eps_b)
    return round(float(bottleneck(dgm_a, dgm_b)), 6)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correct', required=True, help='R_true .npy')
    parser.add_argument('--incorrect', required=True, help='R_flawed .npy')
    parser.add_argument('--hallucination', required=True, help='H .npy')
    parser.add_argument('-o', '--output', required=True, help='Output JSON')
    args = parser.parse_args()

    print("=== TRIPARTITE TEST ===")
    print()

    # Step 1: Individual analysis
    print("--- Individual persistence ---")
    r_true = compute_all(args.correct, 'R_true (correct)')
    r_flawed = compute_all(args.incorrect, 'R_flawed (incorrect)')
    h = compute_all(args.hallucination, 'H (hallucination)')

    for r in [r_true, r_flawed, h]:
        print(f"  {r['label']}: n={r['shape'][0]}, raw={r['beta1_raw']}, "
              f"β₁={r['beta1']}, surv@0.01={r['survival_001']} ({r['survival_rate_001']}%)")

    # Step 2: PHI
    if r_flawed['survival_rate_001'] > 0 and h['survival_rate_001'] > 0:
        phi_flawed = round(r_flawed['survival_rate_001'] / h['survival_rate_001'], 2)
    else:
        phi_flawed = None

    print(f"\n  PHI(R_flawed/H) = {phi_flawed}")

    # Step 3: Union tests
    print("\n--- Union tests ---")
    union_rt_rf = union_test(r_true, r_flawed, args.correct, args.incorrect)
    print(f"  R_true ∪ R_flawed: β₁(U)={union_rt_rf['U_beta1']} "
          f"(sum={union_rt_rf['sum']}) → {union_rt_rf['verdict']} "
          f"overlap={union_rt_rf['overlap_pct']}%")

    union_rt_h = union_test(r_true, h, args.correct, args.hallucination)
    print(f"  R_true ∪ H:        β₁(U)={union_rt_h['U_beta1']} "
          f"(sum={union_rt_h['sum']}) → {union_rt_h['verdict']} "
          f"overlap={union_rt_h['overlap_pct']}%")

    # Step 4: Bottleneck distances
    print("\n--- Bottleneck distances ---")
    bd_rt_rf = bottleneck_dist(r_true['h1_diagram'], r_flawed['h1_diagram'],
                                r_true['eps_pred'], r_flawed['eps_pred'])
    print(f"  R_true ↔ R_flawed: {bd_rt_rf:.4f}")

    bd_rt_h = bottleneck_dist(r_true['h1_diagram'], h['h1_diagram'],
                               r_true['eps_pred'], h['eps_pred'])
    print(f"  R_true ↔ H:        {bd_rt_h:.4f}")

    # Step 5: Verdict
    print("\n=== VERDICT ===")
    predictions = []

    # R_flawed should be nested within R_true
    if union_rt_rf['verdict'] == 'SHARED' and bd_rt_rf < 0.5:
        predictions.append("✅ R_flawed IS nested within R_true — same reasoning manifold, localized error")
    elif union_rt_rf['verdict'] == 'SHARED':
        predictions.append("⚠️  R_flawed shares region but barcode differs — partial structural divergence")
    else:
        predictions.append("❌ R_flawed is SEPARATE from R_true — different manifold (unexpected)")

    # H should be separate from R_true
    if union_rt_h['verdict'] == 'SEPARATE' and bd_rt_h > 0.5:
        predictions.append("✅ H is SEPARATE from R_true — hallucination is a different topological entity")
    elif union_rt_h['verdict'] == 'SEPARATE':
        predictions.append("⚠️  H is separate region but similar barcode — coordinate rotation?")
    else:
        predictions.append("❌ H overlaps with R_true — unexpected")

    # Persistence hierarchy
    if r_flawed['survival_rate_001'] < r_true['survival_rate_001']:
        predictions.append("✅ R_flawed less persistent than R_true — flawed reasoning is structurally fragile")
    else:
        predictions.append("⚠️  R_flawed persistence >= R_true — unexpected")

    for p in predictions:
        print(f"  {p}")

    # Save
    result = {
        'r_true': {k: v for k, v in r_true.items() if k != 'h1_diagram'},
        'r_flawed': {k: v for k, v in r_flawed.items() if k != 'h1_diagram'},
        'hallucination': {k: v for k, v in h.items() if k != 'h1_diagram'},
        'union_Rtrue_Rflawed': union_rt_rf,
        'union_Rtrue_H': union_rt_h,
        'bottleneck_Rtrue_Rflawed': bd_rt_rf,
        'bottleneck_Rtrue_H': bd_rt_h,
        'phi_Rflawed_H': phi_flawed,
        'predictions': predictions,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")

if __name__ == '__main__':
    main()
