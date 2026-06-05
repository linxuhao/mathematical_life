#!/usr/bin/env python3
"""One-shot runner: Cross-family same-dimension Union tests.
Usage:
  python tools/run_cross_family.py \
    --pairs config/cross_family_pairs.json \
    -o results/unions_v3/union_CrossFamily.json
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np

# Add tools/ to path for shared functions
sys.path.insert(0, str(Path(__file__).parent))
from run_unions import load_and_betti, load_and_betti_temp, union_verdict

EPS_FRAC = 0.03

def run_one(label, npy_a, npy_b):
    """Run one cross-family union with independent eps_pred."""
    t0 = time.time()

    points_a, ba, eps_a, eps_pred_a = load_and_betti(npy_a, None)
    if points_a is None:
        return {'label': label, 'error': 'NaN in A'}

    points_b, bb, eps_b, eps_pred_b = load_and_betti(npy_b, None)
    if points_b is None:
        return {'label': label, 'error': 'NaN in B'}

    points_u = np.concatenate([points_a, points_b], axis=0)
    _, bu, eps_u, eps_pred_u = load_and_betti_temp(points_u, None)

    intersection = ba + bb - bu
    overlap_pct = round(intersection / max(ba, bb) * 100, 1) if max(ba, bb) > 0 else 0
    verdict = union_verdict(ba, bb, bu)
    dt = time.time() - t0

    return {
        'label': label,
        'A_beta1': ba, 'B_beta1': bb, 'U_beta1': bu,
        'sum': ba + bb, 'intersection': intersection,
        'overlap_pct': overlap_pct, 'verdict': verdict,
        'eps_pred_A': round(eps_pred_a, 4),
        'eps_pred_B': round(eps_pred_b, 4),
        'eps_pred_U': round(eps_pred_u, 4),
        'dim_A': points_a.shape[1], 'dim_B': points_b.shape[1],
        'n_A': points_a.shape[0], 'n_B': points_b.shape[0], 'n_U': points_u.shape[0],
        'time_s': round(dt, 1),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    with open(args.pairs) as f:
        cfg = json.load(f)

    pairs = cfg['pairs']
    results = []
    for i, pair in enumerate(pairs):
        label = pair['label']
        print(f"[{i+1}/{len(pairs)}] {label}")
        try:
            r = run_one(label, pair['a'], pair['b'])
            results.append(r)
            if 'error' in r:
                print(f"  ERROR: {r['error']}")
            else:
                print(f"  β₁(A)={r['A_beta1']} β₁(B)={r['B_beta1']} β₁(U)={r['U_beta1']} → {r['verdict']}")
                print(f"  eps: A={r['eps_pred_A']} B={r['eps_pred_B']} U={r['eps_pred_U']}")
        except Exception as e:
            print(f"  CRASH: {e}")
            results.append({'label': label, 'error': str(e)})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {output_path}")

if __name__ == '__main__':
    main()
