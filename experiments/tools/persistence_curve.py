#!/usr/bin/env python3
"""Tool 5: Persistence curve — β₁ survival across ε thresholds.
Shows how many H₁ features survive at each filter level, revealing
feature lifetime distribution (ephemeral vs persistent loops).

Usage:
  python tools/persistence_curve.py \
      --files activations_v3/Qwen3.5-9B-Inst_L20_reasoning.npy \
               activations_v3/Qwen3.5-9B-Inst_L20_hallucination.npy \
      --labels "reasoning (β₁=1)" "hallucination (β₁=17)" \
      -o results/curves_v1.json

  # Batch from JSON config:
  python tools/persistence_curve.py \
      --config config/persistence_curve_pairs.json \
      -o results/curves_v1.json --workers 3
"""
import os, json, argparse, time
from pathlib import Path
import numpy as np

N_EPS_STEPS = 50  # number of ε thresholds from 0 to ε_max

def compute_curve(npy_path, label=None):
    """Compute persistence curve for a single .npy file.
    Returns survival counts at N_EPS_STEPS thresholds from 0 to eps_max."""
    t0 = time.time()
    points = np.load(npy_path).astype(np.float32)
    if np.isnan(points).any():
        raise ValueError(f"NaN in {npy_path}")

    from sklearn.metrics import pairwise_distances
    eps_max = float(np.max(pairwise_distances(points)))

    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms = ripser(points, maxdim=1)['dgms']

    # H₁ lifetimes: death - birth for each feature
    h1 = dgms[1]
    lifetimes = h1[:, 1] - h1[:, 0]
    lifetimes = lifetimes[lifetimes > 0]  # filter infinites

    beta1_raw = len(lifetimes)

    # Survival at each ε level
    eps_levels = np.linspace(0, eps_max, N_EPS_STEPS)
    survival = [int(np.sum(lifetimes > eps)) for eps in eps_levels]

    dt = time.time() - t0

    return {
        'name': label or Path(npy_path).stem,
        'file': str(npy_path),
        'shape': list(points.shape),
        'eps_max': round(eps_max, 4),
        'beta1_raw': beta1_raw,
        'eps_levels': [round(e, 4) for e in eps_levels.tolist()],
        'survival': survival,
        'time_s': round(dt, 1),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='*', help='List of .npy files (quick mode)')
    parser.add_argument('--labels', nargs='*', help='Labels for each file')
    parser.add_argument('--config', help='JSON config: [{"file": "...", "label": "..."}, ...]')
    parser.add_argument('-o', '--output', help='Output JSON')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    tasks = []
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        tasks = [(t['file'], t.get('label', Path(t['file']).stem)) for t in cfg]
    elif args.files:
        labels = args.labels or [None] * len(args.files)
        tasks = list(zip(args.files, labels))

    if not tasks:
        parser.print_help()
        return

    results = []
    for i, (npy, label) in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {label or Path(npy).stem}")
        try:
            r = compute_curve(npy, label)
            results.append(r)
            surv_003 = r['survival'][max(0, int(N_EPS_STEPS * 0.03 / (r['eps_max'] or 1)))] if r['eps_max'] > 0 else 0
            print(f"  raw={r['beta1_raw']} β₁(ε=0.03)≈{surv_003} eps_max={r['eps_max']:.2f} time={r['time_s']:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'name': label or str(npy), 'error': str(e)})

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {output_path}")

if __name__ == '__main__':
    main()
