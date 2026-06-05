#!/usr/bin/env python3
"""Tool 2: Compute β₁ for all .npy activation files.
Usage:
  python tools/compute_betti.py -i activations/ -o results/betti_results.json
  python tools/compute_betti.py -i activations/ -o results/betti_results.json --workers 3 --pattern "*_reasoning.npy"
"""
import os, json, argparse, time, gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

EPS_FRAC = 0.03  # frozen in master_protocol.md v3

def compute_one(npy_path):
    """Compute β₁ for a single .npy file. Returns dict."""
    fname = Path(npy_path).stem  # e.g. "Qwen3.5-0.8B-base_L12_reasoning"
    t0 = time.time()
    
    points = np.load(npy_path).astype(np.float32)
    if points.shape[0] < 10:
        return {'file': str(npy_path), 'error': f'too few points: {points.shape[0]}'}
    
    # Check for NaN
    if np.isnan(points).any():
        return {'file': str(npy_path), 'error': 'contains NaN'}
    
    # ε_max from ALL points (n=1319, pairwise ~870K, ~0.1s)
    from sklearn.metrics import pairwise_distances
    eps_max = float(np.max(pairwise_distances(points)))
    eps_pred = EPS_FRAC * eps_max
    
    # ripser
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms = ripser(points, maxdim=1)['dgms']
    
    # Filter H1 features
    beta1_raw = len(dgms[1])
    beta1 = sum(1 for d in dgms[1] if (d[1] - d[0]) > eps_pred)
    beta0 = len(dgms[0])
    
    dt = time.time() - t0
    
    return {
        'file': str(npy_path),
        'name': fname,
        'shape': list(points.shape),
        'beta0': beta0,
        'beta1_raw': beta1_raw,
        'beta1': beta1,
        'eps_max': round(eps_max, 4),
        'eps_pred': round(eps_pred, 4),
        'ripser_time_s': round(dt, 1),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='directory with .npy files')
    parser.add_argument('-o', '--output', required=True, help='output JSON file')
    parser.add_argument('--workers', type=int, default=3, help='parallel workers')
    parser.add_argument('--pattern', default='*.npy', help='glob pattern for filtering')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    npy_files = sorted(input_dir.glob(args.pattern))
    print(f"Found {len(npy_files)} .npy files in {input_dir}")
    
    if len(npy_files) == 0:
        print("No files found, exiting.")
        return
    
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compute_one, str(f)): f.name for f in npy_files}
        for i, future in enumerate(futures):
            fname = futures[future]
            try:
                r = future.result()
                results[r['name']] = r
                if 'error' in r:
                    print(f"  [{i+1}/{len(npy_files)}] {fname}: ERROR - {r['error']}")
                else:
                    print(f"  [{i+1}/{len(npy_files)}] {fname}: β₁={r['beta1']} (raw={r['beta1_raw']}) eps={r['eps_pred']:.4f} time={r['ripser_time_s']:.1f}s")
            except Exception as e:
                print(f"  [{i+1}/{len(npy_files)}] {fname}: CRASH - {e}")
                results[fname] = {'file': str(input_dir / fname), 'error': str(e)}
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    valid = {k: v for k, v in results.items() if 'beta1' in v}
    errors = {k: v for k, v in results.items() if 'error' in v}
    print(f"\nDone: {len(valid)} valid, {len(errors)} errors → {output_path}")

if __name__ == '__main__':
    main()
