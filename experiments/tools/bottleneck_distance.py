#!/usr/bin/env python3
"""Tool 4: Bottleneck distance between persistence barcodes.
Computes bottleneck distance between persistence diagrams of activation point clouds.
Does NOT merge point clouds — compares barcode shapes directly (coordinate-free).

Usage:
  # Test single pair:
  python tools/bottleneck_distance.py --a activations_v3/Qwen3.5-0.8B-base_L12_reasoning.npy \
      --b activations_v3/Qwen3.5-0.8B-base_L12_control.npy

  # Batch from JSON config:
  python tools/bottleneck_distance.py --pairs config/bottleneck_pairs.json \
      -o results/bottleneck_v1.json --workers 3
"""
import os, json, argparse, time, sys
from pathlib import Path
import numpy as np

EPS_FRAC = 0.03

def compute_diagram(npy_path):
    """Run ripser and return full persistence diagram (H₁ only) + metadata."""
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

    dt = time.time() - t0

    # H₁ diagram: list of [birth, death] pairs
    h1_diag = dgms[1].tolist()
    beta1_raw = len(h1_diag)
    beta1 = sum(1 for d in dgms[1] if (d[1] - d[0]) > eps_pred)

    return {
        'name': Path(npy_path).stem,
        'shape': list(points.shape),
        'eps_max': round(eps_max, 4),
        'eps_pred': round(eps_pred, 4),
        'beta1_raw': beta1_raw,
        'beta1': beta1,
        'h1_diagram': h1_diag,
        'ripser_time_s': round(dt, 1),
    }

def bottleneck_distance(diag_a, diag_b, eps_pred_a=None, eps_pred_b=None):
    """Compute bottleneck distance between two H₁ persistence diagrams.
    Uses filtered diagrams (only features with lifetime > eps_pred) if eps_preds given.
    Returns (distance, filtered_diag_a, filtered_diag_b).
    """
    from persim import bottleneck

    def filter_diag(diag, eps_pred):
        if eps_pred is None:
            return np.array(diag) if len(diag) > 0 else np.zeros((0, 2))
        filtered = np.array([[b, d] for b, d in diag if (d - b) > eps_pred])
        return filtered if len(filtered) > 0 else np.zeros((0, 2))

    dgm_a = filter_diag(diag_a, eps_pred_a)
    dgm_b = filter_diag(diag_b, eps_pred_b)

    # persim.bottleneck handles empty diagrams
    dist = bottleneck(dgm_a, dgm_b)
    return round(float(dist), 6)

def compare_pair(path_a, path_b, label_a=None, label_b=None):
    """Compare two .npy files via bottleneck distance. Returns dict."""
    r_a = compute_diagram(path_a)
    r_b = compute_diagram(path_b)

    # Unfiltered bottleneck
    dist_raw = bottleneck_distance(r_a['h1_diagram'], r_b['h1_diagram'])

    # Filtered bottleneck (ε=0.03 × each cloud's own eps_max)
    dist_filtered = bottleneck_distance(
        r_a['h1_diagram'], r_b['h1_diagram'],
        eps_pred_a=r_a['eps_pred'], eps_pred_b=r_b['eps_pred']
    )

    # Normalized: divide by max(eps_pred_a, eps_pred_b) for scale-invariant comparison
    max_eps = max(r_a['eps_pred'], r_b['eps_pred'])
    dist_normalized = round(dist_filtered / max_eps, 4) if max_eps > 0 else dist_filtered

    return {
        'a': label_a or r_a['name'],
        'b': label_b or r_b['name'],
        'a_beta1': r_a['beta1'],
        'b_beta1': r_b['beta1'],
        'a_eps_pred': r_a['eps_pred'],
        'b_eps_pred': r_b['eps_pred'],
        'bottleneck_raw': dist_raw,
        'bottleneck_filtered': dist_filtered,
        'bottleneck_normalized': dist_normalized,
        'time_s': round(r_a['ripser_time_s'] + r_b['ripser_time_s'], 1),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', help='Path to first .npy file (single-pair mode)')
    parser.add_argument('--b', help='Path to second .npy file (single-pair mode)')
    parser.add_argument('--pairs', help='JSON config file with list of {a, b, label_a, label_b}')
    parser.add_argument('-o', '--output', help='Output JSON file (batch mode)')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    args = parser.parse_args()

    if args.a and args.b:
        # Single pair mode
        result = compare_pair(args.a, args.b)
        print(json.dumps(result, indent=2))

    elif args.pairs:
        # Batch mode
        with open(args.pairs) as f:
            pairs = json.load(f)

        results = []
        for i, pair in enumerate(pairs):
            print(f"[{i+1}/{len(pairs)}] {pair.get('label_a', pair['a'])} vs {pair.get('label_b', pair['b'])}")
            try:
                r = compare_pair(pair['a'], pair['b'],
                                label_a=pair.get('label_a'),
                                label_b=pair.get('label_b'))
                results.append(r)
                print(f"  β₁: {r['a_beta1']} vs {r['b_beta1']} | "
                      f"bottleneck(raw)={r['bottleneck_raw']:.4f} "
                      f"bottleneck(filtered)={r['bottleneck_filtered']:.4f} "
                      f"normalized={r['bottleneck_normalized']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({'a': pair.get('label_a', pair['a']),
                               'b': pair.get('label_b', pair['b']),
                               'error': str(e)})

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved {len(results)} results → {output_path}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
