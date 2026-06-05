#!/usr/bin/env python3
"""Analyze label_and_extract.py output: per-trajectory β₁ distribution
AND token-position β₁ curves for correct vs incorrect.

Usage:
  python tools/analyze_tokenwise.py \
      --npz activations_v3/Qwen__Qwen3.5-2B_L12_tokenwise.npz \
      --labels activations_v3/Qwen__Qwen3.5-2B_L12_tokenwise_labels.json \
      -o results/tokenwise_v1.json
"""
import json, argparse, time, os
from pathlib import Path
from collections import defaultdict
import numpy as np

EPS_FRAC = 0.03

def trajectory_betti(acts):
    """Compute β₁ for a single trajectory (n_tokens, hidden_dim)."""
    if acts.shape[0] < 10:
        return None
    from sklearn.metrics import pairwise_distances
    eps_max = float(np.max(pairwise_distances(acts)))
    eps_pred = EPS_FRAC * eps_max

    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms = ripser(acts, maxdim=1)['dgms']
    h1 = dgms[1]
    lt = h1[:, 1] - h1[:, 0]
    lt = lt[lt > 0]

    return {
        'beta1_raw': len(lt),
        'beta1': int(np.sum(lt > eps_pred)),
        'surv_001': int(np.sum(lt > 0.01 * eps_max)),
        'eps_max': round(eps_max, 4),
        'eps_pred': round(eps_pred, 4),
        'max_lifetime': round(float(lt.max()), 4) if len(lt) > 0 else 0,
        'n_tokens': acts.shape[0],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--max-per-group', type=int, default=200,
                       help='Max trajectories per group for speed (0=all)')
    args = parser.parse_args()

    print("=== TOKENWISE ANALYSIS ===")
    print()

    # Load data
    print("Loading data...")
    data = np.load(args.npz)
    acts_flat = data['acts']
    lens = data['lens']
    with open(args.labels) as f:
        labels_data = json.load(f)

    correct_idx = labels_data['correct_idx']
    incorrect_idx = labels_data['incorrect_idx']
    n_total = len(lens)
    print(f"Total: {n_total} prompts, Correct: {len(correct_idx)}, Incorrect: {len(incorrect_idx)}")

    # Reconstruct per-prompt activations
    offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
    hidden_dim = acts_flat.shape[1]
    print(f"Total tokens: {acts_flat.shape[0]}, Hidden dim: {hidden_dim}")

    # --- DIMENSION 1: Per-trajectory β₁ distribution ---
    print("\n--- Dimension 1: Trajectory β₁ distribution ---")

    def compute_group(indices, label, max_n):
        idxs = indices[:max_n] if max_n > 0 else indices
        results = []
        t0 = time.time()
        for k, idx in enumerate(idxs):
            acts = acts_flat[offsets[idx]:offsets[idx]+lens[idx]]
            r = trajectory_betti(acts)
            if r:
                r['prompt_idx'] = int(idx)
                results.append(r)
            if (k+1) % 100 == 0:
                print(f"  {label} [{k+1}/{len(idxs)}] ({time.time()-t0:.0f}s)")
        return results

    max_n = args.max_per_group or None
    correct_traj = compute_group(correct_idx, 'correct', max_n)
    incorrect_traj = compute_group(incorrect_idx, 'incorrect', max_n)

    # Summary stats
    for label, trajs in [('correct', correct_traj), ('incorrect', incorrect_traj)]:
        if not trajs: continue
        b1s = [t['beta1'] for t in trajs]
        raws = [t['beta1_raw'] for t in trajs]
        survs = [t['surv_001'] for t in trajs]
        tokens = [t['n_tokens'] for t in trajs]
        print(f"\n  {label} ({len(trajs)} trajectories):")
        print(f"    β₁: median={np.median(b1s):.0f} mean={np.mean(b1s):.1f} "
              f"std={np.std(b1s):.0f} min={np.min(b1s)} max={np.max(b1s)}")
        print(f"    β₁_raw: median={np.median(raws):.0f} mean={np.mean(raws):.1f}")
        print(f"    surv@0.01: median={np.median(survs):.0f} mean={np.mean(survs):.1f}")
        print(f"    tokens: median={np.median(tokens):.0f} mean={np.mean(tokens):.1f}")

    # --- DIMENSION 2: Cumulative trajectory β₁ (β₁ over prefix) ---
    print("\n--- Dimension 2: Cumulative trajectory β₁ (incremental) ---")
    print("  Computes β₁(0→t) for prefixes of the trajectory.")
    print("  Shows when the 'Aha moment' loop closes during reasoning.")

    def cumulative_betti(acts, step=8):
        """Compute β₁ for prefixes of the trajectory every `step` tokens."""
        results = []
        for t in range(step, acts.shape[0] + 1, step):
            prefix = acts[:t]
            if prefix.shape[0] < 10:
                continue
            r = trajectory_betti(prefix)
            if r:
                r['prefix_len'] = t
                results.append(r)
        return results

    # Pick a representative correct and incorrect trajectory
    if correct_traj and incorrect_traj:
        best_correct = max(correct_traj, key=lambda t: t['beta1'])
        best_incorrect = max(incorrect_traj, key=lambda t: t['beta1'])
        print(f"  Best correct: idx={best_correct['prompt_idx']} β₁={best_correct['beta1']}")
        print(f"  Best incorrect: idx={best_incorrect['prompt_idx']} β₁={best_incorrect['beta1']}")

        acts_c = acts_flat[offsets[best_correct['prompt_idx']]:offsets[best_correct['prompt_idx']]+lens[best_correct['prompt_idx']]]
        acts_i = acts_flat[offsets[best_incorrect['prompt_idx']]:offsets[best_incorrect['prompt_idx']]+lens[best_incorrect['prompt_idx']]]

        cum_c = cumulative_betti(acts_c)
        cum_i = cumulative_betti(acts_i)

        print(f"\n  Correct cumulative β₁ (every 8 tokens):")
        for r in cum_c:
            print(f"    prefix={r['prefix_len']:>4} β₁={r['beta1']:>4} raw={r['beta1_raw']:>4} surv={r['surv_001']:>4}")

        print(f"\n  Incorrect cumulative β₁ (every 8 tokens):")
        for r in cum_i:
            print(f"    prefix={r['prefix_len']:>4} β₁={r['beta1']:>4} raw={r['beta1_raw']:>4} surv={r['surv_001']:>4}")

    # --- DIMENSION 3: Parallel universes (NOT YET IMPLEMENTED) ---
    print("\n--- Dimension 3: Parallel universes (NOT YET DONE) ---")
    print("  To find the topological bifurcation point:")
    print("  1. Take 1 hard GSM8K problem")
    print("  2. Generate ~200 answers with temperature=0.7 (some correct, some wrong)")
    print("  3. At each token position, β₁(all 200 trajectories at pos t)")
    print("  4. Find where β₁ spikes = where paths diverge geometrically")
    print("  This avoids the 'semantic noise blob' trap of mixing unrelated prompts.")

    # --- Save ---
    result = {
        'model': labels_data['model'],
        'layer': labels_data['layer'],
        'n_total': n_total,
        'n_correct': len(correct_idx),
        'n_incorrect': len(incorrect_idx),
        'trajectory_correct': [{k: v for k, v in t.items()} for t in correct_traj],
        'trajectory_incorrect': [{k: v for k, v in t.items()} for t in incorrect_traj],
        'cumulative_correct': cum_c if 'cum_c' in dir() else [],
        'cumulative_incorrect': cum_i if 'cum_i' in dir() else [],
        'dimension_3_note': 'Parallel universes analysis requires separate extraction with temperature=0.7 on a single prompt.',
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.output}")

if __name__ == '__main__':
    main()
