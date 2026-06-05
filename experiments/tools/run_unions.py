#!/usr/bin/env python3
"""Tool 3: Run Union tests (β₁(A), β₁(B), β₁(A∪B)) with Mayer-Vietoris intersection.
Usage:
  python tools/run_unions.py --activations activations/ --pairs config/pairs.json -o results/unions/
  python tools/run_unions.py --activations activations/ --pairs config/pairs_smoke.json -o results/unions/ --workers 2
"""
import os, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

EPS_FRAC = 0.03
EPS_BOUNDARIES = [0.02, 0.05]

def load_and_betti(npy_path, eps_pred_ref=None):
    """Load .npy, compute β₁. If eps_pred_ref provided, use it; else compute from data."""
    points = np.load(npy_path).astype(np.float32)
    if np.isnan(points).any():
        return None, None, None, 'NaN'
    
    from sklearn.metrics import pairwise_distances
    eps_max = float(np.max(pairwise_distances(points)))
    
    if eps_pred_ref is not None:
        eps_pred = eps_pred_ref
    else:
        eps_pred = EPS_FRAC * eps_max
    
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms = ripser(points, maxdim=1)['dgms']
    beta1 = sum(1 for d in dgms[1] if (d[1] - d[0]) > eps_pred)
    
    return points, beta1, eps_max, eps_pred

def load_and_betti_temp(points, eps_pred_ref):
    """Same as load_and_betti but takes pre-loaded points array."""
    if np.isnan(points).any():
        return None, None, None, 'NaN'
    from sklearn.metrics import pairwise_distances
    eps_max = float(np.max(pairwise_distances(points)))
    if eps_pred_ref is not None:
        eps_pred = eps_pred_ref
    else:
        eps_pred = EPS_FRAC * eps_max
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms = ripser(points, maxdim=1)['dgms']
    beta1 = sum(1 for d in dgms[1] if (d[1] - d[0]) > eps_pred)
    return points, beta1, eps_max, eps_pred

def compute_eps_pred_ref(npy_path_a):
    """Compute reference eps_pred from point cloud A's first 200 points."""
    points = np.load(npy_path_a).astype(np.float32)
    from sklearn.metrics import pairwise_distances
    eps_max = float(np.max(pairwise_distances(points)))
    return EPS_FRAC * eps_max

def union_verdict(ba, bb, bu):
    """Determine SEPARATE / SHARED / INDETERMINATE."""
    if ba is None or bb is None or bu is None:
        return 'ERROR'
    if bu >= 0.8 * (ba + bb):
        return 'SEPARATE'
    if bu <= 1.2 * max(ba, bb):
        return 'SHARED'
    return 'MIXED'

def run_single_union(npy_a, npy_b, eps_pred_ref, independent_eps=False):
    """Run one union test.
    independent_eps=True: each cloud uses OWN eps_pred (cross-model unions).
    independent_eps=False: all use ref_eps from A (same-model prompt-type unions)."""
    t0 = time.time()
    
    if independent_eps:
        # Each cloud independently
        points_a, ba, eps_a, _ = load_and_betti(npy_a, None)
        if points_a is None:
            return {'error': 'NaN in A'}
        points_b, bb, eps_b, _ = load_and_betti(npy_b, None)
        if points_b is None:
            return {'error': 'NaN in B'}
        points_u = np.concatenate([points_a, points_b], axis=0)
        _, bu, eps_u, _ = load_and_betti_temp(points_u, None)
        eps_str = "independent"
    else:
        if eps_pred_ref is None:
            eps_pred_ref = compute_eps_pred_ref(npy_a)
        points_a, ba, eps_a, _ = load_and_betti(npy_a, eps_pred_ref)
        if points_a is None:
            return {'error': 'NaN in A'}
        points_b, bb, eps_b, _ = load_and_betti(npy_b, eps_pred_ref)
        if points_b is None:
            return {'error': 'NaN in B'}
        points_u = np.concatenate([points_a, points_b], axis=0)
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        from ripser import ripser
        dgms_u = ripser(points_u, maxdim=1)['dgms']
        bu = sum(1 for d in dgms_u[1] if (d[1] - d[0]) > eps_pred_ref)
        eps_str = str(round(eps_pred_ref, 4))
    
    # Mayer-Vietoris intersection
    intersection = ba + bb - bu
    overlap_pct = round(intersection / max(ba, bb) * 100, 1) if max(ba, bb) > 0 else 0
    
    verdict = union_verdict(ba, bb, bu)
    dt = time.time() - t0
    
    result = {
        'A_beta1': ba, 'B_beta1': bb, 'U_beta1': bu,
        'sum': ba + bb, 'intersection': intersection,
        'overlap_pct': overlap_pct, 'verdict': verdict,
        'eps_pred': eps_str,
        'n_A': points_a.shape[0], 'n_B': points_b.shape[0], 'n_U': points_u.shape[0],
        'time_s': round(dt, 1),
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations', required=True, help='directory with .npy files')
    parser.add_argument('--pairs', required=True, help='pairs.json config')
    parser.add_argument('-o', '--output', required=True, help='output directory for union JSON files')
    parser.add_argument('--workers', type=int, default=3, help='parallel workers')
    args = parser.parse_args()
    
    with open(args.pairs) as f:
        pairs_config = json.load(f)
    
    act_dir = Path(args.activations)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build model name → .npy file mapping
    npy_files = list(act_dir.glob('*.npy'))
    # Map: {(model_name, prompt_type): npy_path}
    npy_map = {}
    models_set = set()
    import re
    for npy in npy_files:
        stem = npy.stem  # e.g. "Qwen3.5-0.8B-base_L12_reasoning" or "...L12_bbh_ff"
        # Parse: split at _L{N}_ pattern (handles prompt types with underscores)
        m = re.match(r'(.+)_(L\d+)_(.+)', stem)
        if m:
            model = m.group(1)
            ptype = m.group(3)
            npy_map[(model, ptype)] = str(npy)
            models_set.add(model)
    
    models_list = sorted(models_set)
    print(f"Activations: {len(npy_files)} .npy, {len(models_list)} models")
    for m in models_list[:5]:
        pts = [k[1] for k in npy_map if k[0] == m]
        print(f"  {m}: {pts}")
    if len(models_list) > 5:
        print(f"  ... and {len(models_list)-5} more models")
    
    # For each pair config, build task list
    tasks = []  # [(pair_name, model_name, npy_a, npy_b, independent_eps)]
    
    for pair_name, pcfg in pairs_config.items():
        pair_type = pcfg.get('type', 'prompt_pair')
        if pair_type == 'model_pair':
            ptype_a = pcfg['prompt']
            ptype_b = pcfg['prompt']
        else:
            ptype_a = pcfg['prompt_a']
            ptype_b = pcfg['prompt_b']
        
        if pair_type == 'model_pair':
            # Base ∪ Instruct for same prompt type
            families = pcfg.get('families', [])
            for model in models_list:
                family = None
                for fam in families:
                    if model.startswith(fam):
                        family = fam
                        break
                if not family:
                    continue
                if 'base' in model.lower():
                    inst_name = model.replace('-base', '-Inst').replace('-Base', '-Inst')
                    if inst_name == model:  # Try alternate patterns
                        inst_name = model.replace('base', 'Inst')
                    if (inst_name, ptype_a) in npy_map and (model, ptype_a) in npy_map:
                        tasks.append((f'{pair_name}/{model}', model, npy_map[(model, ptype_a)], npy_map[(inst_name, ptype_a)], True))
        
        else:
            # Prompt pair: same model, different prompt types
            target_models = pcfg.get('models', 'all')
            if target_models == 'all':
                target_models = models_list
            
            for model in target_models:
                if (model, ptype_a) in npy_map and (model, ptype_b) in npy_map:
                    tasks.append((f'{pair_name}/{model}', model, npy_map[(model, ptype_a)], npy_map[(model, ptype_b)], False))
    
    print(f"\nUnion tasks: {len(tasks)}")
    for pair_name, _, npy_a, npy_b, _ in tasks[:10]:
        print(f"  {pair_name}: {Path(npy_a).name} ∪ {Path(npy_b).name}")
    if len(tasks) > 10:
        print(f"  ... and {len(tasks)-10} more")
    
    if len(tasks) == 0:
        print("No tasks found. Check that .npy files match pairs.json definitions.")
        return
    
    # Run unions in parallel
    # Group by reference eps_pred (same npy_a → same eps_pred_ref)
    # For simplicity, compute eps_pred_ref per task
    all_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for task_id, model_name, npy_a, npy_b, indep in tasks:
            eps_ref = None if indep else compute_eps_pred_ref(npy_a)
            fut = pool.submit(run_single_union, npy_a, npy_b, eps_ref, indep)
            futures[fut] = (task_id, model_name)
        
        for i, fut in enumerate(futures):
            task_id, model_name = futures[fut]
            try:
                r = fut.result()
                r['model'] = model_name
                all_results[task_id] = r
                if 'error' in r:
                    print(f"  [{i+1}/{len(tasks)}] {task_id}: ERROR - {r['error']}")
                else:
                    print(f"  [{i+1}/{len(tasks)}] {task_id}: A={r['A_beta1']} B={r['B_beta1']} U={r['U_beta1']} sum={r['sum']} → {r['verdict']}")
            except Exception as e:
                print(f"  [{i+1}/{len(tasks)}] {task_id}: CRASH - {e}")
                all_results[task_id] = {'error': str(e), 'model': model_name}
    
    # Save per-pair-type results
    for pair_name in pairs_config:
        pair_results = {k: v for k, v in all_results.items() if k.startswith(pair_name + '/')}
        if pair_results:
            # Simplify keys
            simple = {}
            for k, v in pair_results.items():
                model = k.split('/', 1)[1] if '/' in k else k
                simple[model] = {ke: va for ke, va in v.items() if ke != 'model'}
            
            out_file = out_dir / f'union_{pair_name}.json'
            with open(out_file, 'w') as f:
                json.dump(simple, f, indent=2)
            print(f"  Saved: {out_file} ({len(simple)} models)")
    
    print("=== Union tests done ===")

if __name__ == '__main__':
    main()
