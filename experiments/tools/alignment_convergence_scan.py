#!/usr/bin/env python3
"""Does alignment pull DIFFERENT families/models toward a common shape?

Quadruple test: for every cross-group model pair (i, j), compare
  d_base = W1(base_i, base_j)   vs   d_inst = W1(inst_i, inst_j).
Alignment-homogenization predicts d_inst < d_base systematically.
Sign test over all quadruples. Diagrams normalized by own eps_max
(same convention as wasserstein_family_scan.py).
Output: results/alignment_convergence_scan.json
"""
import json, time, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACT = ROOT / 'activations_v3'

GROUPS = {
    'Qwen2.5': ['Qwen2.5-0.5B', 'Qwen2.5-3B'],
    'Qwen3': ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B'],
    'Qwen3.5': ['Qwen3.5-0.8B', 'Qwen3.5-2B', 'Qwen3.5-4B', 'Qwen3.5-9B'],
    'SmolLM2': ['SmolLM2-1.7B'],
}

def find(model, arm):
    hits = glob.glob(str(ACT / f'{model}-{arm}_L*_reasoning.npy'))
    return Path(hits[0]).stem if hits else None

def diagram_from_points(points):
    from sklearn.metrics import pairwise_distances
    from ripser import ripser
    eps_max = float(np.max(pairwise_distances(points)))
    dgm = ripser(points, maxdim=1)['dgms'][1]
    return {'dgm_norm': dgm / eps_max, 'n': len(dgm)}

_cache = {}
def diagram(name):
    if name not in _cache:
        t0 = time.time()
        pts = np.load(ACT / f'{name}.npy').astype(np.float32)
        _cache[name] = diagram_from_points(pts)
        print(f'  diagram {name}: n={_cache[name]["n"]} ({time.time()-t0:.1f}s)',
              flush=True)
    return _cache[name]

def w1(name_a, name_b):
    from persim import wasserstein
    da, db = diagram(name_a), diagram(name_b)
    return float(wasserstein(da['dgm_norm'], db['dgm_norm'])), da['n'], db['n']

def main():
    models = []  # (group, model, base_stem, inst_stem)
    for g, ms in GROUPS.items():
        for m in ms:
            b, i = find(m, 'base'), find(m, 'Inst')
            if b and i:
                models.append((g, m, b, i))
            else:
                print(f'skip {m} (base={bool(b)}, inst={bool(i)})', flush=True)

    quads = []
    for x in range(len(models)):
        for y in range(x + 1, len(models)):
            gx, mx, bx, ix = models[x]
            gy, my, by, iy = models[y]
            if gx == gy:
                continue  # cross-group only
            d_base, nb1, nb2 = w1(bx, by)
            d_inst, ni1, ni2 = w1(ix, iy)
            q = {'a': mx, 'b': my, 'group_a': gx, 'group_b': gy,
                 'd_base': round(d_base, 4), 'd_inst': round(d_inst, 4),
                 'delta': round(d_inst - d_base, 4),
                 'inst_closer': bool(d_inst < d_base),
                 'n_base': [nb1, nb2], 'n_inst': [ni1, ni2]}
            quads.append(q)
            print(f'{mx} x {my}: base={d_base:.2f} inst={d_inst:.2f} '
                  f'{"CONVERGE" if q["inst_closer"] else "diverge"}', flush=True)

    k = sum(q['inst_closer'] for q in quads)
    n = len(quads)
    from scipy.stats import binomtest
    p = binomtest(k, n, 0.5).pvalue
    deltas = [q['delta'] for q in quads]
    summary = {'n_quadruples': n, 'inst_closer': k,
               'sign_test_p': round(float(p), 4),
               'median_delta': round(float(np.median(deltas)), 4),
               'mean_delta': round(float(np.mean(deltas)), 4),
               'note': 'delta = d_inst - d_base; negative = alignment converges shapes'}
    print(f'\nSUMMARY: {k}/{n} quadruples inst-closer, sign-test p={p:.4f}, '
          f'median delta={summary["median_delta"]}', flush=True)

    dest = ROOT / 'results' / 'alignment_convergence_scan.json'
    with open(dest, 'w') as f:
        json.dump({'summary': summary, 'quadruples': quads}, f, indent=1)
    print(f'saved -> {dest}', flush=True)

if __name__ == '__main__':
    main()
