#!/usr/bin/env python3
"""Wasserstein-1 version of the family convergence scan.

Fixes the two failure modes of bottleneck_family_scan.py:
  - saturation (L-inf ignores all but the worst feature) -> W1 sums all costs
  - scale incomparability -> each diagram is divided by its cloud's eps_max
    BEFORE matching, making diagrams dimensionless.
Reports total W1 and W1 per feature (W1 / mean feature count of the pair).
Same pairs + noise floors as the bottleneck scan.
Output: results/wasserstein_family_scan.json
"""
import json, time, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACT = ROOT / 'activations_v3'

def diagram_from_points(points):
    from sklearn.metrics import pairwise_distances
    from ripser import ripser
    eps_max = float(np.max(pairwise_distances(points)))
    dgm = ripser(points, maxdim=1)['dgms'][1]
    return {'dgm_norm': (dgm / eps_max), 'eps_max': eps_max, 'n': len(dgm)}

_cache = {}
def diagram(name):
    if name not in _cache:
        t0 = time.time()
        pts = np.load(ACT / f'{name}.npy').astype(np.float32)
        _cache[name] = diagram_from_points(pts)
        print(f'  diagram {name}: n={_cache[name]["n"]} ({time.time()-t0:.1f}s)',
              flush=True)
    return _cache[name]

def w1_pair(da, db):
    from persim import wasserstein
    t0 = time.time()
    w1 = float(wasserstein(da['dgm_norm'], db['dgm_norm']))
    per_feat = w1 / ((da['n'] + db['n']) / 2)
    return {'w1': round(w1, 4), 'w1_per_feat': round(per_feat, 6),
            'n_a': da['n'], 'n_b': db['n'],
            'match_time_s': round(time.time() - t0, 1)}

def find(model, mode='reasoning'):
    hits = glob.glob(str(ACT / f'{model}_L*_{mode}.npy'))
    return Path(hits[0]).stem if hits else None

FAMILIES = {
    'Qwen3.5-base': [('Qwen3.5-0.8B-base', 0.8), ('Qwen3.5-2B-base', 2),
                     ('Qwen3.5-4B-base', 4), ('Qwen3.5-9B-base', 9)],
    'Qwen3.5-Inst': [('Qwen3.5-0.8B-Inst', 0.8), ('Qwen3.5-2B-Inst', 2),
                     ('Qwen3.5-4B-Inst', 4), ('Qwen3.5-9B-Inst', 9)],
    'Qwen3-base': [('Qwen3-0.6B-base', 0.6), ('Qwen3-1.7B-base', 1.7),
                   ('Qwen3-4B-base', 4), ('Qwen3-8B-base', 8)],
    'Qwen3-Inst': [('Qwen3-0.6B-Inst', 0.6), ('Qwen3-1.7B-Inst', 1.7),
                   ('Qwen3-4B-Inst', 4), ('Qwen3-8B-Inst', 8)],
    'Qwen2.5-base': [('Qwen2.5-0.5B-base', 0.5), ('Qwen2.5-3B-base', 3)],
    'Qwen2.5-Inst': [('Qwen2.5-0.5B-Inst', 0.5), ('Qwen2.5-3B-Inst', 3)],
}

CROSS_FAMILY = [
    ('Qwen3-1.7B-base', 'SmolLM2-1.7B-base'),
    ('Qwen3-1.7B-Inst', 'SmolLM2-1.7B-Inst'),
    ('Qwen3.5-2B-Inst', 'SmolLM2-1.7B-Inst'),
    ('Qwen3.5-2B-Inst', 'Qwen3-1.7B-Inst'),
    ('Qwen3.5-9B-Inst', 'Qwen3-8B-Inst'),
]

NOISE_FLOOR_CLOUDS = ['Qwen3.5-0.8B-Inst', 'Qwen3.5-2B-Inst', 'Qwen3.5-9B-base']

def main():
    out = {'within_family': [], 'cross_family': [], 'noise_floor': []}

    for fam, members in FAMILIES.items():
        names = [(find(m), m, s) for m, s in members]
        names = [(n, m, s) for n, m, s in names if n]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                na, ma, sa = names[i]
                nb, mb, sb = names[j]
                r = w1_pair(diagram(na), diagram(nb))
                r.update({'family': fam, 'a': ma, 'b': mb,
                          'size_a': sa, 'size_b': sb,
                          'pair_scale': round((sa * sb) ** 0.5, 2)})
                out['within_family'].append(r)
                print(f'{fam}: {ma} vs {mb}  W1={r["w1"]} '
                      f'per_feat={r["w1_per_feat"]}', flush=True)

    for ma, mb in CROSS_FAMILY:
        na, nb = find(ma), find(mb)
        if not (na and nb):
            print(f'skip cross {ma} vs {mb} (missing file)', flush=True)
            continue
        r = w1_pair(diagram(na), diagram(nb))
        r.update({'a': ma, 'b': mb})
        out['cross_family'].append(r)
        print(f'CROSS: {ma} vs {mb}  W1={r["w1"]} '
              f'per_feat={r["w1_per_feat"]}', flush=True)

    rng = np.random.default_rng(0)
    for m in NOISE_FLOOR_CLOUDS:
        n = find(m)
        if not n:
            continue
        pts = np.load(ACT / f'{n}.npy').astype(np.float32)
        k = int(0.9 * len(pts))
        d1 = diagram_from_points(pts[rng.choice(len(pts), k, replace=False)])
        d2 = diagram_from_points(pts[rng.choice(len(pts), k, replace=False)])
        r = w1_pair(d1, d2)
        r.update({'cloud': m, 'note': '90% subsample x2, same cloud'})
        out['noise_floor'].append(r)
        print(f'NOISE FLOOR {m}: W1={r["w1"]} per_feat={r["w1_per_feat"]}',
              flush=True)

    dest = ROOT / 'results' / 'wasserstein_family_scan.json'
    with open(dest, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'saved -> {dest}', flush=True)

if __name__ == '__main__':
    main()
