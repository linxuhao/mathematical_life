#!/usr/bin/env python3
"""Within-family cross-size bottleneck scan (reasoning mode).

Question: do same-family models' reasoning-cloud H1 shapes converge,
and does the (normalized) shape distance shrink as scale grows?

Reuses the conventions of tools/bottleneck_distance.py (EPS_FRAC=0.03,
filtered + normalized-by-eps_pred). Adds:
  - diagram caching (each cloud computed once)
  - within-cloud noise floor: 90% subsample (no replacement) x2 -> bottleneck
Output: results/bottleneck_family_scan.json
"""
import json, time, glob, re
from pathlib import Path
import numpy as np

EPS_FRAC = 0.03
ROOT = Path(__file__).resolve().parent.parent
ACT = ROOT / 'activations_v3'

def diagram_from_points(points):
    from sklearn.metrics import pairwise_distances
    from ripser import ripser
    eps_max = float(np.max(pairwise_distances(points)))
    eps_pred = EPS_FRAC * eps_max
    dgm = ripser(points, maxdim=1)['dgms'][1]
    beta1 = int(sum(1 for d in dgm if (d[1] - d[0]) > eps_pred))
    return {'dgm': dgm, 'eps_max': eps_max, 'eps_pred': eps_pred,
            'beta1_raw': len(dgm), 'beta1': beta1}

_cache = {}
def diagram(name):
    if name not in _cache:
        t0 = time.time()
        pts = np.load(ACT / f'{name}.npy').astype(np.float32)
        _cache[name] = diagram_from_points(pts)
        print(f'  diagram {name}: beta1={_cache[name]["beta1"]} '
              f'({time.time()-t0:.1f}s)', flush=True)
    return _cache[name]

def filt(dgm, eps_pred):
    f = np.array([[b, d] for b, d in dgm if (d - b) > eps_pred])
    return f if len(f) else np.zeros((0, 2))

def bn_pair(da, db):
    from persim import bottleneck
    raw = float(bottleneck(np.array(da['dgm']), np.array(db['dgm'])))
    fil = float(bottleneck(filt(da['dgm'], da['eps_pred']),
                           filt(db['dgm'], db['eps_pred'])))
    norm = max(da['eps_pred'], db['eps_pred'])
    return {'raw': round(raw, 4), 'filtered': round(fil, 4),
            'raw_norm': round(raw / norm, 4), 'filt_norm': round(fil / norm, 4)}

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

CROSS_FAMILY = [  # matched-ish size, different family
    ('Qwen3-1.7B-base', 'SmolLM2-1.7B-base'),
    ('Qwen3-1.7B-Inst', 'SmolLM2-1.7B-Inst'),
    ('Qwen3.5-2B-Inst', 'SmolLM2-1.7B-Inst'),
    ('Qwen3.5-2B-Inst', 'Qwen3-1.7B-Inst'),   # cross-generation, same lineage
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
                r = bn_pair(diagram(na), diagram(nb))
                r.update({'family': fam, 'a': ma, 'b': mb,
                          'size_a': sa, 'size_b': sb,
                          'pair_scale': round((sa * sb) ** 0.5, 2)})
                out['within_family'].append(r)
                print(f'{fam}: {ma} vs {mb}  raw_norm={r["raw_norm"]} '
                      f'filt_norm={r["filt_norm"]}', flush=True)

    for ma, mb in CROSS_FAMILY:
        na, nb = find(ma), find(mb)
        if not (na and nb):
            print(f'skip cross {ma} vs {mb} (missing file)', flush=True)
            continue
        r = bn_pair(diagram(na), diagram(nb))
        r.update({'a': ma, 'b': mb})
        out['cross_family'].append(r)
        print(f'CROSS: {ma} vs {mb}  raw_norm={r["raw_norm"]} '
              f'filt_norm={r["filt_norm"]}', flush=True)

    rng = np.random.default_rng(0)
    for m in NOISE_FLOOR_CLOUDS:
        n = find(m)
        if not n:
            continue
        pts = np.load(ACT / f'{n}.npy').astype(np.float32)
        k = int(0.9 * len(pts))
        d1 = diagram_from_points(pts[rng.choice(len(pts), k, replace=False)])
        d2 = diagram_from_points(pts[rng.choice(len(pts), k, replace=False)])
        r = bn_pair(d1, d2)
        r.update({'cloud': m, 'note': '90% subsample x2, same cloud'})
        out['noise_floor'].append(r)
        print(f'NOISE FLOOR {m}: raw_norm={r["raw_norm"]} '
              f'filt_norm={r["filt_norm"]}', flush=True)

    dest = ROOT / 'results' / 'bottleneck_family_scan.json'
    with open(dest, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'saved -> {dest}', flush=True)

if __name__ == '__main__':
    main()
