#!/usr/bin/env python3
"""Center distance: R↔H and R↔C for all 23 models."""
import numpy as np
from pathlib import Path

act_dir = Path('activations_v3')
files = sorted(act_dir.glob('*_reasoning.npy'))

models = {}
for f in files:
    stem = f.stem
    parts = stem.rsplit('_L', 1)
    model = parts[0]
    layer = parts[1].split('_')[0] if '_' in parts[1] else parts[1]
    models[model] = layer

print(f'{"Model":<32s} {"R-H center":>10s} {"R-C center":>10s} {"H surv%":>10s}')
print('-'*68)

for model in sorted(models):
    L = models[model]
    r_f = act_dir / f'{model}_L{L}_reasoning.npy'
    h_f = act_dir / f'{model}_L{L}_hallucination.npy'
    c_f = act_dir / f'{model}_L{L}_control.npy'
    if not r_f.exists() or not h_f.exists():
        continue

    ra = np.load(r_f).astype(np.float32)
    ha = np.load(h_f).astype(np.float32)
    cr = ra.mean(axis=0)
    ch = ha.mean(axis=0)
    d_rh = np.linalg.norm(cr - ch)
    nr = np.linalg.norm(cr)
    nh = np.linalg.norm(ch)
    norm_rh = d_rh / ((nr + nh) / 2)

    norm_rc = 0
    if c_f.exists():
        ca = np.load(c_f).astype(np.float32)
        cc = ca.mean(axis=0)
        d_rc = np.linalg.norm(cr - cc)
        nc = np.linalg.norm(cc)
        norm_rc = d_rc / ((nr + nc) / 2)

    marker = ' *** LOW ***' if norm_rh < 0.15 else ''
    print(f'{model:<32s} {norm_rh:>10.4f} {norm_rc:>10.4f} {marker}')
