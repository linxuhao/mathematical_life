#!/usr/bin/env python3
"""Cumulative β₁(k) "aha-moment" analysis on tokenwise_v2 trajectories.

Tests a DIFFERENT hypothesis than whole-traj β₁ magnitude: does β₁ accumulate
with a localized JUMP that is present in CORRECT trajectories and absent in
INCORRECT ones? We compute β₁ on growing prefixes (stride 16 tokens), then:
  (1) align by % of trajectory length, resample to a common 20-point grid,
      report mean curve for correct vs incorrect (raw + self-normalized to final).
  (2) per-trajectory locate the largest single-step jump (discrete derivative);
      compare the JUMP MAGNITUDE and its POSITION (%) between classes.
Pre-committed read: a real aha-moment ⇒ correct curves show a localized jump
(distribution of max-jump position concentrated) materially larger than incorrect.

Resumable: caches per-traj curves to a json; CPU only.
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/linxuhao/papers/mathematical-life/actopo/src")
from actopo import FROZEN_V5, measure

BASE = Path("/home/linxuhao/papers/mathematical-life/experiments")
TDIR = BASE / "activations_v3/tokenwise_v2"
OUT  = BASE / "results/aha_curves.json"
STRIDE = 16
MIN_TOK = 32

labels = {r["idx"]: r for r in json.load(open(TDIR / "labels.json"))}
cache = json.load(open(OUT)) if OUT.exists() else {}

t0 = time.time(); done = 0
for idx in sorted(labels):
    key = str(idx)
    if key in cache:
        continue
    lab = labels[idx]
    if lab.get("capped", False):
        continue
    f = TDIR / f"traj_{idx}.npy"
    if not f.exists():
        continue
    a = np.load(f).astype("float32")
    n = a.shape[0]
    if n < MIN_TOK:
        continue
    ks = list(range(STRIDE, n + 1, STRIDE))
    if ks[-1] != n:
        ks.append(n)
    curve = []
    for k in ks:
        m = measure(a[:k], FROZEN_V5)
        curve.append(m.beta1)
    cache[key] = {"correct": bool(lab["correct"]), "n": int(n),
                  "ks": ks, "beta1": curve}
    done += 1
    if done % 25 == 0:
        json.dump(cache, open(OUT, "w"))
        print(f"  {done} new ({len(cache)} cached) {time.time()-t0:.0f}s", flush=True)

json.dump(cache, open(OUT, "w"))
print(f"DONE computing curves: {len(cache)} trajectories ({time.time()-t0:.0f}s)", flush=True)

# ---------------- analysis ----------------
GRID = np.linspace(0.05, 1.0, 20)
def resample(ks, ys, n):
    x = np.array(ks, float) / n
    return np.interp(GRID, x, ys)

rawC, rawI, normC, normI, jumpsC, jumpsI = [], [], [], [], [], []
for v in cache.values():
    ys = np.array(v["beta1"], float)
    rs = resample(v["ks"], ys, v["n"])
    nrm = rs / rs[-1] if rs[-1] > 0 else rs
    # max discrete jump on the self-normalized resampled curve + its position
    dd = np.diff(nrm)
    jpos = GRID[1:][int(np.argmax(dd))]
    jmag = float(dd.max())
    if v["correct"]:
        rawC.append(rs); normC.append(nrm); jumpsC.append((jmag, jpos))
    else:
        rawI.append(rs); normI.append(nrm); jumpsI.append((jmag, jpos))

rawC, rawI = np.array(rawC), np.array(rawI)
normC, normI = np.array(normC), np.array(normI)
jC, jI = np.array(jumpsC), np.array(jumpsI)
print(f"\nn correct={len(rawC)}  n incorrect={len(rawI)}")
print("\n% len | rawβ₁ C | rawβ₁ I || normβ₁ C | normβ₁ I  (fraction of final β₁ reached)")
for i, g in enumerate(GRID):
    print(f" {g*100:4.0f}% | {rawC[:,i].mean():6.1f} | {rawI[:,i].mean():6.1f} || "
          f"{normC[:,i].mean():.3f}    | {normI[:,i].mean():.3f}")
print(f"\nLargest localized jump (self-normalized curve):")
print(f"  CORRECT  : mag={jC[:,0].mean():.3f}±{jC[:,0].std():.3f}  pos={jC[:,1].mean()*100:.0f}%±{jC[:,1].std()*100:.0f}%")
print(f"  INCORRECT: mag={jI[:,0].mean():.3f}±{jI[:,0].std():.3f}  pos={jI[:,1].mean()*100:.0f}%±{jI[:,1].std()*100:.0f}%")
from scipy.stats import mannwhitneyu
u,p = mannwhitneyu(jC[:,0], jI[:,0], alternative="two-sided")
print(f"  jump-magnitude correct vs incorrect: Mann-Whitney p={p:.3f}")
# is the jump localized (aha) or just smooth growth? compare max jump to mean step
print(f"  jump concentration: correct max/mean-step ratio={jC[:,0].mean()/(1/19):.2f} (1.0=perfectly smooth)")
