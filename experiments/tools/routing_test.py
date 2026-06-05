#!/usr/bin/env python3
"""Routing / highway test — does the manifold's connectivity structure route reasoning?

Tests the routing interpretation of the topological skeleton (see ROUTING_EXPERIMENT_DESIGN.md).
The correlational P1 test (beta1-count vs capability) cannot falsify the highway hypothesis;
this does. Three observables:
  R1 connectivity (beta0)  -> capacity   : are premise A and conclusion B in one component?
  R2 geodesic (route len)  -> quality     : graph-geodesic A->B, detour = d_geo / d_euclid
  R3 redundancy (beta1)    -> robustness  : does the routing region carry a 1-cycle?
  R4 ablation (causal)     -> capacity    : close the route -> only route-dependent inferences fail

Stage 1 (observational, runnable now): R1+R2 predict per-item correctness beyond Euclidean+difficulty?
Stage 2 (causal): ablate bridge directions, measure reasoning-vs-fluency dissociation.

CPU smoke test (no GPU/model needed):
    python routing_test.py --smoke
Server (real run):
    python routing_test.py --stage 1 --model Qwen/Qwen3.5-2B --items items_gsm8k.jsonl --out routing_s1.json
    python routing_test.py --stage 2 --model Qwen/Qwen3.5-2B --items items_gsm8k.jsonl --out routing_s2.json

items JSONL schema (one obj/line): {"problem": str, "solution": str, "correct": 0/1, "difficulty": float?}
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

# ----------------------------------------------------------------------------- graph / routing core
def build_road_graph(cloud: np.ndarray, k: int = 15):
    """Symmetric kNN graph over the reference activation cloud. Returns (csr_dist, kdtree)."""
    n = len(cloud)
    tree = cKDTree(cloud)
    dist, idx = tree.query(cloud, k=min(k + 1, n))  # +1: self
    rows, cols, vals = [], [], []
    for i in range(n):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):   # skip self
            rows += [i, j]; cols += [j, i]; vals += [d, d]   # symmetrise
    g = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return g, tree


def reduce_graph(cloud, n_pca=64, k=15):
    """PCA-reduce the cloud (kills high-dim distance concentration so kNN neighbourhoods are
    meaningful), build the road graph, and scale eps to the actual edge-length distribution.
    Returns (cloud_reduced, graph, tree, pca, eps_levels, median_knn). Project anchors/trajectories
    with `pca.transform` before calling route_features so they live in the SAME space."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_pca, cloud.shape[1])).fit(cloud)
    Cr = pca.transform(cloud).astype(np.float32)
    graph, tree = build_road_graph(Cr, k=k)
    med = float(np.median(tree.query(Cr, k=2)[0][:, 1]))      # median nearest-neighbour distance
    eps_levels = [mlt * med for mlt in (1.0, 1.5, 2.0, 3.0, 5.0)]
    return Cr, graph, tree, pca, eps_levels, med


def route_features(cloud, graph, tree, A, B, eps_levels, k=15):
    """For premise A and conclusion B (1-D vectors), compute R1/R2 features against road graph.

    A,B are connected into the graph via their k nearest road nodes, then we measure
    connectivity (R1) and geodesic (R2) between them.
    """
    n = cloud.shape[0]
    iA, iB = n, n + 1
    # distances from A and B to their road-node neighbours
    dA, jA = tree.query(A, k=k); dB, jB = tree.query(B, k=k)
    dAB = float(np.linalg.norm(A - B))
    # augmented graph (road + A + B)
    rows, cols, vals = list(graph.nonzero()[0]), list(graph.nonzero()[1]), list(graph.data)
    for d, j in zip(np.atleast_1d(dA), np.atleast_1d(jA)):
        rows += [iA, j]; cols += [j, iA]; vals += [d, d]
    for d, j in zip(np.atleast_1d(dB), np.atleast_1d(jB)):
        rows += [iB, j]; cols += [j, iB]; vals += [d, d]
    # NO direct A-B edge: d_geo is the ROAD-ONLY manifold geodesic (>= Euclidean, or inf if A,B fall
    # in different road components -> that disconnection is itself the R1 capacity signal).
    aug = csr_matrix((vals, (rows, cols)), shape=(n + 2, n + 2))

    d_geo = float(dijkstra(aug, indices=iA, return_predecessors=False)[iB])
    detour = d_geo / dAB if (dAB > 0 and np.isfinite(d_geo)) else float("inf")  # >=1; inf = no road route
    # R1: connectivity at each eps (component test on the eps-thresholded augmented graph)
    route_exists = {}
    merge_eps = np.inf
    for eps in eps_levels:
        mask = aug.copy(); mask.data = (mask.data <= eps).astype(float)
        mask.eliminate_zeros()
        ncomp, lab = connected_components(mask, directed=False)
        same = bool(lab[iA] == lab[iB])
        route_exists[float(eps)] = same
        if same and eps < merge_eps:
            merge_eps = float(eps)
    return {"d_euclid": dAB, "d_geo": d_geo, "detour": detour,
            "merge_eps": merge_eps, "route_exists": route_exists}


# ----------------------------------------------------------------------------- Stage 1 analysis
def stage1_analysis(feats, correct, difficulty, eps_report):
    """5-fold CV AUC: baseline {Euclid, difficulty} vs + routing {detour, merge_eps, route_exists}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from scipy import stats

    y = np.asarray(correct, float)
    eu = np.array([f["d_euclid"] for f in feats])
    diff = np.asarray(difficulty, float)
    det = np.array([f["detour"] for f in feats])
    mrg = np.array([min(f["merge_eps"], 10 * eps_report) for f in feats])  # cap inf
    re = np.array([float(f["route_exists"].get(eps_report, False)) for f in feats])

    def auc(X):
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        p = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5,
                              method="predict_proba")[:, 1]
        return roc_auc_score(y, p)

    base = np.c_[eu, diff]
    full = np.c_[eu, diff, det, mrg, re]
    auc_base, auc_full = auc(base), auc(full)

    def partial(x):
        # corr(x, y | eu, diff) via residuals
        Z = np.c_[np.ones_like(eu), eu, diff]
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        r, p = stats.pearsonr(rx, ry); return float(r), float(p)

    return {
        "n": int(len(y)), "auc_baseline_euclid_diff": round(auc_base, 4),
        "auc_with_routing": round(auc_full, 4), "delta_auc": round(auc_full - auc_base, 4),
        "partial_detour|euclid,diff": partial(det),
        "partial_mergeeps|euclid,diff": partial(mrg),
        "partial_routeexists|euclid,diff": partial(re),
        "verdict": ("ROUTING_PREDICTIVE" if auc_full - auc_base >= 0.03 else "NULL_routing_adds_nothing"),
    }


# ----------------------------------------------------------------------------- Stage 2 ablation
def ablate_directions_hook(model, layer_module, directions):
    """Register a forward hook that projects out `directions` (d x q, orthonormal) from the
    L/2 block output for every token. Returns the handle (call .remove() after eval)."""
    import torch
    P = torch.tensor(directions, dtype=next(model.parameters()).dtype,
                     device=next(model.parameters()).device)  # (d, q)

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out          # (B,T,d)
        coeff = h @ P                                          # (B,T,q)
        h2 = h - coeff @ P.T                                   # remove the routing subspace
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return layer_module.register_forward_hook(hook)


def bridge_directions(waypoint_acts, q):
    """Top-q principal directions of the route waypoints' covariance (the route subspace)."""
    X = waypoint_acts - waypoint_acts.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:q].T                                            # (d, q) orthonormal


def matched_variance_directions(cloud, q, target_var, bridge, rng):
    """Variance-matched confound control: q global-PCA directions whose removed variance ~ target_var
    AND that are ~orthogonal to the bridge subspace (so they remove 'as much structure' but not the
    route). Random orthonormal subspaces cannot match a top-PC subspace's variance, so we search the
    cloud's own spectrum instead."""
    Xc = cloud - cloud.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp_var = (S ** 2) / len(cloud)                       # per-PC removed variance
    align = np.abs(Vt @ bridge).max(1) if bridge is not None else np.zeros(len(S))
    cand = np.where(align < 0.2)[0]                        # PCs not aligned with the route
    # greedily pick q candidate PCs whose summed variance is closest to target_var
    best, order = None, cand[rng.permutation(len(cand))]
    for start in range(0, max(1, len(order) - q)):
        sel = order[start:start + q]
        if len(sel) < q: break
        v = float(comp_var[sel].sum())
        if best is None or abs(v - target_var) < best[0]:
            best = (abs(v - target_var), Vt[sel].T)
    return best[1]


# ----------------------------------------------------------------------------- model glue (server)
def load_model(name, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=getattr(torch, dtype), output_hidden_states=True).eval()
    return model, tok


def l2_block(model):
    """Return the decoder block module at L/2 (for hooks)."""
    layers = model.model.layers
    return layers[len(layers) // 2]


# ----------------------------------------------------------------------------- smoke test (CPU only)
def smoke():
    """Exercise the pure-CPU routing machinery on actopo's bundled example cloud — no GPU/model."""
    from actopo.data import load_example_cloud
    cloud = load_example_cloud("base")           # (600, 896)
    print(f"[smoke] road cloud {cloud.shape}")
    graph, tree = build_road_graph(cloud, k=15)
    rng = np.random.default_rng(0)
    eps_max = float(np.max(tree.query(cloud, k=2)[0][:, 1]) * 20)  # crude scale
    eps_levels = [f * eps_max for f in (0.01, 0.03, 0.05, 0.1, 0.2)]
    feats, correct, diff = [], [], []
    for _ in range(120):                          # synthetic A,B pairs from cloud +/- noise
        a, b = cloud[rng.integers(len(cloud))], cloud[rng.integers(len(cloud))]
        f = route_features(cloud, graph, tree, a, b, eps_levels); feats.append(f)
        # synthetic balanced label with a faint detour signal (just to exercise the regression)
        p = 0.5 + 0.3 * (f["detour"] < 0.999)
        correct.append(int(rng.random() < p))
        diff.append(float(np.linalg.norm(a) + rng.normal()))
    res = stage1_analysis(feats, correct, diff, eps_report=eps_levels[1])
    print("[smoke] stage1:", json.dumps(res, indent=2))
    # stage-2 machinery
    wp = cloud[rng.integers(len(cloud), size=10)]
    B = bridge_directions(wp, q=4)
    Xc = cloud - cloud.mean(0)
    tv = float(np.var(Xc @ B))
    R = matched_variance_directions(cloud, 4, tv, B, rng)
    print(f"[smoke] bridge dirs {B.shape}, matched-var {R.shape}; "
          f"var(bridge)={tv:.3f} var(matched)={np.var(Xc@R):.3f} "
          f"max|align(matched,bridge)|={np.abs(R.T@B).max():.3f}")
    print("[smoke] OK — routing/ablation machinery runs.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", type=int, choices=[1, 2])
    ap.add_argument("--model"); ap.add_argument("--items"); ap.add_argument("--out")
    ap.add_argument("--reasoning_cloud", help="npy of L/2 reasoning activations (road network)")
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()
    if a.smoke:
        smoke(); return
    if a.stage == 1:
        sys.exit("Stage 1 server run: implement item extraction with actopo.extract_activations "
                 "(premise=last-token of problem; conclusion=last-token of problem+solution), build "
                 "road graph from --reasoning_cloud, then stage1_analysis. See ROUTING_EXPERIMENT_DESIGN.md §2.")
    if a.stage == 2:
        sys.exit("Stage 2 server run: identify bridge_directions on route waypoints, ablate via "
                 "ablate_directions_hook, eval reasoning(GSM8K/ARC) vs fluency(LAMBADA/ppl). §3.")


if __name__ == "__main__":
    main()
