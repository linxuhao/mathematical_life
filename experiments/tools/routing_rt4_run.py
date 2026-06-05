#!/usr/bin/env python3
"""Routing RT4 (Stage 2, the capstone) — CAUSAL route ablation.

The decisive test the framework named but never ran (ROUTING_EXPERIMENT_PLAN.md RT4): close the
low-dimensional subspace that holds a premise->conclusion route open and check whether ONLY
route-dependent (reasoning) inferences fail, sparing fluency. Two matched controls:
  - random   : q random orthonormal directions
  - matched  : q non-route directions carrying ~equal variance (the critical confound control)
Crux = DISSOCIATION: (Δreasoning - Δfluency) for bridge vs matched, dose-dependent.

Reasoning metric : ARC-Easy accuracy (likelihood-MC, route-dependent multi-step).
Fluency metric   : LAMBADA last-token accuracy + HellaSwag (pattern/continuation).
All evals are likelihood-based (no generation) so the condition x dose sweep is fast.

  ~/pred1-env/bin/python3 experiments/tools/routing_rt4_run.py --smoke           # tiny
  ~/pred1-env/bin/python3 experiments/tools/routing_rt4_run.py --n_eval 300 --n_route 200
"""
import os, sys, json, time, argparse
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from routing_test import (reduce_graph, route_features, bridge_directions,
                          matched_variance_directions, ablate_directions_hook, l2_block)
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

ROOT = HERE.parent.parent
CLOUD = ROOT / "experiments/activations_v3/Qwen3.5-2B-Inst_L14_reasoning.npy"
OUT = ROOT / "experiments/results/routing_s2.json"
MODEL = "Qwen/Qwen3.5-2B"


# ----------------------------------------------------------------- likelihood evals (no generation)
def choice_logprob(model, tk, layer_mod, context, choice):
    """mean log p(choice tokens | context)."""
    ctx = tk(context, return_tensors="pt").input_ids.cuda()
    full = tk(context + choice, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        logits = model(full).logits[0].float()
    lp = F.log_softmax(logits, -1)
    n_ctx = ctx.shape[1]; tgt = full[0, n_ctx:]
    if len(tgt) == 0: return -1e9
    sel = lp[n_ctx - 1:-1][range(len(tgt)), tgt]
    return float(sel.mean())


def eval_arc(model, tk, items):
    correct = 0
    for it in items:
        q = it["question"]; choices = it["choices"]["text"]; labels = it["choices"]["label"]
        gold = it["answerKey"]
        scores = [choice_logprob(model, tk, None, f"Question: {q}\nAnswer: ", c) for c in choices]
        if labels[int(np.argmax(scores))] == gold: correct += 1
    return correct / len(items)


def eval_hellaswag(model, tk, items):
    correct = 0
    for it in items:
        ctx = it["ctx"]; ends = it["endings"]; gold = int(it["label"])
        scores = [choice_logprob(model, tk, None, ctx + " ", e) for e in ends]
        if int(np.argmax(scores)) == gold: correct += 1
    return correct / len(items)


def eval_lambada(model, tk, items):
    """last-word accuracy: greedy-predict the final token, check match."""
    correct = 0
    for txt in items:
        words = txt.strip().rsplit(" ", 1)
        if len(words) < 2: continue
        ctx, last = words[0], " " + words[1]
        ctx_ids = tk(ctx, return_tensors="pt").input_ids.cuda()
        last_ids = tk(last, add_special_tokens=False, return_tensors="pt").input_ids[0]
        with torch.no_grad():
            logits = model(torch.cat([ctx_ids, last_ids.cuda().unsqueeze(0)], 1)).logits[0]
        n = ctx_ids.shape[1]
        pred = logits[n - 1:-1].argmax(-1).cpu()
        if torch.equal(pred, last_ids): correct += 1
    return correct / len(items)


def run_evals(model, tk, arc, hella, lam):
    return {"reasoning_arc": round(eval_arc(model, tk, arc), 4),
            "fluency_lambada": round(eval_lambada(model, tk, lam), 4),
            "commonsense_hellaswag": round(eval_hellaswag(model, tk, hella), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--n_route", type=int, default=200)
    ap.add_argument("--doses", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--knn", type=int, default=15)
    a = ap.parse_args()
    if a.smoke:
        a.n_eval, a.n_route, a.doses = 20, 40, [4]

    cloud = np.load(CLOUD).astype(np.float32)            # raw 2048-dim road-node activations
    Cr, graph, tree, pca, eps_levels, med = reduce_graph(cloud, n_pca=64, k=a.knn)
    print(f"[load] cloud {cloud.shape} -> PCA{Cr.shape[1]}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                attn_implementation="eager", trust_remote_code=True).cuda().eval()
    tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    layer = len(model.model.layers) // 2
    lblock = l2_block(model)
    gsm = load_dataset("openai/gsm8k", "main", split="train")

    def l2_last(text):
        ids = tk(text, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            return model(ids, output_hidden_states=True).hidden_states[layer][0, -1].float().cpu().numpy()

    # --- collect route waypoints (raw activations of road nodes on each A->B geodesic) ---
    print(f"[route] collecting waypoints for {a.n_route} reasoning items", flush=True)
    wp_idx = set()
    n = cloud.shape[0]
    for qi in range(a.n_route):
        q = gsm[qi]["question"]; sol = gsm[qi]["answer"]
        A = pca.transform(l2_last(q)[None])[0]
        B = pca.transform(l2_last(q + "\n" + sol)[None])[0]
        dA, jA = tree.query(A, k=a.knn); dB, jB = tree.query(B, k=a.knn)
        rows, cols, vals = list(graph.nonzero()[0]), list(graph.nonzero()[1]), list(graph.data)
        iA, iB = n, n + 1
        for d, j in zip(dA, jA): rows += [iA, j]; cols += [j, iA]; vals += [d, d]
        for d, j in zip(dB, jB): rows += [iB, j]; cols += [j, iB]; vals += [d, d]
        aug = csr_matrix((vals, (rows, cols)), shape=(n + 2, n + 2))
        dist, pred = dijkstra(aug, indices=iA, return_predecessors=True)
        j = iB                                            # walk predecessors A<-..<-B
        while j != iA and j >= 0 and pred[j] >= 0:
            if j < n: wp_idx.add(int(j))
            j = pred[j]
    # expand to the route REGION (path nodes + their neighbors) so the bridge subspace is not
    # rank-degenerate; hub-dominated geodesics give very few raw path nodes (itself an RT1-consistent
    # signal that there is no item-specific route structure).
    path_nodes = sorted(wp_idx)
    print(f"[route] {len(path_nodes)} distinct path nodes (pre-expansion)", flush=True)
    if path_nodes:
        nbr = tree.query(Cr[path_nodes], k=min(8, n))[1].reshape(-1)
        wp_idx.update(int(i) for i in nbr)
    wp = cloud[sorted(wp_idx)]                            # raw waypoint activations (route region)
    print(f"[route] {len(wp_idx)} route-region waypoints (after neighbor expansion)", flush=True)

    # eval sets
    arc = list(load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation"))[:a.n_eval]
    hella = list(load_dataset("Rowan/hellaswag", split="validation"))[:a.n_eval]
    lam = [x["text"] for x in load_dataset("EleutherAI/lambada_openai", "en", split="test")][:a.n_eval]

    rng = np.random.default_rng(0)
    # record the hub-domination evidence so the paper's "k of N nodes" claim is reproducible from
    # committed output, not just stdout (n_total_nodes = road-graph node count = #prompts).
    res = {"baseline": run_evals(model, tk, arc, hella, lam), "doses": {},
           "route_structure": {"n_path_nodes": len(path_nodes),
                               "n_route_region_waypoints": len(wp_idx),
                               "n_total_nodes": int(n),
                               "n_route_items": int(a.n_route)}}
    print("[eval] baseline:", res["baseline"], flush=True)
    Xc = cloud - cloud.mean(0)
    d = cloud.shape[1]
    qmax = max(1, min(max(a.doses), len(wp) - 1))                 # cap doses at the route subspace rank
    doses = [q for q in a.doses if q <= qmax] or [qmax]
    for q in doses:
        bridge = bridge_directions(wp, q=q)                       # (d,q) route subspace
        tv = float(np.var(Xc @ bridge))
        randd = np.linalg.qr(rng.standard_normal((d, q)))[0][:, :q]
        matched = matched_variance_directions(cloud, q, tv, bridge, rng)
        conds = {}
        for name, P in (("bridge", bridge), ("random", randd), ("matched", matched)):
            h = ablate_directions_hook(model, lblock, P)
            conds[name] = run_evals(model, tk, arc, hella, lam)
            h.remove()
            print(f"[eval] q={q} {name}: {conds[name]}", flush=True)
        # dissociation: (Δreasoning - Δfluency), bridge vs matched
        b0 = res["baseline"]
        def drop(c, key): return b0[key] - c[key]
        diss = {nm: round(drop(conds[nm], "reasoning_arc") - drop(conds[nm], "fluency_lambada"), 4)
                for nm in conds}
        res["doses"][str(q)] = {"conds": conds, "var_bridge": round(tv, 3),
                                "var_matched": round(float(np.var(Xc @ matched)), 3),
                                "dissociation_reason_minus_fluency": diss}
        json.dump(res, open(OUT, "w"))
    # verdict
    big = max(int(k) for k in res["doses"])
    diss = res["doses"][str(big)]["dissociation_reason_minus_fluency"]
    res["verdict"] = ("ROUTING_CAUSAL" if diss["bridge"] > diss["matched"] + 0.05
                      else "NULL_no_selective_route_damage")
    json.dump(res, open(OUT, "w"))
    print(f"\n[RT4] verdict={res['verdict']}  dissociation@q{big}={diss} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
