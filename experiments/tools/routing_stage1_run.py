#!/usr/bin/env python3
"""Routing Stage 1 server runner — RT1 (connectivity=capacity) + RT3 (redundancy) + RT-S1.

Tests whether the manifold's CONNECTIVITY structure predicts GSM8K solvability beyond
straight-line distance + difficulty (ROUTING_EXPERIMENT_PLAN.md, Stage 1, cross-problem).
RT2 (quality, within-question) is a separate runner reusing C3_sameq.

Model: Qwen/Qwen3.5-2B (chat) to match C3. Road graph: the cached 1319-prompt reasoning
cloud at L/2 (instruct). FROZEN_V5 (L/2, last real token, bf16).

  ~/pred1-env/bin/python3 experiments/tools/routing_stage1_run.py --n 8        # smoke
  ~/pred1-env/bin/python3 experiments/tools/routing_stage1_run.py --n 500 --full

Outputs experiments/results/routing_s1.json : per-item features + RT-S1 adjudication.
"""
import os, sys, json, re, time, argparse
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from routing_test import reduce_graph, route_features              # reuse graph core

ROOT = HERE.parent.parent
CLOUD = ROOT / "experiments/activations_v3/Qwen3.5-2B-Inst_L14_reasoning.npy"
OUT = ROOT / "experiments/results/routing_s1.json"
MODEL = "Qwen/Qwen3.5-2B"

def extract_ans(t):
    m = re.search(r"####\s*(-?\d+[\.,]?\d*)", t)
    if m: return m.group(1).replace(",", "").replace(".0", "")
    nums = re.findall(r"-?\d+[\.,]?\d*", t)
    return nums[-1].replace(",", "").replace(".0", "") if nums else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--maxnew", type=int, default=512)
    a = ap.parse_args()

    print(f"[load] road cloud {CLOUD}", flush=True)
    cloud = np.load(CLOUD).astype(np.float32)
    print(f"[load] cloud {cloud.shape}; PCA->{a.pca}, kNN(k={a.k}) road graph", flush=True)
    Cr, graph, tree, pca, eps_levels, med = reduce_graph(cloud, n_pca=a.pca, k=a.k)
    eps_report = eps_levels[1]
    print(f"[graph] median kNN dist={med:.3f}; eps_levels={[round(e,3) for e in eps_levels]}", flush=True)

    print(f"[load] model {MODEL}", flush=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
            attn_implementation="eager", trust_remote_code=True).cuda().eval()
    tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    layer = len(m.model.layers) // 2
    gsm = load_dataset("openai/gsm8k", "main", split="test")

    def l2_last(ids):
        with torch.no_grad():
            hs = m(ids.unsqueeze(0).cuda(), output_hidden_states=True).hidden_states[layer]
        return hs[0, -1].float().cpu().numpy()

    res = json.load(open(OUT)) if OUT.exists() else {}
    feats, t0 = [], time.time()
    for qi in range(a.n):
        key = str(qi)
        if key in res:                      # resume: skip already-computed items
            feats.append(res[key]); continue
        q = gsm[qi]["question"]; gold = extract_ans(gsm[qi]["answer"]); sol = gsm[qi]["answer"]
        prompt = ("Please solve the following math problem step by step, and give the final "
                  "numerical answer at the very end after '####'.\n\n" + q)
        enc = tk.apply_chat_template([[{"role": "user", "content": prompt}]],
                add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"][0]
        # premise anchor A = L/2 last token of the prompt (problem only), projected into road-graph PCA space
        A = pca.transform(l2_last(ids)[None])[0]
        # conclusion anchor B = L/2 last token of (prompt + gold solution)
        sol_ids = tk(sol, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        B = pca.transform(l2_last(torch.cat([ids, sol_ids]))[None])[0]
        # correctness = model's own greedy answer vs gold
        with torch.no_grad():
            gen = m.generate(ids.unsqueeze(0).cuda(), max_new_tokens=a.maxnew, do_sample=False,
                             pad_token_id=tk.eos_token_id)
        resp = tk.decode(gen[0, ids.shape[0]:], skip_special_tokens=True)
        correct = (extract_ans(resp) == gold) and gold is not None
        # routing features against the road graph
        rf = route_features(Cr, graph, tree, A, B, eps_levels, k=a.k)
        rec = {"qid": qi, "correct": bool(correct), "difficulty": int(ids.shape[0]),
               "d_euclid": rf["d_euclid"], "d_geo": rf["d_geo"], "detour": rf["detour"],
               "merge_eps": rf["merge_eps"], "route_exists": rf["route_exists"]}
        res[key] = rec; feats.append(rec)
        if (qi + 1) % 5 == 0 or qi + 1 == a.n:
            json.dump(res, open(str(OUT) + ".tmp", "w")); os.replace(str(OUT) + ".tmp", str(OUT))
            nc = sum(r["correct"] for r in res.values())
            print(f"  item {qi+1}/{a.n}  correct={nc}/{len(res)}  {time.time()-t0:.0f}s", flush=True)

    # RT-S1 adjudication (only if both classes present)
    R = [v for k, v in res.items() if k != "_RT_S1"]
    y = np.array([r["correct"] for r in R], float)
    if 0 < y.sum() < len(y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        from scipy import stats
        eu = np.array([r["d_euclid"] for r in R]); diff = np.array([r["difficulty"] for r in R], float)
        det = np.array([min(r["detour"], 5.0) for r in R]); mrg = np.array([min(r["merge_eps"], 10*eps_report) for r in R])
        re_ = np.array([float(r["route_exists"].get(str(eps_report), r["route_exists"].get(eps_report, False))) for r in R])
        def auc(X):
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
            cv = min(5, int(y.sum()), int((y == 0).sum()))
            p = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=max(2, cv), method="predict_proba")[:, 1]
            return roc_auc_score(y, p)
        ab = auc(np.c_[eu, diff]); af = auc(np.c_[eu, diff, det, mrg, re_])
        def partial(x):
            Z = np.c_[np.ones_like(eu), eu, diff]
            rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            return [float(v) for v in stats.pearsonr(rx, ry)]
        adj = {"n": len(R), "n_correct": int(y.sum()), "auc_baseline": round(ab, 4),
               "auc_with_routing": round(af, 4), "delta_auc": round(af - ab, 4),
               "partial_detour": partial(det), "partial_mergeeps": partial(mrg),
               "partial_routeexists": partial(re_),
               "verdict": "ROUTING_PREDICTIVE" if af - ab >= 0.03 else "NULL_routing_adds_nothing"}
        res["_RT_S1"] = adj
        json.dump(res, open(OUT, "w"))
        print("\n[RT-S1]", json.dumps(adj, indent=2), flush=True)
    else:
        print(f"\n[RT-S1] skipped: need both classes (correct={int(y.sum())}/{len(y)}). Increase --n.", flush=True)
    print(f"DONE {len(R)} items, {time.time()-t0:.0f}s -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
