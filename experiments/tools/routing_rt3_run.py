#!/usr/bin/env python3
"""Routing RT3 — the one genuinely-topological test: beta1 redundancy = ROBUSTNESS.

Claim (Theory v2): a persistent 1-cycle in an item's local routing region means TWO independent
routes, so beta1>=1 items should be more ROBUST to perturbation than beta1=0 items. A linear probe
cannot express "two independent paths", so this is the only uniquely-topological prediction.

The causal-ablation version is blocked (manifold is hub-dominated, no item-specific routes to ablate,
see RT4). This OBSERVATIONAL version is not: per item, measure beta1 of the local activation region,
then measure robustness as the drop in the gold solution's mean log-prob under Gaussian noise injected
at L/2. Test whether beta1 predicts SMALLER drop (more robust) beyond the confounds.

CONFOUNDS controlled: solution length, baseline log-prob, and -- critically -- local cloud SPREAD
(beta1 can rise just because a cloud is more spread out, exactly like the length confound in RT2).

  ~/pred1-env/bin/python3 experiments/tools/routing_rt3_run.py --n 8        # smoke
  ~/pred1-env/bin/python3 experiments/tools/routing_rt3_run.py --n 300
"""
import os, sys, json, time, argparse
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from routing_test import l2_block
import actopo

ROOT = HERE.parent.parent
CLOUD = ROOT / "experiments/activations_v3/Qwen3.5-2B-Inst_L14_reasoning.npy"
OUT = ROOT / "experiments/results/routing_rt3.json"
MODEL = "Qwen/Qwen3.5-2B"


def noise_hook(model, layer_module, alpha):
    """Add Gaussian noise sigma = alpha * per-token activation norm to the L/2 block output."""
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        sig = alpha * h.norm(dim=-1, keepdim=True) / (h.shape[-1] ** 0.5)
        h2 = h + torch.randn_like(h) * sig
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return layer_module.register_forward_hook(hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=120)         # local-region size for beta1
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()

    cloud = np.load(CLOUD).astype(np.float32)
    tree = cKDTree(cloud)
    print(f"[load] cloud {cloud.shape}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                attn_implementation="eager", trust_remote_code=True).cuda().eval()
    tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    layer = len(model.model.layers) // 2
    lblock = l2_block(model)
    gsm = load_dataset("openai/gsm8k", "main", split="test")

    def sol_logprob(prompt_ids, sol_ids):
        full = torch.cat([prompt_ids, sol_ids]).unsqueeze(0).cuda()
        with torch.no_grad():
            logits = model(full).logits[0].float()
        lp = F.log_softmax(logits, -1)
        n = prompt_ids.shape[0]
        sel = lp[n - 1:-1][range(len(sol_ids)), sol_ids]
        return float(sel.mean())

    res = json.load(open(OUT)) if OUT.exists() else {}
    t0 = time.time()
    for qi in range(a.n):
        key = str(qi)
        if key in res: continue
        q = gsm[qi]["question"]; sol = gsm[qi]["answer"]
        prompt = ("Please solve the following math problem step by step, and give the final "
                  "numerical answer at the very end after '####'.\n\n" + q)
        enc = tk.apply_chat_template([[{"role": "user", "content": prompt}]],
                add_generation_prompt=True, return_tensors="pt", return_dict=True)
        pids = enc["input_ids"][0]
        sol_ids = tk(sol, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # premise anchor A and its local routing region
        with torch.no_grad():
            A = model(pids.unsqueeze(0).cuda(), output_hidden_states=True).hidden_states[layer][0, -1].float().cpu().numpy()
        nn_idx = tree.query(A, k=a.k)[1]
        local = np.vstack([A[None], cloud[nn_idx]])           # raw local cloud
        m = actopo.measure(local)
        spread = float(np.median(np.linalg.norm(local - local.mean(0), axis=1)))
        base_lp = sol_logprob(pids, sol_ids)
        # robustness: mean drop in solution log-prob under noise (averaged over alphas x seeds)
        drops = {}
        for al in a.alphas:
            ds = []
            for s in range(a.seeds):
                torch.manual_seed(1000 * qi + 17 * s + int(al * 10))
                h = noise_hook(model, lblock, al)
                ds.append(base_lp - sol_logprob(pids, sol_ids))   # positive = degraded
                h.remove()
            drops[str(al)] = float(np.mean(ds))
        res[key] = {"qid": qi, "beta1": int(m.beta1), "beta1_raw": int(m.beta1_raw),
                    "spread": round(spread, 3), "sol_len": int(len(sol_ids)),
                    "base_lp": round(base_lp, 4), "drops": drops}
        if (qi + 1) % 10 == 0 or qi + 1 == a.n:
            json.dump(res, open(str(OUT) + ".tmp", "w")); os.replace(str(OUT) + ".tmp", str(OUT))
            print(f"  item {qi+1}/{a.n}  {time.time()-t0:.0f}s", flush=True)
    analyze(res, a.alphas)
    print(f"DONE {len([k for k in res if k!='_RT3'])} items, {time.time()-t0:.0f}s -> {OUT}", flush=True)


def analyze(res, alphas):
    from scipy import stats
    R = [v for k, v in res.items() if k != "_RT3" and isinstance(v, dict) and "drops" in v]
    b1 = np.array([r["beta1"] for r in R]); b1r = np.array([r["beta1_raw"] for r in R])
    spread = np.array([r["spread"] for r in R]); slen = np.array([r["sol_len"] for r in R], float)
    base = np.array([r["base_lp"] for r in R])
    print(f"\n[RT3] n={len(R)}  beta1 dist: 0={np.mean(b1==0):.2f} >=1={np.mean(b1>=1):.2f}  "
          f"beta1_raw median={np.median(b1r):.0f}", flush=True)
    out = {"n": len(R), "frac_b1_0": float(np.mean(b1 == 0)), "frac_b1_ge1": float(np.mean(b1 >= 1))}
    # pick the alpha whose mean drop is most intermediate (so retention has variance to explain)
    best_al = min((str(al) for al in alphas),
                  key=lambda al: abs(np.mean([r["drops"][al] for r in R]) - np.median([r["drops"][al] for r in R])))
    for al in [str(a) for a in alphas]:
        drop = np.array([r["drops"][al] for r in R])
        # robustness = -drop (less degradation = more robust). Does beta1 predict robustness?
        # partial corr of beta1 with (-drop) controlling spread, sol_len, base_lp
        Z = np.c_[np.ones(len(R)), spread, slen, base]
        def resid(x): return x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        rb, pb = stats.pearsonr(resid(b1.astype(float)), resid(-drop))
        # also raw group contrast
        rob = -drop
        g0 = rob[b1 == 0].mean() if (b1 == 0).any() else float("nan")
        g1 = rob[b1 >= 1].mean() if (b1 >= 1).any() else float("nan")
        out[al] = {"mean_drop": round(float(drop.mean()), 3),
                   "partial_corr_beta1_robust|spread,len,base": [round(rb, 3), round(pb, 3)],
                   "robust_b1_0": round(float(g0), 3), "robust_b1_ge1": round(float(g1), 3),
                   "corr_spread_drop": round(float(np.corrcoef(spread, drop)[0, 1]), 3)}
        print(f"  alpha={al}: mean_drop={drop.mean():.3f}  partial(beta1,robust|spread,len,base) "
              f"r={rb:.3f} p={pb:.3f}  robust[b1=0]={g0:.3f} robust[b1>=1]={g1:.3f}", flush=True)
    pc = out[best_al]["partial_corr_beta1_robust|spread,len,base"]
    out["headline_alpha"] = best_al
    # filtered beta1 must have genuine variance across items, else there is no redundancy to test
    if min(out["frac_b1_0"], out["frac_b1_ge1"]) < 0.10:
        out["verdict"] = "NO_REDUNDANCY_SUBSTRATE"     # filtered beta1 ~constant -> vacuous (like RT4 hubs)
    elif pc[0] > 0 and pc[1] < 0.05:
        out["verdict"] = "REDUNDANCY_ROBUSTNESS"
    else:
        out["verdict"] = "NULL_beta1_redundancy_not_robustness"
    out["note"] = ("FILTERED beta1 (persistent loops = independent routes). NO_REDUNDANCY_SUBSTRATE = "
                   "filtered beta1 has no item-level variance (no redundancy exists to test, vacuous). "
                   "REDUNDANCY_ROBUSTNESS = positive partial controlling spread/length/base (the one "
                   "uniquely-topological signal). Else null.")
    res["_RT3"] = out
    json.dump(res, open(OUT, "w"))


if __name__ == "__main__":
    main()
