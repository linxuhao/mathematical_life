#!/usr/bin/env python3
"""Routing RT2 — route EFFICIENCY = quality, tested WITHIN-QUESTION (difficulty-matched).

Reuses the C3 same-question design (ROUTING_EXPERIMENT_PLAN.md RT2): K completions per GSM8K
question at temp>0; keep MIXED questions (both correct & incorrect traces). Same problem ⇒ same
difficulty ⇒ the route difference between a correct and an incorrect trace is confound-free.

Per completion, measure the ACTUAL reasoning trajectory at L/2 and reduce it to two scalars
(no absolute length — that is a difficulty proxy):
  excess_detour = path_len / geodesic(A, B_trace ; road graph G)   # the "useless detour" amount
  offroad_frac  = fraction of trajectory points OFF the skeleton (far from any road node)
Crux: within-question, do CORRECT traces have lower excess_detour / offroad_frac than WRONG ones?
Direction tested, not assumed.

  ~/pred1-env/bin/python3 experiments/tools/routing_rt2_run.py --nq 8 --k 6      # smoke
  ~/pred1-env/bin/python3 experiments/tools/routing_rt2_run.py --nq 150 --k 8    # full
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
from routing_test import reduce_graph, route_features

ROOT = HERE.parent.parent
CLOUD = ROOT / "experiments/activations_v3/Qwen3.5-2B-Inst_L14_reasoning.npy"
OUT = ROOT / "experiments/results/routing_rt2.json"
MODEL = "Qwen/Qwen3.5-2B"

def extract_ans(t):
    m = re.search(r"####\s*(-?\d+[\.,]?\d*)", t)
    if m: return m.group(1).replace(",", "").replace(".0", "")
    nums = re.findall(r"-?\d+[\.,]?\d*", t)
    return nums[-1].replace(",", "").replace(".0", "") if nums else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--maxnew", type=int, default=512)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--traj_max", type=int, default=256)
    a = ap.parse_args()

    cloud = np.load(CLOUD).astype(np.float32)
    Cr, graph, tree, pca, eps_levels, med = reduce_graph(cloud, n_pca=64, k=a.knn)
    tau = med * 1.5                       # off-skeleton threshold = 1.5x median kNN (PCA space)
    print(f"[load] cloud {cloud.shape} -> PCA{Cr.shape[1]}, median kNN={med:.3f}, tau={tau:.3f}", flush=True)

    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
            attn_implementation="eager", trust_remote_code=True).cuda().eval()
    tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    layer = len(m.model.layers) // 2
    gsm = load_dataset("openai/gsm8k", "main", split="test")

    res = json.load(open(OUT)) if OUT.exists() else {}
    t0 = time.time()
    for qi in range(a.nq):
        key = str(qi)
        if key in res: continue
        q = gsm[qi]["question"]; gold = extract_ans(gsm[qi]["answer"])
        prompt = ("Please solve the following math problem step by step, and give the final "
                  "numerical answer at the very end after '####'.\n\n" + q)
        enc = tk.apply_chat_template([[{"role": "user", "content": prompt}]],
                add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {kk: v.cuda() for kk, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = m.generate(**enc, max_new_tokens=a.maxnew, do_sample=True, temperature=a.temp,
                             top_p=0.95, num_return_sequences=a.k, pad_token_id=tk.eos_token_id)
        comps = []
        for s in range(a.k):
            seq = gen[s]
            gen_ids = seq[plen:]
            glen = int((gen_ids != tk.pad_token_id).sum())
            if glen < 3:
                comps.append({"correct": False, "status": "short"}); continue
            seq = seq[:plen + glen]
            resp = tk.decode(seq[plen:], skip_special_tokens=True)
            correct = (extract_ans(resp) == gold) and gold is not None
            # L/2 trajectory over (prompt-end .. last gen token), projected into road-graph PCA space
            with torch.no_grad():
                hs = m(seq.unsqueeze(0), output_hidden_states=True).hidden_states[layer][0]
            traj = hs[plen - 1:].float().cpu().numpy()       # a_1 = prompt-end (A) .. last gen tok
            if len(traj) > a.traj_max:                       # uniform downsample
                idx = np.linspace(0, len(traj) - 1, a.traj_max).astype(int)
                traj = traj[idx]
            traj = pca.transform(traj)                       # same space as the road graph
            A, Btrace = traj[0], traj[-1]
            path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
            straight = float(np.linalg.norm(A - Btrace))
            excess_detour = path_len / straight if straight > 0 else float("nan")   # tortuosity (graph-free)
            d_geo = route_features(Cr, graph, tree, A, Btrace, eps_levels, k=a.knn)["d_geo"]
            road_excess = path_len / d_geo if np.isfinite(d_geo) and d_geo > 0 else float("nan")
            offroad_frac = float((tree.query(traj, k=1)[0] > tau).mean())
            comps.append({"correct": bool(correct), "status": "ok", "n_gen": glen,
                          "excess_detour": excess_detour, "road_excess": road_excess,
                          "offroad_frac": offroad_frac, "path_len": path_len, "d_geo": float(d_geo)})
        nc = sum(c.get("correct", False) for c in comps if c.get("status") == "ok")
        nok = sum(1 for c in comps if c.get("status") == "ok")
        res[key] = {"qid": qi, "gold": gold, "n_correct": nc, "n_ok": nok, "K": a.k, "comps": comps}
        if (qi + 1) % 2 == 0 or qi + 1 == a.nq:
            json.dump(res, open(str(OUT) + ".tmp", "w")); os.replace(str(OUT) + ".tmp", str(OUT))
            mixed = sum(1 for v in res.values() if 0 < v["n_correct"] < v["n_ok"])
            print(f"  q{qi+1}/{a.nq}  mixed={mixed}  {time.time()-t0:.0f}s", flush=True)

    analyze(res)
    print(f"DONE {len(res)} questions, {time.time()-t0:.0f}s -> {OUT}", flush=True)

def analyze(res):
    """Within-question AUC: do correct traces have lower excess_detour / offroad_frac?"""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    FEATS = ["excess_detour", "road_excess", "offroad_frac"]
    rows = []
    for v in res.values():
        if isinstance(v, dict) and "comps" in v and 0 < v["n_correct"] < v.get("n_ok", 0):
            for c in v["comps"]:
                if c.get("status") == "ok":
                    rows.append((v["qid"], int(c["correct"]), c.get("n_gen", np.nan),
                                 *[c.get(f, np.nan) for f in FEATS]))
    if not rows:
        print("[RT2] no mixed questions yet"); return
    qid = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    ng = np.array([r[2] for r in rows], float)                 # completion length = the confound
    feats = {f: np.array([r[3 + i] for r in rows], float) for i, f in enumerate(FEATS)}
    nq = len(set(qid))

    def wq_auc(f):
        a = []
        for q in set(qid):
            m = (qid == q) & np.isfinite(f); yy = y[m]
            if yy.sum() and (~yy.astype(bool)).sum(): a.append(roc_auc_score(yy, -f[m]))
        return (float(np.mean(a)), len(a)) if a else (float("nan"), 0)

    def length_residualize(f):
        """remove within-question linear dependence on completion length (n_gen)."""
        r = np.full_like(f, np.nan)
        for q in set(qid):
            m = (qid == q) & np.isfinite(f) & np.isfinite(ng)
            if m.sum() > 2:
                x = ng[m] - ng[m].mean(); z = f[m] - f[m].mean()
                b = (x @ z) / (x @ x + 1e-9); r[m] = z - b * x
            elif m.sum(): r[m] = f[m] - f[m][np.isfinite(f[m])].mean()
        return r

    print(f"\n[RT2] mixed questions={nq}  completions={len(rows)}  correct={int(y.sum())}", flush=True)
    len_auc = wq_auc(ng)[0]
    print(f"  {'n_gen (LENGTH)':18s} within-Q AUC(lower=correct) = {len_auc:.3f}   <- the confound", flush=True)
    out = {"mixed_q": nq, "n": len(rows), "n_correct": int(y.sum()),
           "length_within_q_auc": round(len_auc, 3)}
    for name, f in feats.items():
        raw, nqf = wq_auc(f)
        ctl, _ = wq_auc(length_residualize(f))                 # length-controlled
        out[name] = {"within_q_auc_raw": round(raw, 3),
                     "within_q_auc_length_controlled": round(ctl, 3), "n_q": nqf}
        print(f"  {name:18s} raw={raw:.3f}  length-controlled={ctl:.3f}  ({nqf} q)", flush=True)
    # verdict is on the LENGTH-CONTROLLED auc (a real routing signal must beat the length confound)
    out["verdict"] = ("QUALITY_SIGNAL" if any(
        np.isfinite(out[n]["within_q_auc_length_controlled"]) and
        out[n]["within_q_auc_length_controlled"] > 0.55 for n in FEATS)
        else "NULL_route_quality_is_length_confound")
    out["note"] = ("length-controlled >0.55 => genuine route-quality signal beyond completion length; "
                   "~0.5 => the apparent signal was completion length (the C3 lesson).")
    res["_RT2"] = out
    json.dump(res, open(OUT, "w"))

if __name__ == "__main__":
    main()
