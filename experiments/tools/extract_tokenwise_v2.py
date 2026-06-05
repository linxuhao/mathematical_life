"""Phase-5 re-extraction, done right.

Fixes vs the original tokenwise run:
  - Generate to COMPLETION (max_new_tokens=1024, stop on EOS) so trajectories
    cover the full reasoning incl. the divergence/answer (old run: 94% truncated
    at 256 → labels were really "finished in 256 tokens", not "correct").
  - Label correctness on the COMPLETE output, with last-number fallback.
  - Save each generated-token trajectory (L/2) + clean labels, RESUMABLE,
    detached. Whole-trajectory β₁ computed inline; cumulative/aha analysis runs
    later on the saved trajectories.
"""
import os, sys, json, re, time
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
sys.path.insert(0, "/home/linxuhao/papers/mathematical-life/actopo/src")
from actopo import FROZEN_V5, measure

MODEL = "Qwen/Qwen3.5-2B"
N = 500
MAXNEW = 1024
OUTDIR = Path("/home/linxuhao/papers/mathematical-life/experiments/activations_v3/tokenwise_v2")
OUTDIR.mkdir(parents=True, exist_ok=True)
LABELS = OUTDIR / "labels.json"


def extract_ans(t):
    m = re.search(r"####\s*(-?\d+[\.,]?\d*)", t)
    if m:
        return m.group(1).replace(",", "").replace(".0", "")
    nums = re.findall(r"-?\d+[\.,]?\d*", t)
    return nums[-1].replace(",", "").replace(".0", "") if nums else None


def atomic(obj, path):
    tmp = f"{path}.tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def main():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="eager", trust_remote_code=True
    ).cuda().eval()
    tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    layer = len(m.model.layers) // 2

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    labels = json.load(open(LABELS)) if LABELS.exists() else []
    done = {r["idx"] for r in labels}
    print(f"resume: {len(done)} done; layer={layer}", flush=True)

    for i in range(N):
        if i in done:
            continue
        t0 = time.time()
        q = gsm[i]["question"]; gold = extract_ans(gsm[i]["answer"])
        prompt = ("Please solve the following math problem step by step, and give "
                  "the final numerical answer at the very end after '####'.\n\n" + q)
        enc = tk.apply_chat_template(
            [[{"role": "user", "content": prompt}]],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.cuda() for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                             pad_token_id=tk.eos_token_id)
        seq = gen[0]; gen_ids = seq[plen:]; n_gen = int(gen_ids.shape[0])
        resp = tk.decode(gen_ids, skip_special_tokens=True)
        pred = extract_ans(resp); correct = (pred is not None and pred == gold)
        capped = n_gen >= MAXNEW
        with torch.no_grad():
            hs = m(seq.unsqueeze(0), output_hidden_states=True).hidden_states[layer][0]
        traj = hs[plen:plen + n_gen].float().cpu().numpy().astype(np.float16)
        np.save(OUTDIR / f"traj_{i}.npy", traj)
        b1 = raw = surv = None
        if traj.shape[0] >= 10:
            r = measure(traj.astype(np.float32), FROZEN_V5)
            b1, raw, surv = r.beta1, r.beta1_raw, r.survival_rate
        labels.append({"idx": i, "correct": bool(correct), "gold": gold, "pred": pred,
                       "n_gen": n_gen, "capped": bool(capped),
                       "beta1": b1, "beta1_raw": raw, "surv": surv})
        atomic(labels, LABELS)
        nc = sum(r["correct"] for r in labels)
        print(f"[{len(labels)}/{N}] idx={i} correct={correct} n_gen={n_gen} "
              f"capped={capped} β₁={b1} | acc={nc}/{len(labels)} ({time.time()-t0:.0f}s)", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
