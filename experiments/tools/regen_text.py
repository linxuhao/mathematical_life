"""Regenerate greedy CoT text + per-token alignment for selected GSM8K idxs.
Greedy (do_sample=False) is deterministic, so text matches the saved trajectories
exactly (verified by n_gen). Saves token list so a text error-location maps to a
token index k for the b1(k) curve.
"""
import os, sys, json, re
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

IDXS = [255,449, 58,438, 89,382, 115,369, 303,333, 306,268, 51,277, 478,200]
MODEL = "Qwen/Qwen3.5-2B"; MAXNEW = 1024

m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
        attn_implementation="eager", trust_remote_code=True).cuda().eval()
tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tk.pad_token is None: tk.pad_token = tk.eos_token
gsm = load_dataset("openai/gsm8k","main",split="test")
labels = {r["idx"]: r for r in json.load(open("activations_v3/tokenwise_v2/labels.json"))}

out = {}
for i in IDXS:
    q = gsm[i]["question"]
    prompt = ("Please solve the following math problem step by step, and give "
              "the final numerical answer at the very end after '####'.\n\n" + q)
    enc = tk.apply_chat_template([[{"role":"user","content":prompt}]],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.cuda() for k, v in enc.items()}
    plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        gen = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tk.eos_token_id)
    gid = gen[0][plen:]
    toks = [tk.decode([t]) for t in gid]
    text = tk.decode(gid, skip_special_tokens=True)
    out[str(i)] = {"idx": i, "correct": labels[i]["correct"], "gold": labels[i]["gold"],
                   "pred": labels[i]["pred"], "n_gen_saved": labels[i]["n_gen"],
                   "n_gen_regen": int(gid.shape[0]), "question": q, "text": text, "tokens": toks}
    print(f"idx={i} correct={labels[i]['correct']} n_gen saved={labels[i]['n_gen']} regen={int(gid.shape[0])} "
          f"{'MATCH' if labels[i]['n_gen']==int(gid.shape[0]) else 'MISMATCH!'}", flush=True)

json.dump(out, open("results/tokenwise_v2_text.json","w"), indent=1)
print("saved results/tokenwise_v2_text.json")
