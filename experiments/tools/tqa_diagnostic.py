"""Settle whether the TruthfulQA 51%->27.5% drop is real truthfulness or position bias.

For step-0 (pre-SFT instruct) and step-14750 (post-SFT), on N questions:
  (a) unshuffled letter-gen  -> reproduces the pipeline's number (correct always 'A')
  (b) SHUFFLED letter-gen     -> correct moved to a random position; if (a)>>(b)~random,
                                 the high score was POSITION BIAS, not truthfulness
  (c) likelihood MC1 (position-free) -> score each answer's TEXT; pick argmax. The
                                 benchmark's real metric. No letter, no position.
"""
import os, random
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

N = 200
CKPTS = [("step0 (pre-SFT)", "Qwen/Qwen3.5-2B"),
         ("step14750 (post-SFT)",
          "/home/linxuhao/papers/mathematical-life/experiments/checkpoints/qwen35_2b_inst_sft/checkpoint-14750")]
tqa = list(load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation"))[:N]


def letter_prompt(q, choices):
    letters = "\n".join(f"{chr(65+k)}. {c}" for k, c in enumerate(choices))
    return f"Q: {q}\n{letters}\nAnswer with the letter:"


def gen_letter(m, tk, prompt):
    enc = tk.apply_chat_template([[{"role": "user", "content": prompt}]],
                                 add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=5, do_sample=False, pad_token_id=tk.eos_token_id)
    out = tk.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return out[0].upper() if out else "?"


def loglik(m, tk, prompt, choice):
    pids = tk(prompt, return_tensors="pt").input_ids
    fids = tk(prompt + " " + choice, return_tensors="pt").input_ids.cuda()
    plen = pids.shape[1]
    with torch.no_grad():
        logits = m(fids).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    pos = torch.arange(plen, fids.shape[1], device=fids.device)
    tok_lp = lp[pos - 1, fids[0, pos]]
    return tok_lp.sum().item(), tok_lp.mean().item()   # total, length-normalized


def run(tag, path):
    print(f"\n############## {tag} ##############", flush=True)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
                                             attn_implementation="eager", trust_remote_code=True).cuda().eval()
    tk = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    a_ok = b_ok = c_tot = c_norm = n = 0
    for qi, q in enumerate(tqa):
        ch = q["mc1_targets"]["choices"]
        if len(ch) < 2:
            continue
        n += 1
        # (a) unshuffled: correct at index 0 -> letter A
        if gen_letter(m, tk, letter_prompt(q["question"], ch)) == "A":
            a_ok += 1
        # (b) shuffled: move correct to a random position
        order = list(range(len(ch))); random.Random(qi).shuffle(order)
        sch = [ch[o] for o in order]; new_correct = order.index(0)
        if gen_letter(m, tk, letter_prompt(q["question"], sch)) == chr(65 + new_correct):
            b_ok += 1
        # (c) likelihood MC1 (position-free): argmax over choice TEXT
        prompt = f"Q: {q['question']}\nA:"
        tots = [loglik(m, tk, prompt, c) for c in ch]
        if int(np.argmax([t[0] for t in tots])) == 0:
            c_tot += 1
        if int(np.argmax([t[1] for t in tots])) == 0:
            c_norm += 1
    print(f"  (a) unshuffled letter-gen : {a_ok/n*100:5.1f}%   <- the pipeline's metric (correct always A)")
    print(f"  (b) SHUFFLED letter-gen   : {b_ok/n*100:5.1f}%   <- if << (a), it was POSITION BIAS")
    print(f"  (c) likelihood MC1 (total): {c_tot/n*100:5.1f}%   <- real truthfulness (position-free)")
    print(f"      likelihood MC1 (norm) : {c_norm/n*100:5.1f}%", flush=True)
    del m; torch.cuda.empty_cache()
    return n


print(f"N={N}  random baseline ~ {np.mean([1/max(2,len(q['mc1_targets']['choices'])) for q in tqa])*100:.1f}%")
for tag, p in CKPTS:
    run(tag, p)
print("\nDONE")
