"""C3 control #3: same-question multiple completions (the clean test).
Sample K completions/question at temp>0. Per completion record: correctness,
decision-token L/2 activation (varies with reasoning), s_pred surprisal. The
prompt-end activation is identical across a question's completions, so
within-question variation isolates the REASONING contribution — free of the
question-difficulty confound. Resumable.
"""
import os, sys, json, re, time
from pathlib import Path
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3.5-2B"; N_Q = 150; K = 8; TEMP = 0.8; MAXNEW = 1024
OUT = Path("/home/linxuhao/papers/mathematical-life/experiments/results/C3_sameq.json")

def extract_ans(t):
    m = re.search(r"####\s*(-?\d+[\.,]?\d*)", t)
    if m: return m.group(1).replace(",", "").replace(".0", "")
    nums = re.findall(r"-?\d+[\.,]?\d*", t)
    return nums[-1].replace(",", "").replace(".0", "") if nums else None

m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True).cuda().eval()
tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tk.pad_token is None: tk.pad_token = tk.eos_token
layer = len(m.model.layers) // 2
gsm = load_dataset("openai/gsm8k", "main", split="test")

res = json.load(open(OUT)) if OUT.exists() else {}
def save():
    tmp = str(OUT)+".tmp"; json.dump(res, open(tmp,"w")); os.replace(tmp, str(OUT))

def ans_pos(gen_ids):
    """token index (in gen space) of the first answer digit after last '####', + the answer token ids."""
    toks = [tk.decode([int(t)]) for t in gen_ids]
    acc = ""; hash_at = -1
    for j, t in enumerate(toks):
        acc += t
        if "####" in acc and hash_at < 0:
            hash_at = j
        if hash_at >= 0 and j >= hash_at and re.search(r"\d", t):
            # first digit-bearing token at/after ####
            k = j
            anstoks = []
            while k < len(toks) and re.search(r"[\d\.,\-]", toks[k]):
                anstoks.append(int(gen_ids[k])); k += 1
            return j, anstoks
    return None, None

t0 = time.time(); did = 0
for qi in range(N_Q):
    key = str(qi)
    if key in res:
        continue
    q = gsm[qi]["question"]; gold = extract_ans(gsm[qi]["answer"])
    prompt = ("Please solve the following math problem step by step, and give "
              "the final numerical answer at the very end after '####'.\n\n" + q)
    enc = tk.apply_chat_template([[{"role":"user","content":prompt}]],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k_: v.cuda() for k_, v in enc.items()}
    plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        gen = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP,
                         top_p=0.95, num_return_sequences=K, pad_token_id=tk.eos_token_id)
    comps = []
    for s in range(K):
        seq = gen[s]
        gen_ids = seq[plen:]
        # strip right pad (eos) for clean decode/position
        resp = tk.decode(gen_ids, skip_special_tokens=True)
        pred = extract_ans(resp); correct = (pred is not None and pred == gold)
        jpos, anstoks = ans_pos(gen_ids)
        rec = {"correct": bool(correct), "pred": pred, "n_gen": int((gen_ids != tk.pad_token_id).sum())}
        if jpos is not None and anstoks:
            p_ans = plen + jpos
            with torch.no_grad():
                out = m(seq[:p_ans+len(anstoks)].unsqueeze(0), output_hidden_states=True)
            dec = out.hidden_states[layer][0, p_ans-1]           # decision token (just before answer)
            logits = out.logits[0].float()
            lp = F.log_softmax(logits, -1)
            s_pred = float(np.mean([-lp[p_ans-1+t, anstoks[t]].item() for t in range(len(anstoks))]))
            rec["dec_act"] = dec.float().cpu().numpy().round(4).tolist()
            rec["s_pred"] = s_pred
            rec["status"] = "ok"
        else:
            rec["status"] = "no_ans"
        comps.append(rec)
    nc = sum(c["correct"] for c in comps)
    res[key] = {"qid": qi, "gold": gold, "n_correct": nc, "K": K, "comps": comps}
    did += 1
    if did % 5 == 0:
        save()
        mixed = sum(1 for v in res.values() if 0 < v["n_correct"] < v["K"])
        print(f"  q{qi} done ({len(res)} total, {mixed} mixed) nc={nc}/{K} {time.time()-t0:.0f}s", flush=True)
save()
mixed = sum(1 for v in res.values() if 0 < v["n_correct"] < v["K"])
print(f"DONE: {len(res)} questions, {mixed} mixed (both ✓&✗) ({time.time()-t0:.0f}s)", flush=True)
