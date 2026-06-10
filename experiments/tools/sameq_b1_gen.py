"""Same-question regeneration v2 — for the within-question cumulative-beta1 test
(the direct adjudication of the precursor hypothesis left open in the paper).

Protocol IDENTICAL to C3_sameq_gen.py (Qwen3.5-2B chat, 150 Qs x K=8, temp 0.8,
top_p 0.95, max_new 1024) with ONE addition: generated token ids are saved per
completion, so trajectories can be re-extracted by teacher-forced prefill.
Also keeps dec_act + s_pred so the B14 endpoint-probe anchor (AUC ~0.58) can be
replicated on this fresh batch. Resumable.

PRE-REGISTERED (2026-06-10, before any data):
  primary endpoint = within-question-centered GroupKFold AUC of FILTERED
  cumulative beta1 at matched k in {128,192,256};
  verdict: >=0.60 sustained across k -> precursor signal real;
  <0.55 -> precursor excluded; else inconclusive, report interval.
"""
import os, sys, json, re, time
from pathlib import Path
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# MAXNEW=2048 (not B14's 1024): the old batch hit the 1024 budget on 40% of all
# completions and 59% of mixed-question completions (temp-0.8 sampling rambles),
# so its labels were partly truncation-driven. N_Q=200 for mixed-count buffer;
# resumable — can stop once mixed questions >= ~80.
MODEL = "Qwen/Qwen3.5-2B"; N_Q = 200; K = 8; TEMP = 0.8; MAXNEW = 2048
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "C3_sameq_v2.json"

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
    tmp = str(OUT) + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, str(OUT))

def ans_pos(gen_ids):
    toks = [tk.decode([int(t)]) for t in gen_ids]
    acc = ""; hash_at = -1
    for j, t in enumerate(toks):
        acc += t
        if "####" in acc and hash_at < 0:
            hash_at = j
        if hash_at >= 0 and j >= hash_at and re.search(r"\d", t):
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
    enc = tk.apply_chat_template([[{"role": "user", "content": prompt}]],
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
        resp = tk.decode(gen_ids, skip_special_tokens=True)
        pred = extract_ans(resp); correct = (pred is not None and pred == gold)
        jpos, anstoks = ans_pos(gen_ids)
        n_gen = int((gen_ids != tk.pad_token_id).sum())
        rec = {"correct": bool(correct), "pred": pred, "n_gen": n_gen,
               "ids": gen_ids[:n_gen].cpu().tolist()}        # <- the v2 addition
        if jpos is not None and anstoks:
            p_ans = plen + jpos
            with torch.no_grad():
                out = m(seq[:p_ans + len(anstoks)].unsqueeze(0), output_hidden_states=True)
            dec = out.hidden_states[layer][0, p_ans - 1]
            logits = out.logits[0].float()
            lp = F.log_softmax(logits, -1)
            s_pred = float(np.mean([-lp[p_ans - 1 + t, anstoks[t]].item()
                                    for t in range(len(anstoks))]))
            rec["dec_act"] = dec.float().cpu().numpy().round(4).tolist()
            rec["s_pred"] = s_pred
            rec["status"] = "ok"
        else:
            rec["status"] = "no_ans"
        comps.append(rec)
    nc = sum(c["correct"] for c in comps)
    res[key] = {"qid": qi, "gold": gold, "n_correct": nc, "K": K, "comps": comps,
                "plen": plen}
    did += 1
    if did % 5 == 0:
        save()
        mixed = sum(1 for v in res.values() if 0 < v["n_correct"] < v["K"])
        print(f"  q{qi} done ({len(res)} total, {mixed} mixed) nc={nc}/{K} "
              f"{time.time()-t0:.0f}s", flush=True)
save()
mixed = sum(1 for v in res.values() if 0 < v["n_correct"] < v["K"])
print(f"DONE: {len(res)} questions, {mixed} mixed ({time.time()-t0:.0f}s)", flush=True)
