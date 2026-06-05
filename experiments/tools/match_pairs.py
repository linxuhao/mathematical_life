import json, numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

tk = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
gsm = load_dataset("openai/gsm8k", "main", split="test")
labels = {r["idx"]: r for r in json.load(open("activations_v3/tokenwise_v2/labels.json"))}

rows = []
for idx, lab in labels.items():
    if lab.get("capped"):
        continue
    q = gsm[idx]["question"]
    prompt = ("Please solve the following math problem step by step, and give "
              "the final numerical answer at the very end after '####'.\n\n" + q)
    enc = tk.apply_chat_template([[{"role": "user", "content": prompt}]],
                                 add_generation_prompt=True, return_tensors="pt", return_dict=True)
    plen = enc["input_ids"].shape[1]
    rows.append({"idx": idx, "correct": lab["correct"], "in": int(plen),
                 "out": int(lab["n_gen"]), "total": int(plen + lab["n_gen"]), "beta1": lab["beta1"]})

C = [r for r in rows if r["correct"]]
I = [r for r in rows if not r["correct"]]
print(f"n_correct={len(C)} n_incorrect={len(I)}")
print(f"input  tok: C mean={np.mean([r['in'] for r in C]):.0f}  I mean={np.mean([r['in'] for r in I]):.0f}")
print(f"output tok: C mean={np.mean([r['out'] for r in C]):.0f}  I mean={np.mean([r['out'] for r in I]):.0f}")
print(f"total  tok: C mean={np.mean([r['total'] for r in C]):.0f}  I mean={np.mean([r['total'] for r in I]):.0f}")

pairs = []
for ic in I:
    for cc in C:
        dt = abs(cc["total"] - ic["total"]); do = abs(cc["out"] - ic["out"])
        pairs.append((dt + do, dt, do, ic, cc))
pairs.sort(key=lambda x: x[0])

print("\nBEST-MATCHED correct/incorrect pairs (by |dtotal|+|doutput|, min total 250):")
seen_i, seen_c, shown = set(), set(), 0
for score, dt, do, ic, cc in pairs:
    if ic["total"] < 250 or cc["total"] < 250:
        continue
    if ic["idx"] in seen_i or cc["idx"] in seen_c:
        continue
    seen_i.add(ic["idx"]); seen_c.add(cc["idx"])
    print(f"  INC idx={ic['idx']:3d} in={ic['in']} out={ic['out']} total={ic['total']} b1={ic['beta1']}  |  "
          f"COR idx={cc['idx']:3d} in={cc['in']} out={cc['out']} total={cc['total']} b1={cc['beta1']}  (dtot={dt} dout={do})")
    shown += 1
    if shown >= 12:
        break
