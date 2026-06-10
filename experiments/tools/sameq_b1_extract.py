"""Teacher-forced prefill re-extraction of L/2 token trajectories for the
within-question beta1 test. Reads C3_sameq_v2.json (must contain 'ids'),
processes MIXED questions only (both correct & incorrect present), and writes
one .npz per question to results/sameq_acts/ (restartable per question).

Validity: for a causal LM, hidden states at the generated-token positions under
prefill are identical to those during decoding of the same tokens (verified in
the villain-probe prefill check: cos-dist ~0.0000 on shared prefixes).
"""
import os, json, time
from pathlib import Path
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3.5-2B"
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "C3_sameq_v2.json"
DST = ROOT / "results" / "sameq_acts"
DST.mkdir(parents=True, exist_ok=True)

res = json.load(open(SRC))
mixed = {k: v for k, v in res.items() if 0 < v["n_correct"] < v["K"]}
print(f"{len(mixed)} mixed questions / {len(res)} total", flush=True)

m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True).cuda().eval()
tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tk.pad_token is None: tk.pad_token = tk.eos_token
layer = len(m.model.layers) // 2
gsm = load_dataset("openai/gsm8k", "main", split="test")

t0 = time.time()
for key, v in sorted(mixed.items(), key=lambda x: int(x[0])):
    out_path = DST / f"q{key}.npz"
    if out_path.exists():
        continue
    q = gsm[v["qid"]]["question"]
    prompt = ("Please solve the following math problem step by step, and give "
              "the final numerical answer at the very end after '####'.\n\n" + q)
    enc = tk.apply_chat_template([[{"role": "user", "content": prompt}]],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
    pids = enc["input_ids"][0].cuda()
    assert len(pids) == v["plen"], f"plen mismatch q{key}: {len(pids)} vs {v['plen']}"
    arrs = {}
    for si, c in enumerate(v["comps"]):
        if not c.get("ids"):
            continue
        gids = torch.tensor(c["ids"], dtype=pids.dtype, device="cuda")
        seq = torch.cat([pids, gids]).unsqueeze(0)
        with torch.no_grad():
            out = m(seq, output_hidden_states=True)
        # states AT the generated-token positions (plen .. plen+n_gen-1)
        h = out.hidden_states[layer][0, len(pids):].float().cpu().numpy().astype(np.float16)
        arrs[f"s{si}"] = h
        arrs[f"s{si}_correct"] = np.array(c["correct"])
    np.savez_compressed(out_path, **arrs)
    print(f"q{key}: {len([a for a in arrs if not a.endswith('_correct')])} traj "
          f"({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
