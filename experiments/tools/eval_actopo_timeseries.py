#!/usr/bin/env python3
"""Re-evaluate an SFT checkpoint series with actopo (correct extraction).

Fixes vs the old inline eval:
  - INSTRUCT models use their native chat template for extraction AND generation
    (matches the v3 protocol; the old run used raw text → β₁ not comparable).
  - Adds per-checkpoint OOD anchors (SVAMP) + TruthfulQA, so the topology gauge
    can be tested against generalization, not just the training target.
  - Writes results INCREMENTALLY and ATOMICALLY after every checkpoint, and is
    RESUMABLE (skips steps already in the output) — survives crashes / SSH drops.

Topology is actopo FROZEN_V5 (ε=0.03, survival-ε=0.01, mask-based last token).
"""
import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# actopo (installed in the env, or add the src path)
sys.path.insert(0, str(Path.home() / "papers/mathematical-life/actopo/src"))
from actopo.protocol import FROZEN_V5
from actopo.topology import measure, phi as compute_phi
from actopo.extract import extract_activations

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------------------------------------------------------- #
def find_decoder_layers(model):
    """Return the transformer block list across common (incl. multimodal) layouts."""
    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers", "model.text_model.layers",
                 "transformer.h"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise RuntimeError("could not locate decoder layers")


def load_ckpt(path, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation="eager", trust_remote_code=True
    ).cuda().eval()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


def extract_answer(text):
    m = re.search(r"####\s*(-?\d+[\.,]?\d*)", text)
    if m:
        return m.group(1).replace(",", "").replace(".0", "")
    nums = re.findall(r"-?\d+[\.,]?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def _encode(tok, texts, instruct, max_length=1024):
    """Chat template for instruct (with generation prompt), raw text for base."""
    if instruct and getattr(tok, "chat_template", None):
        msgs = [[{"role": "user", "content": t}] for t in texts]
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, padding=True, truncation=True, max_length=max_length,
        )
    return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)


def gen_eval(model, tok, questions, answers, instruct, max_new=200, bs=8):
    correct = 0
    for i in range(0, len(questions), bs):
        bq, ba = questions[i:i+bs], answers[i:i+bs]
        inp = {k: v.cuda() for k, v in _encode(tok, bq, instruct).items()}
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        for j in range(len(bq)):
            resp = tok.decode(g[j][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            if extract_answer(resp) == ba[j]:
                correct += 1
    return round(correct / len(questions) * 100, 1)


def eval_truthfulqa(model, tok, mc, instruct, n, bs=8):
    correct = 0
    items = mc[:n]
    for i in range(0, len(items), bs):
        batch = items[i:i+bs]
        prompts, idxs = [], []
        for it in batch:
            ch = it["mc1_targets"]["choices"]; lab = it["mc1_targets"]["labels"]
            ci = lab.index(1) if 1 in lab else 0
            letters = "\n".join(f"{chr(65+k)}. {c}" for k, c in enumerate(ch))
            prompts.append(f"Q: {it['question']}\n{letters}\nAnswer with the letter:")
            idxs.append(ci)
        inp = {k: v.cuda() for k, v in _encode(tok, prompts, instruct).items()}
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=5, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        for j in range(len(batch)):
            resp = tok.decode(g[j][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if resp and resp[0].upper() == chr(65 + idxs[j]):
                correct += 1
    return round(correct / len(items) * 100, 1)


def atomic_dump(obj, path):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--instruct", action="store_true", help="use chat template")
    ap.add_argument("--n-prompts", type=int, default=1319)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="weight dtype; use bfloat16 on ROCm for small models (fp16 -> NaN)")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    base = Path("/home/linxuhao/papers/mathematical-life/experiments")
    pdir = base / "prompts"
    r_prompts = [p["question"] for p in json.load(open(pdir / "reasoning.json"))][:args.n_prompts]
    h_prompts = json.load(open(pdir / "gsm8k_hallucination.json"))["prompts"][:args.n_prompts]

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    g_q = [gsm[i]["question"] for i in range(args.n_eval)]
    g_a = [extract_answer(gsm[i]["answer"]) for i in range(args.n_eval)]
    try:
        svamp = load_dataset("ChilleD/SVAMP", split="test")
        s_q = [f"{svamp[i]['Body']} {svamp[i]['Question']}" for i in range(min(args.n_eval, len(svamp)))]
        s_a = [str(int(svamp[i]["Answer"])) for i in range(min(args.n_eval, len(svamp)))]
    except Exception as e:
        print(f"SVAMP unavailable: {e}"); s_q = s_a = []
    try:
        tqa = list(load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation"))
    except Exception as e:
        print(f"TruthfulQA unavailable: {e}"); tqa = []

    cfg = FROZEN_V5.with_(use_chat_template=args.instruct)

    # ---- resume ----
    out_path = Path(args.output)
    results = json.load(open(out_path)) if out_path.exists() else []
    done = {r["step"] for r in results}
    print(f"Resuming: {len(done)} checkpoints already done")

    # ---- checkpoint list: step 0 = base model, then checkpoint-* ----
    ckpts = [(0, args.base_model)]
    for d in Path(args.checkpoints).glob("checkpoint-*"):
        ckpts.append((int(d.name.split("-")[1]), str(d)))
    ckpts.sort()

    for step, path in ckpts:
        if step in done:
            continue
        print(f"\n=== step {step} ({path}) {time.strftime('%H:%M:%S')} ===", flush=True)
        t0 = time.time()
        try:
            model, tok = load_ckpt(path, dtype=dtype)
            layer = len(find_decoder_layers(model)) // 2
            acts_r = extract_activations(model, tok, r_prompts, cfg, layer=layer)
            acts_h = extract_activations(model, tok, h_prompts, cfg, layer=layer)
            tr, th = measure(acts_r, cfg), measure(acts_h, cfg)
            rec = {
                "step": step, "beta1_r": tr.beta1, "beta1_h": th.beta1,
                "raw_r": tr.beta1_raw, "raw_h": th.beta1_raw,
                "surv_r": tr.survival_rate, "surv_h": th.survival_rate,
                "phi": compute_phi(acts_r, acts_h, cfg),
                "gsm8k_acc": gen_eval(model, tok, g_q, g_a, args.instruct),
                "svamp_acc": gen_eval(model, tok, s_q, s_a, args.instruct, max_new=100) if s_q else None,
                "truthfulqa_acc": eval_truthfulqa(model, tok, tqa, args.instruct, args.n_eval) if tqa else None,
                "layer": layer, "format": "chat" if args.instruct else "raw",
            }
            results.append(rec)
            results.sort(key=lambda r: r["step"])
            atomic_dump(results, out_path)
            print(f"  β₁_R={rec['beta1_r']} β₁_H={rec['beta1_h']} gap={rec['beta1_r']-rec['beta1_h']} "
                  f"PHI={rec['phi']} GSM8K={rec['gsm8k_acc']}% SVAMP={rec['svamp_acc']}% "
                  f"TQA={rec['truthfulqa_acc']}% ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  step {step} FAILED: {e}", flush=True)
        finally:
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()

    print(f"\nDONE: {len(results)} checkpoints → {out_path}")


if __name__ == "__main__":
    main()
