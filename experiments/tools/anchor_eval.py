#!/usr/bin/env python3
"""Behavioral anchors for SFT checkpoints: TruthfulQA (hallucination armor) + SVAMP (OOD collapse).
Usage:
  python tools/anchor_eval.py --checkpoints checkpoints/sft_gsm8k_subset/ -o results/anchors.json
"""
import argparse, re, torch, json, random
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_answer(text):
    """Extract final answer for math questions."""
    m = re.search(r'####\s*(-?\d+[\.,]?\d*)', text)
    if m: return m.group(1).replace(',','').replace('.0','')
    nums = re.findall(r'-?\d+[\.,]?\d*', text)
    return nums[-1].replace(',','') if nums else None

def eval_svamp(model, tokenizer, n_q=200):
    """SVAMP: OOD math — varied question phrasing with distractors."""
    ds = load_dataset('ChilleD/SVAMP', split='test')  # try this, fallback if needed
    indices = random.sample(range(len(ds)), min(n_q, len(ds)))
    correct = 0
    for i in indices:
        q = ds[i]['Body'] + ' ' + ds[i]['Question']
        a = str(ds[i]['Answer'])
        tok = tokenizer(q, return_tensors='pt', truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            gen = model.generate(**tok, max_new_tokens=128, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(gen[0][tok['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(resp)
        if pred and pred == a: correct += 1
    return correct / n_q * 100

def eval_truthfulqa(model, tokenizer, n_q=200):
    """TruthfulQA: MC1 — can model resist common misconceptions?"""
    try:
        ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
    except:
        print("  TruthfulQA not available, skipping")
        return None
    indices = random.sample(range(len(ds)), min(n_q, len(ds)))
    correct = 0
    for i in indices:
        item = ds[i]
        q = item['question']
        choices = item['mc1_targets']['choices']
        labels = item['mc1_targets']['labels']
        # Find correct answer index
        correct_idx = [j for j, l in enumerate(labels) if l == 1][0] if 1 in labels else 0
        # Simple: ask model to pick
        prompt = f"Q: {q}\n"
        for j, c in enumerate(choices):
            prompt += f"{chr(65+j)}. {c}\n"
        prompt += "Answer:"
        tok = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            gen = model.generate(**tok, max_new_tokens=5, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(gen[0][tok['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        # Check if answer letter matches
        if resp and resp[0].upper() == chr(65+correct_idx):
            correct += 1
    return correct / n_q * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--n-eval', type=int, default=200)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints)
    ckpts = sorted(ckpt_dir.glob('checkpoint-*'))
    print(f"Checkpoints: {len(ckpts)}")

    results = []
    for ckpt_path in ckpts:
        step = int(ckpt_path.name.split('-')[1])
        print(f"\n=== Step {step} ===")

        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path), torch_dtype=torch.float16,
            attn_implementation='eager', trust_remote_code=True
        ).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        svamp = eval_svamp(model, tokenizer, args.n_eval)
        tqa = eval_truthfulqa(model, tokenizer, args.n_eval)

        r = {'step': step, 'svamp_acc': round(svamp, 1) if svamp else None,
             'truthfulqa_acc': round(tqa, 1) if tqa else None}
        results.append(r)
        print(f"  SVAMP={svamp:.1f}%  TruthfulQA={tqa}" if svamp and tqa else f"  SVAMP={svamp}")

        del model; torch.cuda.empty_cache()

    # Add base model
    print("\n=== Step 0 (base) ===")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    svamp0 = eval_svamp(model, tokenizer, args.n_eval)
    tqa0 = eval_truthfulqa(model, tokenizer, args.n_eval)
    results.insert(0, {'step': 0, 'svamp_acc': round(svamp0, 1) if svamp0 else None,
                        'truthfulqa_acc': round(tqa0, 1) if tqa0 else None})
    print(f"  SVAMP={svamp0:.1f}%  TruthfulQA={tqa0}")
    del model; torch.cuda.empty_cache()

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

if __name__ == '__main__':
    main()
