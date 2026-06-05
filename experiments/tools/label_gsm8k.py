#!/usr/bin/env python3
"""Fast labeler: generate GSM8K answers, match against ground truth,
output per-prompt correctness labels for splitting existing activations.

Usage:
  python tools/label_gsm8k.py \
      --model Qwen/Qwen3.5-2B \
      --output activations_v3/Qwen__Qwen3.5-2B_L14_reasoning_labels.json

Then split existing .npy:
  python -c "
  import json, numpy as np
  labels = json.load(open('labels.json'))
  acts = np.load('reasoning.npy')
  correct = acts[labels['correct_idx']]
  incorrect = acts[labels['incorrect_idx']]
  np.save('reasoning_correct.npy', correct)
  np.save('reasoning_incorrect.npy', incorrect)
  "
"""
import json, re, time, argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def extract_answer(text):
    m = re.search(r'####\s*(-?\d+[\.,]?\d*)', text)
    if m: return m.group(1).replace(',','').replace('.0','')
    nums = re.findall(r'-?\d+[\.,]?\d*', text)
    return nums[-1].replace(',','') if nums else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-prompts', type=int, default=1319)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    # Load prompts
    prompts_path = Path(__file__).parent.parent / 'prompts' / 'reasoning.json'
    with open(prompts_path) as f:
        prompts_data = json.load(f)
    prompts_data = prompts_data[:args.max_prompts]

    # Load ground truth answers
    gsm8k = load_dataset('openai/gsm8k', 'main', split='test')
    answer_map = {}
    for item in gsm8k:
        m = re.search(r'####\s*(-?\d+[\.,]?\d*)', item['answer'])
        if m:
            answer_map[item['question'].strip()] = m.group(1).replace(',','').replace('.0','')

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda().eval()

    is_inst = 'inst' in args.model.lower() or 'instruct' in args.model.lower()

    correct_idx = []
    incorrect_idx = []
    labels = []

    t0 = time.time()
    for i in range(0, len(prompts_data), args.batch_size):
        batch = prompts_data[i:i+args.batch_size]
        questions = [p['question'] for p in batch]

        if is_inst:
            msgs = [[{'role': 'user', 'content': q}] for q in questions]
            tok = tokenizer.apply_chat_template(msgs, return_tensors='pt', padding=True,
                                               truncation=True, max_length=1024, return_dict=True)
        else:
            tok = tokenizer(questions, return_tensors='pt', padding=True,
                          truncation=True, max_length=1024)
        inputs = {k: v.cuda() for k, v in tok.items()}

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)

        for j, q in enumerate(questions):
            response = tokenizer.decode(gen[j][inputs['input_ids'].shape[1]:],
                                       skip_special_tokens=True)
            pred = extract_answer(response)
            truth = answer_map.get(q.strip(), '')
            is_correct = (pred == truth) if pred else False

            if is_correct:
                correct_idx.append(i + j)
            else:
                incorrect_idx.append(i + j)
            labels.append(is_correct)

        if (i + len(batch)) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + len(batch)) / elapsed * 3600
            print(f"  [{i+len(batch)}/{len(prompts_data)}] correct={len(correct_idx)} "
                  f"incorrect={len(incorrect_idx)} ({rate:.0f}/hr)")

    elapsed = time.time() - t0
    acc = len(correct_idx) / len(labels) * 100
    print(f"\nDone in {elapsed:.0f}s. Acc={acc:.1f}% ({len(correct_idx)}/{len(labels)})")

    out = {
        'model': args.model,
        'correct_idx': correct_idx,
        'incorrect_idx': incorrect_idx,
        'labels': labels,
        'accuracy': round(acc, 2),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f)
    print(f"Saved: {args.output}")

    del model; torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
