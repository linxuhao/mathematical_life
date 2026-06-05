#!/usr/bin/env python3
"""Tool: Extract activations split by GSM8K correctness (correct vs incorrect).
Enables the Tripartite Test: R_true vs R_flawed topological comparison.

Usage (on server with GPU):
  python tools/extract_split_gsm8k.py \
      --model Qwen/Qwen3.5-2B-Inst \
      --output activations_v3/ \
      --max-prompts 1319

Output:
  activations_v3/{Model}_L{layer}_reasoning_correct.npy
  activations_v3/{Model}_L{layer}_reasoning_incorrect.npy
  activations_v3/{Model}_L{layer}_reasoning_labels.json  (per-prompt correctness)
"""
import os, json, argparse, re, time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_final_answer(text):
    """Extract the final numeric answer from GSM8K model output.
    Looks for patterns like '#### 42' or 'The answer is 42'."""
    # Pattern 1: GSM8K standard format
    m = re.search(r'####\s*(-?\d+[\.,]?\d*)', text)
    if m:
        return m.group(1).replace(',', '').replace('.0', '')
    # Pattern 2: last number in the text
    numbers = re.findall(r'-?\d+[\.,]?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def check_correct(model_output, ground_truth):
    """Check if model's final answer matches ground truth."""
    pred = extract_final_answer(model_output)
    truth = str(ground_truth).strip()
    if pred is None:
        return False
    # Normalize: remove trailing .0
    pred = pred.rstrip('0').rstrip('.') if '.' in pred else pred
    truth = truth.rstrip('0').rstrip('.') if '.' in truth else truth
    return pred == truth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='HF model name')
    parser.add_argument('--output', default='activations_v3', help='output directory')
    parser.add_argument('--max-prompts', type=int, default=1319, help='max prompts to process')
    parser.add_argument('--layer', type=int, default=None, help='layer to extract (default: L/2)')
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    model_name = args.model
    short_name = model_name.replace('/', '__')

    # Load prompts
    prompts_path = Path(__file__).parent.parent / 'prompts' / 'reasoning.json'
    with open(prompts_path) as f:
        prompts_data = json.load(f)
    prompts_data = prompts_data[:args.max_prompts]

    print(f"Model: {model_name}")
    print(f"Prompts: {len(prompts_data)}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda().eval()

    n_layers = len(model.model.layers)
    layer = args.layer if args.layer is not None else n_layers // 2
    print(f"Layers: {n_layers}, extracting at L={layer}")

    is_instruct = 'inst' in model_name.lower() or 'instruct' in model_name.lower()

    # Load GSM8K answers from HF for correctness checking
    from datasets import load_dataset
    gsm8k = load_dataset('openai/gsm8k', 'main', split='test')
    # Build answer lookup by question text (first 1319 match our prompts)
    answer_map = {}
    for item in gsm8k:
        q = item['question'].strip()
        a = item['answer'].strip()
        # Extract final numeric answer from GSM8K answer format
        m = re.search(r'####\s*(-?\d+[\.,]?\d*)', a)
        if m:
            answer_map[q] = m.group(1).replace(',', '').replace('.0', '')
    print(f"Loaded {len(answer_map)} GSM8K answers")

    correct_acts = []
    incorrect_acts = []
    labels = []
    n_correct = 0
    n_incorrect = 0

    t0 = time.time()
    for i, item in enumerate(prompts_data):
        prompt = item['question']
        answer = answer_map.get(prompt.strip(), '')

        # Format for model
        if is_instruct:
            msgs = [{'role': 'user', 'content': prompt}]
            inputs = tokenizer.apply_chat_template(
                msgs, return_tensors='pt', truncation=True,
                max_length=args.max_length
            )
        else:
            inputs = tokenizer(prompt, return_tensors='pt',
                              truncation=True, max_length=args.max_length)

        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer][:, -1, :].cpu().float().numpy()[0]

            # Generate answer
            gen = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        is_correct = check_correct(response, str(answer))

        if is_correct:
            correct_acts.append(hidden)
            n_correct += 1
        else:
            incorrect_acts.append(hidden)
            n_incorrect += 1

        labels.append({'idx': i, 'correct': is_correct, 'answer': str(answer)})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts_data) - i - 1) / rate
            print(f"  [{i+1}/{len(prompts_data)}] correct={n_correct} incorrect={n_incorrect} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Correct: {n_correct}, Incorrect: {n_incorrect}")
    print(f"Accuracy: {n_correct/(n_correct+n_incorrect)*100:.1f}%")

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if correct_acts:
        correct_arr = np.stack(correct_acts, axis=0)
        fname = f'{short_name}_L{layer}_reasoning_correct.npy'
        np.save(out_dir / fname, correct_arr)
        print(f"Saved: {fname} shape={correct_arr.shape}")

    if incorrect_acts:
        incorrect_arr = np.stack(incorrect_acts, axis=0)
        fname = f'{short_name}_L{layer}_reasoning_incorrect.npy'
        np.save(out_dir / fname, incorrect_arr)
        print(f"Saved: {fname} shape={incorrect_arr.shape}")

    labels_fname = f'{short_name}_L{layer}_reasoning_labels.json'
    with open(out_dir / labels_fname, 'w') as f:
        json.dump({'model': model_name, 'layer': layer,
                   'n_correct': n_correct, 'n_incorrect': n_incorrect,
                   'labels': labels}, f, indent=2)
    print(f"Saved: {labels_fname}")

    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
