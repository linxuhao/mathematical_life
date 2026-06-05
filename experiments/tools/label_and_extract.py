#!/usr/bin/env python3
"""Labeler + Tokenwise Extractor: one GPU pass does both correctness labels
AND per-token activation extraction for every prompt.

Usage:
  python tools/label_and_extract.py \
      --model Qwen/Qwen3.5-2B \
      --output activations_v3/ \
      --max-prompts 1319 --batch-size 4

Output:
  activations_v3/{model}_L{layer}_tokenwise.npz    (all per-token activations)
  activations_v3/{model}_L{layer}_tokenwise_labels.json  (correctness labels)
"""
import json, re, time, argparse
from pathlib import Path
import torch
import numpy as np
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
    parser.add_argument('--output', default='activations_v3')
    parser.add_argument('--max-prompts', type=int, default=1319)
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=4,
                       help='smaller batch = less GPU memory, more overhead')
    args = parser.parse_args()

    short_name = args.model.replace('/', '__')

    # Load prompts
    prompts_path = Path(__file__).parent.parent / 'prompts' / 'reasoning.json'
    with open(prompts_path) as f:
        prompts_data = json.load(f)
    prompts_data = prompts_data[:args.max_prompts]

    # Load answers
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
    tokenizer.padding_side = 'left'  # important for generation with padding

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda().eval()

    n_layers = len(model.model.layers)
    layer = args.layer if args.layer is not None else n_layers // 2
    is_inst = 'inst' in args.model.lower() or 'instruct' in args.model.lower()

    print(f"Model: {args.model}, L={layer}, Batch={args.batch_size}")
    print(f"Prompts: {len(prompts_data)}")

    # Storage: one array per prompt (variable-length tokens)
    all_acts = []       # list of (n_tokens, hidden_dim) arrays
    all_labels = []     # bool per prompt
    correct_idx = []
    incorrect_idx = []

    t0 = time.time()
    for i in range(0, len(prompts_data), args.batch_size):
        batch = prompts_data[i:i+args.batch_size]
        questions = [p['question'] for p in batch]
        truths = [answer_map.get(q.strip(), '') for q in questions]

        if is_inst:
            msgs = [[{'role': 'user', 'content': q}] for q in questions]
            tok = tokenizer.apply_chat_template(msgs, return_tensors='pt', padding=True,
                                               truncation=True, max_length=1024, return_dict=True)
        else:
            tok = tokenizer(questions, return_tensors='pt', padding=True,
                          truncation=True, max_length=1024)
        inputs = {k: v.cuda() for k, v in tok.items()}

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                output_hidden_states=True, return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # gen.hidden_states: list of tuples, one per generation step
        # gen.hidden_states[t][layer_idx] = tensor (batch, 1, hidden_dim)
        n_gen_steps = len(gen.hidden_states)
        batch_size_actual = len(questions)

        # For each prompt in the batch, collect its token-by-token activations
        for j in range(batch_size_actual):
            acts_j = []
            for t in range(n_gen_steps):
                h = gen.hidden_states[t][layer][j, 0, :].cpu().float().numpy()
                acts_j.append(h)

            # Stop at EOS if generated
            seq = gen.sequences[j]
            input_len = (inputs['attention_mask'][j] == 1).sum().item() if 'attention_mask' in inputs else inputs['input_ids'].shape[1]
            gen_ids = seq[input_len:]
            eos_positions = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                eos_idx = eos_positions[0].item()
                acts_j = acts_j[:eos_idx]  # truncate at first EOS
                gen_ids = gen_ids[:eos_idx]

            acts_arr = np.stack(acts_j, axis=0) if acts_j else np.zeros((0, model.config.hidden_size), dtype=np.float32)
            all_acts.append(acts_arr)

            # Check correctness
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = extract_answer(response)
            is_correct = (pred == truths[j]) if pred else False
            all_labels.append(is_correct)

            if is_correct:
                correct_idx.append(i + j)
            else:
                incorrect_idx.append(i + j)

        done = min(i + args.batch_size, len(prompts_data))
        if done % 50 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed * 3600
            print(f"  [{done}/{len(prompts_data)}] correct={len(correct_idx)} "
                  f"incorrect={len(incorrect_idx)} ({rate:.0f}/hr) "
                  f"avg_tokens={np.mean([a.shape[0] for a in all_acts[-50:]]):.0f}")

    elapsed = time.time() - t0
    acc = len(correct_idx) / len(all_labels) * 100
    total_tokens = sum(a.shape[0] for a in all_acts)
    print(f"\nDone in {elapsed:.0f}s. Acc={acc:.1f}% ({len(correct_idx)}/{len(all_labels)})")
    print(f"Total tokens across all prompts: {total_tokens}")

    # Save per-prompt activations as .npz (flattened + lengths for reconstruction)
    lens = np.array([a.shape[0] for a in all_acts], dtype=np.int32)
    acts_flat = np.concatenate(all_acts, axis=0)
    hidden_dim = acts_flat.shape[1]

    npz_path = Path(args.output) / f'{short_name}_L{layer}_tokenwise.npz'
    np.savez_compressed(npz_path,
                        acts=acts_flat, lens=lens,
                        correct_idx=np.array(correct_idx, dtype=np.int32),
                        incorrect_idx=np.array(incorrect_idx, dtype=np.int32))
    print(f"Saved: {npz_path} ({acts_flat.shape[0]} tokens, {hidden_dim} dim)")

    # Save labels
    labels_path = Path(args.output) / f'{short_name}_L{layer}_tokenwise_labels.json'
    with open(labels_path, 'w') as f:
        json.dump({'model': args.model, 'layer': layer,
                   'n_prompts': len(all_labels),
                   'n_correct': len(correct_idx), 'n_incorrect': len(incorrect_idx),
                   'accuracy': round(acc, 2),
                   'correct_idx': correct_idx, 'incorrect_idx': incorrect_idx,
                   'labels': all_labels}, f)
    print(f"Saved: {labels_path}")

    del model; torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
