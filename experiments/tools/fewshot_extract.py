#!/usr/bin/env python3
"""Few-shot labeler + tokenwise extractor for GSM8K with 5-shot prompting.
Higher accuracy → meaningful R_flawed (near-misses, not catastrophic failures).
Longer max_new_tokens → un-truncated reasoning chains.

Usage:
  python tools/fewshot_extract.py \
      --model Qwen/Qwen3.5-2B \
      --output activations_v3/ \
      --n-shot 5 --max-new-tokens 512 --batch-size 2
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

def build_fewshot_prompt(question, fewshot_examples):
    prefix = ''
    for ex in fewshot_examples:
        prefix += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    return prefix + f"Q: {question}\nA:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', default='activations_v3')
    parser.add_argument('--n-shot', type=int, default=5)
    parser.add_argument('--max-prompts', type=int, default=1319)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()

    short_name = args.model.replace('/', '__')

    # Build few-shot examples from training set
    train = load_dataset('openai/gsm8k', 'main', split='train')
    fewshot = [{'question': train[i]['question'], 'answer': train[i]['answer']}
               for i in range(args.n_shot)]

    # Load test questions + answers
    test = load_dataset('openai/gsm8k', 'main', split='test')
    questions = []
    answers = []
    for item in test:
        questions.append(item['question'])
        m = re.search(r'####\s*(-?\d+[\.,]?\d*)', item['answer'])
        answers.append(m.group(1).replace(',','').replace('.0','') if m else '')

    questions = questions[:args.max_prompts]
    answers = answers[:args.max_prompts]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda().eval()

    n_layers = len(model.model.layers)
    layer = n_layers // 2
    is_inst = 'inst' in args.model.lower() or 'instruct' in args.model.lower()

    print(f"Model: {args.model}, L={layer}, {args.n_shot}-shot, max_tokens={args.max_new_tokens}")
    print(f"Batch: {args.batch_size}, Prompts: {len(questions)}")

    all_acts = []
    all_labels = []
    correct_idx = []
    incorrect_idx = []

    t0 = time.time()
    for i in range(0, len(questions), args.batch_size):
        batch_qs = questions[i:i+args.batch_size]
        batch_as = answers[i:i+args.batch_size]

        # Build few-shot prompts
        full_prompts = [build_fewshot_prompt(q, fewshot) for q in batch_qs]

        if is_inst:
            msgs = [[{'role': 'user', 'content': p}] for p in full_prompts]
            tok = tokenizer.apply_chat_template(msgs, return_tensors='pt', padding=True,
                                               truncation=True, max_length=2048, return_dict=True)
        else:
            tok = tokenizer(full_prompts, return_tensors='pt', padding=True,
                          truncation=True, max_length=2048)

        inputs = {k: v.cuda() for k, v in tok.items()}

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                output_hidden_states=True, return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        n_gen = len(gen.hidden_states)
        batch_n = len(batch_qs)

        for j in range(batch_n):
            acts_j = []
            for t in range(n_gen):
                h = gen.hidden_states[t][layer][j, 0, :].cpu().float().numpy()
                acts_j.append(h)

            seq = gen.sequences[j]
            attn = inputs.get('attention_mask', None)
            input_len = attn[j].sum().item() if attn is not None else inputs['input_ids'].shape[1]
            gen_ids = seq[input_len:]
            eos_pos = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                eos_idx = eos_pos[0].item()
                acts_j = acts_j[:eos_idx]
                gen_ids = gen_ids[:eos_idx]

            acts_arr = np.stack(acts_j, axis=0) if acts_j else np.zeros((0, model.config.hidden_size), dtype=np.float32)
            all_acts.append(acts_arr)

            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = extract_answer(response)
            is_correct = (pred == batch_as[j]) if pred else False
            all_labels.append(is_correct)
            (correct_idx if is_correct else incorrect_idx).append(i + j)

        done = min(i + args.batch_size, len(questions))
        if done % 50 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed * 3600
            avg_tok = np.mean([a.shape[0] for a in all_acts[-50:]]) if all_acts else 0
            print(f"  [{done}/{len(questions)}] correct={len(correct_idx)} "
                  f"incorrect={len(incorrect_idx)} ({rate:.0f}/hr) avg_tok={avg_tok:.0f}")

    elapsed = time.time() - t0
    acc = len(correct_idx) / len(all_labels) * 100 if all_labels else 0
    total_tokens = sum(a.shape[0] for a in all_acts)
    print(f"\nDone in {elapsed:.0f}s. Acc={acc:.1f}% ({len(correct_idx)}/{len(all_labels)})")
    print(f"Correct: {len(correct_idx)}, Incorrect: {len(incorrect_idx)}")
    print(f"Total tokens: {total_tokens}")

    # Save
    lens = np.array([a.shape[0] for a in all_acts], dtype=np.int32)
    acts_flat = np.concatenate(all_acts, axis=0)
    npz_path = Path(args.output) / f'{short_name}_L{layer}_{args.n_shot}shot_tokenwise.npz'
    np.savez_compressed(npz_path, acts=acts_flat, lens=lens,
                        correct_idx=np.array(correct_idx, dtype=np.int32),
                        incorrect_idx=np.array(incorrect_idx, dtype=np.int32))
    print(f"Saved: {npz_path}")

    labels_path = Path(args.output) / f'{short_name}_L{layer}_{args.n_shot}shot_labels.json'
    with open(labels_path, 'w') as f:
        json.dump({'model': args.model, 'layer': layer, 'n_shot': args.n_shot,
                   'n_prompts': len(all_labels), 'accuracy': round(acc,2),
                   'n_correct': len(correct_idx), 'n_incorrect': len(incorrect_idx),
                   'correct_idx': correct_idx, 'incorrect_idx': incorrect_idx,
                   'labels': all_labels}, f)
    print(f"Saved: {labels_path}")

    del model; torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
