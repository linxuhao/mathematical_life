#!/usr/bin/env python3
"""Tool: Extract token-by-token activations for topological trajectory analysis.

Two modes:
  read  — activation at each INPUT token position (batch, fast)
  write — activation at each GENERATED token position (sequential, slower)

Usage (on server with GPU):
  # Write mode: track reasoning trajectory during generation
  python tools/extract_tokenwise.py \
      --model Qwen/Qwen3.5-2B \
      --mode write --max-prompts 500 \
      --output activations_v3/

  # Read mode: track comprehension trajectory during input
  python tools/extract_tokenwise.py \
      --model Qwen/Qwen3.5-2B \
      --mode read --max-prompts 500 \
      --output activations_v3/

Output:
  activations_v3/{model}_L{layer}_{mode}.npz
    Contains per-sample activations + correctness labels
    .npz keys: 'correct_acts', 'incorrect_acts', 'correct_lens', 'incorrect_lens', 'labels'
"""
import os, json, argparse, re, time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def extract_final_answer(text):
    m = re.search(r'####\s*(-?\d+[\.,]?\d*)', text)
    if m:
        return m.group(1).replace(',', '').replace('.0', '')
    numbers = re.findall(r'-?\d+[\.,]?\d*', text)
    return numbers[-1].replace(',', '') if numbers else None

def check_correct(model_output, ground_truth):
    pred = extract_final_answer(model_output)
    truth = str(ground_truth).strip()
    if pred is None:
        return False
    pred = pred.rstrip('0').rstrip('.') if '.' in pred else pred
    truth = truth.rstrip('0').rstrip('.') if '.' in truth else truth
    return pred == truth

def run_read_mode(model, tokenizer, prompts_data, answer_map, layer, args):
    """Batch mode: extract activation at every INPUT token position."""
    is_instruct = 'inst' in args.model.lower() or 'instruct' in args.model.lower()

    correct_acts = []    # list of (n_tokens, hidden_dim) arrays
    incorrect_acts = []
    labels = []

    for i in range(0, len(prompts_data), args.batch_size):
        batch = prompts_data[i:i+args.batch_size]
        questions = [p['question'] for p in batch]
        answers = [answer_map.get(q.strip(), '') for q in questions]

        if is_instruct:
            msgs = [[{'role': 'user', 'content': q}] for q in questions]
            tokenized = tokenizer.apply_chat_template(
                msgs, return_tensors='pt', padding=True,
                truncation=True, max_length=args.max_length, return_dict=True
            )
        else:
            tokenized = tokenizer(questions, return_tensors='pt', padding=True,
                                 truncation=True, max_length=args.max_length)

        inputs = {k: v.cuda() for k, v in tokenized.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
            hidden = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)

        # For each sample in batch, extract per-token activations
        for j in range(len(batch)):
            seq_len = attention_mask[j].sum().item() if attention_mask is not None else input_ids.shape[1]
            acts = hidden[j, :seq_len, :].cpu().float().numpy()  # (seq_len, hidden_dim)

            # Check correctness by generating answer
            single_input = {k: v[j:j+1] for k, v in inputs.items()}
            gen = model.generate(**single_input, max_new_tokens=256, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(gen[0][single_input['input_ids'].shape[1]:],
                                       skip_special_tokens=True)
            is_correct = check_correct(response, answers[j])

            if is_correct:
                correct_acts.append(acts)
            else:
                incorrect_acts.append(acts)

            labels.append({'idx': i+j, 'correct': is_correct,
                          'n_input_tokens': seq_len})

        if (i + len(batch)) % 100 == 0 or i + len(batch) >= len(prompts_data):
            print(f"  [{min(i+len(batch), len(prompts_data))}/{len(prompts_data)}] "
                  f"correct={len(correct_acts)} incorrect={len(incorrect_acts)}")

    return correct_acts, incorrect_acts, labels

def run_write_mode(model, tokenizer, prompts_data, answer_map, layer, args):
    """Generate mode: extract activation at every GENERATED token position."""
    is_instruct = 'inst' in args.model.lower() or 'instruct' in args.model.lower()

    correct_acts = []
    incorrect_acts = []
    labels = []

    t0 = time.time()
    for i, item in enumerate(prompts_data):
        prompt = item['question']
        answer = answer_map.get(prompt.strip(), '')

        if is_instruct:
            msgs = [{'role': 'user', 'content': prompt}]
            tokenized = tokenizer.apply_chat_template(
                msgs, return_tensors='pt', truncation=True,
                max_length=args.max_length
            )
        else:
            tokenized = tokenizer(prompt, return_tensors='pt',
                                 truncation=True, max_length=args.max_length)

        inputs = {k: v.cuda() for k, v in tokenized.items()}

        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # gen_outputs.hidden_states: list of tuples, one per generation step
        # gen_outputs.hidden_states[t][layer] = (batch=1, 1, hidden_dim)
        gen_states = []
        for step_hidden in gen_outputs.hidden_states:
            h = step_hidden[layer][0, 0, :].cpu().float().numpy()  # (hidden_dim,)
            gen_states.append(h)

        acts = np.stack(gen_states, axis=0)  # (n_generated_tokens, hidden_dim)

        # Decode response for correctness check
        response = tokenizer.decode(
            gen_outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        is_correct = check_correct(response, answer)

        if is_correct:
            correct_acts.append(acts)
        else:
            incorrect_acts.append(acts)

        labels.append({'idx': i, 'correct': is_correct,
                       'n_generated_tokens': len(gen_states)})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts_data) - i - 1) / rate
            print(f"  [{i+1}/{len(prompts_data)}] correct={len(correct_acts)} "
                  f"incorrect={len(incorrect_acts)} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    return correct_acts, incorrect_acts, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode', required=True, choices=['read', 'write'])
    parser.add_argument('--output', default='activations_v3')
    parser.add_argument('--max-prompts', type=int, default=1319)
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
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
        q = item['question'].strip()
        a = item['answer'].strip()
        m = re.search(r'####\s*(-?\d+[\.,]?\d*)', a)
        if m:
            answer_map[q] = m.group(1).replace(',', '').replace('.0', '')
    print(f"Loaded {len(answer_map)} answers")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda().eval()

    n_layers = len(model.model.layers)
    layer = args.layer if args.layer is not None else n_layers // 2
    print(f"Model: {args.model}, Layers: {n_layers}, extracting L={layer}")
    print(f"Mode: {args.mode}, Prompts: {len(prompts_data)}")

    t_start = time.time()

    if args.mode == 'read':
        correct_acts, incorrect_acts, labels = run_read_mode(
            model, tokenizer, prompts_data, answer_map, layer, args
        )
    else:
        correct_acts, incorrect_acts, labels = run_write_mode(
            model, tokenizer, prompts_data, answer_map, layer, args
        )

    elapsed = time.time() - t_start
    n_correct = len(correct_acts)
    n_incorrect = len(incorrect_acts)
    print(f"\nDone in {elapsed:.0f}s. Correct: {n_correct}, Incorrect: {n_incorrect}")
    print(f"Accuracy: {n_correct/(n_correct+n_incorrect)*100:.1f}%")

    # Save as .npz (each list element is variable-length, can't stack)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f'{short_name}_L{layer}_{args.mode}.npz'

    # Store lengths for reconstruction
    correct_lens = [a.shape[0] for a in correct_acts]
    incorrect_lens = [a.shape[0] for a in incorrect_acts]

    # Flatten and concatenate
    if correct_acts:
        correct_flat = np.concatenate(correct_acts, axis=0)
    else:
        correct_flat = np.zeros((0, correct_acts[0].shape[1])) if correct_acts else np.zeros((0, 1))

    if incorrect_acts:
        incorrect_flat = np.concatenate(incorrect_acts, axis=0)
    else:
        incorrect_flat = np.zeros((0, incorrect_acts[0].shape[1])) if incorrect_acts else np.zeros((0, 1))

    np.savez_compressed(
        out_dir / fname,
        correct_acts=correct_flat,
        incorrect_acts=incorrect_flat,
        correct_lens=np.array(correct_lens, dtype=np.int32),
        incorrect_lens=np.array(incorrect_lens, dtype=np.int32),
    )

    # Save labels separately
    labels_fname = f'{short_name}_L{layer}_{args.mode}_labels.json'
    with open(out_dir / labels_fname, 'w') as f:
        json.dump({'model': args.model, 'layer': layer, 'mode': args.mode,
                   'n_correct': n_correct, 'n_incorrect': n_incorrect,
                   'labels': labels}, f, indent=2)

    # Summary of sequence lengths
    if correct_lens:
        print(f"Correct: {n_correct} samples, "
              f"tokens: mean={np.mean(correct_lens):.0f} median={np.median(correct_lens):.0f} "
              f"min={np.min(correct_lens)} max={np.max(correct_lens)}")
    if incorrect_lens:
        print(f"Incorrect: {n_incorrect} samples, "
              f"tokens: mean={np.mean(incorrect_lens):.0f} median={np.median(incorrect_lens):.0f} "
              f"min={np.min(incorrect_lens)} max={np.max(incorrect_lens)}")

    print(f"Saved: {fname} ({out_dir / fname})")
    print(f"Saved: {labels_fname}")

    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
