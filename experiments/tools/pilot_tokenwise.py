#!/usr/bin/env python3
"""Quick pilot: token-by-token extraction on 3 prompts (correct/incorrect/hallucination).
Tests the tokenwise trajectory concept before full-scale run.

Usage:
  python tools/pilot_tokenwise.py --model Qwen/Qwen3.5-2B --output activations_v3/
"""
import json, re, time, argparse
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_answer(text):
    m = re.search(r'####\s*(-?\d+[\.,]?\d*)', text)
    if m: return m.group(1).replace(',','').replace('.0','')
    nums = re.findall(r'-?\d+[\.,]?\d*', text)
    return nums[-1].replace(',','') if nums else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', default='activations_v3')
    args = parser.parse_args()

    # Load prompts
    base = Path(__file__).parent.parent
    with open(base/'prompts'/'reasoning.json') as f:
        r_prompts = json.load(f)
    with open(base/'prompts'/'gsm8k_hallucination.json') as f:
        h_data = json.load(f)
    h_list = h_data['prompts']  # list of question strings

    # Pick 3 prompts: one likely correct (easy), one likely wrong (hard),
    # one hallucination (perturbed numbers)
    prompt_easy = r_prompts[0]   # "Janet's ducks..." — easy arithmetic
    prompt_hard = r_prompts[100] # pick a mid-range one
    prompt_hallu_question = h_list[0] if h_list else "No hallucination prompts"

    prompts = [
        ('correct', prompt_easy['question']),
        ('incorrect', prompt_hard['question']),
        ('hallucination', prompt_hallu_question),
    ]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda().eval()

    layer = len(model.model.layers) // 2
    print(f"Model: {args.model}, L={layer}")
    print()

    is_inst = 'inst' in args.model.lower() or 'instruct' in args.model.lower()

    results = {}
    for label, question in prompts:
        print(f"=== {label.upper()} ===")
        print(f"Q: {question[:100]}...")

        if is_inst:
            msgs = [{'role': 'user', 'content': question}]
            tok = tokenizer.apply_chat_template(msgs, return_tensors='pt', truncation=True, max_length=1024)
        else:
            tok = tokenizer(question, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.cuda() for k, v in tok.items()}
        n_input = inputs['input_ids'].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                output_hidden_states=True, return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract per-token hidden states during generation
        token_acts = []
        for step_hidden in gen.hidden_states:
            h = step_hidden[layer][0, 0, :].cpu().float().numpy()
            token_acts.append(h)

        acts = np.stack(token_acts, axis=0)  # (n_gen, hidden_dim)

        # Decode
        response = tokenizer.decode(
            gen.sequences[0][n_input:], skip_special_tokens=True
        )
        answer = extract_answer(response)

        print(f"A: {answer}")
        print(f"Generated tokens: {len(token_acts)}, Hidden dim: {acts.shape[1]}")
        print(f"Activation range: [{acts.min():.4f}, {acts.max():.4f}]")
        print(f"Activation norm (last token): {np.linalg.norm(acts[-1]):.2f}")
        print()

        # Save
        fname = f"pilot_{args.model.replace('/','__')}_L{layer}_{label}.npy"
        np.save(Path(args.output)/fname, acts)
        results[label] = {'n_tokens': len(token_acts), 'dim': acts.shape[1],
                          'answer': answer, 'response_preview': response[:200]}

    # Save summary
    with open(Path(args.output)/f"pilot_{args.model.replace('/','__')}_L{layer}_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("Done.")
    del model; torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
