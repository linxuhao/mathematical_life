#!/usr/bin/env python3
"""DPO Eval: For each checkpoint, extract activations + compute PHI + run mini-eval.
Produces dpo_timeseries.json — the causal proof data.

Usage:
  python tools/dpo_eval.py \
      --checkpoints checkpoints/dpo_05b/ \
      --output results/dpo_timeseries.json \
      --n-prompts 200 --n-eval 200
"""
import json, re, time, os, argparse, glob
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

EPS_FRAC = 0.03

# ---- Extraction ----

def extract_activations(model, tokenizer, prompts, layer, is_instruct, batch_size=8, max_length=1024):
    """Extract last-token hidden states for a list of prompt strings."""
    all_acts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        if is_instruct:
            msgs = [[{'role': 'user', 'content': p}] for p in batch]
            tok = tokenizer.apply_chat_template(msgs, return_tensors='pt', padding=True,
                                               truncation=True, max_length=max_length, return_dict=True)
        else:
            tok = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=max_length)
        inputs = {k: v.cuda() for k, v in tok.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer][:, -1, :].cpu().float().numpy()
        all_acts.append(hidden)
    return np.concatenate(all_acts, axis=0)

# ---- Topology ----

def compute_betti(acts):
    """Compute β₁ and persistence from activation point cloud."""
    from sklearn.metrics import pairwise_distances
    eps_max = float(np.max(pairwise_distances(acts)))
    eps_pred = EPS_FRAC * eps_max
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from ripser import ripser
    dgms = ripser(acts, maxdim=1)['dgms']
    h1 = dgms[1]
    lt = h1[:, 1] - h1[:, 0]
    lt = lt[lt > 0]
    beta1_raw = len(lt)
    beta1 = int(np.sum(lt > eps_pred))
    surv_001 = int(np.sum(lt > 0.01 * eps_max))
    return {
        'beta1_raw': beta1_raw,
        'beta1': beta1,
        'surv_001': surv_001,
        'surv_rate': round(surv_001 / beta1_raw * 100, 1) if beta1_raw > 0 else 0,
        'eps_max': round(eps_max, 4),
        'eps_pred': round(eps_pred, 4),
    }

# ---- Mini-Eval ----

def extract_answer(text):
    m = re.search(r'####\s*(-?\d+[\.,]?\d*)', text)
    if m: return m.group(1).replace(',','').replace('.0','')
    nums = re.findall(r'-?\d+[\.,]?\d*', text)
    return nums[-1].replace(',','') if nums else None

def mini_eval(model, tokenizer, questions, answers, is_instruct, batch_size=8):
    """Run zero-shot GSM8K on a small set."""
    correct = 0
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i:i+batch_size]
        batch_a = answers[i:i+batch_size]
        if is_instruct:
            msgs = [[{'role': 'user', 'content': q}] for q in batch_q]
            tok = tokenizer.apply_chat_template(msgs, return_tensors='pt', padding=True,
                                               truncation=True, max_length=1024, return_dict=True)
        else:
            tok = tokenizer(batch_q, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inputs = {k: v.cuda() for k, v in tok.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        for j in range(len(batch_q)):
            response = tokenizer.decode(gen[j][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(response)
            if pred and pred == batch_a[j]:
                correct += 1
    return correct / len(questions) * 100

# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', required=True, help='Directory with checkpoint-* subdirs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-prompts', type=int, default=200, help='Prompts per type for topology')
    parser.add_argument('--n-eval', type=int, default=200, help='GSM8K questions for mini-eval')
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B',
                       help='HF model name for step 0 baseline')
    args = parser.parse_args()

    # Find checkpoints
    ckpt_dir = Path(args.checkpoints)
    ckpts = sorted(ckpt_dir.glob('checkpoint-*'))
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir}")
        return
    print(f"Found {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")

    # Load prompts (use first N)
    base_dir = Path(__file__).parent.parent
    with open(base_dir / 'prompts' / 'reasoning.json') as f:
        r_all = json.load(f)
    with open(base_dir / 'prompts' / 'gsm8k_hallucination.json') as f:
        h_data = json.load(f)
    h_all = h_data['prompts']  # list of strings
    r_prompts = [p['question'] for p in r_all[:args.n_prompts]]
    h_prompts = h_all[:args.n_prompts]

    # Load mini-eval questions + answers
    gsm8k = load_dataset('openai/gsm8k', 'main', split='test')
    eval_questions = []
    eval_answers = []
    for i, item in enumerate(gsm8k):
        if i >= args.n_eval: break
        eval_questions.append(item['question'])
        m = re.search(r'####\s*(-?\d+[\.,]?\d*)', item['answer'])
        eval_answers.append(m.group(1).replace(',','').replace('.0','') if m else '')

    results = []

    def evaluate_model(model_path, step):
        print(f"\n=== Step {step} ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16,
            attn_implementation='eager', trust_remote_code=True
        ).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        is_instruct = False
        n_layers = len(model.model.layers)
        layer = n_layers // 2

        t0 = time.time()
        acts_r = extract_activations(model, tokenizer, r_prompts, layer, is_instruct)
        acts_h = extract_activations(model, tokenizer, h_prompts, layer, is_instruct)
        topology_r = compute_betti(acts_r)
        topology_h = compute_betti(acts_h)
        phi = round(topology_r['surv_rate'] / topology_h['surv_rate'], 2) if topology_h['surv_rate'] > 0 else None
        topo_time = time.time() - t0

        t0 = time.time()
        acc = mini_eval(model, tokenizer, eval_questions, eval_answers, is_instruct)
        eval_time = time.time() - t0

        r = {
            'step': step, 'beta1_r': topology_r['beta1'], 'beta1_h': topology_h['beta1'],
            'raw_r': topology_r['beta1_raw'], 'raw_h': topology_h['beta1_raw'],
            'surv_r': topology_r['surv_rate'], 'surv_h': topology_h['surv_rate'],
            'phi': phi, 'gsm8k_acc': round(acc, 1),
            'topo_time_s': round(topo_time, 1), 'eval_time_s': round(eval_time, 1),
        }
        results.append(r)
        print(f"  R: β₁={r['beta1_r']} raw={r['raw_r']} surv={r['surv_r']}%")
        print(f"  H: β₁={r['beta1_h']} raw={r['raw_h']} surv={r['surv_h']}%")
        print(f"  PHI={r['phi']}  GSM8K={r['gsm8k_acc']}%")
        del model; torch.cuda.empty_cache()

    # Step 0: base model
    evaluate_model(args.base_model, 0)

    # Steps 125, 250, ...: checkpoints
    for ckpt_path in ckpts:
        step = int(ckpt_path.name.split('-')[1])
        evaluate_model(str(ckpt_path), step)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print summary table
    print(f"\n{'Step':<10s} {'β₁_R':>6s} {'β₁_H':>6s} {'PHI':>6s} {'GSM8K':>8s}")
    print('-'*40)
    for r in results:
        phi_str = f"{r['phi']:.2f}" if r['phi'] else 'N/A'
        print(f"{r['step']:<10d} {r['beta1_r']:>6d} {r['beta1_h']:>6d} {phi_str:>6s} {r['gsm8k_acc']:>7.1f}%")

if __name__ == '__main__':
    main()
