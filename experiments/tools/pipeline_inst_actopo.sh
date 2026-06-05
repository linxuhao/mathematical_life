#!/bin/bash
# Pipeline: Qwen3.5-2B-Instruct SFT → actopo eval
# Runs on GPU0, survives SSH disconnect
set -e

PY=~/pred1-env/bin/python3
MODEL="Qwen/Qwen3.5-2B"
OUTDIR=~/papers/mathematical-life/experiments/checkpoints/qwen35_2b_inst_sft
RESULT=~/papers/mathematical-life/experiments/results/qwen35_2b_inst_actopo.json
LOGFILE=~/papers/mathematical-life/experiments/logs/pipeline_inst_$(date +%Y%m%d_%H%M).log

mkdir -p $(dirname $LOGFILE) $(dirname $RESULT)

exec > >(tee -a $LOGFILE) 2>&1

echo "========================================"
echo "Pipeline: Qwen3.5-2B-Instruct SFT + actopo"
echo "Started: $(date)"
echo "Model: $MODEL"
echo "Output: $OUTDIR"
echo "Result: $RESULT"
echo "========================================"

# Step 1: SFT
echo ""
echo "=== Step 1: SFT Training ==="
$PY tools/sft_gsm8k.py \
    --model $MODEL \
    --output $OUTDIR \
    --batch-size 1 \
    --max-length 512 \
    --save-steps 250

echo ""
echo "SFT done: $(date)"

# Step 2: actopo eval on all checkpoints (Python script)
echo ""
echo "=== Step 2: actopo Eval ==="
$PY -c "
import json, time, os, sys, re
from pathlib import Path
import torch, numpy as np

sys.path.insert(0, str(Path.home() / 'papers/mathematical-life/actopo/src'))
from actopo.extract import last_token_indices
from actopo.topology import measure, phi as compute_phi
from actopo.protocol import FROZEN_V5
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

CKPT_DIR = Path('$OUTDIR')
BASE_MODEL = '$MODEL'
PROMPTS_DIR = Path('papers/mathematical-life/experiments/prompts')
OUTPUT = '$RESULT'
N_PROMPTS = 1319
N_EVAL = 200
BATCH_SIZE = 8
MAX_LENGTH = 1024

with open(PROMPTS_DIR / 'reasoning.json') as f: r_data = json.load(f)
with open(PROMPTS_DIR / 'gsm8k_hallucination.json') as f: h_data = json.load(f)
r_prompts = [p['question'] for p in r_data[:N_PROMPTS]]
h_prompts = h_data['prompts'][:N_PROMPTS]

gsm8k = load_dataset('openai/gsm8k', 'main', split='test')
eval_q, eval_a = [], []
for i, item in enumerate(gsm8k):
    if i >= N_EVAL: break
    eval_q.append(item['question'])
    m = re.search(r'####\s*(-?\d+[\.,]?\d*)', item['answer'])
    eval_a.append(m.group(1).replace(',','').replace('.0','') if m else '')

def extract_acts(model, tokenizer, prompts, layer):
    all_acts = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        tok = tokenizer(batch, return_tensors='pt', padding=True,
                       truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.cuda() for k, v in tok.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer]
            idx = last_token_indices(inputs['attention_mask'])
            idx = idx.to(hidden.device)
            last_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
            last_hidden = last_hidden.cpu().float().numpy()
        all_acts.append(last_hidden)
    return np.concatenate(all_acts, axis=0)

def extract_answer(text):
    m = re.search(r'####\s*(-?\d+[\.,]?\d*)', text)
    if m: return m.group(1).replace(',','').replace('.0','')
    nums = re.findall(r'-?\d+[\.,]?\d*', text)
    return nums[-1].replace(',','') if nums else None

def mini_eval(model, tokenizer, questions, answers):
    correct = 0
    for i in range(0, len(questions), BATCH_SIZE):
        bq, ba = questions[i:i+BATCH_SIZE], answers[i:i+BATCH_SIZE]
        tok = tokenizer(bq, return_tensors='pt', padding=True,
                       truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.cuda() for k, v in tok.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        for j in range(len(bq)):
            resp = tokenizer.decode(gen[j][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(resp)
            if pred and pred == ba[j]: correct += 1
    return correct / len(questions) * 100

results = []
ckpts = sorted([d for d in CKPT_DIR.iterdir() if d.name.startswith('checkpoint-')])
print(f'Found {len(ckpts)} checkpoints')

# Step 0
print('\n=== Step 0 (base instruct model) ===')
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16,
    attn_implementation='eager', trust_remote_code=True
).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
layer = len(model.model.layers) // 2

t0 = time.time()
acts_r = extract_acts(model, tokenizer, r_prompts, layer)
acts_h = extract_acts(model, tokenizer, h_prompts, layer)
topo_r = measure(acts_r, FROZEN_V5)
topo_h = measure(acts_h, FROZEN_V5)
phi_val = compute_phi(acts_r, acts_h, FROZEN_V5)
tt = time.time() - t0
t0 = time.time()
acc = mini_eval(model, tokenizer, eval_q, eval_a)
et = time.time() - t0
r = {'step':0,'beta1_r':topo_r.beta1,'beta1_h':topo_h.beta1,'raw_r':topo_r.beta1_raw,'raw_h':topo_h.beta1_raw,'surv_r':topo_r.survival_rate,'surv_h':topo_h.survival_rate,'phi':phi_val,'gsm8k_acc':round(acc,1)}
results.append(r)
print(f\"  R: β₁={r['beta1_r']} raw={r['raw_r']} surv={r['surv_r']}%\")
print(f\"  H: β₁={r['beta1_h']} raw={r['raw_h']} surv={r['surv_h']}%\")
print(f\"  PHI={r['phi']}  GSM8K={r['gsm8k_acc']}%\")
del model; torch.cuda.empty_cache()

for ckpt_path in ckpts:
    step = int(ckpt_path.name.split('-')[1])
    print(f'\n=== Step {step} ===')
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path), torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    acts_r = extract_acts(model, tokenizer, r_prompts, layer)
    acts_h = extract_acts(model, tokenizer, h_prompts, layer)
    topo_r = measure(acts_r, FROZEN_V5)
    topo_h = measure(acts_h, FROZEN_V5)
    phi_val = compute_phi(acts_r, acts_h, FROZEN_V5)
    tt = time.time() - t0
    t0 = time.time()
    acc = mini_eval(model, tokenizer, eval_q, eval_a)
    et = time.time() - t0
    r = {'step':step,'beta1_r':topo_r.beta1,'beta1_h':topo_h.beta1,'raw_r':topo_r.beta1_raw,'raw_h':topo_h.beta1_raw,'surv_r':topo_r.survival_rate,'surv_h':topo_h.survival_rate,'phi':phi_val,'gsm8k_acc':round(acc,1)}
    results.append(r)
    print(f\"  R: β₁={r['beta1_r']} raw={r['raw_r']} surv={r['surv_r']}%\")
    print(f\"  H: β₁={r['beta1_h']} raw={r['raw_h']} surv={r['surv_h']}%\")
    print(f\"  PHI={r['phi']}  GSM8K={r['gsm8k_acc']}%\")
    del model; torch.cuda.empty_cache()

with open(OUTPUT, 'w') as f: json.dump(results, f, indent=2)
print(f'\nSaved → {OUTPUT}')

steps = sorted(results, key=lambda x: x['step'])
print(f\"\n{'Step':<8s} {'β₁_R':>5s} {'β₁_H':>5s} {'PHI':>6s} {'GSM8K':>7s}\")
print('-'*35)
for r2 in steps:
    phi_str = f\"{r2['phi']:.2f}\" if r2['phi'] else 'N/A'
    print(f\"{r2['step']:<8d} {r2['beta1_r']:>5d} {r2['beta1_h']:>5d} {phi_str:>6s} {r2['gsm8k_acc']:>6.1f}%\")
"

echo ""
echo "========================================"
echo "Pipeline complete: $(date)"
echo "Result: $RESULT"
echo "Log: $LOGFILE"
echo "========================================"
