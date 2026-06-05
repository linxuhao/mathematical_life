#!/usr/bin/env python3
"""Tool 1: Extract L/2 activations for all models × prompt types.
Usage: 
  python tools/extract_activations.py --models config/models.json --prompts prompts/ -o activations/
  python tools/extract_activations.py --models config/models_smoke.json --prompts prompts/ -o activations/ --prompt-types reasoning
"""
import os, sys, json, argparse, time, gc
from pathlib import Path
from multiprocessing import Process

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

PTYPE_FILES = {
    'reasoning': 'reasoning.json',
    'control': 'control.json',
    'narrative': 'narrative.json',
    'hallucination': 'gsm8k_hallucination.json',
    'bbh_ff': 'bbh_ff.json',
    'bbh_nav': 'bbh_nav.json',
    'bbh_ts': 'bbh_ts.json',
    'math500': 'math500.json',
}

def load_prompts(prompts_dir, prompt_types):
    prompts_dir = Path(prompts_dir)
    prompts = {}
    for pt in prompt_types:
        fname = PTYPE_FILES.get(pt, f'{pt}.json')
        fpath = prompts_dir / fname
        if not fpath.exists():
            print(f"WARN: {fpath} not found, skipping {pt}")
            continue
        with open(fpath) as f:
            data = json.load(f)
        if isinstance(data, list):
            items = [item['question'] if isinstance(item, dict) else item for item in data]
        elif 'prompts' in data:
            items = data['prompts']
        else:
            raise ValueError(f"Unknown format: {fpath}")
        prompts[pt] = items
        print(f"  {pt}: {len(items)} prompts")
    return prompts

def do_gpu(gpu_id, models, prompt_types, prompts_dir, output_dir):
    """Run extraction on one GPU (called in subprocess)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    prompts_data = load_prompts(prompts_dir, prompt_types)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for m in models:
        name, hf_id = m['name'], m['hf']
        batch_size, n_layers = m['batch'], m['n_layers']
        L2 = n_layers // 2
        
        # List work needed
        todo = []
        for pt in prompt_types:
            if pt not in prompts_data:
                continue
            npy = out_dir / f'{name}_L{L2}_{pt}.npy'
            if npy.exists():
                print(f"  [{name}/{pt}] exists ({npy.stat().st_size} bytes), skip")
            else:
                todo.append(pt)
        if not todo:
            print(f"  [{name}] all done, skip model")
            continue
        
        print(f"  Loading {hf_id}...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        # Ensure pad_token is set (some models like SmolLM2 don't have it)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.float16, device_map='cuda:0',
            attn_implementation='eager', trust_remote_code=True
        )
        model.eval()
        print(f"  Loaded in {time.time()-t0:.0f}s, {model.get_memory_footprint()/1e9:.1f}GB")
        
        for pt in todo:
            prompts = prompts_data[pt]
            out_file = out_dir / f'{name}_L{L2}_{pt}.npy'
            print(f"  [{name}/{pt}] {len(prompts)} prompts...")
            t1 = time.time()
            
            all_hidden = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                has_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template and 'Inst' in name
                if has_template:
                    msgs = [[{'role': 'user', 'content': p}] for p in batch]
                    inputs = tokenizer.apply_chat_template(
                        msgs, add_generation_prompt=False, tokenize=True,
                        return_tensors='pt', padding=True, truncation=True,
                        max_length=1024
                    )
                else:
                    inputs = tokenizer(
                        batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=1024
                    )
                inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                hidden = outputs.hidden_states[L2]
                mask = inputs['attention_mask']
                last_idx = mask.sum(dim=1) - 1
                last = hidden[torch.arange(len(batch)), last_idx]
                all_hidden.append(last.cpu().to(torch.float32).numpy())
                
                if (i // batch_size) % 20 == 0:
                    torch.cuda.empty_cache()
            
            arr = np.concatenate(all_hidden, axis=0)
            np.save(out_file, arr)
            dt = time.time() - t1
            print(f"  [{name}/{pt}] {arr.shape} saved ({dt:.0f}s, {dt/len(prompts):.3f}s/prompt)")
        
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--prompt-types', default='reasoning,control,narrative,hallucination')
    parser.add_argument('--gpus', default='0,1')
    args = parser.parse_args()
    
    with open(args.models) as f:
        all_models = json.load(f)
    prompt_types = [p.strip() for p in args.prompt_types.split(',')]
    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    
    # Assign models to GPUs
    gpu_models = {g: [] for g in gpu_ids}
    for m in all_models:
        g = m.get('gpu', 0)
        if g in gpu_models:
            gpu_models[g].append(m)
    
    print(f"Models: {len(all_models)}, GPUs: {gpu_ids}, Prompt types: {prompt_types}")
    for g in gpu_ids:
        names = [m['name'] for m in gpu_models[g]]
        print(f"  GPU {g}: {len(names)} models -> {names}")
    
    # Spawn one process per GPU
    procs = []
    for gpu_id in gpu_ids:
        if not gpu_models[gpu_id]:
            continue
        p = Process(target=do_gpu, args=(gpu_id, gpu_models[gpu_id], prompt_types, args.prompts, args.output))
        p.start()
        procs.append((gpu_id, p))
    
    for gpu_id, p in procs:
        p.join()
        print(f"GPU {gpu_id}: exit {p.exitcode}")
    
    print("=== Extraction done ===")

if __name__ == '__main__':
    main()
