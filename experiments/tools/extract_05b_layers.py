#!/usr/bin/env python3
"""Extract Qwen2.5-0.5B base+instruct activations at multiple layers for layer-wise analysis.
Usage (on server with GPU):
  python tools/extract_05b_layers.py --output activations_v3/
"""
import os, argparse, torch, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

MODEL_BASE = "Qwen/Qwen2.5-0.5B"
MODEL_INST = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_TYPES = ["reasoning", "control", "hallucination"]
PROMPT_FILES = {"reasoning": "reasoning.json", "control": "control.json", "hallucination": "gsm8k_hallucination.json"}
LAYERS = [5, 10, 15, 20]  # every ~5 layers of 24
BATCH_SIZE = 8
MAX_LENGTH = 1024

def load_prompts(prompt_type):
    fname = PROMPT_FILES.get(prompt_type, f"{prompt_type}.json")
    path = Path(__file__).parent.parent / "prompts" / fname
    with open(path) as f:
        return json.load(f)

def extract_model(model_name, output_dir, is_instruct):
    print(f"\n=== {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="eager", trust_remote_code=True
    ).cuda().eval()

    n_layers = len(model.model.layers)
    print(f"  Layers: {n_layers}, extracting at: {LAYERS}")

    for ptype in PROMPT_TYPES:
        prompts_data = load_prompts(ptype)
        if isinstance(prompts_data, dict) and 'prompts' in prompts_data:
            prompts_data = prompts_data['prompts']
        if isinstance(prompts_data, list) and len(prompts_data) > 0:
            if isinstance(prompts_data[0], str):
                texts = prompts_data
            else:
                texts = [p.get("question", p.get("prompt", "")) for p in prompts_data]
        else:
            texts = []
        print(f"  {ptype}: {len(texts)} prompts")

        all_activations = {layer: [] for layer in LAYERS}

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            if is_instruct:
                msgs = [[{"role": "user", "content": t}] for t in batch]
                inputs = tokenizer.apply_chat_template(msgs, return_tensors="pt", padding=True,
                                                       truncation=True, max_length=MAX_LENGTH,
                                                       return_dict=True)
            else:
                inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                  truncation=True, max_length=MAX_LENGTH)

            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states  # tuple of (n_layers+1) tensors

                for layer in LAYERS:
                    last_token_hidden = hidden[layer][:, -1, :].cpu().float().numpy()
                    all_activations[layer].append(last_token_hidden)

            if (i // BATCH_SIZE) % 20 == 0:
                print(f"    batch {i//BATCH_SIZE}: {i+len(batch)}/{len(texts)}")

        # Concatenate and save
        for layer in LAYERS:
            acts = np.concatenate(all_activations[layer], axis=0)
            fname = f"{model_name.replace('/', '__')}_L{layer}_{ptype}.npy"
            out_path = Path(output_dir) / fname
            np.save(out_path, acts)
            print(f"    Saved: {fname} shape={acts.shape}")

    del model
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='activations_v3', help='output directory')
    parser.add_argument('--base-only', action='store_true')
    parser.add_argument('--inst-only', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.inst_only:
        extract_model(MODEL_BASE, args.output, is_instruct=False)
    if not args.base_only:
        extract_model(MODEL_INST, args.output, is_instruct=True)

    print("\nDone. Files in:", out_dir)

if __name__ == '__main__':
    main()
