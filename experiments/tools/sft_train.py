#!/usr/bin/env python3
"""SFT Training v2 — dynamic tokenization, minimal memory.
Tests if GPU crash is dataset-related.

Usage:
  python tools/sft_train.py --output checkpoints/sft_05b/ --max-prompts 200
"""
import argparse, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--output', default='checkpoints/sft_05b')
    parser.add_argument('--n-epochs', type=int, default=2)
    parser.add_argument('--max-prompts', type=int, default=5000)
    parser.add_argument('--save-steps', type=int, default=125)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HH-RLHF...")
    ds = load_dataset('Anthropic/hh-rlhf', split='train')
    ds = ds.select(range(min(len(ds), args.max_prompts)))
    texts = ds['chosen']

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation='eager', trust_remote_code=True
    ).cuda()
    model.train()

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_batches = len(texts) // args.batch_size
    total_steps = (n_batches * args.n_epochs) // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, total_steps)

    print(f"Training: {len(texts)} examples, batch={args.batch_size}, grad_accum={args.grad_accum}")
    global_step = 0
    batch_count = 0

    for epoch in range(args.n_epochs):
        # Shuffle
        import random
        indices = list(range(len(texts)))
        random.shuffle(indices)

        for i in range(0, len(texts), args.batch_size):
            batch_texts = [texts[indices[j]] for j in range(i, min(i+args.batch_size, len(texts)))]
            if len(batch_texts) < args.batch_size:
                continue

            tok = tokenizer(batch_texts, truncation=True, max_length=args.max_length,
                          padding='max_length', return_tensors='pt')
            labels = tok['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            batch = {
                'input_ids': tok['input_ids'].cuda(),
                'attention_mask': tok['attention_mask'].cuda(),
                'labels': labels.cuda(),
            }

            try:
                with torch.amp.autocast('cuda'):
                    outputs = model(**batch)
                    loss = outputs.loss / args.grad_accum
                scaler.scale(loss).backward()
            except Exception as e:
                print(f"  ERROR at batch {i}: {e}")
                model.zero_grad()
                continue

            batch_count += 1
            if batch_count % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                loss_val = loss.item() * args.grad_accum
                if global_step % args.save_steps == 0:
                    ckpt_dir = out_dir / f'checkpoint-{global_step}'
                    ckpt_dir.mkdir(exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"  Step {global_step}: loss={loss_val:.4f} saved → {ckpt_dir.name}")

                if global_step % 25 == 0:
                    print(f"  Step {global_step}: loss={loss_val:.4f}")

    final_dir = out_dir / 'final'
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nDone: {final_dir}")

if __name__ == '__main__':
    main()
