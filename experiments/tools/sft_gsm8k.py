#!/usr/bin/env python3
"""SFT on GSM8K training set — teaches step-by-step math reasoning.
Tests causal link: does math training change PHI?

Usage:
  python tools/sft_gsm8k.py --output checkpoints/sft_gsm8k/
"""
import argparse, torch, random
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--output', default='checkpoints/sft_gsm8k')
    parser.add_argument('--n-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--save-steps', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-length', type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GSM8K train set: question + step-by-step answer
    ds = load_dataset('openai/gsm8k', 'main', split='train')
    print(f"GSM8K train: {len(ds)} examples")

    # Format: Q: question\nA: step-by-step reasoning #### answer
    texts = [f"Q: {item['question']}\nA: {item['answer']}" for item in ds]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation='eager', trust_remote_code=True
    ).cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(texts) // args.batch_size * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, total_steps)

    print(f"Training: {len(texts)} examples × {args.n_epochs} epochs, batch={args.batch_size}")

    global_step = 0
    for epoch in range(args.n_epochs):
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

            batch = {k: v.cuda() for k, v in tok.items()}
            batch['labels'] = labels.cuda()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.save_steps == 0:
                ckpt_dir = out_dir / f'checkpoint-{global_step}'
                ckpt_dir.mkdir(exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"  Step {global_step}: loss={loss.item():.4f} saved")

            if global_step % 100 == 0:
                print(f"  Step {global_step}: loss={loss.item():.4f}")

    final_dir = out_dir / 'final'
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nDone: {final_dir}")

if __name__ == '__main__':
    main()
