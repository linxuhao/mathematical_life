#!/usr/bin/env python3
"""DPO training v3 — precomputed reference logprobs + numerical stability.

Usage:
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  python tools/dpo_train.py --output checkpoints/dpo_05b/
"""
import os, argparse, torch, json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'expandable_segments:True')

def compute_logprobs(model, input_ids):
    """Average log-prob per token. Numerically stable version."""
    outputs = model(input_ids)
    logits = outputs.logits.float()
    # Shift: predict token[t] from token[t-1]
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    # Log softmax
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    # Gather per-token log-probs
    per_token = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    # Mask padding tokens
    mask = (shift_labels != 151643).float()  # Qwen pad_token_id
    # Fix: -inf * 0 = NaN. nan_to_num prevents this.
    masked = torch.nan_to_num(per_token * mask, nan=0.0, posinf=0.0, neginf=0.0)
    token_count = mask.sum(-1).clamp(min=1)
    return masked.sum(-1) / token_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--output', default='checkpoints/dpo_05b')
    parser.add_argument('--n-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--max-prompts', type=int, default=5000)
    parser.add_argument('--save-steps', type=int, default=125)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--grad-accum', type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HH-RLHF...")
    ds = load_dataset('Anthropic/hh-rlhf', split='train')
    ds = ds.select(range(min(len(ds), args.max_prompts)))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token_id = 151643

    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda()

    reference = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation='eager', trust_remote_code=True
    ).cuda()
    reference.eval()

    # ---- Pre-compute reference logprobs (once, saves time + memory) ----
    print("Pre-computing reference logprobs...")
    ref_logps_c = []
    ref_logps_r = []
    for i in tqdm(range(0, len(ds), args.batch_size)):
        batch = ds[i:i+args.batch_size]
        c_tok = tokenizer(batch['chosen'], truncation=True, max_length=args.max_length,
                         padding='max_length', return_tensors='pt')
        r_tok = tokenizer(batch['rejected'], truncation=True, max_length=args.max_length,
                         padding='max_length', return_tensors='pt')
        with torch.no_grad():
            lp_c = compute_logprobs(reference, c_tok['input_ids'].cuda())
            lp_r = compute_logprobs(reference, r_tok['input_ids'].cuda())
        ref_logps_c.append(lp_c.cpu())
        ref_logps_r.append(lp_r.cpu())
    ref_c = torch.cat(ref_logps_c)
    ref_r = torch.cat(ref_logps_r)
    print(f"  Reference logprobs: chosen mean={ref_c.mean():.2f}, rejected mean={ref_r.mean():.2f}")
    ref_ratio = ref_c - ref_r
    print(f"  Reference ratio mean={ref_ratio.mean():.2f}")

    del reference
    torch.cuda.empty_cache()

    # ---- Training ----
    def tokenize_fn(examples):
        c_tok = tokenizer(examples['chosen'], truncation=True, max_length=args.max_length,
                         padding='max_length')
        r_tok = tokenizer(examples['rejected'], truncation=True, max_length=args.max_length,
                         padding='max_length')
        return {'c_ids': c_tok['input_ids'], 'r_ids': r_tok['input_ids']}

    ds = ds.map(tokenize_fn, batched=False, with_indices=False)
    # Add index column for reference ratio lookup
    ds = ds.add_column('idx', list(range(len(ds))))
    ds.set_format(type='torch')

    def collate(batch):
        c = torch.stack([b['c_ids'] for b in batch])
        r = torch.stack([b['r_ids'] for b in batch])
        idx = torch.tensor([b['idx'] for b in batch])
        return c, r, idx

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    total_steps = (len(loader) * args.n_epochs) // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, 50, total_steps)

    print(f"Training: {len(ds)} examples, effective batch={args.batch_size*args.grad_accum}")
    global_step = 0
    batch_count = 0
    policy.train()

    for epoch in range(args.n_epochs):
        for c_ids, r_ids, indices in loader:
            c_ids, r_ids = c_ids.cuda(), r_ids.cuda()

            pi_c = compute_logprobs(policy, c_ids)
            pi_r = compute_logprobs(policy, r_ids)

            ref_ratio_batch = ref_ratio[indices].cuda()
            logits_diff = args.beta * (pi_c - pi_r - ref_ratio_batch)
            logits_diff = torch.clamp(logits_diff, -20, 20)

            loss = -torch.nn.functional.logsigmoid(logits_diff).mean()
            loss = loss / args.grad_accum
            loss.backward()

            batch_count += 1

            if batch_count % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                loss_val = loss.item() * args.grad_accum
                if global_step % args.save_steps == 0:
                    ckpt_dir = out_dir / f'checkpoint-{global_step}'
                    ckpt_dir.mkdir(exist_ok=True)
                    policy.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"  Step {global_step}: loss={loss_val:.4f} saved → {ckpt_dir.name}")

                if global_step % 25 == 0:
                    print(f"  Step {global_step}: loss={loss_val:.4f}  pi_c={pi_c.mean().item():.2f}")

    final_dir = out_dir / 'final'
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nDone: {final_dir}")

if __name__ == '__main__':
    main()
