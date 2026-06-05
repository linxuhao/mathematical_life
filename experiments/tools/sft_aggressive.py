#!/usr/bin/env python3
"""AGGRESSIVE arm of the collapse experiment — swing the global hammer.

Identical training recipe to the gentle SFT (sft_gsm8k.py: same GSM8K "Q:/A:"
format, same model = Qwen3.5-2B-Base, fp32 compute, grad-clip 1.0) so the ONLY
changed variable vs the gentle arm is the learning rate (1e-5 -> 5e-5) and the
length of training. Goal: induce catastrophic forgetting on UNRELATED domains
(ARC-Easy, HellaSwag) so we can test whether topology tracks the damage.

Checkpoints saved in bf16 (state_dict cast, live model stays fp32 so optimizer
is untouched) -> ~3.76GB each, matching the gentle 'reduced' ckpts.
Resumable: skips existing checkpoint-* dirs.
"""
import argparse, os, torch, random
from pathlib import Path
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3.5-2B-Base')
    ap.add_argument('--output', default='experiments/checkpoints/qwen35_2b_aggressive')
    ap.add_argument('--n-epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--save-steps', type=int, default=400)   # ~28 ckpts over 3 epochs
    ap.add_argument('--lr', type=float, default=5e-5)        # AGGRESSIVE: 5x gentle
    ap.add_argument('--max-length', type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    done = {int(d.name.split('-')[1]) for d in out_dir.glob('checkpoint-*')}
    if done: print(f"resuming, already have steps: {sorted(done)}", flush=True)

    ds = load_dataset('openai/gsm8k', 'main', split='train')
    texts = [f"Q: {it['question']}\nA: {it['answer']}" for it in ds]
    print(f"GSM8K train: {len(texts)} examples, lr={args.lr}, {args.n_epochs} epochs", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation='eager', trust_remote_code=True).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(texts) // args.batch_size * args.n_epochs
    sched = get_linear_schedule_with_warmup(opt, 50, total)

    def save_bf16(step):
        ck = out_dir / f'checkpoint-{step}'; ck.mkdir(exist_ok=True)
        sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        model.save_pretrained(ck, state_dict=sd)
        tok.save_pretrained(ck)
        print(f"  step {step}: saved (loss={loss.item():.4f})", flush=True)

    gstep = 0
    for ep in range(args.n_epochs):
        idx = list(range(len(texts))); random.shuffle(idx)
        for i in range(0, len(texts), args.batch_size):
            bt = [texts[idx[j]] for j in range(i, min(i+args.batch_size, len(texts)))]
            if len(bt) < args.batch_size: continue
            t = tok(bt, truncation=True, max_length=args.max_length,
                    padding='max_length', return_tensors='pt')
            labels = t['input_ids'].clone(); labels[labels == tok.pad_token_id] = -100
            batch = {k: v.cuda() for k, v in t.items()}; batch['labels'] = labels.cuda()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            gstep += 1
            if gstep % args.save_steps == 0 and gstep not in done:
                save_bf16(gstep)
            if gstep % 100 == 0:
                print(f"  step {gstep}/{total}: loss={loss.item():.4f}", flush=True)
    save_bf16(gstep)
    print(f"\nDONE final step {gstep}", flush=True)

if __name__ == '__main__':
    main()
