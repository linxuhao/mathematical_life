#!/bin/bash
#!/bin/bash
# Full pipeline: Qwen3.5-2B GSM8K SFT → eval → anchors
set -e

PY=~/pred1-env/bin/python3
MODEL="Qwen/Qwen3.5-2B-Base"
PREFIX="qwen35_2b"
N_PROMPTS=1319
N_ANCHOR=200

echo "=== Step 1: SFT Training ==="
$PY tools/sft_gsm8k.py \
    --model $MODEL \
    --output checkpoints/${PREFIX}_sft/ \
    --batch-size 1 \
    --max-length 512 \
    --save-steps 250

echo "=== Step 2: Full Checkpoint Eval (β₁ + PHI + GSM8K) ==="
$PY tools/dpo_eval.py \
    --checkpoints checkpoints/${PREFIX}_sft/ \
    --base-model $MODEL \
    --output results/${PREFIX}_timeseries.json \
    --n-prompts $N_PROMPTS --n-eval $N_ANCHOR

echo "=== Step 3: Anchor Eval (SVAMP + TruthfulQA) ==="
$PY tools/anchor_eval.py \
    --checkpoints checkpoints/${PREFIX}_sft/ \
    -o results/${PREFIX}_anchors.json \
    --n-eval $N_ANCHOR

echo "=== Done ==="
echo "Results: results/${PREFIX}_timeseries.json + results/${PREFIX}_anchors.json"
