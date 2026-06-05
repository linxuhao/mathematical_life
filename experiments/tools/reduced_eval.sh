#!/bin/bash
# Reduced eval: every Nth checkpoint only
set -e

PY=~/pred1-env/bin/python3
SRC=~/papers/mathematical-life/experiments/checkpoints/qwen35_2b_sft
TMP=~/papers/mathematical-life/experiments/checkpoints/qwen35_2b_sft_reduced
N_PROMPTS=1319
N_ANCHOR=200
STRIDE=2   # every 2nd checkpoint

echo "=== Creating reduced checkpoint set (stride=$STRIDE) ==="
rm -rf "$TMP"
mkdir -p "$TMP"

# Link every Nth checkpoint
count=0
for d in $(ls -d "$SRC"/checkpoint-* | sort -V); do
    num=$(basename "$d" | grep -oP '\d+')
    if (( count % STRIDE == 0 )); then
        ln -s "$d" "$TMP/$(basename "$d")"
        echo "  linked $(basename "$d")"
    fi
    ((count++))
done

echo "Total linked: $(ls "$TMP" | wc -l)"

echo ""
echo "=== Step 1: Topology + GSM8K eval ==="
$PY tools/dpo_eval.py \
    --checkpoints "$TMP" \
    --base-model Qwen/Qwen3.5-2B-Base \
    --output results/qwen35_2b_base_timeseries.json \
    --n-prompts $N_PROMPTS --n-eval $N_ANCHOR

echo ""
echo "=== Step 2: Anchor eval (SVAMP + TruthfulQA) ==="
$PY tools/anchor_eval.py \
    --checkpoints "$TMP" \
    -o results/qwen35_2b_base_anchors.json \
    --n-eval $N_ANCHOR

echo ""
echo "=== Done ==="
echo "Results: results/qwen35_2b_base_timeseries.json"
echo "         results/qwen35_2b_base_anchors.json"
