#!/bin/bash
set -e
cd ~/papers/mathematical-life/experiments

echo "============================================"
echo "SMOKE TEST: 2 models, reasoning only"
echo "============================================"

echo ""
echo "--- Step 1: Extract activations ---"
~/pred1-env/bin/python3 tools/extract_activations.py \
    --models config/models_smoke.json \
    --prompts prompts/ \
    -o activations_smoke/ \
    --prompt-types reasoning \
    --gpus 0,1

echo ""
echo "--- Step 2: Compute betti ---"
~/pred1-env/bin/python3 tools/compute_betti.py \
    -i activations_smoke/ \
    -o results/smoke_betti.json \
    --workers 1

echo ""
echo "--- Step 3: Verify against expected ---"
~/pred1-env/bin/python3 -c "
import json, sys
with open('results/smoke_betti.json') as f:
    data = json.load(f)
expected = {'Qwen3.5-0.8B-base_L12_reasoning': 52, 'Qwen3.5-0.8B-Inst_L12_reasoning': 12}
all_ok = True
for name, entry in data.items():
    b1 = entry.get('beta1', None)
    err = entry.get('error', None)
    exp = expected.get(name, None)
    if err:
        print(f'  {name}: ERROR - {err}')
        all_ok = False
    elif b1 is not None and exp is not None:
        status = 'OK' if b1 == exp else 'MISMATCH'
        if status != 'OK':
            all_ok = False
        print(f'  {name}: b1={b1} (expected {exp}) [{status}]')
    else:
        print(f'  {name}: b1={b1}')
if all_ok:
    print('SMOKE TEST PASSED')
    sys.exit(0)
else:
    print('SMOKE TEST FAILED')
    sys.exit(1)
"

echo ""
echo "============================================"
echo "SMOKE TEST COMPLETE"
echo "============================================"
