# Experiments

All measurement goes through the [`actopo`](https://github.com/linxuhao/actopo)
package (`pip install actopo`) under one frozen protocol (`FROZEN_V5`): middle
layer, last non-padding token, Vietoris–Rips persistent homology, β₁ at
ε = 0.03·ε_max. The scripts here are experiment *drivers* that call actopo.

```
tools/         experiment drivers (Python/shell)
results/       committed result JSONs — every number reported in the paper
prompts/       prompt sets used to generate activation clouds
config/        model/pair lists consumed by the drivers
activations_v3/ small label files; the large .npy/.npz activation clouds are
               NOT committed (regenerate from actopo + prompts/)
```

Full experiment × channel inventory: [`../docs/EXPERIMENT_SUMMARY.md`](../docs/EXPERIMENT_SUMMARY.md).

## Paper claim → driver → result

| Paper section | Claim | Driver (`tools/`) | Result (`results/`) |
|---|---|---|---|
| §P1 | β₁ does not track capability | `compute_betti.py`, `reanalyze_tokenwise.py` | `betti_v3.json`, `tokenwise_reanalysis.json` |
| §P1 (threshold) | null survives the persistence-cutoff sweep | `threshold_robustness.py` | `threshold_robustness.json` |
| §P2 | topological collapse is a bystander to forgetting | `sft_aggressive.py`, `aggressive_eval.py`, `gentle_arm.py` | `aggressive_arm.json`, `gentle_arm.json` |
| §P2 (calibration) | forgetting’s signature is over-confidence, not β₁ | `aggressive_confidence.py` | `aggressive_confidence.json` |
| §survives | cognitive modes occupy separable regions | `modesep_eps.py` | `modesep_eps.json` |
| §status (C3) | pre-generation hidden state predicts correctness | `C3_sameq_gen.py`, `C3_sameq_analyze.py` | `C3_sameq.json`, `C3_control1.json` |
| §P1 (supp.) | aha / thermometer / TruthfulQA supplementary nulls | `aha_curve_analysis.py`, `thermometer_analysis.py`, `tqa_diagnostic.py` | `aha_curves.json` |
| §routing | Act 4 routing test — also null under controls | `routing_test.py` + `routing_stage1_run.py`, `routing_rt2_run.py`, `routing_rt3_run.py`, `routing_rt4_run.py` | `routing_s1.json`, `routing_rt2.json`, `routing_rt3.json`, `routing_s2.json` |

## Reproducing

1. `pip install actopo` (pin `actopo==0.1.0` for the exact paper version).
2. The committed `results/*.json` already back every reported number; analysis
   scripts re-run from them directly.
3. To regenerate from scratch, the drivers rebuild activation clouds from
   `prompts/` via actopo; expect multi-GB intermediates (gitignored).

A few intervention figures (notably the Act 4 RT4 ablation) drift mildly under
library/model-snapshot updates while their qualitative null is invariant; the
paper reports the original run.
