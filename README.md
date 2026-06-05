# Mathematical Life — paper, data, and experiments

Reproduction bundle for the paper *Mathematical Life: A Topological Theory of
Reasoning, and Its Falsification* (Xuhao Lin).

The paper is a four-act **self-falsification**: it states a topological theory of
LLM reasoning as two pre-registered predictions (Act 1), falsifies both under
confound-controlled measurement (Act 2), reformulates the theory from cavity
*count* to manifold *connectivity/routing* (Act 3), and tests that reformulation —
finding it **also null** under controls (Act 4). The durable contribution is
methodological: a reproducible, confound-controlled adjudication with open tooling.

- **Paper:** [`mathematical_life.pdf`](mathematical_life.pdf) (LaTeX source: `mathematical_life.tex`)
- **Measurement library:** [`actopo`](https://github.com/linxuhao/actopo) — `pip install actopo`
  (every β₁ / survival / structural-index number in the paper is produced by it)

## Repository layout

```
mathematical_life.tex / .pdf     the manuscript
references.bib                    bibliography
experiments/
  tools/                         experiment drivers (depend on actopo)
    routing_*.py                 Act 4 routing tests (RT1–RT4)
    threshold_robustness.py      persistence-threshold sweep
    aggressive_*.py / gentle_*   the P2 forgetting / collapse experiments
    *.py                         P1, mode-separation, C3, supplementary probes
  results/                       committed result JSONs (every reported number)
  prompts/                       prompt sets used to generate activations
docs/
  EXPERIMENT_SUMMARY.md          experiment × channel inventory (start here)
  COLLAPSE_EXPERIMENT_DESIGN.md  the P2 off-target-forgetting design
  ROUTING_EXPERIMENT_DESIGN.md   the Act 4 routing design
```

## Reproducing the numbers

1. `pip install actopo` (or `pip install actopo==0.1.0` for the exact paper version).
2. The committed `experiments/results/*.json` back every number in the paper; the
   analysis is re-runnable from the scripts in `experiments/tools/`.
3. Raw activation clouds (`experiments/activations_v3/*.npy`, multi-GB) are **not**
   committed — they regenerate from `actopo` + the prompt sets in `experiments/prompts/`.

`docs/EXPERIMENT_SUMMARY.md` maps each experiment to its scripts, data, and the
paper section it supports.

## Reproducibility note

Headline statistics are deterministic and reproduce from the committed JSONs. A few
intervention figures (notably the Act 4 RT4 ablation) drift mildly under library /
model-snapshot updates while their qualitative null is invariant; the paper reports
the original run and flags this.

## Citation

If you use this work, please cite the paper (Zenodo DOI:
[10.5281/zenodo.20298104](https://zenodo.org/records/20298104)) and the `actopo` package.

## License

Code and data: MIT (see `LICENSE`). The manuscript text and figures are © the author.
