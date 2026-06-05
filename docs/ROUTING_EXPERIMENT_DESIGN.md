# Routing / Highway Experiment — Pre-Registered Design

Built 2026-06-04. Tests the **routing interpretation** of the topological skeleton, which the
correlational P1 test (β₁-count vs capability) does **not** address. Decision context:
`/goal` audit established that the P1 null (β₁ magnitude ⊥ capability) is *consistent with* the
highway hypothesis and therefore cannot falsify it. This document specifies the test that can.

## 0. The hypothesis (sharpened)

β₁ (and the skeleton) is **infrastructure, not a score**. The β₁ *count* does not index reasoning
capacity (confirmed: P1 null, §empirical). Instead, three distinct observables carry the claim:

| Obs | Topological quantity | Role | "Highway" gloss |
|-----|----------------------|------|-----------------|
| **R1** | β₀ connectivity: are premise-region A and conclusion-region B in the same component at scale ε? | **capacity** | no highway A→B ⇒ that inference is impossible |
| **R2** | geodesic length of the A→B route through the manifold (graph-geodesic; detour factor d_geo/d_euclid; merge-scale) | **quality** | short highway ⇒ reliable; long detour ⇒ error-prone |
| **R3** | β₁ redundancy: does the A–B routing region contain a persistent 1-cycle (≥2 independent routes)? | **robustness** | redundant highways ⇒ resilient to damage |
| **R4** | causal: ablate the directions that hold the A→B route open | **causal capacity** | close the highway ⇒ only route-dependent inferences fail |

**Master claim:** reasoning is topology-*constrained*. Capacity is set by route existence (R1),
quality by route length (R2), robustness by route redundancy (R3); ablating routes selectively
destroys dependent inferences (R4). The count of cavities is the wrong summary; the **connectivity
structure** is the right one.

## 1. Why this is not P1

P1 tested `corr(β₁_count, benchmark_score)` across models → null. Under the highway hypothesis
that null is **expected**: aggregate cavity count says nothing about whether the *specific* route a
given inference needs exists. So P1 cannot discriminate "topology irrelevant" from "topology routes
reasoning." This program tests the per-inference routing structure directly, with the same
confound-control discipline used elsewhere (partial-on-difficulty, length-norm, Euclidean control,
matched-ablation controls).

---

## 2. Stage 1 — Observational: does routing predict reasoning? (R1 + R2)

**Fully runnable now** (`experiments/tools/routing_test.py`, Stage 1). No retraining; forward passes
+ CPU graph analysis.

### Setup
- **Model**: Qwen3.5-2B-Base (workhorse; replicate on Qwen3.5-0.8B). FROZEN_V5, L/2, last-token, bf16.
- **Road network** G: kNN graph (k=15, Euclidean) over the 1319-prompt reasoning activation cloud at
  L/2 — the manifold's "existing roads."
- **Per item** i (GSM8K, with correctness label from generation):
  - premise anchor A_i = L/2 activation at the last token of the problem statement.
  - conclusion anchor B_i = L/2 activation at the last token of (problem + reference solution).
  - insert A_i, B_i into G; compute:
    - **R1** `route_exists`(ε): A_i, B_i in same connected component of the Rips graph at scale ε
      (sweep ε ∈ {1,3,5,10,20}%·ε_max).
    - **R2** `d_geo` = shortest-path length A_i→B_i through G (Dijkstra); `detour = d_geo / d_euclid`;
      `merge_eps` = smallest ε at which A_i,B_i connect.

### Variables
- **IV / predictors**: route_exists, d_geo, detour, merge_eps.
- **Controls (must be partialled out)**: question difficulty (token length, and an independent
  difficulty proxy), raw Euclidean ‖A_i − B_i‖, anchor norms.
- **DV**: per-item correctness (binary); secondary: solution self-consistency / confidence.

### Analysis
- 5-fold CV ROC-AUC for correctness from {Euclidean + difficulty} (baseline) vs
  {+ routing features}. Report **ΔAUC** and partial correlation of each routing feature with
  correctness | (difficulty, Euclidean).

### Pre-registered outcomes
- **Routing CONFIRMED** if routing features add ΔAUC ≥ 0.03 (bootstrap CI excludes 0) **and**
  `route_exists`/`detour` retain a significant partial correlation with correctness after
  controlling difficulty + Euclidean distance.
- **Routing FALSIFIED (Stage 1)** if ΔAUC CI includes 0 and all routing partials |r| < 0.1 — i.e.,
  graph connectivity/geodesic carries no information beyond straight-line distance and difficulty.
- **Decision rule guard**: because C3 already shows position carries comprehension signal at
  length-controlled AUC ≈ 0.62, the *novel* test here is the **geodesic increment over Euclidean** —
  if geodesic ≈ Euclidean in predictive power, the manifold's *graph* structure is not used, only
  raw distance is, and the "highway" adds nothing.

---

## 3. Stage 2 — Causal: route ablation selectively kills dependent inferences (R4 + R3)

The decisive test. Harder (route→direction attribution); specified here, scaffolded in code.

### Bridge-direction attribution
For a target A→B pair (or a concept-pair cluster), take the shortest-path waypoints in G. The route
occupies a low-dim subspace; estimate it as the top-q principal directions of the covariance of the
waypoint activations (q swept 1–16). These are the **bridge directions**.

### Intervention
Forward hook on the L/2 block output: project out (zero) the bridge directions from the hidden
state, for all tokens, during benchmark eval. (`routing_test.py`, `ablate_directions`.)

### Conditions (each ablates the same number of directions / same removed variance)
1. **bridge** (route-critical directions).
2. **random** directions (matched count).
3. **norm/variance-matched** directions (control for "bridge dirs are just high-variance"): pick
   non-bridge directions matching the bridge set's removed variance. *This is the key confound.*
4. **dose sweep**: q ∈ {1,2,4,8,16} for dose-response.

### DV (the dissociation)
- **reasoning**: GSM8K, ARC accuracy (route-dependent multi-step).
- **fluency / pattern-matching**: LAMBADA last-word acc; perplexity on held-out narrative;
  HellaSwag. (The skeleton claim: ablation hurts reasoning, spares fluency.)
- per condition: Δreasoning, Δfluency.

### R3 redundancy sub-test
Split items by whether their routing region contains a persistent 1-cycle (β₁ ≥ 1 on the
waypoint+neighborhood cloud). Prediction: redundant-route items resist single-bridge ablation;
unique-route items fail. (Tests that β₁ loops = robustness, the one role left for β₁.)

### Pre-registered outcomes
- **Routing CONFIRMED (causal)** if bridge ablation produces Δreasoning ≥ 2σ of the random-ablation
  Δreasoning **and** |Δfluency_bridge − Δfluency_random| < 1σ (selectivity), dose-dependent, and the
  R3 redundancy split holds (redundant items more robust, one-sided test).
- **Routing FALSIFIED (causal)** if bridge ablation degrades reasoning no more than
  norm-matched control, **or** degrades fluency as much as reasoning (no dissociation). This is the
  ablation the paper named as "Prediction 1 in operational form" (§geometry) and never ran.

---

## 4. Confounds & guards (project methodology)
- Always include the **norm/variance-matched** ablation control (Stage 2) and the **Euclidean +
  difficulty** baseline (Stage 1) — the routing signal must survive *beyond* "just distance" and
  "just important/high-variance directions."
- Bootstrap sensor noise: β₀/β₁ on small waypoint clouds is noisy; report CIs (≥1000 resamples).
- Replicate across ≥2 model sizes; do not generalize beyond the families tested.
- Iron Law: never mix base/instruct clouds when building G.

## 5. What each outcome means for the paper
- **Stage 1 confirms + Stage 2 confirms** → the highway ontology survives; β₁-count was the wrong
  observable; reframe the paper around *connectivity-as-routing* (a positive result). The P1/P2 nulls
  become "we falsified the wrong summary statistic; here is the right one."
- **Stage 1 null** → routing structure is not used beyond raw distance; the highway reading joins P1
  as falsified, and the paper's "wrong geometric readout" claim is then fully earned.
- **Stage 1 confirm, Stage 2 null** → connectivity *predicts* but is not *causal* (epiphenomenal,
  like β₁ for forgetting) — a nuanced, publishable middle.

## 6. Status
Stage 1 script: runnable (server). Stage 2: scaffolded; bridge-attribution to validate on a
synthetic 2-hop control set first (`--synthetic` builds A→B→C chains with known routes). Until run,
the paper marks the routing claim **UNTESTED** and cites this protocol as the decisive test.
