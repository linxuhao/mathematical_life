# Experiment Design — Is Catastrophic Forgetting Topological Collapse?
*(v1, 2026-06-02. Built on session lessons: off-target measurement, threshold-sensitivity,
partial-correlation-controlling-step, length-normalization, C3 comprehension probe.)*

## 0. The question (the genesis)
Why can't we leave learning on at deployment? Hypothesis: global back-prop is a "hammer"
that, while producing local gains on the trained topic A, inflicts **collateral damage on
untrained regions B** — observed behaviorally as catastrophic forgetting. The paper's claim
is that this damage is **topological collapse** of B's manifold. Our past SFT (train A,
measure A) was blind to it: local gain masks local damage. **Fix: train A, measure off-target B.**

## 1. What we measure (outcome variables), per checkpoint t, per domain D
1. **Behavioral**: `acc_D(t)` — benchmark accuracy on D.
2. **Topological** (actopo, FROZEN_V5, L/2, last-token), on D's activation cloud:
   - β₁ across a **scale curve** {raw, 1%, 3%, 5%, 10% of ε_max} — NOT a single cutoff (threshold-sensitivity lesson).
   - **survival_rate / mean persistence** = loop *fragility* (the metric that actually moved for 9B).
   - β₀ at 20–30% ε_max (coarse mode/cluster structure).
3. **Comprehension** (C3, the activation-encodes-understanding readout): length-normalized
   probe AUC — can D's *pre-reasoning* prompt activation predict per-item success on D?
   I.e. does the model still "understand/size-up" D-problems. **Candidate cleaner forgetting
   detector than β₁.**
4. **Distributional control**: output entropy + answer-format bias on B (to separate
   "lost B-representation" from "just outputs A-style now").

## 2. Variables
**Manipulated (IV):**
- **Learning rate**: {gentle, AGGRESSIVE}. Aggressive arm ensures the hammer actually lands
  (our prior gentle SFT stayed flat → under-powered).
- **Training time**: dense checkpoint series (forgetting is often fast/early).
- *(optional causal arm)* **update type**: {full-FT, LoRA, replay-mix} — does reducing
  δ_learning reduce forgetting AND topology loss together?

**Constants (controlled):**
- **Single model** (Qwen3.5-2B-base; real β₁=51, fully pretrained) → within-model, so no
  size confound for the "max-HP" test.
- Train domain **A = GSM8K** (narrow, distinct from all B).
- Off-target **B-domains** (≥3, chosen to span DIFFERENT initial topology): e.g.
  commonsense (PIQA/HellaSwag), factual recall (TriviaQA), language/cloze (LAMBADA).
- FROZEN_V5 topology protocol; fixed eval prompts/format; fixed activation prompt sets per
  domain; fixed seed.

## 3. Confound controls (hard-won this session)
| Confound | Control |
|---|---|
| Local gain masks damage | measure **off-target B**, never just A |
| β₁ threshold-fragility | report β₁ **across scales** + persistence, not one ε |
| Training-step drives everything | **partial correlation controlling t** for any topology↔capability claim |
| Correlation ≠ cause | **lead-lag**: does Δtopology *precede* Δacc (Prediction 2)? |
| Length/difficulty | **length-normalized** comprehension probe (C3) |
| Distributional vs representational | measure output entropy/bias on B |
| Size confound for "max-HP" | **within-model**, compare across B-domains |
| Raw vs normalized | report **both**, trust normalized |

## 4. Test methods / analysis
- **Q1 — does the hammer land (forgetting exists)?** slope of `acc_B(t)` < 0, significant.
  (Gentle arm: does realistic LR cause it? Aggressive: confirm the mechanism is real.)
- **Q2 — topological correlate?** slope of `persistence_B(t)`, `β₁_B(t)`, `compreh_B(t)` < 0.
- **Q3 — is topology LOAD-BEARING or epiphenomenal? (the crux)**
  `partial_corr(topology_B, acc_B | t)`. ≈0 ⇒ both just decline with t independently
  (epiphenomenal — our prior, given β₁≠capability). >0 ⇒ coupled.
  PLUS **lead-lag** cross-correlation of ΔtopologyB vs ΔaccB: topology dropping *first* = P2 support.
- **Q4 — "max-HP" (damage ∝ existing topology)?** across B-domains, regress
  forgetting_magnitude(Bᵢ) on **initial topology**(Bᵢ). Positive slope ⇒ HP hypothesis.
- **Q5 — which signal best detects forgetting?** compare how tightly `acc_B` tracks
  β₁_B vs persistence_B vs **comprehension_B** (C3). (Bet: comprehension > β₁.)
- **Q6 — causal (optional arm)**: dose-response (aggressive forgets > gentle) + does
  LoRA/replay reduce forgetting AND topology loss in lockstep?

## 5. Pre-registered predictions & decision rules
- **Topological-collapse-causes-forgetting (H):** acc_B↓ AND topology_B↓ AND
  partial(topo_B,acc_B|t)>0 with **topology leading** AND forgetting∝HP.
- **Epiphenomenal/null (H0, the strong prior):** acc_B↓ (forgetting real) but
  partial(topo_B,acc_B|t)≈0 OR topology_B flat. Topology is a bystander.
- **Distributional (H_dist):** acc_B↓ with output-entropy/bias shift but B-representation
  (topology + comprehension) intact.
- **Falsify topological mechanism if:** B forgets while β₁_B/persistence_B unchanged, OR
  they drop but partial|t ≈ 0 (no leading coupling).

## 6. Interpretation matrix
| acc_B | topo_B | partial\|t & lead | meaning |
|---|---|---|---|
| ↓ | ↓ | >0, topo leads | **topological collapse supported** (real result) |
| ↓ | ↓ | ≈0 | correlate, not cause (epiphenomenal) |
| ↓ | flat | — | forgetting is NOT topological (distributional) |
| flat | — | — | δ too small — increase LR (the hammer didn't land) |

## 7. Feasibility (single researcher, consumer GPU)
2B model + GSM8K SFT (pipeline exists) + standard B-benchmarks (loadable) + actopo topology
on B-activation sets + reusable C3 probe. ~checkpoint series (hours), resumable. The
aggressive-LR arm is the cheap first probe: if even hard hammering leaves B-topology intact
while B-acc falls, the topological-collapse thesis is in serious trouble — and that itself is
a clean, publishable result.

## 8. Honest expectation
Given β₁≠capability and the threshold-fragility of "collapse," the strong prior is **H0**:
B *will* forget (Q1 yes), but topology will likely be a **bystander** (Q3 partial≈0), with
**comprehension (C3)** possibly the one activation signal that tracks forgetting — because it
measures problem-understanding, which is what's actually being lost. Either outcome is a
clean answer; the design is built so we can't fool ourselves about which.
