#!/usr/bin/env python3
"""Is PHI (and the β₁ gauges) a TRAINING THERMOMETER?

Not "does PHI slope with step", but: do the topology gauges read whether the
model keeps ENOUGH (足够多 → β₁_R) and ROBUST (坚挺 → surv_R / PHI) reasoning
circuits — and do they track real reasoning health (OOD generalization) rather
than the training target (GSM8K)?

For each model we merge, by step:
  - topology gauges (actopo, correct extraction): β₁_R, β₁_H, surv_R, surv_H, PHI
  - capability (training target): GSM8K (reliable generation eval)
  - OOD reasoning generalization: SVAMP (NOT trained on)
  - truthfulness: TruthfulQA

Gauges:  PHI, β₁_R, gap=β₁_R−β₁_H, RCH=β₁_R·surv_R/100 (effective robust loops)
Validity: a gauge is a thermometer if it correlates with OOD (SVAMP) more than
          with the memorized target (GSM8K).  r reported with n.
"""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"


def load(name):
    d = json.load(open(RES / name))
    return d if isinstance(d, list) else d.get("data", d)


def by_step(rows, *fields):
    out = {}
    for r in rows:
        out[r["step"]] = {f: r.get(f) for f in fields}
    return out


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3:
        return None, int(m.sum())
    return round(float(np.corrcoef(xs[m], ys[m])[0, 1]), 3), int(m.sum())


def analyse(name, topo_json, gsm_json, anchors_json, gsm_field="gsm8k_acc"):
    topo = by_step(load(topo_json), "beta1_r", "beta1_h", "surv_r", "surv_h", "phi")
    gsm = by_step(load(gsm_json), gsm_field) if gsm_json else {}
    anc = by_step(load(anchors_json), "svamp_acc", "truthfulqa_acc") if anchors_json else {}

    steps = sorted(topo)
    rows = []
    for s in steps:
        t = topo[s]
        if None in (t["beta1_r"], t["beta1_h"], t["surv_r"], t["surv_h"]):
            continue
        rows.append({
            "step": s,
            "phi": t["phi"],
            "beta1_r": t["beta1_r"],
            "gap": t["beta1_r"] - t["beta1_h"],
            "rch": round(t["beta1_r"] * t["surv_r"] / 100, 2),
            "surv_r": t["surv_r"],
            "gsm": gsm.get(s, {}).get(gsm_field),
            "svamp": anc.get(s, {}).get("svamp_acc"),
            "tqa": anc.get(s, {}).get("truthfulqa_acc"),
        })

    def col(k):
        return [r[k] for r in rows]

    print(f"\n{'='*78}\n{name}   ({len(rows)} checkpoints, steps {rows[0]['step']}–{rows[-1]['step']})\n{'='*78}")
    # ---- levels (the reading) ----
    for g in ("phi", "beta1_r", "gap", "rch", "surv_r"):
        v = np.array([x for x in col(g) if x is not None], float)
        print(f"  {g:<9s} start={v[0]:6.2f}  end={v[-1]:6.2f}  mean={v.mean():6.2f}±{v.std():.2f}")

    # ---- trajectory (is it eroding?) ----
    st = col("step")
    print("  trajectory r(step,·):  " + "  ".join(
        f"{g}={pearson(st, col(g))[0]}" for g in ("beta1_r", "gap", "phi", "rch")))

    # ---- INSTRUMENT VALIDITY: gauge vs target ----
    print("  instrument validity  r(gauge, target)   [n]")
    print(f"    {'gauge':<9s} {'GSM8K(target)':>16s} {'SVAMP(OOD)':>14s} {'TruthfulQA':>14s}")
    for g in ("phi", "beta1_r", "gap", "rch"):
        r_gsm = pearson(col(g), col("gsm"))
        r_sv = pearson(col(g), col("svamp"))
        r_tq = pearson(col(g), col("tqa"))
        print(f"    {g:<9s} {str(r_gsm[0]):>12s}[{r_gsm[1]:>2d}] {str(r_sv[0]):>10s}[{r_sv[1]:>2d}] {str(r_tq[0]):>10s}[{r_tq[1]:>2d}]")
    return rows


def maybe(name, topo, gsm, anchors, **kw):
    if not (RES / topo).exists():
        print(f"\n[skip {name}: {topo} not present locally yet]")
        return
    analyse(name, topo, gsm, anchors, **kw)


if __name__ == "__main__":
    maybe("Qwen2.5-0.5B-Base SFT",
          "sft_gsm8k_actopo.json", "sft_gsm8k_timeseries.json", "anchors.json")
    maybe("Qwen3.5-2B-Base SFT",
          "qwen35_2b_base_actopo.json", "qwen35_2b_base_timeseries.json", "anchors_full.json")
    # Fixed instruct run (chat template + embedded per-checkpoint SVAMP/TQA anchors).
    # GSM8K, SVAMP, TruthfulQA all live in the same file → dense OOD for the
    # instrument-validity test (gauge vs OOD), finally available for instruct.
    maybe("Qwen3.5-2B-Instruct SFT (chat, fixed)",
          "qwen35_2b_inst_actopo_chat.json", "qwen35_2b_inst_actopo_chat.json",
          "qwen35_2b_inst_actopo_chat.json")
