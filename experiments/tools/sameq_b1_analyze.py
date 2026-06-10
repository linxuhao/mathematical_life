"""Within-question cumulative-beta1 analysis (CPU) — the pre-registered
adjudication of the precursor hypothesis ("thinking wrong before the visible
error"). Reads results/sameq_acts/*.npz + C3_sameq_v2.json.

Per trajectory, for k in {128, 192, 256} and FULL: cloud = first k token
activations (matched cloud size by construction); beta1_raw = H1 count,
beta1 = count with lifetime > 0.03 * eps_max(cloud) (frozen-protocol filter).

PRE-REGISTERED primary endpoint: within-question-centered GroupKFold(5) AUC of
filtered beta1 at matched k. Verdict: >=0.60 sustained across k -> precursor
real; <0.55 -> excluded; else inconclusive. Secondary: raw beta1 AUC; Wilcoxon
over per-question (mean correct - mean incorrect); full-traj beta1 partial of
correctness given n_gen, within question. Anchor: centered dec_act PCA50 probe
(expect ~0.58, B14).
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, pearsonr

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "C3_sameq_v2.json"
ACTS = ROOT / "results" / "sameq_acts"
KS = [128, 192, 256]

def beta1(cloud):
    from sklearn.metrics import pairwise_distances
    from ripser import ripser
    eps_max = float(np.max(pairwise_distances(cloud)))
    dgm = ripser(cloud.astype(np.float32), maxdim=1)["dgms"][1]
    filt = int(sum(1 for b, d in dgm if (d - b) > 0.03 * eps_max))
    return len(dgm), filt

res = json.load(open(SRC))
rows = []  # qid, si, correct, n_gen, {k: (raw, filt)}, full(raw, filt)
for f in sorted(ACTS.glob("q*.npz"), key=lambda p: int(p.stem[1:])):
    key = f.stem[1:]
    z = np.load(f)
    for si in range(res[key]["K"]):
        if f"s{si}" not in z:
            continue
        h = z[f"s{si}"].astype(np.float32)
        row = {"qid": key, "si": si, "correct": bool(z[f"s{si}_correct"]),
               "n_gen": len(h), "b1": {}}
        for k in KS:
            if len(h) >= k:
                row["b1"][k] = beta1(h[:k])
        row["b1"]["full"] = beta1(h)
        rows.append(row)
print(f"{len(rows)} trajectories from {len(set(r['qid'] for r in rows))} mixed questions")

def centered_auc(feat_rows, label="?"):
    """within-question centering -> GroupKFold(5) logistic AUC."""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = np.array([r["_x"] for r in feat_rows], float).reshape(-1, 1)
    y = np.array([r["correct"] for r in feat_rows])
    g = np.array([r["qid"] for r in feat_rows])
    # center within question
    for q in set(g):
        X[g == q] -= X[g == q].mean(axis=0)
    aucs = []
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(set(y[te])) < 2:
            continue
        p = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))

report = {}
print(f"\n{'k':>6} | {'n':>4} | filt AUC (centered) | raw AUC | Wilcoxon p (per-Q filt)")
for k in KS + ["full"]:
    sub = [dict(r, _x=r["b1"][k][1]) for r in rows if k in r["b1"]]
    if len(sub) < 50:
        print(f"{k:>6} | too few ({len(sub)})"); continue
    a_f, s_f = centered_auc(sub)
    sub_r = [dict(r, _x=r["b1"][k][0]) for r in rows if k in r["b1"]]
    a_r, _ = centered_auc(sub_r)
    # per-question paired test on filtered beta1
    dq = []
    for q in set(r["qid"] for r in sub):
        qs = [r for r in sub if r["qid"] == q]
        c = [r["_x"] for r in qs if r["correct"]]; i = [r["_x"] for r in qs if not r["correct"]]
        if c and i:
            dq.append(np.mean(c) - np.mean(i))
    wp = wilcoxon(dq).pvalue if len(dq) > 10 and any(d != 0 for d in dq) else float("nan")
    print(f"{k:>6} | {len(sub):>4} | {a_f:.3f}±{s_f:.3f}        | {a_r:.3f}   | {wp:.3f} "
          f"(median Δ={np.median(dq):+.1f}, nQ={len(dq)})")
    report[str(k)] = {"n": len(sub), "auc_filt": a_f, "auc_filt_std": s_f,
                      "auc_raw": a_r, "wilcoxon_p": wp,
                      "median_dq": float(np.median(dq)), "n_q": len(dq)}

# full-traj partial: corr(b1_filt, correct | n_gen), all within-question centered
sub = [r for r in rows if "full" in r["b1"]]
if len(sub) > 50:
    b = np.array([r["b1"]["full"][1] for r in sub], float)
    c = np.array([float(r["correct"]) for r in sub])
    n = np.array([float(r["n_gen"]) for r in sub])
    g = np.array([r["qid"] for r in sub])
    for q in set(g):
        idx = g == q
        b[idx] -= b[idx].mean(); c[idx] -= c[idx].mean(); n[idx] -= n[idx].mean()
    def resid(x, z):
        return x - np.poly1d(np.polyfit(z, x, 1))(z)
    pr, pp = pearsonr(resid(b, n), resid(c, n))
    report["partial_full_b1_correct_given_ngen"] = {"r": float(pr), "p": float(pp)}
    print(f"\nfull-traj partial r(b1_filt, correct | n_gen, within-Q) = {pr:+.3f} (p={pp:.4f})")

# anchor: dec_act PCA50 centered probe (B14 expects ~0.58)
try:
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X, y, g = [], [], []
    for key, v in res.items():
        if not (0 < v["n_correct"] < v["K"]):
            continue
        for c in v["comps"]:
            if c.get("status") == "ok" and "dec_act" in c:
                X.append(c["dec_act"]); y.append(c["correct"]); g.append(key)
    X = np.array(X, float); y = np.array(y); g = np.array(g)
    for q in set(g):
        X[g == q] -= X[g == q].mean(axis=0)
    X = PCA(n_components=50).fit_transform(X)
    aucs = []
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(set(y[te])) < 2:
            continue
        p = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    report["anchor_dec_act"] = float(np.mean(aucs))
    print(f"\nANCHOR dec_act centered PCA50 AUC = {np.mean(aucs):.3f} (B14 reference ~0.58)")
except Exception as e:
    print("anchor failed:", e)

# pre-registered verdict on filtered matched-k AUCs
ks_auc = [report[str(k)]["auc_filt"] for k in KS if str(k) in report]
if ks_auc:
    if all(a >= 0.60 for a in ks_auc):
        verdict = "PRECURSOR SIGNAL REAL (>=0.60 sustained)"
    elif all(a < 0.55 for a in ks_auc):
        verdict = "PRECURSOR EXCLUDED (<0.55 across k)"
    else:
        verdict = "INCONCLUSIVE (report interval honestly)"
    report["verdict"] = verdict
    print(f"\nPRE-REGISTERED VERDICT: {verdict}  (filt AUCs at k={KS}: "
          f"{[round(a,3) for a in ks_auc]})")

json.dump(report, open(ROOT / "results" / "sameq_b1_analysis.json", "w"), indent=1)
print(f"saved -> results/sameq_b1_analysis.json")
