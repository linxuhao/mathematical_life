"""Analyze C3 #3: within-question correctness prediction.
Per-question CENTERING of the decision-token activation removes ALL between-question
variance (difficulty/identity) — what remains is the reasoning contribution. If the
centered probe still predicts correctness, the reasoning-state geometry carries
correctness info free of the difficulty confound.
"""
import json, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

d = json.load(open("/home/linxuhao/papers/mathematical-life/experiments/results/C3_sameq.json"))
rows = []
for v in d.values():
    if not (0 < v["n_correct"] < v["K"]):   # mixed questions only
        continue
    for c in v["comps"]:
        if c.get("status") == "ok":
            rows.append((v["qid"], c["correct"], c["dec_act"], c["s_pred"], c["n_gen"]))
qid = np.array([r[0] for r in rows]); y = np.array([1 if r[1] else 0 for r in rows])
act = np.array([r[2] for r in rows], float); sp = np.array([r[3] for r in rows], float)
ng = np.array([r[4] for r in rows], float)
nq = len(set(qid))
print(f"mixed questions: {nq}   completions: {len(rows)}   correct={y.sum()} incorrect={(y==0).sum()}")

def center_by_q(X):
    Xc = X.copy().astype(float)
    if Xc.ndim == 1: Xc = Xc[:, None]
    for q in set(qid):
        mask = qid == q
        Xc[mask] -= Xc[mask].mean(0)
    return Xc

gkf = GroupKFold(5)
def auc(X, pca=None):
    if X.ndim == 1: X = X[:, None]
    steps = [StandardScaler()]
    if pca and X.shape[1] > pca: steps.append(PCA(pca))
    steps.append(LogisticRegression(max_iter=3000, class_weight="balanced"))
    p = cross_val_predict(make_pipeline(*steps), X, y, cv=gkf, groups=qid, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

print("\nPredict correctness, GroupKFold by question:")
print(f"  RAW decision-act (PCA50)          : {auc(act, 50):.3f}   [includes between-Q variance]")
print(f"  CENTERED decision-act (PCA50)     : {auc(center_by_q(act), 50):.3f}   <== within-question REASONING signal")
print(f"  CENTERED s_pred (answer surprisal): {auc(center_by_q(sp)):.3f}")
print(f"  CENTERED n_gen (answer length)    : {auc(center_by_q(ng)):.3f}")

def within_q_auc(feat):
    aucs = []
    for q in set(qid):
        mask = qid == q; yy = y[mask]; ff = feat[mask]
        if yy.sum() and (~yy.astype(bool)).sum():
            aucs.append(roc_auc_score(yy, ff))
    return np.mean(aucs), len(aucs)
m_sp, nqa = within_q_auc(-sp)   # lower surprisal -> correct?
print(f"\n  within-Q mean AUC (lower s_pred => correct): {m_sp:.3f}  over {nqa} questions")
print("\n  >0.5 centered/within-Q => reasoning-state geometry carries correctness info (confound-free).")
print("  ~0.5 => the C3 0.75 was entirely question difficulty; no reasoning-correctness signal.")
