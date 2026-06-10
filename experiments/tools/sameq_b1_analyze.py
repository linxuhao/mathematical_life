"""Within-question cumulative-beta1 analysis (CPU, multiprocess) — the
pre-registered adjudication of the precursor hypothesis ("thinking wrong before
the visible error"). Reads results/sameq_acts/*.npz + C3_sameq_v2.json.

Per trajectory, for k in {128, 192, 256} (+384 exploratory) and FULL: cloud =
first k token activations (matched cloud size by construction); beta1_raw = H1
count, beta1 = count with lifetime > 0.03 * eps_max(cloud).

PRE-REGISTERED primary endpoint: within-question-centered GroupKFold(5) AUC of
filtered beta1 at matched k in {128,192,256}. Verdict: >=0.60 sustained ->
precursor real; <0.55 -> excluded; else inconclusive. Capped trajectories
(n_gen >= MAXNEW) excluded (truncation-corrupted labels). Anchor: dec_act
centered probe — B14 reported 0.58 but on truncation-contaminated labels;
this is the clean re-measurement, not a replication target.
Parallelism is per-question (results identical to the sequential version).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.stats import wilcoxon, pearsonr

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "C3_sameq_v2.json"
ACTS = ROOT / "results" / "sameq_acts"
KS_PRIMARY = [128, 192, 256]   # pre-registered: errors in the 6-pair set occurred
                               # at 47-84% of trajectories (tokens 130-277), so the
                               # pre-error window the precursor hypothesis targets
                               # is covered by these ks
KS_EXTRA = [384]               # exploratory only: conditioning on n_gen>=384 selects
                               # long (within-question error-prone) trajectories
KS = KS_PRIMARY + KS_EXTRA
MAXNEW = 2048                  # capped trajectories carry truncation-corrupted labels
                               # (the v1-pilot failure mode) and are EXCLUDED
WORKERS = 14                   # 8c/16t box: leave a couple of threads for the OS

def beta1(cloud):
    from sklearn.metrics import pairwise_distances
    from ripser import ripser
    eps_max = float(np.max(pairwise_distances(cloud)))
    dgm = ripser(cloud.astype(np.float32), maxdim=1)["dgms"][1]
    filt = int(sum(1 for b, d in dgm if (d - b) > 0.03 * eps_max))
    return len(dgm), filt

def traj_rows(args):
    fpath, key, K = args
    z = np.load(fpath)
    out = []
    for si in range(K):
        if f"s{si}" not in z:
            continue
        h = z[f"s{si}"].astype(np.float32)
        if len(h) >= MAXNEW:   # capped: never reached EOS, label unreliable
            continue
        row = {"qid": key, "si": si, "correct": bool(z[f"s{si}_correct"]),
               "n_gen": len(h), "b1": {}}
        for k in KS:
            if len(h) >= k:
                row["b1"][str(k)] = beta1(h[:k])
        row["b1"]["full"] = beta1(h)
        out.append(row)
    return out

def centered_auc(feat_rows):
    """within-question centering -> GroupKFold(5) logistic AUC."""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = np.array([r["_x"] for r in feat_rows], float).reshape(-1, 1)
    y = np.array([r["correct"] for r in feat_rows])
    g = np.array([r["qid"] for r in feat_rows])
    for q in set(g):
        X[g == q] -= X[g == q].mean(axis=0)
    aucs = []
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(set(y[te])) < 2:
            continue
        p = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))

def main():
    res = json.load(open(SRC))
    cache = ROOT / "results" / "sameq_b1_rows.json"
    if cache.exists():
        rows = json.load(open(cache))
        print(f"loaded {len(rows)} cached rows from {cache.name}")
    else:
        files = sorted(ACTS.glob("q*.npz"), key=lambda p: int(p.stem[1:]))
        tasks = [(str(f), f.stem[1:], res[f.stem[1:]]["K"]) for f in files]
        rows = []
        with Pool(WORKERS) as pool:
            for i, chunk in enumerate(pool.imap_unordered(traj_rows, tasks)):
                rows.extend(chunk)
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(tasks)} questions, {len(rows)} usable traj",
                          flush=True)
        json.dump(rows, open(cache, "w"))
    print(f"{len(rows)} trajectories (uncapped) from "
          f"{len(set(r['qid'] for r in rows))} mixed questions")

    report = {}
    print(f"\n{'k':>6} | {'n':>4} | filt AUC (centered) | raw AUC | Wilcoxon p (per-Q filt)")
    for k in KS + ["full"]:
        kk = str(k)
        sub = [dict(r, _x=r["b1"][kk][1]) for r in rows if kk in r["b1"]]
        if len(sub) < 50:
            print(f"{k:>6} | too few ({len(sub)})"); continue
        a_f, s_f = centered_auc(sub)
        sub_r = [dict(r, _x=r["b1"][kk][0]) for r in rows if kk in r["b1"]]
        a_r, _ = centered_auc(sub_r)
        dq = []
        for q in set(r["qid"] for r in sub):
            qs = [r for r in sub if r["qid"] == q]
            c = [r["_x"] for r in qs if r["correct"]]
            i_ = [r["_x"] for r in qs if not r["correct"]]
            if c and i_:
                dq.append(np.mean(c) - np.mean(i_))
        wp = wilcoxon(dq).pvalue if len(dq) > 10 and any(d != 0 for d in dq) else float("nan")
        print(f"{k:>6} | {len(sub):>4} | {a_f:.3f}±{s_f:.3f}        | {a_r:.3f}   | {wp:.3f} "
              f"(median Δ={np.median(dq):+.1f}, nQ={len(dq)})")
        report[str(k)] = {"n": len(sub), "auc_filt": a_f, "auc_filt_std": s_f,
                          "auc_raw": a_r, "wilcoxon_p": float(wp),
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
        try:
            # partial correlation via the correlation formula (no lstsq/SVD)
            r_bc = pearsonr(b, c)[0]; r_bn = pearsonr(b, n)[0]; r_cn = pearsonr(c, n)[0]
            den = np.sqrt((1 - r_bn**2) * (1 - r_cn**2))
            pr = (r_bc - r_bn * r_cn) / den
            from scipy.stats import t as tdist
            df = len(b) - 3
            tstat = pr * np.sqrt(df / max(1e-12, 1 - pr**2))
            pp = float(2 * tdist.sf(abs(tstat), df))
            report["partial_full_b1_correct_given_ngen"] = {"r": float(pr), "p": pp}
            print(f"\nfull-traj partial r(b1_filt, correct | n_gen, within-Q) = {pr:+.3f} (p={pp:.4f})")
        except Exception as e:
            print("partial failed:", e)

    # anchor: dec_act PCA50 centered probe (clean labels; B14's contaminated value was 0.58)
    try:
        from sklearn.decomposition import PCA
        from sklearn.model_selection import GroupKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        X, y, g = [], [], []
        for key, v in res.items():
            if not (0 < v["n_correct"] < v["K"]):
                continue
            for c_ in v["comps"]:
                if c_.get("status") == "ok" and "dec_act" in c_ and c_["n_gen"] < MAXNEW:
                    X.append(c_["dec_act"]); y.append(c_["correct"]); g.append(key)
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
        print(f"\nANCHOR dec_act centered PCA50 AUC = {np.mean(aucs):.3f} "
              f"(clean labels; B14's contaminated value was 0.58)")
    except Exception as e:
        print("anchor failed:", e)

    # pre-registered verdict on filtered matched-k AUCs (PRIMARY ks only)
    ks_auc = [report[str(k)]["auc_filt"] for k in KS_PRIMARY if str(k) in report]
    if ks_auc:
        if all(a >= 0.60 for a in ks_auc):
            verdict = "PRECURSOR SIGNAL REAL (>=0.60 sustained)"
        elif all(a < 0.55 for a in ks_auc):
            verdict = "PRECURSOR EXCLUDED (<0.55 across k)"
        else:
            verdict = "INCONCLUSIVE (report interval honestly)"
        report["verdict"] = verdict
        print(f"\nPRE-REGISTERED VERDICT: {verdict}  (filt AUCs at k={KS_PRIMARY}: "
              f"{[round(a, 3) for a in ks_auc]})")

    json.dump(report, open(ROOT / "results" / "sameq_b1_analysis.json", "w"), indent=1)
    print("saved -> results/sameq_b1_analysis.json")

if __name__ == "__main__":
    main()
