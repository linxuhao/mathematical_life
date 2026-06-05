#!/usr/bin/env python3
"""Generate a full-dataset LaTeX report (tables + figures) for the Mathematical Life
empirical work. Run on the server (has the JSONs + matplotlib); the resulting
build/ dir is pulled to the Mac and compiled with tectonic.
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path.home()/"papers/mathematical-life/experiments/results"
OUT = Path.home()/"papers/mathematical-life/datasets_report"
FIG = OUT/"figs"; FIG.mkdir(parents=True, exist_ok=True)
def load(f): return json.load(open(R/f))

def esc(s):
    s=str(s)
    for a,b in [("\\","\\textbackslash{}"),("_","\\_"),("&","\\&"),("%","\\%"),
                ("#","\\#"),("$","\\$"),("{","\\{"),("}","\\}"),("~","\\textasciitilde{}"),("^","\\textasciicircum{}")]:
        s=s.replace(a,b)
    return s

def num(x,nd=3):
    if x is None: return "--"
    if isinstance(x,float): return f"{x:.{nd}f}"
    return str(x)

def longtable(headers, rows, caption, label, size="footnotesize", colspec=None, landscape=False):
    nc=len(headers)
    colspec=colspec or ("l"+"r"*(nc-1))
    h=" & ".join("\\textbf{%s}"%x for x in headers)+" \\\\"
    out=["\\begin{landscape}"] if landscape else []
    out+=[f"\\begin{{{size}}}",
         f"\\begin{{longtable}}{{{colspec}}}",
         f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
         "\\toprule", h, "\\midrule", "\\endfirsthead",
         "\\toprule", h, "\\midrule", "\\endhead",
         "\\bottomrule", "\\endlastfoot"]
    for r in rows:
        out.append(" & ".join(esc(c) if not isinstance(c,(int,float)) else num(c) for c in r)+" \\\\")
    out += ["\\bottomrule" if not rows else "", "\\end{longtable}", f"\\end{{{size}}}"]
    if landscape: out+=["\\end{landscape}"]
    return "\n".join(out)

def figure(path, caption, label, width="0.95"):
    return ("\\begin{figure}[H]\\centering\n"
            f"\\includegraphics[width={width}\\textwidth]{{{path}}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n\\end{{figure}}")

S=[]  # document body chunks
def add(x): S.append(x)

# ============================================================ STATIC
add("\\section{Static experiments}")

# --- S1+S5 merged static master (186 rows): betti_v3 + phaseC channels ---
bv=load("betti_v3.json"); pc=load("phaseC_channels.json")
keys=sorted(bv.keys())
rows=[]
for k in keys:
    b=bv[k]; c=pc.get(k,{})
    n,d=b.get("shape",[None,None])
    rows.append([k, n, d, b.get("beta0"), b.get("beta1_raw"), b.get("beta1"),
                 round(b.get("eps_max",0),3), round(c.get("pr",float("nan")),2) if c.get("pr") is not None else "--",
                 round(c.get("evr1",float("nan")),3) if c.get("evr1") is not None else "--"])
add("\\subsection{S1+S5 Static master table (all 186 clouds): homology + channel descriptors}")
add("\\textbf{Variables}: model$\\times$prompt-mode (186 clouds). \\textbf{Constants}: FROZEN\\_V5 ($\\varepsilon{=}0.03\\varepsilon_{\\max}$, L/2, last token). "
    "\\textbf{Measured}: $\\beta_0,\\beta_1^{\\mathrm{raw}},\\beta_1^{\\mathrm{filt}}$, participation ratio (PR), EVR1. \\textbf{Matrix}: static $\\times$ ch4 (homology) + ch3 (PR).")
add(longtable(["cloud","n","d","$\\beta_0$","$\\beta_1$ raw","$\\beta_1$ filt","$\\varepsilon_{\\max}$","PR","EVR1"],
              rows, "S1+S5: full static homology + channel dataset (186 clouds).", "tab:static", size="scriptsize", landscape=True))

# --- S2 bottleneck ---
bn=load("bottleneck_v1.json")
rows=[[r["a"],r["b"],r["a_beta1"],r["b_beta1"],round(r["bottleneck_raw"],4),round(r.get("bottleneck_normalized",float('nan')),4)] for r in bn]
add("\\subsection{S2 Mode separation --- bottleneck distances (full)}")
add("\\textbf{Matrix}: static $\\times$ ch1 (Position).")
add(longtable(["mode A","mode B","$\\beta_1^A$","$\\beta_1^B$","bottleneck","normalized"],
              rows,"S2: full bottleneck dataset (within-basis pairs).","tab:bottleneck"))

# --- S3 PHI / curves summary ---
cv=load("curves_v1.json")
rows=[[r["name"].split(" (")[0], r.get("beta1_raw"), round(r.get("eps_max",0),3), len(r.get("eps_levels",[]))] for r in cv]
add("\\subsection{S3 Persistence curves (14 models)}")
add(longtable(["model/mode","$\\beta_1$ raw","$\\varepsilon_{\\max}$","\\#levels"],rows,
              "S3: persistence-curve inventory (full curve arrays plotted in Fig.~\\ref{fig:phi}).","tab:curves"))

# --- S4 tripartite ---
tp=load("tripartite_v1.json")
rows=[[k, v["beta1_raw"], v["beta1"], v.get("surv_rate"), v.get("n_points")]
      for k,v in tp.items() if isinstance(v,dict) and "beta1_raw" in v]
add("\\subsection{S4 Tripartite test (full)}")
add(longtable(["sub-mode","$\\beta_1$ raw","$\\beta_1$ filt","surv rate","n"],rows,
              "S4: R\\_true vs R\\_flawed vs Hallucination.","tab:tripartite"))

# --- S5b beta0 scale scan (186 rows) ---
b0=load("beta0_prelim.json")
fracs=["0.03","0.1","0.2","0.3","0.5","0.7"]
rows=[[k]+[b0[k].get(fr) for fr in fracs] for k in sorted(b0.keys())]
add("\\subsection{S5b $\\beta_0$ scale scan (all 186 clouds)}")
add(longtable(["cloud"]+["$\\beta_0$@"+f for f in fracs],rows,
              "S5b: $\\beta_0$ across filtration fractions (full).","tab:beta0",size="scriptsize",landscape=True))

# ============================================================ DYNAMIC
add("\\section{Dynamic experiments (training checkpoint series)}")

def ts_table(fn,caption,label,cols):
    d=load(fn)
    headers=[c[0] for c in cols]
    rows=[[ (round(rec.get(c[1]),3) if isinstance(rec.get(c[1]),float) else rec.get(c[1],"--")) for c in cols] for rec in d]
    add(longtable(headers,rows,caption,label))

base_cols=[("step","step"),("$\\beta_1^R$","beta1_r"),("$\\beta_1^H$","beta1_h"),
           ("raw R","raw_r"),("raw H","raw_h"),("surv R","surv_r"),("surv H","surv_h"),
           ("PHI","phi"),("GSM8K","gsm8k_acc")]
add("\\subsection{D1 0.5B-Base SFT (full)}"); ts_table("sft_05b_actopo_anchored.json","D1: 0.5B-Base GSM8K SFT timeseries.","tab:d1",base_cols)
add("\\subsection{D2 2B-Instruct SFT --- raw-text format (full, 60 ckpts)}"); ts_table("qwen35_2b_timeseries.json","D2 raw: format-dependent erosion.","tab:d2raw",base_cols)
add("\\subsection{D2 2B-Instruct SFT --- chat-template format (full, 60 ckpts)}")
ts_table("qwen35_2b_inst_actopo_chat.json","D2 chat: erosion disappears (flat).","tab:d2chat",
         base_cols+[("SVAMP","svamp_acc"),("TQA","truthfulqa_acc")])
add("\\subsection{D3 2B-Base SFT (full, 31 ckpts)}")
ts_table("qwen35_2b_base_actopo_anchored.json","D3: 2B-Base SFT (these ckpts are reused as the gentle arm).","tab:d3",
         base_cols+[("SVAMP","svamp_acc"),("TQA","truthfulqa_acc")])

# --- D5 collapse arms: long format domain column ---
def arm_rows(fn):
    d=load(fn); out=[]
    for s in sorted(d,key=lambda x:int(x)):
        rec=d[s]
        for dom in ["A_gsm8k","B_arc_easy","B_hellaswag"]:
            v=rec.get(dom)
            if not v: continue
            out.append([int(s),dom,round(v["acc"],3),v["beta1"],v["beta1_raw"],
                        round(v["surv"],2),round(v["c3_auc"],3) if v.get("c3_auc") is not None else "--"])
    return out
add("\\subsection{D5 Collapse experiment --- GENTLE arm (full)}")
add("\\textbf{Variables}: step (within), domain A=GSM8K(trained) vs B=ARC/HellaSwag(off-target). \\textbf{Constants}: Qwen3.5-2B-Base, lr=1e-5, FROZEN\\_V5, N=300. \\textbf{Measured}: acc, $\\beta_1$ raw+filt, survival, C3-AUC.")
add(longtable(["step","domain","acc","$\\beta_1$","$\\beta_1$ raw","surv","C3 AUC"],arm_rows("gentle_arm.json"),
              "D5 gentle: full per-step per-domain dataset (no off-target forgetting).","tab:gentle"))
add("\\subsection{D5 Collapse experiment --- AGGRESSIVE arm (full)}")
add("\\textbf{Same as gentle but lr=5e-5} (the global hammer). Forgetting appears in B; topology is a bystander (partial$|$step$\\approx$0).")
add(longtable(["step","domain","acc","$\\beta_1$","$\\beta_1$ raw","surv","C3 AUC"],arm_rows("aggressive_arm.json"),
              "D5 aggressive: full per-step per-domain dataset.","tab:aggr"))

# --- (a) confidence, if available ---
cf_path=R/"aggressive_confidence.json"
if cf_path.exists():
    cf=json.load(open(cf_path))
    if cf:
        rows=[]
        for s in sorted(cf,key=lambda x:int(x)):
            for dom in ["B_arc_easy","B_hellaswag"]:
                v=cf[s].get(dom)
                if not v: continue
                rows.append([int(s),dom,round(v["acc"],3),round(v["mean_conf"],3),
                             round(v["conf_correct"],3) if v.get("conf_correct") else "--",
                             round(v["conf_wrong"],3) if v.get("conf_wrong") else "--",
                             round(v["overconf"],3),round(v["ece"],3)])
        add("\\subsection{(a) Confident-forgetting --- aggressive B domains (full, %d ckpts)}"%len(cf))
        add("\\textbf{Measured}: per-checkpoint accuracy, mean confidence (max softmax over choices), "
            "confidence on correct vs wrong items, overconfidence (conf$-$acc), ECE. \\textbf{Result}: as B-accuracy falls, "
            "confidence RISES ($r{\\approx}{+}0.94$), overconfidence flips negative$\\to$positive, ECE nearly doubles --- "
            "the model becomes CONFIDENTLY WRONG. The forgetting signature is in calibration, not topology.")
        add(longtable(["step","domain","acc","mean conf","conf$|$corr","conf$|$wrong","overconf","ECE"],rows,
                      "(a): confidence + calibration on off-target domains (full).","tab:conf"))

# ============================================================ FIGURES
# Fig 1: collapse accuracy trajectories gentle vs aggressive
fig,ax=plt.subplots(1,2,figsize=(11,4),sharey=True)
for j,(fn,ttl) in enumerate([("gentle_arm.json","Gentle (lr 1e-5)"),("aggressive_arm.json","Aggressive (lr 5e-5)")]):
    d=load(fn); steps=sorted(d,key=lambda x:int(x)); X=[int(s) for s in steps]
    for dom,c in [("A_gsm8k","C0"),("B_arc_easy","C1"),("B_hellaswag","C2")]:
        Y=[d[s][dom]["acc"] for s in steps]
        ax[j].plot(X,Y,marker="o",ms=3,color=c,label=dom.replace("_"," "))
    ax[j].set_title(ttl); ax[j].set_xlabel("training step"); ax[j].grid(alpha=.3)
ax[0].set_ylabel("accuracy"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"acc_traj.pdf"); plt.close()

# Fig 2: aggressive A — normalized capability vs topology channels (raw-vs-filtered story)
d=load("aggressive_arm.json"); steps=sorted(d,key=lambda x:int(x)); X=[int(s) for s in steps]
def series(dom,f): return np.array([d[s][dom][f] for s in steps],float)
fig,ax=plt.subplots(figsize=(7,4.2))
def nz(y):
    y=np.array(y,float); return (y-y.min())/(y.max()-y.min()+1e-9)
acc=series("A_gsm8k","acc"); raw=series("A_gsm8k","beta1_raw"); filt=series("A_gsm8k","beta1"); surv=series("A_gsm8k","surv")
ax.plot(X,nz(acc),marker="o",ms=3,label=f"GSM8K acc ({acc.min():.2f}→{acc.max():.2f})")
ax.plot(X,nz(raw),marker="s",ms=3,label=f"$\\beta_1$ raw ({raw.max():.0f}→{raw.min():.0f})")
ax.plot(X,nz(filt),marker="^",ms=3,label=f"$\\beta_1$ filt ({filt.max():.0f}→{filt.min():.0f})")
ax.plot(X,nz(surv),marker="v",ms=3,label=f"survival ({surv.max():.0f}→{surv.min():.0f})")
ax.set_title("Aggressive, trained task A: capability vs topology (min-max normalized)")
ax.set_xlabel("training step"); ax.set_ylabel("normalized [0,1]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"raw_vs_filt.pdf"); plt.close()

# Fig 3: static beta1 base vs instruct across Qwen3.5 sizes (beta1 != capability)
sizes=["0.8B","2B","4B","9B"]
base=[29,51,46,27]; inst=[40,80,90,1]  # from experiment-findings A table (raw-filtered beta1_R)
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(sizes,base,marker="o",label="base"); ax.plot(sizes,inst,marker="s",label="instruct")
ax.set_title("S1: $\\beta_1$ (reasoning) vs model size --- non-monotone, $\\neq$ capability")
ax.set_xlabel("Qwen3.5 size"); ax.set_ylabel("$\\beta_1$ filtered"); ax.grid(alpha=.3); ax.legend()
plt.tight_layout(); plt.savefig(FIG/"beta1_size.pdf"); plt.close()

# Fig 4: persistence curves (survival count vs eps) for a few models
cv=load("curves_v1.json")
fig,ax=plt.subplots(figsize=(7,4.2))
for r in cv[:6]:
    el=r.get("eps_levels",[]); y=r.get("survival",[])
    if el and y and len(el)==len(y):
        ax.plot(np.array(el)/r["eps_max"], y, label=r["name"][:28])
ax.set_xlabel("$\\varepsilon/\\varepsilon_{\\max}$"); ax.set_ylabel("loops alive"); ax.set_xlim(0,0.3)
ax.set_title("S3: persistence curves"); ax.grid(alpha=.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG/"phi.pdf"); plt.close()

# Fig 5: confident-forgetting (overconfidence + ECE vs accuracy)
if cf_path.exists() and json.load(open(cf_path)):
    cf=json.load(open(cf_path)); cs=sorted(cf,key=lambda x:int(x)); X=[int(s) for s in cs]
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for dom,c in [("B_arc_easy","C1"),("B_hellaswag","C2")]:
        acc=[cf[s][dom]["acc"] for s in cs]; oc=[cf[s][dom]["overconf"] for s in cs]; ece=[cf[s][dom]["ece"] for s in cs]
        ax[0].plot(X,acc,marker="o",ms=3,color=c,ls="-",label=dom.replace("_"," ")+" acc")
        ax[0].plot(X,oc,marker="s",ms=3,color=c,ls="--",label=dom.replace("_"," ")+" overconf")
        ax[1].plot(X,ece,marker="^",ms=3,color=c,label=dom.replace("_"," ")+" ECE")
    ax[0].axhline(0,color="k",lw=.6); ax[0].set_title("Off-target: accuracy (solid) vs overconfidence (dashed)")
    ax[0].set_xlabel("training step"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].set_title("Off-target calibration error (ECE)"); ax[1].set_xlabel("training step"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG/"confident_forgetting.pdf"); plt.close()

# Assemble figures section
add("\\section{Figures}")
if (FIG/"confident_forgetting.pdf").exists():
    add(figure("figs/confident_forgetting.pdf","(a) Confident-forgetting: as off-target accuracy falls (solid), overconfidence (dashed) flips from negative to positive and calibration error (ECE) nearly doubles. The model becomes confidently wrong --- the forgetting signature is in calibration, not the Betti number.","fig:conf"))
add(figure("figs/acc_traj.pdf","Accuracy trajectories: gentle (no off-target forgetting) vs aggressive (B drops).","fig:acc"))
add(figure("figs/raw_vs_filt.pdf","Trained task under aggressive SFT: capability rises while filtered $\\beta_1$/survival collapse but RAW $\\beta_1$ barely moves --- the collapse is filter-fragile; $\\beta_1\\neq$capability.","fig:rawfilt"))
add(figure("figs/beta1_size.pdf","Static $\\beta_1$ vs size, base vs instruct: non-monotone, refutes $\\beta_1$=capability.","fig:size"))
add(figure("figs/phi.pdf","Persistence curves (loops alive vs scale) for representative clouds.","fig:phi"))

# ============================================================ WRITE TEX
HEAD=r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{longtable,booktabs,graphicx,float,amsmath,amssymb}
\usepackage{pdflscape}
\usepackage[hidelinks]{hyperref}
\setlength{\tabcolsep}{4pt}
\title{Mathematical Life --- Empirical Datasets (full)\\\large all experiments: variables, constants, measured, full data}
\date{Generated 2026-06-02}
\begin{document}\maketitle
\noindent This document contains the \textbf{complete datasets} for every experiment in the empirical
program, organized by the static/dynamic $\times$ 5-channel matrix (see EXPERIMENT\_SUMMARY.md).
Channels: 1=Position, 2=Scale (normalized out), 3=Orientation/PR, 4=Connectivity ($\beta_0/\beta_1/\beta_2$), 5=Persistence.
\tableofcontents
\clearpage
"""
TEX=HEAD+"\n\n".join(S)+"\n\\end{document}\n"
(OUT/"datasets_report.tex").write_text(TEX)
print("WROTE",OUT/"datasets_report.tex","("+str(len(TEX))+" chars)")
print("figs:",sorted(p.name for p in FIG.glob("*.pdf")))
