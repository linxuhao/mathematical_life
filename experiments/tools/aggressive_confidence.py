"""(a) Confident-forgetting: confidence + calibration on the off-target B domains
across the AGGRESSIVE checkpoints. For each MC item, confidence is measured AT THE
DECISION (softmax over per-choice avg-logprobs, after the choices are seen) — not at
the question stem. Per checkpoint per domain we log:
  acc, mean_conf, conf_when_correct, conf_when_wrong, overconfidence(=mean_conf-acc),
  ECE(10-bin), and the confident-error gap (conf_wrong - conf_correct).
Question: as the model forgets B, does it lose confidence gracefully, or stay
CONFIDENTLY WRONG (the longitudinal version of the static confident-error finding)?
Forward-only, no training. Resumable.
"""
import os, sys, json, time
from pathlib import Path
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION","11.0.0")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import torch, numpy as np, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE=Path.home()/"papers/mathematical-life/experiments"
CKPT_DIR=BASE/"checkpoints/qwen35_2b_aggressive"
BASE_MODEL="Qwen/Qwen3.5-2B-Base"
OUT=BASE/"results/aggressive_confidence.json"
N=300

arc=load_dataset("allenai/ai2_arc","ARC-Easy",split="validation")
hella=load_dataset("Rowan/hellaswag",split="validation")
def arc_items():
    out=[]
    for i in range(min(N,len(arc))):
        r=arc[i]
        try: lab=r["choices"]["label"].index(r["answerKey"])
        except ValueError: continue
        out.append({"ctx":"Question: "+r["question"]+"\nAnswer:","choices":r["choices"]["text"],"label":lab})
    return out
def hella_items():
    return [{"ctx":hella[i]["ctx"],"choices":hella[i]["endings"],"label":int(hella[i]["label"])} for i in range(min(N,len(hella)))]
DOMAINS={"B_arc_easy":arc_items,"B_hellaswag":hella_items}

def load_model(path):
    m=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,
        attn_implementation="eager",trust_remote_code=True).cuda().eval()
    tk=AutoTokenizer.from_pretrained(path,trust_remote_code=True)
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    return m,tk

@torch.no_grad()
def score_mc(m,tk,items):
    """returns per-item (probs over choices, pred, correct)."""
    confs=[];preds=[];corr=[]
    for it in items:
        scores=[]
        for ch in it["choices"]:
            full=it["ctx"]+" "+ch
            enc=tk(full,return_tensors="pt",truncation=True,max_length=1024)
            ce=tk(it["ctx"],return_tensors="pt",truncation=True,max_length=1024)
            ids=enc["input_ids"].cuda(); clen=ce["input_ids"].shape[1]
            lp=F.log_softmax(m(ids).logits[0].float(),-1)
            s=sum(float(lp[p-1,ids[0,p]]) for p in range(clen,ids.shape[1]))/max(ids.shape[1]-clen,1)
            scores.append(s)
        scores=np.array(scores)
        probs=np.exp(scores-scores.max()); probs/=probs.sum()
        pred=int(scores.argmax())
        confs.append(float(probs[pred])); preds.append(pred); corr.append(int(pred==it["label"]))
    return np.array(confs),np.array(corr)

def ece(conf,corr,bins=10):
    e=0.0
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins
        m=(conf>lo)&(conf<=hi) if b>0 else (conf>=lo)&(conf<=hi)
        if m.sum()==0: continue
        e+=abs(conf[m].mean()-corr[m].mean())*m.sum()/len(conf)
    return float(e)

def ckpts():
    lst=[(0,BASE_MODEL)]+[(int(d.name.split("-")[1]),str(d)) for d in CKPT_DIR.glob("checkpoint-*")]
    lst.sort(); return lst

res=json.load(open(OUT)) if OUT.exists() else {}
items={k:fn() for k,fn in DOMAINS.items()}
for step,path in ckpts():
    if str(step) in res: continue
    print(f"=== step {step} {time.strftime('%H:%M:%S')} ===",flush=True)
    try:
        m,tk=load_model(path); rec={"step":step}
        for dn,its in items.items():
            conf,corr=score_mc(m,tk,its)
            cw=conf[corr==0]; cc=conf[corr==1]
            rec[dn]={"acc":float(corr.mean()),"n":len(corr),
                "mean_conf":float(conf.mean()),
                "conf_correct":float(cc.mean()) if len(cc) else None,
                "conf_wrong":float(cw.mean()) if len(cw) else None,
                "overconf":float(conf.mean()-corr.mean()),
                "confident_error_gap":float(cw.mean()-cc.mean()) if len(cw) and len(cc) else None,
                "ece":ece(conf,corr)}
            print(f"  {dn}: acc={corr.mean():.3f} conf={conf.mean():.3f} conf_wrong={rec[dn]['conf_wrong']} conf_corr={rec[dn]['conf_correct']} overconf={rec[dn]['overconf']:+.3f} ece={rec[dn]['ece']:.3f}",flush=True)
        res[str(step)]=rec
        tmp=str(OUT)+".tmp"; json.dump(res,open(tmp,"w"),indent=1); os.replace(tmp,str(OUT))
    except Exception as e:
        import traceback;traceback.print_exc()
    finally:
        try: del m
        except: pass
        torch.cuda.empty_cache()
print(f"DONE {len(res)} ckpts")
