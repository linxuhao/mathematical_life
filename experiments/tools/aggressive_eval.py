"""Gentle arm — off-target forgetting on EXISTING 2B-base->GSM8K SFT checkpoints.
No training. For each checkpoint, measure on the TRAINED domain A (GSM8K) and
UNRELATED domains B (PIQA, HellaSwag):
  - accuracy
  - L/2 activation topology (β₁ raw + filtered, survival/persistence)
  - C3 comprehension probe: does the prompt-end activation predict per-item success?
Tests whether the global hammer collaterally damages B (behaviorally + topologically),
and whether topology/comprehension track the B-accuracy. Resumable, detached.
"""
import os, sys, json, re, time
from pathlib import Path
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
sys.path.insert(0, str(Path.home()/"papers/mathematical-life/actopo/src"))
from actopo import FROZEN_V5, measure

BASE = Path.home()/"papers/mathematical-life/experiments"
CKPT_DIR = BASE/"checkpoints/qwen35_2b_aggressive"
BASE_MODEL = "Qwen/Qwen3.5-2B-Base"
OUT = BASE/"results/aggressive_arm.json"
N = 300            # items per domain (acc + topology cloud + C3)

def extract_ans(t):
    m = re.search(r"####\s*(-?\d+[\.,]?\d*)", t)
    if m: return m.group(1).replace(",","").replace(".0","")
    nums = re.findall(r"-?\d+[\.,]?\d*", t)
    return nums[-1].replace(",","").replace(".0","") if nums else None

def load_model(path, dtype=torch.bfloat16):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype,
            attn_implementation="eager", trust_remote_code=True).cuda().eval()
    tk = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    return m, tk

# ---- datasets (loaded once) ----
gsm = load_dataset("openai/gsm8k","main",split="test")
arc = load_dataset("allenai/ai2_arc","ARC-Easy",split="validation")
hella = load_dataset("Rowan/hellaswag",split="validation")

def gsm_items():
    out=[]
    for i in range(N):
        out.append({"prompt":"Please solve step by step, final answer after '####'.\n\n"+gsm[i]["question"],
                    "gold":extract_ans(gsm[i]["answer"]), "type":"gen"})
    return out
def arc_items():
    out=[]
    for i in range(min(N,len(arc))):
        r=arc[i]
        try: lab=r["choices"]["label"].index(r["answerKey"])
        except ValueError: continue
        out.append({"ctx":"Question: "+r["question"]+"\nAnswer:","choices":r["choices"]["text"],"label":lab,"type":"mc"})
    return out
def hella_items():
    out=[]
    for i in range(min(N,len(hella))):
        r=hella[i]; out.append({"ctx":r["ctx"],"choices":r["endings"],"label":int(r["label"]),"type":"mc"})
    return out
DOMAINS = {"A_gsm8k":gsm_items, "B_arc_easy":arc_items, "B_hellaswag":hella_items}

@torch.no_grad()
def last_act(m, tk, texts, layer, bs=16):
    acts=[]
    for i in range(0,len(texts),bs):
        b=texts[i:i+bs]
        enc=tk(b,return_tensors="pt",padding=True,truncation=True,max_length=1024)
        enc={k:v.cuda() for k,v in enc.items()}
        h=m(**enc,output_hidden_states=True).hidden_states[layer]
        idx=enc["attention_mask"].sum(1)-1
        acts.append(h[torch.arange(len(b)),idx].float().cpu().numpy())
    return np.concatenate(acts)

@torch.no_grad()
def mc_correct(m, tk, items, bs=8):
    """likelihood MC: pick choice with highest avg per-token logprob given context."""
    res=[]
    for it in items:
        scores=[]
        for ch in it["choices"]:
            full=it["ctx"]+" "+ch
            enc=tk(full,return_tensors="pt",truncation=True,max_length=1024)
            ce=tk(it["ctx"],return_tensors="pt",truncation=True,max_length=1024)
            ids=enc["input_ids"].cuda(); clen=ce["input_ids"].shape[1]
            lg=m(ids).logits[0].float(); lp=F.log_softmax(lg,-1)
            s=0.0; n=0
            for p in range(clen, ids.shape[1]):
                s+=float(lp[p-1, ids[0,p]]); n+=1
            scores.append(s/max(n,1))
        res.append(int(np.argmax(scores)==it["label"]))
    return np.array(res)

@torch.no_grad()
def gen_correct(m, tk, items, bs=8):
    res=[]
    for i in range(0,len(items),bs):
        b=items[i:i+bs]
        enc=tk([x["prompt"] for x in b],return_tensors="pt",padding=True,truncation=True,max_length=1024)
        enc={k:v.cuda() for k,v in enc.items()}
        g=m.generate(**enc,max_new_tokens=256,do_sample=False,pad_token_id=tk.eos_token_id)
        for j,x in enumerate(b):
            resp=tk.decode(g[j][enc["input_ids"].shape[1]:],skip_special_tokens=True)
            pred=extract_ans(resp); res.append(int(pred is not None and pred==x["gold"]))
    return np.array(res)

def c3_auc(acts, y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    if y.sum()<8 or (y==0).sum()<8: return None
    pipe=make_pipeline(StandardScaler(),PCA(min(30,acts.shape[1])),LogisticRegression(max_iter=2000,class_weight="balanced"))
    p=cross_val_predict(pipe,acts,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
    return float(roc_auc_score(y,p))

def ckpts():
    lst=[(0,BASE_MODEL)]
    for d in sorted(CKPT_DIR.glob("checkpoint-*")):
        lst.append((int(d.name.split("-")[1]), str(d)))
    lst.sort(); return lst

res = json.load(open(OUT)) if OUT.exists() else {}
items_cache = {k:fn() for k,fn in DOMAINS.items()}
t0=time.time()
for step, path in ckpts():
    if str(step) in res: continue
    print(f"\n=== step {step} ({path}) {time.strftime('%H:%M:%S')} ===", flush=True)
    try:
        m, tk = load_model(path); layer=len(m.model.layers)//2
        rec={"step":step}
        for dname, items in items_cache.items():
            y = gen_correct(m,tk,items) if items[0]["type"]=="gen" else mc_correct(m,tk,items)
            texts=[it["prompt"] if it["type"]=="gen" else it["ctx"] for it in items]
            acts=last_act(m,tk,texts,layer)
            top=measure(acts.astype(np.float32),FROZEN_V5)
            rec[dname]={"acc":float(y.mean()),"n":len(y),
                        "beta1":top.beta1,"beta1_raw":top.beta1_raw,"surv":top.survival_rate,
                        "c3_auc":c3_auc(acts,y)}
            print(f"  {dname}: acc={y.mean():.3f} β₁={top.beta1}(raw {top.beta1_raw}) surv={top.survival_rate} c3={rec[dname]['c3_auc']}", flush=True)
        res[str(step)]=rec
        tmp=str(OUT)+".tmp"; json.dump(res,open(tmp,"w"),indent=1); os.replace(tmp,str(OUT))
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"  step {step} FAILED: {e}",flush=True)
    finally:
        try: del m
        except: pass
        torch.cuda.empty_cache()
print(f"\nDONE {len(res)} checkpoints ({time.time()-t0:.0f}s)")
