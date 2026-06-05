"""Dynamic PR — effective dimensionality (participation ratio) across training, for
BOTH the gentle and aggressive arms. Tests Neural Collapse: does the representation
manifold's effective dim collapse during over-training, and does it track forgetting
(where filtered beta1 was a bystander)? GPU, forward-only.
PR = (sum lambda)^2 / sum(lambda^2) over covariance eigenvalues (scale-invariant);
EVR1 = lambda_1 / sum(lambda). Same L/2 last-token clouds as the collapse eval (N=300).
"""
import os,sys,json,re,time
from pathlib import Path
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION","11.0.0"); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import torch,numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE=Path.home()/"papers/mathematical-life/experiments"
BASE_MODEL="Qwen/Qwen3.5-2B-Base"
ARMS={"gentle":BASE/"checkpoints/qwen35_2b_sft_reduced","aggressive":BASE/"checkpoints/qwen35_2b_aggressive"}
OUT=BASE/"results/dynamic_pr.json"; N=300

gsm=load_dataset("openai/gsm8k","main",split="test")
arc=load_dataset("allenai/ai2_arc","ARC-Easy",split="validation")
hella=load_dataset("Rowan/hellaswag",split="validation")
def gsm_texts(): return ["Please solve step by step, final answer after '####'.\n\n"+gsm[i]["question"] for i in range(N)]
def arc_texts(): return ["Question: "+arc[i]["question"]+"\nAnswer:" for i in range(min(N,len(arc)))]
def hella_texts(): return [hella[i]["ctx"] for i in range(min(N,len(hella)))]
DOMTEXT={"A_gsm8k":gsm_texts(),"B_arc_easy":arc_texts(),"B_hellaswag":hella_texts()}

def load_model(path):
    m=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation="eager",trust_remote_code=True).cuda().eval()
    tk=AutoTokenizer.from_pretrained(path,trust_remote_code=True)
    if tk.pad_token is None: tk.pad_token=tk.eos_token
    tk.padding_side="left"; return m,tk

@torch.no_grad()
def last_act(m,tk,texts,layer,bs=16):
    out=[]
    for i in range(0,len(texts),bs):
        b=texts[i:i+bs]; enc=tk(b,return_tensors="pt",padding=True,truncation=True,max_length=1024)
        enc={k:v.cuda() for k,v in enc.items()}
        h=m(**enc,output_hidden_states=True).hidden_states[layer]
        idx=enc["attention_mask"].sum(1)-1
        out.append(h[torch.arange(len(b)),idx].float().cpu().numpy())
    return np.concatenate(out)

def pr_evr(acts):
    X=acts-acts.mean(0)
    s=np.linalg.svd(X,compute_uv=False)
    l=s**2
    return float((l.sum()**2)/(l**2).sum()), float(l[0]/l.sum())

def ckpts(cdir):
    lst=[(0,BASE_MODEL)]+[(int(d.name.split("-")[1]),str(d)) for d in cdir.glob("checkpoint-*")]
    lst.sort(); return lst

res=json.load(open(OUT)) if OUT.exists() else {}
for arm,cdir in ARMS.items():
    res.setdefault(arm,{})
    for step,path in ckpts(cdir):
        if str(step) in res[arm]: continue
        try:
            m,tk=load_model(path); layer=len(m.model.layers)//2; rec={}
            for dom,texts in DOMTEXT.items():
                acts=last_act(m,tk,texts,layer); pr,evr=pr_evr(acts)
                rec[dom]={"pr":pr,"evr1":evr,"n":len(acts)}
            res[arm][str(step)]=rec
            print(f"{arm} step {step}: "+" ".join(f"{d}=PR{rec[d]['pr']:.1f}" for d in rec),flush=True)
            tmp=str(OUT)+".tmp"; json.dump(res,open(tmp,"w"),indent=1); os.replace(tmp,str(OUT))
        except Exception as e:
            import traceback; traceback.print_exc()
        finally:
            try: del m
            except: pass
            torch.cuda.empty_cache()
print("DONE")
