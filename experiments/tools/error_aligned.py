import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
txt = json.load(open(ROOT / "results" / "tokenwise_v2_text.json"))
cur = json.load(open(ROOT / "results" / "aha_curves.json"))

# critical-error onset markers (substring where the error begins). Reasoning before
# this point is correct; the error is introduced at/after it.
ERR = {
    "51":  "**Step 3: Calculate the total time",          # answer (5) already computed; needlessly adds outbound 3
    "478": "**Step 3: Verify the role of the tour guides", # then decides to IGNORE the 21 guides
    "115": "Sasha sells the leftover boards at the same price", # ignores the 50% price increase
    "89":  "**2. Calculate the number of gift bags",       # multiplies 0.75 by attending(12) not invited(16)
    "303": "**3. Determine the number of books sold in the second year",  # equation omits current-year sales
    "306": "Since this amount is sold equally",            # extra divide-by-2
}
PAIR = {"51":"277","478":"200","115":"369","89":"382","303":"333","306":"268"}

def tok_index_of(idx, marker):
    toks = txt[idx]["tokens"]
    acc = ""
    for k, t in enumerate(toks):
        acc += t
        if marker in acc:
            return k + 1  # tokens consumed up to & incl. where marker completes
    return None

def b1_curve(idx):
    v = cur[idx]; return np.array(v["ks"], float), np.array(v["beta1"], float)

def b1_at(idx, k):
    ks, ys = b1_curve(idx)
    return float(np.interp(min(k, ks[-1]), ks, ys))

print(f"{'pair':>14} | n_gen | k_err | %pos | β₁@(err-) C vs I | β₁@end C vs I | Δβ₁ pre-err | Δβ₁ end")
for ic, co in PAIR.items():
    n = txt[ic]["n_gen_saved"]
    ke = tok_index_of(ic, ERR[ic])
    if ke is None:
        print(f"  idx {ic}: marker not found"); continue
    kpre = max(ke - 8, 16)  # just BEFORE the error onset
    cpre, ipre = b1_at(co, kpre), b1_at(ic, kpre)
    cend, iend = b1_at(co, n), b1_at(ic, n)
    print(f"  {ic}(I)/{co}(C) | {n:5d} | {ke:5d} | {ke/n*100:3.0f}% | "
          f"{cpre:5.1f} vs {ipre:5.1f}   | {cend:5.1f} vs {iend:5.1f} | "
          f"{cpre-ipre:+5.1f} | {cend-iend:+5.1f}")

# Detailed curve for the cleanest case: idx 51 vs 277, around the error
print("\n=== idx 51 (INCORRECT) vs 277 (CORRECT), β₁(k) around the Step-3 error ===")
ic, co = "51", "277"; ke = tok_index_of(ic, ERR[ic]); n = txt[ic]["n_gen_saved"]
print(f"error onset at token {ke}/{n} ({ke/n*100:.0f}%)")
print(f"{'k':>4} | β₁ correct(277) | β₁ incorrect(51) | Δ(C−I)   {'<-- ERROR ONSET' }")
ksI, ysI = b1_curve(ic); ksC, ysC = b1_curve(co)
for k in range(16, n+1, 16):
    c, i = b1_at(co, k), b1_at(ic, k)
    mark = "  <== error here" if (ke-16) < k <= (ke+16) else ""
    print(f"{k:4d} | {c:6.1f}         | {i:6.1f}          | {c-i:+6.1f}{mark}")
