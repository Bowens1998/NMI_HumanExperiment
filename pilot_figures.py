#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pilot report figure — one panel per hypothesis (H1/H2/H3). Reads ./_pilot/*.json."""
import json, glob, os, random, math, statistics as st
from itertools import combinations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from analyze_ranking import bradley_terry, kendall_tau, PAPER_OVERALL, DISPLAY, ENTITIES

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.load(open(f, encoding="utf-8")) for f in glob.glob(os.path.join(HERE, "_pilot", "*.json"))]
rng = random.Random(0)
SHORT = {"deepseek":"DeepSeek","gemini":"Gemini","gpt":"GPT-5.2","grok":"Grok",
         "qwen":"Qwen","sonnet":"Sonnet","human":"Human"}
# Okabe-Ito (colorblind-safe)
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, GREY = ("#0072B2","#E69F00","#009E73","#D55E00","#CC79A7","#56B4E9","#9aa3ad")
INK, MUT = "#1f2733", "#6b7785"

def pool(rs):
    out=[];
    for d in rs: out+=d["responses"]
    return out

# ---- computations ----
human_order,_,_ = bradley_terry(pool(recs))
tau_h1 = kendall_tau(human_order, PAPER_OVERALL)

def svec(rs):
    _,p,_ = bradley_terry(pool(rs)); return [math.log(max(p[e],1e-9)) for e in ENTITIES]
sh=[]
idx=list(range(len(recs)))
for _ in range(1000):
    rng.shuffle(idx); h=len(idx)//2
    r=spearmanr(svec([recs[i] for i in idx[:h]]), svec([recs[i] for i in idx[h:]])).correlation
    if r==r: sh.append(r)
mean_sh=st.mean(sh); sb=2*mean_sh/(1+mean_sh)

def iorder(d):
    o,_,g=bradley_terry(d["responses"]); return [e for e in o if g[e]>0]
orders=[iorder(d) for d in recs]
pair_taus=[kendall_tau(a,b) for a,b in combinations(orders,2)]
mean_pair=st.mean(pair_taus)

NONH=[e for e in ENTITIES if e!="human"]
brand_rank={e:[] for e in NONH}
for d in recs:
    b=[e for e in d.get("brandRanking",[]) if e in NONH]
    if len(b)==6:
        for e in b: brand_rank[e].append(b.index(e)+1)
brand_mean={e:st.mean(v) for e,v in brand_rank.items() if v}
beh_order=[e for e in human_order if e in NONH]
beh_rank={e:beh_order.index(e)+1 for e in NONH}
common=[e for e in NONH if e in brand_mean]
rho_h3=spearmanr([brand_mean[e] for e in common],[beh_rank[e] for e in common]).correlation
within=[]
for d in recs:
    b=[e for e in d.get("brandRanking",[]) if e in NONH]
    if len(b)<6: continue
    o,_,g=bradley_terry(d["responses"]); beh=[e for e in o if e in NONH]
    within.append(kendall_tau(b,beh))

# ---- figure ----
plt.rcParams.update({"font.size":11,"axes.edgecolor":"#c9d0d8"})
fig=plt.figure(figsize=(13.5,9.6))
gs=fig.add_gridspec(2,2,hspace=0.42,wspace=0.26,left=.07,right=.97,top=.9,bottom=.07)

# --- Panel A : H1 slopegraph AUC vs Human ---
axA=fig.add_subplot(gs[0,0]); n=7
rA={e:i for i,e in enumerate(PAPER_OVERALL)}; rH={e:i for i,e in enumerate(human_order)}
for e in ENTITIES:
    y0,y1=n-rA[e],n-rH[e]
    hi = e=="gpt"; hu=e=="human"
    col=VERM if hi else (ORANGE if hu else GREY); lw=3 if (hi or hu) else 1.6
    axA.plot([0,1],[y0,y1],"-",color=col,lw=lw,alpha=.95 if(hi or hu)else .6,zorder=3 if hi else 1)
    axA.plot([0,1],[y0,y1],"o",color=col,ms=7,zorder=4)
    axA.text(-.03,y0,f"{rA[e]+1} {SHORT[e]}",ha="right",va="center",fontsize=10,
             fontweight="bold" if(hi or hu)else "normal",color=col if(hi or hu)else INK)
    axA.text(1.03,y1,f"{SHORT[e]} {rH[e]+1}",ha="left",va="center",fontsize=10,
             fontweight="bold" if(hi or hu)else "normal",color=col if(hi or hu)else INK)
axA.set_xlim(-.7,1.7); axA.set_ylim(.4,n+.9)
axA.set_xticks([0,1]); axA.set_xticklabels(["AUC metric","Humans"],fontsize=11,fontweight="bold")
axA.set_yticks([]);
for s in ("top","right","left"): axA.spines[s].set_visible(False)
axA.text(.5,n+.7,"↑ more risk-taking",ha="center",fontsize=9,color=MUT)
axA.annotate("GPT-5.2: the metric's\n#1 risk-taker → humans'\n#1 most-cautious",
             xy=(1,n-rH["gpt"]),xytext=(1.15,2.4),fontsize=9,color=VERM,
             arrowprops=dict(arrowstyle="->",color=VERM))
axA.set_title(f"H1 · Humans rank AI risk-taking ≈ opposite the AUC metric\n(Kendall τ = {tau_h1:+.2f}, N={len(recs)})",
              fontsize=12,fontweight="bold",loc="left")

# --- Panel B : H2 crowd vs individuals ---
axB=fig.add_subplot(gs[0,1])
vals=[sb,mean_pair]; labs=["Crowd vs itself\n(split-half)","Two random\nindividuals"]; cols=[BLUE,ORANGE]
bars=axB.bar([0,1],vals,color=cols,width=.6,zorder=3)
for x,v in zip([0,1],vals):
    axB.text(x,v+.03,f"{v:+.2f}",ha="center",fontsize=13,fontweight="bold",color=INK)
axB.axhline(0,color="#c9d0d8",lw=1)
axB.set_ylim(-.05,1.0); axB.set_xticks([0,1]); axB.set_xticklabels(labs,fontsize=11)
axB.set_ylabel("ranking agreement",fontsize=10)
for s in ("top","right"): axB.spines[s].set_visible(False)
axB.set_title("H2 · The crowd agrees with itself; individuals don't\n"
              "→ a reproducible consensus amid individual uncertainty",
              fontsize=12,fontweight="bold",loc="left")

# --- Panel C : H2 support — spread of individual agreement ---
axC=fig.add_subplot(gs[1,0])
axC.hist(pair_taus,bins=12,color=SKY,edgecolor="white",zorder=3)
axC.axvline(mean_pair,color=ORANGE,lw=2.5,zorder=4,label=f"mean = {mean_pair:+.2f}")
axC.axvline(0,color="#c9d0d8",lw=1)
axC.set_xlabel("agreement between two raters (Kendall τ)",fontsize=10)
axC.set_ylabel("# rater pairs",fontsize=10)
for s in ("top","right"): axC.spines[s].set_visible(False)
axC.legend(frameon=False,fontsize=10)
axC.set_title("H2 (support) · Individual raters barely agree\n(pairwise agreement clusters near zero)",
              fontsize=12,fontweight="bold",loc="left")

# --- Panel D : H3 within-person brand-vs-behavior agreement (one dot per rater) ---
axD=fig.add_subplot(gs[1,1])
jit=[rng.uniform(-.2,.2) for _ in within]
axD.axvspan(-.2,.2,color="#eef1f6",zorder=0)                       # "near zero" band
axD.axvline(0,color=MUT,lw=1)
axD.axvline(st.mean(within),color=ORANGE,lw=2.5,zorder=4,label=f"mean τ = {st.mean(within):+.2f}")
axD.scatter(within,jit,s=95,color=BLUE,edgecolor="white",zorder=3)
axD.set_xlim(-1.05,1.05); axD.set_ylim(-1,1); axD.set_yticks([])
axD.set_xlabel("within-person agreement:\nprior brand impression vs behavior ranking (Kendall τ)",fontsize=10)
axD.text(-1.0,.8,"contradict",fontsize=9,color=MUT); axD.text(1.0,.8,"match",fontsize=9,color=MUT,ha="right")
for s in ("top","right","left"): axD.spines[s].set_visible(False)
axD.legend(frameon=False,fontsize=10,loc="lower center")
axD.set_title(f"H3 · A person's prior brand impression is ~unrelated to\n"
              f"their behavior-based ranking (τ ≈ 0; preliminary, n={len(within)})",
              fontsize=12,fontweight="bold",loc="left")

fig.suptitle(f"Pilot (N={len(recs)}) — human perception of AI risk attitudes: evidence for H1–H3",
             fontsize=15,fontweight="bold")
out=os.path.join(HERE,"pilot_H123.png"); fig.savefig(out,dpi=150,bbox_inches="tight")
print("Saved",out)
print(f"H1 tau={tau_h1:+.2f} | H2 split-half SB={sb:+.2f} pairwise={mean_pair:+.2f} | "
      f"H3 rho={rho_h3:+.2f} within={st.mean(within):+.2f}")
