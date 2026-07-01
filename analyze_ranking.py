#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze + visualize human perception-ranking results.

For one or more participant result JSONs it:
  * fits a Bradley-Terry latent risk-taking scale (overall and per task),
  * compares the human-perceived ranking against the paper's AUC ranking
    (overall and, when provided, per task) via Kendall's tau,
  * renders a slopegraph (perceived vs AUC) + per-task panels  -> ranking_comparison.png

Usage:  python analyze_ranking.py [results_dir_or_file]   (default: ./results)
"""
import json, glob, os, sys, math
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITIES = ["deepseek", "gemini", "gpt", "grok", "qwen", "sonnet", "human"]
DISPLAY = {"deepseek":"DeepSeek-V3.2","gemini":"Gemini-3-Pro","gpt":"GPT-5.2",
           "grok":"Grok-4","qwen":"Qwen3-Max","sonnet":"Claude-Sonnet-4.5","human":"Human"}
STAR = {"grok": "*"}     # Grok-4 failed upstream intra-consistency

# --- Paper AUC rankings (Fig. 5a), ordered MOST risk-taking -> MOST risk-averse ---
PAPER_OVERALL = ["gpt", "human", "qwen", "gemini", "deepseek", "sonnet", "grok"]
# Per-task AUC orderings, MOST risk-taking -> MOST risk-averse (task codes: DSB=DNC, TPB=CTD).
PAPER_PERTASK = {
    "DSB": ["human", "grok", "gpt", "qwen", "gemini", "deepseek", "sonnet"],   # DNC
    "TPB": ["gpt", "qwen", "gemini", "deepseek", "sonnet", "human", "grok"],   # CTD
    "FIP": ["gpt", "qwen", "gemini", "deepseek", "sonnet", "human", "grok"],
}

# --------------------------------------------------------------------------- #
def bradley_terry(responses):
    W = {e: 0.0 for e in ENTITIES}
    games = {e: 0 for e in ENTITIES}
    N = {}
    def k(a, b): return (a, b)
    for r in responses:
        a, b, ch = r["leftEntity"], r["rightEntity"], r["choice"]
        N[k(a, b)] = N.get(k(a, b), 0) + 1
        N[k(b, a)] = N.get(k(b, a), 0) + 1
        games[a] += 1; games[b] += 1
        if ch == "left":  W[a] += 1
        elif ch == "right": W[b] += 1
        else: W[a] += .5; W[b] += .5
    p = {e: 1.0 for e in ENTITIES}
    for _ in range(400):
        np_ = {}
        for i in ENTITIES:
            den = sum(N.get((i, j), 0) / (p[i] + p[j]) for j in ENTITIES if N.get((i, j), 0))
            np_[i] = (W[i] / den) if den > 0 else p[i]
        g = math.exp(sum(math.log(max(v, 1e-9)) for v in np_.values()) / len(np_))
        p = {e: v / g for e, v in np_.items()}
    played = [e for e in ENTITIES if games[e] > 0]
    order = sorted(played, key=lambda e: -p[e]) + [e for e in ENTITIES if games[e] == 0]
    return order, p, games


def kendall_tau(order_a, order_b):
    ents = [e for e in order_a if e in order_b]
    ra = {e: i for i, e in enumerate(order_a)}
    rb = {e: i for i, e in enumerate(order_b)}
    c = d = 0
    for a, b in combinations(ents, 2):
        s = (ra[a] - ra[b]) * (rb[a] - rb[b])
        if s > 0: c += 1
        elif s < 0: d += 1
    return (c - d) / (c + d) if (c + d) else float("nan")


def load_results(path):
    files = ([path] if os.path.isfile(path)
             else sorted(glob.glob(os.path.join(path, "*.json"))))
    resp, brands = [], []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        resp += d.get("responses", [])
        if d.get("brandRanking"):
            brands.append(d["brandRanking"])
    return resp, brands, files


# --------------------------------------------------------------------------- #
def slopegraph(ax, left_order, right_order, left_lab, right_lab, title):
    n = len(ENTITIES)
    rl = {e: i for i, e in enumerate(left_order)}
    rr = {e: i for i, e in enumerate(right_order)}
    for e in ENTITIES:
        if e not in rl or e not in rr:
            continue
        y0, y1 = n - rl[e], n - rr[e]
        hu = e == "human"
        col = "#e0792b" if hu else "#9aa7b5"
        ax.plot([0, 1], [y0, y1], "-", color=col, lw=3 if hu else 1.6,
                alpha=.95 if hu else .7, zorder=3 if hu else 1)
        ax.plot([0, 1], [y0, y1], "o", color=col, ms=7 if hu else 5, zorder=4)
        ax.text(-0.04, y0, f"{rl[e]+1}. {DISPLAY[e]}{STAR.get(e,'')}", ha="right",
                va="center", fontsize=10, fontweight="bold" if hu else "normal")
        ax.text(1.04, y1, f"{DISPLAY[e]}{STAR.get(e,'')} ({rr[e]+1})", ha="left",
                va="center", fontsize=10, fontweight="bold" if hu else "normal")
    ax.set_xlim(-0.6, 1.7); ax.set_ylim(0.4, n + 0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels([left_lab, right_lab], fontsize=11, fontweight="bold")
    ax.set_yticks([]); ax.set_title(title, fontsize=12)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.text(0.5, n + 0.5, "↑ more risk-taking", ha="center", fontsize=9, color="#888")


def kendalls_w(orders):
    """Kendall's W (coefficient of concordance) across several ranking lists."""
    common = [e for e in orders[0] if all(e in o for o in orders)]
    m, n = len(orders), len(common)
    if n < 2:
        return float("nan")
    Rsum = {e: sum(o.index(e) + 1 for o in orders) for e in common}
    Rbar = sum(Rsum.values()) / n
    S = sum((Rsum[e] - Rbar) ** 2 for e in common)
    return 12 * S / (m ** 2 * (n ** 3 - n))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
    resp, brands, files = load_results(path)
    print(f"Loaded {len(resp)} comparisons from {len(files)} file(s).")

    # --- per-task BT rankings (mirrors the paper's per-task AUC rankings) ---
    per = {}; task_orders = []
    for task in ("DSB", "TPB", "FIP"):
        rs = [r for r in resp if r.get("task") == task]
        if not rs:
            continue
        o, _, g = bradley_terry(rs)
        played = [e for e in o if g[e] > 0]
        t = kendall_tau(o, PAPER_PERTASK[task]) if PAPER_PERTASK.get(task) else None
        per[task] = (o, played, len(rs), t)
        task_orders.append(o)
        print(f"[{task}] {len(rs)} comparisons | perceived: "
              + " > ".join(DISPLAY[e] for e in played)
              + (f" | tau vs paper-AUC = {t:+.2f}" if t is not None else " | (paper per-task not set)"))

    # --- aggregate by MEAN per-task rank (paper-style: per-task metric, then aggregate) ---
    common = [e for e in ENTITIES if all(e in o for o in task_orders)]
    mean_rank = {e: sum(o.index(e) for o in task_orders) / len(task_orders) for e in common}
    order = sorted(common, key=lambda e: mean_rank[e])      # aggregate human ranking
    W = kendalls_w(task_orders)                             # cross-task rank stability (paper reports W)
    tau = kendall_tau(order, PAPER_OVERALL)
    print("\nAggregate (mean per-task rank, most->least risk-taking):",
          " > ".join(DISPLAY[e] for e in order))
    print("Paper AUC aggregate:", " > ".join(DISPLAY[e] for e in PAPER_OVERALL))
    print(f"Kendall tau (aggregate vs paper) = {tau:+.2f}")
    print(f"Kendall's W across the 3 task rankings (rank stability) = {W:.2f}")

    # ---- figure ----
    ntask = len(per)
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(1, 1 + ntask, width_ratios=[1.5] + [1] * ntask, wspace=0.55)
    ax0 = fig.add_subplot(gs[0, 0])
    slopegraph(ax0, order, PAPER_OVERALL, "Human\n(mean rank)", "Paper\nAUC",
               f"Aggregate ranking   (τ={tau:+.2f}, W={W:.2f})")

    for i, (task, (o, played, ncmp, t)) in enumerate(per.items(), start=1):
        ax = fig.add_subplot(gs[0, i])
        if PAPER_PERTASK.get(task):
            slopegraph(ax, o, PAPER_PERTASK[task], "You", "AUC",
                       f"{task}  (n={ncmp}, τ={t:+.2f})")
        else:
            n = len(ENTITIES)
            for e in played:
                y = n - o.index(e)
                hu = e == "human"
                ax.plot(0, y, "o", color="#e0792b" if hu else "#2f6fed", ms=8)
                ax.text(0.12, y, f"{o.index(e)+1}. {DISPLAY[e]}{STAR.get(e,'')}",
                        va="center", fontsize=10, fontweight="bold" if hu else "normal")
            ax.set_xlim(-0.3, 2.2); ax.set_ylim(0.4, n + 0.6)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
            ax.set_title(f"{task}: your perception\n(n={ncmp}; paper not set)", fontsize=11)

    fig.suptitle("Human-perceived risk-taking ranking vs. paper AUC ranking",
                 fontsize=15, fontweight="bold")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ranking_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
