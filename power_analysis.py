#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulation-based power analysis for the human risk-attitude PERCEPTION ranking study.

Two questions, two analyses:

  A) GROUP RANKING RECOVERY  -- how many participants are needed for the pooled
     human-perceived ranking (Bradley-Terry over all pairwise choices) to recover
     the true latent order reliably?  Driver: total #comparisons = N x C.

  B) DEMOGRAPHIC MODERATION (the binding constraint) -- does a participant's own
     risk attitude (Holt-Laury) predict their per-person perception outcome
     (here: agreement with the AUC ranking)?  This is a between-person regression
     whose power depends on N, comparisons/person C, choice consistency, and the
     true effect size.  Per-person measurement noise from finite C is captured
     by simulating the actual pairwise choices (attenuation is built in).

Everything is generative; key assumptions are in CONFIG and are easy to change.
Outputs: prints power tables + recommended N, saves power_analysis.png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
    TCRIT = lambda df: stats.t.ppf(0.975, df)
except Exception:                                   # normal fallback (~fine for power)
    TCRIT = lambda df: 1.98

rng = np.random.default_rng(7)

# --------------------------------------------------------------------------- #
CONFIG = dict(
    n_entities      = 7,
    spread          = 1.5,      # true latent risk-taking scale spans [-spread, +spread]
    C               = 30,       # comparisons per participant (post fatigue-reduction, 4v4)
    alpha_base      = 1.5,      # choice consistency. 1.0=noisy, 1.5=medium, 2.5=sharp.
                                #   each comparison shows 4v4 trials, which sharpens vs single-trial.
    sigma_nu        = 0.30,     # between-person consistency noise (SD on the alpha multiplier)
    n_grid          = list(range(20, 281, 20)),
    R               = 300,      # Monte-Carlo reps per cell
    alpha_level     = 0.05,
    target_power    = 0.80,
    n_covariates    = 3,        # primary predictor (Holt-Laury) + age + AI-use
    # effect sizes for analysis B: modulation of choice-consistency by risk attitude.
    # realized effect (corr between risk attitude and per-person agreement) is reported.
    effect_mods     = [0.04, 0.08, 0.15, 0.30],   # -> realized r ~ 0.1 / 0.2 / 0.3 / 0.5
)


def latent_scale():
    s = np.linspace(-CONFIG["spread"], CONFIG["spread"], CONFIG["n_entities"])
    return s


def gen_choices(alpha_i, s, C):
    """For each participant (alpha_i vector, len N) generate C random-pair choices.
    Returns per-person agreement tau in [-1,1] vs the true order of s."""
    N = len(alpha_i)
    a = rng.integers(0, len(s), size=(N, C))
    b = rng.integers(0, len(s), size=(N, C))
    same = a == b
    b = np.where(same, (b + 1) % len(s), b)               # avoid self-pairs
    ds = s[a] - s[b]
    p_choose_a = 1.0 / (1.0 + np.exp(-alpha_i[:, None] * ds))
    chose_a = rng.random((N, C)) < p_choose_a
    concordant = np.where(chose_a, ds > 0, ds < 0)        # picked the higher-s entity?
    tau = 2.0 * concordant.mean(axis=1) - 1.0
    return tau


# --------------------------------------------------------------------------- #
def bt_fit(wins, games_pair):
    """Bradley-Terry MM on aggregated pairwise win counts (vectorized). wins[i][j]=#times i beat j."""
    n = wins.shape[0]
    p = np.ones(n)
    W = wins.sum(axis=1)
    for _ in range(200):
        S = games_pair / (p[:, None] + p[None, :])   # p>0 always (floored below) -> no div-by-zero
        np.fill_diagonal(S, 0.0)
        denom = S.sum(axis=1)
        newp = np.where(denom > 0, W / np.where(denom > 0, denom, 1.0), p)
        newp /= np.exp(np.mean(np.log(np.clip(newp, 1e-9, None))))
        p = np.clip(newp, 1e-9, None)                 # keep strengths strictly positive
    return p


def kendall_tau_order(est, true):
    n = len(est); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            sgn = np.sign(est[i] - est[j]) * np.sign(true[i] - true[j])
            if sgn > 0: c += 1
            elif sgn < 0: d += 1
    return (c - d) / (c + d) if (c + d) else 0.0


def analysis_A(s):
    """Group ranking recovery vs N (fixed C, alpha_base)."""
    n = len(s); C = CONFIG["C"]; ab = CONFIG["alpha_base"]
    out = []
    for N in CONFIG["n_grid"]:
        taus = []; exact = 0
        for _ in range(CONFIG["R"]):
            alpha_i = np.clip(ab * (1 + CONFIG["sigma_nu"] * rng.standard_normal(N)), 0.1, None)
            wins = np.zeros((n, n)); games = np.zeros((n, n))
            a = rng.integers(0, n, size=(N, C)); b = rng.integers(0, n, size=(N, C))
            b = np.where(a == b, (b + 1) % n, b)
            ds = s[a] - s[b]
            chose_a = rng.random((N, C)) < 1.0 / (1.0 + np.exp(-alpha_i[:, None] * ds))
            for i in range(N):
                for k in range(C):
                    x, y = a[i, k], b[i, k]
                    if chose_a[i, k]: wins[x, y] += 1
                    else: wins[y, x] += 1
                    games[x, y] += 1; games[y, x] += 1
            p = bt_fit(wins, games)
            t = kendall_tau_order(p, s); taus.append(t)
            if t >= 0.999: exact += 1
        taus = np.array(taus)
        out.append(dict(N=N, mean_tau=taus.mean(),
                        p_tau_ge_090=(taus >= 0.90).mean(),
                        p_exact=exact / CONFIG["R"]))
    return out


def analysis_B(s, rho=1.0):
    """Demographic-moderation regression power vs N, for several effect sizes.
    rho = reliability of the risk-attitude QUESTIONNAIRE (predictor). The trait x_true
    drives perception; we regress on a noisy measurement x_meas = sqrt(rho)x + sqrt(1-rho)e.
    Lower reliability attenuates the observed effect -> needs more participants."""
    C = CONFIG["C"]; ab = CONFIG["alpha_base"]; k = CONFIG["n_covariates"]
    results = {}; realized = {}
    for mod in CONFIG["effect_mods"]:
        Nbig = 4000
        x = rng.standard_normal(Nbig)
        alpha_i = np.clip(ab * (1 + mod * x + CONFIG["sigma_nu"] * rng.standard_normal(Nbig)), 0.1, None)
        tau = gen_choices(alpha_i, s, C)
        x_meas = np.sqrt(rho) * x + np.sqrt(1 - rho) * rng.standard_normal(Nbig)
        realized[mod] = np.corrcoef(x_meas, tau)[0, 1]      # observed (attenuated) effect
        powers = []
        for N in CONFIG["n_grid"]:
            rej = 0
            for _ in range(CONFIG["R"]):
                x = rng.standard_normal(N)
                alpha_i = np.clip(ab * (1 + mod * x + CONFIG["sigma_nu"] * rng.standard_normal(N)), 0.1, None)
                y = gen_choices(alpha_i, s, C)
                x_meas = np.sqrt(rho) * x + np.sqrt(1 - rho) * rng.standard_normal(N)
                X = np.column_stack([np.ones(N), x_meas, rng.standard_normal((N, k - 1))])
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                dof = N - X.shape[1]
                covb = (resid @ resid) / dof * np.linalg.inv(X.T @ X)
                if abs(beta[1] / np.sqrt(covb[1, 1])) > TCRIT(dof): rej += 1
            powers.append(rej / CONFIG["R"])
        results[mod] = powers
    return results, realized


def recommend(grid, powers, target):
    for N, pw in zip(grid, powers):
        if pw >= target:
            return N
    return None


def main():
    s = latent_scale()
    print("Assumptions:", {k: CONFIG[k] for k in
          ("C", "alpha_base", "sigma_nu", "R", "n_covariates", "alpha_level", "target_power")})
    print("\n--- Analysis A: group ranking recovery (Bradley-Terry vs true order) ---")
    A = analysis_A(s)
    print(f"{'N':>5} {'meanTau':>9} {'P(tau>=.90)':>12} {'P(exact)':>9}  (C={CONFIG['C']})")
    for r in A:
        print(f"{r['N']:>5} {r['mean_tau']:>9.3f} {r['p_tau_ge_090']:>12.2f} {r['p_exact']:>9.2f}")
    nA = recommend([r["N"] for r in A], [r["p_tau_ge_090"] for r in A], CONFIG["target_power"])
    print(f"--> N for P(tau>=.90) >= {CONFIG['target_power']:.0%}: {nA}")

    print("\n--- Analysis B: demographic moderation (regression power) ---")
    B, realized = analysis_B(s)
    hdr = "   ".join(f"mod={m} (r~{realized[m]:+.2f})" for m in CONFIG["effect_mods"])
    print(f"{'N':>5}   {hdr}")
    grid = CONFIG["n_grid"]
    for i, N in enumerate(grid):
        print(f"{N:>5}   " + "        ".join(f"{B[m][i]:>5.2f}" for m in CONFIG["effect_mods"]))
    print()
    for m in CONFIG["effect_mods"]:
        nB = recommend(grid, B[m], CONFIG["target_power"])
        print(f"--> effect mod={m} (realized r~{realized[m]:+.2f}): "
              f"N for {CONFIG['target_power']:.0%} power = {nB}")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].plot(grid, [r["p_tau_ge_090"] for r in A], "o-", label="P(τ ≥ 0.90)")
    ax[0].plot(grid, [r["p_exact"] for r in A], "s--", color="#888", label="P(exact order)")
    ax[0].axhline(CONFIG["target_power"], ls=":", color="r")
    ax[0].set_title(f"A. Group ranking recovery  (C={CONFIG['C']}/person)")
    ax[0].set_xlabel("# participants (N)"); ax[0].set_ylabel("probability"); ax[0].set_ylim(0, 1.02); ax[0].legend()
    for m in CONFIG["effect_mods"]:
        ax[1].plot(grid, B[m], "o-", label=f"r ≈ {realized[m]:+.2f}")
    ax[1].axhline(CONFIG["target_power"], ls=":", color="r", label=f"{CONFIG['target_power']:.0%} power")
    ax[1].set_title("B. Detecting a risk-attitude (Holt-Laury) effect")
    ax[1].set_xlabel("# participants (N)"); ax[1].set_ylabel("power"); ax[1].set_ylim(0, 1.02); ax[1].legend()
    fig.suptitle("Power analysis — human perception ranking study", fontweight="bold")
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "power_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")


# ======================================================================================
# SESOI-based design analysis for H1 and H3 (Lakens 2022).
# Power is computed for a TRUE effect equal to the pre-registered SESOI — NOT the pilot
# effect size. The pilot is used ONLY to calibrate the nuisance parameter (choice
# consistency alpha), by matching the pilot's split-half reliability (~0.82 at N=11).
# ======================================================================================
from scipy.stats import spearmanr, wilcoxon

def _group_p(mats):
    W = np.sum(mats, axis=0)
    return bt_fit(W, W + W.T)

def _reliability(mats, rng):
    """Split-half reliability (Spearman-Brown corrected) of the group BT strengths."""
    idx = np.arange(len(mats)); rng.shuffle(idx); h = len(idx) // 2
    p1 = _group_p([mats[i] for i in idx[:h]]); p2 = _group_p([mats[i] for i in idx[h:]])
    r = spearmanr(p1, p2).correlation
    if not np.isfinite(r): r = 0.0
    return 2 * r / (1 + r) if r < 1 else 1.0

def _sim_wins(N, C, s, alpha, rng, sigma_het=0.0):
    """Each rater has a personal scale s_i = s + N(0, sigma_het) (between-rater heterogeneity),
    then makes C noisy pairwise choices (choice consistency alpha)."""
    n = len(s); mats = []
    for _ in range(N):
        si = s + sigma_het * rng.standard_normal(n)
        a = rng.integers(0, n, C); b = rng.integers(0, n, C); b = np.where(a == b, (b + 1) % n, b)
        ca = rng.random(C) < 1.0 / (1.0 + np.exp(-alpha * (si[a] - si[b])))
        win = np.where(ca, a, b); lose = np.where(ca, b, a)
        w = np.zeros((n, n)); np.add.at(w, (win, lose), 1)
        mats.append(w)
    return mats

def _interrater_tau(mats):
    """Mean pairwise Kendall tau between individual raters' BT rankings."""
    ps = [bt_fit(m, m + m.T) for m in mats]
    ts = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            ts.append(kendall_tau_order(ps[i], ps[j]))
    return float(np.mean(ts)) if ts else 0.0

def _human_scale(s_auc, swaps=((1, 2), (4, 5))):
    """A human latent scale that diverges from AUC at ~the SESOI (Kendall tau ~0.8)."""
    order = list(np.argsort(-s_auc))                       # AUC order (strongest first)
    for i, j in swaps: order[i], order[j] = order[j], order[i]
    s = np.zeros(len(s_auc))
    vals = np.sort(s_auc)[::-1]
    for rank, e in enumerate(order): s[e] = vals[rank]
    return s

def run_h1h3():
    rng = np.random.default_rng(11)
    s_auc = latent_scale()
    s_hum = _human_scale(s_auc)
    tau_true = kendall_tau_order(s_hum, s_auc)
    C = 30                                                 # comparisons/rater (10/task x 3)
    print("=== SESOI-based design analysis (H1, H3) ===")
    print(f"true human-vs-AUC Kendall tau at the H1 SESOI boundary = {tau_true:+.2f} "
          f"(=> population D = 1 - tau = {1 - tau_true:.2f} ~ SESOI 0.20)")

    # jointly calibrate (alpha, sigma_het) to BOTH pilot targets at N=11:
    #   split-half reliability ~ 0.82  AND  mean inter-rater tau ~ 0.15
    best = None
    for alpha in [0.8, 1.2, 1.6, 2.0, 2.6, 3.2]:
        for shet in [0.4, 0.8, 1.2, 1.8, 2.4, 3.0]:
            R_, T_ = [], []
            for _ in range(80):
                m = _sim_wins(11, C, s_hum, alpha, rng, shet)
                R_.append(_reliability(m, rng)); T_.append(_interrater_tau(m))
            r, t = np.mean(R_), np.mean(T_)
            loss = (r - 0.82) ** 2 + (t - 0.15) ** 2
            if best is None or loss < best[0]: best = (loss, alpha, shet, r, t)
    _, alpha, shet, r_hat, t_hat = best
    print(f"calibrated alpha={alpha}, sigma_het={shet}  -> N=11 split-half={r_hat:.2f} (target .82), "
          f"inter-rater tau={t_hat:+.2f} (target .15)", flush=True)

    # --- H1 power: P(95% bootstrap CI of D excludes 0), D = R_sh - tau(group, AUC) ---
    for N in (120, 150):
        R, B, rej = 400, 500, 0
        for _ in range(R):
            mats = _sim_wins(N, C, s_hum, alpha, rng, shet)
            Ds = []
            for _ in range(B):
                bs = [mats[i] for i in rng.integers(0, N, N)]
                D = _reliability(bs, rng) - kendall_tau_order(_group_p(bs), s_auc)
                Ds.append(D)
            if np.percentile(Ds, 2.5) > 0: rej += 1
        print(f"H1 power @ N={N}: {rej / R:.3f}  (test: bootstrap 95% CI of D excludes 0)", flush=True)

    # --- H3 power: one-sided Wilcoxon on per-rater gap g=tau_SC - tau_BB, true median = SESOI ---
    print("H3 (one-sided Wilcoxon, true median gap = SESOI 0.15; pilot n=6 too small to pin the")
    print("    per-rater spread, so we sweep plausible SDs):")
    for N in (120,):
        for sd in (0.30, 0.40, 0.50):
            rej = 0; R = 10000
            for _ in range(R):
                g = rng.normal(0.15, sd, N)
                try:
                    p = wilcoxon(g, alternative="greater").pvalue
                    if p < 0.05: rej += 1
                except ValueError:
                    pass
            print(f"   N={N}, gap SD={sd}: power = {rej / R:.3f}", flush=True)


if __name__ == "__main__":
    import sys
    if "h1h3" in sys.argv: run_h1h3()
    else: main()
