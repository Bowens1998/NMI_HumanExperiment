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
    """Bradley-Terry MM on aggregated pairwise win counts. wins[i][j]=#times i beat j."""
    n = wins.shape[0]
    p = np.ones(n)
    W = wins.sum(axis=1)
    for _ in range(200):
        newp = np.zeros(n)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                nij = games_pair[i, j]
                if nij:
                    denom += nij / (p[i] + p[j])
            newp[i] = W[i] / denom if denom > 0 else p[i]
        newp /= np.exp(np.mean(np.log(np.clip(newp, 1e-9, None))))
        p = newp
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


if __name__ == "__main__":
    main()
