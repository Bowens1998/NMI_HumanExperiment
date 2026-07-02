# Pre-registration — Human perception of AI (and human) risk attitudes

**Draft for OSF registration (project osf.io/beq4y). Edit, then freeze before main data collection.**

## 1. Background & aim
Prior work (PNAS) characterized the risk attitudes of 6 LLMs + an aggregate human baseline
across three tasks (DNC/drone navigation, CTD/clinical triage, FIP/financial allocation) using an
Ordered-Logistic-Regression **AUC** measure of the belief→decision mapping. Here we ask how **lay
humans perceive** those same agents' risk-taking from their behavior, via blind pairwise comparison,
and how that perception relates to (a) the normative AUC ranking and (b) raters' own risk attitude.
Thesis: human perception of AI risk is **uncertain and not as internally consistent as people assume**.

## 2. Design
- Anonymized **pairwise forced choice**: "which agent behaves in a more risk-taking way?" (+"about the same").
- Each agent shown via **4 difficulty-matched behavioral exemplars** per comparison (environment + the agent's
  reported belief B_C + final action); no reasoning text. Belief is always shown (it is part of the agent's
  belief→action response, mirroring how AUC is computed).
- **Balanced per-task design**: `comparisonsPerTask` distinct pairs per task per participant, all three tasks.
- Entities (7): DeepSeek-V3.2, Gemini-3-Pro, GPT-5.2, Grok-4, Qwen3-Max, Claude-Sonnet-4.5, Human (aggregate).
- Intake: age, gender (not analyzed), AI-use frequency, **SOEP** general-risk item, **DOSPERT-R (30 items)**,
  a short exploratory AI-risk block, and one **post-task confidence** item.
- Group rankings via **Bradley-Terry** (per task + aggregate by mean per-task rank).

## 3. Confirmatory hypotheses
- **H1 (structured divergence).** Human raters agree with *each other* more than with the AUC ranking:
  inter-rater agreement (group ranking self-consistency) exceeds the human–AUC agreement (Kendall τ).
  *(Not merely "τ < 1", which is trivial under noise.)*
- **H2 (weak-but-real consensus amid individual uncertainty).**
  We do **not** predict that individuals agree. We predict:
  (a) the **aggregate** human ranking is **reproducible** (split-half reliability of Bradley-Terry strengths
  > 0; bootstrap CIs separate at least the extreme entities) — i.e., a shared systematic component exists; AND
  (b) **individual-level agreement is low and within-person inconsistency is non-trivial** (mean pairwise
  inter-rater τ is modest; repeated-pair disagreement > 0 and/or measurable intransitivity) — i.e., the
  consensus is weak and noisy, consistent with cognitive uncertainty. *Low individual agreement is a
  predicted result, not a failure; the only fatal outcome is a non-reproducible aggregate (checked in pilot).*
- **H3 (belief–behavior contradiction).** Within participants, the **brand-impression** ranking of the 6 named
  LLMs diverges from their **behavior-based** ranking **beyond each rater's own noise ceiling**
  (τ_brand,behavior < τ_self-consistency of the behavior ranking).
- **H4 (moderation, two-sided).** A participant's **DOSPERT overall** score is **associated** (no predicted
  direction) with their agreement (Kendall τ) with the AUC ranking, controlling for age and AI-use frequency.

*The paper does not hinge on H4; H1–H3 stand independently.*

## 4. Primary analyses
- Group ranking: Bradley-Terry (per task + aggregate). **Bootstrap over participants** for 95% CIs on ranks/strengths.
- H1: compare inter-rater agreement (mean pairwise Kendall τ between raters, or split-half group τ) vs human–AUC τ.
- H2: (a) permutation test that inter-rater agreement > chance; (b) aggregate repeated-pair consistency and
  intransitive-triad rate, reported with CIs.
- H3: per-rater τ(brand, behavior) vs per-rater behavior self-consistency (split-half / repeated pairs); paired test.
- H4: linear regression `agreement_with_AUC ~ DOSPERT_overall + age + AI_use` (two-sided test on DOSPERT coef).

## 5. Sample size & stopping rule
- **Main sample: recruit ~150, target ~120 analyzable.** Powers detection of a small-to-medium moderation
  effect (r ≈ 0.25) at 80% power, α = .05 (see `power_analysis.py`); group/per-task rankings are far
  over-powered at this N. Stop at the target; no optional continuation.
- **Pilot (separate, calibration only, NOT included in the confirmatory sample):** ~5–8 friends (usability) then
  ~20–25 on Prolific (verify inter-rater signal, calibrate QC thresholds, timing, choice-consistency α).

## 6. Exclusion criteria (fixed in advance)
Exclude a participant if ANY of:
- fails ≥1 instructed-response attention check, OR the DOSPERT directed item;
- DOSPERT straight-lining (near-zero variance / modal-response > 0.9);
- >50% of comparisons faster than `minChoiceMs + fastGraceMs`;
- repeated-pair consistency < 0.5 (with ≥4 repeats);
- total tab-hidden time > [threshold, set after pilot].
(Report results with and without exclusions as a robustness check.)

## 7. Exploratory (NOT confirmatory; reported as tentative)
- Domain-matched moderation (DOSPERT-Financial↔FIP, Health↔CTD, Recreational↔DNC).
- Other demographics (age, AI-use) as predictors; per-entity and per-task breakdowns.
- Projection bias: does DOSPERT predict an overall shift in perceived risk level?
- Confidence × consistency gap (metacognitive calibration); reaction-time patterns.
- Whether human ranking is explained by displayed B_C alone (readout confound check).
- **Country/company prior bias.** Each model's developer company and country are shown ONLY post-task
  (in the brand-impression drag ranking and the debrief), never during the blind pairwise comparisons.
  Exploratory questions: (a) does the **brand-impression** ranking cluster by country of origin (e.g., are
  China-developed models — DeepSeek, Qwen — systematically rated more/less risk-taking than US-developed
  ones)? (b) does country/company predict the **brand-vs-behavior gap** (do priors about a model's origin
  diverge more from its observed behavior)? (c) does a rater's own nationality/region (if collected) interact
  with this? Because origin labels are shown, any such effect reflects a *stated prior*, not behavior-based
  perception, and is interpreted accordingly.

## 8. Data & code
Data collected via DataPipe → OSF (project beq4y / component 7pwm9). Analysis: `analyze_ranking.py`,
`power_analysis.py`. Stimuli generated by `build_stimuli.py` (fixed seed).
