# Human Risk-Attitude **Perception Ranking** Experiment

A parallel human study: a fresh group of raters pairwise-rank the 7 intelligent
entities (6 LLMs + aggregate Human) by **perceived risk-taking**, judging from
*behavior alone*. The resulting human-perceived ranking is then compared
(Kendall τ / Spearman) against the paper's OLR+AUC ranking. Divergence is the
target finding: the AUC integrates a normalized B_C→R_D curve uniformly, whereas
humans integrate behavior holistically (consequence salience, no B_C
normalization).

## Files
- `index.html` — the data-collection webpage (open directly in a browser; reads `stimuli.js`)
- `stimuli.js` / `stimuli.json` — generated stimulus set (`window.STIMULI`)
- `build_stimuli.py` — samples the per-entity trial pools → `stimuli.js/json` + `build_report.txt`
- `analyze_ranking.py` — Bradley-Terry ranking (per-task + aggregate) vs AUC → `ranking_comparison.png`
- `power_analysis.py` — simulation-based power curves → `power_analysis.png`
- `PREREGISTRATION.md` — OSF-ready confirmatory analysis plan
- `backend_google_apps_script.gs` — optional Google Sheets backend (alternative to DataPipe)
- `results/` — drop participant JSONs here for `analyze_ranking.py`

## How to run
1. `python build_stimuli.py`  (regenerates stimuli from the raw data)
2. Open `index.html` in a browser. Flow: consent → (auto ID) → questionnaire → comparisons →
   confidence item → (brand ranking) → blocking save → results. Append `?demo=1` for a fast preview.

## Prolific launch checklist
- [ ] **Backend = DataPipe → OSF** (chosen): set `CONFIG.dataPipeID` to your DataPipe experiment ID and turn
      **off** DataPipe "Enable data validation" (our data is a custom JSON object, not jsPsych rows).
      Data auto-submits at the end and the save is CORS-confirmed. (Google Apps Script via `CONFIG.dataEndpoint`
      + `backend_google_apps_script.gs` remains as a fire-and-forget alternative.)
- [ ] Set `CONFIG.completionCode` to your Prolific study's completion code (until then it shows the `CHANGEME-CC` placeholder).
- [ ] Fill the IRB protocol # / PI on the consent screen (`screenConsent`).
- [ ] Host the folder on **GitHub Pages** (or Netlify) so `index.html` + `stimuli.js` load over https;
      give Prolific the URL `https://<user>.github.io/<repo>/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}`.
- [ ] **Unique IDs:** no manual entry — `subjectId` = the URL `PROLIFIC_PID` (real participants) or an
      auto-generated `anon_*` id (local testing). This removes the duplicate-ID problem.
- [ ] **Pilot first** (see `PREREGISTRATION.md`): ~5–8 friends (usability) then ~20–25 on Prolific to verify
      an aggregate signal is reproducible + calibrate QC/timing; keep pilot data OUT of the confirmatory sample.
      Then re-run `power_analysis.py` and finalize N (~150 recruited / ~120 analyzed).

## Risk-attitude measures collected
- **DOSPERT-R (30 items)** — validated, standard, **scored unchanged** (overall + 5 domain means). Domains map
  to tasks for a domain-matched analysis: Financial↔FIP, Health/Safety↔TPB, Recreational↔DSB.
- **SOEP** single general-risk item (0–10) — quick benchmark.
- **Exploratory AI / situational-risk module (6 items)** — *non-validated*, scored **separately**
  (`demographics.aiRisk`); never mixed into the DOSPERT score. Taps AI-delegation/trust and risk-action under
  uncertainty. Treat as exploratory, not confirmatory.

## Quality control (built in)
- **Attention checks:** `CONFIG.attentionChecks` instructed-response comparison trials ("click Agent ①") +
  one directed DOSPERT item ("select 4"). Pass/fail recorded.
- **Careless-speed guard:** choice buttons stay disabled for `CONFIG.minChoiceMs` (default **2500 ms**, so
  raters must actually view the examples); a response is flagged `fast` if it lands within
  `CONFIG.fastGraceMs` (default 600 ms) of the buttons unlocking.
- **Test–retest consistency:** `CONFIG.repeatFraction` of comparisons are re-shown later (side re-randomised);
  `quality.repeatConsistency` = fraction answered the same way (side-independent).
- **Idle / disengagement:** per-trial RT (a trial > `CONFIG.idleTrialMs` is `idle`); **tab-hidden time + blur
  count are scoped to the comparison task only** (from `state.tStart`), so `hiddenMs` can never exceed task
  wall-time and isn't inflated by idling during consent/questionnaire. Active-vs-wall time also recorded.
  `beforeunload` warns against accidental exit, including on the results page until the save has completed.
- **Straight-lining:** DOSPERT near-zero variance flagged.
- **Metacognition:** one post-task confidence item (`taskConfidence`, 0–10) → enables a confidence × actual-
  consistency gap analysis.
- **Blocking save:** the results screen shows a non-dismissable "Saving…" overlay; the completion code +
  return-to-Prolific button appear only after the save succeeds (prevents exit before data is stored).
- **Quality summary** (`quality` in the export) bundles all of the above and sets a single
  `quality.flagged` boolean for easy exclusion.

## Analysis & pre-registration
- `analyze_ranking.py` → per-task + aggregate Bradley-Terry rankings, Kendall τ vs the per-task/overall AUC
  rankings (`PAPER_PERTASK` / `PAPER_OVERALL`, filled), Kendall's W, and a clean rank heatmap `ranking_comparison.png`.
- `power_analysis.py` → simulation power curves (group ranking recovery + demographic-moderation regression).
- `PREREGISTRATION.md` → OSF-ready plan: confirmatory H1–H4 (H4 two-sided), exclusions, sample size,
  and the exploratory list. Pilot data is calibration-only and excluded from the confirmatory sample.

Tune the session at the top of `index.html` → `CONFIG`:
`tasks`, `maxComparisons` (0 = all 189), `allowEqual`, `doBrandRanking`.

## Design (agreed)
- **No rationale/reasoning text** is ever shown (prevents human/AI identity leak and verbal-style confound).
- **Anonymized blind** pairwise comparison (Agent ① vs ②); true entity keys stored only in the exported data.
- **Flow:** intro → intake → pairwise comparisons → optional brand-impression ranking → results + JSON export.
  Intake measures risk attitude with the **Holt & Laury (2002)** 10-row Multiple Price List (a behavioral/
  revealed-preference lottery task; scored to #safe choices → CRRA risk-aversion class, switch row, and a
  consistency check) plus the **SOEP** single general-risk item (0–10; Dohmen et al. 2011) as a quick
  self-report benchmark. Also collects age and AI-use frequency. (Lotteries are hypothetical by default;
  can be incentivized via a random-row bonus if run on a platform like Prolific.)
- **Trial selection = block-design / Latin-hypercube (10 per entity per task).** Each entity's ~100 trials
  are sorted by an objective difficulty proxy, split into 10 equal blocks (deciles), and **one trial is drawn
  at random per block** (fixed seed). This spans the full range, is reproducible, and is *not* selected on any
  outcome → answers "do 10 trials represent the 100?" and defends against cherry-picking critiques.
    - DSB proxy = total realized `|drift|`; FIP proxy = realized volatility (stdev of high-vol % series).
    - TPB has no continuous proxy → blocks = ground-truth `severity` with a **fixed mix (5 STABLE / 3 HR / 2 LSI)
      identical across entities** (so all agents are judged on equally-sick patients); trials with ≥5 observed
      ticks preferred so vital-sign trends are displayable.
- Each comparison shows the two entities' **10 representative trials each**, spanning the difficulty range, so
  the two sides are matched in distribution without per-trial alignment.
- **B_C is NOT aligned** — perceived risk is shown as part of each entity's behavior; humans weigh it holistically.
- **Human = population aggregate** (pooled across participants), consistent with the paper's baseline.

## Stimulus rendering
- **DSB:** grid with the flown route (blue), start/goal, collisions (red), and **inferred obstacles (black)**
  reconstructed from collisions + blocked horizontal moves *around the trajectory* (the full map has no seed and
  is not recoverable). Each card shows the **designed wind** for its condition — safe ↕[−1,+1]@10%,
  borderline ↕[−2,+2]@20%, dangerous ↕[−3,+3]@30% (human trials → nearest level by realized drift).
- **FIP:** L/M/H price lines on a **fixed shared y-axis** (robust 2nd–98th pct) so volatility is comparable
  across charts; allocation stacked bar (H weight = risk-taking).
- **TPB:** **trend sparklines** for HR, systolic BP, SpO2 over the observed ticks, plus final ESI and perceived
  risk. Glossary explains ESI 1–5 (resuscitation → non-urgent).

## Results screen + analysis
- The participant-facing results screen shows **only the participant's own ranking**, framed neutrally
  ("no right answer"). The **AUC benchmark and Kendall τ are deliberately NOT shown to participants**
  (avoids implying humans are "wrong" / anthropocentric-bias critique); they are still stored in the
  exported data (`kendallTau_vs_paperAUC`, `paperAUCRanking`) for researcher analysis only.
- `analyze_ranking.py` mirrors the paper's structure: a **Bradley-Terry ranking per task** (DSB/TPB/FIP),
  then an **aggregate by mean per-task rank**, plus **Kendall's W** (cross-task rank stability) and **Kendall τ**
  vs the AUC ranking → `ranking_comparison.png`. Fill `PAPER_PERTASK` with the three per-task AUC orderings
  (and `PAPER_OVERALL`) from the paper to light up the per-task human-vs-AUC slopegraphs.
- The webpage uses a **balanced per-task design** (`CONFIG.comparisonsPerTask` distinct pairs per task per
  participant), so each task yields its own clean group ranking — structurally parallel to the per-task AUC.
  Group per-task rankings are recovered at ~7-10 comparisons/task/participant × N≈100-150 (see `power_analysis.py`);
  individual per-task rankings are *not* attempted (too noisy) — everything is aggregated, exactly as the paper
  aggregates per-task AUC across the population.
- Paper AUC overall ranking (Fig. 5a mean), most→least risk-taking:
  `GPT-5.2 > Human > Qwen3-Max > Gemini-3-Pro > DeepSeek-V3.2 > Sonnet-4.5 > Grok-4*` (Grok-4 failed intra-consistency).

## Aggregation (analysis, not in this webpage)
Pairwise "more risk-taking" choices → **Bradley–Terry / Thurstone** latent scale →
perceived ranking per task and overall → compare to AUC ranking.

## Data sources & caveats
- LLM raw: `PNAS_RawDateWithConditionLabel/Data_Clean/{DSB,FIP,TPB} Results/*.json`
  (loader handles both `trials` and `groups[].trials`; Sonnet FIP files use the flat `trials` form).
- Human raw: **`PNAS Human Experiment.xlsx`** (full). The `*_Cleaned.xlsx` copies
  **truncate human FIP** at Excel's 32 767-char cell limit (price series) — do **not** use them for FIP/TPB raw trials.
- Human DSB quality filter: `Quality_Report.DSB Usable` is reliable; its FIP/TPB
  columns were computed on the truncated data and are **all False / unusable** — ignored here.
- **DSB conditions** (safe/borderline/dangerous) are a designed slider manipulation
  (volatility/gust/wall-prob/drift 0.1 → 0.15 → 0.2, drift-prob 0.1→0.2→0.3). The stored
  `factors` field only holds coarse labels and looks constant; the real condition lives
  in the filename and shows up as realized drift (median 2 / 10 / 25). Humans were **not**
  run under these discrete settings, so all entities are placed on a common axis via the
  realized-drift proxy.
- **TPB timeline is not recoverable**: per-agent `steps` only logs the ticks where the
  agent updated its belief (`min_required_updates: 2`), not every patient tick. So TPB shows
  a **uniform patient snapshot** (header + final logged vitals + flags + decision), and the
  sampler prefers trials with more logged ticks (selected median ≈ 8).
