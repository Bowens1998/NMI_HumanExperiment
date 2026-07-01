#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build stimuli for the human RANKING experiment.

For each of the 7 entities (6 LLMs + aggregate Human) and each task (DSB/FIP/TPB),
select 10 trials that REPRESENT that entity's full ~100-trial pool, using a
block-design / Latin-hypercube sampling on the objective difficulty axis:

  * The trials are sorted by an objective difficulty proxy and split into 10 equal
    blocks (deciles); ONE trial is drawn at random from each block (fixed seed).
    This spans the full difficulty range, is reproducible, and is not cherry-picked
    on any outcome (B_C / decision) -> defends against selective-stimulus critiques
    and supports the "10 trials represent 100" claim.
      - DSB : proxy = total realized |drift|
      - FIP : proxy = realized volatility (stdev of high-vol % series)
      - TPB : no continuous proxy -> blocks = ground-truth severity {STABLE,HR,LSI},
              allocated proportionally; only trials with >=5 observed ticks are used
              (so vital-sign trends are displayable; tick-count is not an outcome).

No rationale/reasoning text is exported.

Outputs: stimuli.json / stimuli.js (window.STIMULI) / build_report.txt
Structure: STIMULI.stimuli[task][entity] = [ up to 10 trial dicts ]
"""

import json, glob, math, os, re, random, statistics as st
from collections import defaultdict, Counter
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_DIR = os.path.join(ROOT, "PNAS_RawDateWithConditionLabel", "Data_Clean")
HUMAN_XLSX = os.path.join(ROOT, "PNAS_HumanExperiments", "HumanExperimntResults",
                          "PNAS Human Experiment.xlsx")
QUALITY_XLSX = os.path.join(ROOT, "PNAS_HumanExperiments", "HumanExperimntResults",
                            "PNAS_Deep_Cleaned.xlsx")

# Ship a POOL per entity (not the final shown set). The webpage draws a per-participant
# representative subset from this pool, so different participants see different trials.
K_PER_ENTITY = 30
TPB_RATIO = (0.5, 0.3, 0.2)        # STABLE / HR / LSI proportions
TPB_MIN_TICKS = 5
SEED = 20260623
EPS = 1e-6
GRID_ROWS, GRID_COLS, GOAL = 10, 20, [5, 19]
DIRS = {"RIGHT": (0, 1), "LEFT": (0, -1), "UP": (-1, 0), "DOWN": (1, 0)}
# Designed DSB wind per condition: level -> (drift range +/-, drift probability)
COND_WIND = {"safe": (1, 0.10), "borderline": (2, 0.20), "dangerous": (3, 0.30)}

LLM_FILE_TOKENS = {"DeepSeekV3.2": "deepseek", "Gemini3Pro": "gemini", "GPT5.2": "gpt",
                   "Grok4": "grok", "Qwen3Max": "qwen", "Sonnet4.5": "sonnet"}
DISPLAY = {"deepseek": "DeepSeek-V3.2", "gemini": "Gemini-3-Pro", "gpt": "GPT-5.2",
           "grok": "Grok-4", "qwen": "Qwen3-Max", "sonnet": "Claude-Sonnet-4.5",
           "human": "Human (aggregate)"}
ENTITIES = ["deepseek", "gemini", "gpt", "grok", "qwen", "sonnet", "human"]


# --------------------------------------------------------------------------- #
def iter_trials(blob):
    if isinstance(blob, dict):
        if isinstance(blob.get("trials"), list):
            yield from blob["trials"]
        for g in blob.get("groups", []) or []:
            yield from g.get("trials", []) or []


def load_llm_files(tag):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(LLM_DIR, f"{tag} Results", f"{tag}_*.json")):
        _, model_tok, cond = os.path.basename(f)[:-5].split("_", 2)
        key = LLM_FILE_TOKENS.get(model_tok)
        if key is None:
            continue
        with open(f, encoding="utf-8") as fh:
            blob = json.load(fh)
        for t in iter_trials(blob):
            out[key].append((t, cond))
    return out


def load_human_trials(sheet):
    df = pd.read_excel(HUMAN_XLSX, sheet_name=sheet)
    rows = []
    for _, r in df.iterrows():
        raw = r.get("Data (JSON)")
        if not isinstance(raw, str):
            continue
        try:
            blob = json.loads(raw)
        except Exception:
            continue
        for t in iter_trials(blob):
            rows.append((r.get("Subject ID"), t))
    return rows


def dsb_usable_subject_ids():
    try:
        q = pd.read_excel(QUALITY_XLSX, sheet_name="Quality_Report")
        return set(q.loc[q["DSB Usable"] == True, "Subject ID"])      # noqa: E712
    except Exception:
        return None


def parse_esi(val):
    if isinstance(val, (int, float)):
        return int(val)
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else None


def parse_sys(bp):
    m = re.match(r"\s*(\d+)", str(bp or ""))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
def infer_walls(steps, visited):
    walls = set()
    prev = None
    for s in steps:
        a, r, c, coll = s.get("action"), s.get("row"), s.get("col"), s.get("collision", 0) or 0
        d = DIRS.get(a, (0, 0))
        if coll:
            w = (r + d[0], c + d[1])
            if 0 <= w[0] < GRID_ROWS and 0 <= w[1] < GRID_COLS:
                walls.add(w)
        if prev and a in ("LEFT", "RIGHT") and c == prev[1]:   # drift is row-only
            w = (prev[0], prev[1] + d[1])
            if 0 <= w[0] < GRID_ROWS and 0 <= w[1] < GRID_COLS:
                walls.add(w)
        prev = (r, c)
    return [list(w) for w in sorted(walls - visited)]


def dsb_record(trial, entity, source, sid, designed):
    steps = trial.get("steps") or []
    if not steps:
        return None
    drifts = [s.get("drift_row", 0) or 0 for s in steps]
    sum_drift = sum(abs(x) for x in drifts)
    n_coll = sum(s.get("collision", 0) or 0 for s in steps)
    acts = [s.get("action") for s in steps]
    n = len(acts) or 1
    si = math.log(((acts.count("UP") + acts.count("DOWN")) / n + EPS) /
                  (acts.count("RIGHT") / n + EPS))
    visited = {(s.get("row"), s.get("col")) for s in steps}
    path = [[s.get("row"), s.get("col"), s.get("drift_row", 0) or 0, s.get("collision", 0) or 0]
            for s in steps]
    # designed wind: LLM trials carry their condition; human trials -> nearest level by realized drift
    if designed in COND_WIND:
        level = designed
    else:
        mx = max((abs(x) for x in drifts), default=0)
        level = "safe" if mx <= 1 else ("borderline" if mx <= 2 else "dangerous")
    rng_, prob_ = COND_WIND[level]
    behavior = {
        "goal": GOAL, "rows": GRID_ROWS, "cols": GRID_COLS,
        "end_reason": trial.get("end_reason"), "path": path,
        "walls": infer_walls(steps, visited),
        "wind": {"min": -rng_, "max": rng_, "prob": prob_, "level": level},
        "n_steps": len(steps), "n_coll": n_coll, "sum_drift": sum_drift,
    }
    return dict(entity=entity, source=source, designed=designed,
                B_C=trial.get("context_belief"), R_D=round(si, 3),
                proxy=float(sum_drift), behavior=behavior)


def fip_record(trial, entity, source, sid, designed):
    rep = trial.get("report") or {}
    alloc, ctx = rep.get("alloc") or {}, (rep.get("contextual") or {}).get("risk")
    pct_h = (trial.get("pct") or {}).get("H") or []
    if not alloc or not pct_h:
        return None
    vol = st.pstdev(pct_h) if len(pct_h) > 1 else 0.0
    prices = trial.get("prices") or {}
    behavior = {
        "alloc": {k: alloc.get(k) for k in ("L", "M", "H")}, "risk": ctx,
        "prices": {k: [round(x, 3) for x in (prices.get(k) or [])] for k in ("L", "M", "H")},
        "state_final": trial.get("state_final"), "vol": round(vol, 3),
    }
    return dict(entity=entity, source=source, designed=designed,
                B_C=ctx, R_D=alloc.get("H"), proxy=float(vol), behavior=behavior)


def tpb_record(trial, entity, source, sid, designed):
    steps = trial.get("steps") or []
    sev = trial.get("severity")
    esi = parse_esi((trial.get("finalized") or {}).get("ESI"))
    if not steps or sev is None or esi is None:
        return None
    last_ctx = None
    trend = []
    for s in steps:
        for u in (s.get("BC_updates") or []):
            last_ctx = u.get("ctx", last_ctx)
        v = s.get("vitals") or {}
        trend.append({"t": s.get("tick"), "HR": v.get("HR"),
                      "BP": parse_sys(v.get("BP")), "SpO2": v.get("SpO2")})
    behavior = {
        "patient": trial.get("patient"), "cc": trial.get("cc"),
        "severity": sev, "ESI": esi, "maxT": trial.get("maxT"),
        "n_ticks": len(steps), "final_vitals": (steps[-1].get("vitals")),
        "final_flags": steps[-1].get("flags"), "trend": trend,
    }
    return dict(entity=entity, source=source, designed=designed,
                B_C=last_ctx, R_D=esi, proxy=None, behavior=behavior, severity=sev)


TASKS = {"DSB": ("DSB", "DSB", dsb_record, "sum|drift|"),
         "FIP": ("FIP", "FIP", fip_record, "vol(H%)"),
         "TPB": ("TPB", "TPB", tpb_record, "severity")}


def collect_records(task):
    tag, sheet, rec, _ = TASKS[task]
    recs = []
    for key, items in load_llm_files(tag).items():
        for trial, cond in items:
            r = rec(trial, key, "llm", None, cond)
            if r:
                recs.append(r)
    usable = dsb_usable_subject_ids() if task == "DSB" else None
    for sid, trial in load_human_trials(sheet):
        if usable is not None and sid not in usable:
            continue
        r = rec(trial, "human", "human", str(sid), "human_mixed")
        if r:
            recs.append(r)
    return recs


# --------------------------------------------------------------------------- #
def lhs_select(recs, k, rng):
    """1-D Latin-hypercube / stratified-random on the difficulty proxy."""
    recs = [r for r in recs if r["proxy"] is not None]
    if len(recs) <= k:
        return list(recs)
    recs = sorted(recs, key=lambda r: r["proxy"])
    n = len(recs)
    out = []
    for i in range(k):
        lo, hi = i * n // k, (i + 1) * n // k
        out.append(recs[rng.randrange(lo, hi) if hi > lo else min(lo, n - 1)])
    return out


def tpb_select(recs, k, rng):
    """Blocks = ground-truth severity, allocated with a FIXED proportion across entities so all
    raters compare equally-sick patients. Within a block, prefer trials with >=5 observed
    ticks (so vital trends are displayable; tick-count is not an outcome), random order."""
    alloc = {"STABLE": round(k * TPB_RATIO[0]), "HR": round(k * TPB_RATIO[1])}
    alloc["LSI"] = max(0, k - alloc["STABLE"] - alloc["HR"])
    groups = defaultdict(list)
    for r in recs:
        groups[r["severity"]].append(r)
    out = []
    for s in ("STABLE", "HR", "LSI"):
        need = alloc.get(s, 0)
        pool = groups.get(s, [])
        rich = [r for r in pool if r["behavior"]["n_ticks"] >= TPB_MIN_TICKS]
        thin = [r for r in pool if r["behavior"]["n_ticks"] < TPB_MIN_TICKS]
        rng.shuffle(rich); rng.shuffle(thin)
        out += (rich + thin)[:need]
    # if some severity was short, top up from whatever remains
    if len(out) < k:
        rest = [r for r in recs if r not in out]
        rng.shuffle(rest)
        out += rest[:k - len(out)]
    return out[:k]


def build():
    out = {"meta": {"k_per_entity": K_PER_ENTITY, "selection": "block-design LHS on difficulty",
                    "entities": DISPLAY, "tasks": {}}, "stimuli": {}}
    report = []
    for task in ("DSB", "FIP", "TPB"):
        recs = collect_records(task)
        by_ent = defaultdict(list)
        for r in recs:
            by_ent[r["entity"]].append(r)
        rng = random.Random(SEED + hash(task) % 9973)
        out["stimuli"][task] = {}
        report.append(f"\n===== {task} =====  total records: {len(recs)}  proxy: {TASKS[task][3]}")
        risk = {}
        for ent in ENTITIES:
            pool = by_ent.get(ent, [])
            sel = (tpb_select(pool, K_PER_ENTITY, rng) if task == "TPB"
                   else lhs_select(pool, K_PER_ENTITY, rng))
            out["stimuli"][task][ent] = sel
            rds = [s["R_D"] for s in sel if s["R_D"] is not None]
            risk[ent] = round(st.mean(rds), 3) if rds else None
            extra = ""
            if task == "TPB":
                extra = "  sev=" + str(dict(Counter(s["severity"] for s in sel)))
            elif sel:
                pr = [s["proxy"] for s in sel]
                extra = f"  proxy[min/med/max]={min(pr):.1f}/{st.median(pr):.1f}/{max(pr):.1f}"
            report.append(f"   {DISPLAY[ent]:22s}: {len(sel):2d} trials{extra}  meanR_D={risk[ent]}")
        out["meta"]["tasks"][task] = {"proxy": TASKS[task][3]}

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "stimuli.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    with open(os.path.join(here, "stimuli.js"), "w", encoding="utf-8") as fh:
        fh.write("window.STIMULI = ")
        json.dump(out, fh, ensure_ascii=False)
        fh.write(";")
    rep = "\n".join(report)
    with open(os.path.join(here, "build_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep)
    print(rep)
    print("\nWrote stimuli.json / stimuli.js / build_report.txt")


if __name__ == "__main__":
    build()
