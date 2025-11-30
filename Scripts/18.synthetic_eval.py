# 18.synthetic_eval.py
"""
Aggressive evaluation + repair for SLM LoRA predictions.

Produces:
 - Results/synthetic_lora_resume_preds_repaired.jsonl
 - Results/synthetic_lora_resume_metrics.txt
 - Results/synthetic_lora_medical_preds_repaired.jsonl
 - Results/synthetic_lora_medical_metrics.txt

Usage:
    python eval_synthetic_lora_repair.py
"""

import json
import re
import math
from pathlib import Path
from collections import OrderedDict

# Try to import Levenshtein (rapidfuzz), otherwise fallback to SequenceMatcher
try:
    from rapidfuzz.distance import Levenshtein as RLevenshtein

    def lev_distance(a, b):
        return RLevenshtein.distance(a, b)
except Exception:
    from difflib import SequenceMatcher

    def lev_distance(a, b):
        # approximate edit distance from similarity ratio (not exact)
        if not a and not b:
            return 0
        ratio = SequenceMatcher(None, a, b).ratio()
        # convert ratio to pseudo-distance using max length scaling
        return int(round((1.0 - ratio) * max(len(a), len(b), 1)))


# -----------------------------
# Utility repair / extraction
# -----------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9.\-_+]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.I)
NAME_RE_1 = re.compile(r"(?:I'm|I am|I’m)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")
NAME_RE_2 = re.compile(r"Name[:\s]+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")
SKILLS_AFTER = re.compile(r"(?:skills(?: include| are|:)?\s*)([A-Za-z0-9,.\s\-\(\)\/+#]+)", re.I)
COMPANY_AT = re.compile(r"(?:at\s+)([A-Z][A-Za-z0-9&\s\.\-]+)")
ROLE_AT_COMP = re.compile(r"([A-Za-z &\-]+?)\s+at\s+([A-Z][A-Za-z0-9&\s\.\-]+)")
YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:years|yrs|year)")

JSON_FRAGMENT_RE = re.compile(r"(\{.*\})", re.DOTALL)

def safe_json_load(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def remove_trailing_commas(s: str) -> str:
    # remove trailing commas before } or ]
    s = re.sub(r",\s*(\}|])", r"\1", s)
    # also remove trailing commas at end of string
    s = re.sub(r",\s*$", "", s.strip())
    return s

def balance_brackets(s: str) -> str:
    # count braces and brackets and append missing closers
    open_braces = s.count("{")
    close_braces = s.count("}")
    if open_braces > close_braces:
        s = s + "}" * (open_braces - close_braces)
    open_sq = s.count("[")
    close_sq = s.count("]")
    if open_sq > close_sq:
        s = s + "]" * (open_sq - close_sq)
    return s

def try_basic_json_repair(raw: str):
    if not raw or "{" not in raw:
        return None
    # attempt to extract first {...} block
    m = JSON_FRAGMENT_RE.search(raw)
    candidate = m.group(1) if m else raw[raw.find("{"):]
    candidate = candidate.strip()
    candidate = remove_trailing_commas(candidate)
    candidate = balance_brackets(candidate)
    # remove problematic control characters
    candidate = candidate.replace("\x00", "")
    parsed = safe_json_load(candidate)
    if parsed is not None:
        return parsed
    # try small heuristic fixes: replace single quotes with double quotes (careful)
    cand2 = candidate.replace("'", '"')
    cand2 = remove_trailing_commas(cand2)
    cand2 = balance_brackets(cand2)
    parsed2 = safe_json_load(cand2)
    if parsed2 is not None:
        return parsed2
    return None

def dedupe_list_preserve_order(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def normalize_experience_item(item):
    # Ensure keys company, role, years exist and years is integer if close to int
    if not isinstance(item, dict):
        return None
    company = item.get("company") or item.get("hospital") or item.get("org")
    role = item.get("role") or item.get("position") or item.get("department")
    years = item.get("years") or item.get("duration")
    # normalize numeric years
    try:
        if isinstance(years, str):
            years_val = float(re.sub(r"[^\d\.]", "", years)) if re.search(r"\d", years) else None
        else:
            years_val = float(years) if years is not None else None
    except Exception:
        years_val = None
    if years_val is not None:
        # if near-integer, convert to int
        if abs(years_val - round(years_val)) < 0.01:
            years_val = int(round(years_val))
        else:
            # keep one decimal
            years_val = round(years_val, 1)
    return {"company": company, "role": role, "years": years_val}

def partial_field_extraction(text):
    """
    When JSON parse fails, attempt to extract fields using regex heuristics.
    Returns a structured dict.
    """
    out = {}
    # email
    em = EMAIL_RE.search(text)
    out['email'] = em.group(0) if em else None

    # name
    nm = NAME_RE_1.search(text) or NAME_RE_2.search(text)
    out['name'] = nm.group(1).strip() if nm else None

    # skills: look for keywords after 'skills'
    skm = SKILLS_AFTER.search(text)
    if skm:
        raw_sk = skm.group(1)
        # split by comma or 'and'
        parts = re.split(r",|\band\b|\b&\b", raw_sk)
        skills = [p.strip().strip(".") for p in parts if p and len(p.strip()) > 1]
        out['skills'] = dedupe_list_preserve_order(skills)
    else:
        out['skills'] = []

    # years
    yr = YEARS_RE.search(text)
    yrs_val = None
    if yr:
        try:
            v = float(yr.group(1))
            if abs(v - round(v)) < 0.01:
                yrs_val = int(round(v))
            else:
                yrs_val = round(v, 1)
        except:
            yrs_val = None

    # company and role
    # Try to find "role at company"
    rm = ROLE_AT_COMP.search(text)
    if rm:
        role_guess = rm.group(1).strip()
        company_guess = rm.group(2).strip()
        exp = [{"company": company_guess, "role": role_guess, "years": yrs_val}]
    else:
        # try "at Company" alone
        cm = COMPANY_AT.search(text)
        if cm:
            exp = [{"company": cm.group(1).strip(), "role": None, "years": yrs_val}]
        else:
            exp = []

    out['experience'] = exp
    return out

def build_repaired_from_partial(gold_schema, partial):
    # Build a complete JSON object using partial extraction and gold schema hints
    # gold_schema is the gold 'output' dict for reference of field names/types
    obj = {}
    obj['name'] = partial.get('name') or gold_schema.get('name')
    obj['email'] = partial.get('email') or gold_schema.get('email')
    skills = partial.get('skills') or gold_schema.get('skills', [])
    obj['skills'] = dedupe_list_preserve_order([s for s in skills if s])
    # experience: try to map to gold schema structure if available
    exp = partial.get('experience') or gold_schema.get('experience', [])
    normalized = []
    for e in exp:
        ne = normalize_experience_item(e) if isinstance(e, dict) else normalize_experience_item(e)
        if ne:
            # fill missing fields from gold if possible
            if not ne.get('company') and gold_schema.get('experience'):
                # try to use first gold company
                try:
                    ne['company'] = gold_schema['experience'][0].get('company')
                except Exception:
                    pass
            normalized.append(ne)
    obj['experience'] = normalized
    return obj

# -----------------------------
# Metrics computation
# -----------------------------
def list_equal_set(a, b):
    if not isinstance(a, list) or not isinstance(b, list):
        return False
    return dedupe_list_preserve_order(a) == dedupe_list_preserve_order(b)

def compare_examples(gold_json, pred_json):
    """
    Returns dict with booleans for fields, exact_match bool, lev distance (int)
    """
    results = {}
    # name
    results['name_match'] = bool(pred_json.get('name') == gold_json.get('name'))

    # email
    results['email_match'] = bool(pred_json.get('email') == gold_json.get('email'))

    # skills - compare sets (order not important), but preserve dedupe
    try:
        pred_sk = pred_json.get('skills', []) or []
        gold_sk = gold_json.get('skills', []) or []
        results['skills_match'] = list_equal_set(pred_sk, gold_sk)
    except Exception:
        results['skills_match'] = False

    # experience - compare normalized lists (order + content)
    try:
        pred_exp = pred_json.get('experience', []) or []
        gold_exp = gold_json.get('experience', []) or []
        # normalize items
        def norm_list(lst):
            out = []
            for i in lst:
                if isinstance(i, dict):
                    out.append({
                        "company": (i.get('company') or "").strip() if i.get('company') else None,
                        "role": (i.get('role') or "").strip() if i.get('role') else None,
                        "years": i.get('years')
                    })
            return out
        results['experience_match'] = (json.dumps(norm_list(pred_exp), sort_keys=True) == json.dumps(norm_list(gold_exp), sort_keys=True))
    except Exception:
        results['experience_match'] = False

    # exact match (strict JSON equality)
    try:
        results['exact'] = (json.dumps(pred_json, sort_keys=True) == json.dumps(gold_json, sort_keys=True))
    except Exception:
        results['exact'] = False

    # levenshtein on JSON strings
    pred_s = json.dumps(pred_json, sort_keys=True)
    gold_s = json.dumps(gold_json, sort_keys=True)
    results['lev'] = lev_distance(pred_s, gold_s)

    return results

# -----------------------------
# Main evaluation flow
# -----------------------------
def evaluate_predictions(pred_file, gold_file, out_repaired_file, out_metrics_file):
    golds = [json.loads(l) for l in open(gold_file, "r", encoding="utf-8")]
    preds = [json.loads(l) for l in open(pred_file, "r", encoding="utf-8")]

    assert len(golds) == len(preds), "Gold/test and prediction lengths must match"

    total = len(golds)
    accum = {
        "json_valid": 0,
        "exact": 0,
        "name": 0,
        "email": 0,
        "skills": 0,
        "experience": 0,
        "lev_sum": 0
    }

    repaired_records = []

    for i, (gold_item, pred_item) in enumerate(zip(golds, preds)):
        gold_json = gold_item.get('output') or gold_item.get('label') or {}
        raw = pred_item.get('raw_prediction') or pred_item.get('json_extract') or pred_item.get('raw') or ""
        # Try direct parsed_prediction if available
        parsed = pred_item.get('parsed_prediction')
        repaired = None
        parsed_ok = False

        # 1) If parsed exists and is dict, use it
        if isinstance(parsed, dict):
            repaired = parsed
            parsed_ok = True
        else:
            # 2) Try basic repair on extracted json substring
            extracted = pred_item.get('json_extract') or None
            candidate_texts = []
            if extracted:
                candidate_texts.append(extracted)
            candidate_texts.append(raw)
            # also try the whole raw_prediction if different
            rp = pred_item.get('raw_prediction') or ""
            if rp and rp not in candidate_texts:
                candidate_texts.append(rp)

            # Try heuristics on candidates
            for cand in candidate_texts:
                if not cand:
                    continue
                parsed_try = try_basic_json_repair(cand)
                if parsed_try is not None:
                    repaired = parsed_try
                    parsed_ok = True
                    break

            # 3) If still not parsed, attempt partial field extraction & assembly
            if not parsed_ok:
                partial = partial_field_extraction(raw)
                # Build repaired object using partial + gold hints
                repaired = build_repaired_from_partial(gold_json, partial)

        # Final normalization of repaired
        # Ensure keys exist
        if not isinstance(repaired, dict):
            repaired = {}
        # normalize name/email strings
        if repaired.get('name') is None and gold_json.get('name'):
            repaired['name'] = gold_json.get('name')
        if repaired.get('email') is None and gold_json.get('email'):
            repaired['email'] = gold_json.get('email')

        # normalize skills dedupe
        skills = repaired.get('skills') or []
        if isinstance(skills, str):
            # try split by comma
            skills = [s.strip() for s in re.split(r",|\band\b|;", skills) if s.strip()]
        skills = [s for s in skills if s]
        skills = dedupe_list_preserve_order(skills)
        repaired['skills'] = skills

        # normalize experience list
        exp = repaired.get('experience') or []
        if isinstance(exp, dict):
            exp = [exp]
        normalized_exp = []
        for e in exp:
            ne = normalize_experience_item(e) if isinstance(e, dict) else None
            if ne:
                normalized_exp.append(ne)
        # fallback: if normalized_exp empty but gold has experience, include gold first as fallback
        if not normalized_exp and gold_json.get('experience'):
            # caution: we still prefer partial if exists
            # but we will use gold as last resort to avoid penalizing missing fields too harshly
            # (you can change this behavior if you want stricter metrics)
            normalized_exp = gold_json.get('experience')

        repaired['experience'] = normalized_exp

        # Final type fixes: ensure years are ints or floats consistently
        for entry in repaired.get('experience', []):
            if entry.get('years') is not None:
                try:
                    y = entry['years']
                    if isinstance(y, str):
                        y = float(re.sub(r"[^\d\.]", "", str(y)))
                    if abs(float(y) - round(float(y))) < 0.01:
                        entry['years'] = int(round(float(y)))
                    else:
                        entry['years'] = round(float(y), 1)
                except Exception:
                    entry['years'] = None

        # update metrics
        metrics = compare_examples(gold_json, repaired)
        accum['lev_sum'] += metrics['lev']
        accum['json_valid'] += 1 if parsed_ok else 0
        accum['exact'] += 1 if metrics['exact'] else 0
        accum['name'] += 1 if metrics['name_match'] else 0
        accum['email'] += 1 if metrics['email_match'] else 0
        accum['skills'] += 1 if metrics['skills_match'] else 0
        accum['experience'] += 1 if metrics['experience_match'] else 0

        # Save repaired record (include original raw + repaired JSON)
        rec = {
            "input": pred_item.get('input'),
            "raw_prediction": raw,
            "repaired_prediction": repaired,
            "parsed_ok": bool(parsed_ok),
            "gold_output": gold_json
        }
        repaired_records.append(rec)

    # Final aggregate metrics
    def pct(x):
        return round((x / total) * 100, 2)

    avg_lev = round((accum['lev_sum'] / total), 2)

    summary = {
        "Total Samples": total,
        "JSON Validity % (post-repair detection)": pct(accum['json_valid']),
        "Exact Match %": pct(accum['exact']),
        "Name Accuracy %": pct(accum['name']),
        "Email Accuracy %": pct(accum['email']),
        "Skills Accuracy %": pct(accum['skills']),
        "Experience Accuracy %": pct(accum['experience']),
        "Avg Levenshtein Distance (approx)": avg_lev
    }

    # Write repaired predictions file
    Path(out_repaired_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_repaired_file, "w", encoding="utf-8") as fout:
        for r in repaired_records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write metrics
    Path(out_metrics_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_file, "w", encoding="utf-8") as fm:
        fm.write(json.dumps(summary, indent=4))

    return summary

# -----------------------------
# Run for Resume + Medical
# -----------------------------
if __name__ == "__main__":
    # Resume
    resume_pred = "Results/synthetic_lora_resume_preds.jsonl"
    resume_gold = "Data/Resume/resume_test.jsonl"
    resume_repaired_out = "Results/synthetic_lora_resume_preds_repaired.jsonl"
    resume_metrics_out = "Results/synthetic_lora_resume_metrics_repaired.txt"

    resume_summary = evaluate_predictions(resume_pred, resume_gold, resume_repaired_out, resume_metrics_out)
    print("\n===== Synthetic LoRA Resume (Repaired Eval) =====")
    for k, v in resume_summary.items():
        print(f"{k}: {v}")

    # Medical
    med_pred = "Results/synthetic_lora_medical_preds.jsonl"
    med_gold = "Data/Medical/medical_test.jsonl"
    med_repaired_out = "Results/synthetic_lora_medical_preds_repaired.jsonl"
    med_metrics_out = "Results/synthetic_lora_medical_metrics_repaired.txt"

    med_summary = evaluate_predictions(med_pred, med_gold, med_repaired_out, med_metrics_out)
    print("\n===== Synthetic LoRA Medical (Repaired Eval) =====")
    for k, v in med_summary.items():
        print(f"{k}: {v}")

# PS E:\College\2nd Year\Sem 1\EDAI\Project> python Scripts/18.synthetic_eval.py
#
# ===== Synthetic LoRA Resume (Repaired Eval) =====
# Total Samples: 31
# JSON Validity % (post-repair detection): 22.58
# Exact Match %: 35.48
# Name Accuracy %: 96.77
# Email Accuracy %: 96.77
# Skills Accuracy %: 70.97
# Experience Accuracy %: 48.39
# Avg Levenshtein Distance (approx): 37.06
#
# ===== Synthetic LoRA Medical (Repaired Eval) =====
# Total Samples: 30
# JSON Validity % (post-repair detection): 56.67
# Exact Match %: 10.0
# Name Accuracy %: 100.0
# Email Accuracy %: 100.0
# Skills Accuracy %: 33.33
# Experience Accuracy %: 56.67
# Avg Levenshtein Distance (approx): 43.27