"""
eval_curriculum.py

Evaluation script for structured-output predictions (JSON) produced by any method.

- Designed to match your LoRA evaluation format:
    Exact Match Accuracy
    Name Accuracy
    Email Accuracy
    Skills Accuracy
    Experience Accuracy
    Average Levenshtein Distance

- Input: predictions JSONL where each line is:
    {"input":..., "ground_truth": {...}, "prediction": "..."}

- The script attempts conservative JSON repair on the prediction (strip prefix text before first '{', fix quotes/trailing commas) and then parses.
- Field metrics are computed as percentage of examples where the field is considered correct. For 'skills' and 'experience' we use set/structure-aware comparisons.

Usage example:
python Scripts/eval_curriculum.py \
  --preds Results/curriculum_resume_prediction.jsonl \
  --out Results/curriculum_resume_metrics.json

"""

import argparse
import json
import re
from collections import Counter
from math import inf

# -----------------------------
# Utility: Read JSONL
# -----------------------------
def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    # skip malformed lines
                    continue

# -----------------------------
# JSON repair helpers
# -----------------------------

def strip_before_brace(s: str) -> str:
    if not isinstance(s, str):
        return ''
    i = s.find('{')
    if i >= 0:
        return s[i:]
    return s


def simple_repair(s: str) -> str:
    # Conservative repairs: replace single quotes with double, remove trailing commas before } or ],
    # collapse unicode problematic chars, and ensure balanced braces if possible.
    s = s.strip()
    s = strip_before_brace(s)
    # Replace smart quotes
    s = s.replace("‘", "'").replace("’", "'").replace('“', '"').replace('”', '"')
    # Replace lone single quotes around keys/strings to double quotes (only when safe)
    # best-effort: only replace patterns like 'key': or : 'value'
    s = re.sub(r"'(\w+)':", r'"\1":', s)
    s = re.sub(r":\s*'([^']*)'([,}\]])", r': "\1"\2', s)
    # remove trailing commas before closing brace/bracket
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # remove weird control characters
    s = ''.join(ch for ch in s if ord(ch) >= 32)
    return s


def parse_prediction_to_json(pred_text: str):
    """
    Safely extract a JSON object from a prediction string.
    Avoids recursive regex (?R) which Python re does not support.
    Uses a stack-based brace matcher instead.
    """

    if not isinstance(pred_text, str):
        return None

    # Step 1: Strip everything before first '{'
    s = strip_before_brace(pred_text)
    if not s:
        return None

    # Step 2: Try direct JSON
    try:
        return json.loads(s)
    except:
        pass

    # Step 3: Attempt simple repair
    repaired = simple_repair(s)
    try:
        return json.loads(repaired)
    except:
        pass

    # Step 4: Stack-based extraction of first valid {...} block
    start = s.find('{')
    if start == -1:
        return None

    stack = 0
    for i in range(start, len(s)):
        if s[i] == '{':
            stack += 1
        elif s[i] == '}':
            stack -= 1
            if stack == 0:
                candidate = s[start:i+1]
                try:
                    return json.loads(candidate)
                except:
                    break  # failed, exit loop

    # Nothing worked
    return None

# -----------------------------
# Levenshtein distance (DP)
# -----------------------------

def levenshtein(a: str, b: str) -> int:
    if a is None: a = ''
    if b is None: b = ''
    a = str(a)
    b = str(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # Use optimized 2-row DP
    prev = list(range(m+1))
    cur = [0]*(m+1)
    for i in range(1, n+1):
        cur[0] = i
        ai = a[i-1]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev, cur = cur, prev
    return prev[m]

# -----------------------------
# Field comparators
# -----------------------------

def normalize_name(s: str):
    if s is None: return ''
    return re.sub(r"\s+", ' ', str(s).strip().lower())


def normalize_email(s: str):
    if s is None: return ''
    return str(s).strip().lower()


def normalize_skills(sk):
    # accept list of strings; normalize to lowercase stripped tokens set
    if not sk:
        return set()
    if isinstance(sk, str):
        sk = [sk]
    res = set()
    for x in sk:
        try:
            x = str(x).strip().lower()
            x = re.sub(r"\s+", ' ', x)
            if x:
                res.add(x)
        except Exception:
            continue
    return res


def normalize_experience(exp):
    # experience is expected as list of dicts with company, role, years
    if not exp:
        return []
    out = []
    for item in exp:
        if not isinstance(item, dict):
            continue
        company = str(item.get('company','')).strip().lower()
        role = str(item.get('role','')).strip().lower()
        years = item.get('years', None)
        try:
            years = int(years) if years is not None else None
        except Exception:
            try:
                years = int(float(years))
            except Exception:
                years = None
        out.append({'company': company, 'role': role, 'years': years})
    return out

# -----------------------------
# Main evaluation logic
# -----------------------------

def evaluate(preds_path):
    rows = list(read_jsonl(preds_path))
    total = len(rows)
    if total == 0:
        raise ValueError('No predictions found in file')

    counts = Counter()
    lev_sum = 0
    exact_match_count = 0

    # per-field counts
    name_ok = 0
    email_ok = 0
    skills_ok = 0
    exp_ok = 0

    for r in rows:
        gt = r.get('ground_truth') or r.get('output') or r.get('json')
        pred_text = r.get('prediction') or ''

        # compute levenshtein on raw JSON string (best-effort)
        try:
            gt_str = json.dumps(gt, ensure_ascii=False, sort_keys=True)
        except Exception:
            gt_str = str(gt)

        lev = levenshtein(pred_text, gt_str)
        lev_sum += lev

        # try to parse prediction to JSON
        pred_json = parse_prediction_to_json(pred_text)

        # Exact match: parsed pred json equals gt when both exist
        if pred_json is not None and isinstance(gt, dict):
            # normalize keys order and compare
            try:
                if json.dumps(pred_json, ensure_ascii=False, sort_keys=True) == json.dumps(gt, ensure_ascii=False, sort_keys=True):
                    exact_match_count += 1
            except Exception:
                pass

        # field-wise
        # name
        gt_name = normalize_name(gt.get('name') if isinstance(gt, dict) else None)
        pred_name = normalize_name(pred_json.get('name') if isinstance(pred_json, dict) else None)
        if gt_name and pred_name and gt_name == pred_name:
            name_ok += 1

        # email
        gt_email = normalize_email(gt.get('email') if isinstance(gt, dict) else None)
        pred_email = normalize_email(pred_json.get('email') if isinstance(pred_json, dict) else None)
        if gt_email and pred_email and gt_email == pred_email:
            email_ok += 1

        # skills: treat as set equality or high-overlap
        gt_sk = normalize_skills(gt.get('skills') if isinstance(gt, dict) else None)
        pred_sk = normalize_skills(pred_json.get('skills') if isinstance(pred_json, dict) else None)
        if gt_sk:
            # require at least 0.6 overlap fraction (or exact set equality)
            inter = gt_sk.intersection(pred_sk)
            frac = len(inter) / (len(gt_sk) + 1e-9)
            if frac >= 0.6:
                skills_ok += 1

        # experience: compare list of companies & roles (basic)
        gt_exp = normalize_experience(gt.get('experience') if isinstance(gt, dict) else None)
        pred_exp = normalize_experience(pred_json.get('experience') if isinstance(pred_json, dict) else None)
        if gt_exp:
            # require at least one matching experience entry by company or role
            match = False
            for gentry in gt_exp:
                for pentry in pred_exp:
                    if gentry['company'] and pentry['company'] and gentry['company'] == pentry['company']:
                        match = True
                        break
                    if gentry['role'] and pentry['role'] and gentry['role'] == pentry['role']:
                        match = True
                        break
                if match:
                    break
            if match:
                exp_ok += 1

    metrics = {
        'total': total,
        'exact_match': round(100.0 * exact_match_count / total, 2),
        'name_accuracy': round(100.0 * name_ok / total, 2),
        'email_accuracy': round(100.0 * email_ok / total, 2),
        'skills_accuracy': round(100.0 * skills_ok / total, 2),
        'experience_accuracy': round(100.0 * exp_ok / total, 2),
        'avg_levenshtein': round(lev_sum / total, 2)
    }
    return metrics

# -----------------------------
# CLI
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Results/curriculum_resume_prediction.jsonl", help='Path to predictions JSONL')
    parser.add_argument('--out', type=str, required=False, help='Optional path to save metrics JSON')
    args = parser.parse_args()

    metrics = evaluate(args.preds)
    print("\n=== Curriculum EVALUATION METRICS (with JSON Repair) ===")
    print(f"Total examples (with valid preds): {metrics['total']}")
    print(f"Exact Match Accuracy: {metrics['exact_match']}%")
    print(f"Name Accuracy: {metrics['name_accuracy']}%")
    print(f"Email Accuracy: {metrics['email_accuracy']}%")
    print(f"Skills Accuracy: {metrics['skills_accuracy']}%")
    print(f"Experience Accuracy: {metrics['experience_accuracy']}%")
    print(f"Average Levenshtein Distance: {metrics['avg_levenshtein']}\n")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {args.out}")

# === Curriculum EVALUATION METRICS (with JSON Repair) ===
# Total examples (with valid preds): 31
# Exact Match Accuracy: 9.68%
# Name Accuracy: 93.55%
# Email Accuracy: 96.77%
# Skills Accuracy: 54.84%
# Experience Accuracy: 83.87%
# Average Levenshtein Distance: 788.87
