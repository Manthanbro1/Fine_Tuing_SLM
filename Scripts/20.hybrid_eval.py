# 21.hybrid_eval_final.py
"""
Robust hybrid evaluation:
 - Never crashes on bad JSON lines
 - Handles dict or string predictions
 - Pretty printed output (Like Synthetic LoRA Eval)
"""

import json
from rapidfuzz.distance import Levenshtein

def safe_load(line):
    try:
        return json.loads(line)
    except:
        return None

def normalize_pred(raw):
    """
    raw can be:
    - dict  -> return as is
    - string -> try json loads
    - None -> return None
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except:
            return None
    return None

def skill_match(p, g):
    if not isinstance(p, list) or not isinstance(g, list):
        return False
    p = [x.lower().strip() for x in p]
    g = [x.lower().strip() for x in g]
    return set(p) == set(g)

def exp_match(p, g):
    if not isinstance(p, list) or not isinstance(g, list):
        return False
    if len(p) != len(g):
        return False
    for a, b in zip(p, g):
        if a.get("company","").lower() != b.get("company","").lower():
            return False
        if a.get("role","").lower() != b.get("role","").lower():
            return False
        try:
            if int(a.get("years",0)) != int(b.get("years",0)):
                return False
        except:
            return False
    return True

def evaluate(pred_file, test_file, title="Hybrid Eval"):
    preds_raw = open(pred_file, "r", encoding="utf8").read().splitlines()
    golds = [safe_load(x) for x in open(test_file, "r", encoding="utf8").read().splitlines()]

    preds = [safe_load(x) for x in preds_raw]

    total = len(preds)
    json_valid = 0
    exact = 0
    name_acc = 0
    email_acc = 0
    skills_acc = 0
    exp_acc = 0
    lev_total = 0

    for p, g in zip(preds, golds):
        if p is None or g is None:
            continue

        raw = p.get("raw_prediction", None)
        pred = normalize_pred(raw)
        gold = g.get("output", None)

        if pred is None or gold is None:
            continue

        # valid JSON
        if isinstance(pred, dict):
            json_valid += 1

        # exact match
        if pred == gold:
            exact += 1

        # name
        if pred.get("name","").lower() == gold.get("name","").lower():
            name_acc += 1

        # email
        if pred.get("email","").lower() == gold.get("email","").lower():
            email_acc += 1

        # skills
        if skill_match(pred.get("skills"), gold.get("skills")):
            skills_acc += 1

        # experience
        if exp_match(pred.get("experience"), gold.get("experience")):
            exp_acc += 1

        # levenshtein
        lev_total += Levenshtein.distance(
            json.dumps(pred, sort_keys=True),
            json.dumps(gold, sort_keys=True)
        )

    # pretty format
    print(f"\n===== {title} =====")
    print(f"Total Samples: {total}")
    print(f"JSON Validity %: {json_valid*100/total:.2f}")
    print(f"Exact Match %: {exact*100/total:.2f}")
    print(f"Name Accuracy %: {name_acc*100/total:.2f}")
    print(f"Email Accuracy %: {email_acc*100/total:.2f}")
    print(f"Skills Accuracy %: {skills_acc*100/total:.2f}")
    print(f"Experience Accuracy %: {exp_acc*100/total:.2f}")
    print(f"Avg Levenshtein Distance: {lev_total/total:.2f}\n")


if __name__ == "__main__":

    resume_pred = "E:/College/2nd Year/Sem 1/EDAI/Project/Results/hybrid_Pred_Resume_4.jsonl"
    resume_test = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Resume/resume_test.jsonl"

    med_pred = "E:/College/2nd Year/Sem 1/EDAI/Project/Results/hybrid_Pred_Medical_Final.jsonl"
    med_test = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_test.jsonl"

    evaluate(resume_pred, resume_test, "Hybrid Resume Evaluation")
    evaluate(med_pred, med_test, "Hybrid Medical Evaluation")
