import json
from rapidfuzz.distance import Levenshtein
from pathlib import Path
import os
import argparse


# --- Jaccard Similarity Function (NEW) ---
def jaccard_similarity(predicted_skills, gold_skills):
    """
    Calculates the Jaccard Similarity (Intersection / Union) between two skill lists.
    The comparison is case-insensitive and order-agnostic.
    """
    if not isinstance(predicted_skills, list) or not isinstance(gold_skills, list):
        return 0.0

    pred_set = set(str(s).lower().strip() for s in predicted_skills)
    gold_set = set(str(s).lower().strip() for s in gold_skills)

    if not pred_set and not gold_set:
        return 1.0  # Perfect match if both are empty

    intersection_size = len(pred_set.intersection(gold_set))
    union_size = len(pred_set.union(gold_set))

    # Avoid division by zero, though union_size should be > 0 if one set is not empty
    return intersection_size / union_size if union_size > 0 else 0.0


# --- Helper Functions (From Original Script) ---
def safe_load(line):
    try:
        return json.loads(line)
    except:
        return None


def normalize_pred(raw):
    # ... (Keep original normalize_pred function) ...
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


# NOTE: The original strict skill_match is REMOVED/IGNORED for overlap calculation.

def exp_match(p, g):
    # ... (Keep original strict exp_match function) ...
    if not isinstance(p, list) or not isinstance(g, list):
        return False
    if len(p) != len(g):
        return False
    for a, b in zip(p, g):
        if a.get("company", "").lower() != b.get("company", "").lower():
            return False
        if a.get("role", "").lower() != b.get("role", "").lower():
            return False
        try:
            pred_years = int(float(a.get("years", 0) or 0))
            gold_years = int(float(b.get("years", 0) or 0))
            if pred_years != gold_years:
                return False
        except:
            return False
    return True


# --- MODIFIED EVALUATE FUNCTION ---
def evaluate(pred_file, test_file, title="Hybrid Eval"):
    # ... (Keep file loading logic) ...
    try:
        preds_raw = Path(pred_file).read_text(encoding="utf8").splitlines()
        golds = [safe_load(x) for x in Path(test_file).read_text(encoding="utf8").splitlines()]
    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
        return

    preds = [safe_load(x) for x in preds_raw]
    total = len(preds)

    # --- Counters (UNCHANGED) ---
    json_valid = 0
    exact = 0
    name_acc = 0
    email_acc = 0
    exp_acc = 0
    lev_total = 0

    # --- JACCARD SUMMATION (CHANGED) ---
    skills_jaccard_total = 0.0  # Changed to float for summation

    valid_samples = 0

    for p, g in zip(preds, golds):
        if p is None or g is None: continue

        raw = p.get("raw_prediction", None)
        pred = normalize_pred(raw)
        gold = g.get("output", None)

        if pred is None or gold is None: continue

        valid_samples += 1

        # Valid JSON
        if isinstance(pred, dict): json_valid += 1

        # Exact Match
        if pred == gold: exact += 1

        # Name/Email Accuracies
        if pred.get("name", "").lower() == gold.get("name", "").lower(): name_acc += 1
        if pred.get("email", "").lower() == gold.get("email", "").lower(): email_acc += 1

        # SKILLS: CALCULATE JACCARD SCORE AND SUM IT (CHANGED)
        jaccard_score = jaccard_similarity(pred.get("skills", []), gold.get("skills", []))
        skills_jaccard_total += jaccard_score  # Summing the scores instead of counting a binary match

        # Experience (STRICT MATCH UNCHANGED)
        if exp_match(pred.get("experience"), gold.get("experience")): exp_acc += 1

        # Levenshtein
        try:
            pred_json_str = json.dumps(pred, sort_keys=True)
            gold_json_str = json.dumps(gold, sort_keys=True)
            lev_total += Levenshtein.distance(pred_json_str, gold_json_str)
        except:
            pass

            # --- Pretty Format Output (FINAL OUTPUT LINE CHANGED) ---
    print(f"\n===== {title} =====")
    print(f"Total Samples (Input Lines): {total}")
    print(f"Successfully Compared Samples: {valid_samples}")

    if valid_samples > 0:
        # Calculate Average Jaccard Score
        avg_jaccard_acc = (skills_jaccard_total / valid_samples) * 100

        print(f"JSON Validity %: {json_valid * 100 / total:.2f}")
        print(f"Exact Match %: {exact * 100 / valid_samples:.2f}")
        print(f"Name Accuracy %: {name_acc * 100 / valid_samples:.2f}")
        print(f"Email Accuracy %: {email_acc * 100 / valid_samples:.2f}")
        # NEW LINE FOR AVERAGE JACCARD SCORE
        print(f"Skills Overlap % (Avg Jaccard): {avg_jaccard_acc:.2f}")
        print(f"Experience Accuracy % : {exp_acc * 100 / valid_samples:.2f}")
        print(f"Avg Levenshtein Distance: {lev_total / valid_samples:.2f}\n")
    else:
        print("No valid samples were available for comparison.")


if __name__ == "__main__":
    # Note: I have included argparse to make the script reusable and standard,
    # and replaced the hardcoded paths with the original defaults or examples,
    # but the paths will need to be adjusted to your local file system.

    # Example Paths (Replace with your actual paths):
    DEFAULT_EXAMPLE_PRED = "E:/College/2nd Year/Sem 1/EDAI/Project/Results/hybrid_Pred_Medical_Example.jsonl"
    DEFAULT_EXAMPLE_TEST = "E:/College/2nd Year/Sem 1/EDAI/Project/Example.jsonl"

    parser = argparse.ArgumentParser(description="Robust Hybrid Rerank Evaluation.")
    parser.add_argument("--example_pred", type=str, default=DEFAULT_EXAMPLE_PRED,
                        help="Path to the prediction file for Resume data.")
    parser.add_argument("--example_test", type=str, default=DEFAULT_EXAMPLE_TEST,
                        help="Path to the test/gold standard file for Resume data.")

    # Medical paths are commented out as they were examples in the original script
    parser.add_argument("--res_pred", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Results/hybrid_Pred_Resume_4.jsonl")
    parser.add_argument("--res_test", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Data/Resume/resume_test.jsonl")

    parser.add_argument("--med_pred", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Results/hybrid_pred_medical.jsonl")
    parser.add_argument("--med_test", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_test.jsonl")
    args = parser.parse_args()
    # The script in your prompt only called the example evaluation
    evaluate(args.example_pred, args.example_test, "Hybrid Example Evaluation")
    # The script in your prompt only called the example evaluation
    evaluate(args.res_pred, args.res_test, "Hybrid Resume Evaluation")
    # If you wish to run the medical eval, uncomment and adjust paths:
    evaluate(args.med_pred, args.med_test, "Hybrid Medical Evaluation")