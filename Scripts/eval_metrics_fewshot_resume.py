import json
import re

RESULTS_FILE = "E:/College/2nd Year/Sem 1/EDAI/Project/Results/fewshot_resume.jsonl"


def try_fix_json(bad_json_str):
    """Attempt to fix common JSON formatting issues"""
    if not bad_json_str or not isinstance(bad_json_str, str):
        return None
    try:
        # First, try direct parse
        return json.loads(bad_json_str)
    except:
        # Replace single quotes with double quotes
        fixed = bad_json_str.replace("'", '"')

        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

        # Try parsing again
        try:
            return json.loads(fixed)
        except:
            return None


def normalize_experience(exp_list):
    normalized = []
    for e in exp_list:
        new_e = {}
        new_e["company"] = str(e.get("company", "")).strip()
        new_e["role"] = str(e.get("role", "")).strip()
        try:
            new_e["years"] = float(e.get("years", 0))
        except:
            new_e["years"] = 0
        normalized.append(new_e)
    return normalized


def evaluate():
    total = 0
    valid_preds = 0
    exact_match = 0
    field_correct = {"name": 0, "email": 0, "skills": 0, "experience": 0}

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            pred = ex.get("predicted")
            tgt = ex.get("target")

            # Fix invalid predictions
            if isinstance(pred, dict) and "error" in pred:
                pred = try_fix_json(pred.get("error", ""))
            elif isinstance(pred, str):
                pred = try_fix_json(pred)

            if not pred or not isinstance(pred, dict):
                continue  # skip if still broken

            total += 1
            valid_preds += 1

            pred_exp = normalize_experience(pred.get("experience", []))
            tgt_exp = normalize_experience(tgt.get("experience", []))

            # Exact match check
            if pred == tgt:
                exact_match += 1

            # Field-wise checks
            if pred.get("name", "").strip().lower() == tgt.get("name", "").strip().lower():
                field_correct["name"] += 1
            if pred.get("email", "").strip().lower() == tgt.get("email", "").strip().lower():
                field_correct["email"] += 1
            if set([s.lower() for s in pred.get("skills", [])]) == set([s.lower() for s in tgt.get("skills", [])]):
                field_correct["skills"] += 1
            if pred_exp == tgt_exp:
                field_correct["experience"] += 1

    print("\n=== FEWSHOT EVALUATION METRICS (with JSON Repair) ===")
    print(f"Total examples (with valid preds): {valid_preds}")
    print(f"Exact Match Accuracy: {exact_match / valid_preds:.2%}" if valid_preds else "Exact Match Accuracy: N/A")
    for field, correct in field_correct.items():
        print(
            f"{field.capitalize()} Accuracy: {correct / valid_preds:.2%}" if valid_preds else f"{field.capitalize()} Accuracy: N/A")


if __name__ == "__main__":
    evaluate()
