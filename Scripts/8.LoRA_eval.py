import json
from Levenshtein import distance as levenshtein_distance

# ------------------ CONFIG ------------------
GOLD_FILE = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_LoRA_test.jsonl"
PRED_FILE = "E:/College/2nd Year/Sem 1/EDAI/Project/Results/lora_medical_predictions.jsonl"
# -------------------------------------------

def normalize_value(val):
    """Convert floats that are integers to int, for consistent comparison."""
    if isinstance(val, float) and val.is_integer():
        return int(val)
    elif isinstance(val, list):
        return [normalize_value(v) for v in val]
    elif isinstance(val, dict):
        return {k: normalize_value(v) for k, v in val.items()}
    return val

def exact_match(gold, pred):
    """Check if two JSON objects are exactly the same after normalization."""
    return normalize_value(gold) == normalize_value(pred)

def compute_metrics(gold_data, pred_data):
    total = 0
    exact_matches = 0
    name_acc = 0
    email_acc = 0
    skills_acc = 0
    experience_acc = 0
    lev_distances = []

    for g_obj, p_obj in zip(gold_data, pred_data):
        gold = g_obj['output']
        pred = p_obj.get('prediction') or p_obj.get('target') or {}

        gold_norm = normalize_value(gold)
        pred_norm = normalize_value(pred)

        total += 1
        lev_distances.append(levenshtein_distance(json.dumps(gold_norm), json.dumps(pred_norm)))

        if exact_match(gold_norm, pred_norm):
            exact_matches += 1

        if gold_norm.get("name") == pred_norm.get("name"):
            name_acc += 1
        if gold_norm.get("email") == pred_norm.get("email"):
            email_acc += 1
        if set(gold_norm.get("skills", [])) == set(pred_norm.get("skills", [])):
            skills_acc += 1

        # Experience comparison
        gold_exp = gold_norm.get("experience", [])
        pred_exp = pred_norm.get("experience", [])
        if len(gold_exp) == len(pred_exp):
            exp_match = True
            for g_e, p_e in zip(gold_exp, pred_exp):
                if normalize_value(g_e) != normalize_value(p_e):
                    exp_match = False
                    break
            if exp_match:
                experience_acc += 1

    return {
        "total": total,
        "exact_match": round(100 * exact_matches / total, 2),
        "name_accuracy": round(100 * name_acc / total, 2),
        "email_accuracy": round(100 * email_acc / total, 2),
        "skills_accuracy": round(100 * skills_acc / total, 2),
        "experience_accuracy": round(100 * experience_acc / total, 2),
        "avg_levenshtein": round(sum(lev_distances) / total, 2)
    }

# ------------------ MAIN ------------------
with open(GOLD_FILE, 'r', encoding='utf-8') as f:
    gold_data = [json.loads(line) for line in f if line.strip()]

with open(PRED_FILE, 'r', encoding='utf-8') as f:
    pred_data = [json.loads(line) for line in f if line.strip()]

metrics = compute_metrics(gold_data, pred_data)

print("\n=== LORA EVALUATION METRICS (with JSON Repair) ===")
print(f"Total examples (with valid preds): {metrics['total']}")
print(f"Exact Match Accuracy: {metrics['exact_match']}%")
print(f"Name Accuracy: {metrics['name_accuracy']}%")
print(f"Email Accuracy: {metrics['email_accuracy']}%")
print(f"Skills Accuracy: {metrics['skills_accuracy']}%")
print(f"Experience Accuracy: {metrics['experience_accuracy']}%")
print(f"Average Levenshtein Distance: {metrics['avg_levenshtein']}\n")
