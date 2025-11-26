"""
13.constraint_decode_test.py

Light Repair Constraint Decoding (Inference Only)
-------------------------------------------------
This script:
    • Loads the Curriculum LoRA model (SmolLM2 + LoRA adapters)
    • Loads a TEST JSONL file (same as test_curriculum_lora)
    • Generates predictions
    • Applies LIGHT REPAIR to fix malformed JSON
    • Saves both raw + repaired predictions

NO TRAINING, NO NEW ADAPTERS — inference-only.
This implements Method 4 (Constraint Decoding: Light Repair)

OUTPUT FORMAT (JSONL):
{
  "input": ...,
  "ground_truth": {...},
  "raw_prediction": "...",
  "repaired_prediction": {...}   <-- valid JSON after repair
}

USAGE:
python constraint_decode_test.py \
  --adapter_dir "Models/Curriculum_LoRA_Resume/stage3" \
  --test_file "Data/Resume/resume_test.jsonl" \
  --output_file "Results/curriculum_resume_prediction_repaired.jsonl"
"""

import os
import json
import argparse
from tqdm import tqdm
import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ------------------------------------------------------------
# JSONL Utilities
# ------------------------------------------------------------
def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ------------------------------------------------------------
# Light Repair Helper Functions
# ------------------------------------------------------------
def strip_before_brace(s):
    if not isinstance(s, str): return ''
    i = s.find('{')
    return s[i:] if i != -1 else ''

def simple_repair(s):
    """
    Simple structural cleanup:
    - Remove trailing commas
    - Replace 'key' with "key"
    - Replace 'value' with "value"
    - Remove smart quotes
    """
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace('“', '"').replace('”', '"')

    # Replace 'key': with "key":
    s = re.sub(r"'(\w+)':", r'"\1":', s)

    # Replace : 'value' with : "value"
    s = re.sub(r":\s*'([^']*)'([,}\]])", r': "\1"\2', s)

    # Remove trailing commas
    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s

def brace_based_extract(s):
    """Recover first valid {...} block using a stack parser."""
    s = strip_before_brace(s)
    start = s.find('{')
    if start == -1: return None

    depth = 0
    for i in range(start, len(s)):
        if s[i] == '{': depth += 1
        elif s[i] == '}': depth -= 1
        if depth == 0:
            candidate = s[start:i+1]
            try:
                return json.loads(candidate)
            except:
                return None
    return None

def repair_json(pred_text):
    """
    Main logic used in evaluation:
    1. Strip prefix before first '{'
    2. Attempt direct json.loads
    3. Light repair + json.loads
    4. Brace-based extraction
    """
    if not isinstance(pred_text, str):
        return None

    # 1. strip prefix
    s = strip_before_brace(pred_text)

    # 2. direct parse
    try:
        return json.loads(s)
    except:
        pass

    # 3. simple repair
    repaired = simple_repair(s)
    try:
        return json.loads(repaired)
    except:
        pass

    # 4. brace-based fallback
    return brace_based_extract(s)

# ------------------------------------------------------------
# Prompt formatting
# ------------------------------------------------------------
def format_prompt(example):
    inp = example.get("input") or example.get("text") or ""
    return f"""
Extract structured information from the resume below into JSON with keys:
["name", "email", "skills", "experience"]

Resume:

{inp}

Output JSON:
"""

# ------------------------------------------------------------
# Model Inference
# ------------------------------------------------------------
def generate_output(model, tokenizer, prompt, device, max_new_tokens=200):
    enc = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_dir', type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Models/Curriculum_LoRA_Medical/stage3",
                        help="Path to stage3 folder with LoRA adapter_model.safetensors")
    parser.add_argument('--test_file', type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_test.jsonl")
    parser.add_argument('--output_file', type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Results/contraints_decode_medical_prediction.jsonl")
    parser.add_argument('--max_new_tokens', type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

    print("[INFO] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"[INFO] Loading LoRA adapters from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir, adapter_name="default")
    model.to(device)
    model.eval()

    # Load test data
    print(f"[INFO] Loading test data: {args.test_file}")
    test_data = list(read_jsonl(args.test_file))
    print(f"[INFO] Loaded {len(test_data)} samples.")

    results = []

    print("[INFO] Running inference with Light Repair...")
    for ex in tqdm(test_data):
        prompt = format_prompt(ex)
        raw_pred = generate_output(model, tokenizer, prompt, device, args.max_new_tokens)
        repaired = repair_json(raw_pred)

        results.append({
            "input": ex.get("input"),
            "ground_truth": ex.get("output") or ex.get("json"),
            "raw_prediction": raw_pred,
            "repaired_prediction": repaired
        })

    write_jsonl(args.output_file, results)
    print(f"[DONE] Saved repaired predictions to: {args.output_file}")


if __name__ == '__main__':
    main()
