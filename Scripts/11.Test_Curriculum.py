"""
============================================================
test_curriculum_lora.py  (SmolLM2-Compatible Version)
============================================================

Loads:
    • Base model: HuggingFaceTB/SmolLM2-360M-Instruct
    • LoRA adapters: path/to/Curriculum_LoRA_Resume OR Medical
    • Test JSONL file

Generates:
    • Raw predictions ONLY (no evaluation)
    • Output stored as JSONL:
      {"input": ..., "ground_truth": ..., "prediction": ...}

============================================================
USAGE EXAMPLE
------------------------------------------------------------
python test_curriculum_lora.py ^
    --adapter_dir "E:/College/.../Curriculum_LoRA_Resume" ^
    --test_file "E:/College/.../resume_test.jsonl" ^
    --output_file "E:/College/.../Curriculum_Resume_Prediction.jsonl"
------------------------------------------------------------
"""

import os
import json
import argparse
from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ------------------------------------------------------------
# Load JSONL
# ------------------------------------------------------------
def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ------------------------------------------------------------
# Save JSONL
# ------------------------------------------------------------
def write_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Build prompt
# ------------------------------------------------------------
def format_prompt(example):
    inp = example.get("input") or example.get("text") or ""
    return f"""
Extract structured information from the resume below into JSON with keys:
["name", "email", "skills", "experience"]

Resume:

{example['input']}

Output JSON:
"""


# ------------------------------------------------------------
# Generate prediction
# ------------------------------------------------------------
def generate_output(model, tokenizer, prompt, device, max_new_tokens=300):
    enc = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Models/Curriculum_LoRA_Medical/stage3",
                        help="Directory containing LoRA adapter_model.safetensors")
    parser.add_argument("--test_file", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_test.jsonl")
    parser.add_argument("--output_file", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Results/curriculum_medical_prediction.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=300)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------
    # Load BASE MODEL (SmolLM2-360M-Instruct)
    # ------------------------------------------------------------
    BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

    print("[INFO] Loading base model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.float32)

    # ------------------------------------------------------------
    # Load LoRA ADAPTER
    # ------------------------------------------------------------
    print(f"[INFO] Loading LoRA adapters from: {args.adapter_dir}")

    model = PeftModel.from_pretrained(
        model,
        args.adapter_dir,
        adapter_name="default",
        torch_dtype=torch.float32
    )

    model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------
    print(f"[INFO] Loading test data: {args.test_file}")
    test_data = list(read_jsonl(args.test_file))
    print(f"[INFO] Loaded {len(test_data)} test samples.")

    results = []

    # ------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------
    print("[INFO] Generating predictions...")
    for ex in tqdm(test_data):
        prompt = format_prompt(ex)
        pred = generate_output(model, tokenizer, prompt, device,
                               max_new_tokens=args.max_new_tokens)

        results.append({
            "input": ex.get("input"),
            "ground_truth": ex.get("output") or ex.get("json"),
            "prediction": pred
        })

    # ------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------
    write_jsonl(args.output_file, results)
    print(f"[DONE] Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
