"""
test_synthetic_lora.py

Runs inference using a base SLM + LoRA adapter and saves predictions.

Usage:
    python test_synthetic_lora.py --model_dir Models/Synthetic_LoRA_Resume \
                                 --test_file Data/Resume/resume_test.jsonl \
                                 --out_file Results/synthetic_lora_resume_preds.jsonl
"""
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, PeftModel

# -----------------------
# Helpers
# -----------------------
JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)  # greedy capture of the first {...} block

def extract_json_from_text(text: str):
    """
    Try to extract a JSON-like substring from the generated text.
    Returns the substring or None.
    """
    # First try the simple regex
    m = JSON_RE.search(text)
    if m:
        candidate = m.group(1)
        # Heuristic fixes: sometimes generators omit trailing brace or add text after JSON;
        # We'll try to fix a few common issues (closing braces).
        # Quick attempt to balance braces if they are unmatched.
        open_count = candidate.count("{")
        close_count = candidate.count("}")
        if open_count > close_count:
            candidate = candidate + "}" * (open_count - close_count)
        return candidate.strip()
    return None

def safe_json_load(s):
    try:
        return json.loads(s)
    except Exception:
        return None

# -----------------------
# Main
# -----------------------
def main(args):
    device = torch.device("cpu")

    # Load tokenizer & base model (cpu)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # Load LoRA adapter (if present)
    if args.model_dir:
        # If adapter saved via save_pretrained, load with PeftModel wrapper
        try:
            model = PeftModel.from_pretrained(model, args.model_dir, device_map={"": "cpu"})
            print("Loaded LoRA adapter from", args.model_dir)
        except Exception as e:
            print("Warning: failed to load LoRA adapter with PeftModel:", e)
            print("Trying to load model_dir as full model...")
            model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map={"": "cpu"})

    model.eval()

    # Load test set
    test_items = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            test_items.append(json.loads(line))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / Path(args.out_file).name

    # Generation settings (deterministic)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,           # greedy/deterministic
        top_p=1.0,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )

    with open(out_path, "w", encoding="utf-8") as out_f:
        for item in tqdm(test_items, desc="Generating"):
            inp = item.get("input") if isinstance(item, dict) else str(item)
            # Create prompt exactly like in training
            prompt = f"""
            Extract structured information from the resume below into JSON with keys:
            ["name", "email", "skills", "experience"]
            
            Resume:
            
            {inp}
            
            Output JSON:
            """

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            # Ensure tensors are on CPU
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)

            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            # The model output contains the whole prompt+generation; extract the generated suffix
            # We remove the prompt prefix to keep only model continuation
            if decoded.startswith(prompt):
                raw_gen = decoded[len(prompt):].strip()
            else:
                # Fallback: attempt to remove prompt lines
                raw_gen = decoded.replace(prompt, "").strip()

            # Try to extract JSON substring
            json_sub = extract_json_from_text(raw_gen)
            parsed = None
            if json_sub:
                parsed = safe_json_load(json_sub)

            record = {
                "input": inp,
                "raw_prediction": raw_gen,
                "json_extract": json_sub,
                "parsed_prediction": parsed
            }
            out_f.write(json.dumps(record) + "\n")

    print(f"\nSaved predictions to: {out_path}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct",
                        help="Base HuggingFace model id (same used for training)")
    parser.add_argument("--model_dir", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Models/Synthetic_LoRA_Medical",
                        help="Directory of LoRA adapter (or model dir if merged). Leave empty to use base only.")
    parser.add_argument("--test_file", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_test.jsonl",
                        help="Test JSONL file with {'input', 'output'} format.")
    parser.add_argument("--output_dir", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Results",
                        help="Directory to write predictions.")
    parser.add_argument("--out_file", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/synthetic_lora_medical_preds.jsonl",
                        help="Output filename (inside output_dir).")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate for the output.")
    args = parser.parse_args()
    main(args)


