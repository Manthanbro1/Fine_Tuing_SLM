import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from tqdm import tqdm

# === CONFIG ===
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # same base model you used
LORA_PATH = "E:/College/2nd Year/Sem 1/EDAI/Project/Models/LoRA_medical"   # trained LoRA adapter
TEST_FILE = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_LoRA_test.jsonl"
RESULT_FILE = "E:/College/2nd Year/Sem 1/EDAI/Project/Results/lora_medical_predictions.jsonl"

# === Load Model + LoRA ===
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

pipe = pipeline("text-generation",model=model,tokenizer=tokenizer,device_map="auto",max_new_tokens=256)

# === Load Test Data ===
print("Loading test data...")
dataset = load_dataset("json", data_files=TEST_FILE, split="train")

# === Evaluation Loop ===
results = []
for example in tqdm(dataset, desc="Evaluating"):
    prompt = f"""
Extract structured information from the resume below into JSON with keys:
["name", "email", "skills", "experience"]

Resume:

{example['input']}

Output JSON:
"""
    output = pipe(prompt, do_sample=False, temperature=0.0)[0]["generated_text"]

    # try extracting JSON
    try:
        json_str = output.split("Output JSON:")[-1].strip()
        pred_json = json.loads(json_str)
    except Exception:
        pred_json = {}

    results.append({
        "input": example["input"],
        "target": example["output"],
        "prediction": pred_json,
        "raw_output": output,
    })

# === Save Results ===
os.makedirs("Results", exist_ok=True)
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"Saved predictions to {RESULT_FILE}")

