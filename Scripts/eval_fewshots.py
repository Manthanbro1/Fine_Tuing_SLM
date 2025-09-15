import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return text.strip()  # fallback

# --------------------------
# Config
# --------------------------
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
FEWSHOT_EXAMPLES = [
    {
        "name": "string",
        "email": "string",
        "skills": ["string"],
        "experience": [
            {"company": "string", "role": "string", "years": "number"}
        ]
    }
]

# --------------------------
# Build Prompt
# --------------------------
def build_prompt(new_input: str):
    prompt = "You are an information extractor. Convert resume text into structured JSON.here is the structure you will be following\n\n"
    prompt += str(FEWSHOT_EXAMPLES)
    prompt +=f"Now i will give you one question below to solve it .Solve that and  give me output in same structure"
    prompt += f"Input: {new_input}\nOutput:"
    return prompt

# --------------------------
# Run Few-shot Inference
# --------------------------
def run_fewshot(test_input):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    prompt = build_prompt(test_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        do_sample=False
    )

    text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean text after "Output:"
    raw_text = text_out.split("Output:")[-1].strip()
    return extract_json(raw_text)


# --------------------------
# Example run
# --------------------------
if __name__ == "__main__":
    import os
    """You Writ medical_test.jsonl and fewshot_medical.jsonl to use medical dataset
    You can change the names to resume_test.jsonl and fewshot_resume.jsonl to use resume dataset"""


    test_path = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/medical_test.jsonl"
    output_path = "E:/College/2nd Year/Sem 1/EDAI/Project/Results/fewshot_medical.jsonl"
    os.makedirs("Results", exist_ok=True)

    # Load test data
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    # Run few-shot on all examples
    results = []
    for example in test_data:
        pred = run_fewshot(example["input"])
        try:
            pred_json = json.loads(pred)
        except json.JSONDecodeError:
            pred_json = {"error": pred, "note": "invalid JSON"}  # fallback

        results.append({
            "input": example["input"],
            "predicted": pred_json,
            "target": example["output"]
        })

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Few-shot results saved to {output_path}")
