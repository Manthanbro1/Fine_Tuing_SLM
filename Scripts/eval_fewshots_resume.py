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
    test_resume = "Aarav Patel here, a Project Manager at QuantumLeap for 8 yrs, proficient in Agile, Scrum, JIRA, and Team Leadership. aarav.p@inbox.com"
    result = run_fewshot(test_resume)
    print("\n=== FEW-SHOT RESULT ===")
    print(result)
