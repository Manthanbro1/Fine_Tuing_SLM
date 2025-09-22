import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =============================
#  Config
# =============================
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"   # or whatever base model you used
LORA_PATH = "E:/College/2nd Year/Sem 1/EDAI/Project/Models/LoRA_resume"              # path where LoRA adapter was saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
#  Load Model + LoRA
# =============================
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.to(DEVICE)
model.eval()

# =============================
#  Helper Function
# =============================
def generate_resume_json(text, max_new_tokens=300):
    """
    Generate structured JSON output from resume text.
    """
    prompt = f"""
Extract structured information from the resume below into JSON with keys:
["name", "email", "skills", "experience"]

Resume:
{text}

Output JSON:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =============================
#  Test Run
# =============================
if __name__ == "__main__":
    sample_resume = """
    John Doe
    Email: john.doe@example.com
    Skills: Python, Machine Learning, Data Analysis
    Experience: Worked at ABC Corp as Data Scientist for 3 years.
    """

    result = generate_resume_json(sample_resume)
    print("\n=== MODEL OUTPUT ===\n")
    print(result)
