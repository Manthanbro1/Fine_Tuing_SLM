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
    you have 3 mins to solve this question make sure to extract the info properly, dont halucinate any info.
Extract structured information from the resume below into JSON with keys:
["name" , "email" , "phone" , "Job" , "address" , "username" , "url" , "hobby" ]
And Check Neatly for email and there can be sometimes null value for some keys if not found in the resume.
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
    My name is Aaliyah Popova, and I am a jeweler with 13 years of experience. I remember a very unique and challenging project I had to work on last year. A customer approached me with a precious family heirloom - a diamond necklace that had been passed down through generations. Unfortunately, the necklace was in poor condition, with several loose diamonds and a broken clasp. The customer wanted me to restore it to its former glory, but it was clear that this would be no ordinary repair. Using my specialized tools and techniques, I began the delicate task of dismantling the necklace. Each diamond was carefully removed from its setting, and the damaged clasp was removed. Once the necklace was completely disassembled, I meticulously cleaned each diamond and inspected it for any damage. Fortunately, the diamonds were all in good condition, with no cracks or chips. The next step was to repair the broken clasp. I carefully soldered the broken pieces back together, ensuring that the clasp was sturdy and secure. Once the clasp was repaired, I began the process of reassembling the necklace. Each diamond was carefully placed back into its setting, and the necklace was polished until it sparkled like new. When I presented the restored necklace to the customer, they were overjoyed. They couldn't believe that I had been able to bring their family heirloom back to life. The necklace looked as beautiful as it had when it was first created, and the customer was thrilled to have it back in their possession. If you have a project that you would like to discuss, please feel free to contact me by phone at (95) 94215-7906 or by email at aaliyah.popova4783@aol.edu. I look forward to hearing from you! P.S.: When I'm not creating beautiful jewelry, I enjoy spending time podcasting. I love sharing my knowledge about jewelry and connecting with other people who are passionate about this art form. I also enjoy spending time with my family and exploring new places. If you would like to learn more about me, please feel free to visit my website at [website address] or visit me at my studio located at 97 Lincoln Street.
    """

    result = generate_resume_json(sample_resume)
    print("\n=== MODEL OUTPUT ===\n")
    print(result)
