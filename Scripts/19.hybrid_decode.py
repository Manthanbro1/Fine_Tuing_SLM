# hybrid_rerank.py
"""
Hybrid Rerank (ERR) - Ensemble + Repair + Rerank
Fast, robust, schema-aware reranking for Resume/Medical datasets.

Usage:
    python Scripts/hybrid_rerank.py --base_model HuggingFaceTB/SmolLM2-360M-Instruct \
        --model_dir Models/Curriculum_LoRA_Resume/stage3 \
        --test_file Data/Resume/resume_test.jsonl \
        --out_file Results/hybrid_rerank_resume.jsonl

Important params:
 - --num_beams (default 4)
 - --num_return_sequences (default 4)
 - --max_new_tokens (default 64)
"""

import argparse, json, re
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------------
# Repair + normalization utils
# ---------------------------
def extract_json_substring(text):
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        return m.group(0)
    return None

def quick_repair_json(s):
    if s is None: return None
    t = s.strip()
    t = t.replace(',}', '}').replace(',]', ']')
    # close braces/brackets if needed
    if t.count('{') > t.count('}'):
        t += '}' * (t.count('{') - t.count('}'))
    if t.count('[') > t.count(']'):
        t += ']' * (t.count('[') - t.count(']'))
    # try load
    try:
        return json.loads(t)
    except:
        # try single->double quotes
        t2 = t.replace("'", '"')
        try:
            return json.loads(t2)
        except:
            return None

def normalize_schema(obj):
    # ensure final schema
    if not isinstance(obj, dict):
        obj = {}
    name = obj.get("name", "") or ""
    email = obj.get("email", "") or ""
    skills = obj.get("skills", []) or []
    if isinstance(skills, str):
        skills = [x.strip() for x in re.split(r',| and ', skills) if x.strip()]
    if not isinstance(skills, list):
        skills = []
    exp = obj.get("experience", []) or []
    if isinstance(exp, dict):
        exp = [exp]
    final_exp = []
    for e in exp:
        if isinstance(e, dict):
            company = e.get("company", "") or ""
            role = e.get("role", "") or e.get("position", "") or ""
            years = e.get("years", 0) or 0
            try:
                years = int(float(years))
            except:
                years = 0
            final_exp.append({"company": company, "role": role, "years": years})
    if not final_exp:
        final_exp = [{"company": "", "role": "", "years": 0}]
    return {"name": name, "email": email, "skills": list(dict.fromkeys(skills)), "experience": final_exp}

# ---------------------------
# Lightweight input signals
# ---------------------------
EMAIL_RE = re.compile(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})')
NAME_RE = re.compile(r"(?:my name is|I'm|I am|I’m)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", re.I)
SKILLS_SIGNAL_RE = re.compile(r"skills?(?: include|:)?\s*([A-Za-z0-9, &/+-]+)", re.I)
YEARS_RE = re.compile(r"(\d+)\s+years?")

def extract_signals_from_input(text):
    email = EMAIL_RE.search(text)
    email = email.group(1).lower() if email else ""
    name = NAME_RE.search(text)
    name = name.group(1).strip() if name else ""
    skills_raw = SKILLS_SIGNAL_RE.search(text)
    if skills_raw:
        parts = [p.strip() for p in re.split(r',| and ', skills_raw.group(1)) if p.strip()]
    else:
        parts = []
    years_m = YEARS_RE.search(text)
    years = int(years_m.group(1)) if years_m else None
    # company heuristics
    m = re.search(r'at\s+([A-Z][A-Za-z0-9 &.\-]+)', text)
    company = m.group(1).strip() if m else ""
    return {"name": name, "email": email, "skills": parts, "years": years, "company": company}

# ---------------------------
# Scoring function
# ---------------------------
def jaccard(a,b):
    if not a or not b: return 0.0
    A = set([x.lower() for x in a])
    B = set([x.lower() for x in b])
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def score_candidate(repaired, signals, seq_logprob=None):
    """
    Heuristic scoring:
     + large for json_valid
     + +40 name exact match (fuzzy lower)
     + +40 email exact
     + +30 skill jaccard *100
     + +30 experience/company/years match
     + add seq_logprob (scaled) if present
    """
    score = 0.0
    if repaired is None:
        return -1e6
    # base: valid json
    score += 100.0

    # name
    pred_name = (repaired.get("name") or "").strip().lower()
    sig_name = (signals.get("name") or "").strip().lower()
    if sig_name and pred_name and sig_name in pred_name:
        score += 40.0
    elif sig_name and pred_name:
        # partial fuzzy: token overlap
        if set(sig_name.split()) & set(pred_name.split()):
            score += 15.0

    # email
    pred_email = (repaired.get("email") or "").strip().lower()
    if signals.get("email") and pred_email == signals.get("email"):
        score += 40.0

    # skills overlap
    s_j = jaccard(repaired.get("skills", []), signals.get("skills", []))
    score += s_j * 30.0

    # experience/company/years
    pred_exp = repaired.get("experience", [])
    if pred_exp:
        p0 = pred_exp[0]
        company_ok = False
        years_ok = False
        if signals.get("company"):
            if signals.get("company").lower() in (p0.get("company") or "").lower():
                company_ok = True
                score += 20.0
        if signals.get("years") is not None:
            if int(p0.get("years",0)) == signals.get("years"):
                years_ok = True
                score += 20.0

    # seq logprobs (optional)
    if seq_logprob is not None:
        score += float(seq_logprob) * 1.0  # keep scaling light

    return score

# ---------------------------
# Generation (beam ensemble)
# ---------------------------
def generate_candidates(model, tokenizer, prompt, num_beams, num_return_sequences, max_new_tokens, device):
    """
    Uses generate with beams and returns decoded candidates + optional sequence scores.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
    # Use deterministic beam search (fast)
    gen_out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        do_sample=False,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True
    )
    sequences = gen_out.sequences  # tensor (num_return_sequences)
    scores = None
    if hasattr(gen_out, "sequences_scores"):
        try:
            scores = gen_out.sequences_scores.cpu().tolist()
        except:
            scores = None
    decoded = [tokenizer.decode(s, skip_special_tokens=True) for s in sequences]
    return list(zip(decoded, scores if scores else [None]*len(decoded)))

# ---------------------------
# Runner
# ---------------------------
def main(args):
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, device_map={"": "cpu"}, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, args.model_dir, device_map={"": "cpu"})
    model.eval()

    tests = [json.loads(l) for l in open(args.test_file, "r", encoding="utf8")]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    outp = Path(args.output_dir) / args.out_file

    with open(outp, "w", encoding="utf8") as fout:
        for item in tqdm(tests, desc="ERR decoding"):
            inp = item.get("input")
            signals = extract_signals_from_input(inp)
            # prompt: stronger forcing
            prompt = f"""
                        Extract structured information from the resume below into JSON with keys:
                        ["name", "email", "skills", "experience"]

                        Resume:

                        {inp}

                        Output JSON:
                        """
            # generate small set of candidates
            cand_list = generate_candidates(model, tokenizer, prompt,
                                            num_beams=args.num_beams,
                                            num_return_sequences=args.num_return_sequences,
                                            max_new_tokens=args.max_new_tokens,
                                            device=device)
            # repair + normalize + score
            best_score = -1e9
            best_repaired = None
            for decoded, seq_score in cand_list:
                js = extract_json_substring(decoded)
                repaired = quick_repair_json(js)
                final = normalize_schema(repaired)
                sc = score_candidate(final, signals, seq_score)
                if sc > best_score:
                    best_score = sc
                    best_repaired = final
            # As a safety, if best_repaired is None, fall back to semantic recovery using signals:
            if best_repaired is None:
                best_repaired = {
                    "name": signals.get("name",""),
                    "email": signals.get("email",""),
                    "skills": signals.get("skills",[]),
                    "experience":[{"company":signals.get("company",""), "role":"", "years": signals.get("years") or 0}]
                }
            fout.write(json.dumps({"input": inp, "raw_prediction": best_repaired}, ensure_ascii=False) + "\n")

    print("Saved:", outp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--model_dir", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Models/Curriculum_LoRA_Medical/stage3",
                        help="Directory of Curriculum LoRA adapter (Peft saved adapter path).")
    parser.add_argument("--test_file", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_test.jsonl")
    parser.add_argument("--output_dir", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Results")
    parser.add_argument("--out-file", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Results/hybrid_Pred_Medical_Final.jsonl")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--num_return_sequences", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    main(args)
