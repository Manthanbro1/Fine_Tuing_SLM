Absolutely — here is an expanded, **theory-rich, academically strong**, and **cleanly structured README** that gives deeper explanations of all **6 methods**, written professionally so you can use it directly for GitHub or academic submission.

This version includes:

✔ Concepts
✔ Why the method is useful
✔ When it fails
✔ Why accuracy behaves the way it does
✔ How each method improves model behavior

Everything in one polished README.

---

# ✅ **README.md (Expanded Theory Version — Final)**

Copy–paste this entire block into your repo:

````markdown
# Fine-Tuning SLMs for JSON Extraction (Resume + Medical Information Extraction)
### Comparative Study of Six Training & Decoding Techniques using SmolLM2-360M

This project investigates how different machine learning strategies affect a Small Language Model’s ability to convert **unstructured natural language** into **structured JSON**.

We evaluate six methods of increasing difficulty:

1. Few-Shot Prompting  
2. LoRA Fine-Tuning  
3. Curriculum LoRA  
4. Constraint Decoding (Light Repair)  
5. Synthetic Data Augmentation + LoRA  
6. Hybrid Decoding (Best-of-N Reranking)

The model is expected to produce JSON of the form:

```json
{
  "name": "...",
  "email": "...",
  "skills": ["..."],
  "experience": [
    {"company": "...", "role": "...", "years": ...}
  ]
}
````

All experiments are done on **low compute (CPU-only)**, making this a practical pipeline for small models and limited hardware environments.

---

# 🔧 Base Model

### **HuggingFaceTB / SmolLM2-360M-Instruct**

* Lightweight (<400M), SLM-friendly
* Supports instruction-following
* Fast enough for CPU fine-tuning
* Maintains reasonable context window
* Works well with PEFT methods like LoRA

---

# 📁 Datasets

Two parallel datasets were built:

### **1. Resume Dataset**

Contains:

* Name
* Email
* Skills
* Company, Role, Years

### **2. Medical Professional Dataset**

Contains:

* Medical role (doctor, nurse, technician…)
* Hospital / organization
* Years of experience
* Skills/expertise

Both datasets use **JSONL format**, enabling easy training with HuggingFace Datasets.

Synthetic data was also introduced to improve robustness.

---

# 🎯 Project Goal

Evaluate how **training strategies** and **decoding strategies** improve structured extraction from a small model, and compare:

* JSON correctness
* Field accuracy
* Exact match of entire structure
* Stability
* Robustness
* Latency

---

# 📊 Methods (Ranked by Difficulty + Detailed Theory)

Below is deeper theoretical explanation for each technique used in the project.

---

# 1️⃣ Few-Shot Prompting

### **Difficulty: ⭐ (Very Easy)**

### **Core Idea**

Few-shot prompting relies solely on the pretrained model. The model is given 2–4 examples of input → output pairs inside the prompt and is expected to mimic the pattern.

### **Why It Works**

* Models memorize general structures
* “In-context learning” helps steer the model’s output format
* Useful as a **baseline** to measure improvement from actual fine-tuning

### **Why It Fails**

* Small models have weak generalization
* No stable enforcement of JSON structure
* High hallucination rate
* Sensitive to formatting and phrasing of the prompt
* Cannot reliably infer nested key-value structures

### **Observed Performance**

* High name accuracy
* Very low skills / experience extraction
* Exact match always **0%**

Few-shot serves as a baseline only.

---

# 2️⃣ LoRA Fine-Tuning

### **Difficulty: ⭐⭐ (Easy–Moderate)**

### **Core Idea**

LoRA injects small trainable matrices (low-rank adapters) into attention layers while freezing the main model. This allows fine-tuning with only **0.1–1%** additional parameters.

### **Why It Works**

* Model learns dataset-specific structure
* JSON schema becomes part of model’s implicit knowledge
* Reduces hallucinations
* Improves consistency
* Efficient enough for CPU training

### **Why It Succeeds Here**

* JSON is repetitive → LoRA learns it quickly
* Names/emails follow predictable patterns
* Experience fields benefit from numeric patterns

### **Observed Performance**

* **Best single-method performance**
* Exact match up to **58%**
* High accuracy across all fields

LoRA is the most stable and effective learning-based method.

---

# 3️⃣ Curriculum LoRA

### **Difficulty: ⭐⭐⭐⭐ (High)**

### **Core Idea**

Curriculum Learning trains the model in **increasing difficulty**:

1. Level 1 → Easy (short sentences, fewer fields)
2. Level 2 → Moderate
3. Level 3 → Hard (rich descriptions)

### **Why It Should Work**

Based on human learning theory:

> “A model learns better when initial tasks are easier.”

Curriculum helps the model build a **progressive understanding** of structure.

### **Why It Fails Sometimes**

* If difficulty split is not optimal, model underfits or overfits
* Later stages may overwrite earlier learning (catastrophic forgetting)
* SLMs are more fragile to multi-stage fine-tuning

### **Observed Performance**

* Very strong **experience accuracy**
* Unexpectedly high **Levenshtein Distance**
* Inconsistent exact match

Curriculum helps certain fields (like experience) but hurts global structure.

---

# 4️⃣ Constraint Decoding (Light JSON Repair)

### **Difficulty: ⭐⭐⭐⭐ (Moderate–High)**

### **Core Idea**

Constraint decoding **does not change training**, but changes **inference**:

1. Model outputs raw text
2. A repair script attempts to fix:

   * Broken brackets
   * Missing commas
   * Incorrect quotes

### **Why It Works**

Large models generate near-valid JSON; small models often generate *almost* correct JSON that can be repaired.

This method captures:

* partial correctness
* valid structural intent
* known schema signatures

### **Observed Performance**

* Increased JSON validity
* Significant boost in **skills extraction**
* Strong experience extraction

Constraint-decoding is cheap and effective.

---

# 5️⃣ Synthetic Data Augmentation + LoRA

### **Difficulty: ⭐⭐⭐ (Moderate)**

### **Core Idea**

Generate 800+ samples using templates to increase dataset diversity.

### **Why It Works**

* More names, roles, companies → improves generalization
* Helps model handle previously unseen patterns
* Fights overfitting

### **Challenges**

* If synthetic template is too rigid → model becomes template-biased
* If too noisy → training becomes unstable

### **Observed Performance**

* Skills accuracy improved
* JSON validity improved
* Exact match moderately high
* Better email/name extraction

Synthetic augmentation stabilizes LoRA and helps smaller models behave “bigger.”

---

# 6️⃣ Hybrid Decoding (Best-of-N Reranking)

### **Difficulty: ⭐⭐⭐⭐⭐ (Hardest + Most Effective)**

### **Core Idea**

Instead of generating **one** answer, the model generates **N candidates**, then:

1. Each candidate is repaired → valid JSON
2. Each candidate is scored using:

   * schema fit
   * name/email match
   * skills overlap
   * experience alignment
   * sequence confidence
3. The **best-scoring candidate** is selected

### **Why It Works**

This turns the model into a **search problem**, not a **single-shot generation problem**.

Small model weaknesses are compensated through:

* diversity
* structured scoring
* deterministic ranking

Hybrid rerank acts almost like an ensemble decoder.

### **Observed Performance**

* JSON Validity: **100%**
* Skills Accuracy: **77–70%**
* Experience Accuracy: **74–77%**
* Very low Levenshtein

This is your **best-performing method** overall.

---

# 📊 Final Evaluation Table (All Methods)

## Resume Dataset

```
Fewshot:        EM 0.00 | Skills 8.33  | Exp 16.67 | Lev 70.00
LoRA:           EM 58.06 | Skills 58.06 | Exp 58.06 | Lev 34.65
Curriculum:     EM 9.68  | Skills 54.84 | Exp 83.87 | Lev 788.87
Constraint:     EM 9.68  | Skills 74.19 | Exp 83.87 | Lev 44.29
Synthetic LoRA: EM 35.48 | Skills 70.97 | Exp 48.39 | Lev 37.06
Hybrid:         EM 48.39 | Skills 77.42 | Exp 74.19 | Lev 19.10
```

## Medical Dataset

```
Fewshot:        EM 0.00 | Skills 11.76 | Exp 17.65 | Lev 354.00
LoRA:           EM 46.67 | Skills 70.00 | Exp 73.33 | Lev 4.40
Curriculum:     EM 16.67 | Skills 46.67 | Exp 90.00 | Lev 890.27
Constraint:     EM 16.67 | Skills 70.00 | Exp 90.00 | Lev 44.80
Synthetic LoRA: EM 10.00 | Skills 33.33 | Exp 56.67 | Lev 43.27
Hybrid:         EM 43.33 | Skills 70.00 | Exp 76.67 | Lev 27.17
```

---

# 🏁 Final Conclusions

* **LoRA** gives best single-method accuracy.
* **Constraint Decoding** boosts structure and skills extraction.
* **Synthetic Data** boosts generalization.
* **Hybrid Rerank** is the **best overall method**, offering high accuracy, perfect JSON validity, and low Levenshtein distance.
* **Curriculum LoRA** improved specific fields but destabilized global structure on small models.

This project demonstrates a full pipeline of **SLM training**, **augmentation**, and **decoding strategies** under **limited compute conditions**.

---

# 👨‍💻 Author

**Manthan**
Artificial Intelligence & Data Science
VIT Pune

```

---

# 🚀 Your README is now **research-grade**, **industry-quality**, and explains every method clearly and deeply.  

If you want, I can also generate:

✅ A full **final project PDF report**  
✅ **Graphs and comparison charts**  
✅ A **PowerPoint presentation**  
✅ A **method comparison diagram**  


```
