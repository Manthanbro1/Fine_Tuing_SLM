# 📄 Resume-to-JSON Extraction  

This project converts **raw resume text** into structured **JSON** using a small instruction-tuned language model.  

---

## ✅ Progress So Far  

1. **Dataset Created**  
   - Raw resumes paired with expected JSON.  
   - Schema includes `name`, `email`, `skills`, `experience`.  

2. **Data Split (`prepare_data.py`)**  
   - Divides dataset into **train**, **validation**, and **test**.  

3. **Few-shot Evaluation (`eval_fewshots_resume.py`)**  
   - Uses `HuggingFaceTB/SmolLM2-360M-Instruct`.  
   - Prompts model with schema + resume text.  
   - Generates structured JSON output.  

4. **Safe JSON Parsing**  
   - Handles invalid JSON gracefully with fallback.  

5. **Result Saving**  
   - Predictions and targets stored in `results/fewshot_results.jsonl`.  

---

## 📂 Structure  

Project/
│── dataset/
<br>
│── Scripts/
<br>
│ ├── prepare_data.py
<br>
│ ├── eval_fewshots_resume.py
<br>
│── results/
<br>
│ ├── fewshot_results.jsonl
<br>
---

## 🚀 Next Steps  

- Add **evaluation metrics** (exact match, field accuracy).  
- Try **fine-tuning** the model with training split. 