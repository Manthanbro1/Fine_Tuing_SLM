```
# Resume Information Extraction using LoRA Fine-Tuning

## 📄 Project Overview
This project focuses on **extracting structured information from resumes** using **LoRA fine-tuned language models**. The system takes unstructured text resumes and generates structured JSON outputs containing key information such as:

- `name`  
- `email`  
- `skills`  
- `experience`  
- (Medical dataset: additional fields like `specialization`, `degrees`)  

The project is implemented in **Python**, leveraging Hugging Face Transformers, PEFT (Parameter-Efficient Fine-Tuning), and using methods like Prompting(Few_Shots) LoRA (Low-Rank Adaptation).  

The goal is to compare **baseline few-shot prompting** against **LoRA fine-tuning** for structured resume extraction and measure improvements in accuracy and reliability.

---

## ⚙️ Features
- Converts unstructured resumes into **JSON format**.  
- Supports **general resumes** as well as **medical resumes**.  
- Provides **evaluation metrics** including:  
  - Exact Match Accuracy  
  - Name/Email/Skills/Experience Accuracy  
  - Average Levenshtein Distance  
- LoRA fine-tuning allows efficient model adaptation even on **CPU or low-GPU systems**.

---

## 📝 Methodology
1. **Dataset Preparation**
   - Created general resumes and medical resumes.
   - Annotated data in JSONL format for structured extraction.
   - Split data into `train`, `valid`, and `test` sets.

2. **Few-Shot Prompting Baseline**
   - Tested the model with prompt-based few-shot examples.
   - Measured baseline extraction performance.

3. **LoRA Fine-Tuning**
   - Fine-tuned a pre-trained language model (e.g., LLaMA) using PEFT and LoRA on the resume datasets.
   - Used separate adapters for general resumes and medical resumes.
   - Adjusted training hyperparameters and prompts based on dataset fields.

4. **Evaluation and Benchmarking**
   - Generated predictions on test datasets.
   - Evaluated using Exact Match, field-wise accuracy, and Levenshtein distance.
   - Compared performance of few-shot vs LoRA fine-tuned models.

---

## 📂 Project Structure
```

project/<br>
│<br>
├─ Scripts/<br>
│   ├─ train\_LoRA\_resume.py       # LoRA fine-tuning script for general resumes<br>
│   ├─ train\_LoRA\_medical.py      # LoRA fine-tuning script for medical resumes<br>
│   ├─ eval\_LoRA\_resume.py        # Evaluation script for general resumes<br>
│   ├─ eval\_LoRA\_medical.py       # Evaluation script for medical resumes<br>
│   ├─ eval\_LoRA\_metrics.py       # Metrics calculation script<br>
│<br>
├─ Data/<br>
│   ├─ resume\_train.jsonl<br>
│   ├─ resume\_valid.jsonl<br>
│   ├─ resume\_test.jsonl<br>
│   ├─ medical\_train.jsonl<br>
│   ├─ medical\_valid.jsonl<br>
│   └─ medical\_test.jsonl<br>
│<br>
├─ Models/<br>
│   ├─ LoRA\_resume/               # Saved LoRA adapter for general resumes<br>
│   └─ LoRA\_medical/              # Saved LoRA adapter for medical resumes<br>
│<br>
└─ Results/<br>
├─ lora\_resume\_predictions.jsonl<br>
└─ lora\_medical\_predictions.jsonl<br>

````

---

## 💻 Installation
1. Clone the repository:
```bash
git clone <repo_url>
cd project
````

2. Install dependencies:

```bash
pip install torch transformers datasets accelerate peft scikit-learn
```

3. (Optional) If running on GPU, ensure **CUDA**, **bitsandbytes**, and compatible versions are installed.

---

## 🚀 Usage

### 1. Fine-Tune LoRA Model

```bash
python Scripts/train_LoRA_resume.py  # For general resumes
python Scripts/train_LoRA_medical.py # For medical resumes
```

* Adjust paths for train/valid datasets inside the script.
* Adjust `LORA_SAVE_PATH` to control where the adapter is saved.

### 2. Evaluate LoRA Model

```bash
python Scripts/eval_LoRA_resume.py  # For general resumes
python Scripts/eval_LoRA_medical.py # For medical resumes
```

* Predictions will be saved in `Results/`.

### 3. Compute Metrics

```bash
python Scripts/eval_LoRA_metrics.py
```

* Outputs field-wise accuracy, exact match, and Levenshtein distance.

---

## 📊 Sample LoRA Evaluation Metrics

```
=== LORA EVALUATION METRICS (with JSON Repair) ===
Total examples (with valid preds): 31
Exact Match Accuracy: 58.06%
Name Accuracy: 100.0%
Email Accuracy: 90.32%
Skills Accuracy: 58.06%
Experience Accuracy: 58.06%
Average Levenshtein Distance: 34.65
```

---

## 📚 References

* Hugging Face Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
* PEFT & LoRA: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
* Few-shot prompting: Brown et al., *Language Models are Few-Shot Learners*, 2020
* Levenshtein Distance metric: [https://en.wikipedia.org/wiki/Levenshtein\_distance](https://en.wikipedia.org/wiki/Levenshtein_distance)

---

## ⚡ Next Steps

* Fine-tune on larger medical resume datasets for improved performance.
* Add new structured fields for domain-specific information.
* Experiment with hybrid few-shot + LoRA inference for low-resource settings.

```
```
