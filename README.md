
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


project/<br>
│<br>
├─ Scripts/<br>
│    ├─[1.prepare_data.py](Scripts/1.prepare_data.py)
│    ├─[2.fewshots_train.py](Scripts/2.fewshots_train.py)
│    ├─[3.fewshots_eval.py](Scripts/3.fewshots_eval.py)
│    ├─[4.convert_to_instruct.py](Scripts/4.convert_to_instruct.py)
│    ├─[5.LoRA_train.py](Scripts/5.LoRA_train.py)
│    ├─[6.testing_on_example.py](Scripts/6.testing_on_example.py)
│    ├─[7.testing_on_dataset.py](Scripts/7.testing_on_dataset.py)
│    ├─[8.LoRA_eval.py](Scripts/8.LoRA_eval.py)
│<br>
├─ Data/<br>
[Data](Data)
[Medical](Data/Medical)
[medical_LoRA_test.jsonl](Data/Medical/medical_LoRA_test.jsonl)
[medical_LoRA_train.jsonl](Data/Medical/medical_LoRA_train.jsonl)
[medical_LoRA_valid.jsonl](Data/Medical/medical_LoRA_valid.jsonl)
[medical_raw.jsonl](Data/Medical/medical_raw.jsonl)
[medical_test.jsonl](Data/Medical/medical_test.jsonl)
[medical_train.jsonl](Data/Medical/medical_train.jsonl)
[medical_valid.jsonl](Data/Medical/medical_valid.jsonl)
[Resume](Data/Resume)
[resume_LoRA_test.jsonl](Data/Resume/resume_LoRA_test.jsonl)
[resume_LoRA_train.jsonl](Data/Resume/resume_LoRA_train.jsonl)
[resume_LoRA_valid.jsonl](Data/Resume/resume_LoRA_valid.jsonl)
[resume_raw.jsonl](Data/Resume/resume_raw.jsonl)
[resume_test.jsonl](Data/Resume/resume_test.jsonl)
[resume_train.jsonl](Data/Resume/resume_train.jsonl)
[resume_valid.jsonl](Data/Resume/resume_valid.jsonl)
│<br>
├─ Models/<br>
│   ├─ LoRA_resume/               # Saved LoRA adapter for general resumes<br>
│   └─ LoRA_medical/              # Saved LoRA adapter for medical resumes<br>
│<br>
└─ Results/<br>
[fewshot_medical.jsonl](Results/fewshot_medical.jsonl)
[fewshot_resume.jsonl](Results/fewshot_resume.jsonl)
[lora_medical_predictions.jsonl](Results/lora_medical_predictions.jsonl)
[lora_resume_predictions.jsonl](Results/lora_resume_predictions.jsonl)

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
