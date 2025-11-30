import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------
# 1. Load and combine datasets
# ---------------------------------------------------------------------

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_dataset(real_train, synthetic_path, level1, level2, level3):
    real = load_jsonl(real_train)
    synthetic = load_jsonl(synthetic_path)
    lvl1 = load_jsonl(level1)
    lvl2 = load_jsonl(level2)
    lvl3 = load_jsonl(level3)

    combined = real + synthetic + lvl1 + lvl2 + lvl3
    print(f"Total combined training samples = {len(combined)}")

    return Dataset.from_list(combined)


# ---------------------------------------------------------------------
# 2. Tokenizing Dataset
# ---------------------------------------------------------------------

def tokenize(example):
    instruction = """
Extract structured information from the resume below into a JSON object
with the following keys: ["name", "email", "skills", "experience"].

Resume:
"""
    inp = example["input"]
    out = json.dumps(example["output"])

    # Build final prompt
    prompt = f"{instruction}\n{inp}\n\nOutput JSON:\n{out}"

    # Tokenize
    enc = tokenizer(prompt, truncation=True, max_length=512)

    # Critical: Add labels
    enc["labels"] = enc["input_ids"].copy()

    return enc


# ---------------------------------------------------------------------
# 3. Load model & tokenizer
# ---------------------------------------------------------------------

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# ---------------------------------------------------------------------
# 4. LoRA Config
# ---------------------------------------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
print("LoRA Model Ready.")

# ---------------------------------------------------------------------
# 5. Set DATA PATHS HERE (RESUME or MEDICAL)
# ---------------------------------------------------------------------

REAL_TRAIN = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_train.jsonl"
SYNTHETIC = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_synthetic.jsonl"
LEVEL1 = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/Curriculum/level_1.jsonl"
LEVEL2 = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/Curriculum/level_2.jsonl"
LEVEL3 = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/Curriculum/level_3.jsonl"

# 🔄 For MEDICAL change paths to:
# REAL_TRAIN = "Data/Medical/medical_train.jsonl"
# SYNTHETIC = "Data/Medical/medical_synthetic.jsonl"
# LEVEL1 = "Data/Medical/Curriculum/level1.jsonl"
# LEVEL2 = "Data/Medical/Curriculum/level2.jsonl"
# LEVEL3 = "Data/Medical/Curriculum/level3.jsonl"

SAVE_DIR = "E:/College/2nd Year/Sem 1/EDAI/Project/Models/Synthetic_LoRA_Medical"


# ---------------------------------------------------------------------
# 6. Build Dataset
# ---------------------------------------------------------------------

train_dataset = build_dataset(REAL_TRAIN, SYNTHETIC, LEVEL1, LEVEL2, LEVEL3)
train_dataset = train_dataset.map(tokenize)

# ---------------------------------------------------------------------
# 7. Training Arguments
# ---------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=4e-4,
    num_train_epochs=1,
    logging_steps=400,
    save_strategy="epoch",
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# ---------------------------------------------------------------------
# 8. Train
# ---------------------------------------------------------------------

trainer.train()
model.save_pretrained(SAVE_DIR)

print("\n\n=== Synthetic LoRA Training Complete ===")
print(f"Saved to: {SAVE_DIR}")

# C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe "E:\College\2nd Year\Sem 1\EDAI\Project\Scripts\16.train_synthetic_lora.py"
# 2025-11-28 11:52:36.844595: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-11-28 11:52:50.589338: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING:tensorflow:From C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
#
# `torch_dtype` is deprecated! Use `dtype` instead!
# The 8-bit optimizer is not available on your device, only available on CUDA for now.
# LoRA Model Ready.
# Total combined training samples = 741
# Map: 100%|██████████| 741/741 [00:02<00:00, 260.25 examples/s]
#   0%|          | 0/371 [00:00<?, ?it/s]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  54%|█████▍    | 200/371 [3:31:25<3:07:19, 65.73s/it]{'loss': 0.8602, 'grad_norm': 0.5255451202392578, 'learning_rate': 0.00013908355795148246, 'epoch': 0.54}
# 100%|██████████| 371/371 [6:29:55<00:00, 53.87s/it]{'train_runtime': 23395.3083, 'train_samples_per_second': 0.032, 'train_steps_per_second': 0.016, 'train_loss': 0.7209777009455021, 'epoch': 1.0}
# 100%|██████████| 371/371 [6:29:55<00:00, 63.06s/it]
#
#
# === Synthetic LoRA Training Complete ===
# Saved to: E:/College/2nd Year/Sem 1/EDAI/Project/Models/Synthetic_LoRA_Resume
#
# Process finished with exit code 0
# C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe "E:\College\2nd Year\Sem 1\EDAI\Project\Scripts\16.train_synthetic_lora.py"
# 2025-11-29 10:38:37.291682: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-11-29 10:38:53.525361: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING:tensorflow:From C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
#
# `torch_dtype` is deprecated! Use `dtype` instead!
# The 8-bit optimizer is not available on your device, only available on CUDA for now.
# LoRA Model Ready.
# Total combined training samples = 640
# Map: 100%|██████████| 640/640 [00:03<00:00, 203.23 examples/s]
#   0%|          | 0/320 [00:00<?, ?it/s]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# 100%|██████████| 320/320 [6:24:54<00:00, 73.97s/it]{'train_runtime': 23095.0517, 'train_samples_per_second': 0.028, 'train_steps_per_second': 0.014, 'train_loss': 0.7690901279449462, 'epoch': 1.0}
# 100%|██████████| 320/320 [6:24:55<00:00, 72.17s/it]
#
#
# === Synthetic LoRA Training Complete ===
# Saved to: E:/College/2nd Year/Sem 1/EDAI/Project/Models/Synthetic_LoRA_Medical
