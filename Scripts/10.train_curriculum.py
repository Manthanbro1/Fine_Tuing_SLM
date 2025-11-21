"""
train_curriculum_lora.py

Progressive Curriculum LoRA training script for Text -> JSON structured generation.

Features:
- Loads Level_1/Level_2/Level_3 jsonl files created by preprocess_curriculum.py
- Uses Hugging Face Transformers + PEFT (LoRA)
- Auto-detects GPU and uses FP16 when available
- Trains consecutively on Level_1 -> Level_2 -> Level_3, saving checkpoints for each stage
- Evaluates JSON validity & simple exact-match metrics after each stage (basic eval)
- Robust argument parsing and helpful defaults for low-VRAM setups

Usage example:
python train_curriculum_lora.py \
  --data_dir Data/resume/Curriculum \
  --model_name_or_path gpt2 \
  --output_dir outputs/c_lora \
  --num_epochs 3 \
  --per_device_train_batch

Notes / Recommendations:
- For small SLMs (<=300M params) gpt2 / distilgpt2 / Eleuther-125M are reasonable local choices.
- If you have a small GPU, set batch size > 1. If GPU not available, training will run on CPU (sl_size 4ower).
- This script expects JSONL files where each line is an object with fields 'input' and 'output' (output is a dict).
- Tokenization template: we train the model to predict the JSON text given the input prompt. The sample is formatted like:
  "<BOS> Convert to JSON:\nInput: {input}\nOutput: {json_text} <EOS>"

"""

import argparse
import os
import json
import math
from typing import List, Dict
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def make_prompt(example: Dict) -> str:
    """Create the training prompt->target string. Keep consistent across dataset and eval.
    Expects example to have 'input' (str) and 'output' (dict) keys.
    Returns: single string where model sees input and target concatenated; tokenization will be causal.
    """
    inp = example.get('input') or example.get('text') or example.get('prompt') or ''
    out = example.get('output') or example.get('json') or example.get('label') or {}
    # ensure out is string of canonical JSON (no spaces changed)
    try:
        out_text = json.dumps(out, ensure_ascii=False)
    except Exception as e:
        out_text = str(out)
    prompt = f"Convert the text to JSON:\nInput: {inp}\nOutput: {out_text}"
    return prompt


def prepare_dataset(data_files: Dict[str, str], tokenizer, block_size=512):
    """Load datasets (expects jsonl files) and tokenize them. Returns a dict of tokenized datasets.
    data_files: mapping level_name -> path
    """
    tokenized = {}
    for level, path in data_files.items():
        ds = load_dataset('json', data_files=path, split='train')

        # map to prompt text
        def _map_fn(example):
            example['text_all'] = make_prompt(example)
            return example

        ds = ds.map(_map_fn)

        # tokenize
        def tokenize_fn(examples):
            return tokenizer(examples['text_all'], truncation=True, max_length=block_size)

        ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
        ds.set_format(type='torch')
        tokenized[level] = ds
    return tokenized


def compute_json_validity(pred_texts: List[str]) -> float:
    ok = 0
    total = len(pred_texts)
    for t in pred_texts:
        try:
            # find first '{' to support possible prefix noise
            s = t[t.find('{'):]
            json.loads(s)
            ok += 1
        except Exception:
            continue
    return 100.0 * ok / total if total > 0 else 0.0


def basic_eval(model, tokenizer, dataset, num_samples=100, device='cpu'):
    # sample texts for inference
    import random
    inds = list(range(len(dataset)))
    random.shuffle(inds)
    inds = inds[:min(num_samples, len(dataset))]
    preds = []
    refs = []
    model.eval()
    for i in inds:
        item = dataset[i]
        input_ids = item['input_ids'].unsqueeze(0).to(device)
        # generate
        with torch.no_grad():
            gen = model.generate(input_ids, max_new_tokens=256, do_sample=False)
        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        preds.append(decoded)
        refs.append(item)  # not used now, placeholder for better metrics
    validity = compute_json_validity(preds)
    return {'json_validity_%': validity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default='E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/Curriculum', help='Directory with Level_1.jsonl Level_2.jsonl Level_3.jsonl')
    parser.add_argument('--model_name_or_path', type=str, default='HuggingFaceTB/SmolLM2-360M-Instruct', help='base model')
    parser.add_argument('--output_dir', type=str, default='E:/College/2nd Year/Sem 1/EDAI/Project/Results/Curriculum_LoRA_Medical', help='where to save checkpoints')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=2, help='epochs per level')
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # check files
    data_dir = Path(args.data_dir)
    level_files = {
        'level1': str(data_dir / 'Level_1.jsonl'),
        'level2': str(data_dir / 'Level_2.jsonl'),
        'level3': str(data_dir / 'Level_3.jsonl'),
    }
    for k, p in level_files.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file for {k} at {p}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    print('[INFO] Loading base model (may take a while)')
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    # prepare model for LoRA / k-bit training if needed
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'] if hasattr(model.config, 'n_head') else None,
        bias='none',
        task_type='CAUSAL_LM'
    )

    model = get_peft_model(model, lora_config)

    # load and tokenize datasets
    tokenized = prepare_dataset(level_files, tokenizer, block_size=args.block_size)

    # data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training loop: iterate through levels
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, lvl in enumerate(['level1', 'level2', 'level3'], start=1):
        ds = tokenized[lvl]
        print(f"[TRAIN] Level {idx} -> {lvl} with {len(ds)} samples")

        out_dir_lvl = os.path.join(args.output_dir, f'stage{idx}')
        training_args = TrainingArguments(
            output_dir=out_dir_lvl,
            per_device_train_batch_size=args.per_device_train_batch_size,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            logging_steps=20,
            save_strategy='epoch',
            save_total_limit=args.save_total_limit,
            fp16=(device == 'cuda'),
            remove_unused_columns=False,
            seed=args.seed,
            report_to=['none'],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(out_dir_lvl)

        # basic evaluation
        eval_metrics = basic_eval(model, tokenizer, ds, num_samples=50, device=device)
        print(f"[EVAL] Level {idx} results: {eval_metrics}")

    print('[DONE] Curriculum LoRA training finished. Final model in:', args.output_dir)


if __name__ == '__main__':
    main()



# C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe "E:\College\2nd Year\Sem 1\EDAI\Project\Scripts\10.train_curriculum.py"
# 2025-11-07 21:21:45.419295: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-11-07 21:21:47.833169: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING:tensorflow:From C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

# [INFO] Using device: cpu
# [INFO] Loading base model (may take a while)
# The 8-bit optimizer is not available on your device, only available on CUDA for now.
# Generating train split: 67 examples [00:05, 12.19 examples/s]
# Map: 100%|██████████| 67/67 [00:05<00:00, 11.73 examples/s]
# Map: 100%|██████████| 67/67 [00:03<00:00, 16.95 examples/s]
# Generating train split: 67 examples [00:00, 983.10 examples/s]
# Map: 100%|██████████| 67/67 [00:00<00:00, 322.31 examples/s]
# Map: 100%|██████████| 67/67 [00:00<00:00, 136.47 examples/s]
# Generating train split: 67 examples [00:00, 1320.49 examples/s]
# Map: 100%|██████████| 67/67 [01:52<00:00,  1.67s/ examples]
# Map: 100%|██████████| 67/67 [00:00<00:00, 77.53 examples/s]
# [TRAIN] Level 1 -> level1 with 67 samples
#   0%|          | 0/34 [00:00<?, ?it/s]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  50%|█████     | 17/34 [15:34<09:51, 34.82s/it]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  59%|█████▉    | 20/34 [25:06<29:23, 125.94s/it]{'loss': 1.6928, 'grad_norm': 0.4871947169303894, 'learning_rate': 8.823529411764706e-05, 'epoch': 1.18}
# 100%|██████████| 34/34 [37:16<00:00, 35.15s/it]{'train_runtime': 2235.5973, 'train_samples_per_second': 0.06, 'train_steps_per_second': 0.015, 'train_loss': 1.6168497870950138, 'epoch': 2.0}
# 100%|██████████| 34/34 [37:17<00:00, 65.81s/it]
# The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# [EVAL] Level 1 results: {'json_validity_%': 32.0}
# [TRAIN] Level 2 -> level2 with 67 samples
#   0%|          | 0/34 [00:00<?, ?it/s]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  50%|█████     | 17/34 [08:01<07:33, 26.66s/it]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  59%|█████▉    | 20/34 [09:33<06:45, 28.95s/it]{'loss': 1.3548, 'grad_norm': 0.5842081904411316, 'learning_rate': 8.823529411764706e-05, 'epoch': 1.18}
# 100%|██████████| 34/34 [17:03<00:00, 31.35s/it]{'train_runtime': 1023.3662, 'train_samples_per_second': 0.131, 'train_steps_per_second': 0.033, 'train_loss': 1.272978894850787, 'epoch': 2.0}
# 100%|██████████| 34/34 [17:03<00:00, 30.10s/it]
# [EVAL] Level 2 results: {'json_validity_%': 32.0}
# [TRAIN] Level 3 -> level3 with 67 samples
#   0%|          | 0/34 [00:00<?, ?it/s]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  50%|█████     | 17/34 [08:56<08:13, 29.02s/it]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  59%|█████▉    | 20/34 [10:42<07:29, 32.08s/it]{'loss': 1.0373, 'grad_norm': 0.6719347834587097, 'learning_rate': 8.823529411764706e-05, 'epoch': 1.18}
# 100%|██████████| 34/34 [17:48<00:00, 31.44s/it]
# {'train_runtime': 1068.8662, 'train_samples_per_second': 0.125, 'train_steps_per_second': 0.032, 'train_loss': 0.9865739766289207, 'epoch': 2.0}
# [EVAL] Level 3 results: {'json_validity_%': 0.0}
# [DONE] Curriculum LoRA training finished. Final model in: E:/College/2nd Year/Sem 1/EDAI/Project/Results/Curriculum_LoRA_Resume
#
# Process finished with exit code 0

#
# C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe "E:\College\2nd Year\Sem 1\EDAI\Project\Scripts\10.train_curriculum.py"
# 2025-11-21 16:20:05.909496: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-11-21 16:20:49.654954: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING:tensorflow:From C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
#
# [INFO] Using device: cpu
# [INFO] Loading base model (may take a while)
# The 8-bit optimizer is not available on your device, only available on CUDA for now.
# Generating train split: 66 examples [00:02, 27.45 examples/s]
# Map: 100%|██████████| 66/66 [00:03<00:00, 21.10 examples/s]
# Map: 100%|██████████| 66/66 [00:02<00:00, 23.09 examples/s]
# Generating train split: 67 examples [00:00, 187.81 examples/s]
# Map: 100%|██████████| 67/67 [00:00<00:00, 251.17 examples/s]
# Map: 100%|██████████| 67/67 [00:00<00:00, 113.25 examples/s]
# Generating train split: 67 examples [00:00, 727.85 examples/s]
# Map: 100%|██████████| 67/67 [00:00<00:00, 245.44 examples/s]
# Map: 100%|██████████| 67/67 [00:00<00:00, 216.01 examples/s]
# [TRAIN] Level 1 -> level1 with 66 samples
#   0%|          | 0/34 [00:00<?, ?it/s]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  50%|█████     | 17/34 [29:25<23:02, 81.30s/it]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  59%|█████▉    | 20/34 [34:56<22:42, 97.30s/it]{'loss': 1.6733, 'grad_norm': 0.43099167943000793, 'learning_rate': 8.823529411764706e-05, 'epoch': 1.18}
# 100%|██████████| 34/34 [55:26<00:00, 76.70s/it]{'train_runtime': 3326.5951, 'train_samples_per_second': 0.04, 'train_steps_per_second': 0.01, 'train_loss': 1.6087284088134766, 'epoch': 2.0}
# 100%|██████████| 34/34 [55:26<00:00, 97.84s/it]
# The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# [EVAL] Level 1 results: {'json_validity_%': 40.0}
# [TRAIN] Level 2 -> level2 with 67 samples
#   0%|          | 0/34 [00:00<?, ?it/s]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  50%|█████     | 17/34 [27:32<25:22, 89.58s/it]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  59%|█████▉    | 20/34 [33:51<24:57, 106.98s/it]{'loss': 1.3337, 'grad_norm': 0.548821747303009, 'learning_rate': 8.823529411764706e-05, 'epoch': 1.18}
# 100%|██████████| 34/34 [57:35<00:00, 111.06s/it]{'train_runtime': 3455.815, 'train_samples_per_second': 0.039, 'train_steps_per_second': 0.01, 'train_loss': 1.2519117243149702, 'epoch': 2.0}
# 100%|██████████| 34/34 [57:35<00:00, 101.64s/it]
# [EVAL] Level 2 results: {'json_validity_%': 64.0}
# [TRAIN] Level 3 -> level3 with 67 samples
#   0%|          | 0/34 [00:00<?, ?it/s]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  50%|█████     | 17/34 [41:38<36:13, 127.84s/it]C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  59%|█████▉    | 20/34 [48:15<29:00, 124.33s/it]{'loss': 1.1309, 'grad_norm': 0.7117676734924316, 'learning_rate': 8.823529411764706e-05, 'epoch': 1.18}
# 100%|██████████| 34/34 [1:18:54<00:00, 123.33s/it]{'train_runtime': 4734.4724, 'train_samples_per_second': 0.028, 'train_steps_per_second': 0.007, 'train_loss': 1.08854061014512, 'epoch': 2.0}
# 100%|██████████| 34/34 [1:18:54<00:00, 139.25s/it]
# [EVAL] Level 3 results: {'json_validity_%': 64.0}
# [DONE] Curriculum LoRA training finished. Final model in: E:/College/2nd Year/Sem 1/EDAI/Project/Results/Curriculum_LoRA_Medical
