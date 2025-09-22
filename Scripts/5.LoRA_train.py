# !/usr/bin/env python3

"""
Train a LoRA adapter on the resume instruction dataset.

Expect:
  data/resume/train_inst.jsonl
  data/resume/valid_inst.jsonl

Saves adapter + tokenizer to --output_dir (default: models/lora_resume).
"""
import argparse
import json
import os
import sys
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data_dir", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical")# You can Just Change to "Resume" for tuning instead of "Medical" or Vice-Versa
    p.add_argument("--train_file", type=str, default="medical_LoRA_train.jsonl")# To use , Just change "medical" to "resume" or Vice-versa
    p.add_argument("--valid_file", type=str, default="medical_LoRA_valid.jsonl")# To use , Just change "medical" to "resume" or Vice-versa
    p.add_argument("--output_dir", type=str, default="E:/College/2nd Year/Sem 1/EDAI/Project/Models/LoRA_medical")#"LoRA_medical" to "LoRA_resume" or Vice-versa
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                   help="Comma-separated module names to apply LoRA to. Default: q_proj,v_proj")
    return p.parse_args()


def build_prompt_text(instruction: str, input_text: str, output_text: str):
    # Keep a compact instruction format that will be used to mask prompt tokens later
    return f"Instruction: {instruction}\nInput: {input_text}\n\nOutput:\n{output_text}"


def main():
    args = parse_args()
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    train_path = os.path.join(args.data_dir, args.train_file)
    valid_path = os.path.join(args.data_dir, args.valid_file)

    if not os.path.exists(train_path):
        logger.error(f"Train file not found: {train_path}")
        sys.exit(1)
    if not os.path.exists(valid_path):
        logger.error(f"Valid file not found: {valid_path}")
        sys.exit(1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load model (try 8-bit first)
    model = None
    # Load model safely for CPU-only environment
    if torch.cuda.is_available():
        # GPU available — try 8-bit load (faster)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                load_in_8bit=True,
                device_map="auto"
            )
            logger.info("Loaded model in 8-bit (CUDA) mode.")
        except Exception as e:
            logger.warning("8-bit load failed on CUDA; loading normally: %s", e)
            model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    else:
        # No GPU: load in full precision on CPU (stable)
        logger.info("No CUDA available — loading model on CPU in float32 (this will be slow).")
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="cpu", torch_dtype=torch.float32)

    # Apply LoRA (PEFT)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"Applied LoRA to target_modules={target_modules}")

    # Dataset loading
    data_files = {"train": train_path, "validation": valid_path}
    ds = load_dataset("json", data_files=data_files)

    instruction_field = "instruction"
    input_field = "input"
    output_field = "output"

    max_length = args.max_length
    tokenizer_fn = tokenizer

    def preprocess(example):
        # example['output'] is expected to be an object -> convert to exact JSON string
        instruction = example.get(instruction_field, "Extract structured JSON from the following text.")
        input_text = example.get(input_field, "")
        try:
            output_text = json.dumps(example.get(output_field, {}), ensure_ascii=False)
        except Exception:
            # fallback to string
            output_text = str(example.get(output_field, ""))

        prompt = build_prompt_text(instruction, input_text, "")  # without output
        prompt_ids = tokenizer_fn(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        full_text = prompt + output_text
        # full tokenization with truncation/padding
        tokenized = tokenizer_fn(full_text, truncation=True, max_length=max_length, padding="max_length")
        input_ids = tokenized["input_ids"]

        # If the prompt itself is >= max_length, skip this example (no label tokens available)
        if prompt_len >= max_length:
            # return markers so we can filter later
            return {"input_ids": input_ids, "labels": None, "skip_example": True}

        # Build labels: mask everything up to prompt_len
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        tokenized["labels"] = labels
        tokenized["skip_example"] = False
        return tokenized

    logger.info("Tokenizing and preparing datasets (this may take a minute)...")
    tokenized_train = ds["train"].map(preprocess, remove_columns=ds["train"].column_names)
    tokenized_valid = ds["validation"].map(preprocess, remove_columns=ds["validation"].column_names)

    # Filter out skipped examples
    tokenized_train = tokenized_train.filter(lambda x: x["skip_example"] == False)
    tokenized_valid = tokenized_valid.filter(lambda x: x["skip_example"] == False)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")

    # Training arguments
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving LoRA adapter and tokenizer to %s", output_dir)
    # Save only the adapter weights (PEFT)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
