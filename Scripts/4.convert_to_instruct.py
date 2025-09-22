import json
import os

# --------------------------
# Config
# --------------------------
DATASETS = ["resume", "medical"]   # both datasets
DATA_DIR = "E:/College/2nd Year/Sem 1/EDAI/Project/Data"   # adjust if your path differs

# Instruction prompt template
INSTRUCTION = "Extract structured JSON information from the following text."

def convert_split(dataset: str, split: str):
    """Convert split.jsonl -> split_inst.jsonl for one dataset"""
    in_path = os.path.join(DATA_DIR, dataset, f"{dataset}_{split}.jsonl")
    out_path = os.path.join(DATA_DIR, dataset, f"{dataset}_LoRA_{split}.jsonl")

    if not os.path.exists(in_path):
        print(f"Missing: {in_path}")
        return

    new_data = []
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            new_ex = {
                "instruction": INSTRUCTION,
                "input": ex["input"],
                "output": ex["output"]
            }
            new_data.append(new_ex)

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in new_data:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(new_data)} examples to {out_path}")


if __name__ == "__main__":
    for dataset in DATASETS:
        for split in ["train", "valid", "test"]:
            convert_split(dataset, split)
