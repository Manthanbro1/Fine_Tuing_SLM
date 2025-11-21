import json
import argparse
import os
from tqdm import tqdm

# ----------------------------
# Utility: Read a JSONL file safely
# ----------------------------
def read_jsonl(path):
    """Reads a JSONL file and returns a generator of parsed JSON objects."""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] Skipping invalid JSON line: {line[:50]}...")

# ----------------------------
# Utility: Write a JSONL file safely
# ----------------------------
def write_jsonl(data, path):
    """Writes a list of JSON objects into a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# ----------------------------
# Difficulty Scoring Function
# ----------------------------
def compute_difficulty(entry):
    """
    Computes a simple 'difficulty' score based on:
    - text length
    - JSON nesting depth
    - number of fields
    This helps separate data into curriculum levels.
    """
    text_len = len(entry.get('input', ''))
    output_obj = entry.get('output', {})
    num_fields = len(output_obj.keys())
    nesting = json.dumps(output_obj).count('{')  # rough heuristic

    return text_len * 0.5 + num_fields * 10 + nesting * 5

# ----------------------------
# Main Preprocessing Logic
# ----------------------------
def main(args):
    # If no input file is provided, fallback to default
    if args.input is None:
        args.input = r"E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_raw.jsonl"
        print(f"[INFO] No --input given, using default path: {args.input}")

    # Verify file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"❌ Input file not found: {args.input}")

    print(f"[INFO] Reading dataset from: {args.input}")
    data = list(read_jsonl(args.input))
    print(f"[INFO] Loaded {len(data)} samples")

    # Compute difficulty for each sample
    for item in tqdm(data, desc="Computing difficulty"):
        item['difficulty'] = compute_difficulty(item)

    # Sort by difficulty
    data.sort(key=lambda x: x['difficulty'])

    # Split into three curriculum levels (1→easy, 3→hard)
    n = len(data)
    level1 = data[: n // 3]
    level2 = data[n // 3 : 2 * n // 3]
    level3 = data[2 * n // 3 :]

    # Write outputs
    out_dir = os.path.join(os.path.dirname(args.input), "Curriculum")
    write_jsonl(level1, os.path.join(out_dir, "Level_1.jsonl"))
    write_jsonl(level2, os.path.join(out_dir, "Level_2.jsonl"))
    write_jsonl(level3, os.path.join(out_dir, "Level_3.jsonl"))

    print(f"[DONE] Saved curriculum datasets to: {out_dir}")
    print(f"    Level_1: {len(level1)} samples")
    print(f"    Level_2: {len(level2)} samples")
    print(f"    Level_3: {len(level3)} samples")

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for curriculum fine-tuning")
    parser.add_argument("--input", type=str,default = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/Medical/medical_raw.jsonl")
    args = parser.parse_args()
    main(args)
