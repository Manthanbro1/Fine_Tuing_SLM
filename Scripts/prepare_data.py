import json
import random
import os

# Paths
raw_path = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/resume_raw.jsonl"
train_path = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/resume_train.jsonl"
valid_path = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/resume_valid.jsonl"
test_path = "E:/College/2nd Year/Sem 1/EDAI/Project/Data/resume_test.jsonl"

# Read raw data
with open(raw_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Shuffle to avoid order bias
random.shuffle(data)

# Split sizes
train_size = int(0.7 * len(data))   # 70% for training
valid_size = int(0.15 * len(data))  # 15% for validation
test_size  = len(data) - train_size - valid_size

train_data = data[:train_size]
valid_data = data[train_size:train_size+valid_size]
test_data  = data[train_size+valid_size:]

# Save splits
def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(train_data, train_path)
save_jsonl(valid_data, valid_path)
save_jsonl(test_data, test_path)

print(f"Done! Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
