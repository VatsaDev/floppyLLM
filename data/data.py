import os
import pickle
import numpy as np
import glob
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Setup ---
tok_path = "../nano_8k" 
tok = AutoTokenizer.from_pretrained(tok_path)

# --- Hyperparameters ---
shard_size = 10_000_000  # 1 million tokens per shard
dataset_path = "synth_2"

DATA_CACHE_DIR = dataset_path

print(dataset_path)

input_files = glob.glob(os.path.join(dataset_path, "*.txt"))

if not input_files:
    print(f"Error: No .txt files found in {dataset_path}")
    exit()

print(f"Found {len(input_files)} files. Tokenizing...")

# list to collect tokens, convert to numpy
all_ids_list = []

for file_path in tqdm(input_files, desc="Reading & Tokenizing"):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # tokenizer to handle special tokens (EOS/BOS)
    ids = tok.encode(text, add_special_tokens=True)
    
    all_ids_list.extend(ids)
    
# Convert to numpy array (uint16 is good for vocab < 65535)
print(f"Converting to Numpy array (Current vocab size: {tok.vocab_size})...")
all_ids_np = np.array(all_ids_list, dtype=np.uint16)
print(f"Total tokens: {len(all_ids_np)}")

# train/val
n = len(all_ids_np)
split_idx = int(n * 0.9)
train_ids = all_ids_np[:split_idx]
val_ids = all_ids_np[split_idx:]

# Sharding Logic
def write_datafile(filename, data_shard):

    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1        # version
    header[2] = len(data_shard) # number of tokens in this shard
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(data_shard.tobytes())

def shard_dataset(ids, split_name):
    # Calculate how many shards
    num_shards = int(np.ceil(len(ids) / shard_size))
    
    print(f"Writing {split_name} data into {num_shards} shards...")
    
    for i in tqdm(range(num_shards), desc=f"Sharding {split_name}"):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(ids))
        
        shard_data = ids[start_idx:end_idx]
        
        filename = os.path.join(DATA_CACHE_DIR, f"{dataset_path}_{split_name}_{i:06d}.bin")
        write_datafile(filename, shard_data)

shard_dataset(train_ids, "train")
shard_dataset(val_ids, "val")

meta = {
    'vocab_size': tok.vocab_size,
    'tokenizer_path': tok_path, 
    'vocab_source': 'custom_bpe' 
}

meta_pkl_path = os.path.join(DATA_CACHE_DIR, 'meta.pkl')
with open(meta_pkl_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"Done! Dataset ready in: {DATA_CACHE_DIR}")

gitignore_path = '../.gitignore' # Adjusted path assuming script is running closer to root, adjust if needed
dataset_gitignore_entry = f"{dataset_path}/\n"

if os.path.exists(gitignore_path):
    with open(gitignore_path, 'r') as f:
        lines = f.readlines()
    if dataset_gitignore_entry not in lines:
        with open(gitignore_path, 'a') as f:
            f.write(dataset_gitignore_entry)
            print(f"Added to .gitignore")
else:
    with open(gitignore_path, 'w') as f:
        f.write(dataset_gitignore_entry)
