import os
import pickle
import numpy as np
import glob
import json
from tqdm import tqdm
from transformers import AutoTokenizer

tok_path = "../nano_8k" 
dataset_path = "synthcat"   # Folder containing images and text files
DATA_CACHE_DIR = dataset_path

shard_size = 10_000_000  # 1 million tokens per shard
IMAGE_TOKEN_ID = -100    # The placeholder ID for the Vision Encoder

print(f"Loading tokenizer from {tok_path}...")
tok = AutoTokenizer.from_pretrained(tok_path)
print(f"Vocab Size: {tok.vocab_size}")

print(f"Processing dataset in: {dataset_path}")

# 1. Find all Images first
# We gather images first because they are the anchors for the dataset.
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(dataset_path, ext)))

# SORTING IS CRITICAL for deterministic behavior
image_files.sort() 

if not image_files:
    print(f"Error: No images found in {dataset_path}")
    exit()

print(f"Found {len(image_files)} images. Looking for matching text...")

# Lists to collect data
all_ids_list = []
full_image_manifest = [] # Stores the relative path to the image for every -100 token

def extract_text(content):
    """
    Parses the 'elements' JSON structure from your dataset
    or falls back to raw text if it's a plain txt file.
    """
    try:
        data = json.loads(content)
        # Handle the structure: {"text": ["line1", "line2"]}
        if isinstance(data, dict) and "text" in data:
            if isinstance(data["text"], list):
                return "\n".join(data["text"])
            return data["text"]
    except json.JSONDecodeError:
        pass
    return content # Fallback: return raw string

# --- Main Tokenization Loop ---
for img_path in tqdm(image_files, desc="Tokenizing Pairs"):
    base_name = os.path.splitext(img_path)[0]
    
    # Look for matching text file (.json or .txt)
    text_path = None
    if os.path.exists(base_name + ".json"):
        text_path = base_name + ".json"
    elif os.path.exists(base_name + ".txt"):
        text_path = base_name + ".txt"
        
    if text_path:
        # Read and clean text
        with open(text_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            clean_text = extract_text(raw_content)
        
        # Tokenize text
        text_ids = tok.encode(clean_text, add_special_tokens=True)
        
        # --- THE MULTIMODAL FORMAT ---
        # [IMAGE_TOKEN, text_tokens...]
        # We assume the image comes *before* the text description
        ids = [IMAGE_TOKEN_ID] + text_ids
        
        all_ids_list.extend(ids)
        
        # Store relative path (better for portability)
        rel_path = os.path.relpath(img_path, start=dataset_path)
        full_image_manifest.append(rel_path)

# --- Conversion to Numpy ---
print(f"Converting to Numpy array...")

# Logic to handle negative token IDs based on vocabulary size
if tok.vocab_size > 32767:
    dtype = np.int32
    print(f"Vocab > 32k ({tok.vocab_size}), using int32 to support ID {IMAGE_TOKEN_ID}")
else:
    dtype = np.int16
    print(f"Vocab < 32k ({tok.vocab_size}), using int16 to support ID {IMAGE_TOKEN_ID}")

all_ids_np = np.array(all_ids_list, dtype=dtype)
print(f"Total tokens: {len(all_ids_np)}")
print(f"Total images linked: {len(full_image_manifest)}")

# --- THE EXACT SPLIT LOGIC ---
print("Performing EXACT split calculation...")

# 1. Define split point by token volume (90%)
n_tokens = len(all_ids_np)
split_idx = int(n_tokens * 0.9)

train_ids = all_ids_np[:split_idx]
val_ids = all_ids_np[split_idx:]

# 2. Count EXACTLY how many image tokens are in the Train Set
# This tells us exactly where to slice the Manifest list
num_train_images = np.count_nonzero(train_ids == IMAGE_TOKEN_ID)

# 3. Slice the Manifest using the exact count
train_manifest = full_image_manifest[:num_train_images]
val_manifest = full_image_manifest[num_train_images:]

print(f"--- Split Statistics ---")
print(f"Train Tokens: {len(train_ids):,} | Train Images: {len(train_manifest):,}")
print(f"Val Tokens:   {len(val_ids):,} | Val Images:   {len(val_manifest):,}")

# Double check synchronization
if len(train_manifest) != num_train_images:
    print("CRITICAL ERROR: Train manifest length does not match image token count!")
    exit()

# --- Helper: Write Binary Shards ---
def write_datafile(filename, data_shard):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1        # version
    header[2] = len(data_shard) 
    
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(data_shard.tobytes())

def shard_dataset(ids, split_name):
    num_shards = int(np.ceil(len(ids) / shard_size))
    print(f"Writing {split_name} data into {num_shards} shards...")
    
    for i in tqdm(range(num_shards), desc=f"Sharding {split_name}"):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(ids))
        shard_data = ids[start_idx:end_idx]
        
        filename = os.path.join(DATA_CACHE_DIR, f"{dataset_path}_{split_name}_{i:06d}.bin")
        write_datafile(filename, shard_data)

# Write binaries
shard_dataset(train_ids, "train")
shard_dataset(val_ids, "val")

# --- Save Manifests & Metadata ---
print("Saving manifests and metadata...")

with open(os.path.join(DATA_CACHE_DIR, 'train_manifest.json'), 'w') as f:
    json.dump(train_manifest, f)
    
with open(os.path.join(DATA_CACHE_DIR, 'val_manifest.json'), 'w') as f:
    json.dump(val_manifest, f)

meta = {
    'vocab_size': tok.vocab_size,
    'tokenizer_path': tok_path, 
    'vocab_source': 'custom_bpe',
    'image_token_id': IMAGE_TOKEN_ID,
    'train_images_count': len(train_manifest),
    'val_images_count': len(val_manifest)
}

meta_pkl_path = os.path.join(DATA_CACHE_DIR, 'meta.pkl')
with open(meta_pkl_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"Done! Dataset ready in: {DATA_CACHE_DIR}")

# --- Gitignore Update ---
gitignore_path = '../.gitignore'
dataset_gitignore_entry = f"{dataset_path}/\n"

if os.path.exists(gitignore_path):
    with open(gitignore_path, 'r') as f:
        lines = f.readlines()
    if dataset_gitignore_entry not in lines:
        with open(gitignore_path, 'a') as f:
            f.write(dataset_gitignore_entry)
            print(f"Added {dataset_path} to .gitignore")
else:
    with open(gitignore_path, 'w') as f:
        f.write(dataset_gitignore_entry)
