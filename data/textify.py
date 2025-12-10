import os
import re
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
DATASET_NAME = "PleIAs/SYNTH"
OUTPUT_DIR = "synth_2"
MAX_ROWS = 1_000_000  # Stop after processing this many entries
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 50  # 50mb

# --- The Scrubber Configuration ---
# This regex matches any character that is NOT:
# 1. Standard ASCII (a-z, A-Z, 0-9, punctuation like : / . , etc) (\x20-\x7E)
# 2. Newlines or Tabs (\n, \t)
# 3. The standard Bullet Point (•)
# It replaces everything else with a space.
CLEANER_PATTERN = re.compile(r'[^\x20-\x7E\n\t•]')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    if not text: 
        return ""
    # 1. Remove "random bytes" / non-allowed chars
    text = CLEANER_PATTERN.sub(' ', text)
    # 2. Collapse multiple spaces into one (optional but good for cleaning scrubbed gaps)
    text = re.sub(r' +', ' ', text)
    return text

def get_text_entry(row):
    """
    Extracts fields, cleans them, and formats:
    seed + \n\n + reasoning + \n\n + answer + <|endoftext|>\n
    """
    # Use .get() with defaults
    seed = row.get("query_seed_text") or ""
    reasoning = row.get("synthetic_reasoning") or ""
    answer = row.get("synthetic_answer") or ""
    
    # --- SCRUBBING HAPPENS HERE ---
    seed = clean_text(seed)
    reasoning = clean_text(reasoning)
    answer = clean_text(answer)
    
    # Construct the final string
    text_block = f"{seed.strip()}\n\n{reasoning.strip()}\n\n{answer.strip()}<|endoftext|>\n"
    return text_block

def main():
    print(f"Loading {DATASET_NAME} (streaming mode)...")
    
    # Load dataset in streaming mode
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    file_index = 0
    current_file_path = os.path.join(OUTPUT_DIR, f"synth_part_{file_index:03d}.txt")
    current_file_handle = open(current_file_path, "w", encoding="utf-8")
    current_file_size = 0
    
    row_count = 0
    processed_count = 0
    
    print(f"Scanning rows (Filtering for 'en' only)...")
    
    # Iterate through the dataset
    for row in tqdm(ds, unit="rows"):
        if processed_count >= MAX_ROWS:
            break
            
        # --- LANGUAGE FILTER ---
        # Skip if language is missing or not 'en'
        lang = row.get('language')
        if lang != 'en':
            continue

        # Format and Clean the text
        text_entry = get_text_entry(row)
        
        # Calculate size in bytes (UTF-8)
        entry_size = len(text_entry.encode("utf-8"))
        
        # Check file size limit
        if current_file_size + entry_size > MAX_FILE_SIZE_BYTES:
            current_file_handle.close()
            print(f"\nClosed {current_file_path} (Size: {current_file_size/1024/1024:.2f} MB)")
            
            file_index += 1
            current_file_path = os.path.join(OUTPUT_DIR, f"synth_part_{file_index:03d}.txt")
            current_file_handle = open(current_file_path, "w", encoding="utf-8")
            current_file_size = 0
        
        current_file_handle.write(text_entry)
        current_file_size += entry_size
        
        processed_count += 1
        row_count += 1

    # Cleanup
    current_file_handle.close()
    print(f"\nDone! Processed {processed_count} English rows.")
    print(f"Files saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
