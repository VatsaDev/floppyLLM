import os
import re
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = "processed_math_data"
MAX_ROWS_AUTOMATH = 10_000
MAX_ROWS_REST = 10_000_000
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 50  # 50MB per chunk

# --- The Scrubber Configuration ---
CLEANER_PATTERN = re.compile(r'[^\x20-\x7E\n\t•]')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    if not text: 
        return ""
    # Standard ASCII cleanup
    text = CLEANER_PATTERN.sub(' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def format_chain(seed, reasoning, answer):
    """Constructs: Seed -> Reasoning -> Answer."""
    return f"{seed}\n\n{reasoning}\n\n{answer}<|endoftext|>\n"

# --- Dataset Parsers ---

def parse_automath(row):
    """AutoMathText: Just the 'text' field."""
    text = clean_text(row.get("text", ""))
    if not text: return None
    return f"{text}<|endoftext|>\n"

def parse_asdiv(row):
    """ASDiv: body -> question -> formula -> answer."""
    body = clean_text(row.get("body", ""))
    question = clean_text(row.get("question", ""))
    formula = clean_text(row.get("formula", ""))
    answer = clean_text(row.get("answer", ""))
    
    seed = f"{body} {question}".strip()
    return format_chain(seed, formula, answer)

def parse_mathqa(row):
    """mathQA/train.json: Problem -> Rationale -> correct (answer)."""
    problem = clean_text(row.get("Problem", ""))
    rationale = clean_text(row.get("Rationale", ""))
    answer = clean_text(row.get("correct", ""))
    options = clean_text(row.get("options", ""))
    
    seed = f"{problem}\nOptions: {options}".strip()
    return format_chain(seed, rationale, answer)

# --- Output Management ---

class FileManager:
    def __init__(self, output_dir, prefix):
        self.output_dir = output_dir
        self.prefix = prefix
        self.file_index = 0
        self.current_file_size = 0
        self.handle = None
        self._open_new_file()

    def _open_new_file(self):
        if self.handle: self.handle.close()
        path = os.path.join(self.output_dir, f"{self.prefix}_{self.file_index:03d}.txt")
        self.handle = open(path, "w", encoding="utf-8")
        self.file_index += 1
        self.current_file_size = 0

    def write(self, text):
        if not text: return
        encoded = text.encode("utf-8")
        if self.current_file_size + len(encoded) > MAX_FILE_SIZE_BYTES:
            self._open_new_file()
        self.handle.write(text)
        self.current_file_size += len(encoded)

    def close(self):
        if self.handle: self.handle.close()

# --- Main Script ---

def main():
    writer = FileManager(OUTPUT_DIR, "math_dataset_mix")

    # 1. MathQA (Local JSON)
    # Path: mathQA/train.json
    json_path = os.path.join("mathQA", "train.json")
    if os.path.exists(json_path):
        print(f"Parsing {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Take up to 100k
            for row in tqdm(data[:MAX_ROWS_REST], desc="MathQA"):
                writer.write(parse_mathqa(row))
    else:
        print(f"Skipping MathQA: {json_path} not found.")

    # 2. ASDiv (HF)
    print("\nParsing ASDiv...")
    try:
        asdiv = load_dataset("EleutherAI/asdiv", split="validation")
        # ASDiv is small (~2.3k), so it fits well within 100k limit
        for i, row in enumerate(tqdm(asdiv, desc="ASDiv")):
            if i >= MAX_ROWS_REST: break
            writer.write(parse_asdiv(row))
    except Exception as e:
        print(f"Error loading ASDiv: {e}")

    # 3. AutoMathText (HF - Streaming)
    # Limit: 10,000 rows
    print("\nParsing AutoMathText (Max 10k)...")
    try:
        am_ds = load_dataset("math-ai/AutoMathText", "web-0.50-to-1.00", split="train", streaming=True)
        count = 0
        pbar = tqdm(total=MAX_ROWS_AUTOMATH, desc="AutoMathText")
        for row in am_ds:
            if count >= MAX_ROWS_AUTOMATH: break
            writer.write(parse_automath(row))
            count += 1
            pbar.update(1)
        pbar.close()
    except Exception as e:
        print(f"Error loading AutoMathText: {e}")

    writer.close()
    print(f"\nDone! Files saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
