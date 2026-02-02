import os
import re
import json
import torch
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = "processed_math_data"
MAX_ROWS_AUTOMATH = 100_000 # Keep this low, it's noisy web data
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 50  # 50MB per chunk

# --- The Math-Safe Scrubber ---
# This preserves standard ASCII + common math symbols + LaTeX backslashes
# If you delete these, the model cannot learn the "logic" of the operations.
CLEANER_PATTERN = re.compile(r'[^\x20-\x7E\n\tπθΔΣ±÷×√∞≈≠≤≥^\\{}_]')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    if not text: 
        return ""
    # 1. Remove non-math-safe characters
    text = CLEANER_PATTERN.sub(' ', text)
    # 2. Collapse multiple spaces but keep single spaces for readability
    text = re.sub(r' +', ' ', text)
    return text.strip()

def format_chain(problem, reasoning, answer):
    """
    Constructs a structured prompt. 
    Explicit headers are CRITICAL for a 50-layer/2-head model 
    to stop it from looping or getting lost in the residual stream.
    """
    return (
        f"### Problem: {problem}\n"
        f"### Solution: {reasoning}\n"
        f"### Final Answer: {answer}<|endoftext|>\n"
    )

# --- Dataset Parsers ---

def parse_gsm8k(row):
    """GSM8K: The gold standard for school math reasoning."""
    # GSM8K answer format is "Reasoning steps... #### 42"
    q = clean_text(row.get("question", ""))
    a_field = row.get("answer", "")
    if "####" in a_field:
        reasoning, final_ans = a_field.split("####")
    else:
        reasoning, final_ans = a_field, ""
    return format_chain(q, clean_text(reasoning), clean_text(final_ans))

def parse_asdiv(row):
    """ASDiv: Primary school word problems."""
    body = clean_text(row.get("body", ""))
    question = clean_text(row.get("question", ""))
    formula = clean_text(row.get("formula", ""))
    answer = clean_text(row.get("answer", ""))
    
    problem = f"{body} {question}".strip()
    return format_chain(problem, formula, answer)

def parse_mathqa(row):
    """MathQA: Structured rationale and options."""
    problem = clean_text(row.get("Problem", ""))
    rationale = clean_text(row.get("Rationale", ""))
    answer = clean_text(row.get("correct", ""))
    options = clean_text(row.get("options", ""))
    
    full_problem = f"{problem}\nOptions: {options}"
    return format_chain(full_problem, rationale, answer)

def parse_automath(row):
    """AutoMathText: Raw text format. Use as a 'flavor' supplement."""
    text = clean_text(row.get("text", ""))
    if not text: return None
    # No reasoning/answer split here, so just wrap it
    return f"### Math Content:\n{text}<|endoftext|>\n"

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

# --- Main Logic ---

def main():
    writer = FileManager(OUTPUT_DIR, "math_dataset_mix")

    # 1. GSM8K (Highest Priority for your 1.8 loss goal)
    print("\nParsing GSM8K (Chain of Thought)...")
    try:
        gsm = load_dataset("gsm8k", "main", split="train")
        for row in tqdm(gsm, desc="GSM8K"):
            writer.write(parse_gsm8k(row))
    except Exception as e:
        print(f"Error loading GSM8K: {e}")

    # 2. MathQA (Local or HF)
    print("\nParsing MathQA...")
    try:
        # Checking for local file first as per your previous script
        json_path = os.path.join("mathQA", "train.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for row in tqdm(data, desc="MathQA (Local)"):
                    writer.write(parse_mathqa(row))
        else:
            # Fallback to HF if local not found
            mathqa = load_dataset("math_qa", split="train")
            for row in tqdm(mathqa, desc="MathQA (HF)"):
                writer.write(parse_mathqa(row))
    except Exception as e:
        print(f"Error loading MathQA: {e}")

    # 3. ASDiv
    print("\nParsing ASDiv...")
    try:
        asdiv = load_dataset("EleutherAI/asdiv", split="validation")
        for row in tqdm(asdiv, desc="ASDiv"):
            writer.write(parse_asdiv(row))
    except Exception as e:
        print(f"Error loading ASDiv: {e}")

    # 4. AutoMathText (Streaming 10k rows)
    print(f"\nParsing AutoMathText (Max {MAX_ROWS_AUTOMATH})...")
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
    print(f"\nSuccess! Processed data in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
