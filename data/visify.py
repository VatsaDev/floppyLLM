import os
import json
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "moondream/synthcat" 
OUTPUT_DIR = "synthcat"
MAX_ROWS = 50_000  # good enough

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"Loading {DATASET_NAME} (streaming mode)...")
    
    # Load dataset in streaming mode (no massive download upfront)
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    row_count = 0
    
    print(f"Processing first {MAX_ROWS} rows...")
    print(f"Saving images and json pairs to '{OUTPUT_DIR}/' ...")
    
    # Iterate through the dataset
    for row in tqdm(ds, total=MAX_ROWS, unit="img"):
        if row_count >= MAX_ROWS:
            break
        
        # 1. Handle the Image
        # The 'image' column is usually a PIL object in streaming mode
        if 'image' not in row or row['image'] is None:
            continue
            
        # Convert image to RGB to avoid PNG/RGBA issues
        image = row['image'].convert("RGB")
        
        # Create a deterministic filename
        base_filename = f"doc_{row_count:08d}"
        img_path = os.path.join(OUTPUT_DIR, f"{base_filename}.jpg")
        
        # Save Image
        image.save(img_path, "JPEG", quality=90)
        
        # 2. Handle the Text/Elements
        # Your screenshot shows an 'elements' column or 'sequence'
        # We save this as a JSON file with the SAME basename
        json_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
        
        # Extract the data (keeping it raw as requested)
        # We try to grab 'elements', 'text', or just the whole row minus image
        if 'elements' in row:
            text_data = row['elements']
        elif 'text' in row:
            text_data = {"text": row['text']}
        else:
            # Fallback: Save everything except the PIL image object
            text_data = {k: v for k, v in row.items() if k != 'image'}

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(text_data, f, ensure_ascii=False)
            
        row_count += 1

    print(f"\nDone! Processed {row_count} pairs.")
    print(f"Images and JSONs saved to: {OUTPUT_DIR}/")
    print("You can now run the 'shard_vision_dataset.py' script.")

if __name__ == "__main__":
    main()

