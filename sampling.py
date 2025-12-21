# Methods list 

# Add temprature
# Greedy vs Multinomial option
# stop tokens option 
# Max_new_tokens, is an option, also should be forced to be below ctx len 
# add TopK and TopP
# add a streaming option 
# Get KV cache working
# prefix suffix options 
# add repetition penalty options
# expand this to more tokenizers

# Tail free sampling, - Tail free sampling (TFS) is a text generation technique that aims to reduce the impact of less likely tokens, which may be less relevant, less coherent, or nonsensical, on the output. Similar to Top-P it tries to determine the bulk of the most likely tokens dynamically. But TFS filters out logits based on the second derivative of their probabilities. Adding tokens is stopped after the sum of the second derivatives reaches the parameter z. In short: TFS looks how quickly the probabilities of the tokens decrease and cuts off the tail of unlikely tokens using the parameter z. Typical values for z are in the range of 0.9 to 0.95. A value of 1.0 would include all tokens, and thus disables the effect of TFS.

# Locally Typical Sampling - Locally typical sampling promotes the generation of contextually coherent and diverse text by sampling tokens that are typical or expected based on the surrounding context. By setting the parameter p between 0 and 1, you can control the balance between producing text that is locally coherent and diverse. A value closer to 1 will promote more contextually coherent tokens, while a value closer to 0 will promote more diverse tokens. A value equal to 1 disables locally typical sampling.

# Smooth Sampling / Quadratic Sampling
#    - This sampling method differs from the truncation samplers (Top K, Top P, Min P) in that it is doing something that is fundamentally different to the raw token scores.
#    - We are tweaking the logits using a quadratic transformation, based on each token score's distance from the top token (the transformation centers on the top logit.) The coefficient is decided by the "smoothing factor" value.
#    - This is hard to explain without looking at the visualization, but the idea is that we make the topmost tokens more evenly probable while reducing the probability of extremely unlikely tokens.
#    - Higher values will be more deterministic, but it doesn't work quite like lower temperature would, as the scores of extremely closely competing top tokens will barely change. So if the original probabilities were 50/50 on the top two tokens, they will likely remain that way with higher smoothing factor values.
#    - The idea is that this can be used as an "all in one" sampler by itself, or in tandem with other methods if desired.

# The muse https://github.com/the-crypt-keeper/the-muse
# add beam search 
# Drugs https://github.com/EGjoni/DRUGS 
# minimum bayes risk decoding [https://github.com/ZurichNLP/mbr](https://github.com/ZurichNLP/mbr?scrlybrkr=4c9c022b)

# grammars
# - https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
# - https://github.com/ggerganov/llama.cpp#constrained-output-with-grammars

# Mirostat
# - https://arxiv.org/abs/2007.14966

# EAGLE
# - https://arxiv.org/abs/2401.15077
# - https://github.com/SafeAILab/EAGLE

# Dynamic Temp
# - https://github.com/ggerganov/llama.cpp/issues/3483

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
import model # Assumes your model.py is in the directory

# =========================
# 1. Inference Config
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "/content/floppyLLM/checkpoints/qat_final_tiny.pt"
tokenizer_path = "nano_1k" 
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# =========================
# 2. The Unpacking Engine
# =========================

def load_from_raw_blob(blob_path, model_instance):
    device = next(model_instance.parameters()).device
    meta_path = blob_path + ".meta"
    
    print(f"Reading metadata from {meta_path}...")
    meta = torch.load(meta_path, map_location='cpu', weights_only=False)
    group_size = meta["group_size"]
    
    # 1. Load high-precision params (Norms/Biases)
    model_instance.load_state_dict(meta["high_prec_params"], strict=False)

    # 2. Load the Raw Blob into memory
    print(f"Reading raw bytes from {blob_path}...")
    with open(blob_path, "rb") as f:
        full_blob = f.read()

    # 3. Logic to slice the blob
    # First half of file is weights, second half is scales (based on your export)
    # We calculate the total weight bytes needed
    total_weight_bytes = sum(((s.numel() + 1) // 2) for _, s in meta["weight_order"])
    
    weight_data = np.frombuffer(full_blob[:total_weight_bytes], dtype=np.uint8)
    scale_data = np.frombuffer(full_blob[total_weight_bytes:], dtype=np.float16)
    
    w_ptr = 0
    s_ptr = 0
    sd = model_instance.state_dict()

    for name, shape in meta["weight_order"]:
        num_el = shape.numel()
        bytes_needed = (num_el + 1) // 2
        
        # Unpack the 4-bit pairs
        chunk = weight_data[w_ptr : w_ptr + bytes_needed]
        low = chunk & 0x0F
        high = chunk >> 4
        unpacked = np.stack([low, high], axis=1).flatten()[:num_el]
        
        # Dequantize
        num_groups = num_el // group_size
        scales = scale_data[s_ptr : s_ptr + num_groups].reshape(-1, 1)
        
        # Convert to torch and apply math
        weights = (torch.from_numpy(unpacked).float() - 8).to(device)
        scales_t = torch.from_numpy(scales).to(device)
        
        final_w = (weights.view(-1, group_size) * scales_t).view(shape)
        sd[name].copy_(final_w)
        
        w_ptr += bytes_needed
        s_ptr += num_groups

    # Re-tie head
    model_instance.lm_head.weight = model_instance.transformer.wte.weight
    print("✅ Blob Unpacked. Model is ready for sampling.")
    return model_instance

# =========================
# 3. Execution Logic
# =========================
from model import Transformer, config

# Match your pencil-thin config
config["vocab_size"] = 1024
config["n_embd"] = 64
config["n_layer"] = 40
config["n_head"] = 2

m = Transformer().to(device).half() # Inference in FP16 for speed
m = load_from_raw_blob(checkpoint_path, m)
m.eval()

# --- TEXT GENERATION ---
def generate(prompt, max_tokens=100):
    idx = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    
    print(f"\nPrompt: {prompt}\n" + "-"*30)
    
    # Use the model's built-in generate method
    with torch.no_grad():
        completion = m.generate(idx, max_new_tokens=max_tokens, temperature=0.8, top_k=40)
    
    return tokenizer.decode(completion[0].tolist())

# Run it!
print(generate("In the year 2025,"))
