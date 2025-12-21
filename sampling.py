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
from transformers import AutoTokenizer
from model import Transformer, config

def load_blob_inference(blob_path, meta_path, model_instance):
    device = next(model_instance.parameters()).device
    meta = torch.load(meta_path, map_location='cpu', weights_only=False)
    high_prec = meta["high_prec_params"]
    model_sd = model_instance.state_dict()

    # 1. Inject High-Precision (Embeddings/Norms)
    for mk, weight in high_prec.items():
        clean_mk = mk.replace('_orig_mod.', '')
        for rk in model_sd.keys():
            if clean_mk == rk.replace('_orig_mod.', ''):
                model_sd[rk].copy_(weight.to(device))
                break

    # 2. Unpack Logic Blob
    with open(blob_path, "rb") as f: full_blob = f.read()
    group_size = meta["group_size"]
    total_w_bytes = sum(((s.numel() + 1) // 2) for _, s in meta["weight_order"])
    weight_data = np.frombuffer(full_blob[:total_w_bytes], dtype=np.uint8)
    scale_data = np.frombuffer(full_blob[total_w_bytes:], dtype=np.float16)
    
    w_ptr, s_ptr = 0, 0
    for name, shape in meta["weight_order"]:
        num_el = shape.numel()
        bytes_needed = (num_el + 1) // 2
        chunk = weight_data[w_ptr : w_ptr + bytes_needed]
        
        low, high = chunk & 0x0F, chunk >> 4
        unpacked = np.stack([low, high], axis=1).flatten()[:num_el]
        
        scales = scale_data[s_ptr : s_ptr + (num_el // group_size)].reshape(-1, 1)
        w_t = (torch.from_numpy(unpacked).float() - 8).to(device)
        s_t = torch.from_numpy(scales).to(device)
        
        # FIX: The / 7.0 math MUST match your FakeInt4Weight training math!
        final_w = ((w_t.view(-1, group_size) / 7.0) * s_t).view(shape)
        
        target_key = name if name in model_sd else "_orig_mod." + name
        if target_key in model_sd: model_sd[target_key].copy_(final_w)
        
        w_ptr += bytes_needed
        s_ptr += (num_el // group_size)

    model_instance.lm_head.weight = model_instance.transformer.wte.weight
    strength = model_instance.transformer.wte.weight.abs().mean().item()
    print(f"✅ Ready. Embedding Strength: {strength:.4f}")
    return model_instance

# Setup and Run
config["vocab_size"], config["n_embd"], config["n_layer"] = 1024, 64, 40
tokenizer = AutoTokenizer.from_pretrained("nano_1k")
m = Transformer().to(device)
m = load_blob_inference("checkpoints/qat_final_tiny.pt", "checkpoints/qat_final_tiny.pt.meta", m)
m.eval()

# Generate
prompt = "In the future,"
idx = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
with torch.no_grad():
    out = m.generate(idx, max_new_tokens=100, temperature=0.7, top_k=40)
print(tokenizer.decode(out[0].tolist()))
