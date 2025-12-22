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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
import model
from model import Transformer

# Config must match training
config = {
    "n_embd": 64,
    "n_head": 2,
    "n_layer": 40,
    "dropout": 0.0,
    "vocab_size": 1024,
    "ctx_len": 1024,
    "bias": False,
}
model.config.update(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("nano_1k")

# =========================
# BitLinear for Inference
# =========================
class BitLinearInference(nn.Module):
    """ Inference-only BitLinear with fixed ternary weights """
    def __init__(self, in_features, out_features, ternary_weights, scale, bias=None):
        super().__init__()
        # Store weights as int8 for memory efficiency
        self.register_buffer('weight_ternary', ternary_weights.to(torch.int8))
        self.register_buffer('weight_scale', torch.tensor([scale]))
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x):
        # Reconstruct FP weights from ternary
        w = self.weight_ternary.float() * self.weight_scale
        
        # Simple 8-bit activation quantization for speed
        x_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_quant = torch.round(torch.clamp(x / x_scale * 127, -128, 127))
        x_final = x_quant * x_scale / 127
        
        return F.linear(x_final, w, self.bias)

# =========================
# Unpacking Functions
# =========================
def unpack_ternary_base3(data, target_shape):
    """ Unpacks base-3 encoded ternary weights """
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    
    # Decode base-3: each byte stores 5 ternary values
    total_vals = len(arr) * 5
    w = np.zeros(total_vals, dtype=np.int8)
    w[0::5] = arr % 3
    w[1::5] = (arr // 3) % 3
    w[2::5] = (arr // 9) % 3
    w[3::5] = (arr // 27) % 3
    w[4::5] = (arr // 81) % 3
    
    # Map {0,1,2} back to {-1,0,1}
    w = w - 1
    
    # Reshape and return
    num_elements = np.prod(target_shape)
    return torch.from_numpy(w[:num_elements].reshape(target_shape))

def load_bitnet_model(bin_path, meta_path):
    """ Load BitNet model from binary + metadata """
    print(f"\nLoading BitNet from {bin_path}...")
    
    # Load metadata
    metadata = torch.load(meta_path, map_location='cpu')
    
    # Create base model
    m = Transformer().to(device).float()
    state_dict = m.state_dict()
    
    # Load high-precision weights first
    print("\n--- Loading High-Precision Layers ---")
    for k, v in metadata["high_prec"].items():
        print(f"✅ {k}: {v.shape}")
        # Handle vocab size mismatch
        if "wte" in k and v.shape[0] < config["vocab_size"]:
            full_weight = state_dict[k].clone()
            full_weight[:v.shape[0]] = v.to(device).float()
            state_dict[k].copy_(full_weight)
        else:
            state_dict[k].copy_(v.to(device).float())
    
    # Load quantized weights and replace Linear layers
    print("\n--- Loading Quantized Weights ---")
    with open(bin_path, "rb") as f:
        weight_data = f.read()
    
    offset = 0
    replacements = {}  # Store layer replacements to apply after iteration
    
    for k, shape in metadata["weight_order"]:
        # Calculate size for this weight
        num_elements = np.prod(shape)
        num_bytes = (num_elements + 4) // 5  # 5 ternary values per byte
        
        # Extract and unpack
        chunk = weight_data[offset:offset + num_bytes]
        w_ternary = unpack_ternary_base3(chunk, shape)
        scale = metadata["scales"][k]
        
        print(f"✅ {k}: {shape}, scale={scale:.6f}")
        
        # Parse the key to find the layer
        # Example: "transformer.h.0.attn.c_attn.weight"
        parts = k.split('.')
        
        # Find the parent module and layer name
        if len(parts) >= 2 and parts[-1] == 'weight':
            layer_path = '.'.join(parts[:-1])  # Everything except .weight
            
            try:
                # Get the actual Linear layer
                orig_layer = m.get_submodule(layer_path)
                
                if isinstance(orig_layer, nn.Linear):
                    # Create replacement
                    bias = orig_layer.bias.data if orig_layer.bias is not None else None
                    new_layer = BitLinearInference(
                        orig_layer.in_features,
                        orig_layer.out_features,
                        w_ternary,
                        scale,
                        bias
                    ).to(device)
                    
                    # Store for later replacement
                    replacements[layer_path] = new_layer
            except Exception as e:
                print(f"  ⚠️  Warning: Could not replace {layer_path}: {e}")
        
        offset += num_bytes
    
    # Apply all replacements
    print("\n--- Applying Layer Replacements ---")
    for layer_path, new_layer in replacements.items():
        parts = layer_path.split('.')
        parent_path = '.'.join(parts[:-1])
        layer_name = parts[-1]
        
        if parent_path:
            parent = m.get_submodule(parent_path)
        else:
            parent = m
            
        setattr(parent, layer_name, new_layer)
        print(f"✅ Replaced {layer_path}")
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Replaced {len(replacements)} layers")
    print(f"   Total size: {(len(weight_data) + sum(t.numel()*2 for t in metadata['high_prec'].values())) / 1e6:.2f} MB")
    return m

# =========================
# Sampling
# =========================
@torch.no_grad()
def generate_text(model, prompt="", max_tokens=200, temperature=0.8, top_k=40):
    model.eval()
    
    # Encode prompt
    if prompt:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    else:
        tokens = torch.tensor([[tokenizer.bos_token_id or 0]], device=device)
    
    print(f"\nPrompt: '{prompt}'")
    print("="*60)
    
    # Generate
    output = model.generate(
        tokens, 
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    text = tokenizer.decode(output[0].tolist())
    print(text)
    print("="*60)
    return text

# =========================
# Main
# =========================
def main():
    print("="*60)
    print("BitNet 1.58b Inference")
    print("="*60)
    
    # Load model
    m = load_bitnet_model(
        "checkpoints/bitnet_ultra_tiny.bin",
        "checkpoints/bitnet_ultra_tiny.bin.meta"
    )
    
    # Test prompts
    prompts = [
        "",  # Unconditional generation
        "The quick brown",
        "Once upon a time",
        "In the year 2050",
    ]
    
    print("\n" + "="*60)
    print("GENERATING SAMPLES")
    print("="*60)
    
    for prompt in prompts:
        generate_text(m, prompt, max_tokens=150, temperature=0.8)
        print()

if __name__ == "__main__":
    main()
