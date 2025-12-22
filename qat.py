import os, itertools, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer
import model
from model import Transformer

# =========================
# 1. Config & Setup
# =========================
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

batch_size = 16
block_size = 1024
grad_accum_steps = 2
qat_steps = 2000  # Increased for better convergence
lr = 5e-5  # Lower LR for stability
max_grad_norm = 0.8 
device = "cuda" if torch.cuda.is_available() else "cpu"

fp16_checkpoint = "/content/floppyLLM/checkpoints/epz3VR_14000.pt"
tokenizer = AutoTokenizer.from_pretrained("nano_1k")
data_dir = "synth_3"

# =========================
# 2. FIXED BitNet Module
# =========================
class BitLinear(nn.Module):
    """ Ternary Linear Layer with proper scale handling """
    def __init__(self, in_features, out_features, original_weight, bias=None):
        super().__init__()
        device = original_weight.device
        self.weight = nn.Parameter(original_weight.float().clone())
        self.bias = nn.Parameter(bias.float().clone()) if bias is not None else None
        
        # Store scale as learnable parameter
        init_scale = original_weight.abs().mean().clamp(min=1e-5)
        self.weight_scale = nn.Parameter(torch.tensor([init_scale], device=device))
        
        # Track quantization strength (0=FP32, 1=full quantization)
        self.register_buffer('quant_strength', torch.tensor(0.0, device=device))

    def forward(self, x):
        w = self.weight
        
        # Gradual quantization with learnable scale
        w_norm = w / self.weight_scale.clamp(min=1e-5)
        w_quant = torch.round(torch.clamp(w_norm, -1, 1))
        
        # Mix quantized and original based on strength
        w_mixed = w + self.quant_strength * (w_quant * self.weight_scale - w).detach()

        # Simpler activation quantization (8-bit)
        x_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_quant = torch.round(torch.clamp(x / x_scale * 127, -128, 127))
        x_mixed = x + self.quant_strength * (x_quant * x_scale / 127 - x).detach()

        return F.linear(x_mixed, w_mixed, self.bias)

def convert_to_bitnet(model):
    """ Swaps Linear layers for BitLinear """
    bitnet_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in name:
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Ensure bias handling
            bias_data = m.bias.data if m.bias is not None else None
            new_layer = BitLinear(m.in_features, m.out_features, m.weight.data, bias_data)
            
            setattr(parent, name.rsplit('.', 1)[-1], new_layer)
            bitnet_layers.append(new_layer)
    print(f"✅ Converted {len(bitnet_layers)} layers to BitNet")
    return bitnet_layers

# =========================
# 3. FIXED Export with Scale Storage
# =========================
def pack_ternary_base3(weights):
    """ Packs 5 ternary values into one byte """
    w = (weights.flatten().cpu().numpy().astype(np.int8) + 1).astype(np.uint64)
    
    padding = (5 - (len(w) % 5)) % 5
    if padding > 0:
        w = np.concatenate([w, np.zeros(padding, dtype=np.uint64)])
    
    packed = (w[0::5] + w[1::5]*3 + w[2::5]*9 + w[3::5]*27 + w[4::5]*81).astype(np.uint8)
    return packed.tobytes()

def unpack_ternary_base3(data, target_shape):
    """ Unpacks base-3 encoded ternary weights """
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    
    # Decode base-3
    w = np.zeros(len(arr) * 5, dtype=np.int8)
    w[0::5] = arr % 3
    w[1::5] = (arr // 3) % 3
    w[2::5] = (arr // 9) % 3
    w[3::5] = (arr // 27) % 3
    w[4::5] = (arr // 81) % 3
    
    # Map back to {-1, 0, 1}
    w = w - 1
    return torch.from_numpy(w[:np.prod(target_shape)].reshape(target_shape))

def export_bitnet_optimal(model, path, target_vocab=1024):
    print(f"\nExporting BitNet to {path}...")
    if hasattr(model, '_orig_mod'): 
        model = model._orig_mod
    sd = model.state_dict()
    
    weight_blob = bytearray()
    metadata = {"weight_order": [], "high_prec": {}, "scales": {}}

    for k in sorted(sd.keys()):
        v = sd[k]
        
        # Skip tied lm_head
        if "lm_head.weight" in k:
            continue
            
        # High-precision layers
        if any(x in k for x in ["wte", "wpe", "bias", "ln_"]):
            val = v[:target_vocab].clone() if "wte" in k else v.clone()
            metadata["high_prec"][k] = val.half().cpu()
            print(f"  [FP16] {k}: {val.shape}")
            
        # Quantized weights - store scale separately
        elif ".weight" in k and "weight_scale" not in k:
            scale_key = k.replace(".weight", ".weight_scale")
            scale = sd.get(scale_key, torch.tensor([v.abs().mean().item()]))
            if scale.numel() == 1:
                scale = scale.item()
            else:
                scale = scale.mean().item()
            
            # Quantize using stored scale
            w_q = torch.round(torch.clamp(v / (scale + 1e-5), -1, 1))
            
            weight_blob.extend(pack_ternary_base3(w_q))
            metadata["weight_order"].append((k, tuple(v.shape)))
            metadata["scales"][k] = scale
            print(f"  [1.58b] {k}: {v.shape}, scale={scale:.6f}")

    # Write files
    with open(path, "wb") as f: 
        f.write(weight_blob)
    torch.save(metadata, path + ".meta")
    
    total_size = (os.path.getsize(path) + os.path.getsize(path+".meta")) / 1e6
    print(f"✅ Export complete! Size: {total_size:.2f} MB")
    return total_size

# =========================
# 4. Training utilities
# =========================
def create_gen(pattern, bs, blk):
    files = sorted(list(Path(".").glob(pattern)))
    if not files:
        raise ValueError(f"No files found matching {pattern}")
    it = itertools.cycle(files)
    tokens, pos = None, 0
    while True:
        if tokens is None or pos + bs*blk + 1 >= len(tokens):
            with next(it).open("rb") as f:
                f.read(256*4)
                tokens = torch.frombuffer(f.read(), dtype=torch.uint16).long()
            pos = 0
        yield torch.stack([tokens[pos+i*blk:pos+(i+1)*blk] for i in range(bs)]).to(device), \
              torch.stack([tokens[pos+i*blk+1:pos+(i+1)*blk+1] for i in range(bs)]).to(device)
        pos += bs*blk

@torch.no_grad()
def sample(model, temp=0.8, max_tokens=128):
    model.eval()
    x = torch.tensor([[tokenizer.bos_token_id or 0]], device=device)
    y = model.generate(x, max_new_tokens=max_tokens, temperature=temp, top_k=40)
    text = tokenizer.decode(y[0].tolist())
    print(f"\n{'='*60}\nSAMPLE:\n{text}\n{'='*60}\n")
    model.train()
    return text

# =========================
# 5. MAIN with Gradual Quantization
# =========================
def main():
    m = Transformer().to(device).float()

    print(f"Loading checkpoint: {fp16_checkpoint}")
    ckpt = torch.load(fp16_checkpoint, map_location=device)
    sd = ckpt.get('model', ckpt)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)

    # Convert to BitNet
    bitnet_layers = convert_to_bitnet(m)

    # DON'T compile - it breaks with dynamic quantization strength
    # if hasattr(torch, 'compile'):
    #     print("Compiling model...")
    #     m = torch.compile(m)

    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)
    train_gen = create_gen(f"data/{data_dir}/{data_dir}_train_*.bin", batch_size, block_size)

    print("\n" + "="*60)
    print("Starting Quantization-Aware Training")
    print("="*60)
    
    # Sample before QAT
    print("\nBEFORE QAT:")
    sample(m)

    # Gradual quantization schedule
    warmup_steps = 100
    for step in range(qat_steps):
        # Gradually increase quantization strength
        quant_progress = min(1.0, max(0.0, (step - warmup_steps) / (qat_steps - warmup_steps)))
        for layer in bitnet_layers:
            layer.quant_strength.fill_(quant_progress)
        
        opt.zero_grad(set_to_none=True)
        loss_acc = 0
        
        for _ in range(grad_accum_steps):
            xb, yb = next(train_gen)
            _, loss = m(xb, yb)
            (loss / grad_accum_steps).backward()
            loss_acc += loss.item()
        
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
        opt.step()
        
        if step % 10 == 0:
            print(f"Step {step:3d} | Loss: {loss_acc/grad_accum_steps:.4f} | "
                  f"Quant: {quant_progress*100:.1f}%")
        
        if step % 100 == 0 and step > 0:
            sample(m)

    # Final sample at full quantization
    print("\nAFTER QAT (Full Quantization):")
    sample(m)

    # Export
    export_path = "checkpoints/bitnet_ultra_tiny.bin"
    export_bitnet_optimal(m, export_path)
    
    print("\n" + "="*60)
    print("QAT Complete! Test the exported model with:")
    print("  python inference_bitnet.py")
    print("="*60)

if __name__ == "__main__":
    main()
