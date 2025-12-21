import os, itertools, torch
from pathlib import Path
from transformers import AutoTokenizer

import model
from model import Transformer

# =========================
# 1. Config & Setup
# =========================
# Ensure these match your 2.27M "pencil" model config exactly
model.config["vocab_size"] = 1024 
model.config["n_embd"] = 64
model.config["n_layer"] = 40
model.config["n_head"] = 2
model.config["block_size"] = 1024

batch_size = 16      
block_size = 1024   
grad_accum_steps = 2
qat_steps = 1000    # Adjust based on your patience
lr = 5e-5           
max_grad_norm = 1.0 
group_size = 64     # Standard group size
device = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint paths
fp16_checkpoint = "/content/floppyLLM/checkpoints/UhfF5V_0.pt"
tokenizer = AutoTokenizer.from_pretrained("nano_1k") # Your new 1k tokenizer
data_dir = "synth_3"

# =========================
# 2. Fake Int4 Module (STE + Learnable Scales)
# =========================
class FakeInt4Weight(torch.nn.Module):
    def __init__(self, group_size=64, eps=1e-4):
        super().__init__()
        self.group_size = group_size
        self.scale_param = None
        self.eps = eps 

    def forward(self, w):
        w = w.float() # Ensure FP32 math
        shape = w.shape
        w_flat = w.view(-1, self.group_size)
        
        # Initialize scale on first forward pass
        scale = w_flat.abs().max(dim=1, keepdim=True)[0].clamp(min=self.eps)
        if self.scale_param is None or self.scale_param.shape[0] != scale.shape[0]:
            # Use softplus inverse for initialization
            init_val = torch.log(torch.exp(scale) - 1 + 1e-6)
            self.scale_param = torch.nn.Parameter(init_val)
        
        scale = torch.nn.functional.softplus(self.scale_param)
        
        # Quantize to 4-bit range [-8, 7]
        w_norm = w_flat / scale
        q = torch.clamp(torch.round(w_norm * 7), -8, 7)
        w_quant = (q / 7 * scale).view(shape)
        
        # Straight-Through Estimator (STE)
        return w + (w_quant - w).detach()

def enable_int4_qat(model, group_size=64):
    param_to_quantizer = {}
    for n, m in model.named_modules():
        # We quantize EVERYTHING but ensure Tied Weights share the same scale
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            param_id = id(m.weight)
            if param_id not in param_to_quantizer:
                param_to_quantizer[param_id] = FakeInt4Weight(group_size)
            
            m.fake_q = param_to_quantizer[param_id]
            
            if isinstance(m, torch.nn.Linear):
                def qat_linear_forward(x, m=m):
                    wq = m.fake_q(m.weight)
                    return torch.nn.functional.linear(x, wq, m.bias)
                m.forward = qat_linear_forward
            else: # Embedding
                def qat_embed_forward(x, m=m):
                    wq = m.fake_q(m.weight)
                    return torch.nn.functional.embedding(x, wq, m.padding_idx, m.max_norm, 
                                                        m.norm_type, m.scale_grad_by_freq, m.sparse)
                m.forward = qat_embed_forward

# =========================
# 3. Packing & Clean Export
# =========================
def pack_int4(q):
    # Shift signed [-8, 7] to unsigned [0, 15]
    q = (q + 8).to(torch.uint8) 
    if q.shape[1] % 2 != 0:
        q = torch.cat([q, torch.zeros(q.shape[0], 1, dtype=torch.uint8, device=q.device)], dim=1)
    return (q[:, 0::2] | (q[:, 1::2] << 4))

import numpy as np

def export_int4_blob(model, group_size, path, target_vocab=1024):
    if hasattr(model, '_orig_mod'): model = model._orig_mod
    current_sd = model.state_dict()
    
    # 1. Prepare to store raw bytes
    weight_bytes = bytearray()
    scale_bytes = bytearray()
    
    # 2. Prepare a small metadata dict for Norms, Biases, and Shapes
    metadata = {
        "group_size": group_size,
        "weight_order": [], 
        "high_prec_params": {} 
    }

    for k in sorted(current_sd.keys()):
        v = current_sd[k]
        if any(x in k for x in ["scale_param", "lm_head.weight"]): continue
        
        # Heavy Weights (Linear & Embedding)
        if v.ndim >= 2 and v.dtype.is_floating_point:
            if "wte.weight" in k: v = v[:target_vocab, :].clone()
            
            # Quantize
            w_flat = v.float().view(-1, group_size)
            scale_name = k.replace(".weight", ".fake_q.scale_param")
            scale = torch.nn.functional.softplus(current_sd[scale_name]) if scale_name in current_sd else w_flat.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-2)
            
            # Pack
            q = torch.round(w_flat / scale * 7).clamp(-8, 7).to(torch.int8)
            q_u = (q + 8).cpu().numpy().astype(np.uint8)
            packed = (q_u[0::2] | (q_u[1::2] << 4)).flatten()
            
            # Add to blob
            weight_bytes.extend(packed.tobytes())
            scale_bytes.extend(scale.half().cpu().numpy().tobytes())
            
            metadata["weight_order"].append((k, v.shape))
        else:
            # Norms, Biases, WPE stay in high precision metadata
            metadata["high_prec_params"][k] = v.half().cpu()

    # Write the Big Blob
    with open(path, "wb") as f:
        f.write(weight_bytes)
        f.write(scale_bytes)
        
    # Write the Metadata (Small .pt file)
    torch.save(metadata, path + ".meta")
    print(f"✅ Blob: {os.path.getsize(path)/1e6:.2f}MB | Meta: {os.path.getsize(path+'.meta')/1e6:.2f}MB")

# =========================
# 4. Training Utilities
# =========================
def create_gen(pattern, bs, blk):
    files = sorted(list(Path(".").glob(pattern)))
    it = itertools.cycle(files)
    tokens, pos = None, 0
    while True:
        if tokens is None or pos + bs*blk + 1 >= len(tokens):
            with next(it).open("rb") as f:
                f.read(256*4) # skip header
                tokens = torch.frombuffer(f.read(), dtype=torch.uint16).long()
            pos = 0
        yield torch.stack([tokens[pos+i*blk:pos+(i+1)*blk] for i in range(bs)]).to(device), \
              torch.stack([tokens[pos+i*blk+1:pos+(i+1)*blk+1] for i in range(bs)]).to(device)
        pos += bs*blk

@torch.no_grad()
def sample(model):
    model.eval()
    x = torch.tensor([[tokenizer.bos_token_id or 0]], device=device)
    # Uses the model's internal generate method
    y = model.generate(x, max_new_tokens=128, temperature=0.8, top_k=40)
    print(f"\nSAMPLE:\n{tokenizer.decode(y[0].tolist())}\n")
    model.train()

# =========================
# 5. The Main Loop
# =========================
m = Transformer().to(device).float() # Start in full FP32

# Load and strip 'compiled' prefix
print(f"Loading {fp16_checkpoint}...")
ckpt = torch.load(fp16_checkpoint, map_location=device)
state_dict = ckpt.get('model', ckpt)
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
m.load_state_dict(state_dict, strict=False)

# Setup QAT & Calibrate
enable_int4_qat(m, group_size=group_size)
m.train()
print("Calibrating...")
train_gen = create_gen(f"data/{data_dir}/{data_dir}_train_*.bin", batch_size, block_size)
with torch.no_grad():
    xb, yb = next(train_gen)
    _ = m(xb)

# Compile AFTER patching
if hasattr(torch, 'compile'): 
    print("Compiling...")
    m = torch.compile(m)

# Optimizer (only on weights and new scale params)
scale_params = [p for n, p in m.named_parameters() if "scale_param" in n]
other_params = [p for n, p in m.named_parameters() if "scale_param" not in n]
opt = torch.optim.AdamW([{'params': scale_params}, {'params': other_params}], lr=lr)

print(f"Starting QAT Training on {device}...")
for step in range(qat_steps):
    opt.zero_grad(set_to_none=True)
    loss_acc = 0
    
    for _ in range(grad_accum_steps):
        xb, yb = next(train_gen)
        _, loss = m(xb, yb)
        (loss/grad_accum_steps).backward()
        loss_acc += loss.item()
        
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
    opt.step()
    
    if step % 10 == 0: print(f"step {step} | loss {loss_acc/grad_accum_steps:.4f}")
    if step % 100 == 0: sample(m)

# FINAL CLEAN SAVE
export_int4_blob(m, group_size, "checkpoints/qat_final_tiny.pt", target_vocab=1024)
