import os, itertools, torch
from pathlib import Path
from transformers import AutoTokenizer

import model
from model import Transformer

# =========================
# 1. Config & Setup
# =========================
model.config["vocab_size"] = 8192
model.config["block_size"] = 1024

batch_size = 16      # Context 1024 is heavy, keep batch small
block_size = 1024   
grad_accum_steps = 8 
qat_steps = 100
lr = 4e-5           # Moderate recovery LR
max_grad_norm = 1.0 
group_size = 64  
device = "cuda" if torch.cuda.is_available() else "cpu"

# MAKE SURE THIS PATH IS 100% CORRECT
fp16_checkpoint = "/content/floppyLLM/checkpoints/C19KBM_7000.pt"
tokenizer = AutoTokenizer.from_pretrained("nano_1k")

data_dir = "synth_2"

# =========================
# 2. Fake Int4 Module
# =========================
class FakeInt4Weight(torch.nn.Module):
    def __init__(self, group_size=64, eps=1e-4):
        super().__init__()
        self.group_size = group_size
        self.scale_param = None
        self.eps = eps 

    def forward(self, w):
        w = w.float()
        w_flat = w.view(-1, self.group_size)
        scale = w_flat.abs().max(dim=1, keepdim=True)[0].clamp(min=self.eps)
        if self.scale_param is None or self.scale_param.shape[0] != scale.shape[0]:
            init_val = torch.log(torch.exp(scale) - 1 + 1e-6)
            self.scale_param = torch.nn.Parameter(init_val)
        
        scale = torch.nn.functional.softplus(self.scale_param)
        w_norm = w_flat / scale
        q = torch.clamp(torch.round(w_norm * 7), -8, 7)
        w_quant = (q / 7 * scale).view(w.shape)
        # Straight-Through Estimator
        return w + (w_quant - w).detach()

def enable_int4_qat(model, group_size=64):
    for n, m in model.named_modules():
        # EXCLUDE Embeddings and Head - Keep them high-precision
        if any(x in n for x in ["wte", "wpe", "lm_head"]):
            print(f"Excluding from QAT: {n}")
            continue
            
        if isinstance(m, torch.nn.Linear):
            m.fake_q = FakeInt4Weight(group_size)
            def qat_forward(x, m=m):
                wq = m.fake_q(m.weight)
                return torch.nn.functional.linear(x, wq, m.bias)
            m.forward = qat_forward

# =========================
# 3. Proper Weight Loading
# =========================
m = Transformer().to(device)
m.float() 

print(f"Attempting to load: {fp16_checkpoint}")
checkpoint = torch.load(fp16_checkpoint, map_location=device)

# --- THE FIX: Strip 'compiled' prefix if it exists ---
state_dict = checkpoint.get('model', checkpoint) # Handle full ckpt or just state_dict
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

# Force strict=True to find out EXACTLY why it's failing if it does
try:
    m.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully with strict=True")
except RuntimeError as e:
    print(f"Loading failed! Error: {e}")
    print("Attempting fix with strict=False...")
    m.load_state_dict(state_dict, strict=False)

# DIAGNOSTIC: Check if weights are random or trained
# (Random weights are usually very small, e.g., 0.02. Trained are larger)
print(f"Weight Check (wte): {m.transformer.wte.weight[0, :5].tolist()}")

# =========================
# 4. QAT Logic
# =========================
enable_int4_qat(m, group_size=group_size)

m.train()
# Calibration
print("Calibrating...")
with torch.no_grad():
    # We'll use a manual tiny batch for calibration to ensure scales exist
    dummy_x = torch.randint(0, 8192, (1, 1024), device=device)
    _ = m(dummy_x)

scale_params = [p for n, p in m.named_parameters() if "scale_param" in n]
other_params = [p for n, p in m.named_parameters() if "scale_param" not in n]
opt = torch.optim.AdamW([
    {'params': scale_params, 'lr': lr},
    {'params': other_params, 'lr': lr}
], betas=(0.9, 0.95), weight_decay=0.01)

# =========================
# 5. Data & Loop
# =========================
def _load_data_shard(file: Path):
    with file.open("rb") as f:
        f.read(256*4)
        tokens = torch.frombuffer(f.read(), dtype=torch.uint16)
    return tokens.long()

def create_data_generator(pattern, batch_size, block_size):
    files = sorted(list(Path(".").glob(pattern)))
    it = itertools.cycle(files)
    tokens, pos = None, 0
    while True:
        if tokens is None or pos + batch_size*block_size + 1 >= len(tokens):
            tokens = _load_data_shard(next(it))
            pos = 0
        x = torch.stack([tokens[pos+i*block_size:pos+(i+1)*block_size] for i in range(batch_size)])
        y = torch.stack([tokens[pos+i*block_size+1:pos+(i+1)*block_size+1] for i in range(batch_size)])
        pos += batch_size*block_size
        yield x.to(device), y.to(device)

train_gen = create_data_generator(f"data/{data_dir}/{data_dir}_train_*.bin", batch_size, block_size)

@torch.no_grad()
def generate_sample(model, steps=128):
    model.eval()
    x = torch.tensor([[tokenizer.bos_token_id or 0]], device=device)
    try:
        # Use simple greedy for speed during QAT check
        y = model.generate(x, max_new_tokens=steps, temperature=0.7, top_k=40)
        out = tokenizer.decode(y[0].tolist())
    except: out = "Gen error"
    model.train()
    return out

print("Starting QAT...")
for step in range(qat_steps):
    opt.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for _ in range(grad_accum_steps):
        xb, yb = next(train_gen)
        _, loss = m(xb, yb)
        loss = loss / grad_accum_steps
        loss.backward()
        loss_accum += loss.item()
    
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
    opt.step()
    
    if step % 10 == 0: # Print every step for now so you can see if it drops
        print(f"step {step} | loss {loss_accum:.4f}")
        
    if step % 100 == 0:
        print(f"\nSAMPLE AT STEP {step}:\n{generate_sample(m)}\n")
