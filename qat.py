import os, itertools, torch
from pathlib import Path
from transformers import AutoTokenizer

import model
from model import Transformer

# =========================
# config
# =========================
model.config["vocab_size"] = 8192
model.config["block_size"] = 1024

batch_size = 8
block_size = 1024
grad_accum_steps = 4
qat_steps = 20000
lr = 5e-6
beta1, beta2 = 0.99, 0.95
weight_decay = 0.02
max_grad_norm = 0.3
group_size = 16  
device = "cuda" if torch.cuda.is_available() else "cpu"

# CHANGE: Set dtype to float32
dtype = torch.float32 

fp16_checkpoint = "/content/floppyLLM/checkpoints/1pp88e_inference_fp16.pt"
data_dir = "synth_2"

# =========================
# fake int4 with stochastic rounding + learnable per-channel scale
# =========================
class FakeInt4Weight(torch.nn.Module):
    def __init__(self, group_size=16, learnable_scale=True, eps=1e-4): # Slightly higher eps for stability
        super().__init__()
        self.group_size = group_size
        self.learnable_scale = learnable_scale
        self.scale_param = None
        self.eps = eps 

    def forward(self, w):
        shape = w.shape
        w_flat = w.view(-1, self.group_size)
        
        # 1. Calculate Scale
        scale = w_flat.abs().max(dim=1, keepdim=True)[0].clamp(min=self.eps)
        if self.learnable_scale:
            if self.scale_param is None or self.scale_param.shape[0] != scale.shape[0]:
                init_val = torch.log(torch.exp(scale) - 1)
                self.scale_param = torch.nn.Parameter(init_val)
            scale = torch.nn.functional.softplus(self.scale_param)

        # 2. Quantize (Deterministic is better for this small model)
        # Normalize to [-1, 1], scale to [-7, 7], then round
        w_norm = w_flat / scale
        q = torch.clamp(torch.round(w_norm * 7), -8, 7)
        
        # 3. Dequantize
        w_quant = (q / 7 * scale).view(shape)

        # 4. THE STE TRICK (Crucial)
        # This makes the forward pass use 'w_quant' 
        # but the backward pass skip the rounding and act on 'w' directly.
        return w + (w_quant - w).detach()

def enable_int4_qat(model, group_size=16):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.fake_q = FakeInt4Weight(group_size, learnable_scale=True)
            def qat_forward(x, m=m):
                # Ensure input x is float32
                wq = m.fake_q(m.weight)
                return torch.nn.functional.linear(x.float(), wq, m.bias)
            m.forward = qat_forward

# =========================
# minimal int4 export (unchanged)
# =========================
def pack_int4(q: torch.Tensor) -> torch.Tensor:
    q = (q + 8).to(torch.uint8) 
    if q.shape[1] % 2 != 0:
        q = torch.cat([q, torch.zeros(q.shape[0], 1, dtype=torch.uint8, device=q.device)], dim=1)
    return (q[:, 0::2] | (q[:, 1::2] << 4))

def export_int4_minimal(model, group_size=16):
    sd = {}
    for k, v in model.state_dict().items():
        if k == "lm_head.weight": continue 
        if torch.is_complex(v): continue
        if "bias" in k or "norm" in k:
            sd[k] = v.half()
            continue
        if k == "transformer.wte.weight":
            w = v.float()
            scale = w.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-2)
            q = torch.round(w / scale * 7).clamp(-8,7).to(torch.int8)
            sd[k + ".int4"] = pack_int4(q)
            sd[k + ".scale"] = scale.half()
            sd["lm_head.weight.int4"] = sd[k + ".int4"]
            sd["lm_head.weight.scale"] = sd[k + ".scale"]
            continue
        if v.ndim >= 2 and v.dtype.is_floating_point:
            w = v.float().view(-1, group_size)
            scale = w.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-2)
            q = torch.round(w / scale * 7).clamp(-8,7).to(torch.int8)
            sd[k + ".int4"] = pack_int4(q)
            sd[k + ".scale"] = scale.half()
            continue
        sd[k] = v.half() if v.dtype.is_floating_point else v
    return sd

# =========================
# data loader (tokens cast to long)
# =========================
def _load_data_shard(file: Path):
    with file.open("rb") as f:
        f.read(256*4)
        tokens = torch.frombuffer(f.read(), dtype=torch.uint16)
    return tokens.long()

def create_data_generator(pattern, batch_size, block_size):
    files = list(Path(".").glob(pattern))
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

train_gen = create_data_generator(f"data/{data_dir}/{data_dir}_train_*.bin",
                                  batch_size, block_size)

# =========================
# model load & upcast
# =========================
m = Transformer().to(device)

# CHANGE: Force model to float32 immediately
m.float() 

state_dict = torch.load(fp16_checkpoint, map_location="cpu")
m.load_state_dict(state_dict, strict=False)
m.train()
print(f"model params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M (FP32 Mode)")

# freeze norms
for n, p in m.named_parameters():
    if "norm" in n.lower():
        p.requires_grad = False

# enable qat
enable_int4_qat(m, group_size=group_size)

# split optimizer
scale_params = [p for n,p in m.named_parameters() if "scale_param" in n]
other_params = [p for n,p in m.named_parameters() if "scale_param" not in n]

opt = torch.optim.AdamW([
    {'params': scale_params, 'lr': 1e-6},
    {'params': other_params}
], lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

# REMOVED: Autocast context and GradScaler

# =========================
# qat loop (FP32 Version)
# =========================
for step in range(qat_steps):
    opt.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for _ in range(grad_accum_steps):
        xb, yb = next(train_gen)
        
        # NO Autocast here - we want raw FP32
        _, loss = m(xb, yb)
        loss = loss / grad_accum_steps
        
        # NO scaler here - standard backward
        loss.backward()
        loss_accum += loss.item()
    
    # Clip gradients before stepping
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
    
    # Standard optimizer step
    opt.step()
    
    if step % 100 == 0:
        print(f"qat step {step} | loss {loss_accum:.4f}")

# =========================
# export int4
# =========================
os.makedirs("checkpoints", exist_ok=True)
out_path = "checkpoints/qat_int4.pt"
torch.save(export_int4_minimal(m, group_size=group_size), out_path)
print(f"saved int4 weights → {out_path}")
