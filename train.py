import os
import math
import time
import glob
import torch
import string
import random
import pickle
import numpy as np
import itertools 
from pathlib import Path

# tokenizer
from transformers import AutoTokenizer

# nice wandb style charts
import matplotlib
from matplotlib import pyplot as plt
 
import model
from model import Transformer

# hyperparams
batch_size = 8 #16
block_size = 1024 # ctx_len
eval_interval = 20
grad_accum_steps = 4 # basically microbatch

lr = 1e-3
min_lr = 1e-4

max_iters = 10001
eval_iters = 20
warmup_iters = 10 

train_losses_history = []
val_losses_history = []

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

max_grad_norm = 1.0 

ckpt_iter = 1000
resume = False
resume_checkpoint = "checkpoints/example.pt" 
data_dir = "synth_2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ptdtype = torch.float16
print(f"Using device: {device}")

# Initialize Scaler, needed for FP16 training on T4 to prevent underflow?
from torch.cuda.amp import GradScaler
scaler = GradScaler(enabled=True)

ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)

# ... [Matplotlib settings] ...
plt.style.use('default') 
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.spines.top'] = False 
matplotlib.rcParams['axes.spines.right'] = False 
matplotlib.rcParams['axes.facecolor'] = '#f0f0f0' 
matplotlib.rcParams['figure.facecolor'] = '#f0f0f0' 
matplotlib.rcParams['grid.alpha'] = 0.4 
matplotlib.rcParams['axes.titlesize'] = 12 
matplotlib.rcParams['axes.labelsize'] = 12 
matplotlib.rcParams['xtick.labelsize'] = 10 
matplotlib.rcParams['ytick.labelsize'] = 10 
matplotlib.rcParams['legend.fontsize'] = 10 
matplotlib.rcParams['axes.titlecolor'] = 'grey' 
matplotlib.rcParams['axes.labelcolor'] = 'grey' 
matplotlib.rcParams['xtick.color'] = 'grey' 
matplotlib.rcParams['ytick.color'] = 'grey' 
matplotlib.rcParams['legend.labelcolor'] = 'grey' 

# Run Name
characters = string.ascii_letters + string.digits 
run_name = ''.join(random.choice(characters) for i in range(6))
if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
if not os.path.exists('plots'): os.makedirs('plots')

# encoding 
tok = AutoTokenizer.from_pretrained("nano_8k")
encode = lambda s: tok.encode(s, add_special_tokens=True)
decode = lambda l: tok.decode(l)
vocab_size = tok.vocab_size
print(f"Using AutoTokenizer, vocab_size = {vocab_size}")

# data Loading
def _load_data_shard(file: Path):
    header = torch.from_file(str(file), shared=False, size=256, dtype=torch.int32)
    with file.open("rb") as f:
        header_bytes = f.read(256 * 4)
        header = torch.frombuffer(header_bytes, dtype=torch.int32)
        assert header[0].item() == 20240520, f"magic number mismatch in {file}"
        assert header[1].item() == 1, f"unsupported version in {file}"
        num_tokens = int(header[2].item()) 
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=(device=='cuda'))
        tokens_np = tokens.numpy() 
        nbytes_read = f.readinto(tokens_np.data) 
        expected_bytes = 2 * num_tokens
        assert nbytes_read == expected_bytes, f"read mismatch in {file}"
    return tokens

def create_data_generator(filename_pattern: str, batch_size: int, block_size: int, rank : int = 0, world_size : int = 1):
    files = sorted(glob.glob(filename_pattern))
    if not files: raise FileNotFoundError(f"No data files found: {filename_pattern}")
    print(f"Found {len(files)} data shards for pattern {filename_pattern}")
    file_iter = itertools.cycle([Path(file) for file in files])
    local_batch_size = batch_size // world_size
    current_tokens = None
    current_pos = 0

    while True:
        if current_tokens is None or current_pos + block_size * local_batch_size * world_size + 1 > len(current_tokens):
            next_file = next(file_iter)
            current_tokens = _load_data_shard(next_file)
            current_tokens = current_tokens.to(torch.int64) # Keep on CPU
            current_pos = 0
            if len(current_tokens) <= block_size + 1:
                 current_tokens = None 
                 continue 

        rank_start_offset = current_pos + rank * local_batch_size * block_size
        batch_x, batch_y = [], []
        for i in range(local_batch_size):
             start_idx = rank_start_offset + i * block_size
             end_idx = start_idx + block_size
             if end_idx + 1 > len(current_tokens):
                 current_tokens = None 
                 break 
             x = current_tokens[start_idx : end_idx]
             y = current_tokens[start_idx + 1 : end_idx + 1]
             batch_x.append(x)
             batch_y.append(y)
        if current_tokens is None: continue 

        inputs = torch.stack(batch_x)
        targets = torch.stack(batch_y)
        # Pin memory helps with speed, but non_blocking=True is key
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        current_pos += batch_size * block_size
        yield inputs, targets

train_data_pattern = os.path.join("data", data_dir, f"{data_dir}_train_*.bin")
val_data_pattern = os.path.join("data", data_dir, f"{data_dir}_val_*.bin")

def get_batch_from_shards(split, data_gens):
    if split == 'train': X, Y = next(data_gens['train'])
    else: X, Y = next(data_gens['val'])
    return X, Y

train_data_gen = create_data_generator(train_data_pattern, batch_size, block_size, rank=0, world_size=1)
val_data_gen = create_data_generator(val_data_pattern, batch_size, block_size, rank=0, world_size=1)
data_gens = {'train': train_data_gen, 'val': val_data_gen}

# --- Model Init ---
model.config["vocab_size"] = vocab_size
model.config["block_size"] = block_size 

if resume:
    print(f"Resuming from checkpoint: {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    
    model_instance = Transformer()
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model_instance.load_state_dict(state_dict)
    m = model_instance.to(device)

    opt_muon, opt_adam = m.configure_optimizers(weight_decay, lr, (beta1, beta2), device_type=device)

    if 'opt_adam' in checkpoint and 'opt_muon' in checkpoint:
        opt_adam.load_state_dict(checkpoint['opt_adam'])
        opt_muon.load_state_dict(checkpoint['opt_muon'])

    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])

    start_iter = checkpoint['iter'] + 1 
    run_name = checkpoint['run_name']
    train_losses_history = checkpoint.get('train_losses_history', [])
    val_losses_history = checkpoint.get('val_losses_history', [])

else:
    model_instance = Transformer()
    m = model_instance.to(device)
    
    # Init optimizers for fresh run
    opt_muon, opt_adam = m.configure_optimizers(weight_decay, lr, (beta1, beta2), device_type=device)
    start_iter = 0 
    print(f"Starting new run {run_name} from scratch")

p = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"{p/1e6:.2f} M parameters")

# --- Compile ---
print("compilation step")
if device == "cuda":
    # T4 supports compilation via Triton, though slightly less optimized than Ampere.
    # We keep it as it usually provides a speedup.
    compiled_model = torch.compile(m)
    print("compiled")
else:
    compiled_model = m
    print("skipped compilation")

# --- Loss Estimation ---
@torch.no_grad()
def estimate_loss(model_to_eval, data_gens):
    out = {}
    model_to_eval.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_from_shards(split, data_gens)
            with ctx:
                logits, loss = model_to_eval(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

@torch.no_grad()
def generate_text(model_to_gen, enc, max_new_tokens=1000, temperature=0.8, top_k=10):
    model_to_gen.eval() 
    start = "\n"
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    print("Generating text...")
    with ctx: 
        y = model_to_gen.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    generated_ids = y[0].tolist() 
    print(decode(generated_ids))
    model_to_gen.train()

# --- Training Loop ---
time_s = time.time()
prev_time = time_s 

# Ensure optimizer state is on correct device
for opt in [opt_adam, opt_muon]:
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor): state[k] = v.to(device)

opt_adam.zero_grad(set_to_none=True) 
opt_muon.zero_grad(set_to_none=True)

for iter_num in range(start_iter, max_iters + 1):

    # Learning rate schedule
    lr_iter = min_lr 
    if iter_num < warmup_iters:
        lr_iter = lr * iter_num / warmup_iters
    elif iter_num <= max_iters:
        decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        lr_iter = min_lr + coeff * (lr - min_lr)

    for opt in [opt_adam, opt_muon]:
        for param_group in opt.param_groups:
            param_group['lr'] = lr_iter

    # --- Evaluation ---
    if iter_num % eval_interval == 0 or iter_num == max_iters:
        losses = estimate_loss(m, data_gens)
        val_loss = losses['val']
        val_losses_history.append(val_loss)
        
        time_n = time.time()
        elapsed = time_n - time_s
        dt = time_n - prev_time 
        prev_time = time_n
        mfu = m.estimate_mfu(block_size * batch_size * grad_accum_steps, dt) if hasattr(m, 'estimate_mfu') else 0.0

        print(f"step: {iter_num}, train loss: {losses['train']:.4f}, val loss: {val_loss:.4f}, lr: {lr_iter:.6f}, elapsed: {elapsed/60:.2f} min, MFU: {mfu*100:.2f}%")

        if hasattr(m, 'generate'):
             generate_text(m, tok, max_new_tokens=200) 
        
        # --- Plotting ---
        plt.figure(figsize=(8, 4), dpi=100)
        iterations_eval = range(0, iter_num + 1, eval_interval)
        iterations_train = range(len(train_losses_history)) 
        
        if len(train_losses_history) > 0:
            plt.plot(iterations_train, train_losses_history, label='Train', color='royalblue', alpha=0.8)
        if len(val_losses_history) > 0:
            plt.plot(iterations_eval, val_losses_history, label='Val', color='palevioletred', alpha=0.8)

        plt.title(f"Loss - Run: {run_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"plots/{run_name}_plot_{iter_num}.png", bbox_inches='tight')
        plt.close()

    if iter_num == max_iters: break 

    # --- Training Step ---
    m.train()
    
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch_from_shards('train', data_gens)

        # FP16 Context (T4 safe)
        with ctx:
            logits, loss = compiled_model(xb, yb)
            loss = loss / grad_accum_steps 

        # Scaled Backward Pass (Prevents underflow in FP16)
        scaler.scale(loss).backward()
        loss_accum += loss.item() * grad_accum_steps 

    train_losses_history.append(loss_accum/grad_accum_steps)

    # Unscale before clipping
    scaler.unscale_(opt_adam)
    scaler.unscale_(opt_muon)
    
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)

    # Step with scaler
    scaler.step(opt_adam)
    scaler.step(opt_muon)
    
    scaler.update()

    opt_adam.zero_grad(set_to_none=True)
    opt_muon.zero_grad(set_to_none=True)

    # Checkpoint
    if iter_num % ckpt_iter == 0:
        ckpt_path = f'checkpoints/{run_name}_{iter_num}.pt'
        print(f"Saving checkpoint to {ckpt_path}")
        # Save standard checkpoint (Weights + Optimizer State)
        torch.save({
            'model': m.state_dict(),
            'opt_adam': opt_adam.state_dict(),
            'opt_muon': opt_muon.state_dict(),
            'scaler': scaler.state_dict(),
            'iter': iter_num,
            'run_name': run_name,
            'config': model.config,
            'train_losses_history': train_losses_history,
            'val_losses_history': val_losses_history,
        }, ckpt_path)

        # --- Optimzied FP16 Save (T4 Safe) ---
        print("Saving lightweight FP16 inference model...")
        fp16_inference_path = f'checkpoints/{run_name}_inference_fp16.pt'

        # fp16
        fp16_state_dict = {k: v.half() for k, v in m.state_dict().items()}

        # Weight Tying fix
        if 'transformer.wte.weight' in fp16_state_dict and 'lm_head.weight' in fp16_state_dict:
            fp16_state_dict['lm_head.weight'] = fp16_state_dict['transformer.wte.weight']

        # remove Causal Mask Buffer, mem save
        keys_to_remove = [k for k in fp16_state_dict.keys() if k.endswith('.attn.bias')]
        for k in keys_to_remove:
            del fp16_state_dict[k]

        torch.save(fp16_state_dict, fp16_inference_path)
        print(f"Saved optimized inference model to {fp16_inference_path}")

print('Training finished.')
