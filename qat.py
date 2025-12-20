import torch
import torch.nn as nn
from model import Transformer, config  # Import your architecture
from torchao.quantization import quantize_, Int4WeightOnlyConfig
import os
import time

# --- Settings ---
# Note: Use the FULL checkpoint if possible, but this script works with your inf_fp16
input_checkpoint = "sVtcrs_inference_fp16.pt" 
output_checkpoint = "sVtcrs_qat_int4.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# QAT Hyperparams (Much lower than original training)
qat_iters = 2000       # How many steps to "refine" the 4-bit weights
qat_lr = 5e-6         # Very small LR to prevent weight explosion
batch_size = 4        # Keep small to save VRAM during QAT
block_size = 1024

from model import config 
config["vocab_size"] = 8192  # This was your "nano_8k" size
config["n_embd"] = 64        # Based on your error message: [8192, 64]
config["block_size"] = 1024  # Your context length

# --- 1. Load the Model & Weights ---
print(f"Loading weights from {input_checkpoint}...")
m = Transformer()
state_dict = torch.load(input_checkpoint, map_location=device)

# Handle the Weight Tying and Missing Buffers (if you stripped them)
# Your Transformer likely recreates the causal mask on init, so missing .attn.bias is fine.
m.load_state_dict(state_dict, strict=False) 
m.to(device)

# --- 2. Apply INT4 QAT Transformation ---
print("Applying INT4 QAT nodes...")
# is_qat=True inserts "Fake Quantization" modules into the Linear layers
quantize_(m, int4_weight_only_qat(group_size=128))

# --- 3. Setup QAT Optimizer ---
# We use a simple AdamW for the fine-tuning phase
optimizer = torch.optim.AdamW(m.parameters(), lr=qat_lr)

# --- 4. Mini-QAT Loop ---
# We need your data generator from the original script
# Make sure your data_dir and generators are defined here!
from train import create_data_generator # Assumes train.py is in the same folder
data_pattern = os.path.join("data", "synth_2", "synth_2_train_*.bin")
train_gen = create_data_generator(data_pattern, batch_size, block_size)

print(f"Starting QAT for {qat_iters} iterations...")
m.train()
start_time = time.time()

for i in range(qat_iters):
    xb, yb = next(train_gen)
    
    # We use basic FP16/BF16 autocast during QAT for stability
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits, loss = m(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Gradient clipping is very important in QAT to prevent "NaNs"
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    
    optimizer.step()
    
    if i % 50 == 0:
        print(f"Iter {i} | Loss: {loss.item():.4f}")

print(f"QAT Finished in {(time.time() - start_time)/60:.2f} mins.")

# --- 5. Convert to Real INT4 and Save ---
print("Freezing weights and converting to true INT4 integers...")
m.eval()

# To save a "pure" inference version, we convert the fake-quantized 
# floats into actual int8/int4 storage types.
# For torchao, you often save the state_dict directly after training; 
# the inference engine then handles the 4-bit packing.
save_dict = {
    'model': m.state_dict(),
    'config': config,
    'bit_depth': 'int4'
}

torch.save(save_dict, output_checkpoint)
print(f"Successfully saved INT4 model to {output_checkpoint}")
