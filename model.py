import math 
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

config = {
    "n_embd": 80,         
    "n_head": 2, # at higher depths, more heads is considered better?            
    "n_layer": 50,         
    "dropout": 0.2,         
    "vocab_size": 50257,    # update in train
    "ctx_len": 1024,        # update in train 
    "bias": False,           
}

class RoPE(nn.Module):

    def __init__(self, d_head): # rope changes qk after it is split by attn heads
        super().__init__()

        self.dim = d_head 
        self.ctx = config['ctx_len']

        # pre-compute the theta values and store them

        theta = 10000.0 ** (-2.0 * torch.arange(0, self.dim, 2).float() / self.dim)
        t = torch.arange(self.ctx, dtype=torch.float)

        # shapes t -> (ctx_len, 1), theta -> (1, dim/2) broadcast (ctx_len, dim/2)
        freqs = t.unsqueeze(1) * theta.unsqueeze(0)

        # complex number trick, (cos th + i * sin th)
        freq_cis = torch.polar(torch.ones_like(freqs), freqs)

        self.register_buffer('freq_cis', freq_cis)

    def forward(self, x):

        B, nh, T, hs = x.shape # input is B, nh, T, hs

        x_complex = torch.view_as_complex(x.float().reshape(B, nh, T, hs//2, 2)) # split into 2 groups 

        freq_cis = self.freq_cis[:T].view(1, 1, T, -1)

        x_rot_complex = x_complex * freq_cis

        x_rot = torch.view_as_real(x_rot_complex)
        x_out = x_rot.reshape(B, nh, T, hs)

        return x_out.type_as(x)

class CasualSelfAttn(nn.Module):

    def __init__(self):
        super().__init__()

        assert config['n_embd'] % config['n_head'] == 0

        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=config.get('bias', False))
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=config.get('bias', False))

        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])

        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.dropout = config['dropout']
        self.block_size = config['ctx_len']

        # qk norm and rope share the same size

        self.q_norm = nn.RMSNorm(self.n_embd//self.n_head) 
        self.k_norm = nn.RMSNorm(self.n_embd//self.n_head)
        self.v_norm = nn.RMSNorm(self.n_embd//self.n_head)

        # rope

        self.rope = RoPE(self.n_embd//self.n_head)

        # scaled value resid, Attn @ V + alpha . V

        self.logit_alpha = nn.Parameter(torch.tensor(-2.0))

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                        .view(1, 1, self.block_size, self.block_size))

    def forward(self, x):
        B, T, C = x.size() # bs, ctx_len, n_embd

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # qk norm 

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        v_r = v # raw no norm
        v = self.v_norm(v)

        # qk rope
        q = self.rope(q) 
        k = self.rope(k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # flash_attn
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # value residual

        alpha = torch.sigmoid(self.logit_alpha)
        y = y + alpha * v_r

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y

# Swiglu replaces MLP 

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        n = int((8/3) * config['n_embd'])
        appr = (n + 63) & ~(63) # make it a multiple of 64

        # combine gate and value
        self.gate_value_proj = nn.Linear(config['n_embd'], 2 * appr, bias=False) # Llama uses no bias
        self.linear_out = nn.Linear(appr, config['n_embd'], bias=False)
        self.silu = nn.SiLU()

        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        
        # project input to 2 * appr, split the tensor in half, gate and val
        gate_value = self.gate_value_proj(x)
        gate, value = torch.chunk(gate_value, 2, dim=-1)

        x = self.silu(gate) * value
        x = self.linear_out(x)
        x = self.dropout(x)

        return x

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config['n_embd'])
        self.attn = CasualSelfAttn()
        self.ln_2 = nn.RMSNorm(config['n_embd'])
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.block_size = config['ctx_len']

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']), # tok embd
            wpe = nn.Embedding(self.block_size, config['n_embd']), # pos embd
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block() for _ in range(config['n_layer'])]) 
        ))

        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        # weight-tying
        self.lm_head.weight = self.transformer.wte.weight

        # init all weights
        self.apply(self._init_weights)
        
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))

        print(f"Model initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        
        device = idx.device
        b, t = idx.size()

        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        #x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward final layer norm and lm_head on the very last position
            # Note: This optimization is not needed if using generate method below
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad() 
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 1. Get all parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # 2. Define containers
        muon_params = []
        adam_decay = []
        adam_nodecay = []

        # 3. Sort parameters into groups
        for name, p in param_dict.items():
            # A. 1D Parameters (Biases, LayerNorms) -> AdamW (No Decay)
            if p.dim() < 2:
                adam_nodecay.append(p)
            
            # B. Embeddings and Final Heads -> AdamW (Decay)
            # Muon works by orthogonalizing matrices. This is mathematically wrong for 
            # Embeddings (which are lookup tables) and the final Classifier Head.
            # We check common names for these layers.
            elif any(k in name for k in ["embed", "token", "wte", "wpe", "head", "output"]):
                adam_decay.append(p)
            
            # C. All other 2D Parameters (Attention, MLP weights) -> NorMuon
            else:
                muon_params.append(p)

        # 4. Print stats (sanity check)
        print(f"Muon Params: {len(muon_params)} tensors (Linear/Conv weights)")
        print(f"AdamW Decay Params: {len(adam_decay)} tensors (Embeddings/Heads)")
        print(f"AdamW No-Decay Params: {len(adam_nodecay)} tensors (Norms/Biases)")

        # 5. Create the Optimizers
        
        # --- Optimizer 1: AdamW ---
        # Used for Embeddings, Heads, Norms, Biases
        optim_groups_adam = [
            {'params': adam_decay, 'weight_decay': weight_decay},
            {'params': adam_nodecay, 'weight_decay': 0.0}
        ]
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type.startswith('cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer_adam = torch.optim.AdamW(optim_groups_adam, lr=learning_rate, betas=betas, **extra_args)

        optimizer_muon = torch.optim.Muon(muon_params, lr = 5*learning_rate, weight_decay=weight_decay)

        return [optimizer_muon, optimizer_adam]

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters()) # Use actual parameter count
        cfg = self.transformer.h[0].attn # Get config from an attention block instance if needed
                                         # Or better, access directly if stored on self or use model.config
                                         # For now, let's assume model.config is globally updated
        L, H, Q, T = config['n_layer'], config['n_head'], config['n_embd']//config['n_head'], self.block_size

        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak FLOPS
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # Adjust peak flops based on hardware if necessary
        # A100 = 312 TFLOPS (BF16), H100 = 989 TFLOPS (FP16) / 495 TFLOPS (TF32)
        # T4 = 65 TFLOPS (FP16)
        flops_promised = 312e12 # A100 BF16 peak flops
        if torch.cuda.is_available():
             dev_prop = torch.cuda.get_device_properties(torch.cuda.current_device())
             if dev_prop.major >= 8: # Ampere or newer
                 # Use BF16 peak for A100
                 flops_promised = 312e12
                 if dev_prop.major >= 9: # Hopper
                      # Using FP16 peak for H100, adjust if using TF32
                      flops_promised = 989e12
             elif dev_prop.major == 7: # Volta/Turing (like T4)
                 flops_promised = 65e12 # T4 FP16 peak

        mfu = flops_achieved / flops_promised
        return mfu

