import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from collections import defaultdict

def zeropower_via_newtonschulz5(G, steps=5, epsilon=1e-7):
    """
    Newton-Schulz iteration to compute the approximate zeroth power of the matrix G.
    This is the standard 'Muon' orthogonalization step.
    
    Args:
        G: Input tensor of shape (batch_size, m, n)
        steps: Number of iterations
    """
    assert len(G.shape) == 3
    
    # T4 / FP16 Stability: We must perform the iterative inverse in FP32 
    # or the T4 will overflow/underflow immediately.
    orig_dtype = G.dtype
    X = G.float() 
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    X /= (X.norm(dim=(1, 2), keepdim=True) + epsilon) # Initial normalization

    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    return X.to(orig_dtype)

class NorMuon(Optimizer):
    """
    Single-GPU NorMuon for T4 (FP16).
    
    - Removed distributed scatter/gather logic.
    - Removed hardcoded architecture reshaping.
    - Added FP32 upcast during Newton-Schulz for T4 stability.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95, 
                 ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, 
                        beta2=beta2, ns_steps=ns_steps)
        
        # Group params by shape for efficient batched processing
        params = list(params)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        """
        # We group params by shape dynamically in the step to allow batching
        # This replaces the complex distributed "custom_sizing"
        grouped_params = defaultdict(list)
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    # Muon only works on 2D+ matrices. 
                    # For 1D params (biases/layernorms), you should usually use AdamW.
                    # If you must use this class, we skip them or need a fallback.
                    continue 
                
                grouped_params[p.shape].append(p)

            # Process each shape group as a batch
            for shape, params in grouped_params.items():
                if not params:
                    continue
                
                # 1. Stack gradients: (Batch, Rows, Cols)
                # Detach grads to avoid graph retention
                grads = torch.stack([p.grad for p in params])
                
                # 2. Handle flattened views if params are > 2D (e.g. Conv2D)
                original_shape = grads.shape
                if len(shape) > 2:
                    # Flatten to (Batch, Out, In)
                    grads = grads.view(len(params), shape[0], -1)

                # 3. Momentum
                # We use a shared buffer dict keyed by param object id to store momentum
                state = self.state
                
                # Create/Retrieve momentum buffer
                # We stack the states to match the batching
                mom_buffer_list = []
                for p in params:
                    if 'momentum_buffer' not in state[p]:
                        state[p]['momentum_buffer'] = torch.zeros_like(p.grad)
                    mom_buffer_list.append(state[p]['momentum_buffer'])
                
                # Stack momentum buffers for batched operation
                mom_buffer = torch.stack(mom_buffer_list)
                
                # Apply momentum: buf = buf * momentum + grad * (1-momentum)
                # Note: The original code used lerp which is functionally equivalent
                mom_buffer.lerp_(grads, 1 - group['momentum'])
                
                # Update grads with Nesterov-like view or standard momentum 
                # Original code: updated_grads = grads.lerp(momentum_buffer, momentum)
                updated_grads = grads.lerp(mom_buffer, group['momentum'])

                # 4. Prepare Metadata for NorMuon scaling
                # We calculate shapes based on the flattened view
                H, W = updated_grads.shape[-2], updated_grads.shape[-1]
                
                # Retrieve Second Momentum (NorMuon specific)
                # Tracks variance along the larger dimension
                second_mom_list = []
                use_rows = H >= W
                
                for p in params:
                    if 'second_momentum_buffer' not in state[p]:
                        # Shape is (H, 1) or (1, W) depending on which dim is larger
                        shape_2nd = (H, 1) if use_rows else (1, W)
                        # If original param was >2D, we rely on the flattened view shape here
                        state[p]['second_momentum_buffer'] = torch.zeros(shape_2nd, device=p.device, dtype=p.dtype)
                    second_mom_list.append(state[p]['second_momentum_buffer'])
                    
                second_momentum_buffer = torch.stack(second_mom_list)

                # 5. Orthogonalization (Newton-Schulz / Polar Express)
                # This replaces "polar_express(updated_grads)"
                # CRITICAL for T4: This function internally casts to FP32
                v_chunk = zeropower_via_newtonschulz5(updated_grads, steps=group['ns_steps'])

                # 6. NorMuon Variance Estimation & Scaling
                # Calculate mean of squared updates
                v_mean = v_chunk.square().mean(dim=-1 if use_rows else -2, keepdim=True)
                
                # Update second momentum
                second_momentum_buffer.lerp_(v_mean, 1 - group['beta2'])
                
                # Calculate adaptive step size
                step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
                
                # Scale the orthogonal update
                v_chunk.mul_(step_size)
                
                # Re-normalize (RMS norm correction)
                # v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
                # v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True) 
                # Optimization: calculate correction factor directly
                # Original code logic:
                v_norm_orig = updated_grads.norm(dim=(-2, -1), keepdim=True)
                v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
                v_chunk.mul_(v_norm_orig / v_norm_new.clamp_min(1e-10))

                # 7. Weight Decay and Parameter Update
                
                # Stack params for batched update
                # Note: params are likely fp16.
                param_stack = torch.stack(params)
                if len(original_shape) > 3:
                    param_stack_view = param_stack.view(len(params), shape[0], -1)
                else:
                    param_stack_view = param_stack

                # Effective LR and WD
                # Assuming simple scalar LR/WD here, ignoring the per-param 'lr_mul' complexity 
                # unless you strictly need it (removed for simplicity/speed).
                # If you need per-layer scaling, calculate it here.
                
                # Scaling factor from Muon paper (sqrt(max(H,W)/min(H,W)))
                param_lr_scale = (max(H, W) / min(H, W)) ** 0.5
                eff_lr = group['lr'] * param_lr_scale
                eff_wd = group['weight_decay']

                # Cautious Weight Decay
                # Mask = 1 if update and param have same sign, else 0
                mask = (v_chunk * param_stack_view) >= 0
                
                # Apply WD: param = param - lr * wd * param * mask
                # We do addcmul with negative coeff
                v_chunk.addcmul_(param_stack_view, mask.to(v_chunk.dtype), value=eff_wd)

                # Final Update: param = param - lr * update
                # Using addcmul_ or just add_
                param_stack_view.add_(v_chunk, alpha=-eff_lr)

                # Copy updated params back to originals
                # (Since param_stack_view is a view of param_stack, updates to view reflect in stack)
                # But param_stack is a new tensor, so we must copy back.
                if len(original_shape) > 3:
                    # Unflatten if necessary
                    param_stack = param_stack_view.view(original_shape)
                
                for i, p in enumerate(params):
                    p.copy_(param_stack[i])
                
                # Copy updated momentum buffers back to state
                # (Since we unbind/stack, the state dict tensors aren't updated automatically)
                unstacked_mom = torch.unbind(mom_buffer)
                unstacked_second_mom = torch.unbind(second_momentum_buffer)
                
                for i, p in enumerate(params):
                    state[p]['momentum_buffer'].copy_(unstacked_mom[i])
                    state[p]['second_momentum_buffer'].copy_(unstacked_second_mom[i])

            # Clear the temp dict
            grouped_params.clear()

