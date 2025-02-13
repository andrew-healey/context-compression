from torch import nn
import torch
import math
import torch.nn.functional as F

class CausalSelectiveSelfAttentionForInference(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_head * config.head_dim)
        self.c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd)
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        # Caching mechanism
        self.context_cache = {}
        self.use_cache = True

        # Constants for dynamic pruning
        self.MIN_CONTEXT_FOR_PRUNING = 256  # Only start pruning at this context length
        self.FULL_PRUNING_CONTEXT = 1024    # Context length where we reach maximum pruning

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.config = config

    def get_pruning_ratio(self, context_length):
        """
        Dynamically determine pruning ratio based on context length:
        - Below MIN_CONTEXT_FOR_PRUNING: No pruning (keep all tokens)
        - Between MIN_CONTEXT_FOR_PRUNING and FULL_PRUNING_CONTEXT:
          Linear interpolation from 1/2 to 1/5
        - Above FULL_PRUNING_CONTEXT: Maximum pruning (1/5)
        """
        if self.config.hard_pruning_constant is not None:
            return self.config.hard_pruning_constant

        if context_length < self.MIN_CONTEXT_FOR_PRUNING:
            return 1.0  # Keep all tokens
        elif context_length >= self.FULL_PRUNING_CONTEXT:
            return 0.2  # Keep 1/5 of tokens
        else:
            # Linear interpolation between 1/2 and 1/5
            progress = (context_length - self.MIN_CONTEXT_FOR_PRUNING) / (self.FULL_PRUNING_CONTEXT - self.MIN_CONTEXT_FOR_PRUNING)
            return 0.5 - (0.3 * progress)  # Smoothly transition from 0.5 to 0.2

    def forward(self, x, cache_key=None):
        B, T, C = x.size()

        # Check if cached context is available and should be used
        if self.use_cache and cache_key is not None and cache_key in self.context_cache:
            cached_context = self.context_cache[cache_key]
            pruned_att = cached_context['att']
            pruned_v = cached_context['v']
            pruned_lengths = cached_context['lengths']

            outputs = []

            for b in range(B):
                seq_outputs = []
                for pos in range(T):
                    idx = b * T + pos
                    curr_att = pruned_att[idx]
                    curr_v = pruned_v[idx]
                    curr_att = F.softmax(curr_att, dim=-1)
                    out = (curr_att.unsqueeze(-2) @ curr_v).squeeze(-2)
                    seq_outputs.append(out)
                batch_output = torch.stack(seq_outputs, dim=1)
                outputs.append(batch_output)

            y = torch.stack(outputs, dim=0)
            y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
            y = self.c_proj(y)

            return y, None

        # Existing attention computation logic
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_head * self.head_dim, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Standard attention computation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # Apply selective attention with forgetting
        S = att[:, 0].clone()
        S = F.relu(S)

        S_masked = torch.zeros_like(S)
        S_masked[..., 1:] = S[..., 1:]

        eye_mask = 1 - torch.eye(T, device=S.device)
        S = S_masked * eye_mask

        S_shifted = torch.roll(S, 1, -2)
        S_shifted[..., 0, :] = 0

        FF = torch.cumsum(S_shifted, dim=-2)[:, None]

        # Always apply forgetting regardless of context length
        att -= FF

        # Determine if and how much to prune based on context length
        pruning_ratio = self.get_pruning_ratio(T)
        memory_budget = max(1, int(T * pruning_ratio))

        if pruning_ratio < 1.0:
            # --- BEGIN VECTORIZED COMPUTATION OF Y ---
            device = att.device
            t_range = torch.arange(T, device=device)
            valid_mask = t_range.unsqueeze(0) <= t_range.unsqueeze(1)  # shape: (T, T)
            FF_mod = FF.squeeze(1).masked_fill(~valid_mask, float('inf'))
            sorted_idx = torch.argsort(FF_mod, dim=-1)
            effective_k = torch.minimum(torch.full((T,), memory_budget - 1, device=device),
                                          (t_range + 1))
            r = torch.arange(T, device=device).view(1, 1, T).expand(1, T, T)
            effective_k_exp = effective_k.view(1, T, 1)
            sorted_sel_mask = (r < effective_k_exp)
            keep_indices_v = torch.zeros((B, T, T), dtype=torch.bool, device=device)
            keep_indices_v[:, torch.arange(T), torch.arange(T)] = True
            nz = torch.nonzero(sorted_sel_mask.expand(B, T, T))
            b_idx = nz[:, 0]
            pos_idx = nz[:, 1]
            rank_idx = nz[:, 2]
            selected_j = sorted_idx[b_idx, pos_idx, rank_idx]
            keep_indices_v[b_idx, pos_idx, selected_j] = True
            att_masked = att.masked_fill(~keep_indices_v.unsqueeze(1), float('-inf'))
            y = (F.softmax(att_masked, dim=-1) @ v)  # shape: (B, n_head, T, head_dim)
            # --- END VECTORIZED COMPUTATION OF Y ---

            # Cache the pruned context if requested (vectorized caching)
            if self.use_cache and cache_key is not None:
                pruned_lengths_cache = keep_indices_v.sum(dim=-1)  # shape: (B, T)
                pruned_att_cache = att.masked_fill(~keep_indices_v.unsqueeze(1), float('-inf'))
                pruned_v_cache = v.unsqueeze(2).expand(B, self.n_head, T, T, v.size(-1)) \
                                 .masked_fill(~keep_indices_v.unsqueeze(1).unsqueeze(-1), 0)
                self.context_cache[cache_key] = {
                    'att': pruned_att_cache,
                    'v': pruned_v_cache,
                    'lengths': pruned_lengths_cache
                }

        else:
            # Standard attention for small contexts
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)

        return y, None

    def clear_cache(self):
        """Clear the entire context cache."""
        self.context_cache.clear()

    def remove_cache_entry(self, cache_key):
        """Remove a specific entry from the context cache."""
        if cache_key in self.context_cache:
            del self.context_cache[cache_key]

class CausalSelectiveSelfAttentionWithMemoryPenalty(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_head * config.head_dim)
        self.c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd)
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.tau = 1.0  # Clamping parameter for FF scores

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_head * self.head_dim, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Standard attention computation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # Selective attention computation
        S = att[:, 0].clone()
        S = F.relu(S)

        S_masked = torch.zeros_like(S)
        S_masked[..., 1:] = S[..., 1:]

        eye_mask = 1 - torch.eye(T, device=S.device)
        S = S_masked * eye_mask

        S_shifted = torch.roll(S, 1, -2)
        S_shifted[..., 0, :] = 0

        FF = torch.cumsum(S_shifted, dim=-2)[:, None]

        # Calculate memory requirements M_i^l
        FF_clamped = torch.clamp(FF, 0, self.tau)  # Clamp FF values between 0 and tau
        FF_sum = FF_clamped.sum(dim=-1)  # Sum over k dimension

        # Calculate M_i^l = i - Σ(min(FF^l_(i,k), τ))/τ
        positions = torch.arange(T, device=x.device).float()  # [0, 1, ..., T-1]
        positions = positions.view(1, 1, T, 1)  # Add batch and head dimensions
        M = positions - FF_sum / self.tau  # Calculate memory requirements

        # Continue with attention computation
        att -= FF
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)

        return y, M

class CausalSelectiveSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_head * config.head_dim)
        # output projection
        self.c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_head * self.head_dim, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # Standard attention computation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # Apply selective attention
        S = att[:, 0].clone()  # Select head 0 logits (clone to avoid in-place modification issues)
        S = F.relu(S)  # Only positive selection

        # Use torch.zeros_like to safely modify without inplace ops
        S_masked = torch.zeros_like(S)  # Create a mask to avoid in-place ops
        S_masked[..., 1:] = S[..., 1:]  # Do not mask <BOS> token, leave it unchanged

        eye_mask = 1 - torch.eye(T, device=S.device)  # Do not mask self
        S = S_masked * eye_mask  # Apply the masking to avoid self-attention

        S_shifted = torch.roll(S, 1, -2)  # Shift to mask strictly in the future
        S_shifted[..., 0, :] = 0  # Ensure future masking without inplace

        # Use out-of-place subtraction to preserve computation graph integrity
        cs = torch.cumsum(S_shifted, dim=-2)[:, None]
        att = att - cs

        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        return y, None

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_head * config.head_dim)
        # output projection
        self.c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_head * self.head_dim, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y, None

from enum import StrEnum, auto

class AttentionKind(StrEnum):
    SELECTIVE = auto()
    SELECTIVE_WITH_MEMORY_PENALTY = auto()
    SELF = auto()

def get_attention_cls(kind: AttentionKind, for_inference: bool) -> nn.Module:
    if kind == AttentionKind.SELECTIVE:
        if for_inference:
            return CausalSelectiveSelfAttentionForInference
        else:
            return CausalSelectiveSelfAttention
    elif kind == AttentionKind.SELECTIVE_WITH_MEMORY_PENALTY:
        if for_inference:
            return CausalSelectiveSelfAttentionForInference
        else:
            return CausalSelectiveSelfAttentionWithMemoryPenalty
    elif kind == AttentionKind.SELF:
        return CausalSelfAttention
    else:
        raise ValueError(f"Invalid attention kind: {kind}")
