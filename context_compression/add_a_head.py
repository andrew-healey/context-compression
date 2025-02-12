from .model import GPT, GPTConfig
import torch.nn as nn

def grow_qkv_o(config, model):

    old_n_head = config.n_head

    config.n_head += 1

    for block in model.transformer.h:
        # c_attn
        new_c_attn = nn.Linear(config.n_embd, 3 * config.n_head * config.head_dim, device=model.device, dtype=block.attn.c_attn.weight.dtype)
        new_c_attn_weight = new_c_attn.weight
        assert new_c_attn_weight.shape == (config.n_embd, 3 * config.n_head * config.head_dim)
        new_c_attn_weight = new_c_attn_weight.view(config.n_embd, 3, config.n_head, config.head_dim)
        
        old_c_attn_weight = block.attn.c_attn.weight
        assert old_c_attn_weight.shape == (config.n_embd, 3 * old_n_head * config.head_dim)
        old_c_attn_weight = old_c_attn_weight.view(config.n_embd, 3, old_n_head, config.head_dim)

        new_c_attn_weight[:, :, 1:, :] = old_c_attn_weight

        block.attn.c_attn = new_c_attn

        # c_proj
        new_c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, device=model.device, dtype=block.attn.c_proj.weight.dtype)
        new_c_proj_weight = new_c_proj.weight
        assert new_c_proj_weight.shape == (config.n_head * config.head_dim, config.n_embd)
        new_c_proj_weight = new_c_proj_weight.view(config.n_head, config.head_dim, config.n_embd)
        
        old_c_proj_weight = block.attn.c_proj.weight
        assert old_c_proj_weight.shape == (old_n_head * config.head_dim, config.n_embd)
        old_c_proj_weight = old_c_proj_weight.view(old_n_head, config.head_dim, config.n_embd)

        new_c_proj_weight[:, 1:, :] = old_c_proj_weight

        block.attn.c_proj = new_c_proj

        block.attn.n_head = config.n_head
