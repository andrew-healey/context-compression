from .model import GPT, GPTConfig
import torch
import torch.nn as nn

from dataclasses import dataclass
from enum import Enum, auto

class AddHeadKind(Enum):
    GROW_QKV_O = auto()
    NONE = auto()

@dataclass
class AddHeadConfig:
    add_head_kind: AddHeadKind = AddHeadKind.GROW_QKV_O
    add_head_to_start: bool = True
    zero_out_new_head: bool = False

def add_a_head(
        config: GPTConfig,
        model: GPT,
        add_head_config: AddHeadConfig = AddHeadConfig()
    ):
    if add_head_config.add_head_kind == AddHeadKind.NONE:
        return
    elif add_head_config.add_head_kind == AddHeadKind.GROW_QKV_O:
        grow_qkv_o(config, model, add_head_config)
    else:
        raise ValueError(f"Invalid add head kind: {add_head_config.add_head_kind}")
    

def grow_qkv_o(
        config: GPTConfig,
        model: GPT,
        add_head_config: AddHeadConfig
    ):

    old_n_head = config.n_head

    config.n_head += 1

    for block in model.transformer.h:

        # c_attn
        new_c_attn = nn.Linear(config.n_embd, 3 * config.n_head * config.head_dim, device=block.attn.c_attn.weight.device, dtype=block.attn.c_attn.weight.dtype)
        new_c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, device=block.attn.c_proj.weight.device, dtype=block.attn.c_proj.weight.dtype)

        old_c_attn = block.attn.c_attn
        old_c_proj = block.attn.c_proj
        block.attn.n_head = config.n_head

        with torch.no_grad():

            new_c_attn_weight = new_c_attn.weight.T
            assert new_c_attn_weight.shape == (config.n_embd, 3 * config.n_head * config.head_dim), f"I expect new_c_attn_weight.shape == (config.n_embd, 3 * config.n_head * config.head_dim), but got {new_c_attn_weight.shape} != {(config.n_embd, 3 * config.n_head * config.head_dim)}"
            new_c_attn_weight = new_c_attn_weight.view(config.n_embd, 3, config.n_head, config.head_dim)
            
            old_c_attn_weight = old_c_attn.weight.T
            assert old_c_attn_weight.shape == (config.n_embd, 3 * old_n_head * config.head_dim), f"I expect old_c_attn_weight.shape == (config.n_embd, 3 * old_n_head * config.head_dim), but got {old_c_attn_weight.shape} != {(config.n_embd, 3 * old_n_head * config.head_dim)}"
            old_c_attn_weight = old_c_attn_weight.view(config.n_embd, 3, old_n_head, config.head_dim)

            if add_head_config.zero_out_new_head:
                new_c_attn_weight[:, :, :, :] = 0

            if add_head_config.add_head_to_start:
                new_c_attn_weight[:, :, 1:, :] = old_c_attn_weight
            else:
                new_c_attn_weight[:, :, :-1, :] = old_c_attn_weight
            
            # now the bias
            new_c_attn_bias = new_c_attn.bias
            assert new_c_attn_bias.shape == (3 * config.n_head * config.head_dim,), f"I expect new_c_attn_bias.shape == (3 * config.n_head * config.head_dim), but got {new_c_attn_bias.shape} != {(3 * config.n_head * config.head_dim)}"
            new_c_attn_bias = new_c_attn_bias.view(3, config.n_head, config.head_dim)
            old_c_attn_bias = old_c_attn.bias
            assert old_c_attn_bias.shape == (3 * old_n_head * config.head_dim,), f"I expect old_c_attn_bias.shape == (3 * old_n_head * config.head_dim), but got {old_c_attn_bias.shape} != {(3 * old_n_head * config.head_dim)}"
            old_c_attn_bias = old_c_attn_bias.view(3, old_n_head, config.head_dim)

            # we always zero this bias - otherwise it *kills* the perf of the rest of the model
            # I think the structure of the weight initialization protects them
            # b/c they start out small, and the o is multiplied by the v, so you get small * small = very small contribution to y from the new head.
            new_c_attn_bias[:] = 0
            
            if add_head_config.add_head_to_start:
                new_c_attn_bias[:, 1:, :] = old_c_attn_bias
            else:
                new_c_attn_bias[:, :-1, :] = old_c_attn_bias

            block.attn.c_attn = new_c_attn

            # c_proj
            new_c_proj_weight = new_c_proj.weight.T
            assert new_c_proj_weight.shape == (config.n_head * config.head_dim, config.n_embd), f"I expect new_c_proj_weight.shape == (config.n_head * config.head_dim, config.n_embd), but got {new_c_proj_weight.shape} != {(config.n_head * config.head_dim, config.n_embd)}"
            new_c_proj_weight = new_c_proj_weight.view(config.n_head, config.head_dim, config.n_embd)
            
            old_c_proj_weight = old_c_proj.weight.T
            assert old_c_proj_weight.shape == (old_n_head * config.head_dim, config.n_embd), f"I expect old_c_proj_weight.shape == (old_n_head * config.head_dim, config.n_embd), but got {old_c_proj_weight.shape} != {(old_n_head * config.head_dim, config.n_embd)}"
            old_c_proj_weight = old_c_proj_weight.view(old_n_head, config.head_dim, config.n_embd)

            if add_head_config.zero_out_new_head:
                new_c_proj_weight[:, :, :] = 0

            if add_head_config.add_head_to_start:
                new_c_proj_weight[1:, :, :] = old_c_proj_weight
            else:
                new_c_proj_weight[:-1, :, :] = old_c_proj_weight
            
            # we don't need to modify the bias for c_proj
            new_c_proj.bias = old_c_proj.bias

            block.attn.c_proj = new_c_proj
