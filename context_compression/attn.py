from torch import nn
import torch
import math
import torch.nn.functional as F
import random
import os

#from .protection.protect_and_attack import cumsum_triton, cumsum_bliasson, attack_and_protect_bliasson
from enum import StrEnum, auto

from dataclasses import dataclass
from typing import Optional, Tuple


# construct Q, K, and V from hidden state
class QKVProducer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, hidden_state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("QKVProducer is an abstract class")

class QKVProducerKind(StrEnum):
    INHERIT = auto()
    MHA = auto()

    def get_qkv_producer(self) -> Optional[QKVProducer]:
        if self == QKVProducerKind.MHA:
            return QKVProducerMHA
        else:
            return None

class QKVProducerMHA(QKVProducer):
    def __init__(self, config):
        super().__init__(config)
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_head * config.head_dim + config.n_head * config.dense_attention_config.head_dim_value)
        # set query to zero
        self.c_attn.MUP_INIT_RANGE_TO_ZERO = (0, config.n_head * config.head_dim)

    def forward(self, hidden_state):
        qkv = self.c_attn(hidden_state)
        q, k, v = qkv.split([self.config.n_head * self.config.head_dim, self.config.n_head * self.config.head_dim, self.config.n_head * self.config.dense_attention_config.head_dim_value], dim=-1)
        B, T, _ = v.shape
        v = v.view(B, T, self.config.n_head, self.config.dense_attention_config.head_dim_value).transpose(1, 2)
        return q, k, v

# turn Q and K into attention scores
class AProducer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        raise NotImplementedError("AProducer is an abstract class")
class AProducerKind(StrEnum):
    INHERIT = auto()
    MHA = auto()
    MHA_CONV = auto()

    def get_a_producer(self) -> Optional[AProducer]:
        if self == AProducerKind.MHA:
            return AProducerMHA
        elif self == AProducerKind.MHA_CONV:
            return AProducerConv
        else:
            return None

class AProducerMHA(AProducer):
    def __init__(self, config):
        super().__init__(config)

        self.attn_mult = config.attn_mult

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        B, T, _ = q.shape
        nh = self.config.n_head
        hd = self.config.head_dim
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        A = q @ k.transpose(-2, -1) * self.attn_mult # shape: (B, nh, T, T)
        return A

class AProducerConv(AProducer):
    def __init__(self, config):
        super().__init__(config)

        self.attn_mult = config.attn_mult
        self.linear = nn.Linear(config.n_head,config.n_head)
        # self.linear.EYE_INIT = True

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        B, T, _ = q.shape
        nh = self.config.n_head
        hd = self.config.head_dim
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        A_raw = q @ k.transpose(-2, -1) * self.attn_mult # shape: (B, nh, T, T)
        # dense interaction between all the heads
        A = self.linear(A_raw.transpose(1,3).contiguous()).transpose(1,3).contiguous() # shape: (B, nh, T, T)
        return A

# turn softmax'd attention scores and values into the hidden state residual
class AVCombiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, A: torch.Tensor, v: torch.Tensor):
        raise NotImplementedError("AVCombiner is an abstract class")

class AVCombinerKind(StrEnum):
    INHERIT = auto()
    LINEAR = auto()

    def get_av_combiner(self) -> Optional[AVCombiner]:
        if self == AVCombinerKind.LINEAR:
            return AVCombinerLinear
        else:
            return None

class AVCombinerLinear(AVCombiner):
    def __init__(self, config):
        super().__init__(config)
        self.c_proj = nn.Linear(config.n_head * config.dense_attention_config.head_dim_value, config.n_embd)
        if config.c_proj_scale_init is not None:
            self.c_proj.NANOGPT_SCALE_INIT = config.c_proj_scale_init

    def forward(self, A: torch.Tensor, v: torch.Tensor):
        B, nh, T, T = A.shape
        assert v.shape == (B, nh, T, self.config.dense_attention_config.head_dim_value)
        attn_output = A @ v # shape: (B, nh, T, self.config.dense_attention_config.head_dim_value)
        hidden_state_output = self.c_proj(attn_output.transpose(1, 2).reshape(B, T, nh * self.config.dense_attention_config.head_dim_value))
        return hidden_state_output

# each "kind" is shorthand for a certain config.
class DenseAttentionKind(StrEnum):
    MHA = auto()
    MQA = auto()
    GQA = auto()
    MLA = auto()
    MLPA = auto()
    TPA_NAIVE = auto()
    TPA_EFFICIENT = auto() # should theoretically match TPA_NAIVE outputs, but be faster and lower memory usage, I think.

    def get_qkv_producer(self,config):

        # check for override
        overridden_qkv_producer = config.dense_attention_config.qkv_producer_kind.get_qkv_producer()
        if overridden_qkv_producer is not None:
            return overridden_qkv_producer(config)

        if self == DenseAttentionKind.MHA:
            return QKVProducerMHA(config)
        else:
            raise ValueError(f"No QKVProducer for {self}")

    def get_a_producer(self,config):
        # check for override
        overridden_a_producer = config.dense_attention_config.a_producer_kind.get_a_producer()

        if overridden_a_producer is not None:
            return overridden_a_producer(config)

        if self == DenseAttentionKind.MHA:
            return AProducerMHA(config)
        else:
            raise ValueError(f"No AProducer for {self}")

    def get_av_combiner(self,config):
        # check for override
        overridden_av_combiner = config.dense_attention_config.av_combiner_kind.get_av_combiner()
        if overridden_av_combiner is not None:
            return overridden_av_combiner(config)

        if self == DenseAttentionKind.MHA:
            return AVCombinerLinear(config)
        else:
            raise ValueError(f"No AVCombiner for {self}")
        

class CausalDenseSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense_attention_config = config.dense_attention_config
        self.dense_attention_kind = self.dense_attention_config.dense_attention_kind

        # NOTE: LEFT OFF HERE

        self.qkv_producer = self.dense_attention_kind.get_qkv_producer(config)
        self.a_producer = self.dense_attention_kind.get_a_producer(config)
        self.av_combiner = self.dense_attention_kind.get_av_combiner(config)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.config = config

        self.attn_score = nn.Identity() # for mup coord checking
        self.query = nn.Identity()
        self.key = nn.Identity()
        self.value = nn.Identity()

        self.precision = torch.float32 if self.config.attn_precision == "float32" else torch.bfloat16

        self.attn_mult = config.attn_mult

    def _forward(self, x, ff_cache=None, old_raw_att=None):
        with torch.autocast(device_type=x.device.type,dtype=self.precision):
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

            q_raw, k_raw, v = self.qkv_producer(x)

            # mup coord checking
            q_raw = self.query(q_raw)
            k_raw = self.key(k_raw)
            v = self.value(v)

            # TODO maybe just always gradient checkpoint the a_producer?
            A = self.a_producer(q_raw, k_raw)

            # mup coord checking
            A = self.attn_score(A)

            A = A.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if self.config.stabilize_attn_scores:
                A = A - A.max(dim=-1, keepdim=True)[0]

            att = F.softmax(A, dim=-1)
            hidden_state_output = self.av_combiner(att, v)

            # ok now let's sanity check compare this to SDPA
            if self.config.override_use_sdpa:
                B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
                # calculate query, key, values for all heads in batch and move head forward to be the batch dim
                # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
                # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
                qkv = self.qkv_producer.c_attn(x)
                q, k, v = qkv.split(self.config.n_head * self.config.head_dim, dim=2)
                k = k.view(B, T, self.config.n_head, self.config.head_dim).transpose(1, 2) # (B, nh, T, hs)
                q = q.view(B, T, self.config.n_head, self.config.head_dim).transpose(1, 2) # (B, nh, T, hs)
                v = v.view(B, T, self.config.n_head, self.config.head_dim).transpose(1, 2) # (B, nh, T, hs)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True,scale=self.attn_mult) # flash attention
                y = y.transpose(1, 2).contiguous().view(B, T, self.config.n_head * self.config.head_dim) # re-assemble all head outputs side by side
                y = self.av_combiner.c_proj(y)
                return y, None, None


            return hidden_state_output, None, None

    def forward(self, x, ff_cache=None, old_raw_att=None):
        if self.config.dense_attention_config.ckpt_attn:
            # Create a wrapper that only takes x as input since checkpoint doesn't handle None args well
            def create_custom_forward(module, ff_cache, old_raw_att):
                def custom_forward(x):
                    return module._forward(x, ff_cache, old_raw_att)
                return custom_forward
            
            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self, ff_cache, old_raw_att),
                x,
                use_reentrant=False  # More stable non-reentrant mode
            )
        else:
            return self._forward(x, ff_cache, old_raw_att)

class ProtectionKind(StrEnum):
    HEAD_TWO = auto()
    HEAD_TWO_FP64 = auto()
    LINEAR_COMBO = auto()
    LINEAR_COMBO_HEAD_TWO = auto()
    LEAKY_RELU = auto()
    ZERO = auto()
    ZERO_FP64 = auto()
    NONE = auto()
    NONE_CUSTOM_CUMSUM = auto()
    NONE_CUSTOM_CUMSUM_PARALLEL = auto()
    BIG_CONSTANT = auto()
    NONE_CUSTOM_CUMSUM_BLIASSON = auto()
    NONE_CUSTOM_CUMSUM_BLIASSON_FP64 = auto()
    NONE_TORCH_CUMSUM_FP64 = auto()

class SelectionHeadLinearComboKind(StrEnum):
    NONE = auto()
    TRUE = auto()
    WITH_HEAD_ZERO = auto()
    WITH_HEAD_ZERO_AND_BIAS = auto()
    ONE_MASK_PER_HEAD = auto()
    TWO_MASKS = auto()
    N_SLICED_MASKS = auto()
    N_LATENT_MASKS = auto()
    NONE_WITH_NO_HEAD = auto()
    ATT_CONV_N_LATENT_MASKS = auto()

class AttConvInit(StrEnum):
    NONE = auto()
    EYE = auto()
    DOUBLE_EYE = auto()
    CLIPPED_EYE = auto()
@dataclass
class FFCacheEntry:
    selection_masks: torch.Tensor
    attn_masks: torch.Tensor
    latent_masks: Optional[torch.Tensor] = None
class CausalSelectiveSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.IS_CUSTOM_ATTENTION = True

        # double the number of heads if we're using one mask per head
        # yes, this is very inefficient. but it's just for testing.
        if config.selection_head_linear_combo == SelectionHeadLinearComboKind.ONE_MASK_PER_HEAD:
            self.n_c_attn_heads = config.n_head*2
        elif config.selection_head_linear_combo == SelectionHeadLinearComboKind.NONE_WITH_NO_HEAD:
            self.n_c_attn_heads = config.n_head + 1
        elif config.selection_head_linear_combo == SelectionHeadLinearComboKind.TWO_MASKS:
            self.n_c_attn_heads = config.n_head + 2
        elif config.selection_head_linear_combo == SelectionHeadLinearComboKind.N_SLICED_MASKS:
            if config.one_head_per_latent_mask:
                self.n_c_attn_heads = config.n_head + config.n_sliced_masks
            else:
                self.n_c_attn_heads = config.n_head + 1
        elif config.selection_head_linear_combo == SelectionHeadLinearComboKind.N_LATENT_MASKS:
            if config.one_head_per_latent_mask:
                self.n_c_attn_heads = config.n_head + config.n_latent_masks
            else:
                self.n_c_attn_heads = config.n_head + 1
        else:
            self.n_c_attn_heads = config.n_head
        

        if config.att_conv:
            assert config.att_conv == (config.selection_head_linear_combo == SelectionHeadLinearComboKind.ATT_CONV_N_LATENT_MASKS), "att_conv must be True iff selection_head_linear_combo is ATT_CONV_N_LATENT_MASKS"
            if config.selection_head_linear_combo == SelectionHeadLinearComboKind.ATT_CONV_N_LATENT_MASKS:
                assert config.n_head == self.n_c_attn_heads, "n_head must be equal to n_c_attn_heads"

                self.att_conv_heads = config.n_head + config.n_latent_masks
            else:
                self.att_conv_heads = config.n_head
        else:
            self.att_conv_heads = None

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * self.n_c_attn_heads * config.head_dim)
        # output projection
        n_head_for_c_proj = config.n_head
        self.c_proj = nn.Linear(n_head_for_c_proj * config.head_dim, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.protect_bos_token = config.protect_bos_token
        self.prevent_from_masking_myself = config.prevent_from_masking_myself
        self.config = config

        self.attn_mult = config.attn_mult
        self.attn_score = nn.Identity() # for mup coord checking
        self.query = nn.Identity()
        self.key = nn.Identity()
        self.value = nn.Identity()

        if self.config.selection_head_linear_combo in [SelectionHeadLinearComboKind.ONE_MASK_PER_HEAD, SelectionHeadLinearComboKind.TWO_MASKS, SelectionHeadLinearComboKind.N_SLICED_MASKS]:
            self.selection_head = None
        elif self.config.selection_head_linear_combo in [SelectionHeadLinearComboKind.N_LATENT_MASKS, SelectionHeadLinearComboKind.ATT_CONV_N_LATENT_MASKS]:
            self.selection_head = nn.Linear(config.n_latent_masks, config.n_head, bias=not self.config.disable_selection_head_linear_combo_bias)
            if self.config.init_latent_masks_to_identity:
                self.selection_head.NANOGPT_ONES_INIT = True
            elif self.config.init_latent_masks_to_inverse:
                self.selection_head.NANOGPT_INVERSE_INIT = True
            if self.config.S_layernorm:
                self.S_layernorm = nn.LayerNorm(config.n_head)
        elif self.config.selection_head_linear_combo in [SelectionHeadLinearComboKind.WITH_HEAD_ZERO, SelectionHeadLinearComboKind.WITH_HEAD_ZERO_AND_BIAS, SelectionHeadLinearComboKind.TRUE]:
            use_bias = self.config.selection_head_linear_combo in [SelectionHeadLinearComboKind.WITH_HEAD_ZERO_AND_BIAS]
            self.selection_head = nn.Linear(config.n_head, 1, bias=use_bias)
            if self.config.selection_head_linear_combo in [SelectionHeadLinearComboKind.WITH_HEAD_ZERO, SelectionHeadLinearComboKind.WITH_HEAD_ZERO_AND_BIAS]:
                self.selection_head.ONE_HOT_INIT = 0
        elif self.config.selection_head_linear_combo in [SelectionHeadLinearComboKind.NONE_WITH_NO_HEAD, SelectionHeadLinearComboKind.NONE]:
            self.selection_head = None
        else:
            raise ValueError(f"Invalid selection head linear combo: {self.config.selection_head_linear_combo}")
        
        self.latent_mask_precision = torch.float32 if self.config.latent_mask_precision == "float32" else torch.bfloat16
        self.att_conv_precision = torch.float32 if self.config.att_conv_precision == "float32" else torch.bfloat16
        self.precision = torch.float32 if self.config.attn_precision == "float32" else torch.bfloat16
        
        if self.config.protection_kind in [ProtectionKind.LINEAR_COMBO, ProtectionKind.LINEAR_COMBO_HEAD_TWO]:
            self.protection_head = nn.Linear(config.n_head, 1)
            if self.config.protection_kind == ProtectionKind.LINEAR_COMBO_HEAD_TWO:
                self.protection_head.ONE_HOT_INIT = 1
        else:
            self.protection_head = None
        
        assert (self.config.leaky_relu_alpha is None) == (self.config.protection_kind != ProtectionKind.LEAKY_RELU) == (self.config.leaky_relu_bias is None), "leaky_relu_alpha, protection_kind, and leaky_relu_bias must all match"

        if self.config.mask_layernorm:
            self.mask_layernorm = nn.LayerNorm((config.n_head, config.block_size))
        else:
            self.mask_layernorm = None
        
        if self.config.residual_attention_masks:
            self.raw_att_head = nn.Linear(self.n_c_attn_heads, self.n_c_attn_heads, bias=False) # transforming old raw attention heads to new ones
        else:
            self.raw_att_head = None

        if self.config.selection_head_linear_combo == SelectionHeadLinearComboKind.N_LATENT_MASKS:
            if self.config.one_head_per_latent_mask:
                self.head_split_factor = 1
            else:
                self.head_split_factor = self.config.n_latent_masks
        else:
            if self.config.one_head_per_latent_mask:
                self.head_split_factor = 1
            elif self.config.selection_head_linear_combo == SelectionHeadLinearComboKind.N_SLICED_MASKS:
                self.head_split_factor = self.config.n_sliced_masks
            else:
                self.head_split_factor = 1
        assert self.head_dim % self.head_split_factor == 0, "head_dim must be divisible by n_sliced_masks or n_latent_masks"
    
        if self.config.att_conv:
            # make a linear from n_c_attn_heads to n_c_attn_heads
            assert self.head_split_factor == 1, "head_split_factor must be 1 for att_conv"
            assert self.n_c_attn_heads == config.n_head
            self.att_conv = nn.Linear(config.n_heads, self.att_conv_heads, bias=False)
            if self.config.att_conv_init == AttConvInit.EYE:
                self.att_conv.EYE_INIT = 1
            elif self.config.att_conv_init == AttConvInit.DOUBLE_EYE:
                self.att_conv.DOUBLE_EYE_INIT = 1
            elif self.config.att_conv_init == AttConvInit.CLIPPED_EYE:
                self.att_conv.CLIPPED_EYE_INIT = 1
        else:
            self.att_conv = None
    
    # to preserve precision. honestly not sure if this is necessary!
    @torch._dynamo.disable
    def n_latent_masks_linear(self,S_latent):
        S = self.selection_head(S_latent) # shape: (B, T, T', nh)

        S = S.transpose(1, 3) # shape: (B, nh, T, T')
        return S
    
    @torch._dynamo.disable
    def relu_graph_break(self,S):
        return F.relu(S)

    def forward(self, x,ff_cache=None,old_raw_att=None):
        with torch.autocast(device_type=x.device.type,dtype=self.precision):
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_c_attn_heads * self.head_dim, dim=2)

            # mup - just the identity
            q = self.query(q)
            k = self.key(k)
            v = self.value(v)

            v = v.view(B, T, self.n_c_attn_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

            if self.config.selection_head_linear_combo in [SelectionHeadLinearComboKind.N_SLICED_MASKS, SelectionHeadLinearComboKind.N_LATENT_MASKS]:
                k = k.view(B, T, self.n_c_attn_heads * self.head_split_factor, self.head_dim // self.head_split_factor).transpose(1, 2) # (B, nh, T, hs)
                q = q.view(B, T, self.n_c_attn_heads * self.head_split_factor, self.head_dim // self.head_split_factor).transpose(1, 2) # (B, nh, T, hs)
            else:
                k = k.view(B, T, self.n_c_attn_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
                q = q.view(B, T, self.n_c_attn_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
            

            # Standard attention computation
            att = (q @ k.transpose(-2, -1)) * self.attn_mult
            att = self.attn_score(att)

            if self.att_conv is not None:
                with torch.autocast(device_type=att.device.type,dtype=self.att_conv_precision):
                    att = att.transpose(1,3) # shape: (B, T, T', nh)
                    att = self.att_conv(att) # shape: (B, T, T', nh)
                    att = att.transpose(1,3) # shape: (B, nh, T, T')

            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

            if self.config.residual_attention_masks:
                if old_raw_att is not None:
                    old_raw_att = old_raw_att.masked_fill(self.bias[:,:,:T,:T] == 0, float('0'))
                    old_raw_att_reshaped = old_raw_att.transpose(1,3)
                    att = self.raw_att_head(old_raw_att_reshaped).transpose(1,3).masked_fill(self.bias[:,:,:T,:T] == 0, 0) + att 
                raw_att = att
            else:
                raw_att = None

            # Apply selective attention

            n_head = self.n_head

            if not self.config.disable_selection:

                S_latent = None

                if self.config.selection_head_linear_combo == SelectionHeadLinearComboKind.ONE_MASK_PER_HEAD:
                    # ok let's split att into two halves: the S and the real att
                    S = att[:, :self.n_head, :, :].clone()
                    att = att[:, self.n_head:, :, :]
                    v = v[:, self.n_head:, :, :]
                
                elif self.config.selection_head_linear_combo == SelectionHeadLinearComboKind.TWO_MASKS:
                    # ok let's split att into two halves: the S and the real att
                    with torch.autocast(device_type=att.device.type,dtype=self.latent_mask_precision):
                        S = att[:, :2, :, :].clone()
                        S_latent = S.transpose(1,3)
                        att = att[:, 2:, :, :]
                        v = v[:, 2:, :, :]
                
                elif self.config.selection_head_linear_combo == SelectionHeadLinearComboKind.N_SLICED_MASKS:
                    with torch.autocast(device_type=att.device.type,dtype=self.latent_mask_precision):
                        S = att[:, :self.config.n_sliced_masks, :, :] # shape: (B, n_sliced_masks, T, T')
                        att = att[:, self.config.n_sliced_masks:, :, :] # shape: (B, nh*(n_sliced_masks-1), T, T')
                        att = att.view(B, self.n_head, self.head_split_factor, T, T).sum(dim=2) # shape: (B, nh, T, T')

                        if self.config.one_head_per_latent_mask:
                            v = v[:, self.config.n_sliced_masks:, :, :]
                        else:
                            v = v[:, 1:, :, :]

                elif self.config.selection_head_linear_combo == SelectionHeadLinearComboKind.ATT_CONV_N_LATENT_MASKS:
                    with torch.autocast(device_type=att.device.type,dtype=self.latent_mask_precision):

                        S_latent = att[:, -self.config.n_latent_masks:, :, :] # shape: (B, n_latent_masks, T, T')
                        T_seq = S_latent.shape[2]
                        S_latent = S_latent.masked_fill(self.bias[:,:,:T_seq,:T_seq] == 0, 0) # shape: (B, T, T', n_latent_masks)
                        S_latent = S_latent.transpose(1, 3) # shape: (B, T, T', n_latent_masks)

                        S = self.n_latent_masks_linear(S_latent)

                        if self.config.latent_mask_runtime_multiplier is not None:
                            S = S * self.config.latent_mask_runtime_multiplier

                        att = att[:, :-self.config.n_latent_masks, :, :] # shape: (B, nh*(n_latent_masks-1), T, T')
                        # we're keeping this sum in - I think it might improve numerics?
                        att = att.view(B, self.n_head, self.head_split_factor, T, T).sum(dim=2) # shape: (B, nh, T, T')
                
                elif self.config.selection_head_linear_combo == SelectionHeadLinearComboKind.N_LATENT_MASKS:
                    with torch.autocast(device_type=att.device.type,dtype=self.latent_mask_precision):

                        S_latent = att[:, :self.config.n_latent_masks, :, :] # shape: (B, n_latent_masks, T, T')
                        T_seq = S_latent.shape[2]
                        S_latent = S_latent.masked_fill(self.bias[:,:,:T_seq,:T_seq] == 0, 0) # shape: (B, T, T', n_latent_masks)
                        S_latent = S_latent.transpose(1, 3) # shape: (B, T, T', n_latent_masks)

                        S = self.n_latent_masks_linear(S_latent)

                        if self.config.latent_mask_runtime_multiplier is not None:
                            S = S * self.config.latent_mask_runtime_multiplier

                        att = att[:, self.config.n_latent_masks:, :, :] # shape: (B, nh*(n_latent_masks-1), T, T')
                        att = att.view(B, self.n_head, self.head_split_factor, T, T).sum(dim=2) # shape: (B, nh, T, T')

                        if self.config.one_head_per_latent_mask:
                            v = v[:, self.config.n_latent_masks:, :, :] # vs match. good.
                        else:
                            v = v[:, 1:, :, :] # vs match. good.
                    
                elif self.config.selection_head_linear_combo == SelectionHeadLinearComboKind.NONE_WITH_NO_HEAD:
                    num_heads_to_remove = self.config.n_latent_masks if self.config.one_head_per_latent_mask else 1

                    S =   att[:, 0:num_heads_to_remove,:,:].clone()  # Select head 0 logits (clone to avoid in-place modification issues)
                    att = att[:,num_heads_to_remove:,:,:]

                    v = v[:, num_heads_to_remove:, :, :]

                elif self.config.selection_head_linear_combo != SelectionHeadLinearComboKind.NONE:
                    S = att[:, :, :, :] # shape: (B, n_head, T, T')
                    S = S.transpose(1, 3) # shape: (B, T', T, n_head)
                    S = S.masked_fill(self.bias[0,:,:T,:T,None].transpose(1,2) == 0, 0) # shape: (B, T', T, n_head)
                    S = self.selection_head(S) # shape: (B, T', T, 1)

                    # I copy the output of the selection_head to a fresh tensor
                    # For some reason, when I don't do this, torch.compile causes a CUDA memory alignment error
                    # That's not fixed by .contiguous() or .reshape() or .clone()
                    # But is fixed by i.e. F.sigmoid or by using this copy code
                    S_fresh = torch.zeros(S.shape, device=S.device)
                    S_fresh[:,:,:,:] = S[:,:,:,:]
                    S = S_fresh

                    bias_reshaped = self.bias[0,:,:T,:T].transpose(1,3) # shape: (1, T', T, 1)
                    S = S.masked_fill(bias_reshaped == 0, 0) # shape: (B, T', T, 1)
                    S = S.transpose(1,3).clone() # shape: (B, 1, T, T')
                else:
                    S = att[:, 0:1,:,:].clone()  # Select head 0 logits (clone to avoid in-place modification issues)

                S_pre_relu = S
                S = self.relu_graph_break(S)  # Only positive selection

                # Use torch.zeros_like to safely modify without inplace ops
                S_masked = torch.zeros_like(S)  # Create a mask to avoid in-place ops
                if self.protect_bos_token:
                    S_masked[..., 1:] = S[..., 1:]  # Do not mask <BOS> token, leave it unchanged
                else:
                    S_masked[...] = S

                eye_mask = 1 - torch.eye(T, device=S.device)  # Do not let me hide myself from future tokens
                if self.prevent_from_masking_myself:
                    S = S_masked * eye_mask
                else:
                    S = S_masked

                S_64 = S.to(torch.float64)
                FF_64 = torch.cumsum(S_64, dim=-2)
                if self.config.protection_kind == ProtectionKind.NONE:
                    FF = torch.cumsum(S, dim=-2)
                elif self.config.protection_kind == ProtectionKind.NONE_CUSTOM_CUMSUM:
                    from .protection.protect_and_attack import cumsum_triton
                    FF = cumsum_triton(S, dim=-2)
                elif self.config.protection_kind == ProtectionKind.NONE_CUSTOM_CUMSUM_PARALLEL:
                    from .protection.protect_and_attack import cumsum_triton
                    FF_64 = cumsum_triton(S, dim=-2, parallel_scan=True)
                    FF = FF_64.to(torch.float32)
                elif self.config.protection_kind == ProtectionKind.NONE_CUSTOM_CUMSUM_BLIASSON:
                    from .protection.protect_and_attack import cumsum_bliasson
                    FF = cumsum_bliasson(S, dim=-2)
                elif self.config.protection_kind == ProtectionKind.NONE_CUSTOM_CUMSUM_BLIASSON_FP64:
                    from .protection.protect_and_attack import cumsum_bliasson
                    FF = cumsum_bliasson(S.to(torch.float64),dim=-2,dtype=torch.float64).to(torch.float32)
                elif self.config.protection_kind == ProtectionKind.NONE_TORCH_CUMSUM_FP64:
                    FF = torch.cumsum(S.to(torch.float64), dim=-2).to(torch.float32)
                else:
                    # First, compute Sp
                    from .protection.protect_and_attack import attack_and_protect_bliasson

                    if self.config.protection_kind in [ProtectionKind.HEAD_TWO, ProtectionKind.HEAD_TWO_FP64]:
                        Sp = att[:, 1].clone()
                        Sp = F.relu(Sp)
                    elif self.config.protection_kind in [ProtectionKind.LINEAR_COMBO, ProtectionKind.LINEAR_COMBO_HEAD_TWO]:
                        Sp = att[:,:,:,:] # shape: (B, n_head, T, T')
                        Sp = Sp.transpose(1, 3) # shape: (B, T', T, n_head)
                        Sp = Sp.masked_fill(self.bias[0,:,:T,:T,None].transpose(1,2) == 0, 0) # shape: (B, T', T, n_head)
                        Sp = self.protection_head(Sp) # shape: (B, T', T, 1)

                        # Same copy trick as for the selection head above
                        Sp_fresh = torch.zeros((B, T, T), device=Sp.device)
                        Sp_fresh[:,:,:] = Sp[:,:,:,0]
                        Sp = Sp_fresh

                        Sp = Sp.masked_fill(self.bias[0,:,:T,:T].transpose(1,2) == 0, 0) # shape: (B, T', T, 1)
                        Sp = Sp.squeeze(-1) # shape: (B, T', T)
                        Sp = Sp.transpose(1,2) # shape: (B, T, T')
                        Sp = F.relu(Sp)
                    elif self.config.protection_kind == ProtectionKind.LEAKY_RELU:
                        # we use the "leaky" half of S_pre_relu
                        # This makes our model act as if we used a leaky relu to construct S, BUT with the negative half acting like protection, NOT like healing
                        Sp = (-S_pre_relu * self.config.leaky_relu_alpha + self.config.leaky_relu_bias).relu()
                    elif self.config.protection_kind in [ProtectionKind.ZERO, ProtectionKind.ZERO_FP64]:
                        Sp = torch.zeros_like(S)
                    elif self.config.protection_kind == ProtectionKind.BIG_CONSTANT:
                        Sp = torch.ones_like(S).fill_(50)
                    else:
                        raise NotImplementedError(f"Protection kind {self.config.protection_kind} not implemented")
                    
                    if self.config.protection_head_scaling_factor != 1.0:
                        Sp = Sp * self.config.protection_head_scaling_factor
                    if self.config.protection_head_bias != 0.0:
                        Sp = F.relu(Sp + self.config.protection_head_bias)
                    
                    Sp = Sp[:,None,:,:]

                    # Second, run the protect-and-attack algorithm on Sp and S
                    if self.config.protection_kind in [ProtectionKind.ZERO_FP64, ProtectionKind.HEAD_TWO_FP64]:
                        FF = attack_and_protect_bliasson(S, Sp, dim=-2, dtype=torch.float64) * -1
                        assert FF.dtype == torch.float32
                    else:
                        FF = attack_and_protect_bliasson(S, Sp, dim=-2) * -1

                    # if self.config.protection_kind == ProtectionKind.BIG_CONSTANT:
                    #     assert (FF == 0).all(), "FF should be 0"
                    # elif self.config.protection_kind == ProtectionKind.ZERO:
                    #     assert torch.allclose(FF, torch.cumsum(S, dim=-2)), "FF should be equal to the cumulative sum of S"
                
                # sanity checking section
                # here, we assert that FF is very close to FF_64
                # protection kind=None should be *pretty* close. like 1e-5 seems reasonable.

                threshold = 0.01 if os.environ.get("IS_MINI_MODEL", None) == "true" else 0.0001
                if os.environ.get("DEBUG_CUM_SUM", None) == "true" and random.random() < threshold and self.training:
                    with torch.no_grad():
                        gt_FF_64 = torch.cumsum(S_64.cpu(), dim=-2)
                        max_diff = (FF.cpu() - gt_FF_64).abs().max()
                        print(f"Max diff between FF and gt_FF_64: {max_diff.item()}")
                        import wandb
                        try:
                            wandb.log({"max_diff": max_diff.item()})
                        except:
                            pass

                        # inputs_causing_instability.append((S_64.cpu().detach().numpy(), max_diff.item()))
                        # let's actually just save it to a file
                        # torch.save(inputs_causing_instability, "inputs_causing_instability.pt")

                FF_shifted = torch.roll(FF, 1, -2)
                FF_shifted[..., 0, :] = 0

                if ff_cache is not None:
                    ff_cache.append(FFCacheEntry(selection_masks=FF_shifted.float().detach().cpu().numpy(), attn_masks=att.detach().cpu().numpy(), latent_masks=S_latent.transpose(1,3).detach().cpu().numpy() if S_latent is not None else None))

                # Use out-of-place subtraction to preserve computation graph integrity

                head_split_factor = FF_shifted.shape[1]
                n_head = att.shape[1]
                if head_split_factor < n_head:
                    FF_shifted_even = FF_shifted.repeat_interleave(n_head // head_split_factor, dim=1)[:,:n_head,:,:]
                    if n_head % head_split_factor == 0:
                        FF_shifted = FF_shifted_even
                    else:
                        FF_shifted = torch.cat([FF_shifted_even, FF_shifted[:,:n_head % head_split_factor,:,:]], dim=1)

                if self.config.mask_layernorm:
                    FF_shifted = self.mask_layernorm(FF_shifted.transpose(1, 2)).transpose(1, 2) / torch.arange(T,0,-1,device=FF_shifted.device)[None,None,None,:]

                att = att - FF_shifted[:,:,:,:]
            else:
                if ff_cache is not None:
                    ff_cache.append(FFCacheEntry(selection_masks=torch.zeros_like(att,dtype=torch.float32).detach().cpu().numpy(), attn_masks=att.float().detach().cpu().numpy(), latent_masks=None))

            att = F.softmax(att, dim=-1)

            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, n_head * self.head_dim)
            y = self.c_proj(y)
            return y, None, raw_att

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.IS_CUSTOM_ATTENTION = True
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_head * config.head_dim)
        # output projection
        self.c_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.attn_mult = config.attn_mult
        self.sdpa_iter_size = config.sdpa_iter_size

    def forward(self, x,ff_cache=None,old_raw_att=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_head * self.head_dim, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        if self.sdpa_iter_size is not None and B % self.sdpa_iter_size == 0:
            # we will handle i.e. 16 batch elements at a time
            split_q = q.split(self.sdpa_iter_size, dim=0)
            split_k = k.split(self.sdpa_iter_size, dim=0)
            split_v = v.split(self.sdpa_iter_size, dim=0)
            ys = [F.scaled_dot_product_attention(split_q[i], split_k[i], split_v[i], is_causal=True,scale=self.attn_mult) # flash attention
                  for i in range(len(split_q))]
            y = torch.cat(ys, dim=0)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,scale=self.attn_mult) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y, None, None

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
        self.protect_bos_token = config.protect_bos_token
        self.prevent_from_masking_myself = config.prevent_from_masking_myself

        if self.config.selection_head_linear_combo != SelectionHeadLinearComboKind.NONE:
            self.selection_head = nn.Linear(config.n_head, 1)
        else:
            self.selection_head = None
        
        if self.config.protection_kind != ProtectionKind.NONE:
            raise NotImplementedError("Protection not implemented for inference")

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

    def forward(self, x,ff_cache=None, cache_key=None):
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
        if False and self.config.linear_combo:
            S = att[:, :, :, :] # shape: (B, n_head, T, T')
            S = S.transpose(1, 3) # shape: (B, T', T, n_head)
            S = self.selection_head(S) # shape: (B, T', T, 1)
            S = S.squeeze(-1) # shape: (B, T', T)
            S = S.transpose(1,2) # shape: (B, T, T')
        else:
            S = att[:, 0].clone()

        S = F.relu(S)

        S_masked = torch.zeros_like(S)
        if self.protect_bos_token:
            S_masked[..., 1:] = S[..., 1:]
        else:
            S_masked[...] = S

        eye_mask = 1 - torch.eye(T, device=S.device)
        if self.prevent_from_masking_myself:
            S = S_masked * eye_mask
        else:
            S = S_masked

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
        self.protect_bos_token = config.protect_bos_token
        self.prevent_from_masking_myself = config.prevent_from_masking_myself

        if self.config.selection_head_linear_combo != SelectionHeadLinearComboKind.NONE: raise NotImplementedError("Linear combo not implemented for memory penalty")
        if self.config.protection_kind != ProtectionKind.NONE: raise NotImplementedError("Protection not implemented for memory penalty")

    def forward(self, x,ff_cache=None):
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
        if self.protect_bos_token:
            S_masked[..., 1:] = S[..., 1:]
        else:
            S_masked[...] = S

        eye_mask = 1 - torch.eye(T, device=S.device)
        if self.prevent_from_masking_myself:
            S = S_masked * eye_mask
        else:
            S = S_masked

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

from enum import StrEnum, auto

class AttentionKind(StrEnum):
    SELECTIVE = auto()
    SELECTIVE_WITH_MEMORY_PENALTY = auto()
    SELF = auto()
    DENSE = auto()

def get_attention_cls(kind: AttentionKind, for_inference: bool) -> nn.Module:
    if kind == AttentionKind.DENSE:
        return CausalDenseSelfAttention
    elif kind == AttentionKind.SELECTIVE:
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
