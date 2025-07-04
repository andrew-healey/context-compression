from torch import nn
import torch
import math
import torch.nn.functional as F
from dataclasses import dataclass, field
from .attn import get_attention_cls, AttentionKind, ProtectionKind, SelectionHeadLinearComboKind, AttConvInit, DenseAttentionKind, QKVProducerKind, AProducerKind, AVCombinerKind
import os
import inspect
from .attn import CausalSelectiveSelfAttention
from mup import MuReadout

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = get_attention_cls(config.attention_kind, config.for_inference)(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x,ff_cache=None,old_raw_att=None):
        # Get both attention output and memory requirements
        attn_out, M, raw_att = self.attn(self.ln_1(x),ff_cache=ff_cache,old_raw_att=old_raw_att)
        assert attn_out.shape == x.shape
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, M, raw_att

from typing import Optional

@dataclass
class DenseAttentionConfig:
    head_dim_value: int = 64
    dense_attention_kind: DenseAttentionKind = field(default_factory=lambda: DenseAttentionKind.MHA)
    ckpt_attn: bool = False

    qkv_producer_kind: QKVProducerKind = field(default_factory=lambda: QKVProducerKind.INHERIT)
    a_producer_kind: AProducerKind = field(default_factory=lambda: AProducerKind.INHERIT)
    av_combiner_kind: AVCombinerKind = field(default_factory=lambda: AVCombinerKind.INHERIT)

@dataclass
class GPTConfig:
    attention_kind: AttentionKind
    for_inference: bool
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    head_dim: int = 64
    n_embd: int = 768
    epsilon: float = 0.1  # Weight for memory loss term
    hard_pruning_constant: Optional[float] = None # to fix the pruning during inference
    protect_bos_token: bool = True
    prevent_from_masking_myself: bool = True

    selection_head_linear_combo: SelectionHeadLinearComboKind = SelectionHeadLinearComboKind.NONE
    selection_head_linear_combo_scale: float = 1.0
    disable_selection_head_linear_combo_bias: bool = False
    assert_latent_matches_no_head: bool = False
    n_sliced_masks: Optional[int] = None
    n_latent_masks: Optional[int] = None
    init_latent_masks_to_identity: bool = False
    init_latent_masks_to_inverse: bool = False
    latent_mask_scale: Optional[float] = None
    latent_mask_runtime_multiplier: Optional[float] = None
    latent_mask_sigmoid: bool = False
    latent_mask_precision: str = "bfloat16"
    one_head_per_latent_mask: bool = False


    protection_kind: ProtectionKind = ProtectionKind.NONE
    leaky_relu_alpha: Optional[float] = None
    leaky_relu_bias: Optional[float] = None
    protection_head_scaling_factor: float = 1.0
    protection_head_bias: float = 0.0
    mask_layernorm: bool = False
    residual_attention_masks: bool = False
    disable_selection: bool = False
    use_hf_style_inputs: bool = False
    mup: bool = False
    attn_mult: Optional[float] = None
    readout_zero_init: bool = False # deprecated
    query_zero_init: bool = False # deprecated
    mup_zero_init: bool = False
    l1_loss: bool = False
    S_layernorm: bool = False
    att_conv: bool = False
    att_conv_init: AttConvInit = AttConvInit.NONE
    att_conv_scale: float = 1.0
    att_conv_precision: str = "bfloat16"
    att_conv_weight_decay: bool = True  # Whether to apply weight decay to attention convolution parameters

    attn_precision: str = "bfloat16"

    dense_attention_config: DenseAttentionConfig = field(default_factory=lambda: DenseAttentionConfig())

    sdpa_iter_size: Optional[int] = None
    stabilize_attn_scores: bool = False
    override_use_sdpa: bool = False
    simulate_micro_bs: Optional[int] = None
    c_proj_scale_init: Optional[float] = None

    def __post_init__(self):
        if self.attn_mult is None:
            if self.mup:
                self.attn_mult = (self.head_dim)**-1.0
            else:
                self.attn_mult = (self.head_dim)**-0.5

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        if config.mup:
            from mup import MuReadout
            lm_head_cls = MuReadout
        else:
            lm_head_cls = nn.Linear

        self.lm_head = lm_head_cls(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, MuReadout) and self.config.readout_zero_init:
            module.weight.data.zero_()
        elif isinstance(module, nn.Linear):
            if hasattr(module, 'ONE_HOT_INIT'):
                torch.nn.init.zeros_(module.weight)
                module.weight.data[:,module.ONE_HOT_INIT] = 1.0
                return
            elif hasattr(module, 'NANOGPT_ONES_INIT'):
                torch.nn.init.ones_(module.weight)
                if self.config.latent_mask_scale is not None:
                    module.weight.data.div_(self.config.latent_mask_scale)
                if not self.config.disable_selection_head_linear_combo_bias:
                    torch.nn.init.zeros_(module.bias)
                return
            elif hasattr(module, 'NANOGPT_INVERSE_INIT'):
                print("running inverse _init_weights")
                torch.nn.init.ones_(module.weight)
                module.weight.data.div_(self.config.n_latent_masks)
                if self.config.latent_mask_scale is not None:
                    module.weight.data.div_(self.config.latent_mask_scale)
                if not self.config.disable_selection_head_linear_combo_bias:
                    torch.nn.init.zeros_(module.bias)
                return
            elif hasattr(module, 'EYE_INIT'):
                torch.nn.init.eye_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                return
            elif hasattr(module, 'DOUBLE_EYE_INIT'):
                # blockwise diagonal matrix with 2x2 blocks
                n = module.weight.shape[0]
                for i in range(n//2):
                    right_side = min(2*i+2, n)
                    module.weight.data[2*i:right_side,2*i:right_side] = torch.eye(right_side-2*i) / (right_side-2*i)
                return
            elif hasattr(module, 'CLIPPED_EYE_INIT'):
                print("running clipped eye init")
                # so it's not a square matrix. but we want to make sure every row and every column adds up to 1.
                # so here's what we're gonna do.
                with_more_rows_than_cols = module.weight.T if module.weight.shape[0] < module.weight.shape[1] else module.weight
                assert with_more_rows_than_cols.shape[0] >= with_more_rows_than_cols.shape[1], "with_more_rows_than_cols must have more rows than columns"
                for i in range(with_more_rows_than_cols.shape[0] - 1):
                    module.weight.data[i,i] = 1
                for i in range(with_more_rows_than_cols.shape[1],with_more_rows_than_cols.shape[0]):
                    module.weight.data[-1,i] = 1.0/(with_more_rows_than_cols.shape[0] - with_more_rows_than_cols.shape[1])
                
                # sanity check!
                for i in range(with_more_rows_than_cols.shape[0]):
                    assert with_more_rows_than_cols[i].sum() == 1.0
                for j in range(with_more_rows_than_cols.shape[1]):
                    assert with_more_rows_than_cols[:,i].sum() == 1.0

            std = 0.02

            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
                std *= module.NANOGPT_SCALE_INIT

            # set by mup
            if hasattr(module.weight,'infshape'):
                from mup import normal_
                normal_(module.weight,mean=0.0,std=std)
                module.weight.latest_init = {'std': std,'kind': 'mup','infshape': module.weight.infshape}
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                module.weight.latest_init = {'std': std,'kind': 'normal'}

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        if hasattr(module,'MUP_INIT_RANGE_TO_ZERO') and type(module.MUP_INIT_RANGE_TO_ZERO) == tuple and self.config.mup_zero_init:
            start,end = module.MUP_INIT_RANGE_TO_ZERO
            assert start < end and start >= 0 and end <= module.weight.shape[0]
            module.weight.data[start:end,:] = 0
        
        if hasattr(module,'IS_CUSTOM_ATTENTION') and self.config.query_zero_init:
            fanout, _ = module.c_attn.weight.shape
            assert fanout % 3 == 0
            module.c_attn.weight.data[:fanout//3, :] = 0
    
    def get_latest_inits(self):
        latest_inits = []
        def inner(module):
            if hasattr(module,'weight') and hasattr(module.weight,'latest_init'):
                latest_inits.append(module.weight.latest_init)
        
        self.apply(inner)
        return latest_inits

    def calculate_memory_loss(self, memory_reqs, T):
        """Calculate the memory regularization term"""
        # Get maximum memory requirement for each layer
        layer_maxes = torch.stack([M.max(dim=2)[0] for M in memory_reqs])  # L x B x n_head
        # Sum across layers
        total_memory = layer_maxes.sum(dim=0)  # B x n_head
        # Average across heads and batch
        avg_memory = total_memory.mean()
        # Normalize by sequence length and non-pad tokens (T in this case)
        memory_loss = avg_memory / (T)
        return memory_loss

    def forward(self, idx, targets=None,ff_cache=None):

        # if self.config.use_hf_style_inputs and type(idx) == dict:
        #     hf_style_inputs = idx
        #     assert type(hf_style_inputs) == dict, "hf_style_inputs must be a dict"
        #     assert "idx" in hf_style_inputs, "hf_style_inputs must contain an 'idx' key"
        #     idx = hf_style_inputs["idx"]
        #     targets = hf_style_inputs["targets"]
        #     assert ff_cache is None, "ff_cache must be None when using hf_style_inputs"
        #     assert targets is not None, "targets must not be None when using hf_style_inputs"

        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Get embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        # Collect memory requirements from all layers
        memory_reqs = []
        raw_att = None
        for i, block in enumerate(self.transformer.h):
            x, M, raw_att = block(x,ff_cache=ff_cache,old_raw_att=raw_att)
            memory_reqs.append(M)
        
        del raw_att

        # Final layer norm and get logits
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate losses
        loss = None
        losses = {}
        ce_loss_batched = None
        if targets is not None:
            # Calculate standard cross-entropy loss (L_ppl)
            ce_loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
            if self.config.simulate_micro_bs is not None:
                B = logits.size(0)
                assert B % self.config.simulate_micro_bs == 0
                n_repeats = B // self.config.simulate_micro_bs
                lhs = logits.view(n_repeats,-1, logits.size(-1)).split(1, dim=0)
                rhs = targets.view(n_repeats,-1).split(1, dim=0)
                ce_loss_batched = [F.cross_entropy(l.view(-1,l.size(-1)), r.view(-1)) for l, r in zip(lhs, rhs)]
            loss = ce_loss
            losses = {
                "ce": ce_loss,
            }

            # Combine losses: L_mem = L_ppl + ε * Σ(max_i(M_i^l))/(L * n_nonpad)
            if self.config.attention_kind != AttentionKind.SELF and any([M is not None for M in memory_reqs]):
                # Calculate memory loss
                memory_loss = self.calculate_memory_loss(memory_reqs, T)
                loss = loss + self.config.epsilon * memory_loss
                losses["memory"] = memory_loss
            losses["total"] = loss

            if self.config.l1_loss:
                losses["l1"] = self.l1_loss()
            if loss.isnan().any():
                raise Exception("Oh no! Loss is nan!")

        if self.config.use_hf_style_inputs:
            return {"logits": logits, "loss": loss, "losses": losses, "ce_loss_batched": ce_loss_batched}
        else:
            return logits, loss, losses, ce_loss_batched
        
    def l1_loss(self):
        with torch.no_grad():
            l1_loss = None
            weights = list(self.state_dict().values())
            for weight in weights:
                if l1_loss is None:
                    l1_loss = weight.abs().mean()
                else:
                    l1_loss = l1_loss + weight.abs().mean()
            return l1_loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        selection_head_params = [p for n, p in param_dict.items() if "selection_head" in n]
        att_conv_params = [p for n, p in param_dict.items() if "att_conv" in n]
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and not "selection_head" in n and not "att_conv" in n]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and not "selection_head" in n and not "att_conv" in n and not "raw_att_head" in n]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': selection_head_params, 'weight_decay': 0.0, 'lr': learning_rate * self.config.selection_head_linear_combo_scale},
            {'params': att_conv_params, 'weight_decay': weight_decay if self.config.att_conv_weight_decay else 0.0, 'lr': learning_rate * self.config.att_conv_scale}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_low_lr_selection_head_params = sum(p.numel() for p in selection_head_params)
        master_process = os.environ.get('RANK', 0) == 0
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            print(f"num low-lr selection head parameter tensors: {len(selection_head_params)}, with {num_low_lr_selection_head_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        
        if self.config.mup:
            from mup import MuAdamW
            config_cls = MuAdamW
        else:
            config_cls = torch.optim.AdamW
        optimizer = config_cls(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        # ok now let's lock the original lr into a special key, so we can decay them later
        for param_group in optimizer.param_groups:
            param_group['max_lr'] = param_group.get('lr',learning_rate)

        return optimizer
