from torch import nn
import torch
import math
import torch.nn.functional as F
from dataclasses import dataclass
from .attn import get_attention_cls, AttentionKind, ProtectionKind
import os
import inspect

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

    def forward(self, x):
        # Get both attention output and memory requirements
        attn_out, M = self.attn(self.ln_1(x))
        assert attn_out.shape == x.shape
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, M

from typing import Optional
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
    selection_head_linear_combo: bool = False
    selection_head_linear_combo_scale: float = 1.0
    protection_kind: Optional[ProtectionKind] = None
    leaky_relu_alpha: Optional[float] = None
    leaky_relu_bias: Optional[float] = None
    protection_head_scaling_factor: float = 1.0
    protection_head_bias: float = 0.0

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
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'ONE_HOT_INIT'):
                torch.nn.init.zeros_(module.weight)
                module.weight.data[:,module.ONE_HOT_INIT] = 1.0
                return
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Get embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        # Collect memory requirements from all layers
        memory_reqs = []
        for i, block in enumerate(self.transformer.h):
            x, M = block(x)
            memory_reqs.append(M)

        # Final layer norm and get logits
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate losses
        loss = None
        losses = {}
        if targets is not None:
            # Calculate standard cross-entropy loss (L_ppl)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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

            if loss.isnan().any():
                raise Exception("Oh no! Loss is nan!")

        return logits, loss, losses

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
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and not "selection_head" in n]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and not "selection_head" in n]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': selection_head_params, 'weight_decay': 0.0, 'lr': learning_rate * self.config.selection_head_linear_combo_scale}
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
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
