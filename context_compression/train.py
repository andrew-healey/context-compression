# ANDREWTODO remove slow asserts in protect_and_attack.py before merging into master

import os
import math
import time
import inspect
import json
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import argparse
import shutil
from huggingface_hub import hf_hub_download, HfApi
import wandb
from typing import Optional
from functools import partial
from pathlib import Path

from .data import DataLoaderLite, get_most_likely_row
from .model import GPT, GPTConfig, DenseAttentionConfig
from .attn import AttentionKind, ProtectionKind, SelectionHeadLinearComboKind, AttConvInit, DenseAttentionKind
from .hellaswag import render_example, iterate_examples
from .add_a_head import AddHeadConfig, AddHeadKind, add_a_head, NewHeadInit

from enum import StrEnum, auto

class ProfileKind(StrEnum):
    NONE = auto()
    PYTORCH = auto()

# Parse command-line arguments for custom configuration
parser = argparse.ArgumentParser(description="Train GPT with context compression.")
parser.add_argument("--hellaswag", dest="hellaswag", action="store_true",
                    help="Enable HellaSwag evaluation (default: True)")
parser.set_defaults(hellaswag=True)
parser.add_argument("--no-hellaswag", dest="hellaswag", action="store_false",
                    help="Disable HellaSwag evaluation")

parser.add_argument("--attention_kind", type=lambda x: AttentionKind(x.lower()), default=AttentionKind.SELECTIVE,
                    help="Attention type to use (e.g., self, selective)")
parser.add_argument("--log_dir", type=str, required=True,
                    help="Directory to save logs and checkpoints")
parser.add_argument("--resume_checkpoint", type=str, default=None,
                    help="Checkpoint path to resume training from")
parser.add_argument("--resume_optimizer", action="store_true",
                    help="Resume optimizer state when resuming checkpoint")
parser.add_argument("--add_a_head", action="store_true",
                    help="Add an additional head")
parser.set_defaults(add_a_head=False)
parser.add_argument("--add_head_to_start", action="store_true",
                    help="Place the new head at the start")
parser.set_defaults(add_head_to_start=True)
parser.add_argument("--add_head_to_end", action="store_false", dest="add_head_to_start",
                    help="Place the new head at the end")
parser.add_argument("--new_head_init", type=lambda x: NewHeadInit(x.lower()), default=NewHeadInit.NORMAL,
                    help="Initialization type for the new head (e.g., normal, o_rescaled, o_zero, ko_zero)")
parser.add_argument("--n_heads", type=int, default=13,
                    help="Number of heads")
parser.add_argument("--n_embd", type=int, default=None,
                    help="Number of embeddings")
parser.add_argument("--head_dim", type=int, default=None,
                    help="Number of dimensions per head")
parser.add_argument("--protect_bos_token", action="store_true",
                    help="Protect the BOS token")
parser.set_defaults(protect_bos_token=True)
parser.add_argument("--no_protect_bos_token", dest="protect_bos_token", action="store_false",
                    help="Do not protect the BOS token")
parser.add_argument("--prevent_from_masking_myself", action="store_true",
                    help="Prevent the model from masking itself")
parser.add_argument("--allow_masking_myself", dest="prevent_from_masking_myself", action="store_false",
                    help="Allow each token to mask itself from future tokens")
parser.set_defaults(prevent_from_masking_myself=True)
parser.add_argument("--max_steps", type=int, default=None,
                    help="Maximum number of training steps")
parser.add_argument("--warmup_steps", type=int, default=None,
                    help="Number of warmup steps")
parser.add_argument("--group", type=str, default=None,
                    help="Group name for the run")
parser.add_argument("--no_use_wandb", dest="use_wandb", action="store_false",
                    help="Disable wandb logging")
parser.set_defaults(use_wandb=True)
parser.add_argument("--kill_self_after_run", action="store_true",
                    help="Kill my own instance after run completes")
parser.add_argument("--random_seed", type=int, default=1337,
                    help="Random seed for the run")
parser.add_argument("--memory_penalty_epsilon", type=float, default=0.1,
                    help="Epsilon for the memory penalty")


parser.add_argument("--selection_head_linear_combo", type=lambda x: SelectionHeadLinearComboKind(x.lower()), default=SelectionHeadLinearComboKind.NONE,
                    help="Whether to use a linear combo of attention scores for the selection head")
parser.add_argument("--selection_head_linear_combo_scale", type=float, default=1.0,
                    help="Constant scale (think of alpha in lora) for the selection head linear combo")
parser.add_argument("--disable_selection_head_linear_combo_bias", action="store_true",
                    help="Disable the bias for the selection head linear combo")
parser.set_defaults(disable_selection_head_linear_combo_bias=False)
parser.add_argument("--assert_latent_matches_no_head", action="store_true",
                    help="Assert that the n-latent-masks selection head matches the baseline behavior")
parser.set_defaults(assert_latent_matches_no_head=False)

### LATENT MASKS HERE

parser.add_argument("--n_latent_masks", type=int, default=None,
                    help="Number of latent masks per head")
parser.add_argument("--init_latent_masks_to_identity", action="store_true",
                    help="Initialize the latent masks to the identity matrix")
parser.set_defaults(init_latent_masks_to_identity=False)
parser.add_argument("--init_latent_masks_to_inverse", action="store_true",
                    help="Initialize the latent masks to the 1/n, where n is the number of latent masks")
parser.set_defaults(init_latent_masks_to_inverse=False)
parser.add_argument("--latent_mask_scale", type=float, default=None,
                    help="Initialization scale for the latent masks")
parser.add_argument("--latent_mask_runtime_multiplier", type=float, default=None,
                    help="Multiply latent masks by this constant at runtime")
parser.add_argument("--latent_mask_sigmoid", action="store_true",
                    help="Use tanh on the latent masks")
parser.set_defaults(latent_mask_sigmoid=False)
parser.add_argument("--S_layernorm", action="store_true",
                    help="Use layernorm on the mask")
parser.set_defaults(S_layernorm=False)

parser.add_argument("--one_head_per_latent_mask", action="store_true",
                    help="Use one head per latent mask")
parser.set_defaults(one_head_per_latent_mask=False)

parser.add_argument("--att_conv", action="store_true",
                    help="Use an attention conv")
parser.set_defaults(att_conv=False)
parser.add_argument("--att_conv_init", type=lambda x: AttConvInit(x.lower()), default=AttConvInit.NONE,
                    help="Initialization for the attention conv")
parser.add_argument("--att_conv_scale", type=float, default=1.0,
                    help="Lr scale for the attention conv")
parser.add_argument("--att_conv_weight_decay", action="store_true",
                    help="Apply weight decay to the attention conv")
parser.set_defaults(att_conv_weight_decay=False)

parser.add_argument("--protection_kind", type=lambda x: ProtectionKind(x.lower()), default=ProtectionKind.NONE,
                    help="Kind of protection to use")
parser.add_argument("--leaky_relu_alpha", type=float, default=None,
                    help="Alpha for the leaky relu")
parser.add_argument("--leaky_relu_bias", type=float, default=None,
                    help="Bias for the leaky relu")
parser.set_defaults(use_compile=True)
parser.add_argument("--no_use_compile", action="store_false", dest="use_compile",
                    help="Do not use torch.compile")
parser.add_argument("--use_mini_model", action="store_true",
                    help="Make the model and batch size very small, for fast debugging")
parser.add_argument("--upload_to_hf", action="store_true",
                    help="Upload the model to HuggingFace")
parser.add_argument("--no_upload_to_hf", action="store_false", dest="upload_to_hf",
                    help="Do not upload the model to HuggingFace")
parser.set_defaults(upload_to_hf=True)
parser.add_argument("--seq_len", type=int, default=None,
                    help="Sequence length")
parser.add_argument("--batch_size", type=int, default=None,
                    help="Batch size")
parser.add_argument("--total_batch_size", type=int, default=None,
                    help="Total batch size")
parser.add_argument("--protection_head_scaling_factor", type=float, default=1.0,
                    help="Scaling factor for the protection head")
parser.add_argument("--protection_head_bias", type=float, default=0.0,
                    help="Bias for the protection head")
parser.add_argument("--n_sliced_masks", type=int, default=None,
                    help="Number of sliced masks per head")
parser.add_argument("--mask_layernorm", action="store_true",
                    help="Use layernorm on the mask")
parser.set_defaults(mask_layernorm=False)
parser.add_argument("--residual_attention_masks", action="store_true",
                    help="Use residual attention masks")
parser.set_defaults(residual_attention_masks=False)
parser.add_argument("--compute_base_shapes", action="store_true",
                    help="Compute base shapes")
parser.set_defaults(compute_base_shapes=False)
parser.add_argument("--base_shapes_savefile", type=str, default=None,
                    help="File to save base shapes to")
parser.add_argument("--mup", action="store_true",
                    help="Use Maximum update parametrization")
parser.set_defaults(mup=False)
parser.add_argument("--disable_selection", action="store_true",
                    help="Disable selection")
parser.set_defaults(disable_selection=False)
parser.add_argument("--mup_enable_coord_check_logging", action="store_true",
                    help="Enable coordinate check logging")
parser.set_defaults(mup_enable_coord_check_logging=False)
parser.add_argument("--max_lr", type=float, default=None,
                    help="Maximum learning rate")
parser.add_argument("--no_decay_lr", action="store_false", dest="decay_lr",
                    help="Do not decay the learning rate")
parser.set_defaults(decay_lr=True)
parser.add_argument("--readout_zero_init", action="store_true",
                    help="Zero initialize the readout")
parser.set_defaults(readout_zero_init=False)
parser.add_argument("--query_zero_init", action="store_true",
                    help="Zero initialize the query")
parser.set_defaults(query_zero_init=False)
parser.add_argument("--mup_zero_init", action="store_true",
                    help="Zero initialize the mup parameters")
parser.set_defaults(mup_zero_init=False)
parser.add_argument("--l1_loss", action="store_true",
                    help="Compute L1 loss for debugging purposes")
parser.set_defaults(l1_loss=False)
parser.add_argument("--debugpy", action="store_true",
                    help="Enable debugpy")
parser.set_defaults(debugpy=False)
parser.add_argument("--key", type=str, default=None,
                    help="Key for the run") # for grouping runs with diff seeds but the same hyperparams
parser.add_argument("--latent_mask_precision", type=str, default="bfloat16",
                    help="Precision for the latent masks")
parser.add_argument("--att_conv_precision", type=str, default="bfloat16",
                    help="Precision for the attention conv")
parser.add_argument("--attn_precision", type=str, default="bfloat16",
                    help="Precision for the attention")
parser.add_argument("--profile_kind", type=lambda x:ProfileKind(x.lower()), default=ProfileKind.NONE,
                    help="What kind of profiling to do")

parser.add_argument("--dense_attention_kind", type=lambda x: DenseAttentionKind(x.lower()), default=DenseAttentionKind.MHA,
                    help="What kind of dense attention to use")
parser.add_argument("--head_dim_value", type=int, default=None,
                    help="Head dimension value for dense attention")

parser.add_argument("--sdpa_iter_size", type=int, default=None,
                    help="SDPA iteration size")
parser.add_argument("--stabilize_attn_scores", action="store_true",
                    help="Stabilize the attention scores")
parser.set_defaults(stabilize_attn_scores=False)
parser.add_argument("--override_use_sdpa", action="store_true",
                    help="Override the use of SDPA")
parser.set_defaults(override_use_sdpa=False)
parser.add_argument("--autocast_precision", type=str, default="bfloat16",
                    help="Precision for the autocast")
parser.add_argument("--simulate_micro_bs", type=int, default=None,
                    help="Simulate a smaller micro batch size than the one you're using")
parser.set_defaults(simulate_micro_bs=None)
parser.add_argument("--simulate_micro_bs_2", type=int, default=None,
                    help="Simulate a smaller micro batch size than the one you're using")
parser.set_defaults(simulate_micro_bs_2=None)
parser.add_argument("--ckpt_attn", action="store_true",
                    help="Checkpoint the attention")
parser.set_defaults(ckpt_attn=False)

from .attn import AProducerKind,QKVProducerKind,AVCombinerKind
parser.add_argument("--a_producer_kind", type=lambda x: AProducerKind(x.lower()), default=AProducerKind.MHA,
                    help="What kind of a producer to use")
parser.add_argument("--qkv_producer_kind", type=lambda x: QKVProducerKind(x.lower()), default=QKVProducerKind.MHA,
                    help="What kind of a qkv producer to use")
parser.add_argument("--av_combiner_kind", type=lambda x: AVCombinerKind(x.lower()), default=AVCombinerKind.LINEAR,
                    help="What kind of an av combiner to use")


parser.add_argument("--c_proj_scale_init", type=float, default=None,
                    help="Scale init for the c_proj")
parser.set_defaults(c_proj_scale_init=None)

args = parser.parse_args()

# -----------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


if args.debugpy:
    import debugpy
    debugpy.listen(5678)
    print("Debugpy listening on port 5678. Now is the time to attach!")
    debugpy.wait_for_client()

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.device_count() == int(os.environ['WORLD_SIZE']), "CUDA_VISIBLE_DEVICES must have as many devices as WORLD_SIZE"
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device_id = ddp_local_rank
    device = f'cuda:{device_id}'
    print(f"using device: {device}")
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

    if args.protection_kind != ProtectionKind.NONE:
        torch._dynamo.config.optimize_ddp = False # since we use a pytorch function with a custom backward

else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device_id = -1
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

if args.profile_kind == ProfileKind.PYTORCH and master_process:
    profiler = torch.profiler.profile(
        with_stack=True,
        record_shapes=True,
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2
        ),
    )
else:
    profiler = None


device_type = "cuda" if device.startswith("cuda") else "cpu"
autocast_precision = torch.bfloat16 if args.autocast_precision == "bfloat16" else torch.float32

torch.manual_seed(args.random_seed)
if torch.cuda.is_available():
    import random
    random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

enc = tiktoken.get_encoding("gpt2")

use_mini_model = os.environ.get("USE_MINI_MODEL", "false").lower() == "true" or args.use_mini_model

if use_mini_model:
    total_batch_size = args.total_batch_size or 20480
    B = args.batch_size or 10 # micro batch size
    T = args.seq_len or 512 # sequence length
    args.head_dim = args.head_dim or 64

    args.n_embd = args.n_embd or args.n_heads * args.head_dim
else:
    total_batch_size = args.total_batch_size or 524288 # 2**19, ~0.5M, in number of tokens
    B = args.batch_size or 8 # micro batch size
    T = args.seq_len or 1024 # sequence length
    args.head_dim = args.head_dim or 64

    args.n_embd = args.n_embd or args.n_heads * args.head_dim # just use GPTConfig's default
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

total_valid_batch_size = 131072
val_loss_steps = total_valid_batch_size // (B * T)
assert total_valid_batch_size % (B * T) == 0, "make sure total_valid_batch_size is divisible by B * T"

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
def make_config(args):
    return GPTConfig(
        n_embd=args.n_embd,
        n_head=args.n_heads,
        head_dim=args.head_dim,
        n_layer=12,
        block_size=T,
        attention_kind=args.attention_kind,
        for_inference=False,
        protect_bos_token=args.protect_bos_token,
        prevent_from_masking_myself=args.prevent_from_masking_myself,
        epsilon=args.memory_penalty_epsilon,
        selection_head_linear_combo=args.selection_head_linear_combo,
        selection_head_linear_combo_scale=args.selection_head_linear_combo_scale,
        disable_selection_head_linear_combo_bias=args.disable_selection_head_linear_combo_bias,
        assert_latent_matches_no_head=args.assert_latent_matches_no_head,
        protection_kind=args.protection_kind,
        leaky_relu_alpha=args.leaky_relu_alpha,
        leaky_relu_bias=args.leaky_relu_bias,
        protection_head_scaling_factor=args.protection_head_scaling_factor,
        protection_head_bias=args.protection_head_bias,
        n_sliced_masks=args.n_sliced_masks,
        n_latent_masks=args.n_latent_masks,
        one_head_per_latent_mask=args.one_head_per_latent_mask,
        init_latent_masks_to_identity=args.init_latent_masks_to_identity,
        init_latent_masks_to_inverse=args.init_latent_masks_to_inverse,
        latent_mask_scale=args.latent_mask_scale,
        latent_mask_runtime_multiplier=args.latent_mask_runtime_multiplier,
        latent_mask_sigmoid=args.latent_mask_sigmoid,
        latent_mask_precision=args.latent_mask_precision,
        mask_layernorm=args.mask_layernorm,
        S_layernorm=args.S_layernorm,

        att_conv=args.att_conv,
        att_conv_init=args.att_conv_init,
        att_conv_scale=args.att_conv_scale,
        att_conv_precision=args.att_conv_precision,
        att_conv_weight_decay=args.att_conv_weight_decay,

        residual_attention_masks=args.residual_attention_masks,
        disable_selection=args.disable_selection,
        mup=args.mup,
        readout_zero_init=args.readout_zero_init,
        query_zero_init=args.query_zero_init,
        l1_loss=args.l1_loss,

        attn_precision=args.attn_precision,

        dense_attention_config=DenseAttentionConfig(
            head_dim_value=args.head_dim_value,
            dense_attention_kind=args.dense_attention_kind,
            ckpt_attn=args.ckpt_attn,
            qkv_producer_kind=args.qkv_producer_kind,
            a_producer_kind=args.a_producer_kind,
            av_combiner_kind=args.av_combiner_kind,
        ),
        sdpa_iter_size=args.sdpa_iter_size,
        stabilize_attn_scores=args.stabilize_attn_scores,
        override_use_sdpa=args.override_use_sdpa,
        simulate_micro_bs=args.simulate_micro_bs,
        c_proj_scale_init=args.c_proj_scale_init,
        mup_zero_init=args.mup_zero_init,
    )

config = make_config(args)

model = GPT(config)

if args.mup:
    from mup import set_base_shapes
    if args.compute_base_shapes or args.base_shapes_savefile is None:
        def make_model_from_n_head(n_head,head_dim,n_embd):
            base_args = vars(config)
            base_args['n_head'] = n_head
            base_args['head_dim'] = head_dim
            base_args['n_embd'] = n_embd
            base_config = GPTConfig(**base_args)
            base_model = GPT(base_config)
            return base_model
        base_model = make_model_from_n_head(12,64,768)
        delta_model = make_model_from_n_head(1,64,64)

        model = make_model_from_n_head(args.n_heads,args.head_dim,args.n_embd)

        set_base_shapes(model,base_model,delta=delta_model,rescale_params=False,savefile=args.base_shapes_savefile)
        model.apply(model._init_weights)
        del base_model, delta_model
    else:
        set_base_shapes(model,args.base_shapes_savefile,rescale_params=False)


model.to(device)
use_compile = args.use_compile
non_compiled_model = model
if use_compile:
    model = torch.compile(model)
raw_model = model # always contains the "raw" unwrapped model

max_lr = args.max_lr or 6e-4
min_lr = max_lr * 0.1

if use_mini_model:
    new_max_steps = args.max_steps if args.max_steps is not None else 1000
    warmup_steps = args.warmup_steps or new_max_steps * 715 / 2500
    max_steps = new_max_steps
else:
    warmup_steps = 715
    max_steps = args.max_steps or 2500

def get_lr_scale(it):
    # 1 linear warmup for warmup_iters steps
    if it < warmup_steps:
        return (it+1) / warmup_steps
    # 2 if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr / max_lr
    # 3 in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return (min_lr + coeff * (max_lr - min_lr)) / max_lr

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

# create the log directory we will write checkpoints to and log to
assert args.log_dir is not None, "You have to pass in a log directory for this run"
log_dir = args.log_dir
if master_process:

    # rmdir if it exists
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    os.makedirs(log_dir, exist_ok=False)
    with open(os.path.join(log_dir, f"args.json"), "w") as f:
        # let's just write the args to the config file
        json.dump(vars(args), f)
        
    # Initialize wandb for logging (only for the master process)

    use_wandb = args.use_wandb and not (os.environ.get('SKIP_WANDB', 'false').lower() == 'true')

    if use_wandb:
        if not os.path.exists("wandb-logs"):
            os.makedirs("wandb-logs")

    wandb.init(project="context_compression", 
               config=vars(args), 
               group=args.group,
               name=os.path.basename(args.log_dir),
               dir="wandb-logs/"+log_dir,
               mode="online" if use_wandb else "disabled")

log_file = os.path.join(log_dir, f"log2.txt")
if master_process:
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

resume_checkpoint = args.resume_checkpoint

# Initialize step counter
start_step = 0

# Load checkpoint if resuming
if resume_checkpoint is not None:

    print(f"Resuming from {resume_checkpoint}")
    # Load model checkpoint
    if resume_checkpoint.startswith("hf://"):
        # Download from HuggingFace
        repo_id = "/".join(resume_checkpoint[len("hf://"):].split("/")[:2])
        rel_path = "/".join(resume_checkpoint[len("hf://"):].split("/")[2:])
        print(f"Downloading from {repo_id} to {rel_path}")
        checkpoint = torch.load(hf_hub_download(repo_id, rel_path))
    else:
        checkpoint = torch.load(resume_checkpoint)
    non_compiled_model.load_state_dict(checkpoint['model'], strict=False)

    if args.resume_optimizer:
        start_step = checkpoint['step']  # Resume from next step
        # Load optimizer checkpoint 
        if resume_checkpoint.startswith("hf://"):
            repo_id = "/".join(resume_checkpoint[len("hf://"):].split("/")[:2])
            rel_path = "/".join(resume_checkpoint[len("hf://"):].split("/")[2:])
            optimizer_checkpoint = torch.load(hf_hub_download(repo_id, rel_path.replace('model_', 'optimizer_')))
        else:
            optimizer_checkpoint = torch.load(resume_checkpoint.replace('model_', 'optimizer_'))
        optimizer.load_state_dict(optimizer_checkpoint['optimizer'])

        # Restore RNG state
        torch.set_rng_state(optimizer_checkpoint['rng_state'])
        torch.cuda.set_rng_state(optimizer_checkpoint['cuda_rng_state'])
        del optimizer_checkpoint
    else:
        optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    del checkpoint

    
    # --- ADDED: Restore the dataloader state if available ---
    if resume_checkpoint.startswith("hf://"):
        repo_id = "/".join(resume_checkpoint[len("hf://"):].split("/")[:2])
        rel_path = "/".join(resume_checkpoint[len("hf://"):].split("/")[2:])
        dataloader_path = hf_hub_download(repo_id, rel_path.replace('model_', 'dataloader_'))
        if os.path.exists(dataloader_path):
            print(f"Resuming dataloader state from {dataloader_path}")
            dataloader_state = torch.load(dataloader_path)
            train_loader.set_state(dataloader_state)
            del dataloader_state
    else:
        resume_dataloader_checkpoint = resume_checkpoint.replace('model_', 'dataloader_')
        if os.path.exists(resume_dataloader_checkpoint):
            print(f"Resuming dataloader state from {resume_dataloader_checkpoint}")
            dataloader_state = torch.load(resume_dataloader_checkpoint)
            train_loader.set_state(dataloader_state)
            del dataloader_state

if args.add_a_head:
    if master_process:
        print("ADDING A HEAD")
    assert not args.resume_optimizer, "if adding a head, you're not allowed to reuse your optimizer"
    add_head_to_start = args.add_head_to_start
    new_head_init = NewHeadInit(args.new_head_init.lower())
    add_a_head(config, raw_model, AddHeadConfig(add_head_kind=AddHeadKind.GROW_QKV_O,
                                                 add_head_to_start=add_head_to_start,
                                                 new_head_init=new_head_init))

    # create a new optimizer for the new parameters
    optimizer = raw_model.configure_optimizers(weight_decay=0.1,
                                                 learning_rate=6e-4,
                                                 device_type=device_type)

if ddp:
    model = DDP(model, device_ids=[device_id])

torch.cuda.empty_cache()

print_period = 10
eval_period = 100
save_period = 125000
hellaswag_period = 250

if master_process:
    with open(log_file, "a") as f:
        f.write(f'max_steps: {max_steps}\n')

# Modify your training loop to start from start_step

total_tokens_processed = 0
total_flops = 0

num_params = sum(p.numel() for p in model.parameters())

print("max_steps: ", max_steps)
for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % eval_period == 0 or last_step:
        model.eval()
        val_loader.reset()
        val_losses_accum = {}
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=autocast_precision):
                    logits, loss, losses, _ = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
                for k, v in losses.items():
                    val_losses_accum[k] = val_losses_accum.get(k, 0.0) + v.detach() / val_loss_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            for k in val_losses_accum:
                dist.all_reduce(val_losses_accum[k], op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            validation_perplexity = torch.exp(torch.tensor(val_loss_accum.item()))
            print(f"validation perplexity: {validation_perplexity:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val loss {val_loss_accum.item():.4f}\n")
                f.write(f"{step} val perplexity {validation_perplexity:.4f}\n")
            wandb.log({
                "step": step,
                "val_loss": val_loss_accum.item(),
                **{"val_loss_" + k: v.item() for k, v in val_losses_accum.items()},
                "val_perplexity": validation_perplexity.item(),
                "total_tokens_processed": total_tokens_processed,
                "total_flops": total_flops
            }, step=step)
            if step > 0 and (step % save_period == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': non_compiled_model.state_dict(),
                    'config': non_compiled_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    **{"val_loss_" + k: v.item() for k, v in val_losses_accum.items()}
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
                # save optimizer state and RNG seeds
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state()
                }, os.path.join(log_dir, f"optimizer_{step:05d}.pt"))
                torch.save(train_loader.get_state(), os.path.join(log_dir, f"dataloader_{step:05d}.pt"))

    # once in a while evaluate hellaswag
    if args.hellaswag and not (os.environ.get('SKIP_HELLASWAG', 'false').lower() == 'true') and (step % hellaswag_period == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=autocast_precision):
                    logits, loss, losses, _ = non_compiled_model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
            wandb.log({"step": step, "hellaswag_accuracy": acc_norm}, step=step)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=autocast_precision):
                    logits, loss, losses, _ = non_compiled_model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    total_tokens_processed += total_batch_size
    total_flops += total_batch_size * num_params * 2

    if args.mup_enable_coord_check_logging:
        coord_check_dict = {
            'token_embedding': [],
            'attn': [],
            'attn_score': [],
            'query': [],
            'key': [],
            'value': [],
            'mlp': [],
            'lm_head': [],
        }

        def hook(module, input, output, key):
            with torch.no_grad():
                if type(output) == tuple:
                    output = output[0]
                coord_check_dict[key].append(output.abs().mean().item())
        coord_check_handles = []
        for module_name, module in model.named_modules():
            if module_name == 'transformer.wte':
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='token_embedding')))
            elif module_name.endswith('.attn'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='attn')))
            elif module_name.endswith('.attn_score'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='attn_score')))
            elif module_name.endswith('.query'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='query')))
            elif module_name.endswith('.key'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='key')))
            elif module_name.endswith('.value'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='value')))
            elif module_name.endswith('.mlp'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='mlp')))
            elif module_name == 'lm_head':
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='lm_head')))
    else:
        coord_check_dict = None

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        if args.simulate_micro_bs is not None:
            with torch.autocast(device_type=device_type, dtype=autocast_precision):
                logits, loss, losses, ce_loss_batched = model(x, y)
            num_repeats = len(ce_loss_batched)
            for i in range(num_repeats):
                ce_loss_repeated = ce_loss_batched[i] / num_repeats
                retain_graph = i < num_repeats - 1
                ce_loss_repeated.backward(retain_graph=retain_graph)
        elif args.simulate_micro_bs_2 is not None:
            n_repeats = x.size(0) // args.simulate_micro_bs_2
            assert n_repeats * args.simulate_micro_bs_2 == x.size(0)
            for i in range(n_repeats):
                loss = 0
                losses = {}
                with torch.autocast(device_type=device_type, dtype=autocast_precision):
                    nano_logits, nano_loss, nano_losses, nano_ce_loss_batched = model(x[i*args.simulate_micro_bs_2:(i+1)*args.simulate_micro_bs_2], y[i*args.simulate_micro_bs_2:(i+1)*args.simulate_micro_bs_2])
                ce_loss_repeated = nano_loss / n_repeats
                ce_loss_repeated.backward()
                loss += ce_loss_repeated.detach()
                for k, v in nano_losses.items():
                    losses[k] = losses.get(k, 0) + v.detach() / n_repeats
                
        else:
            with torch.autocast(device_type=device_type, dtype=autocast_precision):
                logits, loss, losses, ce_loss_batched = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()
        loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    if loss_accum.isnan():
        print("Loss is NaN, stopping the run! VERY BAD NEWS!")
        # Optionally perform any cleanup or additional logging:
        try:
            wandb.log({"error": "Loss is NaN, run failed."})
            wandb.finish()  # finish the run gracefully
        except Exception as e:
            pass
        exit(1)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr_scale = get_lr_scale(step) if args.decay_lr else 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_scale * param_group['max_lr']
    optimizer.step()
    # if device_type == "cuda":
    #     torch.cuda.synchronize()

    if args.profile_kind == ProfileKind.PYTORCH and master_process:
        profiler.step()
        if profiler.step_num >= 6:
            profiler.stop()
            print("exporting chrome trace...")
            profiler.export_chrome_trace("small_trace.json")
            print("finished exporting chrome trace")
            exit(1)

    if args.mup_enable_coord_check_logging:
        for handle in coord_check_handles:
            handle.remove()
        
        # sanity check our mup_zero_init implementation
        if step==0 and args.mup_zero_init:
            assert coord_check_dict['attn_score'][0] == 0
        
        wandb.log({
            "step": step,
            **{f"coord_check_{k}": torch.tensor(v).mean().item() for k, v in coord_check_dict.items()}
        }, step=step)

    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        lr = lr_scale*max_lr
        wandb.log({
            "step": step,
            "train_loss": loss_accum.item(),
            **{"train_loss_" + k: v.item() for k, v in losses.items()},
            "lr": lr,
            "grad_norm": norm,
            "dt_ms": dt * 1000,
            "tokens_per_sec": tokens_per_sec
        }, step=step)
        if step % print_period == 0:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f} (lr={lr:.4e}) (hash(x)={x.sum().item()})\n")

if ddp:
    destroy_process_group()

if master_process and args.upload_to_hf:
    api = HfApi()
    log_dir_basename = os.path.basename(log_dir)
    log_dir_path = Path(log_dir).resolve()
    cwd_path = Path.cwd()
    if log_dir_path.is_relative_to(cwd_path):
        repo_path = str(log_dir_path.relative_to(cwd_path))
    else:
        repo_path = log_dir_basename
    
    import debugpy
    if args.debugpy:
        debugpy.breakpoint()

    api.upload_folder(
        repo_id="andrew-healey/context-compression",
        folder_path=log_dir,
        path_in_repo=repo_path,
        repo_type="model"
    )
    wandb.finish()


if master_process and (args.kill_self_after_run or os.environ.get('KILL_SELF_AFTER_RUN', 'false').lower() == 'true'):
    print("Run succeeded, killing my own instance")
    os.system("vastai destroy instance $CONTAINER_ID;")
