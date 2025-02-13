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

from .data import DataLoaderLite, get_most_likely_row
from .model import GPT, GPTConfig
from .attn import AttentionKind
from .hellaswag import render_example, iterate_examples
from .add_a_head import AddHeadConfig, AddHeadKind, add_a_head, NewHeadInit


# Parse command-line arguments for custom configuration
parser = argparse.ArgumentParser(description="Train GPT with context compression.")
parser.add_argument("--hellaswag", dest="hellaswag", action="store_true",
                    help="Enable HellaSwag evaluation (default: True)")
parser.add_argument("--no-hellaswag", dest="hellaswag", action="store_false",
                    help="Disable HellaSwag evaluation")
parser.set_defaults(hellaswag=True)

parser.add_argument("--attention_kind", type=lambda x: AttentionKind(x.lower()), required=True,
                    help="Attention type to use (e.g., self, selective)")
parser.add_argument("--log_dir", type=str, required=True,
                    help="Directory to save logs and checkpoints")
parser.add_argument("--resume_checkpoint", type=str, default=None,
                    help="Checkpoint path to resume training from")
parser.add_argument("--resume_optimizer", action="store_true",
                    help="Resume optimizer state when resuming checkpoint")
parser.add_argument("--add_a_head", action="store_true",
                    help="Add an additional head")
parser.add_argument("--add_head_to_start", action="store_true",
                    help="Place the new head at the start")
parser.add_argument("--new_head_init", type=lambda x: NewHeadInit(x.lower()), default=NewHeadInit.NORMAL,
                    help="Initialization type for the new head (e.g., normal, o_rescaled, o_zero, ko_zero)")
parser.add_argument("--max_steps", type=int, default=10000,
                    help="Maximum number of training steps")
parser.add_argument("--group", type=str, default=None,
                    help="Group name for the run")
parser.add_argument("--no-wandb", dest="use_wandb", action="store_false",
                    help="Disable wandb logging")
parser.add_argument("--kill_self_after_run", action="store_true",
                    help="Kill my own instance after run completes")
parser.set_defaults(use_wandb=True)
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

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 8 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
config = GPTConfig(vocab_size=50304, attention_kind=args.attention_kind, for_inference=False)
model = GPT(config)
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
non_compiled_model = model
if use_compile:
    model = torch.compile(model)
raw_model = model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = args.max_steps

def get_lr(it):
    # 1 linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2 if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3 in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
assert args.log_dir is not None, "You have to pass in a log directory for this run"
log_dir = args.log_dir
if master_process:

    # rmdir if it exists
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    os.makedirs(log_dir, exist_ok=False)
    # ANDREWTODO write config metadata to the log file
    with open(os.path.join(log_dir, f"args.json"), "w") as f:
        # let's just write the args to the config file
        json.dump(vars(args), f)
        
    # Initialize wandb for logging (only for the master process)
    wandb.init(project="context_compression", 
               config=vars(args), 
               group=args.group,
               name=os.path.basename(args.log_dir),
               dir=log_dir,
               mode="online" if args.use_wandb else "disabled")

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
    raw_model.load_state_dict(checkpoint['model'], strict=False)

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

eval_period = 25
save_period = 2500
hellaswag_period = 250

if master_process:
    with open(log_file, "a") as f:
        f.write(f'max_steps: {max_steps}\n')

# Modify your training loop to start from start_step
for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % eval_period == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
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
                "val_perplexity": validation_perplexity.item() # ANDREWTODO be honest abt loss, so I can get good memory loss plots
            }, step=step)
            if step > 0 and (step % save_period == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
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
    if args.hellaswag and (step % hellaswag_period == 0 or last_step) and (not use_compile):
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
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = non_compiled_model(tokens)
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
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = non_compiled_model(xgen) # (B, T, vocab_size)
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
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        wandb.log({
            "step": step,
            "train_loss": loss_accum.item(),
            "lr": lr,
            "grad_norm": norm,
            "dt_ms": dt * 1000,
            "tokens_per_sec": tokens_per_sec
        }, step=step)
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f} (lr={lr:.4e}) (hash(x)={x.sum().item()})\n")

if master_process:
    api = HfApi()
    log_dir_basename = os.path.basename(log_dir)
    api.upload_folder(
        repo_id="andrew-healey/context-compression",
        folder_path=log_dir,
        path_in_repo=log_dir_basename,
        repo_type="model"
    )
    wandb.finish()

if args.kill_self_after_run:
    print("Run succeeded, killing my own instance")
    os.system("vastai stop instance $CONTAINER_ID;")

if ddp:
    destroy_process_group()