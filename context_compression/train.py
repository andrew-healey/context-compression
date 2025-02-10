import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from context_compression.hellaswag import render_example, iterate_examples
from context_compression.data import get_most_likely_row

try_hellaswag = True



# -----------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from data import DataLoaderLite
from model import GPT, GPTConfig
from attn import AttentionKind

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

assert "CUDA_VISIBLE_DEVICES" in os.environ, "You have to pass in CUDA_VISIBLE_DEVICES"
visible_device_ids = [int(id) for id in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
assert len(visible_device_ids) == int(os.environ['WORLD_SIZE']), "CUDA_VISIBLE_DEVICES must have as many devices as WORLD_SIZE"

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
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

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 491520 # 2**19, ~0.5M, in number of tokens
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

assert os.environ["USE_SELECTIVE"] in ["true", "false"], "USE_SELECTIVE must be true or false"
use_selective = os.environ["USE_SELECTIVE"] == "true"

# create model
model = GPT(GPTConfig(vocab_size=50304, attention_kind=AttentionKind.SELECTIVE, for_inference=False))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[device_id])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 10000 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
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
assert os.environ["LOG_DIR"] is not None, "You have to pass in a log directory for this run"
log_dir = os.environ["LOG_DIR"]
if master_process:
    os.makedirs(log_dir, exist_ok=False)
log_file = os.path.join(log_dir, f"log2.txt")
if master_process:
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

# At the start of your training script, add a resume checkpoint argument
resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", None)  # or get from args/env var

# Initialize step counter
start_step = 0

# Load checkpoint if resuming
if resume_checkpoint is not None:
    print(f"Resuming from {resume_checkpoint}")
    # Load model checkpoint
    checkpoint = torch.load(resume_checkpoint)
    raw_model.load_state_dict(checkpoint['model'])
    start_step = checkpoint['step']  # Resume from next step

    del checkpoint
    
    # Load optimizer checkpoint 
    optimizer_checkpoint = torch.load(resume_checkpoint.replace('model_', 'optimizer_'))
    optimizer.load_state_dict(optimizer_checkpoint['optimizer'])

    
    # Restore RNG state
    torch.set_rng_state(optimizer_checkpoint['rng_state'])
    torch.cuda.set_rng_state(optimizer_checkpoint['cuda_rng_state'])
    del optimizer_checkpoint
    
    # --- ADDED: Restore the dataloader state if available ---
    resume_dataloader_checkpoint = resume_checkpoint.replace('model_', 'dataloader_')
    if resume_dataloader_checkpoint is not None:
        print(f"Resuming dataloader state from {resume_dataloader_checkpoint}")
        dataloader_state = torch.load(resume_dataloader_checkpoint)
        train_loader.set_state(dataloader_state)
        del dataloader_state
    
    if ddp:
        model = DDP(model, device_ids=[device_id])

torch.cuda.empty_cache()

eval_period = 250
save_period = 2500

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
                # --- ADDED: Save the dataloader state for resuming ---
                torch.save(train_loader.get_state(), os.path.join(log_dir, f"dataloader_{step:05d}.pt"))

    # once in a while evaluate hellaswag
    if try_hellaswag and (step % 250 == 0 or last_step) and (not use_compile):
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
                    logits, loss = model(tokens)
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
                    logits, loss = model(xgen) # (B, T, vocab_size)
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
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f} (lr={lr:.4e}) (hash(x)={x.sum().item()})\n")

if ddp:
    destroy_process_group()