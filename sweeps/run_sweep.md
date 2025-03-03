```vast:finished
cd /workspace/context-compression && git pull && NNODES=2 NODE_RANK=0 USE_MINI_MODEL=true torchrun --nproc_per_node=8 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh 
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=2 NODE_RANK=1 USE_MINI_MODEL=true torchrun --nproc_per_node=8 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=4 NODE_RANK=0 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=4 NODE_RANK=1 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=4 NODE_RANK=2 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=4 NODE_RANK=3 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=4 NODE_RANK=0 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=4 NODE_RANK=1 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=4 NODE_RANK=2 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=4 NODE_RANK=3 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=0 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=1 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=2 USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

## sweeping lrs and seeds for each alternate arch experiment. requires 3 8-4090 nodes per experiment.

one_mask_per_head_2_latent_vectors:

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=0 USE_MINI_MODEL=true SUFFIX=one_mask_per_head_2_latent_vectors FLAGS="--selection_head_linear_combo n_latent_masks --n_latent_masks 2" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=1 USE_MINI_MODEL=true SUFFIX=one_mask_per_head_2_latent_vectors FLAGS="--selection_head_linear_combo n_latent_masks --n_latent_masks 2" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=2 USE_MINI_MODEL=true SUFFIX=one_mask_per_head_2_latent_vectors FLAGS="--selection_head_linear_combo n_latent_masks --n_latent_masks 2" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

baseline:

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=0 USE_MINI_MODEL=true SUFFIX=baseline FLAGS=" " torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=1 USE_MINI_MODEL=true SUFFIX=baseline FLAGS=" " torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=2 USE_MINI_MODEL=true SUFFIX=baseline FLAGS=" " torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

two_masks_4_heads:

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=0 USE_MINI_MODEL=true SUFFIX=two_masks_4_heads FLAGS="--selection_head_linear_combo two_masks" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=1 USE_MINI_MODEL=true SUFFIX=two_masks_4_heads FLAGS="--selection_head_linear_combo two_masks" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:finished
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=2 USE_MINI_MODEL=true SUFFIX=two_masks_4_heads FLAGS="--selection_head_linear_combo two_masks" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

# results:
# nothing beat the baseline. see results [here](https://wandb.ai/sesamestrong/context_compression/panel/zrk0hi0fm?nw=kumki8h6jwf).
# latent masks was horrible
# 2 masks was slightly worse than the default
# ok so now let's make it be latent mask with n=1
# and let's use many fewer seeds - figure out why latent masks are so horrible