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

```vast:fail/18413204
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=0 USE_MINI_MODEL=true SUFFIX=one_mask_per_head_2_latent_vectors FLAGS="--selection_head_linear_combo n_latent_masks --n_latent_masks 2" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:fail/18424879
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=1 USE_MINI_MODEL=true SUFFIX=one_mask_per_head_2_latent_vectors FLAGS="--selection_head_linear_combo n_latent_masks --n_latent_masks 2" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:fail/18424880
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=2 USE_MINI_MODEL=true SUFFIX=one_mask_per_head_2_latent_vectors FLAGS="--selection_head_linear_combo n_latent_masks --n_latent_masks 2" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

baseline:

```vast:fail/18424885
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=0 USE_MINI_MODEL=true SUFFIX=baseline FLAGS="" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:fail/18425059
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=1 USE_MINI_MODEL=true SUFFIX=baseline FLAGS="" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:fail/18425060
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=2 USE_MINI_MODEL=true SUFFIX=baseline FLAGS="" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

two_masks_4_heads:

```vast:fail/18425063
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=0 USE_MINI_MODEL=true SUFFIX=two_masks_4_heads FLAGS="--selection_head_linear_combo two_masks" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:verified
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=1 USE_MINI_MODEL=true SUFFIX=two_masks_4_heads FLAGS="--selection_head_linear_combo two_masks" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

```vast:verified
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=2 USE_MINI_MODEL=true SUFFIX=two_masks_4_heads FLAGS="--selection_head_linear_combo two_masks" torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```
