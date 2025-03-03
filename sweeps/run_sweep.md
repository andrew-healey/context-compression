```vast
cd /workspace/context-compression && git pull && NNODES=2 NODE_RANK=0 USE_MINI_MODEL=true torchrun --nproc_per_node=8 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh 
```

```vast
cd /workspace/context-compression && git pull && NNODES=2 NODE_RANK=1 USE_MINI_MODEL=true torchrun --nproc_per_node=8 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```
