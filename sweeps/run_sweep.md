```vast
torchrun --nnodes=4 --node_rank=1 --nproc_per_node=8 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```

```vast
torchrun --nnodes=4 --node_rank=2 --nproc_per_node=8 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```
