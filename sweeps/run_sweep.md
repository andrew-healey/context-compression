```vast:verified
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=1 torchrun --nproc_per_node=8 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh 
```

```vast:fail/18411448
cd /workspace/context-compression && git pull && NNODES=3 NODE_RANK=2 torchrun --nproc_per_node=8 sweeps/run_sweep.py sweeps/sweep_wider_is_better.sh
```
