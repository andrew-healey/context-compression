```
rm -rf testing_run; LOG_DIR=testing_run CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 perplexity_loss_only.py clear
```
Then interrupt training.


Replace `model_00002.pt` with the checkpoint you want to resume from.
```
rm -rf testing_run_2; RESUME_CHECKPOINT=testing_run/model_00002.pt LOG_DIR=testing_run_2 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 perplexity_loss_only.py 
```


# The overnight runs

```
rm -rf selective_run_0; USE_SELECTIVE=true LOG_DIR=selective_run_0 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 perplexity_loss_only.py &> selective_run_0.txt
```

```
rm -rf unselective_run_0; USE_SELECTIVE=false LOG_DIR=unselective_run_0 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 perplexity_loss_only.py &> unselective_run_0.txt
```