# Testing commands (outdated)

```
rm -rf testing_run; LOG_DIR=testing_run CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 perplexity_loss_only.py clear
```
Then interrupt training.


Replace `model_00002.pt` with the checkpoint you want to resume from.
```
rm -rf testing_run_2; RESUME_CHECKPOINT=testing_run/model_00002.pt LOG_DIR=testing_run_2 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 perplexity_loss_only.py 
```


# The overnight runs (commands are outdated)

```
rm -rf selective_run_0; USE_SELECTIVE=true LOG_DIR=selective_run_0 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29500 --nproc_per_node=4 perplexity_loss_only.py &> selective_run_0.txt
```

```
rm -rf unselective_run_0; USE_SELECTIVE=false LOG_DIR=unselective_run_0 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29501 --nproc_per_node=4 perplexity_loss_only.py &> unselective_run_0.txt
```

# Fine-tuning runs

```
rm -rf unselective_run_0_continued; RESUME_CHECKPOINT=unselective_run_0/model_07500.pt ATTENTION_KIND=self LOG_DIR=unselective_run_0_continued CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29502 --nproc_per_node=4 -m context_compression.train &> unselective_run_0_continued.txt
```

## Resuming from Yorth

Probably this is overfitting, right? Because he's already trained on this same 10B token dataset.
That's ok, one more epoch is prob fine. We're still not training on the valid set.

```
rm -rf yorth_run_0_continued; RESUME_CHECKPOINT=/root/.cache/huggingface/hub/models--Yorth--selective1/snapshots/1d3d987c90be4b8d6f58de60749ba5823f0ecd29/model.pt RESUME_OPTIMIZER=false ATTENTION_KIND=selective ADD_A_HEAD=false LOG_DIR=yorth_run_0_continued CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 -m context_compression.train &> yorth_run_0_continued.txt
```

## Resuming from Yorth, with a new normal attention head

```
rm -rf yorth_run_0_continued_with_head; RESUME_CHECKPOINT=/root/.cache/huggingface/hub/models--Yorth--selective1/snapshots/1d3d987c90be4b8d6f58de60749ba5823f0ecd29/model.pt RESUME_OPTIMIZER=false ATTENTION_KIND=selective ADD_A_HEAD=true LOG_DIR=yorth_run_0_continued_with_head CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 -m context_compression.train &> yorth_run_0_continued_with_head.txt
```

## Experiments on CPT

Throwing out optimizer state, pretraining the self-attention model on more data (Italy left side)

```
rm -rf unselective_run_0_restarted; RESUME_CHECKPOINT=hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt RESUME_OPTIMIZER=false MAX_STEPS=2500 ATTENTION_KIND=self LOG_DIR=unselective_run_0_restarted CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29501 --nproc_per_node=4 -m context_compression.train &> unselective_run_0_restarted.txt
```

Pretraining the self-attention model on more data, WITH an extra head (Italy right side)

```
rm -rf unselective_run_0_restarted_with_head; RESUME_CHECKPOINT=hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt RESUME_OPTIMIZER=false MAX_STEPS=2500 ATTENTION_KIND=self LOG_DIR=unselective_run_0_restarted_with_head ADD_A_HEAD=true ADD_HEAD_TO_START=true CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29502 --nproc_per_node=4 -m context_compression.train &> unselective_run_0_restarted_with_head.txt
```

Pretraining the selective attention model on more data for 10k steps, WITH a new selective head (DONE, see `self_to_selective_run_0`)

```
rm -rf self_to_selective_run_0; RESUME_CHECKPOINT=unselective_run_0/model_07500.pt RESUME_OPTIMIZER=false MAX_STEPS=2500 ATTENTION_KIND=selective LOG_DIR=self_to_selective_run_0 ADD_A_HEAD=true ADD_HEAD_TO_START=true CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29503 --nproc_per_node=4 -m context_compression.train &> self_to_selective_run_0.txt
```

Pretraining the selective attention model on more data for 2.5k steps, WITH a new zeroed out selective head (Hungary right side)

```
rm -rf self_to_selective_run_0_restarted; RESUME_CHECKPOINT=hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt RESUME_OPTIMIZER=false MAX_STEPS=2500 ATTENTION_KIND=selective LOG_DIR=self_to_selective_run_0_restarted ADD_A_HEAD=true ADD_HEAD_TO_START=true ZERO_OUT_NEW_HEAD=true CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29503 --nproc_per_node=4 -m context_compression.train &> self_to_selective_run_0_restarted.txt
```

Continuing pretraining for the last 2500 steps, with the same optimizer. Should hopefully reproduce the end of the loss curve in unselective_run_0. (Hungary left side)

```
rm -rf unselective_run_0_continued; RESUME_CHECKPOINT=unselective_run_0/model_07500.pt RESUME_OPTIMIZER=true ATTENTION_KIND=self LOG_DIR=unselective_run_0_continued CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29504 --nproc_per_node=4 -m context_compression.train &> unselective_run_0_continued.txt
```

Continuing pretraining for the last 2500 steps, with a new optimizer (and a new lr schedule) (KILLED, dupe of unselective_run_0_restarted)

```
rm -rf unselective_run_0_continued_with_new_optimizer; RESUME_CHECKPOINT=unselective_run_0/model_07500.pt RESUME_OPTIMIZER=false MAX_STEPS=2500 ATTENTION_KIND=self LOG_DIR=unselective_run_0_continued_with_new_optimizer CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29505 --nproc_per_node=4 -m context_compression.train &> unselective_run_0_continued_with_new_optimizer.txt
```

## Experiments on selective-head surgery CPT

We increased the dataset size, so let's restart the run with no modifications.

I expect this'll have basically the same results as unselective_run_0_restarted.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_2 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind self \
  --log_dir unselective_run_1_restarted \
  &> unselective_run_1_restarted.txt
```

Using a 5x-downscaled-O new head.
I expect this'll act like the restarted-run, but with slightly lower loss. It'll have much lower initial validation loss than the full-qkvo restarted run from last experiment, and so will do better relative to this restarted run than the old full-qkvo run did relative to the old restarted run.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_2 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind self \
  --log_dir unselective_run_1_restarted_with_o_rescaled \
  --add_a_head \
  --add_head_to_start \
  --new_head_init o_rescaled \
  &> unselective_run_1_restarted_with_o_rescaled.txt
```

Using a O-zeroed-out new head.

I expect this'll have lower initial loss and lower final loss than the 5x-downscaled-O head. And all its weights will be nonzero.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_2 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind self \
  --log_dir unselective_run_1_restarted_with_o_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init o_zero \
  &> unselective_run_1_restarted_with_o_zero.txt
```

Using a KO-zeroed-out new head.

I expect this'll have a bit worse final loss than the O-zeroed-out head, since it'll have a harder time learning. And all its weights will be nonzero.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_2 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind self \
  --log_dir unselective_run_1_restarted_with_ko_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  &> unselective_run_1_restarted_with_ko_zero.txt
```

Using a KO-zeroed-out new SELECTIVE head.

I expect this'll be the best - it'll be pareto-better than the KO-zeroed-out new head. And all its weights will be nonzero by the end.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_2 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_1_restarted_with_ko_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  &> self_to_selective_run_1_restarted_with_ko_zero.txt
```


## Experiments on selective-head surgery CPT

Using an O-zeroed-out new head.

Hypothesis: this will probably have much higher initial loss than the KO-zeroed-out head. I bet it'll have higher final loss too.

Result: unclear. I'm gonna run some more seeds later.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_3 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_1_restarted_with_o_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init o_zero \
  &> self_to_selective_run_1_restarted_with_o_zero.txt
```

Using a K-zeroed-out new head.

Hypothesis: This will probably be identical to the KO-zeroed-out head.

Result: Unclear. I'm gonna run some more seeds later.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_3 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_1_restarted_with_k_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init k_zero \
  --kill_self_after_run \
  &> self_to_selective_run_1_restarted_with_k_zero.txt
```

Using a memory loss term.

Hypothesis: This will do worse than everything so far on CE loss, but it'll do better than everything else at aggressive memory thresholds.

Result: I still don't know, actually! HF_TOKEN bug means I don't have the model artefact. I have to re-run this to see.

FWIW, it had much CE loss than all the other CPT selective runs (but still better than unselective, which is v interesting! TODO think more later about what this means).

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_3 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective_with_memory_penalty \
  --log_dir self_to_selective_run_1_restarted_with_memory_penalty \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  &> self_to_selective_run_1_restarted_with_memory_penalty.txt
```

With the BOS token unprotected - to compare to the magnitude of differences Leviathan measures.

Hypothesis: this'll be noticeably worse than the default KO-zero runs, like leviathan said it is.

Result: honestly not noticeably different. Makes me think my resolution is super bad compared to his (he pretrains from scratch, for like 50x longer than me). Let's try harder to reproduce his results in future experiments.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_2 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_1_restarted_with_unprotected_bos \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --no_protect_bos_token \
  --kill_self_after_run \
  &> self_to_selective_run_1_restarted_with_unprotected_bos.txt
```


## More experiments on selective-head surgery CPT

(results are [here](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=zx6asbynpfb&panelDisplayName=train_loss_ce&panelSectionName=Charts))



Using a lower penalty for memory in the loss function. e=0.02. (17813137)

Hypothesis: this will do much better than the epsilon=0.1 run, as measured by CE loss on the validation set.
And it'll do slightly better, as measured by CE loss with aggressive pruning.

Result: idk. The CE loss with no pruning is much lower, and the memory loss is much higher. BUT I didn't upload the model OR the logfile to huggingface properly. So I don't know about the important thing, gotta re-run this.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_4 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective_with_memory_penalty \
  --log_dir self_to_selective_run_1_restarted_with_memory_penalty_0.02 \
  --add_a_head \
  --new_head_init ko_zero \
  --memory_penalty_epsilon 0.02 \
  --kill_self_after_run \
  &> self_to_selective_run_1_restarted_with_memory_penalty_0.02.txt
```

Using a normal QKVO init for the new head. (17813138)
Hypothesis: this'll start off much worse, but will end up doing about as well as the KO initialization.

Result: this was acc v bad. Its running-avg training CE loss was better than no selection runs, BUT it's abt as bad as the o-zero init run. (which is to say, p bad!)

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_4 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_1_restarted_with_normal_init \
  --add_a_head \
  --new_head_init normal \
  --kill_self_after_run \
  &> self_to_selective_run_1_restarted_with_normal_init.txt
```

Putting the new head at the end, not the start. (17813139)
Hypothesis: this'll get better loss than an unselective run, but worse than a normal-initted selective run.

Result: this was surprisingly bad. Did just as badly as an unselective run with ko=zero.
I guess models can't work miracles - it's just rly hard to make the existing circuitry for arbitrarily-chosen head=1 *also* be a good selection head.
I do still wonder about the linear layer thing. I'll try it eventually. Maybe after lunch.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_4 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_1_restarted_head_at_end \
  --add_a_head \
  --add_head_to_end \
  --new_head_init ko_zero \
  --kill_self_after_run \
  &> self_to_selective_run_1_restarted_head_at_end.txt
```

Rerun of the o zero init run, with a new random seed. (17813144)

Hypothesis: same hypothesis as before - it'll have a higher initial and final loss than the ko-zero run.

Result: it was a little unclear when looking at the valid losses, but from looking at the running-avg training losses, this looks definitely worse than the ko-zero run and the k-zero runs.

Seems maybe a little better than the normal-init run, but not much. Makes some sense - the head is less confused than normal init at first, b/c it's not contributing to the final attn output. But it's still super bad, b/c the head is mostly just wreaking havoc on the rest of the attn heads.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_4 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_2_restarted_with_o_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init o_zero \
  --random_seed 1338 \
  --kill_self_after_run \
  &> self_to_selective_run_2_restarted_with_o_zero.txt
```

Rerun of the k zero init run, with a new random seed. (17813147)

Hypothesis: same as before - it'll be identical to ko-zero.

Result: seems like maybe. It might be a little better. I'm not sure, will probably do one more ko run and one more k run to be sure. But the strong default is that I'm keeping ko-zero (more stable initial loss, and high switching cost b/c previous experiments used it).

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_4 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_2_restarted_with_k_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init k_zero \
  --random_seed 1338 \
  --kill_self_after_run \
  &> self_to_selective_run_2_restarted_with_k_zero.txt
```

## selective-head surgery CPT 5

Another seed for ko-zero.

Hypothesis: it'll be basically as good as the k-zero runs.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_5 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_2_restarted_with_ko_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1338 \
  --kill_self_after_run \
  &> self_to_selective_run_2_restarted_with_ko_zero.txt
```

Another seed for k-zero.

Hypothesis: it'll be basically as good as the ko-zero runs.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_5 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_3_restarted_with_k_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init k_zero \
  --random_seed 1339 \
  --kill_self_after_run \
  &> self_to_selective_run_3_restarted_with_k_zero.txt
```

Another seed for unprotected BOS token.

Hypothesis: taken with the other run, it'll be a tiny bit worse than the ko-zero baseline.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_5 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_2_restarted_with_unprotected_bos \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1338 \
  --no_protect_bos_token \
  --kill_self_after_run \
  &> self_to_selective_run_2_restarted_with_unprotected_bos.txt
```

A third seed for unprotected BOS token.

Hypothesis: taken with the other runs, it'll be a tiny bit worse than the ko-zero baseline.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_5 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_3_restarted_with_unprotected_bos \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1339 \
  --no_protect_bos_token \
  --kill_self_after_run \
  &> self_to_selective_run_3_restarted_with_unprotected_bos.txt
```

Re-run of the memory loss run with eps=0.1.

Hypothesis: CE loss will be much better than everything else with pruning. But worse than eps=0.02 with pruning.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_5 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective_with_memory_penalty \
  --log_dir self_to_selective_run_2_restarted_with_memory_penalty_0.1 \
  --add_a_head \
  --new_head_init ko_zero \
  --memory_penalty_epsilon 0.1 \
  --kill_self_after_run \
  &> self_to_selective_run_2_restarted_with_memory_penalty_0.1.txt
```

Re-run of the memory loss run with eps=0.02.

Hypothesis: CE loss about as good as eps=0. But better than eps=0.1 with pruning.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_5 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective_with_memory_penalty \
  --log_dir self_to_selective_run_2_restarted_with_memory_penalty_0.02 \
  --add_a_head \
  --new_head_init ko_zero \
  --memory_penalty_epsilon 0.02 \
  --kill_self_after_run \
  &> self_to_selective_run_2_restarted_with_memory_penalty_0.02.txt
```

Allow tokens to mask themselves.

Hypothesis: between the two seeds, this'll show some tiny but nonzero difference vs. the ko-zero baseline. i.e. it's worse.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_5 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_1_restarted_with_allow_masking_myself \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1337 \
  --allow_masking_myself \
  --kill_self_after_run \
  &> self_to_selective_run_1_restarted_with_allow_masking_myself.txt
```

Allow tokens to mask themselves, with a new seed.

Hypothesis: between the two seeds, this'll show some tiny but nonzero difference vs. the ko-zero baseline. i.e. it's worse.

```
torchrun --nproc_per_node=gpu -m context_compression.train \
  --group selective_surgery_5 \
  --resume_checkpoint hf://andrew-healey/context-compression/unselective_run_0/model_07500.pt \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir self_to_selective_run_2_restarted_with_allow_masking_myself \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1338 \
  --allow_masking_myself \
  --kill_self_after_run \
  &> self_to_selective_run_2_restarted_with_allow_masking_myself.txt
```


