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

Graphs showing findings:

k-zero vs ko-zero. [wandb](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=fu6wo6ow9pk). Very unclear which one is better (if either). The train says all the k-zero runs were slightly better than all the k-zero runs. But I'm just gonna go by the size of the variation between them, and say it's not worth switching to k-zero. Even if it *may* possibly give slightly lower val losses.

eps=0.1 vs eps=0.02. [wandb](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=z07nkif4j3), [pruning graph on github](https://raw.githubusercontent.com/andrew-healey/context-compression/9dbbc7bf7798ee290711ebbf22674016011bc93c/imgs/pruning-graphs.png?token=GHSAT0AAAAAAC3UD4HYCUSJY372TOUGKAO2Z5OR2ZA) - eps=0.02 has much better CE loss, and nearly identical performance when pruning. So the memory loss term is probably kinda bad. I bet I can make a better one. TODO try to do that when prune-tuning future overtrained models (and also TODO make some overtrained models to test this on!!).

bos-token unprotected vs protected. [wandb](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=prpu5ytbp7o). Couldn't really see a difference. Again, there was some wide val loss variation, but train loss variation between groups was tiny. So probably let's just ignore this. TODO see if this also holds for pretrained models (i.e. 2500 steps "from random init". haha "from random init").

allow-masking-myself vs disallow-masking-myself. [wandb](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=af08zmdxpwo). Noticeable difference, wow! Ig this sets a standard for difference size needed to be convincing.

<hr>

Another seed for ko-zero. (17817259)

Hypothesis: it'll be basically as good as the k-zero runs.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
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

Another seed for k-zero. (17816804)

Hypothesis: it'll be basically as good as the ko-zero runs.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
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

Another seed for unprotected BOS token. (17816805)

Hypothesis: taken with the other run, it'll be a tiny bit worse than the ko-zero baseline.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
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

A third seed for unprotected BOS token. (17816806)

Hypothesis: taken with the other runs, it'll be a tiny bit worse than the ko-zero baseline.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
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

Re-run of the memory loss run with eps=0.1. (17816814)

Hypothesis: CE loss will be much better than everything else with pruning. But worse than eps=0.02 with pruning.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
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

Re-run of the memory loss run with eps=0.02. (17816816)

Hypothesis: CE loss about as good as eps=0. But better than eps=0.1 with pruning.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
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

Allow tokens to mask themselves. (17816920)

Hypothesis: between the two seeds, this'll show some tiny but nonzero difference vs. the ko-zero baseline. i.e. it's worse.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
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

Allow tokens to mask themselves, with a new seed. (17816921)

Hypothesis: between the two seeds, this'll show some tiny but nonzero difference vs. the ko-zero baseline. i.e. it's worse.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
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


## From-scratch pretraining experiments (with an extra head, based on CPT patterns)

We're using an extra head just because that lets me reuse my add_a_head code for the kv init.


Results:

epsilon=0.02 graphs: [wandb](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=km905uw7die)
- "epsilon=0.02 will basically match epsilon=0 on CE loss with no pruning."
  - Not true! It's worse on CE loss.
  - Hrrm. this is diff from the CPT experiments.
  - IDK if there's enough data to make a general conclusion.
  - Still unclear how much unselective-to-selective loss delta is destroyed by eps=0.1 and eps=0.02, since I didn't try any unselective runs. I'm not planning to, btw.
- "epsilon=0.02 will beat epsilon=0.1 in both CE loss and pruning-ratio-before-parity."
  - yup! [graph](https://raw.githubusercontent.com/andrew-healey/context-compression/refs/heads/master/imgs/pruning-graphs.png?token=GHSAT0AAAAAAC3UD4HZRXXAAEIA7BNCBC4AZ5P4ULA)

bos and self protection: [wandb](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=km905uw7die)
- "no bos protection will be slightly worse than ko zero init"
  - actually, it's a p big degradation!
  - much bigger (relatively) than on CPT. I wonder why?
    - it might be bigger or smaller with normal init. Hrrm. Such is the cost of switching to a new baseline.
- "allow masking myself will be noticeably worse than ko zero init"
  - actually, seems like no!
  - or at least, much tighter delta than the bos protection one. my experiments so far don't show enough separation for any amount of confidence in this.
  - This makes me think it rly is a small difference, just like Leviathan said!!
    - Which would be p good news, ig? If I could basically repro the relative magnitudes of treatment effect deltas.

OK. Our next experiment (From-scratch pretraining experiments 3) will be using a new normal init baseline.


<hr>

Pretrain with an extra head, memory loss with epsilon=0.1. (17822255)

Hypothesis: this'll be as good as the original eps=0.1 run.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head \
  --max_steps 2500 \
  --attention_kind selective_with_memory_penalty \
  --log_dir scratch_selective_run_0_memory_penalty_0.1 \
  --add_a_head \
  --new_head_init ko_zero \
  --memory_penalty_epsilon 0.1 \
  --kill_self_after_run \
  &> scratch_selective_run_0_memory_penalty_0.1.txt
```

Pretrain with memory loss with epsilon=0.02. (17822259)

Hypothesis: this'll be as good as epsilon=0 on CE loss with no pruning. It'll also have a better pruning-ratio-before-parity performance than eps=0.1. Because it did on the CPT experiments.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head \
  --max_steps 2500 \
  --attention_kind selective_with_memory_penalty \
  --log_dir scratch_selective_run_0_memory_penalty_0.02 \
  --add_a_head \
  --new_head_init ko_zero \
  --memory_penalty_epsilon 0.02 \
  --kill_self_after_run \
  &> scratch_selective_run_0_memory_penalty_0.02.txt
```

Pretrain with an extra head, with normal init. (17822262)

Hypothesis: this'll match the other init strats.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_normal_init \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1337 \
  --kill_self_after_run \
  &> scratch_selective_run_0_normal_init.txt
```

Pretrain with an extra head, with k-zero init for the first head. (17822265)

Hypothesis: this'll match the other init strats.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_k_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init k_zero \
  --random_seed 1337 \
  --kill_self_after_run \
  &> scratch_selective_run_0_k_zero.txt
```

Pretrain with an extra head, with ko-zero init for the first head. (17822266)

Hypothesis: this'll match the other init strats.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_ko_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1337 \
  --kill_self_after_run \
  &> scratch_selective_run_0_ko_zero.txt
```

Pretrain with ko-zero init for the first head, with another seed. (17822268)

Hypothesis: this'll match the other init strats.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_ko_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1338 \
  --kill_self_after_run \
  &> scratch_selective_run_1_ko_zero.txt
```

Pretrain with ko-zero init for the first head, with a third seed. (17822271)

Hypothesis: this'll match the other init strats.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_2_ko_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1339 \
  --kill_self_after_run \
  &> scratch_selective_run_2_ko_zero.txt
```

Pretrain with ko-zero init and no bos protection. (17822272)

Hypothesis: it'll be a little worse than the default ko-zero runs.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_ko_zero_no_bos_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1337 \
  --no_protect_bos_token \
  --kill_self_after_run \
  &> scratch_selective_run_0_ko_zero_no_bos_protection.txt
```

Pretrain with ko-zero init and no self protection. (17822274)

Hypothesis: it'll be a little worse than the default ko-zero runs. But it'll be more worse than the no-bos-protection runs (since it was much more worse in the CPT experiments).

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_ko_zero_no_self_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1337 \
  --allow_masking_myself \
  --kill_self_after_run \
  &> scratch_selective_run_0_ko_zero_no_self_protection.txt
```


## From-scratch pretraining experiments 2 (with an extra head)

Pretrain with ko-zero init and memory loss with epsilon=0.02, with a new seed. (17823277, not done)

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective_with_memory_penalty \
  --log_dir scratch_selective_run_1_memory_penalty_0.02 \
  --add_a_head \
  --new_head_init ko_zero \
  --memory_penalty_epsilon 0.02 \
  --random_seed 1338 \
  --kill_self_after_run \
  &> scratch_selective_run_1_memory_penalty_0.02.txt
```

Pretrain with ko-zero init and memory loss with epsilon=0.02, with a third seed. (17823278, not done)

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective_with_memory_penalty \
  --log_dir scratch_selective_run_2_memory_penalty_0.02 \
  --add_a_head \
  --new_head_init ko_zero \
  --memory_penalty_epsilon 0.02 \
  --random_seed 1339 \
  --kill_self_after_run \
  &> scratch_selective_run_2_memory_penalty_0.02.txt
```

Pretrain with ko-zero init and no bos protection, with a new seed. (17823275, not done)

Just to get more signifigance.

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_ko_zero_no_bos_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1338 \
  --no_protect_bos_token \
  --kill_self_after_run \
  &> scratch_selective_run_1_ko_zero_no_bos_protection.txt
```

Pretrain with ko-zero init and no bos protection, with a third seed. (17823339)

Just to get more signifigance.

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_2_ko_zero_no_bos_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1339 \
  --no_protect_bos_token \
  --kill_self_after_run \
  &> scratch_selective_run_2_ko_zero_no_bos_protection.txt
```

Pretrain with ko-zero init and no self protection, with a new seed. (17823274)

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_ko_zero_no_self_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1338 \
  --allow_masking_myself \
  --kill_self_after_run \
  &> scratch_selective_run_1_ko_zero_no_self_protection.txt
```

Pretrain with ko-zero init and no self protection, with a third seed. (17823346)

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_2_ko_zero_no_self_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1339 \
  --allow_masking_myself \
  --kill_self_after_run \
  &> scratch_selective_run_2_ko_zero_no_self_protection.txt
```

Pretrain with ko-zero init, with a fourth seed. (17823348)

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_3_ko_zero \
  --add_a_head \
  --add_head_to_start \
  --new_head_init ko_zero \
  --random_seed 1340 \
  --kill_self_after_run \
  &> scratch_selective_run_3_ko_zero.txt
```

Pretrain with normal init, with a second seed. (17823351)

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_normal_init \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1338 \
  --kill_self_after_run \
  &> scratch_selective_run_1_normal_init.txt
```

Pretrain with normal init, with a third seed. (17823354)

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_2_normal_init \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1339 \
  --kill_self_after_run \
  &> scratch_selective_run_2_normal_init.txt
```

Pretrain with normal init, with a fourth seed. (17823273)

Hypothesis: same as before.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_2 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_3_normal_init \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1340 \
  --kill_self_after_run \
  &> scratch_selective_run_3_normal_init.txt
```

## From-scratch pretraining experiments 3 (with an extra head)

We're not doing any crazy double-relu experiment *yet*.

Hypothesis: no-self-protection will be slightly noticeably worse than self-protection, but only by looking at train losses. This is based on the last from-scratch experiments.

Results: [wandb link](https://wandb.ai/sesamestrong/context_compression/workspace/panel/7i8l8u02w?nw=uykcmzvmij)

No-self-protection is clearly worse on train losses than baselines. It's very stark actually, with normal inits.

I wonder if it has something to do with ko-zero init not encouraging the model to self-mask? So maybe self-masking is less of an issue then for ko-zero.

Hypothesis: no-bos-protection will be noticeably worse than bos-protection, by looking at train losses. Also based on the last from-scratch experiments.

Results: [wandb link](https://wandb.ai/sesamestrong/context_compression/workspace/panel/7i8l8u02w?nw=uykcmzvmij)

Like no-self-protection, no-bos-protection is clearly worse than baselines. I hope all my experiments make things this easy and simple!

Hypothesis: unselective will be much worse than selective, obviously.

Results: [wandb link](https://wandb.ai/sesamestrong/context_compression/workspace/panel/7i8l8u02w?nw=yg2p1tv1pd)

It's kinda surprising *how* bad it is. Like 10x or 20x bigger loss delta than all the other things I'm tweaking (except possibly eps).

Hypothesis: linear head will be a bit better than the normal selection head.

Result: Seems a little worse, surprisingly! [wandb link](https://wandb.ai/sesamestrong/context_compression/workspace/panel/7i8l8u02w?nw=6z9gcoijvxd)

Very confused about this, actually. Doesn't this just strictly improve the capacity of the model? Hrrm. It may acc have acted like a ko-zero init...

Yeah, so ig this should be a pareto improvement if we change the selectivity mask init to maybe be one for head 1. TODO try that next experiment.

Hypothesis: leaky ReLU, in both forms, will have worse performance than baseline.

Result: See [wandb link](https://wandb.ai/sesamestrong/context_compression/workspace/panel/7i8l8u02w?nw=ejl75xkk2). This was right - even though I did the cumsum thing! I'm acc a bit surprised by this. I wrote the hypothesis before modifying my leaky ReLU code (to later run a relu on the cumsum mask) - so after making that modification, I was p optimistic.

Hrrm. It maybe that neighbor-token consumption is just the most important effect for decreasing loss? And that's incredibly local. So have any leakage will wreak havoc. Hrrm yeah, ig anybody that's not explicitly masking a token is then choosing to reinforce it. We rly should try that double-ReLU approach...

Hypothesis: no-ReLU selectivity will diverge completely.

Result: I didn't run a no-ReLU selectivity run, IDT. Didn't see any point.

Hypothesis: move-ReLU-after-cumsum will be better than the selective baseline.

Result: it's worse! It's a subset of leaky-ReLU, so probably it suffers from the same issues.

Hypothesis: move-ReLU-after-cumsum with no-bos-protection will be better than the selective baseline with no-bos-protection.

<hr>

No-self-protection run with normal init. (17860836)

```
./local/run_in_remote.sh 17860836 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_normal_init_no_self_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1337 \
  --allow_masking_myself \
  --kill_self_after_run \
  &> scratch_selective_run_0_normal_init_no_self_protection.txt"
```

No-self-protection run with normal init, with a second seed. (17860837)

```
./local/run_in_remote.sh 17860837 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_normal_init_no_self_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1338 \
  --allow_masking_myself \
  --kill_self_after_run \
  &> scratch_selective_run_1_normal_init_no_self_protection.txt"
```

No-self-protection run with normal init, with a third seed. (17860838)

```
./local/run_in_remote.sh 17860838 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_2_normal_init_no_self_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1339 \
  --allow_masking_myself \
  --kill_self_after_run \
  &> scratch_selective_run_2_normal_init_no_self_protection.txt"
```

No-bos-protection run with normal init. (17860839)

```
./local/run_in_remote.sh 17860839 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_normal_init_no_bos_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1337 \
  --no_protect_bos_token \
  --kill_self_after_run \
  &> scratch_selective_run_0_normal_init_no_bos_protection.txt"
```


No-bos-protection run with normal init, with a second seed. (17860843)

```
./local/run_in_remote.sh 17860843 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_normal_init_no_bos_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1338 \
  --no_protect_bos_token \
  --kill_self_after_run \
  &> scratch_selective_run_1_normal_init_no_bos_protection.txt"
```

Unselective run with normal init. (17860844)

```
./local/run_in_remote.sh 17860844 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind self \
  --log_dir scratch_unselective_run_0_normal_init \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --random_seed 1337 \
  --kill_self_after_run \
  &> scratch_unselective_run_0_normal_init.txt"
```

Replacing ReLU with leaky ReLU for the selection mask, with i.e. 0.1 leak. (17860846)

```
./local/run_in_remote.sh 17860846 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_normal_init_leaky_relu_0.1 \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --relu_leak 0.1 \
  --relu_after_cumsum \
  --random_seed 1338 \
  --kill_self_after_run \
  &> scratch_selective_run_1_normal_init_leaky_relu_0.1.txt"
```

Replacing ReLU with leaky ReLU for the selection mask, with i.e. 0.25 leak. (17860850)

```
./local/run_in_remote.sh 17860850 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_normal_init_leaky_relu_0.25 \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --relu_leak 0.25 \
  --relu_after_cumsum \
  --random_seed 1338 \
  --kill_self_after_run \
  &> scratch_selective_run_1_normal_init_leaky_relu_0.25.txt"
```

Moving ReLU after the cumsum. (17861898)

```
./local/run_in_remote.sh 17861898 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_normal_init_move_relu_after_cumsum \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --relu_leak 1.0 \
  --relu_after_cumsum \
  --random_seed 1337 \
  --kill_self_after_run \
  &> scratch_selective_run_0_normal_init_move_relu_after_cumsum.txt"
```

Moving ReLU after the cumsum, with a second seed. (17861745)

```
./local/run_in_remote.sh 17861745 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_normal_init_move_relu_after_cumsum \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --relu_leak 1.0 \
  --relu_after_cumsum \
  --random_seed 1338 \
  --kill_self_after_run \
  &> scratch_selective_run_1_normal_init_move_relu_after_cumsum.txt"
```

Moving ReLU after the cumsum, with no bos protection. (17861747)

```
./local/run_in_remote.sh 17861747 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_normal_init_move_relu_after_cumsum_no_bos_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --relu_leak 1.0 \
  --relu_after_cumsum \
  --no_protect_bos_token \
  --random_seed 1337 \
  --kill_self_after_run \
  &> scratch_selective_run_0_normal_init_move_relu_after_cumsum_no_bos_protection.txt"
```

Moving ReLU after the cumsum, with no bos protection, with a second seed. (17861748)

```
./local/run_in_remote.sh 17861748 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_normal_init_move_relu_after_cumsum_no_bos_protection \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --relu_leak 1.0 \
  --relu_after_cumsum \
  --no_protect_bos_token \
  --random_seed 1338 \
  --kill_self_after_run \
  &> scratch_selective_run_1_normal_init_move_relu_after_cumsum_no_bos_protection.txt"
```


Replacing selection head with linear combo of attention scores. (17861750)

```
./local/run_in_remote.sh 17861750 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_0_normal_init_linear_combo \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --selection_head_linear_combo \
  --random_seed 1337 \
  --kill_self_after_run \
  &> scratch_selective_run_0_normal_init_linear_combo.txt"
```

Replacing selection head with linear combo of attention scores, with a second seed. (17861752)

```
./local/run_in_remote.sh 17861752 "echo hi && cd /workspace/context-compression && git pull && nohup torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_3 \
  --max_steps 2500 \
  --attention_kind selective \
  --log_dir scratch_selective_run_1_normal_init_linear_combo \
  --add_a_head \
  --add_head_to_start \
  --new_head_init normal \
  --selection_head_linear_combo \
  --random_seed 1338 \
  --kill_self_after_run \
  &> scratch_selective_run_1_normal_init_linear_combo.txt"
```

## From-scratch pretraining experiments 4 (with an extra head)

I have some runs with new linear combo init strategies, since that theoretically should be no worse than the first-head strategy.

I also added some runs for different variants of the protection-token idea. I hope it improves performance!

Hypotheses:

Linear combo selection head initted to head[0] = 1, with bias, will be a bit better than baseline.

Result: it's worse. Almost identically bad to the linear combo with no bias. Maybe no-bias is a little better. Super confused abt this.

Hrrm... Maybe selection is just v spooky and delicate. And all the output-focused heads are fluctuating all the time, possibly bigly.

So maybe they'll just occasionally get unlucky and start messing with masking. Hrrm. Honestly, I could believe this.

What does it mean for me though?

IG maybe it means the model needs to be actually reconfigured to work w/ dramatic new mechanisms (i.e. protection, selection, etc.). It's not enough to expect it'll be the same + a few more allocated circuits.

Hrrm. That's bad news for us though, right? B/c that means every head we spend is costly. Hrrm but maybe we can split the params across two heads.

Linear combo selection head initted to head[0] = 1, with no bias, will be worse than with bias but still better than baseline.

See above.

Head-two protection head will make performance a bit worse, since it's decreasing the # of available heads, for something relatively unimportant.

Result: much worse. Total lobotomization, barely learns at all. Super surprising, smells like a bug. I wonder if it's adding, not subtracting?

The loss decreases slower even in first 25 iters. I'll try directly comparing that way on some 8-4090 setup.

Linear-combo protection head will make performance a bit better, I think - seems like probably just a pareto improvement.

Result: worse.

Linear-combo protection head with head two will make performance worse, since it's basically like a head-two protection head.

Result: worse.

Leaky-relu protection head (with bias) will make performance better than baseline. But probably only by a little bit. IDK how much, relative to the other protection kinds.

Result: worse.

<hr>

Let's examine the shape of the degradation, and check if the code has gotten worse, or maybe my ideas are just bad. Maybe I need to try writing transformer programs by hand?

UPDATE: I think there's a bug, see the next experiment.

I think there's not too much to learn here abt protection, b/c it's buggy.

<hr>

Linear combo selection initted to head[0] = 1, with bias.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_0_lincomb_head0_with_bias \
  --selection_head_linear_combo with_head_zero_and_bias \
  --random_seed 1337 \
  &> run_0_lincomb_head0_with_bias.txt
```

Linear combo selection initted to head[0] = 1, with bias, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_1_lincomb_head0_with_bias \
  --selection_head_linear_combo with_head_zero_and_bias \
  --random_seed 1338 \
  &> run_1_lincomb_head0_with_bias.txt
```

Linear combo selection initted to head[0] = 1, with no bias.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_0_lincomb_head0_no_bias \
  --selection_head_linear_combo with_head_zero \
  --random_seed 1337 \
  &> run_0_lincomb_head0_no_bias.txt
```

Linear combo selection initted to head[0] = 1, with no bias, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_1_lincomb_head0_no_bias \
  --selection_head_linear_combo with_head_zero \
  --random_seed 1338 \
  &> run_1_lincomb_head0_no_bias.txt
```

Head-two protection head.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_0_protection_head2 \
  --protection_kind head_two \
  --random_seed 1337 \
  &> run_0_protection_head2.txt
```

Head-two protection head, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_1_protection_head2 \
  --protection_kind head_two \
  --random_seed 1338 \
  &> run_1_protection_head2.txt
```

Linear combo protection head.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_0_protection_linear_combo \
  --protection_kind linear_combo \
  --random_seed 1337 \
  &> run_0_protection_linear_combo.txt
```

Linear combo protection head, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_1_protection_linear_combo \
  --protection_kind linear_combo \
  --random_seed 1338 \
  &> run_1_protection_linear_combo.txt
```

Linear combo protection head with head two.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_0_protection_linear_combo_head2 \
  --protection_kind linear_combo_head_two \
  --random_seed 1337 \
  &> run_0_protection_linear_combo_head2.txt
```

Linear combo protection head with head two, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_1_protection_linear_combo_head2 \
  --protection_kind linear_combo_head_two \
  --random_seed 1338 \
  &> run_1_protection_linear_combo_head2.txt
```

Leaky-relu protection head (with bias).

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_0_protection_leaky_relu \
  --protection_kind leaky_relu \
  --leaky_relu_alpha 0.1 \
  --leaky_relu_bias -0.05 \
  --random_seed 1337 \
  &> run_0_protection_leaky_relu.txt
```

Leaky-relu protection head (with bias), with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_4 \
  --log_dir run_1_protection_leaky_relu \
  --protection_kind leaky_relu \
  --leaky_relu_alpha 0.1 \
  --leaky_relu_bias -0.05 \
  --random_seed 1338 \
  &> run_1_protection_leaky_relu.txt
```


## From-scratch pretraining experiments 5 (with an extra head)

Let's just very quickly verify that our baselines haven't gotten worse.

And that our protect-and-attack code can perfectly reproduce the baseline results.

<hr>

Quick update: so looks like normal init baseline is all good.

There's some horrible initial divergence for the protection zero run. Makes me think I missed a sign bit somewhere.

OK, I am stopping all these runs early!

<hr>

Run the normal init baseline again.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_5 \
  --log_dir run_5_normal_init \
  --random_seed 1337
```

Run the normal init baseline again, with another seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_5 \
  --log_dir run_6_normal_init \
  --random_seed 1338
```

Run the normal init with protection kind "zero". This should reproduce the baseline results also.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_5 \
  --log_dir run_0_normal_init_protection_zero \
  --protection_kind zero \
  --random_seed 1337
```

Run the normal init with protection kind "zero", with another seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_5 \
  --log_dir run_1_normal_init_protection_zero \
  --protection_kind zero \
  --random_seed 1338
```

## Checking that my bugfix worked

These are the same experiments as before, but with the bugfix.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_5 \
  --log_dir run_2_normal_init_protection_zero \
  --protection_kind zero \
  --random_seed 1337
```

Head-two protection head.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_2_protection_head2 \
  --protection_kind head_two \
  --random_seed 1337
```

Head-two protection head, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_3_protection_head2 \
  --protection_kind head_two \
  --random_seed 1338
```

Linear combo protection head.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_2_protection_linear_combo \
  --protection_kind linear_combo \
  --random_seed 1337
```

Linear combo protection head, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_3_protection_linear_combo \
  --protection_kind linear_combo \
  --random_seed 1338
```

Linear combo protection head with head two.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_2_protection_linear_combo_head2 \
  --protection_kind linear_combo_head_two \
  --random_seed 1337
```

Leaky-relu protection head (with bias).

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_2_protection_leaky_relu \
  --protection_kind leaky_relu \
  --leaky_relu_alpha 0.1 \
  --leaky_relu_bias -0.05 \
  --random_seed 1337
```

Leaky-relu protection head (with bias), with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_3_protection_leaky_relu \
  --protection_kind leaky_relu \
  --leaky_relu_alpha 0.1 \
  --leaky_relu_bias -0.05 \
  --random_seed 1338
```

Linear combo selection head with scale 0.001.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_0_selection_head_linear_combo_scale_0.001 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.001 \
  --random_seed 1337
```


Linear combo selection head with scale 0.001, with another seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_1_selection_head_linear_combo_scale_0.001 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.001 \
  --random_seed 1338
```

Linear combo selection head with scale 0.05.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_2_selection_head_linear_combo_scale_0.05 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.05 \
  --random_seed 1337
```

Linear combo selection head with scale 0.1.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_2_selection_head_linear_combo_scale_0.1 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.1 \
  --random_seed 1337
```

Linear combo selection head with scale 0.25.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_2_selection_head_linear_combo_scale_0.25 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.25 \
  --random_seed 1337
```

Linear combo selection head with scale 0.5.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_2_selection_head_linear_combo_scale_0.5 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.5 \
  --random_seed 1337
```

Let's check the protection zero one more time.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_3_protection_zero \
  --protection_kind zero \
  --random_seed 1337
```

Let's check with a huge constant protection head - this should be equivalent to no selectivity.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group scratch_pretrain_with_extra_head_6 \
  --log_dir run_3_protection_huge_constant \
  --protection_kind big_constant \
  --random_seed 1337
```


## Testing selection head lr/weight decay

Hypothesis: the linear head with no weight decay will perform as well as the normal selective attention.

The effect of init and change in lr will be a wash.

<hr>

Results: Honestly kinda unclear. Big thing is that it's much much closer to the original model.

Head zero, with lr=0, matches normal model, which tells us my code isn't buggy.

Head lr=0.025 seems pretty close to the normal model.

Nothing seems better than the normal model, which is kinda sad. Maybe you rly do just need to allocate a dedicated head. I wonder if there's some batchnorm lesson here?

Hrrm, ig that could possibly make it much more stable. Maybe it's worth trying. After all, it is just one more experiment.

Well, why do I want a linear head? Because I'd like to CPT more easily. And because I'd like to not allocate a dedicated protection head.

So those dreams seem a little bit dashed. normal_init linear head just seems much worse (i.e. 0.02 worse pplx, which is so huge!!) than the normal model.

Hrrm maybe when experimenting w/ protection models, I can just assume they *could* get 0.02 better pplx with a dedicated head. Hrrm. Is this actually true in practice with protection models? Let's check later.

OK, conclusion is just that linear head is prob no worse than a normal model. But I haven't yet found a way that it's better.

Onto debugging the protection models!

<hr>

Normal init, no weight decay.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir normal_head_init \
  --selection_head_linear_combo true \
  --random_seed 1337
```

Init head 0 with 1.0.


```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir head_0_init_1.0 \
  --selection_head_linear_combo with_head_zero \
  --random_seed 1337
```

Init head 0 with 1.0, lr 0.1.

```vastrunning/17952360
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir head_0_init_1.0_lr_0.1 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.1 \
  --random_seed 1337
```

Normal model, no linear head.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir normal_model_no_linear_head \
  --random_seed 1337
```

Init head 0 with 1.0, lr 0.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir head_0_init_1.0_lr_0.0 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.0 \
  --random_seed 1337
```

Init head 0 with 1.0, lr 0.1.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir head_0_init_1.0_lr_0.1 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.1 \
  --random_seed 1337
```

Init head 0 with 1.0, lr 0.1, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir head_0_init_1.0_lr_0.1 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.1 \
  --random_seed 1338
```

Init head 0 with 1.0, lr 0.05.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir head_0_init_1.0_lr_0.05 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.05 \
  --random_seed 1337
```

Init head 0 with 1.0, lr 0.05, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir head_0_init_1.0_lr_0.05 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.05 \
  --random_seed 1338
```

Init head 0 with 1.0, lr 0.025.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir head_0_init_1.0_lr_0.025 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.025 \
  --random_seed 1337
```

Init head 0 with 1.0, lr 0.025, with a second seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_selection_head_lr_weight_decay \
  --log_dir head_0_init_1.0_lr_0.025 \
  --selection_head_linear_combo with_head_zero \
  --selection_head_linear_combo_scale 0.025 \
  --random_seed 1338
```


## Testing protection zero vs. protection none

I'm looking for the protection=zero (i.e. running protect_and_attack_triton with 0 protection, which should be equiv. to just cumsum) loss curves to match the protection=none (i.e. running torch.cumsum directly) loss curves.

I don't think they will, though.

In the course of this experiment, o3-mini told me it might be numeric instability problems. So I also ran a custom Triton cumsum implementation.

Not sure if there are many valid *hypotheses* here, since this run was kinda continuous. I was starting and stopping runs based on partial loss curves, so it's all kinda corrupted.

<hr>

Results: it's definitely a numeric instability problem!

Custom cumsum has the same (maybe a bit worse) instability problems as the protection=zero (protect_and_attack_triton with P=0).

OK, so let's figure out how to fix it. Naive first guess might be to use float64 for my accumulators. This'll prob be super slow, but worth doing.

For now, let's just iterate on my custom cumsum implementation, not using the protection=zero.

<hr>

Protection zero:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_zero \
  --protection_kind zero \
  --random_seed 1337
```

Protection zero, with another seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_zero_2 \
  --protection_kind zero \
  --random_seed 1338
```

Protection none:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_none \
  --protection_kind none \
  --random_seed 1337
```

Protection none, with another seed.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_none_2 \
  --protection_kind none \
  --random_seed 1338
```

Protection zero, with 128 seq len.


```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_zero_128 \
  --protection_kind zero \
  --seq_len 128 \
  --random_seed 1337
```

Protection none, with 128 seq len.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_none_128 \
  --protection_kind none \
  --seq_len 128 \
  --random_seed 1337
```


Protection zero, with 64 seq len.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_zero_64 \
  --protection_kind zero \
  --seq_len 64 \
  --random_seed 1337
```

Protection none, with 64 seq len.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_none_64 \
  --protection_kind none \
  --seq_len 64 \
  --random_seed 1337
```

Protection none, with 32 seq len.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_none_32 \
  --protection_kind none \
  --seq_len 32 \
  --random_seed 1337
```

Protection zero, with 32 seq len.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_zero_32 \
  --protection_kind zero \
  --seq_len 32 \
  --random_seed 1337  
```

Custom cumsum implementation. Should hopefully have the same instability problems as the none protection.

```vast:finished
cd /workspace/context-compression && git checkout andrew/protection-debugging && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection \
  --log_dir protection_none_custom_cumsum \
  --protection_kind none_custom_cumsum \
  --random_seed 1337
```

## Testing local cumsum numeric stability

Can we repro the numeric instability problems on a single 4070S machine?

I'm guessing it shows up most for low lrs. So maybe let's just run some local runs with i.e. 500 max_steps, and plot them on wandb!

Yes, we can!! See [this graph](https://wandb.ai/sesamestrong/context_compression?nw=n5szysj3rim).

<hr>

Custom cumsum impl.

```
SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir custom_cumsum \
  --protection_kind none_custom_cumsum \
  --max_steps 500
```

Torch cumsum impl.

```
SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir baseline \
  --protection_kind none \
  --max_steps 500
```

Custom cumsum impl, as a thin layer on top of torch.cumsum.

```
SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir thin_wrapper_cumsum \
  --protection_kind custom_cumsum \ 
  --max_steps 500
```

Custom cumsum impl, with parallel scan.

```
SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir parallel_scan_cumsum \
  --protection_kind none_custom_cumsum_parallel \
  --max_steps 500
```