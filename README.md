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

