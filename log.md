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

UPDATE: this actually matched the baseline loss curve!!!
Amazing!!!

```
SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir parallel_scan_cumsum \
  --protection_kind none_custom_cumsum_parallel \
  --max_steps 500
```

```
SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir bliasson_cumsum \
  --protection_kind none_custom_cumsum_bliasson \
  --max_steps 500
```

Stable protect-and-attack impl, with protection=zero. Should match the baseline loss curve.

```
SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir stable_attack_and_protect \
  --protection_kind zero \
  --max_steps 500
```


## Testing stable protect-and-attack

Protection zero.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection_2 \
  --log_dir protection_zero_3 \
  --protection_kind zero \
  --random_seed 1337
```

Protection none:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection_2 \
  --log_dir protection_none_3 \
  --protection_kind none \
  --random_seed 1337
```

Protection head2:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection_2 \
  --log_dir protection_head2_3 \
  --protection_kind head_two \
  --random_seed 1337
```

Protection zero (again)

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection_2 \
  --log_dir protection_zero_4 \
  --protection_kind zero \
  --random_seed 1337
```

Protection head2 (again)

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection_2 \
  --log_dir protection_head2_4 \
  --protection_kind head_two \
  --random_seed 1337
```

Protection zero, with no compile

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_stable_protect_and_attack \
  --log_dir protection_zero_0_no_compile \
  --protection_kind zero \
  --max_steps 500 \
  --no_use_compile \
  --batch_size 4
```

Protection head2, with no compile

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_stable_protect_and_attack \
  --log_dir protection_head2_0_no_compile \
  --protection_kind head_two \
  --max_steps 500 \
  --no_use_compile \
  --batch_size 4
```

Protection none, with no compile and cumsum debugging.

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir protection_none_0_no_compile_cumsum_debugging \
  --protection_kind none \
  --max_steps 500 \
  --no_use_compile \
  --batch_size 4
```

Protection zero, with no compile and cumsum debugging.

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir protection_zero_0_no_compile_cumsum_debugging \
  --protection_kind zero \
  --max_steps 500 \
  --no_use_compile \
  --batch_size 4
```

Protection zero, with compile and cumsum debugging.

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir protection_zero_0_compile_cumsum_debugging \
  --protection_kind zero \
  --max_steps 500
  --batch_size 4
```

Protection zero, with compile and fp64 and cumsum debugging.

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir protection_zero_0_compile_fp64_cumsum_debugging \
  --protection_kind zero_fp64 \
  --max_steps 500
```

## OK, let's now figure out an fp64 protect-and-attack impl. And hopefully make torch.compile work again.

FP64 mini-model for cumsum. Should hopefully be super accurate. And hopefully better loss than the fp32 one. OK maybe that's a pipe dream.

```
DEBUG_CUM_SUM=true SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir fp64_bliasson_cumsum \
  --protection_kind none_custom_cumsum_bliasson_fp64 \
  --max_steps 500
```

FP64 mini-model for attack-and-protect zero. Should do everything that cumsum just did.

```
DEBUG_CUM_SUM=true SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir fp64_attack_and_protect_zero \
  --protection_kind zero_fp64 \
  --max_steps 500
```

FP64 torch.cumsum

```
DEBUG_CUM_SUM=true SKIP_WANDB=false python -m context_compression.train \
  --group testing_cumsum_numeric_stability \
  --log_dir fp64_torch_cumsum \
  --protection_kind none_torch_cumsum_fp64 \
  --max_steps 500
```

## OK, FP64 seems ok? protect-and-attack def seems worse than normal cumsum, which I should investigate more. But let's try running protection=0 on the real model.

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection_2 \
  --log_dir protection_zero_1_compile_fp64_cumsum_debugging \
  --protection_kind zero_fp64 \
  --max_steps 500
  --batch_size 2
```

Let's rerun with head_two with no compile and debug enabled.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_protection_2 \
  --log_dir protection_head_two_1_no_compile \
  --protection_kind head_two_fp64 \
  --max_steps 500
  --no_use_compile
```

## OK, it's rly slow. We should fix this somehow.

I think it's slowing down the whole model, which is crazy. We should maybe use a cuda kernel, I think. IDK if there's an easier way.

I wonder if the CUDA kernel would be too crazy hard? ig I could make one in a notebook somewhere, and associative scan is prob a common algo.

Then I'd have to impl a second version for the attack-and-protect scenario. Which would honestly prob take like 15 more minutes. So not horrible.

But I would have to make it generic w.r.t. the dtype. On the one hand, might be educational. On the other, prob a lot of work!

Hrrm. Is there anything we can do to hold this off.

Well what do I know?

Well, I don't know if bliasson cumsum makes a full training run better or worse. B/c it's so damn slow.

I don't know why Bliasson attack-and-protect head2 did so much worse on its most recent run.

I don't know if attack-and-protect with protection=0 is fundamentally worse than bliasson cumsum. (I suspect it's a little worse, since the loss curve was slightly worse on the mini model run.)

I don't know if attack-and-protect with protection=0 gets worse loss on the real model than Bliasson cumsum does.

I'm dealing with like a 4x slowdown by switching from none -> zero_fp64. Debilitating, how can I get it back.

OK, let's go back to basics. Let's run *experiments*, where the goal of *experiments* is to understand my current code better. Success is measured by whether I answered these questions.

## Understanding if/how torch.compile affects real training runs

Let's do a run with protection=none_custom_cumsum_bliasson, with torch.compile enabled vs. disabled. I expect torch.compile will be slightly worse.

Short-run result: [wandb graph](https://wandb.ai/sesamestrong/context_compression?nw=j1cu4nr6b8r).

Warning: I think there's just a lotta inter-run variation, that's a function of the seed, the numerics, etc.

So the numbers were lower for torch.compile disabled. Pretty consistently, actually, but not by much. It might be because torch.compile is worse for bliasson.

A datapoint against that belief might be that for torch.cumsum, the difference between torch.compile enabled and disabled was similarly consistent but in the OPPOSITE direction.

So probably I don't actually know which is better.

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group measuring_instability_2 \
  --log_dir bliasson_cumsum_compiled \
  --protection_kind none_custom_cumsum_bliasson \
  --batch_size 4
```

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group measuring_instability_2 \
  --log_dir bliasson_cumsum_not_compiled \
  --protection_kind none_custom_cumsum_bliasson \
  --no_use_compile \
  --batch_size 4
```

## Understanding grad diffs in real training runs

So our torch.compile experiment will give us grad diffs and loss curves for protection=zero.

Let's get some for protection=none, with compile enabled.

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group measuring_instability_2 \
  --log_dir none_torch_compile \
  --protection_kind none \
  --batch_size 4
```

With another seed:

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group measuring_instability_2 \
  --log_dir none_torch_compile_2 \
  --protection_kind none \
  --batch_size 4 \
  --random_seed 1338
```

You know, and since it's cheap and fast, let's also do protection=none with compile disabled.

I hypothesize that these will be basically identical.

Short-run result: [wandb graph](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=j1cu4nr6b8r).

Not identical. There is a consistent difference over the last steps of the run, but it's not that big.

So I think I don't acc know if they're different or not.

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group measuring_instability_2 \
  --log_dir none_not_compiled \
  --protection_kind none \
  --no_use_compile \
  --batch_size 4
```

With another seed:

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group measuring_instability_2 \
  --log_dir none_not_compiled_2 \
  --protection_kind none \
  --no_use_compile \
  --batch_size 4 \
  --random_seed 1338
```


And let's check for protection=zero, with fp64 and compile enabled. A previous loss curve made this look possibly as good as protection=none.
If this really was as good as protection=none, then an experiment with protection=head_two and fp64 would be in order.

Hypothesis: this run will have similar/lower grad diffs than protection=none, and a very similar/lower loss curve.

Short-run result: [wandb graph](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=j1cu4nr6b8r).

Looks like it, maybe! But again, I don't have a ton of resolution. I need to wait on the longer run.

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group measuring_instability_2 \
  --log_dir zero_fp64_compiled \
  --protection_kind zero_fp64 \
  --batch_size 4
```

With another seed:

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group measuring_instability_2 \
  --log_dir zero_fp64_compiled_2 \
  --protection_kind zero_fp64 \
  --batch_size 4 \
  --random_seed 1338
```

Actually, let's also do a run with protection=head_two and fp64. It's the run I meant to do last night. For random path-dependent reasons, I didn't. But I should have.

Hypothesis: this will be better than protection=none.

Short-run result: [wandb graph](https://wandb.ai/sesamestrong/context_compression/panel/7i8l8u02w?nw=j1cu4nr6b8r).

Nope, it's def worse on this short-run graph. It's possible that it biases the model towards less selectivity... in which case I should maybe try adding diff scaling factors, or maybe biases, to P.

But it's a v short run that goes down to a very high loss. So I should wait for the longer run.


```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group measuring_instability_2 \
  --log_dir head_two_fp64_compiled \
  --protection_kind head_two_fp64 \
  --batch_size 4
```

## Understanding loss curve orderings in the mini model

We can experiment much faster here.

I'm kinda suspicious about ordering the "quality" of numerics based on their loss curve with a single seed.

Since honestly, the max error for most of these runs is small.

OK. Once I launch the 2500-step runs for all of these, I can start experimenting on the mini model.

Update: the 500-step run makes me a little hopeful that my fp64 numerics are good enough to run with. So I'm gonna hold off from these experiments for now, I think.


## Understanding head_two being worse, new baselines losing to old baselines

Well, it's worse on both big models and small models.

I *think* masking is better on both big models and small models. So maybe we can just experiment on small models?

I'm curious what the failure mode is for head_two. Is it that it's too easy to protect ppl from masking? i.e. barely any tokens are masked? Is it that it removes a head?
Is it that almost all masking is just super local, so protection is not necessary? This would be interesting.
Is it that protection is actually a pretty binary decision (i.e. to mask or not to mask myself), so by choosing not to mask, you're doing enough? In which case protection is not necessary? (this may even work w/ the KVs of a given token in just one layer)
Should I expect some theoretical motivation for protection being better? i.e. Leviathan said their motivation for selective attention was from writing transformer programs. I haven't motivated my search in the same way. Should I find inspiration by solving toy problems?

Hrrm. So what am I looking for in my quest for understanding.

IG I could make a visualization of masking patterns with and without protection. Hrrm that would fit in 2 hours.

I could try to think of where a model with selective attn spends unnecessary circuitry.

I could try to look at some examples of maskings and see how much they could be improved. i.e. that is slightly more of a problem-focused and less solution-focused answer.

I could think hard abt what protection needs to be a pareto improvement. i.e. maybe some scale factor or bias that I crank up.

Hrrm ok. I'll do smth in between. I'll kickoff a bunch of variants like this of head two (including an fp32 version), and see if any close the gap.

Then while those are training, I will make a selectivity visualization tool. This should be fun and educational. It would be fun to do it on a code pretraining set. I wonder if fineweb has a code split. I'm guessing no.

Hrrm... after looking at the new loss curves vs. the old loss curves, they seem different and worse! And by a big margin too - what's going on here? Does it show up in mini model loss curves too?

Hrrm... actually the story changes if you look at the ce_loss vs. pplx curves. Why? Is this some weird wandb nonsense with gathering gradients, maybe? I doubt it but it's possible.

Hrrm we can probably just plug the ce loss into an expression for perplexity to see what is the mismatch.

We can maybe repro this inconsistency w/ the mini model, then git bisect it on an 8x4090 rig.

Note: we can prob speed up the mini model by increasing its microbatch size.

Q: should I even go down this rabbithole?

Several possible options:

- I suspect that cumsum debugging is causing it. So I turn that off and everything is fixed.
- I decide it's some random measurement fluke, and the new models are as good as the old models.
  - We can settle this by uploading a model trained with the fastest setting to HF. Hrrm maybe let's just do that? We can figure out the uploading-to-HF stuff at the end.
  - Hrrm. Do we have the old baseline model performances to compare to? Looks like it. OK let's do this thing.
- I decide it's an actual bug that's degrading the model, but it doesn't affect the ordering of my experiments.
- It's a bug, and it does affect the ordering/magnitude of differences.
- It's a bug, but it'll take too long to solve.

<hr>

OK, hypotheses:

- When I train a model with protection=none, its final validation loss, when measured on the same software, will be basically the same as the baselines from the past.

- Using the fp32 Bliasson algorithm will not noticeably widen the gap between protection and non-protection.

- Adding a 1/5x scaling factor to the protection head WILL close the gap somewhat.

- Adding a bias=1 to the protection head WILL close the gap somewhat.

- Setting bos_protection to false will not noticeably change the head_two loss curve. Which will be a good sign.

- Setting bos_protection to false will noticeably change the baseline loss curve. Which is expected (from previous experiments).

<hr>

Protection=none model:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir protection_none_torch_compile \
  --protection_kind none
```

fp32 Bliasson protection=zero (should ~match protection=none):

```vast:finished
cd /workspace/context-compression && git pull && DEBUG_CUM_SUM=true torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir zero_fp32_bliasson_torch_compile \
  --protection_kind zero
```

Protection=head_two_fp64 and 1/5x scaling factor:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir head_two_fp64_torch_compile_1_5x_scaling_factor \
  --protection_kind head_two_fp64
  --protection_head_scaling_factor 0.2 \
  --batch_size 4
```

Protection=head_two_fp64 and bias=-0.1:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir head_two_fp64_torch_compile_bias_minus_0_1 \
  --protection_kind head_two_fp64 \
  --protection_head_bias -0.1
```

Protection=head_two_fp64 with bos_protection=false (failed b/c of bad args):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir head_two_fp64_bos_protection_false \
  --protection_kind head_two_fp64 \
  --no_protect_bos_token
```

Protection=none with bos_protection=false (failed b/c of bad args):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir protection_none_bos_protection_false \
  --protection_kind none \
  --no_protect_bos_token
```

## Super-fast mini model experiments (testing the same things as above, hopefully like 50x faster)

When I come back from lunch, I should a) make my microbatch bigger for the mini model, and then b) run the same experiments as above on my mini model.

Let's try for 50x faster!!!

Protection=none model:

```
SKIP_WANDB=false python -m context_compression.train \
  --group super_fast_mini_model_experiments_1k \
  --log_dir protection_none_torch_compile \
  --protection_kind none
```

fp32 Bliasson protection=zero (should be equally good as protection=none):

```
SKIP_WANDB=false python -m context_compression.train \
  --group super_fast_mini_model_experiments_1k \
  --log_dir zero_fp32_bliasson_torch_compile \
  --protection_kind zero
```

fp32 Bliasson protection=head_two:

```
SKIP_WANDB=false python -m context_compression.train \
  --group super_fast_mini_model_experiments_1k \
  --log_dir head_two_fp32_bliasson_torch_compile \
  --protection_kind head_two
```

fp64 Bliasson protection=head_two (should be equally good as fp32 Bliasson protection=head_two):

```
SKIP_WANDB=false python -m context_compression.train \
  --group super_fast_mini_model_experiments_1k \
  --log_dir head_two_fp64_bliasson_torch_compile \
  --protection_kind head_two_fp64
```

fp32 Bliasson protection=head_two with bos_protection=false (should match normal head_two)

```
SKIP_WANDB=false python -m context_compression.train \
  --group super_fast_mini_model_experiments_1k \
  --log_dir head_two_fp32_bliasson_bos_protection_false \
  --protection_kind head_two \
  --no_protect_bos_token
```

protection=none with bos_protection=false (should be slightly worse than normal none):

```
SKIP_WANDB=false python -m context_compression.train \
  --group super_fast_mini_model_experiments_1k \
  --log_dir protection_none_bos_protection_false \
  --protection_kind none \
  --no_protect_bos_token
```


## Understanding head_two being worse, part 2

It's confusing how protect_bos_token=false performed better on protection=none. Is this just because of one seed? Or is my model different than it was before (when I determined that protect_bos_token=false was worse)? So let's run this another time, with another round of seeds, for both protection=none and protection=head_two.

Protection=none, with a second seed:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir protection_none_2 \
  --protection_kind none \
  --random_seed 1338
```

Protection=head_two_fp64 and 1/50x scaling factor:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir head_two_fp64_1_50x_scaling_factor \
  --protection_kind head_two_fp64 \
  --protection_head_scaling_factor 0.02 \
  --random_seed 1338
```

Protection=head_two_fp64 and bias=-1:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir head_two_fp64_bias_minus_1 \
  --protection_kind head_two_fp64 \
  --protection_head_bias -1.0 \
  --random_seed 1338
```

Protection=head_two_fp64 with bos_protection=false and a second seed:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir head_two_fp64_bos_protection_false_2 \
  --protection_kind head_two_fp64 \
  --no_protect_bos_token \
  --random_seed 1338
```

Protection=none with bos_protection=false and a second seed:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir protection_none_bos_protection_false_2 \
  --protection_kind none \
  --no_protect_bos_token \
  --random_seed 1338
```

Protection=head_two_fp64 and bias=-10:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir head_two_fp64_bias_minus_10 \
  --protection_kind head_two_fp64 \
  --protection_head_bias -10.0 \
  --random_seed 1338
```

Protection=head_two_fp64:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group debugging_head_two_and_baselines \
  --log_dir head_two_fp64 \
  --protection_kind head_two_fp64 \
  --random_seed 1338
```



## Allowing more selection patterns

I didn't write down any hypotheses for this. My memory says I thought one mask per head would be the best. But I was wrong. I'm not sure why - it might just be that there's less positive transfer when you're training lots of independent heads.

BUT having two masks beat baselines. AFAICT, it's the first thing that has beaten baselines. This might just be because it freed up the 13th head, or because it found a way to use the extra FLOPs. And I should check if there was actually any difference between the two masks for each layer.

Ah, *damn* - I accidentally deleted the trained weights for that run. OK, let's try it again with only one new mask, or 2 new masks. And are we comparing to n=12 heads or n=13 heads? What's the most honest comparison moving forward?

Internally (i.e. between my experiments), 13 seems best. Well, I guess the most honest answer is that I'm constraining myself to that total # of flops (basically).

OK, so let's stick with 13 heads, but try to make the various mask patterns fit into that one head worth of KV cache space.

ALSO, I think for now I should stop with the protection experiments. They're 4x slower than other experiments, and they don't seem super promising.

<hr>

One mask per head (with 12 heads):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir one_mask_per_head \
  --selection_head_linear_combo one_mask_per_head \
  --n_heads 12
```

One mask per head, with 13 heads (for consistency):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir one_mask_per_head_13_heads \
  --selection_head_linear_combo one_mask_per_head \
  --n_heads 13 \
  --batch_size 4
```

Two masks, with 13 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir two_masks_13_heads \
  --selection_head_linear_combo two_masks \
  --n_heads 13 \
  --batch_size 4
```

<hr>

Two masks (with 12 heads, so the 13th head is dedicated to selection):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir two_masks_12_heads \
  --selection_head_linear_combo two_masks \
  --n_heads 12
```

One mask, shared across 1 head worth of KV cache space (should be identical to baseline):

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir one_mask_shared_1_head \
  --selection_head_linear_combo n_sliced_masks \
  --n_heads 12 \
  --n_sliced_masks 1
```

Two masks, but shared across 1 head worth of KV cache space:

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir two_masks_shared_1_head \
  --selection_head_linear_combo n_sliced_masks \
  --n_heads 12 \
  --n_sliced_masks 2
```

One mask per head, but all constructed from one latent mask:

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir one_mask_per_head_1_latent_vector \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1
```

One mask per head, but constructed from 2 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir one_mask_per_head_2_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 2 \
  --batch_size 4
```

One mask per head, but constructed from 4 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir one_mask_per_head_4_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 4 \
  --batch_size 4
```

Residual attention masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group residual_attention_masks \
  --log_dir residual_attention_masks \
  --residual_attention_masks
```


## Getting mup to work

Initial run (not shown) running with width=128, seqlen=256, small bs for 2500 steps.

Then a run (not shown) with 2x bigger bs, for 1000 steps. This helped.

Then a run (not shown) with 2x the width of that one. This made loss worse!! This should never happen under mup.


OK, let's try the equivalent with non-selective attention.

```
python -m context_compression.train --group testing_mup   --log_dir unselective_big_bs_1x --disable_selection --n_heads 4
```

```
python -m context_compression.train --group testing_mup   --log_dir unselective_big_bs_2x --disable_selection --n_heads 8
```

```
python -m context_compression.train --group testing_mup   --log_dir unselective_big_bs_4x --disable_selection --n_heads 16 --batch_size 5
```

Damn, 4x is worse than 2x is worse than 1x (see [this view](https://wandb.ai/sesamestrong/context_compression?nw=8hz2r8dk1kd)). Clearly my mup implementation is buggy.

### Testing out coord-checking

Command to make it write coord scales to a CSV:

```
SKIP_WANDB=false python -m context_compression.train --log_dir /tmp/dummy --group making_mup_work --mup_enable_coord_check_logging --attention_kind selective --disable_selection --no_use_compile --mup --max_steps 10 --no_decay_lr
```

After finding a bug (see details in [this report](https://wandb.ai/sesamestrong/context_compression/reports/Coord-checks---VmlldzoxMTU1Mjc0MQ)), let's re-run those commands:

### Part two

```
python -m context_compression.train --group testing_mup_2   --log_dir unselective_big_bs_0.25x --disable_selection --n_heads 1 --mup
```

```
python -m context_compression.train --group testing_mup_2   --log_dir unselective_big_bs_0.5x --disable_selection --n_heads 2 --mup
```

```
python -m context_compression.train --group testing_mup_2   --log_dir unselective_big_bs_1x --disable_selection --n_heads 4 --mup
```

```
python -m context_compression.train --group testing_mup_2   --log_dir unselective_big_bs_2x --disable_selection --n_heads 8 --mup
```

```
python -m context_compression.train --group testing_mup_2   --log_dir unselective_big_bs_4x --disable_selection --n_heads 16 --batch_size 5 --mup
```

Result: yup, 4x beats 2x beats 1x! See [this view](https://wandb.ai/sesamestrong/context_compression?nw=ndftzni7u9).
BUT it seems like the loss for all of them is slightly worse than the previous runs (see [this view](https://wandb.ai/sesamestrong/context_compression?nw=8hz2r8dk1kd)).
Hrrm - let's compare 2x (old) to 2x (new). 2x (new) is worse. BUT 2x (old) descends FASTER than 2x (new)!! What's going on here?

So they had lr warmup. ig we do too... I bet our HPs are just generally worse - it's prob not a bug?

### Part three

Let's try to optimize the HPs on the proxy model. For now, let's use 0.25x for speed. Ooh, should we try doing a HP search on this? Could just use a bash script for it... Would be possibly fun! OK let's try it actually.

### Is the mini model usable with mup on?

See [this report](https://wandb.ai/sesamestrong/context_compression/reports/Mini-model-to-simulate-big-model--VmlldzoxMTQ4Mjk5Ng).

Mup was motivated by the ordering of loss curves for the flexible-selection-pattern experiment. So let's re-run that on the mini model! Hopefully we can a) get much lower loss than before, which would mean we're better-simulating the bigger model and b) recovering the big-model ordering.

Let's run one_mask_per_head_4_latent_vectors:

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group flexible_selection_pattern_mini_model_2 \
  --log_dir one_mask_per_head_4_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 4 \
  --batch_size 4 \
  --mup
```

one_mask_shared_1_head (baseline):

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group flexible_selection_pattern_mini_model_2 \
  --log_dir one_mask_shared_1_head \
  --selection_head_linear_combo n_sliced_masks \
  --n_heads 12 \
  --n_sliced_masks 1 \
  --batch_size 4 \
  --mup
```

two_masks_12_heads (baseline):

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group flexible_selection_pattern_mini_model_2 \
  --log_dir two_masks_12_heads \
  --selection_head_linear_combo two_masks \
  --n_heads 12 \
  --batch_size 4 \
  --mup
```

one_mask_per_head_2_latent_vectors:

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group flexible_selection_pattern_mini_model_2 \
  --log_dir one_mask_per_head_2_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 2 \
  --batch_size 4 \
  --mup
```




## Can I just train the same model with smaller bs and higher lr?

Baseline:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train --group shrinking_big_runs_2 --log_dir baseline --n_heads 12 --total_batch_size 524288 --batch_size 4 --max_lr 6e-5
```

Half-total-bs, sqrt-lr:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train --group shrinking_big_runs_2 --log_dir half_total_bs_sqrt_lr --n_heads 12 --total_batch_size 262144 --batch_size 4 --max_lr 4e-5
```

Same-total-bs, half-seq-len:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train --group shrinking_big_runs_2 --log_dir same_total_bs_half_seq_len --n_heads 12 --total_batch_size 524288 --batch_size 16 --max_lr 6e-5 --seq_len 512
```

4x-smaller-bs, half-lr, half-seq-len:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train --group shrinking_big_runs_2 --log_dir 4x_smaller_bs_half_lr_half_seq_len --n_heads 12 --total_batch_size 131072 --batch_size 16 --max_lr 1.5e-5 --seq_len 512
```


## Comparing past commit performances - bisect major regression (see loss curves [here](https://wandb.ai/sesamestrong/context_compression?nw=0iuwx2zct3i))

Summary: problem was that the attention multiplier was inverted!!

```vast:finished
cd /workspace/context-compression && git fetch && git checkout 88638d4e0d93 && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  88638d4e0d93  --n_heads 12
```

```vast:finished
cd /workspace/context-compression && git pull && git checkout 37758280b && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  ef11f972cd6  --n_heads 12
```

```vast:finished
cd /workspace/context-compression && git fetch && git checkout 55097b0f && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  2fcffe529  --n_heads 12
```

```vast:finished
cd /workspace/context-compression && git fetch && git checkout 18cdb81144 && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  5b9467b8  --n_heads 12
```

```vast:finished
cd /workspace/context-compression && git fetch && git checkout 056a8dc2408 && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  056a8dc2408  --n_heads 12
```

```vast:finished
cd /workspace/context-compression && git fetch && git checkout 1de663ff31 && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  1de663ff31  --n_heads 12
```

REGRESSED AT OR BEFORE THIS COMMIT ^^^ - this is it! 1de663ff31

REGRESSED AFTER THIS COMMIT VVVV

```vast:finished
cd /workspace/context-compression && git fetch && git checkout bbd19ad7267 && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  bbd19ad7267  --n_heads 12
```

```vast:finished
cd /workspace/context-compression && git fetch && git checkout d03690efe30 && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  d03690efe30  --n_heads 12
```

```vast:finished
cd /workspace/context-compression && git pull && git checkout 7fb24853684d6efcd52c13cf39d4f && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  7fb24853684d6  --n_heads 12
````

```vast:finished
cd /workspace/context-compression && git pull && git checkout cb9f28c && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  cb9f28c  --n_heads 12
```


### Hopefully fixing regressions

```vast:finished
cd /workspace/context-compression && git fetch && git checkout 0471b8fde && torchrun --nproc_per_node=gpu -m context_compression.train   --group fix_regressions   --log_dir  0471b8fde  --n_heads 12
```

See all runs [here](https://wandb.ai/sesamestrong/context_compression?nw=0iuwx2zct3i).

The attention multiplier was inverted - no idea how I passed coord checks, crazy.

## Testing my mup (seems pretty good!)

```vast:finished
cd /workspace/context-compression && git fetch && git checkout 24a09f5 && SUFFIX=baseline FLAGS=" " USE_MINI_MODEL=true torchrun --nproc_per_node=4 sweeps/run_sweep.py sweeps/n_heads_12_mini.sh
```

## 2xing bs again on small model

```vast:finished
cd /workspace/context-compression && git fetch && git checkout 649bd3e47a3 && SUFFIX=baseline FLAGS=" " USE_MINI_MODEL=true torchrun --nproc_per_node=2 sweeps/run_sweep.py sweeps/sweep_lrs_and_seeds.sh
```

Update: seems like the best config is total_batch_size=128*1024, lr=12e-4 on 4x4090s - see loss curves [here](https://wandb.ai/sesamestrong/context_compression?nw=06h76kb5qg3).

Notice that the 128*1024 total_batch_size on 4x4090s reaches a lower loss than the 256*1024 total_batch_size on 8x4090s, AND takes about the same time (15 mins).

It *should have* taken 2x longer! I think there's just some communication overhead maybe for 8x4090s? Not sure.

Either way, I'll go with the 128*1024 setting.

## Baseline pretty-fast four-4090 run

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 11e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --n_heads 4 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/baseline \
--key baseline \
--random_seed 1339
```

## Allowing more selection patterns, pt. 2

One mask per head, but constructed from 4 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 11e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --n_heads 4 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/one_mask_per_head_4_latent_masks \
--key one_mask_per_head_4_latent_masks \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 4
```

One mask per head, but constructed from 2 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 11e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --n_heads 4 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/one_mask_per_head_2_latent_masks \
--key one_mask_per_head_2_latent_masks \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2
```

One mask per head, but constructed from 1 latent mask:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 11e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --n_heads 4 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/one_mask_per_head_1_latent_mask \
--key one_mask_per_head_1_latent_mask \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1
```

Two masks, with 4 heads:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 11e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --n_heads 4 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/two_masks_4_heads \
--key two_masks_4_heads \
--random_seed 1339 \
--selection_head_linear_combo two_masks
```

Result of repro attempt: 2 masks is a bit better than baseline, but all the latent stuff is worse than baseline. (in the big-model graph, 2-mask and baseline are ~identical. That's a mismatch)
Difference between latent runs is negligible. (in the big-model graph, 4-latent is much worse than 2-latent. That's a mismatch)

So big important thing: latent masks are fundamentally bad somehow. But the micro details are wrong. Maybe let's try training a 12-head model with a smaller head dim? Hopefully that can be more faithful to the big-model results.

## Allowing more selection patterns, pt. 3

Let's use head_dim=22, n_heads=12 for this.

Baseline run:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 11e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --n_embd 264 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_11e-4 \
--key 12_head_baseline_lr_11e-4 \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 6e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --n_embd 264 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_6e-4 \
--key 12_head_baseline_lr_6e-4 \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 16e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --n_embd 264 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_16e-4 \
--key 12_head_baseline_lr_16e-4 \
--random_seed 1339
```

Result: ~~Seems like the 16e-4 and 11e-4 runs are pretty good. Let's go with 16e-4.~~

OOPS! I accidentally used 64-dim heads! So this was actually a super slow, big model.

Let's run it again with head_dim=22.

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 14e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_14e-4_head_dim_22 \
--key 12_head_baseline_lr_14e-4 \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 20e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_20e-4_head_dim_22 \
--key 12_head_baseline_lr_20e-4 \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 16e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_16e-4_head_dim_22 \
--key 12_head_baseline_lr_16e-4 \
--random_seed 1339
```


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_30e-4_head_dim_22 \
--key 12_head_baseline_lr_30e-4 \
--random_seed 1339
```


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 45e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_45e-4_head_dim_22 \
--key 12_head_baseline_lr_45e-4 \
--random_seed 1339
```


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 70e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_70e-4_head_dim_22 \
--key 12_head_baseline_lr_70e-4 \
--random_seed 1339
```


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 100e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_100e-4_head_dim_22 \
--key 12_head_baseline_lr_100e-4 \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 40e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_40e-4_head_dim_22 \
--key 12_head_baseline_lr_40e-4 \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 35e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_35e-4_head_dim_22 \
--key 12_head_baseline_lr_35e-4 \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 50e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_50e-4_head_dim_22 \
--key 12_head_baseline_lr_50e-4 \
--random_seed 1339
```

Oops - I had a mup bug with custom head_dim. Let's rerun with the fixed bug.

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 8e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_8e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_8e-4_fixed \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 10e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_10e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_10e-4_fixed \
--random_seed 1339
```


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 12e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_12e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_12e-4_fixed \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 14e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_14e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_14e-4_fixed \
--random_seed 1339
```


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 16e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_16e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_16e-4_fixed \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 20e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_20e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_20e-4_fixed \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 25e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_25e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_25e-4_fixed \
--random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_30e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_30e-4_fixed \
--random_seed 1339
```

### Selection pattern experiments on the 12-head mini model

One mask per head, but constructed from 4 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_4_latent_masks \
--key 12_mini_head_one_mask_per_head_4_latent_masks \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 4
```

One mask per head, but constructed from 2 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_2_latent_masks \
--key 12_mini_head_one_mask_per_head_2_latent_masks \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2
```

One mask per head, but constructed from 1 latent mask:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_1_latent_mask \
--key 12_mini_head_one_mask_per_head_1_latent_mask \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1
```

Two masks, with 4 heads:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_two_masks_4_heads \
--key 12_mini_head_two_masks_4_heads \
--random_seed 1339 \
--selection_head_linear_combo two_masks
```

### Selection pattern experiments on the full model

One mask per head (with 12 heads):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head \
  --selection_head_linear_combo one_mask_per_head \
  --n_heads 12 \
  --batch_size 4
```

Two masks (with 12 heads, so the 13th head is dedicated to selection):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/two_masks_12_heads \
  --selection_head_linear_combo two_masks \
  --n_heads 12 \
  --batch_size 4
```

Baseline:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/baseline \
  --selection_head_linear_combo none \
  --n_heads 12 \
  --batch_size 4
```

Two masks, but shared across 1 head worth of KV cache space:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/two_masks_shared_1_head \
  --selection_head_linear_combo n_sliced_masks \
  --n_heads 12 \
  --n_sliced_masks 2 \
  --batch_size 4
```

One mask per head, but all constructed from one latent mask:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_1_latent_vector \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1 \
  --batch_size 4
```

One mask per head, but constructed from 2 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_2_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 2 \
  --batch_size 4
```

One mask per head, but constructed from 4 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_4_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 4 \
  --batch_size 4
```

#### With a second seed:

One mask per head (with 12 heads):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head \
  --selection_head_linear_combo one_mask_per_head \
  --n_heads 12 \
  --batch_size 4 \
  --random_seed 1339
```

Two masks (with 12 heads, so the 13th head is dedicated to selection):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/two_masks_12_heads \
  --selection_head_linear_combo two_masks \
  --n_heads 12 \
  --batch_size 4 \
  --random_seed 1339
```

Baseline:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/baseline \
  --selection_head_linear_combo none \
  --n_heads 12 \
  --batch_size 4 \
  --random_seed 1339
```

Two masks, but shared across 1 head worth of KV cache space:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/two_masks_shared_1_head \
  --selection_head_linear_combo n_sliced_masks \
  --n_heads 12 \
  --n_sliced_masks 2 \
  --batch_size 4 \
  --random_seed 1339
```

One mask per head, but all constructed from one latent mask:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_1_latent_vector \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1 \
  --batch_size 4 \
  --random_seed 1339
```

One mask per head, but constructed from 2 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_2_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 2 \
  --batch_size 4 \
  --random_seed 1339
```

One mask per head, but constructed from 4 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_4_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 4 \
  --batch_size 4 \
  --random_seed 1339
```

### Trying more selection pattern seeds, lrs on the 12-head mini model

Halved lr for 1-latent-mask:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 15e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_1_latent_mask_halved_lr \
--key 12_mini_head_one_mask_per_head_1_latent_mask_halved_lr \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1
```

Another seed for baseline:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_30e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_30e-4_fixed \
--random_seed 1340
```

Another seed for baseline:


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_head_baseline_lr_30e-4_head_dim_22_fixed \
--key 12_head_baseline_lr_30e-4_fixed \
--random_seed 1341
```


Another seed for two-masks-4-heads (which is a misnomer, since there are 12 heads):

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_two_masks_4_heads \
--key 12_mini_head_two_masks_4_heads \
--random_seed 1340 \
--selection_head_linear_combo two_masks
```


Another seed for two-masks-4-heads:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_two_masks_4_heads \
--key 12_mini_head_two_masks_4_heads \
--random_seed 1341 \
--selection_head_linear_combo two_masks
```

### Running selection pattern experiments again on the small model

Baseline:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/baseline_lr_30e-4_seed_1339 \
--key baseline_lr_30e-4 \
--random_seed 1339
```

#### One mask per head, but constructed from 1 latent mask (three diff lrs)

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 15e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_1_latent_mask_15e-4_seed_1339 \
--key 12_mini_head_one_mask_per_head_1_latent_mask_15e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 10e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_1_latent_mask_10e-4_seed_1339 \
--key 12_mini_head_one_mask_per_head_1_latent_mask_10e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 8e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_1_latent_mask_8e-4_seed_1340 \
--key 12_mini_head_one_mask_per_head_1_latent_mask_8e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1
```

#### One mask per head, with 1 latent mask, but with layernorm:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 15e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_1_latent_ln_mask_15e-4_seed_1339 \
--key 12_mini_head_one_mask_per_head_1_latent_ln_mask_15e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--S_layernorm
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 10e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_1_latent_ln_mask_10e-4_seed_1339 \
--key 12_mini_head_one_mask_per_head_1_latent_ln_mask_10e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--S_layernorm
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 8e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group repro_selective_pattern_rankings \
--log_dir logs/repro_selective_pattern_rankings/12_mini_head_one_mask_per_head_1_latent_ln_mask_8e-4_seed_1340 \
--key 12_mini_head_one_mask_per_head_1_latent_ln_mask_8e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--S_layernorm
```

#### Coord-checking 12-head model

Baseline vs. 1-latent-mask vs. 1-latent-mask-init-to-one vs. 2-masks vs. 2-sliced-masks.

See view on [wandb](https://wandb.ai/sesamestrong/context_compression?nw=ebejdivq8ir).

Gist is that 1-latent-mask with no special init seems to cause more coord check variation than the others.

Maybe that means it's less stable or smth? I kinda doubt it honestly.

```
export SEED=1342
CUDA_VISIBLE_DEVICES=0 python -m context_compression.train \
--max_lr 1e-3 --total_batch_size 32768 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group latent_mask_coord_check \
--mup_enable_coord_check_logging --no_decay_lr --max_steps 10 --no_use_compile --no_upload_to_hf \
--log_dir logs/latent_mask_coord_check/baseline_seed_${SEED} \
--key baseline \
--selection_head_linear_combo none \
--random_seed ${SEED} & \
CUDA_VISIBLE_DEVICES=1 python -m context_compression.train \
--max_lr 1e-3 --total_batch_size 32768 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group latent_mask_coord_check \
--mup_enable_coord_check_logging --no_decay_lr --max_steps 10 --no_use_compile --no_upload_to_hf \
--log_dir logs/latent_mask_coord_check/1_latent_mask_seed_${SEED} \
--key 1_latent_mask \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--random_seed ${SEED} & \
CUDA_VISIBLE_DEVICES=2 python -m context_compression.train \
--max_lr 1e-3 --total_batch_size 32768 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group latent_mask_coord_check \
--mup_enable_coord_check_logging --no_decay_lr --max_steps 10 --no_use_compile --no_upload_to_hf \
--log_dir logs/latent_mask_coord_check/1_latent_mask_init_to_one_seed_${SEED} \
--key 1_latent_mask_init_to_one \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity \
--random_seed ${SEED} & \
CUDA_VISIBLE_DEVICES=3 python -m context_compression.train \
--max_lr 1e-3 --total_batch_size 32768 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group latent_mask_coord_check \
--mup_enable_coord_check_logging --no_decay_lr --max_steps 10 --no_use_compile --no_upload_to_hf \
--log_dir logs/latent_mask_coord_check/2_sliced_masks_seed_${SEED} \
--key 2_sliced_masks \
--selection_head_linear_combo n_sliced_masks \
--n_sliced_masks 2 \
--random_seed ${SEED}
```

#### Testing fixes to 1-latent-mask


Baseline, low lr (I'll use 10e-4), init to one, init to one + low-lr param group (with 3 diff lrs), layernorm

Baseline:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/baseline_lr_30e-4_seed_1339 \
--key baseline_lr_30e-4 \
--random_seed 1339 \
--selection_head_linear_combo none
```

1 latent mask:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_seed_1339 \
--key 1_latent_mask_lr_30e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1
```

Low lr:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_10e-4_seed_1339 \
--key 1_latent_mask_lr_10e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1
```

Init to one:


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_ones_lr_30e-4_seed_1339 \
--key 1_latent_mask_ones_lr_30e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

Init to one + low-lr param group (lr=15e-4, lr=10e-4, lr=5e-4):


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 15e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_ones_lr_15e-4_seed_1339 \
--key 1_latent_mask_ones_lr_15e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 10e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_ones_lr_10e-4_seed_1339 \
--key 1_latent_mask_ones_lr_10e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 5e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_ones_lr_5e-4_seed_1339 \
--key 1_latent_mask_ones_lr_5e-4 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

Layernorm:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_ln_seed_1339 \
--key 1_latent_mask_lr_30e-4_ln \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--S_layernorm
```

Results:

- Layernorm lr=30e-4 was good! It did a little worse than 15e-4, which was a little worse than the baseline, BUT it totally solved the divergence problem.
  - Note: it *immediately* took longer to converge than the other guys.
  - And 15e-4 *immediately* took longer to converge than the baseline.
  - I think in general, layernorm might just be a bandaid. But a super useful bandaid!
- Lower lr def helps, even without one-init. Maybe I just need a lower lr for this param group.
- One-init doesn't rly seem to fix things?
  - It still diverges. BUT. before then, one-init with lr=30e-4 has a v similar loss curve to the baseline.
  - So we still need to solve the divergence problem. Hopefully w/ param groups etc.
  - BUT seems like a good init, and not changing global lr, is important!

- Let's try a special param group (or just dividing by a constant, tbh)

#### Fixes to 1-latent-mask, pt. 2

Divide by constant (1/2, 1/4, 1/8):

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_scale_0.5_seed_1339 \
--key 1_latent_mask_lr_30e-4_scale_0.5 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--latent_mask_scale 0.5 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_scale_0.25_seed_1339 \
--key 1_latent_mask_lr_30e-4_scale_0.25 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--latent_mask_scale 0.25 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_scale_0.125_seed_1339 \
--key 1_latent_mask_lr_30e-4_scale_0.125 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--latent_mask_scale 0.125 \
--init_latent_masks_to_identity
```

Multiply by sigmoid:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_sigmoid_seed_1339 \
--key 1_latent_mask_lr_30e-4_sigmoid \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--latent_mask_sigmoid
```

Multiply by sigmoid / 2:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_half_sigmoid_seed_1339 \
--key 1_latent_mask_lr_30e-4_half_sigmoid \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--latent_mask_sigmoid
```

Param group custom lr scale (1/10, 1/100, 1/1000), initted to one. We're looking for at least one of these to repro the baseline.

It really really should!

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_lr_scale_0.1_seed_1339 \
--key 1_latent_mask_lr_30e-4_lr_scale_0.1 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--selection_head_linear_combo_scale 0.1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_lr_scale_0.01_seed_1339 \
--key 1_latent_mask_lr_30e-4_lr_scale_0.01 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--selection_head_linear_combo_scale 0.01 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_lr_scale_0.001_seed_1339 \
--key 1_latent_mask_lr_30e-4_lr_scale_0.001 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--selection_head_linear_combo_scale 0.001 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_lr_scale_0_seed_1339 \
--key 1_latent_mask_lr_30e-4_lr_scale_0 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--selection_head_linear_combo_scale 0 \
--init_latent_masks_to_identity \
--disable_selection_head_linear_combo_bias
```


Baseline, but throwing away the first head (should be the same as lr=0, bias=none, init=1):

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/baseline_lr_30e-4_seed_1339_no_head \
--key baseline_lr_30e-4_no_head \
--random_seed 1339 \
--selection_head_linear_combo none_with_no_head
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_lr_scale_0_seed_1339_retry \
--key 1_latent_mask_lr_30e-4_lr_scale_0_retry \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--selection_head_linear_combo_scale 0 \
--init_latent_masks_to_identity \
--disable_selection_head_linear_combo_bias
```

Baseline:
```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/baseline_lr_30e-4_seed_1339_retry \
--key baseline_lr_30e-4_retry \
--random_seed 1339 \
--selection_head_linear_combo none
```

lr=30e-4:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_seed_1339_retry \
--key 1_latent_mask_lr_30e-4_retry \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

With no compile, b/c it looks like compile messes this up:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_lr_scale_0_seed_1339_no_compile \
--key 1_latent_mask_lr_30e-4_lr_scale_0_no_compile \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--selection_head_linear_combo_scale 0 \
--init_latent_masks_to_identity \
--disable_selection_head_linear_combo_bias \
--no_use_compile \
--assert_latent_matches_no_head
```

lr=30e-4, no compile:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_seed_1339_no_compile \
--key 1_latent_mask_lr_30e-4_no_compile \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity \
--no_use_compile
```

Baseline with no compile:
```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/baseline_lr_30e-4_seed_1339_no_compile \
--key baseline_lr_30e-4_no_compile \
--random_seed 1339 \
--selection_head_linear_combo none \
--no_use_compile
```

lr=30e-4, n_latent_masks=2:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_n_latent_masks_2_seed_1339 \
--key 1_latent_mask_lr_30e-4_n_latent_masks_2 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity \
--no_use_compile
```

lr=30e-4, n_sliced_masks=2:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_n_sliced_masks_2_seed_1339 \
--key 1_latent_mask_lr_30e-4_n_sliced_masks_2 \
--random_seed 1339 \
--selection_head_linear_combo n_sliced_masks \
--n_sliced_masks 2 \
--init_latent_masks_to_identity
```

n_latent_masks=1, lr=30e-4, seed=1340, compile:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_no_compile_seed_1340 \
--key 1_latent_mask_lr_30e-4_no_compile \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity \
--no_use_compile
```

n_latent_masks=2, lr={25e-4, 30e-4, 35e-4}, seed=1340:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 25e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_25e-4_n_latent_masks_2_no_compile_seed_1340 \
--key 1_latent_mask_lr_25e-4_n_latent_masks_2_no_compile \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_30e-4_n_latent_masks_2_no_compile_seed_1340 \
--key 1_latent_mask_lr_30e-4_n_latent_masks_2_no_compile \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 35e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_35e-4_n_latent_masks_2_no_compile_seed_1340 \
--key 1_latent_mask_lr_35e-4_n_latent_masks_2_no_compile \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity \
--no_use_compile
```

#### Fixing torch.compile numeric instability bug

```vast:finished
cd /workspace/context-compression && git fetch && git checkout andrew/high-precision-latent-masks && HIGH_PRECISION_LATENT_MASKS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_compile_bug \
--log_dir logs/fix_compile_bug/autocast_float32 \
--key autocast_float32 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git fetch && git checkout andrew/disable-dynamo && DISABLE_DYNAMO=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_compile_bug \
--log_dir logs/fix_compile_bug/disable_dynamo \
--key disable_dynamo \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git fetch && git checkout andrew/relu-graph-break && RELU_GRAPH_BREAK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_compile_bug \
--log_dir logs/fix_compile_bug/relu_graph_break \
--key relu_graph_break \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

#### Improvements to 1-latent-mask, pt. 3

Seems like compiling is ok now. Seems like lr=25e-4 helped n_latent_masks=2. Does it help n_latent_masks=1 also?

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 25e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_25e-4_n_latent_masks_1_relu_seed_1340 \
--key 1_latent_mask_lr_25e-4_n_latent_masks_1_relu \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 25e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_25e-4_n_latent_masks_2_relu_seed_1340 \
--key 1_latent_mask_lr_25e-4_n_latent_masks_2_relu \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 35e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_35e-4_n_latent_masks_1_relu_seed_1340 \
--key 1_latent_mask_lr_35e-4_n_latent_masks_1_relu \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 35e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_lr_35e-4_n_latent_masks_2_relu_seed_1340 \
--key 1_latent_mask_lr_35e-4_n_latent_masks_2_relu \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity
```

Result: 2 is better than 1 is better than 0, seems like! See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=0zjsohr4exwe).

Let's try to transfer this to the big model!! Fingers crossed.

### Running new latent masks vs. baseline

Baseline:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group trying_new_latent_masks \
  --log_dir trying_new_latent_masks/baseline \
  --key baseline \
  --selection_head_linear_combo none \
  --n_heads 12 \
  --batch_size 4 \
  --random_seed 1337
```


No head:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group trying_new_latent_masks \
  --log_dir trying_new_latent_masks/baseline \
  --key baseline_no_head \
  --selection_head_linear_combo none_with_no_head \
  --n_heads 12 \
  --batch_size 4 \
  --random_seed 1337
```

1 latent mask:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_1_latent_vector \
  --key one_mask_per_head_1_latent_vector \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1 \
  --batch_size 4 \
  --random_seed 1337
```


2 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_2_latent_vectors \
  --key one_mask_per_head_2_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 2 \
  --batch_size 4 \
  --random_seed 1337
```


4 latent masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_4_latent_vectors \
  --key one_mask_per_head_4_latent_vectors \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 4 \
  --batch_size 4 \
  --random_seed 1337
```

1 latent mask, initted to identity:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_1_latent_vector_identity_seed_1337 \
  --key one_mask_per_head_1_latent_vector_identity \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity
```

1 latent mask, initted to identity, with lr scaled to zero, no bias:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_1_latent_vector_degenerate_seed_1337 \
  --key one_mask_per_head_1_latent_vector_degenerate \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias
```

1 latent mask, initted to identity, with lr scaled to zero, no bias, no compile:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_1_latent_vector_degenerate_no_compile_seed_1337 \
  --key one_mask_per_head_1_latent_vector_degenerate_no_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --no_use_compile
```

2 latent masks, initted to identity:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_2_latent_vectors_identity_seed_1337 \
  --key one_mask_per_head_2_latent_vectors_identity \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 2 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity
```

4 latent masks, initted to identity:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_4_latent_vectors_identity_seed_1337 \
  --key one_mask_per_head_4_latent_vectors_identity \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 4 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity
```

8 latent masks, initted to identity:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_8_latent_vectors_identity_seed_1337 \
  --key one_mask_per_head_8_latent_vectors_identity \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 8 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity
```

Results (with "<" means "worse than"): baseline < n latent masks < degenerate (compiled and uncompiled) < 2, latent masks w/ identity init < no head < 1 latent mask w/ identity init.

Suggests that identity init is important.

Maybe I should do 1/n init for 2,4,8 latent masks.

Maybe I should run another seed of no head vs. degenerate. If degenerate is asymptotically worse, then I need higher precision or smth.

Also, I should run fp32-autocast version of 1 latent mask.

Then all of these will yield results in an hour or two. And hopefully I can learn more on the small model.

### Investigating discrepancies on the big model

No head, seed 1338:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group trying_new_latent_masks \
  --log_dir trying_new_latent_masks/baseline_no_head_seed_1338 \
  --key baseline_no_head \
  --selection_head_linear_combo none_with_no_head \
  --n_heads 12 \
  --batch_size 4 \
  --random_seed 1338
```


Degenerate with torch compile, seed 1338:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_1_latent_vector_degenerate_seed_1338 \
  --key one_mask_per_head_1_latent_vector_degenerate \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1 \
  --batch_size 4 \
  --random_seed 1338 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias
```


High-precision latent masks for degenerate, seed 1337:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_1_latent_vector_degenerate_float32_seed_1337 \
  --key one_mask_per_head_1_latent_vector_degenerate_float32 \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --latent_mask_precision float32
```

2 latent masks, 1/n init, seed 1337:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_2_latent_vectors_inverse_seed_1337 \
  --key one_mask_per_head_2_latent_vectors_inverse \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 2 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_inverse
```


Results:

- float32 degenerate (with compile!!!) matches the baseline no head baseline!! Amazing news! Let's use that from now on.
- inverse makes 2 latent masks worse. Honestly no idea why. I feel like I've gotta learn more! Maybe I should be dividing the result by n_latent_masks rather than changing the init from identity?

OK so what to do.
- Rerun 2 latent heads identity with float32.
- Rerun 1 latent head identity with float32.
- Run 2 latent heads identity with float32, divided by n_latent_masks.

2 latent heads, initted to identity, float32:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_2_latent_vectors_identity_float32_seed_1337 \
  --key one_mask_per_head_2_latent_vectors_identity_float32 \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 2 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32
```

1 latent head, initted to identity, float32:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_1_latent_vector_identity_float32_seed_1337 \
  --key one_mask_per_head_1_latent_vector_identity_float32 \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 1 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32
```

2 latent heads, initted to identity, float32, divided by n_latent_masks:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --group allowing_more_selection_patterns \
  --log_dir allowing_more_selection_patterns/one_mask_per_head_2_latent_vectors_identity_float32_seed_1337_div \
  --key one_mask_per_head_2_latent_vectors_identity_float32 \
  --selection_head_linear_combo n_latent_masks \
  --n_heads 12 \
  --n_latent_masks 2 \
  --batch_size 4 \
  --random_seed 1337 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --latent_mask_runtime_multiplier 0.5
```

Results (see [wandb](https://wandb.ai/sesamestrong/context_compression?nw=q6xs31xlsho)):
- float32 identity beats non-float32 identity.
- Divided by n_latent_masks is still worse (even w/ float32)! Mysterious. IDT I can do many more iters on the big model - it's soo slow. Hrrm, maybe I can use some H200s? Let's leave that for later.
- Let's check the little model - does div perform worse than identity for n_heads=2?
- Do we get the same float32 perf boost on the small model?
- If none of these hold up - maybe try scaling up to head_dim=64 and 8 4090s? Or maybe head_dim=32?

### Investigating big-model discrepancies on the little model

1 latent head, initted to identity:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_seed_1339 \
--key 1_latent_mask \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

1 latent head, initted to identity, float32:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_float32_seed_1339 \
--key 1_latent_mask_float32 \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity \
--latent_mask_precision float32
```


2 latent heads, initted to identity, float32:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/2_latent_mask_seed_1339 \
--key 2_latent_mask \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity \
--latent_mask_precision float32
```


2 latent heads, initted to identity, float32, divided by n_latent_masks:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/2_latent_mask_seed_1339_div \
--key 2_latent_mask_div \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity \
--latent_mask_precision float32 \
--latent_mask_runtime_multiplier 0.5
```

2 latent heads, initted to identity, float32, with inverse init for latent heads:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/2_latent_mask_seed_1339_inverse \
--key 2_latent_mask_inverse \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_inverse \
--latent_mask_precision float32
```

Results:

- float32 reduced final perf (!), maybe we need to use more seeds!! Or maybe just not use it for the big model.

- dividing and a lower lr scale both seem to hurt n_latent_heads=2

- n_latent_heads=2 is still better than n_latent_heads=1

Hrrm, what can I interpolate with? Maybe we should try with one full head per latent mask? That should also be better on the little model.

We can also try scaling up to head_dim=64 and 8 4090s. But that will halve our throughput.

OK let's do all 3, all on the little model.

Re-run the float32 vs. bfloat16 comparison with two more seeds.

I expect two heads for two latent masks -> the best performance yet.

I expect two latent masks will still beat one latent mask on a 64-head-dim model.

And while they're running, maybe I should rly look into scaled-fp8 matmuls. See if there's an easy copyable in the speedrun repo.

Also, I found a bug in the no_heads code - it allocates 1 head too few, making the model seem dumber than it rly is. Yet it beat the baseline in my previous runs. What gives? Is this mup shenanigans? Should I be allocating a separate head_proj for the selection head?

#### Checking if float32 beats bfloat16 on the little model

1 latent head, initted to identity, seed={1340,1341}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_seed_1340 \
--key 1_latent_mask \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_seed_1341 \
--key 1_latent_mask \
--random_seed 1341 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

1 latent head, initted to identity, float32, seed={1340,1341}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_float32_seed_1340 \
--key 1_latent_mask_float32 \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity \
--latent_mask_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_float32_seed_1341 \
--key 1_latent_mask_float32 \
--random_seed 1341 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity \
--latent_mask_precision float32
```

Result: They're basically identical. See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=fopxbjobs5r).

Meta-result: what am I doing here? I should be using way more seeds!

#### No-head rerun, with 1 and 2 latent masks

11-effective-head model (should match old no-head results):

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 11 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/no_head_1_effective_11_seed_1339 \
--key no_head_1_effective_11 \
--random_seed 1339 \
--selection_head_linear_combo none_with_no_head \
--one_head_per_latent_mask \
--n_latent_masks 1
```

1 head added:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/no_head_1_seed_1339 \
--key no_head_1 \
--random_seed 1339 \
--selection_head_linear_combo none_with_no_head \
--one_head_per_latent_mask \
--n_latent_masks 1
```

2 heads added:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/no_head_2_seed_1339 \
--key no_head_2 \
--random_seed 1339 \
--selection_head_linear_combo none_with_no_head \
--one_head_per_latent_mask \
--n_latent_masks 2
```

Partial result: 1 head added is ~identical to 2 heads. IG the lrs would be super similar in any case, so ig that makes sense.

My "11-effective-head" model is so much worse than both of them. I think it's b/c it also shrinks the n_embds. Let's fix that now?

11-effective-head model, n_embd=264:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 11 --head_dim 22 --n_embd 264 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/no_head_1_effective_11_n_embd_264_seed_1339 \
--key no_head_1_effective_11_n_embd_264 \
--random_seed 1339 \
--selection_head_linear_combo none_with_no_head \
--one_head_per_latent_mask \
--n_latent_masks 1
```

Result: 1 vs. 2 masks doesn't matter. 

See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=ifwmq2wthz).

Baseline model:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/og_baseline_none_1339 \
--key og_baseline_selective \
--random_seed 1339 \
--selection_head_linear_combo none
```

Baseline model with 13 heads:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 13 --head_dim 22 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/og_baseline_none_13_heads_1339 \
--key og_baseline_selective_13_heads \
--random_seed 1339 \
--selection_head_linear_combo none
```

Baseline model with 13 heads and 264 embd dim:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 13 --head_dim 22 --n_embd 264 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/og_baseline_none_13_heads_264_embd_1339 \
--key og_baseline_selective_13_heads_264_embd \
--random_seed 1339 \
--selection_head_linear_combo none
```



#### Two heads for two latent masks

Compare one head for two latent masks vs. two heads for two latent masks vs. two heads for two sliced masks vs. one head for two sliced masks.

Let's use no float32 for any of them.

Two heads for two latent masks:
```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group two_heads_comparison \
--log_dir logs/two_heads_comparison/two_heads_for_two_latent_masks_seed_1339 \
--key two_heads_for_two_latent_masks \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity \
--one_head_per_latent_mask
```

One head for two latent masks:
```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group two_heads_comparison \
--log_dir logs/two_heads_comparison/one_head_for_two_latent_masks_seed_1339 \
--key one_head_for_two_latent_masks \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity
```

Two heads for two sliced masks:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group two_heads_comparison \
--log_dir logs/two_heads_comparison/two_heads_for_two_sliced_masks_seed_1339 \
--key two_heads_for_two_sliced_masks \
--random_seed 1339 \
--selection_head_linear_combo n_sliced_masks \
--n_sliced_masks 2 \
--one_head_per_latent_mask
```

Two heads (should match the two-heads-for-two-sliced-masks baseline):

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group two_heads_comparison \
--log_dir logs/two_heads_comparison/two_masks_seed_1339 \
--key two_masks \
--random_seed 1339 \
--selection_head_linear_combo two_masks
```

Two heads with 11 heads:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 11 --head_dim 22 --n_embd 264 \
--group two_heads_comparison \
--log_dir logs/two_heads_comparison/two_masks_11_heads_seed_1339 \
--key two_masks_11_heads \
--random_seed 1339 \
--selection_head_linear_combo two_masks
```

Two heads with 10 heads:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 10 --head_dim 22 --n_embd 264 \
--group two_heads_comparison \
--log_dir logs/two_heads_comparison/two_masks_10_heads_seed_1339 \
--key two_masks_10_heads \
--random_seed 1339 \
--selection_head_linear_combo two_masks
```


One head for two sliced masks:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group two_heads_comparison \
--log_dir logs/two_heads_comparison/one_head_for_two_sliced_masks_seed_1339 \
--key one_head_for_two_sliced_masks \
--random_seed 1339 \
--selection_head_linear_combo n_sliced_masks \
--n_sliced_masks 2
```

### Ablation on head-dim 22 model

#### Investigating a suspicious result from previous experiment

(see seed=1339, after March 7th, in groups `fix_1_latent_mask` and `two_heads_comparison`)

Generally, there's a big gap between two clusters of runs (see a big gap [here](https://wandb.ai/sesamestrong/context_compression?nw=zbwxchgxw08)). The runs above the gap mostly include baselines and a few no_head variants. The runs below contain some no_head variants and a bunch of latent mask/two mask/sliced mask experiments.

BUT there is a big mysterious gap between `baseline_lr_30e-4_no_compile` and all the other baselines - it's much better, and falls into the better group.

This makes me worry that most of my progress is from numeric stability on my new codepath and not a better learning representation.
As part of our journey towards understanding the gap (and what explains the jumps), let's first try to repro the no-compile vs. yes-compile gap on two seeds.

Baseline with no compile, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group baseline_compile_comparison \
--log_dir logs/baseline_compile_comparison/baseline_lr_30e-4_seed_1339_no_compile \
--key baseline_lr_30e-4_no_compile \
--random_seed 1339 \
--selection_head_linear_combo none \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group baseline_compile_comparison \
--log_dir logs/baseline_compile_comparison/baseline_lr_30e-4_seed_1340_no_compile \
--key baseline_lr_30e-4_no_compile \
--random_seed 1340 \
--selection_head_linear_combo none \
--no_use_compile
```

Baseline with yes compile, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group baseline_compile_comparison \
--log_dir logs/baseline_compile_comparison/baseline_lr_30e-4_seed_1339_retry \
--key baseline_lr_30e-4_retry \
--random_seed 1339 \
--selection_head_linear_combo none
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 12 --head_dim 22 \
--group baseline_compile_comparison \
--log_dir logs/baseline_compile_comparison/baseline_lr_30e-4_seed_1340_retry \
--key baseline_lr_30e-4_retry \
--random_seed 1340 \
--selection_head_linear_combo none
```

Left off here. Partial results: looks like no-compile really does span the gap.

#### Incrementing the # of selective heads

Heads=12, seed={1339,1340}: (see previous section)

Heads=13, compile=true, seed={1339,1340}:

NOTE: disregard these two runs! I forgot to add the no_compile flag.

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 13 --head_dim 22 --n_embd 264 \
--group heads_bump_comparison \
--log_dir logs/heads_bump_comparison/13_heads_seed_1339 \
--key 13_heads \
--random_seed 1339 \
--selection_head_linear_combo none
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 13 --head_dim 22 --n_embd 264 \
--group heads_bump_comparison \
--log_dir logs/heads_bump_comparison/13_heads_seed_1340 \
--key 13_heads \
--random_seed 1340 \
--selection_head_linear_combo none
```

Heads=13, compile=false, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 13 --head_dim 22 --n_embd 264 \
--group heads_bump_comparison \
--log_dir logs/heads_bump_comparison/13_heads_no_compile_seed_1339 \
--key 13_heads_no_compile \
--random_seed 1339 \
--selection_head_linear_combo none \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 13 --head_dim 22 --n_embd 264 \
--group heads_bump_comparison \
--log_dir logs/heads_bump_comparison/13_heads_no_compile_seed_1340 \
--key 13_heads_no_compile \
--random_seed 1340 \
--selection_head_linear_combo none \
--no_use_compile
```

Partial results: looks like 13 heads are better than 12 heads.

#### Switching from baseline no-compile to "no heads" mode (i.e. throwing away the value head)

Selection kind = default: (see previous section)

Selection kind = no_heads, compile=false, seed={1339,1340}:
(note: disregard! I accidentally used none instead of no_heads)

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
--group heads_bump_comparison \
--log_dir logs/heads_bump_comparison/no_heads_seed_1339 \
--key no_heads \
--random_seed 1339 \
--selection_head_linear_combo none \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
--group heads_bump_comparison \
--log_dir logs/heads_bump_comparison/no_heads_seed_1340 \
--key no_heads \
--random_seed 1340 \
--selection_head_linear_combo none \
--no_use_compile
```

(fixed, actually using kind=no_head)

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
--group no_heads_comparison \
--log_dir logs/no_heads_comparison/no_heads_fixed_seed_1339 \
--key no_heads_fixed \
--random_seed 1339 \
--selection_head_linear_combo none_with_no_head \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
--group no_heads_comparison \
--log_dir logs/no_heads_comparison/no_heads_fixed_seed_1340 \
--key no_heads_fixed \
--random_seed 1340 \
--selection_head_linear_combo none_with_no_head \
--no_use_compile
```

#### Switching from "no heads" mode to "no latent masks degenerate" mode

No heads: (see previous section)

Latent masks degenerate (with torch.compile), with lr={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group latent_masks_degenerate \
  --log_dir latent_masks_degenerate/degen_compile_seed_1339 \
  --key degen_compile_seed_1339 \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --latent_mask_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group latent_masks_degenerate \
  --log_dir latent_masks_degenerate/degen_compile_seed_1340 \
  --key degen_compile_seed_1340 \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --latent_mask_precision float32
```

Latent masks degenerate (without torch.compile), with lr={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group latent_masks_degenerate \
  --log_dir latent_masks_degenerate/degen_no_compile_seed_1339 \
  --key degen_no_compile_seed_1339 \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --latent_mask_precision float32 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group latent_masks_degenerate \
  --log_dir latent_masks_degenerate/degen_no_compile_seed_1340 \
  --key degen_no_compile_seed_1340 \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --latent_mask_precision float32 \
  --no_use_compile
```

#### Switching from degenerate to learnable latent masks

Degenerate (with no torch.compile), lr={1339,1340}: (see previous section)

Degenerate (without torch.compile), lr={1339,1340}: (see previous section)

Learnable (with torch.compile), lr={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group latent_masks_learnable \
  --log_dir latent_masks_learnable/learnable_compile_seed_1339 \
  --key learnable_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group latent_masks_learnable \
  --log_dir latent_masks_learnable/learnable_compile_seed_1340 \
  --key learnable_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32
```

Learnable (without torch.compile), lr={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group latent_masks_learnable \
  --log_dir latent_masks_learnable/learnable_no_compile_seed_1339 \
  --key learnable_no_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group latent_masks_learnable \
  --log_dir latent_masks_learnable/learnable_no_compile_seed_1340 \
  --key learnable_no_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --no_use_compile
```

#### Switching from 1 latent mask to 2 latent masks

1 latent mask: (see previous section)

2 latent masks, lr={1339,1340}: 

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group two_latent_masks \
  --log_dir two_latent_masks/two_latent_masks_seed_1339 \
  --key two_latent_masks \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group two_latent_masks \
  --log_dir two_latent_masks/two_latent_masks_seed_1340 \
  --key two_latent_masks \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --no_use_compile
```

### One vs. two latent masks on a 64-head-dim model (8-GPU run)

#### First set of runs: no float32, with compile

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --n_heads 12 --head_dim 64 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_seed_1339 \
--key 1_latent_mask \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --n_heads 12 --head_dim 64 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/2_latent_mask_seed_1339 \
--key 2_latent_mask \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity
```


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --n_heads 12 --head_dim 64 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/1_latent_mask_seed_1340 \
--key 1_latent_mask \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --n_heads 12 --head_dim 64 \
--group fix_1_latent_mask \
--log_dir logs/fix_1_latent_mask/2_latent_mask_seed_1340 \
--key 2_latent_mask \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity
```

Partial result: 2 latent masks are better than 1. See [wandb](https://wandb.ai/sesamestrong/context_compression/panel/zrk0hi0fm?nw=jlbra05nth).

But I notice I didn't use float32 precision or turn off torch.compile. Let's do another set of runs with no compile and float32 precision.

#### Second set of runs: float32, no compile

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --n_heads 12 --head_dim 64 \
--group 64_head_dim_two_latent_masks \
--log_dir logs/64_head_dim_two_latent_masks/1_latent_mask_seed_1339 \
--key 1_latent_mask \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity \
--latent_mask_precision float32 \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --n_heads 12 --head_dim 64 \
--group 64_head_dim_two_latent_masks \
--log_dir logs/64_head_dim_two_latent_masks/2_latent_mask_seed_1339 \
--key 2_latent_mask \
--random_seed 1339 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity \
--latent_mask_precision float32 \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --n_heads 12 --head_dim 64 \
--group 64_head_dim_two_latent_masks \
--log_dir logs/64_head_dim_two_latent_masks/1_latent_mask_seed_1340 \
--key 1_latent_mask \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 1 \
--init_latent_masks_to_identity \
--latent_mask_precision float32 \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --n_heads 12 --head_dim 64 \
--group 64_head_dim_two_latent_masks \
--log_dir logs/64_head_dim_two_latent_masks/2_latent_mask_seed_1340 \
--key 2_latent_mask \
--random_seed 1340 \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--init_latent_masks_to_identity \
--latent_mask_precision float32 \
--no_use_compile
```

Results: Delta this time is about 0.007. That's bigger. Not clear if this generalizes at all, it's scary that numerics makes such a big difference. [wandb link](https://wandb.ai/sesamestrong/context_compression/panel/ieugzanro?nw=mlefcj092pp).

### Investigating two-masks vs. two-latent-masks

In the many-1339-seed group, I noticed that 12 total heads with two dedicated to masking was better than 13 total heads with one dedicated to masking.

I also noticed that 13 total heads with two dedicated to masking was worse than both of those, which is surprising. Maybe it's smth with the 2 masks not wrapping nicely around the 11 remaining heads?

The ablation tells me the benefit from adding one extra head is about 0.008 - not big, but not tiny. I attribute this up to the bitter lesson.

For reference, the jump from frozen -> learnable latent masks is about 0.002. This is surprisingly small, but I guess I trust it. I have no good intuition for it though - wouldn't this be nearly as good as having two latent masks? Do attn maps frozen -> learned look more similar than attn maps 1 head -> 2 heads?

Actually, let's first look at a bunch of attn maps. Let's put them in a folder and analyze them before running any more experiments.

Also, let's remember to check how 2-masks performed vs. baseline selective attention on the big model. IIRC it made ~no difference, despite adding compute.

Oh, very confusing - "two heads for two sliced masks" scores 0.005 worse than "two masks" in the 1339-mask-scaleup graph. I think this is a super tiny diff, actually.

The two models are identical, i.e. for the same weights, they'll give the same prob distro. The only diff is numerics, I think. So I'm guessing improving numerics (by removing the .sum(), maybe?) will close the gap by a ton.

Let's try a re-run of the two-head comparisons, with float32 and yes/no compile, for two seeds. The gap between sliced and two-masks should drop to a tiny amount. Arguably, 0.005 might already be a tiny difference.

I want to try a variable-scale two-mask run also - I think it'll be better than two latent masks with two heads behind them. Not sure abt this though, since learnable weights didn't help the one-latent-head case much. Honestly, very very confusing.

#### Re-running two-head vs. two-sliced-head comparison with high-precision and yes/no-compile

Two masks, float32, no compile, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group two_heads_sliced_vs_unsliced \
--log_dir logs/two_heads_sliced_vs_unsliced/unsliced_no_compile_seed_1339 \
--key unsliced_no_compile \
--random_seed 1339 \
--selection_head_linear_combo two_masks \
--latent_mask_precision float32 \
--no_use_compile
```

Two heads for two sliced masks, float32, no compile, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group two_heads_sliced_vs_unsliced \
--log_dir logs/two_heads_sliced_vs_unsliced/sliced_no_compile_seed_1339 \
--key sliced_no_compile \
--random_seed 1339 \
--selection_head_linear_combo n_sliced_masks \
--n_sliced_masks 2 \
--one_head_per_latent_mask \
--latent_mask_precision float32 \
--no_use_compile
```


Two masks, float32, yes compile, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group two_heads_sliced_vs_unsliced \
--log_dir logs/two_heads_sliced_vs_unsliced/unsliced_compile_seed_1339 \
--key unsliced_compile \
--random_seed 1339 \
--selection_head_linear_combo two_masks \
--latent_mask_precision float32
```


Two heads for two sliced masks, float32, yes compile, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 \
--group two_heads_sliced_vs_unsliced \
--log_dir logs/two_heads_sliced_vs_unsliced/sliced_compile_seed_1339 \
--key sliced_compile \
--random_seed 1339 \
--selection_head_linear_combo n_sliced_masks \
--n_sliced_masks 2 \
--one_head_per_latent_mask \
--latent_mask_precision float32
```

Result: seems like they basically match! The hypothesis was basically correct - the two models do work the same and train the same, basically.
Maybe compile is a little better than no compile, but I'm really not sure either way. See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=oi6tx1xo8v).

For reference, here's the graph where they initially mismatched: [wandb](https://wandb.ai/sesamestrong/context_compression?nw=2sidl700lb6).

I don't quite remember if torch.compile helps or hurts the final n-latent-mask case... Hrrm, for selective, I think it might hurt? Ah yes, remember step #1 of the tiny-model ablation. I wonder what the comparison is for the 64-dim, 1024-seq-len case.

I also wonder how we can stabilize training so it's always safe to use no-compile with selectivity.

Ideally, I'd like to try my next experiments with float32 and compile, which seems like a reasonable compromise. Well, do I think there's actually a future here? Is it worth optimizing this multi-mask stuff?

TODO think more about this on typehere.

### Ablation on a head_dim-64, seq_len=1024 model

Results: [wandb report](https://wandb.ai/sesamestrong/context_compression/reports/Same-ablation-on-64-1024--VmlldzoxMTc4MzI1OA/).

#### Yes-compile vs. no-compile

Baseline with no compile, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 8 --mup --n_heads 12 --head_dim 64 \
--group 64_baseline_compile_comparison \
--log_dir logs/64_baseline_compile_comparison/baseline_lr_30e-4_seed_1339_no_compile \
--key baseline_lr_30e-4_no_compile \
--random_seed 1339 \
--selection_head_linear_combo none \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 8 --mup --n_heads 12 --head_dim 64 \
--group 64_baseline_compile_comparison \
--log_dir logs/64_baseline_compile_comparison/baseline_lr_30e-4_seed_1340_no_compile \
--key baseline_lr_30e-4_no_compile \
--random_seed 1340 \
--selection_head_linear_combo none \
--no_use_compile
```

Baseline with yes compile, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 8 --mup --n_heads 12 --head_dim 64 \
--group 64_baseline_compile_comparison \
--log_dir logs/64_baseline_compile_comparison/baseline_lr_30e-4_seed_1339_retry \
--key baseline_lr_30e-4_retry \
--random_seed 1339 \
--selection_head_linear_combo none
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 8 --mup --n_heads 12 --head_dim 64 \
--group 64_baseline_compile_comparison \
--log_dir logs/64_baseline_compile_comparison/baseline_lr_30e-4_seed_1340_retry \
--key baseline_lr_30e-4_retry \
--random_seed 1340 \
--selection_head_linear_combo none
```

Result: delta is about -0.005 - nocompile makes it worse! See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=1d9rhfrhhhg).

#### Incrementing the # of selective heads

Heads=12, seed={1339,1340}: (see previous section)

Heads=13, compile=false, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 8 --mup --n_heads 13 --head_dim 64 --n_embd 768 \
--group 64_heads_bump_comparison \
--log_dir logs/64_heads_bump_comparison/13_heads_no_compile_seed_1339 \
--key 13_heads_no_compile \
--random_seed 1339 \
--selection_head_linear_combo none \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 8 --mup --n_heads 13 --head_dim 64 --n_embd 768 \
--group 64_heads_bump_comparison \
--log_dir logs/64_heads_bump_comparison/13_heads_no_compile_seed_1340 \
--key 13_heads_no_compile \
--random_seed 1340 \
--selection_head_linear_combo none \
--no_use_compile
```

Delta: seems like 0.005. See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=1d9rhfrhhhg).

#### Switching from baseline no-compile to "no heads" mode (i.e. throwing away the value head)

Selection kind = default: (see previous section)

Selection kind = no_head, compile=false, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 8 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
--group 64_no_heads_comparison \
--log_dir logs/64_no_heads_comparison/no_heads_fixed_seed_1339 \
--key no_heads_fixed \
--random_seed 1339 \
--selection_head_linear_combo none_with_no_head \
--no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
--max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 8 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
--group 64_no_heads_comparison \
--log_dir logs/64_no_heads_comparison/no_heads_fixed_seed_1340 \
--key no_heads_fixed \
--random_seed 1340 \
--selection_head_linear_combo none_with_no_head \
--no_use_compile
```

Result: delta of about 0.000. See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=68saecckrhw).

#### Switching from "no heads" mode to "no latent masks degenerate" mode

No heads: (see previous section)

Latent masks degenerate (with torch.compile), with lr={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_latent_masks_degenerate \
  --log_dir logs/64_latent_masks_degenerate/degen_compile_seed_1339 \
  --key degen_compile_seed_1339 \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --latent_mask_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_latent_masks_degenerate \
  --log_dir logs/64_latent_masks_degenerate/degen_compile_seed_1340 \
  --key degen_compile_seed_1340 \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --latent_mask_precision float32
```

Latent masks degenerate (without torch.compile), with lr={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_latent_masks_degenerate \
  --log_dir logs/64_latent_masks_degenerate/degen_no_compile_seed_1339 \
  --key degen_no_compile_seed_1339 \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --latent_mask_precision float32 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_latent_masks_degenerate \
  --log_dir logs/64_latent_masks_degenerate/degen_no_compile_seed_1340 \
  --key degen_no_compile_seed_1340 \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --selection_head_linear_combo_scale 0 \
  --disable_selection_head_linear_combo_bias \
  --latent_mask_precision float32 \
  --no_use_compile
```

Result: huge delta of 0.015. Very strange, actually. I would expect 0, so there's some super weird numerical instability thing going on, maybe? Or did I mess up the config? See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=aaqmhq8dj9u).

#### Switching from degenerate to learnable latent masks

Degenerate (with no torch.compile), lr={1339,1340}: (see previous section)

Degenerate (without torch.compile), lr={1339,1340}: (see previous section)

Learnable (with torch.compile), lr={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_latent_masks_learnable \
  --log_dir logs/64_latent_masks_learnable/learnable_compile_seed_1339 \
  --key learnable_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_latent_masks_learnable \
  --log_dir logs/64_latent_masks_learnable/learnable_compile_seed_1340 \
  --key learnable_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32
```

Learnable (without torch.compile), lr={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_latent_masks_learnable \
  --log_dir logs/64_latent_masks_learnable/learnable_no_compile_seed_1339 \
  --key learnable_no_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_latent_masks_learnable \
  --log_dir logs/64_latent_masks_learnable/learnable_no_compile_seed_1340 \
  --key learnable_no_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --no_use_compile
```

Result: moderately big delta! 0.010. [wandb](https://wandb.ai/sesamestrong/context_compression?nw=aaqmhq8dj9u).

#### Switching from 1 latent mask to 2 latent masks

1 latent mask: (see previous section)

2 latent masks, lr={1339,1340}: 

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_two_latent_masks \
  --log_dir logs/64_two_latent_masks/two_latent_masks_seed_1339 \
  --key two_latent_masks \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 1024 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 64 --n_embd 768 \
  --group 64_two_latent_masks \
  --log_dir logs/64_two_latent_masks/two_latent_masks_seed_1340 \
  --key two_latent_masks \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --no_use_compile
```

Result: delta is about 0.003 or 0.004. Small. See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=xgnobbbkhbp).

### Trying an attention conv

#### Checking if I can use torch.compile

Let's check if our pretty-strong baseline (2 latent masks) can be trained with compile without compromising quality.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group two_latent_masks_compile_comparison \
  --log_dir two_latent_masks_compile_comparison/compile_seed_1339 \
  --key compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group two_latent_masks_compile_comparison \
  --log_dir two_latent_masks_compile_comparison/compile_seed_1340 \
  --key compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1340 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32
```

#### Trying attention conv

Seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/a_original_seed_1339 \
  --key a_original \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv
```

Result: worse than baseline.

Init to eye matrix:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/b_eye_init_seed_1339 \
  --key b_eye_init \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye
```

Result: worse even than the first try.

Init to eye matrix, frozen (should match baseline):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/c_degen_seed_1339 \
  --key b_eye_init \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye \
  --att_conv_scale 0
```

Result: seems to diverge?

Init to eye matrix, frozen, no-compile, float32 (should match baseline):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/c_degen_seed_1339 \
  --key b_eye_init \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye \
  --att_conv_scale 0 \
  --no_use_compile \
  --att_conv_precision float32
```

Oops! Looks like I wasn't acc eye-initting them. Let's fix that.

Degen, compile:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/d_degen_seed_1339 \
  --key d_degen \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye \
  --att_conv_scale 0
```

Degen, no-compile, float32:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/d_degen_no_compile_seed_1339 \
  --key d_degen_no_compile \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye \
  --att_conv_scale 0 \
  --no_use_compile \
  --att_conv_precision float32
```

Result: degen is a big improvement over baseline. Like delta=0.010. No idea why.

Learnable, compile, no weight decay:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/e_learnable_compile_no_weight_decay_seed_1339 \
  --key e_learnable_compile_no_weight_decay \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye
```

Learnable, compile, yes weight decay:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/f_learnable_compile_yes_weight_decay_seed_1339 \
  --key f_learnable_compile_yes_weight_decay \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye \
  --att_conv_weight_decay
```

Result: big improvement over degen. Like delta=0.013.

WD seems similar to no WD. Delta = 0.000.

Learnable, compile, yes weight decay, 1 latent mask (should ~match 2 latent masks):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 12 --head_dim 22 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/g_1_latent_mask_seed_1339 \
  --key g_1_latent_mask \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye \
  --att_conv_weight_decay
```

Result: big downgrade from 2 latent masks. Delta=0.009.

More, smaller heads, no weight decay:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 25 --head_dim 11 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/g_small_heads_no_wd_seed_1339 \
  --key g_small_heads_no_wd \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye
```

More, smaller heads, yes weight decay:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 25 --head_dim 11 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/h_small_heads_wd_seed_1339 \
  --key h_small_heads_wd \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye \
  --att_conv_weight_decay
```

Result: when compared with the 1-latent-mask case, these are a good improvement. WD is better than no WD, though.

Delta from smaller-but-more-heads: 0.006.
Delta from disabling WD: -0.003.

The with-WD delta vs. 2-latent-mask, big-few-heads setting is -0.003.
Hopefully we can just add the "delta from smaller-but-more-heads" to our 2-latent-mask, big-few-heads baseline score. That would be nice!


More, smaller heads, yes weight decay, initted to match big-few-heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 25 --head_dim 11 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/i_small_heads_wd_double_eye_seed_1339 \
  --key i_small_heads_wd_double_eye \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 1 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init double_eye \
  --att_conv_weight_decay
```
Result: basically matched the eye-init case.

More, smaller heads, yes weight decay, 2 latent masks again:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 24 --head_dim 11 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/i_small_heads_wd_2_latent_masks_seed_1339 \
  --key i_small_heads_wd_2_latent_masks \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 2 \
  --one_head_per_latent_mask \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye \
  --att_conv_weight_decay
```
Result: small delta=0.002 w.r.t. the 2-latent-mask, big-few-heads setting. I'm still suspicious of this, though. I think I haven't mimicked the model correctly. We're gonna re-run this with more sane inits.

AHA! I was right to be suspicious! These models are not so different after all! Maybe this explains part of the diff between baseline and degen.

Specifically, let's compare this model vs. the 2-latent-mask, big-few-heads model:

This model has "one-head-per-latent-mask" set to True, while the other does not.
This model has 24 heads + 2 heads for 2 latent masks = 26 heads. The other has (12 heads + 1 head) * 2 = 26 heads.

So this is *not* more dense! It just has more diversity of selection.

This model has 24 different selection masks, while the other has 12.
There are also differences in the lr and init, AFAIK, which I should think carefully about.

Now, let's try comparing the degen model vs. the baseline model:

Also, should I be using all the value and attention heads? I think maybe I should.
i.e. if I'm using n latent masks, those should only exist in the inner dimension (i.e. the output of the att_conv). Then all the normal heads should be untouched.

Once we reset that, we can scale up the density much more cleanly (since we don't have any ugly 22*11 math to deal with).

<hr>

More, smaller heads, yes weight decay, 4 latent masks (cannibalized from the existing heads):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 22 --head_dim 11 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/i_small_heads_wd_2_latent_masks_seed_1339 \
  --key i_small_heads_wd_2_latent_masks \
  --selection_head_linear_combo n_latent_masks \
  --n_latent_masks 4 \
  --one_head_per_latent_mask \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye \
  --att_conv_weight_decay
```

Result: pretty bad. Not sure what this was, I think I'll just ignore it entirely.

#### Switching attention conv to a more flexible architecture

Let's train a model that should beat the 2-latent-mask, big-few-heads, wd baseline.

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --max_lr 30e-4 --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --n_heads 26 --head_dim 11 --n_embd 264 \
  --group att_conv_playground \
  --log_dir att_conv_playground/j_att_conv_n_latent_masks_seed_1339 \
  --key j_att_conv_n_latent_masks \
  --selection_head_linear_combo att_conv_n_latent_masks \
  --n_latent_masks 2 \
  --random_seed 1339 \
  --init_latent_masks_to_identity \
  --latent_mask_precision float32 \
  --att_conv \
  --att_conv_init eye
```

Update: I'm not doing this here. I'd rather get some unified measurements of MHA performance wrt. number of heads and head granularity. If that doesn't work, maybe I'll come back to this!


## Understanding MHA scaling laws

#### Finding a strong mini-model MHA baseline

Acc, IDK how head-scaling works. Very scary! Theoretically, mup should make it safe and approachable.

Disappointing that mup hasn't run this exact experiment - they only scale up head count *and hidden state size*, or head dim *and hidden state size*. There's no isolation of just head count scaling, or just head dim scaling.

I'm a little worried this law is diff for diff model scales and sequence lengths. When I find my strong baseline for the small model, I should just try a big model run (on the Yorth 10k step config, I think, or maybe even just the big model 2.5k step config), to see if it generalizes.

OK, on the mini model, let's start w/ 12 heads, head dim 64. And scale the head count to 8,16, and 24.

Augh goddamn. The head dims are not clean. This model has to be either huge or tiny. Maybe we need fewer layers? Oh, but fewer layers is very anti-mup, I think.

IG 12 heads is a weird baseline. Let's fix the baseline head dim to 32 and use 8 heads.

So let's tune the lr for the 8-head setting:

lr=30e-4, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 8 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_baseline_lr_tuning \
  --log_dir mha_baseline_lr_tuning/lr_30e-4_seed_1339 \
  --max_lr 30e-4 \
  --key lr_30e-4 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 8 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_baseline_lr_tuning \
  --log_dir mha_baseline_lr_tuning/lr_30e-4_seed_1340 \
  --max_lr 30e-4 \
  --key lr_30e-4 \
  --random_seed 1340
```


lr=20e-4, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 8 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_baseline_lr_tuning \
  --log_dir mha_baseline_lr_tuning/lr_20e-4_seed_1339 \
  --max_lr 20e-4 \
  --key lr_20e-4 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 8 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_baseline_lr_tuning \
  --log_dir mha_baseline_lr_tuning/lr_20e-4_seed_1340 \
  --max_lr 20e-4 \
  --key lr_20e-4 \
  --random_seed 1340
```


lr=45e-4, seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 8 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_baseline_lr_tuning \
  --log_dir mha_baseline_lr_tuning/lr_45e-4_seed_1339 \
  --max_lr 45e-4 \
  --key lr_45e-4 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 8 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_baseline_lr_tuning \
  --log_dir mha_baseline_lr_tuning/lr_45e-4_seed_1340 \
  --max_lr 45e-4 \
  --key lr_45e-4 \
  --random_seed 1340
```

Result: lr=30e-4 is best. Cool! That's good transfer from selective -> unselective. See [wandb](https://wandb.ai/sesamestrong/context_compression/panel/ieugzanro?nw=nnp8sma6pm).

#### Scaling up and down the number of 32-dim heads

1 head:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_1_seed_1339 \
  --n_heads 1 \
  --key n_heads_1 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_1_seed_1340 \
  --n_heads 1 \
  --key n_heads_1 \
  --random_seed 1340
```


2 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_2_seed_1339 \
  --n_heads 2 \
  --key n_heads_2 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_2_seed_1340 \
  --n_heads 2 \
  --key n_heads_2 \
  --random_seed 1340
```


4 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_4_seed_1339 \
  --n_heads 4 \
  --key n_heads_4 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_4_seed_1340 \
  --n_heads 4 \
  --key n_heads_4 \
  --random_seed 1340
```


8 heads: (see last experiment, lr=30)

8 heads with float32 precision (not to prevent divergence - just to check the perf boost from fp32):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_8_fp32_seed_1339 \
  --n_heads 8 \
  --key n_heads_8_fp32 \
  --random_seed 1339 \
  --attn_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_8_fp32_seed_1340 \
  --n_heads 8 \
  --key n_heads_8_fp32 \
  --random_seed 1340 \
  --attn_precision float32
```

12 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_12_seed_1339 \
  --n_heads 12 \
  --key n_heads_12 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_12_seed_1340 \
  --n_heads 12 \
  --key n_heads_12 \
  --random_seed 1340
```

12 heads with float32 precision (to prevent divergence):


```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_12_fp32_seed_1339 \
  --n_heads 12 \
  --key n_heads_12_fp32 \
  --random_seed 1339 \
  --attn_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_12_fp32_seed_1340 \
  --n_heads 12 \
  --key n_heads_12_fp32 \
  --random_seed 1340 \
  --attn_precision float32
```

Result: fp32 fixes the divergence issue!

12 heads with no compile (to prevent divergence):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_12_no_compile_seed_1339 \
  --n_heads 12 \
  --key n_heads_12_no_compile \
  --random_seed 1339 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_12_no_compile_seed_1340 \
  --n_heads 12 \
  --key n_heads_12_no_compile \
  --random_seed 1340 \
  --no_use_compile
```

Result: no-compile also fixes the divergence issue!

16 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_16_seed_1339 \
  --n_heads 16 \
  --key n_heads_16 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_16_seed_1340 \
  --n_heads 16 \
  --key n_heads_16 \
  --random_seed 1340
```

20 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_20_seed_1339 \
  --n_heads 20 \
  --key n_heads_20 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_20_seed_1340 \
  --n_heads 20 \
  --key n_heads_20 \
  --random_seed 1340
```

20 heads with float32 precision (to prevent divergence):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_20_fp32_seed_1339 \
  --n_heads 20 \
  --key n_heads_20_fp32 \
  --random_seed 1339 \
  --attn_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_20_fp32_seed_1340 \
  --n_heads 20 \
  --key n_heads_20_fp32 \
  --random_seed 1340 \
  --attn_precision float32
```

32 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_32_seed_1339 \
  --n_heads 32 \
  --key n_heads_32 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_32_seed_1340 \
  --n_heads 32 \
  --key n_heads_32 \
  --random_seed 1340
```

64 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_64_seed_1339 \
  --n_heads 64 \
  --key n_heads_64 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_64_seed_1340 \
  --n_heads 64 \
  --key n_heads_64 \
  --random_seed 1340
```

Result: converges!

64 heads with float32 precision (to prevent divergence):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_64_fp32_seed_1339 \
  --n_heads 64 \
  --key n_heads_64_fp32 \
  --random_seed 1339 \
  --attn_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_64_fp32_seed_1340 \
  --n_heads 64 \
  --key n_heads_64_fp32 \
  --random_seed 1340 \
  --attn_precision float32
```

128 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_128_seed_1339 \
  --n_heads 128 \
  --key n_heads_128 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_128_seed_1340 \
  --n_heads 128 \
  --key n_heads_128 \
  --random_seed 1340
```

128 heads with float32 precision (to prevent divergence):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_128_fp32_seed_1339 \
  --n_heads 128 \
  --key n_heads_128_fp32 \
  --random_seed 1339 \
  --attn_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_128_fp32_seed_1340 \
  --n_heads 128 \
  --key n_heads_128_fp32 \
  --random_seed 1340 \
  --attn_precision float32
```

Result: both these 128 heads experiments diverge.

128 heads with float32 precision and no compile (to prevent divergence):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_128_fp32_no_compile_seed_1339 \
  --n_heads 128 \
  --key n_heads_128_fp32_no_compile \
  --random_seed 1339 \
  --attn_precision float32 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_128_fp32_no_compile_seed_1340 \
  --n_heads 128 \
  --key n_heads_128_fp32_no_compile \
  --random_seed 1340 \
  --attn_precision float32 \
  --no_use_compile
```

Result: 128 is still divergent. Ugh! Let's try just no-compile, with no float32. float32 conversion might be messing us up?

128 heads with no compile (to prevent divergence):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_128_no_compile_seed_1339 \
  --n_heads 128 \
  --key n_heads_128_no_compile \
  --random_seed 1339 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_hd_scale_nh \
  --log_dir mha_const_hd_scale_nh/n_heads_128_no_compile_seed_1340 \
  --n_heads 128 \
  --key n_heads_128_no_compile \
  --random_seed 1340 \
  --no_use_compile
```

Result: didn't fix it. I don't know why 128 heads diverges. I wonder if flashattention would solve this.

Result: Generally, seems like loss is basically log-linear in the number of heads. So doubling the number of heads will decrease the loss by a certain margin.

But 32 -> 64 heads is a pretty big jump, so I'm not totally sure abt "log-linear".

#### Scaling up and down the granularity for a fixed number of attention dimensions

Let's do head_dim=2,4,8,16,32,64,128,256.

256 will prob be ridiculous. 2 will prob also be ridiculous, b/c it'll play so badly with RoPE. Unfortunate - maybe I should measure this differently?

IG I don't really understand the behavior for small head_dim (i.e. high granularity) b/c I don't know how to fix its RoPE problems.

So let's just measure normally now. And we will trust the upscaled head_dim curves, but assume the downscaled head_dim curves are an upper bound on the loss (or a lower bound on accuracy).

head_dim=2:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_2_seed_1339 \
  --n_heads 128 \
  --key head_dim_2 \
  --random_seed 1339 \
  --head_dim 2
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_2_seed_1340 \
  --n_heads 128 \
  --key head_dim_2 \
  --random_seed 1340 \
  --head_dim 2
```

head_dim=4:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_4_seed_1339 \
  --n_heads 64 \
  --key head_dim_4 \
  --random_seed 1339 \
  --head_dim 4
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_4_seed_1340 \
  --n_heads 64 \
  --key head_dim_4 \
  --random_seed 1340 \
  --head_dim 4
```


head_dim=8:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_8_seed_1339 \
  --n_heads 32 \
  --key head_dim_8 \
  --random_seed 1339 \
  --head_dim 8
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_8_seed_1340 \
  --n_heads 32 \
  --key head_dim_8 \
  --random_seed 1340 \
  --head_dim 8
```


head_dim=16:


```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_16_seed_1339 \
  --n_heads 16 \
  --key head_dim_16 \
  --random_seed 1339 \
  --head_dim 16
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_8_seed_1340 \
  --n_heads 16 \
  --key head_dim_16 \
  --random_seed 1340 \
  --head_dim 16
```

head_dim=32 (should match previous experiment results):


```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_32_seed_1339 \
  --n_heads 8 \
  --key head_dim_32 \
  --random_seed 1339 \
  --head_dim 32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_32_seed_1340 \
  --n_heads 8 \
  --key head_dim_32 \
  --random_seed 1340 \
  --head_dim 32
```

head_dim=64:


```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_64_seed_1339 \
  --n_heads 4 \
  --key head_dim_64 \
  --random_seed 1339 \
  --head_dim 64
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_64_seed_1340 \
  --n_heads 4 \
  --key head_dim_64 \
  --random_seed 1340 \
  --head_dim 64
```

head_dim=128:


```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_128_seed_1339 \
  --n_heads 2 \
  --key head_dim_128 \
  --random_seed 1339 \
  --head_dim 128
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_128_seed_1340 \
  --n_heads 2 \
  --key head_dim_128 \
  --random_seed 1340 \
  --head_dim 128
```

head_dim=256:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_256_seed_1339 \
  --n_heads 1 \
  --key head_dim_256 \
  --random_seed 1339 \
  --head_dim 256
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_const_width_scale_hd \
  --log_dir mha_const_width_scale_hd/head_dim_256_seed_1340 \
  --n_heads 1 \
  --key head_dim_256 \
  --random_seed 1340 \
  --head_dim 256
```

Results: honestly a little fuzzy. Generally it seems like more granular is better. Except head_dim=16 is high-variance and kinda bad, and head_dim=4 and =2 are both bad.

But head_dim=8 is good, and always beats everything else.

So I suspect some weird RoPE effect is happening, maybe.

Actually, probably not RopE exactly - since we use the GPT method of learnable embeds.

[wandb](https://wandb.ai/sesamestrong/context_compression?nw=hxqcqaoztrj)

#### Repro-ing this performance on my new dense attention codepath

8 heads, head_dim=32:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --n_heads 8 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --max_lr 30e-4 \
  --group dense_attention_repro_old_codepath \
  --log_dir dense_attention_repro_old_codepath/a_first_try \
  --mup_zero_init
```

## TODO fully repeat the two scaleup experiments, run them during breakfast


#### Repro-ing the rough outline of scaleup experiments

2 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_2_seed_1339 \
  --n_heads 2 \
  --key n_heads_2 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_2_seed_1340 \
  --n_heads 2 \
  --key n_heads_2 \
  --random_seed 1340
```


4 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_4_seed_1339 \
  --n_heads 4 \
  --key n_heads_4 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_4_seed_1340 \
  --n_heads 4 \
  --key n_heads_4 \
  --random_seed 1340
```

8 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_8_seed_1339 \
  --n_heads 8 \
  --key n_heads_8 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_8_seed_1340 \
  --n_heads 8 \
  --key n_heads_8 \
  --random_seed 1340
```

16 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_16_seed_1339 \
  --n_heads 16 \
  --key n_heads_16 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_16_seed_1340 \
  --n_heads 16 \
  --key n_heads_16 \
  --random_seed 1340
```

32 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_32_seed_1339 \
  --n_heads 32 \
  --key n_heads_32 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_32_seed_1340 \
  --n_heads 32 \
  --key n_heads_32 \
  --random_seed 1340
```

64 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_64_seed_1339 \
  --n_heads 64 \
  --key n_heads_64 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_64_seed_1340 \
  --n_heads 64 \
  --key n_heads_64 \
  --random_seed 1340
```

Result: converges!

64 heads with float32 precision (to prevent divergence):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_64_fp32_seed_1339 \
  --n_heads 64 \
  --key n_heads_64_fp32 \
  --random_seed 1339 \
  --attn_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_64_fp32_seed_1340 \
  --n_heads 64 \
  --key n_heads_64_fp32 \
  --random_seed 1340 \
  --attn_precision float32
```

128 heads with float32 precision (to prevent divergence):

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_128_fp32_seed_1339 \
  --n_heads 128 \
  --key n_heads_128_fp32 \
  --random_seed 1339 \
  --attn_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group new_mha_const_hd_scale_nh \
  --log_dir new_mha_const_hd_scale_nh/n_heads_128_fp32_seed_1340 \
  --n_heads 128 \
  --key n_heads_128_fp32 \
  --random_seed 1340 \
  --attn_precision float32
```

Result: mostly performs better than the old version, no idea why, spooky. But 128 still diverges. See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=g6z5lybrgdn).

#### Trying to fix the divergence issue

Let's try using att_kind=self.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group fix_128_heads_divergence \
  --log_dir fix_128_heads_divergence/n_heads_32_seed_1339 \
  --n_heads 32 \
  --key n_heads_32 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group fix_128_heads_divergence \
  --log_dir fix_128_heads_divergence/n_heads_64_seed_1339 \
  --n_heads 64 \
  --key n_heads_64 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group fix_128_heads_divergence \
  --log_dir fix_128_heads_divergence/n_heads_32_seed_1339 \
  --n_heads 32 \
  --key n_heads_32 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group fix_128_heads_divergence \
  --log_dir fix_128_heads_divergence/n_heads_128_seed_1339 \
  --n_heads 128 \
  --key n_heads_128 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group fix_128_heads_divergence \
  --log_dir fix_128_heads_divergence/n_heads_256_seed_1339 \
  --n_heads 256 \
  --key n_heads_256 \
  --random_seed 1339
```


```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group fix_128_heads_divergence \
  --log_dir fix_128_heads_divergence/n_heads_512_seed_1339 \
  --n_heads 512 \
  --key n_heads_512 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 8 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group fix_128_heads_divergence \
  --log_dir fix_128_heads_divergence/n_heads_1024_seed_1339 \
  --n_heads 1024 \
  --key n_heads_1024 \
  --random_seed 1339
```

Result: sdpa at n_heads=32 beats all my custom mha impls. And more heads makes it worse. We gotta figure out what's wrong w/ my old impl (and what's wrong with sdpa, to give it bad scaling).
OK, let's check if sdpa matches my impl at inference time.

#### Investigating sdpa vs. naive numeric differences

In the last experiment, I ran sdpa with bs=16 and bs=128. The bs=16 version soundly beat all my custom mha impls. But bs=128 version matches my custom mha impl performance.

Let's re-run all 3 (mha, sdpa 16, sdpa 128) with another seed to try to repro this.

Attention=selective, selection disabled:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --n_embd 256 --attention_kind selective --disable_selection \
  --group mha_numerics \
  --log_dir mha_numerics/selective_disabled_seed_1340 \
  --n_heads 32 \
  --key selective_disabled \
  --random_seed 1340
```

Custom MHA impl:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_seed_1340 \
  --n_heads 32 \
  --key mha_impl \
  --random_seed 1340
```

SDPA, bs=16:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/sdpa_16_seed_1340 \
  --n_heads 32 \
  --key sdpa_16 \
  --random_seed 1340
```

SDPA, bs=128:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/sdpa_128_seed_1340 \
  --n_heads 32 \
  --key sdpa_128 \
  --random_seed 1340
```

Result: yup, it's true. OK, so let's try to match the bs=16 SDPA version with a bs=128 SDPA version, maybe. Then let's try to match both of them with a general (possibly slow) MHA impl.

SDPA, bs=32:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/sdpa_32_seed_1340 \
  --n_heads 32 \
  --key sdpa_32 \
  --random_seed 1340 \
  --sdpa_iter_size 16
```

SDPA, bs=64:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/sdpa_64_seed_1340 \
  --n_heads 32 \
  --key sdpa_64 \
  --random_seed 1340 \
  --sdpa_iter_size 16
```


SDPA, bs=128, split to 16:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/sdpa_128_seed_1340 \
  --n_heads 32 \
  --key sdpa_128 \
  --random_seed 1340 \
  --sdpa_iter_size 16
```

Custom MHA impl, stabilized scores:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_stabilized_2_seed_1340 \
  --n_heads 32 \
  --key mha_impl_stabilized_2 \
  --random_seed 1340 \
  --stabilize_attn_scores
```

SDPA, bs=64, no compile:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/sdpa_64_no_compile_seed_1340 \
  --n_heads 32 \
  --key sdpa_64_no_compile \
  --random_seed 1340 \
  --sdpa_iter_size 16 \
  --no_use_compile
```

I also notice the SDPA runs and the MHA runs separate immediately (i.e. val loss difference of 0.2 at step 100)

Comparing a few configs:

SDPA:
```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_seed_1340 \
  --n_heads 32 \
  --key mha_impl \
  --random_seed 1340
```
val_loss=6.6727.

MHA, mup init zero:
```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_seed_1340 \
  --n_heads 32 \
  --key mha_impl \
  --random_seed 1340
```
val_loss=6.7689.

MHA:
```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_seed_1340 \
  --n_heads 32 \
  --key mha_impl \
  --random_seed 1340
```
val_loss=6.7754.


MHA, but actually using SDPA:
```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_seed_1340 \
  --n_heads 32 \
  --key mha_impl \
  --random_seed 1340 \
  --override_use_sdpa
```
val_loss=6.8320. Hrrm, so is it the init that's killing us? Or maybe smth else...

MHA using SDPA, but with nanogpt c_proj init:
```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_seed_1340 \
  --n_heads 32 \
  --key mha_impl \
  --random_seed 1340 \
  --override_use_sdpa
```
val_loss=6.5995. Great! That was it.

MHA, but with nanogpt c_proj init:
```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_seed_1340 \
  --n_heads 32 \
  --key mha_impl \
  --random_seed 1340
```
val_loss=6.6175.

MHA with mup zero init, stabilized scores and nanogpt c_proj init:
```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_seed_1340 \
  --n_heads 32 \
  --key mha_impl \
  --random_seed 1340 \
  --stabilize_attn_scores
```
val_loss=6.6068. Seems fine still.

Great! So let's do another big run of MHA with this change.

Custom MHA impl, stabilized scores and nanogpt c_proj init:
```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group mha_numerics \
  --log_dir mha_numerics/mha_impl_stabilized_2_seed_1340 \
  --n_heads 32 \
  --key mha_impl_stabilized_2 \
  --random_seed 1340 \
  --stabilize_attn_scores
```

Result: still, nothing matches the SDPA n_heads=32, bs=16 performance. Even this last run. HOWEVER - this last run matches the performance of bs>16 SDPA impls, which is good news. We just would like to make more progress.

Results of runs up to this point: [wandb](https://wandb.ai/sesamestrong/context_compression/panel/ieugzanro?nw=dwzaqcveqnc).

This last run at least matches the early performance of the SDPA impls. But even every SDPA impl but the bs=16 one settles to a higher loss than bs=16. Delta of 0.04 - from 4.289 -> 4.247.

Let's keep investigating - does SDPA bs=16 still behave this way when overriding my MHA impl?

We should be able to see differences by step 1000. So let's do a few more runs.

To sanity check, let's re-run the SDPA bs=16 for seed 1339 and 1340, to make *sure* we're not hallucinating.

Then we'll check, just for seed=1339, whether MHA with impl overridden to SDPA will match that amazing loss curve.

SDPA, bs=16, seed={1339, 1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_16_seed_1339 \
  --n_heads 32 \
  --key sdpa_16 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_16_seed_1340 \
  --n_heads 32 \
  --key sdpa_16 \
  --random_seed 1340
```

MHA, bs=16, SDPA impl, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/mha_overridden_sdpa_16_seed_1339 \
  --n_heads 32 \
  --key mha_overridden_sdpa_16 \
  --random_seed 1339 \
  --override_use_sdpa
```

MHA with no mup zero init, bs=16, SDPA impl, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/mha_nonzero_overridden_sdpa_16_seed_1339 \
  --n_heads 32 \
  --key mha_nonzero_overridden_sdpa_16 \
  --random_seed 1339 \
  --override_use_sdpa
```

MHA, stabilized scores and nanogpt c_proj init, bs=16, seed={1339, 1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/mha_stabilized_16_seed_1339 \
  --n_heads 32 \
  --key mha_stabilized_16 \
  --random_seed 1339 \
  --stabilize_attn_scores
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/mha_stabilized_16_seed_1340 \
  --n_heads 32 \
  --key mha_stabilized_16 \
  --random_seed 1340 \
  --stabilize_attn_scores
```

Results: all five of these runs had similarly great loss curves! Woohoo!

So we don't have to think about spooky SDPA kernel issues. It's just spooky Pytorch gradient-accumulation or DDP issues, maybe.

So let's run the MHA impl with bs=64. I think this will have a bad loss curve.

MHA, stabilized scores and nanogpt c_proj init, bs=64, seed={1339, 1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/mha_stabilized_64_seed_1339 \
  --n_heads 32 \
  --key mha_stabilized_64 \
  --random_seed 1339 \
  --stabilize_attn_scores
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/mha_stabilized_64_seed_1340 \
  --n_heads 32 \
  --key mha_stabilized_64 \
  --random_seed 1340 \
  --stabilize_attn_scores
```

Results: these two runs both have bad loss curves. Spoooky!

Theoretically, we can bisect between these two MHA setups to isolate the issue.

Hrrm. Seems kinda hard to debug this in DDP mode. Let's hope the same result holds for one-gpu inference!

Actually, let's verify that so we can run the bisect on a single-GPU machine. For now, let's just check the discrepancy on SDPA 16 vs. 128, since those are the fastest to train.

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_128_1gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_128_1gpu \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_16_1gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_16_1gpu \
  --random_seed 1339
```

Now let's run it on the MHA impl too. We'll stop this early if the SDPA canaries pass.


```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/mha_64_1gpu_seed_1339 \
  --n_heads 32 \
  --key mha_impl_64 \
  --random_seed 1339 \
  --stabilize_attn_scores
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/mha_16_1gpu_seed_1339 \
  --n_heads 32 \
  --key mha_impl_16 \
  --random_seed 1339 \
  --stabilize_attn_scores
```

Results: there's less degradation from 16 -> 128 on one GPU (for both SDPA and MHA impls). BUT there is still a degradation!!

Hrrrrrrrm. I wonder why this could be happening. We should ask chatgpt when I get back from my run.

Let's try 4-gpu sdpa with float32 precision (for the whole run and the attn part both).

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_64_f32_4gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_64_f32_4gpu \
  --random_seed 1339 \
  --attn_precision float32 \
  --autocast_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_16_f32_4gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_16_f32_4gpu \
  --random_seed 1339 \
  --attn_precision float32 \
  --autocast_precision float32
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_64_bf16_4gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_64_bf16_4gpu \
  --random_seed 1339 \
  --attn_precision bfloat16 \
  --autocast_precision bfloat16
```

Results: I don't notice a difference between the f32 and bf16 64-bs runs. This probably makes sense - I'm guessing the badness comes from the gradient accumulation on the params, not the computed gradients themselves.

However, for some reason, sdpa with bs=64 is less destructive than MHA with bs=64. Let's make sure we didn't accidentally fix sdpa altogether. Specifically, let's do a 128-bs sdpa run.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_128_4gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_128_4gpu \
  --random_seed 1339
```

OK, now let's try simulating 16-micro-bs gradient accumulation using the bs=128 run.

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_64_16_4gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_64_16_4gpu \
  --random_seed 1339 \
  --attn_precision bfloat16 \
  --autocast_precision bfloat16 \
  --simulate_micro_bs 16
```

And with the second method:

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_64_16_v2_4gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_64_16_v2_4gpu \
  --random_seed 1339 \
  --attn_precision bfloat16 \
  --autocast_precision bfloat16 \
  --simulate_micro_bs 16
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_128_16_v2_real_4gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_128_16_v2_real_4gpu \
  --random_seed 1339 \
  --attn_precision bfloat16 \
  --autocast_precision bfloat16 \
  --simulate_micro_bs_2 16
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_128_16_v2_real_no_compile_4gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_128_16_v2_real_no_compile_4gpu \
  --random_seed 1339 \
  --attn_precision bfloat16 \
  --autocast_precision bfloat16 \
  --simulate_micro_bs_2 16 \
  --no_use_compile
```

```vast:finished
cd /workspace/context-compression && git pull && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_spooky \
  --log_dir sdpa_16_spooky/sdpa_64_no_compile_4gpu_seed_1339 \
  --n_heads 32 \
  --key sdpa_64_no_compile_4gpu \
  --random_seed 1339 \
  --attn_precision bfloat16 \
  --autocast_precision bfloat16 \
  --no_use_compile
```

Runs are listed in [wandb](https://wandb.ai/sesamestrong/context_compression/panel/ieugzanro?nw=fua6m82uqmc).

Update: Looks like the difference is just that different batch sizes lead to different validation sets. I think.

So in the morning, let's try to use a shared, deterministic (and big) valid set between all settings.

Then let's check if we've fixed the 128-head stability issue. I hope so, but I worry we haven't.

Then let's tune the 32-head init to work with mup.

#### Re-run the 16-head and 128-head SDPA runs with the new validation set.

I predict their curves will basically match.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_maybe_fixed \
  --log_dir sdpa_16_maybe_fixed/sdpa_16_seed_1339 \
  --n_heads 32 \
  --key sdpa_16 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group sdpa_16_maybe_fixed \
  --log_dir sdpa_16_maybe_fixed/sdpa_128_seed_1339 \
  --n_heads 32 \
  --key sdpa_128 \
  --random_seed 1339
```

Result: yup, they match. [wandb](https://wandb.ai/sesamestrong/context_compression/panel/ieugzanro?nw=bpki4z4pou9).

#### Re-run SDPA head-count-upscaling

Last time I tried using SDPA, increasing n_heads made performance worse. I wonder why? That would violate the bigger-is-better assumptions of mup, right?

Let's try it again.

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group scaling_sdpa_nh \
  --log_dir scaling_sdpa_nh/sdpa_nh_32_seed_1339 \
  --n_heads 32 \
  --key sdpa_nh_32 \
  --random_seed 1339
```

64 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group scaling_sdpa_nh \
  --log_dir scaling_sdpa_nh/sdpa_nh_64_seed_1339 \
  --n_heads 64 \
  --key sdpa_nh_64 \
  --random_seed 1339
```

128 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init \
  --group scaling_sdpa_nh \
  --log_dir scaling_sdpa_nh/sdpa_nh_128_seed_1339 \
  --n_heads 128 \
  --key sdpa_nh_128 \
  --random_seed 1339
```

This time, 128 heads diverges!! I wonder why. Didn't I use the same CLI args as before?

So the same pattern is still happening. Maybe let's try a (local to this machine) coord check. Hrrm, since SDPA is so optimized, maybe we want to coord check MHA impl instead. OK let's do that.

#### MHA head-count-upscaling (should be same as SDPA)

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores \
  --group scaling_mha_nh \
  --log_dir scaling_mha_nh/mha_nh_32_seed_1339 \
  --n_heads 32 \
  --key sdpa_nh_32 \
  --random_seed 1339
```

64 heads:

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores \
  --group scaling_mha_nh \
  --log_dir scaling_mha_nh/mha_nh_64_seed_1339 \
  --n_heads 64 \
  --key sdpa_nh_64 \
  --random_seed 1339
```

128 heads:

```
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores \
  --group scaling_mha_nh \
  --log_dir scaling_mha_nh/mha_nh_128_seed_1339 \
  --n_heads 128 \
  --key sdpa_nh_128 \
  --random_seed 1339
```

Note: looking back at this experiment, it looks like these never ran. That's ok though, since the SDPA diverged, so I had to fix that.

#### Local coord check - 32 vs. 64 vs. 128

For now, let's just assume that MHA has the same bigger-is-worse problem as SDPA. We're running this locally for 10 steps.

```
for n_heads in 2 4 8 16 32 64 128; do
  SKIP_WANDB=false python -m context_compression.train \
    --total_batch_size 8192 --seq_len 256 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores --random_seed 1339 \
    --group mha_coord_check_2 \
    --mup_enable_coord_check_logging --no_decay_lr --max_steps 20 --no_use_compile --no_upload_to_hf \
    --log_dir mha_coord_check_2/mha_nh_${n_heads} \
    --n_heads ${n_heads} \
    --key mha_nh_${n_heads}
done
```

Result: coord check [fails](https://wandb.ai/sesamestrong/context_compression?nw=ebyi79pfjid). Notice how much the attn magnitude scales with n_heads.

Now with nanogpt scale init disabled: (should just be a constant initialization scaling factor, but idk, maybe it'll fix it)

```
for n_heads in 2 4 8 16 32 64 128; do
  SKIP_WANDB=false python -m context_compression.train \
    --total_batch_size 8192 --seq_len 256 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores --random_seed 1339 \
    --group mha_coord_check_3 \
    --mup_enable_coord_check_logging --no_decay_lr --max_steps 20 --no_use_compile --no_upload_to_hf \
    --log_dir mha_coord_check_3/mha_nh_${n_heads} \
    --n_heads ${n_heads} \
    --key mha_nh_${n_heads}
done
```

Result: seems to [pass](https://wandb.ai/sesamestrong/context_compression?nw=tv8lw8bbny).

I'm kinda confused - the only diff between these two coord checks is a constant init scaling factor. But why does that make a difference?

Let's run the passing coord check once more with more coords checked. Then the failing coord check is next.

```
for n_heads in 2 4 8 16 32 64 128; do
  SKIP_WANDB=false python -m context_compression.train \
    --total_batch_size 8192 --seq_len 256 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores --random_seed 1339 \
    --group mha_coord_check_4 \
    --mup_enable_coord_check_logging --no_decay_lr --max_steps 20 --no_use_compile --no_upload_to_hf \
    --log_dir mha_coord_check_4/mha_nh_${n_heads} \
    --n_heads ${n_heads} \
    --key mha_nh_${n_heads}
done
```

Result: Looks like maybe the attn score gets big when using big n_heads. No idea if that's important - it might not be, since we're stabilizing them. But I'm guessing it's a symptom of smth. To check it, just switch to `mha_coord_check_4` in the wandb link above.

I think it's maybe just init effects. At init (i.e. step=0), the attn score seems to be linearly dependent on n_heads. Or maybe sqrt-dependent. Hrrm, why is this happening at all? Aren't I setting mup_zero_init to true? So all attention scores should be zero at init!!

Hrrm - theoretically, attn score should maybe just be a function of the query and the key values, right? 

OK let's solve this problem. Specifically, make the model pass an attn_score coord check.

I'm still not sure, feeling kinda slow rn. Let's just run it with the fixed query-initted-to-zero code. What if that fixes it? It'll surely hide my other bug.

```
for n_heads in 2 4 8 16 32 64 128; do
  SKIP_WANDB=false python -m context_compression.train \
    --total_batch_size 8192 --seq_len 256 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores --random_seed 1339 \
    --group mha_coord_check_5 \
    --mup_enable_coord_check_logging --no_decay_lr --max_steps 20 --no_use_compile --no_upload_to_hf \
    --log_dir mha_coord_check_5/mha_nh_${n_heads} \
    --n_heads ${n_heads} \
    --key mha_nh_${n_heads}
done
```

Result: still fails the coord check. Diverges right after step 0. To check it, just switch to `mha_coord_check_5` in the wandb link above.

I think I might have a fix now, replacing a bad expression for `self.head_dim` with a correct one:

```
for n_heads in 2 4 8 16 32 64 128; do
  SKIP_WANDB=false python -m context_compression.train \
    --total_batch_size 8192 --seq_len 256 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores --random_seed 1339 \
    --group mha_coord_check_6 \
    --mup_enable_coord_check_logging --no_decay_lr --max_steps 20 --no_use_compile --no_upload_to_hf \
    --log_dir mha_coord_check_6/mha_nh_${n_heads} \
    --n_heads ${n_heads} \
    --key mha_nh_${n_heads}
done
```

Result: passes the coord check! To check it, just switch to `mha_coord_check_6` in the wandb link above.

OK, let's re-run the bigger-is-better runs with the new fix.

#### Re-run bigger-is-better runs with the new fix, and nanogpt scale init back on

32

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_nh_2 \
  --log_dir scaling_sdpa_nh_2/sdpa_nh_32_seed_1339 \
  --n_heads 32 \
  --key sdpa_nh_32 \
  --random_seed 1339
```

64 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_nh_2 \
  --log_dir scaling_sdpa_nh_2/sdpa_nh_64_seed_1339 \
  --n_heads 64 \
  --key sdpa_nh_64 \
  --random_seed 1339
```

128 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_nh_2 \
  --log_dir scaling_sdpa_nh_2/sdpa_nh_128_seed_1339 \
  --n_heads 128 \
  --key sdpa_nh_128 \
  --random_seed 1339
```

Result: great! Bigger is better. [wandb](https://wandb.ai/sesamestrong/context_compression?nw=5jkf5ekf8qn).

#### Bigger-is-better runs for MHA, with the new fix (actually - accidentally using SDPA)

1 head:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_2 \
  --log_dir scaling_mha_nh_2/mha_nh_1_seed_1339 \
  --n_heads 1 \
  --key mha_nh_1 \
  --random_seed 1339
```

2 head:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_2 \
  --log_dir scaling_mha_nh_2/mha_nh_2_seed_1339 \
  --n_heads 2 \
  --key mha_nh_2 \
  --random_seed 1339
```

4 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_2 \
  --log_dir scaling_mha_nh_2/mha_nh_4_seed_1339 \
  --n_heads 4 \
  --key mha_nh_4 \
  --random_seed 1339
```

8 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_2 \
  --log_dir scaling_mha_nh_2/mha_nh_8_seed_1339 \
  --n_heads 8 \
  --key mha_nh_8 \
  --random_seed 1339
```

16 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_2 \
  --log_dir scaling_mha_nh_2/mha_nh_16_seed_1339 \
  --n_heads 16 \
  --key mha_nh_16 \
  --random_seed 1339
```

32 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_2 \
  --log_dir scaling_mha_nh_2/mha_nh_32_seed_1339 \
  --n_heads 32 \
  --key mha_nh_32 \
  --random_seed 1339
```

64 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_2 \
  --log_dir scaling_mha_nh_2/mha_nh_64_seed_1339 \
  --n_heads 64 \
  --key mha_nh_64 \
  --random_seed 1339
```

128 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_2 \
  --log_dir scaling_mha_nh_2/mha_nh_128_seed_1339 \
  --n_heads 128 \
  --key mha_nh_128 \
  --random_seed 1339
```

256 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_2 \
  --log_dir scaling_mha_nh_2/mha_nh_256_seed_1339 \
  --n_heads 256 \
  --key mha_nh_256 \
  --random_seed 1339
```

Oops! Those were all running SDPA, actually. That's a shame. Let's try again with MHA. Let's just train 128-head, 32-head, and 4-head, to make sure MHA behaves the same as SDPA.

Result: bigger *is* better! What a logarithmic relationship this is. Wow.

[wandb](https://wandb.ai/sesamestrong/context_compression?nw=c4935zi3io).

#### Bigger-is-better runs for MHA (actually using MHA this time)

4 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 64 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_3 \
  --log_dir scaling_mha_nh_3/mha_nh_4_seed_1339 \
  --n_heads 4 \
  --key mha_nh_4 \
  --random_seed 1339
```

32 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_3 \
  --log_dir scaling_mha_nh_3/mha_nh_32_seed_1339 \
  --n_heads 32 \
  --key mha_nh_32 \
  --random_seed 1339
```

128 heads:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 32 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_mha_nh_3 \
  --log_dir scaling_mha_nh_3/mha_nh_128_seed_1339 \
  --n_heads 128 \
  --key mha_nh_128 \
  --random_seed 1339
```

Result: seems a little bit better than SDPA, but the general pattern is the same. [wandb](https://wandb.ai/sesamestrong/context_compression/panel/ieugzanro?nw=c4935zi3io).

#### SDPA granularity scaling experiments

Let's do head_dim=2,4,8,16,32,64,128,256.

head_dim=2,seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_2_seed_1339 \
  --n_heads 128 --head_dim 2 --head_dim_value 2 \
  --key hd_2 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_2_seed_1340 \
  --n_heads 128 --head_dim 2 --head_dim_value 2 \
  --key hd_2 \
  --random_seed 1340
```

head_dim=4,seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_4_seed_1339 \
  --n_heads 64 --head_dim 4 --head_dim_value 4 \
  --key hd_4 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_4_seed_1340 \
  --n_heads 64 --head_dim 4 --head_dim_value 4 \
  --key hd_4 \
  --random_seed 1340
```

head_dim=8,seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_8_seed_1339 \
  --n_heads 32 --head_dim 8 --head_dim_value 8 \
  --key hd_8 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_8_seed_1340 \
  --n_heads 32 --head_dim 8 --head_dim_value 8 \
  --key hd_8 \
  --random_seed 1340
```

head_dim=16,seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_16_seed_1339 \
  --n_heads 16 --head_dim 16 --head_dim_value 16 \
  --key hd_16 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_16_seed_1340 \
  --n_heads 16 --head_dim 16 --head_dim_value 16 \
  --key hd_16 \
  --random_seed 1340
```

head_dim=32,seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_32_seed_1339 \
  --n_heads 8 --head_dim 32 --head_dim_value 32 \
  --key hd_32 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_32_seed_1340 \
  --n_heads 8 --head_dim 32 --head_dim_value 32 \
  --key hd_32 \
  --random_seed 1340
```

head_dim=64,seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_64_seed_1339 \
  --n_heads 4 --head_dim 64 --head_dim_value 64 \
  --key hd_64 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_64_seed_1340 \
  --n_heads 4 --head_dim 64 --head_dim_value 64 \
  --key hd_64 \
  --random_seed 1340
```

head_dim=128,seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_128_seed_1339 \
  --n_heads 2 --head_dim 128 --head_dim_value 128 \
  --key hd_128 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_128_seed_1340 \
  --n_heads 2 --head_dim 128 --head_dim_value 128 \
  --key hd_128 \
  --random_seed 1340
```

head_dim=256,seed={1339,1340}:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_256_seed_1339 \
  --n_heads 1 --head_dim 256 --head_dim_value 256 \
  --key hd_256 \
  --random_seed 1339
```

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 \
  --group scaling_sdpa_granularity_2 \
  --log_dir scaling_sdpa_granularity_2/hd_256_seed_1340 \
  --n_heads 1 --head_dim 256 --head_dim_value 256 \
  --key hd_256 \
  --random_seed 1340
```

Result: seems like more granularity is worse here. Hrrm, this mismatches my old experiments, remember? Let's check if their loss curves were better or worse than these ones. See [wandb](https://wandb.ai/sesamestrong/context_compression?nw=fvqca8493c9).

Naive guess is that their loss curves were better, and more granular runs were just interacting with my incorrect-head_dim-calculation bug, to make the scale more sane. So naively, more-granular MHA heads actually are worse.

#### MHA granularity scaling experiment

Let's do head_dim=2,8,32,128. Hopefully we get the same results as SDPA.


head_dim=2, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 --ckpt_attn --stabilize_attn \
  --group scaling_mha_granularity_2 \
  --log_dir scaling_mha_granularity_2/hd_2_seed_1339 \
  --n_heads 128 --head_dim 2 --head_dim_value 2 \
  --key hd_2 \
  --random_seed 1339
```

head_dim=8, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 --ckpt_attn --stabilize_attn \
  --group scaling_mha_granularity_2 \
  --log_dir scaling_mha_granularity_2/hd_8_seed_1339 \
  --n_heads 32 --head_dim 8 --head_dim_value 8 \
  --key hd_8 \
  --random_seed 1339
```

head_dim=32, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 --ckpt_attn --stabilize_attn \
  --group scaling_mha_granularity_2 \
  --log_dir scaling_mha_granularity_2/hd_32_seed_1339 \
  --n_heads 8 --head_dim 32 --head_dim_value 32 \
  --key hd_32 \
  --random_seed 1339
```

head_dim=128, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --mup_zero_init --c_proj_scale_init 1.0 --ckpt_attn --stabilize_attn \
  --group scaling_mha_granularity_2 \
  --log_dir scaling_mha_granularity_2/hd_128_seed_1339 \
  --n_heads 2 --head_dim 128 --head_dim_value 128 \
  --key hd_128 \
  --random_seed 1339
```

#### MHA_CONV dense-attention coord check

In the jupyter notebook, seemed to perfectly match behavior when initted to identity.

Let's do a local coord check real quick, comparing to MHA. 8 heads with 32 head_dim.

MHA:

```
for seed in 1339; do
  SKIP_WANDB=false python -m context_compression.train \
    --total_batch_size 8192 --seq_len 256 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores --n_heads 8 \
    --group mha_vanilla_coord_check_4 \
    --mup_enable_coord_check_logging --no_decay_lr --max_steps 20 --no_use_compile --no_upload_to_hf \
    --log_dir mha_vanilla_coord_check_4/mha_seed_$seed \
    --a_producer_kind mha \
    --n_heads 8 \
    --key mha_vanilla_seed_$seed --random_seed $seed
done
```

MHA_CONV:

```
for seed in 1339; do
  SKIP_WANDB=false python -m context_compression.train \
    --total_batch_size 8192 --seq_len 256 --warmup_steps 250 --batch_size 16 --mup --max_lr 30e-4 --head_dim 32 --head_dim_value 32 --n_embd 256 --attention_kind dense --dense_attention_kind mha --mup_zero_init --stabilize_attn_scores --n_heads 8 \
    --group mha_conv_coord_check_4 \
    --mup_enable_coord_check_logging --no_decay_lr --max_steps 20 --no_use_compile --no_upload_to_hf \
    --log_dir mha_conv_coord_check_4/mha_conv_seed_$seed \
    --a_producer_kind mha_conv \
    --n_heads 8 \
    --key mha_conv_seed_$seed --random_seed $seed
done
```

#### MHA_CONV dense-attention pareto-improvement

Let's do another head-granularity scaling experiment, but this time with MHA_CONV.

head_dim=2, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --a_producer_kind mha_conv --mup_zero_init --c_proj_scale_init 1.0 --ckpt_attn --stabilize_attn \
  --group scaling_mha_conv_granularity_2 \
  --log_dir scaling_mha_conv_granularity_2/hd_2_seed_1339 \
  --n_heads 128 --head_dim 2 --head_dim_value 2 \
  --key hd_2 \
  --random_seed 1339
```

head_dim=8, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --a_producer_kind mha_conv --mup_zero_init --c_proj_scale_init 1.0 --ckpt_attn --stabilize_attn \
  --group scaling_mha_conv_granularity_2 \
  --log_dir scaling_mha_conv_granularity_2/hd_8_seed_1339 \
  --n_heads 32 --head_dim 8 --head_dim_value 8 \
  --key hd_8 \
  --random_seed 1339
```

head_dim=32, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --a_producer_kind mha_conv --mup_zero_init --c_proj_scale_init 1.0 --ckpt_attn --stabilize_attn \
  --group scaling_mha_conv_granularity_2 \
  --log_dir scaling_mha_conv_granularity_2/hd_32_seed_1339 \
  --n_heads 8 --head_dim 32 --head_dim_value 32 \
  --key hd_32 \
  --random_seed 1339
```

head_dim=128, seed=1339:

```vast:finished
cd /workspace/context-compression && git pull && torchrun --nproc_per_node=gpu -m context_compression.train \
  --total_batch_size 131072 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 128 --mup --max_lr 30e-4 --n_embd 256 --attention_kind self --dense_attention_kind mha --a_producer_kind mha_conv --mup_zero_init --c_proj_scale_init 1.0 --ckpt_attn --stabilize_attn \
  --group scaling_mha_conv_granularity_2 \
  --log_dir scaling_mha_conv_granularity_2/hd_128_seed_1339 \
  --n_heads 2 --head_dim 128 --head_dim_value 128 \
  --key hd_128 \
  --random_seed 1339
```

Result: it's not much better. And more granular is not better. I wonder if it's init, or there's some inherent source of instability?

I bet it's possible to make it better.