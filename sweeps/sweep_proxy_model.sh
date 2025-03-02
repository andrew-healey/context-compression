#!/bin/bash

# Use: SKIP_WANDB=false torchrun --nproc_per_node=1 sweeps/run_sweep.py

# for attention_kind in selective; do
#   for lr in 1.75e-4 2e-4; do
#     for total_batch_size in 5120; do
#       for seq_len in 256; do
#         for decay_lr in true false; do
#           for warmup_steps in 1 200; do
#             for seed in 1338 1339; do
#               out_dir="proxy_model_sweep_2/lr${lr}_total_batch_size${total_batch_size}_seq_len${seq_len}_decay_lr${decay_lr}_attention_kind${attention_kind}_warmup_steps${warmup_steps}_seed${seed}"
#               python -m context_compression.train --group proxy_model_sweep_2 \
#                 --log_dir $out_dir \
#                 --max_lr $lr \
#                 --total_batch_size $total_batch_size \
#                 --seq_len $seq_len \
#                 --max_steps 5000 \
#                 --warmup_steps $warmup_steps \
#                 --batch_size 20 \
#                 --mup \
#                 $([ "$attention_kind" = "self" ] && echo "--disable_selection") \
#                 $([ "$decay_lr" = "false" ] && echo "--no_decay_lr") \
#                 --n_heads 2 \
#                 --random_seed $seed
#             done
#           done
#         done
#       done
#     done
#   done
# done

# results: See "partial results of HP search". Seems like 200 warmup steps with decay_lr=true is best.
# And seems like lr=1.75e-4 is a little better than lr=2e-4.
# This was done on a 4070Ti, and we're moving up to a 4090, so the optimal bs will prob be slightly different.
# So I won't tune the lr much more - I'll tune the bs first.

# assert that the RANK and WORLD_SIZE env vars are defined
if [ -z "$RANK" ] || [ -z "$WORLD_SIZE" ]; then
  echo "RANK and WORLD_SIZE must be defined"
  exit 1
fi

i=0
for lr in 1.5e-4 1.75e-4 2e-4; do
  for total_batch_size in 5120 10240; do
    for n_heads in 2 4; do
      for seed in 1338 1339 1340; do
        i=$((i + 1))
        if [ $((i % $WORLD_SIZE)) -ne $RANK ]; then
          continue
        fi
        out_dir="proxy_model_sweep_3/lr${lr}_total_batch_size${total_batch_size}_n_heads${n_heads}_seed${seed}"
        # Use CUDA_DEVICE=0 since after setting CUDA_VISIBLE_DEVICES, the only visible device is 0
        CUDA_VISIBLE_DEVICES=$RANK WORLD_SIZE=1 LOCAL_RANK=0 RANK=0 python -m context_compression.train --group proxy_model_sweep_3 \
          --log_dir $out_dir \
          --max_lr $lr \
          --total_batch_size $total_batch_size \
          --seq_len 256 \
          --max_steps 5000 \
          --warmup_steps 200 \
          --batch_size 20 \
          --mup \
          --n_heads $n_heads \
          --key ${lr}_${total_batch_size}_${n_heads}_${seed} \
          --random_seed $seed
      done
    done
  done
done

echo 'done'