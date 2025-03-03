#!/bin/bash

# ok so we have enough vram per gpu to fit two concurrent runs on it.
# so we'll do that.
# so the CUDA_VISIBLE_DEVICES=floor($RANK/2)

if [ -z "$RANK" ] || [ -z "$WORLD_SIZE" ] || [ -z "$NNODES" ] || [ -z "$NODE_RANK" ]; then
  echo "RANK and WORLD_SIZE and NNODES and NODE_RANK must be defined"
  exit 1
fi

i=0
for n_heads in 4; do
    for attention_kind in selective; do
        for lr in 3.25e-5 3.5e-5 3.75e-5 4e-5; do
            for total_batch_size in 30720; do
                batch_size=$((total_batch_size / 256))
                for seed in 1338 1339 1340 1341 1342 1343 1344 1345; do
                    i=$((i + 1))
                    # we gotta be on the right node
                    if [ $((i % $NNODES)) -ne $NODE_RANK ]; then
                        continue
                    fi
                    j=$((i / $NNODES))
                    # we ALSO gotta be on the right gpu
                    if [ $((j % WORLD_SIZE)) -ne $RANK ]; then
                        continue
                    fi
                    out_dir="wider_is_better_9/attention_kind${attention_kind}_n_heads${n_heads}_seed${seed}"
                    # Use CUDA_DEVICE=0 since after setting CUDA_VISIBLE_DEVICES, the only visible device is 0
                    HALF_RANK=$((RANK % 8))
                    { CUDA_VISIBLE_DEVICES=$HALF_RANK WORLD_SIZE=1 LOCAL_RANK=0 RANK=0 python -m context_compression.train --group wider_is_better_9 \
                    --log_dir $out_dir \
                    --max_lr $lr \
                    --total_batch_size $total_batch_size \
                    --seq_len 256 \
                    --max_steps 17500 \
                    --warmup_steps 200 \
                    --batch_size $batch_size \
                    --mup \
                    --n_heads $n_heads \
                    --key ${lr}_${total_batch_size}_${n_heads}_${seed} $([ "$attention_kind" = "selective" ] || echo "--disable_selection") \
                    --random_seed $seed; }
                done
            done
        done
    done
done

# result number 0: [See here](https://wandb.ai/sesamestrong/context_compression?nw=pe07md4nhw).

# For selection=false, wider *IS* better.
# But for selection=true, wider is worse.

# selection=false seems to ALWAYS be worse than selection=true!!
# I think this means my training is very selection-unfriendly.
# And definitely not representative of the real benefits of selection.

# result number 1 (letting all of them run for longer):
# TODO finish this!!!

echo 'done'

# result number 2 (selective, total bs=10240):
# we def have wider-is-better - nice! but not with perfect lr mup transfer.
# for n_heads=2, lr=5e-5 is best
# for n_heads=4, lr=3e-5 is best
# for n_heads=12, lr=2e-5 is better than lr=5e-5. not sure abt the resolution.

# but we gotta improve efficiency - increase total bs as much as I can.
# will try doing it for n_heads=4 rn, hopefully those best bs's can transfer to n_heads=12.