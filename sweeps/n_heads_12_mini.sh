#!/bin/bash

if [ -z "$RANK" ]; then
  export RANK=0
fi
if [ -z "$WORLD_SIZE" ]; then
  export WORLD_SIZE=1
fi

GPUS_PER_RUN=2

i=0
for lr in 14e-4 10e-4 12e-4 20e-4; do
    for total_batch_size in 61440; do
        batch_size=$((total_batch_size / 256 / GPUS_PER_RUN / 2))
        for seed in 1338 1339; do
            i=$((i + 1))
            # we ALSO gotta be on the right gpu
            if [ $((i % WORLD_SIZE)) -ne $RANK ]; then
                continue
            fi
            out_dir="wider_is_better_11/n_heads12_lr${lr}_total_batch_size${total_batch_size}_seed${seed}"
            # Use CUDA_DEVICE=0 since after setting CUDA_VISIBLE_DEVICES, the only visible device is 0
            DOUBLE_RANK=$((RANK * GPUS_PER_RUN))
            DOUBLE_RANK_PLUS_ONE=$((DOUBLE_RANK + 1))
            TORCHRUN_PORT=$((13345 + RANK))
            { CUDA_VISIBLE_DEVICES=$DOUBLE_RANK,$DOUBLE_RANK_PLUS_ONE torchrun --nproc_per_node=$GPUS_PER_RUN --master_port $TORCHRUN_PORT -m context_compression.train --group wider_is_better_11 \
            --log_dir $out_dir \
            --max_lr $lr \
            --total_batch_size $total_batch_size \
            --seq_len 256 \
            --max_steps 8750 \
            --warmup_steps 500 \
            --batch_size $batch_size \
            --mup \
            --n_heads 12 \
            --key n_heads12_${lr}_${total_batch_size} \
            --random_seed $seed; }
        done
    done
done

echo 'done'
