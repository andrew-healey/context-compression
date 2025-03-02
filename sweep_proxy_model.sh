for attention_kind in selective; do
  for lr in 1.75e-4 2e-4; do
    for total_batch_size in 5120; do
      for seq_len in 256; do
        for decay_lr in true false; do
          for warmup_steps in 1 200; do
            for seed in 1338 1339; do
              out_dir="proxy_model_sweep_2/lr${lr}_total_batch_size${total_batch_size}_seq_len${seq_len}_decay_lr${decay_lr}_attention_kind${attention_kind}_warmup_steps${warmup_steps}_seed${seed}"
              python -m context_compression.train --group proxy_model_sweep_2 \
                --log_dir $out_dir \
                --max_lr $lr \
                --total_batch_size $total_batch_size \
                --seq_len $seq_len \
                --max_steps 5000 \
                --warmup_steps $warmup_steps \
                --batch_size 20 \
                --mup \
                $([ "$attention_kind" = "self" ] && echo "--disable_selection") \
                $([ "$decay_lr" = "false" ] && echo "--no_decay_lr") \
                --n_heads 2 \
                --random_seed $seed
            done
          done
        done
      done
    done
  done
done
