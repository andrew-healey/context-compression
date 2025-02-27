for lr in 1e-4 2e-4 4e-4 6e-4 8e-4 1e-3 1.5e-3 2e-3 3e-3; do
  for total_batch_size in 5120 10240 20480 40960; do
    for seq_len in 128 256 512 1024; do
        out_dir="proxy_model_sweep/lr${lr}_total_batch_size${total_batch_size}_seq_len${seq_len}"
        python -m context_compression.train --group proxy_model_sweep \
          --log_dir $out_dir \
          --max_lr $lr \
          --total_batch_size $total_batch_size \
          --seq_len $seq_len \
          --mup \
          --disable_selection \
          --n_heads 2
    done
  done
done
