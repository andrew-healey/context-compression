```
python -m context_compression.train \
--max_lr 30e-4 --total_batch_size 4096 --seq_len 256 --max_steps 4375 --warmup_steps 250 --batch_size 4 --mup --n_heads 12 --head_dim 22 \
--log_dir /tmp/dummy \
--selection_head_linear_combo n_latent_masks \
--n_latent_masks 2 \
--latent_mask_precision float32
```