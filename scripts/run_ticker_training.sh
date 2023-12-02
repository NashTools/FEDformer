export CUDA_VISIBLE_DEVICES=0,1

python -u run.py \
  --is_training 1 \
  --root_path /content/drive/MyDrive/training_data/ \
  --task_id Ticker \
  --model FEDformer \
  --data multi \
  --data_prefix trial \
  --features MS \
  --seq_len 128 \
  --label_len 64 \
  --pred_len 64 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des Tr \
  --itr 1 \
  --freq u \
  --detail_freq u \
  --use_gpu True \
  --d_ff 64 \
  --d_model 64 \
  --target log_return_s \
  --modes 32 \
  --use_amp

#   --root_path /content/drive/MyDrive/training_data/ \