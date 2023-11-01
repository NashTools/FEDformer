export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --is_training 1 \
  --root_path /content/drive/MyDrive/training_data/ \
  --data_path BTCUSDT-bookTicker-2023-06-s-50k-processed.csv \
  --task_id Ticker \
  --model FEDformer \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Tr' \
  --itr 3 \
  --do_predict \
  --freq u \
  --detail_freq u
