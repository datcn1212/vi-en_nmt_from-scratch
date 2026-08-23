#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
source .venv/bin/activate
DATA=data/processed

# Same as run_rnn_luong.sh (Xavier + Luong bilinear) but with the score divided
# by sqrt(hidden_dim) - conditional follow-up run, only justified because the
# unscaled run's real attention entropy collapsed to ~0.07 (near one-hot),
# confirmed against the trained checkpoint, not assumed.
python src/train.py --arch rnn \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --epochs 12 --batch_size 32 --xavier_init --attention_type luong --luong_scale \
  --save_dir checkpoints/rnn_luong_scaled \
  > logs/train_rnn_luong_scaled.log 2>&1
