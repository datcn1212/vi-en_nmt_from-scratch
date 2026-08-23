#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
source .venv/bin/activate
DATA=data/processed

# RNN + Xavier + Luong (bilinear) attention - built on top of the RNN+Xavier
# result: only the attention score function changes, everything else
# (Xavier init, 12 epochs, batch 32, lr) stays fixed, so any difference from
# rnn_xavier isolates the scoring function itself.
python src/train.py --arch rnn \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --epochs 12 --batch_size 32 --xavier_init --attention_type luong \
  --save_dir checkpoints/rnn_luong \
  > logs/train_rnn_luong.log 2>&1
