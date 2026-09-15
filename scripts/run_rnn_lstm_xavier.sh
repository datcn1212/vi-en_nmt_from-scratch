#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
source .venv/bin/activate
DATA=data/processed

# RNN with LSTMCell instead of GRUCell - one flag changed from checkpoints/rnn_xavier,
# everything else (12 epochs, batch 32, lr 1e-3, seed 42, dropout 0.1, Xavier init,
# Bahdanau attention) held fixed for a clean single-axis comparison.
python src/train.py --arch rnn \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --epochs 12 --batch_size 32 --xavier_init --cell_type lstm \
  --save_dir checkpoints/rnn_lstm_xavier \
  > logs/train_rnn_lstm_xavier.log 2>&1
