#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
source .venv/bin/activate
DATA=data/processed

# Second seed of the best config (Transformer, warmup + Xavier) - only --seed
# changes, everything else identical to checkpoints/transformer_warmup_xavier.
# Needed for a same-architecture ensemble pair.
python src/train.py --arch transformer \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --epochs 12 --batch_size 32 --warmup_steps 4000 --xavier_init --seed 123 \
  --save_dir checkpoints/transformer_warmup_xavier_seed123 \
  > logs/train_transformer_warmup_xavier_seed123.log 2>&1
