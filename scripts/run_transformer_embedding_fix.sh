#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
source .venv/bin/activate
DATA=data/processed

# Isolates the embedding/positional-encoding scale fix from full --xavier_init:
# flat lr, default init everywhere except the embeddings, which are re-drawn
# from N(0, 1/d_model) instead of nn.Embedding's default N(0,1). Compares
# directly against checkpoints/transformer_baseline (same flat lr, no fix at
# all) and checkpoints/transformer_xavier (same flat lr, everything Xavier'd).
python src/train.py --arch transformer \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --epochs 12 --batch_size 32 --fix_embedding_init \
  --save_dir checkpoints/transformer_embedding_fix \
  > logs/train_transformer_embedding_fix.log 2>&1
