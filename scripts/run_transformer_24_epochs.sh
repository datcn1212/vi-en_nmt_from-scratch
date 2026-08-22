#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
source .venv/bin/activate
DATA=data/processed

# convergence check - best factorial config (warmup + Xavier), 24 epochs
# instead of 12, nothing else changed. Never goes into the architecture
# comparison table (budget mismatch), only into its own labeled convergence panel.
python src/train.py --arch transformer \
  --train_src $DATA/train.vi --train_tgt $DATA/train.en \
  --dev_src $DATA/dev.vi --dev_tgt $DATA/dev.en \
  --src_spm $DATA/spm/src_spm.model --tgt_spm $DATA/spm/tgt_spm.model \
  --epochs 24 --batch_size 32 --warmup_steps 4000 --xavier_init \
  --save_dir checkpoints/transformer_warmup_xavier_24ep \
  > logs/train_transformer_warmup_xavier_24ep.log 2>&1