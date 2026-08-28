# Vi-En NMT from Scratch

Vietnamese to English neural machine translation, two architectures implemented from scratch in plain PyTorch: an RNN/GRU encoder-decoder with attention (Bahdanau and Luong variants), and a Transformer encoder-decoder with multi-head self-attention and sinusoidal positional encoding. No `torch.nn.Transformer`, no `torch.nn.MultiheadAttention`, no pretrained weights.

## Data

IWSLT15 English-Vietnamese (TED talk transcripts): 133,317 raw train pairs, 133,222 after a length filter applied to the train split only, plus 1553 dev pairs and 1268 test pairs left unfiltered. SentencePiece BPE, vocab size 8000, one model per language, trained on the filtered train split only. All text normalized to Unicode NFC before tokenization.

## Results (test set, 1268 sentences, official one-time pass)

| system | decode | BLEU |
|---|---|---|
| 3-member ensemble, consensus | beam-5 | 24.30 |
| 3-member ensemble, majority | beam-5 | 24.02 |
| Transformer (warmup + Xavier init) | beam-5 | 21.99 |
| Transformer (warmup + Xavier init) | greedy | 21.14 |
| RNN + Bahdanau (Xavier init) | beam-5 | 20.28 |
| RNN + Bahdanau (Xavier init) | greedy | 18.64 |
| RNN baseline (default init) | beam-5 | 18.37 |
| Transformer baseline (default init, flat lr) | beam-5 | 8.76 |

The frozen decode settings and the list of systems scored on test are in `EVAL_PROTOCOL.md`.

## Repo layout

```
src/            model code, training loop, decoding, evaluation, diagnostics
ensemble/       ensemble decoding and evaluation (majority voting, consensus building)
scripts/        one shell script per training run
tests/          pytest suite (correctness gates: overfit tests, masking, autoregression)
data/           raw and preprocessed data, tokenizers, fixed dev/test index files
checkpoints/    trained model weights (git-ignored; reproducible from scripts/ and logs/)
logs/           training logs, one per run
results/        BLEU scores, hypotheses, figures, significance tests, error analysis
report/         write-up and generated tables
```

## Running it

```
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_data.sh     # data/raw/{train,tst2012,tst2013}.{en,vi}
python src/prepare_iwslt.py --raw_dir data/raw --out_dir data/processed \
  --max_words <chosen from data_stats.txt percentiles, not a fixed default> \
  --vocab_size 8000
python src/train.py --arch rnn --xavier_init ...
python src/train.py --arch transformer --warmup_steps 4000 --xavier_init ...
python src/evaluate.py --checkpoint <ckpt> --split test --decode beam --beam_width 5 ...
python -m pytest tests/
```

See each script's `--help` for the full flag list; `scripts/*.sh` are the exact commands used for every reported result.
