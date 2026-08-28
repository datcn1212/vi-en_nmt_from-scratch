# Evaluation protocol

Locked in before any results exist

- Comparing configurations (hyperparameters, architectures, attention variants, anything) is judged on the dev set only.
- The test set is scored exactly once, at the end, after everything else is already decided. Precisely: each system gets exactly one test score, produced after the frozen list below was committed. Decoding is deterministic, so where a system was decoded more than once (the ensemble script re-scores each member alongside the ensemble) the runs return byte-identical output and the same single score - repeated compute, never a second chance at a number.
- Whenever two models are compared, they use the same decoding settings (same beam width, same max length).

## Decisions log

- Vocab size fixed at 8000, SentencePiece BPE, one model per language, trained on the filtered train split only, then frozen (dev/test only ever encoded with it, never retrained).
- Decode settings frozen: beam_width=5, max_len=50, length_penalty=0.0, detokenize with the target SentencePiece model's own decode() (never join subwords by hand). Same settings for every architecture whenever they're compared.

## Frozen system list for the one-time test pass

Decided and committed before any test-set score exists for any of these systems.

Architecture comparison - beam-5 and greedy each:
- checkpoints/rnn_xavier (best RNN)
- checkpoints/transformer_warmup_xavier (best Transformer, seed 42)

Baseline reference - beam-5 only, to check whether the largest dev effect found (Xavier init) is still significant on held-out data:
- checkpoints/rnn_baseline
- checkpoints/transformer_baseline

Ensemble - beam-5, both combination rules, to check whether the smallest dev effect found (majority vs consensus) is significant; scores every member (transformer_warmup_xavier, transformer_warmup_xavier_seed123, rnn_xavier) alongside the ensemble in the same run:
- 3-member ensemble (transformer_warmup_xavier + transformer_warmup_xavier_seed123 + rnn_xavier), majority
- same 3 members, consensus

Every other trained checkpoint (rnn_luong, rnn_luong_scaled, rnn_xavier_dropout, transformer_xavier, transformer_warmup, transformer_embedding_fix, transformer_warmup_xavier_24ep) stays dev-only.
