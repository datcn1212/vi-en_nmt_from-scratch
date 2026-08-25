# Paired bootstrap significance

- architecture: best Transformer vs best RNN: transformer_warmup_xavier_test_beam 21.99 (+/-0.93) vs rnn_xavier_test_beam 20.28 (+/-0.92), p=0.0010
- Xavier init effect (largest effect found): transformer_warmup_xavier_test_beam 21.99 (+/-0.93) vs transformer_baseline_test_beam 8.76 (+/-0.58), p=0.0010
- ensemble combine rule (smallest effect found): ens_3member_consensus_test_beam 24.30 (+/-1.01) vs ens_3member_majority_test_beam 24.02 (+/-1.01), p=0.0669
- seed 42 vs seed 123 (training-seed noise floor): transformer_warmup_xavier_test_beam 21.99 (+/-0.93) vs transformer_warmup_xavier_seed123_test_beam 21.93 (+/-0.97), p=0.3367
