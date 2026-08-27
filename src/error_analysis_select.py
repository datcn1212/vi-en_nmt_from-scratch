"""Pre-committed selection rule for the report's qualitative error analysis. Compares the 2 best systems
(transformer_warmup_xavier, rnn_xavier) at matched beam-5 decode on test.

Rule, fixed before looking at any translation:
    1. highest min(bleu_a, bleu_b)  -> both systems right
    2. lowest max(bleu_a, bleu_b)   -> both systems wrong
    3. highest |bleu_a - bleu_b|    -> systems disagree most
    4-5. two indices sampled uniformly at random, seed 42
"""
import random

from sacrebleu import sentence_bleu

from data import read_parallel


def load_hyps(name):
    with open(f"results/hyps/{name}.txt") as f:
        return [line.rstrip("\n") for line in f]


def main():
    test_pairs = read_parallel("data/processed/test.vi", "data/processed/test.en")
    refs = [p[1] for p in test_pairs]

    hyp_tf = load_hyps("transformer_warmup_xavier_test_beam")
    hyp_rnn = load_hyps("rnn_xavier_test_beam")

    scores_tf = [sentence_bleu(h, [r]).score for h, r in zip(hyp_tf, refs)]
    scores_rnn = [sentence_bleu(h, [r]).score for h, r in zip(hyp_rnn, refs)]
    n = len(refs)

    both_correct = max(range(n), key=lambda i: min(scores_tf[i], scores_rnn[i]))
    both_wrong = min(range(n), key=lambda i: max(scores_tf[i], scores_rnn[i]))
    differ_most = max(range(n), key=lambda i: abs(scores_tf[i] - scores_rnn[i]))

    random.seed(42)
    random_idx = random.sample(range(n), 2)

    for label, idx in [("both_correct", both_correct), ("both_wrong", both_wrong),
                        ("differ_most", differ_most), ("random_1", random_idx[0]),
                        ("random_2", random_idx[1])]:
        print(f"{label}: idx={idx} bleu_tf={scores_tf[idx]:.2f} bleu_rnn={scores_rnn[idx]:.2f}")


if __name__ == "__main__":
    main()
