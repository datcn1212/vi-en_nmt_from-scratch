"""Paired bootstrap significance testing between pairs of already-decoded
systems, using sacrebleu's own resampler on the saved hypothesis files.
"""
import argparse

from sacrebleu import BLEU
from sacrebleu.significance import PairedTest

from data import read_parallel

COMPARISONS = [
    ("transformer_warmup_xavier_test_beam", "rnn_xavier_test_beam",
     "architecture: best Transformer vs best RNN"),
    ("transformer_warmup_xavier_test_beam", "transformer_baseline_test_beam",
     "Xavier init effect (largest effect found)"),
    ("ens_3member_consensus_test_beam", "ens_3member_majority_test_beam",
     "ensemble combine rule (smallest effect found)"),
    ("transformer_warmup_xavier_test_beam", "transformer_warmup_xavier_seed123_test_beam",
     "seed 42 vs seed 123 (training-seed noise floor)"),
]


def load_hyps(name):
    with open(f"results/hyps/{name}.txt") as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    pairs = read_parallel(f"data/processed/{args.split}.vi", f"data/processed/{args.split}.en")
    refs = [p[1] for p in pairs]

    lines = ["# Paired bootstrap significance", ""]
    for name_a, name_b, label in COMPARISONS:
        hyps_a = load_hyps(name_a)
        hyps_b = load_hyps(name_b)
        named_systems = [(name_a, hyps_a), (name_b, hyps_b)]
        pt = PairedTest(named_systems, {"bleu": BLEU()}, [refs], test_type="bs",
                         n_samples=args.n_samples)
        _, results = pt()
        result_a, result_b = results["BLEU"]
        line = (f"{label}: {name_a} {result_a.score:.2f} (+/-{result_a.ci:.2f}) vs "
                f"{name_b} {result_b.score:.2f} (+/-{result_b.ci:.2f}), p={result_b.p_value:.4f}")
        print(line)
        lines.append(f"- {line}")

    with open("results/significance.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results/significance.md")


if __name__ == "__main__":
    main()
