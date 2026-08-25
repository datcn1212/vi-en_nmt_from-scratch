"""Loss-curve figures for the report, parsed straight from logs/
"""
import re

import matplotlib.pyplot as plt

LOG_RE = re.compile(r"epoch (\d+) train_loss ([\d.]+) dev_loss ([\d.]+)")

PANEL_A = [
    ("logs/train_rnn_baseline.log", "RNN baseline"),
    ("logs/train_rnn_xavier.log", "RNN + Xavier"),
    ("logs/train_transformer_baseline.log", "Transformer, flat lr"),
    ("logs/train_transformer_xavier.log", "Transformer, flat lr + Xavier"),
    ("logs/train_transformer_warmup.log", "Transformer, warmup"),
    ("logs/train_transformer_warmup_xavier.log", "Transformer, warmup + Xavier"),
]

PANEL_B = [
    ("logs/train_transformer_warmup_xavier.log", "12 epochs"),
    ("logs/train_transformer_warmup_xavier_24ep.log", "24 epochs"),
]


def parse_dev_loss(log_path):
    epochs, dev_losses = [], []
    with open(log_path) as f:
        for line in f:
            m = LOG_RE.match(line)
            if m:
                epochs.append(int(m.group(1)))
                dev_losses.append(float(m.group(3)))
    return epochs, dev_losses


def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    for log_path, label in PANEL_A:
        epochs, dev_losses = parse_dev_loss(log_path)
        ax_a.plot(epochs, dev_losses, marker="o", markersize=3, label=label)
    ax_a.set_xlabel("epoch")
    ax_a.set_ylabel("dev loss")
    ax_a.set_title("6 configurations, 12 epochs each")
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)

    for log_path, label in PANEL_B:
        epochs, dev_losses = parse_dev_loss(log_path)
        ax_b.plot(epochs, dev_losses, marker="o", markersize=3, label=label)
    ax_b.set_xlabel("epoch")
    ax_b.set_ylabel("dev loss")
    ax_b.set_title("best config, budget extension (not an architecture comparison)")
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/loss_curves.png", dpi=150)
    print("wrote results/loss_curves.png")


if __name__ == "__main__":
    main()
