"""Training loop: teacher-forced train/dev epochs, checkpointing on best dev
loss. Shared across architectures; --arch only changes which model gets built.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import read_parallel, TranslationDataset, collate_fn
from init_utils import scale_embedding_init
from rnn_model import RNNSeq2Seq
from transformer_model import TransformerSeq2Seq
from vocab import PAD_ID, Vocab


def build_model(arch, src_vocab_size, tgt_vocab_size, xavier_init=False, attention_type="bahdanau",
                 luong_scale=False, dropout=0.1):
    if arch == "rnn":
        return RNNSeq2Seq(src_vocab_size, tgt_vocab_size, pad_id=PAD_ID, xavier_init=xavier_init,
                           attention_type=attention_type, luong_scale=luong_scale, dropout=dropout)
    if arch == "transformer":
        return TransformerSeq2Seq(src_vocab_size, tgt_vocab_size, pad_id=PAD_ID, xavier_init=xavier_init,
                                   dropout=dropout)
    raise ValueError(f"unknown arch: {arch}")


def noam_lr_lambda(step, d_model, warmup_steps):
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n_batches = 0.0, 0
    for batch in loader:
        src = batch["src"].to(device)
        src_pad_mask = batch["src_pad_mask"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(src, src_pad_mask, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["rnn", "transformer"])
    parser.add_argument("--train_src", required=True)
    parser.add_argument("--train_tgt", required=True)
    parser.add_argument("--dev_src", required=True)
    parser.add_argument("--dev_tgt", required=True)
    parser.add_argument("--src_spm", required=True)
    parser.add_argument("--tgt_spm", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None,
                         help="truncate train/dev to this many pairs, for a quick pipeline check")
    parser.add_argument("--warmup_steps", type=int, default=0,
                         help="Noam warmup, only takes effect when > 0 and --arch transformer")
    parser.add_argument("--xavier_init", action="store_true")
    parser.add_argument("--attention_type", default="bahdanau", choices=["bahdanau", "luong"],
                         help="only affects --arch rnn")
    parser.add_argument("--luong_scale", action="store_true",
                         help="divide the luong bilinear score by sqrt(hidden_dim); only affects --attention_type luong")
    parser.add_argument("--fix_embedding_init", action="store_true",
                         help="re-draw embeddings from N(0, 1/d_model) instead of the nn.Embedding "
                              "default N(0,1); only meaningful for --arch transformer, and meant to be "
                              "used alone (not with --xavier_init) to isolate this one change")
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    lr = args.lr if args.lr is not None else (1e-3 if args.arch == "rnn" else 3e-4)

    src_vocab = Vocab(args.src_spm)
    tgt_vocab = Vocab(args.tgt_spm)

    train_pairs = read_parallel(args.train_src, args.train_tgt)
    dev_pairs = read_parallel(args.dev_src, args.dev_tgt)
    if args.max_examples is not None:
        train_pairs = train_pairs[:args.max_examples]
        dev_pairs = dev_pairs[:args.max_examples]

    train_loader = DataLoader(
        TranslationDataset(train_pairs, src_vocab, tgt_vocab),
        batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, PAD_ID),
    )
    dev_loader = DataLoader(
        TranslationDataset(dev_pairs, src_vocab, tgt_vocab),
        batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, PAD_ID),
    )

    model = build_model(args.arch, len(src_vocab), len(tgt_vocab), xavier_init=args.xavier_init,
                         attention_type=args.attention_type, luong_scale=args.luong_scale,
                         dropout=args.dropout)
    if args.fix_embedding_init:
        assert args.arch == "transformer", "--fix_embedding_init only makes sense for --arch transformer"
        scale_embedding_init(model.src_embedding, model.d_model)
        scale_embedding_init(model.tgt_embedding, model.d_model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    use_warmup = args.warmup_steps > 0 and args.arch == "transformer"
    if use_warmup:
        # lr=1.0 here is not a multiplier - noam_lr_lambda already returns the full
        # learning rate value, so Adam's own lr just has to be 1 to pass it through
        # unscaled. The easiest line in this file to misread.
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        d_model = model.d_model
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: noam_lr_lambda(step, d_model, args.warmup_steps)
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None

    os.makedirs(args.save_dir, exist_ok=True)
    hyperparams = {
        "arch": args.arch, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": lr, "seed": args.seed, "warmup_steps": args.warmup_steps, "xavier_init": args.xavier_init,
        "attention_type": args.attention_type, "luong_scale": args.luong_scale,
        "fix_embedding_init": args.fix_embedding_init, "dropout": args.dropout,
    }

    best_dev_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scheduler=scheduler)
        dev_loss = run_epoch(model, dev_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch} train_loss {train_loss:.4f} dev_loss {dev_loss:.4f} lr {current_lr:.6f}",
              flush=True)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "arch": args.arch,
                "hyperparams": hyperparams,
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
                "pad_id": PAD_ID,
            }, os.path.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()
