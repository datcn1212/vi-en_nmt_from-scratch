"""Cheap checks on a batch of decoded hypotheses, and cheap checks on a model
before committing to another multi-hour training run.
"""
import math

import torch
import torch.nn as nn


def degeneracy_rate(hyps):
    degenerate = sum(1 for h in hyps if len(h) <= 2)
    return degenerate / len(hyps)

def truncation_rate(model, srcs, src_pad_masks, sos_id, eos_id, max_len, decode_fn):
    truncated = 0
    for src, mask in zip(srcs, src_pad_masks):
        hyp = decode_fn(model, src, mask, sos_id=sos_id, eos_id=eos_id, max_len=max_len)
        if len(hyp) == 0 or hyp[-1] != eos_id:
            truncated += 1
    return truncated / len(srcs)


def repetition_rate(hyps):
    # hyps: detokenized strings, same as evaluate.py already produces.
    repetitive = 0
    for hyp in hyps:
        tokens = hyp.split()
        counts = {}
        for i in range(len(tokens) - 2):
            trigram = tuple(tokens[i:i + 3])
            counts[trigram] = counts.get(trigram, 0) + 1
        if any(c >= 3 for c in counts.values()):
            repetitive += 1
    return repetitive / len(hyps)


def epoch1_dev_loss(log_path):
    with open(log_path, encoding="utf-8") as f:
        first_line = f.readline()
    parts = first_line.split()
    return float(parts[parts.index("dev_loss") + 1])


def compare_epoch1_loss(rnn_log_path, transformer_log_path):
    return {
        "rnn": epoch1_dev_loss(rnn_log_path),
        "transformer": epoch1_dev_loss(transformer_log_path),
    }


def gradient_norm_stats(model, loader, criterion, optimizer, n_steps=100, max_norm=1.0):
    norms = []
    step = 0
    for batch in loader:
        if step >= n_steps:
            break
        optimizer.zero_grad()
        logits = model(batch["src"], batch["src_pad_mask"], batch["tgt_in"])
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1))
        loss.backward()
        # clip_grad_norm_ returns the norm as it was BEFORE clipping, even though
        # it also performs the clip - one call gives both the measurement and the
        # normal training step's own clipping behavior.
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()
        optimizer.step()
        norms.append(total_norm)
        step += 1
    clipped = sum(1 for n in norms if n > max_norm)
    return {
        "mean_grad_norm": sum(norms) / len(norms),
        "clip_fraction": clipped / len(norms),
        "n_steps": len(norms),
    }


def logit_scale_at_init(model, src, src_pad_mask, tgt_in):
    with torch.no_grad():
        logits = model(src, src_pad_mask, tgt_in)
    return logits.std().item()


def embedding_norm_ratio(model):
    # ||token embedding * sqrt(d_model)|| versus ||positional encoding||, averaged
    # per position/token rather than over the whole matrix - vocab_size and max_len
    # differ, so a whole-matrix norm would compare table sizes, not signal amplitude.
    emb_norms = (model.src_embedding.weight * math.sqrt(model.d_model)).norm(dim=-1)
    pe_norms = model.pos_encoding.pe.norm(dim=-1)
    return (emb_norms.mean() / pe_norms.mean()).item()


def member_agreement(hyps_a, hyps_b):
    # Fraction of sentences where two ensemble members produce the exact same
    # hypothesis - a cheap proxy for how much diversity they actually offer.
    agree = sum(1 for a, b in zip(hyps_a, hyps_b) if a == b)
    return agree / len(hyps_a)
