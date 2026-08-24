"""Decode with several models at once, combining their per-step distributions.
Only requires each model to have encode() and decode_step(), so members can
mix architectures freely.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from decode import _score


def _ensemble_log_probs(models, memory_list, src_pad_mask, tgt_so_far, combine="majority"):
    log_probs_per_model = [
        torch.log_softmax(model.decode_step(memory, src_pad_mask, tgt_so_far), dim=-1)
        for model, memory in zip(models, memory_list)
    ]
    stacked = torch.stack(log_probs_per_model, dim=0)  # [M, B, V]

    if combine == "majority":
        # Arithmetic mean of probabilities - mean of distributions that each sum
        # to 1 still sums to 1, so this is a properly normalized distribution.
        probs = stacked.exp().mean(dim=0)
        return probs.clamp_min(1e-12).log()
    if combine == "consensus":
        # Geometric mean of probabilities, computed as a mean of log-probs to
        # match exactly and avoid underflow. Not itself a normalized
        # distribution (the missing normalizer is a genuine gap), but that
        # normalizer is the same constant for every token at this step, so it
        # cannot change the argmax or the top-k ranking used below.
        return stacked.mean(dim=0)
    raise ValueError(f"unknown combine: {combine}")


def ensemble_greedy_decode(models, src, src_pad_mask, sos_id=1, eos_id=2, max_len=50, combine="majority"):
    with torch.no_grad():
        memory_list = [model.encode(src, src_pad_mask) for model in models]
        tgt = [sos_id]
        result = []
        for _ in range(max_len):
            log_probs = _ensemble_log_probs(models, memory_list, src_pad_mask, torch.tensor([tgt]), combine)
            next_id = log_probs.argmax(dim=-1).item()
            result.append(next_id)
            tgt.append(next_id)
            if next_id == eos_id:
                break
    return result


def ensemble_beam_search_decode(models, src, src_pad_mask, sos_id=1, eos_id=2, max_len=50,
                                 beam_width=5, length_penalty=0.0, combine="majority"):
    with torch.no_grad():
        memory_list = [model.encode(src, src_pad_mask) for model in models]
        beams = [([sos_id], 0.0)]
        finished = []

        for _ in range(max_len):
            if not beams:
                break
            candidates = []
            for seq, cum_logprob in beams:
                log_probs = _ensemble_log_probs(models, memory_list, src_pad_mask, torch.tensor([seq]), combine)
                top_logprobs, top_ids = log_probs.squeeze(0).topk(beam_width)
                for lp, tok_id in zip(top_logprobs.tolist(), top_ids.tolist()):
                    candidates.append((seq + [tok_id], cum_logprob + lp))

            candidates.sort(key=lambda c: _score(c[0], c[1], length_penalty), reverse=True)
            candidates = candidates[:beam_width]

            beams = []
            for seq, cum_logprob in candidates:
                if seq[-1] == eos_id:
                    finished.append((seq, cum_logprob))
                else:
                    beams.append((seq, cum_logprob))

        # Same rule as single-model beam search: never select from finished
        # alone, for the same reason (UC6).
        pool = finished + beams
        best_seq, _ = max(pool, key=lambda c: _score(c[0], c[1], length_penalty))
    return best_seq[1:]
