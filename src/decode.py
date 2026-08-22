"""Shared decoding: only calls model.encode() and model.decode_step(), so
every architecture is decoded the same way.
"""
import torch


def greedy_decode(model, src, src_pad_mask, sos_id=1, eos_id=2, max_len=50):
    with torch.no_grad():
        memory = model.encode(src, src_pad_mask)
        tgt = [sos_id]
        result = []
        for _ in range(max_len):
            logits = model.decode_step(memory, src_pad_mask, torch.tensor([tgt]))
            next_id = logits.argmax(dim=-1).item()
            result.append(next_id)
            tgt.append(next_id)
            if next_id == eos_id:
                break
    return result


def _score(seq, cum_logprob, length_penalty):
    if length_penalty > 0:
        return cum_logprob / (len(seq) ** length_penalty)
    return cum_logprob


def beam_search_decode(model, src, src_pad_mask, sos_id=1, eos_id=2, max_len=50,
                        beam_width=5, length_penalty=0.0):
    with torch.no_grad():
        memory = model.encode(src, src_pad_mask)
        beams = [([sos_id], 0.0)]
        finished = []

        for _ in range(max_len):
            if not beams:
                break
            candidates = []
            for seq, cum_logprob in beams:
                logits = model.decode_step(memory, src_pad_mask, torch.tensor([seq]))
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                top_logprobs, top_ids = log_probs.topk(beam_width)
                for logp, tok_id in zip(top_logprobs.tolist(), top_ids.tolist()):
                    candidates.append((seq + [tok_id], cum_logprob + logp))

            candidates.sort(key=lambda c: _score(c[0], c[1], length_penalty), reverse=True)
            candidates = candidates[:beam_width]

            beams = []
            for seq, cum_logprob in candidates:
                if seq[-1] == eos_id:
                    finished.append((seq, cum_logprob))
                else:
                    beams.append((seq, cum_logprob))

        # Never select from finished alone: cumulative log-prob is negative and
        # strictly decreases with length, so a beam that emitted eos immediately
        # would always beat any longer
        pool = finished + beams
        best_seq, _ = max(pool, key=lambda c: _score(c[0], c[1], length_penalty))
    return best_seq[1:]
