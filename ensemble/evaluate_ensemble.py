"""Score an ensemble of checkpoints - any mix of architectures, each just
needs encode() and decode_step() - together with every individual member, in
one run so the comparison uses the same split and decode settings throughout.
"""
import argparse
import datetime
import json
import os
import sys

import torch
from sacrebleu import BLEU

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import read_parallel
from decode import beam_search_decode, greedy_decode
from evaluate import load_model
from vocab import EOS_ID, SOS_ID, Vocab

from decode_ensemble import ensemble_beam_search_decode, ensemble_greedy_decode

DATA_DIR = "data/processed"
SPM_DIR = os.path.join(DATA_DIR, "spm")


def _member_name(checkpoint_path):
    return os.path.basename(os.path.dirname(checkpoint_path))


def _write_result(system_name, split, decode, hyps, bleu_score, checkpoints, extra):
    os.makedirs("results/hyps", exist_ok=True)
    hyps_path = f"results/hyps/{system_name}_{split}_{decode}.txt"
    with open(hyps_path, "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(h + "\n")

    record = {
        "system": system_name, "split": split, "decode": decode,
        "bleu": bleu_score.score, "signature": str(bleu_score.get_signature())
        if hasattr(bleu_score, "get_signature") else None,
        "n_sentences": len(hyps), "checkpoints": checkpoints,
        "timestamp": datetime.datetime.now().isoformat(),
        **extra,
    }
    result_path = f"results/{system_name}_{split}_{decode}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return result_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--combine", required=True, choices=["majority", "consensus"])
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--indices_file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decode", required=True, choices=["greedy", "beam"])
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--system_name", required=True)
    args = parser.parse_args()

    src_vocab = Vocab(os.path.join(SPM_DIR, "src_spm.model"))
    tgt_vocab = Vocab(os.path.join(SPM_DIR, "tgt_spm.model"))
    models = [load_model(c) for c in args.checkpoints]

    pairs = read_parallel(os.path.join(DATA_DIR, f"{args.split}.vi"), os.path.join(DATA_DIR, f"{args.split}.en"))
    if args.indices_file is not None:
        with open(args.indices_file, encoding="utf-8") as f:
            indices = [int(line) for line in f]
        pairs = [pairs[i] for i in indices]
    elif args.limit is not None:
        pairs = pairs[:args.limit]
    refs = [tgt for _, tgt in pairs]

    beam_width = args.beam_width if args.decode == "beam" else None

    def decode_one(model, src, mask):
        if args.decode == "greedy":
            return greedy_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID, max_len=args.max_len)
        return beam_search_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID, max_len=args.max_len,
                                   beam_width=args.beam_width, length_penalty=args.length_penalty)

    def decode_ensemble(src, mask):
        if args.decode == "greedy":
            return ensemble_greedy_decode(models, src, mask, sos_id=SOS_ID, eos_id=EOS_ID,
                                           max_len=args.max_len, combine=args.combine)
        return ensemble_beam_search_decode(models, src, mask, sos_id=SOS_ID, eos_id=EOS_ID,
                                            max_len=args.max_len, beam_width=args.beam_width,
                                            length_penalty=args.length_penalty, combine=args.combine)

    print(f"BLEU (each member, {args.decode}):")
    member_hyps = []
    for ckpt, model in zip(args.checkpoints, models):
        hyps = []
        for src_text, _ in pairs:
            src = torch.tensor([src_vocab.encode(src_text)])
            mask = torch.zeros_like(src, dtype=torch.bool)
            hyps.append(tgt_vocab.decode(decode_one(model, src, mask)))
        member_hyps.append(hyps)
        bleu_score = BLEU().corpus_score(hyps, [refs])
        name = _member_name(ckpt)
        result_path = _write_result(name, args.split, args.decode, hyps, bleu_score, [ckpt],
                                     {"beam_width": beam_width})
        print(f"  {name}: {bleu_score.score:.2f}  (wrote {result_path})")

    ensemble_hyps = []
    for src_text, _ in pairs:
        src = torch.tensor([src_vocab.encode(src_text)])
        mask = torch.zeros_like(src, dtype=torch.bool)
        ensemble_hyps.append(tgt_vocab.decode(decode_ensemble(src, mask)))
    ensemble_bleu = BLEU().corpus_score(ensemble_hyps, [refs])
    result_path = _write_result(args.system_name, args.split, args.decode, ensemble_hyps, ensemble_bleu,
                                 args.checkpoints, {"beam_width": beam_width, "combine": args.combine})
    print(f"BLEU (ensemble, {args.combine}): {ensemble_bleu.score:.2f}  (wrote {result_path})")


if __name__ == "__main__":
    main()
