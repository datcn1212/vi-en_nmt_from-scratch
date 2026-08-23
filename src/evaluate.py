"""Score a trained checkpoint on dev or test: decode, detokenize with the
tokenizer's own decode(), score with sacrebleu, write a results json and a
hypotheses txt file.
"""
import argparse
import datetime
import json
import os

import torch
from sacrebleu import BLEU

from data import read_parallel
from decode import beam_search_decode, greedy_decode
from rnn_model import RNNSeq2Seq
from transformer_model import TransformerSeq2Seq
from vocab import EOS_ID, SOS_ID, Vocab

DATA_DIR = "data/processed"
SPM_DIR = os.path.join(DATA_DIR, "spm")


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt["arch"] == "rnn":
        attention_type = ckpt["hyperparams"].get("attention_type", "bahdanau")
        luong_scale = ckpt["hyperparams"].get("luong_scale", False)
        model = RNNSeq2Seq(ckpt["src_vocab_size"], ckpt["tgt_vocab_size"], pad_id=ckpt["pad_id"],
                            attention_type=attention_type, luong_scale=luong_scale)
    elif ckpt["arch"] == "transformer":
        model = TransformerSeq2Seq(ckpt["src_vocab_size"], ckpt["tgt_vocab_size"], pad_id=ckpt["pad_id"])
    else:
        raise ValueError(f"unknown arch in checkpoint: {ckpt['arch']}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="use only the first N pairs")
    parser.add_argument("--indices_file", default=None,
                         help="one 0-based line index per line - for a fixed, reproducible subset "
                              "like dev500 (random sample, not just a prefix)")
    parser.add_argument("--decode", required=True, choices=["greedy", "beam"])
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--system_name", required=True)
    args = parser.parse_args()

    src_vocab = Vocab(os.path.join(SPM_DIR, "src_spm.model"))
    tgt_vocab = Vocab(os.path.join(SPM_DIR, "tgt_spm.model"))
    model = load_model(args.checkpoint)

    pairs = read_parallel(os.path.join(DATA_DIR, f"{args.split}.vi"), os.path.join(DATA_DIR, f"{args.split}.en"))
    if args.indices_file is not None:
        with open(args.indices_file, encoding="utf-8") as f:
            indices = [int(line) for line in f]
        pairs = [pairs[i] for i in indices]
    elif args.limit is not None:
        pairs = pairs[:args.limit]

    hyps = []
    for src_text, _ in pairs:
        src = torch.tensor([src_vocab.encode(src_text)])
        mask = torch.zeros_like(src, dtype=torch.bool)  # batch of 1: nothing to pad
        if args.decode == "greedy":
            hyp_ids = greedy_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID, max_len=args.max_len)
        else:
            hyp_ids = beam_search_decode(model, src, mask, sos_id=SOS_ID, eos_id=EOS_ID,
                                          max_len=args.max_len, beam_width=args.beam_width,
                                          length_penalty=args.length_penalty)
        hyps.append(tgt_vocab.decode(hyp_ids))

    refs = [tgt for _, tgt in pairs]
    bleu_metric = BLEU()
    score = bleu_metric.corpus_score(hyps, [refs])

    os.makedirs("results/hyps", exist_ok=True)
    hyps_path = f"results/hyps/{args.system_name}_{args.split}_{args.decode}.txt"
    with open(hyps_path, "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(h + "\n")

    record = {
        "system": args.system_name,
        "split": args.split,
        "decode": args.decode,
        "beam_width": args.beam_width if args.decode == "beam" else None,
        "bleu": score.score,
        "signature": str(bleu_metric.get_signature()),
        "n_sentences": len(pairs),
        "checkpoint": args.checkpoint,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    result_path = f"results/{args.system_name}_{args.split}_{args.decode}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"BLEU: {score.score:.2f}")
    print(f"wrote {result_path}")
    print(f"wrote {hyps_path}")


if __name__ == "__main__":
    main()
