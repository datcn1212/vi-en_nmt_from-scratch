"""Preprocessing for the IWSLT15 en-vi corpus: HTML unescape, NFC normalisation, 
a length filter applied to the train split, SentencePiece BPE trained on the filtered train split only, 
and subword-length stats overevery split.
"""
import argparse
import html
import os
import unicodedata
import numpy as np
import sentencepiece as spm

# Fixed control-symbol ids; whatever wraps this tokenizer later must agree
# with these exact values.
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def clean_lines(path):
    with open(path, encoding="utf-8") as f:
        return [unicodedata.normalize("NFC", html.unescape(line.rstrip("\n"))) for line in f]


def write_lines(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def filter_by_length(src_lines, tgt_lines, max_words):
    keep = [i for i in range(len(src_lines))
            if len(src_lines[i].split()) <= max_words
            and len(tgt_lines[i].split()) <= max_words]
    dropped = len(src_lines) - len(keep)
    print(f"train: dropped {dropped}/{len(src_lines)} pairs longer than {max_words} words "
          f"({100 * dropped / len(src_lines):.2f}%)")
    return [src_lines[i] for i in keep], [tgt_lines[i] for i in keep]


def train_sentencepiece(input_path, model_prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        input=input_path, model_prefix=model_prefix, vocab_size=vocab_size,
        model_type="bpe", pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID, unk_id=UNK_ID,
        # Default 0.9995 silently drops the rarest characters to <unk>, which
        # is unrecoverable on decode - hits capitalized proper-noun letters
        # in this corpus (e.g. "Ả" in "Ả Rập"). 1.0 keeps every character seen.
        character_coverage=1.0,
    )
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix + ".model")
    return sp


def subword_length_stats(lines, sp):
    lengths = np.array([len(sp.encode(line)) + 2 for line in lines])  # +2 for bos/eos
    p50, p90, p99 = np.percentile(lengths, [50, 90, 99])
    return {"p50": p50, "p90": p90, "p99": p99, "max": int(lengths.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_words", type=int, required=True,
                         help="drop train pairs where either side has more whitespace words "
                              "than this - pick from the corpus's own word-length percentiles, "
                              "not a copied default")
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits = [("train", "train"), ("tst2012", "dev"), ("tst2013", "test")]
    for raw_name, out_name in splits:
        vi_lines = clean_lines(os.path.join(args.raw_dir, f"{raw_name}.vi"))
        en_lines = clean_lines(os.path.join(args.raw_dir, f"{raw_name}.en"))
        assert len(vi_lines) == len(en_lines), f"{raw_name}: line count mismatch"

        if out_name == "train":
            vi_lines, en_lines = filter_by_length(vi_lines, en_lines, args.max_words)

        write_lines(vi_lines, os.path.join(args.out_dir, f"{out_name}.vi"))
        write_lines(en_lines, os.path.join(args.out_dir, f"{out_name}.en"))
        print(f"{out_name}: {len(vi_lines)} pairs written")

    # Train only on the filtered train split; dev/test are only ever encoded
    # with this frozen model, never used to (re)train it.
    spm_dir = os.path.join(args.out_dir, "spm")
    os.makedirs(spm_dir, exist_ok=True)
    src_sp = train_sentencepiece(os.path.join(args.out_dir, "train.vi"),
                                  os.path.join(spm_dir, "src_spm"), args.vocab_size)
    tgt_sp = train_sentencepiece(os.path.join(args.out_dir, "train.en"),
                                  os.path.join(spm_dir, "tgt_spm"), args.vocab_size)
    print(f"SentencePiece trained: {spm_dir}/src_spm.model, {spm_dir}/tgt_spm.model")

    report = []
    overall_max = 0
    for _, out_name in splits:
        vi_lines = open(os.path.join(args.out_dir, f"{out_name}.vi"), encoding="utf-8").read().splitlines()
        en_lines = open(os.path.join(args.out_dir, f"{out_name}.en"), encoding="utf-8").read().splitlines()
        report.append(f"=== {out_name} ({len(vi_lines)} lines) ===")
        for lang, lines, sp in [("vi", vi_lines, src_sp), ("en", en_lines, tgt_sp)]:
            s = subword_length_stats(lines, sp)
            overall_max = max(overall_max, s["max"])
            report.append(f"  {lang}: subword length (+bos/eos) p50={s['p50']:.0f} "
                           f"p90={s['p90']:.0f} p99={s['p99']:.0f} max={s['max']}")
    report.append(f"\nmax subword length across all splits: {overall_max}")
    report.append("(lower bound for the model's max sequence length - dev/test are unfiltered, "
                   "so this can exceed the train-side max_words filter)")

    text = "\n".join(report)
    print()
    print(text)
    os.makedirs("results", exist_ok=True)
    with open("results/subword_stats.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
