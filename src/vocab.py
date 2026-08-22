"""Wraps a trained SentencePiece model with the project's fixed control-symbol ids and add/strip-special-token logic.
"""
import sentencepiece as spm

PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class Vocab:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text, add_special=True):
        ids = self.sp.encode(text, out_type=int)
        if add_special:
            return [SOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids):
        # sos/eos/pad carry no content; unk is left in
        ids = [i for i in ids if i not in (PAD_ID, SOS_ID, EOS_ID)]
        return self.sp.decode(ids)

    def __len__(self):
        return self.sp.get_piece_size()
