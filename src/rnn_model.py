"""RNN encoder-decoder building blocks."""
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or hidden_dim
        self.W1 = nn.Linear(hidden_dim, attn_dim)
        self.W2 = nn.Linear(2 * hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_pad_mask):
        # decoder_hidden: [B, H]
        # encoder_outputs: [B, S, 2H]
        # src_pad_mask: [B, S] bool (True = pad)
        query = self.W1(decoder_hidden).unsqueeze(1)
        keys = self.W2(encoder_outputs)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)

        scores = scores.masked_fill(src_pad_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, alpha


class LuongAttention(nn.Module):
    """Bilinear ("general") score: score(q, k) = q^T W k. Mask and softmax are
    identical to BahdanauAttention - only the score formula differs.

    scale=True divides by sqrt(hidden_dim): unlike Bahdanau's tanh, the bilinear
    form has nothing capping the raw score's magnitude, so on trained weights it
    saturates softmax to near one-hot (measured: raw score std ~88, max softmax
    prob ~0.97, entropy ~0.07, versus Bahdanau's std ~6.6, prob ~0.51, entropy
    ~1.4 on the same real batch) - same mechanism as scaled dot-product attention.
    """
    def __init__(self, hidden_dim, scale=False):
        super().__init__()
        self.W = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** 0.5 if scale else None

    def forward(self, decoder_hidden, encoder_outputs, src_pad_mask):
        scores = torch.bmm(self.W(encoder_outputs), decoder_hidden.unsqueeze(2)).squeeze(2)
        if self.scale is not None:
            scores = scores / self.scale

        scores = scores.masked_fill(src_pad_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, alpha


class RNNSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=128, hidden_dim=256,
                 pad_id=0, dropout=0.1, xavier_init=False, attention_type="bahdanau",
                 luong_scale=False):
        super().__init__()
        assert attention_type in ("bahdanau", "luong"), f"unknown attention_type: {attention_type}"

        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        self.encoder_gru = nn.GRU(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.enc_to_dec = nn.Linear(2 * hidden_dim, hidden_dim)

        if attention_type == "bahdanau":
            self.attention = BahdanauAttention(hidden_dim)
        else:
            self.attention = LuongAttention(hidden_dim, scale=luong_scale)
        self.decoder_cell = nn.GRUCell(emb_dim + 2 * hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim + 2 * hidden_dim, tgt_vocab_size)

        if xavier_init:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def encode(self, src, src_pad_mask):
        embedded = self.dropout(self.src_embedding(src))
        encoder_outputs, h_n = self.encoder_gru(embedded)
        # h_n: [2, B, H], index 0 = forward direction, index 1 = backward (PyTorch convention).
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)
        hidden = torch.tanh(self.enc_to_dec(h_cat))
        return encoder_outputs, hidden

    def _step(self, tgt_token_ids, hidden, encoder_outputs, src_pad_mask):
        embedded = self.dropout(self.tgt_embedding(tgt_token_ids))
        context, _ = self.attention(hidden, encoder_outputs, src_pad_mask)
        hidden = self.decoder_cell(torch.cat([embedded, context], dim=-1), hidden)
        logits = self.output_layer(torch.cat([hidden, context], dim=-1))
        return logits, hidden

    def forward(self, src, src_pad_mask, tgt_in):
        encoder_outputs, hidden = self.encode(src, src_pad_mask)
        all_logits = []
        for t in range(tgt_in.size(1)):
            logits, hidden = self._step(tgt_in[:, t], hidden, encoder_outputs, src_pad_mask)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=1)

    def decode_step(self, memory, src_pad_mask, tgt_so_far):
        # Stateless from the caller's side: re-unrolls from the initial hidden state
        # through the full prefix every call, so this matches whatever interface a
        # Transformer's decode_step exposes and greedy/beam search can stay
        # architecture-agnostic.
        encoder_outputs, hidden = memory
        for t in range(tgt_so_far.size(1)):
            logits, hidden = self._step(tgt_so_far[:, t], hidden, encoder_outputs, src_pad_mask)
        return logits
