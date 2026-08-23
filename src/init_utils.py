"""Standalone re-initialization helpers, applied to an already-built model
from the outside. Never imported by rnn_model.py or transformer_model.py
themselves, so an experiment that uses one can be added or dropped without
touching either model file.
"""
import torch.nn as nn


def scale_embedding_init(embedding, d_model):
    # nn.Embedding defaults to N(0, 1). TransformerSeq2Seq later multiplies the
    # embedding by sqrt(d_model), which only lands at the positional encoding's
    # own ~1 amplitude if the embedding itself has variance ~1/d_model going in.
    # Re-drawing from N(0, 1/d_model) here fixes just that one thing, leaving
    # every other weight (attention, FFN) at its default init.
    std = d_model ** -0.5
    nn.init.normal_(embedding.weight, mean=0.0, std=std)
