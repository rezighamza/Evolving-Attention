# benchmarks/models/text_transformer.py

import torch
import torch.nn as nn
import math
from .vit import BenchmarkEncoderLayer  # We can reuse the same encoder layer!


class PositionalEncoding(nn.Module):
    """ Standard positional encoding for Transformers. """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TextTransformer(nn.Module):
    """ A Transformer Encoder model for text classification. """

    def __init__(self, n_tokens, d_model, n_head, dim_feedforward, n_layers,
                 attention_module_class, attention_graph_def=None, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.layers = nn.ModuleList([
            BenchmarkEncoderLayer(
                d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward,
                dropout=dropout,
                # d_head is d_model // n_head, which matches the ViT implementation
                attention_module=attention_module_class(graph_def=attention_graph_def, d_model=d_model // n_head)
            ) for _ in range(n_layers)
        ])

        self.d_model = d_model
        # We classify using the output of the first token (like a CLS token)
        self.classifier = nn.Linear(d_model, 1)  # Binary classification (output is a logit)

    def forward(self, src, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # The underlying attention modules in BenchmarkEncoderLayer expect no mask,
        # so we will pass the tensors directly. For a full implementation, you'd pass the mask.
        for layer in self.layers:
            src = layer(src)

        # Use the first token's output for classification
        output = self.classifier(src[0, :, :])
        return output.squeeze(-1)