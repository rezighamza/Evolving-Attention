# benchmarks/models/text_transformer.py

import torch
import torch.nn as nn
import math
from .vit import BenchmarkEncoderLayer  # Reuses the updated, projection-aware encoder!


class PositionalEncoding(nn.Module):
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
    """ A Transformer Encoder model for text, now projection-aware. """

    def __init__(self, n_tokens, d_model, n_head, dim_feedforward, n_layers,
                 attention_module_class, attention_graph_def, proj_config, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # --- UPDATED: Pass the full configuration to the encoder layer ---
        self.layers = nn.ModuleList([
            BenchmarkEncoderLayer(
                d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_module_class=attention_module_class,
                attention_graph_def=attention_graph_def,
                proj_config=proj_config
            ) for _ in range(n_layers)
        ])

        self.d_model = d_model
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, src, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        for layer in self.layers:
            src = layer(src)

        output = self.classifier(src[0, :, :])
        return output.squeeze(-1)