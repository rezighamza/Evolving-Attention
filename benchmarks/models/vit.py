# benchmarks/models/vit.py

import torch
import torch.nn as nn


class BenchmarkEncoderLayer(nn.Module):
    """
    A standard Transformer Encoder Layer, now projection-aware to match the co-evolutionary search.
    """

    def __init__(self, d_model, n_head, dim_feedforward, dropout, attention_module_class,
                 attention_graph_def, proj_config):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_head = d_model // n_head

        # Instantiate the attention module with its graph and head dimension
        self.self_attn = attention_module_class(graph_def=attention_graph_def, d_model=self.d_head)

        # --- UPDATED: Conditionally create learnable projection layers ---
        self.Wq = nn.Linear(d_model, d_model, bias=False) if proj_config['has_wq'] else nn.Identity()
        self.Wk = nn.Linear(d_model, d_model, bias=False) if proj_config['has_wk'] else nn.Identity()
        self.Wv = nn.Linear(d_model, d_model, bias=False) if proj_config['has_wv'] else nn.Identity()
        self.Wo = nn.Linear(d_model, d_model) if proj_config['has_wo'] else nn.Identity()

        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        x_norm = self.norm1(src)
        batch_size, seq_len, _ = x_norm.shape

        q = self.Wq(x_norm).reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = self.Wk(x_norm).reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = self.Wv(x_norm).reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        attn_output = self.self_attn(q, k, v)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.Wo(attn_output)

        src = src + self.dropout(attn_output)

        x_norm = self.norm2(src)
        ffn_output = self.ffn(x_norm)
        src = src + self.dropout(ffn_output)

        return src


class VisionTransformer(nn.Module):
    """ Vision Transformer updated to accept projection configurations. """

    def __init__(self, attention_module_class, attention_graph_def, proj_config,
                 img_size=32, patch_size=4, d_model=256, n_layers=4, n_head=4,
                 dim_feedforward=512, n_classes=100, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)

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

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, patches), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x[:, 0]))