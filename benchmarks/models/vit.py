import torch
import torch.nn as nn


class BenchmarkEncoderLayer(nn.Module):
    """ A standard Transformer Encoder Layer. It's given a complete attention module. """

    def __init__(self, d_model, n_head, dim_feedforward, dropout, attention_module):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.self_attn = attention_module  # Plug in the attention mechanism

        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_norm = self.norm1(src)
        batch_size, seq_len, _ = src_norm.shape
        d_head = self.d_model // self.n_head

        # Create Q, K, V from the same input source, then reshape for multi-head
        # In a full implementation, these would be learnable projections.
        # For a fair comparison of the core mechanism, we can share them.
        q = src_norm.reshape(batch_size, seq_len, self.n_head, d_head).transpose(1, 2)
        k = src_norm.reshape(batch_size, seq_len, self.n_head, d_head).transpose(1, 2)
        v = src_norm.reshape(batch_size, seq_len, self.n_head, d_head).transpose(1, 2)

        attn_output = self.self_attn(q, k, v)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        src = src + self.dropout(attn_output)
        src = src + self.dropout(self.ffn(self.norm2(src)))
        return src


class VisionTransformer(nn.Module):
    """ Vision Transformer for benchmarking, simplified for plug-and-play attention. """

    def __init__(self, attention_module_class, attention_graph_def=None,
                 img_size=32, patch_size=4, d_model=256, n_layers=4, n_head=4,
                 dim_feedforward=512, n_classes=100, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        d_head = d_model // n_head

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            BenchmarkEncoderLayer(
                d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_module=attention_module_class(graph_def=attention_graph_def, d_model=d_head)
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