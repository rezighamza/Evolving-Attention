# src/evaluation/proxy_tasks.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from src.search_space.symbolic_graph import SymbolicAttention


class CustomEncoderLayer(nn.Module):
    """A simplified Transformer Encoder Layer that uses our SymbolicAttention."""

    def __init__(self, d_model, n_head, dim_feedforward, dropout, chromosome):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.proj_config = chromosome['proj_config']

        # The symbolic attention part remains the same
        self.self_attn = SymbolicAttention(chromosome['graph_def'], self.d_head)

        # --- Conditionally create projection layers ---
        self.Wq = nn.Linear(d_model, d_model) if self.proj_config['has_wq'] else nn.Identity()
        self.Wk = nn.Linear(d_model, d_model) if self.proj_config['has_wk'] else nn.Identity()
        self.Wv = nn.Linear(d_model, d_model) if self.proj_config['has_wv'] else nn.Identity()
        self.Wo = nn.Linear(d_model, d_model) if self.proj_config['has_wo'] else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        x_norm = self.norm1(src)
        batch_size, seq_len, _ = x_norm.shape

        # Apply projections (or identity)
        q = self.Wq(x_norm).reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = self.Wk(x_norm).reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = self.Wv(x_norm).reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        attn_output = self.self_attn(q, k, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        # Apply output projection (or identity)
        attn_output = self.Wo(attn_output)
        src = src + self.dropout(attn_output)

        ffn_output = self.ffn(self.norm2(src))
        src = src + self.dropout(ffn_output)
        return src


class ProxyVisionTransformer(nn.Module):
    """The Proxy Model: A tiny Vision Transformer for CIFAR-10."""
    def __init__(self, chromosome, img_size=32, patch_size=4, d_model=64, n_classes=10):
        super().__init__()
        self.d_model = d_model
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding now accepts 3 input channels for color images
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        self.encoder = CustomEncoderLayer(
            d_model=d_model, n_head=1, dim_feedforward=d_model * 4,
            dropout=0.1, chromosome=chromosome
        )

        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, patches), dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        cls_output = x[:, 0]
        return self.head(cls_output)


def run_proxy_evaluation(model, device, num_epochs=3):
    """
    Trains and evaluates a model on a small subset of CIFAR-10 for a few epochs.
    This provides a more robust fitness signal than single-batch training.
    """
    model.to(device)

    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])

    # Use a larger subset for more stable training
    train_dataset = Subset(CIFAR10(root='./data', train=True, download=True, transform=transform), range(4096))
    val_dataset = Subset(CIFAR10(root='./data', train=False, download=True, transform=transform), range(2048))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop for a few full epochs
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Check for non-finite loss which indicates instability
            if not torch.isfinite(loss):
                return 0.0 # Return worst score if unstable
            loss.backward()
            optimizer.step()

    # Final evaluation on the validation set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0: return 0.0
    return correct / total