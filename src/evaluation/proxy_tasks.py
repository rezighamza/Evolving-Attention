# src/evaluation/proxy_tasks.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
# --- CHANGE 1: Import CIFAR10 and different transforms ---
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from src.search_space.symbolic_graph import SymbolicAttention

# ... (CustomEncoderLayer class remains the same) ...
class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout, graph_def):
        super().__init__()
        self.self_attn = SymbolicAttention(graph_def, d_model)
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
        src_norm = self.norm1(src)
        attn_output = self.self_attn(src_norm, src_norm, src_norm)
        src = src + self.dropout(attn_output)
        ffn_output = self.ffn(self.norm2(src))
        src = src + self.dropout(ffn_output)
        return src

# --- The Proxy Model: A tiny Vision Transformer (ViT) ---
class ProxyVisionTransformer(nn.Module):
    # --- CHANGE 2: Default img_size to 32 for CIFAR-10 ---
    def __init__(self, graph_def, img_size=32, patch_size=4, d_model=64, n_classes=10):
        super().__init__()
        self.d_model = d_model
        num_patches = (img_size // patch_size) ** 2

        # --- CHANGE 3: Accept 3 input channels for color images ---
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        self.encoder = CustomEncoderLayer(
            d_model=d_model, n_head=1, dim_feedforward=d_model * 4,
            dropout=0.1, graph_def=graph_def
        )
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        patches = self.patch_embed(x)
        patches = patches.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, patches), dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        cls_output = x[:, 0]
        return self.head(cls_output)

# --- CHANGE 4: The evaluation function now uses CIFAR-10 ---
def run_proxy_evaluation(model, device, train_steps=200, eval_steps=50):
    """
    Trains and evaluates a model on a small subset of CIFAR-10 for speed.
    """
    model.to(device)

    # Use standard CIFAR-10 transforms
    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])

    # Use a small, fixed subset of CIFAR-10
    train_dataset = Subset(CIFAR10(root='./data', train=True, download=True, transform=transform), range(2048))
    val_dataset = Subset(CIFAR10(root='./data', train=False, download=True, transform=transform), range(1024))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    step = 0
    train_iter = iter(train_loader)
    while step < train_steps:
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        step += 1

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        eval_iter = iter(val_loader)
        for i in range(eval_steps):
            try:
                images, labels = next(eval_iter)
            except StopIteration:
                break # Stop if validation set is exhausted
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0: return 0.0
    return correct / total