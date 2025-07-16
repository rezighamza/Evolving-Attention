# src/search_space/operations.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sparsemax import Sparsemax

# A registry to hold all our operations, mapping string names to classes
OPERATIONS_REGISTRY = {}


# ==============================================================================
# A. SCORING OPERATIONS (How to compare Q and K) - Mostly Binary
# ==============================================================================

class ScaledDotProductOp(nn.Module):
    """The classic: (Q @ K.T) / sqrt(d_k)"""

    def forward(self, q, k):
        d_k = q.size(-1)
        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)


class BilinearOp(nn.Module):
    """Learnable similarity: Q @ W @ K.T"""

    def __init__(self, feature_dim):
        super().__init__()
        self.W = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, q, k):
        return torch.matmul(self.W(q), k.transpose(-2, -1))


class AdditiveAttentionScoreOp(nn.Module):
    """Bahdanau-style attention score: v.T * tanh(W1*q + W2*k)"""

    def __init__(self, feature_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feature_dim
        self.W1 = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, q, k):
        q_proj = self.W1(q).unsqueeze(3)
        k_proj = self.W2(k).unsqueeze(2)
        scores = self.v(torch.tanh(q_proj + k_proj)).squeeze(-1)
        return scores


class ConvolutionalScoreOp(nn.Module):
    """Captures local n-gram like similarity."""

    def __init__(self, feature_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=kernel_size,
            groups=feature_dim,
            padding='same'
        )

    def forward(self, q, k):
        q_flat = q.reshape(-1, q.size(-2), q.size(-1))
        k_flat = k.reshape(-1, k.size(-2), k.size(-1))
        q_conv = self.conv(q_flat.transpose(1, 2))
        k_conv = self.conv(k_flat.transpose(1, 2))
        q_out = q_conv.transpose(1, 2).reshape_as(q)
        k_out = k_conv.transpose(1, 2).reshape_as(k)
        return torch.matmul(q_out, k_out.transpose(-2, -1))


# ==============================================================================
# B. NORMALIZATION & GATING OPERATIONS - Mostly Unary (Robust to extra inputs)
# ==============================================================================

class SoftmaxOp(nn.Module):
    """The standard choice for normalization."""

    def forward(self, *args):
        # Takes only the first input, robust to EA connecting more.
        return F.softmax(args[0], dim=-1)


class SparsemaxOp(nn.Module):
    """Forces sparsity. Many weights will be exactly zero."""

    def __init__(self):
        super().__init__()
        # dim=-1 means it operates on the last dimension (the sequence length)
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, *args):
        return self.sparsemax(args[0])


class SigmoidOp(nn.Module):
    """Creates independent gates [0, 1]. Does NOT sum to 1."""

    def forward(self, *args):
        return torch.sigmoid(args[0])


class TopKOp(nn.Module):
    """A hard-selection mechanism. Zeros out all but the top k scores."""

    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def forward(self, *args):
        x = args[0]
        # Ensure k is not larger than the sequence dimension
        k_actual = min(self.k, x.size(-1))
        if k_actual <= 0: return x

        top_k_vals, _ = torch.topk(x, k=k_actual, dim=-1)
        kth_val = top_k_vals[..., -1].unsqueeze(-1)
        # Create a mask where values less than the k-th value are -inf
        mask = torch.full_like(x, -torch.inf)
        mask[x >= kth_val] = 0
        return x + mask


# ==============================================================================
# C. AGGREGATION & VALUE MANIPULATION OPERATIONS
# ==============================================================================

class WeightedSumOp(nn.Module):
    """The classic aggregation: weights @ V. (Binary)"""

    def forward(self, weights, v):
        return torch.matmul(weights, v)


class GatedValueOp(nn.Module):
    """A gated transformation on V before aggregation: V * sigmoid(Linear(V)). (Unary)"""

    def __init__(self, feature_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, *args):
        v = args[0]
        return v * torch.sigmoid(self.linear(v))


# ==============================================================================
# D. STRUCTURAL & UTILITY OPERATIONS
# ==============================================================================

class AddOp(nn.Module):
    """Adds two tensors element-wise. (Binary)"""

    def forward(self, x1, x2):
        return x1 + x2


class ElementwiseMultiplyOp(nn.Module):
    """Multiplies two tensors element-wise. (Binary)"""

    def forward(self, x1, x2):
        return x1 * x2


class LayerNormOp(nn.Module):
    """Applies Layer Normalization. (Unary)"""

    def __init__(self, feature_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, *args):
        return self.norm(args[0])


class GeluOp(nn.Module):
    """Applies GELU activation. (Unary)"""

    def forward(self, *args):
        return F.gelu(args[0])


class LinearOp(nn.Module):
    """A learnable linear projection. (Unary)"""

    def __init__(self, feature_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, *args):
        return self.linear(args[0])


# ==============================================================================
# REGISTER ALL OPERATIONS
# ==============================================================================

# Scoring
OPERATIONS_REGISTRY['scaled_dot_product'] = ScaledDotProductOp
OPERATIONS_REGISTRY['bilinear'] = BilinearOp
OPERATIONS_REGISTRY['additive_attention_score'] = AdditiveAttentionScoreOp
OPERATIONS_REGISTRY['convolutional_score'] = ConvolutionalScoreOp

# Normalization
OPERATIONS_REGISTRY['softmax'] = SoftmaxOp
OPERATIONS_REGISTRY['sparsemax'] = SparsemaxOp
OPERATIONS_REGISTRY['sigmoid'] = SigmoidOp
OPERATIONS_REGISTRY['topk_filter'] = TopKOp

# Aggregation & Value Manipulation
OPERATIONS_REGISTRY['weighted_sum'] = WeightedSumOp
OPERATIONS_REGISTRY['gated_value'] = GatedValueOp

# Structural (some binary, some unary)
OPERATIONS_REGISTRY['add'] = AddOp
OPERATIONS_REGISTRY['multiply'] = ElementwiseMultiplyOp
OPERATIONS_REGISTRY['layer_norm'] = LayerNormOp
OPERATIONS_REGISTRY['gelu'] = GeluOp
OPERATIONS_REGISTRY['linear'] = LinearOp