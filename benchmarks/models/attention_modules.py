import torch
import torch.nn as nn
import math
from src.search_space.symbolic_graph import SymbolicAttention


class StandardAttention(nn.Module):
    """ The classic scaled dot-product attention. """

    def __init__(self, graph_def=None, d_model=None):  # Args for API compatibility
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        weights = self.softmax(scores)
        return torch.matmul(weights, v)


# We can re-use our SymbolicAttention for the discovered candidate
DiscoveredAttention = SymbolicAttention