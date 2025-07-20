# src/evaluation/fitness.py

import torch
from fvcore.nn import FlopCountAnalysis

from src.evaluation.proxy_tasks import ProxyVisionTransformer
from src.evaluation.proxy_tasks import run_proxy_evaluation
from src.search_space.symbolic_graph import SymbolicAttention

# Define sets of operations by their function for the validator
SCORING_OPS = {'scaled_dot_product', 'bilinear', 'additive_attention_score', 'convolutional_score'}
NORMALIZATION_OPS = {'softmax', 'sparsemax', 'sigmoid'}
AGGREGATION_OPS = {'weighted_sum'}
# --- NEW: Define ops that are ALLOWED to be on the value path ---
ALLOWED_VALUE_OPS = {'linear', 'gelu', 'layer_norm', 'gated_value', 'add', 'multiply'}


class FitnessCalculator:
    """
    Calculates fitness with a definitive, path-aware semantic validator.
    """

    def __init__(self, d_model=64, img_size=32, patch_size=4, device='cpu'):
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.device = device
        print(f"FitnessCalculator initialized with DEFINITIVE PATH-AWARE validator.")

    def _is_value_branch_valid(self, start_node_idx, graph_def):
        """
        Recursively checks if the entire branch leading to this node is valid for a 'value' stream.
        It ensures no scoring or normalization ops are on this path.
        """
        if not isinstance(start_node_idx, int):
            return start_node_idx == 'v'  # The branch must originate from 'v'

        node = graph_def[start_node_idx]
        op = node['op']

        # If this operation is invalid for a value stream, fail immediately.
        if op not in ALLOWED_VALUE_OPS:
            return False

        # Recursively check all inputs to this node
        for inp in node.get('inputs', []):
            if not self._is_value_branch_valid(inp, graph_def):
                return False

        return True

    def _is_graph_valid(self, graph_def):
        """Performs the final, strict semantic check on the graph's structure."""
        if not graph_def: return False

        final_node = graph_def[-1]
        if final_node['op'] not in AGGREGATION_OPS: return False
        if len(final_node['inputs']) != 2: return False

        # --- Validate the 'weights' branch (must be Score -> Normalize) ---
        weight_provider_idx = final_node['inputs'][0]
        if not isinstance(weight_provider_idx, int): return False

        norm_node = graph_def[weight_provider_idx]
        if norm_node['op'] not in NORMALIZATION_OPS: return False
        if len(norm_node['inputs']) != 1: return False

        score_provider_idx = norm_node['inputs'][0]
        if not isinstance(score_provider_idx, int): return False

        score_node = graph_def[score_provider_idx]
        if score_node['op'] not in SCORING_OPS: return False

        # --- Validate the 'values' branch using the new path-aware function ---
        value_provider_idx = final_node['inputs'][1]
        if not self._is_value_branch_valid(value_provider_idx, graph_def):
            return False

        return True

    def calculate_fitness(self, chromosome):
        graph_def = chromosome['graph_def']
        if not self._is_graph_valid(graph_def):
            return (0.0, -float('inf'))

        try:
            proxy_model = ProxyVisionTransformer(
                chromosome=chromosome, d_model=self.d_model,
                img_size=self.img_size, patch_size=self.patch_size
            )
            encoder_layer = proxy_model.encoder
            seq_len = (self.img_size // self.patch_size) ** 2 + 1
            dummy_encoder_input = torch.randn(1, seq_len, self.d_model)
            flops = FlopCountAnalysis(encoder_layer.to(self.device), (dummy_encoder_input.to(self.device),)).total()
            accuracy = run_proxy_evaluation(proxy_model, self.device)
            return (accuracy, -flops / 1e6)
        except Exception:
            return (0.0, -float('inf'))