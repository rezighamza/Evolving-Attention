# src/evaluation/fitness.py

import torch
from fvcore.nn import FlopCountAnalysis

from src.evaluation.proxy_tasks import ProxyVisionTransformer, run_proxy_evaluation
from src.search_space.symbolic_graph import SymbolicAttention

# Define sets of operations by their function for the validator
SCORING_OPS = {'scaled_dot_product', 'bilinear', 'additive_attention_score', 'convolutional_score'}
NORMALIZATION_OPS = {'softmax', 'sparsemax', 'sigmoid'}
AGGREGATION_OPS = {'weighted_sum'}
VALUE_TRANSFORM_OPS = {'linear', 'gelu', 'layer_norm', 'gated_value'}  # Ops that can validly transform V


class FitnessCalculator:
    """
    Calculates the multi-objective fitness of a candidate attention graph.
    Includes a very strict validator to ensure the graph is semantically correct.
    """

    def __init__(self, d_model=64, img_size=32, patch_size=4, device='cpu'):
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.device = device
        print(f"FitnessCalculator initialized with FINAL STRICT validator on CIFAR-10 proxy task.")

    def _trace_input_ancestry(self, start_node_idx, graph_def):
        """Traces the primary input source of a node back to its origin ('q', 'k', 'v', or an op type)."""
        current_idx = start_node_idx
        visited_indices = {current_idx}

        while isinstance(current_idx, int):
            # Check for cycles, which are invalid.
            if len(visited_indices) > len(graph_def): return 'cycle_error'

            provider_node = graph_def[current_idx]
            # We trace the first input as the primary ancestor
            if not provider_node['inputs']: return 'no_input_error'

            current_idx = provider_node['inputs'][0]
            if isinstance(current_idx, int):
                if current_idx in visited_indices: return 'cycle_error'
                visited_indices.add(current_idx)

        return current_idx  # Will be 'q', 'k', or 'v'

    def _is_graph_valid(self, graph_def):
        """
        Performs a strict semantic check on the graph's structure by validating
        the final aggregation node and tracing its inputs.
        """
        if not graph_def: return False

        # --- Rule 1: The final node must be a valid aggregation op. ---
        final_node = graph_def[-1]
        if final_node['op'] not in AGGREGATION_OPS:
            return False

        # --- Rule 2: The final aggregation op must have exactly two inputs. ---
        if len(final_node['inputs']) != 2:
            return False

        # --- Rule 3: Validate the 'weights' input branch (Score -> Normalize -> Aggregate) ---
        weight_provider_idx = final_node['inputs'][0]
        if not isinstance(weight_provider_idx, int): return False  # Weights must come from a node

        norm_node = graph_def[weight_provider_idx]
        if norm_node['op'] not in NORMALIZATION_OPS: return False  # Must be a normalization op
        if len(norm_node['inputs']) != 1: return False  # Normalization should have one input

        score_provider_idx = norm_node['inputs'][0]
        if not isinstance(score_provider_idx, int): return False  # Normalized value must come from a node

        score_node = graph_def[score_provider_idx]
        if score_node['op'] not in SCORING_OPS: return False  # Must be a scoring op

        # --- Rule 4: Validate the 'values' input branch (Should connect back to V) ---
        value_provider_idx = final_node['inputs'][1]
        # The value branch can be complex, so we trace its ultimate origin.
        # It can be 'v' directly, or a node that transforms 'v'.
        if isinstance(value_provider_idx, int):
            value_origin = self._trace_input_ancestry(value_provider_idx, graph_def)
            if value_origin != 'v':
                return False
        elif isinstance(value_provider_idx, str):
            if value_provider_idx != 'v':
                return False
        else:
            return False

        return True

    def calculate_fitness(self, graph_def):
        """
        Takes a graph definition and returns its fitness tuple.
        """
        if not self._is_graph_valid(graph_def):
            return (0.0, -float('inf'))

        try:
            attn_module_for_flops = SymbolicAttention(graph_def, d_model=self.d_model)
            seq_len = (self.img_size // self.patch_size) ** 2 + 1
            dummy_q = torch.randn(1, seq_len, self.d_model)
            dummy_k = torch.randn(1, seq_len, self.d_model)
            dummy_v = torch.randn(1, seq_len, self.d_model)
            flops = FlopCountAnalysis(attn_module_for_flops, (dummy_q, dummy_k, dummy_v)).total()

            proxy_model = ProxyVisionTransformer(
                graph_def=graph_def, d_model=self.d_model,
                img_size=self.img_size, patch_size=self.patch_size
            )
            accuracy = run_proxy_evaluation(proxy_model, self.device)

            return (accuracy, -flops / 1e6)

        except Exception:
            return (0.0, -float('inf'))