import torch
from fvcore.nn import FlopCountAnalysis

from src.evaluation.proxy_tasks import ProxyVisionTransformer, run_proxy_evaluation


class FitnessCalculator:
    """
    Calculates the multi-objective fitness of a candidate attention graph.
    Fitness = (Performance, -Efficiency)
    """

    def __init__(self, d_model=64, img_size=28, patch_size=7, device='cpu'):
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.device = device
        print(f"FitnessCalculator initialized on device: '{self.device}'")

    def calculate_fitness(self, graph_def):
        """
        Takes a graph definition and returns its fitness tuple.
        Returns a very poor fitness score if any error occurs.
        """
        try:
            # --- 1. Calculate Efficiency (FLOPs) ---
            # Instantiate only the attention module for FLOPs calculation
            attention_module = ProxyVisionTransformer(graph_def, d_model=self.d_model).encoder.self_attn

            # Create dummy inputs
            seq_len = (self.img_size // self.patch_size) ** 2 + 1
            dummy_q = torch.randn(1, seq_len, self.d_model)
            dummy_k = torch.randn(1, seq_len, self.d_model)
            dummy_v = torch.randn(1, seq_len, self.d_model)

            # Analyze FLOPs
            flops = FlopCountAnalysis(attention_module, (dummy_q, dummy_k, dummy_v)).total()

            # --- 2. Calculate Performance (Accuracy) ---
            # Instantiate the full proxy model
            proxy_model = ProxyVisionTransformer(
                graph_def=graph_def,
                d_model=self.d_model,
                img_size=self.img_size,
                patch_size=self.patch_size
            )

            accuracy = run_proxy_evaluation(proxy_model, self.device)

            # Return the fitness tuple. We want to maximize both.
            # So, we return accuracy and NEGATIVE flops.
            # We scale flops to be on a similar order of magnitude to accuracy for stability.
            return (accuracy, -flops / 1e6)  # Flops in Millions

        except Exception as e:
            # Any failure during instantiation, FLOPs count, or training
            # means the candidate is invalid or unstable.
            print(f"Evaluation failed for a candidate: {e}")
            # Return the worst possible score
            return (0.0, -float('inf'))