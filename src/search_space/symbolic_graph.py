import torch
import torch.nn as nn
from .operations import OPERATIONS_REGISTRY


class SymbolicAttention(nn.Module):
    """
    A PyTorch module that dynamically builds an attention mechanism
    from a symbolic graph definition.
    """

    def __init__(self, graph_def, d_model):
        """
        Args:
            graph_def (list of dict): The "chromosome" defining the graph.
                Each dict is a "gene" with keys "op", "inputs".
                e.g., [{'op': 'scaled_dot_product', 'inputs': [0, 1]}]
            d_model (int): The feature dimension of the model (e.g., 512).
        """
        super().__init__()
        self.graph_def = graph_def
        self.d_model = d_model

        # This will hold the instantiated nn.Module for each node in the graph
        self.nodes = nn.ModuleList()

        self._compile_graph()

    def _compile_graph(self):
        """
        Parses the graph_def and instantiates the necessary nn.Module objects.
        """
        for gene in self.graph_def:
            op_name = gene['op']
            if op_name not in OPERATIONS_REGISTRY:
                raise ValueError(f"Unknown operation: {op_name}")

            op_class = OPERATIONS_REGISTRY[op_name]

            # Instantiate operations that require the feature dimension
            if op_name in ['bilinear', 'layer_norm', 'linear']:
                self.nodes.append(op_class(self.d_model))
            else:
                self.nodes.append(op_class())

    def forward(self, q, k, v):
        """
        Executes the forward pass of the compiled attention graph.

        Args:
            q, k, v (torch.Tensor): The query, key, and value tensors.

        Returns:
            torch.Tensor: The output of the final node in the graph.
        """
        # Store the initial inputs for easy access.
        # We use a dictionary for clarity.
        initial_inputs = {'q': q, 'k': k, 'v': v}

        # This list will store the output tensor of each node as we compute it.
        node_outputs = []

        for i, gene in enumerate(self.graph_def):
            # Get the actual operation module we instantiated
            operation = self.nodes[i]

            # Gather the input tensors for this operation
            input_tensors = []
            for input_source in gene['inputs']:
                if isinstance(input_source, str):  # 'q', 'k', or 'v'
                    input_tensors.append(initial_inputs[input_source])
                elif isinstance(input_source, int):  # Index of a previous node
                    if input_source >= len(node_outputs):
                        raise IndexError(f"Node {i} tries to access output of future node {input_source}")
                    input_tensors.append(node_outputs[input_source])
                else:
                    raise TypeError(f"Invalid input source type: {type(input_source)}")

            # Execute the operation
            try:
                result = operation(*input_tensors)
            except Exception as e:
                print(f"Error executing node {i} ('{gene['op']}') with inputs from {gene['inputs']}")
                # Optional: print shapes for debugging
                for j, t in enumerate(input_tensors):
                    print(f"  Input {j} shape: {t.shape}")
                raise e

            node_outputs.append(result)

        # The final output is the output of the last node in the graph.
        if not node_outputs:
            raise ValueError("Graph is empty and produced no output.")

        return node_outputs[-1]