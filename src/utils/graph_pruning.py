# src/utils/graph_pruning.py

from collections import deque
from copy import deepcopy


def prune_graph(graph_def: list) -> list:
    """
    Automatically removes "dead code" from a symbolic graph.

    It works by starting from the final node (the output) and traversing
    backwards to find all nodes that contribute to the final result. Any nodes
    not visited during this traversal are considered dead code and are pruned.

    Args:
        graph_def (list): The original, potentially unpruned graph definition.

    Returns:
        list: A new, pruned graph definition with re-mapped indices.
    """
    if not graph_def:
        return []

    # 1. Find all necessary nodes by traversing backwards from the last node
    nodes_to_keep = set()
    q = deque()

    # The last node is always the output and is therefore necessary
    final_node_idx = len(graph_def) - 1
    nodes_to_keep.add(final_node_idx)
    q.append(final_node_idx)

    while q:
        current_idx = q.popleft()
        node = graph_def[current_idx]

        for inp in node.get('inputs', []):
            if isinstance(inp, int) and inp not in nodes_to_keep:
                nodes_to_keep.add(inp)
                q.append(inp)

    # 2. Rebuild the graph using only the necessary nodes
    if not nodes_to_keep:
        return []

    pruned_graph = []
    # A map from old, sparse indices to new, dense indices
    old_to_new_idx_map = {}

    # Iterate through the original graph to maintain topological order
    for i, node in enumerate(graph_def):
        if i in nodes_to_keep:
            new_node = deepcopy(node)

            # Remap the input indices for the new graph
            new_inputs = []
            for inp in new_node['inputs']:
                if isinstance(inp, int):
                    # This input must exist in our map because we process in order
                    new_inputs.append(old_to_new_idx_map[inp])
                else:
                    new_inputs.append(inp)  # Keep 'q', 'k', 'v' as is

            new_node['inputs'] = new_inputs
            pruned_graph.append(new_node)

            # Add the new, dense index to our map for future nodes to use
            old_to_new_idx_map[i] = len(pruned_graph) - 1

    return pruned_graph