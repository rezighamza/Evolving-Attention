import random
from copy import deepcopy
from src.search_space.operations import OPERATIONS_REGISTRY

# List of operations that can be used for mutation
MUTATABLE_OPS = list(OPERATIONS_REGISTRY.keys())
# Define input sources: initial inputs + a placeholder for node outputs
VALID_INPUT_SOURCES = ['q', 'k', 'v']


def mutate_graph(graph_def):
    """
    Applies one of several possible mutations to a graph definition.
    """
    mutated_graph = deepcopy(graph_def)

    # Don't mutate empty or very small graphs to avoid errors
    if not mutated_graph:
        return mutated_graph

    mutation_type = random.choice(['mutate_op', 'mutate_connection', 'add_node', 'remove_node'])

    if mutation_type == 'mutate_op':
        node_idx = random.randrange(len(mutated_graph))
        mutated_graph[node_idx]['op'] = random.choice(MUTATABLE_OPS)

    elif mutation_type == 'mutate_connection':
        node_idx = random.randrange(len(mutated_graph))
        # Ensure the selected node has inputs to mutate
        if not mutated_graph[node_idx]['inputs']:
            return mutated_graph

        input_idx_to_mutate = random.randrange(len(mutated_graph[node_idx]['inputs']))

        # New source can be 'q', 'k', 'v' or the output of a PREVIOUS node
        max_source_node = node_idx - 1
        possible_sources = VALID_INPUT_SOURCES + list(range(max_source_node + 1))

        if not possible_sources:  # Should not happen in a valid graph
            return mutated_graph

        new_source = random.choice(possible_sources)
        mutated_graph[node_idx]['inputs'][input_idx_to_mutate] = new_source

    elif mutation_type == 'add_node':
        new_op = random.choice(MUTATABLE_OPS)

        # Find a random edge to split (node_idx -> its input)
        node_idx = random.randrange(len(mutated_graph))
        if not mutated_graph[node_idx]['inputs']:
            return mutated_graph

        input_idx_to_split = random.randrange(len(mutated_graph[node_idx]['inputs']))
        original_source = mutated_graph[node_idx]['inputs'][input_idx_to_split]

        # Insert the new node right before the chosen node
        insertion_point = node_idx
        # The new node takes the original source as its input
        # Note: This is a simple version. A more robust implementation would check arity.
        new_node = {'op': new_op, 'inputs': [original_source]}

        # Insert the new node into the graph
        mutated_graph.insert(insertion_point, new_node)

        # The original node now points to the new node
        mutated_graph[node_idx + 1]['inputs'][input_idx_to_split] = insertion_point

        # We need to shift indices for all subsequent nodes
        for i in range(node_idx + 2, len(mutated_graph)):
            for j, inp in enumerate(mutated_graph[i]['inputs']):
                if isinstance(inp, int) and inp >= insertion_point:
                    mutated_graph[i]['inputs'][j] += 1

    elif mutation_type == 'remove_node':
        if len(mutated_graph) <= 1:  # Don't remove the only node
            return mutated_graph

        node_idx_to_remove = random.randrange(len(mutated_graph))

        # A simple removal strategy: remove the node and try to have later nodes
        # point to its first input source. This can fail if arities don't match,
        # but for evolution, sometimes "breaking" things is okay.
        input_source_to_remap = 'q'  # Default fallback
        if mutated_graph[node_idx_to_remove]['inputs']:
            input_source_to_remap = mutated_graph[node_idx_to_remove]['inputs'][0]

        # Remove the node
        mutated_graph.pop(node_idx_to_remove)

        # Remap and shift indices for all subsequent nodes
        for i in range(node_idx_to_remove, len(mutated_graph)):
            for j, inp in enumerate(mutated_graph[i]['inputs']):
                if isinstance(inp, int):
                    if inp == node_idx_to_remove:
                        mutated_graph[i]['inputs'][j] = input_source_to_remap  # Remap
                    elif inp > node_idx_to_remove:
                        mutated_graph[i]['inputs'][j] -= 1  # Shift index

    return mutated_graph