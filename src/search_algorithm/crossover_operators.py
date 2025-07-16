import random
from copy import deepcopy


def crossover_graphs(graph1, graph2):
    """
    Performs a simple one-point crossover on two graph definitions.
    """
    g1, g2 = deepcopy(graph1), deepcopy(graph2)

    # Choose a crossover point in both graphs
    if not g1 or not g2:
        return g1, g2  # Cannot perform crossover

    p1 = random.randint(1, len(g1))
    p2 = random.randint(1, len(g2))

    # The new children are combinations of the parents' parts
    child1_graph = g1[:p1] + g2[p2:]
    child2_graph = g2[:p2] + g1[p1:]

    # --- Fix invalid connections ---
    # A simple but crucial step. Any connection pointing to a node that no longer
    # exists after crossover is invalid. We'll remap it to a valid source.

    def fix_graph(graph):
        for i, node in enumerate(graph):
            for j, inp in enumerate(node['inputs']):
                # If an input points to a node index beyond the graph's length, it's invalid
                if isinstance(inp, int) and inp >= i:
                    # Remap to a valid previous node or a default input
                    if i > 0:
                        node['inputs'][j] = random.randrange(i)
                    else:
                        node['inputs'][j] = 'q'  # Fallback to 'q'
        return graph

    return fix_graph(child1_graph), fix_graph(child2_graph)