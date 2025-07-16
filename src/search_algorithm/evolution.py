# src/search_algorithm/evolution.py

import random
from tqdm import tqdm
from copy import deepcopy
import sys

from src.evaluation.fitness import FitnessCalculator
from src.search_algorithm.mutation_operators import mutate_graph
from src.search_algorithm.crossover_operators import crossover_graphs


# --- NSGA-II Helper Functions ---

def fast_non_dominated_sort(fitnesses):
    """
    Sorts the population into fronts based on dominance.
    Args:
        fitnesses (list of tuples): A list of (performance, -flops) for each individual.
    Returns:
        list of lists: A list of fronts, where each front is a list of indices.
    """
    population_size = len(fitnesses)
    fronts = [[]]

    # S[p] = list of solutions dominated by p
    # n[p] = number of solutions that dominate p
    S = [[] for _ in range(population_size)]
    n = [0] * population_size

    for p in range(population_size):
        for q in range(population_size):
            if p == q:
                continue
            # Check for dominance
            # fitnesses[p] dominates fitnesses[q] if...
            if (fitnesses[p][0] >= fitnesses[q][0] and fitnesses[p][1] >= fitnesses[q][1]) and \
                    (fitnesses[p][0] > fitnesses[q][0] or fitnesses[p][1] > fitnesses[q][1]):
                S[p].append(q)
            elif (fitnesses[q][0] >= fitnesses[p][0] and fitnesses[q][1] >= fitnesses[p][1]) and \
                    (fitnesses[q][0] > fitnesses[p][0] or fitnesses[q][1] > fitnesses[p][1]):
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]  # The last front is always empty


def crowding_distance_assignment(fitnesses, front):
    """
    Calculates the crowding distance for each individual in a front.
    """
    if not front:
        return {}

    pop_size = len(front)
    distances = {i: 0 for i in front}

    for m in range(len(fitnesses[0])):  # For each objective (accuracy, -flops)
        # Sort the front by the current objective
        sorted_front = sorted(front, key=lambda i: fitnesses[i][m])

        # Assign infinite distance to boundary solutions
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        if pop_size > 2:
            # Range of the objective values in this front
            obj_range = fitnesses[sorted_front[-1]][m] - fitnesses[sorted_front[0]][m]
            if obj_range == 0:
                obj_range = sys.float_info.epsilon  # Avoid division by zero

            # Add distances for intermediate solutions
            for i in range(1, pop_size - 1):
                distances[sorted_front[i]] += (fitnesses[sorted_front[i + 1]][m] - fitnesses[sorted_front[i - 1]][
                    m]) / obj_range

    return distances


# --- Main NSGA-II Class ---

class NSGAIISearch:
    def __init__(self, fitness_calculator: FitnessCalculator, population_size=50, generations=20, mutation_rate=0.8,
                 crossover_rate=0.5):
        self.fitness_calculator = fitness_calculator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _initialize_population(self):
        initial_graph = [
            {'op': 'scaled_dot_product', 'inputs': ['q', 'k']},
            {'op': 'softmax', 'inputs': [0]},
            {'op': 'weighted_sum', 'inputs': [1, 'v']},
        ]
        return [deepcopy(initial_graph) for _ in range(self.population_size)]

    def run(self):
        """Executes the full NSGA-II search process."""
        print("Initializing population...")
        population = self._initialize_population()

        for gen in range(self.generations):
            print(f"\n--- Generation {gen + 1}/{self.generations} ---")

            # --- Create Offspring ---
            offspring_population = []
            # For simplicity, we create a full new population via crossover/mutation
            # A more traditional approach uses tournament selection based on rank/distance
            for _ in range(self.population_size // 2):
                parent1 = random.choice(population)  # Simplified selection
                parent2 = random.choice(population)  # Simplified selection

                if random.random() < self.crossover_rate:
                    child1, child2 = crossover_graphs(parent1, parent2)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)

                if random.random() < self.mutation_rate:
                    child1 = mutate_graph(child1)
                if random.random() < self.mutation_rate:
                    child2 = mutate_graph(child2)

                offspring_population.extend([child1, child2])

            # --- Evaluation ---
            combined_population = population + offspring_population
            fitnesses = [self.fitness_calculator.calculate_fitness(ind) for ind in
                         tqdm(combined_population, desc="Evaluating Fitness")]

            # --- Selection using NSGA-II principles ---
            fronts = fast_non_dominated_sort(fitnesses)

            new_population = []
            front_num = 0
            while len(new_population) + len(fronts[front_num]) <= self.population_size:
                new_population.extend([combined_population[i] for i in fronts[front_num]])
                front_num += 1
                if front_num >= len(fronts): break

            # If the last front needs to be sorted by crowding distance
            if len(new_population) < self.population_size and front_num < len(fronts):
                remaining_needed = self.population_size - len(new_population)

                # Calculate crowding distance for the last front
                distances = crowding_distance_assignment(fitnesses, fronts[front_num])

                # Sort the front by descending crowding distance
                sorted_by_crowding = sorted(fronts[front_num], key=lambda i: distances[i], reverse=True)

                # Add the most diverse individuals
                new_population.extend([combined_population[i] for i in sorted_by_crowding[:remaining_needed]])

            population = new_population

            # --- Logging ---
            best_front_indices = fronts[0]
            print(f"Pareto Front (Front 0) has {len(best_front_indices)} members.")
            # Print one of the best for a quick check
            best_acc_idx = max(best_front_indices, key=lambda i: fitnesses[i][0])
            print(
                f"  - Top Accuracy on Front: {fitnesses[best_acc_idx][0]:.4f} (FLOPs: {-fitnesses[best_acc_idx][1]:.4f}M)")
            best_eff_idx = max(best_front_indices, key=lambda i: fitnesses[i][1])
            print(
                f"  - Top Efficiency on Front: {fitnesses[best_eff_idx][0]:.4f} (FLOPs: {-fitnesses[best_eff_idx][1]:.4f}M)")

        # --- Final Result ---
        print("\nSearch Complete! Calculating final Pareto front...")
        final_fitnesses = [self.fitness_calculator.calculate_fitness(ind) for ind in
                           tqdm(population, desc="Final Evaluation")]
        final_fronts = fast_non_dominated_sort(final_fitnesses)

        pareto_front_indices = final_fronts[0]
        pareto_front_solutions = [
            {
                "graph": population[i],
                "fitness": {
                    "accuracy": final_fitnesses[i][0],
                    "negative_flops_M": final_fitnesses[i][1]
                }
            } for i in pareto_front_indices
        ]

        print(f"Discovered a Pareto front with {len(pareto_front_solutions)} optimal solutions.")
        return pareto_front_solutions