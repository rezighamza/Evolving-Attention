# src/search_algorithm/evolution.py

import random
from tqdm import tqdm
from copy import deepcopy
import sys

from src.evaluation.fitness import FitnessCalculator
from src.search_algorithm.mutation_operators import mutate_graph
from src.search_algorithm.crossover_operators import crossover_graphs

from src.search_algorithm.mutation_operators import mutate_chromosome
from src.search_algorithm.crossover_operators import crossover_chromosomes


def fast_non_dominated_sort(fitnesses):
    population_size = len(fitnesses)
    fronts = [[]]
    S = [[] for _ in range(population_size)]
    n = [0] * population_size

    for p in range(population_size):
        for q in range(population_size):
            if p == q: continue
            # Check for dominance: p dominates q
            if (fitnesses[p][0] >= fitnesses[q][0] and fitnesses[p][1] >= fitnesses[q][1]) and \
                    (fitnesses[p][0] > fitnesses[q][0] or fitnesses[p][1] > fitnesses[q][1]):
                S[p].append(q)
            # Check for dominance: q dominates p
            elif (fitnesses[q][0] >= fitnesses[p][0] and fitnesses[q][1] >= fitnesses[p][1]) and \
                    (fitnesses[q][0] > fitnesses[p][0] or fitnesses[q][1] > fitnesses[p][1]):  # CORRECTED LINE
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
    return fronts[:-1]


def crowding_distance_assignment(fitnesses, front):
    if not front: return {}
    pop_size = len(front)
    distances = {i: 0 for i in front}
    num_objectives = len(fitnesses[0])

    for m in range(num_objectives):
        sorted_front = sorted(front, key=lambda i: fitnesses[i][m])
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        if pop_size > 2:
            obj_range = fitnesses[sorted_front[-1]][m] - fitnesses[sorted_front[0]][m]
            if obj_range == 0: obj_range = sys.float_info.epsilon

            for i in range(1, pop_size - 1):
                distances[sorted_front[i]] += (fitnesses[sorted_front[i + 1]][m] - fitnesses[sorted_front[i - 1]][
                    m]) / obj_range
    return distances


def tournament_selection(population_indices, fitnesses, fronts, crowding_distances, tournament_size):
    """Selects a winning index from a random tournament based on rank and crowding."""
    contender_indices = random.sample(population_indices, tournament_size)
    best_contender_idx = contender_indices[0]

    for i in range(1, tournament_size):
        contender_idx = contender_indices[i]

        # Find ranks (the lower the front index, the better the rank)
        rank_best = next(j for j, front in enumerate(fronts) if best_contender_idx in front)
        rank_contender = next(j for j, front in enumerate(fronts) if contender_idx in front)

        if rank_contender < rank_best:
            best_contender_idx = contender_idx
        elif rank_contender == rank_best:
            if crowding_distances[contender_idx] > crowding_distances[best_contender_idx]:
                best_contender_idx = contender_idx

    return best_contender_idx


class NSGAIISearch:
    def __init__(self, fitness_calculator, population_size=50, generations=20, mutation_rate=0.8,
                 crossover_rate=0.5, tournament_size=3):
        self.fitness_calculator = fitness_calculator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

    def _initialize_population(self):
        """Initializes a population of chromosomes (graph + proj_config)."""
        population = []
        for _ in range(self.population_size):
            # Start with standard attention graph
            graph_def = [
                {'op': 'scaled_dot_product', 'inputs': ['q', 'k']},
                {'op': 'softmax', 'inputs': [0]},
                {'op': 'weighted_sum', 'inputs': [1, 'v']},
            ]
            # And a standard full projection config
            proj_config = {'has_wq': True, 'has_wk': True, 'has_wv': True, 'has_wo': True}

            # Create a chromosome and mutate it to create diversity
            chromosome = {'graph_def': graph_def, 'proj_config': proj_config}
            population.append(mutate_chromosome(chromosome))
        return population

    def run(self):
        print("Initializing and evaluating initial population...")
        population = self._initialize_population()
        fitnesses = [self.fitness_calculator.calculate_fitness(ind) for ind in
                     tqdm(population, desc="Initial Evaluation")]

        for gen in range(self.generations):
            print(f"\n--- Generation {gen + 1}/{self.generations} ---")

            # --- Pre-calculate ranks and distances for selection ---
            parent_indices = list(range(len(population)))
            fronts = fast_non_dominated_sort(fitnesses)
            crowding_distances = {}
            for front in fronts:
                crowding_distances.update(crowding_distance_assignment(fitnesses, front))

            # --- Create Offspring using proper Tournament Selection ---
            offspring_population = []
            for _ in range(self.population_size // 2):
                p1_idx = tournament_selection(parent_indices, fitnesses, fronts, crowding_distances,
                                              self.tournament_size)
                p2_idx = tournament_selection(parent_indices, fitnesses, fronts, crowding_distances,
                                              self.tournament_size)
                parent1, parent2 = population[p1_idx], population[p2_idx]

                if random.random() < self.crossover_rate:
                    child1, child2 = crossover_chromosomes(parent1, parent2)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)

                offspring_population.append(mutate_chromosome(child1))
                offspring_population.append(mutate_chromosome(child2))

            # --- Evaluate offspring and create combined pool for selection ---
            offspring_fitnesses = [self.fitness_calculator.calculate_fitness(ind) for ind in
                                   tqdm(offspring_population, desc="Evaluating Offspring")]

            combined_population = population + offspring_population
            combined_fitnesses = fitnesses + offspring_fitnesses

            # --- Elitist Selection for the next generation ---
            combined_fronts = fast_non_dominated_sort(combined_fitnesses)

            new_population, new_fitnesses = [], []
            front_num = 0
            while len(new_population) + len(combined_fronts[front_num]) <= self.population_size:
                current_front_indices = combined_fronts[front_num]
                new_population.extend([combined_population[i] for i in current_front_indices])
                new_fitnesses.extend([combined_fitnesses[i] for i in current_front_indices])
                front_num += 1
                if front_num >= len(combined_fronts): break

            if len(new_population) < self.population_size and front_num < len(combined_fronts):
                remaining_needed = self.population_size - len(new_population)
                last_front_indices = combined_fronts[front_num]

                last_front_distances = crowding_distance_assignment(combined_fitnesses, last_front_indices)
                sorted_by_crowding = sorted(last_front_indices, key=lambda i: last_front_distances[i], reverse=True)

                for i in sorted_by_crowding[:remaining_needed]:
                    new_population.append(combined_population[i])
                    new_fitnesses.append(combined_fitnesses[i])

            population = new_population
            fitnesses = new_fitnesses

            # --- Logging ---
            final_fronts_indices = fast_non_dominated_sort(fitnesses)
            if not final_fronts_indices or not final_fronts_indices[0]:
                print("No valid solutions in population.")
                continue

            best_front_indices = final_fronts_indices[0]
            print(f"Pareto Front (Front 0) has {len(best_front_indices)} members.")
            best_acc_idx = max(best_front_indices, key=lambda i: fitnesses[i][0])
            print(
                f"  - Top Accuracy on Front: {fitnesses[best_acc_idx][0]:.4f} (FLOPs: {-fitnesses[best_acc_idx][1]:.4f}M)")

        print("\nSearch Complete! Final Pareto front identified.")
        final_front_indices = fast_non_dominated_sort(fitnesses)[0]
        pareto_front_solutions = [{
            "graph": population[i],
            "fitness": {"accuracy": fitnesses[i][0], "negative_flops_M": fitnesses[i][1]}
        } for i in final_front_indices]

        print(f"Discovered a Pareto front with {len(pareto_front_solutions)} optimal solutions.")
        return pareto_front_solutions