# src/main_search.py

import torch
import yaml
import json
import os
from datetime import datetime
from pathlib import Path

from src.evaluation.fitness import FitnessCalculator
# Updated import to the new class
from src.search_algorithm.evolution import NSGAIISearch


def main():
    """ Main function to run the NSGA-II search for attention mechanisms. """
    ROOT_DIR = Path(__file__).resolve().parent.parent
    config_path = ROOT_DIR / 'configs' / 'search_config.yaml'

    print(f"Loading configuration from: {config_path}")
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        return
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = ROOT_DIR / config['logging']['results_path'] / f"search_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    fitness_calculator = FitnessCalculator(d_model=config['proxy_task']['d_model'], device=device)

    # Use the new NSGAIISearch class
    searcher = NSGAIISearch(
        fitness_calculator=fitness_calculator,
        population_size=config['ea']['population_size'],
        generations=config['ea']['generations'],
        mutation_rate=config['ea']['mutation_rate'],
        crossover_rate=config['ea']['crossover_rate']
    )

    print("\nStarting NSGA-II search...")
    # The output is now a list of solutions
    pareto_front = searcher.run()

    # Save the entire Pareto front
    final_results = {
        'pareto_front': pareto_front,
        'config': config
    }

    results_filepath = os.path.join(results_dir, 'pareto_front.json')
    with open(results_filepath, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\nPareto front with {len(pareto_front)} solutions saved to {results_filepath}")
    print("--- Evolving Attention Search Finished ---")


if __name__ == '__main__':
    main()