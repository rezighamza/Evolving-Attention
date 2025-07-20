# src/main_search.py

import torch
import yaml
import json
import os
from datetime import datetime
from pathlib import Path

from src.evaluation.fitness import FitnessCalculator
from src.search_algorithm.evolution import NSGAIISearch

def main():
    """ Main function to run the NSGA-II co-evolutionary search. """
    ROOT_DIR = Path(__file__).resolve().parent.parent
    config_path = ROOT_DIR / 'configs' / 'search_config.yaml'

    print(f"Loading configuration from: {config_path}")
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        return
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- UPDATED: Check for new tournament_size parameter ---
    if 'tournament_size' not in config['ea']:
        print("Error: 'tournament_size' not found in search_config.yaml under 'ea' section.")
        print("Please add it (e.g., tournament_size: 3).")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = ROOT_DIR / config['logging']['results_path'] / f"search_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    fitness_calculator = FitnessCalculator(d_model=config['proxy_task']['d_model'], device=device)

    # --- UPDATED: Pass tournament_size to the searcher ---
    searcher = NSGAIISearch(
        fitness_calculator=fitness_calculator,
        population_size=config['ea']['population_size'],
        generations=config['ea']['generations'],
        mutation_rate=config['ea']['mutation_rate'],
        crossover_rate=config['ea']['crossover_rate'],
        tournament_size=config['ea']['tournament_size'] # Pass the new parameter
    )

    print("\nStarting NSGA-II co-evolutionary search...")
    pareto_front = searcher.run()

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
    # Reminder: Run this from the project root using `python -m src.main_search`
    main()