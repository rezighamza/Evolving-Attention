# src/main_benchmark.py

import json
import torch
from pathlib import Path
import argparse

from benchmarks.models.vit import VisionTransformer
from benchmarks.models.attention_modules import StandardAttention, DiscoveredAttention
from benchmarks.tasks.image_classification import run_benchmark
# --- NEW: Import the automated pruner ---
from src.utils.graph_pruning import prune_graph


def main():
    parser = argparse.ArgumentParser(description="Benchmark discovered attention mechanisms against a baseline.")
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="The timestamped directory of the search run to benchmark (e.g., 'search_...')."
    )
    parser.add_argument(
        "--candidate_idx", type=int, default=0,
        help="The index of the candidate to select from the pareto_front.json file (default: 0)."
    )
    args = parser.parse_args()

    print("--- Starting ViT Benchmark on CIFAR-100 ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmark on device: {device}")

    # --- 1. Load the discovered attention graph ---
    results_file = Path("results/search_artifacts") / args.run_dir / "pareto_front.json"

    if not results_file.is_file():
        print(f"Error: Could not find results file at {results_file}.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    if args.candidate_idx >= len(data['pareto_front']):
        print(
            f"Error: candidate_idx {args.candidate_idx} is out of bounds for the Pareto front (size: {len(data['pareto_front'])}).")
        return

    discovered_candidate = data['pareto_front'][args.candidate_idx]
    original_graph = discovered_candidate['graph']

    # --- NEW: Automatically prune the graph ---
    pruned_graph = prune_graph(original_graph)

    print(f"\nLoaded candidate {args.candidate_idx} from run '{args.run_dir}':")
    print(f"  - Fitness: {discovered_candidate['fitness']}")
    print("\nOriginal Graph:")
    print(json.dumps(original_graph, indent=2))
    print("\nAutomatically Pruned Graph:")
    print(json.dumps(pruned_graph, indent=2))

    # --- 2. Define Benchmark Models ---
    vit_params = {
        'img_size': 32, 'patch_size': 4, 'd_model': 192, 'n_layers': 6,
        'n_head': 3, 'dim_feedforward': 192 * 4, 'n_classes': 100
    }

    print("\nInitializing model with Standard Attention...")
    baseline_vit = VisionTransformer(attention_module_class=StandardAttention, **vit_params)

    print("Initializing model with Discovered Attention...")
    discovered_vit = VisionTransformer(attention_module_class=DiscoveredAttention,
                                       # --- Use the pruned graph ---
                                       attention_graph_def=pruned_graph,
                                       **vit_params)

    # --- 3. Run Benchmarks ---
    epochs = 25
    lr = 1e-3

    print(f"\n--- Benchmarking Baseline Model ({epochs} epochs) ---")
    baseline_accuracy = run_benchmark(baseline_vit, epochs=epochs, lr=lr, device=device)

    print(f"\n--- Benchmarking Discovered Model ({epochs} epochs) ---")
    discovered_accuracy = run_benchmark(discovered_vit, epochs=epochs, lr=lr, device=device)

    # --- 4. Report Final Results ---
    print("\n" + "=" * 40)
    print("      FINAL BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Standard Attention Accuracy: {baseline_accuracy:.2f}%")
    print(f"Discovered Attention Accuracy: {discovered_accuracy:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    main()