# src/main_benchmark.py

import json
import torch
from pathlib import Path
import argparse  # Import the argument parsing library

from benchmarks.models.vit import VisionTransformer
from benchmarks.models.attention_modules import StandardAttention, DiscoveredAttention
from benchmarks.tasks.image_classification import run_benchmark


def main():
    # --- New: Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Benchmark discovered attention mechanisms against a baseline.")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="The timestamped directory of the search run to benchmark (e.g., 'search_20250716-053346')."
    )
    parser.add_argument(
        "--candidate_idx",
        type=int,
        default=0,
        help="The index of the candidate to select from the pareto_front.json file (default: 0)."
    )
    args = parser.parse_args()

    print("--- Starting ViT Benchmark on CIFAR-100 ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmark on device: {device}")

    # --- 1. Load the discovered attention graph (using the command-line argument) ---
    ROOT_DIR = Path(__file__).resolve().parent.parent
    results_file = ROOT_DIR / "results" / "search_artifacts" / args.run_dir / "pareto_front.json"

    print(f"Attempting to load results from: {results_file}")
    if not results_file.is_file():
        print(f"Error: Could not find results file at the specified path.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    # Select the specific candidate from the Pareto front
    if args.candidate_idx >= len(data['pareto_front']):
        print(
            f"Error: candidate_idx {args.candidate_idx} is out of bounds for the Pareto front (size: {len(data['pareto_front'])}).")
        return

    discovered_candidate = data['pareto_front'][args.candidate_idx]
    discovered_graph = discovered_candidate['graph']
    print(f"Loaded candidate {args.candidate_idx} from run '{args.run_dir}':")
    print(f"  - Graph: {discovered_graph}")
    print(f"  - Fitness: {discovered_candidate['fitness']}")

    # --- 2. Define Benchmark Models ---
    vit_params = {
        'img_size': 32, 'patch_size': 4, 'd_model': 192, 'n_layers': 6,
        'n_head': 3, 'dim_feedforward': 192 * 4, 'n_classes': 100
    }

    print("\nInitializing model with Standard Attention...")
    baseline_vit = VisionTransformer(attention_module_class=StandardAttention, **vit_params)

    print("Initializing model with Discovered Attention...")
    discovered_vit = VisionTransformer(attention_module_class=DiscoveredAttention, attention_graph_def=discovered_graph,
                                       **vit_params)

    # --- 3. Run Benchmarks ---
    epochs = 25
    lr = 1e-3

    print(f"\n--- Benchmarking Baseline Model ({epochs} epochs) ---")
    baseline_accuracy = run_benchmark(baseline_vit, epochs=epochs, lr=lr, device=device)

    print(f"\n--- Benchmarking Discovered Model ({epochs} epochs) ---")
    discovered_accuracy = run_benchmark(discovered_vit, epochs=epochs, lr=lr, device=device)

    # --- 4. Report Final Results ---
    print("\n--- Benchmark Complete ---")
    print(f"Run Directory: {args.run_dir}")
    print(f"Candidate Index: {args.candidate_idx}")
    print(f"Standard Attention Accuracy: {baseline_accuracy:.2f}%")
    print(f"Discovered Attention Accuracy: {discovered_accuracy:.2f}%")


if __name__ == '__main__':
    main()