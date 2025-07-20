# src/main_benchmark_multi.py

import json
import torch
import argparse
from pathlib import Path

from benchmarks.models.vit import VisionTransformer
from benchmarks.tasks.image_classification import run_benchmark as run_vision_benchmark

from benchmarks.models.text_transformer import TextTransformer
from benchmarks.tasks.text_classification import run_text_benchmark, get_imdb_dataloaders

from benchmarks.models.attention_modules import StandardAttention, DiscoveredAttention
from src.utils.graph_pruning import prune_graph
from src.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser(description="Run a multi-modal benchmark suite on a co-evolved chromosome.")
    parser.add_argument("--run_dir", type=str, required=True, help="Timestamped directory of the search run.")
    parser.add_argument("--candidate_idx", type=int, default=0, help="Index of the candidate in pareto_front.json.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Starting Co-Evolved Benchmark Suite on device: {device} with Seed: {args.seed} ---")

    # --- 1. Load the Champion Chromosome ---
    base_dir = Path(__file__).resolve().parent.parent
    results_file = base_dir / "results/search_artifacts" / args.run_dir / "pareto_front.json"

    with open(results_file, 'r') as f:
        data = json.load(f)

    if args.candidate_idx >= len(data['pareto_front']):
        print(f"Error: Candidate index {args.candidate_idx} is out of bounds.")
        return

    # --- THIS IS THE FIX ---
    # The chromosome data is nested under the 'graph' key.
    champion_chromosome = data['pareto_front'][args.candidate_idx]['graph']

    # Now the following lines will work correctly
    original_graph = champion_chromosome['graph_def']
    proj_config = champion_chromosome['proj_config']

    # Prune the graph part of the chromosome
    pruned_graph = prune_graph(original_graph)
    champion_chromosome['graph_def'] = pruned_graph

    print("\n" + "=" * 50)
    print("Loaded and Pruned Champion Chromosome:")
    print(f"  - Projection Config: {proj_config}")
    print(f"  - Pruned Graph: {json.dumps(pruned_graph, indent=2)}")
    # We need to go up one level to get the fitness
    print(f"  - Fitness on Proxy Task: {data['pareto_front'][args.candidate_idx]['fitness']}")
    print("=" * 50 + "\n")

    # The rest of the file is correct and does not need to be changed.
    # ... (BENCHMARK 1 and BENCHMARK 2 sections are the same) ...
    # ... (FINAL REPORT section is the same) ...

    # === BENCHMARK 1: IMAGE CLASSIFICATION (CIFAR-10) ===
    print("\n--- Benchmark 1: Vision Transformer on CIFAR-10 ---")
    vit_params = {'img_size': 32, 'patch_size': 4, 'd_model': 192, 'n_layers': 6, 'n_head': 3,
                  'dim_feedforward': 192 * 4, 'n_classes': 10}

    baseline_proj_config = {'has_wq': True, 'has_wk': True, 'has_wv': True, 'has_wo': True}
    baseline_vit = VisionTransformer(attention_module_class=StandardAttention,
                                     attention_graph_def=None,  # Baseline has no graph_def
                                     proj_config=baseline_proj_config, **vit_params)

    champion_vit = VisionTransformer(attention_module_class=DiscoveredAttention,
                                     attention_graph_def=pruned_graph,
                                     proj_config=proj_config, **vit_params)

    print("\n-- Running Baseline ViT --")
    baseline_vision_acc = run_vision_benchmark(baseline_vit, epochs=15, lr=1e-3, device=device)
    print("\n-- Running Champion ViT --")
    champion_vision_acc = run_vision_benchmark(champion_vit, epochs=15, lr=1e-3, device=device)

    # === BENCHMARK 2: TEXT CLASSIFICATION (IMDB) ===
    print("\n\n--- Benchmark 2: Text Transformer on IMDB ---")
    _, _, n_tokens = get_imdb_dataloaders()
    text_params = {'n_tokens': n_tokens, 'd_model': 128, 'n_head': 4, 'dim_feedforward': 128 * 4, 'n_layers': 4}

    baseline_text = TextTransformer(attention_module_class=StandardAttention,
                                    attention_graph_def=None,
                                    proj_config=baseline_proj_config, **text_params)
    champion_text = TextTransformer(attention_module_class=DiscoveredAttention,
                                    attention_graph_def=pruned_graph,
                                    proj_config=proj_config, **text_params)

    print("\n-- Running Baseline Text Transformer --")
    baseline_text_acc = run_text_benchmark(baseline_text, epochs=5, lr=1e-4, device=device)
    print("\n-- Running Champion Text Transformer --")
    champion_text_acc = run_text_benchmark(champion_text, epochs=5, lr=1e-4, device=device)

    # --- FINAL REPORT ---
    print("\n\n" + "=" * 50)
    print("        CO-EVOLVED MULTI-MODAL BENCHMARK RESULTS")
    print("=" * 50)
    print("\n[Vision Benchmark - CIFAR-10]")
    print(f"  Standard Attention Accuracy: {baseline_vision_acc:.2f}%")
    print(f"  Discovered Champion Accuracy: {champion_vision_acc:.2f}%")
    print("\n[Text Benchmark - IMDB]")
    print(f"  Standard Attention Accuracy: {baseline_text_acc:.2f}%")
    print(f"  Discovered Champion Accuracy: {champion_text_acc:.2f}%")
    print("\n" + "=" * 50)


if __name__ == '__main__':
    main()