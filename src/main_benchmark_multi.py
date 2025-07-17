# src/main_benchmark_multi.py

import json
import torch
import argparse
from pathlib import Path

# Vision imports
from benchmarks.models.vit import VisionTransformer
from benchmarks.tasks.image_classification import run_benchmark as run_vision_benchmark

# Text imports
from benchmarks.tasks.text_classification import run_text_benchmark, get_imdb_dataloaders
from benchmarks.models.text_transformer import TextTransformer

# Common imports
from benchmarks.models.attention_modules import StandardAttention, DiscoveredAttention
from src.utils.graph_pruning import prune_graph


def main():
    parser = argparse.ArgumentParser(description="Run a multi-modal benchmark suite.")
    parser.add_argument("--run_dir", type=str, required=True, help="Timestamped directory of the search run.")
    parser.add_argument("--candidate_idx", type=int, default=0, help="Index of the candidate in pareto_front.json.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Starting Multi-Modal Benchmark Suite on device: {device} ---")

    # --- 1. Load and Prune the Champion Graph ---
    base_dir = Path(__file__).resolve().parent.parent
    results_file = base_dir / "results/search_artifacts" / args.run_dir / "pareto_front.json"

    with open(results_file, 'r') as f:
        data = json.load(f)

    if args.candidate_idx >= len(data['pareto_front']):
        print(f"Error: Candidate index {args.candidate_idx} is out of bounds.")
        return

    champion = data['pareto_front'][args.candidate_idx]
    pruned_graph = prune_graph(champion['graph'])

    print("\n" + "=" * 50)
    print("Loaded and Pruned Champion Graph:")
    print(json.dumps(pruned_graph, indent=2))
    print(f"Fitness on Proxy Task: {champion['fitness']}")
    print("=" * 50 + "\n")

    # === BENCHMARK 1: IMAGE CLASSIFICATION (CIFAR-10) ===
    # --- CHANGE 1: Updated print statement for clarity ---
    print("\n--- Benchmark 1: Vision Transformer on CIFAR-10 ---")
    # The n_classes=10 parameter you set is now correct for CIFAR-10
    vit_params = {'img_size': 32, 'patch_size': 4, 'd_model': 192, 'n_layers': 6, 'n_head': 3,
                  'dim_feedforward': 192 * 4, 'n_classes': 10}

    baseline_vit = VisionTransformer(attention_module_class=StandardAttention, **vit_params)
    champion_vit = VisionTransformer(attention_module_class=DiscoveredAttention, attention_graph_def=pruned_graph,
                                     **vit_params)

    print("\n-- Running Baseline ViT --")
    baseline_vision_acc = run_vision_benchmark(baseline_vit, epochs=15, lr=1e-3, device=device)
    print("\n-- Running Champion ViT --")
    champion_vision_acc = run_vision_benchmark(champion_vit, epochs=15, lr=1e-3, device=device)

    # === BENCHMARK 2: TEXT CLASSIFICATION (IMDB) ===
    print("\n\n--- Benchmark 2: Text Transformer on IMDB ---")
    _, _, n_tokens = get_imdb_dataloaders()
    text_params = {'n_tokens': n_tokens, 'd_model': 128, 'n_head': 4, 'dim_feedforward': 128 * 4, 'n_layers': 4}

    baseline_text = TextTransformer(attention_module_class=StandardAttention, **text_params)
    champion_text = TextTransformer(attention_module_class=DiscoveredAttention, attention_graph_def=pruned_graph,
                                    **text_params)

    print("\n-- Running Baseline Text Transformer --")
    baseline_text_acc = run_text_benchmark(baseline_text, epochs=5, lr=1e-4, device=device)
    print("\n-- Running Champion Text Transformer --")
    champion_text_acc = run_text_benchmark(champion_text, epochs=5, lr=1e-4, device=device)

    # --- FINAL REPORT ---
    print("\n\n" + "=" * 50)
    print("        MULTI-MODAL BENCHMARK RESULTS")
    print("=" * 50)
    # --- CHANGE 2: Updated print label for the vision benchmark ---
    print("\n[Vision Benchmark - CIFAR-10]")
    print(f"  Standard Attention Accuracy: {baseline_vision_acc:.2f}%")
    print(f"  Discovered Champion Accuracy: {champion_vision_acc:.2f}%")
    print("\n[Text Benchmark - IMDB]")
    print(f"  Standard Attention Accuracy: {baseline_text_acc:.2f}%")
    print(f"  Discovered Champion Accuracy: {champion_text_acc:.2f}%")
    print("\n" + "=" * 50)


if __name__ == '__main__':
    main()