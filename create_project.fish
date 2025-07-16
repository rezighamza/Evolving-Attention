#!/usr/bin/env fish

# ==============================================================================
# Fish script to create the directory structure for the Evolving Attention project
# ==============================================================================

set PROJECT_NAME "evolving-attention"

# Check if directory already exists
if test -d $PROJECT_NAME
    echo "Error: Directory '$PROJECT_NAME' already exists."
    exit 1
end

echo "Creating project structure for '$PROJECT_NAME'..."

# Create root project directory and cd into it
mkdir $PROJECT_NAME
cd $PROJECT_NAME

# 1. Source code directory
mkdir -p src/search_space src/evaluation src/search_algorithm src/utils

# 2. Benchmarking suite
mkdir -p benchmarks/models benchmarks/tasks

# 3. Configuration files
mkdir -p configs

# 4. Data storage (to be ignored by git)
mkdir -p data

# 5. Notebooks for exploration and analysis
mkdir -p notebooks

# 6. Results and outputs (to be ignored by git)
mkdir -p results/search_artifacts results/benchmark_logs

# 7. Scripts for running tasks
mkdir -p scripts

# 8. Documentation
mkdir -p docs

# --- Create initial files ---

# Source code files
touch src/__init__.py
touch src/main_search.py
touch src/main_benchmark.py

touch src/search_space/__init__.py
touch src/search_space/operations.py
touch src/search_space/symbolic_graph.py

touch src/evaluation/__init__.py
touch src/evaluation/fitness.py
touch src/evaluation/proxy_tasks.py

touch src/search_algorithm/__init__.py
touch src/search_algorithm/evolution.py
touch src/search_algorithm/mutation_operators.py
touch src/search_algorithm/crossover_operators.py

touch src/utils/__init__.py
touch src/utils/logging_utils.py
touch src/utils/model_compiler.py

# Benchmark files
touch benchmarks/__init__.py
touch benchmarks/models/__init__.py
touch benchmarks/models/vit.py
touch benchmarks/models/gpt.py
touch benchmarks/tasks/language_modeling.py
touch benchmarks/tasks/image_classification.py

# Config files
touch configs/search_config.yaml
touch configs/benchmark_wikitext.yaml
touch configs/benchmark_cifar100.yaml

# Scripts
touch scripts/run_search.sh
touch scripts/run_benchmark.sh

# Root files
touch README.md
touch LICENSE
touch requirements.txt
touch .gitignore

# --- Populate key files with initial content ---

# README.md
echo "# Evolving Attention: A Mathematical Search Approach" > README.md

# .gitignore
echo """
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# IDE / Editor
.vscode/
.idea/
*.swp
*.swo

# Data and Results
/data/
/results/
/notebooks/.ipynb_checkpoints

# OS files
.DS_Store
Thumbs.db
""" > .gitignore

# requirements.txt
echo """
# Core ML/DL
torch
torchvision
torchaudio

# Scientific Computing
numpy

# Utilities
tqdm         # Progress bars
pyyaml       # For handling .yaml configs
matplotlib   # For plotting results
seaborn      # For prettier plots
pandas       # For results analysis
fvcore       # For FLOPs and parameter counting

# Optional: For evolutionary algorithm
deap         # A popular EA framework we might use
""" > requirements.txt


# Add .gitkeep to empty directories that we want to track in git
touch data/.gitkeep
touch results/.gitkeep
touch docs/.gitkeep

echo "Project structure created successfully in '$PROJECT_NAME/'."
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. git init"
echo "3. Create a Python virtual environment (e.g., python -m venv .venv)"
echo "4. Activate the environment (e.g., source .venv/bin/activate.fish)"
echo "5. Install dependencies: pip install -r requirements.txt"