# Layout Optimization System

Optimize layouts of items and item-pairs with advanced single- and multi-objective algorithms.

**Repository**: https://github.com/binarybottle/optimize_layouts.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

## Usage

```bash
# Prepare input data
python normalize_input.py     # Normalize raw input data
python analyze_input.py       # Analyze and plot raw/normalized data

# Single-objective optimization (default)
python optimize_layout.py --config config.yaml

# Multi-objective optimization (Pareto front of candidate solutions)
python optimize_layout.py --config config.yaml --moo

# Detailed MOO analysis with comprehensive validation
python optimize_layout.py --moo --max-solutions 50 --detailed --validate --verbose

# Analyze results
python analyze_results.py

# Score a specific layout
python score_layout.py --items "etaoi" --positions "FJDSV"
```

For running parallel processes, 
see **Running parallel processes** below.

## Overview
This code optimizes a layout, an arrangement of items,
and takes into account scores for individual items, 
item-pairs (a,b and b,a), positions, and position-pairs.
To efficiently prune vast combinatorial layouts, 
it uses a depth-first search with a branch-and-bound algorithm.

The initial intended use-case is keyboard layout optimization 
for touch typing (see **keyboards/README_keyboards.md**):
  - Items and item-pairs correspond to letters and letter-pairs.
  - Positions and position-pairs correspond to keys and key-pairs.  
  - Item scores and item-pair scores correspond to frequency 
    of letters and frequency of letter-pairs in a given language.
  - Position scores and position-pair scores correspond to 
    a measure of comfort when typing single keys or pairs of keys 
    (in any language, as usage frequencies are regressed out).

## Architecture
The system is built with a modular, maintainable architecture:
  - config.py: Configuration management and validation
  - scoring.py: Unified scoring system with JIT optimization
  - search.py: Branch-and-bound and Pareto search algorithms  
  - display.py: Keyboard visualization and results formatting
  - validation.py: Comprehensive testing and validation suite
  - optimize_layout.py: Main CLI interface

## Optimization Modes

  ### Single-Objective Optimization (SOO)
    - Algorithm: Branch-and-bound with tight upper bounds
    - Goal: Find top N highest-scoring layouts
    - Scoring: Combined item × pair score or individual components
    - Output: Ranked list of best solutions with detailed breakdowns

  ```bash
  # Find top 5 layouts (default)
  python optimize_layout.py --config config.yaml

  # Find top 10 layouts with detailed scoring
  python optimize_layout.py --config config.yaml --n-solutions 10 --verbose
  ```

  ### Multi-Objective Optimization (MOO)
    - Algorithm: Pareto-optimal search
    - Goal: Find non-dominated solutions across multiple objectives
    - Objectives: Item scores, internal pair scores, cross-interaction scores
    - Output: Pareto front with trade-off analysis

  ```bash
  # Discover Pareto front
  python optimize_layout.py --config config.yaml --moo

  # Limit solutions and time
  python optimize_layout.py --config config.yaml --moo --max-solutions 100 --time-limit 300
  ```

## Inputs
The system accepts normalized score files in CSV format.
The code will accept any set of letters (or special characters) 
to represent items, item-pairs, positions, and position-pairs, 
so long as each item-pair is represented by two item characters 
and each position-pair is represented by two position characters.

  - raw_item_scores_file:           
    ```
    item, score
    a, 0.1
    b, 0.5
    c, 0.2       
    ```
  - raw_item_pair_scores_file:      
    ```
    item_pair, score
    ab, 0.1
    ba, 0.3
    ac, 0.2       
    ca, 0.1       
    bc, 0.4       
    cb, 0.2       
    ```
  - raw_position_scores_file:       
    ```
    position, score
    X, 0.4
    Y, 0.3
    Z, 0.1       
    ```
  - raw_position_pair_scores_file:  
    ```
    position_pair, score
    XY, 0.2
    YX, 0.3
    XZ, 0.2
    ZX, 0.5
    YZ, 0.4
    ZY, 0.5
    ```

A configuration file (config.yaml) specifies input filenames
(raw and normalized) as well as text strings for the variables:
  - Required:
    - items_to_assign: items to arrange in positions_to_assign
    - positions_to_assign: available positions
  - Optional:
    - items_assigned: items already assigned to positions
    - positions_assigned: positions that are already filled 
    - items_to_constrain: subset of items_to_assign 
      to arrange every way within positions_to_constrain
    - positions_to_constrain: subset of positions_to_assign 
      to constrain the possible assignment of items_to_constrain

## Scoring system
Layouts are scored based on normalized item and item-pair scores 
and corresponding normalized position and position-pair scores, 
where direction (sequence of a given pair) matters.

  1. item_score = Σ(item_score_i × position_score_i) / N_items
  2. pair_score = Σ(item_pair_score_ij × position_pair_score_ij) / N_pairs

  **MOO (Multi-Objective Optimization) scoring**
  - multi_objective: Separate objectives for MOO
  **SOO (Single-Objective Optimization) scoring**
  - item_only: Only individual item-position matches
  - pair_only: Only pair interactions (internal + cross)
  - combined: Multiplicative combination (item × total_pairs)

## Branch-and-bound optimization
  - Calculates exact scores for placed letters
  - Uses provable upper bounds for unplaced letters
  - Prunes branches that cannot exceed best known solution
  - Depth-first search maintains optimality & reduces search space
  - Uses numba-optimized array operations
  - Progress tracking and statistics
  - Validation of input configurations
  - Optional constraints for a subset of items

## Output

  **Console Output**
  - Configuration summary and search space analysis
  - Real-time progress with pruning statistics
  - Top item-to-position solutions
  - Detailed score breakdowns (with --verbose)
  - Performance metrics and timing
  - Optional ASCII art visualization of keyboard layouts

  **CSV Results**
  Automatically saved timestamped files:
  - SOO: soo_results_config_YYYYMMDD_HHMMSS.csv
  - MOO: moo_results_config_YYYYMMDD_HHMMSS.csv

## Optional: Running parallel processes locally or on SLURM
Two commands can be modified to parallelize layout optimization:

  ### Run parallel jobs locally
  You can generate configuration files in output/configs1/
  by creating your own generate_configs.py script,
  following the example in keyboards/generate_configs1.py,
  then modify the run_jobs_local.py script for your needs:
  ```bash
  python3 generate_configs.py
  python3 run_jobs_local.py
  ```

  ### Run parallel jobs on a linux cluster (slurm)
  Connect and set up the code environment: 
  ```bash
  # Log in (example: Pittsburgh Supercomputing Center's Bridge-2 cluster)
  ssh username@bridges2.psc.edu

  # Create directory for the project
  mkdir -p optimizer; cd optimizer

  # Clone repository and make scripts executable
  git clone https://github.com/binarybottle/optimize_layouts.git
  cd optimize_layouts
  chmod +x *.py *.sh

  # Make output directories
  mkdir -p output/layouts output/outputs output/errors output/configs1

  # Test that anaconda3 module works (no virtual environment needed)
  module load anaconda3
  python3 --version
  python3 -c "import numpy, pandas, yaml; print('Required packages available')"

  # Generate config files as described in the local parallel job submission example
  python3 generate_configs.py
  ```

  Configure, submit, monitor, and cancel jobs:
  ```bash
  # Use screen to keep session active
  screen -S submission

  # Automatically manage submissions (checks every 5 minutes)
  nohup bash run_job_slurm.sh

  # Check all running jobs
  squeue -u $USER

  # Watch jobs in real-time
  watch squeue -u $USER

  # See how many jobs are running vs. pending
  squeue -j <job_array_id> | awk '{print $5}' | sort | uniq -c

  # Check number of output files
  sh count_files.sh output/layouts

  # Cancel all your jobs at once, or for a specific job ID
  scancel -u $USER
  scancel <job_array_id>
  ```
