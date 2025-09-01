# Software for optimizing item-to-position layouts

Optimize layouts of items and item-pairs with advanced single- and multi-objective algorithms.

**Repository**: https://github.com/binarybottle/optimize_layouts.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

## Usage
```bash
# Prepare input data (normalize raw files to standard format)
python normalize_input.py     # Reads raw_* files, outputs to output/normalized_input/
python analyze_input.py       # Analyze and plot raw vs normalized data

# Single-objective optimization (default)
python optimize_layout.py --config config.yaml

# Multi-objective optimization (Pareto front of candidate solutions)
python optimize_layout.py --config config.yaml --moo

# Detailed MOO analysis with comprehensive validation
python optimize_layout.py --moo --max-solutions 50 --detailed --validate --verbose

# General MOO with arbitrary objectives from keypair tables
python optimize_layout_general.py --config config.yaml \
  --objectives comfort_score,time_total,accuracy_score \
  --keypair-table input/keypair_scores_detailed.csv \
  --processes 16 --max-solutions 200 --time-limit 3600

# Analyze results
python analyze_results.py

# Score a specific layout
python score_layout.py --items "etaoi" --positions "FJDSV"
```

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
  - optimize_layout_general.py: General MOO with arbitrary objectives from keypair tables

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

  ### General Multi-Objective Optimization (General MOO)
    - Algorithm**: Branch-and-bound search with arbitrary objectives
    - Goal: Optimize for any number of objectives from keypair score tables
    - Input: CSV tables with keypair scores for different metrics
    - Output: Pareto front across all specified objectives

  ```bash
  # Optimize for multiple metrics from keypair table
  python optimize_layout_general.py --config config.yaml \
    --objectives engram8_columns,engram8_curl,engram8_home,engram8_hspan \
    --keypair-table input/keypair_scores_detailed.csv

  # With custom weights and directions
  python optimize_layout_general.py --config config.yaml \
    --objectives comfort_score,time_total,accuracy_score \
    --keypair-table data/keypair_scores.csv \
    --weights 1.0,2.0,0.5 \
    --maximize true,false,true

  # Brute force for small problems or validation
  python optimize_layout_general.py --config config.yaml \
    --objectives comfort_score,time_total \
    --keypair-table data/keypair_scores.csv \
    --brute-force --max-solutions 100
  ```

## Inputs
The system accepts normalized score files in CSV format.
The code will accept any set of letters (or special characters) 
to represent items, item-pairs, positions, and position-pairs, 
so long as each item-pair is represented by two item characters 
and each position-pair is represented by two position characters.

  ### Normalized scores
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

  The script detects different distribution types and applies appropriate methods.
  For example, in a keyboard layout optimization problem:
    - Item & position scores: Robust scaling (1st-99th percentile clipping)
    - Item-pair scores: Heavy-tailed distribution → square root transformation
    - Position-pair scores: Symmetric distribution → standard min-max scaling

  ### Keypair tables for general MOO
  For general multi-objective optimization, provide a keypair table with multiple objective columns:

  ```csv
  key_pair,comfort_score,time_total,accuracy_score,engram8_columns,engram8_curl
  qw,0.85,0.23,0.91,0.78,0.65
  we,0.72,0.31,0.88,0.82,0.71
  er,0.68,0.28,0.85,0.75,0.69
  ```

  ### Configuration file
  A configuration file (config.yaml) specifies raw input filenames. 
  The system automatically generates normalized versions in 
  `output/normalized_input/` with standardized names:

  ```yaml
  paths:
    input:
      raw_item_scores_file: "data/sources/letter-counts.csv"
      raw_item_pair_scores_file: "data/sources/bigram-counts.csv" 
      raw_position_scores_file: "data/comfort/key_comfort_estimates.csv"
      raw_position_pair_scores_file: "data/comfort/key_pair_comfort_estimates.csv"
  ```

  Required optimization variables:
      - items_to_assign: items to arrange in positions_to_assign
      - positions_to_assign: available positions
  Optional:
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

Note: Since the scoring accepts any characters for items and any positions,
there could be a problem if item or position files are incomplete.
Layouts with items or positions that are missing in these files 
will be assigned default scores:
  - prepare_scoring_arrays() will assign default scores of 1.0 for any missing pairs:
    - missing_item_pair_score: float = 1.0,      # ← DEFAULT SCORE FOR MISSING ITEM PAIRS
    - missing_position_pair_score: float = 1.0   # ← DEFAULT SCORE FOR MISSING POSITION PAIRS
For an expanded, standalone "engram" scorer intended for keyboard layouts, 
see https://github.com/binarybottle/keyboard_layout_scorers.

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
  - Top item-to-position solutions
  - Detailed score breakdowns (with --verbose)
  - Performance metrics and timing
  - Optional ASCII art visualization of keyboard layouts

  **CSV Results**
  Automatically saved timestamped files:
  - SOO: soo_results_config_YYYYMMDD_HHMMSS.csv
  - MOO: moo_results_config_YYYYMMDD_HHMMSS.csv
  - General MOO: branch_and_bound_moo_results_config_YYYYMMDD_HHMMSS.csv

## Running many configurations
You can generate configuration files in output/configs1/
by creating your own generate_configs.py script,
following the example in keyboards/generate_configs1.py,
then modify the run_jobs_local.py script for your needs
(see run_jobs_slurm.sh for running jobs on a linux cluster):

```bash
python3 generate_configs.py
python3 run_jobs_local.py
```

Example for 16 of 24 items: etaoinsrhldcumfpgwybvkxj
arranged in 16 of 24 positions: FDESVRWACQZXJKILMUO;,P/.
We can optimize arrangements of 16 items by running 
parallel jobs with 5 fixed items in each config file: 
items (fixed)     config files               permutations 
   10 (6)     12!/6! = 665,280            3,628,800 (10!)
   11 (5)     12!/7! =  95,040           79,833,600 (11!) 
   12 (4)     12!/8! =  11,880          479,001,600 (12!)
   16 (0)                    1   20,922,789,888,000 (16!)
