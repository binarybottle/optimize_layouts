# Layout Optimization System

Optimize layouts of items and item-pairs with advanced single- and multi-objective algorithms.

---

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

# Single-objective with custom parameters  
python optimize_layout.py --config config.yaml --n-solutions 10 --verbose

# Multi-objective optimization (Pareto front)
python optimize_layout.py --config config.yaml --moo --max-solutions 50

# With comprehensive validation
python optimize_layout.py --config config.yaml --validate --verbose

# Analyze results
python analyze_results.py

# Score a specific layout
python display_and_score_layout.py --items "etaoi" --positions "FJDSV"

For running parallel processes, 
see **Running parallel processes** below.
```

## Overview
-------------------------------------------------------------------
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
-------------------------------------------------------------------
The system is built with a modular, maintainable architecture:
  - config.py: Configuration management and validation
  - scoring.py: Unified scoring system with JIT optimization
  - search.py: Branch-and-bound and Pareto search algorithms  
  - display.py: Keyboard visualization and results formatting
  - validation.py: Comprehensive testing and validation suite
  - optimize_layout.py: Main CLI interface

## Optimization Modes
-------------------------------------------------------------------
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
-------------------------------------------------------------------
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
-------------------------------------------------------------------
Layouts are scored based on normalized item and item-pair scores 
and corresponding normalized position and position-pair scores, 
where direction (sequence of a given pair) matters.

### Score Components
  - Item Component: sum(item_score × position_score) / num_items
  - Pair Component: sum(item_pair_score × position_pair_score) / num_pairs

### Scoring Modes

*** SOO ***
  - item_only: Only individual item-position matches
  - pair_only: Only pair interactions (internal + cross)
  - combined: Multiplicative combination (item × total_pairs)

*** MOO ***
  - multi_objective: Separate objectives for MOO

### Score Calculation Formula
item_component = Σ(item_score_i × position_score_i) / N_items

pair_component = Σ(item_pair_score_ij × position_pair_score_ij) / N_pairs

cross_component = Σ(assigned_interactions) / N_cross_pairs

## Branch-and-bound optimization
-------------------------------------------------------------------
  - Calculates exact scores for placed letters
  - Uses provable upper bounds for unplaced letters
  - Prunes branches that cannot exceed best known solution
  - Depth-first search maintains optimality & reduces search space
  - Uses numba-optimized array operations
  - Progress tracking and statistics
  - Validation of input configurations
  - Optional constraints for a subset of items

## Output
-------------------------------------------------------------------
### Console Output
  - Configuration summary and search space analysis
  - Real-time progress with pruning statistics
  - Top item-to-position solutions
  - Detailed score breakdowns (with --verbose)
  - Performance metrics and timing
  - Optional ASCII art visualization of keyboard layouts

### CSV Results
Automatically saved timestamped files:
  - SOO: soo_results_config_YYYYMMDD_HHMMSS.csv
  - MOO: moo_results_config_YYYYMMDD_HHMMSS.csv

## Optional: Running parallel processes on SLURM
-------------------------------------------------------------------
The example commands below to parallelize layout optimization
were used as part of a keyboard layout optimization study supported 
by NSF and Pittsburgh Supercomputing Center computing resources 
(see **README_keyboards**).

  ### Connect and set up the code environment
    ```bash
    # Log in
    ssh username@bridges2.psc.edu

    # Create directory for the project
    mkdir -p keyboard_optimizer; cd keyboard_optimizer

    # Load Python module, create virtual environment, 
    # and make activate script executable
    module load python/3.8.6
    python -m venv keyboard_env
    source keyboard_env/bin/activate
    chmod +x $HOME/keyboard_optimizer/keyboard_env/bin/activate

    # Install required packages
    pip install pyyaml numpy pandas tqdm numba psutil matplotlib
  
    # Clone repository, make scripts executable, 
    # and make output directories
    git clone https://github.com/binarybottle/optimize_layouts.git
    cd optimize_layouts
    chmod +x *.py *.sh
    mkdir -p output
    mkdir -p output/layouts
    mkdir -p output/outputs
    mkdir -p output/errors
    ```

  ### Generate config files and prepare to submit jobs
  You can generate configuration files in output/configs/ 
  by creating your own generate_configs.py script, 
  following the example in generate_keyboard_configs1.py:
  `python generate_configs.py`

  ### Set up slurm parameters:  
  Replace slurm parameters listed below according to your setup.
  Replace <ALLOCATION_ID> with the actual allocation ID,
  and <TOTAL_CONFIGS> with the total number of config files.
  
  slurm_array_processor.sh:

    ```bash
  # SLURM configuration
    #===================================================================
    #SBATCH --time=4:00:00       # Time limit per configuration
    #SBATCH --ntasks-per-node=1  # Number of tasks per node
    #SBATCH --cpus-per-task=2    # Number of CPUs per task   
    #SBATCH --mem=2GB            # Memory allocation
    #SBATCH --job-name=layout    # Job name
    #SBATCH --output=output/outputs/layout_%A_%a.out # Output file
    #SBATCH --error=output/errors/layout_%A_%a.err # Error file
    #SBATCH -p RM-shared         # Regular Memory-shared
    #SBATCH -A <ALLOCATION_ID>   # Your allocation ID (e.g., med250002p)
    #===================================================================
    ```

  slurm_quota_smart_array_submit.sh:

    ```bash
    # Configuration
    TOTAL_CONFIGS=<TOTAL_CONFIGS>      # Total configurations (e.g., 65520)
    BATCH_SIZE=1000                    # Configs per batch file
    ARRAY_SIZE=1000                    # Maximum array tasks per job
    MAX_CONCURRENT=500                 # Maximum concurrent tasks
    CHUNK_SIZE=4                       # Number of array jobs to submit at once
    config_pre=output/configs1/config_ # Config file path prefix
    config_post=.yaml                  # Config file suffix
    ```

  ### Run scripts

    ```bash
    # Use screen
    screen -S submission

    # Run all batches as a slurm job
    bash slurm_quota_smart_array_submit.sh [--rescan]
    ```

  ### Monitor jobs

    ```bash
    # Check all your running jobs
    squeue -u $USER

    # See how many jobs are running vs. pending
    squeue -j <job_array_id> | awk '{print $5}' | sort | uniq -c

    # Check number of output files
    sh count_files.sh output/layouts
    ```

  ### Cancel jobs

    ```bash
    # Cancel all your jobs at once
    scancel -u $USER

    # Cancel a specific job ID, such as:
    scancel <job_array_id>
    ```