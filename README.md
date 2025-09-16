Multi-Objective Keyboard Layout Optimizer

A streamlined framework for finding Pareto-optimal layouts using weighted multi-objective optimization.

**Repository**: https://github.com/binarybottle/optimize_layouts.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

This framework discovers Pareto-optimal layouts by optimizing multiple objectives simultaneously.
It combines direct key-pair score lookup with multi-objective search to find layouts that represent 
the best trade-offs across an arbitrary number of competing criteria.

## Multi-Objective Optimization (MOO)
  ```bash
    # Basic MOO with default settings in config.yaml (uses exhaustive search)
    python optimize_moo.py --config config.yaml

    # MOO with specific objectives (overrides config defaults)
    python optimize_moo.py --config config.yaml \
        --objectives engram_keys,engram_rows,engram_columns,engram_order

    # MOO with custom weights and directions
    python optimize_moo.py --config config.yaml \
        --objectives engram_keys,engram_rows,engram_columns,engram_order \
        --weights 1.0,2.0,0.5,0.75 --maximize true,true,false,true \
        --max-solutions 100 --time-limit 3600 --verbose

    # Force branch-and-bound search (faster for 12+ items, but more complex)
    python optimize_moo.py --config config.yaml \
        --objectives engram_keys,engram_rows,engram_columns,engram_order \
        --search-mode branch-bound

    # Validation run
    python optimize_moo.py --config config.yaml --validate --dry-run
  ```

## Analysis and (keyboard layout) visualization
  ```bash
    # Analyze optimization results
    python analyze_results.py --results-dir output/layouts

    # Compare input data (raw vs normalized)
    python analyze_input.py --config config.yaml

    # Visualize keyboard layout as ASCII
    python display_layout.py --letters "etaoinsrhl" --positions "FDESVRJKIL"

    # Generate rich HTML keyboard
    python display_layout.py --letters "etaoinsrhl" --positions "FDESVRJKIL" --html
  ```

## Architecture
optimize_layouts/
├── README.md                            # This file
│ 
│ # Configuration
├── config.yaml                          # Main configuration file
│ 
│ # Core MOO system
├── optimize_moo.py                      # Main entry point for optimization
├── moo_scoring.py                       # Item-pair-weighted MOO scoring
├── moo_search.py                        # Pareto-optimal search algorithms
├── config.py                            # Configuration management
├── consolidate_moo.py                   # Select global Pareto solutions from separate files
├── visualize_moo.py                     # Analyze and visualize MOO results
│
│ # I/O (inputs are for a keyboard optimization study)
├── input/
│   ├── engram_2key_scores.csv          # Position-pair scoring (bigram)
│   ├── engram_3key_order_scores.csv    # Position-triple scoring (trigram)
│   └── frequency/
│       ├── english-letter-pair-counts-google-ngrams_normalized.csv
│       └── english-letter-triple-counts-google-ngrams_normalized.csv
├── run_jobs.py                          # Run multiple jobs
├── output/                              
│   └── layouts/                         # MOO optimization results
│
│ # Optional utilities
├── utilities/
│   ├── normalize_and_analyze.py         # Normalize data and compare raw vs normalized data
│   ├── generate_command.py              # Generate command to score and visualize layouts
│   ├── run_jobs_slurm.sh                # SLURM cluster job submission
│   ├── calc_positions_items.py          # Calculate permutations
│   └── count_files.py                   # Count files in a folder (with find command)
│
│ # Keyboard study
└── keyboard_study/
    ├── README_keyboards.md              # README for keyboard layout optimization study
    ├── analyze_frequencies.py           # Analyze item-pair (bigram) frequencies
    ├── generate_configs1.py             # Generate config files for parallel optimization phase 1
    ├── generate_configs1.py             # Generate config files for parallel optimization phase 2
    └── display_layout.py                # Keyboard visualization


## Inputs
The system accepts normalized score files in CSV format.
The code will accept any set of letters (or special characters) 
to represent item-pairs and position-pairs, 
so long as each item-pair is represented by two item characters 
and each position-pair is represented by two position characters.

  ### Normalized scores
  Item-pair (normalized) scores file:      
    ```csv
      item_pair, score
      ab, 0.1
      ba, 0.3
      ac, 0.2       
      ca, 0.1       
      bc, 0.4       
      cb, 0.2       
    ```
  Position-pair (normalized multi-objective) scores file:       
    ```csv
      position_pair,engram_keys,engram_rows,engram_columns,engram_order
      FD,0.85,0.72,0.91,0.78,0.65,0.88
      DE,0.72,0.68,0.88,0.82,0.71,0.85
      ES,0.68,0.71,0.85,0.75,0.69,0.82
    ```

## Multi-objective search algorithms
The system provides two Pareto-optimal search algorithms with different performance characteristics:
Exhaustive Search (Default)

Guaranteed complete: Finds ALL Pareto-optimal solutions
Simple and reliable: No complex pruning logic or upper bound calculations
Optimal for ≤10 items: Faster than branch-and-bound for smaller problems
Direct evaluation: Tests every possible permutation with minimal overhead

## Branch-and-Bound Search
Intelligent pruning: Uses upper bounds to eliminate dominated solution branches
Optimal for ≥12 items: Essential when factorial growth becomes prohibitive
Complex but scalable: More overhead per node but massive savings on large problems
Preserves optimality: Mathematically sound pruning ensures no optimal solutions are lost

## Bigram and Trigram Scoring
The system supports both bigram (2-key) and trigram (3-key) objective scoring; 
it automatically detects which objectives are bigram vs trigram based on input scoring tables.

- **Bigram objectives**: Scored using position-pair table (e.g., `engram_keys,engram_rows,engram_columns,engram_order`,...)
- **Trigram objectives**: Scored using position-triple table (e.g., `engram_3key_order`)
- **Weighting**: Bigram and trigram scores are weighted by item-pair/triple scores

## Output
  **Console output**
  - Configuration summary and search space analysis
  - Pareto front analysis with objective ranges
  - Top solutions with detailed score breakdowns
  - Performance metrics and timing statistics
  - Optional ASCII art visualization of keyboard layouts

  **CSV results**
  - Complete Pareto front with all objectives: moo_results_config_YYYYMMDD_HHMMSS.csv

  **Analysis outputs**
  - Objective correlation plots and scatter visualizations
  - Input data distribution comparisons
  - Global Pareto front analysis across multiple runs

## Running many configurations
You can generate configuration files
by creating your own generate_configs.py script,
following the example in keyboards/generate_configs1.py,
then modify the run_jobs_local.py script for your needs
(see run_jobs_slurm.sh for running jobs on a linux cluster):

```bash
  python generate_configs.py
  python run_jobs_local.py
```
