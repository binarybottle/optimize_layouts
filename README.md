Multi-Objective Keyboard Layout Optimizer

A streamlined framework for finding Pareto-optimal layouts using frequency-weighted multi-objective optimization.

**Repository**: https://github.com/binarybottle/optimize_layouts.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

This framework discovers Pareto-optimal layouts by optimizing multiple objectives simultaneously using English bigram frequency weighting. It combines direct keypair score lookup with multi-objective search to find layouts that represent the best trade-offs across an arbitrary number of competing criteria.

## Multi-Objective Optimization (MOO)
  ```bash
    # Basic MOO with default settings in config.yaml
    python optimize_moo.py --config config.yaml

    # MOO with specific objectives (overrides config defaults)
    python optimize_moo.py --config config.yaml \
        --objectives engram7_load,engram7_strength,engram7_position,engram7_vspan,engram7_hspan,engram7_sequence

    # MOO with custom weights and directions
    python optimize_moo.py --config config.yaml \
        --objectives engram7_load,engram7_strength,engram7_position \
        --weights 1.0,2.0,0.5 --maximize true,true,false \
        --max-solutions 50 --time-limit 1800

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
├── README.md                          # This file
│ 
├── # Core MOO System
├── optimize_moo.py                    # Main entry point for optimization
├── moo_scoring.py                     # Frequency-weighted MOO scoring
├── moo_search.py                      # Pareto-optimal search algorithms
├── config.py                          # Configuration management
│
├── # Analysis and Utilities
├── normalize_input.py                 # Input data normalization
├── analyze_input.py                   # Compare raw vs normalized input data
├── analyze_results.py                 # Analyze MOO results with visualizations
├── select_global_moo_solutions.py     # Global Pareto front selection
├── analyze_global_moo_solutions.py    # Global Pareto analysis across files
├── display_layout.py                  # Rich keyboard visualization
│
├── # Job Running Scripts
├── run_jobs_local.py                  # Local parallel job execution
├── run_jobs_slurm.sh                  # SLURM cluster job submission
│
├── # Configuration
├── config.yaml                        # Main configuration file
│
├── # Data Directories
├── input/
│   ├── keypair_engram7_scores.csv     # Position-pair scoring table (required)
│   └── frequency/
│       └── normalized-english-letter-pair-counts-google-ngrams.csv
│
└── output/                            # Generated results
    └── layouts/                       # MOO optimization results

## Inputs
The system accepts normalized score files in CSV format.
The code will accept any set of letters (or special characters) 
to represent item-pairs and position-pairs, 
so long as each item-pair is represented by two item characters 
and each position-pair is represented by two position characters.

  ### Normalized scores
  Item-pair (normalized frequency) scores file:      
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
      position_pair,engram7_load,engram7_strength,engram7_position,engram7_vspan,engram7_hspan,engram7_sequence
      FD,0.85,0.72,0.91,0.78,0.65,0.88
      DE,0.72,0.68,0.88,0.82,0.71,0.85
      ES,0.68,0.71,0.85,0.75,0.69,0.82
    ```

## Multi-objective search algorithm
  - Pareto-optimal search finds non-dominated solutions across multiple objectives
  - Branch-and-bound optimization with exact score calculation and pruning
  - Frequency-weighted scoring using English bigram frequencies
  - Constraint handling for partial assignments and position restrictions
  - Progress tracking with statistics and time limits

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
