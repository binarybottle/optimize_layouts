Multi-Objective Keyboard Layout Optimizer

A streamlined framework for finding Pareto-optimal layouts using frequency-weighted multi-objective optimization.

**Repository**: https://github.com/binarybottle/optimize_layouts.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

This framework discovers Pareto-optimal layouts by optimizing multiple objectives simultaneously using English bigram frequency weighting. It combines direct keypair score lookup with multi-objective search to find layouts that represent the best trade-offs across an arbitrary number of competing criteria.

## Basic Multi-Objective Optimization
  ```bash
    # Basic MOO with default settings in config.yaml
    python optimize_moo.py --config config.yaml \\

    # Basic MOO with six of the Engram-7 objectives
    python optimize_moo.py --config config.yaml \\
        --objectives engram7_load,engram7_strength,engram7_position,engram7_vspan,engram7_hspan,engram7_sequence \\

    # With custom settings
    python optimize_moo.py --config config.yaml \\
        --objectives engram7_load,engram7_strength,engram7_position,engram7_vspan,engram7_hspan,engram7_sequence \\
        --keypair-table input/keypair_scores_detailed.csv \\
        --frequency-file input/custom_frequencies.csv \\
        --weights 1.0,2.0,0.5,0.5,0.5,0.5 --maximize true,true,false,false,false,false
        --max-solutions 50 --time-limit 1800

    # Validation run
    python optimize_moo.py --config config.yaml \\
        --validate --dry-run
  ```

## Optional visualization for keyboard layouts
  ```bash
    # Visualize keyboard layout as ASCII
    python display_layout.py --letters "etaoinsrhl" --positions "FDESVRJKIL"

    # Generate rich HTML keyboard
    python display_layout.py --letters "etaoinsrhl" --positions "FDESVRJKIL" --html

    # Show empty key positions
    python display_layout.py --letters "etao" --positions "FDES" --show-empty
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
├── # Optional Utilities
├── display_layout.py                  # Rich keyboard visualization
├── normalize_input.py                 # Input data normalization
│
├── # Configuration
├── config.yaml                        # Main configuration file
│
├── # Data Directories
├── tables/
│   ├── keypair_scores_detailed.csv    # Unified scoring table (required)
│   └── english-letter-pair-frequencies-google-ngrams.csv
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
      key_pair,comfort_score,time_total,accuracy_score,engram8_columns,engram8_curl
      qw,0.85,0.23,0.91,0.78,0.65
      we,0.72,0.31,0.88,0.82,0.71
      er,0.68,0.28,0.85,0.75,0.69
    ```

## Branch-and-bound optimization
  - Calculates exact scores for placed items
  - Uses provable upper bounds for unplaced items
  - Prunes branches that cannot exceed best known solution
  - Depth-first search maintains optimality & reduces search space
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
  - MOO: moo_results_config_YYYYMMDD_HHMMSS.csv

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
