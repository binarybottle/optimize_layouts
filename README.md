# README
Optimize layouts of items and item-pairs. 
===================================================================
https://github.com/binarybottle/optimize_layouts.git
Author: Arno Klein (arnoklein.info)
License: MIT License (see LICENSE)

## Usage
-------------------------------------------------------------------
  ```bash
  python normalize_input.py # Normalize input data
  python analyze_input.py   # Analyze and plot raw/normalized data
  python optimize_layout.py # Optimize layouts based on config file
  python analyze_results.py # Analyze resulting layouts
  ```
For running parallel processes, 
see **Running parallel processes** below.

## Context
-------------------------------------------------------------------
This code optimizes a layout, an arrangement of items,
and takes into account scores for individual items, 
item-pairs (a,b and b,a), positions, and position-pairs.
To efficiently prune vast combinatorial layouts, 
it uses a branch-and-bound algorithm.

The initial intended use-case is keyboard layout optimization 
for touch typing (see **keyboards/README_keyboards.md**):
  - Items and item-pairs correspond to letters and bigrams.
  - Positions and position-pairs correspond to keys and key-pairs.  
  - Item scores and item-pair scores correspond to frequency 
    of letters and frequency of bigrams in a given language.
  - Position scores and position-pair scores correspond to 
    a measure of comfort when typing single keys or pairs of keys 
    (in any language, as usage frequencies are regressed out).

## Inputs
-------------------------------------------------------------------
The input files are constructed as in the example below.
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

## Layout scoring
-------------------------------------------------------------------
Layouts are scored based on normalized item and item-pair scores 
and corresponding normalized position and position-pair scores, 
where direction (sequence of a given pair) matters.
Below, `I` is the number of items, `P` is the number of item_pairs:

    ```
    item_component = sum[i,I](item_score_i * position_score_i) / I
    
    item_pair_component = sum[j,P](
        item_pair_score_j_seq1 * position_pair_score_j_seq1 +
        item_pair_score_j_seq2 * position_pair_score_j_seq2) / P
 
    score = item_component * item_pair_component
    ```

To calculate the score for a specific layout:
  `python calculate_score.py --items "etaoi" --positions "FJDSV"`

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
  - Top-scoring layouts:
    - Item-to-position mappings
    - Total score and item and item-pair scores
  - Detailed command-line output and CSV file:
    - Configuration parameters
    - Search space statistics
    - Pruning statistics
    - Complete scoring breakdown
  - Optional visual mapping of layouts as a partial keyboard:
    - ASCII art visualization of the layout
    - Clear marking of constrained positions

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
    chmod +x generate_configs.py
    chmod +x slurm_batchmaking.sh
    chmod +x slurm_submit_batches.sh
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
  
  slurm_batchmaking.sh:

    ```bash
    #SBATCH --time=08:00:00       
    #SBATCH --array=0-999%1000       
    #SBATCH --ntasks-per-node=1   
    #SBATCH --cpus-per-task=2     
    #SBATCH --mem=2GB          
    #SBATCH --job-name=layouts     
    #SBATCH --output=output/outputs/layouts_%A_%a.out
    #SBATCH --error=output/errors/layouts_%A_%a.err
    #SBATCH -p RM-shared              
    #SBATCH -A <ALLOCATION_ID>   
    TOTAL_CONFIGS=<TOTAL_CONFIGS> 
    BATCH_SIZE=1000       
    ```

  slurm_submit_batches.sh:

    ```bash
    TOTAL_CONFIGS=<TOTAL_CONFIGS>
    BATCH_SIZE=1000                   
    CHUNK_SIZE=5                        
    ```

  ### Run scripts

    ```bash
    # Use screen
    screen -S submission

    # Run a single test job
    sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh

    # Run all batches as a slurm job
    sbatch --time=8:00:00 slurm_submit_batches.sh
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