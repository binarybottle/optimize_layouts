# README
Optimize layouts of items and item-pairs. 
===================================================================
https://github.com/binarybottle/optimize_layouts.git
Author: Arno Klein (arnoklein.info)
License: MIT License (see LICENSE)

## Usage

  ```bash
  python optimize_layout.py
  python analyze_results.py --top 5 --config config.yaml
  ```
For running parallel processes, 
see **Running parallel processes** below.

## Context
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
  - Position scores and position-pair scores correspond to a measure 
    of comfort when typing single keys or pairs of keys (any language).

## Inputs
The input files are constructed as in the example below.
The code will accept any set of letters (and special characters) 
to represent items, item pairs, positions, and position pairs, 
so long as item pair characters are composed of two item characters 
and position pair characters are composed of two position characters.

  - item_scores_file:           
    ```
    item, score
    a, 0.1
    b, 0.5
    c, 0.2       
    ```
  - item_pair_scores_file:      
    ```
    item_pair, score
    ab, 0.1
    ba, 0.3
    ac, 0.2       
    ca, 0.1       
    bc, 0.4       
    cb, 0.2       
    ```
  - position_scores_file:       
    ```
    position, score
    X, 0.4
    Y, 0.3
    Z, 0.1       
    ```
  - position_pair_scores_file:  
    ```
    position_pair, score
    XY, 0.2
    YX, 0.3
    XZ, 0.2
    ZX, 0.5
    YZ, 0.4
    ZY, 0.5
    ```

A configuration file (config.yaml) specifies input filenames,
as well as text strings for the following variables:
  - Required:
    - items_to_assign: items to arrange in positions_to_assign
    - positions_to_assign: available positions
  - Optional:
    - items_assigned: items already assigned to positions
    - positions_assigned: positions that are already filled 
    - items_to_constrain: subset of items_to_assign 
      to arrange within positions_to_constrain
    - positions_to_constrain: subset of positions_to_assign 
      to constrain items_to_constrain

## Layout scoring
Layouts are scored based on item and item_pair scores 
and corresponding position and position_pair scores, 
where direction (sequence of a given pair) matters.
Below, I is the number of items, P is the number of item_pairs:

    item_component = sum[i,I](item_score_i * position_score_i) / I
    
    item_pair_component = sum[j,P](
        item_pair_score_j_sequence1 * position_pair_score_j_sequence1 +
        item_pair_score_j_sequence2 * position_pair_score_j_sequence2) / P
 
    score = item_weight * item_component + item_pair_weight * item_pair_component

To calculate the score for a specific layout, you can use calculate_score.py:
  `python calculate_score.py --items "etaoi" --positions "FJDSV"`

## Branch-and-bound optimization
  - Calculates exact scores for placed letters
  - Uses provable upper bounds for unplaced letters
  - Prunes branches that cannot exceed best known solution
  - Depth-first search maintains optimality while reducing search space
  - Uses numba-optimized array operations
  - Detailed progress tracking and statistics
  - Comprehensive validation of input configurations
  - Optional constraints for a subset of items

## Output
  - Top-scoring layouts:
    - Item-to-position mappings
    - Total score and unweighted item and item-pair scores
  - Detailed command-line output and CSV file:
    - Configuration parameters
    - Search space statistics
    - Pruning statistics
    - Complete scoring breakdown
  - Optional visual mapping of layouts as a partial keyboard:
    - ASCII art visualization of the layout
    - Clear marking of constrained positions

## Running parallel processes on SLURM
The examples below will use Pittsburgh Super Computing Bridges-2 resources.
NSF ACCESS granted access for a keyboard layout optimization study 
(see **README_keyboards**). 

  ### Connect and set up the code environment
    ```bash
    # Log in
    ssh username@bridges2.psc.edu

    # Create a directory for the project
    mkdir -p keyboard_optimizer
    cd keyboard_optimizer

    # Load Python module
    module load python/3.8.6

    # Create a virtual environment and make the activate script executable
    python -m venv keyboard_env
    source keyboard_env/bin/activate
    chmod +x $HOME/keyboard_optimizer/keyboard_env/bin/activate

    # Install required packages
    pip install pyyaml numpy pandas tqdm numba psutil matplotlib
  
      # Clone the repository
    git clone https://github.com/binarybottle/optimize_layouts.git
    cd optimize_layouts

    # Make the scripts executable
    chmod +x generate_configs.py
    chmod +x slurm_batchmaking.sh
    chmod +x slurm_submit_batches.sh

    # Make output directories if they don't exist
    mkdir -p output
    mkdir -p output/layouts
    mkdir -p output/outputs
    mkdir -p output/errors
    ```

  ### Generate config files and prepare to submit jobs
    ```bash
    # You can generate configuration files by creating your own
    # generate_configs.py script, following the example in 
    # generate_keyboard_configs1.py:
    python generate_configs.py

    # Replace <YOUR_ALLOCATION_ID> with your actual allocation ID
    # In the code below, replace abc123 with your allocation ID
    # (you can find this using the 'projects' command):
    sed -i 's/<YOUR_ALLOCATION_ID>/abc123/g' slurm_batchmaking.sh

    # Make any other changes needed in the slurm scripts, 
    # and don't forget to update TOTAL_CONFIGS (number of config files) 
    # in slurm_submit_batches.sh and slurm_batchmaking.sh
    ```

  ### Run scripts in ```screen```
    ```bash
    # Use screen
    screen -S submission

    # Run a single test job
    sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh
    # Check the output files to see if any error messages are generated
    ls -la output/outputs/
    cat output/outputs/layouts_*.out

    # Run just one batch (1001-2000) manually:
    sbatch --export=BATCH_NUM=1 slurm_batchmaking.sh

    # Run all batches as a slurm job
    sbatch --time=00:10:00 slurm_submit_batches.sh
    ```

  ### Monitor jobs
    ```bash
    # See the current progress
    cat batch_submission_progress.txt

    # Check the log files
    ls -la submission_logs/

    # Check all your running jobs
    squeue -u $USER

    # Check a specific job array status
    squeue -j <job_array_id>

    # See how many jobs are running vs. pending
    squeue -j <job_array_id> | awk '{print $5}' | sort | uniq -c

    # Check if there are any failed jobs
    ls -la output/layouts/config_*/job_failed.txt | wc -l
    # For very large directories:
    for i in {1..10}; do 
      find output/layouts/config_${i}* -name "job_failed.txt" | wc -l
    done | awk '{sum+=$1} END {print sum}'

    # View detailed information about a job
    scontrol show job <job_id>

    # Check the number of completed jobs
    ls -la output/layouts/config_*/job_completed.txt | wc -l
    # For very large directories:
    for i in {1..10}; do 
      find output/layouts/config_${i}* -name "job_completed.txt" | wc -l
    done | awk '{sum+=$1} END {print sum}'

    # Check which specific jobs have completed
    ls -la output/layouts/config_*/job_completed.txt | head -10
    ```

  ### Cancel jobs
    ```bash
    # Cancel a specific job ID, such as:
    scancel 29556406_1

    # Cancel all your jobs at once
    scancel -u $USER
    ```