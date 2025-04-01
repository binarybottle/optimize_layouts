# README
Optimize layouts of items and item-pairs. 
===================================================================

https://github.com/binarybottle/optimize_layouts.git

Author: Arno Klein (binarybottle.com)

## Context
This code uses a branch-and-bound algorithm to find optimal positions 
for items by jointly considering two scoring components: 
item/item_pair scores and position/position_pair scores (see "Layout scoring").

The initial intended use-case is keyboard layout optimization for touch typing
(see **README_keyboard_application.md**):
  - Items and item-pairs correspond to letters and bigrams.
  - Positions and position-pairs correspond to keys and key-pairs.  
  - Item scores and item-pair scores correspond to frequency of occurrence 
    of letters and frequency of bigrams in a given language.
  - Position scores and position-pair scores correspond to a measure of comfort  
    when typing single keys or pairs of keys (for any language).

## Layout scoring
Layouts are scored based on item and item_pair scores 
and corresponding position and position_pair scores,
where below, I is the number of items, and P is the number of item_pairs:

    item_component = sum_i_I(item_score_i * position_score_i) / I
    
    item_pair_component = sum_j_P(
        item_pair_score_j_sequence1 * position_pair_score_j_sequence1 +
        item_pair_score_j_sequence2 * position_pair_score_j_sequence2) / P
 
    score = item_weight * item_component + item_pair_weight * item_pair_component

Scoring System:
  - Calculates separate item and item-pair scores
  - Distribution detection and normalization of all scores to 0-1 range
  - Considers both directions for each item-pair
  - Provides detailed scoring breakdowns

Branch and Bound Optimization:
  - Calculates exact scores for placed letters
  - Uses provable upper bounds for unplaced letters
  - Prunes branches that cannot exceed best known solution
  - Depth-first search maintains optimality while reducing search space
  - Uses numba-optimized array operations
  - Detailed progress tracking and statistics
  - Comprehensive validation of input configurations
  - Optional constraints for a subset of items

## Setup
The code expects four input files with the following column names,
where the 1st column is a letter or letter-pair, and the 2nd column is a number:
  - item_scores_file:           item, score       
  - item_pair_scores_file:      item_pair, score
  - position_scores_file:       position, score
  - position_pair_scores_file:  position_pair, score

The code will accept any set of letters (and special characters) 
to represent items, item pairs, positions, and position pairs, 
so long as item pair characters are composed of two item characters 
and position pair characters are composed of two position characters.

A configuration file (config.yaml) specifies these filenames,
as well as text strings that represent items/positions to arrange or constrain:
  - Required:
    - items_to_assign: items to arrange in positions_to_assign
    - positions_to_assign: available positions
  - Optional:
    - items_assigned: items already assigned to positions
    - positions_assigned: positions that are already filled 
    - items_to_constrain: subset of items_to_assign to arrange within positions_to_constrain
    - positions_to_constrain: subset of positions_to_assign to constrain items_to_constrain

## Output
  - Top-scoring layouts:
    - Item-to-position mappings
    - Total score and unweighted item and item-pair scores
  - Detailed command-line output and CSV file:
    - Configuration parameters
    - Search space statistics
    - Pruning statistics
    - Complete scoring breakdown
  - Progress updates during execution:
    - Permutations processed per second
    - Estimated time remaining
  - Optional visual mapping of layouts as a partial keyboard:
    - ASCII art visualization of the layout
    - Clear marking of constrained positions

## Running parallel processes on an NSF ACCESS cluster

### Set up the code environment on Bridges-2
```
# Log in to Bridges-2
ssh username@bridges2.psc.edu

# Create a directory for the project
mkdir -p keyboard_optimizer
cd keyboard_optimizer

# Clone the repository if git is available
git clone https://github.com/binarybottle/optimize_layouts.git

# If git is not available, create the file structure manually
mkdir -p optimize_layouts
mkdir -p optimize_layouts/input
mkdir -p optimize_layouts/output/layouts
mkdir -p optimize_layouts/output/outputs
mkdir -p optimize_layouts/output/errors
```

### Install required Python packages
```
# Load Python module on Bridges-2
module load python/3.8.6

# Create a virtual environment
python -m venv keyboard_env
source keyboard_env/bin/activate

# Install required packages
pip install pyyaml numpy pandas tqdm numba psutil matplotlib
```

### Run the scripts on Bridges-2
```
cd ~/keyboard_optimizer/optimize_layouts

# Make the scripts executable
#chmod +x generate_configs.py
chmod +x slurm_batchmaking.sh
chmod +x slurm_submit_batches.sh

# Generate configuration files
#python generate_configs.py

# Replace <YOUR_ALLOCATION_ID> with your actual allocation ID
# In the code below, replace abc123 with your allocation ID
# (you can find this using the 'projects' command):
sed -i 's/<YOUR_ALLOCATION_ID>/abc123/g' slurm_batchmaking.sh

# Update the number of config files in:
#   slurm_batchmaking.sh: MAX_CONFIG
#   slurm_submit_batches.sh: TOTAL_CONFIGS

# Submit the job to the scheduler
sbatch slurm_submit_batches.sh

# Alternatively, if you want to run just one batch (1001-2000) manually:
sbatch --export=BATCH_NUM=1 slurm_batchmaking.sh
```

### Monitoring jobs
```
# Check all your running jobs
squeue -u $USER

# Check a specific job array status
squeue -j <job_array_id>

# See how many jobs are running vs. pending
squeue -j <job_array_id> | awk '{print $5}' | sort | uniq -c

# View detailed information about a job
scontrol show job <job_id>
```

### Cancel jobs
```
# Cancel a specific job ID
scancel 29556406_1

# Cancel all your jobs at once
scancel -u $USER
```

### Analyze the results
```
# Check if all jobs are done
ls -l output/layouts/config_*/job_completed.txt | wc -l

# Run the analysis
python analyze_results.py --top 10
```