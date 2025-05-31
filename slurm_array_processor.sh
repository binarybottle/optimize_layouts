#!/bin/bash
# Process a single configuration as an array task
# This script is intended to be run as a SLURM array job, 
# called by slurm_array_submit.sh.
# It calls optimize_layout.py for each configuration.

# SLURM configuration - static parameters only
#===================================================================
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --job-name=layout           # Job name
#SBATCH --output=output/outputs/layout_%A_%a.out # Output file with array job and task IDs
#SBATCH --error=output/errors/layout_%A_%a.err   # Error file with array job and task IDs
#===================================================================

# Configuration - can be overridden by environment variables
config_pre=${CONFIG_PREFIX:-output/configs1/config_}  # Config file path prefix
config_post=${CONFIG_SUFFIX:-.yaml}                   # Config file suffix

# The batch file must be provided when submitting
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE not set. Use: sbatch --export=CONFIG_FILE=X array_processor.sh"
    exit 1
fi

# Read the configuration IDs from the batch file
mapfile -t CONFIG_IDS < "$CONFIG_FILE"
TOTAL_CONFIGS=${#CONFIG_IDS[@]}

# Calculate which config ID to process based on array task ID
if [ $SLURM_ARRAY_TASK_ID -ge $TOTAL_CONFIGS ]; then
    echo "Array task ID ($SLURM_ARRAY_TASK_ID) exceeds number of configurations ($TOTAL_CONFIGS). Nothing to do."
    exit 0
fi

CONFIG_ID=${CONFIG_IDS[$SLURM_ARRAY_TASK_ID]}
echo "Array task $SLURM_ARRAY_TASK_ID processing configuration ID: $CONFIG_ID"

# Load required modules
module purge
module load anaconda3

echo "=== Environment Info ==="
echo "Hostname: $(hostname)"
echo "Python: $(python3 --version)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE"
module list

# Set environment for HPC parallel processing
export OMP_NUM_THREADS=1              # Prevent thread oversubscription
export PYTHONHASHSEED=0               # Deterministic hashing
export NUMBA_NUM_THREADS=1            # Control Numba threading

# Set working directory
cd $HOME/keyboard_optimizer/optimize_layouts

# Check if config file exists
if [ ! -f "${CONFIG_PREFIX}${CONFIG_ID}${CONFIG_SUFFIX}" ]; then
    echo "Configuration file ${CONFIG_PREFIX}${CONFIG_ID}${CONFIG_SUFFIX} not found!"
    exit 1
fi

# Check if output already exists
FIND_PATTERN="*results_${CONFIG_ID}_[0-9]*"
if find output/layouts -name "$FIND_PATTERN*.csv" | grep -q .; then
    echo "Output file already exists for config ${CONFIG_ID}. Skipping optimization."
    exit 0
fi

# Run the PROVEN optimization with passed parameters
echo "Running optimization for config ${CONFIG_ID} using optimize_layout.py..."
echo "  Mode: ${MODE:---moo}"
echo "  Processes: ${PROCESSES:-$SLURM_CPUS_PER_TASK}"
if [ -n "$MAX_SOLUTIONS" ]; then echo "  Max solutions: $MAX_SOLUTIONS"; fi
if [ -n "$N_SOLUTIONS" ]; then echo "  N solutions: $N_SOLUTIONS"; fi
if [ -n "$TIME_LIMIT" ]; then echo "  Time limit: ${TIME_LIMIT}s"; fi

# ✅ CHANGE THIS LINE: Use your proven script instead of parallel_optimize_layout.py
python3 optimize_layout.py \
    --config ${CONFIG_PREFIX}${CONFIG_ID}${CONFIG_SUFFIX} \
    ${MODE:---moo} \
    --processes ${PROCESSES:-$SLURM_CPUS_PER_TASK} \
    ${MAX_SOLUTIONS:+--max-solutions $MAX_SOLUTIONS} \
    ${TIME_LIMIT:+--time-limit $TIME_LIMIT} \
    ${N_SOLUTIONS:+--n-solutions $N_SOLUTIONS}

if [ $? -eq 0 ]; then
    echo "Optimization completed successfully for config ${CONFIG_ID}"
else
    echo "Optimization failed with error code $? for config ${CONFIG_ID}"
    exit 1
fi