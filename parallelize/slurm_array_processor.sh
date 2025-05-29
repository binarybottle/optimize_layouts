#!/bin/bash
# Process a single configuration as an array task
# This script is intended to be run as a SLURM array job, 
# called by slurm_quota_smart_array_submit.sh

# SLURM configuration
#===================================================================
#SBATCH --time=4:00:00              # Time limit per configuration
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=2           # Number of CPUs per task   
#SBATCH --mem=2GB                   # Memory allocation
#SBATCH --job-name=layout           # Job name
#SBATCH --output=output/outputs/layout_%A_%a.out   # Output file with array job and task IDs
#SBATCH --error=output/errors/layout_%A_%a.err    # Error file with array job and task IDs
#SBATCH -p RM-shared                # Regular Memory-shared
#SBATCH -A <ALLOCATION_ID>          # Your allocation ID (e.g., med250002p)
#===================================================================

# Configuration
config_pre=output/configs1/config_  # Config file path prefix
config_post=.yaml                   # Config file suffix

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
module load python/3.8.6

# Use the Python directly from the virtual environment
export PATH="$HOME/keyboard_optimizer/keyboard_env/bin:$PATH"
export PYTHONPATH="$HOME/keyboard_optimizer:$PYTHONPATH"

# Set working directory
cd $HOME/keyboard_optimizer/optimize_layouts

# Check if config file exists
if [ ! -f "${config_pre}${CONFIG_ID}${config_post}" ]; then
    echo "Configuration file ${config_pre}${CONFIG_ID}${config_post} not found!"
    exit 1
fi

# Check if output already exists
FIND_PATTERN="layout_results_${CONFIG_ID}_[0-9]*"
if find output/layouts -name "$FIND_PATTERN*.csv" | grep -q .; then
    echo "Output file already exists for config ${CONFIG_ID}. Skipping optimization."
    exit 0
fi

# Run the optimization
echo "Running optimization for config ${CONFIG_ID}..."
python optimize_layout.py --config ${config_pre}${CONFIG_ID}${config_post} --moo

if [ $? -eq 0 ]; then
    echo "Optimization completed successfully for config ${CONFIG_ID}"
else
    echo "Optimization failed with error code $? for config ${CONFIG_ID}"
    exit 1
fi
