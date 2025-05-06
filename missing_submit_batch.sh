#!/bin/bash
# missing_submit_batch.sh

# SLURM configuration
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --job-name=missing_batch
#SBATCH --output=output/outputs/missing_batch_%A_%a.out
#SBATCH --error=output/errors/missing_batch_%A_%a.err
#SBATCH -p RM-shared
#SBATCH -A <YOUR_ALLOCATION>

# The batch file must be provided when submitting:
# sbatch --export=BATCH_FILE=missing_batches/batch_1.txt submit_batch.sh
if [ -z "$BATCH_FILE" ]; then
    echo "Error: BATCH_FILE not set. Use: sbatch --export=BATCH_FILE=missing_batches/batch_X.txt"
    exit 1
fi

# Get the config ID from the batch file based on array task ID
CONFIG_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $BATCH_FILE)

if [ -z "$CONFIG_ID" ]; then
    echo "Error: No config ID found in $BATCH_FILE for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Processing CONFIG_ID: $CONFIG_ID"

# Load required modules
module purge
module load python/3.8.6

# Use the Python directly from the virtual environment
export PATH="$HOME/keyboard_optimizer/keyboard_env/bin:$PATH"
export PYTHONPATH="$HOME/keyboard_optimizer:$PYTHONPATH"

# Set working directory
cd $HOME/keyboard_optimizer/optimize_layouts

# Set config file paths
config_file="output/configs1/config_${CONFIG_ID}.yaml"

# Check if config file exists
if [ ! -f $config_file ]; then
    echo "Error: Configuration file $config_file not found!"
    exit 1
fi

# Run the optimization with the specific configuration
python optimize_layout.py --config $config_file

echo "Job finished at: $(date)"