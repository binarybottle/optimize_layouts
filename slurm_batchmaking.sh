#!/bin/bash
# CONFIGURATION 
#===================================================================
#SBATCH --time=08:00:00            # Time limit (max hours per job)
#SBATCH --array=0-999%1000         # Job array: 0-999 indices (use %1000 to limit concurrent jobs)
#SBATCH --ntasks-per-node=1        # Run one task per job
#SBATCH --cpus-per-task=2          # Cores per optimization process
#SBATCH --mem=2GB                  # Memory per job
#SBATCH --job-name=layouts         # Job name
#SBATCH --output=output/outputs/layouts_%A_%a.out # Output file with job and array IDs
#SBATCH --error=output/errors/layouts_%A_%a.err   # Error file with job and array IDs
#SBATCH -p RM-shared               # Partition (queue) - use shared to save SUs
#SBATCH -A abc123                  # Replace abc123 with your allocation ID
TOTAL_CONFIGS=720720  # Total number of configurations
BATCH_SIZE=1000       # SLURM array limit per batch
#===================================================================

# The batch number must be provided when submitting:
# sbatch --export=BATCH_NUM=0 batch_slurm_mapping.sh
if [ -z "$BATCH_NUM" ]; then
    echo "Error: BATCH_NUM not set. Use: sbatch --export=BATCH_NUM=X slurm_batchmaking.sh"
    exit 1
fi

# Calculate the actual config ID based on batch number and array task ID
CONFIG_ID=$((BATCH_NUM * BATCH_SIZE + SLURM_ARRAY_TASK_ID + 1))  # +1 to convert from 0-based to 1-based

# Check if the CONFIG_ID is within valid range
if [ $CONFIG_ID -gt $TOTAL_CONFIGS ]; then
    echo "Warning: CONFIG_ID $CONFIG_ID exceeds maximum of $TOTAL_CONFIGS. Exiting."
    exit 0
fi

# Load required modules
module purge
module load python/3.8.6

# Use the Python directly from the virtual environment
export PATH="$HOME/keyboard_optimizer/keyboard_env/bin:$PATH"
export PYTHONPATH="$HOME/keyboard_optimizer:$PYTHONPATH"

# Set working directory
cd $HOME/keyboard_optimizer/optimize_layouts

# Create a directory for this specific job's output
mkdir -p output/layouts/config_${CONFIG_ID}

# Echo job information
echo "Starting job array ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Processing config ID: ${CONFIG_ID}"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Configuration file: output/configs/config_${CONFIG_ID}.yaml"

# Check if config file exists
if [ ! -f output/configs/config_${CONFIG_ID}.yaml ]; then
    echo "Error: Configuration file output/configs/config_${CONFIG_ID}.yaml not found!"
    exit 1
fi

# Run the optimization with the specific configuration
python optimize_layout.py --config output/configs/config_${CONFIG_ID}.yaml 

# Save completion status
if [ $? -eq 0 ]; then
    echo "Job completed successfully at $(date)" > output/layouts/config_${CONFIG_ID}/job_completed.txt
else
    echo "Job failed with error code $? at $(date)" > output/layouts/config_${CONFIG_ID}/job_failed.txt
fi

echo "Job finished at: $(date)"
