#!/bin/bash
# CONFIGURATION ("EM" for Extreme Memory nodes on Bridges-2)
#===================================================================
#SBATCH -p RM-shared           # Regular Memory-shared    (EM: EM)   
#SBATCH --time=4:00:00         # Time limit for the job
#SBATCH --array=0-999%1000     # Array job with 1000 tasks per batch
#SBATCH --ntasks-per-node=1    # Number of tasks per node
#SBATCH --cpus-per-task=2      # Number of CPUs per task  (EM: 24)  
#SBATCH --mem=2GB              # Memory allocation        (EM: 512GB) 
#SBATCH --job-name=layouts     # Job name
#SBATCH --output=output/outputs/layouts_%A_%a.out # Output file
#SBATCH --error=output/errors/layouts_%A_%a.err   # Error file
#SBATCH -A <ALLOCATION_ID>     # Replace <ALLOCATION_ID> with your allocation ID
TOTAL_CONFIGS=<TOTAL_CONFIGS>  # Replace <TOTAL_CONFIGS> with total number of configurations
BATCH_SIZE=1000                # SLURM array limit per batch
config_pre=output/configs2/per_config/step2_from_config_  # config file path and prepend before ID (phase 1: output/configs1/config_)
config_post=_rank_1.yaml       # config file append after ID (phase 1: .yaml)
output_pre="output/layouts/layout_results_"
output_post="_*.csv"
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
echo "Configuration file: ${config_pre}${CONFIG_ID}${config_post}"

# Check if config file exists
if [ ! -f ${config_pre}${CONFIG_ID}${config_post} ]; then
    echo "Error: Configuration file ${config_pre}${CONFIG_ID}${config_post} not found!"
    exit 1
fi

# Define the output layout file path
LAYOUT_FILE="${output_pre}${CONFIG_ID}${output_post}"

# Check if the output file already exists
if [ -f "$LAYOUT_FILE" ]; then
    echo "Output layout file already exists for config ${CONFIG_ID}. Skipping optimization."
else
    echo "Output layout file not found. Running optimization for config ${CONFIG_ID}."
    # Run the optimization with the specific configuration
    python optimize_layout.py --config ${config_pre}${CONFIG_ID}${config_post}
    
    # Save completion status
    if [ $? -eq 0 ]; then
        echo "Job completed successfully at $(date)" > output/layouts/config_${CONFIG_ID}/job_completed.txt
    else
        echo "Job failed with error code $? at $(date)" > output/layouts/config_${CONFIG_ID}/job_failed.txt
    fi
fi

echo "Job finished at: $(date)"