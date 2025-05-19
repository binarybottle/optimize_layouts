#!/bin/bash
# slurm_batch_processor.sh - Process configs from a batch file

# SLURM configuration
#===================================================================
#SBATCH --time=4:00:00              # Time limit for the job
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=2           # Number of CPUs per task   
#SBATCH --mem=2GB                   # Memory allocation
#SBATCH --job-name=layouts          # Job name
#SBATCH --output=output/outputs/layout_batch_%j.out # Output file
#SBATCH --error=output/errors/layout_batch_%j.err   # Error file
#SBATCH -p RM-shared                # Regular Memory-shared
#SBATCH -A med250002p               # Your allocation ID
#===================================================================

# Configuration
config_pre=output/configs1/config_  # Config file path prefix
config_post=.yaml                   # Config file suffix
output_pre=output/layouts/layout_results_  # Output file prefix
output_post=_$(date +%Y%m%d_%H%M%S).csv    # Output file suffix with timestamp

# Check if CONFIG_FILE is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE not set. Use: sbatch --export=CONFIG_FILE=X slurm_batch_processor.sh"
    exit 1
fi

# Check if the file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found"
    exit 1
fi

# Read configuration IDs from the file
mapfile -t CONFIG_IDS < "$CONFIG_FILE"
TOTAL_CONFIGS=${#CONFIG_IDS[@]}

echo "Starting batch processor for $TOTAL_CONFIGS configurations"
echo "Running on node: $(hostname)"
echo "Batch file: $CONFIG_FILE"
echo "Start time: $(date)"

# Load required modules
module purge
module load python/3.8.6

# Use the Python directly from the virtual environment
export PATH="$HOME/keyboard_optimizer/keyboard_env/bin:$PATH"
export PYTHONPATH="$HOME/keyboard_optimizer:$PYTHONPATH"

# Set working directory
cd $HOME/keyboard_optimizer/optimize_layouts

# Process each configuration
PROCESSED=0
SKIPPED=0
FAILED=0

for ((i=0; i<TOTAL_CONFIGS; i++)); do
    CONFIG_ID=${CONFIG_IDS[i]}
    
    echo "[$((i+1))/$TOTAL_CONFIGS] Processing configuration ID: $CONFIG_ID"
    
    # Check if config file exists
    if [ ! -f "${config_pre}${CONFIG_ID}${config_post}" ]; then
        echo "  Config file not found, skipping"
        FAILED=$((FAILED+1))
        continue
    fi
    
    # Double-check if output already exists
    FIND_PATTERN="layout_results_${CONFIG_ID}_[0-9]*"
    if find output/layouts -name "$FIND_PATTERN*.csv" | grep -q .; then
        echo "  Output file already exists, skipping"
        SKIPPED=$((SKIPPED+1))
        continue
    fi
    
    # Run the optimization
    echo "  Running optimization..."
    python optimize_layout.py --config ${config_pre}${CONFIG_ID}${config_post}
    
    if [ $? -eq 0 ]; then
        echo "  Optimization completed successfully"
        PROCESSED=$((PROCESSED+1))
    else
        echo "  Optimization failed with error code $?"
        FAILED=$((FAILED+1))
    fi
done

echo "Batch processing complete"
echo "Configurations processed successfully: $PROCESSED"
echo "Configurations skipped (already done): $SKIPPED"
echo "Configurations failed: $FAILED"
echo "End time: $(date)"
