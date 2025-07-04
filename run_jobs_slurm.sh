#!/bin/bash
# Auto submitter using individual jobs (no arrays)
# Properly tracks submitted configs

MAX_JOBS=16
CHECK_INTERVAL=300
CONFIG_PREFIX="output/configs1/config_"
CONFIG_SUFFIX=".yaml"
OUTPUT_DIR="output/layouts"
TOTAL_CONFIGS=95040

echo "=== Auto Individual Job Submitter ==="
echo "Max jobs: $MAX_JOBS"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Avoids all array limits!"
echo ""

# Create temp file to track submitted configs this session
SUBMITTED_FILE="/tmp/submitted_configs_$$"
> "$SUBMITTED_FILE"

# Find next config to process
find_next_config() {
    for ((i=4001; i<=TOTAL_CONFIGS; i++)); do
        CONFIG_FILE="${CONFIG_PREFIX}${i}${CONFIG_SUFFIX}"
        
        # Check if config exists
        if [ ! -f "$CONFIG_FILE" ]; then
            continue
        fi
        
        # Check if already completed
        if find "$OUTPUT_DIR" -name "*config_${i}_*.csv" 2>/dev/null | grep -q .; then
            continue
        fi
        
        # Check if already submitted this session
        if grep -q "^${i}$" "$SUBMITTED_FILE" 2>/dev/null; then
            continue
        fi
        
        # Mark as submitted and return
        echo "$i" >> "$SUBMITTED_FILE"
        echo $i
        return
    done
    echo ""
}

submit_individual_job() {
    local config_id=$1
    
    # Create job script
    local job_script=$(mktemp)
    cat > "$job_script" << JOBEOF
#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --time=2:00:00
#SBATCH --partition=RM-shared
#SBATCH --account=med250002p
#SBATCH --job-name=cfg_${config_id}
#SBATCH --output=output/outputs/cfg_${config_id}.out
#SBATCH --error=output/errors/cfg_${config_id}.err

module purge
module load anaconda3

export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export NUMBA_NUM_THREADS=1

cd \$HOME/keyboard_optimizer/optimize_layouts

# Double-check not completed (race condition)
if find output/layouts -name "*config_${config_id}_*.csv" 2>/dev/null | grep -q .; then
    echo "Already completed, exiting"
    exit 0
fi

echo "Processing config ${config_id} with 8 processes..."
python3 optimize_layout.py \\
    --config ${CONFIG_PREFIX}${config_id}${CONFIG_SUFFIX} \\
    --moo \\
    --processes 8

echo "Completed config ${config_id}"
JOBEOF

    # Submit and clean up
    local job_output=$(sbatch "$job_script" 2>&1)
    rm "$job_script"
    
    if [[ $job_output == *"Submitted batch job"* ]]; then
        local job_id=$(echo "$job_output" | grep -o '[0-9]\+')
        echo "  ✓ Submitted config $config_id as job $job_id"
        return 0
    else
        echo "  ✗ Failed to submit config $config_id: $job_output"
        return 1
    fi
}

# Clean up on exit
trap "rm -f '$SUBMITTED_FILE'" EXIT

# Main loop
while true; do
    # Check current status
    CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
    COMPLETED=$(find "$OUTPUT_DIR" -name "*config_*" | wc -l)
    
    echo "$(date): Jobs running: $CURRENT_JOBS, Completed: $COMPLETED/$TOTAL_CONFIGS"
    
    # Submit more jobs if we have capacity
    if [ $CURRENT_JOBS -lt $MAX_JOBS ]; then
        JOBS_TO_SUBMIT=$((MAX_JOBS - CURRENT_JOBS))
        echo "  Submitting $JOBS_TO_SUBMIT individual jobs..."
        
        SUBMITTED=0
        for ((j=0; j<JOBS_TO_SUBMIT; j++)); do
            NEXT_CONFIG_TO_SUBMIT=$(find_next_config)
            if [ -z "$NEXT_CONFIG_TO_SUBMIT" ]; then
                echo "  No more configs to process!"
                break
            fi
            
            if submit_individual_job $NEXT_CONFIG_TO_SUBMIT; then
                SUBMITTED=$((SUBMITTED + 1))
                sleep 1  # Prevent overwhelming scheduler
            fi
        done
        
        echo "  Successfully submitted $SUBMITTED jobs"
    else
        echo "  At job limit, waiting for completions..."
    fi
    
    # Check if done
    if [ $COMPLETED -ge $TOTAL_CONFIGS ]; then
        echo "🎉 All configurations completed!"
        break
    fi
    
    echo "  Sleeping ${CHECK_INTERVAL}s..."
    echo ""
    sleep $CHECK_INTERVAL
done
