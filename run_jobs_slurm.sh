#!/bin/bash
"""
Run individual jobs (no arrays) on a (slurm) linux cluster.
Properly tracks submitted configs.

Connect and set up the code environment: 

  ```bash
  # Log in (example: Pittsburgh Supercomputing Center's Bridge-2 cluster)
  ssh username@bridges2.psc.edu

  # Create directory for the project
  mkdir -p optimizer; cd optimizer

  # Clone repository and make scripts executable
  git clone https://github.com/binarybottle/optimize_layouts.git
  cd optimize_layouts
  chmod +x *.py *.sh

  # Make output directories
  mkdir -p output/layouts2 output/outputs2 output/errors2 output/configs2

  # Test that anaconda3 module works (no virtual environment needed)
  module load anaconda3
  python3 --version
  python3 -c "import numpy, pandas, yaml; print('Required packages available')"

  # Generate config files as described in either of the config generation scripts
  python3 generate_configs2.py
  ```

Configure, submit, monitor, and cancel jobs:

  ```bash
  # Use screen to keep session active
  screen -S submission

  # Automatically manage submissions (checks every 5 minutes)
  nohup bash run_job_slurm.sh

  # Check all running jobs
  squeue -u $USER

  # Watch jobs in real-time
  watch squeue -u $USER

  # See how many jobs are running vs. pending
  squeue -j <job_array_id> | awk '{print $5}' | sort | uniq -c

  # Check number of output files
  sh count_files.sh output/layouts

  # Cancel all your jobs at once, or for a specific job ID
  scancel -u $USER
  scancel <job_array_id>
  ```
"""

MAX_JOBS=16
CHECK_INTERVAL=300
CONFIG_PREFIX="output/configs2/config_"
CONFIG_SUFFIX=".yaml"
OUTPUT_DIR="output/layouts2"
TOTAL_CONFIGS=95040

echo "=== Auto Individual Job Submitter ==="
echo "Max jobs: $MAX_JOBS"
echo "Check interval: ${CHECK_INTERVAL}s"
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
#SBATCH --cpus-per-task=24
#SBATCH --mem=500GB
#SBATCH --time=8:00:00
#SBATCH --partition=EM
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

echo "Processing config ${config_id} with ${cpus-per-task} processes..."
python3 optimize_layout.py \\
    --config ${CONFIG_PREFIX}${config_id}${CONFIG_SUFFIX} \\
    --moo \\
    --processes ${cpus-per-task}

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
