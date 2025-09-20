#!/bin/bash
#
# SLURM job submission script for MOO Layout Optimization
#
# Usage:
#   # Basic MOO with default objectives from config
#   bash run_jobs_slurm.sh
#   
#   # MOO with custom objectives  
#   bash run_jobs_slurm.sh --objectives engram_key_preference,engram_row_separation,engram_same_row,engram_same_finger
#
# Connect and set up the code environment: 
#
#   # Log in (example: Pittsburgh Supercomputing Center's Bridge-2 cluster)
#   ssh username@bridges2.psc.edu
#
#   # Create directory for the project
#   mkdir -p optimizer; cd optimizer
#
#   # Clone repository and make scripts executable
#   git clone https://github.com/binarybottle/optimize_layouts.git
#   cd optimize_layouts
#   chmod +x *.py *.sh
#
#   # Make output directories
#   mkdir -p output/layouts output/outputs output/errors output/configs
#
#   # Test that anaconda3 module works (no virtual environment needed)
#   module load anaconda3
#   python3 --version
#   python3 -c "import numpy, pandas, yaml; print('Required packages available')"
#
#   # Generate config files as described in generate_configs.py
#   python3 generate_configs.py
#
# Configure, submit, monitor, and cancel jobs:
#   
#   # Use screen to keep session active
#   screen -S submission
#   
#   # Automatically manage submissions (checks every 5 minutes)
#   # Make sure to adjust SLURM and SBATCH settings below!
#   chmod +x run_jobs_slurm.sh
#   nohup bash run_jobs_slurm.sh > submission.log 2>&1 &
#   echo $! > run_jobs.pid # If necessary, kill using the saved PID: kill $(cat run_jobs.pid)
#   
#   # Check all running jobs
#   squeue -u $USER
#   
#   # Watch jobs in real-time
#   watch squeue -u $USER
#   
#   # See how many jobs are running vs. pending
#   squeue -j <job_array_id> | awk '{print $5}' | sort | uniq -c
#   
#   # Check number of output files
#   ls output/layouts/moo_results_config_*.csv | wc -l
# 
#   # Cancel all your jobs at once, or for a specific job ID
#   scancel -u $USER
#   scancel <job_array_id>

#--------------------------------------------------------------
# SLURM settings (adjust for a given cluster)
#--------------------------------------------------------------
MAX_JOBS=16
CHECK_INTERVAL=300
CONFIG_PREFIX="output/configs/config_"
CONFIG_SUFFIX=".yaml"
OUTPUT_DIR="output/layouts"
TOTAL_CONFIGS=1

SCRIPT_PATH="optimize_layouts.py"
OBJECTIVES=""
WEIGHTS=""
MAXIMIZE=""
MAX_SOLUTIONS=""
TIME_LIMIT=""

#--------------------------------------------------------------
# Parse command line arguments
#--------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --objectives)
                OBJECTIVES="$2"
                shift 2
                ;;
            --weights)
                WEIGHTS="$2"
                shift 2
                ;;
            --maximize)
                MAXIMIZE="$2"
                shift 2
                ;;
            --max-solutions)
                MAX_SOLUTIONS="$2"
                shift 2
                ;;
            --time-limit)
                TIME_LIMIT="$2"
                shift 2
                ;;
            --max-jobs)
                MAX_JOBS="$2"
                shift 2
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

print_usage() {
    cat << EOF
SLURM MOO Job Submitter

Usage:
  # Basic MOO with config defaults
  bash run_jobs_slurm.sh
  
  # MOO with custom objectives
  bash run_jobs_slurm.sh --objectives engram_key_preference,engram_row_separation,engram_same_row,engram_same_finger

Arguments:
  --objectives        Comma-separated objective names (use config defaults if not specified)
  --weights           Comma-separated objective weights (optional)
  --maximize          Comma-separated true/false flags (optional)
  --max-solutions     Maximum Pareto solutions per config (optional)
  --time-limit        Time limit per config in seconds (optional)
  --max-jobs          Maximum concurrent SLURM jobs (default: 16)
  --help              Show this help
EOF
}

#--------------------------------------------------------------
# Validation
#--------------------------------------------------------------
validate_args() {
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "Error: Script not found: $SCRIPT_PATH"
        exit 1
    fi
}

#--------------------------------------------------------------
# Output file detection
#--------------------------------------------------------------
is_config_completed() {
    local config_id=$1
    local pattern="moo_results_config_${config_id}_*.csv"
    
    find "$OUTPUT_DIR" -name "$pattern" 2>/dev/null | grep -q .
}

#--------------------------------------------------------------
# Job submission
#--------------------------------------------------------------
build_optimization_command() {
    local config_id=$1
    local config_file="${CONFIG_PREFIX}${config_id}${CONFIG_SUFFIX}"
    
    local cmd="python3 optimize_moo.py --config $config_file"
    
    if [[ -n "$OBJECTIVES" ]]; then
        cmd="$cmd --objectives $OBJECTIVES"
    fi
    
    if [[ -n "$WEIGHTS" ]]; then
        cmd="$cmd --weights $WEIGHTS"
    fi
    
    if [[ -n "$MAXIMIZE" ]]; then
        cmd="$cmd --maximize $MAXIMIZE"
    fi
    
    if [[ -n "$MAX_SOLUTIONS" ]]; then
        cmd="$cmd --max-solutions $MAX_SOLUTIONS"
    fi
    
    if [[ -n "$TIME_LIMIT" ]]; then
        cmd="$cmd --time-limit $TIME_LIMIT"
    fi
    
    echo "$cmd"
}

submit_individual_job() {
    local config_id=$1
    
    # Create job script
    local job_script=$(mktemp)
    local optimization_cmd=$(build_optimization_command $config_id)
    
    # Resource settings for MOO optimization
    local cpus=24
    local memory="500GB"
    local time_limit="8:00:00"
    
    cat > "$job_script" << JOBEOF
#!/bin/bash
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$memory
#SBATCH --time=$time_limit
#SBATCH --partition=EM
#SBATCH --account=med250002p
#SBATCH --job-name=moo_${config_id}
#SBATCH --output=output/outputs/moo_cfg_${config_id}.out
#SBATCH --error=output/errors/moo_cfg_${config_id}.err

module purge
module load anaconda3

export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export NUMBA_NUM_THREADS=1

cd \$HOME/optimizer/optimize_layouts

# Double-check not completed (race condition)
if find output/layouts -name "moo_results_config_${config_id}_*.csv" 2>/dev/null | grep -q .; then
    echo "Already completed, exiting"
    exit 0
fi

echo "Processing config ${config_id} using MOO system..."
echo "Command: $optimization_cmd"

# Run optimization
$optimization_cmd

echo "Completed config ${config_id}"
JOBEOF

    # Submit job
    local job_output=$(sbatch "$job_script" 2>&1)
    rm "$job_script"
    
    if [[ $job_output == *"Submitted batch job"* ]]; then
        local job_id=$(echo "$job_output" | grep -o '[0-9]\+')
        echo "  Submitted config $config_id as job $job_id (MOO)"
        return 0
    else
        echo "  Failed to submit config $config_id: $job_output"
        return 1
    fi
}

find_next_config() {
    for ((i=1; i<=TOTAL_CONFIGS; i++)); do
        CONFIG_FILE="${CONFIG_PREFIX}${i}${CONFIG_SUFFIX}"
        
        # Check if config exists
        if [ ! -f "$CONFIG_FILE" ]; then
            continue
        fi
        
        # Check if already completed
        if is_config_completed $i; then
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

#--------------------------------------------------------------
# Main execution
#--------------------------------------------------------------

# Parse arguments
parse_args "$@"

# Validate configuration
validate_args

echo "=== SLURM MOO Job Submitter ==="
echo "Script: $SCRIPT_PATH"
if [[ -n "$OBJECTIVES" ]]; then
    echo "Objectives: $OBJECTIVES"
fi
if [[ -n "$WEIGHTS" ]]; then
    echo "Weights: $WEIGHTS"
fi
if [[ -n "$MAXIMIZE" ]]; then
    echo "Maximize: $MAXIMIZE"
fi
echo "Max jobs: $MAX_JOBS"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Create temp file to track submitted configs
SUBMITTED_FILE="/tmp/submitted_configs_moo_$$"
> "$SUBMITTED_FILE"

# Clean up on exit
trap "rm -f '$SUBMITTED_FILE'" EXIT

# Main submission loop
while true; do
    # Check current status
    CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
    
    # Count completed configs
    COMPLETED=$(find "$OUTPUT_DIR" -name "moo_results_config_*" | wc -l)
    
    echo "$(date): Jobs running: $CURRENT_JOBS, Completed: $COMPLETED/$TOTAL_CONFIGS (MOO)"
    
    # Submit more jobs if we have capacity
    if [ $CURRENT_JOBS -lt $MAX_JOBS ]; then
        JOBS_TO_SUBMIT=$((MAX_JOBS - CURRENT_JOBS))
        echo "  Submitting $JOBS_TO_SUBMIT jobs..."
        
        SUBMITTED=0
        for ((j=0; j<JOBS_TO_SUBMIT; j++)); do
            NEXT_CONFIG_TO_SUBMIT=$(find_next_config)
            if [ -z "$NEXT_CONFIG_TO_SUBMIT" ]; then
                echo "  No more configs to process!"
                break
            fi
            
            if submit_individual_job $NEXT_CONFIG_TO_SUBMIT; then
                SUBMITTED=$((SUBMITTED + 1))
                sleep 1
            fi
        done
        
        echo "  Successfully submitted $SUBMITTED jobs"
    else
        echo "  At job limit, waiting for completions..."
    fi
    
    # Check if done
    if [ $COMPLETED -ge $TOTAL_CONFIGS ]; then
        echo "All configurations completed!"
        break
    fi
    
    echo "  Sleeping ${CHECK_INTERVAL}s..."
    echo ""
    sleep $CHECK_INTERVAL
done