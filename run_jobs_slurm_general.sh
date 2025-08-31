#!/bin/bash
#
# SLURM job submission script for General MOO System
# 
# This extends your existing run_jobs_slurm.sh to support both:
# 1. Current system (optimize_layout.py --moo)  
# 2. General MOO system (optimize_layout_general.py)
#
# Usage:
#   # Current system (unchanged)
#   bash run_jobs_slurm_general.sh
#   
#   # General MOO system
#   bash run_jobs_slurm_general.sh --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized,engram8_score_normalized

#--------------------------------------------------------------
# SLURM settings (adjust for your cluster)
#--------------------------------------------------------------
MAX_JOBS=16
CHECK_INTERVAL=300
CONFIG_PREFIX="output/configs2/config_"
CONFIG_SUFFIX=".yaml"
OUTPUT_DIR="output/layouts"
TOTAL_CONFIGS=84

# Default settings (current system)
SCRIPT_MODE="current"
SCRIPT_PATH="optimize_layout.py"
KEYPAIR_TABLE=""
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
            --mode)
                SCRIPT_MODE="$2"
                shift 2
                ;;
            --keypair-table)
                KEYPAIR_TABLE="$2"
                shift 2
                ;;
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
SLURM General MOO Job Submitter

Usage:
  # Current system (unchanged)
  bash run_jobs_slurm_general.sh
  
  # General MOO with 2 objectives
  bash run_jobs_slurm_general.sh --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized
  
  # With custom weights and directions
  bash run_jobs_slurm_general.sh --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized --weights 1.0,2.0 --maximize true,false

Arguments:
  --mode              current or general (default: current)
  --keypair-table     Path to key-pair scoring table (required for general mode)
  --objectives        Comma-separated objective columns (required for general mode)
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
    if [[ "$SCRIPT_MODE" == "general" ]]; then
        if [[ -z "$KEYPAIR_TABLE" ]]; then
            echo "Error: --keypair-table required for general mode"
            exit 1
        fi
        
        if [[ -z "$OBJECTIVES" ]]; then
            echo "Error: --objectives required for general mode"
            exit 1
        fi
        
        if [[ ! -f "$KEYPAIR_TABLE" ]]; then
            echo "Error: Key-pair table not found: $KEYPAIR_TABLE"
            exit 1
        fi
        
        SCRIPT_PATH="optimize_layout_general.py"
    fi
    
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "Error: Script not found: $SCRIPT_PATH"
        exit 1
    fi
}

#--------------------------------------------------------------
# Output file detection
#--------------------------------------------------------------
get_output_pattern() {
    local config_id=$1
    
    if [[ "$SCRIPT_MODE" == "general" ]]; then
        echo "general_moo_results_config_${config_id}_*.csv"
    else
        echo "moo_results_config_${config_id}_*.csv"
    fi
}

is_config_completed() {
    local config_id=$1
    local pattern=$(get_output_pattern $config_id)
    
    find "$OUTPUT_DIR" -name "$pattern" 2>/dev/null | grep -q .
}

#--------------------------------------------------------------
# Job submission
#--------------------------------------------------------------
build_optimization_command() {
    local config_id=$1
    local config_file="${CONFIG_PREFIX}${config_id}${CONFIG_SUFFIX}"
    
    if [[ "$SCRIPT_MODE" == "general" ]]; then
        # General MOO system
        local cmd="python3 optimize_layout_general.py --config $config_file --mode general --keypair-table $KEYPAIR_TABLE --objectives $OBJECTIVES"
        
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
        
        # Use all allocated CPUs for parallel search
        cmd="$cmd --processes \$SLURM_CPUS_PER_TASK"
        
    else
        # Current system (unchanged)
        local cmd="python3 optimize_layout.py --config $config_file --moo --processes \$SLURM_CPUS_PER_TASK"
    fi
    
    echo "$cmd"
}

submit_individual_job() {
    local config_id=$1
    
    # Create job script
    local job_script=$(mktemp)
    local optimization_cmd=$(build_optimization_command $config_id)
    local output_pattern=$(get_output_pattern $config_id)
    
    cat > "$job_script" << JOBEOF
#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --mem=500GB
#SBATCH --time=8:00:00
#SBATCH --partition=EM
#SBATCH --account=med250002p
#SBATCH --job-name=${SCRIPT_MODE}_${config_id}
#SBATCH --output=output/outputs/${SCRIPT_MODE}_cfg_${config_id}.out
#SBATCH --error=output/errors/${SCRIPT_MODE}_cfg_${config_id}.err

module purge
module load anaconda3

export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export NUMBA_NUM_THREADS=1

cd \$HOME/optimizer/optimize_layouts

# Double-check not completed (race condition)
if find output/layouts -name "$output_pattern" 2>/dev/null | grep -q .; then
    echo "Already completed, exiting"
    exit 0
fi

echo "Processing config ${config_id} using ${SCRIPT_MODE} mode with \$SLURM_CPUS_PER_TASK processes..."
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
        echo "  Submitted config $config_id as job $job_id ($SCRIPT_MODE mode)"
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

echo "=== SLURM General MOO Job Submitter ==="
echo "Mode: $SCRIPT_MODE"
echo "Script: $SCRIPT_PATH"
if [[ "$SCRIPT_MODE" == "general" ]]; then
    echo "Key-pair table: $KEYPAIR_TABLE"
    echo "Objectives: $OBJECTIVES"
    if [[ -n "$WEIGHTS" ]]; then
        echo "Weights: $WEIGHTS"
    fi
    if [[ -n "$MAXIMIZE" ]]; then
        echo "Maximize: $MAXIMIZE"
    fi
fi
echo "Max jobs: $MAX_JOBS"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Create temp file to track submitted configs
SUBMITTED_FILE="/tmp/submitted_configs_${SCRIPT_MODE}_$$"
> "$SUBMITTED_FILE"

# Clean up on exit
trap "rm -f '$SUBMITTED_FILE'" EXIT

# Main submission loop (unchanged logic from your original)
while true; do
    # Check current status
    CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
    
    # Count completed configs (use appropriate pattern)
    if [[ "$SCRIPT_MODE" == "general" ]]; then
        COMPLETED=$(find "$OUTPUT_DIR" -name "general_moo_results_config_*" | wc -l)
    else
        COMPLETED=$(find "$OUTPUT_DIR" -name "moo_results_config_*" | wc -l)
    fi
    
    echo "$(date): Jobs running: $CURRENT_JOBS, Completed: $COMPLETED/$TOTAL_CONFIGS ($SCRIPT_MODE mode)"
    
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