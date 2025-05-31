#!/bin/bash
# Quota-aware array job submission for HPC optimization.
# Calls slurm_array_processor.sh to process configurations as array tasks.

# Use preset configurations for common scenarios:
#   bash slurm_array_submit.sh --account "med250002p" --preset standard --moo
#
# Custom resource allocation and optimization settings:  
#   bash slurm_array_submit.sh --account "med250002p" --preset extreme-memory \
#       --moo --total-configs 1 \
#       --config-prefix "output/configs1/config_" --config-suffix ".yaml" 
#
#   bash slurm_array_submit.sh --cpus 16 --mem 200GB --time 6:00:00 --moo --processes 48

# Default configuration (extreme-memory preset)
#===================================================================
SLURM_CPUS=24
SLURM_MEM="500GB"
SLURM_TIME="4:00:00"
SLURM_PARTITION="EM"
SLURM_ACCOUNT="med250002p"               # Default allocation ID
TOTAL_CONFIGS=11880                
BATCH_SIZE=500                     
ARRAY_SIZE=500                     
MAX_CONCURRENT=8                   
CHUNK_SIZE=2                       
CONFIG_PREFIX="output/configs1/config_"  # Config file path prefix
CONFIG_SUFFIX=".yaml"                    # Config file suffix

# Optimization settings
OPT_MODE="--moo"                         # Default to MOO
OPT_MAX_SOLUTIONS=""
OPT_N_SOLUTIONS=""
OPT_TIME_LIMIT=""
OPT_PROCESSES=""

# Resource presets
declare -A PRESETS
PRESETS[standard]="8,40GB,2:00:00,RM-shared,4"         # cpus,mem,time,partition,concurrent
PRESETS[extreme-memory]="24,500GB,4:00:00,EM,8"        # Maximum performance
PRESETS[debug]="4,20GB,0:30:00,RM-shared,2"            # Quick testing

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            IFS=',' read -ra PRESET_VALUES <<< "${PRESETS[$2]}"
            SLURM_CPUS="${PRESET_VALUES[0]}"
            SLURM_MEM="${PRESET_VALUES[1]}"
            SLURM_TIME="${PRESET_VALUES[2]}"
            SLURM_PARTITION="${PRESET_VALUES[3]}"
            MAX_CONCURRENT="${PRESET_VALUES[4]}"
            shift 2
            ;;
        --cpus)
            SLURM_CPUS="$2"
            shift 2
            ;;
        --mem)
            SLURM_MEM="$2"
            shift 2
            ;;
        --time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --partition)
            SLURM_PARTITION="$2"
            shift 2
            ;;
        --account)
            SLURM_ACCOUNT="$2"
            shift 2
            ;;
        --concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --total-configs)
            TOTAL_CONFIGS="$2"
            shift 2
            ;;
        --config-prefix)
            CONFIG_PREFIX="$2"
            shift 2
            ;;
        --config-suffix)
            CONFIG_SUFFIX="$2"
            shift 2
            ;;
        --moo)
            OPT_MODE="--moo"
            shift
            ;;
        --soo)
            OPT_MODE="--soo"
            shift
            ;;
        --max-solutions)
            OPT_MAX_SOLUTIONS="$2"
            shift 2
            ;;
        --n-solutions)
            OPT_N_SOLUTIONS="$2"
            shift 2
            ;;
        --time-limit)
            OPT_TIME_LIMIT="$2"
            shift 2
            ;;
        --processes)
            OPT_PROCESSES="$2"
            shift 2
            ;;
        --rescan)
            shift
            ;;
        --help)
            echo "Usage: bash slurm_array_submit.sh [OPTIONS]"
            echo ""
            echo "Resource Presets:"
            echo "  --preset standard         8 CPUs,  40GB, 2h,  RM-shared"
            echo "  --preset extreme-memory  24 CPUs, 500GB, 4h,  EM (default)"
            echo "  --preset debug            4 CPUs,  20GB, 30m, RM-shared"
            echo ""
            echo "Custom Resources:"
            echo "  --cpus N                CPUs per task"
            echo "  --mem SIZE              Memory per task (e.g., 40GB, 500GB)"
            echo "  --time TIME             Time limit (e.g., 2:00:00, 6:00:00)"
            echo "  --partition PART        SLURM partition (RM-shared, EM)"
            echo "  --account ID            SLURM account/allocation ID"
            echo "  --concurrent N          Max concurrent array tasks"
            echo ""
            echo "Configuration:"
            echo "  --total-configs N       Total number of configurations (default: 11880)"
            echo "  --config-prefix PATH    Config file prefix (default: output/configs1/config_)"
            echo "  --config-suffix EXT     Config file suffix (default: .yaml)"
            echo ""
            echo "Optimization:"
            echo "  --moo (default), --soo, --max-solutions N, --n-solutions N, --time-limit SEC"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use --help for options)"
            exit 1
            ;;
    esac
done

# Set default processes if not specified
OPT_PROCESSES=${OPT_PROCESSES:-$SLURM_CPUS}

#===================================================================

# Check for --rescan flag in the arguments to determine if we need to rescan
RESCAN=false
for arg in "$@"; do
    if [[ $arg == "--rescan" ]]; then
        RESCAN=true
        break
    fi
done

# Set up directories
mkdir -p output/batches output/outputs output/errors output/logs

# Set up log file 
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="output/logs/submit_${TIMESTAMP}.log"

# If rescan flag is set, remove existing batch files and rescan
if [ "$RESCAN" = true ]; then
    echo "Rescanning configurations and recreating batch files..."
    rm -f output/batches/batch_*.txt
fi

# Scan for pending configurations and create batch files
echo "Scanning for pending configurations..." | tee -a "$LOG_FILE"

BATCH_NUM=1
CURRENT_BATCH_SIZE=0
CURRENT_BATCH_FILE="output/batches/batch_${BATCH_NUM}.txt"
TOTAL_PENDING=0
TOTAL_MISSING=0

# Only scan if batch files don't exist or rescan was requested
if [ ! -f "$CURRENT_BATCH_FILE" ] || [ "$RESCAN" = true ]; then
    echo "Creating new batch files..." | tee -a "$LOG_FILE"
    > "$CURRENT_BATCH_FILE"  # Create/clear first batch file
    
    # Scan all configurations
    for ((CONFIG_ID=1; CONFIG_ID<=TOTAL_CONFIGS; CONFIG_ID++)); do
        # Check if the config file exists
        if [ ! -f "${CONFIG_PREFIX}${CONFIG_ID}${CONFIG_SUFFIX}" ]; then
            TOTAL_MISSING=$((TOTAL_MISSING+1))
            continue
        fi
        
        # Check if output already exists
        FIND_PATTERN="layout_results_${CONFIG_ID}_[0-9]*"
        if find output/layouts -name "$FIND_PATTERN*.csv" 2>/dev/null | grep -q .; then
            continue  # Skip if output exists
        fi
        
        # Add to current batch
        echo "$CONFIG_ID" >> "$CURRENT_BATCH_FILE"
        CURRENT_BATCH_SIZE=$((CURRENT_BATCH_SIZE+1))
        TOTAL_PENDING=$((TOTAL_PENDING+1))
        
        # Start new batch file if current one is full
        if [ $CURRENT_BATCH_SIZE -ge $BATCH_SIZE ]; then
            BATCH_NUM=$((BATCH_NUM+1))
            CURRENT_BATCH_FILE="output/batches/batch_${BATCH_NUM}.txt"
            > "$CURRENT_BATCH_FILE"
            CURRENT_BATCH_SIZE=0
        fi
    done
    
    # Remove empty final batch file if it exists
    if [ $CURRENT_BATCH_SIZE -eq 0 ] && [ -f "$CURRENT_BATCH_FILE" ]; then
        rm "$CURRENT_BATCH_FILE"
        BATCH_NUM=$((BATCH_NUM-1))
    fi
else
    echo "Using existing batch files..." | tee -a "$LOG_FILE"
    # Count pending configurations from existing batch files
    for BATCH_FILE in output/batches/batch_*.txt; do
        if [ -f "$BATCH_FILE" ]; then
            BATCH_COUNT=$(wc -l < "$BATCH_FILE" 2>/dev/null || echo 0)
            TOTAL_PENDING=$((TOTAL_PENDING + BATCH_COUNT))
        fi
    done
    BATCH_NUM=$(ls output/batches/batch_*.txt 2>/dev/null | wc -l)
fi

TOTAL_BATCHES=$BATCH_NUM

if [ $TOTAL_PENDING -eq 0 ]; then
    echo "No pending configurations found. All work may be complete!" | tee -a "$LOG_FILE"
    if [ $TOTAL_MISSING -gt 0 ]; then
        echo "Note: $TOTAL_MISSING configuration files were missing." | tee -a "$LOG_FILE"
    fi
    exit 0
fi

# Determine which batches to submit
CURRENT_BATCH=1
END_BATCH=$((CURRENT_BATCH + CHUNK_SIZE - 1))
if [ $END_BATCH -gt $TOTAL_BATCHES ]; then
    END_BATCH=$TOTAL_BATCHES
fi

echo "Configuration: $SLURM_CPUS CPUs, $SLURM_MEM memory, $SLURM_TIME time" | tee -a "$LOG_FILE"
echo "Cluster: $SLURM_PARTITION partition, account $SLURM_ACCOUNT" | tee -a "$LOG_FILE"
echo "Configs: $TOTAL_CONFIGS total, prefix=$CONFIG_PREFIX, suffix=$CONFIG_SUFFIX" | tee -a "$LOG_FILE"
echo "Optimization: $OPT_MODE, processes=$OPT_PROCESSES" | tee -a "$LOG_FILE"
if [ -n "$OPT_MAX_SOLUTIONS" ]; then echo "MOO max solutions: $OPT_MAX_SOLUTIONS" | tee -a "$LOG_FILE"; fi
if [ -n "$OPT_N_SOLUTIONS" ]; then echo "SOO solutions: $OPT_N_SOLUTIONS" | tee -a "$LOG_FILE"; fi
if [ -n "$OPT_TIME_LIMIT" ]; then echo "Time limit: ${OPT_TIME_LIMIT}s" | tee -a "$LOG_FILE"; fi
echo "Total pending configurations: $TOTAL_PENDING" | tee -a "$LOG_FILE"
echo "Total batch files: $TOTAL_BATCHES" | tee -a "$LOG_FILE"
echo "Submitting batches $CURRENT_BATCH through $END_BATCH" | tee -a "$LOG_FILE"
echo "Array size: $ARRAY_SIZE" | tee -a "$LOG_FILE"
echo "Max concurrent tasks: $MAX_CONCURRENT" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Submit array jobs for each batch
for ((i=CURRENT_BATCH-1; i<END_BATCH; i++)); do
    BATCH_FILE="output/batches/batch_$((i+1)).txt"
    
    if [ ! -f "$BATCH_FILE" ]; then
        echo "Batch file $BATCH_FILE not found, skipping..." | tee -a "$LOG_FILE"
        continue
    fi
    
    CONFIG_COUNT=$(wc -l < "$BATCH_FILE")
    if [ $CONFIG_COUNT -eq 0 ]; then
        echo "Batch file $BATCH_FILE is empty, skipping..." | tee -a "$LOG_FILE"
        continue
    fi
    
    # Determine array range (0-indexed)
    ARRAY_END=$((CONFIG_COUNT-1))
    if [ $ARRAY_END -ge $ARRAY_SIZE ]; then
        ARRAY_END=$((ARRAY_SIZE-1))
    fi
    ARRAY_RANGE="0-$ARRAY_END"
    
    echo "Submitting HPC batch $((i+1))/$TOTAL_BATCHES with $CONFIG_COUNT configurations as array $ARRAY_RANGE..." | tee -a "$LOG_FILE"
    echo "  Resources: $SLURM_CPUS CPUs × $SLURM_MEM memory on $SLURM_PARTITION" | tee -a "$LOG_FILE"
    
    # Build export variables for optimization parameters
    EXPORT_VARS="CONFIG_FILE=$BATCH_FILE,MODE=$OPT_MODE,PROCESSES=$OPT_PROCESSES"
    EXPORT_VARS="$EXPORT_VARS,CONFIG_PREFIX=$CONFIG_PREFIX,CONFIG_SUFFIX=$CONFIG_SUFFIX"
    if [ -n "$OPT_MAX_SOLUTIONS" ]; then EXPORT_VARS="$EXPORT_VARS,MAX_SOLUTIONS=$OPT_MAX_SOLUTIONS"; fi
    if [ -n "$OPT_N_SOLUTIONS" ]; then EXPORT_VARS="$EXPORT_VARS,N_SOLUTIONS=$OPT_N_SOLUTIONS"; fi
    if [ -n "$OPT_TIME_LIMIT" ]; then EXPORT_VARS="$EXPORT_VARS,TIME_LIMIT=$OPT_TIME_LIMIT"; fi
    
    # Submit array job with configured resources
    JOB_OUTPUT=$(sbatch \
        --cpus-per-task=$SLURM_CPUS \
        --mem=$SLURM_MEM \
        --time=$SLURM_TIME \
        --partition=$SLURM_PARTITION \
        --account=$SLURM_ACCOUNT \
        --export="$EXPORT_VARS" \
        --array=$ARRAY_RANGE%$MAX_CONCURRENT \
        slurm_array_processor.sh 2>&1)
    
    if [[ $JOB_OUTPUT == *"Submitted batch job"* ]]; then
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+')
        echo "  ✓ Job submitted successfully: $JOB_ID" | tee -a "$LOG_FILE"
    else
        echo "  ✗ Job submission failed: $JOB_OUTPUT" | tee -a "$LOG_FILE"
    fi
    
    echo "$JOB_OUTPUT" >> "$LOG_FILE"
    
    # Small delay between submissions
    sleep 1
done

echo "Job submission complete. Check log file: $LOG_FILE"