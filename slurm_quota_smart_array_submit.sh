#!/bin/bash
# Quota-aware array job submission for HPC optimization.
# Calls slurm_array_processor.sh to process configurations as array tasks.
#
# If you want to start fresh with a new scan:
#     bash slurm_quota_smart_array_submit.sh --rescan
# Otherwise:
#     bash slurm_quota_smart_array_submit.sh

# Configuration (RM-shared vs. EM)
TOTAL_CONFIGS=65520                # Total configurations (adjust as needed)
BATCH_SIZE=500                     # Configs per batch file 
ARRAY_SIZE=500                     # Maximum array tasks per job
MAX_CONCURRENT=8                   # Maximum concurrent tasks (4 vs. 8)
CHUNK_SIZE=2                       # Number of array jobs to submit at once
config_pre=output/configs1/config_ # Config file path prefix
config_post=.yaml                  # Config file suffix

# Create needed directories
mkdir -p output/outputs output/errors submission_logs batch_files output/layouts

# Decide whether to scan for new configurations or use existing batches
if [ "$1" == "--rescan" ] || [ ! -f "pending_configs.txt" ]; then
    echo "=== Scanning for pending configurations ==="
    echo "This may take several minutes for $TOTAL_CONFIGS configurations..."
    
    # Clear existing files
    > pending_configs.txt
    rm -f batch_files/*
    
    # Track counts
    TOTAL_PENDING=0
    TOTAL_COMPLETED=0
    TOTAL_MISSING=0
    
    # Scan all configurations
    for ((CONFIG_ID=1; CONFIG_ID<=TOTAL_CONFIGS; CONFIG_ID++)); do
        # Check if the config file exists
        if [ ! -f "${config_pre}${CONFIG_ID}${config_post}" ]; then
            TOTAL_MISSING=$((TOTAL_MISSING+1))
            continue
        fi
        
        # Check if output already exists and contains actual data
        FIND_PATTERN="layout_results_${CONFIG_ID}_[0-9]*"
        HAS_VALID_OUTPUT=false

        # Find all matching CSV files for this config
        while IFS= read -r -d '' file; do
            if [ -s "$file" ]; then  # Check if file is not empty
                # Count lines in the file (excluding header)
                LINE_COUNT=$(tail -n +2 "$file" | wc -l)
                if [ "$LINE_COUNT" -gt 0 ]; then
                    HAS_VALID_OUTPUT=true
                    break  # Found at least one valid file, no need to check others
                fi
            fi
        done < <(find output/layouts -name "$FIND_PATTERN*.csv" -print0 2>/dev/null)

        if [ "$HAS_VALID_OUTPUT" = true ]; then
            TOTAL_COMPLETED=$((TOTAL_COMPLETED+1))
        else
            # Add to pending list
            echo $CONFIG_ID >> pending_configs.txt
            TOTAL_PENDING=$((TOTAL_PENDING+1))
        fi

        # Show progress periodically
        if [ $((CONFIG_ID % 5000)) -eq 0 ]; then
            echo "Scanned $CONFIG_ID / $TOTAL_CONFIGS configs..."
            echo "  Completed so far: $TOTAL_COMPLETED"
            echo "  Pending so far: $TOTAL_PENDING"
        fi
    done
    
    echo "Scan complete!"
    echo "Configurations completed: $TOTAL_COMPLETED"
    echo "Configurations pending: $TOTAL_PENDING"
    echo "Configuration files missing: $TOTAL_MISSING"
    
    # Create batch files if there are pending configurations
    if [ $TOTAL_PENDING -gt 0 ]; then
        echo "Creating batch files for $TOTAL_PENDING configurations..."
        
        # Create batches
        CURRENT_BATCH=1
        CURRENT_COUNT=0
        BATCH_FILE="batch_files/batch_${CURRENT_BATCH}.txt"
        > $BATCH_FILE
        
        # Add each pending config to a batch file
        while read -r CONFIG_ID; do
            echo $CONFIG_ID >> $BATCH_FILE
            CURRENT_COUNT=$((CURRENT_COUNT+1))
            
            # Start a new batch if this one is full
            if [ $CURRENT_COUNT -eq $BATCH_SIZE ]; then
                CURRENT_BATCH=$((CURRENT_BATCH+1))
                CURRENT_COUNT=0
                BATCH_FILE="batch_files/batch_${CURRENT_BATCH}.txt"
                > $BATCH_FILE
            fi
        done < pending_configs.txt
        
        # Count total batches
        TOTAL_BATCHES=$(ls -1 batch_files/batch_*.txt 2>/dev/null | wc -l)
        echo "Created $TOTAL_BATCHES batch files"
        
        # Reset progress file for new batch list
        echo "0" > batch_submission_progress.txt
    else
        echo "No pending configurations found. All work is complete!"
        exit 0
    fi
else
    # Use existing batch files
    TOTAL_BATCHES=$(ls -1 batch_files/batch_*.txt 2>/dev/null | wc -l)
    TOTAL_PENDING=$(wc -l < pending_configs.txt)
    echo "Using existing $TOTAL_BATCHES batch files with $TOTAL_PENDING pending configurations"
fi

# Read the current progress or start fresh
PROGRESS_FILE="batch_submission_progress.txt"
if [ -f "$PROGRESS_FILE" ]; then
    CURRENT_BATCH=$(cat "$PROGRESS_FILE")
else
    CURRENT_BATCH=0
    echo "0" > "$PROGRESS_FILE"
fi

# Calculate end batch for this chunk
END_BATCH=$((CURRENT_BATCH + CHUNK_SIZE - 1))
if [ $END_BATCH -ge $TOTAL_BATCHES ]; then
    END_BATCH=$((TOTAL_BATCHES - 1))
fi

# Log file
LOG_FILE="submission_logs/array_submission_$(date +%Y%m%d_%H%M%S)_chunk${CURRENT_BATCH}.log"

echo "=== SLURM HPC Array Job Submission ===" | tee -a "$LOG_FILE"
echo "Total pending configurations: $TOTAL_PENDING" | tee -a "$LOG_FILE"
echo "Total batch files: $TOTAL_BATCHES" | tee -a "$LOG_FILE"
echo "Submitting batches $CURRENT_BATCH through $END_BATCH" | tee -a "$LOG_FILE"
echo "Array size: $ARRAY_SIZE" | tee -a "$LOG_FILE"
echo "Max concurrent tasks: $MAX_CONCURRENT (8 CPUs each = $((MAX_CONCURRENT * 8)) total CPUs)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Array to store job IDs from this chunk
declare -a CHUNK_JOB_IDS

# Submit array jobs for this chunk
for ((i=CURRENT_BATCH; i<=END_BATCH; i++)); do
    BATCH_FILE="batch_files/batch_$((i+1)).txt"  # +1 because batch files are 1-indexed
    
    if [ ! -f "$BATCH_FILE" ]; then
        echo "Warning: Batch file $BATCH_FILE not found" | tee -a "$LOG_FILE"
        continue
    fi
    
    # Count configs in this batch
    CONFIG_COUNT=$(wc -l < $BATCH_FILE)
    
    # Calculate array indices
    if [ $CONFIG_COUNT -lt $ARRAY_SIZE ]; then
        ARRAY_RANGE="0-$((CONFIG_COUNT-1))"
    else
        ARRAY_RANGE="0-$((ARRAY_SIZE-1))"
    fi
    
    echo "Submitting HPC batch $((i+1))/$TOTAL_BATCHES with $CONFIG_COUNT configurations as array $ARRAY_RANGE..." | tee -a "$LOG_FILE"
    echo "  Resource usage: $CONFIG_COUNT × 8 CPUs × 15GB = moderate workload" | tee -a "$LOG_FILE"
    
    # Submit array job
    JOB_OUTPUT=$(sbatch --export=CONFIG_FILE=$BATCH_FILE --array=$ARRAY_RANGE%$MAX_CONCURRENT slurm_array_processor.sh 2>&1)
    JOB_STATUS=$?
    
    if [ $JOB_STATUS -eq 0 ]; then
        JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
        echo "  Success: Job ID $JOB_ID" | tee -a "$LOG_FILE"
        CHUNK_JOB_IDS+=($JOB_ID)
    else
        echo "  Failed: $JOB_OUTPUT" | tee -a "$LOG_FILE"
    fi
    
    sleep 5  # Delay between submissions
done

# Update progress for next run
NEXT_BATCH=$((END_BATCH + 1))
echo $NEXT_BATCH > "$PROGRESS_FILE"

# Check if we've completed all batches
if [ $NEXT_BATCH -ge $TOTAL_BATCHES ]; then
    echo "All HPC batches have been submitted. Submission complete!" | tee -a "$LOG_FILE"
    echo "Final job IDs: ${CHUNK_JOB_IDS[@]}" | tee -a "$LOG_FILE"
    echo "Monitor progress with: squeue -u $USER" | tee -a "$LOG_FILE"
else
    # Schedule the next chunk with a delay
    echo "Scheduling next HPC chunk (batches $NEXT_BATCH to $((NEXT_BATCH+CHUNK_SIZE-1)))" | tee -a "$LOG_FILE"
    
    # Submit this script as a job that depends on the completion of this chunk's jobs
    if [ ${#CHUNK_JOB_IDS[@]} -gt 0 ]; then
        # Use afterany dependency to continue even if some jobs fail
        DEPENDENCY_LIST=$(IFS=:; echo "afterany:${CHUNK_JOB_IDS[*]}")
        NEXT_MANAGER=$(sbatch --dependency=$DEPENDENCY_LIST --time=00:10:00 --wrap="cd $PWD && bash $0" | awk '{print $4}')
        echo "Next manager job scheduled with ID: $NEXT_MANAGER" | tee -a "$LOG_FILE"
    else
        # If no jobs were submitted in this chunk, continue anyway with a short delay
        echo "No jobs were submitted in this chunk. Scheduling next manager immediately." | tee -a "$LOG_FILE"
        NEXT_MANAGER=$(sbatch --time=00:10:00 --wrap="cd $PWD && bash $0" | awk '{print $4}')
        echo "Next manager job scheduled with ID: $NEXT_MANAGER" | tee -a "$LOG_FILE"
    fi
fi

echo "This HPC submission manager completed at $(date)" | tee -a "$LOG_FILE"