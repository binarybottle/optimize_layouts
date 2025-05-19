#!/bin/bash
# slurm_smart_submit.sh - Smart configuration detection with managed chunked submissions

# CONFIGURATION
#===================================================================
TOTAL_CONFIGS=65520                # Total configurations
BATCH_SIZE=1000                    # SLURM array limit per batch
CHUNK_SIZE=4                       # Number of batches to submit in each run
config_pre=output/configs1/config_ # Config file path prefix
config_post=.yaml                  # Config file suffix
#===================================================================

# Create needed directories
mkdir -p output/outputs output/errors submission_logs batch_files

# Staging file to record which configurations need processing
PENDING_FILE="pending_configs.txt"
BATCHES_FILE="batch_list.txt"  # List of batch files to be processed
PROGRESS_FILE="batch_submission_progress.txt"  # Track which batch we're on

# Check if we should rescan or use existing batch files
RESCAN=1  # Default to rescan

# If batches file exists and no --rescan flag, use existing batches
if [ -f "$BATCHES_FILE" ] && [ "$1" != "--rescan" ]; then
    echo "Found existing batch list. Using it for submission."
    echo "(Run with --rescan to force a new configuration scan)"
    RESCAN=0
fi

# Scan for pending configurations if needed
if [ $RESCAN -eq 1 ]; then
    echo "=== Scanning for pending configurations ==="
    echo "This may take several minutes for $TOTAL_CONFIGS configurations..."
    
    # Clear existing files
    > $PENDING_FILE
    > $BATCHES_FILE
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
        
        # Check if output already exists using find (proven reliable method)
        FIND_PATTERN="layout_results_${CONFIG_ID}_[0-9]*"
        FILE_COUNT=$(find output/layouts -name "$FIND_PATTERN*.csv" | wc -l)
        
        if [ "$FILE_COUNT" -gt 0 ]; then
            TOTAL_COMPLETED=$((TOTAL_COMPLETED+1))
        else
            # Add to pending list
            echo $CONFIG_ID >> $PENDING_FILE
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
                echo $BATCH_FILE >> $BATCHES_FILE
                CURRENT_BATCH=$((CURRENT_BATCH+1))
                CURRENT_COUNT=0
                BATCH_FILE="batch_files/batch_${CURRENT_BATCH}.txt"
                > $BATCH_FILE
            fi
        done < $PENDING_FILE
        
        # Add the last batch if it has any configs
        if [ $CURRENT_COUNT -gt 0 ]; then
            echo $BATCH_FILE >> $BATCHES_FILE
        fi
        
        # Count total batches
        TOTAL_BATCHES=$(wc -l < $BATCHES_FILE)
        echo "Created $TOTAL_BATCHES batch files"
        
        # Reset progress for the new batch list
        echo "0" > $PROGRESS_FILE
    else
        echo "No pending configurations found. All work is complete!"
        exit 0
    fi
fi

# Read the current progress
if [ -f "$PROGRESS_FILE" ]; then
    CURRENT_BATCH=$(cat "$PROGRESS_FILE")
else
    CURRENT_BATCH=0
    echo "0" > $PROGRESS_FILE
fi

# Count total batches
TOTAL_BATCHES=$(wc -l < $BATCHES_FILE)

# Log file for this submission run
LOG_FILE="submission_logs/submission_$(date +%Y%m%d_%H%M%S)_chunk${CURRENT_BATCH}.log"

echo "=== SLURM Managed Submission Chunk ===" | tee -a "$LOG_FILE"
echo "Total batches: $TOTAL_BATCHES" | tee -a "$LOG_FILE"
echo "Starting from batch: $CURRENT_BATCH" | tee -a "$LOG_FILE"
echo "Chunk size: $CHUNK_SIZE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Calculate end batch for this chunk
END_BATCH=$((CURRENT_BATCH + CHUNK_SIZE - 1))
if [ $END_BATCH -ge $TOTAL_BATCHES ]; then
    END_BATCH=$((TOTAL_BATCHES - 1))
fi

echo "This run will submit batches $CURRENT_BATCH through $END_BATCH" | tee -a "$LOG_FILE"

# Array to store job IDs from this chunk
declare -a CHUNK_JOB_IDS

# Submit batches for this chunk
for ((i=CURRENT_BATCH; i<=END_BATCH; i++)); do
    # Get the batch file path (i+1 because file lines are 1-indexed)
    BATCH_FILE=$(sed -n "$((i+1))p" $BATCHES_FILE)
    
    if [ ! -f "$BATCH_FILE" ]; then
        echo "Warning: Batch file $BATCH_FILE not found" | tee -a "$LOG_FILE"
        continue
    fi
    
    CONFIG_COUNT=$(wc -l < $BATCH_FILE)
    echo "Submitting batch $((i+1))/$TOTAL_BATCHES with $CONFIG_COUNT configurations..." | tee -a "$LOG_FILE"
    
    # Submit the job
    JOB_OUTPUT=$(sbatch --export=CONFIG_FILE=$BATCH_FILE slurm_batch_processor.sh 2>&1)
    JOB_STATUS=$?
    
    if [ $JOB_STATUS -eq 0 ]; then
        JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
        echo "  Success: Job ID $JOB_ID" | tee -a "$LOG_FILE"
        CHUNK_JOB_IDS+=($JOB_ID)
    else
        echo "  Failed: $JOB_OUTPUT" | tee -a "$LOG_FILE"
    fi
    
    sleep 2  # Short delay between submissions
done

# Update the progress file for the next run
NEXT_BATCH=$((END_BATCH + 1))
echo $NEXT_BATCH > "$PROGRESS_FILE"

# Check if we've completed all batches
if [ $NEXT_BATCH -ge $TOTAL_BATCHES ]; then
    echo "All batches have been submitted. Submission complete!" | tee -a "$LOG_FILE"
    echo "Final job IDs: ${CHUNK_JOB_IDS[@]}" | tee -a "$LOG_FILE"
else
    # Schedule the next chunk with a delay
    echo "Scheduling next chunk (batches $NEXT_BATCH to $((NEXT_BATCH+CHUNK_SIZE-1)))" | tee -a "$LOG_FILE"
    
    # Submit this script as a job that depends on the completion of this chunk's jobs
    if [ ${#CHUNK_JOB_IDS[@]} -gt 0 ]; then
        # Use afterany dependency to continue even if some jobs fail
        DEPENDENCY_LIST=$(IFS=:; echo "afterany:${CHUNK_JOB_IDS[*]}")
        NEXT_MANAGER=$(sbatch --dependency=$DEPENDENCY_LIST --time=00:10:00 "$0" | awk '{print $4}')
        echo "Next manager job scheduled with ID: $NEXT_MANAGER" | tee -a "$LOG_FILE"
    else
        # If no jobs were submitted in this chunk, continue anyway with a short delay
        echo "No jobs were submitted in this chunk. Scheduling next manager immediately." | tee -a "$LOG_FILE"
        NEXT_MANAGER=$(sbatch --time=00:10:00 "$0" | awk '{print $4}')
        echo "Next manager job scheduled with ID: $NEXT_MANAGER" | tee -a "$LOG_FILE"
    fi
fi

echo "This submission manager completed at $(date)" | tee -a "$LOG_FILE"