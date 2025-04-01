#!/bin/bash
# Script to submit batches of keyboard layout optimization jobs with persistence
# Save the current batch number in a file for resuming later

# Total number of configurations
TOTAL_CONFIGS=65520

# SLURM array limit per batch
BATCH_SIZE=1000

# Calculate number of batches needed (ceiling division)
NUM_BATCHES=$(( (TOTAL_CONFIGS + BATCH_SIZE - 1) / BATCH_SIZE ))

# Maximum number of concurrent batches to submit
MAX_CONCURRENT_BATCHES=5

# File to store progress
PROGRESS_FILE="batch_submission_progress.txt"

# Check if progress file exists and we're resuming
if [ -f "$PROGRESS_FILE" ]; then
    batch=$(cat "$PROGRESS_FILE")
    echo "Resuming from batch $batch"
else
    batch=0
    echo "Starting new submission from batch 0"
fi

echo "Submitting $NUM_BATCHES batches for $TOTAL_CONFIGS configurations"
echo "Maximum concurrent batches: $MAX_CONCURRENT_BATCHES"
echo "Current batch: $batch"

# Array to store job IDs
declare -a JOB_IDS

# Create a log file
LOG_FILE="batch_submission_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Detailed log in: $LOG_FILE"

# Function to log messages
log_message() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

log_message "Starting submission process for batches $batch to $((NUM_BATCHES-1))"

while [ $batch -lt $NUM_BATCHES ]; do
    # Save progress
    echo $batch > "$PROGRESS_FILE"
    
    # Check number of currently running/pending jobs
    CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
    
    # If we're at the concurrent job limit, wait and check again
    if [ $CURRENT_JOBS -ge $MAX_CONCURRENT_BATCHES ]; then
        log_message "Current job count: $CURRENT_JOBS - Waiting for jobs to complete before submitting more..."
        sleep 300  # Wait 5 minutes before checking again
        continue
    fi
    
    # Calculate start and end indices for this batch
    START_IDX=$((batch * BATCH_SIZE + 1))
    END_IDX=$((START_IDX + BATCH_SIZE - 1))
    
    # Make sure we don't exceed the total
    if [ $END_IDX -gt $TOTAL_CONFIGS ]; then
        END_IDX=$TOTAL_CONFIGS
    fi
    
    # Calculate array size for this batch
    ARRAY_SIZE=$((END_IDX - START_IDX + 1))
    
    log_message "Submitting batch $((batch+1))/$NUM_BATCHES: configs $START_IDX-$END_IDX"
    
    # Submit with appropriate array range for the last batch
    if [ $batch -eq $((NUM_BATCHES-1)) ] && [ $ARRAY_SIZE -lt $BATCH_SIZE ]; then
        # For the last batch, we might need fewer array indices
        LAST_ARRAY_IDX=$((ARRAY_SIZE-1))
        JOB_ID=$(sbatch --array=0-$LAST_ARRAY_IDX%1000 --export=BATCH_NUM=$batch slurm_batchmaking.sh | awk '{print $4}')
    else
        # Full batch
        JOB_ID=$(sbatch --export=BATCH_NUM=$batch slurm_batchmaking.sh | awk '{print $4}')
    fi
    
    # Check if job submission was successful
    if [[ -n $JOB_ID && $JOB_ID =~ ^[0-9]+$ ]]; then
        log_message "  Submitted job $JOB_ID"
        JOB_IDS+=($JOB_ID)
        batch=$((batch+1))
    else
        log_message "  Job submission failed, will retry in 60 seconds..."
        sleep 60
        continue
    fi
    
    # Optional: add small delay between submissions
    sleep 5
done

log_message "All batches submitted!"
log_message "Job IDs: ${JOB_IDS[@]}"

# Remove progress file once completed
rm -f "$PROGRESS_FILE"