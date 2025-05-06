#!/bin/bash
# SLURM Submission manager with checkpoint
# This script submits a small chunk of batches and schedules itself to continue

CONFIGURATION
===================================================================
TOTAL_CONFIGS=<YOUR_TOTAL_CONFIGS>  # Replace <YOUR_TOTAL_CONFIGS> with total number of configurations
BATCH_SIZE=1000                     # SLURM array limit per batch
CHUNK_SIZE=3                        # Number of batches to submit in each manager run
===================================================================

# Calculate number of batches needed (ceiling division)
NUM_BATCHES=$(( (TOTAL_CONFIGS + BATCH_SIZE - 1) / BATCH_SIZE ))

# Create a log directory if it doesn't exist
mkdir -p submission_logs

# Progress file
PROGRESS_FILE="batch_submission_progress.txt"

# Read the current batch from the progress file
if [ -f "$PROGRESS_FILE" ]; then
    START_BATCH=$(cat "$PROGRESS_FILE")
else
    START_BATCH=0
fi

# Log file for this run
LOG_FILE="submission_logs/batch_submission_$(date +%Y%m%d_%H%M%S)_chunk${START_BATCH}.log"

echo "=== SLURM Submission Manager ===" | tee -a "$LOG_FILE"
echo "Total configurations: $TOTAL_CONFIGS" | tee -a "$LOG_FILE"
echo "Total batches: $NUM_BATCHES" | tee -a "$LOG_FILE"
echo "Starting from batch: $START_BATCH" | tee -a "$LOG_FILE"
echo "Chunk size: $CHUNK_SIZE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Calculate end batch for this chunk
END_BATCH=$((START_BATCH + CHUNK_SIZE - 1))
if [ $END_BATCH -ge $NUM_BATCHES ]; then
    END_BATCH=$((NUM_BATCHES - 1))
fi

echo "This run will submit batches $START_BATCH through $END_BATCH" | tee -a "$LOG_FILE"

# Array to store job IDs from this chunk
declare -a CHUNK_JOB_IDS

# Submit batches for this chunk
for ((batch=START_BATCH; batch<=END_BATCH; batch++)); do
    # Calculate start and end indices for this batch
    START_IDX=$((batch * BATCH_SIZE + 1))
    END_IDX=$((START_IDX + BATCH_SIZE - 1))
    
    # Make sure we don't exceed the total
    if [ $END_IDX -gt $TOTAL_CONFIGS ]; then
        END_IDX=$TOTAL_CONFIGS
    fi
    
    # Calculate array size for this batch
    ARRAY_SIZE=$((END_IDX - START_IDX + 1))
    
    echo "Submitting batch $((batch+1))/$NUM_BATCHES: configs $START_IDX-$END_IDX" | tee -a "$LOG_FILE"
    
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
        echo "  Submitted job $JOB_ID" | tee -a "$LOG_FILE"
        CHUNK_JOB_IDS+=($JOB_ID)
    else
        echo "  Job submission failed for batch $batch" | tee -a "$LOG_FILE"
    fi
    
    # Add a small delay between submissions
    sleep 30
done

# Update the progress file for the next run
NEXT_BATCH=$((END_BATCH + 1))
echo $NEXT_BATCH > "$PROGRESS_FILE"

# Check if we've completed all batches
if [ $NEXT_BATCH -ge $NUM_BATCHES ]; then
    echo "All batches have been submitted. Submission complete!" | tee -a "$LOG_FILE"
    echo "Final job IDs: ${CHUNK_JOB_IDS[@]}" | tee -a "$LOG_FILE"
else
    # Schedule the next chunk with a short delay to avoid overwhelming the scheduler
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