#!/bin/bash
# Script to submit batches of keyboard layout optimization jobs with dependency handling

# NOTE: Set TOTAL_CONFIGS below to the total number of config files!

# Total number of configurations
TOTAL_CONFIGS=65520

# SLURM array limit per batch
BATCH_SIZE=1000

# Calculate number of batches needed (ceiling division)
NUM_BATCHES=$(( (TOTAL_CONFIGS + BATCH_SIZE - 1) / BATCH_SIZE ))

# Maximum number of concurrent batches to submit (adjust based on your QoS limits)
MAX_CONCURRENT_BATCHES=5

echo "Submitting $NUM_BATCHES batches for $TOTAL_CONFIGS configurations"
echo "Maximum concurrent batches: $MAX_CONCURRENT_BATCHES"

# Array to store job IDs
declare -a JOB_IDS

batch=0
while [ $batch -lt $NUM_BATCHES ]; do
    # Check number of currently running/pending jobs
    CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
    
    # If we're at the concurrent job limit, wait and check again
    if [ $CURRENT_JOBS -ge $MAX_CONCURRENT_BATCHES ]; then
        echo "Current job count: $CURRENT_JOBS - Waiting for jobs to complete before submitting more..."
        sleep 60  # Wait 1 minute before checking again
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
    
    echo "Submitting batch $((batch+1))/$NUM_BATCHES: configs $START_IDX-$END_IDX"
    
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
        echo "  Submitted job $JOB_ID"
        JOB_IDS+=($JOB_ID)
        batch=$((batch+1))
    else
        echo "  Job submission failed, will retry in 60 seconds..."
        sleep 60
        continue
    fi
    
    # Optional: add small delay between submissions
    sleep 5
done

echo "All batches submitted!"
echo "Job IDs: ${JOB_IDS[@]}"
