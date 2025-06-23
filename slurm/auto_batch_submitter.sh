#!/bin/bash
# Automated batch submitter that respects quota limits

MAX_JOBS_ALLOWED=8  # Keep below your quota limit
CHUNK_SIZE=8        # Submit this many at once
CHECK_INTERVAL=300  # Check every 5 minutes (300 seconds)

echo "=== Automated Batch Submitter ==="
echo "Max jobs allowed: $MAX_JOBS_ALLOWED"
echo "Chunk size: $CHUNK_SIZE" 
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

while true; do
    # Check current job count
    CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
    echo "$(date): Current jobs in queue: $CURRENT_JOBS"
    
    # If we have room for more jobs, submit more
    if [ $CURRENT_JOBS -lt $MAX_JOBS_ALLOWED ]; then
        JOBS_TO_SUBMIT=$((MAX_JOBS_ALLOWED - CURRENT_JOBS))
        if [ $JOBS_TO_SUBMIT -gt $CHUNK_SIZE ]; then
            JOBS_TO_SUBMIT=$CHUNK_SIZE
        fi
        
        echo "  Submitting $JOBS_TO_SUBMIT more batches..."
        bash slurm/slurm_array_submit.sh --chunk-size $JOBS_TO_SUBMIT
        
        # Check if submission was successful
        NEW_JOB_COUNT=$(squeue -u $USER -h | wc -l)
        if [ $NEW_JOB_COUNT -gt $CURRENT_JOBS ]; then
            echo "  ✓ Successfully submitted $((NEW_JOB_COUNT - CURRENT_JOBS)) jobs"
        else
            echo "  ! No new jobs submitted (might be done or hit quota)"
        fi
    else
        echo "  At job limit, waiting for completions..."
    fi
    
    # Check if we're done
    TOTAL_COMPLETED=$(find output/layouts -name "*config_*" | wc -l)
    echo "  Progress: $TOTAL_COMPLETED / 95040 configurations completed"
    
    if [ $TOTAL_COMPLETED -ge 95040 ]; then
        echo "🎉 All configurations completed!"
        break
    fi
    
    echo "  Sleeping ${CHECK_INTERVAL}s..."
    echo ""
    sleep $CHECK_INTERVAL
done