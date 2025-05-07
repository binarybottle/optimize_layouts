#!/bin/bash

total_batches=$(ls missing_batches/batch_*.txt | wc -l)
current=1
failed=0

for batch_file in missing_batches/batch_*.txt; do
    lines=$(wc -l < $batch_file)
    if [ $lines -gt 0 ]; then
        echo "Submitting batch $batch_file with $lines configs"
        
        max_retries=3
        retry=0
        success=0
        
        while [ $retry -lt $max_retries ] && [ $success -eq 0 ]; do
            submission=$(sbatch --array=1-$lines%50 --export=BATCH_FILE=$batch_file missing_submit_batch.sh 2>&1)
            
            if [[ "$submission" == *"Submitted batch job"* ]]; then
                job_id=$(echo "$submission" | awk '{print $4}')
                echo "  Submitted as job $job_id"
                success=1
            else
                retry=$((retry + 1))
                echo "  Submission attempt $retry failed: $submission"
                sleep 60  # Wait longer between retries
            fi
        done
        
        if [ $success -eq 0 ]; then
            echo "$batch_file" >> failed_submissions.txt
            echo "  Failed to submit after $max_retries attempts"
        fi
        
        sleep 30  # Delay between batches
    fi
done

echo "Completed. $((current-1-failed)) batches submitted successfully, $failed batches failed."
if [ $failed -gt 0 ]; then
    echo "Failed batch files are listed in failed_submissions.txt"
fi