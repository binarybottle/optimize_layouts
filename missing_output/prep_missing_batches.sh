#!/bin/bash

# Create directory for batch files
mkdir -p missing_batches

# Group configs into batches of 100 (adjust as needed)
BATCH_SIZE=100
line_count=0
batch_num=1
batch_file="missing_batches/batch_${batch_num}.txt"

> $batch_file  # Initialize first batch file

while read config_id; do
    # Skip comment lines
    [[ "$config_id" =~ ^#.*$ ]] && continue
    
    echo $config_id >> $batch_file
    line_count=$((line_count + 1))
    
    if [ $line_count -eq $BATCH_SIZE ]; then
        batch_num=$((batch_num + 1))
        batch_file="missing_batches/batch_${batch_num}.txt"
        > $batch_file  # Initialize new batch file
        line_count=0
    fi
done < missing_config_ids.txt

# Count total batches
total_batches=$batch_num
if [ $line_count -eq 0 ]; then
    total_batches=$((total_batches - 1))
fi

echo "Created $total_batches batch files in missing_batches/"