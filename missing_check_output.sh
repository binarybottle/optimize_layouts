#!/bin/bash
# missing_check_output.sh

# Set your parameters
TOTAL_CONFIGS=65520 # Total number of configurations, adjust as needed
config_pre="output/configs1/config_" # Config file path prefix
config_post=".yaml" # Config file suffix
output_dir="output/layouts" # Directory where layout results are stored
missing_layouts_file="missing_layouts.txt" # Output file to list missing layouts

# Define the expected output file or directory pattern
output_pre="layout_results_"
output_post="_*.csv"

echo "Starting to check for missing layouts..." > $missing_layouts_file
echo "Script started at $(date)" >> $missing_layouts_file
echo "-------------------------------------" >> $missing_layouts_file

# Create a file to store just the config IDs
echo "# Missing configuration IDs" > missing_config_ids.txt

# Counter for progress tracking
processed=0
missing=0

# Check each configuration
for ((config_id=1; config_id<=TOTAL_CONFIGS; config_id++)); do
    # Check if the config file exists
    config_file="${config_pre}${config_id}${config_post}"
    if [ ! -f "$config_file" ]; then
        echo "Config file missing: $config_file" >> $missing_layouts_file
        continue
    fi

    # Calculate the batch number for reference
    batch_num=$(( (config_id - 1) / 1000 ))
    array_task_id=$(( (config_id - 1) % 1000 ))
    
    # Look for layout file with the pattern layout_results_[configID]_*.csv
    layout_pattern="${output_dir}/${output_pre}${config_id}_*.csv"
    if ! ls $layout_pattern 1> /dev/null 2>&1; then
        echo "Missing layout: config_${config_id} (Batch: ${batch_num}, Array task: ${array_task_id})" >> $missing_layouts_file
        echo "${config_id}" >> missing_config_ids.txt
        missing=$((missing + 1))
    fi
    
    processed=$((processed + 1))
    
    # Print progress every 1000 configs
    if [ $((processed % 1000)) -eq 0 ]; then
        echo "Checked $processed/$TOTAL_CONFIGS configurations, found $missing missing"
    fi
done

echo "-------------------------------------" >> $missing_layouts_file
echo "Total missing layouts: $missing out of $processed checked" >> $missing_layouts_file
echo "Script completed at $(date)" >> $missing_layouts_file

echo "Check complete. Found $missing missing layouts out of $processed checked."
echo "Results saved to $missing_layouts_file"
echo "Config IDs only saved to missing_config_ids.txt"