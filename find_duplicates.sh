#!/bin/bash

# Directory containing layout files
LAYOUTS_DIR="output/layouts"

# Output file for results
OUTPUT_FILE="duplicate_layouts_report.txt"

# Temporary directory for processing
TEMP_DIR="temp_layout_analysis"
mkdir -p "$TEMP_DIR"

echo "Finding duplicate layouts based on Positions field..."
echo "This may take some time depending on the number of files."

# Process each CSV file
find "$LAYOUTS_DIR" -name "layout_results_*.csv" | while read file; do
    # Extract the configuration ID from filename
    config_id=$(echo "$file" | grep -o "layout_results_[0-9]*" | cut -d'_' -f3)
    
    # Extract the "Positions" value from the last line of the file
    # We need to handle CSV format carefully - the Positions field should be the 2nd column in the last line
    positions=$(tail -n 1 "$file" | awk -F'",' '{print $2}' | tr -d '"')
    
    # Skip files with empty positions (might be corrupted files)
    if [ -z "$positions" ]; then
        echo "Warning: Empty positions in file $file" >> "$OUTPUT_FILE.warnings"
        continue
    fi
    
    # Create a file with config_id as content and positions as filename (for easy grouping)
    # We need to ensure the positions are filesystem-safe
    safe_positions=$(echo "$positions" | md5sum | cut -d ' ' -f1)
    echo "$config_id:$file:$positions" >> "$TEMP_DIR/$safe_positions"
done

echo "Analyzing results..."

# Find files with duplicate positions
echo "=== Duplicate Layout Files Report ===" > "$OUTPUT_FILE"
echo "Generated on $(date)" >> "$OUTPUT_FILE"
echo "---------------------------------------" >> "$OUTPUT_FILE"

duplicates_found=0

# Process each unique position
for position_file in "$TEMP_DIR"/*; do
    # Count files with this position
    count=$(wc -l < "$position_file")
    
    # If more than one file has this position, it's a duplicate
    if [ "$count" -gt 1 ]; then
        # Get the actual position value from the first line
        first_line=$(head -n 1 "$position_file")
        actual_position=$(echo "$first_line" | cut -d':' -f3)
        
        echo "Position: $actual_position" >> "$OUTPUT_FILE"
        echo "Found in $count files:" >> "$OUTPUT_FILE"
        
        # List all files with this position
        cat "$position_file" | while read line; do
            config_id=$(echo "$line" | cut -d':' -f1)
            filename=$(echo "$line" | cut -d':' -f2)
            echo "  - Config ID: $config_id, File: $filename" >> "$OUTPUT_FILE"
        done
        
        echo "---------------------------------------" >> "$OUTPUT_FILE"
        duplicates_found=$((duplicates_found + 1))
    fi
done

# Summary
if [ "$duplicates_found" -eq 0 ]; then
    echo "No duplicate layouts found." >> "$OUTPUT_FILE"
else
    echo "Found $duplicates_found sets of duplicate layouts." >> "$OUTPUT_FILE"
fi

echo "Analysis complete. Results saved to $OUTPUT_FILE"

# Clean up temporary files
rm -rf "$TEMP_DIR"