#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files to run keyboard layout optimizations in parallel 
with specific letter-to-key constraints specified in each file.

This is Step 2 in a process to optimally arrange the 24 most frequent letters 
in the 24 keys of the home block of a keyboard. 

There are two versions of this script:
  - generate_configs1.py generates an initial set of sparse keyboard layouts as config files.
  - optimize_layout.py generates optimal keyboard layouts for a given config file.
  - generate_configs2.py generates a new set of config files based on the optimal keyboard layouts.

Usage (input files assumed to be in output/layouts1/):

    All unique layouts from all config files (default), with optional removal of letters in specified positions:
    ``python generate_configs2.py --remove-positions "A;" --file-pattern "moo_results_config_*.csv"``

    Top layouts per config (number of top-scoring layouts per config file):
    ``python generate_configs2.py --remove-positions "A;" --layouts-per-config 10``

    Top layouts across all configs (number of top-scoring layouts across all config files):
    ``python generate_configs2.py --remove-positions "A;" --top-across-all 1000``

See **README_keyboards.md** for a full description.

See **README.md** for instructions to run batches of config files in parallel.
"""
import os
import glob
import csv
import argparse
from pathlib import Path

def remove_specified_positions(positions, items, positions_to_remove):
    """
    Remove specified positions from a layout.
    PRESERVES the original item order by only removing items at the specified positions.
    
    Args:
        positions: String of positions from Step 1 (e.g., "FDESRJKUIMVLA;OW")
        items: String of items from Step 1 (e.g., "etaoinsrhldcumfp")
        positions_to_remove: List of position characters to remove (e.g., ['A', ';'])
        
    Returns:
        tuple: (remaining_positions, remaining_items, removed_positions_str)
    """
    
    # Convert to lists
    pos_list = list(positions.strip())
    item_list = list(items.strip())
    
    if len(pos_list) != len(item_list):
        print(f"Warning: Position/item length mismatch: {len(pos_list)} vs {len(item_list)}")
        return positions, items, ""
    
    #print(f"Removing positions: {positions_to_remove}")
    
    # Find indices of positions to remove
    remove_indices = set()
    removed_positions = []
    removed_items = []
    
    for i, pos in enumerate(pos_list):
        if pos in positions_to_remove:
            remove_indices.add(i)
            removed_positions.append(pos)
            removed_items.append(item_list[i])
    
    #print(f"Found and removing: {removed_positions} (items: {removed_items})")
    
    # Check if we found all requested positions
    missing_positions = [pos for pos in positions_to_remove if pos not in pos_list]
    if missing_positions:
        print(f"Warning: Requested positions not found in layout: {missing_positions}")
    
    # Build remaining lists in ORIGINAL order
    remaining_positions = []
    remaining_items = []
    
    for i, (pos, item) in enumerate(zip(pos_list, item_list)):
        if i not in remove_indices:
            remaining_positions.append(pos)
            remaining_items.append(item)
    
    # Convert back to strings
    remaining_positions_str = ''.join(remaining_positions)
    remaining_items_str = ''.join(remaining_items)
    removed_positions_str = ''.join(removed_positions)
    
    return remaining_positions_str, remaining_items_str, removed_positions_str

def parse_layout_results(results_path, top_n=None):
    """
    Parse layout results files to extract layouts from Step 1.
    
    Args:
        results_path: Path to CSV file with layout results
        top_n: Number of top layouts to extract (None = all)
        
    Returns:
        List of dictionaries with layout information
    """
    layouts = []
    
    try:
        with open(results_path, 'r') as f:
            reader = csv.reader(f)
            
            # Skip first 4 lines: title, items info, positions info, empty line
            for i in range(4):
                next(reader, None)
            
            # Read header row (line 5)
            header = next(reader, None)
            if not header:
                print(f"Warning: Could not find header row in {results_path}")
                return layouts
            
            # Read data rows (starting from line 6)
            count = 0
            for row_num, row in enumerate(reader):
                if not row or len(row) < 3:  # Need at least rank, items, positions
                    continue
                
                if top_n is not None and count >= top_n:
                    break
                
                try:
                    # CSV format: Rank, Items, Positions, Opt Item Score, Opt Item-Pair Score, Opt Combined, ...
                    rank = int(row[0])
                    items = row[1]
                    positions = row[2]
                    
                    # Use "Opt Combined" score (column 5) as the main score
                    score = float(row[5]) if len(row) > 5 else 0.0
                    
                    # Clean up positions
                    positions = positions.replace('[semicolon]', ';')
                    
                    layout = {
                        'items': items,
                        'positions': positions,
                        'score': score,
                        'rank': rank
                    }
                    
                    layouts.append(layout)
                    count += 1
                    
                except Exception as row_error:
                    print(f"Warning: Error processing row {row_num+6}: {row_error}")
                    continue
                
        return layouts
        
    except Exception as e:
        print(f"Error parsing {results_path}: {e}")
        return layouts

def generate_config_content(items_assigned, positions_assigned, items_to_assign, positions_to_assign):
    """Generate YAML configuration content for Step 2."""
    
    config_template = f"""# optimize_layouts/config.yaml
# Configuration file for item-to-position layout optimization - Step 2
# Generated from Step 1 results

#-----------------------------------------------------------------------
# Paths
#-----------------------------------------------------------------------
paths:
  input:
    raw_item_scores_file:          "input/letter_frequencies_english.csv"
    raw_item_pair_scores_file:     "input/letter_pair_frequencies_english.csv"
    raw_position_scores_file:      "input/key_comfort_estimates.csv"
    raw_position_pair_scores_file: "input/key_pair_comfort_estimates.csv"    
    item_scores_file:              "output/normalized_input/normalized_item_scores.csv"
    item_pair_scores_file:         "output/normalized_input/normalized_item_pair_scores.csv"
    position_scores_file:          "output/normalized_input/normalized_position_scores.csv"
    position_pair_scores_file:     "output/normalized_input/normalized_position_pair_scores.csv"
  output:
    layout_results_folder:         "output/layouts"

#-----------------------------------------------------------------------
# Optimization Settings
#-----------------------------------------------------------------------
optimization:   
  items_assigned:       "{items_assigned}"
  positions_assigned:   "{positions_assigned}"
  items_to_assign:      "{items_to_assign}"
  positions_to_assign:  "{positions_to_assign}"

  # Subset constraints (empty for Step 2)
  items_to_constrain:      ""   
  positions_to_constrain:  ""  

#-----------------------------------------------------------------------
# Visualization Settings
#-----------------------------------------------------------------------
visualization: 
  print_keyboard: True
"""
    return config_template

def main():
    """Main function to process layouts and generate Step 2 configurations."""
    args = parse_arguments()
    
    if args.remove_positions:
        positions_to_remove = list(args.remove_positions)
        print(f"Will remove positions: {positions_to_remove}")
    else:
        positions_to_remove = []
        print("No positions specified for removal - using original layouts")

    # Find all Step 1 result files and sort them to ensure consistent order
    results_pattern = os.path.join(args.results_path, args.file_pattern)
    result_files = sorted(glob.glob(results_pattern))  # Sort for consistent order
    
    if not result_files:
        print(f"No result files found matching: {results_pattern}")
        return
    
    print(f"Found {len(result_files)} CSV files (processing in order)")
    
    # Parse all layouts while preserving file order and row order
    all_layouts = []
    
    for file_idx, file_path in enumerate(result_files):
        print(f"Processing file {file_idx + 1}: {os.path.basename(file_path)}")
        if args.layouts_per_config:
            layouts = parse_layout_results(file_path, top_n=args.layouts_per_config)
        else:
            layouts = parse_layout_results(file_path)
        
        # Add file source information to each layout
        for layout in layouts:
            layout['source_file'] = os.path.basename(file_path)
            layout['file_index'] = file_idx
            
        all_layouts.extend(layouts)
        print(f"  Extracted {len(layouts)} layouts")
    
    if not all_layouts:
        print("No layouts extracted from result files")
        return
    
    # Apply selection strategy
    if args.top_across_all:
        print(f"Selecting top {args.top_across_all} layouts across all configs")
        print(f"WARNING: This will change the order - config files will NOT correspond to original rows!")
        all_layouts.sort(key=lambda x: x['score'], reverse=True)
        selected_layouts = all_layouts[:args.top_across_all]
        print(f"Selected {len(selected_layouts)} layouts using top-across-all approach")
    else:
        selected_layouts = all_layouts
        print(f"Extracted {len(selected_layouts)} layouts preserving original file/row order")
    
    # Generate Step 2 configurations
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Processing {len(selected_layouts)} layouts to generate Step 2 configurations...")
    print(f"Removing positions: {positions_to_remove}")
    
    configs_generated = 0
    error_count = 0
    
    for i, layout in enumerate(selected_layouts):
        try:
            config_num = f"{i+1}"
            items = layout['items']
            positions = layout['positions']
            
            # Validate layout
            if len(items) != len(positions):
                print(f"Warning: Items/positions length mismatch in config {config_num}")
                error_count += 1
                continue
            
            # Filter out non-16 item layouts (outliers)
            if len(items) != 16 or len(positions) != 16:
                if len(items) != 16:  # Only warn if it's not the expected size
                    print(f"Info: Skipping config {config_num} with {len(items)} items (expected 16)")
                error_count += 1
                continue
            
            # Remove specified positions (if any)
            if positions_to_remove:
                remaining_positions, remaining_items, removed_positions = remove_specified_positions(
                    positions, items, positions_to_remove
                )
            else:
                remaining_positions, remaining_items, removed_positions = positions, items, ""

            # Get unassigned items and positions, maintaining original frequency order
            all_positions = "FDESVRWACQZXJKILMUO;,P/."  # 24 positions
            all_items_24 = "etaoinsrhldcumfpgwybvkxj"  # Exactly 24 letters in frequency order
            
            # Build items_assigned and items_to_assign in correct frequency order
            items_assigned_ordered = ""
            items_to_assign_ordered = ""
            
            for item in all_items_24:
                if item in remaining_items:
                    items_assigned_ordered += item
                else:
                    items_to_assign_ordered += item
            
            unassigned_positions = ''.join([c for c in all_positions if c not in remaining_positions])
            
            # Debug: ensure we have the right counts
            if len(items_assigned_ordered) + len(items_to_assign_ordered) != 24:
                print(f"Warning: Item count mismatch. Assigned: {len(items_assigned_ordered)}, Unassigned: {len(items_to_assign_ordered)}, Total: {len(items_assigned_ordered) + len(items_to_assign_ordered)}")
            if len(remaining_positions) + len(unassigned_positions) != 24:
                print(f"Warning: Position count mismatch. Assigned: {len(remaining_positions)}, Unassigned: {len(unassigned_positions)}, Total: {len(remaining_positions) + len(unassigned_positions)}")
            
            # Generate config content using properly ordered items
            config_content = generate_config_content(
                items_assigned_ordered, remaining_positions,
                items_to_assign_ordered, unassigned_positions
            )
            
            # Write config file
            config_file = os.path.join(args.output_path, f"config_{config_num}.yaml")
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            configs_generated += 1
            
            # Show mapping for first few configs
            if configs_generated <= 3:
                print(f"\nConfig #{configs_generated}:")
                print(f"  Source: {layout.get('source_file', 'unknown')} row {layout.get('rank', '?')}")
                print(f"  Original layout: {items} -> {positions}")
                print(f"  After removing {removed_positions}: {remaining_items} -> {remaining_positions}")
                print(f"  Items assigned (freq order): {items_assigned_ordered}")
                print(f"  Items to assign (freq order): {items_to_assign_ordered}")
                print(f"  Config file: {os.path.basename(config_file)}")
        
        except Exception as e:
            print(f"Error processing layout {i+1}: {e}")
            error_count += 1
            continue
    
    print(f"\nStep 2 complete!")
    print(f"Generated {configs_generated} configuration files in '{args.output_path}'")
    if error_count > 0:
        print(f"Skipped {error_count} layouts due to errors")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Step 2 configurations from Step 1 results')
    
    parser.add_argument('--results-path', type=str, default='../output/layouts1',
                       help='Path to Step 1 results directory')
    parser.add_argument('--output-path', type=str, default='../output/configs2',
                       help='Path to output Step 2 configurations')
    parser.add_argument('--file-pattern', type=str, default='moo_results_config_*.csv',
                       help='File pattern to match (default: "moo_results_config_*.csv")')
    parser.add_argument('--remove-positions', type=str, required=False,
                       help='Positions to remove (e.g., "A;" for A and semicolon)')
    parser.add_argument('--layouts-per-config', type=int, default=None,
                       help='Number of top layouts to take from each config file')
    parser.add_argument('--top-across-all', type=int, default=None,
                       help='Take top N layouts across all configs (overrides per-config)')
    
    return parser.parse_args()

if __name__ == "__main__":
    main()