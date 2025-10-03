#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files from global Pareto optimal keyboard layouts.

This is Step 2 in a process to optimally arrange the 24 most frequent letters 
in the 24 keys of the home block of a keyboard. 

This script takes the output from run_jobs.py 
(and optionally layouts_[consolidate, compare, filter_scores, filter_patterns]) 
and generates configuration files for the second optimization round with specific 
letter-to-key constraints. If --remove-positions is specified, it will remove 
positions from the global Pareto set, and remove resulting redundant layouts.

Usage:
    # Use all global Pareto solutions, after removing items from specified positions
    python generate_configs2.py --input-file ../output/global_moo_solutions.csv --remove-positions "A;"
    
    # Use "top" N solutions from global Pareto set with specific ranking criteria
    python generate_configs2.py --input-file ../output/layouts_consolidate_moo_solutions.csv --top-n 50 --sort-by global_rank

    # Study
    poetry run python3 generate_configs2.py --input-file ../output/phase1/layouts_filtered_patterns_and_scores.csv

See **README_keyboards.md** for a full description.
"""
import os
import pandas as pd
import argparse
from pathlib import Path


def parse_global_pareto_csv(filepath: str) -> pd.DataFrame:
    """Parse the global Pareto results CSV, handling metadata headers."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find where the actual data starts (after metadata)
        data_start_idx = -1
        for i, line in enumerate(lines):
            line = line.strip()
            # Look for the actual column headers (not metadata)
            if ('config_id' in line or 'items' in line or 'positions' in line or 
                'item_score' in line or 'Complete Item' in line) and ',' in line:
                data_start_idx = i
                break
        
        if data_start_idx == -1:
            # Try alternative approach - skip first several lines and look for CSV data
            print("Could not find header, trying to skip metadata lines...")
            for i in range(10):  # Check first 10 lines
                if i >= len(lines):
                    break
                line = lines[i].strip()
                if line and ',' in line and not line.startswith('"'):
                    data_start_idx = i
                    break
        
        if data_start_idx == -1:
            raise ValueError("Could not find data header in file")
        
        print(f"Found data starting at line {data_start_idx + 1}")
        
        # Read just the data portion
        data_lines = ''.join(lines[data_start_idx:])
        from io import StringIO
        return pd.read_csv(StringIO(data_lines))
        
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        print("Attempting fallback parsing...")
        
        # Fallback: try to read the file directly, skipping problematic rows
        try:
            # Skip first several rows that might be metadata
            df = pd.read_csv(filepath, skiprows=6, on_bad_lines='skip')
            print(f"Fallback parsing successful, loaded {len(df)} rows")
            return df
        except Exception as e2:
            print(f"Fallback parsing also failed: {e2}")
            raise e


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
    
    # Find indices of positions to remove
    remove_indices = set()
    removed_positions = []
    removed_items = []
    
    for i, pos in enumerate(pos_list):
        if pos in positions_to_remove:
            remove_indices.add(i)
            removed_positions.append(pos)
            removed_items.append(item_list[i])
    
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


def generate_config_content(items_assigned, positions_assigned, items_to_assign, positions_to_assign,
                            least_frequent_items="", worst_positions=""):
    """Generate YAML configuration content for Step 2."""
    
    config_template = f"""# Configuration file for item-to-position layout optimization.

#-----------------------------------------------------------------------
# Paths
#-----------------------------------------------------------------------
paths:
  item_pair_score_table:        "input/frequency/english-letter-pair-counts-google-ngrams_normalized.csv"
  item_triple_score_table:      "input/frequency/english-letter-triple-counts-google-ngrams_normalized.csv"
  position_pair_score_table:    "input/engram_2key_scores.csv"
  position_triple_score_table:  None  # "input/engram_3key_order_scores.csv"
  layout_results_folder:        "output/layouts"

#-----------------------------------------------------------------------
# MOO (Multi-Objective Optimization) settings
#-----------------------------------------------------------------------
moo:
  default_objectives: 
    - "engram_key_preference"
    #- "engram_avg4_score"  # engram_[row_separation, same_row, same_finger, outside]
    - "engram_row_separation"
    - "engram_same_row"
    - "engram_same_finger"
    #- "engram_outside"  # Irrelevant to this optimization code
  default_weights: [1.0, 1.0, 1.0, 1.0]
  default_maximize: [true, true, true, true]
  default_max_solutions: 10000
  default_time_limit: 100000
  show_progress_bar: true
  save_detailed_results: true

#-----------------------------------------------------------------------
# Optimization settings
#-----------------------------------------------------------------------
#   _to_assign:    Items to arrange in available positions
#   _assigned:     Items already assigned to positions
#   _to_constrain: Subset of items_to_assign to arrange in positions_to_constrain,
#                  and subset of positions_to_assign to constrain items_to_constrain
optimization:   
  items_assigned:       "{items_assigned}"
  positions_assigned:   "{positions_assigned}"
  items_to_assign:      "{items_to_assign}"
  positions_to_assign:  "{positions_to_assign}"
  items_to_constrain:      "{least_frequent_items}"   
  positions_to_constrain:  "{worst_positions}"  

#-----------------------------------------------------------------------
# Visualization settings
#-----------------------------------------------------------------------
visualization: 
  print_keyboard: True
"""
    return config_template


def main():
    """Main function to process global Pareto layouts and generate Step 2 configurations."""
    args = parse_arguments()
    
    if args.remove_positions:
        positions_to_remove = list(args.remove_positions)
        print(f"Will remove positions: {positions_to_remove}")
    else:
        positions_to_remove = []
        print("No positions specified for removal - using original layouts")

    # Load global Pareto solutions
    print(f"Loading global Pareto solutions from: {args.input_file}")
    try:
        df = parse_global_pareto_csv(args.input_file)
        print(f"Loaded {len(df)} global Pareto solutions")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Validate required columns
    required_cols = ['items', 'positions']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Add score column for sorting if needed
    score_col = None
    possible_score_cols = ['weighted_score', 'global_rank']
    for col in possible_score_cols:
        if col in df.columns:
            score_col = col
            break
    
    if score_col is None:
        print("Warning: No score column found, using original order")
        df['score'] = range(len(df))  # Use index as score
        score_col = 'score'
    
    # Sort by specified column
    if args.sort_by and args.sort_by in df.columns:
        sort_col = args.sort_by
        df = df.sort_values(sort_col, ascending=False)
        print(f"Sorted by {sort_col} (descending)")
    else:
        # Default: sort by score (higher is better)
        df = df.sort_values(score_col, ascending=False)
        print(f"Sorted by {score_col} (descending)")
    
    # Select top N if specified
    if args.top_n:
        df = df.head(args.top_n)
        print(f"Selected top {len(df)} solutions")
    
    # Convert to list of dictionaries for processing
    layouts = []
    for idx, row in df.iterrows():
        layout = {
            'items': row['items'],
            'positions': row['positions'],
            'score': row.get(score_col, 0),
            'rank': idx + 1,
            'source_info': f"Global Pareto #{idx+1}"
        }
        
        # Add additional info if available
        if 'source_file' in row:
            layout['source_file'] = row['source_file']
        if 'global_rank' in row:
            layout['global_rank'] = row['global_rank']
            
        layouts.append(layout)
    
    print(f"Processing {len(layouts)} layouts to generate Step 2 configurations...")
    
    # Generate Step 2 configurations
    os.makedirs(args.output_path, exist_ok=True)
    
    configs_generated = 0
    error_count = 0
    skipped_duplicates = 0
    unique_layouts = {}  # Track unique layouts to avoid duplicates
    
    for i, layout in enumerate(layouts):
        try:
            config_num = f"{i+1}"
            items = layout['items']
            positions = layout['positions']
            
            # Validate layout
            if len(items) != len(positions):
                print(f"Warning: Items/positions length mismatch in config {config_num}")
                error_count += 1
                continue
            
            # Remove specified positions (if any)
            if positions_to_remove:
                remaining_positions, remaining_items, removed_positions = remove_specified_positions(
                    positions, items, positions_to_remove
                )
            else:
                remaining_positions, remaining_items, removed_positions = positions, items, ""

            # Get unassigned items and positions, maintaining original order
            all_positions = "FJDKEISLVMRUWOA;C,Z/QPX."  # 24 positions
            all_items_24 = "etaoinsrhldcumfpgwybvkxj"  # Exactly 24 letters in order

            least_frequent_items = "xj"
            worst_positions = "QPX."

            # Build items_assigned and items_to_assign in correct order
            items_assigned_ordered = ""
            items_to_assign_ordered = ""

            for item in all_items_24:
                # All items should be included, regardless of least_frequent_items
                if item in remaining_items:
                    items_assigned_ordered += item
                else:
                    items_to_assign_ordered += item

            # Unassigned positions should only exclude remaining_positions, not worst_positions
            # worst_positions are part of positions_to_assign (they're just constrained)
            unassigned_positions = ''.join([c for c in all_positions if c not in remaining_positions])

            # Check for duplicate layouts after position removal
            layout_key = (items_assigned_ordered, remaining_positions, items_to_assign_ordered, unassigned_positions)
            
            if layout_key in unique_layouts:
                print(f"Duplicate layout found for config {config_num}, skipping (same as config {unique_layouts[layout_key]['config_num']})")
                skipped_duplicates += 1
                continue
            
            # Store this unique layout
            unique_layouts[layout_key] = {
                'config_num': config_num,
                'layout': layout
            }
            
            # Ensure we have the right counts
            total_items = len(items_assigned_ordered) + len(items_to_assign_ordered)
            total_positions = len(remaining_positions) + len(unassigned_positions)
            if total_items != 24:
                print(f"Warning: Item count mismatch in config {config_num}. Assigned: {len(items_assigned_ordered)}, Unassigned: {len(items_to_assign_ordered)}, Total: {total_items}")
            if total_positions != 24:
                print(f"Warning: Position count mismatch in config {config_num}. Assigned: {len(remaining_positions)}, Unassigned: {len(unassigned_positions)}, Total: {total_positions}")
            
            # Generate config content using properly ordered items
            config_content = generate_config_content(
                items_assigned_ordered, remaining_positions,
                items_to_assign_ordered, unassigned_positions,
                least_frequent_items, worst_positions
            )
            
            # Write config file
            config_file = os.path.join(args.output_path, f"config_{config_num}.yaml")
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            configs_generated += 1
            
            # Show mapping for first few configs
            if configs_generated <= 3:
                print(f"\nConfig #{configs_generated}:")
                print(f"  Source: {layout.get('source_info', 'unknown')}")
                print(f"  Original layout: {items} -> {positions}")
                if removed_positions:
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
    if skipped_duplicates > 0:
        print(f"Skipped {skipped_duplicates} duplicate layouts after position removal")
    
    # Summary statistics
    if configs_generated > 0:
        print(f"\nSummary:")
        print(f"- Input: {len(layouts)} global Pareto optimal solutions")
        print(f"- Duplicates after position removal: {skipped_duplicates}")
        print(f"- Unique configurations generated: {configs_generated}")
        print(f"- Deduplication efficiency: {skipped_duplicates/(len(layouts))*100:.1f}% reduction")
        print(f"- Next step: Run optimization on these {configs_generated} unique configs")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate Step 2 configurations from global Pareto optimal solutions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all global Pareto solutions, after removing items from specified positions
  python generate_configs2.py --input-file ../output/global_moo_solutions.csv --remove-positions "A;"
    
  # Use "top" N solutions from global Pareto set with specific ranking criteria
  python generate_configs2.py --input-file ../output/global_moo_solutions.csv --top-n 50 --sort-by global_rank
        """
    )
    
    parser.add_argument('--input-file', type=str, required=True,
                       help='Path to global Pareto solutions CSV file')
    parser.add_argument('--output-path', type=str, default='../output/configs2',
                       help='Path to output Step 2 configurations')
    parser.add_argument('--remove-positions', type=str, required=False,
                       help='Positions to remove (e.g., "A;" for A and semicolon)')
    parser.add_argument('--top-n', type=int, default=None,
                       help='Number of top solutions to use (None = all)')
    parser.add_argument('--sort-by', type=str, default=None,
                       choices=['global_rank', 'item_score', 'item_pair_score', 'total_score'],
                       help='Column to sort by before selecting top-n')
    
    return parser.parse_args()


if __name__ == "__main__":
    main()