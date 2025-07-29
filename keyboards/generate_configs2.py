#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files from global Pareto optimal keyboard layouts.

This script takes the output from select_global_moo_solutions.py and generates
configuration files for the next round of optimization with specific 
letter-to-key constraints. If --remove-positions is specified, it will remove 
positions from the global Pareto set, and remove redundant layouts.

This is Step 2 in the process, but now using globally optimal solutions
instead of processing thousands of individual result files.

Usage:
    # Use all global Pareto solutions
    python generate_configs2.py --input-file ../output/global_moo_solutions.csv --remove-positions "A;OW"
    
    # Use top N solutions from global Pareto set
    python generate_configs2.py --input-file ../output/global_moo_solutions.csv --top-n 20 --remove-positions "A;"
    
    # Use solutions with specific ranking criteria
    python generate_configs2.py --input-file ../output/global_moo_solutions.csv --top-n 50 --sort-by global_rank

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


def generate_config_content(items_assigned, positions_assigned, items_to_assign, positions_to_assign):
    """Generate YAML configuration content for Step 2."""
    
    config_template = f"""# optimize_layouts/config.yaml
# Configuration file for item-to-position layout optimization - Step 2
# Generated from global Pareto optimal solutions

#-----------------------------------------------------------------------
# Paths
#-----------------------------------------------------------------------
paths:
  input:
    raw_item_scores_file:          "input/frequency/spanish-letter-counts-leipzig.csv"
    raw_item_pair_scores_file:     "input/frequency/spanish-letter-pair-counts-leipzig.csv"
    raw_position_scores_file:      "input/comfort/key-comfort-scores.csv"
    raw_position_pair_scores_file: "input/comfort/key-pair-comfort-scores.csv"
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
    possible_score_cols = ['global_rank', 'item_score', 'item_pair_score', 'total_score']
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
        # For global_rank, lower is better; for scores, higher is better
        ascending = (sort_col == 'global_rank')
        df = df.sort_values(sort_col, ascending=ascending)
        print(f"Sorted by {sort_col} ({'ascending' if ascending else 'descending'})")
    elif score_col == 'global_rank':
        # Default: sort by global_rank if available (lower is better)
        df = df.sort_values('global_rank', ascending=True)
        print("Sorted by global_rank (ascending)")
    else:
        # Default: sort by score (higher is better)
        ascending = (score_col == 'global_rank')
        df = df.sort_values(score_col, ascending=ascending)
        print(f"Sorted by {score_col}")
    
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
            
            # Check for expected layout size (should be 16 for optimal layouts)
            if len(items) != 16 or len(positions) != 16:
                if len(items) != 16:  # Only warn if it's not the expected size
                    print(f"Info: Layout {config_num} has {len(items)} items (expected 16)")
                # Continue processing - global Pareto might have different sizes
            
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
            
            # Debug: ensure we have the right counts
            total_items = len(items_assigned_ordered) + len(items_to_assign_ordered)
            total_positions = len(remaining_positions) + len(unassigned_positions)
            
            if total_items != 24:
                print(f"Warning: Item count mismatch in config {config_num}. Assigned: {len(items_assigned_ordered)}, Unassigned: {len(items_to_assign_ordered)}, Total: {total_items}")
            if total_positions != 24:
                print(f"Warning: Position count mismatch in config {config_num}. Assigned: {len(remaining_positions)}, Unassigned: {len(unassigned_positions)}, Total: {total_positions}")
            
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
                print(f"  Source: {layout.get('source_info', 'unknown')}")
                if 'global_rank' in layout:
                    print(f"  Global rank: {layout['global_rank']}")
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
  # Use all global Pareto solutions
  python generate_configs2.py --input-file ../output/global_moo_solutions.csv --remove-positions "A;"
  
  # Use top 20 solutions by global rank
  python generate_configs2.py --input-file ../output/global_moo_solutions.csv --top-n 20 --remove-positions "A;"
  
  # Sort by item score and take top 50
  python generate_configs2.py --input-file ../output/global_moo_solutions.csv --top-n 50 --sort-by item_score
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