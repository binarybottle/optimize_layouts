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

See **README_keyboards.md** for a full description.

See **README.md** for instructions to run batches of config files in parallel.

Usage:

    Per-config approach (default: 1 layout per config file):
    ``python generate_keyboard_configs1.py --layouts-per-config 10``

    Across-all approach (top 1,000 layouts across all config files):
    ``python generate_keyboard_configs2.py --top-across-all 1000``

    Both approaches together:
    ``python generate_keyboard_configs2.py --layouts-per-config 10 --top-across-all 1000``

"""
import os
import yaml
import csv
import argparse
import glob
from collections import defaultdict
import sys

# Configuration: output directory and number of layouts per configuration
OUTPUT_DIR = '../output/configs2'
CONFIG_FILE = '../config.yaml'
nlayouts = 10 

# Positions (left right): FDESVRWACQZX JKILMUO;,P/.  
#MOST_COMFORTABLE_KEYS  = "FDESVRWJKILMUO" # 14 most comfortable keys
#LEAST_COMFORTABLE_KEYS = "ACQZX;,P/."     # 10 least comfortable keys
MOST_COMFORTABLE_KEYS  = "FDESVRJKILMU"  # 12 most comfortable keys
LEAST_COMFORTABLE_KEYS = "WACQZXO;,P/."  # 12 least comfortable keys

# Least frequent of 24 letters (`vkxj` in English) to add to items_to_assign
least_frequent_items_of_24 = 'vkxj'

# Base configuration from the original config file
with open(CONFIG_FILE, 'r') as f:
    base_config = yaml.safe_load(f)

def parse_layout_results(results_path, top_n=1):
    """
    Parse layout results files to extract the top N layouts from Step #1.
    
    Args:
        results_path: Path to CSV file with layout results
        top_n: Number of top layouts to extract
        
    Returns:
        List of dictionaries with layout information
    """
    layouts = []
    
    try:
        with open(results_path, 'r') as f:
            reader = csv.reader(f)
            
            # Skip configuration info (first 9 lines)
            for _ in range(9):
                next(reader, None)
            
            # Read header row (line 9)
            header = next(reader, None)
            if not header:
                print(f"Warning: Could not find header row in {results_path}")
                return layouts
            
            # Debug: Print header
            # print(f"Header found: {header}")
            
            # Read top N data rows
            for _ in range(top_n):
                layout_row = next(reader, None)
                if not layout_row or len(layout_row) < 5:  # Ensure row has enough data
                    # Debug: print(f"No data row found or insufficient columns: {layout_row}")
                    break
                
                # Extract all items and positions (including assigned ones)
                # Your CSV has "Items","Positions","Optimized Items","Optimized Positions"
                all_items = layout_row[2]
                all_positions = layout_row[3]
                all_positions = all_positions.replace('[semicolon]', ';')
                rank = int(layout_row[4]) if len(layout_row) > 4 else 0
                score = float(layout_row[5]) if len(layout_row) > 5 else 0
                
                layout = {
                    'items': all_items,
                    'positions': all_positions,
                    'score': score,
                    'rank': rank
                }
                
                layouts.append(layout)
                
        return layouts
        
    except Exception as e:
        print(f"Error parsing {results_path}: {e}")
        return layouts
    
def find_layout_results(step1_dir, layouts_per_config=1):
    """
    Find layout result files from Step #1, taking the top N layouts from each config.
    
    Args:
        step1_dir: Directory containing Step #1 results
        layouts_per_config: Number of top layouts to take from each config file
        
    Returns:
        List of dictionaries containing layout information
    """
    results = []
    # Group result files by config number
    config_results = defaultdict(list)
    
    pattern = os.path.join(step1_dir, "layout_results_*.csv")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"No layout result files found in {step1_dir}")
        return results
    
    print(f"Found {len(result_files)} result files from Step #1")
    
    # First, organize files by config number
    for file_path in result_files:
        try:
            # Extract config number from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            if len(parts) > 2:
                try:
                    config_num = int(parts[2])
                except ValueError:
                    config_num = 0
            else:
                config_num = 0

            config_results[config_num].append(file_path)
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not extract config number from {file_path}: {e}")
            config_results[0].append(file_path)
                
    print(f"Found results for {len(config_results)} different Step #1 configurations")
    
    # Process each config's results
    for config_num, file_paths in config_results.items():
        # Sort by timestamp to get the latest result file for this config
        file_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Take the most recent file for this config
        latest_file = file_paths[0]
        
        try:
            # Parse the CSV file based on the format we've seen
            with open(latest_file, 'r') as f:
                reader = csv.reader(f)
                
                # Skip configuration info (first 9 lines)
                for _ in range(9):
                    next(reader, None)
                
                # Read header row
                header = next(reader, None)
                if not header:
                    continue
                
                # Read data rows for this config
                layouts_found = 0
                for row in reader:
                    if not row or len(row) < 6:  # Ensure row has enough data
                        continue
                    
                    # Extract full layout data (all items and positions)
                    all_items     = row[0]  # "Items"
                    all_positions = row[1]  # "Positions"
                    all_positions = all_positions.replace('[semicolon]', ';')
                    rank = int(row[4]) if len(row) > 4 else 0
                    score = float(row[5]) if len(row) > 5 else 0
                    
                    layout = {
                        'items': all_items,
                        'positions': all_positions,
                        'score': score,
                        'rank': rank,
                        'config': config_num,
                        'source_file': latest_file
                    }
                    
                    results.append(layout)
                    layouts_found += 1
                    
                    if layouts_found >= layouts_per_config:
                        break

        except Exception as e:
            print(f"Error processing {latest_file}: {e}")
    
    print(f"Extracted {len(results)} layouts from {len(config_results)} configurations")
    return results

def generate_constraint_sets(layouts):
    """Generate configurations based on top-scoring layouts from Step #1."""
    configs = []
    unique_item_sets = set()  # Track unique sets of letters in most comfortable keys
    duplicates_count = 0      # Track number of duplicates found
    error_count = 0           # Track number of layouts with errors
    
    for layout in layouts:
        try:
            # Extract items and positions from the layout
            items = layout['items']  # This is the Items column (e.g., "etaoinsrhldcumfp")
            positions = layout['positions']  # This is the Positions column (e.g., "FDESVJKILMUO;RWA")
            config_num = layout.get('config', 0)
            rank = layout.get('rank', 1)
            
            # Validate that items and positions have the same length
            if len(items) != len(positions):
                print(f"Warning: Items and positions have different lengths in layout from config_{config_num}:")
                print(f"  Items: '{items}' (length: {len(items)})")
                print(f"  Positions: '{positions}' (length: {len(positions)})")
                error_count += 1
                continue
            
            # Create mapping of positions to items
            pos_to_item = {}
            for i, pos in enumerate(positions):
                pos_to_item[pos] = items[i]
            
            # Identify items in the most comfortable keys
            items_in_most_comfortable = ""
            for pos in MOST_COMFORTABLE_KEYS:
                if pos in pos_to_item:
                    items_in_most_comfortable += pos_to_item[pos]
            
            # Skip if there are no items in most comfortable keys
            if not items_in_most_comfortable:
                print(f"Warning: Layout from config_{config_num} has no items in most comfortable keys")
                error_count += 1
                continue
            
            # Check for duplicates
            items_key = items_in_most_comfortable
            if items_key in unique_item_sets:
                duplicates_count += 1
                continue

            # Add to unique sets
            unique_item_sets.add(items_key)
            
            # Fill the to_assign items with letters not in items_in_most_comfortable
            to_assign_items = ""
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            for letter in alphabet:
                if letter not in items_in_most_comfortable and len(to_assign_items) < len(LEAST_COMFORTABLE_KEYS):
                    to_assign_items += letter
            
            # Create config
            config = {
                'items_assigned': items_in_most_comfortable,
                'positions_assigned': MOST_COMFORTABLE_KEYS,
                'items_to_assign': to_assign_items,
                'positions_to_assign': LEAST_COMFORTABLE_KEYS,
                'items_to_constrain': "",
                'positions_to_constrain': "",
                'source_config': config_num,
                'source_rank': rank
            }
            
            configs.append(config)
                
        except Exception as e:
            print(f"Error processing layout from config_{layout.get('config', 0)}: {e}")
            error_count += 1
            continue
            
    print(f"Skipped {duplicates_count} duplicate configurations and {error_count} configurations with errors")
    return configs

def debug_available_positions(layouts):
    all_positions = set()
    for layout in layouts:
        positions = layout['positions']
        for pos in positions:
            all_positions.add(pos)
    print(f"\nAvailable positions in layouts: {''.join(sorted(all_positions))}")
    return all_positions

def create_config_files(configs, output_subdir=""):
    """Create individual config file for each configuration."""
    # Create output directory
    output_dir = OUTPUT_DIR
    if output_subdir:
        output_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {len(configs)} configuration files in {output_dir}...")
    
    for i, config_params in enumerate(configs, 1):
        # Create a copy of the base config
        config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
        
        # Update optimization parameters
        for param, value in config_params.items():
            if param not in ['source_config', 'source_rank', 'source_file']:
                config['optimization'][param] = value
        
        config['optimization']['nlayouts'] = nlayouts
        
        # Determine file naming based on approach
        if output_subdir == "per_config":
            # Per-config approach - use source config and rank
            source_config = config_params.get('source_config', i)
            source_rank = config_params.get('source_rank', 1)
            
            # Set up unique output path based on source config
            config_filename = f"{output_dir}/step2_from_config_{source_config}_rank_{source_rank}.yaml"
        else:
            # Across-all approach - use top N naming
            config_filename = f"{output_dir}/step2_top_{i}.yaml"
        
        # Write the configuration to a YAML file
        with open(config_filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Print progress for every 10 files or at the end
        if i % 10 == 0 or i == len(configs):
            print(f"  Created {i}/{len(configs)} configuration files")
        
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate configurations from existing layouts.')
    parser.add_argument('--step1-dir', type=str, default='../output/layouts',
                        help='Directory containing existing layouts')
    parser.add_argument('--layouts-per-config', type=int, default=1,
                        help='Number of top layouts to take from each existing config (default: 1)')
    parser.add_argument('--top-across-all', type=int, default=0,
                        help='Number of top layouts to take across all existing configs (default: 0)')
    parser.add_argument('--both-approaches', action='store_true',
                        help='Run both per-config and across-all approaches')
    args = parser.parse_args()

    print("Step 2: Generating keyboard layout configurations for least comfortable keys...")
    
    # Find layout results from Step #1, taking the top N layouts from each config
    layouts_per_config = find_layout_results(args.step1_dir, args.layouts_per_config)
    
    # Debug available positions
    available_positions = debug_available_positions(layouts_per_config)

    if not layouts_per_config:
        print("Error: No layout results found from Step #1")
        sys.exit(1)

    # Generate configurations from per-config layouts
    configs_per_config = generate_constraint_sets(layouts_per_config)
    
    # Group layouts by config number for reporting
    configs_represented = len({layout['config'] for layout in layouts_per_config})
    print(f"Found {len(layouts_per_config)} layouts from {configs_represented} Step #1 configurations")
    print(f"Generated {len(configs_per_config)} valid configurations using per-config approach")
    
    # Create directories for different approaches
    os.makedirs(f"{OUTPUT_DIR}/per_config", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/across_all", exist_ok=True)
    
    # If using both approaches or just the per-config approach
    if args.both_approaches or (args.layouts_per_config > 0 and args.top_across_all == 0):
        # Create config files for per-config approach
        print("\nCreating configuration files for per-config approach...")
        create_config_files(configs_per_config, output_subdir="per_config")
    
    # If using both approaches or just the across-all approach
    if args.both_approaches or args.top_across_all > 0:
        # Get top N layouts across all configs
        all_layouts = []
        for file_path in glob.glob(os.path.join(args.step1_dir, "layout_results_*.csv")):
            config_layouts = parse_layout_results(file_path, top_n=100)  # Get many layouts from each file

            for layout in config_layouts:
                try:
                    # Try to extract config number if possible, otherwise use a default
                    if 'config_' in file_path:
                        config_num = int(file_path.split('config_')[1].split('/')[0])
                    else:
                        # Extract number from layout_results_NUMBER_date.csv
                        filename = os.path.basename(file_path)
                        parts = filename.split('_')
                        if len(parts) > 2:
                            try:
                                config_num = int(parts[2])
                            except ValueError:
                                config_num = 0
                        else:
                            config_num = 0

                    layout['config'] = config_num
                    layout['source_file'] = file_path
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not extract config number from {file_path}: {e}")
                    # Still add config information
                    layout['config'] = 0
                    layout['source_file'] = file_path

            all_layouts.extend(config_layouts)
        
        # Sort all layouts by score and take top N
        all_layouts.sort(key=lambda x: x['score'], reverse=True)
        top_layouts = all_layouts[:args.top_across_all]
        
        print(f"\nTaking top {len(top_layouts)} layouts across all {len(all_layouts)} layouts from Step #1")
        
        # Generate configurations from top N layouts (with duplicate prevention)
        configs_across_all = generate_constraint_sets(top_layouts)
        print(f"Generated {len(configs_across_all)} valid configurations using across-all approach")
        
        # Create config files for across-all approach
        print("Creating configuration files for across-all approach...")
        create_config_files(configs_across_all, output_subdir="across_all")
    
    print(f"\nAll configuration files have been generated in the '{OUTPUT_DIR}' directory.")