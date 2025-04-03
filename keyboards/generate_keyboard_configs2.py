#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files to run keyboard layout optimizations in parallel 
with specific letter-to-key constraints specified in each file.

This is one step in a process to optimally arrange the 24 most frequent letters 
in the 24 keys of the home block of a keyboard. 

There are two versions of this script:
  - generate_keyboard_configs1.py generates an initial set of sparse keyboard layouts as config files.
  - optimize_layout.py generates optimal keyboard layouts for a given config file.
  - generate_keyboard_configs2.py generates a new set of config files based on the optimal keyboard layouts.

See **README_keyboards.md** for a full description.

See **README.md** for instructions to run batches of config files in parallel.

Usage:

    Per-config approach (default: 1 layout per config file):
    ``python generate_keyboard_configs2.py --layouts-per-config 10``

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
OUTPUT_DIR = '../output/configs'
CONFIG_FILE = '../config.yaml'
nlayouts = 100 

# Positions
MOST_COMFORTABLE_KEYS = "FDRSEVJKULIM"   # 12 most comfortable keys
LEAST_COMFORTABLE_KEYS = "AWCQXZ;O,P./"  # 12 least comfortable keys

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
            
            # Skip configuration info (variable number of rows until we find the header)
            row = next(reader, None)
            while row and not ('Items' in row and 'Positions' in row and 'Rank' in row):
                row = next(reader, None)
                
            # If we never found the header, return empty
            if not row:
                print(f"Warning: Could not find header row in {results_path}")
                return layouts
                
            # Read top N data rows
            for _ in range(top_n):
                layout_row = next(reader, None)
                if not layout_row or len(layout_row) < 4:
                    break
                    
                # Extract items (letters) and positions
                items = layout_row[0]
                positions = layout_row[1]
                rank = int(layout_row[2]) if len(layout_row) > 2 else 0
                score = float(layout_row[3]) if len(layout_row) > 3 else 0
                
                layout = {
                    'items': items,
                    'positions': positions,
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
    
    pattern = os.path.join(step1_dir, "config_*/layout_results_*.csv")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"No layout result files found in {step1_dir}")
        return results
    
    print(f"Found {len(result_files)} result files from Step #1")
    
    # First, organize files by config number
    for file_path in result_files:
        try:
            config_num = int(file_path.split('config_')[1].split('/')[0])
            config_results[config_num].append(file_path)
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not extract config number from {file_path}: {e}")
    
    print(f"Found results for {len(config_results)} different Step #1 configurations")
    
    # Process each config's results
    for config_num, file_paths in config_results.items():
        # Sort by timestamp to get the latest result file for this config
        file_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Take the most recent file for this config
        latest_file = file_paths[0]
        
        # Parse layouts from this file
        with open(latest_file, 'r') as f:
            reader = csv.reader(f)
            
            # Skip configuration info
            for _ in range(10):  # Skip first rows of config info
                next(reader, None)
            
            # Read header row
            header = next(reader, None)
            if not header:
                continue
            
            # Read top N data rows for this config
            layouts_found = 0
            for row in reader:
                if not row or len(row) < 4:  # Ensure row has enough data
                    continue
                
                # Extract items (letters) and positions
                items = row[0]
                positions = row[1]
                rank = int(row[2]) if len(row) > 2 else 0
                score = float(row[3]) if len(row) > 3 else 0
                
                layout = {
                    'items': items,
                    'positions': positions,
                    'score': score,
                    'rank': rank,
                    'config': config_num,
                    'source_file': latest_file
                }
                
                results.append(layout)
                layouts_found += 1
                
                if layouts_found >= layouts_per_config:
                    break
    
    print(f"Extracted {len(results)} layouts from {len(config_results)} configurations")
    return results

def generate_constraint_sets(layouts):
    """Generate configurations based on top-scoring layouts from Step #1."""
    configs = []
    unique_item_sets = set()  # Track unique sets of letters in most comfortable keys
    duplicates_count = 0      # Track number of duplicates found
    
    for layout in layouts:
        # Extract items and positions from the layout
        items = layout['items']
        positions = layout['positions']
        config_num = layout.get('config', 0)
        rank = layout.get('rank', 1)
        
        # Create mapping of positions to items
        pos_to_item = {}
        for i, pos in enumerate(positions):
            pos_to_item[pos] = items[i]
        
        # Create mapping of items to positions
        item_to_pos = {}
        for i, item in enumerate(items):
            item_to_pos[item] = positions[i]
        
        # Identify items in the most/least comfortable keys
        items_in_most_comfortable = ''.join([pos_to_item.get(pos, '') for pos in MOST_COMFORTABLE_KEYS])
        
        # Check if we've already seen this set of most comfortable letters
        items_key = ''.join(sorted(items_in_most_comfortable))
        if items_key in unique_item_sets:
            # Skip this layout as it would produce a duplicate config
            duplicates_count += 1
            continue
        
        # Add to our set of unique letter combinations
        unique_item_sets.add(items_key)
        
        items_in_least_comfortable = ''.join([pos_to_item.get(pos, '') for pos in LEAST_COMFORTABLE_KEYS])
        
        # Add vkxj to items_to_assign (as specified in the docstring)
        fixed_items_to_add = 'vkxj'
        items_to_assign = fixed_items_to_add + items_in_least_comfortable
        items_to_assign = ''.join(sorted(set(items_to_assign), key=items_to_assign.index))  # Remove duplicates
        
        # Create config
        config = {
            'items_to_assign': items_to_assign[:14],  # Take first 14 characters
            'positions_to_assign': LEAST_COMFORTABLE_KEYS,
            'items_assigned': items_in_most_comfortable,
            'positions_assigned': MOST_COMFORTABLE_KEYS,
            'items_to_constrain': "",  # No constraints for these configurations
            'positions_to_constrain': "",
            'source_config': config_num,  # Store source config for reference
            'source_rank': rank  # Store rank in original config
        }
        
        # Validate config
        if len(config['items_to_assign']) == 14 and len(config['items_assigned']) == 10:
            configs.append(config)
        else:
            print(f"Warning: Invalid configuration generated from config_{config_num}, rank {rank}:")
            print(f"  items_to_assign: '{config['items_to_assign']}' (length: {len(config['items_to_assign'])})")
            print(f"  items_assigned: '{config['items_assigned']}' (length: {len(config['items_assigned'])})")
            
            # Try to fix the configuration
            if len(config['items_to_assign']) < 14:
                # Add more common letters until we have 14
                common_letters = 'zqxjkvbpygwfmcdhrlstnaoeiu'  # Least to most common
                for letter in common_letters:
                    if letter not in config['items_to_assign'] and letter not in config['items_assigned']:
                        config['items_to_assign'] += letter
                        if len(config['items_to_assign']) == 14:
                            break
                            
                print(f"  Fixed items_to_assign: '{config['items_to_assign']}' (length: {len(config['items_to_assign'])})")
                
                # Add if fixed successfully
                if len(config['items_to_assign']) == 14 and len(config['items_assigned']) == 10:
                    configs.append(config)
                    print("  Configuration fixed and added.")
    
    print(f"Skipped {duplicates_count} duplicate configurations")
    return configs

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
            output_folder = f"output/layouts/step2_per_config/from_config_{source_config}_rank_{source_rank}"
            config_filename = f"{output_dir}/step2_from_config_{source_config}_rank_{source_rank}.yaml"
        else:
            # Across-all approach - use top N naming
            output_folder = f"output/layouts/step2_across_all/top_{i}"
            config_filename = f"{output_dir}/step2_top_{i}.yaml"
        
        # Set output folder in config
        #config['paths']['output']['layout_results_folder'] = output_folder
        #os.makedirs(output_folder, exist_ok=True)
        
        # Write the configuration to a YAML file
        with open(config_filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Print progress for every 10 files or at the end
        if i % 10 == 0 or i == len(configs):
            print(f"  Created {i}/{len(configs)} configuration files")
        
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate configurations from existing layouts.')
    parser.add_argument('--step1-dir', type=str, default='output/layouts',
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
        for file_path in glob.glob(os.path.join(args.step1_dir, "config_*/layout_results_*.csv")):
            config_layouts = parse_layout_results(file_path, top_n=100)  # Get many layouts from each file
            for layout in config_layouts:
                try:
                    config_num = int(file_path.split('config_')[1].split('/')[0])
                    layout['config'] = config_num
                    layout['source_file'] = file_path
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not extract config number from {file_path}: {e}")
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