#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files to run keyboard layout optimizations in parallel 
with specific letter-to-key constraints specified in each file.

This is one step in a process to optimally arrange the 24 most frequent letters 
in the 24 keys of the home block of a keyboard. 

There are two versions of this script:
  - generate_configs1.py generates an initial set of sparse layouts as config files.
  - optimize_layout.py generates optimal layouts for a given config file.
  - generate_configs2.py generates a new set of config files based on the optimal layouts.
  
See **README_keyboard_application.md** for a full description.

See **README.md** for instructions to run batches of config files in parallel.

"""
import os
import yaml
import math
import itertools

# Configuration
OUTPUT_DIR = 'configs'

# Define fixed items for all configurations
items_assigned  = "etaoi"            # 5 most frequent letters (etaoi in English)
items_to_assign = "nsrhldcumfpgwyb"  # next 15 most frequent letters (nsrhldcumfpgwyb in English)

# Position constraints
positions_for_item1 = ["F","D"] # 2 most comfortable left qwerty keys for the most frequent letter
positions_for_items_2thru5 = ["F","D","R","S","E","V","A","W",
                              "J","K","U","L","I","M",";","O"] # 16 most comfortable qwerty keys

# Keys
ALL_KEYS = "FDRSEVAWCQJKULIM;O,P"  # 20 most comfortable keys

# Base configuration from the original config file
with open('config.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

def generate_constraint_sets():
    """Generate all valid configurations based on the constraints."""

    # Generate all valid configurations
    configs = []
    
    # Loop through all possible position combinations
    n_items_to_assign = len(items_to_assign)
    for pos1 in positions_for_item1:

        # Find all available positions for 4 remaining items in items_assigned
        remaining_positions = [pos for pos in positions_for_items_2thru5 if pos not in [pos1]]
        
        # Generate all combinations of 4 positions
        for combo4 in itertools.combinations(remaining_positions, 4):

            # Generate all permutations of these combinations
            for perm4 in itertools.permutations(combo4):
                pos2, pos3, pos4, pos5 = perm4
                
                # Create final position assignments
                positions = {items_assigned[0]: pos1, 
                             items_assigned[1]: pos2, 
                             items_assigned[2]: pos3, 
                             items_assigned[3]: pos4, 
                             items_assigned[4]: pos5}
                
                # Create the positions_assigned string (must match the order of items_assigned)
                positions_assigned = ''.join([positions[letter] for letter in items_assigned])
                
                # Create positions_to_assign (15 keys not used in positions_assigned)
                used_positions = set(positions_assigned)
                positions_to_assign = ''.join([pos for pos in ALL_KEYS if pos not in used_positions])
                
                # Add to configs if valid and positions_to_assign has the correct number of positions
                if len(positions_to_assign) == n_items_to_assign:
                    configs.append({
                        'items_to_assign': items_to_assign,
                        'positions_to_assign': positions_to_assign,
                        'items_assigned': items_assigned,
                        'positions_assigned': positions_assigned,
                        'items_to_constrain': "",  # No constraints for these configurations
                        'positions_to_constrain': ""
                    })

    return configs

def create_config_files(configs, nlayouts=100):
    """Create individual config file for each configuration."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Print the total number of configs we're working with
    print(f"Creating {len(configs)} configuration files...")
    
    for i, config_params in enumerate(configs, 1):
        # Create a copy of the base config
        config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
        
        # Update optimization parameters
        for param, value in config_params.items():
            config['optimization'][param] = value
        
        # Set nlayouts
        config['optimization']['nlayouts'] = nlayouts
        
        # Set up unique output path
        #config['paths']['output']['layout_results_folder'] = f"output/layouts/config_{i}"
        #os.makedirs(config['paths']['output']['layout_results_folder'], exist_ok=True)
        
        # Write the configuration to a YAML file
        config_filename = f"{OUTPUT_DIR}/config_{i}.yaml"
        with open(config_filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Print progress feedback for every 100 files
        if i % 100 == 0:
            print(f"  Created {i}/{len(configs)} configuration files...")
        
if __name__ == "__main__":
    nlayouts = 100
    print("Generating keyboard layout configurations...")
    configs = generate_constraint_sets()
    print(f"Found {len(configs)} valid configurations based on the constraints.")
    create_config_files(configs, nlayouts)
    print(f"All configuration files have been generated in the '{OUTPUT_DIR}' directory.")
    
    # Print details about the first few configs
    num_examples = min(3, len(configs))
    print(f"\nShowing details for first {num_examples} configurations:")
    for i in range(num_examples):
        config = configs[i]
        print(f"\nConfig {i+1}:")
        print(f"  items_assigned: {config['items_assigned']}")
        print(f"  positions_assigned: {config['positions_assigned']}")
        print(f"  positions_to_assign: {config['positions_to_assign']}")
        
        # Map letters to positions for clarity
        letter_positions = {config['items_assigned'][j]: config['positions_assigned'][j] 
                          for j in range(len(config['items_assigned']))}
        print("  Letter mappings:")
        for letter, pos in letter_positions.items():
            print(f"    {letter} -> {pos}")
            
    # Calculate the number of possible combinations 
    positions_for_item1 = 2
    t_valid_positions = 5  # 6 keys minus 1 where e is placed
    aoi_combinations = math.comb(14, 3)  # 364 ways to choose 3 from 14
    perm4utations = 6  # 3! = 6 ways to arrange a,o,i
    
    print(f"\nTheoretical maximum configurations: {positions_for_item1 * t_valid_positions * aoi_combinations * perm4utations}")
    print(f"Actual configurations: {len(configs)}")
